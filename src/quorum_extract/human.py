"""Human-review queue and safe apply-back of resolved values.

A simple, auditable, file-backed queue (plan 3.5):

* :class:`ReviewQueue` is a JSONL of ``(doc_id, path, candidates)`` items for
  fields that reached ``needs_review``.
* A human resolves an item, which appends a ``(doc_id, path, value)`` line to a
  **separate** ``resolved-overrides.jsonl``.
* :func:`apply_overrides` merges those overrides into records **at read/report
  time** -- the original ``results.jsonl`` is never rewritten in place (safer and
  auditable). A merged field is marked ``resolved``.

Because of the field-completeness invariant upstream, an override can never
target a field that does not exist on a record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import EscalationStatus, RecordResult


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One field awaiting human resolution."""

    doc_id: str
    path: str
    candidates: list[Any]

    def to_json(self) -> str:
        return json.dumps({"doc_id": self.doc_id, "path": self.path, "candidates": self.candidates})

    @classmethod
    def from_json(cls, line: str) -> ReviewItem:
        d = json.loads(line)
        return cls(doc_id=d["doc_id"], path=d["path"], candidates=list(d.get("candidates", [])))


@dataclass(frozen=True, slots=True)
class Override:
    """A resolved value to merge back into a record at report time."""

    doc_id: str
    path: str
    value: Any

    def to_json(self) -> str:
        return json.dumps({"doc_id": self.doc_id, "path": self.path, "value": self.value})

    @classmethod
    def from_json(cls, line: str) -> Override:
        d = json.loads(line)
        return cls(doc_id=d["doc_id"], path=d["path"], value=d["value"])


class ReviewQueue:
    """A JSONL-backed queue of fields needing human review."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def push(self, item: ReviewItem) -> None:
        """Append one item to the queue file (created if absent)."""
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(item.to_json() + "\n")

    def extend(self, items: Iterable[ReviewItem]) -> None:
        """Append many items."""
        with self.path.open("a", encoding="utf-8") as fh:
            for item in items:
                fh.write(item.to_json() + "\n")

    def load(self) -> list[ReviewItem]:
        """Read all queued items (empty list if the file does not exist)."""
        if not self.path.exists():
            return []
        return [
            ReviewItem.from_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def items_for_review(records: Sequence[RecordResult]) -> list[ReviewItem]:
    """Build review items for every ``needs_review`` field across records.

    Candidates are the distinct non-missing raw values seen in the field's
    votes, preserving first-seen order for determinism.
    """
    items: list[ReviewItem] = []
    for record in records:
        for path in record.contested_paths():
            fr = record.fields[path]
            if fr.status is not EscalationStatus.NEEDS_REVIEW:
                continue
            seen: list[Any] = []
            for v in fr.votes:
                if v.missing:
                    continue
                if v.raw_value not in seen:
                    seen.append(v.raw_value)
            items.append(ReviewItem(doc_id=record.doc_id, path=path, candidates=seen))
    return items


def write_override(path: str | Path, override: Override) -> None:
    """Append a resolved override to the overrides JSONL (never rewrites results)."""
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(override.to_json() + "\n")


def load_overrides(path: str | Path) -> list[Override]:
    """Read all overrides (empty if the file does not exist)."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        Override.from_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def apply_overrides(
    records: Sequence[RecordResult], overrides: Sequence[Override]
) -> list[RecordResult]:
    """Merge overrides into records at read time (returns new records).

    A matched field gets the resolved value and ``status = resolved``. The
    originals are not mutated. Later overrides for the same ``(doc_id, path)``
    win (last-write-wins), matching append-only resolution.
    """
    by_key: dict[tuple[str, str], Any] = {}
    for ov in overrides:
        by_key[(ov.doc_id, ov.path)] = ov.value

    out: list[RecordResult] = []
    for record in records:
        new_fields = dict(record.fields)
        for (doc_id, path), value in by_key.items():
            if doc_id != record.doc_id:
                continue
            if path not in new_fields:
                # Invariant guarantees this should not happen; skip defensively.
                continue  # pragma: no cover
            new_fields[path] = new_fields[path].model_copy(
                update={"value": value, "status": EscalationStatus.RESOLVED}
            )
        out.append(record.model_copy(update={"fields": new_fields}))
    return out
