"""Corpus-level disagreement diagnostics.

Over a corpus, compute each field path's **contention rate** -- the fraction of
records where it was contested (status != ``accepted``) -- and the most common
disagreement patterns, so the user learns which fields are *systematically* hard
(a prompt/schema problem) versus a per-doc fluke (plan 3.6).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .types import EscalationStatus, RecordResult


@dataclass(frozen=True, slots=True)
class FieldDiagnostic:
    """Contention summary for one field path across a corpus."""

    path: str
    n_records: int
    n_contested: int
    contention_rate: float
    mean_agreement: float
    top_disagreement_keys: list[tuple[str, int]]
    """The most common winning ``normalized_key`` values among contested records
    (key, count), descending. Surfaces recurrent wrong/ambiguous buckets."""


def field_contention(records: Sequence[RecordResult]) -> list[FieldDiagnostic]:
    """Compute per-field contention diagnostics over a corpus.

    Returns one :class:`FieldDiagnostic` per field path, sorted by descending
    contention rate then path (deterministic).
    """
    paths: list[str] = []
    seen: set[str] = set()
    for record in records:
        for path in record.fields:
            if path not in seen:
                seen.add(path)
                paths.append(path)

    out: list[FieldDiagnostic] = []
    for path in paths:
        n = 0
        contested = 0
        agreement_sum = 0.0
        keys: Counter[str] = Counter()
        for record in records:
            fr = record.fields.get(path)
            if fr is None:
                continue
            n += 1
            agreement_sum += fr.agreement
            if fr.status is not EscalationStatus.ACCEPTED:
                contested += 1
                if fr.winning_key is not None:
                    keys[fr.winning_key] += 1
        rate = contested / n if n else 0.0
        mean_agreement = agreement_sum / n if n else 0.0
        out.append(
            FieldDiagnostic(
                path=path,
                n_records=n,
                n_contested=contested,
                contention_rate=rate,
                mean_agreement=mean_agreement,
                top_disagreement_keys=keys.most_common(3),
            )
        )

    out.sort(key=lambda d: (-d.contention_rate, d.path))
    return out


def systematically_contested(records: Sequence[RecordResult], threshold: float = 0.5) -> list[str]:
    """Field paths whose contention rate meets ``threshold`` (sorted)."""
    return [d.path for d in field_contention(records) if d.contention_rate >= threshold]


@dataclass(frozen=True, slots=True)
class LabelSuggestion:
    """A ``(doc_id, path)`` pick to label next, ranked by informativeness."""

    doc_id: str
    path: str
    agreement: float


def suggest_labels(
    records: Sequence[RecordResult],
    *,
    n: int | None = None,
    boundary: float = 0.5,
) -> list[LabelSuggestion]:
    """Rank ``(doc_id, path)`` records by how informative labeling them would be.

    Active-learning first slice (plan: "grow the calibration set"): the most
    informative labels are the ones the ensemble is least sure about -- agreement
    sitting right on the accept ``boundary`` (the quorum threshold). Contested
    fields break ties (they cost a human/escalation either way), then a stable
    ``(doc_id, path)`` order makes the output fully deterministic and offline.

    Args:
        records: The corpus results (from :func:`read_results`).
        n: Return at most this many suggestions (all when ``None``); negative
            values yield an empty list.
        boundary: The accept threshold agreement is measured against.

    Returns:
        Suggestions ordered most-informative first.
    """
    scored: list[tuple[float, int, str, str, float]] = []
    for record in records:
        for path in sorted(record.fields):
            fr = record.fields[path]
            distance = abs(fr.agreement - boundary)
            contested = 0 if fr.status is not EscalationStatus.ACCEPTED else 1
            scored.append((distance, contested, record.doc_id, path, fr.agreement))
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    out = [
        LabelSuggestion(doc_id=doc_id, path=path, agreement=agreement)
        for _, _, doc_id, path, agreement in scored
    ]
    if n is not None:
        return out[: max(n, 0)]
    return out
