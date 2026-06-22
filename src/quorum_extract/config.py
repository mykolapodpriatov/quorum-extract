"""Project configuration: schema + extractors + quorum/calibration/budget.

A project is described by a :class:`ProjectConfig`. Because extractors are
arbitrary Python callables (and may hold provider clients), config is supplied
as a **Python module** that defines a top-level ``config`` (or ``CONFIG``)
object -- not a static data file. :func:`load_config` imports such a module by
path.

This module also owns (de)serialization of :class:`~quorum_extract.types.RecordResult`
to/from JSONL so ``run`` can persist results and ``report`` can read them back.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from .calibration import LabeledExample

from .schema import LeafPath, leaf_paths
from .types import (
    ExtractorSpec,
    FieldResult,
    FieldVote,
    QuorumConfig,
    RecordResult,
)


@dataclass
class ProjectConfig:
    """Everything ``run`` needs to process a corpus.

    Attributes:
        schema: The target Pydantic v2 model.
        extractors: The cheap ensemble (tier 0). At least one is required.
        quorum: Reconciliation/escalation policy.
        strong_extractor: Optional escalation extractor (higher tier).
        calibration_groups: Optional map of field path -> group name for
            per-group calibration (global fallback when sparse).
        load_docs: Optional callable that yields ``(doc_id, payload)`` pairs from
            a docs directory/path, used by the CLI ``run`` command.
    """

    schema: type[BaseModel]
    extractors: list[ExtractorSpec]
    quorum: QuorumConfig = field(default_factory=QuorumConfig)
    strong_extractor: ExtractorSpec | None = None
    calibration_groups: dict[str, str] = field(default_factory=dict)
    load_docs: Any = None

    def __post_init__(self) -> None:
        if not self.extractors:
            raise ValueError("ProjectConfig.extractors must contain at least one extractor")

    def leaf_paths(self) -> list[LeafPath]:
        """The schema's leaf paths under this project's ``list_key`` declarations."""
        return leaf_paths(self.schema, self.quorum.list_key)

    def leaf_path_strings(self) -> list[str]:
        return [lp.path for lp in self.leaf_paths()]

    def extractor_names(self) -> list[str]:
        names = [e.name for e in self.extractors]
        if self.strong_extractor is not None:
            names.append(self.strong_extractor.name)
        return names


def load_config(path: str | Path) -> ProjectConfig:
    """Import a Python config module and return its ``config``/``CONFIG`` object.

    Args:
        path: Path to a ``.py`` file defining a top-level ``config`` (or
            ``CONFIG``) :class:`ProjectConfig`.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the module defines no usable config object.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    # A stable, unique module name derived from the path. Registering the module
    # in sys.modules *before* executing it is required so that Pydantic models in
    # the config can resolve forward references to each other (which arise when
    # the config uses ``from __future__ import annotations``).
    mod_name = f"quorum_extract_user_config_{abs(hash(str(p.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ValueError(f"could not load config module from {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise
    cfg = getattr(module, "config", None) or getattr(module, "CONFIG", None)
    if cfg is None:
        raise ValueError(
            f"config module {p} must define a top-level `config` (or `CONFIG`) ProjectConfig"
        )
    if not isinstance(cfg, ProjectConfig):
        raise ValueError(f"`config` in {p} must be a ProjectConfig, got {type(cfg).__name__}")
    return cfg


# --------------------------------------------------------------------------- #
# RecordResult <-> JSONL
# --------------------------------------------------------------------------- #


def _vote_to_dict(v: FieldVote) -> dict[str, Any]:
    return {
        "extractor": v.extractor,
        "raw_value": v.raw_value,
        "normalized_key": v.normalized_key,
        "missing": v.missing,
    }


def record_to_dict(record: RecordResult) -> dict[str, Any]:
    """Serialize a :class:`RecordResult` to a JSON-able dict."""
    return {
        "doc_id": record.doc_id,
        "cost_usd": record.cost_usd,
        "fields": {
            path: {
                "path": fr.path,
                "value": fr.value,
                "agreement": fr.agreement,
                "confidence": fr.confidence,
                "status": fr.status.value,
                "winning_key": fr.winning_key,
                "votes": [_vote_to_dict(v) for v in fr.votes],
            }
            for path, fr in record.fields.items()
        },
    }


def record_from_dict(d: dict[str, Any]) -> RecordResult:
    """Deserialize a :class:`RecordResult` from a dict produced by
    :func:`record_to_dict`."""
    fields: dict[str, FieldResult] = {}
    for path, fd in d.get("fields", {}).items():
        votes = [
            FieldVote(
                extractor=v["extractor"],
                raw_value=v.get("raw_value"),
                normalized_key=v["normalized_key"],
                missing=v.get("missing", False),
            )
            for v in fd.get("votes", [])
        ]
        fields[path] = FieldResult(
            path=fd["path"],
            value=fd.get("value"),
            votes=votes,
            agreement=fd["agreement"],
            confidence=fd.get("confidence"),
            status=fd["status"],
            winning_key=fd.get("winning_key"),
        )
    return RecordResult(
        doc_id=d["doc_id"],
        fields=fields,
        cost_usd=d.get("cost_usd", 0.0),
    )


def write_results(path: str | Path, records: Iterable[RecordResult]) -> None:
    """Write records to a JSONL file (one record per line)."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record_to_dict(record)) + "\n")


def read_results(path: str | Path) -> list[RecordResult]:
    """Read records from a JSONL file produced by :func:`write_results`."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    return [record_from_dict(json.loads(line)) for line in lines if line.strip()]


def write_labeled(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write labeled calibration rows to JSONL.

    Each row is ``{"winning_share": float, "k": int, "entropy": float,
    "correct": bool, "group"?: str}``.
    """
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def read_labeled(path: str | Path) -> list[LabeledExample]:
    """Read labeled calibration rows from JSONL into :class:`LabeledExample`.

    Expected per-line schema: ``{"winning_share": float, "k": int,
    "entropy": float, "correct": bool, "group"?: str}``. Missing ``k``/``entropy``
    default to ``1`` / ``0.0``.

    Raises:
        ValueError: if a row lacks a required key (``winning_share``/``correct``).
    """
    from .agreement import AgreementFeatures
    from .calibration import LabeledExample

    out: list[LabeledExample] = []
    p = Path(path)
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        if "winning_share" not in row or "correct" not in row:
            raise ValueError(f"labeled row {lineno} in {p} must have 'winning_share' and 'correct'")
        group_val = row.get("group")
        out.append(
            LabeledExample(
                features=AgreementFeatures(
                    winning_share=float(row["winning_share"]),
                    k=int(row.get("k", 1)),
                    entropy=float(row.get("entropy", 0.0)),
                ),
                correct=bool(row["correct"]),
                group=str(group_val) if group_val is not None else None,
            )
        )
    return out
