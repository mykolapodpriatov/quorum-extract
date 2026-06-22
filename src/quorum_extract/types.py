"""Core domain types for quorum extraction.

These are intentionally small, explicit, and (mostly) Pydantic-validated so the
whole pipeline -- voting, calibration, cascade, budget, reporting -- speaks one
vocabulary.

Key invariants encoded here:

* ``ExtractorSpec.cost_usd`` is the cost of *one invocation* that extracts the
  whole record. There is no per-field billing; an extractor that fills 30
  fields still costs ``cost_usd`` once.
* A failed/timed-out extractor is represented as a ``FieldVote`` with
  ``missing=True`` for every field, so the agreement denominator ``K`` is the
  number of extractors *configured*, never the number that happened to succeed.
* ``EscalationStatus`` is a closed enum of lifecycle states a field can reach.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# A document handed to an extractor. Deliberately permissive: it may be raw
# text, a path, or a structured payload -- the extractor decides how to read it.
DocInput = Any

# The merge policies for cost-aware escalation (see cascade.py / plan 3.5).
EscalationMerge = Literal["strong_wins", "re_quorum", "consensus"]


class EscalationStatus(StrEnum):
    """Lifecycle state of a reconciled field.

    :class:`enum.StrEnum` (Python 3.11+) keeps these JSON-serializable as their
    value and comparable to plain strings, which keeps report/CLI code simple.
    """

    ACCEPTED = "accepted"
    """Quorum (and confidence, if a calibrator is set) was satisfied by the
    cheap extractors; the value is taken as-is."""

    ESCALATED_MODEL = "escalated_model"
    """The field was contested and resolved by a higher-tier model invocation."""

    NEEDS_REVIEW = "needs_review"
    """Still contested after available escalation (or past the budget cap);
    queued for a human. The field is never dropped."""

    RESOLVED = "resolved"
    """A human (or an override file) supplied the final value."""


class ExtractorSpec(BaseModel):
    """One extractor in the ensemble.

    Attributes:
        name: Unique, stable identifier used in votes, reports, and fingerprints.
        fn: Callable mapping a document to a flat-ish dict of extracted values.
            It extracts the *whole* record in one call.
        cost_usd: Cost of a single invocation (per-document, not per-field).
        tier: Cost/strength tier. ``0`` = cheap baseline; higher = stronger and
            more expensive. Used for deterministic tie-breaks and to pick the
            escalation target.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    name: str = Field(min_length=1)
    fn: Callable[[DocInput], dict[str, Any]]
    cost_usd: float = Field(ge=0.0)
    tier: int = Field(default=0, ge=0)

    @field_validator("fn")
    @classmethod
    def _fn_is_callable(cls, value: Callable[[DocInput], dict[str, Any]]) -> Callable[..., Any]:
        if not callable(value):
            raise ValueError("ExtractorSpec.fn must be callable")
        return value


class FieldVote(BaseModel):
    """A single extractor's vote for a single field path.

    ``normalized_key`` is the bucket the vote falls into for equality purposes;
    two votes agree iff their ``normalized_key`` matches. ``missing`` marks the
    unified absence bucket (absent key / ``None`` / empty / whitespace, or a
    failed extractor).
    """

    model_config = ConfigDict(frozen=True)

    extractor: str
    raw_value: Any = None
    normalized_key: str
    missing: bool = False


class FieldResult(BaseModel):
    """Reconciled outcome for one field path of one document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    value: Any = None
    votes: list[FieldVote] = Field(default_factory=list)
    agreement: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    status: EscalationStatus = EscalationStatus.ACCEPTED
    winning_key: str | None = None
    """The ``normalized_key`` of the bucket whose value was chosen (or
    ``"__missing__"``). Kept for diagnostics and audit."""


class RecordResult(BaseModel):
    """All reconciled fields for one document, plus what it cost to produce."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    doc_id: str
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    cost_usd: float = Field(default=0.0, ge=0.0)

    def contested_paths(self) -> list[str]:
        """Paths whose status is not ``accepted`` (sorted, deterministic)."""
        return sorted(
            path for path, fr in self.fields.items() if fr.status is not EscalationStatus.ACCEPTED
        )


class QuorumConfig(BaseModel):
    """Tunable policy for reconciliation, calibration gating, and escalation.

    A failed/timed-out extractor always counts as a ``missing`` vote (it never
    reduces ``K``), so the agreement denominator is constant regardless of which
    extractors succeed.
    """

    model_config = ConfigDict(frozen=True)

    min_agreement: float = Field(default=0.5, ge=0.0, le=1.0)
    """Quorum threshold: accept a field iff ``agreement >= min_agreement``."""

    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """If set (and a calibrator is present), also require
    ``confidence >= min_confidence`` to accept."""

    escalate_tier: int | None = Field(default=None, ge=0)
    """Tier of the extractor used to resolve contested fields. ``None`` disables
    model escalation (contested fields go straight to ``needs_review``)."""

    escalation_merge: EscalationMerge = "strong_wins"
    """How the strong model's result is merged with the cheap votes."""

    list_key: dict[str, str] = Field(default_factory=dict)
    """Maps a list-of-objects field path to the sub-field used to align rows
    across extractors (e.g. ``{"line_items": "sku"}``). Without an entry, a
    list is treated as a single structural leaf."""

    max_cost_usd: float | None = Field(default=None, ge=0.0)
    """Optional escalation budget cap. Once exceeded, remaining contested fields
    are marked ``needs_review`` (never dropped)."""

    calibrator_path: str | None = None
    """Optional path to a persisted calibrator JSON to load for confidence."""

    @field_validator("min_agreement")
    @classmethod
    def _agreement_in_unit(cls, value: float) -> float:
        # ge/le already enforce [0,1]; this validator documents intent and is a
        # hook for future stricter checks.
        return value


class BudgetReport(BaseModel):
    """Cost accounting for one run."""

    cheap_cost_usd: float = 0.0
    escalation_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    all_frontier_cost_usd: float = 0.0
    """Hypothetical cost of running the strong model on *every* document."""
    saved_usd: float = 0.0
    """``all_frontier_cost_usd - (cheap escalation actually spent)`` -- the
    measurable value proposition."""
    docs_total: int = 0
    docs_escalated: int = 0
    docs_over_budget: int = 0


class RunReport(BaseModel):
    """Top-level result of a ``run``: per-record results plus aggregate stats."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    records: list[RecordResult] = Field(default_factory=list)
    budget: BudgetReport = Field(default_factory=BudgetReport)
    leaf_paths: list[str] = Field(default_factory=list)
    n_accepted: int = 0
    n_escalated: int = 0
    n_needs_review: int = 0
    n_resolved: int = 0
