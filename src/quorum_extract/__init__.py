"""quorum-extract: field-level quorum extraction with calibrated confidence.

Run the same Pydantic schema against K cheap extractors, reconcile **per field**
(not per record), calibrate inter-model agreement into a real per-field
confidence, and escalate only the contested fields to a stronger model or a
human -- with a cost report.

Public API (stable surface):

* Types: :class:`ExtractorSpec`, :class:`FieldVote`, :class:`FieldResult`,
  :class:`RecordResult`, :class:`QuorumConfig`, :class:`EscalationStatus`,
  :class:`RunReport`, :class:`BudgetReport`.
* Schema: :func:`leaf_paths`, :class:`LeafPath`.
* Normalization: :func:`normalize_value`, :data:`MISSING_KEY`.
* Quorum: :func:`quorum_record`, :func:`quorum_field`, :class:`ExtractorOutput`.
* Calibration: :class:`AgreementCalibrator`, :class:`LabeledExample`,
  :class:`CalibrationError`, :class:`AgreementFeatures`.
* Cascade/budget: :func:`cascade_corpus`, :class:`Document`, :class:`BudgetTracker`.
* Extractors: :class:`FakeExtractor`, :func:`run_extractors`,
  provider helpers (``openai_extractor`` / ``anthropic_extractor`` /
  ``ollama_extractor``).
* Config/pipeline: :class:`ProjectConfig`, :func:`load_config`,
  :func:`run_project`.
"""

from __future__ import annotations

from .agreement import AgreementFeatures, features_for
from .budget import BudgetTracker
from .calibration import (
    AgreementCalibrator,
    CalibrationError,
    Fingerprint,
    LabeledExample,
    fingerprint_for,
)
from .cascade import CascadeResult, Document, cascade_corpus
from .config import ProjectConfig, load_config, read_results, write_results
from .diagnostics import FieldDiagnostic, field_contention, systematically_contested
from .extractors import (
    Extractor,
    ExtractorFailure,
    FakeExtractor,
    anthropic_extractor,
    ollama_extractor,
    openai_extractor,
    run_extractors,
)
from .human import (
    Override,
    ReviewItem,
    ReviewQueue,
    apply_overrides,
    items_for_review,
    load_overrides,
)
from .normalize import MISSING_KEY, is_missing, normalize_value
from .pipeline import run_project
from .quorum import ExtractorOutput, quorum_field, quorum_record
from .schema import LeafKind, LeafPath, SchemaError, leaf_path_strings, leaf_paths
from .types import (
    BudgetReport,
    EscalationStatus,
    ExtractorSpec,
    FieldResult,
    FieldVote,
    QuorumConfig,
    RecordResult,
    RunReport,
)

__version__ = "0.1.0"

__all__ = [
    "MISSING_KEY",
    "AgreementCalibrator",
    "AgreementFeatures",
    "BudgetReport",
    "BudgetTracker",
    "CalibrationError",
    "CascadeResult",
    "Document",
    "EscalationStatus",
    "Extractor",
    "ExtractorFailure",
    "ExtractorOutput",
    "ExtractorSpec",
    "FakeExtractor",
    "FieldDiagnostic",
    "FieldResult",
    "FieldVote",
    "Fingerprint",
    "LabeledExample",
    "LeafKind",
    "LeafPath",
    "Override",
    "ProjectConfig",
    "QuorumConfig",
    "RecordResult",
    "ReviewItem",
    "ReviewQueue",
    "RunReport",
    "SchemaError",
    "__version__",
    "anthropic_extractor",
    "apply_overrides",
    "cascade_corpus",
    "features_for",
    "field_contention",
    "fingerprint_for",
    "is_missing",
    "items_for_review",
    "leaf_path_strings",
    "leaf_paths",
    "load_config",
    "load_overrides",
    "normalize_value",
    "ollama_extractor",
    "openai_extractor",
    "quorum_field",
    "quorum_record",
    "read_results",
    "run_extractors",
    "run_project",
    "systematically_contested",
    "write_results",
]
