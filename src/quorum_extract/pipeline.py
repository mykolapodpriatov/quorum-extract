"""High-level ``run`` orchestration tying config -> cascade -> report together.

Keeps the CLI thin: :func:`run_project` takes a :class:`ProjectConfig` and a
list of documents and returns a :class:`~quorum_extract.types.RunReport` plus the
raw records, ready to persist or render.
"""

from __future__ import annotations

from collections.abc import Sequence

from .calibration import AgreementCalibrator, Fingerprint, fingerprint_for
from .cascade import Document, cascade_corpus
from .config import ProjectConfig
from .extractors import run_extractors
from .types import EscalationStatus, RecordResult, RunReport


def _load_calibrator(config: ProjectConfig) -> AgreementCalibrator | None:
    """Load the configured calibrator (if any), warning on fingerprint mismatch."""
    path = config.quorum.calibrator_path
    if not path:
        return None
    expected: Fingerprint = fingerprint_for(
        schema_paths=config.leaf_path_strings(),
        extractor_names=config.extractor_names(),
        labeled_path=None,
    )
    # We compare on schema + extractor set; labeled-set hash differs per fit and
    # is informational, so we only pass the parts we can reconstruct here.
    loaded = AgreementCalibrator.load(path)
    if (
        loaded.fingerprint.schema_hash and loaded.fingerprint.schema_hash != expected.schema_hash
    ) or (
        loaded.fingerprint.extractor_set_hash
        and loaded.fingerprint.extractor_set_hash != expected.extractor_set_hash
    ):
        import warnings

        warnings.warn(
            "loaded calibrator fingerprint does not match the current config "
            "(schema/extractor set); confidence may be invalid.",
            stacklevel=2,
        )
    return loaded


def run_project(
    config: ProjectConfig,
    documents: Sequence[Document],
    *,
    calibrator: AgreementCalibrator | None = None,
) -> tuple[RunReport, list[RecordResult]]:
    """Execute the full pipeline for a corpus.

    Args:
        config: The project configuration.
        documents: Documents to process (with stable ids).
        calibrator: Optional pre-loaded calibrator; if omitted, one is loaded
            from ``config.quorum.calibrator_path`` when set.

    Returns:
        ``(report, records)`` where ``report`` aggregates counts + budget and
        ``records`` are the per-document results (also embedded in the report).
    """
    if calibrator is None:
        calibrator = _load_calibrator(config)

    result = cascade_corpus(
        documents,
        config.leaf_paths(),
        config.extractors,
        config.quorum,
        strong_spec=config.strong_extractor,
        calibrator=calibrator,
        calibration_groups=config.calibration_groups,
        extract_fn=run_extractors,
    )

    n_accepted = n_escalated = n_review = n_resolved = 0
    for record in result.records:
        for fr in record.fields.values():
            if fr.status is EscalationStatus.ACCEPTED:
                n_accepted += 1
            elif fr.status is EscalationStatus.ESCALATED_MODEL:
                n_escalated += 1
            elif fr.status is EscalationStatus.NEEDS_REVIEW:
                n_review += 1
            elif fr.status is EscalationStatus.RESOLVED:
                n_resolved += 1

    report = RunReport(
        records=result.records,
        budget=result.budget.report(),
        leaf_paths=config.leaf_path_strings(),
        n_accepted=n_accepted,
        n_escalated=n_escalated,
        n_needs_review=n_review,
        n_resolved=n_resolved,
    )
    return report, result.records
