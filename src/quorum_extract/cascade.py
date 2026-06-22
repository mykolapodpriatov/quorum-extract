"""Cost-aware cascade: escalate contested fields per document, then merge.

Escalation is **per-document** to match real billing (plan 3.5): a document with
*any* contested field triggers exactly **one** higher-tier invocation; we then
read only the contested fields' values from it. A doc with five contested fields
costs one strong call, not five. Fully-accepted docs are never escalated.

Three typed merge policies decide how the strong model's result combines with
the cheap votes:

* ``strong_wins`` (default) -- take the strong model's value for each contested
  field.
* ``re_quorum`` -- add the strong vote to the cheap votes and re-run quorum with
  denominator ``K + 1``.
* ``consensus`` -- accept only if the strong model's normalized value equals the
  cheap plurality's; on agreement the **stored value is the cheap plurality's**
  normalized value, else ``needs_review``.

A ``max_cost_usd`` cap stops escalation deterministically (docs ordered by id);
every still-contested field past the cap becomes ``needs_review`` -- never
dropped. The **field-completeness invariant** holds: every leaf path appears in
every record after the cascade.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from .agreement import features_for
from .budget import BudgetTracker
from .calibration import AgreementCalibrator
from .normalize import MISSING_KEY
from .quorum import (
    ExtractorOutput,
    _build_list_indices,
    quorum_field,
    quorum_record,
)
from .schema import LeafPath
from .types import (
    DocInput,
    EscalationStatus,
    ExtractorSpec,
    FieldResult,
    FieldVote,
    QuorumConfig,
    RecordResult,
)

# Type of the per-document extraction step (cheap extractors -> outputs).
ExtractFn = Callable[[Sequence[ExtractorSpec], DocInput], list[ExtractorOutput]]


@dataclass(frozen=True, slots=True)
class Document:
    """A document plus a stable id used for ordering and reporting."""

    doc_id: str
    payload: DocInput


def _apply_calibration(
    result: FieldResult,
    calibrator: AgreementCalibrator | None,
    config: QuorumConfig,
    group: str | None = None,
) -> FieldResult:
    """Attach calibrated confidence and apply the ``min_confidence`` gate.

    A field that met quorum but falls below ``min_confidence`` is *demoted* to
    contested (so the cascade can escalate it). Calibration never *promotes* a
    field that failed quorum.

    ``group`` selects the per-group calibrator for this field path (from
    ``ProjectConfig.calibration_groups``). The calibrator falls back to the
    global model when the group is ``None`` or its group model is absent /
    under-trained.
    """
    if calibrator is None or not calibrator.is_fitted:
        return result
    conf = calibrator.predict_one(features_for(result), group)
    new_status = result.status
    if (
        result.status is EscalationStatus.ACCEPTED
        and config.min_confidence is not None
        and conf < config.min_confidence
    ):
        new_status = EscalationStatus.NEEDS_REVIEW
    return result.model_copy(update={"confidence": conf, "status": new_status})


def _strong_output_for(
    doc: Document, strong_spec: ExtractorSpec, extract_fn: ExtractFn
) -> ExtractorOutput:
    """Run the single strong invocation for one contested document."""
    outputs = extract_fn([strong_spec], doc.payload)
    return outputs[0]


def _merge_strong_wins(
    leaf: LeafPath,
    base: FieldResult,
    strong_output: ExtractorOutput,
    config: QuorumConfig,
) -> FieldResult:
    """Take the strong model's value for the contested field."""
    from .quorum import collect_votes

    indices = _build_list_indices([strong_output], [leaf])
    strong_votes = collect_votes(leaf, [strong_output], indices)
    sv = strong_votes[0]
    status = (
        EscalationStatus.NEEDS_REVIEW
        if sv.normalized_key == MISSING_KEY
        else EscalationStatus.ESCALATED_MODEL
    )
    return base.model_copy(
        update={
            "value": sv.raw_value,
            "status": status,
            "winning_key": sv.normalized_key,
            "votes": [*base.votes, sv],
        }
    )


def _merge_re_quorum(
    leaf: LeafPath,
    base: FieldResult,
    cheap_outputs: Sequence[ExtractorOutput],
    strong_output: ExtractorOutput,
    config: QuorumConfig,
) -> FieldResult:
    """Add the strong vote to the cheap pool and re-run quorum (denominator K+1)."""
    combined = [*cheap_outputs, strong_output]
    indices = _build_list_indices(combined, [leaf])
    re = quorum_field(leaf, combined, config, indices)
    # Quorum decided acceptance against the enlarged pool; mark the source.
    if re.status is EscalationStatus.ACCEPTED:
        status = EscalationStatus.ESCALATED_MODEL
        value = re.value
    else:
        status = EscalationStatus.NEEDS_REVIEW
        value = re.value
    return re.model_copy(update={"status": status, "value": value})


def _merge_consensus(
    leaf: LeafPath,
    base: FieldResult,
    strong_output: ExtractorOutput,
    config: QuorumConfig,
) -> FieldResult:
    """Accept iff the strong value matches the cheap plurality's normalized key.

    On agreement the stored value remains the cheap plurality's value (``base``);
    otherwise the field needs human review. A ``missing == missing`` agreement
    (the strong model AND the cheap plurality both find the field absent) is a
    genuine agreement: it is accepted with value ``None``, not forced to review.
    """
    from .quorum import collect_votes

    indices = _build_list_indices([strong_output], [leaf])
    sv = collect_votes(leaf, [strong_output], indices)[0]
    cheap_key = base.winning_key
    if sv.normalized_key == cheap_key:
        # Strong and cheap plurality agree -- including missing == missing, where
        # base.value is already None. Accept.
        status = EscalationStatus.ESCALATED_MODEL
        value = base.value
    else:
        status = EscalationStatus.NEEDS_REVIEW
        value = base.value
    return base.model_copy(update={"status": status, "value": value, "votes": [*base.votes, sv]})


def _merge_field(
    leaf: LeafPath,
    base: FieldResult,
    cheap_outputs: Sequence[ExtractorOutput],
    strong_output: ExtractorOutput,
    config: QuorumConfig,
) -> FieldResult:
    """Dispatch to the configured ``escalation_merge`` policy."""
    policy = config.escalation_merge
    if policy == "strong_wins":
        return _merge_strong_wins(leaf, base, strong_output, config)
    if policy == "re_quorum":
        return _merge_re_quorum(leaf, base, cheap_outputs, strong_output, config)
    if policy == "consensus":
        return _merge_consensus(leaf, base, strong_output, config)
    raise ValueError(f"unknown escalation_merge policy: {policy!r}")  # pragma: no cover


@dataclass
class CascadeResult:
    """Result of cascading one corpus: records plus the budget tracker used."""

    records: list[RecordResult]
    budget: BudgetTracker


def cascade_corpus(
    docs: Sequence[Document],
    leaves: Sequence[LeafPath],
    cheap_specs: Sequence[ExtractorSpec],
    config: QuorumConfig,
    *,
    strong_spec: ExtractorSpec | None = None,
    calibrator: AgreementCalibrator | None = None,
    calibration_groups: Mapping[str, str] | None = None,
    extract_fn: ExtractFn,
) -> CascadeResult:
    """Run the full cheap-vote -> calibrate -> escalate pipeline over a corpus.

    Args:
        docs: Documents (with stable ids). Processed in a deterministic order
            (sorted by ``doc_id``) so the budget cap point is reproducible.
        leaves: The schema's leaf paths (the complete field set).
        cheap_specs: The cheap extractor ensemble (tier 0).
        config: Quorum/calibration/escalation policy.
        strong_spec: The escalation extractor. If ``None`` (or no
            ``escalate_tier``), contested fields go straight to ``needs_review``.
        calibrator: Optional fitted calibrator for confidence + gating.
        calibration_groups: Optional map of leaf path -> calibration group name.
            Each field is scored with its group's calibrator, falling back to the
            global model when the path has no group or the group model is
            absent / under-trained.
        extract_fn: Callable that invokes specs against a document and returns
            outputs (failures captured as ``ok=False``). Injected for testability.

    Returns:
        A :class:`CascadeResult`. Every record satisfies the field-completeness
        invariant (one :class:`FieldResult` per leaf path).
    """
    strong_cost = strong_spec.cost_usd if strong_spec is not None else 0.0
    tracker = BudgetTracker(strong_cost_usd=strong_cost, max_cost_usd=config.max_cost_usd)
    groups: Mapping[str, str] = calibration_groups or {}

    ordered = sorted(docs, key=lambda d: d.doc_id)
    records: list[RecordResult] = []

    for doc in ordered:
        cheap_outputs = extract_fn(cheap_specs, doc.payload)
        cheap_cost = sum(o.spec.cost_usd for o in cheap_outputs)
        tracker.charge_cheap(doc.doc_id, cheap_cost)

        field_results = quorum_record(leaves, cheap_outputs, config)
        # Apply calibration + confidence gate (may demote accepted -> review).
        # Each field path is scored with its calibration group (global fallback).
        field_results = {
            path: _apply_calibration(fr, calibrator, config, groups.get(path))
            for path, fr in field_results.items()
        }

        contested = sorted(
            path for path, fr in field_results.items() if fr.status is not EscalationStatus.ACCEPTED
        )

        if contested and strong_spec is not None and config.escalate_tier is not None:
            field_results, doc_cost = _escalate_document(
                doc=doc,
                leaves=leaves,
                contested=contested,
                field_results=field_results,
                cheap_outputs=cheap_outputs,
                strong_spec=strong_spec,
                config=config,
                tracker=tracker,
                extract_fn=extract_fn,
            )
        else:
            doc_cost = cheap_cost

        record = RecordResult(
            doc_id=doc.doc_id,
            fields=field_results,
            cost_usd=round(doc_cost, 10),
        )
        _assert_complete(record, leaves)
        records.append(record)

    return CascadeResult(records=records, budget=tracker)


def _escalate_document(
    *,
    doc: Document,
    leaves: Sequence[LeafPath],
    contested: Sequence[str],
    field_results: dict[str, FieldResult],
    cheap_outputs: Sequence[ExtractorOutput],
    strong_spec: ExtractorSpec,
    config: QuorumConfig,
    tracker: BudgetTracker,
    extract_fn: ExtractFn,
) -> tuple[dict[str, FieldResult], float]:
    """Escalate one contested document with a single strong invocation.

    Honors the budget cap: if charging the strong call would exceed
    ``max_cost_usd``, no call is made and all contested fields become
    ``needs_review``. Returns the updated field map and the doc's total cost.
    """
    cheap_cost = sum(o.spec.cost_usd for o in cheap_outputs)

    if not tracker.charge_escalation():
        # Over budget: mark remaining contested fields needs_review (never drop).
        for path in contested:
            field_results[path] = field_results[path].model_copy(
                update={"status": EscalationStatus.NEEDS_REVIEW}
            )
        return field_results, cheap_cost

    # Exactly ONE strong invocation, regardless of how many fields are contested.
    strong_output = _strong_output_for(doc, strong_spec, extract_fn)
    leaf_by_path = {lp.path: lp for lp in leaves}
    for path in contested:
        leaf = leaf_by_path[path]
        field_results[path] = _merge_field(
            leaf, field_results[path], cheap_outputs, strong_output, config
        )
    return field_results, cheap_cost + strong_spec.cost_usd


def _assert_complete(record: RecordResult, leaves: Sequence[LeafPath]) -> None:
    """Enforce the field-completeness invariant for one record.

    Raises:
        AssertionError: if any leaf path is missing from the record (which would
            mean a field was silently dropped -- forbidden by design).
    """
    expected = {lp.path for lp in leaves}
    got = set(record.fields)
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise AssertionError(
            "field-completeness invariant violated for "
            f"doc {record.doc_id!r}: missing={sorted(missing)} extra={sorted(extra)}"
        )


def find_contested(record: RecordResult) -> list[FieldVote]:
    """Convenience for diagnostics: the votes of every contested field."""
    out: list[FieldVote] = []
    for path in record.contested_paths():
        out.extend(record.fields[path].votes)
    return out
