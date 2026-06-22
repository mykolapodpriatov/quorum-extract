"""Tests for the cost-aware cascade: batching, merge policies, budget, invariant."""

from __future__ import annotations

import random

from quorum_extract import (
    AgreementCalibrator,
    Document,
    EscalationStatus,
    FakeExtractor,
    LabeledExample,
    QuorumConfig,
    cascade_corpus,
    leaf_paths,
    run_extractors,
)
from quorum_extract.agreement import AgreementFeatures

from ._helpers import Flat


def specs_from(*fakes: FakeExtractor) -> list:  # type: ignore[type-arg]
    return [f.to_spec() for f in fakes]


def cheap_trio(
    outputs_a: dict, outputs_b: dict, outputs_c: dict, cost: float = 0.001
) -> list[FakeExtractor]:
    return [
        FakeExtractor("cheap-a", outputs=outputs_a, cost_usd=cost, tier=0),
        FakeExtractor("cheap-b", outputs=outputs_b, cost_usd=cost, tier=0),
        FakeExtractor("cheap-c", outputs=outputs_c, cost_usd=cost, tier=0),
    ]


LEAVES = leaf_paths(Flat)


# --------------------------------------------------------------------------- #
# Per-document batching: ONE strong call per contested doc, never per field.
# --------------------------------------------------------------------------- #


def test_one_strong_call_per_contested_doc() -> None:
    # doc "d1" has TWO contested fields (b and c); doc "d2" is fully accepted.
    a = FakeExtractor(
        "cheap-a",
        outputs={
            "d1": {"a": "x", "b": 1, "c": 1.0},
            "d2": {"a": "y", "b": 2, "c": 2.0},
        },
        tier=0,
    )
    b = FakeExtractor(
        "cheap-b",
        outputs={
            "d1": {"a": "x", "b": 99, "c": 99.0},  # disagrees on b and c
            "d2": {"a": "y", "b": 2, "c": 2.0},
        },
        tier=0,
    )
    strong = FakeExtractor(
        "strong",
        outputs={
            "d1": {"a": "x", "b": 1, "c": 1.0},
            "d2": {"a": "y", "b": 2, "c": 2.0},
        },
        cost_usd=0.05,
        tier=1,
    )
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1, escalation_merge="strong_wins")
    docs = [Document("d1", {"id": "d1"}), Document("d2", {"id": "d2"})]
    cascade_corpus(
        docs,
        LEAVES,
        specs_from(a, b),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    # d1 contested (2 fields) -> exactly ONE strong call; d2 accepted -> zero.
    assert strong.call_count == 1
    assert strong.called_doc_ids == ["d1"]


def test_no_strong_call_when_all_accepted() -> None:
    a = FakeExtractor("cheap-a", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    b = FakeExtractor("cheap-b", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    strong = FakeExtractor(
        "strong", outputs={"d1": {"a": "z", "b": 9, "c": 9.0}}, cost_usd=0.05, tier=1
    )
    cfg = QuorumConfig(min_agreement=0.5, escalate_tier=1)
    docs = [Document("d1", {"id": "d1"})]
    cascade_corpus(
        docs, LEAVES, specs_from(a, b), cfg, strong_spec=strong.to_spec(), extract_fn=run_extractors
    )
    assert strong.call_count == 0


# --------------------------------------------------------------------------- #
# Merge policies
# --------------------------------------------------------------------------- #


def _contested_setup(
    strong_value: dict,
) -> tuple[list[FakeExtractor], FakeExtractor, list[Document]]:
    # cheap plurality says b=1 (2 votes) vs b=2 (1 vote) -> agreement 0.667.
    cheaps = cheap_trio(
        {"d1": {"a": "x", "b": 1, "c": 1.0}},
        {"d1": {"a": "x", "b": 1, "c": 1.0}},
        {"d1": {"a": "x", "b": 2, "c": 1.0}},
    )
    strong = FakeExtractor("strong", outputs={"d1": strong_value}, cost_usd=0.05, tier=1)
    return cheaps, strong, [Document("d1", {"id": "d1"})]


def test_strong_wins_takes_strong_value() -> None:
    cheaps, strong, docs = _contested_setup({"a": "x", "b": 7, "c": 1.0})
    cfg = QuorumConfig(min_agreement=0.8, escalate_tier=1, escalation_merge="strong_wins")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.value == 7  # strong's value wins
    assert fr.status is EscalationStatus.ESCALATED_MODEL


def test_re_quorum_uses_k_plus_one_denominator() -> None:
    # cheap votes: b=1,1,2. Strong votes b=1 -> 3/4 agree => accepted at 0.75.
    cheaps, strong, docs = _contested_setup({"a": "x", "b": 1, "c": 1.0})
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1, escalation_merge="re_quorum")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    # 3 of 4 -> 0.75, meets threshold against the enlarged pool.
    assert fr.agreement == 0.75
    assert fr.status is EscalationStatus.ESCALATED_MODEL
    assert fr.value == 1


def test_re_quorum_below_threshold_needs_review() -> None:
    # cheap b=1,1,2; strong b=3 -> buckets {1:2, 2:1, 3:1} of 4 => max 0.5 < 0.75.
    cheaps, strong, docs = _contested_setup({"a": "x", "b": 3, "c": 1.0})
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1, escalation_merge="re_quorum")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.status is EscalationStatus.NEEDS_REVIEW


def test_consensus_agrees_keeps_cheap_plurality_value() -> None:
    # strong b=1 equals cheap plurality b=1 -> accept, store cheap plurality value.
    cheaps, strong, docs = _contested_setup({"a": "x", "b": 1, "c": 1.0})
    cfg = QuorumConfig(min_agreement=0.8, escalate_tier=1, escalation_merge="consensus")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.status is EscalationStatus.ESCALATED_MODEL
    assert fr.value == 1  # cheap plurality value retained


def test_consensus_missing_on_missing_is_accepted() -> None:
    # Cheap plurality says field 'b' is ABSENT (2 missing vs 1 value) at 0.667,
    # below the 0.75 threshold -> contested. The strong model ALSO returns nothing
    # for 'b'. missing == missing is a genuine agreement -> accepted as None,
    # NOT forced to needs_review.
    # Regression: _merge_consensus excluded missing==missing via `!= MISSING_KEY`.
    cheaps = [
        FakeExtractor("cheap-a", outputs={"d1": {"a": "x", "c": 1.0}}, tier=0),  # no b
        FakeExtractor("cheap-b", outputs={"d1": {"a": "x", "c": 1.0}}, tier=0),  # no b
        FakeExtractor("cheap-c", outputs={"d1": {"a": "x", "b": 5, "c": 1.0}}, tier=0),  # b=5
    ]
    strong = FakeExtractor(
        "strong", outputs={"d1": {"a": "x", "c": 1.0}}, cost_usd=0.05, tier=1
    )  # strong also omits b
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1, escalation_merge="consensus")
    docs = [Document("d1", {"id": "d1"})]
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.winning_key == "__missing__"
    assert fr.status is EscalationStatus.ESCALATED_MODEL
    assert fr.value is None


def test_consensus_disagrees_needs_review() -> None:
    # strong b=5 != cheap plurality b=1 -> needs_review.
    cheaps, strong, docs = _contested_setup({"a": "x", "b": 5, "c": 1.0})
    cfg = QuorumConfig(min_agreement=0.8, escalate_tier=1, escalation_merge="consensus")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.status is EscalationStatus.NEEDS_REVIEW


def test_strong_missing_value_needs_review() -> None:
    # Strong returns nothing for b -> missing -> needs_review under strong_wins.
    cheaps, strong, docs = _contested_setup({"a": "x", "c": 1.0})  # no b
    cfg = QuorumConfig(min_agreement=0.8, escalate_tier=1, escalation_merge="strong_wins")
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(*cheaps),
        cfg,
        strong_spec=strong.to_spec(),
        extract_fn=run_extractors,
    )
    fr = res.records[0].fields["b"]
    assert fr.status is EscalationStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# No escalation target -> needs_review
# --------------------------------------------------------------------------- #


def test_no_strong_extractor_contested_needs_review() -> None:
    cheaps, _, docs = _contested_setup({})
    cfg = QuorumConfig(min_agreement=0.8)  # no escalate_tier / strong_spec
    res = cascade_corpus(docs, LEAVES, specs_from(*cheaps), cfg, extract_fn=run_extractors)
    assert res.records[0].fields["b"].status is EscalationStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# Budget cap -> needs_review (never dropped), deterministic by doc id
# --------------------------------------------------------------------------- #


def test_budget_cap_marks_overflow_needs_review() -> None:
    # Three contested docs; budget allows exactly ONE strong call.
    a = FakeExtractor(
        "cheap-a", outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2", "d3")}, tier=0
    )
    b = FakeExtractor(
        "cheap-b", outputs={d: {"a": "x", "b": 99, "c": 1.0} for d in ("d1", "d2", "d3")}, tier=0
    )
    strong = FakeExtractor(
        "strong",
        outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2", "d3")},
        cost_usd=0.05,
        tier=1,
    )
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1, max_cost_usd=0.05)
    docs = [
        Document("d3", {"id": "d3"}),
        Document("d1", {"id": "d1"}),
        Document("d2", {"id": "d2"}),
    ]
    res = cascade_corpus(
        docs, LEAVES, specs_from(a, b), cfg, strong_spec=strong.to_spec(), extract_fn=run_extractors
    )
    # Deterministic order by doc id: d1 escalates (budget), d2/d3 over budget.
    assert strong.called_doc_ids == ["d1"]
    by_id = {r.doc_id: r for r in res.records}
    assert by_id["d1"].fields["b"].status is EscalationStatus.ESCALATED_MODEL
    assert by_id["d2"].fields["b"].status is EscalationStatus.NEEDS_REVIEW
    assert by_id["d3"].fields["b"].status is EscalationStatus.NEEDS_REVIEW
    rep = res.budget.report()
    assert rep.docs_escalated == 1
    assert rep.docs_over_budget == 2


# --------------------------------------------------------------------------- #
# $ saved
# --------------------------------------------------------------------------- #


def test_saved_usd_computed_correctly() -> None:
    # 3 docs, only 1 contested -> strong runs once; all-frontier would be 3x.
    a = FakeExtractor(
        "cheap-a",
        outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2", "d3")},
        cost_usd=0.001,
        tier=0,
    )
    b = FakeExtractor(
        "cheap-b",
        outputs={
            "d1": {"a": "x", "b": 99, "c": 1.0},  # contested
            "d2": {"a": "x", "b": 1, "c": 1.0},
            "d3": {"a": "x", "b": 1, "c": 1.0},
        },
        cost_usd=0.001,
        tier=0,
    )
    strong = FakeExtractor(
        "strong",
        outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2", "d3")},
        cost_usd=0.05,
        tier=1,
    )
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1)
    docs = [Document(d, {"id": d}) for d in ("d1", "d2", "d3")]
    res = cascade_corpus(
        docs, LEAVES, specs_from(a, b), cfg, strong_spec=strong.to_spec(), extract_fn=run_extractors
    )
    rep = res.budget.report()
    assert rep.docs_total == 3
    assert rep.docs_escalated == 1
    assert rep.all_frontier_cost_usd == 0.15  # 0.05 * 3
    assert rep.escalation_cost_usd == 0.05
    assert rep.saved_usd == 0.10  # 0.15 - 0.05


# --------------------------------------------------------------------------- #
# Field-completeness invariant for every record
# --------------------------------------------------------------------------- #


def test_field_completeness_invariant_all_records() -> None:
    a = FakeExtractor(
        "cheap-a", outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2")}, tier=0
    )
    b = FakeExtractor(
        "cheap-b", outputs={"d1": {"a": "x", "b": 2}, "d2": {"a": "x", "b": 1, "c": 1.0}}, tier=0
    )
    strong = FakeExtractor(
        "strong",
        outputs={d: {"a": "x", "b": 1, "c": 1.0} for d in ("d1", "d2")},
        cost_usd=0.05,
        tier=1,
    )
    cfg = QuorumConfig(min_agreement=0.75, escalate_tier=1)
    docs = [Document("d1", {"id": "d1"}), Document("d2", {"id": "d2"})]
    res = cascade_corpus(
        docs, LEAVES, specs_from(a, b), cfg, strong_spec=strong.to_spec(), extract_fn=run_extractors
    )
    n_leaves = len(LEAVES)
    for record in res.records:
        assert len(record.fields) == n_leaves
        assert set(record.fields) == {lp.path for lp in LEAVES}


# --------------------------------------------------------------------------- #
# Calibration confidence gate can demote an accepted field
# --------------------------------------------------------------------------- #


def test_min_confidence_gate_demotes_low_confidence_field() -> None:
    rng = random.Random(0)
    rows: list[LabeledExample] = []
    # Train so that share ~0.67 maps to LOW confidence.
    for _ in range(60):
        rows.append(LabeledExample(AgreementFeatures(0.67, 3, 0.0), rng.random() < 0.2))
    for _ in range(60):
        rows.append(LabeledExample(AgreementFeatures(1.0, 3, 0.0), rng.random() < 0.95))
    for _ in range(40):
        rows.append(LabeledExample(AgreementFeatures(0.33, 3, 0.0), rng.random() < 0.1))
    cal = AgreementCalibrator("isotonic", min_examples=50).fit(rows)

    # A field that meets quorum (0.667) but low confidence -> demoted to review
    # (no strong extractor) when min_confidence is high.
    a = FakeExtractor("cheap-a", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    b = FakeExtractor("cheap-b", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    c = FakeExtractor(
        "cheap-c", outputs={"d1": {"a": "x", "b": 2, "c": 1.0}}, tier=0
    )  # b contested
    cfg = QuorumConfig(min_agreement=0.6, min_confidence=0.5)
    docs = [Document("d1", {"id": "d1"})]
    res = cascade_corpus(
        docs, LEAVES, specs_from(a, b, c), cfg, calibrator=cal, extract_fn=run_extractors
    )
    fr = res.records[0].fields["b"]
    assert fr.confidence is not None and fr.confidence < 0.5
    assert fr.status is EscalationStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# calibration_groups: per-group calibrator is actually used (with global
# fallback for ungrouped fields).
# --------------------------------------------------------------------------- #


def _grouped_calibrator() -> AgreementCalibrator:
    """A calibrator where, at share=1.0, the global model is LOW (~0.28) but the
    'money' group model is HIGH (~1.0), so the two are unambiguously distinct."""
    rng = random.Random(1)
    rows: list[LabeledExample] = []
    # Large global=None population: high share is strongly WRONG.
    for share, p in [(0.25, 0.05), (0.5, 0.1), (0.75, 0.12), (1.0, 0.1)]:
        for _ in range(80):
            rows.append(LabeledExample(AgreementFeatures(share, 3, 0.0), rng.random() < p))
    # Group 'money': high share is strongly RIGHT (enough rows + decile span).
    for share, p in [(0.25, 0.85), (0.5, 0.9), (0.75, 0.95), (1.0, 0.98)]:
        for _ in range(20):
            rows.append(
                LabeledExample(AgreementFeatures(share, 3, 0.0), rng.random() < p, group="money")
            )
    return AgreementCalibrator("isotonic", min_examples=50).fit(rows)


def test_calibration_group_threaded_into_scoring() -> None:
    cal = _grouped_calibrator()
    assert "money" in cal.group_names
    # Both fields are unanimous (share 1.0) -> identical features; only the
    # calibration GROUP can make their confidences differ.
    a = FakeExtractor("cheap-a", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    b = FakeExtractor("cheap-b", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    cfg = QuorumConfig(min_agreement=0.5)  # no min_confidence gate
    docs = [Document("d1", {"id": "d1"})]
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(a, b),
        cfg,
        calibrator=cal,
        calibration_groups={"b": "money"},  # field 'b' -> 'money' group; a/c ungrouped
        extract_fn=run_extractors,
    )
    fields = res.records[0].fields
    conf_b = fields["b"].confidence  # money group -> HIGH
    conf_a = fields["a"].confidence  # no group -> global fallback -> LOW
    assert conf_b is not None and conf_a is not None
    # 'b' got the money group's confidence; 'a' fell back to the global model.
    assert conf_b > 0.7, conf_b
    assert conf_a < 0.4, conf_a
    assert conf_b == cal.predict_one(AgreementFeatures(1.0, 2, 0.0), group="money")
    assert conf_a == cal.predict_one(AgreementFeatures(1.0, 2, 0.0))


def test_unknown_group_falls_back_to_global() -> None:
    cal = _grouped_calibrator()
    a = FakeExtractor("cheap-a", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    b = FakeExtractor("cheap-b", outputs={"d1": {"a": "x", "b": 1, "c": 1.0}}, tier=0)
    cfg = QuorumConfig(min_agreement=0.5)
    docs = [Document("d1", {"id": "d1"})]
    res = cascade_corpus(
        docs,
        LEAVES,
        specs_from(a, b),
        cfg,
        calibrator=cal,
        # 'b' assigned to a group with NO fitted model -> global fallback.
        calibration_groups={"b": "does-not-exist"},
        extract_fn=run_extractors,
    )
    fields = res.records[0].fields
    # Both 'a' (ungrouped) and 'b' (unknown group) use the global model.
    assert fields["b"].confidence == fields["a"].confidence
    assert fields["b"].confidence is not None and fields["b"].confidence < 0.4
