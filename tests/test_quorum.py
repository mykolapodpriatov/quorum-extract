"""Tests for per-field quorum: thresholds, tie-breaks, failed votes, alignment."""

from __future__ import annotations

from pydantic import BaseModel

from quorum_extract import (
    MISSING_KEY,
    EscalationStatus,
    QuorumConfig,
    leaf_paths,
    quorum_field,
    quorum_record,
)
from quorum_extract.quorum import collect_votes, tally
from quorum_extract.schema import LeafKind, LeafPath

from ._helpers import Flat, Invoice, make_outputs


def flat_leaf(path: str, annotation: object) -> LeafPath:
    return LeafPath(path=path, kind=LeafKind.SCALAR, annotation=annotation)


# --------------------------------------------------------------------------- #
# Acceptance thresholds
# --------------------------------------------------------------------------- #


def test_unanimous_field_accepted() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "x"}), ("e3", {"a": "x"})])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    assert fr.value == "x"


def test_acceptance_at_threshold() -> None:
    leaf = flat_leaf("a", str)
    # 2 of 3 agree -> agreement 0.667.
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "x"}), ("e3", {"a": "y"})])
    accepted = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.66))
    assert accepted.status is EscalationStatus.ACCEPTED
    rejected = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.67))
    assert rejected.status is EscalationStatus.NEEDS_REVIEW


def test_fully_split_field_low_agreement() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "y"}), ("e3", {"a": "z"})])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1 / 3
    assert fr.status is EscalationStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# Deterministic tie-break
# --------------------------------------------------------------------------- #


def test_tie_break_prefers_lowest_tier() -> None:
    leaf = flat_leaf("a", str)
    # Two buckets, each one vote; e1 is tier 0, e2 is tier 1 -> e1 wins.
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "y"})], tiers=[0, 1])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.0))
    assert fr.value == "x"
    assert fr.winning_key is not None


def test_tie_break_is_deterministic_repeated() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "y"})], tiers=[1, 0])
    # e2 is tier 0 -> should win every time.
    for _ in range(5):
        fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.0))
        assert fr.value == "y"


def test_tie_break_prefers_earliest_at_equal_tier() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "y"})], tiers=[0, 0])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.0))
    assert fr.value == "x"


# --------------------------------------------------------------------------- #
# Failed extractor = missing vote with K preserved
# --------------------------------------------------------------------------- #


def test_failed_extractor_is_missing_vote_k_preserved() -> None:
    leaf = flat_leaf("a", str)
    # e3 failed (ok=False). K must remain 3, not drop to 2.
    outputs = make_outputs(
        [("e1", {"a": "x"}), ("e2", {"a": "x"}), ("e3", {"a": "x"})],
        ok=[True, True, False],
    )
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert len(fr.votes) == 3  # K preserved
    # 2 real "x" votes + 1 missing vote -> agreement 2/3.
    assert fr.agreement == 2 / 3
    missing_votes = [v for v in fr.votes if v.missing]
    assert len(missing_votes) == 1
    assert missing_votes[0].extractor == "e3"
    assert missing_votes[0].normalized_key == MISSING_KEY


def test_all_failed_yields_missing_winner() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "x"})], ok=[False, False])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.winning_key == MISSING_KEY
    assert fr.value is None
    assert fr.agreement == 1.0  # both agree it is missing


def test_unanimous_missing_is_accepted_as_none() -> None:
    leaf = flat_leaf("note", str)
    outputs = make_outputs([("e1", {}), ("e2", {}), ("e3", {})])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    # Everyone agrees the field is absent -> accepted, value None.
    assert fr.status is EscalationStatus.ACCEPTED
    assert fr.value is None


# --------------------------------------------------------------------------- #
# List alignment by key
# --------------------------------------------------------------------------- #


def test_keyed_list_agreement_when_rows_match() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 5}]}
    rows_b = {"line_items": [{"sku": "B", "qty": 5}, {"sku": "A", "qty": 2}]}  # reordered
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    # Aligned by sku, reordering does not matter -> full agreement.
    assert fr.agreement == 1.0


def test_list_length_mismatch_is_full_disagreement() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"sku": "A", "qty": 2}]}
    rows_b = {"line_items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 5}]}
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    # Key sets differ -> full disagreement, no spurious partial match.
    assert fr.agreement == 0.5  # winning bucket size 1 of 2, both unique buckets
    keys = {v.normalized_key for v in fr.votes}
    assert len(keys) == 2  # two distinct disagreement buckets


def test_duplicate_key_is_full_disagreement() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"sku": "A", "qty": 2}, {"sku": "A", "qty": 9}]}  # dup A
    rows_b = {"line_items": [{"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    # e1 has a duplicate key -> its list is a full-disagreement bucket.
    assert fr.status is EscalationStatus.NEEDS_REVIEW
    assert fr.agreement < 1.0


def test_missing_key_on_row_is_full_disagreement() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"qty": 2}]}  # missing sku
    rows_b = {"line_items": [{"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    assert fr.status is EscalationStatus.NEEDS_REVIEW


def test_keyed_list_value_disagreement_on_matched_rows() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"sku": "A", "qty": 2}]}
    rows_b = {"line_items": [{"sku": "A", "qty": 99}]}  # same key, different qty
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    # Rows align but values differ -> disagreement.
    assert fr.agreement == 0.5
    assert fr.status is EscalationStatus.NEEDS_REVIEW


# --------------------------------------------------------------------------- #
# Record-level
# --------------------------------------------------------------------------- #


def test_quorum_record_is_field_complete() -> None:
    leaves = leaf_paths(Flat)
    outputs = make_outputs(
        [("e1", {"a": "x", "b": 1, "c": 1.0}), ("e2", {"a": "x", "b": 1, "c": 1.0})]
    )
    results = quorum_record(leaves, outputs, QuorumConfig())
    assert set(results) == {lp.path for lp in leaves}


def test_tally_counts_and_k() -> None:
    leaf = flat_leaf("a", str)
    outputs = make_outputs([("e1", {"a": "x"}), ("e2", {"a": "x"}), ("e3", {"a": "y"})])
    votes = collect_votes(leaf, outputs)
    _key, count, k = tally(votes, outputs)
    assert count == 2
    assert k == 3


# --------------------------------------------------------------------------- #
# List alignment: failed extractor, non-sequence value, lazy index build.
# --------------------------------------------------------------------------- #


def test_failed_extractor_in_keyed_list_is_missing_vote() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows = {"line_items": [{"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", rows), ("e2", rows)], ok=[True, False])
    votes = collect_votes(leaf, outputs)
    assert len(votes) == 2  # K preserved
    assert votes[1].missing is True


def test_keyed_list_non_sequence_value_is_disagreement() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows_a = {"line_items": [{"sku": "A", "qty": 2}]}
    rows_b = {"line_items": "not-a-list"}  # non-sequence -> malformed
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    assert fr.status is EscalationStatus.NEEDS_REVIEW


def test_collect_votes_builds_indices_when_absent() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows = {"line_items": [{"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", rows), ("e2", rows)])
    # No precomputed list_indices passed -> collect_votes builds them.
    votes = collect_votes(leaf, outputs)
    assert len({v.normalized_key for v in votes}) == 1  # both agree


def test_all_malformed_lists_full_disagreement() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    # Both extractors have duplicate keys -> both malformed -> all disagree.
    rows = {"line_items": [{"sku": "A", "qty": 1}, {"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", rows), ("e2", rows)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    assert fr.status is EscalationStatus.NEEDS_REVIEW
    keys = {v.normalized_key for v in fr.votes}
    assert len(keys) == 2  # two distinct disagreement buckets


def test_empty_keyed_lists_agree_as_missing() -> None:
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    # Both have empty line_items -> no rows -> unified ``missing`` bucket. An
    # empty object-list means "no rows", the same as an absent/None list, so it
    # lands in MISSING_KEY (not a concrete ``rows:{}`` bucket) and is accepted.
    rows: dict[str, list] = {"line_items": []}
    outputs = make_outputs([("e1", rows), ("e2", rows)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    assert fr.winning_key == MISSING_KEY
    assert all(v.missing for v in fr.votes)


def test_empty_keyed_list_unifies_with_absent_and_none() -> None:
    # An object-list field's "no rows" forms must all unify: an extractor
    # returning ``[]`` agrees (unanimous missing, agreement 1.0) with one
    # returning an absent list and one returning ``None`` -- all map to the
    # single ``missing`` bucket. Regression: an empty ``[]`` previously became a
    # concrete ``rows:{}`` bucket (missing=False) and scored 1/3, not 1.0.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    outputs = make_outputs(
        [
            ("e1", {"line_items": []}),  # empty list
            ("e2", {}),  # absent key
            ("e3", {"line_items": None}),  # explicit None
        ]
    )
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.winning_key == MISSING_KEY
    assert all(v.missing for v in fr.votes)
    assert {v.normalized_key for v in fr.votes} == {MISSING_KEY}
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    assert fr.value is None


def test_empty_keyed_list_distinct_from_malformed_list() -> None:
    # The "no rows" unification must NOT swallow a genuinely *malformed* list: an
    # empty ``[]`` (missing) and a duplicate-key list (unique ``__disagree__``)
    # are two distinct buckets -> agreement 0.5, not a spurious match.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    dup = {"line_items": [{"sku": "A", "qty": 1}, {"sku": "A", "qty": 2}]}  # dup key
    outputs = make_outputs([("e1", {"line_items": []}), ("e2", dup)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    assert fr.agreement == 0.5
    keys = {v.normalized_key for v in fr.votes}
    assert MISSING_KEY in keys
    assert any(k.startswith("__disagree__") for k in keys)
    assert len(keys) == 2  # empty (missing) and malformed (disagree) stay apart


def test_absent_list_field_unifies_as_missing() -> None:
    # Two ok extractors that BOTH omit the object-list key entirely must agree it
    # is absent (unified missing bucket), not each get a unique disagreement key.
    # Regression: previously an absent list -> per-extractor `__disagree__:<name>`
    # so two unanimous omitters scored agreement 0.5 instead of 1.0.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    outputs = make_outputs([("e1", {}), ("e2", {})])  # no line_items key at all
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.winning_key == MISSING_KEY
    assert all(v.missing for v in fr.votes)
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    assert fr.value is None


def test_none_list_field_unifies_as_missing() -> None:
    # Same unification when the list key is present but explicitly None.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    outputs = make_outputs([("e1", {"line_items": None}), ("e2", {"line_items": None})])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.winning_key == MISSING_KEY
    assert all(v.missing for v in fr.votes)
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED


def test_absent_list_field_distinct_from_malformed_list() -> None:
    # An absent list (missing) must NOT coalesce with a *malformed* list
    # (duplicate key -> unique disagreement). One missing + one disagreement =>
    # two distinct buckets, agreement 0.5.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    dup = {"line_items": [{"sku": "A", "qty": 1}, {"sku": "A", "qty": 2}]}
    outputs = make_outputs([("e1", {}), ("e2", dup)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.6))
    assert fr.agreement == 0.5
    keys = {v.normalized_key for v in fr.votes}
    assert MISSING_KEY in keys
    assert len(keys) == 2


def test_keyed_row_value_ending_in_missing_suffix_not_dropped() -> None:
    # A real sub-value whose canonical form merely *ends* with "__missing__"
    # (e.g. the literal "hello__missing__") must NOT be treated as absent.
    # Regression: an endswith(MISSING_KEY) suffix check on the composite
    # "'sku'=str:hello__missing__" part string mis-classified it as missing and
    # silently dropped the field to None.
    leaves = {lp.path: lp for lp in leaf_paths(Invoice, list_key={"line_items": "sku"})}
    leaf = leaves["line_items[*].qty"]
    rows = {"line_items": [{"sku": "A", "qty": "hello__missing__"}]}
    outputs = make_outputs([("e1", rows), ("e2", rows)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    # The bucket is a real value bucket, NOT the unified missing bucket.
    assert fr.winning_key != MISSING_KEY
    assert all(not v.missing for v in fr.votes)


def test_keyed_row_nested_dict_subfield_agrees_on_key_order() -> None:
    # A keyed-row sub-field that is itself a dict must be canonicalized
    # structurally, so two payloads differing only in key order AGREE.
    # Regression: the OBJECT_LIST_FIELD path used scalar repr canonicalization,
    # which is key-order sensitive and under-counted agreement.
    class Row(BaseModel):
        sku: str
        meta: dict[str, int] = {}

    class Doc3(BaseModel):
        items: list[Row] = []

    leaves = {lp.path: lp for lp in leaf_paths(Doc3, list_key={"items": "sku"})}
    leaf = leaves["items[*].meta"]
    rows_a = {"items": [{"sku": "A", "meta": {"x": 1, "y": 2}}]}
    rows_b = {"items": [{"sku": "A", "meta": {"y": 2, "x": 1}}]}  # reordered keys
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1.0
    assert fr.status is EscalationStatus.ACCEPTED
    assert len({v.normalized_key for v in fr.votes}) == 1


def test_keyed_list_integer_keys() -> None:
    class Item(BaseModel):
        idx: int
        val: str

    class Doc2(BaseModel):
        items: list[Item] = []

    leaves = {lp.path: lp for lp in leaf_paths(Doc2, list_key={"items": "idx"})}
    leaf = leaves["items[*].val"]
    rows_a = {"items": [{"idx": 1, "val": "a"}, {"idx": 2, "val": "b"}]}
    rows_b = {"items": [{"idx": 2, "val": "b"}, {"idx": 1, "val": "a"}]}
    outputs = make_outputs([("e1", rows_a), ("e2", rows_b)])
    fr = quorum_field(leaf, outputs, QuorumConfig(min_agreement=0.5))
    assert fr.agreement == 1.0
