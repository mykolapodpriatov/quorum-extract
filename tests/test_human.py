"""Tests for the human-review queue and safe override apply-back."""

from __future__ import annotations

from quorum_extract import (
    EscalationStatus,
    FieldResult,
    FieldVote,
    Override,
    RecordResult,
    ReviewItem,
    ReviewQueue,
    apply_overrides,
    items_for_review,
    load_overrides,
)
from quorum_extract.human import write_override


def make_record(doc_id: str, status: EscalationStatus, value: object = None) -> RecordResult:
    fr = FieldResult(
        path="vendor",
        value=value,
        votes=[
            FieldVote(extractor="e1", raw_value="ACME", normalized_key="str:acme", missing=False),
            FieldVote(
                extractor="e2", raw_value="Globex", normalized_key="str:globex", missing=False
            ),
        ],
        agreement=0.5,
        status=status,
        winning_key="str:acme",
    )
    return RecordResult(doc_id=doc_id, fields={"vendor": fr})


def test_queue_push_and_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    q = ReviewQueue(tmp_path / "queue.jsonl")
    q.push(ReviewItem(doc_id="d1", path="vendor", candidates=["ACME", "Globex"]))
    q.push(ReviewItem(doc_id="d2", path="total", candidates=[100, 200]))
    items = q.load()
    assert len(items) == 2
    assert items[0].doc_id == "d1"
    assert items[1].candidates == [100, 200]


def test_load_missing_queue_is_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert ReviewQueue(tmp_path / "nope.jsonl").load() == []


def test_items_for_review_from_needs_review_fields() -> None:
    records = [make_record("d1", EscalationStatus.NEEDS_REVIEW)]
    items = items_for_review(records)
    assert len(items) == 1
    assert items[0].doc_id == "d1"
    assert items[0].path == "vendor"
    assert items[0].candidates == ["ACME", "Globex"]


def test_items_for_review_skips_accepted() -> None:
    records = [make_record("d1", EscalationStatus.ACCEPTED, value="ACME")]
    assert items_for_review(records) == []


def test_apply_overrides_marks_resolved_and_sets_value(tmp_path) -> None:  # type: ignore[no-untyped-def]
    records = [make_record("d1", EscalationStatus.NEEDS_REVIEW)]
    ov_path = tmp_path / "overrides.jsonl"
    write_override(ov_path, Override(doc_id="d1", path="vendor", value="ACME"))
    merged = apply_overrides(records, load_overrides(ov_path))
    fr = merged[0].fields["vendor"]
    assert fr.value == "ACME"
    assert fr.status is EscalationStatus.RESOLVED


def test_apply_overrides_does_not_mutate_originals() -> None:
    records = [make_record("d1", EscalationStatus.NEEDS_REVIEW)]
    apply_overrides(records, [Override(doc_id="d1", path="vendor", value="ACME")])
    # Original record untouched (apply-back is read-time merge, not in-place).
    assert records[0].fields["vendor"].status is EscalationStatus.NEEDS_REVIEW
    assert records[0].fields["vendor"].value is None


def test_override_last_write_wins() -> None:
    records = [make_record("d1", EscalationStatus.NEEDS_REVIEW)]
    merged = apply_overrides(
        records,
        [
            Override(doc_id="d1", path="vendor", value="First"),
            Override(doc_id="d1", path="vendor", value="Second"),
        ],
    )
    assert merged[0].fields["vendor"].value == "Second"


def test_override_for_other_doc_ignored() -> None:
    records = [make_record("d1", EscalationStatus.NEEDS_REVIEW)]
    merged = apply_overrides(records, [Override(doc_id="other", path="vendor", value="X")])
    assert merged[0].fields["vendor"].status is EscalationStatus.NEEDS_REVIEW


def test_review_item_json_roundtrip() -> None:
    item = ReviewItem(doc_id="d1", path="vendor", candidates=["A", "B"])
    assert ReviewItem.from_json(item.to_json()) == item


def test_override_json_roundtrip() -> None:
    ov = Override(doc_id="d1", path="vendor", value=42)
    assert Override.from_json(ov.to_json()) == ov
