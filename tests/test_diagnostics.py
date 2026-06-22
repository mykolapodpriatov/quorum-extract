"""Tests for corpus-level contention diagnostics."""

from __future__ import annotations

from quorum_extract import (
    EscalationStatus,
    FieldResult,
    RecordResult,
    field_contention,
    systematically_contested,
)


def record(
    doc_id: str, statuses: dict[str, EscalationStatus], agreements: dict[str, float]
) -> RecordResult:
    fields = {
        path: FieldResult(path=path, agreement=agreements[path], status=status, winning_key="k")
        for path, status in statuses.items()
    }
    return RecordResult(doc_id=doc_id, fields=fields)


def test_contention_rate_per_field() -> None:
    # "vendor" contested in 2 of 3 records; "total" never contested.
    records = [
        record(
            "d1",
            {"vendor": EscalationStatus.NEEDS_REVIEW, "total": EscalationStatus.ACCEPTED},
            {"vendor": 0.4, "total": 1.0},
        ),
        record(
            "d2",
            {"vendor": EscalationStatus.NEEDS_REVIEW, "total": EscalationStatus.ACCEPTED},
            {"vendor": 0.5, "total": 1.0},
        ),
        record(
            "d3",
            {"vendor": EscalationStatus.ACCEPTED, "total": EscalationStatus.ACCEPTED},
            {"vendor": 1.0, "total": 1.0},
        ),
    ]
    diags = {d.path: d for d in field_contention(records)}
    assert diags["vendor"].contention_rate == 2 / 3
    assert diags["vendor"].n_contested == 2
    assert diags["total"].contention_rate == 0.0


def test_diagnostics_sorted_by_contention_desc() -> None:
    records = [
        record(
            "d1",
            {"a": EscalationStatus.ACCEPTED, "b": EscalationStatus.NEEDS_REVIEW},
            {"a": 1.0, "b": 0.3},
        ),
    ]
    diags = field_contention(records)
    # b (rate 1.0) before a (rate 0.0).
    assert [d.path for d in diags] == ["b", "a"]


def test_systematically_contested_threshold() -> None:
    records = [
        record(
            "d1",
            {"a": EscalationStatus.NEEDS_REVIEW, "b": EscalationStatus.ACCEPTED},
            {"a": 0.3, "b": 1.0},
        ),
        record(
            "d2",
            {"a": EscalationStatus.NEEDS_REVIEW, "b": EscalationStatus.ACCEPTED},
            {"a": 0.3, "b": 1.0},
        ),
    ]
    assert systematically_contested(records, threshold=0.5) == ["a"]
    assert systematically_contested(records, threshold=1.1) == []


def test_mean_agreement_reported() -> None:
    records = [
        record("d1", {"a": EscalationStatus.NEEDS_REVIEW}, {"a": 0.4}),
        record("d2", {"a": EscalationStatus.ACCEPTED}, {"a": 0.8}),
    ]
    d = field_contention(records)[0]
    assert abs(d.mean_agreement - 0.6) < 1e-9


def test_top_disagreement_keys_tracked() -> None:
    records = [
        record("d1", {"a": EscalationStatus.NEEDS_REVIEW}, {"a": 0.4}),
        record("d2", {"a": EscalationStatus.NEEDS_REVIEW}, {"a": 0.4}),
    ]
    d = field_contention(records)[0]
    assert d.top_disagreement_keys == [("k", 2)]


def test_empty_corpus() -> None:
    assert field_contention([]) == []
