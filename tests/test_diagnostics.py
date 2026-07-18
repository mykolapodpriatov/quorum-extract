"""Tests for corpus-level contention diagnostics."""

from __future__ import annotations

import csv
import json
from io import StringIO

from quorum_extract import (
    EscalationStatus,
    FieldResult,
    RecordResult,
    field_contention,
    suggest_labels,
    systematically_contested,
)
from quorum_extract.report import (
    _DIAGNOSTIC_COLUMNS,
    render_diagnostics,
    render_diagnostics_csv,
    render_diagnostics_json,
    render_diagnostics_md,
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


# --------------------------------------------------------------------------- #
# Dashboard export (diagnose): CSV / Markdown / JSON
# --------------------------------------------------------------------------- #


def _dashboard_corpus() -> list[RecordResult]:
    # "vendor" contested 2/2 (rate 1.0); "total" contested 0/2 (rate 0.0).
    return [
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
    ]


def test_csv_stable_column_order_and_row_order() -> None:
    rows = list(csv.reader(StringIO(render_diagnostics_csv(_dashboard_corpus()))))
    # Header column order is stable and matches the module constant.
    assert tuple(rows[0]) == _DIAGNOSTIC_COLUMNS
    # Rows are ordered by descending contention: vendor (1.0) before total (0.0).
    assert [r[0] for r in rows[1:]] == ["vendor", "total"]


def test_csv_threshold_flags_systematically_contested() -> None:
    corpus = _dashboard_corpus()
    rows = {r[0]: r for r in csv.reader(StringIO(render_diagnostics_csv(corpus, threshold=0.5)))}
    flag_col = _DIAGNOSTIC_COLUMNS.index("systematically_contested")
    # vendor is contested in every record (rate 1.0 >= 0.5); total is never.
    assert rows["vendor"][flag_col] == "true"
    assert rows["total"][flag_col] == "false"
    # Raising the threshold above 1.0 flags nothing.
    high = {r[0]: r for r in csv.reader(StringIO(render_diagnostics_csv(corpus, threshold=1.1)))}
    assert high["vendor"][flag_col] == "false"


def test_csv_top_disagreement_keys_cell() -> None:
    corpus = _dashboard_corpus()
    rows = {r[0]: r for r in csv.reader(StringIO(render_diagnostics_csv(corpus)))}
    keys_col = _DIAGNOSTIC_COLUMNS.index("top_disagreement_keys")
    # Both contested vendor records share winning_key "k".
    assert rows["vendor"][keys_col] == "k (2)"
    # An accepted-only field surfaces no disagreement buckets.
    assert rows["total"][keys_col] == ""


def test_empty_results_yield_header_only_csv() -> None:
    rows = list(csv.reader(StringIO(render_diagnostics_csv([]))))
    assert rows == [list(_DIAGNOSTIC_COLUMNS)]


def test_diagnostics_md_lists_systematic_fields() -> None:
    md = render_diagnostics_md(_dashboard_corpus(), threshold=0.5)
    assert md.startswith("# Per-field reliability dashboard")
    assert "top disagreement keys" in md
    assert "`vendor`" in md
    # vendor row is flagged systematic ("yes"); total is not.
    vendor_line = next(line for line in md.splitlines() if line.startswith("| `vendor`"))
    assert "| yes |" in vendor_line


def test_diagnostics_md_empty_reports_none() -> None:
    assert "_none_" in render_diagnostics_md([])


def test_diagnostics_json_schema() -> None:
    payload = json.loads(render_diagnostics_json(_dashboard_corpus(), threshold=0.5))
    assert payload["threshold"] == 0.5
    assert payload["systematically_contested"] == ["vendor"]
    vendor = next(f for f in payload["fields"] if f["path"] == "vendor")
    assert vendor["systematically_contested"] is True
    assert vendor["top_disagreement_keys"] == [["k", 2]]
    # Deterministic descending-contention field order.
    assert [f["path"] for f in payload["fields"]] == ["vendor", "total"]


def test_suggest_labels_ranks_nearest_boundary_first() -> None:
    records = [
        record(
            "d1",
            {
                "near": EscalationStatus.NEEDS_REVIEW,  # agreement 0.5 -> distance 0.0
                "far": EscalationStatus.ACCEPTED,  # agreement 1.0 -> distance 0.5
            },
            {"near": 0.5, "far": 1.0},
        ),
        record(
            "d2",
            {"mid": EscalationStatus.NEEDS_REVIEW},  # agreement 0.7 -> distance 0.2
            {"mid": 0.7},
        ),
    ]
    ranked = suggest_labels(records, boundary=0.5)
    assert [(s.doc_id, s.path) for s in ranked] == [
        ("d1", "near"),
        ("d2", "mid"),
        ("d1", "far"),
    ]
    assert ranked[0].agreement == 0.5


def test_suggest_labels_contested_breaks_ties() -> None:
    # Two fields equidistant from the boundary; the contested one ranks first.
    records = [
        record(
            "d1",
            {
                "accepted": EscalationStatus.ACCEPTED,
                "contested": EscalationStatus.NEEDS_REVIEW,
            },
            {"accepted": 0.6, "contested": 0.6},
        ),
    ]
    ranked = suggest_labels(records, boundary=0.5)
    assert [s.path for s in ranked] == ["contested", "accepted"]


def test_suggest_labels_stable_tiebreak_and_limit() -> None:
    # Same distance and status across docs -> deterministic (doc_id, path) order.
    records = [
        record("d2", {"x": EscalationStatus.NEEDS_REVIEW}, {"x": 0.5}),
        record("d1", {"x": EscalationStatus.NEEDS_REVIEW}, {"x": 0.5}),
    ]
    ranked = suggest_labels(records, n=1)
    assert [(s.doc_id, s.path) for s in ranked] == [("d1", "x")]


def test_suggest_labels_empty_and_nonpositive_n() -> None:
    assert suggest_labels([]) == []
    one = [record("d1", {"x": EscalationStatus.NEEDS_REVIEW}, {"x": 0.5})]
    assert suggest_labels(one, n=0) == []
    assert suggest_labels(one, n=-3) == []


def test_render_diagnostics_dispatch_and_unknown_format() -> None:
    corpus = _dashboard_corpus()
    assert render_diagnostics(corpus, "csv") == render_diagnostics_csv(corpus)
    assert render_diagnostics(corpus, "md") == render_diagnostics_md(corpus)
    assert render_diagnostics(corpus, "json") == render_diagnostics_json(corpus)
    try:
        render_diagnostics(corpus, "bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown format")
