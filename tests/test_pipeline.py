"""Tests for the high-level run_project orchestration and report rendering."""

from __future__ import annotations

import random
import warnings

from quorum_extract import (
    AgreementCalibrator,
    Document,
    EscalationStatus,
    FakeExtractor,
    Fingerprint,
    LabeledExample,
    ProjectConfig,
    QuorumConfig,
    run_project,
)
from quorum_extract.agreement import AgreementFeatures
from quorum_extract.report import render, render_markdown, render_terminal

from ._helpers import Invoice


def build_config(**quorum_kwargs: object) -> tuple[ProjectConfig, list[Document]]:
    a = FakeExtractor(
        "cheap-a",
        outputs={
            "d1": {
                "vendor": "ACME",
                "total": 100,
                "issued": "2020-01-01",
                "address": {"city": "NYC", "zip": "10001"},
                "currency": "USD",
            },
            "d2": {
                "vendor": "Globex",
                "total": 50,
                "issued": "2021-02-02",
                "address": {"city": "LA", "zip": "90001"},
            },
        },
        cost_usd=0.001,
        tier=0,
    )
    b = FakeExtractor(
        "cheap-b",
        outputs={
            "d1": {
                "vendor": "ACME",
                "total": 100,
                "issued": "2020-01-01",
                "address": {"city": "NYC", "zip": "10001"},
            },
            "d2": {
                "vendor": "Globex",
                "total": 999,
                "issued": "2021-02-02",
                "address": {"city": "LA", "zip": "90001"},
            },  # total contested
        },
        cost_usd=0.001,
        tier=0,
    )
    strong = FakeExtractor(
        "strong",
        outputs={
            "d1": {
                "vendor": "ACME",
                "total": 100,
                "issued": "2020-01-01",
                "address": {"city": "NYC", "zip": "10001"},
            },
            "d2": {
                "vendor": "Globex",
                "total": 50,
                "issued": "2021-02-02",
                "address": {"city": "LA", "zip": "90001"},
            },
        },
        cost_usd=0.05,
        tier=1,
    )
    cfg = ProjectConfig(
        schema=Invoice,
        extractors=[a.to_spec(), b.to_spec()],
        strong_extractor=strong.to_spec(),
        quorum=QuorumConfig(min_agreement=0.75, escalate_tier=1, **quorum_kwargs),  # type: ignore[arg-type]
    )
    docs = [Document("d1", {"id": "d1"}), Document("d2", {"id": "d2"})]
    return cfg, docs


def test_run_project_counts_and_budget() -> None:
    cfg, docs = build_config()
    report, records = run_project(cfg, docs)
    assert len(records) == 2
    # d2 total contested -> escalated. Each record is field-complete.
    n_leaves = len(cfg.leaf_paths())
    for rec in records:
        assert len(rec.fields) == n_leaves
    assert report.n_escalated >= 1
    assert report.budget.docs_escalated == 1
    assert report.budget.saved_usd > 0
    assert report.leaf_paths == cfg.leaf_path_strings()


def test_run_project_status_tally_consistency() -> None:
    cfg, docs = build_config()
    report, records = run_project(cfg, docs)
    total_fields = sum(len(r.fields) for r in records)
    assert (
        report.n_accepted + report.n_escalated + report.n_needs_review + report.n_resolved
        == total_fields
    )


def test_run_project_loads_calibrator_from_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Fit and persist a calibrator, point the config at it, and confirm it is
    # loaded and confidence is populated.
    rng = random.Random(0)
    rows: list[LabeledExample] = []
    for share, p in [(0.5, 0.55), (0.75, 0.8), (1.0, 0.78), (0.25, 0.3)]:
        for _ in range(40):
            rows.append(LabeledExample(AgreementFeatures(share, 2, 0.0), rng.random() < p))
    cal = AgreementCalibrator("isotonic", min_examples=50).fit(rows)
    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)

    cfg, docs = build_config(calibrator_path=str(cal_path))
    _, records = run_project(cfg, docs)
    # Confidence should be populated on accepted fields.
    accepted = [
        fr for r in records for fr in r.fields.values() if fr.status is EscalationStatus.ACCEPTED
    ]
    assert any(fr.confidence is not None for fr in accepted)


def test_run_project_fingerprint_mismatch_warns(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = random.Random(1)
    rows = [
        LabeledExample(AgreementFeatures(s, 2, 0.0), rng.random() < p)
        for s, p in [(0.5, 0.5), (0.75, 0.8), (1.0, 0.78), (0.25, 0.3)]
        for _ in range(40)
    ]
    # Persist with a deliberately wrong schema fingerprint.
    cal = AgreementCalibrator(
        "isotonic",
        fingerprint=Fingerprint(schema_hash="WRONG", extractor_set_hash="WRONG"),
        min_examples=50,
    ).fit(rows)
    cal_path = tmp_path / "cal.json"
    cal.save(cal_path)

    cfg, docs = build_config(calibrator_path=str(cal_path))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_project(cfg, docs)
    assert any("fingerprint" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# Report renderers
# --------------------------------------------------------------------------- #


def test_render_terminal_and_markdown_and_json() -> None:
    cfg, docs = build_config()
    report, records = run_project(cfg, docs)

    term = render_terminal(records, report.budget)
    assert "Budget" in term

    md = render_markdown(records, report.budget)
    assert "# Quorum extraction report" in md
    assert "Saved" in md

    js = render(records, report.budget, "json")
    assert '"records"' in js
    assert '"budget"' in js


def test_render_unknown_format_raises() -> None:
    import pytest

    cfg, docs = build_config()
    _, records = run_project(cfg, docs)
    with pytest.raises(ValueError, match="unknown report format"):
        render(records, None, "xml")


def test_render_markdown_reports_over_budget() -> None:
    cfg, docs = build_config(max_cost_usd=0.0)  # no escalation budget at all
    report, records = run_project(cfg, docs)
    md = render_markdown(records, report.budget)
    # With a zero cap, the contested doc cannot escalate -> over budget noted.
    assert "Over budget" in md or report.budget.docs_over_budget >= 1
