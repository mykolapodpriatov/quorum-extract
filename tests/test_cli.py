"""End-to-end CLI tests (offline, FakeExtractors): run -> report -> calibrate -> review."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from quorum_extract.cli import app

runner = CliRunner()


PROJECT_SRC = """
from pydantic import BaseModel
from quorum_extract import ProjectConfig, QuorumConfig, FakeExtractor

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

def load_docs(path):
    # Two docs: d1 has a contested currency, d2 is unanimous.
    yield ("d1", {"id": "d1"})
    yield ("d2", {"id": "d2"})

cheap_a = FakeExtractor("cheap-a", outputs={
    "d1": {"vendor": "ACME", "total": "100.0", "currency": "USD"},
    "d2": {"vendor": "Globex", "total": 50, "currency": "EUR"},
}, cost_usd=0.001, tier=0)
cheap_b = FakeExtractor("cheap-b", outputs={
    "d1": {"vendor": "acme", "total": 100, "currency": "EUR"},
    "d2": {"vendor": "globex", "total": 50.0, "currency": "EUR"},
}, cost_usd=0.001, tier=0)
cheap_c = FakeExtractor("cheap-c", outputs={
    "d1": {"vendor": " ACME ", "total": 100.0, "currency": "GBP"},
    "d2": {"vendor": "Globex", "total": 50, "currency": "EUR"},
}, cost_usd=0.001, tier=0)
strong = FakeExtractor("strong", outputs={
    "d1": {"vendor": "ACME", "total": 100, "currency": "USD"},
    "d2": {"vendor": "Globex", "total": 50, "currency": "EUR"},
}, cost_usd=0.05, tier=1)

config = ProjectConfig(
    schema=Invoice,
    extractors=[cheap_a.to_spec(), cheap_b.to_spec(), cheap_c.to_spec()],
    strong_extractor=strong.to_spec(),
    quorum=QuorumConfig(min_agreement=0.66, escalate_tier=1, escalation_merge="strong_wins"),
    load_docs=load_docs,
)
"""


def write_project(tmp_path) -> str:  # type: ignore[no-untyped-def]
    p = tmp_path / "project.py"
    p.write_text(PROJECT_SRC, encoding="utf-8")
    return str(p)


def test_run_json_and_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project = write_project(tmp_path)
    docs = tmp_path  # load_docs ignores the path content here
    out = tmp_path / "results.jsonl"

    result = runner.invoke(
        app, ["run", str(docs), "--config", project, "--out", str(out), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "records" in payload
    assert "budget" in payload
    assert len(payload["records"]) == 2

    # d1 currency was contested -> escalated; vendor/total unanimous -> accepted.
    recs = {r["doc_id"]: r for r in payload["records"]}
    assert recs["d1"]["fields"]["vendor"]["status"] == "accepted"
    assert recs["d1"]["fields"]["currency"]["status"] == "escalated_model"
    # budget: one strong call on d1 only, saving one strong call vs all-frontier.
    assert payload["budget"]["docs_escalated"] == 1
    assert payload["budget"]["saved_usd"] > 0

    # results file persisted; report renders it.
    assert out.exists()
    rep = runner.invoke(app, ["report", str(out), "--format", "md"])
    assert rep.exit_code == 0
    assert "# Quorum extraction report" in rep.output
    assert "Diagnostics" in rep.output


def test_diagnose_csv_md_json_and_bad_format(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project = write_project(tmp_path)
    out = tmp_path / "results.jsonl"
    run_res = runner.invoke(
        app, ["run", str(tmp_path), "--config", project, "--out", str(out), "--format", "json"]
    )
    assert run_res.exit_code == 0, run_res.output

    csv_res = runner.invoke(app, ["diagnose", str(out)])  # default --format csv
    assert csv_res.exit_code == 0, csv_res.output
    assert "path,contention_rate,mean_agreement" in csv_res.output
    assert "currency" in csv_res.output

    md_res = runner.invoke(app, ["diagnose", str(out), "--format", "md", "--threshold", "0.4"])
    assert md_res.exit_code == 0
    assert "# Per-field reliability dashboard" in md_res.output

    json_res = runner.invoke(app, ["diagnose", str(out), "--format", "json"])
    assert json_res.exit_code == 0
    payload = json.loads(json_res.output)
    assert "fields" in payload and "systematically_contested" in payload

    bad = runner.invoke(app, ["diagnose", str(out), "--format", "bogus"])
    assert bad.exit_code == 2


def test_diagnose_empty_results_header_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    empty = tmp_path / "results.jsonl"
    empty.write_text("", encoding="utf-8")
    res = runner.invoke(app, ["diagnose", str(empty)])
    assert res.exit_code == 0
    assert res.output.startswith("path,contention_rate")


def test_run_term_format(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project = write_project(tmp_path)
    result = runner.invoke(app, ["run", str(tmp_path), "--config", project])
    assert result.exit_code == 0
    assert "saved" in result.output.lower()


def test_run_queue_and_review_and_override(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Force a needs_review by disabling escalation so contested -> needs_review.
    src = PROJECT_SRC.replace(
        'QuorumConfig(min_agreement=0.66, escalate_tier=1, escalation_merge="strong_wins")',
        "QuorumConfig(min_agreement=0.66)",
    )
    project = tmp_path / "project.py"
    project.write_text(src, encoding="utf-8")
    out = tmp_path / "results.jsonl"
    queue = tmp_path / "queue.jsonl"
    overrides = tmp_path / "overrides.jsonl"

    run_res = runner.invoke(
        app,
        [
            "run",
            str(tmp_path),
            "--config",
            str(project),
            "--out",
            str(out),
            "--queue",
            str(queue),
            "--format",
            "json",
        ],
    )
    assert run_res.exit_code == 0
    assert queue.exists()

    # `review --list` lists pending items non-interactively.
    rev = runner.invoke(app, ["review", str(queue), "--list", "--overrides", str(overrides)])
    assert rev.exit_code == 0
    assert "currency" in rev.output

    # Simulate a human resolution by writing an override, then report merges it.
    overrides.write_text(
        json.dumps({"doc_id": "d1", "path": "currency", "value": "USD"}) + "\n",
        encoding="utf-8",
    )
    rep = runner.invoke(
        app, ["report", str(out), "--overrides", str(overrides), "--format", "json"]
    )
    assert rep.exit_code == 0
    payload = json.loads(rep.output)
    recs = {r["doc_id"]: r for r in payload["records"]}
    assert recs["d1"]["fields"]["currency"]["status"] == "resolved"
    assert recs["d1"]["fields"]["currency"]["value"] == "USD"


def test_review_resolve_single(tmp_path) -> None:  # type: ignore[no-untyped-def]
    queue = tmp_path / "queue.jsonl"  # need not exist for --resolve
    overrides = tmp_path / "overrides.jsonl"
    res = runner.invoke(
        app,
        ["review", str(queue), "--overrides", str(overrides), "--resolve", "d1:currency=USD"],
    )
    assert res.exit_code == 0, res.output
    assert "Recorded 1 override" in res.output
    rows = [json.loads(line) for line in overrides.read_text().splitlines() if line.strip()]
    assert rows == [{"doc_id": "d1", "path": "currency", "value": "USD"}]


def test_review_resolve_many(tmp_path) -> None:  # type: ignore[no-untyped-def]
    queue = tmp_path / "queue.jsonl"
    overrides = tmp_path / "overrides.jsonl"
    res = runner.invoke(
        app,
        [
            "review",
            str(queue),
            "--overrides",
            str(overrides),
            "--resolve",
            "d1:currency=USD",
            # value may itself contain ':' and '=' (split is on the first of each).
            "--resolve",
            "d2:note=key=val:extra",
        ],
    )
    assert res.exit_code == 0, res.output
    rows = [json.loads(line) for line in overrides.read_text().splitlines() if line.strip()]
    assert rows == [
        {"doc_id": "d1", "path": "currency", "value": "USD"},
        {"doc_id": "d2", "path": "note", "value": "key=val:extra"},
    ]


def test_review_resolve_malformed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    queue = tmp_path / "queue.jsonl"
    overrides = tmp_path / "overrides.jsonl"
    res = runner.invoke(
        app,
        ["review", str(queue), "--overrides", str(overrides), "--resolve", "d1-currency-USD"],
    )
    assert res.exit_code == 2
    assert "invalid --resolve spec" in res.output.lower()
    # Nothing is written when any spec is malformed.
    assert not overrides.exists()


def test_review_resolve_empty_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    queue = tmp_path / "queue.jsonl"
    overrides = tmp_path / "overrides.jsonl"
    res = runner.invoke(
        app,
        ["review", str(queue), "--overrides", str(overrides), "--resolve", "d1:=USD"],
    )
    assert res.exit_code == 2
    assert "empty doc_id or path" in res.output.lower()


def test_calibrate_happy_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import random

    labeled = tmp_path / "labels.jsonl"
    rng = random.Random(0)
    rows = []
    for share, p, n in [(0.25, 0.3, 40), (0.5, 0.55, 40), (0.75, 0.8, 40), (1.0, 0.78, 60)]:
        for _ in range(n):
            rows.append(
                {"winning_share": share, "k": 4, "entropy": 0.0, "correct": rng.random() < p}
            )
    labeled.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cal.json"
    res = runner.invoke(
        app, ["calibrate", "--labeled", str(labeled), "--out", str(out), "--method", "isotonic"]
    )
    assert res.exit_code == 0, res.output
    assert out.exists()
    cal = json.loads(out.read_text())
    assert cal["method"] == "isotonic"
    assert cal["fingerprint"]["labeled_set_hash"] != ""


def test_calibrate_refuses_too_few(tmp_path) -> None:  # type: ignore[no-untyped-def]
    labeled = tmp_path / "labels.jsonl"
    labeled.write_text(
        "\n".join(
            json.dumps({"winning_share": 0.5, "k": 4, "correct": i % 2 == 0}) for i in range(10)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "cal.json"
    res = runner.invoke(app, ["calibrate", "--labeled", str(labeled), "--out", str(out)])
    assert res.exit_code == 1
    assert "refused" in res.output.lower()


def test_calibrate_bad_method(tmp_path) -> None:  # type: ignore[no-untyped-def]
    labeled = tmp_path / "labels.jsonl"
    labeled.write_text('{"winning_share": 1.0, "k": 4, "correct": true}\n', encoding="utf-8")
    out = tmp_path / "cal.json"
    res = runner.invoke(
        app, ["calibrate", "--labeled", str(labeled), "--out", str(out), "--method", "bogus"]
    )
    assert res.exit_code == 2


def test_review_empty_queue(tmp_path) -> None:  # type: ignore[no-untyped-def]
    queue = tmp_path / "queue.jsonl"
    queue.write_text("", encoding="utf-8")
    res = runner.invoke(app, ["review", str(queue), "--list"])
    assert res.exit_code == 0
    assert "empty" in res.output.lower()


def test_no_args_shows_help() -> None:
    res = runner.invoke(app, [])
    assert res.exit_code in (0, 2)
    assert "run" in res.output and "calibrate" in res.output


# Config WITHOUT a load_docs hook -> the CLI text-directory fallback is used.
TEXTDIR_PROJECT = """
from pydantic import BaseModel
from quorum_extract import ProjectConfig, QuorumConfig, FakeExtractor

class Note(BaseModel):
    word_count: int

def _fn(doc):
    return {"word_count": len(doc["text"].split())}

e1 = FakeExtractor("e1", fn=_fn, cost_usd=0.0)
e2 = FakeExtractor("e2", fn=_fn, cost_usd=0.0)

config = ProjectConfig(
    schema=Note,
    extractors=[e1.to_spec(), e2.to_spec()],
    quorum=QuorumConfig(min_agreement=0.5),
)
"""


def test_run_text_directory_fallback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project = tmp_path / "project.py"
    project.write_text(TEXTDIR_PROJECT, encoding="utf-8")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("one two three", encoding="utf-8")
    (docs_dir / "b.md").write_text("hello world", encoding="utf-8")
    (docs_dir / "ignore.json").write_text("{}", encoding="utf-8")  # not loaded

    res = runner.invoke(app, ["run", str(docs_dir), "--config", str(project), "--format", "json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    ids = {r["doc_id"] for r in payload["records"]}
    assert ids == {"a", "b"}  # only .txt/.md, by stem


def test_run_no_documents_errors(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project = tmp_path / "project.py"
    project.write_text(TEXTDIR_PROJECT, encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    res = runner.invoke(app, ["run", str(empty), "--config", str(project)])
    assert res.exit_code != 0


def test_calibrate_with_config_fingerprint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import random

    project = tmp_path / "project.py"
    project.write_text(TEXTDIR_PROJECT, encoding="utf-8")
    labeled = tmp_path / "labels.jsonl"
    rng = random.Random(0)
    rows = []
    for share, p, n in [(0.25, 0.3, 40), (0.5, 0.55, 40), (0.75, 0.8, 40), (1.0, 0.78, 60)]:
        for _ in range(n):
            rows.append(
                {"winning_share": share, "k": 2, "entropy": 0.0, "correct": rng.random() < p}
            )
    labeled.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cal.json"
    res = runner.invoke(
        app,
        ["calibrate", "--labeled", str(labeled), "--out", str(out), "--config", str(project)],
    )
    assert res.exit_code == 0, res.output
    cal = json.loads(out.read_text())
    # With --config, schema + extractor-set fingerprints are recorded.
    assert cal["fingerprint"]["schema_hash"] != ""
    assert cal["fingerprint"]["extractor_set_hash"] != ""


def test_calibrate_invalid_labeled_row(tmp_path) -> None:  # type: ignore[no-untyped-def]
    labeled = tmp_path / "labels.jsonl"
    labeled.write_text('{"k": 4}\n', encoding="utf-8")  # missing required keys
    out = tmp_path / "cal.json"
    res = runner.invoke(app, ["calibrate", "--labeled", str(labeled), "--out", str(out)])
    assert res.exit_code == 2
    assert "invalid labeled set" in res.output.lower()
