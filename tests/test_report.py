"""Tests for report rendering, focused on the diagnostics surface.

These assert that ``top_disagreement_keys`` -- the "which wrong buckets recur"
signal computed in :mod:`quorum_extract.diagnostics` -- actually reaches both the
JSON and Markdown reports.
"""

from __future__ import annotations

import json

from quorum_extract import EscalationStatus, FieldResult, RecordResult
from quorum_extract.report import render_json, render_markdown


def _contested_corpus() -> list[RecordResult]:
    # "vendor" is contested in both records and both times the winning bucket is
    # "acme"; "total" is always accepted so it has no disagreement buckets.
    def record(doc_id: str) -> RecordResult:
        return RecordResult(
            doc_id=doc_id,
            fields={
                "vendor": FieldResult(
                    path="vendor",
                    agreement=0.4,
                    status=EscalationStatus.NEEDS_REVIEW,
                    winning_key="acme",
                ),
                "total": FieldResult(
                    path="total",
                    agreement=1.0,
                    status=EscalationStatus.ACCEPTED,
                    winning_key="100",
                ),
            },
        )

    return [record("d1"), record("d2")]


def test_json_report_includes_top_disagreement_keys() -> None:
    payload = json.loads(render_json(_contested_corpus()))
    diags = {d["path"]: d for d in payload["diagnostics"]}
    # Keys are serialized as sorted [key, count] pairs for the contested field.
    assert diags["vendor"]["top_disagreement_keys"] == [["acme", 2]]
    # Accepted-only field carries an empty (but present, schema-stable) list.
    assert diags["total"]["top_disagreement_keys"] == []


def test_markdown_report_includes_top_disagreement_keys() -> None:
    md = render_markdown(_contested_corpus())
    assert "top disagreement keys" in md
    # The "acme (2)" bucket cell appears only in the diagnostics table row.
    diag_line = next(
        line for line in md.splitlines() if line.startswith("| `vendor`") and "acme (2)" in line
    )
    assert diag_line.endswith("acme (2) |")
