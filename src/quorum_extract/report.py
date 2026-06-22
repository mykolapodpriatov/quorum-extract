"""Annotated record output and run reports (terminal / Markdown / JSON).

Renders the ``(value, agreement, confidence, escalation)`` annotation per field,
the budget summary, and corpus diagnostics, in three formats:

* ``term`` -- a Rich table for interactive use.
* ``md`` -- GitHub-flavored Markdown for docs/PRs.
* ``json`` -- machine-readable, schema-stable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from io import StringIO

from rich.console import Console
from rich.table import Table

from .config import record_to_dict
from .diagnostics import field_contention
from .types import BudgetReport, EscalationStatus, RecordResult

_STATUS_STYLE = {
    EscalationStatus.ACCEPTED: "green",
    EscalationStatus.ESCALATED_MODEL: "yellow",
    EscalationStatus.NEEDS_REVIEW: "red",
    EscalationStatus.RESOLVED: "cyan",
}


def _fmt_value(value: object) -> str:
    if value is None:
        return "∅"
    return str(value)


def _fmt_conf(conf: float | None) -> str:
    return "-" if conf is None else f"{conf:.2f}"


def render_budget_dict(budget: BudgetReport) -> dict[str, float | int]:
    """Budget summary as a plain dict (used by JSON output)."""
    return {
        "cheap_cost_usd": budget.cheap_cost_usd,
        "escalation_cost_usd": budget.escalation_cost_usd,
        "total_cost_usd": budget.total_cost_usd,
        "all_frontier_cost_usd": budget.all_frontier_cost_usd,
        "saved_usd": budget.saved_usd,
        "docs_total": budget.docs_total,
        "docs_escalated": budget.docs_escalated,
        "docs_over_budget": budget.docs_over_budget,
    }


def render_json(records: Sequence[RecordResult], budget: BudgetReport | None = None) -> str:
    """Machine-readable JSON of records (+ optional budget + diagnostics)."""
    payload: dict[str, object] = {"records": [record_to_dict(r) for r in records]}
    if budget is not None:
        payload["budget"] = render_budget_dict(budget)
    payload["diagnostics"] = [
        {
            "path": d.path,
            "contention_rate": d.contention_rate,
            "mean_agreement": d.mean_agreement,
            "n_contested": d.n_contested,
            "n_records": d.n_records,
        }
        for d in field_contention(records)
    ]
    return json.dumps(payload, indent=2)


def render_markdown(records: Sequence[RecordResult], budget: BudgetReport | None = None) -> str:
    """GitHub-flavored Markdown report."""
    lines: list[str] = ["# Quorum extraction report", ""]
    if budget is not None:
        lines += [
            "## Budget",
            "",
            f"- Total spent: **${budget.total_cost_usd:.4f}**",
            f"- Escalation spent: ${budget.escalation_cost_usd:.4f} "
            f"on {budget.docs_escalated}/{budget.docs_total} docs",
            f"- All-frontier hypothetical: ${budget.all_frontier_cost_usd:.4f}",
            f"- **Saved: ${budget.saved_usd:.4f}**",
        ]
        if budget.docs_over_budget:
            lines.append(f"- Over budget (needs review): {budget.docs_over_budget} docs")
        lines.append("")
    for record in records:
        lines += [
            f"## {record.doc_id}",
            "",
            "| field | value | agreement | confidence | status |",
            "| --- | --- | --- | --- | --- |",
        ]
        for path in sorted(record.fields):
            fr = record.fields[path]
            lines.append(
                f"| `{path}` | {_fmt_value(fr.value)} | {fr.agreement:.2f} | "
                f"{_fmt_conf(fr.confidence)} | {fr.status.value} |"
            )
        lines.append("")
    diags = field_contention(records)
    if diags:
        lines += [
            "## Diagnostics (contention by field)",
            "",
            "| field | contention rate | mean agreement |",
            "| --- | --- | --- |",
        ]
        for d in diags:
            lines.append(f"| `{d.path}` | {d.contention_rate:.2f} | {d.mean_agreement:.2f} |")
        lines.append("")
    return "\n".join(lines)


def render_terminal(
    records: Sequence[RecordResult],
    budget: BudgetReport | None = None,
    *,
    console: Console | None = None,
) -> str:
    """Rich-rendered terminal report; returns the captured text."""
    buf = StringIO()
    con = console or Console(file=buf, force_terminal=False, width=100)
    if budget is not None:
        con.print(
            f"[bold]Budget[/bold]: spent ${budget.total_cost_usd:.4f} "
            f"(escalation ${budget.escalation_cost_usd:.4f} on "
            f"{budget.docs_escalated}/{budget.docs_total} docs) — "
            f"[green]saved ${budget.saved_usd:.4f}[/green] vs all-frontier "
            f"${budget.all_frontier_cost_usd:.4f}"
        )
        if budget.docs_over_budget:
            con.print(f"[red]{budget.docs_over_budget} docs over budget -> needs_review[/red]")
    for record in records:
        table = Table(title=f"{record.doc_id}  (cost ${record.cost_usd:.4f})")
        table.add_column("field", style="bold")
        table.add_column("value")
        table.add_column("agreement", justify="right")
        table.add_column("confidence", justify="right")
        table.add_column("status")
        for path in sorted(record.fields):
            fr = record.fields[path]
            style = _STATUS_STYLE.get(fr.status, "")
            table.add_row(
                path,
                _fmt_value(fr.value),
                f"{fr.agreement:.2f}",
                _fmt_conf(fr.confidence),
                f"[{style}]{fr.status.value}[/{style}]" if style else fr.status.value,
            )
        con.print(table)
    return buf.getvalue()


def render(
    records: Sequence[RecordResult],
    budget: BudgetReport | None = None,
    fmt: str = "term",
) -> str:
    """Dispatch to a renderer by format name (``term`` | ``md`` | ``json``)."""
    if fmt == "json":
        return render_json(records, budget)
    if fmt == "md":
        return render_markdown(records, budget)
    if fmt == "term":
        return render_terminal(records, budget)
    raise ValueError(f"unknown report format: {fmt!r} (use term|md|json)")
