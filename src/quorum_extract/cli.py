"""Command-line interface: ``run | calibrate | review | report``.

Thin wrappers over the library. All commands are offline-friendly; the heavy
lifting lives in the library modules. The CLI prints a cost report on ``run`` and
machine-readable JSON when asked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from . import report as report_mod
from .calibration import (
    AgreementCalibrator,
    CalibrationError,
    fingerprint_for,
)
from .cascade import Document
from .config import load_config, read_labeled, read_results, write_results
from .diagnostics import suggest_labels
from .human import (
    Override,
    ReviewQueue,
    apply_overrides,
    items_for_review,
    load_overrides,
    write_override,
)
from .pipeline import run_project

app = typer.Typer(
    name="quorum-extract",
    help="Field-level quorum extraction: vote, calibrate, escalate.",
    no_args_is_help=True,
    add_completion=False,
)
_err = Console(stderr=True)
_out = Console()


def _load_documents(docs_path: Path, config: object) -> list[Document]:
    """Load documents using the project's ``load_docs`` hook, or a text-dir fallback.

    Fallback: every ``*.txt``/``*.md`` file in ``docs_path`` becomes a document
    whose id is the filename stem and whose payload is ``{"id": stem, "text": ...}``.
    """
    loader = getattr(config, "load_docs", None)
    if callable(loader):
        return [Document(doc_id=str(did), payload=payload) for did, payload in loader(docs_path)]
    docs: list[Document] = []
    if docs_path.is_dir():
        for fp in sorted(docs_path.iterdir()):
            if fp.suffix.lower() in (".txt", ".md"):
                text = fp.read_text(encoding="utf-8")
                docs.append(Document(doc_id=fp.stem, payload={"id": fp.stem, "text": text}))
    if not docs:
        raise typer.BadParameter(
            f"no documents found under {docs_path} and config has no load_docs() hook"
        )
    return docs


@app.command()
def run(
    docs: Annotated[Path, typer.Argument(help="Documents directory or file.")],
    config: Annotated[Path, typer.Option("--config", "-c", help="Project config .py.")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write results JSONL here.")
    ] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="term|md|json.")] = "term",
    queue: Annotated[
        Path | None,
        typer.Option("--queue", help="Append needs-review items to this JSONL queue."),
    ] = None,
) -> None:
    """Extract + reconcile + cascade a corpus; print the cost report."""
    cfg = load_config(config)
    documents = _load_documents(docs, cfg)
    run_report, records = run_project(cfg, documents)

    if out is not None:
        write_results(out, records)
    if queue is not None:
        items = items_for_review(records)
        if items:
            ReviewQueue(queue).extend(items)

    rendered = report_mod.render(records, run_report.budget, fmt)
    if fmt == "term":
        _out.print(rendered, end="")
    else:
        typer.echo(rendered)


@app.command()
def calibrate(
    labeled: Annotated[Path, typer.Option("--labeled", "-l", help="Labeled rows JSONL.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Write calibrator JSON here.")],
    method: Annotated[str, typer.Option("--method", "-m", help="isotonic|platt.")] = "isotonic",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Project config (for fingerprinting)."),
    ] = None,
    min_examples: Annotated[int, typer.Option(help="Minimum labeled examples.")] = 50,
) -> None:
    """Fit an agreement->confidence calibrator from a labeled set.

    Leakage warning: the labeled set MUST be drawn from documents NOT in any run
    you will score with this calibrator. The calibrator is fingerprinted to the
    labeled-set hash (and, with --config, the schema + extractor set) so reuse
    against a different config/labeled set is detectable.
    """
    if method not in ("isotonic", "platt"):
        _err.print(f"[red]unknown method {method!r}; use isotonic|platt[/red]")
        raise typer.Exit(code=2)

    try:
        examples = read_labeled(labeled)
    except ValueError as exc:
        _err.print(f"[red]invalid labeled set:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    schema_paths: list[str] = []
    extractor_names: list[str] = []
    if config is not None:
        cfg = load_config(config)
        schema_paths = cfg.leaf_path_strings()
        extractor_names = cfg.extractor_names()
    fp = fingerprint_for(
        schema_paths=schema_paths, extractor_names=extractor_names, labeled_path=str(labeled)
    )

    cal = AgreementCalibrator(method=method, fingerprint=fp, min_examples=min_examples)  # type: ignore[arg-type]
    try:
        cal.fit(examples)
    except CalibrationError as exc:
        _err.print(f"[red]calibration refused:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    cal.save(out)
    _out.print(f"[green]Fitted {method} calibrator[/green] on {len(examples)} examples -> {out}")


@app.command()
def report(
    results: Annotated[Path, typer.Argument(help="Results JSONL from `run`.")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="term|md|json.")] = "term",
    overrides: Annotated[
        Path | None,
        typer.Option("--overrides", help="resolved-overrides.jsonl to merge at read time."),
    ] = None,
) -> None:
    """Render an annotated report (+ diagnostics), merging resolved overrides."""
    records = read_results(results)
    if overrides is not None and overrides.exists():
        records = apply_overrides(records, load_overrides(overrides))
    rendered = report_mod.render(records, None, fmt)
    if fmt == "term":
        _out.print(rendered, end="")
    else:
        typer.echo(rendered)


def _parse_resolve(spec: str) -> Override:
    """Parse a ``"doc_id:path=value"`` resolution spec into an :class:`Override`.

    Splits on the first ``:`` (doc id) and then the first ``=`` (path / value), so
    values may contain ``:`` and ``=``. ``doc_id`` and ``path`` must be non-empty.

    Raises:
        ValueError: if the spec is missing its ``:`` or ``=``, or has an empty
            doc id or path.
    """
    doc_part, sep, rest = spec.partition(":")
    if not sep:
        raise ValueError(f"missing ':' in resolve spec {spec!r} (expected 'doc_id:path=value')")
    path_part, eq, value = rest.partition("=")
    if not eq:
        raise ValueError(f"missing '=' in resolve spec {spec!r} (expected 'doc_id:path=value')")
    doc_id = doc_part.strip()
    path = path_part.strip()
    if not doc_id or not path:
        raise ValueError(f"empty doc_id or path in resolve spec {spec!r}")
    return Override(doc_id=doc_id, path=path, value=value)


@app.command()
def diagnose(
    results: Annotated[Path, typer.Argument(help="Results JSONL from `run`.")],
    fmt: Annotated[str, typer.Option("--format", "-f", help="csv|md|json.")] = "csv",
    threshold: Annotated[
        float,
        typer.Option(help="Contention rate at/above which a field is flagged systematic."),
    ] = 0.5,
) -> None:
    """Export the per-field reliability dashboard (CSV/MD/JSON).

    One row per field path, sorted by descending contention. CSV is the point:
    it drops straight into a spreadsheet for the "which fields are systematically
    hard" review. An empty results file yields a header-only export, not an error.
    """
    records = read_results(results)
    try:
        rendered = report_mod.render_diagnostics(records, fmt, threshold)
    except ValueError as exc:
        _err.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    typer.echo(rendered)


@app.command()
def review(
    queue: Annotated[Path, typer.Argument(help="Review queue JSONL.")],
    overrides: Annotated[
        Path, typer.Option("--overrides", help="Append resolved values here.")
    ] = Path("resolved-overrides.jsonl"),
    non_interactive: Annotated[
        bool, typer.Option("--list", help="List pending items and exit (no prompts).")
    ] = False,
    resolve: Annotated[
        list[str] | None,
        typer.Option(
            "--resolve",
            help='Non-interactive: record "doc_id:path=value" and exit (repeatable).',
        ),
    ] = None,
) -> None:
    """Work the human-review queue; resolutions append to an overrides file."""
    if resolve:
        try:
            parsed = [_parse_resolve(spec) for spec in resolve]
        except ValueError as exc:
            _err.print(f"[red]invalid --resolve spec:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        for override in parsed:
            write_override(overrides, override)
        _out.print(f"[green]Recorded {len(parsed)} override(s)[/green] -> {overrides}")
        return
    rq = ReviewQueue(queue)
    items = rq.load()
    if not items:
        _out.print("[green]Review queue is empty.[/green]")
        return
    if non_interactive or not sys.stdin.isatty():
        for item in items:
            _out.print(f"{item.doc_id}  {item.path}  candidates={item.candidates}")
        return
    for item in items:  # pragma: no cover - interactive path
        _out.print(f"\n[bold]{item.doc_id}[/bold] / [cyan]{item.path}[/cyan]")
        for i, cand in enumerate(item.candidates):
            _out.print(f"  [{i}] {cand!r}")
        choice = typer.prompt("Pick index, type a value, or 's' to skip", default="s")
        if choice == "s":
            continue
        if choice.isdigit() and int(choice) < len(item.candidates):
            value = item.candidates[int(choice)]
        else:
            value = choice
        write_override(overrides, Override(doc_id=item.doc_id, path=item.path, value=value))
        _out.print(f"  [green]resolved[/green] -> {value!r}")


@app.command(name="suggest-labels")
def suggest_labels_command(
    results: Annotated[Path, typer.Argument(help="Results JSONL from `run`.")],
    n: Annotated[int, typer.Option("--n", help="Number of suggestions to emit.")] = 10,
    boundary: Annotated[
        float, typer.Option(help="Accept boundary (quorum threshold) to rank against.")
    ] = 0.5,
) -> None:
    """Rank which doc/field records to label next to most improve calibration.

    Prints the top-N as JSONL of ``{doc_id, path, agreement}``, most informative
    first (agreement nearest the accept boundary, contested fields breaking ties).
    Fully offline and deterministic.
    """
    records = read_results(results)
    for suggestion in suggest_labels(records, n=n, boundary=boundary):
        typer.echo(
            json.dumps(
                {
                    "doc_id": suggestion.doc_id,
                    "path": suggestion.path,
                    "agreement": suggestion.agreement,
                }
            )
        )


if __name__ == "__main__":  # pragma: no cover
    app()
