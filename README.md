# quorum-extract

> Field-level quorum extraction — run multiple cheap models, accept a field only when they agree, route disagreements to an expensive model or human, with calibrated per-field confidence.

![status](https://img.shields.io/badge/status-alpha-orange) ![language](https://img.shields.io/badge/language-Python-blue) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Runs the same Pydantic schema against K cheap models/configs and reconciles **per field** (not per record). A field is accepted when enough extractors agree; contested fields escalate to a stronger model or are marked needs-review. Inter-model agreement is turned into a statistically calibrated per-field confidence score.

## Why

Single-model extraction gives you no trustworthy confidence signal. Cross-model agreement, properly calibrated, does — and it's cheaper than always calling a frontier model. `quorum-extract` orchestrates the ensemble, votes per field with type-aware normalization, calibrates agreement into a real probability of correctness, and spends frontier-model budget only where the cheap models actually disagree.

## Features

- **Per-field quorum** across K models/configs with type-aware vote equality (numbers within tolerance, dates as instants, case/whitespace-insensitive strings, order-insensitive scalar lists, key-aligned lists of objects).
- **Unified `missing` bucket** — absent key / `None` / `""` / whitespace all count as the same vote, distinct from any real value; a failed extractor counts as `missing` so the agreement denominator `K` is preserved.
- **Calibrated confidence** — turn agreement into a real probability via isotonic (monotonic, 1-D on vote share) or Platt (logistic over share, K, entropy, with a persisted `StandardScaler`). Honesty guards refuse to emit a degenerate calibrator.
- **Cost-aware cascade** — one strong invocation per *contested document* (never per field, never for accepted docs), three documented merge policies, a budget cap that marks the remainder `needs_review` (never dropped), and a `$ saved` report vs running the frontier model everywhere.
- **Diagnostics** — which fields are *systematically* contested across the corpus.
- **Human queue** — contested fields are queued to JSONL; resolutions are merged at report time from a separate overrides file (the results file is never rewritten in place).

## Install

```bash
pip install quorum-extract
# optional provider helpers:
pip install "quorum-extract[openai]"      # OpenAI structured outputs
pip install "quorum-extract[anthropic]"   # Anthropic tool-use extraction
```

Requires Python 3.11+.

## Quickstart

Run the fully offline demo (deterministic `FakeExtractor`s, no network, no keys):

```bash
git clone https://github.com/mykolapodpriatov/quorum-extract
cd quorum-extract
pip install -e ".[dev]"
python -m examples.demo
```

It extracts three invoices with three cheap extractors plus one strong extractor, shows the per-field quorum, escalates only the contested fields (one strong call per doc), and prints the `$ saved` cost report and contention diagnostics. See [`examples/invoice_project.py`](examples/invoice_project.py) and [`examples/demo.py`](examples/demo.py).

### Library

```python
from pydantic import BaseModel
from quorum_extract import (
    Document, QuorumConfig, FakeExtractor, leaf_paths, cascade_corpus, run_extractors,
)

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str

cheap = [
    FakeExtractor("a", outputs={"d1": {"vendor": "ACME", "total": "100.0", "currency": "USD"}}, tier=0),
    FakeExtractor("b", outputs={"d1": {"vendor": "acme", "total": 100,     "currency": "EUR"}}, tier=0),
    FakeExtractor("c", outputs={"d1": {"vendor": " ACME ", "total": 100.0, "currency": "GBP"}}, tier=0),
]
strong = FakeExtractor("strong", outputs={"d1": {"vendor": "ACME", "total": 100, "currency": "USD"}}, cost_usd=0.05, tier=1)

cfg = QuorumConfig(min_agreement=0.66, escalate_tier=1, escalation_merge="strong_wins")
result = cascade_corpus(
    [Document("d1", {"id": "d1"})],
    leaf_paths(Invoice),
    [e.to_spec() for e in cheap],
    cfg,
    strong_spec=strong.to_spec(),
    extract_fn=run_extractors,
)
for path, fr in sorted(result.records[0].fields.items()):
    print(path, fr.value, f"agreement={fr.agreement:.2f}", fr.status.value)
# vendor / total agree after normalization (accepted); currency is escalated.
```

### CLI

```bash
quorum-extract run docs/ --config project.py --out results.jsonl   # extract + reconcile + cascade
quorum-extract calibrate --labeled labels.jsonl --method isotonic --out calibrator.json
quorum-extract report results.jsonl --format md                    # annotated output + diagnostics
quorum-extract review queue.jsonl --list                           # work the human queue
```

A project config is a Python module exposing a `config = ProjectConfig(...)` (extractors are arbitrary callables, so config is code, not a static file). See `examples/invoice_project.py`.

## How it works

1. **Schema → leaf paths.** A Pydantic v2 model is flattened into votable leaf paths (`address.city`, `tags`, `line_items[*].sku`). Lists of objects expand only when you declare a `list_key` for alignment.
2. **Extract & normalize.** Each extractor fills the schema; values are normalized into buckets so equivalent formats agree and absence is unified.
3. **Vote.** `agreement = winning_bucket / K` with a deterministic tie-break (lowest tier, then earliest extractor).
4. **Calibrate.** A global calibrator maps agreement to a real probability of correctness, validated against held-out accuracy.
5. **Cascade.** Contested documents get one strong invocation; merged by `strong_wins` / `re_quorum` (denominator `K+1`) / `consensus`. A budget cap marks the rest `needs_review`. The report quantifies `$ saved` vs all-frontier.

## Calibration & leakage

A calibrator is fit on a small labeled set and is **not transferable by default**: it is fingerprinted to the schema, the extractor set, and the labeled-set file hash, and a mismatch on load warns. The labeled set **must** come from documents *not* in the run you score — high agreement is not certainty when a corpus has correlated errors, and the guards/tests enforce that confidence never spuriously approaches 1.

## Tech stack

- Python 3.11+, [Pydantic v2](https://docs.pydantic.dev/)
- [scikit-learn](https://scikit-learn.org/) (isotonic / logistic + `StandardScaler`), NumPy
- [Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/) CLI
- Optional: OpenAI / Anthropic SDKs (import-guarded), Ollama for local models

## Development

```bash
pip install -e ".[dev]"
ruff check && ruff format --check
mypy src
pytest -q --cov=quorum_extract
```

The full pipeline — voting, calibration, cascade, budget — is deterministic and tested offline (no network). CI runs ruff, format, mypy, and pytest on Python 3.11–3.13.

## Status & roadmap

**Alpha.** Per-field voting, calibration, cost-aware cascade, budget reporting, the human queue, and diagnostics are implemented and tested offline. Provider extractor helpers are import-guarded.

- [x] Per-field multi-model voting over a Pydantic schema
- [x] Agreement-to-confidence calibration (isotonic / Platt)
- [x] Cost-aware escalation cascade + budget report
- [x] Human-review queue + corpus diagnostics
- [ ] Active-learning loop to grow the calibration set
- [x] Per-field reliability dashboard export (`diagnose` — CSV/MD/JSON)

## License

[MIT](LICENSE) © 2026 Mykola Podpriatov
