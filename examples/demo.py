"""End-to-end offline demo of quorum-extract (no network, fully deterministic).

Run it:

    python -m examples.demo
    # or
    python examples/demo.py

It uses the FakeExtractors in ``examples.invoice_project`` to:

1. Extract three invoices with three cheap extractors + one strong extractor.
2. Reconcile every field by quorum (type-aware normalization collapses
   formatting differences into agreement).
3. Escalate only the genuinely contested fields, one strong call per doc.
4. Print an annotated terminal report plus the cost ("$ saved") summary and the
   corpus contention diagnostics.

No API keys, no network -- the FakeExtractors return canned data deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a plain script (``python examples/demo.py``) by making the
# examples package importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.invoice_project import config

from quorum_extract import Document, field_contention, run_project
from quorum_extract.report import render_terminal


def main() -> None:
    documents = [Document(doc_id=did, payload=payload) for did, payload in config.load_docs(None)]
    report, records = run_project(config, documents)

    print(render_terminal(records, report.budget))

    print("Field reconciliation summary:")
    print(f"  accepted     : {report.n_accepted}")
    print(f"  escalated    : {report.n_escalated}")
    print(f"  needs review : {report.n_needs_review}")
    print()

    print("Contention diagnostics (which fields are systematically hard):")
    for diag in field_contention(records):
        bar = "#" * round(diag.contention_rate * 20)
        print(f"  {diag.path:<18} {diag.contention_rate:>5.0%}  {bar}")
    print()

    b = report.budget
    print(
        f"Cost: spent ${b.total_cost_usd:.4f} "
        f"(escalation ${b.escalation_cost_usd:.4f} on {b.docs_escalated}/{b.docs_total} docs). "
        f"Running the frontier model on every doc would have cost "
        f"${b.all_frontier_cost_usd:.4f} -- saved ${b.saved_usd:.4f}."
    )


if __name__ == "__main__":
    main()
