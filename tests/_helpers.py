"""Shared synthetic models and deterministic helpers for the test suite.

Kept separate from ``conftest.py`` so tests can import these directly
(``from tests._helpers import ...``) without the conftest-import anti-pattern.
Everything is offline and deterministic.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from quorum_extract import (
    AgreementFeatures,
    Document,
    ExtractorOutput,
    ExtractorSpec,
    LabeledExample,
)

# --------------------------------------------------------------------------- #
# Synthetic schemas
# --------------------------------------------------------------------------- #


class Address(BaseModel):
    city: str
    zip: str


class LineItem(BaseModel):
    sku: str
    qty: int
    price: float


class Invoice(BaseModel):
    """A schema exercising scalars, nested objects, scalar lists, object lists."""

    vendor: str
    total: float
    issued: str  # date-like string
    address: Address
    note: str | None = None
    tags: list[str] = []
    line_items: list[LineItem] = []


class Flat(BaseModel):
    """A minimal flat schema for focused quorum/cascade tests."""

    a: str
    b: int
    c: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_spec(name: str, *, cost_usd: float = 0.0, tier: int = 0) -> ExtractorSpec:
    """A no-op spec (its fn is never called; used only for vote construction)."""
    return ExtractorSpec(name=name, fn=lambda doc: {}, cost_usd=cost_usd, tier=tier)


def make_outputs(
    payloads: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    ok: Sequence[bool] | None = None,
    tiers: Sequence[int] | None = None,
) -> list[ExtractorOutput]:
    """Build :class:`ExtractorOutput`s from ``(name, data)`` pairs."""
    n = len(payloads)
    ok_flags = list(ok) if ok is not None else [True] * n
    tier_vals = list(tiers) if tiers is not None else [0] * n
    out: list[ExtractorOutput] = []
    for (name, data), is_ok, tier in zip(payloads, ok_flags, tier_vals, strict=True):
        out.append(ExtractorOutput(spec=make_spec(name, tier=tier), data=dict(data), ok=is_ok))
    return out


def fake_doc(doc_id: str, **fields: Any) -> Document:
    """A document payload carrying its id (for FakeExtractor.doc_id)."""
    return Document(doc_id=doc_id, payload={"id": doc_id, **fields})


def synthetic_labeled(seed: int = 20240601) -> list[LabeledExample]:
    """A seeded labeled set spanning the agreement range, with both classes.

    Accuracy increases with share, BUT high-agreement cases include correlated
    errors (share=1.0 only ~75% correct), so an honest calibrator must not map
    high agreement to ~1.0.
    """
    rng = random.Random(seed)
    examples: list[LabeledExample] = []
    plan = [
        (0.25, 0.30, 40),
        (0.40, 0.45, 40),
        (0.50, 0.55, 40),
        (0.60, 0.65, 40),
        (0.75, 0.82, 40),
        (1.00, 0.75, 80),
    ]
    for share, p_correct, n in plan:
        for _ in range(n):
            examples.append(
                LabeledExample(
                    features=AgreementFeatures(winning_share=share, k=4, entropy=0.0),
                    correct=rng.random() < p_correct,
                )
            )
    return examples
