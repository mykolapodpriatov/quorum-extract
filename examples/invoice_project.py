"""Offline example project for ``quorum-extract`` (no network, fully deterministic).

Defines an invoice schema, three cheap :class:`FakeExtractor`s that disagree on a
couple of fields, and one strong escalation extractor. Import this as the
``--config`` for the CLI, or run :mod:`examples.demo` to see the whole pipeline.

The cheap extractors deliberately format the same facts differently (``"100.0"``
vs ``100`` vs ``100.00``; ``"ACME"`` vs ``"acme"`` vs ``" ACME "``) so the
type-aware normalizer collapses them into agreement -- while genuinely contested
fields (currency on ``inv-001``, tax on ``inv-003``) are escalated.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from quorum_extract import FakeExtractor, ProjectConfig, QuorumConfig


class Address(BaseModel):
    city: str
    country: str


class Invoice(BaseModel):
    """The target schema the ensemble extracts."""

    vendor: str
    total: float
    currency: str
    issued: str
    tax: float
    address: Address


# Each FakeExtractor returns a canned, deterministic dict keyed by document id.
# Cheap extractors agree on most fields (after normalization) and disagree on a
# few, so the cascade has something to escalate.
_CHEAP_A = FakeExtractor(
    "cheap-a",
    outputs={
        "inv-001": {
            "vendor": "ACME",
            "total": "100.0",
            "currency": "USD",
            "issued": "2024-01-15",
            "tax": 8.0,
            "address": {"city": "NYC", "country": "US"},
        },
        "inv-002": {
            "vendor": "Globex",
            "total": 250,
            "currency": "EUR",
            "issued": "Feb 2, 2024",
            "tax": 21.0,
            "address": {"city": "Berlin", "country": "DE"},
        },
        "inv-003": {
            "vendor": "Initech",
            "total": 99.5,
            "currency": "GBP",
            "issued": "2024-03-20",
            "tax": 19.9,
            "address": {"city": "London", "country": "GB"},
        },
    },
    cost_usd=0.0008,
    tier=0,
)
_CHEAP_B = FakeExtractor(
    "cheap-b",
    outputs={
        "inv-001": {
            "vendor": "acme",
            "total": 100,
            "currency": "EUR",
            "issued": "Jan 15, 2024",
            "tax": 8.0,
            "address": {"city": "New York", "country": "US"},
        },
        "inv-002": {
            "vendor": "GLOBEX",
            "total": 250.0,
            "currency": "EUR",
            "issued": "2024-02-02",
            "tax": 21.0,
            "address": {"city": "Berlin", "country": "DE"},
        },
        "inv-003": {
            "vendor": "Initech",
            "total": 99.5,
            "currency": "GBP",
            "issued": "2024-03-20",
            "tax": 17.0,
            "address": {"city": "London", "country": "GB"},
        },
    },
    cost_usd=0.0008,
    tier=0,
)
_CHEAP_C = FakeExtractor(
    "cheap-c",
    outputs={
        "inv-001": {
            "vendor": " ACME ",
            "total": 100.00,
            "currency": "GBP",
            "issued": "2024-01-15",
            "tax": 8.0,
            "address": {"city": "NYC", "country": "US"},
        },
        "inv-002": {
            "vendor": "Globex",
            "total": 250,
            "currency": "EUR",
            "issued": "2024-02-02",
            "tax": 21.0,
            "address": {"city": "Berlin", "country": "DE"},
        },
        "inv-003": {
            "vendor": "Initech",
            "total": 99.5,
            "currency": "GBP",
            "issued": "2024-03-20",
            "tax": 22.5,
            "address": {"city": "London", "country": "GB"},
        },
    },
    cost_usd=0.0008,
    tier=0,
)
# The strong (frontier) extractor: more expensive, higher tier, used only to
# resolve contested fields.
_STRONG = FakeExtractor(
    "frontier",
    outputs={
        "inv-001": {
            "vendor": "ACME",
            "total": 100,
            "currency": "USD",
            "issued": "2024-01-15",
            "tax": 8.0,
            "address": {"city": "NYC", "country": "US"},
        },
        "inv-002": {
            "vendor": "Globex",
            "total": 250,
            "currency": "EUR",
            "issued": "2024-02-02",
            "tax": 21.0,
            "address": {"city": "Berlin", "country": "DE"},
        },
        "inv-003": {
            "vendor": "Initech",
            "total": 99.5,
            "currency": "GBP",
            "issued": "2024-03-20",
            "tax": 19.9,
            "address": {"city": "London", "country": "GB"},
        },
    },
    cost_usd=0.04,
    tier=1,
)


def load_docs(_path: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(doc_id, payload)`` pairs. The payload only needs an ``id`` here
    because the FakeExtractors look documents up by id (a real loader would read
    file/text content)."""
    for doc_id in ("inv-001", "inv-002", "inv-003"):
        yield doc_id, {"id": doc_id}


config = ProjectConfig(
    schema=Invoice,
    extractors=[_CHEAP_A.to_spec(), _CHEAP_B.to_spec(), _CHEAP_C.to_spec()],
    strong_extractor=_STRONG.to_spec(),
    quorum=QuorumConfig(
        min_agreement=0.66,  # accept a field when >= 2 of 3 cheap extractors agree
        escalate_tier=1,
        escalation_merge="strong_wins",
    ),
    load_docs=load_docs,
)
