"""Pytest fixtures (thin wrappers over :mod:`tests._helpers`).

All fixtures are deterministic and network-free.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from pydantic import BaseModel

from quorum_extract import FakeExtractor, LabeledExample

from ._helpers import Flat, Invoice
from ._helpers import synthetic_labeled as _synthetic_labeled


@pytest.fixture
def invoice_model() -> type[BaseModel]:
    return Invoice


@pytest.fixture
def flat_model() -> type[BaseModel]:
    return Flat


@pytest.fixture
def synthetic_labeled() -> list[LabeledExample]:
    return _synthetic_labeled()


@pytest.fixture
def fake_extractor_factory() -> Callable[..., FakeExtractor]:
    """Factory producing :class:`FakeExtractor`s with canned per-doc outputs."""

    def _make(
        name: str,
        outputs: Mapping[str, Mapping[str, Any]],
        *,
        cost_usd: float = 0.0,
        tier: int = 0,
        fail_on: set[str] | None = None,
    ) -> FakeExtractor:
        return FakeExtractor(name, outputs=outputs, cost_usd=cost_usd, tier=tier, fail_on=fail_on)

    return _make
