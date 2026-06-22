"""Agreement features that feed calibration.

A reconciled field exposes a small, fixed feature vector that a calibrator maps
to a probability of correctness:

* ``winning_share`` -- ``agreement`` (winning bucket size / K), the sole 1-D
  feature isotonic regression consumes.
* ``k`` -- the number of extractors (denominator), a Platt-only feature.
* ``entropy`` -- normalized Shannon entropy of the vote distribution in
  ``[0, 1]`` (0 = unanimous, 1 = maximally split), a Platt-only feature.

Order matters and is frozen in :data:`FEATURE_NAMES`; persistence relies on it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .types import FieldResult, FieldVote

#: Frozen feature order. ``winning_share`` is first so isotonic (1-D) can slice
#: column 0 unambiguously.
FEATURE_NAMES: tuple[str, ...] = ("winning_share", "k", "entropy")


@dataclass(frozen=True, slots=True)
class AgreementFeatures:
    """The fixed feature vector for one reconciled field."""

    winning_share: float
    k: int
    entropy: float

    def as_row(self) -> list[float]:
        """Dense row in :data:`FEATURE_NAMES` order."""
        return [self.winning_share, float(self.k), self.entropy]


def _normalized_entropy(votes: Sequence[FieldVote]) -> float:
    """Shannon entropy of the bucket distribution normalized to ``[0, 1]``.

    Normalized by ``log(K)`` so it is comparable across different ``K``. With
    ``K <= 1`` there is no uncertainty, so it is ``0``.
    """
    counts: dict[str, int] = {}
    for v in votes:
        counts[v.normalized_key] = counts.get(v.normalized_key, 0) + 1
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p)
    max_ent = math.log(total)
    return ent / max_ent if max_ent > 0 else 0.0


def features_for(result: FieldResult) -> AgreementFeatures:
    """Compute :class:`AgreementFeatures` for a reconciled field."""
    k = len(result.votes)
    return AgreementFeatures(
        winning_share=result.agreement,
        k=k,
        entropy=_normalized_entropy(result.votes),
    )
