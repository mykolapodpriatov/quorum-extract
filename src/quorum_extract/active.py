"""The feedback half of the active-learning loop.

``suggest-labels`` ranks which records to label next. This module is what
happens after you label them: fold the new labels into an existing set, refit,
and get a number that says whether calibration actually improved.

Two rules shape everything here:

* **Score on data the calibrator has not seen.** Fitting and scoring on the same
  labels always looks good and means nothing, so evaluation is always against a
  held-out split.
* **Deterministic.** The split is seeded and the metrics are closed-form, so two
  runs over the same labels give the same answer. The rest of this package is
  offline and reproducible and this is not the place to break that.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .calibration import AgreementCalibrator, LabeledExample

__all__ = [
    "CalibrationScores",
    "brier_score",
    "evaluate_calibrator",
    "expected_calibration_error",
    "merge_labeled",
    "split_labeled",
]

#: Default number of equal-width bins for expected calibration error.
DEFAULT_ECE_BINS = 10


def _identity(example: LabeledExample) -> tuple[str, str] | None:
    """The ``(doc_id, path)`` a row can be deduplicated on, or ``None``.

    Rows written before those fields existed carry neither, and two such rows
    cannot be told apart, so they are never merged into each other.
    """
    if example.doc_id is None or example.path is None:
        return None
    return (example.doc_id, example.path)


def merge_labeled(
    base: Sequence[LabeledExample], new: Sequence[LabeledExample]
) -> list[LabeledExample]:
    """Fold ``new`` labels into ``base``, newer labels winning.

    Deduplication is on ``(doc_id, path)``. A re-labelled record **replaces** the
    old one rather than being kept alongside it: a corrected label is the whole
    reason anyone relabels, and keeping both would train the calibrator on a
    contradiction.

    Rows without both ``doc_id`` and ``path`` cannot be identified, so they are
    appended and never deduplicated. Ordering is stable: ``base`` order first
    (with replaced rows updated in place), then whatever in ``new`` is genuinely
    new, so a merge does not reshuffle the set and change a fingerprint for no
    reason.
    """
    merged = list(base)
    positions: dict[tuple[str, str], int] = {}
    for index, example in enumerate(merged):
        key = _identity(example)
        if key is not None:
            positions[key] = index

    for example in new:
        key = _identity(example)
        if key is not None and key in positions:
            merged[positions[key]] = example
            continue
        if key is not None:
            positions[key] = len(merged)
        merged.append(example)
    return merged


def split_labeled(
    examples: Sequence[LabeledExample], *, holdout: float, seed: int
) -> tuple[list[LabeledExample], list[LabeledExample]]:
    """Split into ``(train, test)`` with a seeded shuffle.

    ``holdout`` is the test fraction in ``(0, 1)``. The split is over a shuffled
    index list rather than the input order, because a labelled set built by
    appending batches is ordered by when it was labelled, and slicing that
    directly would put one whole labelling session on one side.

    Raises:
        ValueError: if ``holdout`` is outside ``(0, 1)``, or if it leaves either
            side empty (an empty test split cannot be scored, and an empty train
            split cannot be fit).
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be strictly between 0 and 1, got {holdout!r}")
    n = len(examples)
    n_test = round(n * holdout)
    if n_test < 1 or n_test >= n:
        raise ValueError(
            f"holdout={holdout} over {n} example(s) leaves an empty split; "
            "label more rows or lower the holdout fraction"
        )
    order = list(range(n))
    random.Random(seed).shuffle(order)
    test_idx = set(order[:n_test])
    train = [examples[i] for i in range(n) if i not in test_idx]
    test = [examples[i] for i in range(n) if i in test_idx]
    return train, test


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between predicted probability and outcome.

    Lower is better; 0 is perfect. Unlike accuracy it punishes a confident wrong
    answer harder than an unsure one, which is the property that matters for a
    confidence score people route on.

    Raises:
        ValueError: if the inputs differ in length or are empty.
    """
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels differ in length: {len(probabilities)} vs {len(labels)}"
        )
    if not labels:
        raise ValueError("cannot score an empty set")
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray([1.0 if v else 0.0 for v in labels], dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[bool], *, bins: int = DEFAULT_ECE_BINS
) -> float:
    """Weighted average gap between predicted confidence and observed accuracy.

    Predictions are put into ``bins`` equal-width buckets over ``[0, 1]``; each
    bucket contributes ``|accuracy - mean confidence|`` weighted by its share of
    the data. Empty buckets contribute nothing.

    This is the number that answers "when this says 0.9, is it right 90% of the
    time?", which Brier alone does not: a model can have a good Brier score and
    still be systematically overconfident.

    Raises:
        ValueError: if the inputs differ in length, are empty, or ``bins < 1``.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    if len(probabilities) != len(labels):
        raise ValueError(
            f"probabilities and labels differ in length: {len(probabilities)} vs {len(labels)}"
        )
    if not labels:
        raise ValueError("cannot score an empty set")

    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray([1.0 if v else 0.0 for v in labels], dtype=np.float64)
    n = len(p)
    # Right-closed bins so a prediction of exactly 1.0 lands in the last bucket
    # rather than falling off the end.
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = which == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        total += (count / n) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return total


@dataclass(frozen=True, slots=True)
class CalibrationScores:
    """Held-out quality of one calibrator. Lower is better on both metrics."""

    n: int
    brier: float
    ece: float

    def to_dict(self) -> dict[str, float | int]:
        return {"n": self.n, "brier": self.brier, "ece": self.ece}


def evaluate_calibrator(
    calibrator: AgreementCalibrator,
    examples: Iterable[LabeledExample],
    *,
    bins: int = DEFAULT_ECE_BINS,
) -> CalibrationScores:
    """Score ``calibrator`` against labelled rows it was not fit on.

    Raises:
        ValueError: if ``examples`` is empty.
    """
    rows = list(examples)
    if not rows:
        raise ValueError("cannot score an empty set")
    probabilities = [
        calibrator.predict_one(example.features, group=example.group) for example in rows
    ]
    labels = [example.correct for example in rows]
    return CalibrationScores(
        n=len(rows),
        brier=brier_score(probabilities, labels),
        ece=expected_calibration_error(probabilities, labels, bins=bins),
    )
