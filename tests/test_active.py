"""Tests for the feedback half of the active-learning loop.

Everything here is offline and closed-form: labelled rows in, numbers out.
"""

from __future__ import annotations

import math
import random

import pytest

from quorum_extract import (
    AgreementCalibrator,
    AgreementFeatures,
    LabeledExample,
    brier_score,
    evaluate_calibrator,
    expected_calibration_error,
    merge_labeled,
    split_labeled,
)

from ._helpers import synthetic_labeled


def _row(
    doc_id: str | None, path: str | None, *, correct: bool, share: float = 0.5
) -> LabeledExample:
    return LabeledExample(
        features=AgreementFeatures(winning_share=share, k=4, entropy=0.0),
        correct=correct,
        doc_id=doc_id,
        path=path,
    )


# --------------------------------------------------------------------- merging


def test_merge_appends_genuinely_new_rows() -> None:
    base = [_row("d1", "a", correct=True), _row("d1", "b", correct=False)]
    new = [_row("d2", "a", correct=True)]

    merged = merge_labeled(base, new)

    assert [(e.doc_id, e.path) for e in merged] == [("d1", "a"), ("d1", "b"), ("d2", "a")]


def test_relabelling_replaces_rather_than_duplicating() -> None:
    """A corrected label is why anyone relabels; keeping both would train on a
    contradiction."""
    base = [_row("d1", "a", correct=True), _row("d1", "b", correct=True)]
    new = [_row("d1", "a", correct=False)]

    merged = merge_labeled(base, new)

    assert len(merged) == 2
    corrected = next(e for e in merged if (e.doc_id, e.path) == ("d1", "a"))
    assert corrected.correct is False


def test_merge_is_idempotent() -> None:
    base = [_row("d1", "a", correct=True), _row("d2", "b", correct=False)]

    once = merge_labeled(base, base)
    twice = merge_labeled(once, base)

    assert len(once) == len(base)
    assert [(e.doc_id, e.path, e.correct) for e in twice] == [
        (e.doc_id, e.path, e.correct) for e in base
    ]


def test_merge_preserves_base_order() -> None:
    """A merge must not reshuffle the set, or the labeled-set fingerprint churns
    for no reason."""
    base = [_row(f"d{i}", "a", correct=True) for i in range(5)]
    new = [_row("d2", "a", correct=False), _row("d9", "a", correct=True)]

    merged = merge_labeled(base, new)

    assert [e.doc_id for e in merged] == ["d0", "d1", "d2", "d3", "d4", "d9"]


def test_unidentifiable_rows_are_never_deduplicated() -> None:
    """Two hand-written rows carrying no ids cannot be told apart, so neither is
    silently dropped."""
    base = [_row(None, None, correct=True)]
    new = [_row(None, None, correct=True)]

    assert len(merge_labeled(base, new)) == 2


# -------------------------------------------------------------------- splitting


def test_split_is_deterministic_for_a_seed() -> None:
    examples = synthetic_labeled()

    a_train, a_test = split_labeled(examples, holdout=0.25, seed=7)
    b_train, b_test = split_labeled(examples, holdout=0.25, seed=7)

    assert [e.correct for e in a_train] == [e.correct for e in b_train]
    assert [e.correct for e in a_test] == [e.correct for e in b_test]


def test_split_is_disjoint_and_complete() -> None:
    examples = synthetic_labeled()
    train, test = split_labeled(examples, holdout=0.25, seed=1)

    assert len(train) + len(test) == len(examples)
    assert {id(e) for e in train}.isdisjoint({id(e) for e in test})
    assert {id(e) for e in train} | {id(e) for e in test} == {id(e) for e in examples}


def test_different_seeds_give_different_splits() -> None:
    examples = synthetic_labeled()
    _, first = split_labeled(examples, holdout=0.25, seed=1)
    _, second = split_labeled(examples, holdout=0.25, seed=2)

    assert [id(e) for e in first] != [id(e) for e in second]


@pytest.mark.parametrize("holdout", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_an_out_of_range_holdout(holdout: float) -> None:
    with pytest.raises(ValueError, match="holdout"):
        split_labeled(synthetic_labeled(), holdout=holdout, seed=0)


def test_split_rejects_a_holdout_that_empties_a_side() -> None:
    tiny = [_row("d1", "a", correct=True), _row("d2", "a", correct=False)]
    with pytest.raises(ValueError, match="empty split"):
        split_labeled(tiny, holdout=0.01, seed=0)


# --------------------------------------------------------------------- metrics


def test_brier_matches_hand_computed_values() -> None:
    # (0.9-1)^2 + (0.2-0)^2 + (0.5-1)^2 = 0.01 + 0.04 + 0.25 = 0.30, over 3.
    assert brier_score([0.9, 0.2, 0.5], [True, False, True]) == pytest.approx(0.30 / 3)


def test_brier_is_zero_for_perfect_confident_predictions() -> None:
    assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)


def test_brier_is_one_for_confidently_wrong_predictions() -> None:
    assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)


def test_brier_punishes_confident_errors_harder_than_unsure_ones() -> None:
    confident_wrong = brier_score([0.99], [False])
    unsure_wrong = brier_score([0.55], [False])
    assert confident_wrong > unsure_wrong


def test_ece_is_zero_for_a_perfectly_calibrated_set() -> None:
    # Bin [0.6, 0.7): mean confidence 0.65, and exactly 65 of 100 are correct.
    probs = [0.65] * 100
    labels = [True] * 65 + [False] * 35
    assert expected_calibration_error(probs, labels, bins=10) == pytest.approx(0.0)


def test_ece_catches_systematic_overconfidence() -> None:
    # Says 0.95, right half the time.
    probs = [0.95] * 100
    labels = [True] * 50 + [False] * 50
    assert expected_calibration_error(probs, labels, bins=10) == pytest.approx(0.45)


def test_ece_weights_bins_by_their_share() -> None:
    # 90 well-calibrated rows and 10 wildly overconfident ones: the small bin's
    # gap of 1.0 contributes only its 10% share.
    probs = [0.5] * 90 + [1.0] * 10
    labels = [True] * 45 + [False] * 45 + [False] * 10
    assert expected_calibration_error(probs, labels, bins=10) == pytest.approx(0.10)


def test_a_prediction_of_exactly_one_lands_in_the_last_bin() -> None:
    """Off-by-one at the top edge would silently drop the most confident rows."""
    assert expected_calibration_error([1.0], [True], bins=10) == pytest.approx(0.0)
    assert expected_calibration_error([1.0], [False], bins=10) == pytest.approx(1.0)


def test_single_example_sets_are_scoreable() -> None:
    assert brier_score([0.25], [False]) == pytest.approx(0.0625)
    assert math.isfinite(expected_calibration_error([0.25], [False]))


@pytest.mark.parametrize("fn", [brier_score, expected_calibration_error])
def test_metrics_reject_empty_and_mismatched_input(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        fn([], [])
    with pytest.raises(ValueError):
        fn([0.5], [True, False])


def test_ece_rejects_a_bin_count_below_one() -> None:
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error([0.5], [True], bins=0)


# ------------------------------------------------------------------ evaluation


def test_evaluate_scores_a_fitted_calibrator_on_held_out_rows() -> None:
    examples = synthetic_labeled()
    train, test = split_labeled(examples, holdout=0.25, seed=3)
    cal = AgreementCalibrator(method="isotonic").fit(train)

    scores = evaluate_calibrator(cal, test)

    assert scores.n == len(test)
    assert 0.0 <= scores.brier <= 1.0
    assert 0.0 <= scores.ece <= 1.0


def test_evaluate_rejects_an_empty_set() -> None:
    cal = AgreementCalibrator(method="isotonic").fit(synthetic_labeled())
    with pytest.raises(ValueError, match="empty"):
        evaluate_calibrator(cal, [])


def _monotone_labeled(seed: int, n_per_share: int) -> list[LabeledExample]:
    """A clean set where accuracy rises with agreement, no correlated errors.

    ``synthetic_labeled`` deliberately breaks monotonicity at share=1.0, which is
    the right shape for testing honesty but the wrong one for asking whether more
    data helps.
    """
    rng = random.Random(seed)
    rows: list[LabeledExample] = []
    for share in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
        for _ in range(n_per_share):
            rows.append(
                LabeledExample(
                    features=AgreementFeatures(winning_share=share, k=4, entropy=0.0),
                    correct=rng.random() < share,
                )
            )
    return rows


def test_more_labels_beat_fewer_on_the_same_holdout() -> None:
    """The property the whole feature exists to expose: a calibrator fit on more
    of the same distribution scores better on held-out rows.

    Averaged over seeds rather than asserted on one: a single thin sample can get
    lucky, and a test that depends on that luck is a test that flakes.
    """
    thin_total = 0.0
    full_total = 0.0
    seeds = range(10)
    for seed in seeds:
        train = _monotone_labeled(seed, n_per_share=100)
        test = _monotone_labeled(1000 + seed, n_per_share=60)
        thin = random.Random(seed).sample(train, 60)

        # Guards relaxed on purpose: this measures fit quality, not the honesty
        # guards, and a 60-row sample trips the per-bin minimum by design.
        def fit(rows: list[LabeledExample]) -> AgreementCalibrator:
            return AgreementCalibrator(method="isotonic", min_examples=20, min_per_bin=1).fit(rows)

        thin_total += evaluate_calibrator(fit(thin), test).brier
        full_total += evaluate_calibrator(fit(train), test).brier

    assert full_total / len(seeds) < thin_total / len(seeds)


def test_scores_round_trip_through_a_dict() -> None:
    examples = synthetic_labeled()
    train, test = split_labeled(examples, holdout=0.25, seed=3)
    cal = AgreementCalibrator(method="isotonic").fit(train)

    payload = evaluate_calibrator(cal, test).to_dict()

    assert set(payload) == {"n", "brier", "ece"}
    assert payload["n"] == len(test)
