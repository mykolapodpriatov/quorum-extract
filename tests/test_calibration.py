"""Tests for agreement->confidence calibration (isotonic + Platt).

Covers monotonicity, empirical-accuracy agreement on held-out data, the
adversarial high-agreement-but-WRONG case, the honesty guards, isotonic's
multi-feature rejection, persist/load roundtrip, and fingerprint behavior.
"""

from __future__ import annotations

import random
import warnings

import pytest

from quorum_extract import (
    AgreementCalibrator,
    AgreementFeatures,
    CalibrationError,
    Fingerprint,
    LabeledExample,
    fingerprint_for,
)

from ._helpers import synthetic_labeled


def feat(share: float, k: int = 4, entropy: float = 0.0) -> AgreementFeatures:
    return AgreementFeatures(winning_share=share, k=k, entropy=entropy)


# --------------------------------------------------------------------------- #
# Isotonic monotonicity over winning_share (1-D)
# --------------------------------------------------------------------------- #


def test_isotonic_monotone_over_share() -> None:
    cal = AgreementCalibrator("isotonic").fit(synthetic_labeled())
    shares = [i / 20 for i in range(21)]
    preds = [cal.predict_one(feat(s)) for s in shares]
    assert all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1))


def test_isotonic_predictions_in_unit_interval() -> None:
    cal = AgreementCalibrator("isotonic").fit(synthetic_labeled())
    for s in (0.0, 0.3, 0.5, 0.9, 1.0):
        p = cal.predict_one(feat(s))
        assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# Calibrated confidence ~ empirical accuracy on HELD-OUT data
# --------------------------------------------------------------------------- #


def test_confidence_matches_empirical_accuracy_held_out() -> None:
    rng = random.Random(7)

    def make_set(n_per: int) -> tuple[list[LabeledExample], dict[float, float]]:
        plan = {0.25: 0.30, 0.5: 0.55, 0.75: 0.80, 1.0: 0.78}
        rows: list[LabeledExample] = []
        for share, p in plan.items():
            for _ in range(n_per):
                rows.append(LabeledExample(feat(share), rng.random() < p))
        return rows, plan

    train, _ = make_set(200)
    holdout, true_rates = make_set(400)
    cal = AgreementCalibrator("isotonic").fit(train)

    # For each share bucket in the holdout, compare predicted vs empirical.
    by_share: dict[float, list[bool]] = {}
    for ex in holdout:
        by_share.setdefault(ex.features.winning_share, []).append(ex.correct)
    for share, labels in by_share.items():
        empirical = sum(labels) / len(labels)
        predicted = cal.predict_one(feat(share))
        assert abs(predicted - empirical) < 0.12, (share, predicted, empirical)
    # Sanity: the true generative rates are reflected (not exactly, but close).
    assert true_rates[1.0] < 0.95


# --------------------------------------------------------------------------- #
# Adversarial: all-K-agree but WRONG => confidence not spuriously ~1
# --------------------------------------------------------------------------- #


def test_high_agreement_wrong_not_overconfident() -> None:
    """The labeled set has many unanimous-but-wrong rows (correlated errors).

    The calibrator must learn that share=1.0 is far from certain.
    """
    rng = random.Random(11)
    rows: list[LabeledExample] = []
    # Mid agreement is honest; high agreement is poisoned (40% wrong).
    for _ in range(80):
        rows.append(LabeledExample(feat(0.5), rng.random() < 0.5))
    for _ in range(80):
        rows.append(LabeledExample(feat(0.75), rng.random() < 0.7))
    for _ in range(120):
        rows.append(LabeledExample(feat(1.0), rng.random() < 0.60))  # correlated errors

    cal = AgreementCalibrator("isotonic").fit(rows)
    conf_at_1 = cal.predict_one(feat(1.0))
    assert conf_at_1 < 0.8, conf_at_1
    # And Platt agrees it is not ~1 either.
    cal_p = AgreementCalibrator("platt").fit(rows)
    assert cal_p.predict_one(feat(1.0)) < 0.85


# --------------------------------------------------------------------------- #
# Honesty guards
# --------------------------------------------------------------------------- #


def test_too_few_examples_raises() -> None:
    rows = [LabeledExample(feat(0.5), i % 2 == 0) for i in range(10)]
    with pytest.raises(CalibrationError, match="at least 50"):
        AgreementCalibrator("isotonic", min_examples=50).fit(rows)


def test_single_class_raises() -> None:
    rows = [LabeledExample(feat(0.5 + (i % 5) / 10), True) for i in range(80)]
    with pytest.raises(CalibrationError, match="both correct and incorrect"):
        AgreementCalibrator("isotonic").fit(rows)


def test_sparse_decile_bin_raises() -> None:
    rng = random.Random(3)
    rows: list[LabeledExample] = []
    # Plenty of data at 0.5, but only 2 rows at share 0.95 (sparse bin).
    for _ in range(80):
        rows.append(LabeledExample(feat(0.5), rng.random() < 0.5))
    rows.append(LabeledExample(feat(0.95), True))
    rows.append(LabeledExample(feat(0.95), False))
    with pytest.raises(CalibrationError, match="too sparse"):
        AgreementCalibrator("isotonic", min_per_bin=5).fit(rows)


def test_empty_examples_raises() -> None:
    with pytest.raises(CalibrationError, match="no labeled examples"):
        AgreementCalibrator("isotonic").fit([])


# --------------------------------------------------------------------------- #
# Isotonic rejects multi-feature
# --------------------------------------------------------------------------- #


def test_isotonic_rejects_multi_feature() -> None:
    rows = synthetic_labeled()
    with pytest.raises(CalibrationError, match="1-D"):
        AgreementCalibrator("isotonic").fit(rows, features=["winning_share", "k"])


def test_isotonic_accepts_explicit_share_only() -> None:
    rows = synthetic_labeled()
    cal = AgreementCalibrator("isotonic").fit(rows, features=["winning_share"])
    assert cal.is_fitted


def test_unknown_feature_raises() -> None:
    rows = synthetic_labeled()
    with pytest.raises(CalibrationError, match="unknown calibration feature"):
        AgreementCalibrator("platt").fit(rows, features=["bogus"])


def test_platt_accepts_multi_feature() -> None:
    rows = synthetic_labeled()
    cal = AgreementCalibrator("platt").fit(rows, features=["winning_share", "k", "entropy"])
    assert cal.is_fitted


# --------------------------------------------------------------------------- #
# Persist / load roundtrip
# --------------------------------------------------------------------------- #


def test_isotonic_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cal = AgreementCalibrator("isotonic").fit(synthetic_labeled())
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = AgreementCalibrator.load(path)
    for s in (0.2, 0.5, 0.7, 1.0):
        assert abs(cal.predict_one(feat(s)) - loaded.predict_one(feat(s))) < 1e-9


def test_platt_roundtrip_restores_scaler(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cal = AgreementCalibrator("platt").fit(synthetic_labeled())
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = AgreementCalibrator.load(path)
    for s in (0.2, 0.5, 1.0):
        f = feat(s, k=4, entropy=0.3)
        assert abs(cal.predict_one(f) - loaded.predict_one(f)) < 1e-9


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #


def test_fingerprint_records_labeled_hash(tmp_path) -> None:  # type: ignore[no-untyped-def]
    labeled = tmp_path / "labels.jsonl"
    labeled.write_text('{"winning_share": 1.0, "k": 4, "correct": true}\n', encoding="utf-8")
    fp = fingerprint_for(
        schema_paths=["a", "b"], extractor_names=["e1", "e2"], labeled_path=str(labeled)
    )
    assert fp.labeled_set_hash != ""
    assert fp.schema_hash != ""
    assert fp.extractor_set_hash != ""


def test_fingerprint_mismatch_warns_on_load(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fp = Fingerprint(schema_hash="aaa", extractor_set_hash="bbb", labeled_set_hash="ccc")
    cal = AgreementCalibrator("isotonic", fingerprint=fp).fit(synthetic_labeled())
    path = tmp_path / "cal.json"
    cal.save(path)
    other = Fingerprint(schema_hash="zzz", extractor_set_hash="bbb", labeled_set_hash="ccc")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AgreementCalibrator.load(path, expected=other)
    assert any("fingerprint" in str(w.message) for w in caught)


def test_fingerprint_match_no_warning(tmp_path) -> None:  # type: ignore[no-untyped-def]
    fp = Fingerprint(schema_hash="aaa", extractor_set_hash="bbb", labeled_set_hash="ccc")
    cal = AgreementCalibrator("isotonic", fingerprint=fp).fit(synthetic_labeled())
    path = tmp_path / "cal.json"
    cal.save(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        AgreementCalibrator.load(path, expected=fp)
    assert not any("fingerprint" in str(w.message) for w in caught)


# --------------------------------------------------------------------------- #
# Per-group calibration with global fallback
# --------------------------------------------------------------------------- #


def test_group_fallback_when_sparse() -> None:
    rng = random.Random(5)
    rows: list[LabeledExample] = []
    # A well-populated global signal across deciles.
    for share, p in [(0.25, 0.3), (0.5, 0.55), (0.75, 0.8), (1.0, 0.78)]:
        for _ in range(40):
            rows.append(LabeledExample(feat(share), rng.random() < p, group=None))
    # A sparse group "rare" with only a handful of rows -> must fall back.
    rows.append(LabeledExample(feat(1.0), True, group="rare"))
    rows.append(LabeledExample(feat(0.5), False, group="rare"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cal = AgreementCalibrator("isotonic", min_examples=50).fit(rows)
    assert any("falls back to the global" in str(w.message) for w in caught)
    # Predicting with the sparse group uses the global model (no crash).
    p_group = cal.predict_one(feat(1.0), group="rare")
    p_global = cal.predict_one(feat(1.0))
    assert p_group == p_global
    assert "rare" not in cal.group_names


def test_predict_unfitted_raises() -> None:
    cal = AgreementCalibrator("isotonic")
    with pytest.raises(CalibrationError, match="not fitted"):
        cal.predict_one(feat(1.0))


# --------------------------------------------------------------------------- #
# predict(): groups/features length must match (no silently-dropped tail)
# --------------------------------------------------------------------------- #


def test_predict_groups_features_length_mismatch_raises() -> None:
    cal = AgreementCalibrator("isotonic").fit(synthetic_labeled())
    features = [feat(0.5), feat(0.75), feat(1.0)]
    # A shorter groups list previously dropped the tail feature(s) silently.
    with pytest.raises(CalibrationError, match="length mismatch"):
        cal.predict(features, groups=[None])


def test_predict_returns_one_confidence_per_feature() -> None:
    cal = AgreementCalibrator("isotonic").fit(synthetic_labeled())
    features = [feat(0.25), feat(0.5), feat(0.75), feat(1.0)]
    # Equal-length groups: one confidence per feature, in order.
    out = cal.predict(features, groups=[None, None, None, None])
    assert len(out) == len(features)
    # Default (groups=None) also yields one-per-feature.
    assert len(cal.predict(features)) == len(features)
    # And matches per-feature predict_one.
    for f, conf in zip(features, out, strict=True):
        assert conf == cal.predict_one(f)
