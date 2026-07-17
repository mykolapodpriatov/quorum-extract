"""Agreement -> calibrated probability of correctness.

One **global** calibrator maps agreement features to a real probability that a
reconciled field is correct (plan 3.4). Two honest methods:

* ``isotonic`` (default) -- monotonic, 1-D, consumes only ``winning_share``.
  ``sklearn.isotonic.IsotonicRegression`` guarantees that higher share never
  *decreases* predicted confidence.
* ``platt`` -- logistic regression over all three features (share, K, entropy).
  Because the features have mismatched magnitudes, a :class:`StandardScaler` is
  fit first and **persisted alongside the weights** so no feature is numerically
  dominated and load restores identical behavior.

Honesty guards (``fit`` *refuses* rather than emitting a degenerate mapping):
require enough examples, both classes, and a minimum count per agreement decile.
A calibrator trained on a labeled set that includes *high-agreement-but-wrong*
cases learns that high agreement is not certainty when the corpus has correlated
errors -- so confidence does not spuriously approach 1.

Determinism: fit is seeded; persistence is JSON keyed by a fingerprint that
records the schema, extractor set, and the **labeled-set file hash** so reusing
a calibrator against a different config/labeled set is detectable and warned.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .agreement import FEATURE_NAMES, AgreementFeatures

CalibrationMethod = Literal["isotonic", "platt"]

#: Default honesty thresholds (plan 3.4).
DEFAULT_MIN_EXAMPLES = 50
DEFAULT_MIN_PER_BIN = 5
N_DECILE_BINS = 10

#: Fixed RNG seed so fits are reproducible.
RANDOM_STATE = 1234

#: Sentinel group name for the single global calibrator.
GLOBAL_GROUP = "__global__"

_FORMAT_VERSION = 1


class CalibrationError(ValueError):
    """Raised when a calibrator cannot be honestly fit (guards failed)."""


@dataclass(frozen=True, slots=True)
class LabeledExample:
    """One labeled calibration row: features + ground-truth correctness.

    ``group`` selects an optional per-group calibrator; ``None`` uses the global
    one.
    """

    features: AgreementFeatures
    correct: bool
    group: str | None = None


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """Identifies the configuration a calibrator was trained against.

    A mismatch on load warns (a calibrator is not transferable by default).
    """

    schema_hash: str = ""
    extractor_set_hash: str = ""
    labeled_set_hash: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_hash": self.schema_hash,
            "extractor_set_hash": self.extractor_set_hash,
            "labeled_set_hash": self.labeled_set_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> Fingerprint:
        return cls(
            schema_hash=d.get("schema_hash", ""),
            extractor_set_hash=d.get("extractor_set_hash", ""),
            labeled_set_hash=d.get("labeled_set_hash", ""),
        )


def hash_text(text: str) -> str:
    """Stable short hash used for fingerprint components."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fingerprint_for(
    *, schema_paths: Sequence[str], extractor_names: Sequence[str], labeled_path: str | None
) -> Fingerprint:
    """Build a :class:`Fingerprint` from a run's identifying parts.

    The labeled-set hash is the hash of the file *contents* when a path is given
    (so a changed labeled set is detected), else empty.
    """
    schema_hash = hash_text("|".join(sorted(schema_paths)))
    ext_hash = hash_text("|".join(sorted(extractor_names)))
    labeled_hash = ""
    if labeled_path:
        p = Path(labeled_path)
        if p.exists():
            labeled_hash = hash_text(p.read_text(encoding="utf-8"))
    return Fingerprint(
        schema_hash=schema_hash,
        extractor_set_hash=ext_hash,
        labeled_set_hash=labeled_hash,
    )


def _check_guards(
    shares: np.ndarray, labels: np.ndarray, *, min_examples: int, min_per_bin: int
) -> None:
    """Enforce the honesty guards; raise :class:`CalibrationError` on failure."""
    n = len(labels)
    if n < min_examples:
        raise CalibrationError(
            f"need at least {min_examples} labeled examples to calibrate honestly, got {n}"
        )
    classes = np.unique(labels)
    if classes.size < 2:
        only = bool(classes[0]) if classes.size else None
        raise CalibrationError(
            "calibration requires both correct and incorrect examples; "
            f"the labeled set has only one class (correct={only})"
        )
    # Minimum count per agreement decile bin so the fit generalizes.
    bins = np.clip((shares * N_DECILE_BINS).astype(int), 0, N_DECILE_BINS - 1)
    occupied = np.bincount(bins, minlength=N_DECILE_BINS)
    sparse = [i for i, c in enumerate(occupied) if 0 < c < min_per_bin]
    if sparse:
        raise CalibrationError(
            "some agreement decile bins are too sparse to calibrate honestly "
            f"(bins {sparse} have fewer than {min_per_bin} examples); "
            "collect more labeled data spanning the agreement range"
        )


@dataclass
class _GroupModel:
    """Internal per-group fitted model (isotonic or platt)."""

    method: CalibrationMethod
    isotonic: IsotonicRegression | None = None
    logistic: LogisticRegression | None = None
    scaler: StandardScaler | None = None

    def predict(self, rows: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            assert self.isotonic is not None
            shares = rows[:, 0]
            out = self.isotonic.predict(shares)
            return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)
        assert self.logistic is not None and self.scaler is not None
        scaled = self.scaler.transform(rows)
        proba = self.logistic.predict_proba(scaled)[:, 1]
        return np.clip(np.asarray(proba, dtype=np.float64), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method}
        if self.method == "isotonic":
            assert self.isotonic is not None
            iso = self.isotonic
            d["isotonic"] = {
                "x_thresholds": np.asarray(iso.X_thresholds_).tolist(),
                "y_thresholds": np.asarray(iso.y_thresholds_).tolist(),
                "x_min": float(iso.X_min_),
                "x_max": float(iso.X_max_),
                "increasing": bool(iso.increasing_),
            }
        else:
            assert self.logistic is not None and self.scaler is not None
            d["platt"] = {
                "coef": np.asarray(self.logistic.coef_).tolist(),
                "intercept": np.asarray(self.logistic.intercept_).tolist(),
                "classes": np.asarray(self.logistic.classes_).tolist(),
                "scaler_mean": np.asarray(self.scaler.mean_).tolist(),
                "scaler_scale": np.asarray(self.scaler.scale_).tolist(),
                "n_features": int(self.scaler.n_features_in_),
            }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> _GroupModel:
        method: CalibrationMethod = d["method"]
        if method == "isotonic":
            payload = d["isotonic"]
            iso = IsotonicRegression(out_of_bounds="clip")
            xt = np.asarray(payload["x_thresholds"], dtype=float)
            yt = np.asarray(payload["y_thresholds"], dtype=float)
            # Reconstruct the fitted interpolator from persisted thresholds.
            iso.X_thresholds_ = xt
            iso.y_thresholds_ = yt
            iso.X_min_ = float(payload["x_min"])
            iso.X_max_ = float(payload["x_max"])
            iso.increasing_ = bool(payload["increasing"])
            from scipy.interpolate import interp1d  # local import; scipy via sklearn

            iso.f_ = interp1d(xt, yt, kind="linear", bounds_error=False, fill_value=(yt[0], yt[-1]))
            return cls(method=method, isotonic=iso)
        payload = d["platt"]
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
        scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = int(payload["n_features"])
        log = LogisticRegression()
        log.coef_ = np.asarray(payload["coef"], dtype=float)
        log.intercept_ = np.asarray(payload["intercept"], dtype=float)
        log.classes_ = np.asarray(payload["classes"])
        log.n_features_in_ = int(payload["n_features"])
        return cls(method=method, logistic=log, scaler=scaler)


class AgreementCalibrator:
    """Maps agreement features to a calibrated probability of correctness.

    Use :meth:`fit` to train (raises :class:`CalibrationError` if the data is too
    weak), :meth:`predict_one` / :meth:`predict` to score, and
    :meth:`save` / :meth:`load` to persist. Optional per-group calibrators fall
    back to the global model when a group lacks enough data.
    """

    def __init__(
        self,
        method: CalibrationMethod = "isotonic",
        *,
        fingerprint: Fingerprint | None = None,
        min_examples: int = DEFAULT_MIN_EXAMPLES,
        min_per_bin: int = DEFAULT_MIN_PER_BIN,
    ) -> None:
        self.method: CalibrationMethod = method
        self.fingerprint = fingerprint or Fingerprint()
        self.min_examples = min_examples
        self.min_per_bin = min_per_bin
        self._models: dict[str, _GroupModel] = {}

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        examples: Sequence[LabeledExample],
        *,
        features: Sequence[str] | None = None,
    ) -> AgreementCalibrator:
        """Fit the global (and any per-group) calibrator from labeled examples.

        Args:
            examples: Labeled rows (features + ground-truth correctness).
            features: Optional explicit feature selection (subset of
                :data:`~quorum_extract.agreement.FEATURE_NAMES`). ``isotonic`` is
                strictly 1-D and therefore rejects any selection other than
                ``("winning_share",)``; ``platt`` defaults to all features.

        Raises:
            CalibrationError: if the global model fails an honesty guard, if an
                unknown feature is requested, or if ``isotonic`` is asked to use
                more than the single ``winning_share`` feature.
        """
        if not examples:
            raise CalibrationError("no labeled examples provided")

        self._validate_features(features)

        # Global model from all rows.
        self._models = {}
        self._models[GLOBAL_GROUP] = self._fit_group(examples)

        # Optional per-group models; degenerate groups fall back to global.
        groups: dict[str, list[LabeledExample]] = {}
        for ex in examples:
            if ex.group is not None:
                groups.setdefault(ex.group, []).append(ex)
        for gname, gex in groups.items():
            try:
                self._models[gname] = self._fit_group(gex)
            except CalibrationError as exc:
                warnings.warn(
                    f"calibration group {gname!r} falls back to the global calibrator: {exc}",
                    stacklevel=2,
                )
        return self

    def _validate_features(self, features: Sequence[str] | None) -> None:
        """Reject unknown features and multi-feature isotonic.

        ``IsotonicRegression`` is genuinely 1-D, so the monotonicity guarantee
        only holds for the single ``winning_share`` feature; allowing more would
        silently break that contract.
        """
        if features is None:
            return
        unknown = [f for f in features if f not in FEATURE_NAMES]
        if unknown:
            raise CalibrationError(
                f"unknown calibration feature(s) {unknown}; valid: {list(FEATURE_NAMES)}"
            )
        if self.method == "isotonic" and list(features) != ["winning_share"]:
            raise CalibrationError(
                "isotonic calibration is 1-D and only supports the "
                f"'winning_share' feature; got {list(features)}. Use method='platt' "
                "for multi-feature calibration."
            )

    def _fit_group(self, examples: Sequence[LabeledExample]) -> _GroupModel:
        rows = np.array([ex.features.as_row() for ex in examples], dtype=float)
        labels = np.array([1 if ex.correct else 0 for ex in examples], dtype=int)
        shares = rows[:, 0]
        _check_guards(
            shares,
            labels,
            min_examples=self.min_examples,
            min_per_bin=self.min_per_bin,
        )
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True, y_min=0.0, y_max=1.0)
            iso.fit(shares, labels)
            return _GroupModel(method="isotonic", isotonic=iso)
        # Platt: standardize, then logistic over all features.
        scaler = StandardScaler()
        scaled = scaler.fit_transform(rows)
        log = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        log.fit(scaled, labels)
        return _GroupModel(method="platt", logistic=log, scaler=scaler)

    # -- prediction --------------------------------------------------------
    def predict(
        self, features: Sequence[AgreementFeatures], groups: Sequence[str | None] | None = None
    ) -> list[float]:
        """Score a batch of feature vectors. Missing/unknown groups use global.

        Raises:
            CalibrationError: if the calibrator is not fitted, or if ``groups`` is
                provided with a length that does not match ``features`` (a length
                mismatch would silently drop tail feature vectors).
        """
        if not self._models:
            raise CalibrationError("calibrator is not fitted")
        if groups is None:
            groups = [None] * len(features)
        elif len(groups) != len(features):
            raise CalibrationError(
                "predict(): groups and features length mismatch "
                f"(len(groups)={len(groups)}, len(features)={len(features)}); "
                "pass exactly one group per feature vector (or None)"
            )
        rows = np.array([f.as_row() for f in features], dtype=float)
        out: list[float] = []
        for i in range(len(features)):
            model = self._models.get(groups[i] or GLOBAL_GROUP) or self._models[GLOBAL_GROUP]
            out.append(float(model.predict(rows[i : i + 1])[0]))
        return out

    def predict_one(self, features: AgreementFeatures, group: str | None = None) -> float:
        """Score a single feature vector."""
        return self.predict([features], [group])[0]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": _FORMAT_VERSION,
            "method": self.method,
            "min_examples": self.min_examples,
            "min_per_bin": self.min_per_bin,
            "feature_names": list(FEATURE_NAMES),
            "fingerprint": self.fingerprint.to_dict(),
            "models": {name: m.to_dict() for name, m in self._models.items()},
        }

    def save(self, path: str | Path) -> None:
        """Persist the calibrator to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgreementCalibrator:
        cal = cls(
            method=d["method"],
            fingerprint=Fingerprint.from_dict(d.get("fingerprint", {})),
            min_examples=int(d.get("min_examples", DEFAULT_MIN_EXAMPLES)),
            min_per_bin=int(d.get("min_per_bin", DEFAULT_MIN_PER_BIN)),
        )
        cal._models = {name: _GroupModel.from_dict(m) for name, m in d.get("models", {}).items()}
        return cal

    @classmethod
    def load(cls, path: str | Path, *, expected: Fingerprint | None = None) -> AgreementCalibrator:
        """Load a calibrator; warn on a fingerprint mismatch with ``expected``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = cls.from_dict(data)
        if expected is not None and cal.fingerprint != expected:
            warnings.warn(
                "loaded calibrator fingerprint does not match the current "
                "config/labeled set; confidence may be invalid. "
                f"(calibrator={cal.fingerprint}, expected={expected})",
                stacklevel=2,
            )
        return cal

    @property
    def is_fitted(self) -> bool:
        return bool(self._models)

    @property
    def group_names(self) -> list[str]:
        """Names of fitted per-group models (excluding the global sentinel)."""
        return sorted(n for n in self._models if n != GLOBAL_GROUP)
