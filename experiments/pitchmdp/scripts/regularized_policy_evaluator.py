"""One July-only regularized residual-logistic evaluator; no neural fitting."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.model import CountBaseline, outcome_labels


class RegularizedPolicyEvaluator:
    """Count/hand log-probability offsets plus shared pitch-type/context effects.

    All counts, categories, normalization, and residual coefficients use July
    1–21. July 22–31 selects one scalar convex mixture without any later refit.
    No player IDs or realized pitch physics enter the feature matrix.
    """
    SUPPORT_KEYS = [*CountBaseline.keys, "pitch_type"]
    PROFILE_COLUMNS = [*STYLE_COLUMNS, *RELIABILITY_COLUMNS]
    NUMERIC_NAMES = ["balls", "strikes", "outs", "inning", "score_difference",
                     "top_half", "first_base", "second_base", "third_base",
                     "batter_right", "pitcher_right", *PROFILE_COLUMNS]

    def __init__(self, l2=.05, max_training_rows=80000):
        if l2 != .05:
            raise ValueError("This bounded candidate fixes L2 at 0.05; no setting search")
        if type(max_training_rows) is not int or not 1 <= max_training_rows <= 80000:
            raise ValueError("max_training_rows must be an integer in 1..80000")
        self.l2, self.max_training_rows = float(l2), max_training_rows

    def _numeric(self, frame):
        required = {"balls", "strikes", "outs_when_up", "inning", "home_score", "away_score",
                    "inning_topbot", "bases", "stand", "p_throws", "pitch_type", *self.PROFILE_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing evaluator context: {sorted(missing)}")
        if not frame.inning_topbot.isin(["Top", "Bot"]).all() or any(
            not frame[key].isin(["L", "R"]).all() for key in ("stand", "p_throws")
        ) or frame.pitch_type.isna().any():
            raise ValueError("Invalid evaluator side or pitch type")
        integer = {}
        for key, low, high in [("balls", 0, 3), ("strikes", 0, 2), ("outs_when_up", 0, 2),
                               ("bases", 0, 7), ("inning", 1, 99),
                               ("home_score", 0, 999), ("away_score", 0, 999)]:
            values = frame[key].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values % 1 != 0).any() or (values < low).any() or (values > high).any():
                raise ValueError(f"Invalid evaluator {key}")
            integer[key] = values
        bases = integer["bases"].astype(int)
        profiles = frame[self.PROFILE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(profiles).all() or (profiles < 0).any() or (
            profiles > np.array([1, 1, 1, 1, 3, 1, *([1] * 6)])
        ).any():
            raise ValueError("Invalid continuous batter profile")
        return np.column_stack([integer["balls"], integer["strikes"], integer["outs_when_up"],
            integer["inning"], integer["home_score"] - integer["away_score"],
            frame.inning_topbot.eq("Top").to_numpy(float), bases & 1, (bases >> 1) & 1,
            (bases >> 2) & 1, frame.stand.eq("R").to_numpy(float), frame.p_throws.eq("R").to_numpy(float),
            profiles])

    def _features(self, frame):
        numeric = (self._numeric(frame) - self.mean) / self.scale
        kinds = frame.pitch_type.astype(str).to_numpy()
        indicators = np.column_stack([kinds == kind for kind in self.pitch_types])
        return np.ascontiguousarray(np.column_stack([np.ones(len(frame)), numeric, indicators]), dtype=float)

    @staticmethod
    def _loss(probabilities, y):
        return float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-30, 1)).mean())

    def fit(self, frame):
        if not len(frame) or "game_date" not in frame or "game_pk" not in frame:
            raise ValueError("Evaluator fitting requires dated July rows and game IDs")
        dates = pd.to_datetime(frame.game_date, errors="raise").dt.normalize()
        if dates.isna().any() or not dates.between("2025-07-01", "2025-07-31").all():
            raise ValueError("Independent evaluator fitting is restricted to July 2025")
        if frame.game_pk.isna().any():
            raise ValueError("Missing evaluator game IDs")
        y = outcome_labels(frame)
        if (y < 0).any():
            raise ValueError("Fit only eligible, labeled July pitches")
        self._numeric(frame)  # Validate required context before fitting anything.
        keys = ["game_date", "game_pk", *[key for key in ("at_bat_number", "pitch_number") if key in frame]]
        train = frame.loc[dates.le("2025-07-21")].sort_values(keys, kind="stable")
        calibration = frame.loc[dates.ge("2025-07-22")].sort_values(keys, kind="stable")
        if not len(train) or not len(calibration):
            raise ValueError("Need both July 1–21 fitting and July 22–31 calibration rows")
        fit_games = sorted(int(game) for game in train.game_pk.unique())
        calibration_games = sorted(int(game) for game in calibration.game_pk.unique())
        if set(fit_games).intersection(calibration_games):
            raise ValueError("Fit/calibration games must be disjoint")
        self.parent = CountBaseline().fit(train)
        self.support_counts = train.groupby(self.SUPPORT_KEYS, observed=True).size().to_dict()
        self.pitch_types = sorted(train.pitch_type.astype(str).unique().tolist())
        numeric = self._numeric(train)
        self.mean, self.scale = numeric.mean(axis=0), np.maximum(numeric.std(axis=0), .01)
        if len(train) > self.max_training_rows:
            selected = np.sort(np.random.default_rng(42).choice(len(train), self.max_training_rows, replace=False))
            residual_train = train.iloc[selected]
        else:
            residual_train = train
        x = self._features(residual_train)
        labels = outcome_labels(residual_train)
        offset = np.log(self.parent.predict(residual_train))
        n_features = x.shape[1]

        def objective(flat):
            coefficients = flat.reshape(n_features, 10)
            logits = offset + x @ coefficients
            loss = (logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]).mean()
            gradient_logits = softmax(logits, axis=1)
            gradient_logits[np.arange(len(labels)), labels] -= 1
            gradient = x.T @ gradient_logits / len(labels) + self.l2 * coefficients
            return float(loss + .5 * self.l2 * np.square(coefficients).sum()), gradient.ravel()

        fitted = minimize(objective, np.zeros(n_features * 10), method="L-BFGS-B", jac=True,
                          options={"maxiter": 150, "ftol": 1e-9, "gtol": 1e-6})
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise RuntimeError(f"Regularized evaluator optimization failed: {fitted.message}")
        self.coefficients = fitted.x.reshape(n_features, 10)
        parent_p = self.parent.predict(calibration)
        adjusted = softmax(np.log(parent_p) + self._features(calibration) @ self.coefficients, axis=1)
        calibration_y = outcome_labels(calibration)

        def mixture_loss(weight):
            return self._loss((1 - weight) * parent_p + weight * adjusted, calibration_y)

        selected = minimize_scalar(mixture_loss, bounds=(0., 1.), method="bounded", options={"xatol": 1e-7})
        if not selected.success:
            raise RuntimeError("July-only scalar mixture calibration failed")
        self.weight = float(min((0., float(selected.x), 1.), key=mixture_loss))
        self.report = {"kind": "count_offset_regularized_multiclass_residual_logistic", "l2": self.l2,
            "count_prior_source": "All eligible July 1–21 rows only; CountBaseline pseudocounts fixed",
            "fit_date_min": pd.Timestamp(train.game_date.min()).date().isoformat(),
            "fit_date_max": pd.Timestamp(train.game_date.max()).date().isoformat(),
            "calibration_date_min": pd.Timestamp(calibration.game_date.min()).date().isoformat(),
            "calibration_date_max": pd.Timestamp(calibration.game_date.max()).date().isoformat(),
            "fit_game_ids": fit_games, "calibration_game_ids": calibration_games,
            "residual_fit_game_ids": sorted(int(game) for game in residual_train.game_pk.unique()),
            "count_fit_rows": len(train), "residual_fit_rows": len(residual_train),
            "calibration_rows": len(calibration), "sampling_seed": 42,
            "residual_weight": self.weight, "calibration_count_log_loss": mixture_loss(0.),
            "calibration_adjusted_log_loss": mixture_loss(1.), "calibration_mixed_log_loss": mixture_loss(self.weight),
            "feature_names": ["intercept", *self.NUMERIC_NAMES, *[f"pitch_type={kind}" for kind in self.pitch_types]],
            "optimizer_iterations": int(fitted.nit), "optimizer_success": bool(fitted.success),
            "support": "Observed first-July-21 count/hand/pitch-type cell count; zero for unseen cells",
            "origin": {"1": "observed action cell", "0": "known type, unseen cell", "-1": "unknown type"},
            "scope": "One fixed candidate; July-only fitting/calibration, no refit, no August outcomes, "
                     "no player-ID or realized-delivery features; independent guard validation still required"}
        return self

    def predict(self, frame):
        x = self._features(frame)
        if not len(frame):
            return np.empty((0, 10), dtype=float)
        parent = self.parent.predict(frame)
        adjusted = softmax(np.log(parent) + x @ self.coefficients, axis=1)
        probabilities = (1 - self.weight) * parent + self.weight * adjusted
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict_with_support(self, frame):
        probabilities = self.predict(frame)
        support = np.array([self.support_counts.get(key, 0) for key in
            frame[self.SUPPORT_KEYS].itertuples(index=False, name=None)], dtype=np.int64)
        known = frame.pitch_type.astype(str).isin(self.pitch_types).to_numpy()
        origin = np.where(support > 0, 1, np.where(known, 0, -1)).astype(np.int64)
        return probabilities, support, origin
