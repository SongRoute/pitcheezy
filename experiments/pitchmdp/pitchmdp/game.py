"""Game-state transitions, empirical runner advancement and learned continuation.

The continuation model learns actual game winners from training games. Provider
win-expectancy columns are neither features nor labels. Regular MLB games under
the 2023--25 extra-inning automatic-runner rule are the supported game format.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .planner import TERMINALS


@dataclass(frozen=True)
class GameState:
    inning: int
    half: str
    outs: int
    bases: int
    home_score: int
    away_score: int
    winner: str | None = None

    def __post_init__(self):
        if self.inning < 1 or self.half not in ("Top", "Bot") or not 0 <= self.outs <= 2:
            raise ValueError("invalid inning, half or outs")
        if not 0 <= self.bases <= 7 or min(self.home_score, self.away_score) < 0:
            raise ValueError("invalid bases or score")
        if self.winner not in (None, "home", "away"):
            raise ValueError("winner must be home, away or None")

    @property
    def defender_is_home(self):
        return self.half == "Top"

    @classmethod
    def from_row(cls, row: Mapping):
        bases = row.get("bases")
        if bases is None or pd.isna(bases):
            bases = sum((1 << i) for i, col in enumerate(("on_1b", "on_2b", "on_3b")) if pd.notna(row.get(col)))
        return cls(int(row["inning"]), str(row["inning_topbot"]), int(row["outs_when_up"]),
                   int(bases), int(row["home_score"]), int(row["away_score"]))


def _advance(state: GameState, outs_added: int, bases: int, runs: int, *, home_run=False) -> GameState:
    if state.winner is not None:
        return state
    if outs_added < 0 or runs < 0 or state.outs + outs_added > 3:
        raise ValueError("invalid terminal advancement")
    home = state.home_score + (runs if state.half == "Bot" else 0)
    away = state.away_score + (runs if state.half == "Top" else 0)
    if state.half == "Bot" and state.inning >= 9 and home > away:
        # All runs count on a home run. Other walkoffs end at the winning run.
        home = home if home_run else min(home, away + 1)
        return replace(state, home_score=home, away_score=away, winner="home", bases=bases,
                       outs=min(state.outs + outs_added, 2))
    outs = state.outs + outs_added
    if outs < 3:
        return replace(state, outs=outs, bases=bases, home_score=home, away_score=away)
    if state.half == "Top":
        if state.inning >= 9 and home > away:
            return replace(state, outs=2, bases=0, home_score=home, away_score=away, winner="home")
        return GameState(state.inning, "Bot", 0, 2 if state.inning >= 10 else 0, home, away)
    if state.inning >= 9 and away > home:
        return replace(state, outs=2, bases=0, home_score=home, away_score=away, winner="away")
    inning = state.inning + 1
    return GameState(inning, "Top", 0, 2 if inning >= 10 else 0, home, away)


def apply_terminal(state: GameState, event: str) -> GameState:
    """Deterministic fallback; hits advance each runner exactly hit-distance.

    Ordinary outs hold runners. With a runner on first and fewer than two outs,
    a double play retires batter and first-base runner. Otherwise its probability
    is projected to an ordinary out. EmpiricalAdvancement replaces those coarse
    approximations whenever observed transitions for the base/out/event exist.
    """
    if state.winner is not None:
        return state
    if event in ("walk", "hbp"):
        bases = state.bases
        runs = int(bases == 7)
        if bases & 1:
            if bases & 2:
                bases |= 4
            bases |= 2
        bases |= 1
        return _advance(state, 0, bases, runs)
    if event in ("out", "strikeout"):
        return _advance(state, 1, state.bases, 0)
    if event == "double_play":
        if state.outs < 2 and state.bases & 1:
            return _advance(state, 2, state.bases & ~1, 0)
        return _advance(state, 1, state.bases, 0)
    distance = {"single": 1, "double": 2, "triple": 3, "home_run": 4}.get(event)
    if distance is None:
        raise ValueError(f"unsupported terminal event: {event}")
    bases, runs = (0, 1) if distance == 4 else (1 << (distance - 1), 0)
    for base_index in range(3):
        if state.bases & (1 << base_index):
            destination = base_index + 1 + distance
            if destination >= 4:
                runs += 1
            else:
                bases |= 1 << (destination - 1)
    return _advance(state, 0, bases, runs, home_run=event == "home_run")


_EVENT_GROUP = {
    "field_out": "out", "force_out": "out", "fielders_choice_out": "out",
    "sac_fly": "out", "sac_bunt": "out", "other_out": "out", "out": "out",
    "grounded_into_double_play": "double_play", "double_play": "double_play",
    "sac_fly_double_play": "double_play", "single": "single", "double": "double",
    "triple": "triple", "home_run": "home_run", "walk": "walk",
    "intent_walk": "walk", "hit_by_pitch": "hbp", "hbp": "hbp", "strikeout": "strikeout",
}


class EmpiricalAdvancement:
    """Train-only event/base/out runner transition frequencies with fallback.

    Counts are pooled over league, batters and pitchers. Five fallback pseudo-
    observations shrink small cells. Fitting excludes observed walkoffs, whose
    official run/base truncation does not transfer to non-walkoff situations.
    """
    def __init__(self, prior_strength: float = 5.0):
        self.prior_strength = float(prior_strength)
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be nonnegative")
        self.counts = {}
        self.report = {}

    def fit(self, frame: pd.DataFrame):
        data = frame
        if "split" in data:
            data = data.loc[data["split"].eq("train")]
        if "supported_pa" in data:
            data = data.loc[data["supported_pa"].fillna(False)]
        if "is_pa_terminal" in data:
            data = data.loc[data["is_pa_terminal"].fillna(False)]
        required = ["next_outs", "next_bases", "next_home_score", "next_away_score", "next_inning", "next_half"]
        candidates = len(data)
        data = data.dropna(subset=required)
        counts = defaultdict(Counter)
        used = 0
        event_col = "terminal_event" if "terminal_event" in data else "events"
        cols = ["inning", "inning_topbot", "outs_when_up", "bases", "home_score", "away_score", event_col, *required]
        for row in data[cols].itertuples(index=False, name=None):
            inning, half, outs, bases, home, away, raw, no, nb, nh, na, ni, n_half = row
            event = _EVENT_GROUP.get(raw)
            if event not in ("out", "single", "double", "triple", "double_play"):
                continue
            runs = int(nh - home if half == "Bot" else na - away)
            if runs < 0 or runs > 4 or (half == "Bot" and inning >= 9 and nh > na):
                continue
            same_half = ni == inning and n_half == half
            out_delta = int(no - outs) if same_half else int(3 - outs)
            # Only the immediately following legal half is an inning transition.
            legal_flip = ((half == "Top" and ni == inning and n_half == "Bot") or
                          (half == "Bot" and ni == inning + 1 and n_half == "Top"))
            if (not same_half and not legal_flip) or not 0 <= out_delta <= 3 - outs:
                continue
            if event == "double_play" and out_delta != 2:
                continue
            if event == "out" and out_delta != 1:
                continue
            # Runner outs on hits are supported if observed; no mid-PA changes.
            counts[(event, int(bases), int(outs))][(out_delta, int(nb) if same_half else 0, runs)] += 1
            used += 1
        self.counts = dict(counts)
        self.report = {"candidate_terminal_rows": candidates, "used_rows": used,
                       "observed_event_base_out_cells": len(self.counts),
                       "prior_strength": self.prior_strength,
                       "training_scope": "train only; league pooled event/base/out; observed walkoffs excluded"}
        return self

    def distribution(self, state: GameState, event: str):
        counts = self.counts.get((event, state.bases, state.outs))
        fallback = apply_terminal(state, event)
        if not counts:
            return [(1.0, fallback)]
        total = sum(counts.values()) + self.prior_strength
        distribution = defaultdict(float)
        for (outs, bases, runs), n in counts.items():
            distribution[_advance(state, outs, bases, runs)] += n / total
        if self.prior_strength:
            distribution[fallback] += self.prior_strength / total
        return [(probability, next_state) for next_state, probability in distribution.items()]


def _we_features(frame: pd.DataFrame) -> np.ndarray:
    inning = frame["inning"].to_numpy(float)
    bot = frame["inning_topbot"].eq("Bot").to_numpy(float)
    outs = frame["outs_when_up"].to_numpy(float)
    bases = frame["bases"].to_numpy(int)
    runners = np.column_stack([(bases & (1 << i) > 0).astype(float) for i in range(3)])
    diff = frame["home_score"].to_numpy(float) - frame["away_score"].to_numpy(float)
    # Remaining regulation half innings plus the remainder of this half.
    remaining = np.maximum(2 * (9 - inning) + (1 - bot) + (3 - outs) / 3, 0.35)
    batting_sign = 2 * bot - 1
    scale = np.sqrt(remaining + 0.5)
    return np.column_stack([
        np.ones(len(frame)), diff / scale, diff, bot, inning / 9,
        outs / 3, bot * outs / 3, (inning >= 9).astype(float),
        batting_sign / scale, batting_sign * outs / (3 * scale),
        runners * batting_sign[:, None] / scale[:, None],
        runners * batting_sign[:, None] * (3 - outs[:, None]) / (3 * scale[:, None]),
    ])


class WinExpectancy:
    """Regularized logistic game-winner continuation, fitted on PA-start states.

    The caller must invoke fit sequentially with other model training. Score,
    inning, half, outs, and bases are the entire feature set. Extra innings use
    the same late-game feature family; complete regular-season games only.
    """
    def __init__(self, l2: float = 1e-3, max_rows: int = 120000, seed: int = 42):
        self.l2, self.max_rows, self.seed = l2, max_rows, seed

    @staticmethod
    def labeled_states(frame: pd.DataFrame) -> pd.DataFrame:
        if not {"final_home_win", "complete_game"}.issubset(frame.columns):
            raise ValueError("WE requires final_home_win and complete_game from actual final scores")
        data = frame.loc[frame["complete_game"].fillna(False) & frame["final_home_win"].notna()]
        if "game_type" in data:
            data = data.loc[data["game_type"].eq("R")]
        data = data.sort_values(["game_pk", "at_bat_number", "pitch_number"])
        data = data.drop_duplicates(["game_pk", "at_bat_number"], keep="first")
        return data.loc[data["outs_when_up"].between(0, 2) & data["bases"].between(0, 7)]

    def fit(self, frame: pd.DataFrame):
        data = frame.loc[frame["split"].eq("train")] if "split" in frame else frame
        data = self.labeled_states(data)
        full_rows = len(data)
        if len(data) > self.max_rows:
            data = data.sample(self.max_rows, random_state=self.seed)
        if len(data) < 100 or data["final_home_win"].nunique() != 2:
            raise ValueError("not enough complete training-game states for WE")
        x = _we_features(data)
        self.feature_scale = np.std(x, axis=0)
        self.feature_scale[self.feature_scale < 1e-6] = 1
        x = x / self.feature_scale
        y = data["final_home_win"].to_numpy(float)
        penalty = np.full(x.shape[1], self.l2)
        penalty[0] = 0

        def loss_gradient(coef):
            logits = x @ coef
            loss = np.mean(np.logaddexp(0, logits) - y * logits) + 0.5 * np.sum(penalty * coef**2)
            grad = x.T @ (expit(logits) - y) / len(y) + penalty * coef
            return loss, grad

        # Positive score coefficients enforce that increasing the home lead
        # cannot decrease its continuation probability at fixed other features.
        bounds = [(None, None)] * x.shape[1]
        bounds[1] = bounds[2] = (0, None)
        result = minimize(loss_gradient, np.zeros(x.shape[1]), method="L-BFGS-B", jac=True,
                          bounds=bounds, options={"maxiter": 300, "ftol": 1e-10})
        if not result.success:
            raise RuntimeError(f"WE fit did not converge: {result.message}")
        self.coef = result.x
        self.training_report = {
            "available_pa_start_rows": full_rows, "fit_rows": len(data),
            "fit_games": int(data["game_pk"].nunique()), "iterations": int(result.nit),
            "training_min_date": str(data["game_date"].min()),
            "training_max_date": str(data["game_date"].max()),
            "features": "score difference, inning, half, outs, bases and remaining-game interactions",
            "labels": "actual final winner of complete regular-season train games",
            "provider_we_used": False, "score_monotonicity": "positive constrained score coefficients",
            "l2": self.l2,
        }
        return self

    def predict_home(self, states: GameState | pd.DataFrame):
        if isinstance(states, GameState):
            if states.winner is not None:
                return float(states.winner == "home")
            frame = pd.DataFrame([{"inning": states.inning, "inning_topbot": states.half,
                                   "outs_when_up": states.outs, "bases": states.bases,
                                   "home_score": states.home_score, "away_score": states.away_score}])
            return float(expit((_we_features(frame) / self.feature_scale) @ self.coef)[0])
        return expit((_we_features(states) / self.feature_scale) @ self.coef)

    def predict_defense(self, state: GameState, defender_is_home: bool) -> float:
        home = self.predict_home(state)
        return home if defender_is_home else 1 - home

    def evaluate(self, frame: pd.DataFrame) -> dict:
        data = self.labeled_states(frame)
        if not len(data):
            return {"rows": 0, "games": 0}
        y = data["final_home_win"].to_numpy(float)
        p = np.clip(self.predict_home(data), 1e-12, 1 - 1e-12)
        bins = np.minimum((p * 10).astype(int), 9)
        calibration = []
        for bin_index in range(10):
            mask = bins == bin_index
            if mask.any():
                calibration.append({"bin": bin_index, "n": int(mask.sum()),
                                    "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
        return {"rows": len(data), "games": int(data["game_pk"].nunique()),
                "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
                "brier": float(np.mean((p - y)**2)),
                "ece10": float(sum(c["n"] * abs(c["predicted"] - c["observed"]) for c in calibration) / len(data)),
                "calibration": calibration,
                "min_date": str(data["game_date"].min()), "max_date": str(data["game_date"].max()),
                "unit": "PA-start state; game outcomes repeat within each game"}


def terminal_values(state: GameState, we: WinExpectancy,
                    advancement: EmpiricalAdvancement | None = None) -> dict[str, float]:
    """Absolute terminal W for the team defending at the *initial* state."""
    defender_is_home = state.defender_is_home
    result = {}
    for event in TERMINALS:
        distribution = advancement.distribution(state, event) if advancement else [(1., apply_terminal(state, event))]
        result[event] = float(sum(prob * we.predict_defense(next_state, defender_is_home)
                                  for prob, next_state in distribution))
    return result
