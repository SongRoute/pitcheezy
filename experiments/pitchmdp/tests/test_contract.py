"""Independent information-boundary and probability-contract checks.

These tests do not fit a model and do not inspect any external/raw data.
Run from the repository root with PYTHONPATH=experiments/pitchmdp.
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pitchmdp.model import DeliveryDistribution, Encoder, OUTCOMES, metrics, outcome_labels
from pitchmdp.recommend import probability_tensor
from pitchmdp.game import GameState, terminal_values
from pitchmdp.planner import TERMINALS, solve_pa


def pitch_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "pitch_type": ["FF", "FF", "SL", "SL"],
        "pitcher": [1, 1, 1, 1], "batter": [20, 21, 20, 21],
        "stand": ["R"] * 4, "p_throws": ["R"] * 4,
        "prev_pitch_type": ["START", "FF", "FF", "SL"],
        "balls": [0, 1, 2, 3], "strikes": [0, 1, 2, 2],
        "outs_when_up": [0, 1, 2, 2], "inning": [1, 3, 6, 8],
        "bases": [0, 1, 3, 7], "home_score": [0, 2, 4, 4],
        "away_score": [0, 1, 4, 3], "inning_topbot": ["Top"] * 4,
        "plate_x": [-0.8, 0.2, 0.7, 1.0], "plate_z": [2.1, 2.8, 2.0, 3.1],
        "batter_pa_prior": [0, 10, 100, 1000],
        "batter_obp_prior": [.32, .30, .31, .34],
        "batter_k_prior": [.23, .24, .21, .20],
    })


class LocationResponse:
    """Deterministic stand-in whose predictions visibly depend on location."""
    def predict(self, frame):
        p = np.zeros((len(frame), len(OUTCOMES)))
        p[:, 0] = 1 / (1 + np.exp(-frame.plate_x.to_numpy()))
        p[:, 1] = 1 - p[:, 0]
        return p


class InformationBoundaryTests(unittest.TestCase):
    def test_postpitch_and_future_columns_are_not_model_inputs(self):
        frame = pitch_frame()
        forbidden = [
            "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "sz_top", "sz_bot",
            "description", "events", "type", "zone", "launch_speed", "launch_angle",
            "estimated_woba_using_speedangle", "home_win_exp", "bat_win_exp",
            "delta_home_win_exp", "post_home_score", "post_away_score",
            "pitcher_days_until_next_game", "batter_days_until_next_game",
        ]
        for col in forbidden:
            frame[col] = 0
        changed = frame.copy()
        for col in forbidden:
            changed[col] = ["future", 123456, None, -123456]
        for variant in ("full", "minus_b", "minus_c"):
            encoder = Encoder(variant).fit(frame)
            expected = encoder.transform(frame)
            actual = encoder.transform(changed)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_prepitch_delivery_integration_ignores_realized_current_location(self):
        train = pitch_frame()
        distribution = DeliveryDistribution().fit(train, draws=3, seed=7)
        evaluated = pitch_frame()
        changed = evaluated.copy()
        changed["plate_x"] = [-100, 100, np.nan, 10]
        changed["plate_z"] = [100, -100, 10, np.nan]
        expected = distribution.predict(LocationResponse(), evaluated)
        actual = distribution.predict(LocationResponse(), changed)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=1), 1)
        self.assertTrue(np.isfinite(actual).all())

    def test_ablation_removes_only_declared_information(self):
        frame = pitch_frame()
        changed = frame.copy()
        changed["batter"] = 99999
        changed["batter_pa_prior"] = 999999
        changed["batter_obp_prior"] = .99
        changed["batter_k_prior"] = .01
        encoder = Encoder("minus_b").fit(frame)
        before, after = encoder.transform(frame), encoder.transform(changed)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        encoder = Encoder("minus_c").fit(frame)
        changed = frame.copy()
        changed["prev_pitch_type"] = "DIFFERENT"
        before, after = encoder.transform(frame), encoder.transform(changed)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])

    def test_target_planner_never_uses_logged_current_action_or_delivery(self):
        row = pitch_frame().iloc[1].copy()
        actions = [
            {"pitch_type": "FF", "target_x_ft": -.7, "target_z_ft": 2.45,
             "training_pitch_type_n": 400},
            {"pitch_type": "FF", "target_x_ft": .7, "target_z_ft": 2.45,
             "training_pitch_type_n": 400},
            {"pitch_type": "SL", "target_x_ft": 0., "target_z_ft": 1.65,
             "training_pitch_type_n": 100},
        ]
        expected, previous, next_previous, baseline = probability_tensor(LocationResponse(), row, actions)
        row["pitch_type"], row["plate_x"], row["plate_z"] = "UNKNOWN", 9999., -9999.
        actual, _, _, _ = probability_tensor(LocationResponse(), row, actions)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual.sum(axis=-1), 1)
        np.testing.assert_allclose(baseline, [.4, .4, .2])
        self.assertEqual([previous[i] for i in next_previous], ["FF", "FF", "SL"])


class OutcomeMetricTests(unittest.TestCase):
    def test_logloss_and_multiclass_brier_match_hand_calculation(self):
        p = np.zeros((2, len(OUTCOMES)))
        p[0, :2] = [.75, .25]
        p[1, :2] = [.4, .6]
        result = metrics([0, 1], p)
        self.assertAlmostEqual(result["log_loss"], -(np.log(.75) + np.log(.6)) / 2)
        self.assertAlmostEqual(result["brier_multiclass"], ((.25**2) * 2 + (.4**2) * 2) / 2)

    def test_invalid_probability_mass_is_rejected(self):
        with self.assertRaises(AssertionError):
            metrics([0], np.ones((1, len(OUTCOMES))))
        with self.assertRaises(AssertionError):
            metrics([0], np.full((1, len(OUTCOMES)), np.nan))

    def test_two_strike_bunt_foul_is_not_repeatable_foul(self):
        frame = pd.DataFrame({
            "description": ["foul", "foul_bunt", "foul_bunt"],
            "events": [None, None, "strikeout"], "strikes": [2, 1, 2],
        })
        self.assertEqual(outcome_labels(frame).tolist(), [OUTCOMES.index(x) for x in ["foul", "foul", "strike"]])


class FixedHomeWinExpectancy:
    def predict_defense(self, state, defender_is_home):
        if state.winner is not None:
            home = float(state.winner == "home")
        else:
            home = .8 if state.half == "Bot" else .6
        return home if defender_is_home else 1 - home


class DefensiveValueContractTests(unittest.TestCase):
    def test_terminal_value_preserves_defensive_team_through_half_change(self):
        we = FixedHomeWinExpectancy()
        home_defending = GameState(5, "Top", 2, 0, 0, 0)
        away_defending = GameState(5, "Bot", 2, 0, 0, 0)
        self.assertAlmostEqual(terminal_values(home_defending, we)["out"], .8)
        self.assertAlmostEqual(terminal_values(away_defending, we)["out"], .4)

    def test_game_ending_out_and_walkoff_have_exact_defensive_boundaries(self):
        we = FixedHomeWinExpectancy()
        last_out = GameState(9, "Top", 2, 0, 1, 0)
        walkoff = GameState(9, "Bot", 0, 0, 0, 0)
        self.assertEqual(terminal_values(last_out, we)["out"], 1.)
        self.assertEqual(terminal_values(walkoff, we)["home_run"], 0.)

    def test_two_strike_foul_loop_matches_exact_constant_terminal_value(self):
        p = np.zeros((4, 3, 2, 2, len(OUTCOMES)))
        p[..., OUTCOMES.index("foul")] = .8
        p[..., OUTCOMES.index("out")] = .2
        terminals = {name: .37 for name in TERMINALS}
        result = solve_pa(p, terminals, [0, 1])
        np.testing.assert_allclose(result.values, .37, atol=1e-10)
        np.testing.assert_allclose(result.baseline_values, .37, atol=1e-10)

    def test_full_plan_dominates_one_pitch_improvement_and_baseline(self):
        rng = np.random.default_rng(203)
        p = rng.dirichlet(np.ones(len(OUTCOMES)), size=(4, 3, 3, 4))
        terminals = {name: float(value) for name, value in zip(TERMINALS, [.2, .8, .7, .4, .3, .2, .1, .2, .9])}
        result = solve_pa(p, terminals, [0, 1, 2, 0], np.array([.1, .2, .3, .4]))
        myopic = result.myopic_q_values.max(axis=-1)
        self.assertTrue(np.all(result.values >= myopic - 1e-10))
        self.assertTrue(np.all(myopic >= result.baseline_values - 1e-10))
        self.assertTrue(np.all(result.values >= min(terminals.values()) - 1e-10))
        self.assertTrue(np.all(result.values <= max(terminals.values()) + 1e-10))
        self.assertLess(result.diagnostics["max_bellman_residual"], 1e-10)


if __name__ == "__main__":
    unittest.main()
