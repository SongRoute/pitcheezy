"""Small hand-checkable contracts; no model fitting or real-data access."""
import unittest

import numpy as np
import pandas as pd

from pitchmdp.game import EmpiricalAdvancement, GameState, apply_terminal
from pitchmdp.planner import OUTCOMES, TERMINALS, solve_pa


class GameTests(unittest.TestCase):
    def test_walk_only_advances_forced_runners(self):
        expected = {0: (1, 0), 1: (3, 0), 2: (3, 0), 3: (7, 0),
                    4: (5, 0), 5: (7, 0), 6: (7, 0), 7: (7, 1)}
        for bases, (next_bases, runs) in expected.items():
            with self.subTest(bases=bases):
                state = GameState(4, "Top", 1, bases, 0, 0)
                next_state = apply_terminal(state, "walk")
                self.assertEqual((next_state.bases, next_state.away_score), (next_bases, runs))

    def test_double_play_changes_half(self):
        state = GameState(5, "Top", 1, 5, 2, 1)
        next_state = apply_terminal(state, "double_play")
        self.assertEqual(next_state, GameState(5, "Bot", 0, 0, 2, 1))

    def test_illegal_double_play_projects_to_out(self):
        for bases, outs in ((0, 0), (2, 1), (7, 2)):
            state = GameState(3, "Bot", outs, bases, 0, 0)
            self.assertEqual(apply_terminal(state, "double_play"), apply_terminal(state, "out"))

    def test_extra_innings_automatic_runner(self):
        state = GameState(9, "Bot", 2, 7, 3, 3)
        self.assertEqual(apply_terminal(state, "out"), GameState(10, "Top", 0, 2, 3, 3))
        state = GameState(10, "Top", 2, 2, 3, 3)
        self.assertEqual(apply_terminal(state, "out"), GameState(10, "Bot", 0, 2, 3, 3))

    def test_walkoff_home_run_counts_all_runs(self):
        state = GameState(9, "Bot", 1, 7, 2, 2)
        homer = apply_terminal(state, "home_run")
        double = apply_terminal(state, "double")
        self.assertEqual((homer.winner, homer.home_score), ("home", 6))
        self.assertEqual((double.winner, double.home_score), ("home", 3))

    def test_empirical_advance_train_only_and_mass(self):
        rows = []
        for split, next_bases, next_away in (("train", 1, 1), ("dev", 3, 0)):
            rows.append(dict(split=split, supported_pa=True, is_pa_terminal=True,
                             inning=3, inning_topbot="Top", outs_when_up=0, bases=2,
                             home_score=0, away_score=0, terminal_event="single",
                             next_outs=0, next_bases=next_bases, next_home_score=0,
                             next_away_score=next_away, next_inning=3, next_half="Top"))
        advancement = EmpiricalAdvancement(prior_strength=0).fit(pd.DataFrame(rows))
        distribution = advancement.distribution(GameState(3, "Top", 0, 2, 0, 0), "single")
        self.assertEqual(advancement.report["used_rows"], 1)
        self.assertAlmostEqual(sum(prob for prob, _ in distribution), 1)
        self.assertEqual(distribution, [(1., GameState(3, "Top", 0, 1, 0, 1))])


class PlannerTests(unittest.TestCase):
    @staticmethod
    def values():
        return {name: (1. if name in ("out", "strikeout", "double_play") else 0.) for name in TERMINALS}

    def test_hand_optimum_and_one_pitch_improvement(self):
        p = np.zeros((4, 3, 1, 2, len(OUTCOMES)))
        p[..., 0, 3] = .8
        p[..., 0, 7] = .2
        p[..., 1, 3] = .3
        p[..., 1, 7] = .7
        result = solve_pa(p, self.values(), [0, 0])
        np.testing.assert_allclose(result.values, .8)
        np.testing.assert_allclose(result.baseline_values, .55)
        np.testing.assert_allclose(result.myopic_q_values[..., 0], .8)
        self.assertTrue((result.policy == 0).all())

    def test_two_strike_foul_updates_preceding_pitch(self):
        p = np.zeros((4, 3, 2, 1, len(OUTCOMES)))
        p[..., 3] = 1
        p[0, 2, 0, 0, :] = 0
        p[0, 2, 0, 0, 2] = .5
        p[0, 2, 0, 0, 7] = .5
        result = solve_pa(p, self.values(), [1])
        self.assertAlmostEqual(result.values[0, 2, 0], .5)
        self.assertAlmostEqual(result.baseline_values[0, 2, 0], .5)

    def test_four_balls_end_walk_and_three_strikes_end_out(self):
        p = np.zeros((4, 3, 1, 2, len(OUTCOMES)))
        p[..., 0, 0] = 1
        p[..., 1, 1] = 1
        result = solve_pa(p, self.values(), [0, 0], np.array([1., 0.]))
        np.testing.assert_allclose(result.baseline_values, 0)
        np.testing.assert_allclose(result.values, 1)
        self.assertEqual(result.myopic_q_values[0, 0, 0, 1], 0)
        self.assertEqual(result.myopic_q_values[0, 2, 0, 1], 1)

    def test_invalid_probability_mass_rejected(self):
        p = np.ones((4, 3, 1, 1, len(OUTCOMES)))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            solve_pa(p, self.values(), [0])

    def test_nonabsorbing_foul_policy_rejected(self):
        p = np.zeros((4, 3, 1, 1, len(OUTCOMES)))
        p[..., 2] = 1
        with self.assertRaisesRegex(ValueError, "no guaranteed terminal"):
            solve_pa(p, self.values(), [0])


if __name__ == "__main__":
    unittest.main()
