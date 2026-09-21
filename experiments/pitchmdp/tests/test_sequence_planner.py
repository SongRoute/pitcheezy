"""Hand-checkable sequence lookahead contracts; no training/data required."""
import unittest

import numpy as np

from pitchmdp.planner import OUTCOMES, TERMINALS, solve_pa
from pitchmdp.sequence_planner import solve_lookahead


def utilities():
    return {name: float(name in ("out", "strikeout", "double_play")) for name in TERMINALS}


class SequencePlannerTests(unittest.TestCase):
    def test_second_pitch_depends_on_simulated_physics(self):
        calls = []

        def predictor(histories, counts, deliveries):
            calls.append((histories, counts.copy(), deliveries.copy()))
            p = np.zeros((len(histories), len(OUTCOMES)))
            for i, h in enumerate(histories):
                if len(h) == 0:
                    p[i, 0] = 1
                else:
                    p[i, 3] = .6 if h[-1, 0] == 1 else .9
                    p[i, 7] = 1 - p[i, 3]
            return p

        def tail(b, s, h):
            return .5 if len(h) == 0 else (.7 if h[-1, 0] == 1 else .2)

        result = solve_lookahead(np.empty((0, 1)), 0, 0, np.array([[[1]], [[2]]]),
                                 predictor, utilities(), tail)
        np.testing.assert_allclose(result.one_step_q_values, [.7, .2])
        np.testing.assert_allclose(result.q_values, [.6, .9])
        self.assertEqual(result.policy, 1)
        self.assertEqual(result.diagnostics["prediction_calls"], 2)
        self.assertEqual(result.diagnostics["prediction_rows"], 6)
        self.assertFalse(result.diagnostics["exact_full_history_pa_mdp"])
        np.testing.assert_array_equal(calls[1][1], [[1, 0]] * 4)
        np.testing.assert_array_equal([h[-1, 0] for h in calls[1][0]], [1, 1, 2, 2])

    def test_two_strike_foul_keeps_count_and_updates_last_five(self):
        original = np.arange(1, 6, dtype=float)[:, None]
        seen = []

        def predictor(histories, counts, deliveries):
            seen.append((histories[0].copy(), counts[0].copy()))
            p = np.zeros((len(histories), 10))
            p[:, 2 if histories[0][-1, 0] == 5 else 3] = 1
            return p

        result = solve_lookahead(original, 0, 2, np.array([[[9.]]]), predictor,
                                 utilities(), lambda b, s, h: 0)
        self.assertEqual(result.value, 1)
        self.assertEqual(result.one_step_q_values[0], 0)
        np.testing.assert_array_equal(seen[1][0][:, 0], [2, 3, 4, 5, 9])
        np.testing.assert_array_equal(seen[1][1], [0, 2])
        np.testing.assert_array_equal(original[:, 0], [1, 2, 3, 4, 5])

    def test_all_outcomes_count_updates_and_terminal_mass(self):
        seen = []

        def tail(b, s, h):
            seen.append((b, s))
            return (b + s) / 10

        result = solve_lookahead(np.empty((0, 1)), 2, 1, np.array([[[5.]]]),
                                 lambda h, c, d: np.full((len(h), 10), .1),
                                 utilities(), tail, max_depth=1)
        # Ball 3-1, strike/foul 2-2 each continue at .4. Out/DP have value 1.
        self.assertAlmostEqual(result.value, .32)
        self.assertCountEqual(seen, [(2, 1), (3, 1), (2, 2), (2, 2)])
        np.testing.assert_array_equal(result.q_values, result.one_step_q_values)

    def test_fourth_ball_and_third_strike_are_terminal(self):
        terminal = utilities()
        terminal["walk"] = .2

        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[:, 0] = .4
            p[:, 1] = .6
            return p

        result = solve_lookahead(np.empty((0, 1)), 3, 2, np.array([[[0.]]]),
                                 predictor, terminal, lambda b, s, h: .99)
        self.assertAlmostEqual(result.value, .68)
        self.assertEqual(result.diagnostics["tail_calls"], 1)
        self.assertEqual(result.diagnostics["prediction_calls"], 1)

    def test_delivery_is_integrated_before_action_selection(self):
        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[:, 3] = d[:, 0] / 10
            p[:, 7] = 1 - p[:, 3]
            return p

        result = solve_lookahead(np.empty((0, 1)), 0, 0,
                                 np.array([[[0.], [10.]], [[4.], [4.]]]), predictor,
                                 utilities(), lambda b, s, h: .5, weights=np.array([.75, .25]))
        np.testing.assert_allclose(result.q_values, [.25, .4])
        self.assertEqual(result.policy, 1)
        self.assertEqual(result.topk(1)[0]["action_index"], 1)

    def test_consistent_count_tail_matches_existing_one_pitch_policy_improvement(self):
        probabilities = np.zeros((4, 3, 1, 2, 10))
        probabilities[..., 0] = .25
        probabilities[..., 1] = .25
        probabilities[..., 2] = .1
        probabilities[..., 0, 3] = .2
        probabilities[..., 1, 3] = .3
        probabilities[..., 0, 7] = .2
        probabilities[..., 1, 7] = .1
        exact = solve_pa(probabilities, utilities(), [0, 0])

        def predictor(h, c, d):
            return probabilities[c[:, 0], c[:, 1], 0, d[:, 0].astype(int)]

        result = solve_lookahead(np.empty((0, 1)), 1, 2, np.array([[[0.]], [[1.]]]),
                                 predictor, utilities(),
                                 lambda b, s, h: exact.baseline_values[b, s, 0])
        np.testing.assert_allclose(result.one_step_q_values, exact.myopic_q_values[1, 2, 0])
        self.assertAlmostEqual(result.continuation_reference, exact.baseline_values[1, 2, 0])
        self.assertTrue((result.q_values >= result.one_step_q_values - 1e-12).all())
        self.assertTrue((result.q_values <= exact.q_values[1, 2, 0] + 1e-12).all())

    def test_optional_token_encoder_receives_pre_pitch_count_and_outcome(self):
        encoded = []

        def append(h, d, outcome, b, s):
            encoded.append((outcome, b, s))
            return np.concatenate([h, [[d[0], b, s]]])

        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[:, 2] = 1
            return p

        result = solve_lookahead(np.empty((0, 3)), 1, 1, np.array([[[8.]]]), predictor,
                                 utilities(), lambda b, s, h: float(s), append_token=append)
        self.assertEqual(encoded, [("foul", 1, 1), ("foul", 1, 2)])
        self.assertEqual(result.value, 2)

    def test_invalid_mass_and_work_budget_rejected(self):
        args = (np.empty((0, 1)), 0, 0, np.array([[[1.]], [[2.]]]))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            solve_lookahead(*args, lambda h, c, d: np.ones((len(h), 10)),
                            utilities(), lambda b, s, h: 0)

        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[:, 0] = 1
            return p

        with self.assertRaisesRegex(ValueError, "max_prediction_rows"):
            solve_lookahead(*args, predictor, utilities(), lambda b, s, h: 0,
                            max_prediction_rows=3)
        with self.assertRaisesRegex(ValueError, "tail values must be finite"):
            solve_lookahead(*args, predictor, utilities(), lambda b, s, h: np.nan)
        with self.assertRaisesRegex(ValueError, "quadrature weights must sum"):
            solve_lookahead(*args, predictor, utilities(), lambda b, s, h: 0,
                            weights=np.array([.5]))


if __name__ == "__main__":
    unittest.main()
