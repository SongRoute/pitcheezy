"""CPU-only terminal, CRN, history, independent selection and censor contracts."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sequence_rollout import common_uniforms, simulate, summarize
from pitchmdp.planner import TERMINALS


def utility():
    return {event: float(event in ('strikeout', 'out', 'double_play')) for event in TERMINALS}


def certain(outcome):
    def predictor(h, c, d):
        p = np.zeros((len(h), 10))
        p[:, outcome] = 1
        return p
    return predictor


class RolloutTests(unittest.TestCase):
    def test_constant_terminal_utility_and_identical_policy_contrast(self):
        arguments = (np.empty((0, 1)), 0, 0, np.array([[[1.]], [[2.]]]), np.ones(1), np.array([.3, .7]),
                     certain(3), {event: .37 for event in TERMINALS})
        selection = simulate(*arguments, common_uniforms(32, 32, 142))
        evaluation = simulate(*arguments, common_uniforms(64, 32, 242))
        np.testing.assert_allclose(evaluation['lower_returns'], .37)
        self.assertFalse(evaluation['censored'].any())
        np.testing.assert_array_equal(evaluation['pitches'], 1)
        report = summarize(selection, evaluation, np.array([.3, .7]))
        self.assertAlmostEqual(report['paired_zero_imputed_contrast']['mean'], 0)
        self.assertAlmostEqual(report['paired_zero_imputed_contrast']['mc_standard_error'], 0)

    def test_fourth_ball_and_third_strike_terminate_with_correct_utility(self):
        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[np.arange(len(h)), d[:, 0].astype(int)] = 1
            return p
        terminal = utility()
        terminal['walk'] = .2
        result = simulate(np.empty((0, 1)), 3, 2, np.array([[[0.]], [[1.]]]), np.ones(1), np.array([.5, .5]),
                          predictor, terminal, common_uniforms(4, 4, 142))
        np.testing.assert_allclose(result['lower_returns'][0], .2)
        np.testing.assert_allclose(result['lower_returns'][1], 1)
        np.testing.assert_array_equal(result['terminal_event_index'], [[0]*4, [1]*4])
        np.testing.assert_array_equal(result['pitches'], 1)

    def test_two_strike_foul_changes_history_not_count(self):
        seen = []
        def predictor(histories, counts, delivered):
            seen.append(([h.copy() for h in histories], counts.copy()))
            p = np.zeros((len(histories), 10))
            for i, history in enumerate(histories):
                p[i, 2 if history[-1, 0] == 5 else 1] = 1
            return p
        original = np.arange(1, 6, dtype=float)[:, None]
        result = simulate(original, 1, 2, np.array([[[9.]]]), np.ones(1), np.ones(1),
                          predictor, utility(), common_uniforms(2, 3, 142))
        np.testing.assert_allclose(result['lower_returns'], 1)
        np.testing.assert_array_equal(result['pitches'], 2)
        np.testing.assert_array_equal(seen[1][1], [[1, 2], [1, 2]])
        np.testing.assert_array_equal(seen[1][0][0][:, 0], [2, 3, 4, 5, 9])
        np.testing.assert_array_equal(original[:, 0], [1, 2, 3, 4, 5])

    def test_crn_pairs_identical_arms_and_streams_are_independent(self):
        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            p[:, [0, 1, 2, 3, 7]] = [.15, .15, .1, .3, .3]
            return p
        uniforms = common_uniforms(128, 32, 142)
        self.assertFalse(np.array_equal(uniforms, common_uniforms(128, 32, 242)))
        result = simulate(np.empty((0, 1)), 0, 0, np.array([[[1.]], [[2.]]]), np.ones(1), np.array([.5, .5]),
                          predictor, utility(), uniforms)
        np.testing.assert_array_equal(result['lower_returns'][0], result['lower_returns'][1])
        np.testing.assert_array_equal(result['pitches'][0], result['pitches'][1])
        np.testing.assert_array_equal(result['terminal_event_index'][0], result['terminal_event_index'][1])

    def test_continuation_samples_fixed_policy_after_forced_root(self):
        def predictor(h, c, d):
            p = np.zeros((len(h), 10))
            # Root action 0 yields a ball, root action 1 immediately yields out.
            # On the next pitch, policy action 0 yields out, action 1 home run.
            root = c[:, 0] == 0
            event = np.where(root, np.where(d[:, 0] == 0, 0, 3),
                             np.where(d[:, 0] == 0, 3, 7))
            p[np.arange(len(h)), event] = 1
            return p
        uniforms = common_uniforms(128, 3, 142)
        result = simulate(np.empty((0, 1)), 0, 0, np.array([[[0.]], [[1.]]]), np.ones(1), np.array([.25, .75]),
                          predictor, utility(), uniforms)
        np.testing.assert_array_equal(result['lower_returns'][0], uniforms[1, :, 0] < .25)
        np.testing.assert_array_equal(result['lower_returns'][1], 1)
        np.testing.assert_array_equal(result['pitches'][0], 2)
        np.testing.assert_array_equal(result['pitches'][1], 1)

    def test_evaluation_cannot_reselect_action_after_selection(self):
        selection = {'lower_returns': np.array([[1., 1.], [0., 0.]]), 'censored': np.zeros((2, 2), bool)}
        evaluation = {'lower_returns': np.array([[0., 0.], [1., 1.]]), 'censored': np.zeros((2, 2), bool)}
        report = summarize(selection, evaluation, np.array([.5, .5]))
        self.assertEqual(report['selected_action_index'], 0)
        self.assertEqual(report['paired_zero_imputed_contrast']['mean'], -.5)
        self.assertEqual(report['paired_censor_contrast_bounds'], [-.5, -.5])

    def test_nonterminal_cap_reports_explicit_bounds_not_tail(self):
        result = simulate(np.empty((0, 1)), 0, 2, np.array([[[1.]], [[2.]]]), np.ones(1), np.array([.5, .5]),
                          certain(2), utility(), common_uniforms(4, 3, 142))
        self.assertTrue(result['censored'].all())
        np.testing.assert_array_equal(result['lower_returns'], 0)
        np.testing.assert_array_equal(result['terminal_event_index'], -1)
        np.testing.assert_array_equal(result['pitches'], 3)
        report = summarize(result, result, np.array([.5, .5]))
        self.assertEqual(report['selected_value_bounds'], [0, 1])
        self.assertEqual(report['baseline_value_bounds'], [0, 1])
        self.assertEqual(report['paired_censor_contrast_bounds'], [-.5, .5])
        self.assertEqual(report['paired_censor_expanded_mc95_normal'], [-.5, .5])

    def test_invalid_probability_mass_and_utility_rejected(self):
        args = (np.empty((0, 1)), 0, 0, np.array([[[1.]]]), np.ones(1), np.ones(1))
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            simulate(*args, lambda h, c, d: np.ones((len(h), 10)), utility(), common_uniforms(2, 2, 1))
        invalid = utility()
        invalid['out'] = 2
        with self.assertRaisesRegex(ValueError, 'lie in'):
            simulate(*args, certain(3), invalid, common_uniforms(2, 2, 1))

    def test_identical_single_action_policy_censoring_cancels_exactly(self):
        result = simulate(np.empty((0, 1)), 0, 2, np.array([[[1.]]]), np.ones(1), np.ones(1),
                          certain(2), utility(), common_uniforms(4, 3, 142))
        report = summarize(result, result, np.ones(1))
        self.assertEqual(report['paired_censor_contrast_bounds'], [0, 0])
        self.assertEqual(report['paired_censor_expanded_mc95_normal'], [0, 0])


if __name__ == '__main__':
    unittest.main()
