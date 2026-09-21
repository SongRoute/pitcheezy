"""CPU-only contracts for fixed-candidate sequence policy sensitivity."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sequence_policy_sensitivity import (controlled_points, evaluate_case,
                                             squared_distance_medoid, summarize_stability)


class Delivery:
    TIERS = [('pitcher', 'pitch_type')]

    def sample(self, frame):
        pool = np.zeros((3, 8))
        pool[:, 2:4] = [[1, 0], [0, 1], [-1, 0]]
        return np.repeat(pool[None], len(frame), axis=0), np.zeros(len(frame), dtype=int)


class SensitivityTests(unittest.TestCase):
    def test_medoid_is_retained_joint_member_and_minimizes_squared_distance(self):
        pool = np.random.default_rng(5).normal(size=(7, 8))
        index, selected, distance = squared_distance_medoid(pool)
        pairwise = ((pool[:, None, :6] - pool[None, :, :6]) ** 2).sum(axis=-1).sum(axis=-1)
        self.assertEqual(index, int(pairwise.argmin()))
        np.testing.assert_array_equal(selected, pool[index])
        self.assertGreaterEqual(distance, 0)
        with self.assertRaisesRegex(ValueError, 'nonempty finite'):
            squared_distance_medoid(np.full((2, 8), np.nan))

    def test_mean_and_medoid_keep_identical_targets_weights_and_fix_circular_mean(self):
        row = pd.Series(dict(pitcher=1, pitch_type='FF'))
        actions = [dict(pitch_type='FF', target_x_ft=.2, target_z_ft=2.4)]
        normalizer = SimpleNamespace(mean=np.zeros(8), scale=np.ones(8))
        mean, wm, dm = controlled_points(row, actions, Delivery(), normalizer, .3, 'mean')
        medoid, wd, dd = controlled_points(row, actions, Delivery(), normalizer, .3, 'train_medoid')
        np.testing.assert_array_equal(wm, wd)
        np.testing.assert_array_equal(mean[..., 6:], medoid[..., 6:])
        self.assertAlmostEqual(dm[0]['spin_sine_cosine_norm_after_inverse_standardization'], 1/3)
        self.assertAlmostEqual(dd[0]['spin_sine_cosine_norm_after_inverse_standardization'], 1)
        np.testing.assert_array_equal(medoid[0, :, 2:4], np.tile([0, 1], (9, 1)))

    def test_stability_shortfall_compares_actions_within_same_scenario(self):
        scenarios = [
            {'scenario_id': 'mean_sigma0.30_depth2', 'best_action_index': 0, 'best_model_internal_defense_we': .8,
             'ranked_actions': [{'action_index': 0, 'model_internal_defense_we': .8}, {'action_index': 1, 'model_internal_defense_we': .7}]},
            {'scenario_id': 'other', 'best_action_index': 1, 'best_model_internal_defense_we': .6,
             'ranked_actions': [{'action_index': 1, 'model_internal_defense_we': .6}, {'action_index': 0, 'model_internal_defense_we': .55}]},
        ]
        report = summarize_stability(scenarios)
        self.assertEqual(report['unique_best_actions'], 2)
        self.assertEqual(report['reference_action_best_in_scenarios'], 1)
        self.assertAlmostEqual(report['largest_reference_action_shortfall_pp'], 5)
        self.assertAlmostEqual(scenarios[1]['reference_action_shortfall_in_scenario_pp'], 5)

    def test_case_evaluates_twelve_scenarios_with_identical_candidates(self):
        class Model:
            temperature = 1.

            def predict(self, arrays):
                p = np.zeros((len(arrays[0]), 10))
                p[:, 0:3] = [.2, .2, .1]
                p[:, 3] = .5
                return p

        class Baseline:
            def predict(self, frame):
                p = np.zeros((len(frame), 10))
                p[:, 3] = 1
                return p

        class We:
            def predict_defense(self, state, defender_is_home):
                return .6

        row = pd.Series(dict(pitcher=1, batter=2, pitch_type='FF', pitch_number=1, balls=0, strikes=0,
                             supported_pa=True, stand='R', inning=1, inning_topbot='Top', outs_when_up=0,
                             bases=0, home_score=0, away_score=0, game_pk=3, at_bat_number=1, game_date='2025-06-01'))
        actions = [dict(pitch_type='FF', target_x_ft=x, target_z_ft=2.4,
                        training_local_n=n, training_pitch_type_n=200) for x, n in [(0., 80), (.7, 50)]]
        encoders = {'delivery': Delivery(), 'normalizer': SimpleNamespace(mean=np.zeros(8), scale=np.ones(8)),
                    'context': SimpleNamespace(transform=lambda frame: np.zeros((len(frame), 2), dtype=np.float32))}
        planning = {'we': We(), 'advancement': None, 'baseline': Baseline(), 'actions': {(1, 'R'): actions}}
        result = evaluate_case(Model(), encoders, planning, row)
        self.assertEqual(len(result['scenarios']), 12)
        self.assertEqual(len(result['computations']), 6)
        self.assertTrue(result['stability']['all_scenarios_same_best_action'])
        for scenario in result['scenarios']:
            self.assertEqual([a['action_index'] for a in scenario['ranked_actions']], [0, 1])
            self.assertEqual([a['training_local_n'] for a in scenario['ranked_actions']], [80, 50])
            self.assertAlmostEqual(scenario['best_model_internal_defense_we'], .6)


if __name__ == '__main__':
    unittest.main()
