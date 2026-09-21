"""CPU contracts for matched known-type frequency controls and train-only fitting."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
from run_sequence_frequency_baselines import (HierarchicalFrequencyBaseline,
                                              fit_temperature, temperature_predictions)


def observations():
    return pd.DataFrame({
        'pitcher': [1, 1, 2, 2, 2, 3], 'pitch_type': ['FF', 'FF', 'FF', 'SL', 'SL', 'CH'],
        'balls': [0, 0, 0, 0, 1, 1], 'strikes': [0]*6,
        'stand': ['R']*6, 'p_throws': ['R']*6,
        'game_date': pd.to_datetime(['2024-04-01']*6), 'split': ['train']*6,
        'description': ['ball', 'called_strike', 'ball', 'foul', 'ball', 'called_strike'],
        'events': [None]*6, 'batter': [99]*6,
    })


class FrequencyBaselineContracts(unittest.TestCase):
    def test_exact_hierarchical_smoothing_uses_declared_parent_counts(self):
        frame = observations()
        model = HierarchicalFrequencyBaseline(include_pitcher=True).fit(frame)
        raw_global = np.array([3., 2., 1.]+[0.]*7)
        global_p = (raw_global+1)/16
        count_p = (np.array([2., 1., 1.]+[0.]*7)+50*global_p)/54
        type_p = (np.array([2., 1., 0.]+[0.]*7)+50*count_p)/53
        pitcher_p = (np.array([1., 1., 0.]+[0.]*7)+100*type_p)/102
        np.testing.assert_allclose(model.parent.global_p, global_p)
        np.testing.assert_allclose(model.parent.predict(frame.iloc[[0]])[0], count_p)
        np.testing.assert_allclose(HierarchicalFrequencyBaseline().fit(frame).predict(frame.iloc[[0]])[0], type_p)
        actual, origin = model.predict_with_origin(frame.iloc[[0]])
        np.testing.assert_allclose(actual[0], pitcher_p)
        self.assertEqual(origin[0], 3)

    def test_unseen_pitcher_type_and_count_backoffs_are_exact(self):
        frame = observations()
        model = HierarchicalFrequencyBaseline(include_pitcher=True).fit(frame)
        queries = pd.concat([frame.iloc[[0]]]*4, ignore_index=True)
        queries.loc[1:, 'pitcher'] = 999
        queries.loc[2:, 'pitch_type'] = 'UNKNOWN'
        queries.loc[3, ['balls', 'strikes']] = [3, 2]
        probability, origin = model.predict_with_origin(queries)
        np.testing.assert_array_equal(origin, [3, 2, 1, 0])
        np.testing.assert_allclose(probability[1], HierarchicalFrequencyBaseline().fit(frame).predict(queries.iloc[[1]])[0])
        np.testing.assert_array_equal(probability[2], model.parent.predict(queries.iloc[[2]])[0])
        np.testing.assert_array_equal(probability[3], model.parent.global_p)
        np.testing.assert_allclose(probability.sum(1), 1.)
        self.assertTrue((probability > 0).all())

    def test_query_outcomes_batter_id_current_delivery_and_future_state_are_unused(self):
        frame = observations()
        model = HierarchicalFrequencyBaseline(include_pitcher=True).fit(frame)
        expected = model.predict(frame)
        changed = frame.copy()
        for column in ['batter', 'description', 'events', 'plate_x', 'plate_z', 'release_speed',
                       'post_home_score', 'next_outs', 'home_win_exp']:
            changed[column] = 'UNAVAILABLE_INFORMATION'
        np.testing.assert_array_equal(model.predict(changed), expected)
        # The type-only model also cannot exploit pitcher identity.
        type_model = HierarchicalFrequencyBaseline().fit(frame)
        changed['pitcher'] = -999
        np.testing.assert_array_equal(type_model.predict(changed), type_model.predict(frame))

    def test_fitting_rejects_nontrain_dates_and_unknown_targets(self):
        for mode in ['split', 'date', 'target']:
            frame = observations()
            if mode == 'split':
                frame.loc[0, 'split'] = 'calibration'
            elif mode == 'date':
                frame.loc[0, 'game_date'] = pd.Timestamp('2025-05-01')
            else:
                frame.loc[0, 'description'] = 'UNKNOWN'
            with self.assertRaises(ValueError):
                HierarchicalFrequencyBaseline().fit(frame)

    def test_transform_is_frozen_and_independent_of_query_batch(self):
        frame = observations()
        model = HierarchicalFrequencyBaseline(include_pitcher=True).fit(frame)
        table = model.pitcher_table.copy(deep=True)
        expected = model.predict(frame.iloc[[0]])
        noisy = pd.concat([frame.iloc[[0]], frame.iloc[[3]]]*100, ignore_index=True)
        model.predict(noisy)
        np.testing.assert_array_equal(model.predict(frame.iloc[[0]]), expected)
        pd.testing.assert_frame_equal(table, model.pitcher_table)

    def test_temperature_one_preserves_probabilities_and_fit_uses_given_calibration(self):
        probability = np.array([[.8, .2], [.7, .3], [.3, .7], [.2, .8]])
        np.testing.assert_allclose(temperature_predictions(probability, 1.), probability)
        labels = np.array([0, 1, 0, 1])
        result = fit_temperature(probability, labels)
        self.assertGreaterEqual(result['temperature'], .5)
        self.assertLessEqual(result['temperature'], 2.5)
        self.assertLessEqual(result['calibrated_log_loss'], result['raw_log_loss']+1e-7)


if __name__ == '__main__':
    unittest.main()
