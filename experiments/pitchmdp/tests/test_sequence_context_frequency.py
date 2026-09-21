"""CPU-only contracts for legal-state frequency controls and fixed backoff."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline
from run_sequence_context_frequency import ContextFrequencyBaseline


def observations():
    return pd.DataFrame({
        'pitcher': [1, 1, 2, 2, 2, 3], 'pitch_type': ['FF']*6,
        'balls': [0]*6, 'strikes': [0]*6, 'stand': ['R']*6, 'p_throws': ['R']*6,
        'outs_when_up': [0, 0, 0, 0, 0, 2], 'bases': [1, 1, 1, 2, 2, 3],
        'game_date': pd.to_datetime(['2024-04-01']*6), 'split': ['train']*6,
        'description': ['ball', 'called_strike', 'ball', 'foul', 'ball', 'called_strike'],
        'events': [None]*6, 'batter': [99]*6,
    })


class ContextFrequencyContracts(unittest.TestCase):
    def test_exact_context100_and_pitcher100_smoothing(self):
        frame = observations()
        parent = HierarchicalFrequencyBaseline().fit(frame)
        parent_p = parent.predict(frame.iloc[[0]])[0]
        context_p = (np.array([2., 1., 0.]+[0.]*7)+100*parent_p)/103
        pitcher_p = (np.array([1., 1., 0.]+[0.]*7)+100*context_p)/102
        league = ContextFrequencyBaseline(parent, include_pitcher=False).fit(frame)
        individual = ContextFrequencyBaseline(parent, include_pitcher=True).fit(frame)
        np.testing.assert_allclose(league.predict(frame.iloc[[0]])[0], context_p)
        np.testing.assert_allclose(individual.predict(frame.iloc[[0]])[0], pitcher_p)

    def test_exact_backoff_from_pitcher_to_context_type_count_and_global(self):
        frame = observations()
        parent = HierarchicalFrequencyBaseline().fit(frame)
        model = ContextFrequencyBaseline(parent, include_pitcher=True).fit(frame)
        query = pd.concat([frame.iloc[[0]]]*5, ignore_index=True)
        query.loc[1:, 'pitcher'] = 999
        query.loc[2:, ['outs_when_up', 'bases']] = [2, 0]
        query.loc[3:, 'pitch_type'] = 'UNSEEN'
        query.loc[4, ['balls', 'strikes']] = [3, 2]
        probability, origin = model.predict_with_origin(query)
        np.testing.assert_array_equal(origin, [5, 4, 2, 1, 0])
        league = ContextFrequencyBaseline(parent).fit(frame)
        np.testing.assert_array_equal(probability[1], league.predict(query.iloc[[1]])[0])
        np.testing.assert_array_equal(probability[2:], parent.predict(query.iloc[2:]))
        np.testing.assert_allclose(probability.sum(1), 1.)

    def test_base_occupancy_bits_remain_distinct_categories(self):
        frame = observations()
        model = ContextFrequencyBaseline(HierarchicalFrequencyBaseline().fit(frame)).fit(frame)
        query = pd.concat([frame.iloc[[0]]]*2, ignore_index=True)
        query.loc[1, 'bases'] = 2  # One runner on second is not one runner on first.
        p, origin = model.predict_with_origin(query)
        np.testing.assert_array_equal(origin, [4, 4])
        self.assertGreater(float(np.max(np.abs(p[0]-p[1]))), 1e-4)

    def test_query_outcomes_ids_future_and_physical_fields_are_not_predictors(self):
        frame = observations()
        parent = HierarchicalFrequencyBaseline().fit(frame)
        for with_pitcher in [False, True]:
            model = ContextFrequencyBaseline(parent, include_pitcher=with_pitcher).fit(frame)
            expected = model.predict(frame)
            changed = frame.copy()
            for column in ['batter', 'description', 'events', 'post_home_score', 'next_outs',
                           'next_bases', 'home_win_exp', 'plate_x', 'plate_z', 'release_speed']:
                changed[column] = 'FUTURE_INFORMATION'
            if not with_pitcher:
                changed['pitcher'] = 999
            np.testing.assert_array_equal(model.predict(changed), expected)

    def test_only_train_legal_states_and_valid_labels_can_fit(self):
        frame = observations()
        parent = HierarchicalFrequencyBaseline().fit(frame)
        mutations = [('split', 'calibration'), ('game_date', pd.Timestamp('2025-05-01')),
                     ('description', 'UNKNOWN'), ('outs_when_up', 3), ('outs_when_up', -1),
                     ('bases', 8), ('bases', -1)]
        for column, value in mutations:
            changed = frame.copy()
            changed.loc[0, column] = value
            with self.assertRaises(ValueError):
                ContextFrequencyBaseline(parent).fit(changed)

    def test_query_batches_do_not_refit_or_mutate_parent(self):
        frame = observations()
        parent = HierarchicalFrequencyBaseline().fit(frame)
        parent_before = parent.predict(frame)
        model = ContextFrequencyBaseline(parent, include_pitcher=True).fit(frame)
        before = model.predict(frame.iloc[[0]])
        extra = pd.concat([frame.iloc[[3]]]*100, ignore_index=True)
        model.predict(extra)
        np.testing.assert_array_equal(model.predict(frame.iloc[[0]]), before)
        np.testing.assert_array_equal(parent.predict(frame), parent_before)


if __name__ == '__main__':
    unittest.main()
