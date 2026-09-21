"""CPU-only exact empirical delivery integration contracts."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_exact_delivery_diagnostic import exact_memberships, ragged_probabilities


class ExactDeliveryTests(unittest.TestCase):
    def test_ragged_softmax_means_keep_each_query_pool_and_fixed_temperature(self):
        logits = np.array([[3., 0.], [0., 1.], [1., 2.], [0., 2.], [4., 1.]])
        expected = np.array([softmax(logits[:2]/1.7, axis=1).mean(axis=0),
                             softmax(logits[2:]/1.7, axis=1).mean(axis=0)])
        result = ragged_probabilities(logits, np.array([0, 2, 5]), 1.7)
        np.testing.assert_allclose(result, expected, atol=1e-15)
        self.assertFalse(np.allclose(result[0], softmax(logits[:2].mean(axis=0)/1.7)))
        np.testing.assert_allclose(result.sum(axis=1), 1.)

    def test_uniform_repeated_finite_pool_matches_exact(self):
        logits = np.array([[2., -1.], [-3., 2.], [0., .5]])
        exact = ragged_probabilities(logits, np.array([0, 3]), .8)
        repeated = np.tile(logits, (7, 1))
        np.testing.assert_allclose(exact, ragged_probabilities(repeated, np.array([0, 21]), .8))

    def test_pool_priority_thresholds_and_full_membership(self):
        train = pd.DataFrame({'split': ['train']*100, 'pitcher': [1]*100,
                              'pitch_type': ['FF']*100, 'p_throws': ['R']*100,
                              'stand': ['R']*100, 'balls': [0]*25+[1]*75,
                              'strikes': [0]*100}, index=np.arange(100)+200)
        queries = train.iloc[[0, 0, 0, 0]].copy().reset_index(drop=True)
        queries.loc[1, 'balls'] = 3  # no count support: pitcher/no-count wins
        queries.loc[2, 'pitcher'] = 9  # league/count wins
        queries.loc[3, 'pitch_type'] = 'CU'  # no tier, full eligible TRAIN
        members, offsets, tiers, _ = exact_memberships(train, queries)
        np.testing.assert_array_equal(tiers, [3, 2, 1, -1])
        np.testing.assert_array_equal(np.diff(offsets), [25, 100, 25, 100])
        for i, expected in enumerate([np.arange(200, 225), np.arange(200, 300),
                                      np.arange(200, 225), np.arange(200, 300)]):
            np.testing.assert_array_equal(members[offsets[i]:offsets[i+1]], expected)
        # Nineteen count rows cannot satisfy the twenty-row threshold.
        train.loc[219:224, 'balls'] = 1
        _, offsets, tiers, _ = exact_memberships(train, queries.iloc[:1])
        self.assertEqual(tiers[0], 2)
        self.assertEqual(offsets[-1], 100)
        train.loc[200, 'split'] = 'dev'
        with self.assertRaises(ValueError):
            exact_memberships(train, queries)

    def test_invalid_ragged_segments_or_temperature_fail(self):
        logits = np.zeros((3, 2))
        for offsets in [[0, 0, 3], [1, 3], [0, 2], [0., 3.]]:
            with self.assertRaises(ValueError):
                ragged_probabilities(logits, np.asarray(offsets), 1.)
        with self.assertRaises(ValueError):
            ragged_probabilities(logits, np.array([0, 3]), 0.)


if __name__ == '__main__':
    unittest.main()
