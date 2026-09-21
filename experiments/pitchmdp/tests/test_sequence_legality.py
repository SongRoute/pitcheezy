"""Synthetic tests for common impossible-DP probability conditioning."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_sequence_legality import condition_on_legality


class LegalConditioningContracts(unittest.TestCase):
    def test_removes_dp_and_preserves_all_other_class_odds(self):
        raw = np.array([[.05, .15, .1, .2, .04, .06, .01, .04, .05, .3]])
        original = raw.copy()
        constrained = condition_on_legality(raw, [True])
        self.assertEqual(constrained[0, 9], 0.)
        np.testing.assert_allclose(constrained[0, :9], raw[0, :9]/.7)
        self.assertAlmostEqual(constrained[0, 3]/constrained[0, 1], raw[0, 3]/raw[0, 1])
        np.testing.assert_array_equal(raw, original)
        self.assertNotEqual(constrained[0, 3], raw[0, 3]+raw[0, 9])

    def test_legal_rows_unchanged_and_valid_labels_cannot_lose_log_probability(self):
        raw = np.full((3, 10), .1)
        constrained = condition_on_legality(raw, [True, False, True])
        np.testing.assert_array_equal(constrained[1], raw[1])
        np.testing.assert_allclose(constrained.sum(1), 1.)
        self.assertTrue(np.all(constrained[[0, 2], :9] >= raw[[0, 2], :9]))
        labels = np.array([0, 9, 8])  # The DP target appears only in a legal row.
        original_labels = labels.copy()
        raw_loss = -np.log(raw[np.arange(3), labels])
        constrained_loss = -np.log(constrained[np.arange(3), labels])
        self.assertTrue(np.all(constrained_loss <= raw_loss))
        np.testing.assert_array_equal(labels, original_labels)

    def test_all_mass_on_impossible_event_cannot_be_conditioned(self):
        raw = np.zeros((1, 10)); raw[:, 9] = 1.
        with self.assertRaises(ValueError):
            condition_on_legality(raw, [True])
        np.testing.assert_array_equal(condition_on_legality(raw, [False]), raw)


if __name__ == '__main__':
    unittest.main()
