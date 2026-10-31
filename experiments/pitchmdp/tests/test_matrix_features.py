"""Synthetic contracts for the shared, strictly prior-outcome H5 adapter."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.matrix_features import MatrixHistoryStore


def frame():
    n = 8
    return pd.DataFrame({'game_date': pd.to_datetime(['2024-04-01'] * 6 + ['2024-04-02'] * 2),
                         'game_pk': [1] * 6 + [2] * 2, 'at_bat_number': [1] * 3 + [2] * 3 + [1] * 2,
                         'pitch_number': [1, 2, 3, 1, 2, 3, 1, 2], 'split': ['train'] * 6 + ['dev'] * 2,
                         'pitch_type': ['FF', 'SL', 'FF', 'SL', 'FF', 'SL', 'NEW', 'FF'],
                         'description': ['ball', 'automatic_strike', 'foul', 'ball', 'ball', 'foul', 'ball', 'ball'],
                         'events': [None] * n, 'supported_pa': [True, False] + [True] * 6,
                         'strikes': [0, 0, 1, 0, 0, 0, 0, 0], 'balls': [0, 1, 1, 0, 1, 2, 0, 1],
                         'pitcher': [10] * n, 'p_throws': ['R'] * n, 'stand': ['L'] * n,
                         'effective_speed': np.arange(n) + 90., 'release_spin_rate': np.arange(n) + 2100.,
                         'spin_axis': np.arange(n) * 10., 'pfx_x': [.1] * n, 'pfx_z': [.2] * n,
                         'plate_x': np.arange(n) / 10, 'plate_z': [2.5] * n})


class MatrixFeaturesTests(unittest.TestCase):
    def test_prior_only_labels_and_unknown_unmapped_outcome(self):
        store = MatrixHistoryStore.from_frame(frame())
        tokens, mask = store.gather([2])
        self.assertEqual(mask.tolist(), [[False, False, False, True, True, True]])
        self.assertEqual(tokens[0, 3, -11:].argmax(), 0)
        self.assertEqual(tokens[0, 4, -11:].argmax(), 10)
        self.assertTrue((tokens[:, -1, -11:] == 0).all())
        self.assertTrue((tokens[~mask] == 0).all())
        self.assertNotIn('NEW', store.type_vocabulary)
        unknown, _ = store.gather([6])
        self.assertEqual(unknown[0, -1, 8], 1.)

    def test_current_and_future_outcomes_do_not_change_current_prediction_inputs(self):
        original = frame()
        changed = original.copy()
        changed.loc[2:, 'description'] = 'hit_by_pitch'
        left = MatrixHistoryStore.from_frame(original)
        right = MatrixHistoryStore.from_frame(changed, normalizer=left.normalizer, type_vocabulary=left.type_vocabulary)
        np.testing.assert_array_equal(left.gather([2])[0], right.gather([2])[0])

    def test_retrospective_pa_support_never_changes_tokens(self):
        original = frame()
        changed = original.copy()
        changed['supported_pa'] = ~changed.supported_pa
        left = MatrixHistoryStore.from_frame(original)
        right = MatrixHistoryStore.from_frame(changed, normalizer=left.normalizer, type_vocabulary=left.type_vocabulary)
        np.testing.assert_array_equal(left.gather([1, 2, 3, 7])[0], right.gather([1, 2, 3, 7])[0])

    def test_current_physics_override_and_candidate_type_follow_query(self):
        store = MatrixHistoryStore.from_frame(frame())
        supplied = np.full((1, 8), 2., dtype=np.float32)
        before, _ = store.gather([2], current=supplied)
        store.frame.loc[2, 'pitch_type'] = 'SL'
        after, _ = store.gather([2], current=supplied)
        np.testing.assert_array_equal(after[:, -1, :8], supplied)
        self.assertFalse(np.array_equal(before[:, -1, 8:-11], after[:, -1, 8:-11]))
        explicit, _ = store.gather([2], current=supplied, candidate_pitch_types=['FF'])
        np.testing.assert_array_equal(before, explicit)

    def test_pa_reset_and_zero_history(self):
        store = MatrixHistoryStore.from_frame(frame())
        _, mask = store.gather([3])
        self.assertEqual(mask.sum(), 1)
        empty = MatrixHistoryStore.from_frame(frame(), history_length=0)
        tokens, mask = empty.gather([2, 3])
        self.assertEqual(tokens.shape[1], 1)
        self.assertTrue(mask.all())
        self.assertEqual(empty.report()['history_length'], 0)


if __name__ == '__main__':
    unittest.main()
