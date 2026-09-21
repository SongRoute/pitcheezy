"""Adapter checks for hypothetical tokens, support selection and control weights."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recommend_sequence import action_deliveries, recommend, select_actions, sequence_predictor


class AdapterTests(unittest.TestCase):
    def test_action_selection_covers_types_before_extra_targets(self):
        actions = [dict(pitch_type=t, training_local_n=n, target_x_ft=x, target_z_ft=2.5)
                   for t, n, x in [('FF', 100, 0), ('FF', 90, 1), ('SL', 50, 0), ('SL', 40, 1)]]
        selected = select_actions(actions, 3)
        self.assertEqual([(a['pitch_type'], a['training_local_n']) for a in selected],
                         [('FF', 100), ('SL', 50), ('FF', 90)])

    def test_control_points_use_train_pool_and_hypothetical_target(self):
        class Delivery:
            TIERS = [('pitcher', 'pitch_type')]

            def sample(self, frame):
                return np.array([[[1.] * 8, [3.] * 8]]), np.array([0])

        normalizer = SimpleNamespace(mean=np.zeros(8), scale=np.full(8, 2.))
        points, weights, levels = action_deliveries(
            pd.Series(dict(pitcher=1, pitch_type='FF', plate_x=999, plate_z=999)),
            [dict(pitch_type='SL', target_x_ft=.4, target_z_ft=2.4)], Delivery(), normalizer, .3)
        np.testing.assert_allclose(points[0, :, :6], 2)
        np.testing.assert_allclose(weights @ points[0, :, 6:], [.2, 1.2])
        raw_x = points[0, :, 6] * 2
        self.assertAlmostEqual(weights @ ((raw_x - .4) ** 2), .3 ** 2)
        self.assertAlmostEqual(weights.sum(), 1)

    def test_predictor_uses_branch_histories_counts_and_masks_illegal_dp(self):
        seen = []

        class Model:
            def predict(self, arrays):
                seen.append(arrays)
                p = np.zeros((len(arrays[0]), 10))
                p[:, 9] = 1
                return p

        context = SimpleNamespace(transform=lambda frame: np.array([[0., 0., 123.]], dtype=np.float32))
        row = pd.Series(dict(outs_when_up=2, bases=1))
        predict = sequence_predictor(Model(), context, row)
        p = predict([np.empty((0, 8)), np.full((2, 8), 7.)], np.array([[0, 1], [3, 2]]),
                    np.full((2, 8), 9.))
        tokens, valid, transformed = seen[0]
        np.testing.assert_array_equal(valid[0], [False] * 5 + [True])
        np.testing.assert_array_equal(valid[1], [False] * 3 + [True] * 3)
        np.testing.assert_allclose(tokens[1, 3:5], 7)
        np.testing.assert_allclose(tokens[:, -1], 9)
        np.testing.assert_allclose(transformed, [[0, .5, 123], [1, 1, 123]])
        np.testing.assert_allclose(p[:, 3], 1)
        np.testing.assert_allclose(p[:, 9], 0)

    def test_recommend_rejects_missing_actual_history_for_later_pitch(self):
        with self.assertRaisesRegex(ValueError, 'observed histories'):
            recommend(None, {}, {}, pd.Series(dict(pitch_number=2, balls=1, strikes=0)))


if __name__ == '__main__':
    unittest.main()
