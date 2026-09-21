"""CPU-only integration/calibration archive contracts; no fitted model required."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import softmax
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_sequence_integration_reevaluation import probability_pair, tensor_hash, ensemble_calibration


class IntegrationTests(unittest.TestCase):
    def test_average_probabilities_after_temperature_not_average_logits(self):
        logits = np.array([[[3., 0.], [0., 1.]]])
        result = probability_pair(logits, 1.5)
        expected = (softmax(logits[0, 0]/1.5) + softmax(logits[0, 1]/1.5))/2
        np.testing.assert_allclose(result['delivery_integrated_calibrated'][0], expected)
        self.assertFalse(np.allclose(expected, softmax(logits.mean(axis=1)[0]/1.5)))
        np.testing.assert_allclose(result['delivery_integrated_uncalibrated'].sum(axis=1), 1)

    def test_tensor_identity_ignores_temperature_but_detects_weight_change(self):
        model = SimpleNamespace(net=torch.nn.Linear(2, 2), delivery_temperature=1.)
        original = tensor_hash(model)
        model.delivery_temperature = 1.3
        self.assertEqual(tensor_hash(model), original)
        with torch.no_grad():
            model.net.weight[0, 0] += .1
        self.assertNotEqual(tensor_hash(model), original)

    def test_compatible_calibration_archives_keep_five_members_and_exact_rows(self):
        y = np.array([0, 1, 0, 1])
        games = np.array([1, 1, 2, 2])
        keys = np.column_stack([games, np.arange(4), np.ones(4, int)])
        seed_p = np.array([[[.7, .3], [.2, .8], [.6, .4], [.3, .7]] for _ in range(5)])
        baseline = np.full((4, 2), .5)
        config = {'source_hashes': {}, 'reference_hashes': {'pool.pkl': 'hash'},
                  'delivery100_path': 'pool.pkl', 'pool_randomness': 'not nested',
                  'temperature_row_hash': 'temperaturehash'}
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            ensemble_calibration(directory, directory, directory, None, y, y, keys, keys, games, games,
                                 'calhash', 'devhash', baseline, baseline, config, seed_p, seed_p)
            result = json.loads((directory/'calibration100/results.json').read_text())
            self.assertEqual(result['calibration_row_hash'], 'calhash')
            self.assertEqual(result['delivery_draws'], 100)
            for filename in ['predictions.npz', 'calibration_predictions.npz']:
                with np.load(directory/'calibration100'/filename) as saved:
                    np.testing.assert_array_equal(saved['pitch_keys'], keys)
                    np.testing.assert_array_equal(saved['y'], y)
                    np.testing.assert_allclose(saved['ensemble'], saved['seed_predictions'].mean(axis=0))
                    self.assertEqual(saved['seed_predictions'].shape, (5, 4, 2))


if __name__ == '__main__':
    unittest.main()
