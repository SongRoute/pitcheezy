"""CPU-only robustness aggregation, mask, and identity contracts; never fit models."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))

from run_sequence_robustness import (FixedMaskedContext, _completed, _verified_predictions,
                                     aggregate_comparison, per_pitch_scores)
from run_sequence_ablations import CONTEXT_SCHEMA, MaskedContext


def probabilities_of_true_class(values):
    p = np.asarray(values, dtype=float)
    return np.stack((p, 1-p), axis=-1)


class RobustnessAggregationContracts(unittest.TestCase):
    def test_average_single_model_loss_is_not_ensemble_loss(self):
        model = probabilities_of_true_class([[.9, .9], [.1, .1]])
        reference = np.full_like(model, .5)
        result = aggregate_comparison(np.zeros(2, dtype=int), model, reference,
                                      np.array([100, 200]), [42, 43], replicates=200)
        expected = -(np.log(.9)+np.log(.1))/2
        self.assertAlmostEqual(result['log_loss']['mean_model'], expected)
        self.assertAlmostEqual(result['log_loss']['model_minus_reference'], expected-np.log(2))
        self.assertGreater(result['log_loss']['mean_model'], np.log(2)+.5)
        self.assertEqual(result['games'], 2)

    def test_per_pitch_log_loss_and_brier_match_hand_calculation(self):
        p = probabilities_of_true_class([[.75, .6], [.25, .4]])
        labels = np.array([0, 1])
        scores = per_pitch_scores(labels, p)
        expected_true_p = np.array([[.75, .4], [.25, .6]])
        np.testing.assert_allclose(scores['log_loss'], -np.log(expected_true_p))
        np.testing.assert_allclose(scores['brier_multiclass'], 2*(1-expected_true_p)**2)

    def test_identical_seed_matched_models_have_exact_zero_difference_and_intervals(self):
        p = probabilities_of_true_class([[.8, .3, .6], [.4, .7, .9], [.2, .9, .7]])
        result = aggregate_comparison(np.zeros(3, dtype=int), p, p.copy(),
                                      np.array([900, 900, 901]), [42, 43, 44], replicates=200)
        for metric in ['log_loss', 'brier_multiclass']:
            item = result[metric]
            self.assertEqual(item['model_minus_reference'], 0.)
            np.testing.assert_array_equal(list(item['per_seed_difference'].values()), 0.)
            np.testing.assert_array_equal(item['game_only_bootstrap95'], [0., 0.])
            np.testing.assert_array_equal(item['crossed_seed_game_bootstrap95'], [0., 0.])

    def test_unequal_game_sizes_keep_pitch_weighted_target(self):
        p = probabilities_of_true_class([[.9, .2, .2, .2], [.9, .2, .2, .2]])
        result = aggregate_comparison(np.zeros(4, dtype=int), p, np.full_like(p, .5),
                                      np.array([10, 20, 20, 20]), [42, 43], replicates=200)
        differences = -np.log(np.array([.9, .2]))-np.log(2)
        expected = (differences[0]+3*differences[1])/4
        self.assertAlmostEqual(result['log_loss']['model_minus_reference'], expected)
        self.assertNotAlmostEqual(expected, differences.mean())
        self.assertEqual(result['games'], 2)
        np.testing.assert_allclose(result['log_loss']['game_only_bootstrap95'],
                                   [differences.min(), differences.max()])

    def test_crossed_interval_retains_training_variation_when_games_identical(self):
        p = probabilities_of_true_class(np.repeat(np.array([.2, .3, .5, .7, .8])[:, None], 6, axis=1))
        result = aggregate_comparison(np.zeros(6, dtype=int), p, np.full_like(p, .5),
                                      np.arange(6), list(range(42, 47)), replicates=1000)
        item = result['log_loss']
        np.testing.assert_allclose(item['game_only_bootstrap95'], [item['model_minus_reference']]*2)
        self.assertGreater(np.ptp(item['crossed_seed_game_bootstrap95']), .1)
        self.assertEqual(len(item['per_seed_difference']), 5)
        self.assertEqual(result['games'], 6)

    def test_crossed_bootstrap_matches_explicit_jointly_paired_row_resampling(self):
        p = probabilities_of_true_class([[.2, .4, .8, .3], [.7, .6, .9, .2], [.5, .2, .4, .8]])
        q = probabilities_of_true_class([[.6, .3, .4, .5], [.4, .7, .6, .3], [.3, .6, .5, .4]])
        y, games, replicates = np.zeros(4, dtype=int), np.array([10, 20, 20, 30]), 200
        result = aggregate_comparison(y, p, q, games, [42, 43, 44], replicates=replicates)
        rng = np.random.default_rng(42)
        game_draws = rng.integers(0, 3, size=(replicates, 3))
        seed_draws = rng.integers(0, 3, size=(replicates, 3))
        differences = -np.log(p[:, :, 0])+np.log(q[:, :, 0])
        game_values, crossed_values = [], []
        for sampled_games, sampled_seeds in zip(game_draws, seed_draws):
            rows = np.concatenate([np.flatnonzero(games == np.unique(games)[game]) for game in sampled_games])
            game_values.append(differences[:, rows].mean())
            crossed_values.append(differences[sampled_seeds][:, rows].mean())
        np.testing.assert_allclose(result['log_loss']['game_only_bootstrap95'], np.quantile(game_values, [.025, .975]))
        np.testing.assert_allclose(result['log_loss']['crossed_seed_game_bootstrap95'], np.quantile(crossed_values, [.025, .975]))

    def test_invalid_probabilities_and_duplicate_seed_ids_are_rejected(self):
        labels, games = np.zeros(2, dtype=int), np.array([1, 2])
        p = np.full((2, 2, 2), .5)
        with self.assertRaises(ValueError):
            aggregate_comparison(labels, p, p, games, [42, 42], replicates=100)
        for value in [-1., np.nan, 1.5]:
            invalid = p.copy()
            invalid[0, 0, 0] = value
            with self.assertRaises(ValueError):
                per_pitch_scores(labels, invalid)


class RobustnessContextMaskContracts(unittest.TestCase):
    def test_context_mask_cannot_erase_legal_count_handedness_or_mutate_shared_arrays(self):
        raw = np.arange(1, 85, dtype=np.float32).reshape(3, 28)
        class Encoder:
            def report(self):
                return {'features': CONTEXT_SCHEMA}
            def transform(self, frame):
                return raw
        original = raw.copy()
        for variant, remove in [('no_game_context', list(range(2, 9))),
                                ('no_batter_style', list(range(11, 28))),
                                ('no_clusters', list(range(23, 28)))]:
            encoded = FixedMaskedContext(Encoder(), variant).transform(None)
            keep = [column for column in range(28) if column not in remove]
            np.testing.assert_array_equal(encoded[:, keep], original[:, keep])
            np.testing.assert_array_equal(encoded[:, remove], 0.)
            np.testing.assert_array_equal(raw, original)
            np.testing.assert_array_equal(encoded[:, [0, 1, 9, 10]], original[:, [0, 1, 9, 10]])


class RobustnessReuseContracts(unittest.TestCase):
    def test_completed_identity_rejects_wrong_seed_variant_and_mutated_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(_completed(directory, 43, 'full_transformer'))
            (directory/'model.pt').write_bytes(b'synthetic-checkpoint-no-torch-data')
            (directory/'predictions.npz').write_bytes(b'synthetic-predictions')
            result = {'seed': 43, 'variant': 'full_transformer',
                      'artifact_hashes': {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                                          for name in ['model.pt', 'predictions.npz']}}
            (directory/'result.json').write_text(json.dumps(result))
            self.assertEqual(_completed(directory, 43, 'full_transformer'), result)
            for seed, variant in [(44, 'full_transformer'), (43, 'flatten_mlp')]:
                with self.assertRaises(ValueError):
                    _completed(directory, seed, variant)
            (directory/'model.pt').write_bytes(b'changed-checkpoint')
            with self.assertRaises(ValueError):
                _completed(directory, 43, 'full_transformer')

    def test_prediction_reuse_requires_exact_labels_keys_and_games(self):
        y = np.array([0, 1])
        keys = np.array([[90, 1, 1], [91, 1, 1]])
        games = np.array([90, 91])
        p = np.array([[.7, .3], [.4, .6]])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'predictions.npz'
            np.savez(path, y=y, pitch_keys=keys, game_pk=games, delivery_integrated_calibrated=p)
            restored = _verified_predictions(path, y, keys, games)
            np.testing.assert_array_equal(restored['delivery_integrated_calibrated'], p)
            for changed_y, changed_keys, changed_games in [(y[::-1], keys, games),
                                                           (y, keys[::-1], games),
                                                           (y, keys, games[::-1])]:
                with self.assertRaises(ValueError):
                    _verified_predictions(path, changed_y, changed_keys, changed_games)


if __name__ == '__main__':
    unittest.main()
