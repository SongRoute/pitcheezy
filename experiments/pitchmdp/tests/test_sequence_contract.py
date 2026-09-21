"""Independent synthetic contracts; no training, external data, or downloads."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.archetypes import INITIAL_RATES, RELIABILITY_COLUMNS, STYLE_COLUMNS
from pitchmdp.sequence_data import HistoryStore, PHYSICAL_COLUMNS, PhysicalNormalizer
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceContext, SequenceModel, SequenceNetwork, classification_metrics


def synthetic_frame():
    n = 12
    frame = pd.DataFrame({
        'game_pk': [1]*8 + [2]*4, 'at_bat_number': [1]*8 + [1]*4,
        'pitch_number': list(range(1, 9)) + list(range(1, 5)),
        'game_date': pd.to_datetime(['2024-04-01']*8 + ['2024-04-02']*4),
        'split': ['train']*n, 'batter': np.arange(n), 'pitcher': [200]*n,
        'balls': np.arange(n)%4, 'strikes': np.arange(n)%3,
        'outs_when_up': np.arange(n)%3, 'inning': np.arange(n)%9+1,
        'home_score': np.arange(n)%5, 'away_score': [1]*n,
        'bases': np.arange(n)%8, 'inning_topbot': ['Top', 'Bot']*6,
        'stand': ['R', 'L']*6, 'p_throws': ['R']*n, 'pitch_type': ['FF']*n,
    })
    for j, name in enumerate(PHYSICAL_COLUMNS):
        frame[name] = np.arange(n, dtype=float) + 2*j
    frame['spin_axis'] = np.linspace(5., 345., n)
    for j, name in enumerate(STYLE_COLUMNS):
        frame[name] = INITIAL_RATES[j] + np.arange(n)*.001
    for name in RELIABILITY_COLUMNS:
        frame[name] = .5
    return frame


class SequenceInformationContracts(unittest.TestCase):
    def test_six_tokens_mean_five_prior_and_current_and_reset_at_game_boundary(self):
        store = HistoryStore.from_frame(synthetic_frame())
        tokens, valid = store.gather([0, 6, 8])
        self.assertEqual(tokens.shape, (3, 6, 8))
        np.testing.assert_array_equal(valid[0], [False]*5+[True])
        np.testing.assert_array_equal(valid[2], [False]*5+[True])
        np.testing.assert_array_equal(tokens[1], store.physical[1:7])
        np.testing.assert_array_equal(tokens[0, :-1], 0.)

    def test_candidate_override_and_prior_tokens_ignore_current_and_future_physics(self):
        frame = synthetic_frame()
        normalizer = PhysicalNormalizer().fit(frame)
        before = HistoryStore.from_frame(frame, normalizer)
        changed = frame.copy()
        changed.loc[6:, list(PHYSICAL_COLUMNS)] = 1e6
        after = HistoryStore.from_frame(changed, normalizer)
        candidate = np.arange(8, dtype=np.float32)[None, :]
        expected = before.gather([6], current=candidate)
        actual = after.gather([6], current=candidate)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(actual[0][:, -1], candidate)
        self.assertFalse(np.array_equal(before.gather([6])[0], after.gather([6])[0]))

    def test_current_outcomes_and_ids_never_enter_physical_or_context_features(self):
        frame = synthetic_frame()
        context = SequenceContext().fit(frame)
        normalizer = PhysicalNormalizer().fit(frame)
        changed = frame.copy()
        forbidden = ['batter', 'pitcher', 'description', 'events', 'next_outs',
                     'next_bases', 'next_inning', 'next_home_score', 'next_away_score',
                     'post_home_score', 'post_away_score', 'launch_speed', 'launch_angle',
                     'home_win_exp', 'bat_win_exp', 'delta_home_win_exp', 'pitch_outcome']
        for column in forbidden:
            changed[column] = 'FORBIDDEN_FUTURE_INFORMATION'
        np.testing.assert_array_equal(context.transform(frame), context.transform(changed))
        np.testing.assert_array_equal(normalizer.transform(frame), normalizer.transform(changed))
        # Even the current physical realization is absent from the context branch.
        changed.loc[:, list(PHYSICAL_COLUMNS)] = -1e9
        np.testing.assert_array_equal(context.transform(frame), context.transform(changed))

    def test_normalization_and_context_reject_calibration_fit(self):
        frame = synthetic_frame()
        frame.loc[0, 'split'] = 'calibration'
        for encoder in (PhysicalNormalizer(), SequenceContext()):
            with self.assertRaises(ValueError):
                encoder.fit(frame)

    def test_spin_axis_is_periodic_and_transform_does_not_refit(self):
        frame = synthetic_frame()
        normalizer = PhysicalNormalizer().fit(frame)
        mean, scale = normalizer.mean.copy(), normalizer.scale.copy()
        changed = frame.copy()
        changed['spin_axis'] += 720
        np.testing.assert_allclose(normalizer.transform(changed), normalizer.transform(frame), atol=1e-6)
        changed.loc[:, list(PHYSICAL_COLUMNS)] = np.nan
        self.assertTrue(np.isfinite(normalizer.transform(changed)).all())
        np.testing.assert_array_equal(mean, normalizer.mean)
        np.testing.assert_array_equal(scale, normalizer.scale)


class SequenceNetworkContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def arrays(self):
        torch.manual_seed(10)
        return (torch.randn(2, 6, 8),
                torch.tensor([[False]*5+[True], [False, False, True, True, True, True]]),
                torch.randn(2, 7))

    def test_padding_payload_does_not_change_any_architecture(self):
        tokens, valid, context = self.arrays()
        changed = tokens.clone()
        changed[~valid] = float('nan')
        for kind in ('transformer', 'flatten_mlp', 'current_only'):
            net = SequenceNetwork(kind, 7, width=16).eval()
            with torch.no_grad():
                expected, actual = net(tokens, valid, context), net(changed, valid, context)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(actual).all())

    def test_memoryless_model_ignores_history_but_ordered_models_can_use_it(self):
        tokens, valid, context = self.arrays()
        valid[:] = True
        changed = tokens.clone()
        changed[:, :5] = tokens[:, :5].flip(1)
        for kind in ('transformer', 'flatten_mlp', 'current_only'):
            net = SequenceNetwork(kind, 7, width=16).eval()
            with torch.no_grad():
                expected, actual = net(tokens, valid, context), net(changed, valid, context)
            if kind == 'current_only':
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                self.assertGreater(float((actual-expected).abs().max()), 1e-7)

    def test_binary_and_ten_class_heads_return_finite_probabilities(self):
        arrays = self.arrays()
        for classes in (2, 10):
            for kind in ('transformer', 'flatten_mlp', 'current_only'):
                net = SequenceNetwork(kind, 7, n_classes=classes, width=16).eval()
                with torch.no_grad():
                    p = net(*arrays).softmax(-1)
                self.assertEqual(p.shape, (2, classes))
                self.assertTrue(torch.isfinite(p).all())
                torch.testing.assert_close(p.sum(-1), torch.ones(2))


class SequenceMetricContracts(unittest.TestCase):
    def test_logloss_brier_and_auc_match_small_example(self):
        p = np.array([[.75, .25], [.4, .6]])
        result = classification_metrics([0, 1], p)
        self.assertAlmostEqual(result['log_loss'], -(np.log(.75)+np.log(.6))/2)
        self.assertAlmostEqual(result['brier_multiclass'], (.25**2+.4**2))
        self.assertEqual(result['auc'], 1.)

    def test_calibration_bins_count_boundaries_once(self):
        result = classification_metrics([0, 1], np.array([[.5, .5], [.5, .5]]))
        self.assertEqual(result['top_label_ece10'], 0.)
        result = classification_metrics([0, 0], np.array([[.5, .5], [.5, .5]]))
        self.assertAlmostEqual(result['top_label_ece10'], .5)

    def test_invalid_mass_is_rejected(self):
        for p in (np.array([[.9, .9]]), np.array([[np.nan, np.nan]]), np.array([[-.1, 1.1]])):
            with self.assertRaises(ValueError):
                classification_metrics([0], p)


class PhysicalResponseStub:
    temperature = 1.

    def logits(self, arrays):
        tokens, valid, context = arrays
        score = tokens[:, -1, 0] + .2*tokens[:, :-1, 0].sum(1) + .1*context[:, 0]
        return np.column_stack([score, -score])


class SequenceDeliveryContracts(unittest.TestCase):
    def test_delivery_pools_use_train_only_joint_vectors(self):
        frame = synthetic_frame()
        normalizer = PhysicalNormalizer().fit(frame)
        delivery = JointDelivery().fit(frame, normalizer, draws=7, seed=9)
        samples, levels = delivery.sample(frame)
        physical = normalizer.transform(frame)
        for vector in samples.reshape(-1, physical.shape[1]):
            self.assertTrue(np.any(np.all(physical == vector, axis=1)))
        changed = frame.copy()
        changed.loc[0, 'split'] = 'dev'
        with self.assertRaises(ValueError):
            JointDelivery().fit(changed, normalizer, draws=7, seed=9)

    def test_integrated_prediction_ignores_realized_current_physics(self):
        frame = synthetic_frame()
        normalizer = PhysicalNormalizer().fit(frame)
        context = SequenceContext().fit(frame)
        delivery = JointDelivery().fit(frame, normalizer, draws=3, seed=9)
        before = HistoryStore.from_frame(frame, normalizer)
        changed = frame.copy()
        # Change only evaluated current rows; their histories are unchanged.
        changed.loc[[6, 11], list(PHYSICAL_COLUMNS)] = 1e6
        after = HistoryStore.from_frame(changed, normalizer)
        expected = delivery.predict(PhysicalResponseStub(), before, context, [6, 11])
        actual = delivery.predict(PhysicalResponseStub(), after, context, [6, 11])
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_allclose(actual.sum(1), 1.)

    def test_delivery_calibration_survives_checkpoint_round_trip(self):
        model = SequenceModel('current_only', width=16, n_classes=2)
        model.device = 'cpu'
        model.net = SequenceNetwork('current_only', 7, width=16, n_classes=2)
        model.temperature = 1.2
        model.delivery_temperature = 2.1
        model.report = {'delivery_temperature': model.delivery_temperature}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            model.save(path)
            restored = SequenceModel.load(path)
        self.assertEqual(restored.temperature, model.temperature)
        self.assertEqual(restored.delivery_temperature, model.delivery_temperature)


if __name__ == '__main__':
    unittest.main()
