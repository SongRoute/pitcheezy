"""Synthetic CPU-only contracts for batter representations and history windows."""
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
import numpy as np
import pandas as pd
import torch

from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.sequence_model import SequenceContext
from representation_adapters import BatterContext, IDModel, IDNetwork, WindowStore


def observations():
    n = 40
    rng = np.random.default_rng(77)
    frame = pd.DataFrame({'game_date': pd.date_range('2024-03-01', periods=n), 'split': 'train',
        'game_pk': np.arange(n), 'at_bat_number': 1, 'pitch_number': 1,
        'batter': np.arange(n) % 10 + 100, 'balls': np.arange(n) % 4,
        'strikes': np.arange(n) % 3, 'outs_when_up': np.arange(n) % 3,
        'inning': np.arange(n) % 9 + 1, 'bases': np.arange(n) % 8,
        'home_score': 2, 'away_score': 3, 'inning_topbot': ['Top', 'Bot']*(n//2),
        'stand': ['L', 'R']*(n//2), 'p_throws': ['R', 'L']*(n//2)})
    frame[list(STYLE_COLUMNS)] = rng.uniform(.1, .8, (n, 6))
    frame[list(RELIABILITY_COLUMNS)] = rng.uniform(.1, .9, (n, 6))
    return frame


class RepresentationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = observations()
        cls.base = SequenceContext().fit(cls.train)

    def test_reference_and_continuous_exactly_preserve_frozen_channels(self):
        original = self.base.transform(self.train)
        for mode, expected in [('reference', original), ('continuous', original[:, :23]),
                               ('hand', original[:, :11])]:
            adapter = BatterContext(self.base, mode).fit(self.train)
            np.testing.assert_array_equal(adapter.transform(self.train), expected)
            self.assertEqual(adapter.report()['n_context'], expected.shape[1])

    def test_hand_and_id_ignore_outcomes_and_style_values(self):
        changed = self.train.copy()
        changed[list(STYLE_COLUMNS)+list(RELIABILITY_COLUMNS)] = np.nan
        changed['description'], changed['events'] = 'FUTURE_OUTCOME', 'home_run'
        for mode in ('hand', 'id'):
            adapter = BatterContext(self.base, mode).fit(self.train, id_train=self.train.iloc[:3])
            expected = adapter.transform(self.train)
            np.testing.assert_array_equal(adapter.transform(changed), expected)
        hand = BatterContext(self.base, 'hand').fit(self.train)
        changed['batter'] = 999999
        np.testing.assert_array_equal(hand.transform(changed), hand.transform(self.train))

    def test_id_vocabulary_uses_only_neural_train_and_unknown_is_zero(self):
        adapter = BatterContext(self.base, 'id').fit(self.train, id_train=self.train.iloc[:2])
        self.assertEqual(adapter.vocab_size, 3)
        query = self.train.iloc[:4].copy()
        query['batter'] = [100., 101., 102., np.nan]
        encoded = adapter.transform(query)
        np.testing.assert_array_equal(encoded[:, -1], [1, 2, 0, 0])
        query['batter'] = [999, 999, 999, 999]
        np.testing.assert_array_equal(adapter.transform(query)[:, -1], np.zeros(4))
        self.assertEqual(adapter.vocab_size, 3)
        network = IDNetwork(adapter.vocab_size)
        np.testing.assert_array_equal(network.embedding(torch.tensor([0])).detach().numpy(), np.zeros((1, 16)))

    def test_only_train_can_fit_clusters_or_id_vocabulary(self):
        future = self.train.copy()
        future['split'] = 'dev'
        with self.assertRaises(ValueError):
            BatterContext(self.base, 'clusters', 3).fit(future)
        with self.assertRaises(ValueError):
            BatterContext(self.base, 'id').fit(self.train, id_train=future)
        future['split'], future['game_date'] = 'train', pd.Timestamp('2025-07-01')
        with self.assertRaises(ValueError):
            BatterContext(self.base, 'clusters', 3).fit(future)
        outsider = self.train.iloc[:1].copy()
        outsider['game_pk'] = 99999
        with self.assertRaisesRegex(ValueError, 'belong'):
            BatterContext(self.base, 'id').fit(self.train, id_train=outsider)
        later = self.train.copy()
        later['game_date'] = later.game_date + pd.DateOffset(years=1)
        later_base = SequenceContext().fit(later)
        with self.assertRaisesRegex(ValueError, 'geometry extends'):
            BatterContext(later_base, 'continuous').fit(self.train)
        with self.assertRaisesRegex(ValueError, 'dates extend'):
            BatterContext(self.base, 'id').fit(self.train, id_train=later)

    def test_clusters_are_soft_only_and_do_not_refit_on_query(self):
        for count in (3, 5, 10, 20):
            adapter = BatterContext(self.base, 'clusters', count).fit(self.train)
            original = adapter.transform(self.train)
            self.assertEqual(original.shape, (len(self.train), 11+count))
            np.testing.assert_allclose(original[:, 11:].sum(1), 1., atol=1e-6)
            query = self.train.copy()
            query['game_date'], query['split'] = pd.Timestamp('2025-09-01'), 'dev'
            query['batter'] = 999999
            np.testing.assert_array_equal(adapter.transform(query), original)
            np.testing.assert_array_equal(adapter.transform(self.train), original)
            self.assertLessEqual(adapter.report()['archetypes']['fit_date_max'], '2024-04-30')
        k5 = BatterContext(self.base, 'clusters', 5).fit(self.train)
        np.testing.assert_array_equal(k5.transform(self.train)[:, 11:], self.base.transform(self.train)[:, 23:])

    def test_window_preserves_current_override_and_does_not_mutate_store(self):
        class Store:
            frame = pd.DataFrame({'dummy': [1, 2]})
            normalizer = object()
            tokens = np.arange(96, dtype=np.float32).reshape(2, 6, 8)
            valid = np.array([[True]*6, [False, False, True, True, True, True]])
            def gather(self, rows, current=None):
                result = self.tokens.copy()
                if current is not None:
                    result[:, -1] = current
                return result, self.valid
        store = Store()
        before = store.tokens.copy()
        valid_before = store.valid.copy()
        override = np.full((2, 8), -9, dtype=np.float32)
        for count in range(1, 6):
            adapter = WindowStore(store, count)
            tokens, valid = adapter.gather([0, 1], current=override)
            self.assertIs(adapter.frame, store.frame)
            self.assertIs(adapter.normalizer, store.normalizer)
            np.testing.assert_array_equal(tokens[:, -1], override)
            np.testing.assert_array_equal(tokens[:, :5-count], np.zeros((2, 5-count, 8)))
            self.assertFalse(valid[:, :5-count].any())
            np.testing.assert_array_equal(tokens[:, 5-count:-1], before[:, 5-count:-1])
            np.testing.assert_array_equal(valid[:, 5-count:], valid_before[:, 5-count:])
        np.testing.assert_array_equal(store.tokens, before)
        np.testing.assert_array_equal(store.valid, valid_before)
        for bad in (0, 6, 1.5, True):
            with self.assertRaises(ValueError):
                WindowStore(store, bad)

    def test_id_cpu_fit_padding_checkpoint_and_roundtrip(self):
        adapter = BatterContext(self.base, 'id').fit(self.train, id_train=self.train.iloc[:4])
        context = adapter.transform(self.train.iloc[:8])
        rng = np.random.default_rng(8)
        tokens = rng.normal(size=(8, 6, 8)).astype(np.float32)
        valid = np.ones((8, 6), dtype=bool)
        data, labels = (tokens, valid, context), np.arange(8) % 3
        with tempfile.TemporaryDirectory() as directory:
            model = IDModel(adapter.vocab_size, seed=42, width=16)
            model.device = 'cpu'
            self.assertEqual(model.device, 'cpu')
            checkpoint = Path(directory)/'best.pt'
            model.fit(data, labels, data, labels, epochs=1, patience=1, batch_size=4,
                      learning_rate=.0005, checkpoint=checkpoint)
            self.assertTrue(checkpoint.exists())
            np.testing.assert_array_equal(model.net.embedding.weight[0].detach().numpy(), np.zeros(16))
            expected = model.predict(data)
            path = Path(directory)/'model.pt'
            model.delivery_temperature = 1.25
            model.save(path)
            restored = IDModel.load(path, device='cpu')
            np.testing.assert_array_equal(restored.predict(data), expected)
            self.assertEqual(restored.delivery_temperature, 1.25)
            bad_context = context.copy()
            bad_context[0, -1] = .5
            with self.assertRaises(ValueError):
                restored.logits((tokens, valid, bad_context))


if __name__ == '__main__':
    unittest.main()
