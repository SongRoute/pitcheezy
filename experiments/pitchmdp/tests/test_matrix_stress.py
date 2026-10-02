"""Tiny CPU-only T3 contracts, with no real observations or neural training."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.archetypes import INITIAL_RATES, STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.matrix_benchmark import predict_streamed
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_sharing import SharingContext, SharingPredictor
from pitchmdp.matrix_stress import (SCENARIOS, StressHistoryStore, StressSharingContext,
                                   predict_stress_streamed, protocol)
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceContext


def fixture():
    n = 36
    frame = pd.DataFrame({'game_pk': np.repeat(np.arange(1, 7), 6), 'at_bat_number': 1,
        'pitch_number': np.tile(np.arange(1, 7), 6),
        'game_date': np.repeat(pd.date_range('2024-04-01', periods=6), 6),
        'split': 'train', 'pitcher': 10, 'batter': np.repeat(np.arange(6), 6),
        'pitch_type': np.tile(['FF', 'SL'], 18), 'description': 'ball', 'events': None,
        'balls': 0, 'strikes': 0, 'p_throws': 'R', 'stand': 'L', 'outs_when_up': 0,
        'inning': 1, 'inning_topbot': 'Top', 'home_score': 0, 'away_score': 0, 'bases': 0,
        'effective_speed': np.linspace(85, 95, n), 'release_spin_rate': np.linspace(1800, 2400, n),
        'spin_axis': np.linspace(0, 350, n), 'pfx_x': np.linspace(-1, 1, n),
        'pfx_z': np.linspace(.5, 1.5, n), 'plate_x': np.linspace(-1, 1, n),
        'plate_z': np.linspace(1, 4, n)})
    for j, name in enumerate(STYLE_COLUMNS):
        frame[name] = INITIAL_RATES[j] + np.linspace(-.05, .05, n)
    for name in RELIABILITY_COLUMNS:
        frame[name] = .7
    base_context = SequenceContext().fit(frame)
    clusters = {'columns': ['synthetic'], 'pitcher_profiles': {'10': [2.]},
                'pitcher_cluster': {'10': 0}, 'pitcher_counts': {'10': 500},
                'cluster_counts': {'0': 500}}
    return MatrixHistoryStore.from_frame(frame), SharingContext(base_context, clusters)


class FixedModel:
    def __init__(self, label=0):
        self.label = label
        self.delivery_temperature = 1.3

    def logits(self, arrays):
        tokens, mask, context = arrays
        result = np.zeros((len(tokens), 10))
        result[:, self.label] = 2. + tokens[:, -1, 0] / 20.
        return result


class StressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store, cls.context = fixture()

    def test_all_scenarios_preserve_current_candidate_and_source(self):
        original = self.store.gather([0, 5])
        frame = self.store.frame.copy(deep=True)
        physical = self.store.physical.copy()
        current = np.full((2, 8), 7., dtype=np.float32)
        clean, _ = self.store.gather([0, 5], current, ['SL', 'FF'])
        for scenario in SCENARIOS:
            wrapper = StressHistoryStore(self.store, scenario)
            tokens, mask = wrapper.gather([0, 5], current, ['SL', 'FF'])
            np.testing.assert_array_equal(tokens[:, -1], clean[:, -1])
            self.assertTrue(mask[:, -1].all())
            self.assertTrue((tokens[~mask] == 0.).all())
            self.assertTrue((tokens[:, -1, -11:] == 0.).all())
            with self.assertRaises(ValueError):
                wrapper.gather([0])
        pd.testing.assert_frame_equal(self.store.frame, frame)
        np.testing.assert_array_equal(self.store.physical, physical)
        np.testing.assert_array_equal(self.store.gather([0, 5])[0], original[0])

    def test_history_lengths_and_unknown_fields_have_distinct_masks(self):
        current = np.zeros((1, 8))
        for scenario, expected in [('H0', 1), ('H2', 3), ('unknown_type_outcome', 6)]:
            tokens, mask = StressHistoryStore(self.store, scenario).gather([5], current)
            self.assertEqual(mask.sum(), expected)
            if scenario == 'unknown_type_outcome':
                self.assertTrue((tokens[0, :5, 8] == 1).all())
                self.assertTrue((tokens[0, :5, -1] == 1).all())
                self.assertTrue((tokens[0, :5, 8:].sum(1) == 2).all())

    def test_current_and_future_realizations_never_enter_stressed_history(self):
        changed = self.store.frame.copy()
        changed.loc[2:, 'description'] = 'hit_by_pitch'
        changed.loc[2:, 'effective_speed'] = 999.
        other = MatrixHistoryStore.from_frame(changed, self.store.normalizer,
                                              type_vocabulary=self.store.type_vocabulary)
        current = np.ones((1, 8), dtype=np.float32)
        for scenario in SCENARIOS:
            expected = StressHistoryStore(self.store, scenario).gather([2], current)
            actual = StressHistoryStore(other, scenario).gather([2], current)
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)

    def test_repeated_draws_queries_and_batching_share_historical_noise(self):
        wrapper = StressHistoryStore(self.store, 'sensor03')
        rows = np.repeat([4, 5], 400)
        tokens, _ = wrapper.gather(rows, np.zeros((800, 8)))
        np.testing.assert_array_equal(tokens[0, :-1], tokens[399, :-1])
        # Pitch row 0 occurs at different H5 positions for the two queries.
        np.testing.assert_array_equal(tokens[0, 1], tokens[400, 0])
        reversed_tokens, _ = wrapper.gather([5, 4], np.zeros((2, 8)))
        np.testing.assert_array_equal(reversed_tokens[0], tokens[400])
        np.testing.assert_array_equal(reversed_tokens[1], tokens[0])

    def test_masks_nested_and_key_based_across_queries(self):
        rows = np.array([4, 5, 4, 5])
        current = np.zeros((4, 8))
        _, a = StressHistoryStore(self.store, 'mask20').gather(rows, current)
        _, b = StressHistoryStore(self.store, 'mask50').gather(rows, current)
        self.assertTrue((~b | a).all())
        self.assertEqual(a[0, 1], a[1, 0])
        np.testing.assert_array_equal(a[:2], a[2:])
        # All test keys need not be dropped, but the two protocols cannot be aliases.
        rows = np.arange(36)
        _, b = StressHistoryStore(self.store, 'mask50').gather(rows, np.zeros((36, 8)))
        self.assertLess(b.sum(), self.store.gather(rows)[1].sum())

    def test_sensor_scale_and_circular_geometry(self):
        current = np.zeros((1, 8))
        clean, _ = self.store.gather([5], current)
        low, _ = StressHistoryStore(self.store, 'sensor01').gather([5], current)
        high, _ = StressHistoryStore(self.store, 'sensor03').gather([5], current)
        non_angle = [0, 1, 4, 5, 6, 7]
        np.testing.assert_allclose((high-clean)[0, :5, non_angle],
                                   3*(low-clean)[0, :5, non_angle], atol=4e-7)
        mean, scale = self.store.normalizer.mean[2:4], self.store.normalizer.scale[2:4]
        original = clean[0, :5, 2:4]*scale + mean
        rotated = high[0, :5, 2:4]*scale + mean
        np.testing.assert_allclose(np.square(original).sum(1), np.square(rotated).sum(1), atol=2e-7)
        np.testing.assert_array_equal(high[0, :5, 8:], clean[0, :5, 8:])

    def test_unknown_pitcher_disables_cluster_and_personal_routes(self):
        frame = self.store.frame.iloc[[5]]
        original = self.context.transform(frame)
        missing = StressSharingContext(self.context, 'unknown_pitcher').transform(frame)
        np.testing.assert_array_equal(missing[:, :-9], original[:, :-9])
        np.testing.assert_array_equal(missing[:, -9:], [[0., 0., 0., 0., 0., 0., 1., -1., -1.]])
        model = SharingPredictor('G4-partial', FixedModel(0), self.context.clusters,
            cluster_models={0: FixedModel(1)}, personal_models={10: FixedModel(2)})
        tokens, valid = self.store.gather([5])
        unknown = model.logits((tokens, valid, missing))
        global_only = SharingPredictor('G0-global', model.global_model, self.context.clusters)
        np.testing.assert_array_equal(unknown, global_only.logits((tokens, valid, missing)))
        self.assertFalse(np.array_equal(unknown, model.logits((tokens, valid, original))))
        np.testing.assert_array_equal(self.context.transform(frame), original)

    def test_missing_batter_reencodes_all_rates_evidence_memberships(self):
        frame = self.store.frame.iloc[[5, 11]].copy()
        original = frame.copy(deep=True)
        transformed = StressSharingContext(self.context, 'unknown_batter').transform(frame)
        encoder = self.context.base.archetypes
        expected = (INITIAL_RATES - encoder.mean) / encoder.scale
        np.testing.assert_allclose(transformed[:, 11:17], np.tile(expected, (2, 1)), rtol=1e-6)
        np.testing.assert_array_equal(transformed[:, 17:23], np.zeros((2, 6)))
        np.testing.assert_allclose(transformed[:, 23:28], .2)
        ordinary = self.context.transform(frame)
        np.testing.assert_array_equal(transformed[:, :11], ordinary[:, :11])
        np.testing.assert_array_equal(transformed[:, 28:], ordinary[:, 28:])
        pd.testing.assert_frame_equal(frame, original)

    def test_adapter_clean_equivalence_400_draws_and_no_recalibration(self):
        delivery = JointDelivery().fit(self.store.frame, self.store.normalizer, draws=400)
        model = FixedModel()
        expected = predict_streamed(model, delivery, self.store, self.context, [4, 5])
        actual = predict_stress_streamed(model, delivery, self.store, self.context, [4, 5], 'clean', chunk_size=1)
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
        stressed = predict_stress_streamed(model, delivery, self.store, self.context, [4, 5], 'unknown_pitcher')
        np.testing.assert_array_equal(expected[2], stressed[2])
        self.assertEqual(model.delivery_temperature, 1.3)
        self.assertEqual(protocol()['draws'], 400)
        delivery.draws = 20
        with self.assertRaises(ValueError):
            predict_stress_streamed(model, delivery, self.store, self.context, [4], 'H0')

    def test_invalid_contracts_rejected(self):
        with self.assertRaises(ValueError):
            StressHistoryStore(self.store, 'unregistered')
        with self.assertRaises(ValueError):
            StressHistoryStore(MatrixHistoryStore.from_frame(self.store.frame, history_length=2), 'H0')
        with self.assertRaises(ValueError):
            StressSharingContext(self.context.base, 'clean')


if __name__ == '__main__':
    unittest.main()
