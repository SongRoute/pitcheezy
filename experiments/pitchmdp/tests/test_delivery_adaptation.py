"""Independent CPU tests for strictly prior delivery adaptation; no inference."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT/'scripts'))
from run_delivery_adaptation import (CHANNELS, RAW_CHANNELS, adaptive_shift,
                                     build_prior_physics, complete_physics, fit_reference,
                                     reference_for_rows, shifted_delivery_logits)


def example():
    frame = pd.DataFrame({
        'game_pk': [1]*7+[2, 2], 'at_bat_number': [1, 1, 1, 2, 2, 3, 3, 1, 1],
        'pitch_number': [1, 2, 3, 1, 2, 1, 2, 1, 2],
        'pitcher': [10, 10, 10, 10, 20, 10, 10, 10, 10],
        'pitch_type': ['FF', 'SL', 'FF', 'FF', 'FF', 'FF', 'FF', 'FF', 'FF'],
        'game_date': pd.to_datetime(['2024-06-01']*7+['2024-06-02']*2),
        'split': ['train']*9,
    })
    for column in RAW_CHANNELS:
        frame[column] = np.arange(1, 10, dtype=float)
    physical = np.arange(1, 10, dtype=np.float32)[:, None]*np.ones((1, 8), dtype=np.float32)
    return frame, physical


class AsOfDeliveryContracts(unittest.TestCase):
    def test_history_crosses_pa_but_resets_game_pitcher_and_pitch_type(self):
        frame, physical = example()
        count, means = build_prior_physics(frame, physical)
        np.testing.assert_array_equal(count, [0, 0, 1, 2, 0, 3, 4, 0, 1])
        np.testing.assert_allclose(means[:, 0], [0, 0, 1, 2, 0, 8/3, 3.5, 0, 8])
        shift = adaptive_shift(count, means, np.ones_like(means), 10.)
        np.testing.assert_array_equal(shift[count == 0], 0.)
        self.assertNotEqual(float(shift[3, 0]), 0.)  # New PA retains earlier same-game FFs.

    def test_current_future_physics_and_outcomes_cannot_change_current_shift(self):
        frame, physical = example()
        before = build_prior_physics(frame, physical)
        changed_frame, changed_physical = frame.copy(), physical.copy()
        changed_frame.loc[3:, list(RAW_CHANNELS)] = 1e15
        changed_physical[3:] = 1e15
        changed_frame['events'], changed_frame['description'] = 'home_run', 'hit_into_play'
        after = build_prior_physics(changed_frame, changed_physical)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left[:4], right[:4])
        # The changed delivery becomes observable for the later matching pitch.
        self.assertGreater(float(after[1][5, 0]), 1e10)

    def test_shuffled_rows_return_same_asof_features_aligned_to_input(self):
        frame, physical = example()
        count, means = build_prior_physics(frame, physical)
        order = np.array([5, 0, 8, 4, 3, 7, 2, 1, 6])
        shuffled = build_prior_physics(frame.iloc[order], physical[order])
        np.testing.assert_array_equal(shuffled[0], count[order])
        np.testing.assert_array_equal(shuffled[1], means[order])

    def test_missing_physics_is_not_a_median_imputed_observation(self):
        frame, physical = example()
        frame.loc[2, 'effective_speed'] = np.nan
        physical[2] = 999.  # Imputed finite normalizer output must not count as evidence.
        valid = complete_physics(frame)
        self.assertFalse(valid[2])
        count, means = build_prior_physics(frame, physical)
        self.assertEqual(count[3], 1)
        np.testing.assert_array_equal(means[3], [1.]*4)

    def test_shrinkage_and_infinite_k_static_are_exact(self):
        counts = np.array([0, 10, 30])
        means, reference = np.full((3, 4), 5.), np.ones((3, 4))
        shifted = adaptive_shift(counts, means, reference, 10.)
        np.testing.assert_array_equal(shifted, [[0.]*4, [2.]*4, [3.]*4])
        np.testing.assert_array_equal(adaptive_shift(counts, means, reference, np.inf), 0.)

    def test_reference_is_train_only_frozen_and_has_auditable_fallbacks(self):
        train, physical = example()
        reference = fit_reference(train, physical, minimum=2)
        stored_mean = reference['global'].copy()
        queried = train.iloc[[0, 4, 1]].copy()
        queried.loc[queried.index[1], 'pitcher'] = 9999
        queried.loc[queried.index[2], 'pitch_type'] = 'UNKNOWN'
        means, origins = reference_for_rows(queried, reference)
        np.testing.assert_array_equal(origins, [2, 1, 0])
        np.testing.assert_array_equal(means[2], stored_mean.astype(np.float32))
        queried.loc[:, list(RAW_CHANNELS)] = 1e12
        queried['events'] = 'home_run'
        np.testing.assert_array_equal(reference_for_rows(queried, reference)[0], means)
        np.testing.assert_array_equal(reference['global'], stored_mean)
        train.loc[0, 'split'] = 'dev'
        with self.assertRaises(ValueError):
            fit_reference(train, physical, minimum=2)
        train['split'] = 'train'
        train.loc[0, 'game_date'] = pd.Timestamp('2025-07-01')
        with self.assertRaises(ValueError):
            fit_reference(train, physical, minimum=2)

    def test_duplicate_keys_and_forbidden_year_are_rejected(self):
        frame, physical = example()
        with self.assertRaises(ValueError):
            build_prior_physics(pd.concat([frame, frame.iloc[[0]]]), np.concatenate([physical, physical[[0]]]))
        frame.loc[0, 'game_date'] = pd.Timestamp('2026-01-01')
        with self.assertRaises(ValueError):
            build_prior_physics(frame, physical)


class TranslationBoundaryContracts(unittest.TestCase):
    def test_translation_changes_only_four_candidates_channels_not_history_or_pool(self):
        frame, physical = example()
        pool = np.arange(16, dtype=np.float32).reshape(2, 8)
        original_pool = pool.copy()
        shifts = np.arange(len(frame)*4, dtype=np.float32).reshape(-1, 4)
        rows = np.array([1, 5])
        recorded = []
        class Delivery:
            draws = 2
            def sample(self, selected):
                return np.broadcast_to(pool, (len(selected), *pool.shape)), np.zeros(len(selected))
        class Store:
            def __init__(self):
                self.frame = frame
            def gather(self, selected, current):
                tokens = np.zeros((len(selected), 6, 8), dtype=np.float32)
                tokens[:, :-1] = 123.
                tokens[:, -1] = current
                return tokens, np.ones((len(selected), 6), dtype=bool)
        class Context:
            def transform(self, selected):
                return np.zeros((len(selected), 28), dtype=np.float32)
        class Model:
            def logits(self, arrays):
                recorded.append(arrays[0].copy())
                return np.zeros((len(arrays[0]), 10), dtype=np.float32)
        logits = shifted_delivery_logits(Model(), Store(), Context(), Delivery(), rows, shifts, chunk_size=1)
        self.assertEqual(logits.shape, (2, 2, 10))
        tokens = np.concatenate(recorded)
        np.testing.assert_array_equal(tokens[:, :-1], 123.)
        expected = np.tile(pool, (2, 1))
        expected[:, CHANNELS] += np.repeat(shifts[rows], 2, axis=0)
        np.testing.assert_array_equal(tokens[:, -1], expected)
        np.testing.assert_array_equal(tokens[:, -1, [2, 3, 6, 7]], np.tile(pool, (2, 1))[:, [2, 3, 6, 7]])
        np.testing.assert_array_equal(pool, original_pool)


if __name__ == '__main__':
    unittest.main()
