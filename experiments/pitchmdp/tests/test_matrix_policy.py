"""Synthetic adapter equivalence, leakage and denominator tests; no real fits."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.sequence_data import PHYSICAL_COLUMNS, PhysicalNormalizer
from pitchmdp.matrix_policy import (PolicyInputs, SupportedBC, FrozenGEnsemble,
    safe_rows, state_from_row, select_pa_requests, context_key, fit_bc, game_policy_comparisons)
from pitchmdp.rollout_policy import PAState, PastPitch, BCRecord, CategoricalBC
from run_ml_policy import config_check


def frame():
    rows = []
    for i in range(3):
        rows.append(dict(game_pk=1, at_bat_number=1, pitch_number=i+1,
            game_date='2025-04-20', pitcher=9, batter=2, stand='L', p_throws='R',
            balls=i, strikes=0, inning=1, inning_topbot='Top', outs_when_up=0,
            bases=0, home_score=0, away_score=0, pitch_type=('FF', 'SL', 'FF')[i],
            description='ball', events=None, supported_pa=True, split='train',
            **{k: float(j+i) for j, k in enumerate(PHYSICAL_COLUMNS)},
            **{k: .5 for k in STYLE_COLUMNS}, **{k: .3 for k in RELIABILITY_COLUMNS}))
    return pd.DataFrame(rows)


class Context:
    def transform(self, rows):
        # First count channels match frozen SequenceContext layout; routing at end.
        out = np.zeros((len(rows), 20), dtype=np.float32)
        out[:, 0], out[:, 1] = rows.balls/3, rows.strikes/2
        out[:, 3] = rows.inning
        out[:, -2], out[:, -1] = 0, rows.pitcher
        return out


class Model:
    cell = 'G0-global'
    def __init__(self): self.seen = []
    def logits(self, arrays):
        self.seen.append(tuple(a.copy() for a in arrays))
        z = np.zeros((len(arrays[0]), 10))
        z[:, 0] = arrays[0][:, -1, 0]
        z[:, 1] = arrays[2][:, 0]
        return z


class Baseline:
    def __init__(self): self.seen = []
    def predict(self, rows):
        self.seen.append(rows.copy())
        z = np.zeros((len(rows), 10)); z[:, 1] = rows.balls
        return softmax(z, axis=-1)


def inputs_for(data):
    model = CategoricalBC().fit([BCRecord(PAState(0, 0, '9', 'L'), 'FF', 'train'),
                                BCRecord(PAState(0, 0, '9', 'L'), 'SL', 'train')])
    delivery = SimpleNamespace(draws=400, TIERS=[('pitch_type', 'p_throws', 'stand')],
        pools={(0, ('FF', 'R', 'L')): np.ones((400, 8))}, fallback=np.ones((400, 8)))
    return PolicyInputs(data, Context(), delivery, ['FF', 'SL'], 'synthetic-hash', model)


class AdapterTests(unittest.TestCase):
    def test_generated_arrays_equal_frozen_matrix_gather_for_observed_past(self):
        data = frame(); normalizer = PhysicalNormalizer().fit(data)
        store = MatrixHistoryStore.from_frame(data, normalizer, type_vocabulary=['FF', 'SL'])
        inputs = inputs_for(data)
        state = state_from_row(store, 2)
        physics = np.arange(8, dtype=float)[None]
        expected, valid = store.gather([2], current=physics, candidate_pitch_types=['SL'])
        actual, actual_valid, context = inputs.arrays([state], ['SL'], physics)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_valid, valid)
        self.assertEqual(context[0, 0], np.float32(2/3))
        self.assertTrue((actual[0, -1, -11:] == 0).all())
        # State advancement replaces counts AND generated history, without using
        # another observed dataframe row or its current physical measurements.
        generated = PAState(3, 1, '9', 'L', state.history+(PastPitch('FF', tuple(physics[0]), 'strike', 2, 0),), state.context_key)
        updated, mask, context = inputs.arrays([generated], ['FF'], physics+100)
        self.assertEqual(context[0, 0], 1)
        self.assertEqual(context[0, 1], .5)
        np.testing.assert_array_equal(updated[0, -2, :8], physics[0])
        self.assertEqual(updated[0, -2, -10], 1)  # strike channel
        self.assertEqual(mask.sum(), 4)

    def test_current_realization_and_outcome_not_in_safe_context(self):
        data = frame(); original = safe_rows(data)
        data.loc[:, list(PHYSICAL_COLUMNS)] = 99999
        data.loc[:, 'pitch_type'] = 'POISON'
        data.loc[:, 'events'] = 'home_run'
        data.loc[:, 'description'] = 'hit_into_play'
        pd.testing.assert_frame_equal(original, safe_rows(data))
        self.assertFalse({'pitch_type', 'plate_x', 'events', 'supported_pa'} & set(original.columns))

    def test_common_support_intersection_rejects_global_alltype_fallback(self):
        data = frame(); inputs = inputs_for(data)
        state = PAState(0, 0, '9', 'L', (), context_key(data.iloc[0]))
        model = SupportedBC(inputs)
        np.testing.assert_array_equal(model.support(state), [True, False])
        np.testing.assert_array_equal(model.probabilities(state), [1, 0])
        np.testing.assert_array_equal(model.probabilities(state, frequency=True), [1, 0])
        with self.assertRaisesRegex(ValueError, 'action-specific'):
            inputs.pool(state, 'SL')
        self.assertEqual(inputs.pool(state, 'FF').values.shape, (400, 8))
        for b in range(4):
            for s in range(3):
                np.testing.assert_array_equal(model.support(PAState(b, s, '9', 'L', (), state.context_key)), [True, False])

    def test_conditional_ensemble_uses_frozen_baseline_temperature_and_blend(self):
        data = frame(); inputs = inputs_for(data)
        s = PAState(2, 1, '9', 'L', (), context_key(data.iloc[0]))
        models, baseline = [Model() for _ in range(3)], Baseline()
        ensemble = FrozenGEnsemble(inputs, models, [1, 2, 3], baseline, 2., .7)
        physics = np.full((1, 8), 2.)
        p = ensemble([s], ['FF'], physics)
        raw = np.zeros((1, 10)); raw[:, 0] = 2; raw[:, 1] = 2/3
        neural = np.mean([softmax(raw/t, axis=1) for t in (1, 2, 3)], axis=0)
        f = np.zeros((1, 10)); f[:, 1] = 1  # balls2 / baseline_temperature2
        np.testing.assert_allclose(p, .7*neural+.3*softmax(f, axis=1), atol=1e-8)
        self.assertEqual(baseline.seen[0].iloc[0].pitch_type, 'FF')
        self.assertEqual(baseline.seen[0].iloc[0].balls, 2)
        self.assertNotIn('plate_x', baseline.seen[0])
        self.assertEqual(ensemble.actual_neural_network_rows, 3)

    def test_pa_selection_preserves_denominator_and_ignores_outcome_support(self):
        data = pd.concat([frame().assign(game_pk=i, pitcher=9 if i%2 else 10,
            game_date='2025-07-05', split='dev') for i in range(1, 9)], ignore_index=True)
        panel = {'pitcher_ids': [9, 10], 'train_players': [dict(pitcher=9, train_role='starter'),
                                                         dict(pitcher=10, train_role='relief')]}
        first = select_pa_requests(data, panel, 'dev', 4, 99)
        data['supported_pa'] = False; data['events'] = 'home_run'; data['plate_x'] = np.nan
        second = select_pa_requests(data, panel, 'dev', 4, 99)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(first.selected.sum(), 4)
        self.assertEqual(first.loc[first.selected, 'train_role'].tolist(), ['starter', 'relief']*2)
        with self.assertRaises(ValueError):
            select_pa_requests(data.assign(game_date='2026-07-05'), panel, 'dev', 4, 99)

    def test_game_bootstrap_weights_pa_starts_and_keeps_three_test_family(self):
        base = np.full((4, 3), .5)
        changed = base + np.array([.1, .1, .1, -.1])[:, None]
        rows = game_policy_comparisons({'P0': base, 'P1': changed, 'P2': changed, 'P3': changed},
                                       [1, 1, 1, 2], draws=100, seed=7)
        self.assertAlmostEqual(rows[0]['mean_delta_we'], .05)
        self.assertEqual(rows[0]['games'], 2)
        self.assertEqual(rows[0]['pa_starts'], 4)
        self.assertEqual(len(rows), 3)
        self.assertIsNone(rows[1]['holm_p'])
        self.assertFalse(rows[1]['model_internal_P_screen'])
        self.assertIsNone(rows[0]['causal_P'])

    def test_p_screen_requires_positive_worst_case_truncation_bound(self):
        values = {'P0': np.full((60, 3), .5), 'P1': np.full((60, 3), .51),
                  'P2': np.full((60, 3), .51), 'P3': np.full((60, 3), .51)}
        flags = {k: np.zeros_like(v, dtype=bool) for k, v in values.items()}
        flags['P1'][:4] = True
        rows = game_policy_comparisons(values, np.arange(60), truncated=flags, draws=100, seed=7)
        self.assertTrue(rows[0]['model_internal_P_screen'])
        self.assertFalse(rows[0]['untruncated_pa_improvement_confirmed'])
        self.assertEqual(rows[0]['model_internal_screen'], 'tail_assumption_dependent')
        self.assertLess(rows[0]['worst_case_mean_delta_lower'], 0)
        flags['P1'][:] = False
        rows = game_policy_comparisons(values, np.arange(60), truncated=flags, draws=100, seed=7)
        self.assertTrue(rows[0]['untruncated_pa_improvement_confirmed'])

    def test_bc_fit_date_rejection_and_no_cross_pa_history(self):
        data = frame(); store = MatrixHistoryStore.from_frame(data, type_vocabulary=['FF', 'SL'])
        bc = fit_bc(store, data)
        self.assertEqual(sum(bc.league.values()), 3)
        with self.assertRaises(ValueError): fit_bc(store, data.assign(game_date='2025-06-02'))
        store.indices[2, -1] = 2
        with self.assertRaisesRegex(ValueError, 'strictly previous'):
            state_from_row(store, 2)


if __name__ == '__main__': unittest.main()
