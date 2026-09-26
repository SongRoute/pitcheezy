"""Synthetic Bellman/mask/PA-boundary tests, no data or actual experiment fits."""
import tempfile
import shutil
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from pitchmdp.game import GameState, EmpiricalAdvancement, terminal_values
from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_offline_rl import (FIXED, METHODS, audit_trajectories, legal_terminal_next,
    terminal_rewards, LazyFeatures, OfflinePolicy, expectile_loss, masked_log_probs,
    state_features, rl_comparisons)
from pitchmdp.matrix_policy import state_from_row
from pitchmdp.rollout_policy import rollouts
from test_matrix_policy import frame, Context, inputs_for
from test_rollout_policy import simulator, state
from run_ml_offline_rl import verify_family, seal, family_seconds, train_job
from unittest.mock import patch
from pitchmdp.data import hash_file
from pitchmdp.matrix_data import canonical_hash


def trajectory_frame():
    data = frame().iloc[:2].copy()
    data['balls'] = [0, 1]
    data['pitch_type'] = ['FF', 'SL']
    data['description'] = ['ball', 'hit_into_play']
    data['events'] = [None, 'field_out']
    data['terminal_event'] = [None, 'field_out']
    data['is_pa_terminal'] = [False, True]
    data['next_outs'] = [np.nan, 1]
    data['next_bases'] = [np.nan, 0]
    data['next_home_score'] = [np.nan, 0]
    data['next_away_score'] = [np.nan, 0]
    data['next_inning'] = [np.nan, 1]
    data['next_half'] = [None, 'Top']
    return data.reset_index(drop=True)


def audit(data):
    return audit_trajectories(data, data.index, ('FF', 'SL'), lambda r: np.array([1, 1], bool), {9: 'starter'})


class FakeWE:
    def predict_home(self, value):
        if isinstance(value, GameState): return .5+.05*value.outs
        return .5+.05*value.outs_when_up.to_numpy()
    def predict_defense(self, value, home):
        p = self.predict_home(value)
        return p if home else 1-p


class OfflineTests(unittest.TestCase):
    def test_complete_pa_transition_and_strict_terminal_missing_rejection(self):
        data = trajectory_frame(); records, report = audit(data)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['next_index'], 1)
        self.assertEqual(records[1]['next_index'], -1)
        self.assertEqual(records[1]['terminal_event'], 'out')
        self.assertTrue(report.retained.iloc[0])
        broken = data.copy(); broken.loc[1, 'next_outs'] = np.nan
        records, report = audit(broken)
        self.assertEqual(records, [])
        self.assertIn('missing_terminal_next_state', report.reasons.iloc[0])
        # Even game-ending-looking scores never create an exemption.
        broken['inning'] = 9; broken['inning_topbot'] = 'Bot'; broken['home_score'] = 9
        records, _ = audit(broken)
        self.assertEqual(records, [])

    def test_audit_excludes_whole_pa_on_gap_support_or_bad_next_count(self):
        for change, reason in ((lambda d: d.assign(pitch_number=[1, 3]), 'nonconsecutive'),
                               (lambda d: d.assign(balls=[0, 2]), 'count_or_event')):
            records, report = audit(change(trajectory_frame()))
            self.assertEqual(len(records), 0); self.assertIn(reason, report.reasons.iloc[0])
        data = trajectory_frame()
        records, report = audit_trajectories(data, [0], ('FF', 'SL'), lambda r: np.array([1, 1], bool), {})
        self.assertEqual(records, []); self.assertIn('not_complete_d100', report.reasons.iloc[0])
        records, report = audit_trajectories(data, [0, 1], ('FF', 'SL'), lambda r: np.array([1, 0], bool), {})
        self.assertEqual(records, []); self.assertIn('outside_common_support', report.reasons.iloc[0])
        with self.assertRaises(ValueError): audit(data.assign(game_date='2025-06-01'))

    def test_terminal_rewards_match_frozen_expected_advancement_and_initial_defender(self):
        data = trajectory_frame(); records, _ = audit(data)
        advance, we = EmpiricalAdvancement(), FakeWE()
        reward = terminal_rewards(records, data, we, advance)
        expected = terminal_values(GameState.from_row(data.iloc[0]), we, advance)['out']
        np.testing.assert_allclose(reward, [0, expected])
        data['inning_topbot'] = 'Bot'; data.loc[1, 'next_half'] = 'Bot'
        records, _ = audit(data)
        np.testing.assert_allclose(terminal_rewards(records, data, we, advance), [0, 1-expected])
        bad = data.iloc[1].copy(); bad['next_home_score'] = -1
        self.assertIsNotNone(legal_terminal_next(bad, 'out'))

    def test_lazy_features_match_online_generated_encoder_and_never_current_token(self):
        data = trajectory_frame(); records, _ = audit(data)
        store = MatrixHistoryStore.from_frame(data, type_vocabulary=['FF', 'SL'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'features'
            LazyFeatures.write(path, store, Context(), records, np.array([0, .55], np.float32))
            bank = LazyFeatures(path)
            states = [state_from_row(store, i) for i in range(2)]
            online = state_features(inputs_for(data), states)
            np.testing.assert_array_equal(bank.features([0, 1]), online)
            batch = bank.batch([0, 1])
            np.testing.assert_array_equal(batch['next_x'][0], online[1])
            np.testing.assert_array_equal(batch['next_x'][1], 0)
            self.assertFalse(batch['next_mask'][1].any())
            # Poison the CURRENT observed token only. Its own state is invariant;
            # the next state legitimately sees the newly observed previous token.
            current = np.load(path / 'tokens.npy'); current[0] = 999
            np.save(path / 'tokens.npy', current)
            changed = LazyFeatures(path)
            np.testing.assert_array_equal(changed.features([0]), online[:1])
            self.assertFalse(np.array_equal(changed.features([1]), online[1:]))

    def test_expectile_weights_and_masked_actor_loss(self):
        error = torch.tensor([-2., 2.])
        self.assertAlmostEqual(float(expectile_loss(error, .7)), 2.)
        logits = torch.tensor([[0., 999., 0.]], requires_grad=True)
        mask = torch.tensor([[True, False, True]])
        logp = masked_log_probs(logits, mask)
        self.assertEqual(float(logp.exp()[0, 1].detach()), 0)
        loss = -logp[0, 0]; loss.backward()
        self.assertEqual(float(logits.grad[0, 1]), 0)
        with self.assertRaises(ValueError): masked_log_probs(logits, torch.zeros_like(mask))

    @staticmethod
    def batch():
        return dict(x=np.zeros((4, 7), np.float32), next_x=np.zeros((4, 7), np.float32),
            action=np.array([0, 1, 0, 1]), reward=np.array([.2, .4, .6, .8], np.float32),
            done=np.ones(4, dtype=bool), mask=np.array([[1, 1, 0]]*4, dtype=bool),
            next_mask=np.zeros((4, 3), dtype=bool))

    def test_all_fixed_losses_accept_terminal_empty_next_support_and_serialize(self):
        for method in METHODS:
            model = OfflinePolicy(method, 7, 3, 0, device='cpu')
            metrics = model.train_step(self.batch())
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
            p = model.probabilities(self.batch()['x'], self.batch()['mask'])
            np.testing.assert_array_equal(p[:, 2], 0)
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
            if method != 'NNBC':
                self.assertAlmostEqual(metrics['target_min'], .2, places=6)
                self.assertAlmostEqual(metrics['target_max'], .8, places=6)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.pt'; model.save(path)
                restored = OfflinePolicy.load(path, device='cpu')
                np.testing.assert_array_equal(restored.probabilities(self.batch()['x'], self.batch()['mask']), p)
                self.assertEqual(restored.updates, 1)

    def test_nonfinite_aborts_and_nonterminal_missing_support_aborts(self):
        data = self.batch(); data['reward'][0] = np.nan
        with self.assertRaises(FloatingPointError): OfflinePolicy('IQL', 7, 3, 0, 'cpu').train_step(data)
        data = self.batch(); data['done'][0] = False
        with self.assertRaises(ValueError): OfflinePolicy('CQL', 7, 3, 0, 'cpu').train_step(data)

    def test_joint_four_gate_requires_matched_bc_and_minimum_samples(self):
        values = {'planner': np.full((60, 3), .5), 'NNBC': np.full((60, 3), .52),
                  'IQL': np.full((60, 3), .51), 'CQL': np.full((60, 3), .53)}
        flags = {k: np.zeros_like(v, bool) for k, v in values.items()}
        results = rl_comparisons(values, flags, np.repeat(np.arange(30), 2), draws=200)
        self.assertFalse(results[0]['rl_gain_over_both_controls'])
        self.assertTrue(results[1]['rl_gain_over_both_controls'])
        self.assertTrue(all(r['family_size'] == 4 for r in results))
        small = rl_comparisons({k: v[:10] for k, v in values.items()},
            {k: v[:10] for k, v in flags.items()}, np.arange(10), draws=100)
        self.assertTrue(all(r['model_internal_screen'] == 'descriptive_only' for r in small))

    def test_full_family_rejects_copied_seed_checkpoint_even_with_resealed_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'preparation.json').write_text('{}')
            prep, execution = {'feature_width': 7, 'actions': ['A', 'B', 'C']}, {'updates': 3}
            for method in METHODS:
                for seed in (0, 1, 2):
                    path = root/'members'/method/f'seed{seed}'; path.mkdir(parents=True)
                    expected = dict(preparation_sha256=hash_file(root/'preparation.json'),
                        execution_sha256=canonical_hash(execution), method=method, seed=seed)
                    (path/'started.json').write_text(json.dumps(expected))
                    (path/'state.json').write_text(json.dumps({**expected, 'completed': True}))
                    model = OfflinePolicy(method, 7, 3, seed, 'cpu'); model.updates = 3; model.save(path/'model.pt')
                    seal(path)
            dependencies = verify_family(root, prep, execution)
            self.assertEqual(sum(k.endswith('model.pt') for k in dependencies), 9)
            first, second = root/'members/NNBC/seed0', root/'members/NNBC/seed1'
            shutil.copyfile(first/'model.pt', second/'model.pt')
            (second/'manifest.json').unlink(); seal(second)
            with self.assertRaisesRegex(ValueError, 'checkpoint method/seed'):
                verify_family(root, prep, execution)

    def test_failure_time_survives_checkpoint_error_and_hard_kills_block_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); member = root/'members/IQL/seed0'; member.mkdir(parents=True)
            (member/'started.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Unaccounted interrupted'):
                family_seconds(root)
            fake = SimpleNamespace(updates=0, train_step=lambda b: (_ for _ in ()).throw(RuntimeError('fit failed')),
                save=lambda p: (_ for _ in ()).throw(OSError('disk failed')))
            bank = SimpleNamespace(width=7, support=np.ones((2, 3)), size=2, batch=lambda i: {})
            with patch('run_ml_offline_rl.OfflinePolicy', return_value=fake):
                with self.assertRaisesRegex(RuntimeError, 'fit failed'):
                    train_job(bank, 'IQL', 0, 1, member, 100)
            runtime = json.loads((member/'runtime.json').read_text())
            self.assertFalse(runtime['completed'])
            self.assertEqual(runtime['checkpoint_error'], 'disk failed')
            self.assertEqual(family_seconds(root), runtime['seconds'])

    def test_batched_common_evaluator_matches_scalar_policy_with_same_crn(self):
        def prediction(states, actions, physical):
            p = np.zeros((len(states), 10)); p[:, 3] = .5; p[:, 7] = .5
            return p
        calls = []
        class Policy:
            def batch_probabilities(self, states, depth):
                calls.append(len(states)); return ('A', 'B'), np.tile([.2, .8], (len(states), 1))
        u = np.random.default_rng(5).random((8, 4, 3))
        scalar = rollouts(simulator(prediction), [state()]*8, lambda s, d: (('A', 'B'), [.2, .8]), u, lambda s: .5)
        batched = rollouts(simulator(prediction), [state()]*8, Policy(), u, lambda s: .5)
        np.testing.assert_array_equal(scalar.values, batched.values)
        self.assertEqual(calls, [8])


if __name__ == '__main__': unittest.main()
