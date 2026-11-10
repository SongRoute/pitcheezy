"""Explicit device option for the fixed-epoch T trainer. Generated random arrays only; no real data read."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.ws_fullseason import train
from experiments.ws_fullseason.data import key_digest

RECIPE = Path(__file__).resolve().parents[1] / 'configs/EXP-WS-FULLSEASON-T-001.json'
HAS_MPS = torch.backends.mps.is_available()


def synthetic_package(games=12, pitches=40, dim=16, oof_games=3, seed=0):
    """Random in-memory package with the trainer's array contract. Roles are split by game, so a
    same-pitcher-same-game context never crosses FIT/OOF. oof_games=0 gives a FIT-only package."""
    rng = np.random.default_rng(seed); n = games * pitches
    game = np.repeat(np.arange(700000, 700000 + games), pitches).astype(np.int32)
    i = np.tile(np.arange(pitches), games)
    at_bat = (i // 4 + 1).astype(np.int16); pitch_no = (i % 4 + 1).astype(np.int16)
    y10 = rng.integers(0, 10, n).astype(np.int64)
    whiff = (y10 == 1).astype(np.int8)
    bip = np.isin(y10, (3, 4, 5, 6, 7, 9)); gb = (bip & (rng.random(n) < .45)).astype(np.int8)
    y7 = rng.integers(0, 7, n).astype(np.int64)
    evaluation = np.isin(game, np.arange(700000, 700000 + games)[games - oof_games:]) if oof_games else np.zeros(n, bool)
    zeros = np.zeros(n, bool)
    c = {'keys': np.column_stack([game, at_bat, pitch_no]).astype(np.int64), 'y10': y10, 'y7': y7,
         'games': game.astype(np.int64), 'fit': ~evaluation, 'tuning': zeros.copy(), 'calibration': zeros.copy(),
         'evaluation': evaluation, 'narrow_whiff': whiff, 'ground_ball_per_pitch': gb}
    s = {'game_pk': game, 'at_bat_number': at_bat, 'pitch_number': pitch_no,
         'pitcher': (game % 7 + 100).astype(np.int32), 'batter': rng.integers(200, 260, n).astype(np.int32),
         'pitch_type': rng.integers(0, 19, n).astype(np.int16), 'outcome10': y10.astype(np.int16),
         'balls': rng.integers(0, 4, n).astype(np.int16), 'strikes': rng.integers(0, 3, n).astype(np.int16),
         'stand': rng.integers(0, 2, n).astype(np.int16)}
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x, c, s


def small_recipe(dim=16, epochs=2, batch=64, model=None):
    r = json.loads(RECIPE.read_text())
    r['base']['inputs']['feature_dimension'] = dim
    r['base']['model'].update(model or dict(trunk_hidden=[16, 8], token_dim=8, encoder_layers=1, encoder_heads=2, encoder_ff=16))
    r['base']['training'].update(batch_size=batch, eval_batch_size=2 * batch)
    r['training_rule']['epochs'] = epochs
    return r


def oof_probabilities(recipe, package, device, synthetic=True):
    with tempfile.TemporaryDirectory() as tmp:
        r = train.run(recipe, 0, package, tmp, synthetic=synthetic, device=device)
        with np.load(Path(tmp) / 'oof-predictions.npz', allow_pickle=False) as z:
            return r, z['keys'].copy(), z['probabilities'].copy()


class DeviceGuards(unittest.TestCase):
    def setUp(self):
        self.recipe = small_recipe(); self.package = synthetic_package()

    def test_unknown_device_refused(self):
        for name in ['cpu', 'gpu', 'mps:0', 'CUDA', '', None]:
            with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'unknown device'):
                train.run(self.recipe, 0, self.package, tmp, synthetic=True, device=name)

    def test_fallback_env_refused_before_any_work(self):
        for value in ['1', 'true', 'false', '0 ', 'False']:
            with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': value}):
                with self.assertRaisesRegex(ValueError, 'PYTORCH_ENABLE_MPS_FALLBACK'):
                    train.run(self.recipe, 0, self.package, tmp, synthetic=True, device='mps')
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_fallback_env_off_values_pass_the_guard(self):
        for value in ['', '0']:
            with patch.dict(os.environ, {'PYTORCH_ENABLE_MPS_FALLBACK': value}), \
                    patch.object(torch.backends.mps, 'is_available', return_value=True):
                self.assertEqual(train.resolve_device('mps', True).type, 'mps')

    def test_missing_mps_refused(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.backends.mps, 'is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'MPS required'):
                train.run(self.recipe, 0, self.package, tmp, synthetic=True, device='mps')

    def test_default_is_still_cuda_required_and_schema_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CUDA required'):
                train.run(self.recipe, 0, self.package, tmp)
            with self.assertRaisesRegex(ValueError, 'CUDA required'):
                train.run(self.recipe, 0, self.package, tmp, device='cuda')
        r, _, _ = oof_probabilities(self.recipe, self.package, 'cuda', synthetic=True)
        self.assertEqual(r['backend'], {'deterministic': True, 'warn_only': True, 'cudnn_benchmark': False, 'tf32': False, 'dtype': 'fp32'})
        self.assertEqual(set(r['versions']), {'torch', 'cuda', 'numpy'})


@unittest.skipUnless(HAS_MPS, 'Apple MPS not available')
class MpsRun(unittest.TestCase):
    def test_oof_and_final_refit_complete_on_mps(self):
        recipe = small_recipe(); x, c, s = synthetic_package()
        r, keys, q = oof_probabilities(recipe, (x, c, s), 'mps')
        self.assertEqual(r['status'], 'COMPLETED'); self.assertEqual(r['execution_stage'], 'OOF')
        self.assertEqual(r['backend']['device'], 'mps'); self.assertFalse(r['backend']['mps_fallback_env'])
        self.assertIs(r['backend']['deterministic'], False); self.assertIs(r['backend']['deterministic_guaranteed'], False)
        self.assertIn('macos', r['versions']['mps']); self.assertGreater(r['versions']['mps']['driver_allocated_bytes'], 0)
        np.testing.assert_array_equal(keys, c['keys'][c['evaluation']])
        self.assertEqual(q.shape, (int(c['evaluation'].sum()), 7))
        self.assertTrue(np.isfinite(q).all()); self.assertFalse((q < 0).any())
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-7)
        self.assertEqual(len(r['consumption']), recipe['training_rule']['epochs'])
        for epoch, ev in enumerate(r['consumption'], 1):
            self.assertEqual((ev['epoch'], ev['rows'], ev['every_key_once']), (epoch, int(c['fit'].sum()), True))
            self.assertEqual(ev['ordered_keys_sha256'], key_digest(c['keys'][c['fit']]))
        steps = -(-int(c['fit'].sum()) // recipe['base']['training']['batch_size'])
        self.assertEqual(r['optimization_steps'], steps * recipe['training_rule']['epochs'])
        final = copy.deepcopy(recipe); final['execution_stage'] = 'FINAL_REFIT'
        package = synthetic_package(oof_games=0)
        with tempfile.TemporaryDirectory() as tmp:
            f = train.run(final, 0, package, tmp, synthetic=True, device='mps')
            self.assertEqual((f['execution_stage'], f['OOF_rows'], f['backend']['device']), ('FINAL_REFIT', 0, 'mps'))
            self.assertEqual(f['sample_presentations'], len(package[0]) * 2)
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, {'model-state.pt'})
            state = torch.load(Path(tmp) / 'model-state.pt', weights_only=True, map_location='cpu')
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))

    def test_cpu_vs_mps_and_mps_repeat_difference_measured(self):
        recipe = small_recipe(); package = synthetic_package()
        _, _, cpu = oof_probabilities(recipe, package, 'cuda', synthetic=True)   # synthetic default path = CPU
        _, _, m1 = oof_probabilities(recipe, package, 'mps')
        _, _, m2 = oof_probabilities(recipe, package, 'mps')
        cross = float(np.abs(cpu - m1).max()); repeat = float(np.abs(m1 - m2).max())
        print(f'\n[MPS-LANE] OOF max|p_cpu - p_mps| = {cross:.3e}; max|p_mps1 - p_mps2| = {repeat:.3e} '
              f'(rows={len(package[0])}, oof={len(cpu)}, epochs=2, batch=64)')
        # Loose bound only: dropout masks come from different device generators, so CPU and MPS differ by design.
        self.assertLess(cross, .25); self.assertLess(repeat, .25)


if __name__ == '__main__': unittest.main()
