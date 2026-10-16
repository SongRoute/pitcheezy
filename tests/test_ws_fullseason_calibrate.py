"""Generated probabilities/keys only; no file, model or real-data inputs."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
from scipy.special import softmax
from experiments.ws_fullseason import calibrate as cal
from experiments.ws_fullseason import calibrate_worker as worker


def windows(sizes=(334, 333, 333)):
    result = {}; offset = 0
    for fold, n in zip(cal.FOLDS, sizes):
        keys = np.column_stack([np.arange(offset + 1, offset + n + 1), np.ones(n, int), np.ones(n, int)])
        q = np.full((n, 7), 1 / 7)
        result[fold] = {'keys': keys, 'labels': np.arange(offset, offset + n) % 7,
                        'seeds': {s: (keys.copy(), q.copy()) for s in cal.SEEDS}}
        offset += n
    return result


class PooledTemperature(unittest.TestCase):
    def test_canonical_pool_and_equal_pitch_weighting(self):
        w = windows((2, 4, 7))
        for fold in w.values():
            fold['keys'] = fold['keys'][::-1]
            fold['labels'] = fold['labels'][::-1]
            fold['seeds'] = {s: (k[::-1], q[::-1]) for s, (k, q) in fold['seeds'].items()}
        keys, labels, q = cal.pool(w)
        np.testing.assert_array_equal(keys[:, 0], np.arange(1, 14))
        np.testing.assert_array_equal(labels, np.arange(13) % 7)
        self.assertEqual(q.shape, (13, 7))
        w = windows((1, 1, 998))
        with patch.object(cal.base, 'temperature', return_value=1.25) as fit:
            result = cal.fit(w)
        self.assertEqual(result['weighting'], 'equal_pitch')
        self.assertEqual(fit.call_args.args[0].shape, (1000, 7))
        np.testing.assert_array_equal(fit.call_args.args[1], np.arange(1000) % 7)

    def test_support_boundary_and_identity_preserves_zeros(self):
        w = windows((333, 333, 333))
        with patch.object(cal.base, 'temperature') as fit:
            c = cal.fit(w)
        fit.assert_not_called()
        self.assertEqual((c['kind'], c['reason'], c['optimizer_calls']), ('identity', 'insufficient_support', 0))
        q = np.eye(7)[[0, 3]]
        np.testing.assert_array_equal(cal.apply({s: q for s in cal.SEEDS}, c), q)
        with patch.object(cal.base, 'temperature', return_value=1.5) as fit:
            c = cal.fit(windows())
        fit.assert_called_once()
        self.assertEqual((c['kind'], c['rows'], c['optimizer_calls']), ('temperature', 1000, 1))

    def test_mean_before_temperature_and_stored_application(self):
        w = windows()
        with patch.object(cal.base, 'temperature', return_value=2.0):
            c = cal.fit(w)
        qs = {0: np.array([[.8, .1, .02, .02, .02, .02, .02]]),
              1: np.array([[.02, .02, .02, .02, .02, .1, .8]]),
              2: np.full((1, 7), 1 / 7)}
        mean = np.mean(np.stack(list(qs.values())), axis=0)
        expected = softmax(np.log(np.clip(mean, cal.CLIP, 1)) / 2, axis=1)
        wrong_order = np.mean([softmax(np.log(q) / 2, axis=1) for q in qs.values()], axis=0)
        actual = cal.apply(qs, c)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(np.max(np.abs(actual - wrong_order)), .001)

    def test_existing_optimizer_recovers_known_temperature(self):
        w = windows((1000, 1000, 800))
        true = np.arange(1, 8) / 28
        raw = softmax(np.log(true) * 2)
        labels = np.tile(np.repeat(np.arange(7), np.arange(1, 8)), 100)
        offset = 0
        for window in w.values():
            n = len(window['keys'])
            window['labels'] = labels[offset:offset + n]
            for seed in cal.SEEDS:
                window['seeds'][seed] = (window['keys'].copy(), np.tile(raw, (n, 1)))
            offset += n
        result = cal.fit(w)
        self.assertEqual(result['kind'], 'temperature')
        self.assertAlmostEqual(result['temperature'], 2, places=5)

    def test_boundary_and_optimizer_stop_use_recorded_identity(self):
        # All outcomes favor the slightly largest probability: optimum is below
        # the allowed lower bound, which the existing optimizer must reject.
        w = windows()
        for window in w.values():
            window['labels'][:] = 0
            for _, q in window['seeds'].values():
                q[:] = [.16, .14, .14, .14, .14, .14, .14]
        c = cal.fit(w)
        self.assertEqual((c['kind'], c['reason'], c['optimizer_calls']), ('identity', 'optimizer_failure', 1))
        with patch.object(cal.base, 'temperature', side_effect=cal.base.Stop('optimizer did not converge')):
            self.assertEqual(cal.fit(windows())['reason'], 'optimizer_failure')

    def test_unexpected_errors_do_not_become_identity(self):
        with patch.object(cal.base, 'temperature', side_effect=RuntimeError('programming failure')):
            with self.assertRaises(RuntimeError):
                cal.fit(windows())
        with patch.object(cal.base, 'temperature', return_value=np.nan):
            with self.assertRaises(ValueError):
                cal.fit(windows())

    def test_missing_extra_window_or_seed_stop(self):
        for kind in ('missing_window', 'extra_window', 'missing_seed', 'extra_seed'):
            w = windows((1, 1, 1))
            if kind == 'missing_window': del w['july']
            elif kind == 'extra_window': w['october'] = w['july']
            elif kind == 'missing_seed': del w['july']['seeds'][2]
            else: w['july']['seeds'][3] = w['july']['seeds'][0]
            with self.subTest(kind=kind), self.assertRaises(ValueError): cal.fit(w)

    def test_duplicate_crossfold_and_misaligned_keys_stop(self):
        w = windows((2, 2, 2)); w['july']['seeds'][1] = (w['july']['keys'][::-1], np.full((2, 7), 1 / 7))
        with self.assertRaises(ValueError): cal.fit(w)
        w = windows((2, 2, 2)); w['august'] = copy.deepcopy(w['july'])
        with self.assertRaises(ValueError): cal.fit(w)
        w = windows((2, 2, 2)); w['july']['keys'][1] = w['july']['keys'][0]
        with self.assertRaises(ValueError): cal.fit(w)

    def test_bad_keys_labels_and_probabilities_stop_before_optimizer(self):
        for value in (np.nan, -1, 1.1):
            w = windows((1, 1, 1)); w['july']['seeds'][0][1][0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError): cal.fit(w)
        for labels in (np.array([7]), np.array([-1]), np.array([1.0]), np.array([True])):
            w = windows((1, 1, 1)); w['july']['labels'] = labels
            with self.assertRaises(ValueError): cal.fit(w)
        for keys in (np.array([[0, 1, 1]]), np.array([[1., 1., 1.]]), np.array([[2**63, 1, 1]], np.uint64)):
            with self.assertRaises(ValueError): cal.checked_keys(keys)

    def test_probability_shape_and_row_sum_are_strict(self):
        for q in (np.zeros((0, 7)), np.full((1, 6), 1 / 6), np.full((1, 7), (1 + 1e-6) / 7), np.full((1, 7), 'x')):
            with self.assertRaises(ValueError): cal.probabilities(q)
        w = windows((1, 1, 1)); w['july']['seeds'][0] = (w['july']['keys'], np.full((2, 7), 1 / 7))
        with self.assertRaises(ValueError): cal.fit(w)

    def test_changed_calibrator_contract_rejected(self):
        c = cal.fit(windows((1, 1, 1))); qs = {s: np.full((1, 7), 1 / 7) for s in cal.SEEDS}
        for name, value in [('clip', 1e-8), ('order', 'calibrate_then_mean'), ('temperature', 2),
                            ('kind', 'other'), ('temperature', True), ('reason', 'fitted')]:
            changed = dict(c); changed[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError): cal.apply(qs, changed)


def loader_fixture(root, order='C'):
    def bind(path):
        body = path.read_bytes()
        return {'path': str(path), 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
    def document(path, obj):
        path.write_text(json.dumps(obj)); return bind(path)
    reg = {'folds': {}}
    for f, fold in enumerate(cal.FOLDS):
        folder = root / fold; folder.mkdir()
        keys = np.array([[f * 10 + i + 1, 1, 1] for i in range(4)], order=order)
        fit = np.array([True, True, False, False]); keep = ~fit
        corpus = {n: np.zeros(4, dtype=np.int64) for n in worker.CORPUS}
        corpus.update(keys=keys, y7=np.arange(4), fit=fit, evaluation=keep,
                      tuning=np.zeros(4, bool), calibration=np.zeros(4, bool))
        cp = folder / 'corpus.npz'; np.savez(cp, **corpus); cb = bind(cp)
        recipe = {'arm': 'T', 'seeds': [0, 1, 2], 'data_role': '2026 EXPOSED DEVELOPMENT',
                  'window': worker.WINDOWS[fold], 'training_rule': {'epochs': 30}}
        rb = document(folder / 'recipe.json', recipe)
        manifest = {'status': 'COMPLETED', 'recipe_sha256': rb['sha256'], 'roles': {'FIT': 2, 'OOF': 2},
                    'files': {'corpus-subset.npz': {k: cb[k] for k in ('bytes', 'sha256')}}}
        mb = document(folder / 'manifest.json', manifest)
        hashes = {'FIT': worker.key_digest(keys[fit]), 'OOF': worker.key_digest(keys[keep])}
        grant = {'commit': 'synthetic', 'seeds': [0, 1, 2], 'input_manifest_sha256': mb['sha256'],
                 'expected_rows': {'FIT': 2, 'OOF': 2}, 'expected_key_sha256': hashes}
        gb = document(folder / 'grant.json', grant)
        seeds = {}; records = []
        for seed in cal.SEEDS:
            pp = folder / f'predictions-{seed}.npz'; np.savez(pp, keys=keys[keep], probabilities=np.full((2, 7), 1 / 7)); pb = bind(pp)
            receipt = {'status': 'COMPLETED', 'seed': seed, 'source_commit': 'synthetic', 'grant_sha256': gb['sha256'],
                       'recipe_sha256': rb['sha256'], 'input_manifest_sha256': mb['sha256'],
                       'outputs_sha256': {'oof-predictions.npz': pb['sha256']}, 'OOF_rows': 2,
                       'OOF_keys_sha256': hashes['OOF'], 'epochs': 30, 'temperature': 1, 'calibration_fits': 0,
                       'SELECT_queries': 0, 'checkpoint': 'final epoch', 'consumption': [
                           {'epoch': i, 'rows': 2, 'ordered_keys_sha256': hashes['FIT'],
                            'every_key_once': True, 'positive_parent_weight_every_row': True} for i in range(1, 31)]}
            sb = document(folder / f'receipt-{seed}.json', receipt)
            sup = {'status': 'COMPLETED', 'child_reaped': True, 'returncode': 0, 'receipt_sha256': sb['sha256'], 'grant_sha256': gb['sha256']}
            ub = document(folder / f'supervisor-{seed}.json', sup)
            seeds[str(seed)] = {'receipt': sb, 'supervisor': ub, 'predictions': pb}
            records.append({'seed': seed, 'receipt_sha256': sb['sha256'], 'supervisor_sha256': ub['sha256']})
        parent = {'status': 'COMPLETED', 'fits': 3, 'calibration_fits': 0, 'grant_sha256': gb['sha256'], 'seeds': records}
        reg['folds'][fold] = {'files': {'manifest': mb, 'corpus': cb, 'recipe': rb, 'fit_grant': gb,
            'parent': document(folder / 'parent.json', parent), 'acceptance': document(folder / 'acceptance.json', {'verdict': 'PASS_GENERATED'})},
            'seeds': seeds, 'expected': {'rows': 4, 'FIT': 2, 'OOF': 2, 'key_sha256': hashes,
                                       'source_commit': 'synthetic', 'window': recipe['window']}}
    return reg


class BoundedLoader(unittest.TestCase):
    def test_producer_fortran_keys_equal_c_order(self):
        import pandas as pd
        columns = ['game_pk', 'at_bat_number', 'pitch_number']
        frame = pd.DataFrame({columns[0]: [1, 2, 3, 4], columns[1]: [1] * 4, columns[2]: [1] * 4})
        keys = frame[columns].to_numpy(dtype='<i8')
        self.assertTrue(keys.flags.f_contiguous)
        self.assertFalse(keys.flags.c_contiguous)
        stream = io.BytesIO(); np.savez(stream, keys=keys)
        worker.header_check(stream, {'keys'}, {'keys': ((4, 3), 'iu')})
        with np.load(stream, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['keys'], keys)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            c = cal.pool(worker.load_windows(loader_fixture(Path(a), order='C')))
            f = cal.pool(worker.load_windows(loader_fixture(Path(b), order='F')))
            for left, right in zip(c, f):
                np.testing.assert_array_equal(left, right)

    def test_generated_registered_worker_writes_only_pending_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); reg = loader_fixture(root); out = root / 'out'; out.mkdir()
            recipe = root / 'calibration-recipe.json'; recipe.write_text(json.dumps(worker.RECIPE))
            reg.update(author='generated-author', recipe=str(recipe), calibration_recipe=worker.RECIPE,
                       source_pins={str(recipe): worker.sha(recipe)}, accepted_folds=list(cal.FOLDS), output=str(out),
                       bounds={'wall_seconds': 300, 'rss_bytes': 2 << 30, 'free_memory_bytes': 4 << 30,
                               'free_disk_bytes': 1 << 30, 'output_bytes': 64 << 10})
            rp = root / 'registration.json'; rp.write_text(json.dumps(reg))
            gp = root / 'grant.json'; gp.write_text(json.dumps({'execute': True, 'verdict': 'PASS',
                'registration_sha256': worker.sha(rp), 'author': reg['author'], 'reviewer': 'generated-reviewer', 'authorized_by': 'root'}))
            with (root / 'lock').open('w') as lock, patch.dict(os.environ, PITCHEEZY_FULLSEASON_LOCK_FD=str(lock.fileno())), \
                    patch.object(worker.signal, 'signal'), patch.object(worker.signal, 'alarm'):
                worker.run(rp, gp)
                self.assertEqual({p.name for p in out.iterdir()}, {'calibrator.json', 'receipt.pending.json'})
                result = json.loads((out / 'receipt.pending.json').read_text())
                self.assertEqual((result['status'], result['optimizer_calls'], result['metrics_computed']), ('COMPLETED', 0, False))
                with patch.object(worker.cal, 'fit') as fit, self.assertRaises(FileExistsError): worker.run(rp, gp)
                fit.assert_not_called()

    def test_generated_complete_loader_produces_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            reg = loader_fixture(Path(directory)); w = worker.load_windows(reg)
            c = cal.fit(w)
            self.assertEqual((c['kind'], c['rows'], c['optimizer_calls']), ('identity', 6, 0))

    def test_all_file_hashes_before_any_numpy_array_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            reg = loader_fixture(Path(directory))
            reg['folds']['september']['seeds']['2']['predictions']['sha256'] = '0' * 64
            with patch.object(worker.np, 'load') as load, self.assertRaisesRegex(ValueError, 'custody'):
                worker.load_windows(reg)
            load.assert_not_called()

    def test_wrong_receipt_generation_and_missing_member_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            reg = loader_fixture(Path(directory)); changed = copy.deepcopy(reg)
            changed['folds']['july']['expected']['source_commit'] = 'other'
            with self.assertRaisesRegex(ValueError, 'identity'): worker.load_windows(changed)
            del reg['folds']['august']['seeds']['2']
            with self.assertRaises(ValueError): worker.load_windows(reg)

    def test_header_rejects_object_compression_and_shape_before_load(self):
        for kwargs, compress in [({'keys': np.array([object()], object)}, False),
                                 ({'keys': np.ones((2, 3), np.int64)}, True),
                                 ({'keys': np.ones((3, 3), np.int64)}, False)]:
            stream = io.BytesIO()
            (np.savez_compressed if compress else np.savez)(stream, **kwargs)
            with self.assertRaises(ValueError): worker.header_check(stream, {'keys'}, {'keys': ((2, 3), 'iu')})

    def test_header_exact_member_byte_length_and_duplicate_member(self):
        npy = io.BytesIO(); np.save(npy, np.ones((2, 3), np.int64), allow_pickle=False)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive: archive.writestr('keys.npy', npy.getvalue() + b'extra')
        with self.assertRaisesRegex(ValueError, 'byte-length'): worker.header_check(stream, {'keys'}, {'keys': ((2, 3), 'iu')})
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('keys.npy', npy.getvalue())
            with self.assertWarns(UserWarning): archive.writestr('keys.npy', npy.getvalue())
        with self.assertRaises(ValueError): worker.header_check(stream, {'keys'}, {'keys': ((2, 3), 'iu')})

    def test_input_caps_and_symlink_reject_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            reg = loader_fixture(Path(directory)); cb = reg['folds']['july']['files']['corpus']
            original = cb['path']; link = Path(directory) / 'linked.npz'; link.symlink_to(original); cb['path'] = str(link)
            with self.assertRaisesRegex(ValueError, 'nonsymlink'): worker.load_windows(reg)
            cb['path'] = original; cb['bytes'] = worker.FILE_CAP + 1
            with self.assertRaisesRegex(ValueError, 'byte bound'): worker.load_windows(reg)


if __name__ == '__main__':
    unittest.main()
