"""Generated probabilities/keys only; no file, model or real-data inputs."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import softmax
from experiments.ws_fullseason import calibrate as cal


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


if __name__ == '__main__':
    unittest.main()
