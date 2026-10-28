"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 phase-2 drivers: the new S-2' driver is bit-identical to the
EXP-OPE-RES-01 ``run_s5x`` on a fixture world at the same A2 seeds (per-run b_hat and SE, repetitions, pairs), the A2
and A2b seed schemes share one code path (only the seeds differ), fresh-only operator-blind pooling with fixed run
counts, the G-INT-S2EQ integrity record (absolute 1e-12 exact form, auxiliary statistical form, paths), the TAB
re-run on the A2 world's fixed draws with A2b replication streams and the new size, and the exact-value cell rule.
The fixture world is the existing synthetic candidate runtime (fake G0 + WE, test_policy_validation); no real data.
"""
import copy
import math
from pathlib import Path
import shutil
import struct
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

from ope_a2_fixtures import synthetic_tab_marginals, synthetic_table  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp import ope_synthetic_a2b as sx  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402


def bits(value):
    if isinstance(value, dict):
        return [(str(k), bits(value[k])) for k in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    if isinstance(value, np.ndarray):
        return bits(value.tolist())
    if isinstance(value, (float, np.floating)):
        return struct.pack('<d', float(value))
    return value


def tab_marginals(seed=0):
    mg = synthetic_tab_marginals(seed)
    shares = mg.archetype_shares
    shares[shares.sum(axis=1) == 0] = 1.0
    mg.archetype_shares = shares
    return mg


def run_of_k(k):
    return next(r for r in reg.S5X_RUNS if r['k'] == k)


class S2PrimeDriverTests(unittest.TestCase):
    """The fixture world of the EXP-OPE-RES-01 S5-X tests (cap 4 for speed; the registered run uses cap 12)."""
    @classmethod
    def setUpClass(cls):
        import test_policy_validation as tpv
        cls.tpv = tpv

    def fixture(self, name, *, kappa=0.3, zeta=0.0):
        from pitchmdp.rollout_policy import RowBudget
        fx = self.tpv.SemiSyntheticTests('test_v2_matches_world_truth_and_is_not_vacuous')
        fx.setUp()
        self.addCleanup(shutil.rmtree, fx.root, True)
        runtime, components, starts = fx.world(name)
        rng = np.random.default_rng(0)
        layers = osx.S5XLayers.from_components(
            components, fx.syn.game_values, groups=reg.S5X_GROUPS, x_dist=rng.dirichlet(np.ones(reg.N_X) * .5, size=(5, 2)),
            class_table=synthetic_table(rng), signs=[1, -1, 1, 0, 0], signs_prime=[0, 1, -1, 1, 0], kappa=kappa, zeta=zeta)
        law = lambda s: (runtime.bc.actions, runtime.bc.probabilities(s))
        return runtime, components, starts, layers, law, RowBudget(10 ** 8, seed_count=5)

    def pair_of_runs(self, k, cell, *, kappa, zeta):
        runtime, components, starts, layers, law, budget = self.fixture(f'a2-k{k}', kappa=kappa, zeta=zeta)
        run = dict(next(r for r in reg.S5X_RUNS if r['k'] == k))
        old = osx.run_s5x(runtime, components, starts, run=run, law=law, layers=layers, cap=4, gamma=True,
                          invariant=run['zeta'] == '0', budget=budget)
        runtime2, components2, starts2, layers2, law2, budget2 = self.fixture(f'new-k{k}', kappa=kappa, zeta=zeta)
        new = sx.run_s2_prime(runtime2, components2, starts2, k=k, scheme='A2', cell=cell, law=law2, layers=layers2,
                              cap=4, budget=budget2)
        return old, new

    def test_bit_identity_with_run_s5x_on_the_fixture_world(self):
        records = {}
        for k, cell, kappa, zeta in ((4, '2*kappa_p', 0.3, 0.0), (5, '2*zeta_p', 0.0, 1.1)):
            with self.subTest(k=k):
                old, new = self.pair_of_runs(k, cell, kappa=kappa, zeta=zeta)
                records[k] = new
                self.assertEqual(new['world_seed'], reg.V2_ROLE_SEED ^ k)
                for comp in ('R1_a', 'R1_b'):
                    self.assertEqual(bits(new['s2'][comp]), bits(old['s2'][comp]), (k, comp))
                for key in ('pairs', 'repetitions', 's2_pairs', 'invariant', 'kept_starts', 'excluded_starts',
                            'inherited_s5_se'):
                    self.assertEqual(bits(new[key]), bits(old[key]), (k, key))
                self.assertEqual(new['run']['path'], 'a2_order_with_log_phase')
                self.assertEqual((new['run']['kappa'], new['run']['zeta']), (run_of_k(k)['kappa'], run_of_k(k)['zeta']))
        self.assertNotEqual(records[4]['s2']['R1_b']['b_hat'], 0.0)  # kappa > 0 moves R1_b (not a vacuous identity)
        self.assertGreater(records[4]['s2']['R1_b']['se'], 0.0)
        self.assertEqual(records[4]['invariant']['outcome'], 'PASS')
        self.assertIsNone(records[5]['invariant'])  # the zeta run has no gamma_a invariant

    def test_the_two_seed_schemes_share_one_code_path(self):
        runtime, components, starts, layers, law, budget = self.fixture('a2-scheme')
        a2_run = sx.run_s2_prime(runtime, components, starts, k=4, scheme='A2', cell='2*kappa_p', law=law,
                                 layers=layers, cap=4, budget=budget)
        runtime2, components2, starts2, layers2, law2, budget2 = self.fixture('a2b-seeded-like-a2')
        real = sx.scheme_seeds
        with mock.patch.object(sx, 'scheme_seeds', lambda scheme, k: real('A2', 4)):  # an A2b run fed the A2 seeds
            same = sx.run_s2_prime(runtime2, components2, starts2, k=11, scheme='A2b', cell='2*kappa_p', law=law2,
                                   layers=layers2, cap=4, budget=budget2)
        for key in ('s2', 'pairs', 'repetitions', 'invariant'):
            self.assertEqual(bits(same[key]), bits(a2_run[key]), key)
        runtime3, components3, starts3, layers3, law3, budget3 = self.fixture('a2b-own')
        fresh = sx.run_s2_prime(runtime3, components3, starts3, k=11, scheme='A2b', cell='2*kappa_p', law=law3,
                                layers=layers3, cap=4, budget=budget3)
        self.assertEqual(fresh['world_seed'], rb.world_seed(11))
        self.assertNotEqual(bits(fresh['s2']['R1_b']), bits(a2_run['s2']['R1_b']))
        self.assertEqual(fresh['s2']['R1_a']['b_hat'], 0.0)  # the gamma_a invariant at zeta = 0
        self.assertEqual(fresh['invariant']['outcome'], 'PASS')
        with self.assertRaisesRegex(IntegrityError, 'not a registered run of'):
            sx.run_s2_prime(runtime3, components3, starts3, k=3, scheme='A2b', cell='2*kappa_p', law=law3,
                            layers=layers3, cap=4, budget=budget3)
        with self.assertRaisesRegex(IntegrityError, 'not a recorded run of the 2\\*zeta_p dose family'):
            sx.run_s2_prime(runtime3, components3, starts3, k=4, scheme='A2', cell='2*zeta_p', law=law3,
                            layers=layers3, cap=4, budget=budget3)  # k = 4 belongs to the kappa family
        with self.assertRaises(IntegrityError):
            sx.scheme_seeds('A2', 16)
        with self.assertRaises(IntegrityError):
            sx.scheme_seeds('A2b', 71)

    def test_the_s2_prime_budget_split_is_required(self):
        """MINOR-5 (guard :67): logs per start = repetitions and truth rollouts = 2 x repetitions."""
        from types import SimpleNamespace
        components = object()
        runtime = SimpleNamespace(components=components, improvement=object())
        for logs, truth in ((5, 12), (6, 10), (6, 13), (7, 14)):
            with self.subTest(logs=logs, truth=truth), \
                    self.assertRaisesRegex(IntegrityError, 'repetitions split the S5 budget'):
                sx.run_s2_prime(runtime, components, [], k=11, scheme='A2b', cell='2*kappa_p', law=None, layers=None,
                                logs_per_start=logs, truth_rollouts=truth)

    def test_a_gamma_a_violation_at_zeta_zero_is_an_integrity_failure(self):
        """MINOR-5 (guard :140-142): at zeta = 0 every gamma_a is exactly 0; one off by 1e-9 raises."""
        runtime, components, starts, layers, law, budget = self.fixture('gamma-violation')
        real = sx.osx._gamma

        def off(*args, **kwargs):
            g_a, g_b = real(*args, **kwargs)
            return g_a + 1e-9, g_b
        with mock.patch.object(sx.osx, '_gamma', off), self.assertRaisesRegex(IntegrityError, 'invariant violated'):
            sx.run_s2_prime(runtime, components, starts, k=11, scheme='A2b', cell='2*kappa_p', law=law, layers=layers,
                            cap=4, budget=budget)

    def test_gint_s2eq_record(self):
        runs = {}
        for k in (4, 9, 10):
            runtime, components, starts, layers, law, budget = self.fixture(f'eq-k{k}')
            runs[k] = sx.run_s2_prime(runtime, components, starts, k=k, scheme='A2', cell='2*kappa_p', law=law,
                                      layers=layers, cap=4, budget=budget)
        recorded = {k: {c: (r['s2'][c]['b_hat'], r['s2'][c]['se']) for c in ('R1_a', 'R1_b')} for k, r in runs.items()}
        # NOTE-2: an A2-scheme run carries the recorded run's own levels (metadata; b_hat and SE are the A2 values)
        self.assertEqual({k: (r['run']['kappa'], r['run']['zeta']) for k, r in runs.items()},
                         {4: ('2*kappa_p', '0'), 9: ('kappa_p', '0'), 10: ('5*kappa_p', '0')})
        self.assertTrue(all(r['invariant']['outcome'] == 'PASS' for r in runs.values()))  # zeta = 0 runs
        reference = {'path': 'results/EXP-OPE-RES-01.json', 'sha256': 'a' * 64}
        record = sx.gint_s2eq_record(['2*kappa_p'], runs, recorded, reference=reference)
        self.assertEqual((record['result'], record['max_abs_delta']), ('PASS', 0.0))
        self.assertEqual(record['run_indices'], [4, 9, 10])
        self.assertEqual(record['failure_scope'], 'all')  # NOTE-8: written explicitly
        self.assertEqual(set(record['paths'].values()), {'a2_order_with_log_phase'})
        self.assertEqual(record['auxiliary_statistical_form']['result'], 'PASS')
        self.assertFalse(record['auxiliary_statistical_form']['decision_input'])
        limit = record['auxiliary_statistical_form']['detection_limit_B']['k4']['R1_b']
        se = runs[4]['s2']['R1_b']['se']
        self.assertAlmostEqual(limit, 3 * math.sqrt(2) * se / rb.B, places=9)
        self.assertEqual(record['auxiliary_statistical_form']['detection_limit_B']['k4']['R1_a'], 0.0)  # invariant zero
        self.assertTrue(sx.gint_s2eq_scope(record))
        near = copy.deepcopy(recorded)
        near[9]['R1_b'] = (near[9]['R1_b'][0] + 5e-13, near[9]['R1_b'][1])
        self.assertEqual(sx.gint_s2eq_record(['2*kappa_p'], runs, near, reference=reference)['result'], 'PASS')
        far = copy.deepcopy(recorded)
        far[9]['R1_b'] = (far[9]['R1_b'][0] + 2e-12, far[9]['R1_b'][1])
        bad = sx.gint_s2eq_record(['2*kappa_p'], runs, far, reference=reference)
        self.assertEqual(bad['result'], 'FAIL')
        self.assertEqual(bad['mismatched_keys'], ['k9.R1_b.b_hat'])
        with self.assertRaisesRegex(IntegrityError, 'G-INT-S2EQ failed'):
            sx.gint_s2eq_scope(bad)
        defect = copy.deepcopy(record)
        defect['auxiliary_statistical_form']['result'] = 'FAIL'
        with self.assertRaisesRegex(IntegrityError, 'comparison defect'):
            sx.gint_s2eq_scope(defect)
        with self.assertRaisesRegex(IntegrityError, 'exactly the recorded runs'):
            sx.gint_s2eq_record(['2*kappa_p', '2*zeta_p'], runs, recorded, reference=reference)
        self.assertTrue(rb.validate_integrity_record(record, 'gint_s2eq'))
        self.assertEqual(record['reference_cross_check']['sealed_a2_s2_run_records'], 'not supplied')
        crossed = sx.gint_s2eq_record(['2*kappa_p'], runs, recorded, reference=reference, sealed=copy.deepcopy(recorded))
        self.assertEqual(crossed['reference_cross_check'], {'sealed_a2_s2_run_records': 'compared', 'equal': True,
                                                            'mismatched_keys': []})
        extracted = copy.deepcopy(recorded)
        extracted[10]['R1_b'] = (extracted[10]['R1_b'][0], extracted[10]['R1_b'][1] * 1.01)  # a PM-extraction slip
        slip = sx.gint_s2eq_record(['2*kappa_p'], runs, recorded, reference=reference, sealed=extracted)
        self.assertEqual(slip['reference_cross_check']['mismatched_keys'], ['k10.R1_b.se'])
        self.assertEqual(slip['result'], 'PASS')  # the registered reference (results file) still decides
        floats = sorted(v for _, v in rb.leaves(record) if isinstance(v, float))
        self.assertTrue(all(v in (0.0, 1e-12) or v == limit or v >= 0 for v in floats))
        self.assertNotIn('b_hat', str(record['auxiliary_statistical_form']))


class PoolingTests(unittest.TestCase):
    def runs(self, cell, *, bad=None, pair_invalid=None):
        sealed = {}
        rng = np.random.default_rng(3)
        for k in rb.RUN_INDEX[cell]:
            zeta_zero = rb.S2P_CELL_LEVELS[cell][1] == '0'
            record = {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S2a', 'cell': cell, 'k': k,
                      'world_seed': rb.world_seed(k), 'path': rb.S2P_PATH, 'kept_starts': 147,
                      's2': {'R1_a': {'b_hat': 0.0 if zeta_zero else float(rng.normal(5e-6, 3e-6)),
                                      'se': 0.0 if zeta_zero else 3.1e-6},
                             'R1_b': {'b_hat': float(rng.normal(0, 2e-5)), 'se': 2.3e-5}},
                      'invariant': {'max_abs_gamma_a': 0.0, 'outcome': 'PASS'} if zeta_zero else None,
                      'pair_valid': {p: not (pair_invalid == (k, p)) for p in reg.PAIRS},
                      'valid': k != bad}
            sealed[k] = sx.seal_run(record)
        return sealed

    def test_pooling_waits_for_every_registered_run(self):
        sealed = self.runs('2*zeta_p')
        pooled = sx.pool_cell('2*zeta_p', sealed)
        records = [__import__('json').loads(sealed[k][0]) for k in rb.RUN_INDEX['2*zeta_p']]
        b = [r['s2']['R1_a']['b_hat'] for r in records]
        self.assertEqual(pooled['n_prime'], 10)
        self.assertEqual(pooled['components']['R1_a']['b_hat'], sum(b) / 10)
        self.assertAlmostEqual(pooled['components']['R1_a']['se'], math.sqrt(10 * 3.1e-6 ** 2) / 10, places=18)
        self.assertTrue(pooled['decision_input'])
        partial = dict(sealed)
        partial.pop(7)
        with self.assertRaisesRegex(IntegrityError, 'operator-blind pooling'):
            sx.pool_cell('2*zeta_p', partial)
        extra = dict(sealed)
        extra[11] = sealed[10]
        with self.assertRaisesRegex(IntegrityError, 'operator-blind pooling'):
            sx.pool_cell('2*zeta_p', extra)
        tampered = dict(sealed)
        tampered[3] = (sealed[3][0] + b' ', sealed[3][1])
        with self.assertRaisesRegex(IntegrityError, 'differs from its seal'):
            sx.pool_cell('2*zeta_p', tampered)

    def test_an_invalid_run_invalidates_the_cell_and_pair_validity_carries(self):
        pooled = sx.pool_cell('2*kappa_p', self.runs('2*kappa_p', bad=20))
        self.assertEqual(pooled['components']['R1_b']['outcome'], 'INVALID')
        self.assertEqual(pooled['components']['R1_b']['invalid_runs'], [20])
        ok = sx.pool_cell('2*kappa_p', self.runs('2*kappa_p', pair_invalid=(30, 'R1_bxR2_1')))
        self.assertEqual(ok['components']['R1_a']['outcome'], 'PASS')  # the exact gamma_a invariant at zeta = 0
        self.assertEqual(ok['tuples']['R1_bxR2_1xB0']['outcome'], 'INCONCLUSIVE')
        self.assertTrue(ok['tuples']['R1_bxR2_1xB0']['invalid_repetition'])
        self.assertEqual(ok['tuples']['R1_bxR2_0xB0']['outcome'], ok['components']['R1_b']['outcome'])
        self.assertEqual(ok['n_prime'], 55)
        descriptive = sx.pool_cell('descriptive_kappa_prime', self.runs('descriptive_kappa_prime'))
        self.assertFalse(descriptive['decision_input'])

    def test_a_run_is_valid_only_with_finite_values_and_its_invariant(self):
        """MINOR-5 (run_record :200-204): an invariant that did not PASS, or a nonfinite b_hat / SE, makes the run
        invalid (its cell INVALID at pooling)."""
        result = {'run': {'scheme': 'A2b', 'cell': '2*kappa_p', 'k': 11, 'path': rb.S2P_PATH},
                  'world_seed': rb.world_seed(11), 'kept_starts': 3,
                  's2': {'R1_a': {'b_hat': 0.0, 'se': 0.0}, 'R1_b': {'b_hat': 1e-5, 'se': 2e-5}},
                  'invariant': {'max_abs_gamma_a': 0.0, 'outcome': 'PASS'},
                  'pairs': {p: {'valid': True} for p in reg.PAIRS}}
        self.assertTrue(sx.run_record(result)['valid'])
        self.assertTrue(sx.run_record({**result, 'invariant': None})['valid'])  # a zeta run has no invariant
        self.assertFalse(sx.run_record({**result, 'invariant': {'max_abs_gamma_a': 1e-9, 'outcome': 'FAIL'}})['valid'])
        nan = copy.deepcopy(result)
        nan['s2']['R1_b']['se'] = math.nan
        self.assertFalse(sx.run_record(nan)['valid'])

    def test_only_fresh_a2b_runs_are_sealed(self):
        result = {'run': {'scheme': 'A2', 'cell': '2*kappa_p', 'k': 4, 'path': rb.S2P_PATH}}
        with self.assertRaisesRegex(IntegrityError, 'never pooled'):
            sx.run_record(result)
        public = sx.run_public({'cell': '2*zeta_p', 'k': 1, 'valid': True}, 'a' * 64, wall_seconds=400.0,
                               peak_rss_bytes=10)
        self.assertNotIn('b_hat', public)
        self.assertTrue(rb.validate_run_public(public))


class TabRerunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mg = tab_marginals()
        cell = dict(rb.TAB_ACCEPTANCE[0], index=0, role='acceptance', drift=False)
        cls.cell = cell
        cls.world = sx.tab_world(cls.mg, cell, eta_p=0.16)

    def test_fixed_draws_are_the_a2_world_and_replications_the_a2b_stream(self):
        a2_world = osx.TabWorld(self.mg, law='V2', eta=reg.scale('eta_p', 0.16), kappa=0.0, zeta=0.0)
        self.assertEqual(bits(self.world.eps), bits(a2_world.eps))  # role_sequence('tab_world', 0, 3) of A2
        self.assertEqual(bits(self.world.q_hat), bits(a2_world.q_hat))
        new = sx.tab_replication(self.world, 0, 2, n_train=self.mg.n_train_bip, n_pas=600)
        old = osx.tab_replication(self.world, 0, 2, n_train=self.mg.n_train_bip)
        self.assertNotEqual(bits(new[0]), bits(old[0]))  # A2b replication stream, not the A2 one
        again = sx.tab_replication(self.world, 0, 2, n_train=self.mg.n_train_bip, n_pas=600)
        self.assertEqual(bits(new), bits(again))
        rng = np.random.default_rng(rb.role_sequence('tab_world', 1, 0, 2))
        self.world.synthetic_train(rng, self.mg.n_train_bip)
        data = self.world.simulate(rng, n_pas=600)
        self.assertEqual(len(data['reward']), 600)  # the new dataset size

    def test_cells_and_the_exact_value_rule(self):
        record = sx.tab_cell(self.world, self.cell, n_train=self.mg.n_train_bip, n_pas=400, replications=3)
        self.assertEqual(record['dataset_pas'], 400)
        self.assertEqual(record['pairs'][reg.BASELINE]['outcome'], 'EXACT_DP')
        real = osx.pipeline

        def tied(data, rewards, **kwargs):  # R1_0 x R2_1 equals the baseline in every replication
            out, selected, invalid = real(data, rewards, **kwargs)
            return {**out, 'R1_0xR2_1': out['R1_0xR2_0']}, selected, invalid
        with mock.patch.object(osx, 'pipeline', tied):
            exact = sx.tab_cell(self.world, self.cell, n_train=self.mg.n_train_bip, n_pas=400, replications=3)
        entry = exact['pairs']['R1_0xR2_1']
        self.assertEqual(entry['outcome'], 'EXACT_PASS')  # V2: beta_0 = 0 by DP; A2 would say INCONCLUSIVE (SE = 0)
        self.assertTrue(entry['exact'])
        v3 = dict(rb.TAB_ACCEPTANCE[1], index=1, role='acceptance', drift=False)
        world3 = sx.tab_world(self.mg, v3, eta_p=0.16)
        with mock.patch.object(osx, 'pipeline', tied):
            relative = sx.tab_cell(world3, v3, n_train=self.mg.n_train_bip, n_pas=400, replications=2)
        self.assertEqual(relative['pairs']['R1_0xR2_1']['outcome'], 'EXACT_PASS')  # |beta_0| <= |beta_0| + B
        self.assertEqual(relative['pairs']['R1_0xR2_1']['rule'], 'relative')

    def test_kappa_prime_uses_the_four_large_groups(self):
        rng = np.random.default_rng(4)
        groups = np.array(rng.choice(['FB', 'FC', 'BR', 'OS', 'other'], size=4000), dtype=object)
        hit = rng.uniform(0.2, 0.4, size=4000)
        diag = {g: {'n': 800, 'mean_residual': (0.02 if g == 'other' else 0.004), 'se': (0.011 if g == 'other' else 0.002)}
                for g in reg.S5X_GROUPS}
        prime = sx.kappa_prime(hit, groups, diag)
        full = __import__('pitchmdp.ope_surrogate', fromlist=['x']).calibrate_kappa(hit, groups, diag, reg.S5X_GROUPS)
        self.assertEqual(prime['groups'], ['FB', 'FC', 'BR', 'OS'])
        self.assertLess(prime['kappa_p'], full['kappa_p'])  # 'other' no longer sets the target


if __name__ == '__main__':
    unittest.main()
