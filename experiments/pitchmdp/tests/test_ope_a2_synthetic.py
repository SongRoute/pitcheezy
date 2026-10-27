"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-01 certification harness: the TAB exact-DP world (baseline
expectation = truth under V2, simulator vs DP, calibration targets, full replications, cell outcomes)
and the S5-X layered world (per-run streams, repetition slices, CRN, the gamma_a invariant, end to end
on the existing synthetic candidate runtime with fake G0/WE, and the X2 single-member predictor).
No real data, model weights or 2026 access.
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

from ope_a2_fixtures import ToyAdvancement, ToyWE, synthetic_tab_marginals, synthetic_table  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_synthetic_a2 as osx  # noqa: E402
from pitchmdp import ope_variants as ov  # noqa: E402
from pitchmdp.ope_diagnostics import affine_padded  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

INF = math.inf


def marginals(seed=0):
    mg = synthetic_tab_marginals(seed)
    shares = mg.archetype_shares
    shares[shares.sum(axis=1) == 0] = 1.0
    mg.archetype_shares = shares
    return mg


class TabWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mg = marginals()

    def test_v2_baseline_dp_expectation_equals_truth(self):
        for kw in (dict(eta=0.0, kappa=0.0, zeta=0.0), dict(eta=0.3, kappa=0.4, zeta=0.0),
                   dict(eta=0.15, kappa=0.0, zeta=0.9), dict(eta=0.15, kappa=0.0, zeta=0.0, drift=True)):
            world = osx.TabWorld(self.mg, law='V2', **kw)
            with self.subTest(**kw):
                self.assertLessEqual(abs(world.expected_delta_hat_0 - world.delta_true), 1e-12)
                self.assertTrue(osx.tab_integrity(world))
        v3 = osx.TabWorld(self.mg, law='V3', eta=0.15, kappa=0.0, zeta=0.0)
        self.assertGreater(abs(v3.beta_0), 1e-9)  # V3 misspecifies the denominator on purpose
        broken = osx.TabWorld(self.mg, law='V2', eta=0.0, kappa=0.0, zeta=0.0)
        broken.expected_delta_hat_0 += 1e-9
        with self.assertRaisesRegex(IntegrityError, 'DP expectation'):
            osx.tab_integrity(broken)

    def test_simulator_matches_the_dp_world(self):
        """Mean realized WE, the R1_a value and the plain DR of simulated logs agree with exact DP."""
        world = osx.TabWorld(self.mg, law='V3', eta=0.15, kappa=0.4, zeta=0.9)
        shape = world.pi_b_full.shape
        logging = np.broadcast_to(world.logging[:, None], shape)
        v_log = osx._solve(logging, world.pb, world.ps, world.pf, world.imm_world, world.walk, world.strikeout)[0][0, 0].mean()
        tv_bip = world.tv[:, [osx.T9[e] for e in reg.BIP_CLASSES]]
        dp = np.array([s.dp for s in self.mg.situations]).astype(int)
        pcls = np.moveaxis(world.pcls[:, world.ts.astype(int)][:, :, dp], 0, 2)  # [12, S, G, 6]
        imm_a = world.p5[:, None][..., 3] * world.hbp[None, :, None, None] + world.p5[:, None][..., 4] * (
            (pcls * tv_bip[None, :, None]).sum(-1))[:, :, None, :]
        v_r1a = osx._solve(logging, world.pb, world.ps, world.pf, imm_a, world.walk, world.strikeout)[0][0, 0].mean()
        rewards, r1a, deltas = [], [], []
        for j in range(30):
            data = world.simulate(np.random.default_rng(reg.role_sequence('smoke', 7, j)), n_pas=4000)
            rewards.append(data['reward'].mean())
            r1a.append(osx.tab_rewards(world, data, self.mg.class_table)['R1_a'].mean())
            deltas.append(osx.pipeline(data, osx.tab_rewards(world, data, self.mg.class_table))[0][reg.BASELINE])
        for name, values, target in (('realized WE', rewards, v_log), ('R1_a', r1a, v_r1a),
                                     ('plain DR', deltas, world.expected_delta_hat_0)):
            se = np.std(values, ddof=1) / math.sqrt(len(values))
            with self.subTest(name):
                self.assertLess(abs(np.mean(values) - target), 4.5 * se + 1e-12, (np.mean(values), target, se))
        self.assertGreater(abs(v_log - v_r1a), 1e-6)  # zeta moves the value of outs with a runner on third

    def test_calibration_targets(self):
        eta_p = osx.TabWorld.eta_p(self.mg)
        ece = lambda eta: osx.top_label_ece(osx.TabWorld(self.mg, law='V2', eta=eta, kappa=0.0, zeta=0.0).model10,
                                            base.world10, weights)
        base = osx.TabWorld(self.mg, law='V2', eta=eta_p, kappa=0.0, zeta=0.0)
        weights = base.visits[..., None] * base.pi_b_full
        self.assertGreaterEqual(ece(eta_p), 2 * 0.0087)  # the smallest bracketed eta reaching 2 x 0.0087
        self.assertLess(ece(eta_p * (1 - 1e-9)), 2 * 0.0087)
        self.assertLess(abs(ece(eta_p) - 2 * 0.0087), 5e-4)
        pi = base.pi_c
        with np.errstate(divide='ignore', invalid='ignore'):
            kl = np.where(base.mask & (pi > 0), pi * np.log(pi / base.pi_b_full), 0.).sum(-1)
        self.assertAlmostEqual(float((base.visits * kl).sum() / base.visits.sum()), 0.0074, places=9)

    def test_replications_are_seeded_and_complete(self):
        world = osx.TabWorld(self.mg, law='V2', eta=0.1, kappa=0.0, zeta=0.0)
        a, sel_a, bad_a = osx.tab_replication(world, 0, 0, n_train=self.mg.n_train_bip)
        b, sel_b, _ = osx.tab_replication(world, 0, 0, n_train=self.mg.n_train_bip)
        c, _, _ = osx.tab_replication(world, 0, 1, n_train=self.mg.n_train_bip)
        d, _, _ = osx.tab_replication(world, 1, 0, n_train=self.mg.n_train_bip)
        self.assertEqual(a, b)  # same seed, bit-identical
        self.assertEqual((sel_a, bad_a), (sel_b, []))
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)
        self.assertEqual(set(a), set(reg.PAIRS))
        self.assertIn(sel_a['m'], reg.M_SMOOTH_GRID)
        smoke_world = osx.TabWorld(self.mg, law='V2', eta=0.0, kappa=0.0, zeta=0.0, seed_role='smoke')
        e, sel_e, _ = osx.tab_replication(smoke_world, 0, 0, n_train=self.mg.n_train_bip,
                                          smoke={'m': 20, 'lambda': 30.0, 'M': 5.0})
        self.assertEqual(sel_e['m'], 20)
        self.assertNotEqual(e, osx.tab_replication(osx.TabWorld(self.mg, law='V2', eta=0.0, kappa=0.0, zeta=0.0), 0, 0,
                                                   n_train=self.mg.n_train_bip, smoke={'m': 20, 'lambda': 30.0, 'M': 5.0})[0])

    def test_cell_outcomes_and_invalid_replications(self):
        world = osx.TabWorld(self.mg, law='V2', eta=0.1, kappa=0.0, zeta=0.0)
        cell = dict(reg.tab_cells()[0])
        record = osx.tab_cell(world, cell, n_train=self.mg.n_train_bip, replications=3)
        base = record['pairs'][reg.BASELINE]
        self.assertEqual((base['b_hat'], base['se'], base['outcome']), (world.beta_0, 0.0, 'PASS'))
        for pair in reg.TOTAL_ORDER:
            self.assertIn(record['pairs'][pair]['outcome'], ('PASS', 'FAIL', 'INCONCLUSIVE'))
        real = osx.pipeline

        def broken(data, rewards, **kw):
            out, selected, invalid = real(data, rewards, **kw)
            return {**out, 'R1_bxR2_3': math.nan}, selected, invalid + ['R1_bxR2_3']
        with mock.patch.object(osx, 'pipeline', broken):
            record = osx.tab_cell(world, cell, n_train=self.mg.n_train_bip, replications=2)
        self.assertEqual(record['pairs']['R1_bxR2_3']['outcome'], 'INVALID')  # never dropped and re-averaged
        self.assertEqual(record['pairs']['R1_bxR2_3']['invalid_replications'], 2)
        descriptive = osx.tab_cell(world, dict(reg.tab_cells()[10]), n_train=self.mg.n_train_bip, replications=2)
        self.assertNotIn('outcome', descriptive['pairs'][reg.BASELINE])

    def test_synthetic_train_folds(self):
        world = osx.TabWorld(self.mg, law='V2', eta=0.0, kappa=0.4, zeta=0.0)
        a, b = world.synthetic_train(np.random.default_rng(0), 5000)
        self.assertEqual(int(a.sum() + b.sum()), 5000)
        self.assertEqual(int(a[0::2, 5].sum() + b[0::2, 5].sum()), 0)  # no double play at dp = 0

    def test_vectorized_dr_is_bit_identical_to_the_scalar_recursion(self):
        rng = np.random.default_rng(4)
        lines = []
        for _ in range(40):
            T = int(rng.integers(1, 7))
            rho = rng.uniform(0, 2, T) * (rng.random(T) > .1)
            lines.append(tuple((float(rng.uniform(.3, .7)), float(rng.uniform(.3, .7)), float(rng.uniform(.3, .7)),
                                float(r), float(rng.uniform(.5, 1.5))) for r in rho))
        steps, lengths = osx._padded_lines([{'steps': s} for s in lines])
        u = np.concatenate([[1.0], rng.uniform(.8, 1.2, 7)])
        for f_kind, param in (('identity', None), ('shrink', 3.0), ('switch', 1.0)):
            f_scalar = ov.transform(f_kind, param)
            f_array = osx._transform_array(f_kind, param)
            rho = steps['rho_c'] if f_array is None else f_array(steps['rho_c'])
            base, slope = affine_padded(steps['v_c'], steps['q_a'], rho, lengths)
            for i, s in enumerate(lines):
                self.assertEqual((base[i], slope[i]), ov.affine_t(s, 0, f_scalar))
        base, slope = affine_padded(steps['v_c'], steps['q_a'], steps['rho_c'], lengths, u)
        for i, s in enumerate(lines):
            self.assertEqual((base[i], slope[i]), ov.affine_sn(s, 0, u))


class StreamTests(unittest.TestCase):
    def test_repetitions_partition_the_s5_budget(self):
        logs, truth = [], []
        for r in range(1, 7):
            streams = osx.repetition_streams(r)
            logs.append(streams['log'])
            truth.extend(streams['truth'])
        self.assertEqual(logs, list(range(6)))
        self.assertEqual(truth, list(range(12)))
        with self.assertRaises(IntegrityError):
            osx.repetition_streams(7)

    def test_world_streams_are_disjoint(self):
        seeds = {r['k']: reg.world_seed(r['k']) for r in reg.S5X_RUNS}
        draws = set()
        for k, ws in seeds.items():
            for key in ([ws, 0, 3, 1], [ws, 1, 3]):
                draws.add(tuple(np.random.default_rng(np.random.SeedSequence(key)).random(3)))
        self.assertEqual(len(draws), 30)


class Pool:
    def __init__(self, values):
        self.values = values


class LayerWorldTests(unittest.TestCase):
    def layers(self, kappa=0.0, zeta=0.0, situation=None):
        from pitchmdp.game import GameState
        rng = np.random.default_rng(2)
        sit = situation or osx.Situation.from_models(GameState(6, 'Top', 1, 5, 2, 3), ToyWE(), ToyAdvancement())
        return osx.S5XLayers(groups=reg.S5X_GROUPS, x_dist=rng.dirichlet(np.ones(reg.N_X), size=(5, 2)),
                             class_table=synthetic_table(rng), signs=[1, -1, 1, 0, 0], signs_prime=[1, 1, 0, -1, 0],
                             kappa=kappa, zeta=zeta, situations={'ctx': sit})

    def world(self, layers, outcome='single'):
        p = np.zeros(10)
        p[osx.OUTCOMES.index(outcome)] = 1.0
        return osx.QualityLayerWorld(lambda s, a: Pool(np.zeros((400, 2))), lambda st, ac, ph: np.tile(p, (len(st), 1)),
                                     lambda s, t: 0.5, SimpleNamespace(consume=lambda n: None), layers)

    def state(self, strikes=0):
        from pitchmdp.rollout_policy import PAState
        return PAState(0, strikes, '9', 'R', (), 'ctx')

    def test_in_play_layers_follow_the_layer_uniforms(self):
        world = self.world(self.layers(kappa=0.5, zeta=0.8))
        u = np.full((8, 2), 0.3)
        lu = np.random.default_rng(5).random((8, 3))
        states, actions = [self.state()] * 8, ['FF', 'SL', 'CH', 'FC', 'FF', 'SL', 'CH', 'FC']
        first = [s.value for s in world.step(states, actions, u, layer_u=lu)]
        again = [s.value for s in world.step(states, actions, u, layer_u=lu)]
        other = [s.value for s in world.step(states, actions, u, layer_u=np.random.default_rng(6).random((8, 3)))]
        self.assertEqual(first, again)  # common random numbers: the same layer draws give the same outcomes
        self.assertNotEqual(first, other)
        self.assertTrue(all(0 <= v <= 1 for v in first))
        world.bind(lu)
        logged = [world.step([self.state()], [a], u[:1])[0].value for a in actions[:3]]
        self.assertEqual(logged, first[:3])  # the log path consumes its bound stream row by row
        with self.assertRaisesRegex(IntegrityError, 'pitch group'):
            world.step([self.state()], ['XX'], u[:1], layer_u=lu[:1])

    def test_gamma_a_invariant_and_gamma_b(self):
        for kappa in (0.0, 0.7, 2.0):
            layers = self.layers(kappa=kappa, zeta=0.0)
            for g in range(5):
                for strikes in (0, 2):
                    d_a, d_b = layers.gamma(self.state(strikes), g)
                    self.assertEqual(d_a, 0.0)  # exact: matching advancement model and zeta = 0
        layers = self.layers(kappa=0.0, zeta=1.2)
        self.assertNotEqual(layers.gamma(self.state(), 0)[0], 0.0)  # runner on third, one out: zeta moves outs

    def test_layers_must_match_the_frozen_terminal_values(self):
        from pitchmdp.game import GameState, terminal_values
        row = {'inning': 6, 'inning_topbot': 'Top', 'outs_when_up': 1, 'bases': 5, 'home_score': 2, 'away_score': 3}
        state = GameState.from_row(row)
        good = SimpleNamespace(inputs=SimpleNamespace(rows={'ctx': row}),
                               we=SimpleNamespace(terminal_tables={'ctx': terminal_values(state, ToyWE(), ToyAdvancement())}))
        kwargs = dict(groups=reg.S5X_GROUPS, x_dist=np.full((5, 2, reg.N_X), 1 / reg.N_X),
                      class_table=synthetic_table(np.random.default_rng(0)), signs=[0] * 5, signs_prime=[0] * 5,
                      kappa=0.0, zeta=0.0)
        osx.S5XLayers.from_components(good, {'we': ToyWE(), 'advancement': ToyAdvancement()}, **kwargs)

        class Other(ToyAdvancement):
            def distribution(self, state, event):
                if event == 'triple':
                    from pitchmdp.game import apply_terminal
                    return [(.5, apply_terminal(state, 'triple')), (.5, apply_terminal(state, 'home_run'))]
                return super().distribution(state, event)
        with self.assertRaisesRegex(IntegrityError, 'frozen terminal values'):
            osx.S5XLayers.from_components(good, {'we': ToyWE(), 'advancement': Other()}, **kwargs)


class S5XRunTests(unittest.TestCase):
    """End to end on the existing synthetic candidate runtime (fake G0 + WE, test_policy_validation)."""
    @classmethod
    def setUpClass(cls):
        import test_policy_validation as tpv
        cls.tpv = tpv

    def fixture(self, name):
        fx = self.tpv.SemiSyntheticTests('test_v2_matches_world_truth_and_is_not_vacuous')
        fx.setUp()
        runtime, components, starts = fx.world(name)
        rng = np.random.default_rng(0)
        layers = osx.S5XLayers.from_components(
            components, fx.syn.game_values, groups=reg.S5X_GROUPS, x_dist=rng.dirichlet(np.ones(reg.N_X) * .5, size=(5, 2)),
            class_table=synthetic_table(rng), signs=[1, -1, 1, 0, 0], signs_prime=[0] * 5, kappa=0.3, zeta=0.0)
        return fx, runtime, components, starts, layers

    def test_run_end_to_end(self):
        from pitchmdp.rollout_policy import RowBudget
        fx, runtime, components, starts, layers = self.fixture('s5x-run')
        law = lambda s: (runtime.bc.actions, runtime.bc.probabilities(s))
        run = dict(reg.S5X_RUNS[3])
        result = osx.run_s5x(runtime, components, starts, run=run, law=law, layers=layers, cap=4, gamma=True,
                             invariant=True, budget=RowBudget(10 ** 8, seed_count=5))
        self.assertEqual(result['world_seed'], reg.V2_ROLE_SEED ^ 4)
        self.assertEqual(len(result['repetitions']), 6)
        self.assertTrue(all(result['pairs'][p]['valid'] for p in reg.PAIRS))
        gaps = [rep['gaps'][reg.BASELINE] for rep in result['repetitions']]
        self.assertAlmostEqual(result['pairs'][reg.BASELINE]['se'], float(np.std(gaps, ddof=1) / math.sqrt(6)), places=15)
        self.assertEqual(result['invariant']['outcome'], 'PASS')
        self.assertEqual(set(result['s2']), {'R1_a', 'R1_b'})
        self.assertEqual(result['s2']['R1_a']['b_hat'], 0.0)  # zeta = 0 and matching advancement: no R1_a bias
        ledger_pas = {r['pa_id'] for r in runtime.ledger.iter_decisions()}
        self.assertEqual(len(ledger_pas), 6 * result['kept_starts'])
        summary = __import__('run_ope_a2').s5x_summary([result])
        self.assertEqual(summary['s2']['invariant'], 'PASS')
        self.assertEqual(summary['nets']['S1'][reg.BASELINE]['outcome'], 'INVALID')  # runs 1-3 not present

    def test_invariant_violation_is_a_deterministic_failure(self):
        from pitchmdp.rollout_policy import RowBudget
        fx, runtime, components, starts, layers = self.fixture('s5x-invariant')
        law = lambda s: (runtime.bc.actions, runtime.bc.probabilities(s))
        with mock.patch.object(osx.S5XLayers, 'gamma', lambda self, state, g: (1e-9, 0.0)):
            with self.assertRaisesRegex(IntegrityError, 'invariant'):
                osx.run_s5x(runtime, components, starts, run=dict(reg.S5X_RUNS[3]), law=law, layers=layers, cap=4,
                            invariant=True, budget=RowBudget(10 ** 8, seed_count=5))

    def test_single_member_world_predictor(self):
        fx, runtime, components, starts, layers = self.fixture('s5x-x2')
        g0 = components.g0
        state = starts[0][0]
        actions = [a for a, ok in zip(runtime.bc.actions, runtime.reference.support(state)) if ok]
        physical = np.stack([components.inputs.pool(state, a).values[0] for a in actions])
        member = osx.SingleMemberPredictor(g0, g0.models[0].seed)
        p = member([state] * len(actions), actions, physical)
        self.assertEqual(p.shape, (len(actions), 10))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-12)
        self.assertFalse(np.allclose(p, g0([state] * len(actions), actions, physical)))
        with self.assertRaises(IntegrityError):
            osx.SingleMemberPredictor(g0, 99)


if __name__ == '__main__':
    unittest.main()
