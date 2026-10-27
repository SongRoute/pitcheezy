"""SYNTHETIC-ONLY tests for EXP-OPE-RES-01 estimators: the baseline path bit-identical to the registered
policy_estimator on toy ledgers, R2 transform math, self-normalisation, R1 surrogate definitions and
TRAIN guards, the cluster bootstraps, the sealed evaluator's typed public allowlist, G-INT and the
decision labels. No real data, ledger, model or file outside a temp dir is read.
"""
import copy
import json
import math
from pathlib import Path
import struct
import sys
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ope_a2_fixtures import ToyAdvancement, ToyWE, synthetic_ledger, synthetic_pa  # noqa: E402
from pitchmdp import ope_bootstrap as ob  # noqa: E402
from pitchmdp import ope_decision as odc  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp import ope_sealed as osd  # noqa: E402
from pitchmdp import ope_surrogate as osg  # noqa: E402
from pitchmdp import ope_variants as ov  # noqa: E402
from pitchmdp import policy_estimator as est  # noqa: E402
from pitchmdp.game import GameState, terminal_values  # noqa: E402
from pitchmdp.ope_synthetic_a2 import Situation  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

MIN = {'games': 1, 'pa_starts': 1}
INF = math.inf


def bits(value):
    """Exact float bytes of every float in a nested structure (bit-identity, not closeness)."""
    if isinstance(value, dict):
        return [(k, bits(value[k])) for k in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [bits(v) for v in value]
    if isinstance(value, float):
        return struct.pack('<d', value)
    return value


class BaselineIdentityTests(unittest.TestCase):
    def test_variant_path_is_bit_identical_to_the_registered_estimator(self):
        for seed in (0, 1, 2):
            decisions, facts = synthetic_ledger(seed=seed, n_pas=160)
            ref, ref_rows = est.estimate(decisions, facts, draws=200, seed=11, invalid_share_max=1., minimum=MIN,
                                         censoring='l1r')
            for kind, param in (('identity', None), ('shrink', INF), ('switch', INF)):
                got, rows = ov.estimate_variant(decisions, facts, f=ov.transform(kind, param), draws=200, seed=11,
                                                invalid_share_max=1., minimum=MIN)
                with self.subTest(seed=seed, kind=kind):
                    for key in got:
                        self.assertEqual(bits(got[key]), bits(ref[key]), key)
                    self.assertEqual([bits(r) for r in rows], [bits(r) for r in ref_rows])
            resolutions = ref['censoring_rule']['resolutions']
            self.assertTrue({'rho_zero', 'natural_course', 'worst_case_residual', 'shared_end_unknown'} <= set(resolutions))

    def test_per_pa_rows_equal_pa_value(self):
        decisions, facts = synthetic_ledger(seed=4, n_pas=120)
        for pa_id, info, rows in est.by_pa(decisions, facts):
            rows = sorted(rows, key=lambda d: d['decision_index'])
            self.assertEqual(bits(ov.pa_value_variant(rows, info)), bits(est.pa_value(rows, info, 'l1r')))

    def test_all_unit_ratios_make_sn_equal_dr(self):
        rng = np.random.default_rng(3)
        decisions, facts = [], {}
        for i in range(30):
            rows, info = synthetic_pa(rng, f'{i % 5}:{i}', i % 5, 7, 'complete')
            for r in rows:  # candidate = reference = logging law on the full support: every rho = 1
                res = r['result']
                res.update(mask=[True] * 4, reference=list(res['logging']), candidate=list(res['logging']),
                           q_reference=[float(rng.uniform(.3, .7)) for _ in range(4)], rho_reference=1.0, rho_candidate=1.0)
                r['status'] = 'SUPPORTED'
            decisions.extend(rows)
            facts[f'{i % 5}:{i}'] = info
        lines = [ov.pa_line(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        n_bar, valid = ov.sn_normalizers(lines)
        self.assertTrue(valid and np.array_equal(n_bar, np.ones_like(n_bar)))
        u = (1 / n_bar[0], 1 / n_bar[1])
        for line in lines:
            self.assertEqual(bits(ov.line_value(line, None, u)), bits(ov.line_value(line)))


class TransformTests(unittest.TestCase):
    def test_transform_values(self):
        shrink = ov.transform('shrink', 3.0)
        self.assertEqual(shrink(1.0), 0.75)  # lambda = 3 shrinks rho = 1 to 0.75 (design R2_1)
        self.assertEqual(shrink(0.0), 0.0)
        self.assertAlmostEqual(shrink(2.0), 3 * 2 / (4 + 3), places=15)
        switch = ov.transform('switch', 2.0)
        self.assertEqual((switch(2.0), switch(2.0000001), switch(0.5)), (2.0, 0.0, 0.5))
        self.assertIsNone(ov.transform('shrink', INF))
        self.assertIsNone(ov.transform('switch', INF))
        with self.assertRaises(IntegrityError):
            ov.transform('clip', 3.0)
        with self.assertRaises(IntegrityError):
            ov.transform('shrink', -1.0)

    def test_switch_uses_the_model_value_and_drops_later_corrections(self):
        steps = ((.40, .30, .50, 3.0, 1.0), (.45, .35, .55, 0.5, 1.0))  # (v_c, v_r, q_a, rho_c, rho_r)
        base, slope = ov.affine_t(steps, 0, ov.transform('switch', 2.0))
        self.assertEqual((base, slope), (.40, 0.))  # step 0 switched: V_0 = v_hat(H_0), no later term
        base, slope = ov.affine_t(steps, 0)
        self.assertAlmostEqual(base, .40 + 3.0 * ((.45 + 0.5 * (0 - .55)) - .50), places=15)
        self.assertAlmostEqual(slope, 1.5, places=15)

    def test_sn_coefficients_match_the_recursion(self):
        rng = np.random.default_rng(5)
        steps = tuple((float(rng.uniform(.3, .7)), float(rng.uniform(.3, .7)), float(rng.uniform(.3, .7)),
                       float(rng.uniform(0, 2)), float(rng.uniform(.5, 1.5))) for _ in range(4))
        line = {'steps': steps, 'end': 'complete', 'reward': .6, 'reward_source': 'observed'}
        u = rng.uniform(.8, 1.2, size=6)
        for which in (0, 1):
            coef, w_t, T = ov.sn_coefficients(line, which, 5)
            base, slope = ov.affine_sn(steps, which, u)
            self.assertAlmostEqual(coef @ u, base, places=13)
            self.assertAlmostEqual(w_t * u[T], slope, places=13)
        self.assertEqual(ov.affine_sn(steps, 0, np.ones(6)), ov.affine_t(steps, 0))  # u = 1 is the plain DR, bit for bit

    def test_normalizers_hold_weights_and_detect_zero(self):
        a = {'in_population': True, 'steps': ((0., 0., 0., 2.0, 1.0), (0., 0., 0., .5, 1.0))}
        b = {'in_population': True, 'steps': ((0., 0., 0., 0.0, 1.0),)}  # ends at depth 1 with W = 0
        out = {'in_population': False, 'steps': ((0., 0., 0., 9.0, 9.0),)}
        n_bar, valid = ov.sn_normalizers([a, b, out])
        self.assertTrue(valid)
        np.testing.assert_array_equal(n_bar[0], [1., 1., 0.5])  # (2 + 0)/2 then (1 + 0)/2: b holds 0
        np.testing.assert_array_equal(n_bar[1], [1., 1., 1.])
        zero = {'in_population': True, 'steps': ((0., 0., 0., 0.0, 0.0),)}
        self.assertFalse(ov.sn_normalizers([zero])[1])

    def test_selection_rule_and_ties(self):
        clusters = [g for g in range(10) for _ in range(5)]
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1e-3, 50)
        base -= base.mean()
        by_value = {v: base.copy() for v in reg.LAMBDA_GRID}  # identical deltas: every objective ties -> inf
        chosen, objectives = ov.select_parameter('shrink', by_value, clusters)
        self.assertEqual(chosen, INF)
        s2 = reg.cluster_se(base, clusters) ** 2
        by_value[3.0] = base * 0.9  # B^2 + V = (0.1 * 1.96)^2 s^2 + 0.81 s^2 = 0.848 s^2 < s^2
        chosen, objectives = ov.select_parameter('shrink', by_value, clusters)
        self.assertEqual(chosen, 3.0)
        self.assertAlmostEqual(objectives['shrink:3.0'] / s2, (0.1 * 1.96) ** 2 + 0.81, places=9)
        by_value[3.0] = base * 0.5  # B^2 + V = (0.5 * 1.96)^2 s^2 + 0.25 s^2 > s^2: the bias term dominates
        self.assertEqual(ov.select_parameter('shrink', by_value, clusters)[0], INF)
        by_value[3.0] = np.full(50, np.nan)
        chosen, objectives = ov.select_parameter('shrink', by_value, clusters)
        self.assertEqual(objectives['shrink:3.0'], math.inf)
        self.assertEqual(chosen, INF)


class SurrogateTests(unittest.TestCase):
    def setUp(self):
        self.we, self.adv = ToyWE(), ToyAdvancement()
        self.state = GameState(7, 'Bot', 1, 5, 3, 4)  # runners on first and third, one out

    def test_r1a_uses_terminal_values_and_passes_other_ends_through(self):
        tv = terminal_values(self.state, self.we, self.adv)
        for event, cls in (('strikeout', 'strikeout'), ('walk', 'walk'), ('intent_walk', 'walk'),
                           ('hit_by_pitch', 'hbp'), ('home_run', 'home_run'), ('sac_fly', 'out'),
                           ('grounded_into_double_play', 'double_play')):
            self.assertEqual(osg.r1a_value(tv, event), (tv[cls], 'class'))
        for event in ('field_error', 'fielders_choice', 'caught_stealing_2b', 'strikeout_double_play', None):
            self.assertEqual(osg.r1a_value(tv, event), (None, 'pass_through'))

    def test_surrogate_rewards_paths_and_facts_unchanged(self):
        table = np.full((reg.N_CELLS, 6), 1 / 6)
        table[0::2, 5] = 0
        table[0::2] /= table[0::2].sum(axis=1, keepdims=True)
        row = {'inning': 7, 'inning_topbot': 'Bot', 'outs_when_up': 1, 'bases': 5, 'home_score': 3, 'away_score': 4,
               'stand': 'R'}
        rows = {'bip': {**row, 'events': 'single', 'description': 'hit_into_play', 'launch_speed': 101.0,
                        'launch_angle': 14.0, 'hc_x': 80.0, 'hc_y': 120.0},
                'bip_missing': {**row, 'events': 'single', 'description': 'hit_into_play', 'launch_speed': np.nan,
                                'launch_angle': 14.0, 'hc_x': 80.0, 'hc_y': 120.0},
                'k': {**row, 'events': 'strikeout', 'description': 'swinging_strike'},
                'odd': {**row, 'events': 'caught_stealing_home', 'description': 'ball'}}
        facts = {k: {'reward': .55} for k in rows}
        facts['censored'] = {'reward': None}
        before = copy.deepcopy(facts)
        terminal = osg.TerminalValues(self.we, self.adv)
        out = osg.surrogate_rewards(facts, rows, terminal, table)
        self.assertEqual(facts, before)  # the input facts are never modified
        self.assertNotIn('censored', out)  # no surrogate without an observed R1-0 reward (P unchanged)
        tv = terminal_values(self.state, self.we, self.adv)
        cell = int(osg.quality_cells([101.], [14.], [80.], [120.], ['R'], [5], [1])[0])
        self.assertEqual(out['bip']['R1_b'], (sum(table[cell][k] * tv[e] for k, e in enumerate(reg.BIP_CLASSES)),
                                              'bip_quality'))
        self.assertEqual(out['bip']['R1_a'], (tv['single'], 'class'))
        self.assertEqual(out['bip_missing']['R1_b'], (tv['single'], 'bip_missing_ev_la_fallback'))
        self.assertEqual(out['k']['R1_b'], (tv['strikeout'], 'non_bip_class'))
        self.assertEqual(out['odd']['R1_a'], (.55, 'pass_through'))
        shares = osg.path_shares(out)
        self.assertEqual((shares['pas_with_reward'], shares['bip_with_ev_la'], shares['pass_through']), (4, 1, 1))

    def test_bins_spray_and_dp(self):
        self.assertEqual(osg._edge_bin([59.9, 60.0, 109.99, 110.0, np.nan], reg.EV_EDGES).tolist(), [0, 1, 10, 11, -1])
        self.assertEqual(osg._edge_bin([-21, -20, 49.9, 50], reg.LA_EDGES).tolist(), [0, 1, 8, 9])
        phi = osg.spray_angle([60., 60., 125.42, 125.42, np.nan], [130., 130., 100., 100., 100.], ['R', 'L', 'R', None, 'R'])
        self.assertGreater(phi[0], 15)  # left field is pull for a right-handed batter
        self.assertLess(phi[1], -15)  # and opposite field for a left-handed batter
        self.assertEqual(osg.spray_bin(phi).tolist(), [0, 2, 1, 3, 3])
        self.assertEqual(osg.dp_flag([1, 1, 2, 5], [1, 2, 0, 0]).tolist(), [True, False, False, True])
        cells = osg.quality_cells([100.], [20.], [60.], [130.], ['R'], [1], [0])
        ev, la, spray, dp = (int(v[0]) for v in osg.cell_parts(cells))
        self.assertEqual((ev, la, spray, dp), (7, 5, 0, 1))
        self.assertEqual(reg.N_CELLS, 960)

    def test_hierarchical_shrinkage_by_hand(self):
        cells = np.array([osg.quality_cells([100.], [20.], [60.], [130.], ['R'], [1], [0])[0]] * 3 +
                         [osg.quality_cells([100.], [20.], [190.], [130.], ['R'], [1], [0])[0]] +
                         [osg.quality_cells([70.], [-15.], [100.], [130.], ['L'], [0], [2])[0]])  # one dp = 0 row
        classes = np.array([1, 1, 0, 2, 0])
        counts = osg.class_counts(cells, classes)
        m = 5.0
        table = osg.fit_table(counts, m)
        root = np.array([1, 2, 1, 0, 0, 0]) / 4  # dp = 1 stratum only
        la_level = (np.array([1, 2, 1, 0, 0, 0]) + m * root) / (4 + m)
        ev_level = (np.array([1, 2, 1, 0, 0, 0]) + m * la_level) / (4 + m)
        leaf = (np.array([1, 2, 0, 0, 0, 0]) + m * ev_level) / (3 + m)
        np.testing.assert_allclose(table[cells[0]], leaf, rtol=0, atol=1e-15)
        self.assertTrue((table[0::2, 5] == 0).all())
        with self.assertRaises(IntegrityError):
            osg.fit_table(osg.class_counts(np.array([0]), np.array([5])), m)  # double play at dp = 0

    def test_train_guard_and_mapping(self):
        with self.assertRaisesRegex(IntegrityError, '2026'):
            osg.guard_train(['2024-06-01', '2026-04-01'])
        with self.assertRaisesRegex(IntegrityError, 'outside TRAIN'):
            osg.guard_train(['2024-06-01', '2025-05-02'])
        with self.assertRaisesRegex(IntegrityError, 'outside TRAIN'):
            osg.guard_train(['2023-05-14'])
        frame = pd.DataFrame({
            'game_date': ['2024-06-01'] * 5, 'description': ['hit_into_play'] * 4 + ['ball'],
            'events': ['field_error', 'triple_play', 'grounded_into_double_play', 'sac_bunt_double_play', 'walk'],
            'launch_speed': [90.] * 5, 'launch_angle': [10.] * 5, 'hc_x': [100.] * 5, 'hc_y': [120.] * 5,
            'stand': ['R'] * 5, 'bases': [0, 1, 0, 1, 0], 'outs_when_up': [0, 0, 1, 0, 0]})
        cells, classes, index, report = osg.train_bip_rows(frame)
        self.assertEqual(index.tolist(), [0, 1, 2])  # sac_bunt_double_play is not in the TRAIN mapping
        self.assertEqual([reg.BIP_CLASSES[c] for c in classes], ['single', 'double_play', 'out'])  # dp = 0 GIDP -> out
        self.assertEqual(report['dp_ineligible_double_play_to_out'], 1)
        self.assertEqual(report['bip_unmapped_event'], 1)

    def test_m_selection_by_cv_and_ties_to_larger_m(self):
        rng = np.random.default_rng(0)
        cells = rng.integers(0, reg.N_CELLS, 6000)
        classes = rng.choice(6, 6000, p=[.6, .2, .1, .02, .05, .03])
        classes[(cells % 2 == 0) & (classes == 5)] = 0
        fit = osg.fit_r1b(cells, classes, rng.integers(0, 2, 6000))
        self.assertIn(fit['m'], reg.M_SMOOTH_GRID)
        self.assertEqual(fit['m'], reg.select_minimizer(reg.M_SMOOTH_GRID, [fit['cv_log_loss'][str(m)] for m in reg.M_SMOOTH_GRID]))
        fixed = osg.fit_r1b(cells, classes, rng.integers(0, 2, 6000), m=20)
        self.assertIsNone(fixed['cv_log_loss'])  # the smoke never computes a log-loss
        self.assertEqual(len(osg.table_sha256(fixed['table'], 20)), 64)

    def test_gamma_a_is_zero_when_zeta_is_zero(self):
        sit = Situation.from_models(GameState(5, 'Top', 1, 4, 2, 2), self.we, self.adv)  # runner on third, 1 out
        for sign in (-1, 0, 1):
            ev = sit.class_values(0.0, sign)
            np.testing.assert_array_equal(ev, [sit.tv[e] for e in reg.BIP_CLASSES])  # exact, any kappa
        self.assertNotEqual(sit.class_values(0.7, 1)[0], sit.tv['out'])  # zeta tilts the out value
        self.assertEqual(sit.tv, {e: v for e, v in terminal_values(sit.state, self.we, self.adv).items()})


class BootstrapTests(unittest.TestCase):
    def test_game_resampler_matches_the_registered_game_bootstrap_draws(self):
        games = [3, 1, 2, 10, 1, 3]
        r = ob.Resampler(games, unit='game', draws=50, seed=7, minimum_clusters=1, minimum_pa_starts=1)
        self.assertEqual(r.order, sorted(set(games), key=str))
        sample = np.random.default_rng(7).integers(len(r.order), size=(50, len(r.order)))
        expected = np.stack([np.bincount(row, minlength=len(r.order)) for row in sample]).astype(float)
        np.testing.assert_array_equal(r.m, expected)

    def test_pitcher_equals_game_when_each_pitcher_has_one_game(self):
        decisions, facts = synthetic_ledger(seed=9, n_pas=150, n_games=15)
        lines = []
        for pa_id, info, rows in est.by_pa(decisions, facts):
            line = ov.pa_line(sorted(rows, key=lambda d: d['decision_index']), info)
            lines.append(line)
        e0 = [l for l in lines if l['in_population']]
        point = np.array([ov.is_point(l) for l in e0])
        keys = [l['game'] for l in e0]
        rows = [ov.row_of(l) for l in e0]
        pairs = [ob.fixed_pair(reg.BASELINE, rows, point), ob.sn_pair('sn', e0, point)]
        out = {}
        for unit, k in (('game', keys), ('pitcher', [f'p{g}' for g in keys])):
            r = ob.Resampler(k, unit=unit, draws=300, seed=5, minimum_clusters=1, minimum_pa_starts=1)
            stats, valid = ob.run_pairs(r, pairs, point, reg.BASELINE)
            out[unit] = ob.summarize(stats, valid, r)
        for name in (reg.BASELINE, 'sn'):
            self.assertEqual(out['game'][name]['mean_p']['half_width'], out['pitcher'][name]['mean_p']['half_width'])
        r1 = ob.Resampler(keys, unit='game', draws=300, seed=5, minimum_clusters=1, minimum_pa_starts=1)
        r2 = ob.Resampler(keys, unit='game', draws=300, seed=5, minimum_clusters=1, minimum_pa_starts=1)
        np.testing.assert_array_equal(r1.m, r2.m)  # seed reproducible

    def test_sn_normalizers_are_recomputed_per_replicate(self):
        decisions, facts = synthetic_ledger(seed=2, n_pas=200, n_games=20, weight_scale=1.5)
        e0 = [ov.pa_line(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        e0 = [l for l in e0 if l['in_population']]
        point = np.array([ov.is_point(l) for l in e0])
        pair = ob.sn_pair('sn', e0, point)
        r = ob.Resampler([l['game'] for l in e0], unit='game', draws=200, seed=3, minimum_clusters=1, minimum_pa_starts=1)
        stats, valid = ob.run_pairs(r, [ob.fixed_pair(reg.BASELINE, [ov.row_of(l) for l in e0], point), pair], point,
                                    reg.BASELINE)
        n_bar, ok = ov.sn_normalizers(e0)
        self.assertTrue(ok)
        fixed_deltas, _, _ = ob.sn_values(pair, 1 / n_bar)
        p_rows = np.flatnonzero(point)
        fixed = (r.m @ r.sums(fixed_deltas[p_rows], p_rows)) / (r.m @ r.sums(np.ones(len(p_rows)), p_rows))
        self.assertFalse(np.allclose(stats['sn']['mean_p'], fixed))  # not the fixed full-sample normaliser
        m0 = r.m[0][r.row_cluster]  # replicate 0 by hand: normalisers from the resampled E0 PAs only
        held = pair.sn['held']
        nb = np.array([(m0 @ held[k]) / m0.sum() for k in (0, 1)])
        d0, _, _ = ob.sn_values(pair, 1 / nb)
        self.assertAlmostEqual(stats['sn']['mean_p'][0], float(m0[p_rows] @ d0[p_rows] / m0[p_rows].sum()), places=12)
        full, _, _ = ob.sn_values(pair, 1 / n_bar)
        for i in np.flatnonzero(point)[:20]:  # the linear form equals the SN recursion per PA
            line = e0[i]
            value = ov.line_value(line, None, (1 / n_bar[0], 1 / n_bar[1]))
            self.assertAlmostEqual(full[i], value['delta'], places=12)

    def test_validity_minimums_and_invalid_share(self):
        r = ob.Resampler(['a', 'b'], unit='game', draws=10, seed=1, minimum_clusters=30, minimum_pa_starts=50)
        self.assertFalse(r.eligible)
        values = np.zeros(100)
        valid = np.ones(100, dtype=bool)
        valid[:6] = False
        self.assertIsNone(ob.interval(values, valid, 100)['ci95'])  # 6% invalid > 5%
        out = ob.interval(np.linspace(0, 1, 100), np.ones(100, dtype=bool), 100)
        self.assertTrue(out['valid'] and out['half_width'] > 0)
        bad = np.full(100, np.inf)
        self.assertFalse(ob.interval(bad, np.ones(100, dtype=bool), 100)['valid'])

    def test_decomposition_adds_up(self):
        decisions, facts = synthetic_ledger(seed=6, n_pas=200, n_games=20)
        e0 = [ov.pa_line(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        e0 = [l for l in e0 if l['in_population']]
        point = np.array([ov.is_point(l) for l in e0])
        r = ob.Resampler([l['game'] for l in e0], unit='game', draws=400, seed=8, minimum_clusters=1, minimum_pa_starts=1)
        stats, valid = ob.run_pairs(r, [ob.fixed_pair(reg.BASELINE, [ov.row_of(l) for l in e0], point)], point, reg.BASELINE)
        dec = ob.summarize(stats, valid, r)[reg.BASELINE]['decomposition']
        self.assertAlmostEqual(dec['var_lower_endpoint'], dec['var_point'] + dec['var_censored'] + 2 * dec['cov_point_censored'],
                               places=12)


def _evaluate(seed=5, gint=None, **kw):
    decisions, facts = synthetic_ledger(seed=seed, n_pas=300, n_games=20, n_pitchers=12)
    rng = np.random.default_rng(1)
    sur = {pa: {'R1_a': (float(np.clip(f['reward'] + rng.normal(0, .02), 0, 1)), 'class'),
                'R1_b': (float(np.clip(f['reward'] + rng.normal(0, .01), 0, 1)), 'bip_quality'), 'spray_missing': False}
           for pa, f in facts.items() if f.get('reward') is not None}
    return osd.rehearsal_evaluate(lambda: iter(decisions), facts, surrogates=sur, gint_reference=gint,
                                  train_residuals={'hit': {'FB': {'n': 10, 'mean_residual': -0.01, 'se': 0.005}}},
                                  r1_paths=osg.path_shares(sur), r3_status='NOT_RUN:R3', draws=200, minimum_games=5,
                                  minimum_pitchers=4, minimum_pa_starts=10, placebos=4, **kw), decisions, facts


class BlindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.public, cls.sealed, _), cls.decisions, cls.facts = _evaluate()

    def test_public_file_holds_only_allowlisted_types(self):
        public_bytes, sealed_bytes = osd.seal(self.public, self.sealed)
        public = json.loads(public_bytes)
        self.assertEqual(public['sealed_sha256'], osd.sha256_bytes(sealed_bytes))
        self.assertTrue(osd.validate_public(public))
        text = json.dumps(public)
        for word in ('selected', 'center', '"delta"', 'shift', 'objective', 'placebo_delta'):
            self.assertNotIn(word, text)
        for pair in reg.PAIRS:
            self.assertEqual(set(public['pairs'][pair]), {'H_point', 'SE', 'endpoint_half_width', 'valid', 'invalid_replicates'})
            self.assertTrue(all(v is None or v >= 0 for v in public['pairs'][pair]['H_point'].values()))
        self.assertIn('selected', json.dumps(self.sealed['selection']))  # the selection is sealed, not lost

    def test_validator_rejects_leaks(self):
        good = json.loads(osd.seal(self.public, self.sealed)[0])
        leaks = {
            'signed width': lambda d: d['pairs'][reg.BASELINE]['H_point'].__setitem__('game', -1e-4),
            'effect center key': lambda d: d['pairs'][reg.BASELINE].__setitem__('delta', 1e-4),
            'shift endpoints': lambda d: d['pairs']['R1_axR2_0'].__setitem__('shift_ci95', [-1e-4, 2e-4]),
            'CI under a width': lambda d: d['pairs']['R1_axR2_0']['SE'].__setitem__('game', [-1e-4, 2e-4]),
            'selected lambda': lambda d: d['R5']['b'].__setitem__('selected_lambda', 3.0),
            'S5 value': lambda d: d['S5']['R1_axR2_0'].__setitem__('game', 0.0003),
            'unknown section': lambda d: d.__setitem__('centers', {}),
            'signed diagnostic outside g/i': lambda d: d['R5']['a'].__setitem__('mean_shift', -0.2),
            'negative variance': lambda d: d['r1_variance'].__setitem__('R1_0', -1.0),
        }
        for name, mutate in leaks.items():
            doc = copy.deepcopy(good)
            mutate(doc)
            with self.subTest(name), self.assertRaises(IntegrityError):
                osd.validate_public(doc)
        allowed = copy.deepcopy(good)
        allowed['R5']['g']['candidate']['mean_pa_end_weight_ci95'] = [-0.5, 1.2]  # reward-free weight CI: allowed
        allowed['R5']['i']['train_residuals']['hit']['FB']['mean_residual'] = -0.03  # TRAIN-only signed: allowed
        allowed['R5']['f'][reg.BASELINE]['game']['cov_point_censored'] = -1e-9  # a covariance may be negative
        self.assertTrue(osd.validate_public(allowed))

    def test_unblind_needs_the_committed_record_and_matching_seals(self):
        public_bytes, sealed_bytes = osd.seal(self.public, self.sealed)
        sha = json.loads(public_bytes)['sealed_sha256']
        record = b'{"label": "NO_PRECISION_GAIN"}\n'
        with self.assertRaisesRegex(IntegrityError, 'committed'):
            osd.unblind(sealed_bytes, sha, record, osd.sha256_bytes(record), committed=False)
        with self.assertRaisesRegex(IntegrityError, 'decision record'):
            osd.unblind(sealed_bytes, sha, record, 'f' * 64, committed=True)
        with self.assertRaisesRegex(IntegrityError, 'sealed file'):
            osd.unblind(sealed_bytes + b' ', sha, record, osd.sha256_bytes(record), committed=True)
        opened = osd.unblind(sealed_bytes, sha, record, osd.sha256_bytes(record), committed=True)
        self.assertIn('selection', opened)

    def test_s5_and_inf_identity_are_reported(self):
        self.assertEqual(self.public['integrity']['inf_variants_reproduce_baseline'], 'PASS')
        self.assertEqual(self.public['S5'][reg.BASELINE], {'game': 'BASELINE', 'pitcher': 'BASELINE'})
        self.assertTrue(all(v in ('PASS', 'FAIL', 'INVALID') for p in reg.TOTAL_ORDER for v in self.public['S5'][p].values()))
        self.assertEqual(set(self.sealed['placebo']), {'placebo_delta_hat', 'tau_p', 'placebo_bootstrap_se', 'placebo_sd'})
        self.assertEqual(len(self.sealed['placebo']['placebo_delta_hat']), 4)


class GintTests(unittest.TestCase):
    def test_gint_reproduction_and_failure_stops_before_new_arms(self):
        decisions, facts = synthetic_ledger(seed=5, n_pas=300, n_games=20, n_pitchers=12)
        result, _ = est.estimate(decisions, facts, draws=200, seed=reg.GAME_BOOTSTRAP_SEED, invalid_share_max=0.05,
                                 minimum={'games': 5, 'pa_starts': 10}, censoring='l1r')
        rule, boot = result['censoring_rule'], result['bootstrap']
        reference = {'l1r_delta_bounds': result['layers']['L1_start_population']['delta_bounds'],
                     'l1r_lower_endpoint_ci95': boot['L1_lower_endpoint_ci95'], 'l2_ci95': boot['L2_conditional_ci95'],
                     'l2_delta_mean': result['layers']['L2_complete_conditional']['delta_mean'],
                     'worst_case_residual_share_s': rule['worst_case_residual_share'], 'resolutions': rule['resolutions'],
                     'ess_all_e0_candidate': result['ess_all_e0']['candidate']}
        self.assertEqual(osd.gint_compare(result, reference), [])
        bad = copy.deepcopy(reference)
        bad['l2_ci95'] = [bad['l2_ci95'][0] + 2e-12, bad['l2_ci95'][1]]
        self.assertEqual(osd.gint_compare(result, bad), ['l2_ci95'])
        ok = copy.deepcopy(reference)
        ok['l2_ci95'] = [ok['l2_ci95'][0] + 5e-13, ok['l2_ci95'][1]]
        self.assertEqual(osd.gint_compare(result, ok), [])
        (_, _, integrity), _, _ = _evaluate(gint=reference)
        self.assertEqual(integrity['gint_failures'], [])
        with self.assertRaisesRegex(IntegrityError, 'G-INT failed'):
            _evaluate(gint=bad)
        config = json.loads((reg.REPO / reg.CONFIG_PATH).read_text())
        quoted = osd.registered_gint_reference(config)
        self.assertEqual(quoted['resolutions'], {'rho_zero': 401, 'natural_course': 55, 'worst_case_residual': 6,
                                                 'shared_end_unknown': 1})
        self.assertEqual(quoted['l1r_delta_bounds'], [-0.0006478399045356745, 0.000539688464028487])


def _decision_inputs(h=None, s5='PASS', net='PASS', tab='PASS', s2='PASS', valid=True):
    h = h or {}
    public = {'pairs': {}, 'S5': {}, 'R5': {'h': {'flag': None}}}
    for pair in reg.PAIRS:
        width = h.get(pair, 1.0)
        public['pairs'][pair] = {'H_point': {'game': width, 'pitcher': width}, 'valid': {'game': valid, 'pitcher': valid}}
        public['S5'][pair] = {'game': 'BASELINE' if pair == reg.BASELINE else s5,
                              'pitcher': 'BASELINE' if pair == reg.BASELINE else s5}
    tab_doc = {'acceptance': {c['id']: {p: {'outcome': tab} for p in reg.PAIRS} for c in reg.TAB_ACCEPTANCE}}
    s5x = {'nets': {n: {p: {'outcome': net} for p in reg.PAIRS} for n in ('S1', 'S4')},
           's2': {'R1_a': {'2*kappa_p': {'outcome': s2}, '2*zeta_p': {'outcome': s2}},
                  'R1_b': {'2*kappa_p': {'outcome': s2}, '2*zeta_p': {'outcome': s2}}, 'invariant': 'PASS'}}
    return public, tab_doc, s5x


class DecisionTests(unittest.TestCase):
    def test_label_precedence_and_selection(self):
        decide = lambda *a, **k: odc.decide(*a, r3_status='NOT_RUN:R3', **k)
        public, tab, s5x = _decision_inputs()
        self.assertEqual(decide(public, tab, s5x, integrity_failures=['A2-S1'])['label'], 'FAILED_INTEGRITY')
        public['pairs'][reg.BASELINE]['valid']['pitcher'] = False
        self.assertEqual(decide(public, tab, s5x)['label'], 'NOT_EVALUABLE')
        self.assertEqual(decide(*_decision_inputs())['label'], 'NO_PRECISION_GAIN')  # equal widths: no 30% cut
        gains = {'R1_bxR2_0': 0.69, 'R1_axR2_3': 0.68, 'R1_0xR2_1': 0.60}
        out = decide(*_decision_inputs(h=gains))
        self.assertEqual(out['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_1')  # 0.68/0.69 are outside the 2% window
        gains = {'R1_bxR2_0': 0.61, 'R1_axR2_3': 0.60, 'R1_0xR2_1': 0.605}
        self.assertEqual(decide(*_decision_inputs(h=gains))['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_bxR2_0')  # total order
        self.assertEqual(decide(*_decision_inputs(h=gains, tab='INCONCLUSIVE'))['label'],
                         'BIAS_NOT_CERTIFIED:R1_bxR2_0,R1_axR2_3,R1_0xR2_1')
        self.assertEqual(decide(*_decision_inputs(h=gains, s5='FAIL'))['label'][:19], 'BIAS_NOT_CERTIFIED:')
        self.assertEqual(decide(*_decision_inputs(h={'R1_0xR2_3': 0.7000001}))['label'], 'NO_PRECISION_GAIN')
        self.assertEqual(decide(*_decision_inputs(h={'R1_0xR2_3': 0.7}))['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_3')
        public, tab, s5x = _decision_inputs(h={'R1_axR2_0': 0.5, 'R1_0xR2_3': 0.5})
        s5x['s2']['R1_a']['2*zeta_p']['outcome'] = 'FAIL'  # S-2 binds only the R1_a / R1_b pairs
        self.assertEqual(decide(public, tab, s5x)['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_3')
        public, tab, s5x = _decision_inputs(h={'R1_axR2_0': 0.5})
        s5x['nets']['S1'][reg.BASELINE]['outcome'] = 'FAIL'  # the baseline's shared net binds every pair
        self.assertEqual(decide(public, tab, s5x)['label'], 'BIAS_NOT_CERTIFIED:R1_axR2_0')
        public['R5']['h']['flag'] = 'SE_CALIBRATION_WARNING'
        self.assertEqual(decide(public, tab, s5x)['flags'], ['SE_CALIBRATION_WARNING'])


if __name__ == '__main__':
    unittest.main()
