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
                'odd': {**row, 'events': 'caught_stealing_home', 'description': 'ball'},
                'k_with_ev': {**row, 'events': 'strikeout', 'description': 'foul_tip', 'launch_speed': 80.0,
                              'launch_angle': 5.0, 'hc_x': np.nan, 'hc_y': np.nan},
                'bip_no_spray': {**row, 'events': 'single', 'description': 'hit_into_play', 'launch_speed': 101.0,
                                 'launch_angle': 14.0, 'hc_x': np.nan, 'hc_y': np.nan}}
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
        self.assertEqual((shares['pas_with_reward'], shares['bip_with_ev_la'], shares['pass_through']), (6, 2, 1))
        # review #14: a spray-missing count only among PAs on the bip_quality path (a non-BIP row with EV/LA is not one)
        self.assertEqual(out['k_with_ev']['R1_b'][1], 'non_bip_class')
        self.assertFalse(out['k_with_ev']['spray_missing'])
        self.assertTrue(out['bip_no_spray']['spray_missing'])
        self.assertEqual(shares['spray_missing_among_bip'], 1)

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
        fixed_deltas, _, _, _ = ob.sn_values(pair, 1 / n_bar)
        p_rows = np.flatnonzero(point)
        fixed = (r.m @ r.sums(fixed_deltas[p_rows], p_rows)) / (r.m @ r.sums(np.ones(len(p_rows)), p_rows))
        self.assertFalse(np.allclose(stats['sn']['mean_p'], fixed))  # not the fixed full-sample normaliser
        m0 = r.m[0][r.row_cluster]  # replicate 0 by hand: normalisers from the resampled E0 PAs only
        held = pair.sn['held']
        nb = np.array([(m0 @ held[k]) / m0.sum() for k in (0, 1)])
        d0, _, _, _ = ob.sn_values(pair, 1 / nb)
        self.assertAlmostEqual(stats['sn']['mean_p'][0], float(m0[p_rows] @ d0[p_rows] / m0[p_rows].sum()), places=12)
        full, _, _, _ = ob.sn_values(pair, 1 / n_bar)
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


GATE = {'draws': 200, 'minimum_games': 5, 'minimum_pa_starts': 10}
CENSUS = {'train_rows': 1000, 'bip_rows': 200, 'bip_missing_ev_or_la': 9, 'bip_unmapped_event': 3, 'bip_used': 188,
          'dp_ineligible_double_play_to_out': 2, 'spray_missing_among_used': 4,
          'mapping_counts': {'field_out': 120, 'single': 50, 'double': 18}}  # synthetic A2-S0 census shape


def _evaluate(seed=5, references=None, extra=(), **kw):
    """G-INT (gint_gate) first, then the sealed evaluator, on a synthetic ledger (+ ``extra`` PAs)."""
    decisions, facts = synthetic_ledger(seed=seed, n_pas=300, n_games=20, n_pitchers=12)
    for rows, pa_id, info in extra:
        decisions.extend(rows)
        facts[pa_id] = info
    rng = np.random.default_rng(1)
    sur = {pa: {'R1_a': (float(np.clip(f['reward'] + rng.normal(0, .02), 0, 1)), 'class'),
                'R1_b': (float(np.clip(f['reward'] + rng.normal(0, .01), 0, 1)), 'bip_quality'), 'spray_missing': False}
           for pa, f in facts.items() if f.get('reward') is not None}
    stream = lambda: iter(decisions)
    result0, rows0, gint = osd.gint_gate(stream, facts, references=references or {}, **GATE)
    return osd.rehearsal_evaluate(stream, facts, baseline=(result0, rows0), surrogates=sur, gint=gint,
                                  train_residuals={'hit': {'FB': {'n': 10, 'mean_residual': -0.01, 'se': 0.005}}},
                                  r1_paths={'rehearsal': osg.path_shares(sur), 'train_census': CENSUS},
                                  r3_status='NOT_RUN:R3', minimum_pitchers=4, placebos=4, **GATE, **kw), decisions, facts


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
            # the reviewer's probes (review #7): an unlisted metric name never passes, whatever its type
            'R5.g.effect_ci95': lambda d: d['R5']['g'].__setitem__('effect_ci95', [-0.0001, 0.0002]),
            'R5.a.mean_shift': lambda d: d['R5']['a'].__setitem__('mean_shift', 0.0002),
            'R5.g.lambda': lambda d: d['R5']['g'].__setitem__('lambda', 30),
            'unlisted R5.c metric': lambda d: d['R5']['c']['weights']['candidate'].__setitem__('q999', 2.0),
            'unlisted grid value': lambda d: d['R5']['c']['switch_trigger_rate'].__setitem__('7.0', {'candidate': 0.1}),
            'unlisted policy': lambda d: d['R5']['g'].__setitem__('planner', {'mean_rho_logged': 1.0}),
            'R5-g indicator not a bool': lambda d: d['R5']['g']['candidate'].__setitem__('mean_rho_logged_ci_excludes_1',
                                                                                        0.0),
            'unlisted top-level key': lambda d: d.__setitem__('note', 'x'),
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
        for policy in ('candidate', 'reference'):  # review #14: the typed CI-excludes-1 indicators are public
            g = good['R5']['g'][policy]
            self.assertIsInstance(g['mean_rho_logged_ci_excludes_1'], bool)
            self.assertIsInstance(g['mean_pa_end_weight_ci_excludes_1'], bool)
            lo, hi = g['mean_rho_logged_ci95']
            self.assertEqual(g['mean_rho_logged_ci_excludes_1'], not (lo <= 1 <= hi))

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
        self.assertTrue(all(v in ('PASS', 'FAIL', 'INCONCLUSIVE', 'INVALID')
                            for p in reg.TOTAL_ORDER for v in self.public['S5'][p].values()))
        # review round 2 #17: a pair that selected lambda = inf or M = inf equals the baseline, so its shift interval
        # is [0, 0] in both bootstraps: INCONCLUSIVE, never PASS
        identical = [name for name, entry in self.sealed['selection'].items() if entry['selected'] == 'inf'
                     and name.startswith('R1_0')]
        self.assertTrue(identical)
        for pair in identical:
            for b in ('game', 'pitcher'):
                ci = self.sealed['pairs'][pair]['bootstrap'][b]['shift_p_ci95']
                self.assertEqual(ci, [0.0, 0.0])
                self.assertEqual(self.public['S5'][pair][b], 'INCONCLUSIVE')
        self.assertEqual(set(self.sealed['placebo']) - {'population'},
                         {'placebo_delta_hat', 'tau_p', 'placebo_bootstrap_se', 'placebo_sd'})
        self.assertEqual(len(self.sealed['placebo']['placebo_delta_hat']), 4)

    def test_missing_descriptive_output_is_a_stage_integrity_failure(self):
        # PM decision (review #10): every required descriptive output, else IntegrityError (never silent)
        for path in (('R5', 'f'), ('r1_variance',), ('validity',)):
            public = copy.deepcopy(self.public)
            node = public
            for part in path[:-1]:
                node = node[part]
            del node[path[-1]]
            with self.subTest(path), self.assertRaisesRegex(IntegrityError, 'descriptive output missing'):
                osd.require_descriptive(public, self.sealed)
        for key in ('pigeonhole', 'gint', 'p_nonpoint', 'grid_points'):
            sealed = dict(self.sealed)
            del sealed[key]
            with self.subTest(key), self.assertRaisesRegex(IntegrityError, 'sealed descriptive output missing'):
                osd.require_descriptive(self.public, sealed)
        sealed = copy.deepcopy(self.sealed)
        del sealed['pairs']['R1_axR2_1']['shift_center']
        with self.assertRaisesRegex(IntegrityError, 'paired shift missing'):
            osd.require_descriptive(self.public, sealed)
        with self.assertRaisesRegex(IntegrityError, 'G-INT record is required'):
            osd.rehearsal_evaluate(lambda: iter(self.decisions), self.facts, baseline=(None, None), surrogates={},
                                   train_residuals=None, r1_paths={}, r3_status='NOT_RUN:R3', gint=None)


class GintTests(unittest.TestCase):
    def setUp(self):
        self.decisions, self.facts = synthetic_ledger(seed=5, n_pas=300, n_games=20, n_pitchers=12)
        self.stream = lambda: iter(self.decisions)
        result, _ = est.estimate(self.decisions, self.facts, draws=200, seed=reg.GAME_BOOTSTRAP_SEED,
                                 invalid_share_max=reg.INVALID_SHARE_MAX, minimum={'games': 5, 'pa_starts': 10},
                                 censoring='l1r')
        self.recorded = osd.recorded_gint_reference({**result, 'games': [1000, 1001]})  # the S6 dr.json layout

    def test_every_listed_value_is_compared_against_the_pinned_record(self):
        self.assertEqual(set(self.recorded), set(osd.GINT_KEYS))
        self.assertTrue(all(self.recorded[k] is not None for k in osd.GINT_KEYS))
        quoted = {k: v for k, v in self.recorded.items() if k not in ('l1r_upper_endpoint_ci95', 'ess_all_e0_reference')}
        _, _, record = osd.gint_gate(self.stream, self.facts, references={'registration_quoted': quoted,
                                                                          's6_dr_a5_record': self.recorded}, **GATE)
        self.assertEqual(record['registration_quoted']['not_comparable'], ['ess_all_e0_reference', 'l1r_upper_endpoint_ci95'])
        self.assertEqual(record['s6_dr_a5_record']['compared'], sorted(osd.GINT_KEYS))
        self.assertEqual(record['s6_dr_a5_record']['not_comparable'], [])
        self.assertEqual(record['listed_keys_never_compared'], [])
        _, _, alone = osd.gint_gate(self.stream, self.facts, references={'registration_quoted': quoted}, **GATE)
        self.assertEqual(alone['listed_keys_never_compared'], ['ess_all_e0_reference', 'l1r_upper_endpoint_ci95'])

    def test_a_mismatch_in_any_listed_value_stops_the_stage(self):
        def perturb(value, eps):
            if isinstance(value, list):
                return [value[0] + eps, *value[1:]]
            if isinstance(value, dict):
                key = sorted(value)[0]
                return {**value, key: value[key] + (1 if isinstance(value[key], int) else eps)}
            return value + eps
        for key in osd.GINT_KEYS:
            bad = {**self.recorded, key: perturb(self.recorded[key], 2e-12)}
            with self.subTest(key), self.assertRaisesRegex(IntegrityError, 'G-INT failed'):
                osd.gint_gate(self.stream, self.facts, references={'s6_dr_a5_record': bad}, **GATE)
        close = {**self.recorded, 'l2_ci95': perturb(self.recorded['l2_ci95'], 5e-13)}
        osd.gint_gate(self.stream, self.facts, references={'s6_dr_a5_record': close}, **GATE)  # within 1e-12

    def test_registration_quotes(self):
        config = json.loads((reg.REPO / reg.CONFIG_PATH).read_text())
        quoted = osd.registered_gint_reference(config)
        self.assertEqual(quoted['resolutions'], {'rho_zero': 401, 'natural_course': 55, 'worst_case_residual': 6,
                                                 'shared_end_unknown': 1})
        self.assertEqual(quoted['l1r_delta_bounds'], [-0.0006478399045356745, 0.000539688464028487])
        self.assertNotIn('l1r_upper_endpoint_ci95', quoted)  # not quoted: compared against the pinned S6 record only


class PointSetTests(unittest.TestCase):
    """PM decision (review #12): P = E0 PAs whose BASELINE paired delta bounds coincide exactly."""
    @staticmethod
    def _decision(pa_id, t, candidate, reference, q, logged=0):
        logging = [0.25, 0.25, 0.25, 0.25]
        return {'pa_id': pa_id, 'decision_index': t, 'pitcher': '501', 'status': 'SUPPORTED', 'balls': 0, 'strikes': t,
                'result': {'mask': [True] * 4, 'logging': logging, 'reference': reference, 'candidate': candidate,
                           'logged_index': logged, 'q_reference': q,
                           'rho_reference': reference[logged] / logging[logged],
                           'rho_candidate': candidate[logged] / logging[logged]}}

    def _shared_end_pa(self, pa_id, steps):
        """A shared-end PA (one unknown end value for both policies) from dyadic probabilities (exact sums)."""
        rows = [self._decision(pa_id, t, c, r, [0.5, 0.75, 0.25, 0.5]) for t, (c, r) in enumerate(steps)]
        info = {'game': 999, 'in_population': True, 'reward': None, 'kind': est.TERMINAL_VALUE_MISSING,
                'reason': 'end_state_gap', 'start_pitcher': '501', 'problem': None, 'problem_index': None}
        return rows, pa_id, info

    def test_equal_nonzero_shared_slopes_are_in_p(self):
        quarter = [0.25] * 4
        # the candidate keeps the reference probability of the logged action (equal ratios, slope 1 for both) but moves
        # the other mass, so the paired delta is a nonzero point while each policy's own value stays an interval
        rows, pa_id, info = self._shared_end_pa('999:0', [([0.25, 0.5, 0.25, 0.0], quarter), ([0.25, 0.0, 0.5, 0.25], quarter)])
        row = est.pa_value(rows, info, 'l1r')
        self.assertEqual(row['resolution'], 'shared_end_unknown')
        lo, hi = row['delta_bounds']
        self.assertEqual(lo, hi)
        self.assertNotEqual(lo, 0.0)
        self.assertLess(row['candidate_bounds'][0], row['candidate_bounds'][1])
        self.assertEqual(row['weight_candidate'], row['weight_reference'])
        self.assertNotEqual(row['weight_candidate'], 0.0)
        self.assertTrue(reg.in_point_set(row))
        line = ov.pa_line(rows, info)
        self.assertFalse(ov.is_point(line))  # its end is unknown: the old end-label rule would have excluded it
        self.assertEqual(ov.point_deltas([line], None, None, None).tolist(), [lo])

    def test_a_transformation_that_breaks_the_cancellation_invalidates_the_pair(self):
        quarter = [0.25] * 4
        # ratios (2, 1/2) for the candidate and (1, 1) for the reference: equal slope products at the baseline,
        # unequal after a shrinkage or switch transform (and under the policy-specific SN normalisers)
        extra = [self._shared_end_pa('999:0', [([0.5, 0.25, 0.25, 0.0], quarter), ([0.125, 0.375, 0.25, 0.25], quarter)])]
        rows, pa_id, info = extra[0]
        base = est.pa_value(rows, info, 'l1r')
        self.assertTrue(reg.in_point_set(base))
        line = ov.pa_line(rows, info)
        for kind, value in (('shrink', 3.0), ('switch', 1.5)):
            lo, hi = ov.line_value(line, ov.transform(kind, value))['delta_bounds']
            self.assertNotEqual(lo, hi)
            self.assertTrue(np.isnan(ov.point_deltas([line], ov.transform(kind, value), None, None)).all())
        (public, sealed, _), _, _ = _evaluate(extra=extra)
        self.assertGreaterEqual(sealed['p_counts']['p_shared_unknown_end'], 1)
        grid = {name: count for name, count in sealed['p_nonpoint'].items() if name.startswith('grid:')}
        shrink = {name: count for name, count in grid.items() if 'R2_1:' in name}
        switch = {name: count for name, count in grid.items() if 'R2_2:' in name}
        self.assertTrue(shrink and all(count == 1 for count in shrink.values()), grid)  # every finite lambda breaks it
        self.assertTrue(switch and all(count == 0 for count in switch.values()), grid)  # M >= 2 keeps both ratios
        for comp in reg.R1:
            # a broken grid value has no P mean (nonfinite objective) and is never selected
            self.assertEqual(sealed['selection'][f'{comp}xR2_1']['selected'], 'inf')
        for pair in ('R1_0xR2_3', 'R1_axR2_3', 'R1_bxR2_3'):  # SN: per-policy normalisers break it as well
            self.assertEqual(public['pairs'][pair]['valid'], {'game': False, 'pitcher': False})
            self.assertEqual(public['S5'][pair], {'game': 'INVALID', 'pitcher': 'INVALID'})
        self.assertEqual(sealed['p_nonpoint'][reg.BASELINE], 0)
        self.assertTrue(public['pairs'][reg.BASELINE]['valid']['game'])
        for pair in ('R1_0xR2_3', 'R1_axR2_3'):  # the endpoints stay defined although the P mean is not
            self.assertIsNotNone(public['pairs'][pair]['endpoint_half_width']['lower']['game'])
            self.assertIsNotNone(public['pairs'][pair]['endpoint_half_width']['upper']['game'])

    def test_sn_keeps_a_p_row_whose_sn_paired_bounds_coincide(self):
        # PM decision (review round 2 #12): P applies to SN as to every estimator. With every ratio equal to one the
        # SN normalisers are 1 for both policies, so a shared unknown end still cancels: the P rows stay points.
        quarter = [0.25] * 4
        decisions, facts = [], {}
        rng = np.random.default_rng(7)
        for i in range(120):
            game, pa_id = 2000 + i % 30, f'{2000 + i % 30}:{i}'
            steps = []
            for _ in range(int(rng.integers(1, 4))):
                other = [0.25, 0.25, 0.25, 0.25]
                j = int(rng.integers(1, 4))  # move the candidate mass away from the logged action 0 (ratio stays 1)
                other[j], other[(j % 3) + 1] = 0.5, 0.0
                steps.append((other, quarter))
            rows, _, info = self._shared_end_pa(pa_id, steps)
            if i % 2:  # half complete with an observed end, half with a shared unknown end
                info = {**info, 'reward': float(rng.uniform(.2, .8)), 'kind': None, 'reason': None}
            info = {**info, 'game': game}
            decisions.extend(rows)
            facts[pa_id] = info
        stream = lambda: iter(decisions)
        result0, rows0, gint = osd.gint_gate(stream, facts, references={}, draws=300, minimum_games=5,
                                             minimum_pa_starts=10)
        lines = [ov.pa_line(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        e0 = [l for l in lines if l['in_population']]
        point = np.array([reg.in_point_set(r) for r, l in zip(rows0, lines) if l['in_population']])
        unknown = np.array([not ov.is_point(l) for l in e0])
        self.assertEqual(int((point & unknown).sum()), 60)  # 60 shared unknown ends, all in P
        n_bar, ok = ov.sn_normalizers(e0)
        self.assertTrue(ok and all(np.array_equal(nb, np.ones_like(nb)) for nb in n_bar))
        sn = ob.sn_pair('sn', e0, point, u=np.vstack((1 / n_bar[0], 1 / n_bar[1])))
        self.assertEqual(sn.p_nonpoint, 0)
        dr = ob.fixed_pair(reg.BASELINE, [ov.row_of(l) for l in e0], point)
        r = ob.Resampler([l['game'] for l in e0], unit='game', draws=300, seed=5, minimum_clusters=1, minimum_pa_starts=1)
        stats, valid = ob.run_pairs(r, [dr, sn], point, reg.BASELINE)
        self.assertEqual(int(valid['sn']['p'].sum()), int(valid[reg.BASELINE]['p'].sum()))
        self.assertEqual(int(valid['sn']['p'].sum()), 300)
        np.testing.assert_allclose(stats['sn']['mean_p'], stats[reg.BASELINE]['mean_p'], rtol=0, atol=1e-12)
        deltas, lo, hi, nonpoint = ob.sn_values(sn, np.vstack((1 / n_bar[0], 1 / n_bar[1])))
        self.assertFalse(nonpoint.any())
        np.testing.assert_allclose(deltas[point], dr.deltas[point], rtol=0, atol=1e-12)
        # a row outside P with an unknown end only widens the endpoints; it never invalidates the P statistics
        censored = point.copy()
        censored[np.flatnonzero(point & unknown)[:5]] = False
        sn_c = ob.sn_pair('sn', e0, censored, u=np.vstack((1 / n_bar[0], 1 / n_bar[1])))
        dr_c = ob.fixed_pair(reg.BASELINE, [ov.row_of(l) for l in e0], censored)
        stats, valid = ob.run_pairs(r, [dr_c, sn_c], censored, reg.BASELINE)
        self.assertEqual(int(valid['sn']['p'].sum()), 300)
        self.assertTrue(valid['sn']['end'].all())

    def test_endpoint_validity_is_separate_from_the_p_mean(self):
        decisions, facts = synthetic_ledger(seed=3, n_pas=200, n_games=20)
        lines = [ov.pa_line(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        e0 = [l for l in lines if l['in_population']]
        rows = [ov.row_of(l) for l in e0]
        point = np.array([r['delta_bounds'][0] == r['delta_bounds'][1] for r in rows])
        broken = [dict(r) for r in rows]
        i = int(np.flatnonzero(point)[0])
        broken[i]['delta_bounds'] = [broken[i]['delta_bounds'][0], broken[i]['delta_bounds'][0] + 1e-3]  # a non-point
        pair = ob.fixed_pair('broken', broken, point)
        self.assertEqual(pair.p_nonpoint, 1)
        r = ob.Resampler([l['game'] for l in e0], unit='game', draws=200, seed=2, minimum_clusters=1, minimum_pa_starts=1)
        stats, valid = ob.run_pairs(r, [ob.fixed_pair(reg.BASELINE, rows, point), pair], point, reg.BASELINE)
        summary = ob.summarize(stats, valid, r)['broken']
        self.assertFalse(summary['mean_p']['valid'] or summary['shift_p']['valid'])
        self.assertTrue(summary['lower_endpoint']['valid'] and summary['upper_endpoint']['valid'])


class VariantPointRuleTests(unittest.TestCase):
    """PM decision (review round 3 #1/#4): the P rule applies to SN and to every variant exactly as to DR, in the
    full sample, in each replicate and for a small P."""
    @staticmethod
    def _ledger(seed=1729, n=200):
        """The reviewer's counterexample: shared unknown ends, non-dyadic probabilities and q, and every logged
        ratio (hence every SN normaliser) exactly one; the candidate moves only the mass of the other actions."""
        rng = np.random.default_rng(seed)
        decisions, facts = [], {}
        for i in range(n):
            game, pitcher = 3000 + i % 40, str(500 + i % 35)
            pa_id = f'{game}:{i}'
            for t in range(int(rng.integers(1, 4))):
                logging = rng.dirichlet(np.ones(4) * 2.0)
                a = int(rng.choice(4, p=logging))
                candidate = np.empty(4)
                candidate[a] = logging[a]
                candidate[[j for j in range(4) if j != a]] = (1 - logging[a]) * rng.dirichlet(np.ones(3))
                result = {'mask': [True] * 4, 'logging': logging.tolist(), 'reference': logging.tolist(),
                          'candidate': candidate.tolist(), 'logged_index': a,
                          'q_reference': rng.uniform(.3, .7, 4).tolist(), 'rho_reference': 1.0, 'rho_candidate': 1.0}
                decisions.append({'pa_id': pa_id, 'decision_index': t, 'pitcher': pitcher, 'status': 'SUPPORTED',
                                  'result': result, 'balls': 0, 'strikes': min(t, 2)})
            facts[pa_id] = {'game': game, 'in_population': True, 'reward': None, 'kind': est.TERMINAL_VALUE_MISSING,
                            'reason': 'end_state_gap', 'start_pitcher': pitcher, 'problem': None, 'problem_index': None}
        return decisions, facts

    def test_equal_transformed_slopes_cancel_in_every_path(self):
        decisions, facts = self._ledger()
        triples = [(sorted(r, key=lambda d: d['decision_index']), info) for _, info, r in est.by_pa(decisions, facts)]
        lines = [ov.pa_line(rows, info) for rows, info in triples]
        point = np.array([reg.in_point_set(est.pa_value(rows, info, 'l1r')) for rows, info in triples])
        self.assertTrue(0 < point.sum() < len(point))  # the baseline P is the registered rule, unchanged
        n_bar, ok = ov.sn_normalizers(lines)
        self.assertTrue(ok and all(np.array_equal(nb, np.ones_like(nb)) for nb in n_bar))
        u = np.vstack((1 / n_bar[0], 1 / n_bar[1]))
        sn = ob.sn_pair('sn', lines, point, u=u)
        s = sn.sn
        base = np.einsum('pnd,pd->pn', s['coef'], u)
        slope = s['slope'] * u[np.arange(2)[:, None], s['end'][None]]
        p_rows = np.flatnonzero(point)
        self.assertTrue(np.array_equal(slope[0][p_rows], slope[1][p_rows]))  # equal transformed slopes
        split = [i for i in p_rows if (base[0][i] - base[1][i]) != (base[0][i] + slope[0][i] - base[1][i] - slope[1][i])]
        self.assertTrue(split)  # the rounding-prone end formula would have split these P rows
        self.assertEqual(sn.p_nonpoint, 0)
        deltas, lo, hi, nonpoint = ob.sn_values(sn, u)
        self.assertFalse(nonpoint.any())
        self.assertTrue(np.array_equal(lo[p_rows], hi[p_rows]))
        dr = ob.fixed_pair(reg.BASELINE, [ov.row_of(line) for line in lines], point)
        r = ob.Resampler([line['game'] for line in lines], unit='game', draws=500, seed=42, minimum_clusters=1,
                         minimum_pa_starts=1)
        stats, valid = ob.run_pairs(r, [dr, sn], point, reg.BASELINE)
        self.assertEqual((int(valid[reg.BASELINE]['p'].sum()), int(valid['sn']['p'].sum())), (500, 500))
        np.testing.assert_allclose(stats['sn']['mean_p'], stats[reg.BASELINE]['mean_p'], rtol=0, atol=1e-12)
        p_lines = [line for line, keep in zip(lines, point) if keep]
        for kind, value in (('shrink', 3.0), ('switch', 2.0)):  # fixed-transform variants follow the same rule
            self.assertTrue(np.isfinite(ov.point_deltas(p_lines, ov.transform(kind, value), None, None)).all())
            pair = ob.fixed_pair(kind, [ov.row_of(line, ov.transform(kind, value)) for line in lines], point)
            self.assertEqual(pair.p_nonpoint, 0)
        unequal = ov.shared_end_bounds(0.3, 0.1, 0.5, 0.5 + 1e-15)  # genuinely unequal slopes are never merged
        self.assertNotEqual(unequal[0], unequal[1])

    def test_small_p_invalidates_the_p_statistics_of_every_pair_only(self):
        decisions, facts = synthetic_ledger(seed=5, n_pas=300, n_games=20, n_pitchers=12)
        stream = lambda: iter(decisions)
        result0, rows0, gint = osd.gint_gate(stream, facts, references={}, **GATE)
        e0 = [row for row in rows0 if row['in_population']]
        p = sum(reg.in_point_set(row) for row in e0)
        minimum = (p + len(e0)) // 2 + 1
        self.assertTrue(p < minimum <= len(e0))  # P below the minimum, E0 above it
        rng = np.random.default_rng(1)
        sur = {pa: {'R1_a': (float(np.clip(f['reward'] + rng.normal(0, .02), 0, 1)), 'class'),
                    'R1_b': (float(np.clip(f['reward'] + rng.normal(0, .01), 0, 1)), 'bip_quality'), 'spray_missing': False}
               for pa, f in facts.items() if f.get('reward') is not None}
        public, sealed, _ = osd.rehearsal_evaluate(
            stream, facts, baseline=(result0, rows0), surrogates=sur, gint=gint,
            train_residuals={'hit': {'FB': {'n': 10, 'mean_residual': -0.01, 'se': 0.005}}},
            r1_paths={'rehearsal': osg.path_shares(sur), 'train_census': CENSUS}, r3_status='NOT_RUN:R3', draws=200,
            minimum_games=5, minimum_pitchers=4, minimum_pa_starts=minimum, placebos=2)
        self.assertEqual(public['validity'], {'point_set_meets_minimum': False, 'game_bootstrap_eligible': True,
                                              'pitcher_bootstrap_eligible': True})
        for pair in reg.PAIRS:  # SN and DR alike
            self.assertEqual(public['pairs'][pair]['valid'], {'game': False, 'pitcher': False})
            self.assertIsNone(public['pairs'][pair]['H_point']['game'])
        for pair in (reg.BASELINE, 'R1_axR2_0', 'R1_0xR2_3'):  # the all-E0 endpoint summaries are kept
            for b in ('game', 'pitcher'):
                self.assertIsNotNone(public['pairs'][pair]['endpoint_half_width']['lower'][b])
                self.assertIsNotNone(public['R5']['e'][pair][b]['mde80_current_rule'])
                self.assertIsNone(public['R5']['e'][pair][b]['mde80_point'])
        self.assertTrue(osd.validate_public(json.loads(osd.seal(public, sealed)[0])))


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
           's2': {p: {'2*kappa_p': {'outcome': s2}, '2*zeta_p': {'outcome': s2}, 'invariant': 'PASS'}
                  for p in reg.PAIRS if reg.split_pair(p)[0] in ('R1_a', 'R1_b')}}
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
        s5x['s2']['R1_axR2_0']['2*zeta_p']['outcome'] = 'FAIL'  # S-2 binds only the R1_a / R1_b pairs
        self.assertEqual(decide(public, tab, s5x)['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_0xR2_3')
        public, tab, s5x = _decision_inputs(h={'R1_axR2_0': 0.5})
        s5x['nets']['S1'][reg.BASELINE]['outcome'] = 'FAIL'  # the baseline's shared net binds every pair
        self.assertEqual(decide(public, tab, s5x)['label'], 'BIAS_NOT_CERTIFIED:R1_axR2_0')
        public['R5']['h']['flag'] = 'SE_CALIBRATION_WARNING'
        self.assertEqual(decide(public, tab, s5x)['flags'], ['SE_CALIBRATION_WARNING'])

    def test_invalid_repetitions_never_support_adoption(self):
        # PM decision (review #1): an invalid repetition or replication in an acceptance-relevant run or cell makes
        # that cell INCONCLUSIVE (never PASS) and the pair non-adoptable
        decide = lambda *a: odc.decide(*a, r3_status='NOT_RUN:R3')
        gains = {'R1_axR2_3': 0.5}
        self.assertEqual(decide(*_decision_inputs(h=gains))['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_3')
        cases = {
            'S-2 run 4 invalid repetition': lambda s, t: s['s2']['R1_axR2_3'].__setitem__(
                '2*kappa_p', reg.inconclusive_invalid('1 invalid repetitions in run 4')),
            'S-2 run 5 missing': lambda s, t: s['s2']['R1_axR2_3'].pop('2*zeta_p'),
            'S-1 net with an invalid repetition': lambda s, t: s['nets']['S1'].__setitem__(
                'R1_axR2_3', reg.net_outcome([0.0, 0.0, 0.0], [0.001] * 3, valid=False)),
            'S-4 net without an outcome': lambda s, t: s['nets']['S4'].__setitem__('R1_axR2_3', {'outcome': None}),
            'invariant invalid': lambda s, t: s['s2']['R1_axR2_3'].__setitem__('invariant', 'INVALID'),
            'invalid flag on a PASS': lambda s, t: s['s2']['R1_axR2_3'].__setitem__(
                '2*zeta_p', {'outcome': 'PASS', 'invalid_repetition': True}),
            'TAB cell with invalid replications': lambda s, t: t['acceptance']['T-A3'].__setitem__(
                'R1_axR2_3', reg.inconclusive_invalid('2 invalid replications')),
        }
        for name, mutate in cases.items():
            public, tab, s5x = _decision_inputs(h=gains)
            mutate(s5x, tab)
            out = decide(public, tab, s5x)
            with self.subTest(name):
                self.assertEqual(out['label'], 'BIAS_NOT_CERTIFIED:R1_axR2_3')
                self.assertFalse(out['pairs']['R1_axR2_3']['mandatory_valid'])
        # review #11: an R1_b table failure is NOT_EVALUABLE for the R1_b pair only; the other pairs continue
        public, tab, s5x = _decision_inputs(h={'R1_bxR2_0': 0.5, 'R1_axR2_0': 0.55})
        tab['acceptance']['T-A1']['R1_bxR2_0'] = reg.cell_outcome(None, None, not_evaluable=True)
        self.assertEqual(decide(public, tab, s5x)['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_0')

    def test_degenerate_half_widths_are_inconclusive(self):
        # PM decision (review #13): (ii) is the registered inequality evaluated directly; a zero half-width is
        # INCONCLUSIVE (never a pass); the ratio is reporting only
        public, tab, s5x = _decision_inputs(h={'R1_axR2_3': 0.0})
        ii = odc.condition_ii(public, 'R1_axR2_3')
        self.assertEqual((ii['outcome'], ii['pass'], ii['degenerate']), ('INCONCLUSIVE', False, True))
        self.assertEqual(odc.decide(public, tab, s5x, r3_status='NOT_RUN:R3')['label'], 'NO_PRECISION_GAIN')
        public, _, _ = _decision_inputs(h={reg.BASELINE: 0.0, 'R1_axR2_3': 0.0})
        ii = odc.condition_ii(public, 'R1_axR2_3')
        self.assertEqual((ii['outcome'], ii['pass'], ii['ratio']['game']), ('INCONCLUSIVE', False, None))
        public, _, _ = _decision_inputs(h={'R1_axR2_3': 0.69})
        ii = odc.condition_ii(public, 'R1_axR2_3')
        self.assertEqual((ii['outcome'], ii['pass']), ('PASS', True))
        self.assertAlmostEqual(ii['ratio']['game'], 0.69, places=15)
        public['pairs']['R1_axR2_3']['valid']['pitcher'] = False
        self.assertEqual(odc.condition_ii(public, 'R1_axR2_3')['outcome'], 'INVALID')


if __name__ == '__main__':
    unittest.main()
