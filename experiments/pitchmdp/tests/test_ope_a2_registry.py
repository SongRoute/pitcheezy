"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-01 registry: config check rejections, fail-closed execution
gate, digest canonicalization with nulled paths, seeds and streams, the fixed decision rules (minimizer
ties, cluster SE, cell outcome, nets, pair selection) and the run-window rule. No data is read.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

CONFIG = reg.REPO / reg.CONFIG_PATH


def load():
    return json.loads(CONFIG.read_text())


def registered(config=None):
    """The config with the registration step applied and synthetic (non-real) pins filled."""
    c = copy.deepcopy(config or load())
    c.update(status=reg.REGISTERED, registered=True, preregistration_complete=True)
    c['execution'] = {**c['execution'], 'enabled': True}
    pins = c['registration_input_pins']
    pins.update(s6_dr_a5_ledger_path='/tmp/synthetic/ledger.jsonl', s6_dr_a5_ledger_sha256='a' * 64,
                s6_dr_a5_manifest_sha256='b' * 64, train_raw_files_sha256={n: 'c' * 64 for n in reg.RAW_ALLOWLIST},
                processed_frame_identity={'synthetic': True}, x2_world_predictor_identity={'kind': 'g0_single_member', 'seed': 0},
                execution_source_commit='d' * 40, new_files_sha256={f: 'e' * 64 for f in reg.NEW_FILES},
                digest_manifest_path='/tmp/synthetic/digest.jsonl', output_root='/tmp/synthetic/out',
                r3_status='NOT_RUN:R3', a3_behavior_policy_sha256='NOT_APPLICABLE',
                a3_provenance_manifest_sha256='NOT_APPLICABLE', smoke_train_subset={'game_dates': ['2024-06-01', '2024-06-30']},
                smoke_tab_replications_per_cell=5)
    return c


class ConfigCheckTests(unittest.TestCase):
    def test_materialized_config_is_a_registry_conform_draft(self):
        config = load()
        self.assertEqual(reg.config_check(config), reg.DRAFT)
        self.assertEqual((config['registered'], config['preregistration_complete'], config['execution']['enabled']),
                         (False, False, False))
        self.assertTrue(all(v is None for k, v in config['stage_output_seals'].items() if k not in ('A2-S0_contents', 'rule')))
        self.assertEqual(reg.config_check(registered()), reg.REGISTERED)

    def test_deviations_are_rejected(self):
        def mutate(path, value):
            c = load()
            node = c
            *head, last = path
            for part in head:
                node = node[part]
            node[last] = value
            return c
        cases = {
            'B unrounded': (('synthetic_bias_check', 'acceptance', 'bias_bound_B', 'value'), 1.2612e-05),
            'X 20%': (('decision_rules', 'condition_ii_half_width', 'X', 'value'), 0.2),
            'lambda grid': (('arms', 'R2_estimator', 'R2_1_dr_shrinkage', 'lambda_grid', 'value'), ['inf', 1000, 100, 30, 10]),
            'M grid': (('arms', 'R2_estimator', 'R2_2_switch_dr', 'M_grid', 'value'), ['inf', 20, 10, 5, 3, 1.5]),
            'm grid': (('arms', 'R1_reward', 'R1_b_bip_quality_expected', 'shrinkage', 'm_grid', 'value'), [5, 20]),
            'EV edges': (('arms', 'R1_reward', 'R1_b_bip_quality_expected', 'keys', 'ev_mph_edges', 'value'), [60, 70]),
            'total order': (('decision_rules', 'selection', 'total_order', 'value'), list(reversed(reg.TOTAL_ORDER))),
            'pitcher seed': (('arms', 'R4_bootstrap', 'pitcher', 'seed'), 1),
            'game seed': (('seeds', 'game_bootstrap', 'value'), 3060201744),
            'min pitchers': (('decision_rules', 'validity_gates', 'new', 'minimum_pitchers_pitcher_bootstrap', 'value'), 20),
            'invalid share': (('decision_rules', 'validity_gates', 'inherited', 'invalid_replicate_share_max'), 0.1),
            'repetitions': (('synthetic_bias_check', 'S5X_world', 'repetitions', 'value'), 12),
            'placebos': (('arms', 'R5_diagnostics_before_any_delta', 'h_placebo_se_check', 'placebos'), 50),
            'rss cap': (('cost_budget', 'rss_cap_gib', 'value'), 16),
            'base commit': (('base_commit',), 'f' * 40),
            'protocol': (('protocol',), 'exp_ope_res_01_registration_draft_v1'),
            'R3 statuses': (('registration_input_pins', 'r3_status_allowed'), ['ENABLED_DIAGNOSTIC_ONLY']),
            'seal filled': (('stage_output_seals', 'A2-S0_outputs_manifest_sha256'), 'a' * 64),
            'protected list': (('governing_rules', 'not_modified_paths'), ['experiments/pitchmdp/pitchmdp/policy_estimator.py']),
        }
        for name, (path, value) in cases.items():
            with self.subTest(name), self.assertRaises(IntegrityError):
                reg.config_check(mutate(path, value))
        c = load()
        c['seeds']['roles']['placebo']['value'] += 1
        with self.assertRaisesRegex(IntegrityError, 'seed role placebo'):
            reg.config_check(c)
        c = load()
        c['synthetic_bias_check']['synthetic_matrix']['S5X_independent_streams']['world_seed_values_derived'][0] = reg.V2_ROLE_SEED
        with self.assertRaisesRegex(IntegrityError, 'world seeds'):
            reg.config_check(c)
        c = load()
        c['synthetic_bias_check']['synthetic_matrix']['TAB_acceptance_cells'][0]['R'] = 1000
        with self.assertRaisesRegex(IntegrityError, 'TAB acceptance'):
            reg.config_check(c)
        c = load()
        c['cost_budget']['hang_guards_seconds']['A2-S4']['value'] = 7200
        with self.assertRaisesRegex(IntegrityError, 'hang guards'):
            reg.config_check(c)
        c = load()
        c['blinding']['public_before_decision_allowlist'][0]['public_type'] = 'signed_effect'
        with self.assertRaisesRegex(IntegrityError, 'allowlist'):
            reg.config_check(c)
        c = load()
        c['registration']['config_digest']['value'] = 'a' * 64
        with self.assertRaisesRegex(IntegrityError, 'digest value'):
            reg.config_check(c)
        c = load()
        c['registration']['config_sha256'] = 'a' * 64
        with self.assertRaisesRegex(IntegrityError, 'in-config digest'):
            reg.config_check(c)
        c = load()
        c['implementation']['tab_world']['archetype_perturbation_log_sd'] = 0.2
        with self.assertRaisesRegex(IntegrityError, 'implementation'):
            reg.config_check(c)
        c = load()
        c['registration']['config_digest']['nulled_json_paths'] = list(reg.NULLED_PATHS[:-1])
        with self.assertRaises(IntegrityError):
            reg.config_check(c)

    def test_status_flags_must_agree(self):
        c = load()
        c['execution']['enabled'] = True
        with self.assertRaisesRegex(IntegrityError, 'draft is unregistered'):
            reg.config_check(c)
        c = registered()
        c['preregistration_complete'] = False
        with self.assertRaisesRegex(IntegrityError, 'REGISTERED needs'):
            reg.config_check(c)
        c = load()
        c['status'] = 'REGISTRATION_PENDING'
        with self.assertRaisesRegex(IntegrityError, 'unknown status'):
            reg.config_check(c)


class ExecutionGateTests(unittest.TestCase):
    def test_draft_refuses_fail_closed(self):
        with self.assertRaisesRegex(IntegrityError, 'not REGISTERED and execution-enabled'):
            reg.require_executable(load())

    def test_registered_needs_every_pin(self):
        reg.require_executable(registered())
        for key in ('s6_dr_a5_ledger_sha256', 'execution_source_commit', 'digest_manifest_path', 'output_root',
                    'r3_status', 'smoke_train_subset', 'x2_world_predictor_identity', 'processed_frame_identity'):
            c = registered()
            c['registration_input_pins'][key] = None
            with self.subTest(key), self.assertRaises(IntegrityError):
                reg.require_executable(c)
        c = registered()
        c['registration_input_pins']['new_files_sha256'].pop(reg.NEW_FILES[0])
        with self.assertRaisesRegex(IntegrityError, 'new_files_sha256'):
            reg.require_executable(c)

    def test_pin_shapes(self):
        c = registered()
        c['registration_input_pins']['a3_behavior_policy_sha256'] = 'a' * 64
        with self.assertRaisesRegex(IntegrityError, 'NOT_APPLICABLE'):
            reg.check_pins(c)
        c = registered()
        c['registration_input_pins'].update(r3_status='ENABLED_DIAGNOSTIC_ONLY')
        with self.assertRaisesRegex(IntegrityError, 'A3 artifact'):
            reg.check_pins(c)
        c = registered()
        c['registration_input_pins']['smoke_train_subset'] = {'game_dates': ['2025-05-01', '2025-05-10']}
        with self.assertRaisesRegex(IntegrityError, 'inside TRAIN'):
            reg.check_pins(c)
        c = registered()
        c['registration_input_pins']['output_root'] = str(reg.REPO / 'results' / 'a2')
        with self.assertRaisesRegex(IntegrityError, 'outside the repository results'):
            reg.check_pins(c)
        c = registered()
        c['registration_input_pins']['smoke_tab_replications_per_cell'] = 0
        with self.assertRaises(IntegrityError):
            reg.check_pins(c)


class DigestTests(unittest.TestCase):
    def test_canonical_serialization_and_nulled_paths(self):
        config = load()
        obj = copy.deepcopy(config)
        for key in ('status', 'registered', 'preregistration_complete', 'execution', 'stage_output_seals'):
            obj[key] = None
        for key in ('registered_kst', 'review_reference', 'code_review_reference'):
            obj['registration'][key] = None
        for gate in obj['pre_run_gates']:
            gate['status'] = None
        text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        self.assertEqual(reg.config_digest(config), hashlib.sha256(text.encode('utf-8')).hexdigest())
        self.assertTrue(reg.canonical_digest_text(config).endswith('}\n') and not reg.canonical_digest_text(config).endswith('\n\n'))

    def test_registration_changes_only_nulled_paths(self):
        draft = load()
        done = registered(draft)
        done['registration_input_pins'] = draft['registration_input_pins']  # pins are filled before G-CODE
        done['registration'].update(registered_kst='2026-10-28T14:00+09:00', review_reference='REVIEW-A2-ASTRA-005',
                                    code_review_reference='CODE-REVIEW-A2-001')
        for gate in done['pre_run_gates']:
            gate['status'] = 'PASS'
        self.assertEqual(reg.config_digest(done), reg.config_digest(draft))
        other = copy.deepcopy(draft)
        other['task'] = other['task'] + ' '
        self.assertNotEqual(reg.config_digest(other), reg.config_digest(draft))
        pins = copy.deepcopy(draft)
        pins['registration_input_pins']['output_root'] = '/tmp/x'
        self.assertNotEqual(reg.config_digest(pins), reg.config_digest(draft))
        missing = copy.deepcopy(draft)
        del missing['registration']['review_reference']
        with self.assertRaisesRegex(IntegrityError, 'nulled path missing'):
            reg.config_digest(missing)
        unicode = copy.deepcopy(draft)
        unicode['task'] = '설계'
        self.assertIn('설계', reg.canonical_digest_text(unicode))  # ensure_ascii=False: raw UTF-8, not escapes


class SeedTests(unittest.TestCase):
    def test_roles_world_seeds_and_streams(self):
        self.assertEqual(reg.derive_seed(reg.S6_SEED_BASE, 1), 3060201743)
        self.assertEqual(reg.derive_seed(reg.S6_SEED_BASE, 2), 4115122098)
        self.assertEqual({r: reg.role_seed(r) for r in reg.ROLES}, reg.ROLE_VALUES)
        seeds = [reg.world_seed(r['k']) for r in reg.S5X_RUNS]
        self.assertEqual(seeds, [4115122098 ^ k for k in range(1, 16)])
        self.assertEqual(len(set(seeds)), 15)
        self.assertEqual(seeds, load()['synthetic_bias_check']['synthetic_matrix']['S5X_independent_streams'][
            'world_seed_values_derived'])
        draws = {key: np.random.default_rng(reg.role_sequence('s5x_world', *key)).random(4).tolist()
                 for key in ((4, 0, 3, 0), (4, 1, 3, 0), (5, 0, 3, 0), (4, 0, 3, 1))}
        self.assertEqual(len({json.dumps(v) for v in draws.values()}), 4)  # log/truth namespaces and runs differ
        self.assertNotEqual(reg.role_sequence('smoke', 1, 0).entropy, None)
        with self.assertRaises(IntegrityError):
            reg.role_sequence('s5x_world', -1)
        with self.assertRaises(IntegrityError):
            reg.world_seed(16)

    def test_tab_matrix(self):
        cells = reg.tab_cells()
        self.assertEqual(len(cells), 57)
        self.assertEqual([c['id'] for c in cells[:4]], ['T-A1', 'T-A2', 'T-A3', 'T-A4'])
        self.assertTrue(all(c['R'] == 2000 for c in cells[:4]) and all(c['R'] == 300 for c in cells[4:]))
        self.assertEqual(sum(c['R'] for c in cells), 23900)
        self.assertEqual([c['index'] for c in cells], list(range(57)))
        self.assertTrue(cells[-1]['drift'] and cells[-1]['law'] == 'V2' and cells[-1]['eta'] == 'eta_p')
        combos = {(c['law'], c['eta'], c['kappa'], c['zeta']) for c in cells[:56]}
        self.assertEqual(len(combos), 56)
        self.assertEqual([c['law'] for c in cells[4:56]], ['V2'] * 25 + ['V3'] * 27)  # 28 - 3 and 28 - 1
        self.assertEqual(reg.scale('2*kappa_p', .3), .6)
        self.assertEqual(reg.scale('eta_p/2', .3), .15)
        self.assertEqual(reg.scale('0', None), 0.0)


class RuleTests(unittest.TestCase):
    def test_minimizer_ties(self):
        inf = math.inf
        self.assertEqual(reg.select_minimizer([inf, 1000., 100.], [0.0, 0.0, 1.]), inf)  # exact tie at v_min = 0
        self.assertEqual(reg.select_minimizer([inf, 1000., 100.], [2.0, 1.0 + 5e-13, 1.0]), 1000.)
        self.assertEqual(reg.select_minimizer([inf, 1000., 100.], [2.0, 1.0 + 2e-12, 1.0]), 100.)
        self.assertEqual(reg.select_minimizer([inf, 3.], [math.nan, 5.]), 3.)
        self.assertIsNone(reg.select_minimizer([inf, 3.], [math.inf, math.nan]))
        self.assertEqual(reg.select_minimizer([5, 20, 80], [1.3, 1.3, 1.4]), 20)  # larger m inside the tie set
        self.assertEqual(reg.select_minimizer([inf, 1e6], [1e6, 1e6 + 1e-7]), inf)  # relative tolerance x |v_min|

    def test_cluster_se_matches_hand_computation(self):
        values = [1., 2., 3., 4., 10.]
        clusters = ['a', 'a', 'b', 'c', 'c']
        t = {'a': 3., 'b': 3., 'c': 14.}
        n = {'a': 2, 'b': 1, 'c': 2}
        mean = 20. / 5
        hand = math.sqrt(3 / 2 * sum((t[g] - n[g] * mean) ** 2 for g in t) / 25)
        self.assertAlmostEqual(reg.cluster_se(values, clusters), hand, places=15)
        self.assertAlmostEqual(reg.cluster_se_index(values, np.array([0, 0, 1, 2, 2]), 3), hand, places=15)
        self.assertTrue(math.isnan(reg.cluster_se([1., 2.], ['a', 'a'])))
        self.assertTrue(math.isnan(reg.cluster_se([1., math.inf], ['a', 'b'])))

    def test_cell_outcome_precision_gate_and_bounds(self):
        B = reg.B
        self.assertEqual(B, 1.26e-05)
        self.assertEqual(reg.cell_outcome(0.0, B / 2 + 1e-12)['outcome'], 'INCONCLUSIVE')  # never PASS without precision
        self.assertEqual(reg.cell_outcome(0.0, 1e-6)['outcome'], 'PASS')
        self.assertEqual(reg.cell_outcome(B - 1.96 * 1e-6, 1e-6)['outcome'], 'PASS')
        self.assertEqual(reg.cell_outcome(B + 1.96 * 1e-6 + 1e-9, 1e-6)['outcome'], 'FAIL')
        self.assertEqual(reg.cell_outcome(B, 1e-6)['outcome'], 'INCONCLUSIVE')
        self.assertEqual(reg.cell_outcome(-(B - 1.96 * 1e-6), 1e-6)['outcome'], 'PASS')  # |b|
        self.assertEqual(reg.cell_outcome(5e-5, 1e-6, beta_0=4.5e-5)['outcome'], 'PASS')  # V3: U <= |beta_0| + B
        self.assertEqual(reg.cell_outcome(9e-5, 1e-6, beta_0=4.5e-5)['outcome'], 'FAIL')
        self.assertEqual(reg.cell_outcome(None, 1e-6)['outcome'], 'INVALID')
        self.assertEqual(reg.cell_outcome(0.0, 1e-6, valid=False)['outcome'], 'INVALID')
        # the unrounded derivation 1.2612e-5 is never the bound (it would PASS these)
        self.assertEqual(reg.cell_outcome(1.2605e-5, 1e-10)['outcome'], 'FAIL')
        self.assertEqual(reg.cell_outcome(1.2595e-5, 1e-10)['outcome'], 'PASS')

    def test_nets(self):
        out = reg.net_outcome([.001, -.002, .0005], [.001, .001, .001])
        self.assertEqual(out['outcome'], 'PASS')
        self.assertAlmostEqual(out['se'], math.sqrt(3e-6) / 3, places=15)
        self.assertEqual(reg.net_outcome([.02, .02, .02], [.02, .02, .02])['outcome'], 'FAIL')  # |gap| > 0.01
        self.assertEqual(reg.net_outcome([.0, .0, .0], [.009, .009, .009])['outcome'], 'FAIL')  # SE > 0.005
        self.assertEqual(reg.net_outcome([.004, .004, .004], [.001, .001, .001])['outcome'], 'FAIL')  # excludes 0
        self.assertEqual(reg.net_outcome([.0, .0], [.001, .001], valid=False)['outcome'], 'INVALID')
        z = reg.net_outcome([0.00129], [0.000576])  # 0.00129 - 2.2414 * 0.000576 = 0.00000 (inside at 97.5%)
        self.assertEqual(z['outcome'], 'PASS')

    def test_pair_selection_window(self):
        h = {p: 1.0 for p in reg.TOTAL_ORDER}
        h.update({'R1_bxR2_0': 0.50, 'R1_0xR2_3': 0.505, 'R1_axR2_1': 0.49})
        passing = {'R1_bxR2_0', 'R1_0xR2_3', 'R1_axR2_1'}
        self.assertEqual(reg.select_pair(h, passing, 1.0), 'R1_bxR2_0')  # within 0.02 x 1.0 of the minimum
        self.assertEqual(reg.select_pair(h, passing, 0.4), 'R1_axR2_1')  # narrower window: only the minimum
        h2 = {**h, 'R1_axR2_0': 0.5, 'R1_bxR2_0': 0.509, 'R1_0xR2_3': 0.518}
        self.assertEqual(reg.select_pair(h2, {'R1_bxR2_0', 'R1_0xR2_3'}, 0.5), 'R1_bxR2_0')  # no chaining
        self.assertIsNone(reg.select_pair(h, set(), 1.0))


class WindowTests(unittest.TestCase):
    def kst(self, day, hour, minute=0):
        return datetime(2026, 10, day, hour, minute, tzinfo=reg.KST)

    def test_stage_window(self):
        self.assertTrue(reg.stage_window_ok(self.kst(27, 13, 30), 7200))
        self.assertTrue(reg.stage_window_ok(self.kst(27, 23, 0), 7616))
        self.assertFalse(reg.stage_window_ok(self.kst(28, 5, 0), 7200))  # would run into 06:30
        self.assertFalse(reg.stage_window_ok(self.kst(28, 12, 0), 60))  # inside the protection window
        self.assertFalse(reg.stage_window_ok(self.kst(28, 4, 0), 9667))  # 06:41 end
        self.assertTrue(reg.stage_window_ok(self.kst(28, 3, 40), 9667))  # 06:21 end
        self.assertTrue(reg.stage_window_ok(self.kst(28, 4, 30), 7200))  # ends exactly at 06:30
        self.assertFalse(reg.stage_window_ok(self.kst(27, 13, 29), 60))
        utc = self.kst(27, 14, 0).astimezone(timezone.utc)
        self.assertTrue(reg.stage_window_ok(utc, 3600))
        with self.assertRaises(IntegrityError):
            reg.stage_window_ok(datetime(2026, 10, 27, 14, 0), 60)  # naive time refused


class SourceTests(unittest.TestCase):
    def test_new_modules_stay_outside_the_policy_identity_closure(self):
        from pitchmdp import ope_bootstrap, ope_decision, ope_diagnostics, ope_sealed, ope_surrogate  # noqa: F401
        from pitchmdp import ope_synthetic_a2, ope_variants  # noqa: F401
        from pitchmdp import policy_identity as pid
        closure = pid.project_sources([])  # the fixed runtime call path and its module-level imports
        self.assertTrue(closure)
        self.assertFalse([path for path in closure if '/ope_' in path or path.endswith('run_ope_a2.py')], closure)
        for rel in reg.PROTECTED_SOURCES:
            self.assertFalse(rel.split('/')[-1].startswith('ope_'))

    def test_protected_sources_and_new_files(self):
        self.assertEqual(reg.verify_protected_sources(), reg.PROTECTED_SOURCES)
        for rel in reg.NEW_FILES:
            self.assertTrue((reg.REPO / rel).is_file(), rel)
        self.assertEqual(load()['governing_rules']['not_modified_paths'], list(reg.PROTECTED_SOURCES))


if __name__ == '__main__':
    unittest.main()
