"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 (A2b) registry: seeds (roles, S-2' world seeds, disjointness from
EXP-OPE-RES-01), the 25 rows and the S6 12-row split, labels, the integrity comparison (absolute 1e-12, counts and
lambda / M by equality incl. inf), condition (ii), the TAB reuse and exact-value cell rules, the run window and
hang guards, the S-2' cost gate, the typed public allowlist and integrity records, the config check, the pins and
the digest. Every config is a synthetic fixture (never the real draft or configs/EXP-OPE-RES-02.json).
"""
import copy
from datetime import datetime
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ope_a2b_fixtures import draft_config, registered_config  # noqa: E402
from pitchmdp import ope_redesign_registry as a2  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

KST = rb.KST


def kst(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=KST)


class SeedTests(unittest.TestCase):
    def test_roles_derive_from_the_a2b_base(self):
        self.assertTrue(rb.verify_seeds())
        for role, index in rb.ROLES.items():
            value = int(np.random.SeedSequence([20261028, index]).generate_state(1)[0])
            self.assertEqual(value, rb.ROLE_VALUES[role], role)
        self.assertEqual(rb.A2B_SELECTION_SALT, 759396736)
        self.assertEqual(rb.S6_SELECTION_SALT_RECOMPUTED, int(np.random.SeedSequence([20260929, 3]).generate_state(1)[0]))
        # the same rule reproduces the EXP-OPE-RES-01 / S6 role values (design 9.4)
        self.assertEqual(rb.derive_seed(20261027, 0), 3624346705)
        self.assertEqual(rb.derive_seed(20260929, 1), 3060201743)

    def test_world_seeds_match_the_registered_list_and_never_collide(self):
        derived = [int(np.random.SeedSequence([20261028, 5, k]).generate_state(1)[0]) for k in range(1, 77)]
        self.assertEqual(derived, list(rb.WORLD_SEEDS_DERIVED))
        self.assertEqual([rb.world_seed(k) for k in range(1, 77)], derived)
        self.assertEqual(len(set(derived)), 76)
        a2_seeds = {a2.world_seed(k) for k in range(1, 16)}
        self.assertEqual(len(a2_seeds), 15)
        self.assertFalse(set(derived) & a2_seeds)
        self.assertNotIn(a2.V2_ROLE_SEED, derived)
        with self.assertRaises(IntegrityError):
            rb.world_seed(0)
        with self.assertRaises(IntegrityError):
            rb.world_seed(77)

    def test_registered_run_map_and_fixed_counts(self):
        self.assertEqual(rb.N_PRIME, {'2*zeta_p': 10, '2*kappa_p': 55, 'descriptive_kappa_prime': 5})
        self.assertEqual(rb.registered_run(1), '2*zeta_p')
        self.assertEqual(rb.registered_run(10), '2*zeta_p')
        self.assertEqual(rb.registered_run(11), '2*kappa_p')
        self.assertEqual(rb.registered_run(65), '2*kappa_p')
        self.assertEqual(rb.registered_run(70), 'descriptive_kappa_prime')
        for k in (0, 71, 76):  # 71..76 belonged to the non-selected S5-X-R3
            with self.assertRaises(IntegrityError):
                rb.registered_run(k)
        self.assertEqual(rb.GINT_S2EQ_RUNS, {'2*zeta_p': (5, 11, 12), '2*kappa_p': (4, 9, 10)})
        self.assertTrue(all(a2.world_seed(k) == a2.V2_ROLE_SEED ^ k for ks in rb.GINT_S2EQ_RUNS.values() for k in ks))

    def test_evaluator_seeds_by_mode(self):
        self.assertEqual(rb.EVALUATOR_SEEDS[rb.MODE_S6]['game'], 3060201743)
        self.assertEqual(rb.EVALUATOR_SEEDS[rb.MODE_S6]['pitcher'], 3624346705)
        self.assertEqual(rb.EVALUATOR_SEEDS[rb.MODE_REHEARSAL]['game'], 3287529750)
        self.assertEqual(rb.EVALUATOR_SEEDS[rb.MODE_REHEARSAL]['pitcher'], 1876336626)
        a2_stream = rb.placebo_stream(('EXP-OPE-RES-01', 'placebo'))(3)
        self.assertEqual(list(a2_stream.entropy), [20261027, 1, 3])  # EXP-OPE-RES-01 role placebo (index 1)
        new = rb.placebo_stream(('EXP-OPE-RES-02', 'placebo'))(3)
        self.assertEqual(list(new.entropy), [20261028, 2, 3])  # A2b role placebo (index 2)
        with self.assertRaises(IntegrityError):
            rb.placebo_stream(('EXP-OPE-RES-03', 'placebo'))


class RowTests(unittest.TestCase):
    def test_rows_candidates_and_the_s6_split(self):
        self.assertEqual(len(rb.ROWS), 25)
        self.assertEqual(rb.BASELINE, 'R1_0xR2_0xB0')
        self.assertEqual(rb.CANDIDATES, ('R1_axR2_0xB0', 'R1_bxR2_0xB0', 'R1_0xR2_1xB0', 'R1_axR2_1xB0', 'R1_bxR2_1xB0'))
        self.assertEqual(rb.TOTAL_ORDER, rb.CANDIDATES)
        self.assertEqual(len(rb.S6_MODE_ROWS), 12)
        self.assertEqual(len(rb.S6_REFUSED_ROWS), 13)
        self.assertEqual(set(rb.S6_MODE_ROWS) | set(rb.S6_REFUSED_ROWS), set(rb.ROWS))
        self.assertEqual([rb.a2_pair(r) for r in rb.S6_MODE_ROWS], list(a2.PAIRS))
        self.assertIsNone(rb.a2_pair('R1_0xR2_0xB2'))
        self.assertEqual(rb.split_row(rb.LAMBDA_FIXED_ROW), ('R1_b', 'R2_1', 'B0'))
        self.assertEqual(len(rb.DESCRIPTIVE_ROWS), 19)
        self.assertEqual(rb.s2p_cells_for(['R1_axR2_1xB0']), ('2*zeta_p',))
        self.assertEqual(rb.s2p_cells_for(['R1_bxR2_0xB0']), ('2*zeta_p', '2*kappa_p', 'descriptive_kappa_prime'))
        self.assertEqual(rb.s2p_cells_for(['R1_0xR2_1xB0']), ())

    def test_labels_in_precedence(self):
        self.assertEqual(rb.LABELS[0], 'FAILED_INTEGRITY')
        self.assertEqual(rb.LABELS[3], 'DENOMINATOR_NOT_CERTIFIED:<B2 tuples>')
        self.assertEqual(rb.LABELS[-1], 'NO_PRECISION_GAIN')
        self.assertEqual(len(rb.LABELS), 6)


class RuleTests(unittest.TestCase):
    def test_integrity_tolerance_is_absolute_and_counts_are_exact(self):
        self.assertTrue(rb.integrity_equal(1e-5 + 9e-13, 1e-5))
        self.assertFalse(rb.integrity_equal(1e-5 + 2e-12, 1e-5))
        self.assertTrue(rb.integrity_equal(9125.0, 9125.0))
        self.assertFalse(rb.integrity_equal(9125.000000000002, 9125.0))  # ESS scale: the next double differs
        self.assertTrue(rb.integrity_equal(math.inf, math.inf))  # lambda / M = inf by equality (A2 _same rejects it)
        self.assertFalse(rb.integrity_equal(math.inf, 1e300))
        self.assertFalse(rb.integrity_equal(math.nan, math.nan))
        self.assertTrue(rb.integrity_equal('inf', 'inf'))
        self.assertFalse(rb.integrity_equal('100.0', '30.0'))
        self.assertTrue(rb.integrity_equal(7, 7))
        self.assertFalse(rb.integrity_equal(7, 8))
        self.assertFalse(rb.integrity_equal(True, 1))
        self.assertTrue(rb.integrity_equal(None, None))
        self.assertFalse(rb.integrity_equal(None, 0.0))

    def test_compare_documents(self):
        expected = {'a': {'x': 1.0, 'n': 3}, 'sel': 'inf', 'ci': [0.1, 0.2]}
        got = {'a': {'x': 1.0 + 5e-13, 'n': 3, 'extra': 2.0}, 'sel': 'inf', 'ci': [0.1, 0.2 + 2e-12]}
        compared, not_comparable, mismatched, max_delta = rb.compare_documents(got, expected)
        self.assertEqual(compared, ['a.n', 'a.x', 'ci.[0]', 'ci.[1]', 'sel'])
        self.assertEqual(not_comparable, ['a.extra'])
        self.assertEqual(mismatched, ['ci.[1]'])
        self.assertGreater(max_delta, 1e-12)
        _, _, missing, _ = rb.compare_documents({'a': {'x': 1.0}}, expected)
        self.assertIn('a.n', missing)

    def test_condition_ii_per_bootstrap(self):
        base = {'game': 1.0, 'pitcher': 1.0}
        self.assertTrue(rb.condition_ii({'game': 0.69, 'pitcher': 0.70}, base)['pass'])
        out = rb.condition_ii({'game': 0.69, 'pitcher': 0.7001}, base)
        self.assertFalse(out['pass'])
        self.assertEqual(out['per_bootstrap'], {'game': True, 'pitcher': False})
        self.assertEqual(rb.condition_ii({'game': None, 'pitcher': 0.5}, base)['outcome'], 'INVALID')
        self.assertEqual(rb.condition_ii({'game': 0.0, 'pitcher': 0.5}, base)['outcome'], 'INCONCLUSIVE')

    def test_tab_reuse_rule(self):
        self.assertTrue(rb.tab_reuse(10131)['reuse'])
        self.assertTrue(rb.tab_reuse(12157)['reuse'])  # +20.0%
        self.assertFalse(rb.tab_reuse(12158)['reuse'])
        self.assertTrue(rb.tab_reuse(8105)['reuse'])
        self.assertFalse(rb.tab_reuse(8104)['reuse'])
        self.assertEqual(rb.tab_reuse(8104)['rerun_dataset_pas'], 8104)

    def test_exact_value_cell_rule(self):
        self.assertEqual(rb.exact_cell_outcome(1.0e-5)['outcome'], 'EXACT_PASS')
        self.assertEqual(rb.exact_cell_outcome(1.3e-5)['outcome'], 'EXACT_FAIL')
        self.assertEqual(rb.exact_cell_outcome(2.0e-5, beta_0=0.9e-5)['outcome'], 'EXACT_PASS')  # relative bound
        self.assertEqual(rb.exact_cell_outcome(-2.2e-5, beta_0=0.9e-5)['outcome'], 'EXACT_FAIL')
        self.assertIn('EXACT_PASS', rb.PASSING)
        # the inherited rule keeps a degenerate SE = 0 INCONCLUSIVE (why the exact rule exists)
        self.assertEqual(rb.cell_outcome(1e-6, 0.0)['outcome'], 'INCONCLUSIVE')

    def test_window_and_hang_guards(self):
        self.assertTrue(rb.window_ok(kst(10, 13, 30), 'A2b-L'))  # 13:30 + 4 h
        self.assertTrue(rb.window_ok(kst(11, 2, 29), 'A2b-L'))  # ends 06:29
        self.assertFalse(rb.window_ok(kst(11, 2, 31), 'A2b-L'))  # would end after 06:30
        self.assertFalse(rb.window_ok(kst(11, 7, 0), 'A2b-S3'))  # inside the protection window
        self.assertFalse(rb.window_ok(kst(11, 13, 0), 'A2b-G'))  # 13:00 + 600 s overlaps the window
        self.assertTrue(rb.window_ok(kst(11, 6, 0), 'A2b-G'))  # 06:00 + 10 min ends before 06:30
        self.assertTrue(rb.window_ok(kst(11, 5, 59), 'A2b-S2a'))  # 1,800 s per run
        self.assertFalse(rb.window_ok(kst(11, 6, 1), 'A2b-S2a'))
        self.assertEqual(rb.guard_seconds('A2b-S2a'), 1800)
        self.assertEqual(rb.guard_seconds('A2b-L'), 14400)
        with self.assertRaises(IntegrityError):
            rb.guard_seconds('A2b-S2b')  # not applicable (D192)

    def test_s2_prime_cost_gate(self):
        ok = rb.s2p_cost_gate('2*kappa_p', 867.8)
        self.assertTrue(ok['continue'])
        self.assertEqual(ok['limit_seconds'], 2 * 55 * 433.9)
        stop = rb.s2p_cost_gate('2*kappa_p', 867.9)
        self.assertFalse(stop['continue'])
        self.assertIn('decision needed', stop['action'])
        self.assertTrue(rb.s2p_cost_gate('2*zeta_p', 500.0)['continue'])
        self.assertFalse(rb.s2p_cost_gate('2*zeta_p', 900.0)['continue'])
        self.assertTrue(rb.smoke_gate(3500.0, 'A2b-S3'))
        self.assertFalse(rb.smoke_gate(3601.0, 'A2b-S3'))


class PublicSchemaTests(unittest.TestCase):
    def test_rows_and_new_items_are_typed(self):
        doc = {'experiment': 'EXP-OPE-RES-02', 'stage': 'A2b-S3', 'r3_status': 'DIAGNOSTIC_ONLY:DECIDED',
               'pairs': {r: {'H_point': {'game': 1e-4, 'pitcher': 2e-4}} for r in rb.ROWS},
               'S5': {'R1_axR2_0xB2': {'game': 'PASS', 'pitcher': 'FAIL'}},
               'B2': {'validity': {'pass': True, 'decisions_checked': 12},
                      'R5_g': {'candidate': {'mean_rho_logged_ci95': [-0.2, 1.1]}},
                      'kl_b2_vs_b0': {'mean': 0.05}, 'E5': {'pass': True, 'thresholds': {'pa': 100, 'game': 30}}},
               'flags': {'B2_PRECISION_PASS_DIAGNOSTIC': False}}
        self.assertTrue(rb.validate_public(doc))
        leaks = {'center': lambda d: d['pairs'][rb.BASELINE].__setitem__('center', 1e-4),
                 'signed width': lambda d: d['pairs'][rb.BASELINE]['H_point'].__setitem__('game', -1.0),
                 'unknown row': lambda d: d['pairs'].__setitem__('R1_cxR2_0xB0', {'H_point': {'game': 1.0}}),
                 'selected lambda': lambda d: d['B2'].__setitem__('selected_lambda', 100.0),
                 'shift': lambda d: d['S5'][rb.BASELINE if False else 'R1_axR2_0xB2'].__setitem__('game', 0.001),
                 'A2 experiment id': lambda d: d.__setitem__('experiment', 'EXP-OPE-RES-01'),
                 'other r3 status': lambda d: d.__setitem__('r3_status', 'ADOPTION_ELIGIBLE')}
        for name, mutate in leaks.items():
            bad = copy.deepcopy(doc)
            mutate(bad)
            with self.subTest(name), self.assertRaises(IntegrityError):
                rb.validate_public(bad)

    def _record(self, kind):
        common = {'experiment': 'EXP-OPE-RES-02', 'result': 'PASS', 'max_abs_delta': 0.0,
                  'tolerance': {'float_absolute': 1e-12, 'counts_strings_selected_lambda_M': 'equality (inf by equality)'},
                  'compared_keys': ['a'], 'mismatched_keys': [], 'reference': {'path': 'x', 'sha256': 'y'}}
        if kind == 'gint_eval':
            return {**common, 'check': 'G-INT-EVAL', 'not_comparable_keys': [], 'rows': list(rb.S6_MODE_ROWS),
                    'refused_rows': list(rb.S6_REFUSED_ROWS)}
        return {**common, 'check': 'G-INT-S2EQ', 'run_indices': [4, 9, 10], 'cells': ['2*kappa_p'],
                'paths': {'exact_form': rb.S2P_PATH, 'statistical_form': rb.S2P_PATH, 'new_seed_runs': rb.S2P_PATH},
                'auxiliary_statistical_form': {'result': 'PASS', 'detection_limit_B': {'k4': {'R1_b': 7.8}},
                                               'decision_input': False}}

    def test_integrity_records_hold_only_their_type(self):
        for kind in ('gint_eval', 'gint_s2eq'):
            record = self._record(kind)
            self.assertTrue(rb.validate_integrity_record(record, kind))
            for mutate in (lambda d: d.__setitem__('center', 1e-4), lambda d: d.__setitem__('b_hat', 1e-5),
                           lambda d: d.__setitem__('tolerance', {'float_relative': 1e-12}),
                           lambda d: d.__setitem__('max_abs_delta', -1.0)):
                bad = copy.deepcopy(record)
                mutate(bad)
                with self.subTest(kind), self.assertRaises(IntegrityError):
                    rb.validate_integrity_record(bad, kind)
        bad = self._record('gint_eval')
        bad['rows'] = list(rb.ROWS)  # the 13 B2 / lambda-fixed rows are never part of the S6 record
        with self.assertRaises(IntegrityError):
            rb.validate_integrity_record(bad, 'gint_eval')

    def test_run_public_line(self):
        doc = {'experiment': 'EXP-OPE-RES-02', 'stage': 'A2b-S2a', 'cell': '2*kappa_p', 'k': 11, 'wall_seconds': 431.0,
               'peak_rss_bytes': 10, 'integrity': 'PASS', 'sealed_sha256': 'a' * 64}
        self.assertTrue(rb.validate_run_public(doc))
        with self.assertRaises(IntegrityError):
            rb.validate_run_public({**doc, 'b_hat': 1e-5})


class ConfigTests(unittest.TestCase):
    def test_synthetic_draft_passes_and_deviations_refuse(self):
        config = draft_config()
        self.assertEqual(rb.config_check(config), rb.DRAFT)
        mutations = {
            'seed role value': lambda c: c['seeds']['roles']['placebo'].__setitem__('value', 1),
            'world seed list': lambda c: c['seeds']['world_seeds_derived']['k_1_to_76'].__setitem__(3, 1),
            'X': lambda c: c['decision_rules']['condition_ii_half_width']['X'].__setitem__('value', 0.25),
            'B': lambda c: c['synthetic_bias_check']['acceptance']['B'].__setitem__('value', 1.3e-5),
            'N prime': lambda c: c['synthetic_bias_check']['S2_prime']['N_prime_decided'].__setitem__('kappa_2x', 60),
            'label order': lambda c: c['decision_rules']['labels_in_precedence'].reverse(),
            'total order': lambda c: c['decision_rules']['selection']['total_order']['value'].reverse(),
            'flags': lambda c: c.__setitem__('registered', True),
            'pins keys': lambda c: c['registration_input_pins'].__setitem__('extra', None),
            'nulled paths': lambda c: c['registration']['config_digest']['nulled_json_paths'].pop(),
            'self digest': lambda c: c['registration'].__setitem__('config_sha256', 'f' * 64),
            'seal filled': lambda c: c['stage_output_seals'].__setitem__('A2b-L_ledger_sha256', 'f' * 64),
            'hang guard': lambda c: c['cost_budget']['hang_guards_seconds']['A2b-L'].__setitem__('value', 20000),
            'stages': lambda c: c['stages'].pop(),
            'protected paths': lambda c: c['governing_rules']['not_modified_paths'].pop(),
        }
        for name, mutate in mutations.items():
            bad = copy.deepcopy(config)
            mutate(bad)
            with self.subTest(name), self.assertRaises(IntegrityError):
                rb.config_check(bad)
        foreign = copy.deepcopy(config)  # another config's sha256 under 'config_sha256' elsewhere is allowed
        foreign['a3_record']['config_sha256'] = {'value': 'e' * 64}
        self.assertEqual(rb.config_check(foreign), rb.DRAFT)

    def test_draft_refuses_execution_fail_closed(self):
        with self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
            rb.require_executable(draft_config())

    def test_registered_pins_and_d192_single_values(self):
        root = Path(tempfile.mkdtemp(prefix='a2b-reg-')).resolve()
        config = registered_config(root)
        self.assertEqual(rb.config_check(config), rb.REGISTERED)
        self.assertIs(rb.require_executable(config), config)
        checks = {
            'S6 reuse': lambda p: p.__setitem__('rehearsal_population', 'S6_150_REUSE'),
            'r3 eligible': lambda p: p.__setitem__('r3_status', 'ADOPTION_ELIGIBLE'),
            'N prime': lambda p: p.__setitem__('s2_prime_runs', {'zeta_2x': 5, 'kappa_2x': 36, 'descriptive_kappa_prime': 5}),
            'new files': lambda p: p['new_files_sha256'].pop(rb.NEW_FILES[0]),
            'aligned source': lambda p: p['processed_frame_identity'].__setitem__('sha256', 'd' * 64),
            'chain': lambda p: p['policy_registration_chain'].pop(),
            'row budget': lambda p: p['runtime_setting'].__setitem__('row_budget', 1),
            'style as-of': lambda p: p['dev_style_snapshot'].__setitem__('as_of_exclusive', '2025-06-01'),
            'lightgbm': lambda p: p['b2_inference_environment'].__setitem__('lightgbm', '4.5.0'),
            'output root in repo': lambda p: p.__setitem__('output_root', str(rb.REPO / 'outputs')),
            'results pin': lambda p: p['a2_results_file'].__setitem__('sha256', 'd' * 64),
            'smoke subset': lambda p: p['smoke_train_subset'].__setitem__('game_dates', ['2025-06-01', '2025-06-02']),
        }
        for name, mutate in checks.items():
            bad = copy.deepcopy(config)
            mutate(bad['registration_input_pins'])
            with self.subTest(name), self.assertRaises(IntegrityError):
                rb.config_check(bad)
        missing = copy.deepcopy(config)
        missing['registration']['codex_reconfirmation_reference'] = None  # Codex re-confirmation required (D192 6 1)
        with self.assertRaisesRegex(IntegrityError, 'codex_reconfirmation_reference'):
            rb.require_executable(missing)

    def test_digest_nulls_exactly_the_registered_paths(self):
        config = draft_config()
        digest = rb.config_digest(config)
        same = copy.deepcopy(config)
        same.update(status=rb.REGISTERED, registered=True, preregistration_complete=True)
        same['execution'] = {'enabled': True}
        same['registration'].update(registered_kst='x', review_reference='y', codex_reconfirmation_reference='z',
                                    code_review_reference='w', code_review_phase2_reference='v')
        for gate in same['pre_run_gates']:
            gate['status'] = 'PASS'
        same['stage_output_seals']['A2b-L_ledger_sha256'] = 'f' * 64
        self.assertEqual(rb.config_digest(same), digest)
        other = copy.deepcopy(config)
        other['registration']['target'] = 'configs/other.json'
        self.assertNotEqual(rb.config_digest(other), digest)


if __name__ == '__main__':
    unittest.main()
