"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-01 stage runner: every real-data command refuses on the draft
config (fail closed), the external append-only digest manifest and stage seals, output location rules,
the run-window, hang and memory guards with a mocked clock, attempt accounting, and the A2-S0 / A2-S1 /
smoke / A2-S4 code paths on synthetic TRAIN rows and synthetic worlds. No real data is read.
"""
import copy
from datetime import datetime
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

from ope_a2_fixtures import ToyAdvancement, ToyWE, synthetic_train_frame  # noqa: E402
from pitchmdp import ope_redesign_registry as reg  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402
import run_ope_a2 as roa  # noqa: E402
from test_ope_a2_estimators import _decision_inputs  # noqa: E402
from test_ope_a2_registry import registered  # noqa: E402

GAME_VALUES = {'we': ToyWE(), 'advancement': ToyAdvancement()}
AFTERNOON = lambda: datetime(2026, 10, 27, 14, 0, tzinfo=reg.KST)


class FailClosedTests(unittest.TestCase):
    def test_every_real_data_command_refuses_the_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            for command in ('smoke', 's0', 's1', 's2', 's3', 'decide', 'unblind', 'register-digest'):
                argv = [command, '--output', str(Path(tmp) / command), '--expect-digest', 'x']
                with self.subTest(command), self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                    roa.main(argv)
            with self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                roa.main(['register-addendum', '--stage', 'A2-S0', '--failing-test', 't', '--review', 'r'])
            with self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                roa.main(['record-review', '--gate', 'G-CODE', '--review', 'r'])
            self.assertEqual(list(Path(tmp).iterdir()), [])  # nothing was created

    def test_data_free_commands(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(roa.main(['check']), 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report['status'], reg.DRAFT)
        self.assertEqual(report['config_digest'], reg.config_digest(json.loads((reg.REPO / reg.CONFIG_PATH).read_text())))
        out = io.StringIO()
        with redirect_stdout(out):
            roa.main(['pins'])
        self.assertEqual(set(json.loads(out.getvalue())['new_files_sha256']), set(reg.NEW_FILES))

    def test_x2_identity_must_name_a_supported_loader(self):
        with self.assertRaisesRegex(IntegrityError, 'X2'):
            roa.x2_world_predictor({'kind': 'some_member'})
        self.assertTrue(callable(roa.x2_world_predictor({'kind': 'g0_single_member', 'seed': 2})))


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = roa.DigestManifest(self.tmp / 'digest.jsonl')

    def test_append_only_chain_and_tamper_detection(self):
        a = self.manifest.append('config_digest', {'digest': 'a' * 64})
        b = self.manifest.append('smoke_record', {'record_sha256': 'b' * 64})
        self.assertEqual([e['seq'] for e in self.manifest.entries()], [0, 1])
        self.assertEqual(b['prev'], a['sha256'])
        self.assertEqual(self.manifest.latest('config_digest')['payload']['digest'], 'a' * 64)
        lines = (self.tmp / 'digest.jsonl').read_text().splitlines()
        (self.tmp / 'digest.jsonl').write_text(lines[0].replace('aaaa', 'aaab', 1) + '\n' + lines[1] + '\n')
        with self.assertRaisesRegex(IntegrityError, 'tampered'):
            self.manifest.entries()
        with self.assertRaisesRegex(IntegrityError, 'unknown manifest entry'):
            roa.DigestManifest(self.tmp / 'x.jsonl').append('note', {})
        with self.assertRaisesRegex(IntegrityError, 'outside the repository results'):
            roa.DigestManifest(reg.REPO / 'results' / 'digest.jsonl')

    def test_config_digest_must_match(self):
        config = json.loads((reg.REPO / reg.CONFIG_PATH).read_text())
        with self.assertRaisesRegex(IntegrityError, 'config digest'):
            roa.require_config_digest(self.manifest, config)
        self.manifest.append('config_digest', {'digest': reg.config_digest(config)})
        roa.require_config_digest(self.manifest, config)
        changed = copy.deepcopy(config)
        changed['task'] += '!'
        with self.assertRaisesRegex(IntegrityError, 'config digest'):
            roa.require_config_digest(self.manifest, changed)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.manifest = roa.DigestManifest(self.tmp / 'digest.jsonl')
        self.identity = {'config_digest': 'c' * 64, 'git_commit': 'd' * 40, 'seeds': {'a2_base': reg.A2_BASE}}

    def test_success_writes_sums_manifest_and_seal(self):
        out = self.tmp / 'A2-S1-a1'
        with roa.stage(out, 'A2-S1', self.identity, manifest=self.manifest, now=AFTERNOON, alarm=False) as d:
            roa.dump(d / 'tab.json', {'cells': {}})
        manifest = json.loads((out / 'manifest.json').read_text())
        self.assertEqual(set(manifest), {'stage', 'cost', 'artifact_sha256', 'config_digest', 'git_commit', 'seeds'})
        self.assertEqual(set(manifest['cost']), {'wall_seconds', 'cpu_seconds', 'peak_rss_bytes'})
        self.assertIn('SHA256SUMS', manifest['artifact_sha256'])
        sums = (out / 'SHA256SUMS').read_text().splitlines()
        self.assertTrue(any(line.endswith('  tab.json') for line in sums))
        sealed, entry = roa.require_seal(self.manifest, 'A2-S1')
        self.assertEqual(sealed, out)
        (out / 'tab.json').write_text('{"cells": {"x": 1}}\n')
        with self.assertRaisesRegex(IntegrityError, 'differs from its seal'):
            roa.require_seal(self.manifest, 'A2-S1')

    def test_window_refusal_creates_nothing(self):
        early = lambda: datetime(2026, 10, 28, 5, 30, tzinfo=reg.KST)
        with self.assertRaisesRegex(IntegrityError, 'protection window'):
            with roa.stage(self.tmp / 'x', 'A2-S0', self.identity, manifest=self.manifest, now=early, alarm=False):
                pass
        self.assertFalse((self.tmp / 'x').exists())
        self.assertEqual(self.manifest.entries(), [])

    def test_failures_are_classified_and_recorded(self):
        with self.assertRaises(IntegrityError):
            with roa.stage(self.tmp / 'f1', 'A2-S2', self.identity, manifest=self.manifest, now=AFTERNOON, alarm=False):
                raise IntegrityError('synthetic deterministic defect')
        failure = json.loads(next((self.tmp / 'f1').glob('failure-*.json')).read_text())
        self.assertEqual((failure['class'], failure['citable']), ('deterministic_integrity', False))
        with self.assertRaises(roa.HangGuardExceeded):
            with roa.stage(self.tmp / 'f2', 'A2-S2', self.identity, manifest=self.manifest, now=AFTERNOON,
                           guard_seconds=0.05):
                time.sleep(1.0)
        failure = json.loads(next((self.tmp / 'f2').glob('failure-*.json')).read_text())
        self.assertEqual(failure['class'], 'infrastructure')
        kinds = [(e['kind'], e['payload']['class']) for e in self.manifest.entries()]
        self.assertEqual(kinds, [('stage_failure', 'deterministic_integrity'), ('stage_failure', 'infrastructure')])
        with self.assertRaisesRegex(IntegrityError, 'at most two attempts'):
            roa.attempt_gate(self.manifest, 'A2-S2')

    def test_attempt_gate(self):
        roa.attempt_gate(self.manifest, 'A2-S0')
        self.manifest.append('stage_failure', {'stage': 'A2-S0', 'class': 'deterministic_integrity'})
        with self.assertRaisesRegex(IntegrityError, 'bug-fix addendum'):
            roa.attempt_gate(self.manifest, 'A2-S0')
        self.manifest.append('bugfix_addendum', {'stage': 'A2-S0', 'fix_commit': 'e' * 40, 'new_files_sha256': {}})
        roa.attempt_gate(self.manifest, 'A2-S0')
        self.manifest.append('stage_failure', {'stage': 'A2-S0', 'class': 'deterministic_integrity'})
        with self.assertRaisesRegex(IntegrityError, 'FAILED_INTEGRITY'):
            roa.attempt_gate(self.manifest, 'A2-S0')
        self.assertEqual(roa.failed_integrity(self.manifest), ['A2-S0'])
        self.manifest.append('stage_seal', {'stage': 'A2-S1', 'output_dir': str(self.tmp), 'manifest_sha256': 'f' * 64})
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            roa.attempt_gate(self.manifest, 'A2-S1')
        config = registered()
        pins = roa.code_pins(config, self.manifest)
        self.assertEqual(pins['execution_source_commit'], 'e' * 40)  # the recorded fix replaces the code identity

    def test_smoke_gate(self):
        with self.assertRaisesRegex(IntegrityError, 'no smoke record'):
            roa.smoke_gate(self.manifest, 'A2-S0')
        self.manifest.append('smoke_record', {'record_sha256': 'a' * 64,
                                              'projection': {'within_guard': {'A2-S0': True, 'A2-S1': False}}})
        roa.smoke_gate(self.manifest, 'A2-S0')
        with self.assertRaisesRegex(IntegrityError, 'decision needed'):
            roa.smoke_gate(self.manifest, 'A2-S1')

    def test_output_location(self):
        config = registered()
        config['registration_input_pins']['output_root'] = str(self.tmp)
        self.assertEqual(roa.check_output(self.tmp / 'A2-S0-a1', config), (self.tmp / 'A2-S0-a1').resolve())
        with self.assertRaisesRegex(IntegrityError, 'directly under'):
            roa.check_output(self.tmp / 'a' / 'b', config)
        (self.tmp / 'exists').mkdir()
        with self.assertRaisesRegex(IntegrityError, 'directly under'):
            roa.check_output(self.tmp / 'exists', config)
        with self.assertRaisesRegex(IntegrityError, 'results/'):
            roa.check_output(reg.REPO / 'results' / 'a2', config)

    def test_guards(self):
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.Guards(rss=lambda: 9 * roa.GIB, free=lambda: 10 * roa.GIB)()
        with self.assertRaises(roa.MemoryGuardExceeded):
            roa.Guards(rss=lambda: roa.GIB, free=lambda: 3 * roa.GIB)()
        roa.Guards(rss=lambda: roa.GIB, free=lambda: None)()
        self.assertGreater(roa.peak_rss_bytes(), 0)

    def test_enforce_sources_with_a_mocked_git_state(self):
        config = registered()
        config['registration_input_pins']['new_files_sha256'] = roa.new_file_hashes()
        ok = {'head': 'a' * 40, 'ancestor': True, 'changed': [], 'dirty': False, 'config_committed': True}
        roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state=ok)
        for key, value in (('changed', ['experiments/x.py']), ('ancestor', False), ('dirty', True),
                           ('config_committed', False)):
            with self.subTest(key), self.assertRaises(IntegrityError):
                roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state={**ok, key: value})
        config['registration_input_pins']['new_files_sha256'][reg.NEW_FILES[0]] = '0' * 64
        with self.assertRaisesRegex(IntegrityError, 'registration pins'):
            roa.enforce_sources(config, reg.REPO / reg.CONFIG_PATH, state=ok)


class StageLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic_train_frame()

    def test_s0_full_and_smoke(self):
        s0 = roa.build_s0(self.train, GAME_VALUES, min_pitches=100)
        self.assertEqual(s0['r1_table_sha256'], roa.osg.table_sha256(s0['r1_table'], s0['m']))
        self.assertIn(s0['m'], reg.M_SMOOTH_GRID)
        self.assertEqual(set(s0['kappa_p']), {'S5X', 'TAB'})
        self.assertTrue(all(v > 0 for v in s0['kappa_p'].values()))
        self.assertEqual(set(s0['train_residuals']['hit']), set(reg.S5X_GROUPS))
        marginals = roa.marginals_from_doc(json.loads(json.dumps(s0['tab_marginals'], default=roa._default)))
        self.assertEqual(len(marginals.situations), reg.TAB_SITUATIONS)
        smoke = roa.build_s0(self.train, GAME_VALUES, smoke=True, seed_role='smoke', fixed_m=20, min_pitches=50)
        self.assertIsNone(smoke['cv_log_loss'])
        for key in ('train_residuals', 'kappa_calibration', 'zeta_calibration'):
            self.assertNotIn(key, smoke)
        self.assertEqual((smoke['kappa_p'], smoke['zeta_p']), ({'S5X': 0.0, 'TAB': 0.0}, {'S5X': 0.0, 'TAB': 0.0}))
        bad = self.train.copy()
        bad.loc[0, 'game_date'] = '2026-04-01'
        with self.assertRaisesRegex(IntegrityError, '2026'):
            roa.build_s0(bad, GAME_VALUES, min_pitches=100)

    def test_smoke_records_costs_and_never_a_metric(self):
        smoke = roa.build_s0(self.train, GAME_VALUES, smoke=True, seed_role='smoke', fixed_m=20, min_pitches=50)
        tab = roa.smoke_tab(smoke, replications=2)
        self.assertEqual(tab['integrity'], {'tab_v2_dp_expectation_equals_truth': 'PASS', 'same_seed_bit_reproducible': 'PASS'})
        projection = roa.smoke_projection(1.0, smoke['timings']['table_fit'], tab['variant_estimate_unit_seconds'],
                                          tab['replication_seconds'], 10.0)
        self.assertEqual(projection['label'], 'projection (not a measurement)')
        self.assertAlmostEqual(projection['projection_term_seconds_per_replication'],
                               6 * smoke['timings']['table_fit'] + 30 * tab['variant_estimate_unit_seconds'], places=12)
        text = json.dumps({**tab, **projection})
        for word in ('b_hat', 'rmse', 'H_point', '"sd"', 'gap', 'delta_true', 'log_loss'):
            self.assertNotIn(word, text)

    def test_s1_cells_order_and_not_run(self):
        doc = roa.build_s0(self.train, GAME_VALUES, min_pitches=100)
        doc = json.loads(json.dumps(doc, default=roa._default))
        cells = [reg.tab_cells()[0], reg.tab_cells()[5], reg.tab_cells()[30]]
        written = []
        results = roa.stage_s1(doc, write_cell=lambda cell, record: written.append(cell['id']), cells=cells, replications=2)
        self.assertEqual(written, [c['id'] for c in cells])
        self.assertEqual(set(results['first_step_v2_dp_check']), {c['id'] for c in cells if c['law'] == 'V2'})
        self.assertTrue(all(abs(v['expected_delta_hat_0_minus_truth']) <= 1e-12
                            for v in results['first_step_v2_dp_check'].values()))
        self.assertEqual(set(results['acceptance']), {'T-A1'})
        calls = {'n': 0}

        def deadline():
            calls['n'] += 1
            if calls['n'] > 2:  # the guard trips during the descriptive cells
                raise roa.HangGuardExceeded('synthetic')
        results = roa.stage_s1(doc, write_cell=lambda cell, record: None, cells=cells, replications=2, deadline=deadline)
        self.assertEqual(results['not_run'], [cells[1]['id'], cells[2]['id']])
        calls['n'] = 0
        with self.assertRaises(roa.HangGuardExceeded):  # before the acceptance cells are sealed: infrastructure
            roa.stage_s1(doc, write_cell=lambda cell, record: None, cells=cells, replications=4, deadline=deadline)

    def test_s4_decision(self):
        public, tab, s5x = _decision_inputs(h={'R1_axR2_3': 0.6})
        record = roa.stage_s4(public, tab, s5x, integrity_failures=(), r3_status='NOT_RUN:R3')
        self.assertEqual(record['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_3')
        self.assertEqual(record['r3_status'], 'NOT_RUN:R3')
        del tab['acceptance']['T-A2'][reg.BASELINE]
        with self.assertRaisesRegex(IntegrityError, 'T-A2'):
            roa.stage_s4(public, tab, s5x, integrity_failures=(), r3_status='NOT_RUN:R3')


class CommittedTests(unittest.TestCase):
    def test_files_outside_the_repository_are_never_committed(self):
        with tempfile.NamedTemporaryFile() as handle:
            self.assertFalse(roa.committed(handle.name))


if __name__ == '__main__':
    unittest.main()
