"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 stage runner: fail-closed refusals of a draft, the run window and the
A2b hang guards, the registration gate (digest and review receipts), the gate command on a synthetic sealed public
file, G-GATE (phase 2 and the decision need the gate record committed at HEAD byte for byte), the S-2' order, the
G-INT-S2EQ precondition and the cost gate, operator-blind pooling, the decision and the committed-record unblind,
and the manifest commands. Every config, manifest and stage directory is synthetic and lives in a temporary
directory; no real data, ledger or model is read (real-data hooks raise in the test environment).
"""
from contextlib import nullcontext
import argparse
from unittest import mock
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'scripts'))

from ope_a2b_fixtures import (a2_nets, a2_tab, draft_config, pooled, public_doc, registered_config,  # noqa: E402
                              synthetic_a2_s0)
import run_ope_a2 as roa  # noqa: E402
import run_ope_a2b as rr  # noqa: E402
from pitchmdp import ope_decision_a2b as dc  # noqa: E402
from pitchmdp import ope_registry_a2b as rb  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

GOOD = (0.6e-3, 0.65e-3)


def kst(day, hour, minute=0):
    return datetime(2026, 11, day, hour, minute, tzinfo=rb.KST)


AFTERNOON = kst(16, 14, 0)


def tmpdir(test):
    path = Path(tempfile.mkdtemp(prefix='a2b-runner-')).resolve()
    test.addCleanup(shutil.rmtree, path, True)
    return path


class ClockEnv(rr.Environment):
    """No data: a fixed clock, no SIGALRM, no memory sampling; every real-data hook raises."""
    alarm = False

    def __init__(self, now=AFTERNOON):
        super().__init__()
        self._now, self.clock = now, 0.0

    def now(self):
        return self._now

    def monotonic(self):
        return self.clock

    def guards(self):
        return lambda: None

    def _real(self, *args, **kwargs):
        raise AssertionError('a real-data hook was called by a synthetic test')
    s5x_inputs = rehearsal_inputs = s6_record = policy_registration = ledger_world = b2_dev_table = _real
    train = train_plan = a2_sealed = a2_results = _real


class WorkEnv(ClockEnv):
    def __init__(self, ws, now=AFTERNOON):
        super().__init__(now)
        self.ws, self.committed_paths = ws, set()

    def git_state(self, config_path, code):
        return {'head': code['execution_source_commit'], 'ancestor': True, 'changed': [], 'dirty': False,
                'config_committed': True}

    def chain_documents(self, config):
        return [{'le2025_validation_plan': {'local_config': {'sha256': 'synthetic'}}}]

    def local_config(self, base):
        return {'artifact_root': str(self.ws.root / 'artifacts'), 'raw_root': str(self.ws.root / 'raw')}

    def protected_roots(self, config, local, chain):
        return {'the repository': rr.REPO, 'the synthetic inputs': self.ws.root / 'inputs'}

    def heavy_lock(self, local):
        return nullcontext()

    def nice(self):
        pass

    def committed(self, path):
        return Path(path).resolve() in self.committed_paths


class Workspace:
    """A synthetic REGISTERED config with the real new-file hashes, a digest manifest with the config digest and
    the G-CODE / G-REG receipts, and helpers to seal synthetic upstream stages."""
    def __init__(self, test, *, draft=False, pins=None, identities=None):
        self.test = test
        self.root = tmpdir(test)
        (self.root / 'inputs').mkdir()
        (self.root / 'outputs').mkdir()
        config = draft_config() if draft else registered_config(self.root)
        if not draft:
            config['registration_input_pins']['new_files_sha256'] = rr.new_file_hashes()
            config['registration_input_pins'].update(pins or {})
            config['identities'].update(identities or {})
        self.config = config
        self.config_path = self.root / 'EXP-OPE-RES-02.json'
        self.config_path.write_text(json.dumps(config, indent=1))
        self.env = WorkEnv(self)
        if not draft:
            self.manifest = rr.A2bManifest(config['registration_input_pins']['digest_manifest_path'])
            digest = rb.config_digest(config)
            self.manifest.append('config_digest', {'digest': digest, 'config_path': str(self.config_path)})
            for gate in ('G-CODE', 'G-REG'):
                self.manifest.append('review_reference', {'gate': gate, 'reference': f'{gate}-SYN', 'verdict': 'PASS',
                                                          'notes_closed': True, 'commit': 'c' * 40, 'config_digest': digest})

    def output(self, name):
        return self.root / 'outputs' / name

    def argv(self, command, name, *extra):
        return [command, '--config', str(self.config_path), '--output', str(self.output(name)), *extra]

    def seal(self, stage_id, files):
        out = self.output(f'sealed-{stage_id.replace(":", "_").replace("*", "x")}')
        with roa.stage(out, stage_id, {'config_digest': rb.config_digest(self.config)}, manifest=self.manifest,
                       final_check=lambda: None):
            for name, value in files.items():
                (out / name).write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return out

    def commit(self, path):
        self.env.committed_paths.add(Path(path).resolve())


def public_bytes(doc):
    doc = {**doc, 'sealed_sha256': 'a' * 64}
    return (json.dumps(doc, sort_keys=True, indent=1) + '\n').encode()


def run_of(ws, command, **kw):
    args = argparse.Namespace(command=command, gate_record=kw.get('gate_record'), cell=kw.get('cell'), k=kw.get('k'),
                              decision_record=kw.get('decision_record'))
    return rr.Run(args, ws.env, ws.config, {}, {}, ws.manifest, lambda: None)


class FailClosedTests(unittest.TestCase):
    def test_every_real_data_and_manifest_command_refuses_the_draft(self):
        ws = Workspace(self, draft=True)
        extra = {'s2a': ('--k', '11'), 's2a-gint': ('--cell', '2*kappa_p', '--k', '4'),
                 's2a-gint-record': ('--cell', '2*kappa_p'), 's2a-pool': ('--cell', '2*zeta_p')}
        for command in ('smoke', 's0', 'ledger', 's3-r3', 's3', 'gate', 's2a', 's2a-gint', 's2a-gint-record', 's2a-pool',
                        's1', 'decide', 'unblind', 'register-digest', 'record-review', 'register-addendum',
                        'reconcile-attempt'):
            with self.subTest(command), self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
                rr.main(ws.argv(command, command, *extra.get(command, ())), ws.env)

    def test_data_free_commands(self):
        ws = Workspace(self, draft=True)
        import contextlib
        import io
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(rr.main(['check', '--config', str(ws.config_path)], ws.env), 0)
        self.assertEqual(json.loads(buffer.getvalue())['status'], rb.DRAFT)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            rr.main(['pins', '--config', str(ws.config_path)], ws.env)
        pins = json.loads(buffer.getvalue())
        self.assertEqual(set(pins['new_files_sha256']), set(rb.NEW_FILES))
        self.assertIn('a3_weight_settings', pins['added_to_the_draft_pin_keys'])
        self.assertTrue({'g0_bundle_sha256', 'a2_execution_source_commit', 'a3_reader_commit',
                         'x2_world_predictor_identity', 'smoke_train_subset', 'processed_frame_aligned'}
                        <= set(pins['record_only_pins']))
        self.assertTrue(all(v.startswith('record-only') for v in pins['record_only_pins'].values()))
        self.assertTrue({'runtime_setting', 'dev_style_snapshot', 'b2_inference_environment'}
                        <= set(pins['pins_compared_at_run']))
        self.assertEqual(set(pins['registration_time_pins']), {'ml_policy_d127_chain', 's2_prime_runs',
                                                               'rehearsal_population'})  # NOTE-D
        classes = ('pins_compared_at_run', 'registration_time_pins', 'record_only_pins')
        self.assertEqual(sorted(k for c in classes for k in pins[c]),
                         sorted(pins['registration_input_pins_expected_keys']))

    def test_the_select_helper_takes_the_heavy_lock_and_nice(self):
        """G-CODE MINOR-2: the G-REG selection helper (draft config allowed) loads data only under os.nice(10) and the
        common heavy lock, inside the run window."""
        import contextlib
        import io
        from test_ope_a2b_ledger import selection_frame
        import run_policy_validation as rpv
        ws = Workspace(self, draft=True)
        calls = []

        class LockEnv(WorkEnv):
            def nice(self):
                calls.append('nice')

            @contextlib.contextmanager
            def heavy_lock(self, local):
                calls.append('lock')
                yield
                calls.append('unlock')

            def ledger_world(self, config, local, base, checkpoint):
                calls.append('data')
                frame = selection_frame()
                s6 = rpv.select_games(frame, 'dev', 6, rb.S6_SELECTION_SALT_RECOMPUTED)
                return rr.LedgerWorld(rpv, {}, {'frame': frame}, s6, frozenset(), {})
        env = LockEnv(ws)
        buffer = io.StringIO()
        with mock.patch.object(rr.lg, 'N_GAMES', 6), contextlib.redirect_stdout(buffer):
            self.assertEqual(rr.main(['select', '--config', str(ws.config_path)], env), 0)
        self.assertEqual(calls, ['nice', 'lock', 'data', 'unlock'])
        self.assertEqual(set(json.loads(buffer.getvalue())), {'selection_s6_game_list_sha256', 'a2b_new_game_list_sha256',
                                                              'month_quota_table', 'straddling_games_excluded', 'salts'})
        env._now = kst(17, 7, 0)  # inside the protection window: refused before any data
        calls.clear()
        with self.assertRaisesRegex(IntegrityError, 'protection window'):
            rr.main(['select', '--config', str(ws.config_path)], env)
        self.assertEqual(calls, [])

    def test_window_and_hang_guards_refuse_before_preflight(self):
        ws = Workspace(self, draft=True)
        ws.env._now = kst(17, 6, 45)
        with self.assertRaisesRegex(IntegrityError, 'protection window'):
            rr.main(ws.argv('smoke', 'w1'), ws.env)
        ws.env._now = kst(17, 2, 31)
        with self.assertRaisesRegex(IntegrityError, 'protection window'):
            rr.main(ws.argv('ledger', 'w2'), ws.env)  # 14,400 s would end after 06:30
        ws.env._now = kst(17, 2, 29)
        with self.assertRaisesRegex(IntegrityError, 'not REGISTERED'):
            rr.main(ws.argv('ledger', 'w3'), ws.env)
        env = ClockEnv()
        self.assertEqual(rr.CommandGuard('A2b-L', env).seconds, 14400)
        self.assertEqual(rr.CommandGuard('A2b-S2a:k11', env).seconds, 1800)
        self.assertEqual(rr.CommandGuard('A2b-S2a:gint:2*kappa_p', env).seconds, 1800)
        self.assertEqual(rr.CommandGuard('A2b-G', env).seconds, 600)
        self.assertEqual(rr.CommandGuard('A2b-S3', env).seconds, 3600)

    def test_attempt_stage_ids_and_arguments(self):
        self.assertEqual(rr.attempt_stage(rr.parse(['s2a', '--k', '11'])), 'A2b-S2a:k11')
        self.assertEqual(rr.attempt_stage(rr.parse(['s2a-gint', '--cell', '2*kappa_p', '--k', '9'])),
                         'A2b-S2a:gint:2*kappa_p:k9')
        self.assertEqual(rr.attempt_stage(rr.parse(['s2a-gint-record', '--cell', '2*kappa_p'])), 'A2b-S2a:gint:2*kappa_p')
        self.assertEqual(rr.attempt_stage(rr.parse(['s3-r3'])), 'A2b-S3:R3')
        with self.assertRaises(IntegrityError):
            rr.parse(['s2a-gint', '--cell', '2*kappa_p', '--k', '5'])  # k = 5 belongs to the 2 zeta_p family
        self.assertEqual(rr.attempt_stage(rr.parse(['s2a-pool', '--cell', '2*zeta_p'])), 'A2b-S2a:pool:2*zeta_p')
        self.assertEqual(rr.attempt_stage(rr.parse(['ledger'])), 'A2b-L')
        with self.assertRaises(IntegrityError):
            rr.parse(['s2a', '--k', '71'])  # 71..76 never registered
        with self.assertRaises(IntegrityError):
            rr.parse(['s2a-pool'])


class RegisteredTests(unittest.TestCase):
    def test_the_registration_gate_and_the_receipts(self):
        ws = Workspace(self)
        config, manifest, state = rr.preflight(rr.parse(ws.argv('gate', 'g0')), ws.env)
        self.assertEqual(config['status'], rb.REGISTERED)
        bad = copy.deepcopy(ws.config)
        bad['registration_input_pins']['new_files_sha256'][rb.NEW_FILES[0]] = 'f' * 64
        path = ws.root / 'bad.json'
        path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(IntegrityError, 'digest manifest|differ'):
            rr.preflight(rr.parse(['gate', '--config', str(path), '--output', str(ws.output('x'))]), ws.env)
        other = Workspace(self)
        other.manifest.append('review_reference', {'gate': 'G-CODE', 'reference': 'later', 'verdict': 'PASS_WITH_NOTES',
                                                   'notes_closed': False, 'commit': 'c' * 40,
                                                   'config_digest': rb.config_digest(other.config)})
        with self.assertRaisesRegex(IntegrityError, 'G-CODE receipt is not a closed pass'):
            rr.preflight(rr.parse(other.argv('gate', 'g1')), other.env)

    def sealed_phase_1(self, ws, widths, e0=10131):
        ws.seal('A2b-SM', {'smoke_record.json': {'stage': 'A2b-SM'}})
        ws.seal('A2b-S0', {'s0_record.json': {'stage': 'A2b-S0', 'reused': {'tab_acceptance': a2_tab(), 'nets': a2_nets()}}})
        ws.seal('A2b-L', {'ledger_record.json': {'stage': 'A2b-L', 'e0_count': e0}})
        ws.seal('A2b-S3', {'public.json': public_bytes(public_doc(widths)),
                           'centers.sealed.json': b'{"sealed": true}\n'})

    def test_gate_then_g_gate_for_phase_2(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_axR2_1xB0': GOOD})
        self.assertEqual(rr.main(ws.argv('gate', 'gate'), ws.env), 0)
        record_path = ws.output('gate') / 'gate_record.json'
        record = json.loads(record_path.read_text())
        self.assertEqual(record['gated'], ['R1_axR2_1xB0'])
        self.assertEqual(record['work_list']['S2_prime_cells'], ['2*zeta_p'])
        with self.assertRaisesRegex(IntegrityError, 'G-GATE'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', k=5, gate_record=record_path))
        copy_path = ws.root / 'committed_gate_record.json'
        copy_path.write_bytes(record_path.read_bytes())
        ws.commit(copy_path)
        plan = rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', k=5, gate_record=copy_path))
        self.assertEqual(plan['cell'], '2*zeta_p')
        with self.assertRaisesRegex(IntegrityError, 'did not open'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*kappa_p', k=4, gate_record=copy_path))
        with self.assertRaisesRegex(IntegrityError, 'did not require a TAB re-run'):
            rr.plan_phase_2(run_of(ws, 's1', gate_record=copy_path))
        tampered = ws.root / 'tampered.json'
        tampered.write_bytes(record_path.read_bytes().replace(b'R1_axR2_1xB0', b'R1_bxR2_1xB0', 1))
        ws.commit(tampered)
        with self.assertRaisesRegex(IntegrityError, 'G-GATE'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', k=5, gate_record=tampered))
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            rr.main(ws.argv('gate', 'gate-again'), ws.env)

    def test_s2_prime_order_precondition_and_cost_gate(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_bxR2_1xB0': GOOD})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            rr.plan_phase_2(run_of(ws, 's2a', k=12, gate_record=record))  # G-INT-S2EQ first
        ws.seal('A2b-S2a:gint:2*kappa_p', {'gint_s2eq.json': {'result': 'FAIL'}})
        with self.assertRaisesRegex(IntegrityError, 'did not pass'):
            rr.plan_phase_2(run_of(ws, 's2a', k=12, gate_record=record))
        ws2 = Workspace(self)
        self.sealed_phase_1(ws2, {'R1_bxR2_1xB0': GOOD})
        rr.main(ws2.argv('gate', 'gate'), ws2.env)
        record2 = ws2.root / 'gate.json'
        record2.write_bytes((ws2.output('gate') / 'gate_record.json').read_bytes())
        ws2.commit(record2)
        ws2.seal('A2b-S2a:gint:2*kappa_p', {'gint_s2eq.json': {'result': 'PASS'}})
        self.assertIsNone(rr.plan_phase_2(run_of(ws2, 's2a', k=11, gate_record=record2)).get('none'))  # the first run
        with self.assertRaisesRegex(IntegrityError, 'runs in order'):
            rr.plan_phase_2(run_of(ws2, 's2a', k=12, gate_record=record2))
        public = {'experiment': rb.EXPERIMENT_ID, 'stage': 'A2b-S2a', 'cell': '2*kappa_p', 'k': 11, 'wall_seconds': 900.0,
                  'peak_rss_bytes': 1, 'integrity': 'PASS', 'sealed_sha256': 'a' * 64}
        ws2.manifest.append('run_public', {'stage': 'A2b-S2a:k11', **public})
        with self.assertRaisesRegex(IntegrityError, 'cost gate'):
            rr.plan_phase_2(run_of(ws2, 's2a', k=12, gate_record=record2))
        stops = [e for e in ws2.manifest.entries() if e['kind'] == 'cost_gate_stop']
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]['payload']['n_prime'], 55)
        ws2.manifest.append('run_public', {'stage': 'A2b-S2a:k11', **{**public, 'wall_seconds': 800.0}})
        self.assertEqual(rr.plan_phase_2(run_of(ws2, 's2a', k=12, gate_record=record2))['cell'], '2*kappa_p')

    def test_the_r3_only_stage_scopes_its_failure_to_the_b2_rows(self):
        import numpy as np
        ws = Workspace(self)
        with self.assertRaisesRegex(IntegrityError, 'run s3-r3 first'):
            rr.plan_s3(run_of(ws, 's3'))
        for attempt in (1, 2):  # two attempts aborted by a deterministic integrity failure (the stage is exhausted)
            out = ws.output(f'r3-{attempt}')
            ws.manifest.append('attempt_start', {'stage': rr.R3_STAGE, 'output_dir': str(out), 'started_utc': 'x'})
            ws.manifest.append('stage_failure', {'stage': rr.R3_STAGE, 'class': 'deterministic_integrity',
                                                 'output_dir': str(out), 'aborted': True, 'during_publication': False})
        plan = rr.plan_s3(run_of(ws, 's3'))
        self.assertEqual(plan['r3'], 'FAILED_INTEGRITY')
        run = run_of(ws, 's3')
        run.plan = plan
        b2, stream, gint_r3 = rr.r3_inputs(run, [], set())
        self.assertFalse(b2['validity']['pass'])
        self.assertTrue(b2['validity']['r3_stage_failed_integrity'])
        self.assertIsNone(stream)
        self.assertEqual(gint_r3, {'result': 'FAILED_INTEGRITY'})
        ws2 = Workspace(self)
        decisions = [{'pa_id': '1:1', 'decision_index': 0, 'status': 'SUPPORTED',
                      'result': {'logging': [0.5, 0.5], 'candidate': [0.4, 0.6], 'reference': [0.5, 0.5],
                                 'logged_index': 1, 'rho_candidate': 1.2, 'rho_reference': 1.0, 'mask': [True, True]}}]
        out = ws2.seal(rr.R3_STAGE, {'b2_keys.json': [['1:1', 0]],
                                     'b2_record.json': {'validity': {'pass': True}, 'weights': None,
                                                        'gint_r3': {'result': 'PASS'}}})
        np.save(out / 'b2_vectors.npy', np.array([[0.25, 0.75]]))
        run = run_of(ws2, 's3')
        run.plan = {'r3': 'sealed', 'r3_dir': out}
        b2, stream, gint_r3 = rr.r3_inputs(run, decisions, {'1:1'})
        self.assertEqual(gint_r3, {'result': 'PASS'})
        row = list(stream())[0]
        self.assertEqual(row['result']['logging'], [0.25, 0.75])
        self.assertEqual(row['result']['rho_candidate'], 0.6 / 0.75)

    def test_g_int_s2eq_runs_one_by_one_then_the_record(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_bxR2_0xB0': GOOD})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        self.assertEqual(rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*kappa_p', k=9, gate_record=record))['cell'],
                         '2*kappa_p')
        with self.assertRaisesRegex(IntegrityError, 'not a recorded run'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*kappa_p', k=11, gate_record=record))
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            rr.plan_phase_2(run_of(ws, 's2a-gint-record', cell='2*kappa_p', gate_record=record))
        for k in (4, 9, 10):
            ws.seal(f'A2b-S2a:gint:2*kappa_p:k{k}', {'run.json': {'k': k}})
        self.assertEqual(rr.plan_phase_2(run_of(ws, 's2a-gint-record', cell='2*kappa_p', gate_record=record))['cell'],
                         '2*kappa_p')

    def test_pooling_waits_for_every_run_seal(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_axR2_0xB0': GOOD})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            rr.main([*ws.argv('s2a-pool', 'pool', '--cell', '2*zeta_p'), '--gate-record', str(record)], ws.env)

    def test_decide_and_the_committed_unblind(self):
        ws = Workspace(self)
        widths = {'R1_axR2_1xB0': GOOD}
        ws.seal('A2b-SM', {'smoke_record.json': {'stage': 'A2b-SM'}})
        ws.seal('A2b-S0', {'s0_record.json': {'stage': 'A2b-S0', 'reused': {'tab_acceptance': a2_tab(), 'nets': a2_nets()}}})
        ws.seal('A2b-L', {'ledger_record.json': {'stage': 'A2b-L', 'e0_count': 10131}})
        fixed = {'row': rb.LAMBDA_FIXED_ROW, 'lambda': 100.0, 'pairs': {'H_point': {'game': 7e-4, 'pitcher': 8e-4}},
                 'S5': {'game': 'PASS', 'pitcher': 'PASS'}, 'R5': {'e': {}, 'f': {}}, 'equals_R1_bxR2_1xB0': False,
                 'decision_input': False, 'disclosure': 'sealed until A2b-UB'}
        sealed_bytes = (json.dumps({'pairs': {}, 'selection': {'R1_axR2_1xB0': {'selected': '100.0'}},
                                    'lambda_fixed_row_public': fixed}) + '\n').encode()
        doc = {**public_doc(widths), 'sealed_sha256': hashlib.sha256(sealed_bytes).hexdigest()}
        ws.seal('A2b-S3', {'public.json': (json.dumps(doc, sort_keys=True) + '\n').encode(),
                           'centers.sealed.json': sealed_bytes})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        for k in rb.GINT_S2EQ_RUNS['2*zeta_p']:
            ws.seal(f'A2b-S2a:gint:2*zeta_p:k{k}', {'run.json': {'k': k}})
        ws.seal('A2b-S2a:gint:2*zeta_p', {'gint_s2eq.json': {'result': 'PASS'}})
        for k in rb.RUN_INDEX['2*zeta_p']:
            ws.seal(f'A2b-S2a:k{k}', {'run.sealed.json': {'k': k}})
        with self.assertRaisesRegex(IntegrityError, 'not complete'):  # the pooled cell is still missing
            rr.main([*ws.argv('decide', 'decide-early'), '--gate-record', str(record)], ws.env)
        ws.seal('A2b-S2a:pool:2*zeta_p', {'pooled.json': pooled('2*zeta_p')})
        self.assertEqual(rr.main([*ws.argv('decide', 'decide'), '--gate-record', str(record)], ws.env), 0)
        decision = json.loads((ws.output('decide') / 'decision_record.json').read_text())
        self.assertEqual(decision['label'], 'ADOPT_FOR_2027_PROTOCOL:R1_axR2_1xB0')
        self.assertEqual(decision['gate_record_sha256'], dc.gate_sha256(json.loads(record.read_text())))
        committed = ws.root / 'decision_record.json'
        committed.write_bytes((ws.output('decide') / 'decision_record.json').read_bytes())
        with self.assertRaisesRegex(IntegrityError, 'committed before unblinding'):
            rr.main([*ws.argv('unblind', 'ub0'), '--decision-record', str(committed)], ws.env)
        ws.commit(committed)
        self.assertEqual(rr.main([*ws.argv('unblind', 'ub1'), '--decision-record', str(committed)], ws.env), 0)
        opened = json.loads((ws.output('ub1') / 'unblinded.json').read_text())
        self.assertEqual(opened['selection'], {'R1_axR2_1xB0': {'selected': '100.0'}})
        self.assertEqual(opened['lambda_fixed_row_public'], fixed)  # G-CODE MAJOR-1: opened with the selection
        self.assertNotIn(rb.LAMBDA_FIXED_ROW, json.dumps(json.loads(
            (roa.require_seal(ws.manifest, 'A2b-S3')[0] / 'public.json').read_text())))

    def test_decide_needs_the_pooled_cells_the_gate_opened(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_axR2_1xB0': GOOD})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        with self.assertRaisesRegex(IntegrityError, 'not complete'):
            rr.main([*ws.argv('decide', 'decide'), '--gate-record', str(record)], ws.env)
        for attempt in (1, 2):  # an exhausted deterministic failure of a phase-2 stage: FAILED_INTEGRITY is available
            out = ws.output(f'gint-record-{attempt}')
            ws.manifest.append('attempt_start', {'stage': 'A2b-S2a:gint:2*zeta_p', 'output_dir': str(out),
                                                 'started_utc': 'x'})
            ws.manifest.append('stage_failure', {'stage': 'A2b-S2a:gint:2*zeta_p', 'class': 'deterministic_integrity',
                                                 'output_dir': str(out), 'aborted': True, 'during_publication': False})
        self.assertEqual(rr.main([*ws.argv('decide', 'decide-terminal'), '--gate-record', str(record)], ws.env), 0)
        decision = json.loads((ws.output('decide-terminal') / 'decision_record.json').read_text())
        self.assertEqual(decision['label'], 'FAILED_INTEGRITY')
        self.assertEqual(decision['failed_integrity'], ['A2b-S2a:gint:2*zeta_p'])
        self.assertEqual(decision['blocked'], sorted([*(f'A2b-S2a:k{k}' for k in rb.RUN_INDEX['2*zeta_p']),
                                                      'A2b-S2a:pool:2*zeta_p']))
        self.assertEqual(decision['not_run'], sorted(f'A2b-S2a:gint:2*zeta_p:k{k}' for k in (5, 11, 12)))
        self.assertEqual(decision['gate_record_sha256'], dc.gate_sha256(json.loads(record.read_text())))

    def exhaust(self, ws, stage, cls='deterministic_integrity'):
        """Two attempts of ``stage`` aborted by failures of ``cls`` (the stage's attempts are used up)."""
        for attempt in (1, 2):
            out = ws.output(f'{stage.replace(":", "_").replace("*", "x")}-failed-{attempt}')
            ws.manifest.append('attempt_start', {'stage': stage, 'output_dir': str(out), 'started_utc': 'x'})
            ws.manifest.append('stage_failure', {'stage': stage, 'class': cls, 'output_dir': str(out), 'aborted': True,
                                                 'during_publication': False})

    def test_an_exhausted_common_stage_is_terminal_without_a_gate_record(self):
        """G-CODE MAJOR-3 (the A2 rule, review round 2 #16): A2b-L exhausted by deterministic integrity failures ->
        the decision records FAILED_INTEGRITY at once, with no A2b-G seal and no gate record; its unsealed
        descendants are BLOCKED. An exhausted infrastructure failure is a decision need, never a label."""
        ws = Workspace(self)
        ws.seal('A2b-SM', {'smoke_record.json': {'stage': 'A2b-SM'}})
        ws.seal('A2b-S0', {'s0_record.json': {'stage': 'A2b-S0'}})
        self.exhaust(ws, 'A2b-L')
        self.assertEqual(rr.main(ws.argv('decide', 'decide-terminal'), ws.env), 0)  # no --gate-record
        decision = json.loads((ws.output('decide-terminal') / 'decision_record.json').read_text())
        self.assertEqual(decision['label'], 'FAILED_INTEGRITY')
        self.assertEqual(decision['failed_integrity'], ['A2b-L'])
        self.assertEqual(decision['blocked'], ['A2b-G', 'A2b-S3'])
        self.assertEqual(decision['not_run'], [])
        self.assertIsNone(decision['gate_record_sha256'])
        self.assertEqual({k: decision['stage_status'][k] for k in ('A2b-SM', 'A2b-S0', 'A2b-L')},
                         {'A2b-SM': 'sealed', 'A2b-S0': 'sealed', 'A2b-L': 'FAILED_INTEGRITY'})
        self.assertTrue(decision['phase_2'].startswith('NOT_OPENED'))
        with self.assertRaisesRegex(IntegrityError, 'already sealed'):
            rr.main(ws.argv('decide', 'decide-again'), ws.env)
        early = Workspace(self)  # an exhausted A2b-S0: everything after it is blocked, A2b-SM never ran
        self.exhaust(early, 'A2b-S0')
        self.assertEqual(rr.main(early.argv('decide', 'decide-s0'), early.env), 0)
        record = json.loads((early.output('decide-s0') / 'decision_record.json').read_text())
        self.assertEqual((record['failed_integrity'], record['blocked'], record['not_run']),
                         (['A2b-S0'], ['A2b-G', 'A2b-L', 'A2b-S3'], ['A2b-SM']))
        infra = Workspace(self)
        infra.seal('A2b-SM', {'smoke_record.json': {'stage': 'A2b-SM'}})
        infra.seal('A2b-S0', {'s0_record.json': {'stage': 'A2b-S0'}})
        self.exhaust(infra, 'A2b-L', cls='infrastructure')
        with self.assertRaisesRegex(IntegrityError, 'A2b-G is not sealed'):  # refused before the attempt (no label)
            rr.main(infra.argv('decide', 'decide-infra'), infra.env)
        self.assertFalse(infra.output('decide-infra').exists())
        self.assertEqual(roa.stage_status(infra.manifest, ('A2b-L', 'A2b-S4')),
                         {'A2b-L': 'DECISION_NEEDED', 'A2b-S4': 'incomplete'})

    def test_an_exhausted_r3_stage_voids_the_b2_rows_only(self):
        """G-CODE MAJOR-3: A2b-S3:R3 exhausted by integrity failures is not terminal - A2b-S3 may still run (B2 rows
        invalid) and the decision carries the r3_stage_failed_integrity flag."""
        ws = Workspace(self)
        self.exhaust(ws, rr.R3_STAGE)
        self.assertEqual(rr.plan_s3(run_of(ws, 's3'))['r3'], 'FAILED_INTEGRITY')  # A2b-S3 may proceed
        self.sealed_phase_1(ws, {})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        self.assertEqual(rr.main([*ws.argv('decide', 'decide'), '--gate-record', str(record)], ws.env), 0)
        decision = json.loads((ws.output('decide') / 'decision_record.json').read_text())
        self.assertEqual(decision['label'], 'NO_PRECISION_GAIN')
        self.assertIn('r3_stage_failed_integrity', decision['flags'])

    def test_manifest_commands(self):
        ws = Workspace(self)
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            rr.main(['record-review', '--config', str(ws.config_path), '--gate', 'G-REV', '--review', 'R', '--verdict',
                     'PASS'], ws.env)
        self.assertEqual(ws.manifest.latest('review_reference', gate='G-REV')['payload']['verdict'], 'PASS')
        with self.assertRaisesRegex(IntegrityError, 'registered once'):
            rr.main(['register-digest', '--config', str(ws.config_path), '--expect-digest', rb.config_digest(ws.config)],
                    ws.env)
        with self.assertRaisesRegex(IntegrityError, 'reviewed digest'):
            rr.main(['register-digest', '--config', str(ws.config_path), '--expect-digest', 'f' * 64], ws.env)
        with self.assertRaisesRegex(IntegrityError, 'follows a recorded deterministic-integrity failure'):
            rr.main(['register-addendum', '--config', str(ws.config_path), '--stage', 'A2b-S3', '--failing-test', 't',
                     '--review', 'r'], ws.env)


class KappaPrimeTests(unittest.TestCase):
    def test_kappa_prime_is_the_sealed_a2_s0_tab_calibration(self):
        """G-CODE MINOR-1: A2-S0 calibrated kappa for the TAB world with the same hit probabilities, groups and
        cross-fitted residuals as for S5-X (run_ope_a2.py:1355-1357) over the four large groups - the kappa_p' rule -
        so the sealed kappa_calibration.TAB equals the removed TRAIN re-read path bit for bit."""
        import numpy as np
        from pitchmdp import ope_redesign_registry as a2
        from pitchmdp import ope_surrogate as osg
        from pitchmdp import ope_synthetic_a2b as sx
        train, doc = synthetic_a2_s0()
        cells, classes, index, census = osg.train_bip_rows(train)
        hit = osg.hit_probability(np.asarray(doc['r1_table']), cells)
        groups = roa._groups(train.iloc[index], a2.S5X_GROUPS)
        recomputed = sx.kappa_prime(hit, groups, doc['train_residuals']['hit'])  # the A2b TRAIN path it replaces
        prime = rr.kappa_prime_from_s0(doc)
        self.assertEqual(prime, recomputed)
        self.assertEqual(prime['groups'], ['FB', 'FC', 'BR', 'OS'])
        self.assertEqual(a2.TAB_GROUPS, rb.KAPPA_PRIME_GROUPS)
        for mutate in (lambda d: d['kappa_calibration']['TAB']['signs'].__setitem__('other', 1),
                       lambda d: d['kappa_p'].__setitem__('TAB', d['kappa_p']['TAB'] * 2),
                       lambda d: d['kappa_calibration']['TAB'].__setitem__('extra', 1)):
            bad = copy.deepcopy(doc)
            mutate(bad)
            with self.assertRaisesRegex(IntegrityError, 'kappa calibration is malformed'):
                rr.kappa_prime_from_s0(bad)


# ---------------------------------------------------------------- the whole chain (G-CODE MINOR-5, second part)
class ChainData:
    """The synthetic inputs of the full chain, generated by the registered EXP-OPE-RES-01 / ML-POLICY code on
    synthetic data (no real file): the ML-POLICY registration (dispatch chain, S6 = runtime-v3 dr-evaluate of 2 of 4
    synthetic DEV games), a sealed-A2-S0 document (build_s0 on synthetic TRAIN), the S6 facts and the A2 rehearsal
    record (results-file layout), the recorded A2 dose-family runs k = 5, 11, 12 (run_s5x on the S-2' fixture world),
    a stub frozen B2 (deterministic tilt of the B0 logging law inside its support; no LightGBM here)."""
    SMALL = {'draws': 200, 'minimum_games': 1, 'minimum_pitchers': 1, 'minimum_pa_starts': 1, 'placebos': 2}
    RECORDED = (5, 11, 12)  # the 2 zeta_p family the gate is steered to open

    def __init__(self, cleanup):
        import itertools
        from ope_a2b_fixtures import ml_policy_world, read_s6, synthetic_a2_s0
        from pitchmdp import ledger_a2b as lg
        self.world = world = ml_policy_world()
        cleanup(shutil.rmtree, world.root, True)
        self.header, self.s6_decisions, self.s6_pins = read_s6(world)
        self.s6_ledger = world.s6 / 'ledger-dev-0.jsonl'
        self.s6_games = lg.ledger_games(self.s6_decisions)
        rpv, policy = world.rpv, world.policy
        self.pre = rpv.prerequisites('dr-evaluate', policy)
        self.selection = lg.selection(world.inputs['frame'], rpv.select_games, rpv.straddling_games,
                                      s6_ledger_games=self.s6_games, n_games=2)
        self.train, self.s0 = synthetic_a2_s0()
        self.raw = self.raw_rows(world.frame)
        self.b2_vectors = {}
        fx = world.tpv.SemiSyntheticTests('test_v2_matches_world_truth_and_is_not_vacuous')
        fx.setUp()
        cleanup(shutil.rmtree, fx.root, True)
        names = itertools.count()

        class S2World:  # the S-2' fixture world (fake G0 + WE) in place of the S5 DEV selection
            game_values = fx.syn.game_values

            def candidate(self, spec_run):
                return fx.world(f"chain-k{spec_run['k']:02d}-{next(names)}")
        self.s5x_world = S2World()
        self.recorded = self.recorded_runs()

    @staticmethod
    def raw_rows(frame):
        """Approved-raw-file columns of every synthetic game (quality columns on the in-play pitches)."""
        import numpy as np
        out = frame[['game_pk', 'at_bat_number', 'pitch_number', 'game_date']].copy()
        play = frame['description'].eq('hit_into_play').to_numpy()
        out['hc_x'] = np.where(play, 120.0, np.nan)
        out['hc_y'] = np.where(play, 140.0, np.nan)
        out['bb_type'] = np.where(play, 'line_drive', None)
        out['launch_speed'] = np.where(play, 92.0, np.nan)
        out['launch_angle'] = np.where(play, 12.0, np.nan)
        out['estimated_woba_using_speedangle'] = np.where(play, 0.4, np.nan)
        return out.reset_index(drop=True)

    def recorded_runs(self):
        """The recorded A2 runs of the 2 zeta_p family on the fixture world with the EXP-OPE-RES-01 ``run_s5x``
        (cap 4), layers built exactly as the A2b runner builds them from the sealed s5x_layers."""
        from pitchmdp import ope_redesign_registry as a2
        from pitchmdp import ope_synthetic_a2 as osx
        doc = self.s0['s5x_layers']
        s0_record = {'s5x_layers': doc, 'kappa_p_prime': rr.kappa_prime_from_s0(self.s0)}
        out = {}
        for k in self.RECORDED:
            spec = next(r for r in a2.S5X_RUNS if r['k'] == k)
            runtime, components, starts = self.s5x_world.candidate({'k': k, 'planning_seed': 0})
            layers = rr.s2_layers(components, self.s5x_world, s0_record, a2.scale(spec['kappa'], doc['kappa_p']),
                                  a2.scale(spec['zeta'], doc['zeta_p']))
            out[k] = osx.run_s5x(runtime, components, starts, run=spec, law=rr._v2_law(runtime), layers=layers,
                                 logs_per_start=6, truth_rollouts=12, cap=4, gamma=True, invariant=spec['zeta'] == '0')
        return out

    def a2_outputs(self, root, env, config):
        """The sealed A2-S0 / A2-S1 / A2-S2 directories, the S6 record (as A2-S3 recomputes it) and the results file
        (A2 rehearsal on the synthetic S6 at the small settings, the recorded dose-family runs)."""
        from pitchmdp import ope_redesign_registry as a2
        from pitchmdp import ope_sealed as osd
        dirs = {}
        for stage, files in (('A2-S0', {'s0.json': self.s0}),
                             ('A2-S1', {'tab.json': {'eta_p': rb.ETA_P, 'acceptance': a2_tab()}}),
                             ('A2-S2', {'s5x.json': {'runs': [], 'nets': a2_nets(), 's2': {}},
                                        **{f'run-k{k:02d}.json': {'s2': r['s2']} for k, r in self.recorded.items()}})):
            dirs[stage] = root / stage
            dirs[stage].mkdir()
            for name, value in files.items():
                (dirs[stage] / name).write_text(json.dumps(value, default=roa._default))
        work = root / 'a2-s3'
        work.mkdir()
        data = env.rehearsal_inputs(config, {}, {}, self.s6_ledger, {'games': self.s6_games}, work, lambda: None)
        result0, _, _ = osd.gint_gate(data.decisions, data.facts, references={}, ess_gate=data.ess_gate,
                                      **{k: self.SMALL[k] for k in ('draws', 'minimum_games', 'minimum_pa_starts')})
        self.s6_dr = roa.json_safe({**result0, 'games': self.s6_games})
        public, sealed, _, _ = roa.rehearsal(
            data.decisions, data.facts, data.last_rows, raw_join=lambda: self.raw, game_values=data.game_values, s0=self.s0,
            references={'s6_dr_a5_record': osd.recorded_gint_reference(self.s6_dr)}, ess_gate=data.ess_gate,
            r3_status='NOT_RUN:R3', **self.SMALL)
        pairs = {p: {k: sealed['pairs'][p][k] for k in ('bootstrap', 'center', 'shift_center') if k in sealed['pairs'][p]}
                 for p in a2.PAIRS}
        runs = [{'k': k, 'per_r1_component': {c: {'b_hat': r['s2'][c]['b_hat'], 'se': r['s2'][c]['se']}
                                               for c in ('R1_a', 'R1_b')}} for k, r in self.recorded.items()]
        self.results = {'S3_public': json.loads(osd.seal(public, sealed)[0]),
                        'UB_unblinded': {'selection': osd._jsonable(sealed['selection']), 'pairs': osd._jsonable(pairs)},
                        'S2_s5x_summary': {'dose_response_s2_by_run': {'runs': runs}}}
        self.a2_dirs = dirs

    def b2_table(self, decisions):
        """The frozen pre-pitch B2 context rows of the decisions (BehaviorTable) and the stub B2 vectors."""
        import numpy as np
        import pandas as pd
        from pitchmdp import policy_estimator as est
        from pitchmdp.behavior_model import START, BehaviorTable
        rows = []
        for row in decisions:
            result = row.get('result')
            if row['status'] not in est.EVALUATED or not isinstance(result, dict):
                continue
            game, ab, pitch = (int(v) for v in str(row['context_key']).split(':'))
            rows.append({'game_pk': game, 'at_bat_number': ab, 'pitch_number': pitch, 'pitcher': int(row['pitcher']),
                         'batter_side': row['batter_side'], 'balls': int(row['balls']), 'strikes': int(row['strikes']),
                         'previous': row['history_actions'][-1] if row['history_actions'] else START})
            logging = np.asarray(result['logging'], dtype=np.float64)
            tilt = np.where(logging > 0, logging * np.exp(0.3 * np.sin(np.arange(len(logging)) + pitch)), 0.)
            self.b2_vectors[(game, ab, pitch)] = tilt / tilt.sum()
        return BehaviorTable(pd.DataFrame(rows).drop_duplicates(['game_pk', 'at_bat_number', 'pitch_number']))

    class StubB2:
        def __init__(self, vectors):
            self.vectors = vectors

        def batch(self, table, positions):
            import numpy as np
            frame = table.frame.iloc[np.asarray(positions)]
            return np.array([self.vectors[(int(g), int(a), int(p))] for g, a, p in
                             zip(frame.game_pk, frame.at_bat_number, frame.pitch_number)])


class ChainEnv(WorkEnv):
    """Every real-data hook served by the synthetic ChainData; the registered code path otherwise (the ML-POLICY
    hooks of run_ope_a2's Environment, rehearsal_inputs and the ledger readers run for real on synthetic files)."""
    def __init__(self, ws, data):
        super().__init__(ws)
        self.data = data

    def policy_registration(self, config):
        return self.data.world.rpv, self.data.world.policy

    def policy_inputs(self, config, local, rpv, policy):
        return self.data.world.inputs

    def game_values(self, base):
        return self.data.world.syn.game_values

    def ledger_world(self, config, local, base, checkpoint):
        return rr.Environment.ledger_world(self, config, local, base, checkpoint)

    def s6_allowlisted(self, config):
        return rr.Environment.s6_allowlisted(self, config)

    def rehearsal_inputs(self, config, local, base, ledger_path, s6_dr, out, checkpoint):
        return roa.Environment.rehearsal_inputs(self, config, local, base, ledger_path, s6_dr, out, checkpoint)

    def bc_vector(self, world):
        from pitchmdp.behavior_model import VectorBC
        from pitchmdp.policy_artifacts import load_train_bc
        artifact = load_train_bc(*world.rpv.registered_path(world.policy, 'bc'))
        return VectorBC.from_artifact(artifact, vocabulary=tuple(artifact.bc.actions))  # the 3 synthetic codes

    def repertoire(self, world):
        return None  # the stub B2 needs no repertoire

    def b2_model(self, config, repertoire):
        return ChainData.StubB2(self.data.b2_vectors)

    def b2_dev_table(self, config, local, world, games):
        if sorted(games) == sorted(self.data.s6_games):
            return self.data.b2_table(self.data.s6_decisions)
        ledger_dir, _ = roa.require_seal(self.ws.manifest, 'A2b-L')
        return self.data.b2_table(list(self.ledger_decisions(ledger_dir / 'ledger-dev-0.jsonl')()))

    def a3_s6_weights(self, config):
        """The 'recorded' A3 S6 B2 diagnostics: the A3 weight code on the synthetic S6 ledger and the stub B2."""
        from pitchmdp import behavior_weights as bw
        from pitchmdp import ope_denominator_a2b as dn
        world = self.ledger_world(config, {}, {}, lambda: None)
        bc, support, hands = rr.b0_artifacts(world)
        decisions = self.data.s6_decisions
        e0 = bw.e0_from_ledger(bw.by_pa(decisions), bc, support, hands)
        table = self.data.b2_table(decisions)
        vectors = dn.decision_vectors(self.b2_model(config, None), table, dn.b2_decision_rows(decisions, e0))
        s = config['registration_input_pins']['a3_weight_settings']
        return dn.b2_weights(decisions, e0, vectors, large=s['large_weight_threshold'], mei=s['mei'],
                             draws=s['bootstrap_draws'], seed=s['bootstrap_seed'], ess_min=s['E5_thresholds'])[0]

    def a2_sealed(self, config, stage_id):
        return self.data.a2_dirs[stage_id]

    def a2_results(self, config):
        return self.data.results, {'path': 'results/EXP-OPE-RES-01.json',
                                   'sha256': config['identities']['exp_ope_res_01']['results_file_sha256']}

    def s6_record(self, config):
        return self.data.s6_ledger, self.data.s6_dr

    def raw_join(self, config, local, window, checkpoint):
        return self.data.raw

    def s5x_inputs(self, config, local, base, out, checkpoint):
        return self.data.s5x_world


class ChainCase(unittest.TestCase):
    """The synthetic chain inputs (ChainData) built once per class, and per test a fresh registered workspace with
    the chain's patches: the synthetic candidate identity in the config expectations and the minimal sizes (the
    registered draws, minimums, TAB R and dataset size, S-2' cap). No real data."""
    @classmethod
    def setUpClass(cls):
        cls.data = ChainData(cls.addClassCleanup)
        root = Path(tempfile.mkdtemp(prefix='a2b-chain-')).resolve()
        cls.addClassCleanup(shutil.rmtree, root, True)
        cls.data.a2_outputs(root, ChainEnv(None, cls.data), None)  # the synthetic hooks never read the config

    def chain_workspace(self, *extra_patches):
        import functools
        from pitchmdp import ledger_a2b as lg
        from pitchmdp import ope_sealed_a2b as sa
        from pitchmdp import ope_synthetic_a2b as sx
        data = self.data
        rpv, policy = data.world.rpv, data.world.policy
        style_path, style_sha = rpv.registered_path(policy, 'style_dev')
        pins = {'s6_dr_a5_ledger_path': str(data.s6_ledger), 's6_dr_a5_ledger_sha256': data.s6_pins['ledger_sha256'],
                's6_dr_a5_manifest_sha256': data.s6_pins['manifest_sha256'],
                's6_runtime_sha256_header': data.s6_pins['runtime_sha256'],
                'selection': {k: data.selection[k] for k in ('selection_s6_game_list_sha256', 'a2b_new_game_list_sha256',
                                                             'month_quota_table')},
                'runtime_setting': lg.runtime_setting_used(policy, data.pre),
                'dev_style_snapshot': {'path': str(style_path), 'sha256': style_sha, 'as_of_exclusive': '2025-07-01'},
                'b2_inference_environment': {'lightgbm': '4.6.0',
                                             'pyproject_sha256': rb.file_sha256(rr.REPO / 'pyproject.toml')},
                'smoke_tab_replications_per_cell': 1}
        candidate = data.s6_pins['candidate_sha256']  # the synthetic candidate identity replaces c13cc989...
        expectations = tuple((p, candidate if p == '$.identities.candidate_policy_sha256' else v)
                             for p, v in rb.CONFIG_EXPECTATIONS)
        patches = [mock.patch.object(rb, 'CONFIG_EXPECTATIONS', expectations),
                   mock.patch.object(lg, 'N_GAMES', 2), mock.patch.object(rb, 'TAB_DATASET_PAS', 300),
                   mock.patch.dict(rb.S2P_WORLD, {'cap': 4}),
                   mock.patch.object(roa, 'rehearsal', functools.partial(roa.rehearsal, **ChainData.SMALL)),
                   mock.patch.object(sa, 'evaluate', functools.partial(sa.evaluate, **ChainData.SMALL)),
                   mock.patch.object(roa, 'require_s5x_consistent', lambda s5x: True),  # the synthetic A2-S2 has no runs
                   mock.patch.object(sx, 'tab_rerun', functools.partial(sx.tab_rerun, replications=2)), *extra_patches]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        ws = Workspace(self, pins=pins, identities={'candidate_policy_sha256': candidate})
        env = ws.env = ChainEnv(ws, data)  # ws.commit records into this environment
        return ws, env, lambda command, name, *extra: rr.main([*ws.argv(command, name), *extra], env)

    def stages(self, main, *commands):
        for command in commands:
            self.assertEqual(main(command, command), 0)

    @staticmethod
    def sealed(ws, stage):
        return roa.require_seal(ws.manifest, stage)[0]

    def failures(self, ws, stage):
        """The aborted attempts of ``stage`` (each a deterministic integrity failure; the stage never sealed)."""
        entries = ws.manifest.entries()
        self.assertFalse([e for e in entries if e['kind'] == 'stage_seal' and e['payload']['stage'] == stage])
        failed = [e['payload'] for e in entries if e['kind'] == 'stage_failure' and e['payload']['stage'] == stage]
        self.assertTrue(failed)
        self.assertEqual({f['class'] for f in failed}, {'deterministic_integrity'})
        return [Path(f['output_dir']) for f in failed]


class FullChainTests(ChainCase):
    """G-CODE MINOR-5 (second part; the A2 FullChainTests pattern): every A2b stage body runs once through ``main``
    on a SYNTHETIC environment - SM, S0, L (with the driver equivalence, path A), S3:R3, S3 (G-INT-A2, G-INT-EVAL, the
    25 rows), G, a small phase 2 (the G-INT-S2EQ runs and record of the 2 zeta_p family, its 10 new-seed runs, the
    pooling, the TAB re-run), S4 and UB, with minimal sizes; the gate is steered (S-5 and (ii) patched) to open the
    R1_a tuples. The run takes a few seconds (measured about 6 s on the Mac mini; no wall-clock assertion, NOTE-E)."""
    def test_smoke_to_unblind(self):
        from pitchmdp import ledger_a2b as lg
        data = self.data
        steer = lambda view, row: ((True, {'game': 'PASS', 'pitcher': 'PASS'}) if row.startswith('R1_axR2_')
                                   else (False, {'game': 'FAIL', 'pitcher': 'FAIL'}))
        real_ii = dc.condition_ii

        def steer_ii(view, row):  # one synthetic pitcher: the pitcher bootstrap is degenerate (zero width)
            out = real_ii(view, row)
            if not row.startswith('R1_axR2_'):
                return out
            return {**out, 'valid': True, 'degenerate': False, 'pass': True, 'outcome': 'PASS',
                    'per_bootstrap': {'game': True, 'pitcher': True}}
        ws, env, run = self.chain_workspace(mock.patch.object(dc, 'condition_ii', steer_ii),
                                            mock.patch.object(dc, 's5_pass', steer))
        main = lambda command, name, *extra: self.assertEqual(run(command, name, *extra), 0)
        for command in ('smoke', 's0', 'ledger', 's3-r3', 's3', 'gate'):
            main(command, command)
        sealed = lambda stage: self.sealed(ws, stage)
        s0_record = json.loads((sealed('A2b-S0') / 's0_record.json').read_text())
        self.assertEqual(s0_record['kappa_p_prime']['groups'], ['FB', 'FC', 'BR', 'OS'])  # no TRAIN row was read
        ledger = json.loads((sealed('A2b-L') / 'ledger_record.json').read_text())
        self.assertTrue(rb.validate_ledger_record(ledger))  # NOTE-A: the written record is its type
        self.assertEqual(json.loads((sealed('A2b-L') / 'driver_equivalence.json').read_text()),
                         ledger['driver_equivalence'])
        equivalence = ledger['driver_equivalence']
        self.assertEqual((equivalence['path'], equivalence['pass'], equivalence['style_source'], equivalence['games']),
                         ('s6_two_game_replay', True, 'header', [lg.game_of(data.s6_decisions[0]),
                                                                 lg.game_of(data.s6_decisions[-1])]))
        self.assertEqual(equivalence['max_abs_delta'], 0.0)
        self.assertEqual(ledger['registered_pins_compared']['runtime_setting'],
                         ws.config['registration_input_pins']['runtime_setting'])
        self.assertTrue(ledger['m10']['pass'] and all(ledger['identity']['checks'].values()))
        r3 = json.loads((sealed('A2b-S3:R3') / 'b2_record.json').read_text())
        self.assertEqual((r3['gint_r3']['result'], r3['validity']['pass']), ('PASS', True))
        s3 = sealed('A2b-S3')
        for name, check in (('gint_a2.json', 'G-INT-A2'), ('gint_eval.json', 'G-INT-EVAL')):
            record = json.loads((s3 / name).read_text())
            self.assertEqual((record['check'], record['result'], record['mismatched_keys']), (check, 'PASS', []))
        public = json.loads((s3 / 'public.json').read_text())
        self.assertEqual(set(public['pairs']), set(rb.PUBLIC_ROWS))
        self.assertTrue(public['B2']['validity']['pass'])
        gate = json.loads((sealed('A2b-G') / 'gate_record.json').read_text())
        self.assertEqual(gate['gated'], ['R1_axR2_0xB0', 'R1_axR2_1xB0'])
        self.assertEqual(gate['work_list']['S2_prime_cells'], ['2*zeta_p'])
        self.assertTrue(gate['work_list']['tab']['rerun'])
        record = ws.root / 'gate.json'
        record.write_bytes((sealed('A2b-G') / 'gate_record.json').read_bytes())
        ws.commit(record)
        gated = ('--gate-record', str(record))
        for k in rb.GINT_S2EQ_RUNS['2*zeta_p']:
            main('s2a-gint', f'gint-k{k}', '--cell', '2*zeta_p', '--k', str(k), *gated)
        main('s2a-gint-record', 'gint-record', '--cell', '2*zeta_p', *gated)
        s2eq = json.loads((sealed('A2b-S2a:gint:2*zeta_p') / 'gint_s2eq.json').read_text())
        self.assertEqual((s2eq['result'], s2eq['max_abs_delta'], s2eq['failure_scope']), ('PASS', 0.0, 'all'))
        self.assertEqual(s2eq['reference_cross_check']['equal'], True)
        for k in rb.RUN_INDEX['2*zeta_p']:
            main('s2a', f'run-k{k}', '--k', str(k), *gated)
        main('s2a-pool', 'pool', '--cell', '2*zeta_p', *gated)
        main('s1', 'tab', *gated)
        main('decide', 'decide', *gated)
        decision_dir = sealed('A2b-S4')
        decision = json.loads((decision_dir / 'decision_record.json').read_text())
        self.assertTrue(decision['label'].startswith(('ADOPT_FOR_2027_PROTOCOL:R1_axR2_', 'BIAS_NOT_CERTIFIED:'))
                        or decision['label'] == 'NO_PRECISION_GAIN', decision['label'])
        committed = ws.root / 'decision_record.json'
        committed.write_bytes((decision_dir / 'decision_record.json').read_bytes())
        ws.commit(committed)
        main('unblind', 'unblind', '--decision-record', str(committed))
        opened = json.loads((sealed('A2b-UB') / 'unblinded.json').read_text())
        self.assertEqual(opened['lambda_fixed_row_public']['row'], rb.LAMBDA_FIXED_ROW)
        self.assertEqual(set(opened['pairs']), set(rb.ROWS))
        stages = [e['payload']['stage'] for e in ws.manifest.entries() if e['kind'] == 'stage_seal']
        self.assertEqual(stages, ['A2b-SM', 'A2b-S0', 'A2b-L', 'A2b-S3:R3', 'A2b-S3', 'A2b-G',
                                  *(f'A2b-S2a:gint:2*zeta_p:k{k}' for k in (5, 11, 12)), 'A2b-S2a:gint:2*zeta_p',
                                  *(f'A2b-S2a:k{k}' for k in range(1, 11)), 'A2b-S2a:pool:2*zeta_p', 'A2b-S1', 'A2b-S4',
                                  'A2b-UB'])
        self.assertFalse([e for e in ws.manifest.entries() if e['kind'] == 'stage_failure'])
        pooled_doc = json.loads((sealed('A2b-S2a:pool:2*zeta_p') / 'pooled.json').read_text())
        self.assertEqual((pooled_doc['n_prime'], pooled_doc['runs']), (10, list(rb.RUN_INDEX['2*zeta_p'])))
        self.assertEqual(len([e for e in ws.manifest.entries() if e['kind'] == 'run_public']), 10)
        self.assertEqual(set(json.loads((sealed('A2b-S1') / 'tab_rerun.json').read_text())['acceptance']),
                         {c['id'] for c in rb.TAB_ACCEPTANCE})


class FailClosedWiringTests(ChainCase):
    """G-CODE 2nd closure MINOR-A: the runner's fail-closed wires on the synthetic chain - removing any of them makes
    a test here fail (a failed driver equivalence, a path-B identity that is not the new ledger runtime, a pyproject
    pin mismatch, a G-INT key record or an A2b-L record outside its type)."""
    def test_a_failed_driver_equivalence_never_seals_the_ledger_stage(self):
        from pitchmdp import ledger_a2b as lg
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0')
        real = lg.driver_equivalence

        def failing(*args, **kwargs):  # a well-typed failing record (one mismatched field)
            record = real(*args, **kwargs)
            return {**record, 'pass': False, 'mismatched_fields': 1,
                    'mismatched_keys_shown': ['x#0.result.logging.[0]']}
        with mock.patch.object(lg, 'driver_equivalence', failing), \
                self.assertRaisesRegex(IntegrityError, 'driver equivalence failed on path s6_two_game_replay'):
            main('ledger', 'ledger-fail')
        out, = self.failures(ws, 'A2b-L')
        written = json.loads((out / 'driver_equivalence.json').read_text())  # written before its pass is required
        self.assertEqual((written['pass'], written['mismatched_fields']), (False, 1))
        self.assertFalse((out / 'ledger-dev-0.jsonl').exists())  # stopped before the new DEV ledger
        self.assertFalse((out / 'ledger_record.json').exists())

    def test_the_a2b_l_records_are_validated_before_they_are_written(self):
        from pitchmdp import ledger_a2b as lg
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0')
        real = lg.driver_equivalence
        with mock.patch.object(lg, 'driver_equivalence', lambda *a, **k: {**real(*a, **k), 'reward': 0.5}), \
                self.assertRaisesRegex(IntegrityError, 'driver equivalence record: keys outside its type'):
            main('ledger', 'ledger-untyped')
        out, = self.failures(ws, 'A2b-L')
        self.assertFalse((out / 'driver_equivalence.json').exists())

    def test_a_ledger_record_outside_its_type_is_never_written(self):
        from pitchmdp import ledger_a2b as lg
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0')
        real = lg.require_runtime_pins
        with mock.patch.object(lg, 'require_runtime_pins', lambda *a, **k: {**real(*a, **k), 'extra_pin': 'x'}), \
                self.assertRaisesRegex(IntegrityError, 'A2b-L record: registered pins compared'):
            main('ledger', 'ledger-record-untyped')
        out, = self.failures(ws, 'A2b-L')
        self.assertTrue((out / 'driver_equivalence.json').exists())
        self.assertFalse((out / 'ledger_record.json').exists())

    def test_path_b_must_have_run_the_new_ledger_runtime(self):
        from pitchmdp import ledger_a2b as lg
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0')
        real = lg.driver_equivalence

        def other_identity(*args, **kwargs):  # a well-typed passing path-B record of another runtime
            record = real(*args, **kwargs)
            return {**record, 'path': 'same_request_twice', 'reason': 'runtime_sha_not_reproduced',
                    'runtime_sha256_reproduced': False, 'games': record['games'][:1], 'runtime_sha256': 'e' * 64}
        with mock.patch.object(lg, 'driver_equivalence', other_identity), \
                self.assertRaisesRegex(IntegrityError, 'path-B runtimes are not the new ledger runtime'):
            main('ledger', 'ledger-other-b')
        self.failures(ws, 'A2b-L')

    def test_the_real_path_b_seals_the_ledger_stage(self):
        """The positive control of the path-B identity wire: the S6 header without its style report and with another
        runtime sha256 sends the runner down path B, whose runtimes are the new ledger's."""
        from pitchmdp import ledger_a2b as lg
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0')
        real = lg.driver_equivalence

        def unreproducible(*args, **kwargs):
            header = kwargs['s6_header']
            provenance = {k: v for k, v in header['pins']['provenance'].items() if k != 'style_report'}
            header = {**header, 'runtime_sha256': 'e' * 64, 'pins': {**header['pins'], 'provenance': provenance}}
            return real(*args, **{**kwargs, 's6_header': header})
        with mock.patch.object(lg, 'driver_equivalence', unreproducible):
            self.stages(main, 'ledger')
        ledger = json.loads((self.sealed(ws, 'A2b-L') / 'ledger_record.json').read_text())
        equivalence = ledger['driver_equivalence']
        self.assertEqual((equivalence['path'], equivalence['pass'], equivalence['reason']),
                         ('same_request_twice', True, 'runtime_sha_not_reproduced'))
        self.assertEqual(equivalence['runtime_sha256'], ledger['runtime_sha256'])

    def test_a_pyproject_mismatch_voids_only_the_b2_rows(self):
        ws, env, main = self.chain_workspace(mock.patch.object(ChainEnv, 'pyproject_sha256', lambda env: 'f' * 64))
        self.stages(main, 'smoke', 's0', 'ledger')
        with self.assertRaisesRegex(IntegrityError, 'pyproject.toml differs from the B2 inference environment pin'):
            main('s3-r3', 'r3-1')
        ws.manifest.append('bugfix_addendum', {'stage': rr.R3_STAGE, 'fix_commit': 'c' * 40,
                                               'failing_test': 'synthetic', 'new_files_sha256': rr.new_file_hashes(),
                                               'review_reference': 'synthetic'})
        with self.assertRaisesRegex(IntegrityError, 'pyproject.toml differs from the B2 inference environment pin'):
            main('s3-r3', 'r3-2')  # the second registered attempt
        self.assertEqual(len(self.failures(ws, rr.R3_STAGE)), 2)
        self.assertEqual(roa.stage_status(ws.manifest, (rr.R3_STAGE,))[rr.R3_STAGE], 'FAILED_INTEGRITY')
        self.stages(main, 's3')  # A2b-S3 still runs: the B2 rows only are void (design 9.6)
        public = json.loads((self.sealed(ws, 'A2b-S3') / 'public.json').read_text())
        self.assertFalse(public['B2']['validity']['pass'])
        for row in rb.B2_ROWS:
            self.assertEqual(public['pairs'][row]['valid'], {'game': False, 'pitcher': False}, row)
        self.assertTrue(public['pairs'][rb.BASELINE]['valid']['game'])

    def test_a_gint_r3_record_outside_its_type_is_never_written(self):
        from pitchmdp import ope_denominator_a2b as dn
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0', 'ledger')
        real = dn.gint_r3
        with mock.patch.object(dn, 'gint_r3', lambda *a, **k: {**real(*a, **k), 'center': 1e-4}), \
                self.assertRaisesRegex(IntegrityError, 'gint_r3: keys outside its type'):
            main('s3-r3', 'r3-untyped')
        out, = self.failures(ws, rr.R3_STAGE)
        self.assertFalse((out / 'gint_r3.json').exists())

    def test_a_gint_a2_record_outside_its_type_is_never_written(self):
        ws, env, main = self.chain_workspace()
        self.stages(main, 'smoke', 's0', 'ledger', 's3-r3')
        real = rr.gint_a2_record
        with mock.patch.object(rr, 'gint_a2_record', lambda *a, **k: {**real(*a, **k), 'b_hat': 1e-5}), \
                self.assertRaisesRegex(IntegrityError, 'gint_a2: keys outside its type'):
            main('s3', 's3-untyped')
        out, = self.failures(ws, 'A2b-S3')
        self.assertFalse((out / 'gint_a2.json').exists())

if __name__ == '__main__':
    unittest.main()
