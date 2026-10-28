"""SYNTHETIC-ONLY tests for the EXP-OPE-RES-02 stage runner: fail-closed refusals of a draft, the run window and the
A2b hang guards, the registration gate (digest and review receipts), the gate command on a synthetic sealed public
file, G-GATE (phase 2 and the decision need the gate record committed at HEAD byte for byte), the S-2' order, the
G-INT-S2EQ precondition and the cost gate, operator-blind pooling, the decision and the committed-record unblind,
and the manifest commands. Every config, manifest and stage directory is synthetic and lives in a temporary
directory; no real data, ledger or model is read (real-data hooks raise in the test environment).
"""
from contextlib import nullcontext
import argparse
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
    def __init__(self, test, *, draft=False):
        self.test = test
        self.root = tmpdir(test)
        (self.root / 'inputs').mkdir()
        (self.root / 'outputs').mkdir()
        config = draft_config() if draft else registered_config(self.root)
        if not draft:
            config['registration_input_pins']['new_files_sha256'] = rr.new_file_hashes()
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
        self.assertEqual(set(pins['record_only_pins']), {'g0_bundle_sha256', 'a2_execution_source_commit',
                                                        'a3_reader_commit', 'x2_world_predictor_identity',
                                                        'smoke_train_subset', 'processed_frame_aligned'})
        self.assertTrue(all(v.startswith('record-only') for v in pins['record_only_pins'].values()))
        self.assertEqual(set(pins['pins_compared_at_run']), {'runtime_setting', 'dev_style_snapshot',
                                                            'b2_inference_environment'})

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


if __name__ == '__main__':
    unittest.main()
