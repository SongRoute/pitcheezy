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

from ope_a2b_fixtures import a2_nets, a2_tab, draft_config, pooled, public_doc, registered_config  # noqa: E402
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
    kappa_prime_inputs = train = a2_sealed = a2_results = _real


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
        extra = {'s2a': ('--k', '11'), 's2a-gint': ('--cell', '2*kappa_p'), 's2a-pool': ('--cell', '2*zeta_p')}
        for command in ('smoke', 's0', 'ledger', 's3', 'gate', 's2a', 's2a-gint', 's2a-pool', 's1', 'decide', 'unblind',
                        'register-digest', 'record-review', 'register-addendum', 'reconcile-attempt'):
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
        self.assertEqual(rr.attempt_stage(rr.parse(['s2a-gint', '--cell', '2*kappa_p'])), 'A2b-S2a:gint:2*kappa_p')
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
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', gate_record=record_path))
        copy_path = ws.root / 'committed_gate_record.json'
        copy_path.write_bytes(record_path.read_bytes())
        ws.commit(copy_path)
        plan = rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', gate_record=copy_path))
        self.assertEqual(plan['cell'], '2*zeta_p')
        with self.assertRaisesRegex(IntegrityError, 'did not open'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*kappa_p', gate_record=copy_path))
        with self.assertRaisesRegex(IntegrityError, 'did not require a TAB re-run'):
            rr.plan_phase_2(run_of(ws, 's1', gate_record=copy_path))
        tampered = ws.root / 'tampered.json'
        tampered.write_bytes(record_path.read_bytes().replace(b'R1_axR2_1xB0', b'R1_bxR2_1xB0', 1))
        ws.commit(tampered)
        with self.assertRaisesRegex(IntegrityError, 'G-GATE'):
            rr.plan_phase_2(run_of(ws, 's2a-gint', cell='2*zeta_p', gate_record=tampered))
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
        sealed_bytes = b'{"pairs": {}, "selection": {"R1_axR2_1xB0": {"selected": "100.0"}}}\n'
        doc = {**public_doc(widths), 'sealed_sha256': hashlib.sha256(sealed_bytes).hexdigest()}
        ws.seal('A2b-S3', {'public.json': (json.dumps(doc, sort_keys=True) + '\n').encode(),
                           'centers.sealed.json': sealed_bytes})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
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

    def test_decide_needs_the_pooled_cells_the_gate_opened(self):
        ws = Workspace(self)
        self.sealed_phase_1(ws, {'R1_axR2_1xB0': GOOD})
        rr.main(ws.argv('gate', 'gate'), ws.env)
        record = ws.root / 'gate.json'
        record.write_bytes((ws.output('gate') / 'gate_record.json').read_bytes())
        ws.commit(record)
        with self.assertRaisesRegex(IntegrityError, 'not sealed'):
            rr.main([*ws.argv('decide', 'decide'), '--gate-record', str(record)], ws.env)

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


if __name__ == '__main__':
    unittest.main()
