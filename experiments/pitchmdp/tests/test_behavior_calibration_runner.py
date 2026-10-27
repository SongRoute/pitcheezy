"""EXP-BC-CAL-01 runner: fail-closed registration gate (immutable pin schema, reviewed scope with
pyproject.toml, dependency pins), destination rules, caps and the protection window (mock clock /
RSS), the attempt budget persisted across commands, supervised children, fixed stage order with
chain-anchored stage verification, revalidation before every stage, DEV-label and DEV-metric gates,
a terminal non-citable final.json for every unsuccessful attempt, freeze-manifest determinism and
tamper detection, Holm / tie rule / exhaustive labels. Status-neutral: explicit DRAFT / REGISTERED
fixtures, never the committed config's status. The end-to-end synthetic attempt and the registered
preflight -> smoke launch run in isolated interpreters (one OpenMP runtime)."""
import copy
from datetime import datetime
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_behavior_calibration as runner  # noqa: E402
from behavior_calibration_synthetic import (NESTED, RUNNER_SHA256, as_draft, as_registered,  # noqa: E402
                                            digest_entries_for, extra_rows, load_config, make_frame, run_isolated,
                                            write_processed, write_raw)
from pitchmdp import behavior_calibration_registry as reg  # noqa: E402
from pitchmdp.behavior_data import A3Inputs, ProcessedReader, RawKeysReader, ReadLog  # noqa: E402
from pitchmdp.behavior_features import SEALED  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

GIT_OK = {'head': 'a' * 40, 'scope': list(reg.REVIEWED_CODE_SCOPE), 'dirty': False, 'commit_is_ancestor': True,
          'changed_since_commit': [], 'config_committed': True}
AFTERNOON = datetime(2026, 10, 27, 15, 0, tzinfo=runner.KST)
MORNING = datetime(2026, 10, 28, 6, 30, tzinfo=runner.KST)
nested = pytest.mark.skipif(bool(os.environ.get(NESTED)), reason='inside a test-started preflight run')


@pytest.fixture()
def draft():
    return as_draft(load_config())


def registered(config, runner_sha=RUNNER_SHA256, **kwargs):
    return as_registered(config, runner_sha256=runner_sha, **kwargs)


def gate(config, *, git, runner_sha256, entries=None):
    """The registration gate with a digest-manifest chain that reconciles with ``config`` by default."""
    return runner.registration_gate(config, git=git, runner_sha256=runner_sha256,
                                    digest_entries=digest_entries_for(config) if entries is None else entries)


def _set(config, dotted, value):
    node = config
    *parents, last = dotted.split('.')
    for part in parents:
        node = node[part]
    node[last] = value


# ---------------------------------------------------------------- registration gate

def test_draft_config_is_refused_before_any_data(draft):
    with pytest.raises(runner.Refused, match='not REGISTERED') as error:
        gate(draft, git=GIT_OK, runner_sha256=RUNNER_SHA256)
    assert error.value.terminal


def test_registered_config_passes_and_every_missing_pin_is_refused(draft):
    good = registered(draft)
    assert gate(good, git=GIT_OK, runner_sha256=RUNNER_SHA256)
    cases = [
        (lambda c: c.update(execute=False), 'not REGISTERED'),
        (lambda c: c['registration'].update(code_review_reference=None), 'code_review_reference missing'),
        (lambda c: c['registration'].update(registered_kst='2026-10-28T05:00:00+00:00'), 'registered_kst'),
        (lambda c: c['data']['s6_ledger'].update(ledger_sha256=None), 'data.s6_ledger.ledger_sha256'),
        (lambda c: c['data']['s6_ledger'].update(manifest_sha256='Z' * 64), 'data.s6_ledger.manifest_sha256'),
        (lambda c: c['data']['s6_ledger']['runtime_sha256'].update(value='Z' * 64), 'runtime_sha256.value'),
        (lambda c: c['data']['s6_ledger'].update(candidate_sha256='0' * 64), 'candidate_sha256'),
        (lambda c: c['data']['s6_ledger'].update(candidate_sha256=None), 'candidate_sha256 missing'),
        (lambda c: c['code'].update(commit=None), 'code.commit'),
        (lambda c: c['code'].update(commit='a' * 39), 'code.commit'),
        (lambda c: c['runner'].update(output_root='/tmp/elsewhere'), 'config_sha256'),
    ]
    for mutate, message in cases:
        changed = copy.deepcopy(good)
        mutate(changed)
        with pytest.raises(runner.Refused, match=message):
            gate(changed, git=GIT_OK, runner_sha256=RUNNER_SHA256)
    with pytest.raises(runner.Refused, match='runner_sha256'):
        gate(good, git=GIT_OK, runner_sha256='f' * 64)
    for git in ({'dirty': True}, {'commit_is_ancestor': False}, {'changed_since_commit': ['experiments/x.py']},
                {'changed_since_commit': ['pyproject.toml']}, {'config_committed': False},
                {'config_committed': None}, {'scope': ['experiments', 'src', 'scripts']}):
        with pytest.raises(runner.Refused):
            gate(good, git={**GIT_OK, **git}, runner_sha256=RUNNER_SHA256)
    deviating = copy.deepcopy(good)
    deviating['candidates']['B3'] = {'mandatory': True}
    with pytest.raises(runner.Refused, match='deviates'):
        gate(deviating, git=GIT_OK, runner_sha256=RUNNER_SHA256)


def test_review_reproduction_empty_pin_list_null_s6_pins_and_recomputed_digest_is_refused(draft):
    """Review finding 4: the config's editable list cannot make pins optional."""
    weakened = registered(draft)
    weakened['registration']['fields_pinned_at_registration'] = []
    weakened['data']['s6_ledger'].update(ledger_sha256=None, manifest_sha256=None)
    weakened['data']['s6_ledger']['runtime_sha256']['value'] = None
    weakened['code']['config_sha256'] = runner.config_content_sha256(weakened)
    with pytest.raises(runner.Refused) as error:
        gate(weakened, git=GIT_OK, runner_sha256=RUNNER_SHA256)
    assert 'fields_pinned_at_registration' in str(error.value) and error.value.terminal
    weakened['registration']['fields_pinned_at_registration'] = list(reg.DRAFT_PIN_FIELDS)
    weakened['code']['config_sha256'] = runner.config_content_sha256(weakened)
    with pytest.raises(runner.Refused, match='ledger_sha256 missing'):
        gate(weakened, git=GIT_OK, runner_sha256=RUNNER_SHA256)


def test_the_gate_refuses_void_registrations_and_unreconciled_digest_chains(draft, monkeypatch):
    """Bug-fix retry 1: the gate refuses the void registration's commits and digest, and the digest
    manifest must hold one original entry then exactly one reconciled re-registration per retry record
    (an ordered bijection), the last being this config at this commit."""
    good = registered(draft)
    record = good['bug_fix_retries'][0]  # the committed config carries the retry record
    assert gate(good, git=GIT_OK, runner_sha256=RUNNER_SHA256)
    for commit in (record['void_registration']['registered_commit'], record['void_registration']['registration_commit']):
        with pytest.raises(runner.Refused, match='void registration'):
            gate(registered(draft, commit=commit), git=GIT_OK, runner_sha256=RUNNER_SHA256)
    entries = digest_entries_for(good)
    original, rereg = entries
    cases = {'only the original': [original], 'an extra entry': [original, rereg, dict(rereg)],
             'no original': [rereg], 'two originals': [original, original],
             'previous digest': [original, {**rereg, 'previous_digest': '0' * 64}],
             'retry index': [original, {**rereg, 'retry': 2}],
             'void registration': [original, {**rereg, 'void_registration': {**record['void_registration'],
                                                                               'registration_commit': '0' * 40}}],
             'aborted attempt': [original, {**rereg, 'aborted_attempt': '/elsewhere/attempt-9'}],
             'reason': [original, {**rereg, 'reason': ' '}],
             'original at another commit': [{**original, 'commit': '0' * 40}, rereg],
             'active digest': [original, {**rereg, 'config_content_sha256': '0' * 64}],
             'active commit': [original, {**rereg, 'commit': '0' * 40}],
             'unreadable manifest': lambda config: (_ for _ in ()).throw(OSError('no such file'))}
    for name, chain in cases.items():
        with pytest.raises(runner.Refused, match='digest') as error:
            gate(good, git=GIT_OK, runner_sha256=RUNNER_SHA256, entries=chain)
        assert error.value.terminal, name
    monkeypatch.setattr(runner, 'config_content_sha256', lambda config: record['void_registration']['config_digest'])
    with pytest.raises(runner.Refused, match='void registration'):
        gate(good, git=GIT_OK, runner_sha256=RUNNER_SHA256)


def test_git_scope_includes_pyproject_and_dependency_pins(tmp_path):
    state = runner.git_state(None)
    assert state['scope'] == list(reg.REVIEWED_CODE_SCOPE) and 'pyproject.toml' in state['scope']
    pins = runner.dependency_pins()
    assert pins['group'] == reg.DEPENDENCY_GROUP and pins['pinned'] == pins['installed'] == reg.DEPENDENCY_PINS
    assert len(pins['pyproject_sha256']) == 64
    for body, message in (('lightgbm==4.5.0', 'pins'), ('', 'pins')):
        fake = tmp_path / f'pyproject-{len(body)}.toml'
        fake.write_text(f'[project]\nname = "x"\n[project.optional-dependencies]\nresearch-a3 = ["{body}"]\n')
        with pytest.raises(runner.Refused, match=message):
            runner.dependency_pins(fake)


def test_config_sha_ignores_only_its_own_field_and_appended_addenda(draft):
    base = runner.config_content_sha256(draft)
    later = copy.deepcopy(draft)
    later['code']['config_sha256'] = 'f' * 64
    later['registration']['addenda_records'] = [{'addendum': 'smoke'}]
    assert runner.config_content_sha256(later) == base
    later['seeds']['bootstrap'] += 1
    assert runner.config_content_sha256(later) != base


def test_main_refuses_a_draft_config_without_touching_data(tmp_path, capsys, draft):
    config = copy.deepcopy(draft)
    config['runner'].update(output_root=str(tmp_path), heavy_lock=str(tmp_path / '.heavy.lock'))
    path = tmp_path / 'draft.json'
    path.write_text(json.dumps(config))
    destination = tmp_path / 'attempt'
    code = runner.main(['--config', str(path), '--local-config', str(tmp_path / 'missing.json'),
                        '--destination', str(destination), 'smoke'])
    assert code == 2 and not destination.exists() and not (tmp_path / '.heavy.lock').exists()
    assert 'not REGISTERED' in capsys.readouterr().err


def test_destination_rules(tmp_path):
    with pytest.raises(runner.Refused, match='results'):
        runner.check_destination(runner.REPO / 'results' / 'EXP-BC-CAL-01')
    fake = tmp_path / 'repo'
    (fake / '.git').mkdir(parents=True)
    with pytest.raises(runner.Refused, match='results'):
        runner.check_destination(fake / 'results' / 'a3')
    with pytest.raises(runner.Refused, match='ML-POLICY-VAL-v1'):
        runner.check_destination(tmp_path / 'ML-POLICY-VAL-v1' / 'S6-x' / 'a3')
    root = tmp_path / 'EXP-BC-CAL-01'
    with pytest.raises(runner.Refused, match='directly under'):
        runner.check_destination(tmp_path / 'other' / 'attempt', output_root=root)
    assert runner.check_destination(root / 'attempt-1', output_root=root) == (root / 'attempt-1').resolve()


# ---------------------------------------------------------------- caps, window, budget, children

class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t


def test_guard_caps_and_protection_window():
    clock, rss = Clock(), {'v': 1 << 30}
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, clock=clock, now=lambda: AFTERNOON,
                         rss=lambda: rss['v'], alarm=False)
    guard.start_run()
    guard.start_stage('fit')
    guard.check('ok')
    assert guard.soft_b2(3_599, 'x') is None
    warning = guard.soft_b2(3_601, 'fit b2')
    assert warning['warning'] == 'B2_FIT_SOFT_CAP_EXCEEDED' and guard.warnings == [warning]
    guard.check('soft cap is warn-only')
    rss['v'] = reg.CAPS['abort_rss_bytes'] + 1
    with pytest.raises(runner.Abort) as error:
        guard.check('rss')
    assert error.value.reason == 'RSS_CAP'
    rss['v'] = 1 << 30
    clock.t = reg.CAPS['stage_hang_guard_seconds'] + 1
    with pytest.raises(runner.Abort) as error:
        guard.check('stage')
    assert error.value.reason == 'STAGE_HANG_GUARD'
    guard.start_stage('next')
    clock.t = reg.CAPS['total_wall_cap_seconds'] + 1
    with pytest.raises(runner.Abort) as error:
        guard.check('total')
    assert error.value.reason == 'TOTAL_WALL_CAP'
    morning = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: MORNING, alarm=False)
    with pytest.raises(runner.Refused, match='protection window'):
        morning.start_run()
    with pytest.raises(runner.Abort) as error:
        morning.check('window')
    assert error.value.reason == 'PROTECTION_WINDOW'
    edge = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: datetime(2026, 10, 28, 13, 30, tzinfo=runner.KST),
                        alarm=False)
    edge.start_run()  # 13:30 is outside [06:30, 13:30)
    assert guard.seconds_to_window() == 15.5 * 3600
    early = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: datetime(2026, 10, 28, 5, 0, tzinfo=runner.KST),
                         alarm=False)
    assert early.seconds_to_window() == 1.5 * 3600


def test_attempt_budget_persists_across_commands(tmp_path):
    """Review finding 5: the 14,400 s total covers the whole attempt (smoke command + run command)."""
    root = tmp_path / 'attempt'
    root.mkdir()
    assert runner.consumed_seconds(root) == 0
    start = runner.begin_command(root, 'smoke', 0.)
    with pytest.raises(runner.Refused, match='no end record') as error:
        runner.consumed_seconds(root)  # a killed command leaves the budget unknown
    assert error.value.terminal
    runner.end_command(root, start, 1200.)
    second = runner.begin_command(root, 'run', 1200.)
    runner.end_command(root, second, 300.5)
    assert runner.consumed_seconds(root) == 1500.5
    clock = Clock()
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, clock=clock, now=lambda: AFTERNOON, alarm=False,
                         consumed=runner.consumed_seconds(root))
    guard.start_run()
    assert guard.remaining() == reg.CAPS['total_wall_cap_seconds'] - 1500.5
    clock.t = reg.CAPS['total_wall_cap_seconds'] - 1500.
    with pytest.raises(runner.Abort, match='TOTAL_WALL_CAP'):
        guard.check('second command')
    spent = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: AFTERNOON, alarm=False,
                         consumed=reg.CAPS['total_wall_cap_seconds'])
    with pytest.raises(runner.Abort, match='TOTAL_WALL_CAP'):
        spent.start_run()
    lines = (root / runner.BUDGET).read_bytes().splitlines()
    (root / runner.BUDGET).write_bytes(lines[0] + b'\n' + lines[1].replace(b'1200.0', b'12.0') + b'\n' +
                                       b'\n'.join(lines[2:]) + b'\n')
    with pytest.raises(IntegrityError, match='chain broken'):
        runner.consumed_seconds(root)


class Now:
    """AFTERNOON for the first ``calls`` reads, then 06:30 KST (the window begins)."""
    def __init__(self, calls):
        self.calls, self.n = calls, 0

    def __call__(self):
        self.n += 1
        return AFTERNOON if self.n <= self.calls else MORNING


def test_supervised_children_stop_at_the_window_and_on_timeout(tmp_path):
    """Review finding 6: the memory probe / preflight child is checked against the same window and
    budget; only the runner-owned child (its own process group) is stopped."""
    pid_file = tmp_path / 'child.pid'
    child = [sys.executable, '-c', f'import os, time; open({str(pid_file)!r}, "w").write(str(os.getpid())); '
                                   'time.sleep(60)']
    bystander = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=Now(3), alarm=False)
        guard.start_run()
        began = time.monotonic()
        with pytest.raises(runner.Abort, match='PROTECTION_WINDOW'):
            runner.supervised(child, guard, timeout=120, where='preflight', poll=0.2)
        assert time.monotonic() - began < 20
        for _ in range(50):
            if pid_file.exists() and pid_file.read_text():
                break
            time.sleep(0.05)
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)  # the runner-owned child is gone
        assert bystander.poll() is None  # another process is never touched
        calm = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: AFTERNOON, alarm=False)
        calm.start_run()
        with pytest.raises(runner.Refused, match='timeout'):
            runner.supervised([sys.executable, '-c', 'import time; time.sleep(60)'], calm, timeout=0.5,
                              where='memory probe', poll=0.1)
        done = runner.supervised([sys.executable, '-c', 'print("System-wide memory free percentage: 63%")'], calm,
                                 timeout=30, where='memory probe', poll=0.1)
        assert runner.parse_free_share(done['stdout']) == 0.63 and done['returncode'] == 0
    finally:
        bystander.kill()
        bystander.wait()


# ---------------------------------------------------------------- stages, records, final.json

class StubSource:
    """Records every per-stage revalidation; refuses at ``refuse_at``; never opens data."""
    def __init__(self, refuse_at=None):
        self.calls, self.refuse_at = [], refuse_at

    def revalidate(self, run, stage):
        self.calls.append(stage)
        if stage == self.refuse_at:
            raise runner.Refused(f'registration gate: revalidation failed before {stage}', terminal=True)
        return {'stage': stage, 'registration_gate': 'stub'}

    def open(self, run):
        raise AssertionError('stub stages never open inputs')


def make_run(tmp_path, config, source=None, now=AFTERNOON):
    destination = tmp_path / 'EXP-BC-CAL-01' / 'attempt-1'
    destination.mkdir(parents=True)
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: now, alarm=False)
    return runner.Run(config, destination=destination, source=source or StubSource(), guard=guard, commit='a' * 40)


def stub(result=None, error=None, artifact=None):
    def stage(run, out):
        if artifact:
            out.json(artifact, {'partial': True})
        if error is not None:
            raise error
        return result or {}
    return stage


def test_fixed_stage_order_and_fail_closed_after_a_failure(tmp_path, draft, monkeypatch):
    run = make_run(tmp_path, draft)
    with pytest.raises(runner.Refused, match='needs a passed smoke'):
        runner.execute_stage(run, 'census')
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', stub())
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'census', stub())
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'fit', stub(error=IntegrityError('B1 counts differ')))
    assert runner.execute_stage(run, 'smoke') == 'PASS'
    with pytest.raises(runner.Refused, match='already exists'):
        runner.execute_stage(run, 'smoke')
    with pytest.raises(runner.Refused, match='needs a passed fit'):
        runner.execute_stage(run, 'calibrate')
    changed = copy.deepcopy(draft)
    changed['seeds']['bootstrap'] += 1
    other = runner.Run(changed, destination=run.destination, source=StubSource(), guard=run.guard, commit='a' * 40)
    with pytest.raises(runner.Refused, match='config changed since smoke'):
        runner.execute_stage(other, 'census')
    assert runner.execute_stage(run, 'census') == 'PASS'
    assert runner.execute_stage(run, 'fit') == 'FAILED_INTEGRITY'
    final = json.loads((run.destination / 'final.json').read_text())
    assert final['label'] == 'FAILED_INTEGRITY' and final['citable'] is False
    with pytest.raises(runner.Refused, match='closed'):
        runner.execute_stage(run, 'calibrate')
    assert [r['status'] for r in run.attempt.records()] == ['PASS', 'PASS', 'FAILED_INTEGRITY']


@pytest.mark.parametrize('error,status', [
    (runner.Refused('freeze-xf manifest not pinned'), 'REFUSED'),
    (ValueError('a bug'), 'FAILED_RUNTIME'),
    (IntegrityError('R2 failed'), 'FAILED_INTEGRITY'),
    (runner.Abort('RSS_CAP', 'fit'), 'ABORTED'),
    (KeyboardInterrupt(), 'ABORTED')])
def test_every_unsuccessful_stage_writes_a_terminal_final(tmp_path, draft, monkeypatch, error, status):
    """Review finding 10: REFUSED / FAILED_RUNTIME / ABORTED / FAILED_INTEGRITY all close the attempt."""
    run = make_run(tmp_path, draft)
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', stub(error=error))
    assert runner.execute_stage(run, 'smoke') == status
    final = json.loads((run.destination / 'final.json').read_text())
    assert final['outcome'] == final['label'] == status and status in reg.TERMINAL_FAILURES
    assert final['citable'] is False and final['stage'] == 'smoke' and final['reason']
    assert final['config_sha256'] == run.config_sha256
    record = json.loads((run.attempt.stage_path('smoke') / 'stage.json').read_text())
    assert record['status'] == status and record['error']
    with pytest.raises(IntegrityError, match='unregistered'):
        runner.write_final(tmp_path, outcome='FAILED', stage='x', reason='y', config_sha256='z')


def test_registration_is_revalidated_before_every_stage(tmp_path, draft, monkeypatch):
    """Review finding 4: the gate runs again before each stage (cached inputs included)."""
    source = StubSource(refuse_at='calibrate')
    run = make_run(tmp_path, draft, source=source)
    for stage in reg.STAGES:
        monkeypatch.setitem(runner.STAGE_FUNCTIONS, stage, stub())
    statuses = [runner.execute_stage(run, s) for s in ('smoke', 'census', 'fit', 'calibrate')]
    assert statuses == ['PASS', 'PASS', 'PASS', 'REFUSED'] and source.calls == ['smoke', 'census', 'fit', 'calibrate']
    record = json.loads((run.attempt.stage_path('fit') / 'stage.json').read_text())
    assert record['revalidation'] == {'stage': 'fit', 'registration_gate': 'stub'}
    assert json.loads((run.destination / 'final.json').read_text())['label'] == 'REFUSED'


def launch_env(**overrides):
    env = dict(git_state=lambda commit, path=None: dict(GIT_OK), runner_sha256=lambda: RUNNER_SHA256,
               native_check=lambda: {'pinned': dict(reg.DEPENDENCY_PINS)},
               memory_command=(sys.executable, '-c', 'print("System-wide memory free percentage: 80%")'),
               preflight_command=lambda: [sys.executable, '-c', 'print("3 passed")'],
               source_factory=lambda config, local, path, env: StubSource(), now=lambda: AFTERNOON, alarm=False,
               digest_entries=digest_entries_for)
    env.update(overrides)
    return runner.LaunchEnv(**env)


def test_real_source_revalidation_checks_file_git_and_runner(tmp_path, draft):
    config = registered(draft)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    state = {'git': dict(GIT_OK)}
    env = launch_env(git_state=lambda commit, p=None: dict(state['git']))
    source = runner.RealSource(config, {}, path, env)
    assert source.revalidate(None, 'fit')['registration_gate'] == 'PASS'
    state['git']['dirty'] = True
    with pytest.raises(runner.Refused, match='uncommitted'):
        source.revalidate(None, 'calibrate')
    state['git']['dirty'] = False
    path.write_text(json.dumps({**config, 'execute': False}))
    with pytest.raises(runner.Refused, match='config file changed'):
        source.revalidate(None, 'calibrate')


ADOPT = 'ADOPT_FOR_2027_REGISTRATION:B1'


def stub_all(monkeypatch, clock=None, seconds=0., label=ADOPT):
    def make(name):
        def stage(run, out):
            if clock is not None:
                clock.t += seconds
            if name == 'decide':
                out.json('decision.json', {'label': label, 'citable': False})
                return {'label': label}
            return {}
        return stage
    for name in reg.STAGES:
        monkeypatch.setitem(runner.STAGE_FUNCTIONS, name, make(name))


def write_registered(tmp_path, draft):
    root = tmp_path / 'EXP-BC-CAL-01'
    config = registered(draft, runner={'output_root': str(root), 'heavy_lock': str(tmp_path / '.heavy.lock')})
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, root / 'attempt-1'


def test_launch_smoke_then_run_shares_one_attempt_budget(tmp_path, draft, monkeypatch, capsys):
    path, destination = write_registered(tmp_path, draft)
    clock = Clock()
    stub_all(monkeypatch, clock, seconds=1400.)  # each stub stage spends 1,400 s of the mock clock
    assert runner.launch(path, tmp_path / 'local.json', destination, 'smoke', launch_env(clock=clock)) == 0
    assert runner.consumed_seconds(destination) == 1400.
    preflight = json.loads((destination / 'preflight-smoke.json').read_text())
    assert preflight['tests']['returncode'] == 0 and preflight['free_share'] == 0.8
    # census .. decide: 10 stages * 1,400 s fit a fresh 14,400 s budget, but not the attempt's remaining
    # 13,000 s (the smoke command's 1,400 s count): decide is aborted by the total cap
    code = runner.launch(path, tmp_path / 'local.json', destination, 'run', launch_env(clock=clock))
    assert code == 3
    rows = runner.read_chain(destination / runner.BUDGET)
    assert [r['event'] for r in rows] == ['start', 'end', 'start', 'end'] and rows[2]['consumed_before'] == 1400.
    records = runner.Attempt(destination).records()
    assert [r['status'] for r in records] == ['PASS'] * 10 + ['ABORTED'] and records[-1]['stage'] == 'decide'
    record = json.loads((runner.Attempt(destination).stage_path('decide') / 'stage.json').read_text())
    assert record['error']['reason'] == 'TOTAL_WALL_CAP'
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'ABORTED' and final['citable'] is False
    assert runner.launch(path, tmp_path / 'local.json', destination, 'run', launch_env(clock=clock)) == 2
    assert 'closed' in capsys.readouterr().err


def test_launch_refusals_and_terminal_outcomes(tmp_path, draft, monkeypatch):
    path, destination = write_registered(tmp_path, draft)
    local = tmp_path / 'local.json'
    stub_all(monkeypatch)
    # inside the window: refused before any directory exists
    assert runner.launch(path, local, destination, 'smoke', launch_env(now=lambda: MORNING)) == 2
    assert not destination.exists()
    # memory below the registered minimum: transient, recorded, the attempt stays open (no stage ran)
    low = (sys.executable, '-c', 'print("System-wide memory free percentage: 10%")')
    assert runner.launch(path, local, destination, 'smoke', launch_env(memory_command=low)) == 2
    assert not (destination / 'final.json').exists()
    refusals = runner.read_chain(destination / 'refusals.jsonl')
    assert refusals[-1]['terminal'] is False and 'free share' in refusals[-1]['refused']
    # preflight failure: terminal, the attempt is closed REFUSED
    failing = lambda: [sys.executable, '-c', 'import sys; print("1 failed"); sys.exit(1)']
    assert runner.launch(path, local, destination, 'smoke', launch_env(preflight_command=failing)) == 2
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'REFUSED' and 'tests failed' in final['reason']
    assert json.loads((destination / 'preflight-smoke.json').read_text())['tests']['returncode'] == 1
    assert runner.consumed_seconds(destination) >= 0  # every command closed its budget record


def test_launch_aborts_a_child_at_the_window_and_closes_the_attempt(tmp_path, draft, monkeypatch):
    path, destination = write_registered(tmp_path, draft)
    stub_all(monkeypatch)
    slow = (sys.executable, '-c', 'import time; time.sleep(60)')
    code = runner.launch(path, tmp_path / 'local.json', destination, 'smoke',
                         launch_env(memory_command=slow, now=Now(2)))
    assert code == 3
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'ABORTED' and final['reason']['reason'] == 'PROTECTION_WINDOW'
    assert final['reason']['where'] == 'memory probe' and not runner.Attempt(destination).started()


def test_launch_gate_failure_on_a_running_attempt_is_terminal_and_busy_lock_is_not(tmp_path, draft, monkeypatch):
    path, destination = write_registered(tmp_path, draft)
    local = tmp_path / 'local.json'
    stub_all(monkeypatch)
    assert runner.launch(path, local, destination, 'smoke', launch_env()) == 0
    with open(tmp_path / '.heavy.lock', 'a+b') as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runner.launch(path, local, destination, 'run', launch_env()) == 2
    assert not (destination / 'final.json').exists()
    assert runner.read_chain(destination / 'refusals.jsonl')[-1]['terminal'] is False
    dirty = lambda commit, p=None: {**GIT_OK, 'dirty': True}
    assert runner.launch(path, local, destination, 'run', launch_env(git_state=dirty)) == 2
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'REFUSED' and 'uncommitted' in final['reason']


@pytest.mark.parametrize('damage', ['missing', 'empty', 'cut', 'torn', 'replaced', 'shortened', 'nan_string',
                                    'negative', 'boolean', 'numeric_string', 'not_monotone'])
def test_a_started_attempt_needs_its_bound_budget_history(tmp_path, draft, monkeypatch, damage):
    """Review round 2, #5: after a passed smoke, a missing, empty, cut, torn, replaced or shortened
    budget history refuses (fail closed); a fresh 14,400 s budget is never granted."""
    path, destination = write_registered(tmp_path, draft)
    clock = Clock()
    stub_all(monkeypatch, clock, seconds=100.)
    assert runner.launch(path, tmp_path / 'local.json', destination, 'smoke', launch_env(clock=clock)) == 0
    budget = destination / runner.BUDGET
    lines = budget.read_bytes().splitlines()
    if damage == 'missing':
        budget.unlink()
    elif damage == 'empty':
        budget.write_bytes(b'')
    elif damage == 'cut':
        budget.write_bytes(lines[0] + b'\n')  # the end record is gone
    elif damage == 'torn':
        budget.write_bytes(lines[0] + b'\n' + lines[1][:len(lines[1]) // 2] + b'\n')
    elif damage == 'replaced':  # a valid chain, but not the one the stage records are bound to
        budget.unlink()
        start = runner.append_chain(budget, {'event': 'start', 'command': 'smoke', 'consumed_before': 0.,
                                             'utc': 'replaced'})
        runner.end_command(destination, start, 100.)
    elif damage == 'shortened':  # the same start record, a forged shorter end record
        budget.unlink()
        start = runner.append_chain(budget, {k: v for k, v in json.loads(lines[0]).items()
                                             if k not in ('seq', 'prev', 'sha256')})
        runner.end_command(destination, start, 1.)
    elif damage == 'not_monotone':  # a second command claims it started with nothing consumed
        start = runner.append_chain(budget, {'event': 'start', 'command': 'run', 'consumed_before': 0.,
                                             'utc': 'x'})
        runner.append_chain(budget, {'event': 'end', 'command': 'run', 'start_sha256': start['sha256'],
                                     'seconds': 5., 'utc': 'y'})
    else:  # a correctly rehashed end record with a malformed duration (review round 3, #3)
        bad = {'nan_string': 'nan', 'negative': -100., 'boolean': True, 'numeric_string': '100'}[damage]
        budget.unlink()
        start = runner.append_chain(budget, {k: v for k, v in json.loads(lines[0]).items()
                                             if k not in ('seq', 'prev', 'sha256')})
        runner.append_chain(budget, {'event': 'end', 'command': 'smoke', 'start_sha256': start['sha256'],
                                     'seconds': bad, 'utc': 'z'})
    code = runner.launch(path, tmp_path / 'local.json', destination, 'run', launch_env(clock=clock))
    assert code == 2 and not runner.Attempt(destination).stage_path('census').exists()
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] in ('REFUSED', 'FAILED_INTEGRITY') and final['citable'] is False
    assert any(word in final['reason'] for word in ('budget', 'chain', 'unreadable', 'end record', 'bound',
                                                    'duration', 'monotone'))


def test_the_citable_outcome_is_the_last_write_after_sealing_and_budget_closure(tmp_path, draft, monkeypatch):
    """Review round 3, #2: a complete run publishes the citable final.json last, after decide is sealed
    and the run command's budget record is closed."""
    path, destination = write_registered(tmp_path, draft)
    stub_all(monkeypatch)
    assert runner.launch(path, tmp_path / 'local.json', destination, 'smoke', launch_env()) == 0
    seen = {}
    real_write_final = runner.write_final

    def spy(root, **kwargs):
        rows = runner.read_chain(Path(root) / runner.BUDGET)
        seen.update(budget_closed=rows[-1]['event'] == 'end', decide=runner.Attempt(root).status('decide'),
                    citable=kwargs.get('citable'))
        return real_write_final(root, **kwargs)
    monkeypatch.setattr(runner, 'write_final', spy)
    assert runner.launch(path, tmp_path / 'local.json', destination, 'run', launch_env()) == 0
    assert seen == {'budget_closed': True, 'decide': 'PASS', 'citable': True}
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == ADOPT and final['citable'] is True
    assert final['budget_start_sha256'] == runner.read_chain(destination / runner.BUDGET)[-2]['sha256']
    decide = json.loads((runner.Attempt(destination).stage_path('decide') / 'stage.json').read_text())
    assert decide['citable'] is False and 'final.json' in decide['citable_outcome']
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: AFTERNOON, alarm=False)
    again = runner.Run(json.loads(path.read_text()), destination=destination, source=StubSource(), guard=guard,
                       commit='a' * 40)
    with pytest.raises(runner.Refused, match='closed'):
        runner.publish_final(again)  # an outcome is published once


@pytest.mark.parametrize('failure', ['stage_json', 'checksum_window', 'chain', 'budget_closure', 'publish_window'])
def test_a_failure_during_finalization_never_leaves_a_citable_adoption(tmp_path, draft, monkeypatch, failure):
    """Review round 3, #2: failures or a window crossing while sealing decide, closing the command
    budget or publishing end non-citable (FAILED_RUNTIME / ABORTED); no ADOPT label is ever citable."""
    path, destination = write_registered(tmp_path, draft)
    stub_all(monkeypatch)
    assert runner.launch(path, tmp_path / 'local.json', destination, 'smoke', launch_env()) == 0
    decide_dir = runner.Attempt(destination).stage_path('decide')
    now = lambda: AFTERNOON
    if failure == 'stage_json':
        real_json = runner.StageDir.json

        def broken(self, name, value):
            if name == 'stage.json' and self.path == decide_dir:
                raise OSError('disk full while writing stage.json')
            return real_json(self, name, value)
        monkeypatch.setattr(runner.StageDir, 'json', broken)
    elif failure == 'checksum_window':  # the window begins while the decide checksums are computed
        now = lambda: MORNING if (decide_dir / 'stage.json').exists() else AFTERNOON
    elif failure == 'chain':
        real_append = runner.append_chain

        def broken(target, body):
            if Path(target).name == 'stages.jsonl' and body.get('stage') == 'decide':
                raise OSError('disk full while appending the stage chain')
            return real_append(target, body)
        monkeypatch.setattr(runner, 'append_chain', broken)
    elif failure == 'budget_closure':
        real_end = runner.end_command

        def broken(root, start, seconds):
            if start['command'] == 'run':
                raise OSError('disk full while closing the command budget')
            return real_end(root, start, seconds)
        monkeypatch.setattr(runner, 'end_command', broken)
    else:  # the window begins after the budget closure, before the citable write
        now = lambda: MORNING if len(runner.read_chain(destination / runner.BUDGET)) >= 4 else AFTERNOON
    code = runner.launch(path, tmp_path / 'local.json', destination, 'run', launch_env(now=now))
    assert code != 0
    final = json.loads((destination / 'final.json').read_text())
    assert final['citable'] is False and final['label'] in ('FAILED_RUNTIME', 'ABORTED')
    assert final['label'] == ('ABORTED' if 'window' in failure else 'FAILED_RUNTIME')


def test_the_stage_cap_still_applies_while_a_pass_is_sealed(tmp_path, draft, monkeypatch):
    """Review round 4, #1: a stage body ending at 7,199 s whose checksums reach 7,201 s is a non-citable
    STAGE_HANG_GUARD abort (the stage clock stays on through sealing), and the preserved record skips its
    checksums once the cap is reached."""
    destination = tmp_path / 'EXP-BC-CAL-01' / 'attempt-1'
    destination.mkdir(parents=True)
    smoke_dir = runner.Attempt(destination).stage_path('smoke')

    class SealingClock:
        """Mocked monotonic clock: +7,199 s inside the stage body, +2 s more once stage.json exists."""
        def __init__(self):
            self.t = 0.

        def __call__(self):
            return self.t + (2. if (smoke_dir / 'stage.json').exists() else 0.)
    clock = SealingClock()
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, clock=clock, now=lambda: AFTERNOON, alarm=False)
    run = runner.Run(draft, destination=destination, source=StubSource(), guard=guard, commit='a' * 40)

    def body(run, out):
        out.json('artifact.json', {'rows': 1})
        clock.t += reg.CAPS['stage_hang_guard_seconds'] - 1  # the body ends at 7,199 s: within the cap
        return {}
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', body)
    assert runner.execute_stage(run, 'smoke') == 'ABORTED'
    record = json.loads((smoke_dir / 'stage.json').read_text())
    assert record['status'] == 'ABORTED' and record['citable'] is False
    assert record['error']['sealing']['reason'] == 'STAGE_HANG_GUARD' and record['error']['stage'] is None
    assert 'skipped' in record['checksums'] and not (smoke_dir / 'SHA256SUMS').exists()
    assert [r['status'] for r in run.attempt.records()] == ['ABORTED'] and not run.attempt.passed('smoke')
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'ABORTED' and final['citable'] is False
    assert guard.stage_start is None and guard.armed is False  # cleared only after sealing


def test_stage_records_are_bound_to_the_command_budget(tmp_path, draft, monkeypatch):
    path, destination = write_registered(tmp_path, draft)
    stub_all(monkeypatch)
    assert runner.launch(path, tmp_path / 'local.json', destination, 'smoke', launch_env()) == 0
    start = runner.read_chain(destination / runner.BUDGET)[0]
    row = runner.Attempt(destination).record('smoke')
    assert row['budget_start_sha256'] == start['sha256'] and row['attempt_seconds'] >= 0
    record = json.loads((runner.Attempt(destination).stage_path('smoke') / 'stage.json').read_text())
    assert record['budget']['command_start_sha256'] == start['sha256'] and 'read_summary' in record


def test_loader_gate_opens_no_file_for_a_draft_or_changed_config(tmp_path, draft, monkeypatch):
    """Review round 2, #4: the real loader re-checks the stage state and the registration itself;
    called directly, it opens no data or artifact file (only the config is read by the gate)."""
    import builtins
    import io
    opened = []
    real_open = io.open

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def loader_for(config):
        path = tmp_path / f'config-{len(opened)}-{config["status"]}.json'
        path.write_text(json.dumps(config))
        source = runner.RealSource(config, {'artifact_root': str(tmp_path / 'nowhere'), 'raw_root': str(tmp_path),
                                            'raw_allowlist': []}, path, launch_env())
        run = make_run(tmp_path / f'run-{len(opened)}-{config["status"]}', config, source=source)
        return run, path

    run, path = loader_for(draft)
    monkeypatch.setattr(io, 'open', spy_open)
    monkeypatch.setattr(builtins, 'open', spy_open)
    with pytest.raises(runner.Refused, match='inside an executing stage'):
        run.inputs()  # not inside a stage: refused before anything is read
    assert opened == []
    run.current_stage = 'smoke'
    run.attempt.stage_path('smoke').mkdir()
    with pytest.raises(runner.Refused, match='not REGISTERED'):
        run.inputs()
    assert set(opened) == {str(path)}
    good = registered(draft)
    run, path = loader_for(good)
    run.current_stage = 'smoke'
    run.attempt.stage_path('smoke').mkdir()
    path.write_text(json.dumps({**good, 'execute': False}))  # changed on disk after the command started
    opened.clear()
    with pytest.raises(runner.Refused, match='config file changed'):
        run.inputs()
    assert set(opened) == {str(path)} and run.log.entries == []


def test_the_memory_rule_counts_the_supervised_child_tree(tmp_path):
    """Review round 2, #11: RSS of the child tree (a grandchild in its own session included) counts
    against the registered cap together with the runner's own peak; the whole owned tree is stopped."""
    pid_file = tmp_path / 'grandchild.pid'
    hog = (f'import os, time; open({str(pid_file)!r}, "w").write(str(os.getpid())); data = b"x" * (160 << 20); '
           'time.sleep(60)')
    child = [sys.executable, '-c', 'import subprocess, sys, time; '
             f'subprocess.Popen([sys.executable, "-c", {hog!r}], start_new_session=True); time.sleep(60)']
    caps = {**reg.CAPS, 'abort_rss_bytes': runner.peak_rss_bytes() + (96 << 20)}
    guard = runner.Guard(caps, reg.PROTECTION_WINDOW_KST, now=lambda: AFTERNOON, alarm=False)
    guard.start_run()
    assert guard.rss() <= guard.rss_cap  # the runner alone is under the cap
    with pytest.raises(runner.Abort, match='RSS_CAP') as error:
        runner.supervised(child, guard, timeout=60, where='preflight', poll=0.2)
    assert 'child tree' in error.value.where
    pid = int(pid_file.read_text())
    for _ in range(100):
        if not runner._alive(pid):
            break
        time.sleep(0.05)
    assert not runner._alive(pid)  # the grandchild (own session) was stopped too
    tree = runner.process_tree_rss(os.getpid())
    assert tree and os.getpid() in tree and all(v >= 0 for v in tree.values())


def test_inter_stage_verification_runs_under_the_window_guard(tmp_path, draft, monkeypatch):
    """Review round 2, #12: hashing earlier stages checks the guard inside its loops."""
    run = make_run(tmp_path, draft)
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', stub(artifact='smoke.json'))
    assert runner.execute_stage(run, 'smoke') == 'PASS'
    run.guard.now = Now(2)  # the window begins inside the first hash loop of the verification
    with pytest.raises(runner.Abort) as error:
        runner.execute_stage(run, 'census')
    assert error.value.reason == 'PROTECTION_WINDOW' and error.value.where == 'verification'
    assert not run.attempt.stage_path('census').exists()


def test_the_pinned_a3_input_must_be_certified_by_its_identity_manifest(tmp_path, draft):
    """The real loader reads the split-aligned A3 input only when its pinned identity manifest records a
    PASS rewrite of the pinned processed parquet into exactly the pinned file at the split dates."""
    from behavior_calibration_synthetic import extra_rows, prepare_input
    world = make_frame(2, games={'train': 4, 'earlystop': 1, 'temperature': 1, 'dev': 2})
    record = prepare_input(tmp_path / 'data', pd.concat([world, extra_rows()], ignore_index=True))
    manifest_path = Path(record['rewritten']['path']).with_name('pitches.split-aligned.identity.json')
    good = json.loads(manifest_path.read_text())
    good['source']['sha256'] = reg.DATA_PINS['processed_sha256']  # as for the real pinned source

    def config_for(manifest):
        path = tmp_path / f'manifest-{len(list(tmp_path.glob("manifest-*")))}.json'
        path.write_text(json.dumps(manifest))
        config = copy.deepcopy(draft)
        config['data']['a3_input'].update(identity_manifest_path=str(path),
                                          identity_manifest_sha256=runner.sha256_file(path),
                                          rechunked_sha256=record['rewritten']['sha256'])
        return config

    path, checked = runner.verify_a3_input(config_for(good))
    assert path == Path(draft['data']['a3_input']['path']) and all(checked['checks'].values())
    damages = [lambda m: m.update(result='FAIL'), lambda m: m['identity'].update(result='FAIL'),
               lambda m: m.update(aligned=False), lambda m: m['source'].update(sha256='0' * 64),
               lambda m: m['rewritten'].update(sha256='0' * 64), lambda m: m.update(boundaries=['2025-07-01']),
               lambda m: m['identity']['columns']['pitch_type'].update(equal=False),
               lambda m: m['identity'].update(differing_columns=['pitch_type']),
               lambda m: m['rewritten'].update(rows=1)]
    for damage in damages:
        bad = copy.deepcopy(good)
        damage(bad)
        with pytest.raises(IntegrityError, match='does not certify'):
            runner.verify_a3_input(config_for(bad))
    config = config_for(good)
    config['data']['a3_input']['identity_manifest_sha256'] = 'f' * 64
    with pytest.raises(IntegrityError, match='registration pin'):
        runner.verify_a3_input(config)
    with pytest.raises(runner.Refused, match='inputs/ folder'):
        runner.check_destination(tmp_path / 'EXP-BC-CAL-01' / 'inputs', output_root=tmp_path / 'EXP-BC-CAL-01')


APPLEDOUBLE = bytes.fromhex('00051607') + bytes(4092)  # a 4096-byte macOS AppleDouble sidecar


def sidecar(path):
    path.write_bytes(APPLEDOUBLE)
    return path


@pytest.mark.parametrize('outcome', ['PASS', 'ABORTED'])
def test_appledouble_sidecars_never_enter_or_break_a_seal(tmp_path, draft, monkeypatch, outcome):
    """Bug-fix retry 1: the exFAT T7 volume writes AppleDouble sidecars beside files, before and after
    sealing (._SHA256SUMS included); they are excluded from SHA256SUMS, verification and freeze-manifest
    file lists, while a ._ file without the AppleDouble magic stays an artifact (refused after sealing)."""
    run = make_run(tmp_path, draft)

    def body(run, out):
        out.json('smoke.json', {'rows': 1})
        sidecar(out.path / '._smoke.json')  # before sealing
        if outcome == 'ABORTED':
            raise runner.Abort('RSS_CAP', 'smoke')
        return {}
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', body)
    assert runner.execute_stage(run, 'smoke') == outcome
    path = run.attempt.stage_path('smoke')
    sums = (path / 'SHA256SUMS').read_text()
    assert '._' not in sums and 'smoke.json' in sums and 'stage.json' in sums
    for name in ('._SHA256SUMS', '._stage.json', '._layout.json'):  # written by the volume after sealing
        sidecar(path / name)
    assert runner.verify_sha256sums(path)
    assert [p.name for p in runner.StageDir(path).artifacts()] == ['smoke.json']
    impostor = path / '._payload.json'  # a ._ name without the AppleDouble magic is an artifact
    if outcome == 'PASS':
        assert run.attempt.passed('smoke')
        monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'census', stub())
        assert runner.execute_stage(run, 'census') == 'PASS'  # the next stage verifies the sealed smoke
        impostor.write_text('{"payload": true}')
        with pytest.raises(IntegrityError, match='SHA256SUMS'):
            run.attempt.passed('smoke')
    else:
        assert json.loads((run.destination / 'final.json').read_text())['label'] == 'ABORTED'
        impostor.write_text('{"payload": true}')
        assert not runner.verify_sha256sums(path)
    assert impostor in runner.StageDir(path).artifacts()


def test_abort_keeps_outputs_and_stops(tmp_path, draft, monkeypatch):
    run = make_run(tmp_path, draft)
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', stub())
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'census',
                        stub(error=runner.Abort('RSS_CAP', 'census'), artifact='partial.json'))
    runner.execute_stage(run, 'smoke')
    assert runner.execute_stage(run, 'census') == 'ABORTED'
    path = run.attempt.stage_path('census')
    record = json.loads((path / 'stage.json').read_text())
    assert record['status'] == 'ABORTED' and record['error']['reason'] == 'RSS_CAP' and record['citable'] is False
    assert (path / 'partial.json').exists() and runner.verify_sha256sums(path)
    assert json.loads((run.destination / 'final.json').read_text())['label'] == 'ABORTED'
    with pytest.raises(runner.Refused):
        runner.execute_stage(run, 'fit')


def test_protection_window_reached_mid_run_aborts_the_stage(tmp_path, draft, monkeypatch):
    destination = tmp_path / 'EXP-BC-CAL-01' / 'attempt-1'
    destination.mkdir(parents=True)
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=Now(4), alarm=False)
    run = runner.Run(draft, destination=destination, source=StubSource(), guard=guard, commit='a' * 40)

    def long_stage(run, out):
        out.json('progress.json', {'step': 1})
        run.guard.check('loop')  # a cooperative check inside a long loop
        run.guard.check('loop')
        return {}
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', long_stage)
    assert runner.execute_stage(run, 'smoke') == 'ABORTED'
    record = json.loads((run.attempt.stage_path('smoke') / 'stage.json').read_text())
    assert record['error']['reason'] == 'PROTECTION_WINDOW'
    assert (run.attempt.stage_path('smoke') / 'progress.json').exists()


def pin_stage(run, stage, status='PASS', manifest=True):
    """Write a minimal stage directory with SHA256SUMS + chain entry (and a pinned manifest)."""
    path = run.attempt.stage_path(stage)
    path.mkdir()
    out = runner.StageDir(path)
    if manifest:
        (path / 'manifest.json').write_bytes(runner.canonical_bytes({'manifest': stage}))
        runner.append_chain(run.attempt.root / 'addenda.jsonl',
                            {'addendum': stage, 'manifest_sha256': runner.sha256_file(path / 'manifest.json')})
    out.json('stage.json', {'stage': stage, 'status': status})
    out.text('SHA256SUMS', runner.sha256sums(path))
    runner.append_chain(run.attempt.root / 'stages.jsonl',
                        {'stage': stage, 'status': status, 'stage_json_sha256': runner.sha256_file(path / 'stage.json'),
                         'sha256sums_sha256': runner.sha256_file(path / 'SHA256SUMS')})


def test_stage_verification_uses_the_chain_anchored_digests(tmp_path, draft, monkeypatch):
    """Review finding 8: an artifact changed together with a regenerated SHA256SUMS is detected."""
    run = make_run(tmp_path, draft)
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'smoke', stub(artifact='smoke.json'))
    monkeypatch.setitem(runner.STAGE_FUNCTIONS, 'census', stub())
    assert runner.execute_stage(run, 'smoke') == 'PASS' and run.attempt.passed('smoke')
    path = run.attempt.stage_path('smoke')
    (path / 'smoke.json').write_text('{"partial": false}\n')
    assert not runner.verify_sha256sums(path)
    (path / 'SHA256SUMS').write_text(runner.sha256sums(path))  # regenerate; the chain is untouched
    assert runner.verify_sha256sums(path)
    with pytest.raises(IntegrityError, match='SHA256SUMS differs from the digest anchored'):
        run.attempt.passed('smoke')
    with pytest.raises(IntegrityError):
        runner.execute_stage(run, 'census')
    other = make_run(tmp_path / 'b', draft)
    assert runner.execute_stage(other, 'smoke') == 'PASS'
    record = other.attempt.stage_path('smoke') / 'stage.json'
    record.write_text(record.read_text().replace('"citable": true', '"citable": false'))
    (other.attempt.stage_path('smoke') / 'SHA256SUMS').write_text(runner.sha256sums(other.attempt.stage_path('smoke')))
    with pytest.raises(IntegrityError, match='stage.json differs'):
        other.attempt.passed('smoke')


def test_dev_labels_and_dev_metrics_are_gated(tmp_path, draft):
    run = make_run(tmp_path, draft)
    world = make_frame(1, games={'train': 3, 'earlystop': 1, 'temperature': 1, 'dev': 2})
    frame = pd.concat([world, extra_rows()], ignore_index=True)
    path, sha, _ = write_processed(tmp_path / 'data', frame)
    root, pins = write_raw(tmp_path / 'data', frame)
    log = ReadLog()
    inputs = A3Inputs(reader=ProcessedReader(path, sha256=sha, log=log),
                      raw_reader=RawKeysReader(root, pins, log), b0=None, support=None, hands=None, archetypes=None,
                      ledger=None, identity={}, log=log)
    loaded = inputs.pre_gate_frame('fit')
    dev = loaded.split.eq('dev')
    assert dev.any() and loaded.loc[dev, 'pitch_type'].eq(SEALED).all()
    with pytest.raises(runner.Refused, match='freeze-primary manifest pinned'):
        inputs.release_dev(run.gate, 'xf-fit')
    with pytest.raises(runner.Refused, match='freeze-primary'):
        runner.stage_s6_integrity(run, None)
    pin_stage(run, 'freeze-primary')
    with pytest.raises(runner.Refused, match='s6-integrity'):
        inputs.release_dev(run.gate, 'xf-fit')
    assert log.rows_at_or_after('2025-07-01', guarded_only=True) == 0
    pin_stage(run, 's6-integrity', manifest=False)
    inputs.release_dev(run.gate, 'xf-fit')
    original = world.loc[world.split.eq('dev')].sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number'])
    clean = lambda s: s.astype(object).where(s.notna(), None).tolist()
    assert clean(loaded.loc[dev, 'pitch_type']) == clean(original.pitch_type)
    for stage in (runner.stage_score_dev, runner.stage_s6_weights):
        with pytest.raises(runner.Refused, match='freeze-xf manifest pinned'):
            stage(run, None)
    (run.attempt.stage_path('freeze-primary') / 'manifest.json').write_bytes(b'{"tampered":true}\n')
    with pytest.raises(IntegrityError, match='pinned sha256'):
        run.gate.require_pinned('freeze-primary', 'test')


def test_freeze_manifest_is_deterministic_and_detects_tampering(tmp_path):
    root = tmp_path / 'attempt'
    stage = runner.StageDir(root / '02-fit')
    files = [stage.npy('b1/league.npy', np.arange(18)), stage.json('b1/b1.json', {'alphas': {'pitcher': 0}}),
             stage.text('b2/model.txt', 'tree\n')]
    first = runner.canonical_bytes(runner.build_manifest(root, files, name='freeze-primary', groups={'B1': {}},
                                                         extra={'config_sha256': 'c'}))
    second = runner.canonical_bytes(runner.build_manifest(root, list(reversed(files)), name='freeze-primary',
                                                          groups={'B1': {}}, extra={'config_sha256': 'c'}))
    assert first == second
    manifest = json.loads(first)
    assert runner.verify_manifest(root, manifest)
    assert np.array_equal(runner.load_frozen_npy(root, manifest, '02-fit/b1/league.npy'), np.arange(18))
    stage.npy('b1/league.npy', np.arange(18) + 1)
    with pytest.raises(IntegrityError, match='changed'):
        runner.verify_manifest(root, manifest)
    with pytest.raises(IntegrityError, match='changed'):
        runner.load_frozen_npy(root, manifest, '02-fit/b1/league.npy')
    with pytest.raises(IntegrityError):
        runner.load_frozen_text(root, manifest, '02-fit/b2/missing.txt')


def test_chains_are_tamper_evident(tmp_path):
    path = tmp_path / 'stages.jsonl'
    runner.append_chain(path, {'stage': 'smoke', 'status': 'PASS'})
    runner.append_chain(path, {'stage': 'census', 'status': 'PASS'})
    assert [r['seq'] for r in runner.read_chain(path)] == [0, 1]
    lines = path.read_bytes().splitlines()
    path.write_bytes(lines[0].replace(b'PASS', b'FAIL') + b'\n' + lines[1] + b'\n')
    with pytest.raises(IntegrityError, match='chain broken'):
        runner.read_chain(path)


def test_holm_tie_rule_and_exhaustive_labels(draft):
    family = draft['verdict_rule']['eligibility']['E1_nll_superiority']['holm_family']
    good = {'status': 'measured', 'nll': {'delta': -0.01, 'ci95': [-0.02, -0.001], 'p_less': 0.01},
            'ece': {'ci95': [-0.001, 0.001]}, 'p1': {'ci95': [-1e-4, 1e-4]}}
    calibration = {'B1': {'status': 'OK'}, 'B2': {'status': 'OK'}}
    holm = runner.holm_adjust([0.01, 0.04])
    assert holm == [0.02, 0.04]
    partial = runner.partial_eligibility(draft, {'B1': good, 'B2': good}, holm, family, calibration)
    assert partial['B1']['E1'] and partial['B2']['E1'] and partial['B1']['E2'] and partial['B1']['E3']
    small = {**good, 'nll': {**good['nll'], 'delta': -0.004}}
    wide = {**good, 'nll': {**good['nll'], 'ci95': [-0.02, 0.0]}}
    worse_ece = {**good, 'ece': {'ci95': [-0.001, 0.0021]}}
    worse_p1 = {**good, 'p1': {'ci95': [0, 0.0006]}}
    out = runner.partial_eligibility(draft, {'B1': small, 'B2': wide}, [0.01, 0.01], family, calibration)
    assert not out['B1']['E1'] and not out['B2']['E1']
    out = runner.partial_eligibility(draft, {'B1': good, 'B2': good}, [0.051, 0.05], family, calibration)
    assert not out['B1']['E1'] and out['B2']['E1']
    out = runner.partial_eligibility(draft, {'B1': worse_ece, 'B2': worse_p1}, [0.01, 0.01], family, calibration)
    assert not out['B1']['E2'] and not out['B2']['E3']
    failed = runner.partial_eligibility(draft, {'B1': good}, [None, 0.01], family,
                                        {'B1': {'status': 'CALIBRATION_FAILED'}, 'B2': {'status': 'OK'}})
    assert failed['B1'] == {'status': 'CALIBRATION_FAILED'} and failed['B2']['status'] == 'NOT_EVALUABLE'
    status, _ = runner.candidate_status({'status': 'CALIBRATION_FAILED'}, True, True)
    assert status == 'CALIBRATION_FAILED'
    assert runner.candidate_status(partial['B1'], None, True)[0] == 'NOT_EVALUABLE'
    assert runner.candidate_status(partial['B1'], False, True)[0] == 'INELIGIBLE'
    assert runner.candidate_status(partial['B1'], True, True)[0] == 'ELIGIBLE'
    tie = draft['verdict_rule']['selection']['tie_margin']
    label = lambda s, nll, ok=True: runner.final_label(ok, s, nll, tie_margin=tie, simpler='B1')
    both = {'B1': 'ELIGIBLE', 'B2': 'ELIGIBLE'}
    assert label(both, {'B1': 1.0, 'B2': 1.0}) == 'ADOPT_FOR_2027_REGISTRATION:B1'
    assert label(both, {'B1': 1.0, 'B2': 1.0 - 2 ** -10}) == 'ADOPT_FOR_2027_REGISTRATION:B1'  # < 0.001: simpler
    assert label(both, {'B1': 1.0, 'B2': 1.0 - 2 ** -9}) == 'ADOPT_FOR_2027_REGISTRATION:B2'
    assert label(both, {'B1': 0.251, 'B2': 0.25}) == 'ADOPT_FOR_2027_REGISTRATION:B2'  # 0.001 is not < 0.001
    assert label(both, {'B1': 0.25, 'B2': 0.251}) == 'ADOPT_FOR_2027_REGISTRATION:B1'
    assert label({'B1': 'INELIGIBLE', 'B2': 'ELIGIBLE'}, {'B1': 0.1, 'B2': 1.}) == 'ADOPT_FOR_2027_REGISTRATION:B2'
    assert label({'B1': 'CALIBRATION_FAILED', 'B2': 'INELIGIBLE'}, {'B1': None, 'B2': 1.}) == 'KEEP_B0'
    assert label({'B1': 'NOT_EVALUABLE', 'B2': 'NOT_EVALUABLE'}, {'B1': None, 'B2': None}) == 'NOT_EVALUABLE'
    assert label(both, {'B1': 1., 'B2': 1.}, ok=False) == 'FAILED_INTEGRITY'
    seen = set()
    for s1, s2, ok, gap in itertools.product(reg.CANDIDATE_STATUSES, reg.CANDIDATE_STATUSES, (True, False),
                                             (0.0, 2 ** -10, 0.01, -0.01)):
        result = label({'B1': s1, 'B2': s2}, {'B1': 1.0, 'B2': 1.0 + gap}, ok)
        assert result in reg.DECISION_LABELS
        seen.add(result)
    assert seen == set(reg.DECISION_LABELS)
    assert set(reg.RUN_LABELS) == set(reg.DECISION_LABELS) | set(reg.TERMINAL_FAILURES)


def test_native_runtime_rule(monkeypatch):
    import importlib.util
    torch_lib = Path(importlib.util.find_spec('torch').origin).resolve().parent / 'lib'
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', 'TRUE')
    with pytest.raises(runner.Refused, match='KMP_DUPLICATE_LIB_OK'):
        runner.validate_native_runtime()
    monkeypatch.delenv('KMP_DUPLICATE_LIB_OK')
    monkeypatch.setenv('DYLD_LIBRARY_PATH', str(torch_lib))
    assert runner.validate_native_runtime()
    if sys.platform == 'darwin':
        monkeypatch.setenv('DYLD_LIBRARY_PATH', '/opt/homebrew/opt/libomp/lib')
        with pytest.raises(runner.Refused, match='one OpenMP runtime'):
            runner.validate_native_runtime()


@nested
def test_end_to_end_synthetic_attempt_isolated():
    """smoke -> decide on synthetic data in a fresh interpreter (isolated_behavior_calibration_runner.py)."""
    done = run_isolated('isolated_behavior_calibration_runner.py')
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert ' passed' in done.stdout and 'failed' not in done.stdout


@nested
def test_registered_preflight_then_smoke_isolated():
    """Review finding 1: launch() with a REGISTERED config runs the real preflight test files (against
    that registered config) and then the smoke on synthetic data (isolated_behavior_calibration_launch.py)."""
    done = run_isolated('isolated_behavior_calibration_launch.py', timeout=1500)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert ' passed' in done.stdout and 'failed' not in done.stdout
