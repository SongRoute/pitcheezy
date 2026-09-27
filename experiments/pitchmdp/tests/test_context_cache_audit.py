"""Cache audit runner: synthetic CPU guards only; no real data, MPS or fits of registered size."""
from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from pitchmdp.data import hash_file
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_observed_context_cache import FrozenObservedContext
from pitchmdp.matrix_sharing import ContinuousPitcherContext
from pitchmdp.sequence_model import SequenceContext
import pitchmdp.matrix_context_cache_audit as audit
import audit_ml_context_cache as runner
from test_observed_context_cache import fixture
from test_lazy_delivery_reuse import FrozenSyntheticPool

COUNTS = {'train': 24, 'train_evaluation': 6, 'earlystop': 8, 'temperature': 3}
SCRIPT = PROJECT/'scripts'/'audit_ml_context_cache.py'
COMMON = ['--config', 'c', '--local-config', 'l', '--output', 'o']


def batches(length):
    """Seventy chronological synthetic rows: five date-shifted copies of the shared 14-row fixture."""
    base_frame, _ = fixture()
    copies = []
    for k in range(5):
        part = base_frame.copy()
        part['game_date'] = part['game_date']+pd.Timedelta(days=10*k); part['game_pk'] = part['game_pk']+1000*k
        copies.append(part)
    frame = pd.concat(copies, ignore_index=True)
    frame['split'] = 'train'; frame.loc[40:, 'split'] = 'earlystop'; frame.loc[60:, 'split'] = 'temperature'
    clusters = {'columns': [f'profile{i}' for i in range(23)], 'pitcher_cluster': {'11': 0},
                'pitcher_profiles': {'11': np.arange(23).tolist()}, 'pitcher_counts': {'11': len(frame)}}
    base = ContinuousPitcherContext(SequenceContext().fit(frame.loc[frame.split.eq('train')]), clusters)
    store = MatrixLongHistoryStore.from_frame(frame, long_length=length)
    return {name: LazyPitchBatch(store, base, np.flatnonzero(frame.split.eq(name).to_numpy()))
            for name in ('train', 'earlystop', 'temperature')}


def draft():
    return json.loads((PROJECT.parents[1]/'configs'/'EXP-P4-002-v3-cache-audit-v2.yaml').read_text())


def prior_ledger(tmp_path, preparation, *, seconds=1., mutate=None):
    """Valid five-category owner ledger v2 with one verified evidence file."""
    evidence = tmp_path/'evidence.json'; evidence.write_text('{"queue": true}')
    entries = [{'category': category, 'attempt': f'{category}/1', 'step': 'x', 'seconds': seconds, 'exit_code': 0,
                'evidence': [{'path': str(evidence), 'sha256': hash_file(evidence)}]} for category in sorted(audit.PRIOR_LEDGER_CATEGORIES)]
    ledger = {'protocol': audit.PRIOR_LEDGER_PROTOCOL, 'preparation_sha256': preparation, 'entries': entries,
              'elapsed_seconds_total': seconds*5, 'entry_count': 5, 'category_seconds': {c: seconds for c in audit.PRIOR_LEDGER_CATEGORIES}}
    if mutate: mutate(ledger)
    path = tmp_path/'prior.json'; runner.dump(path, ledger)
    return path


# ------------------------------------------------------------------ configuration

def test_draft_config_accepted_only_as_draft_and_no_relaxation_or_reduction():
    config = draft()
    assert runner.config_check(config, real=False)['device'] == 'mps'
    with pytest.raises(ValueError, match='Null identity pin'): runner.config_check(config, real=True)
    assert set(config['sources']) == set(runner.SOURCES)
    assert config['contracts']['optional_cache']['sha256'] == hash_file(PROJECT.parents[1]/runner.CONTRACTS['optional_cache'])
    assert config['contracts']['runner']['sha256'] is None and config['limits']['command_seconds']['stage3'] <= 600
    for mutate in [lambda c: c['stage2']['tolerances']['gradients'].update(atol=1e-3),
                   lambda c: c['stage3']['tolerances']['probability'].update(atol=1e-3),
                   lambda c: c['stage3']['tolerances'].pop('calibration_objective'),
                   lambda c: c['samples']['train'].update(n=8192),
                   lambda c: c['stage3']['budget'].update(epochs=1),
                   lambda c: c.update(device='cpu'),
                   lambda c: c.update(arms=['F4-H0']),
                   lambda c: c['limits']['command_seconds'].update(stage3=601),
                   lambda c: c['limits']['command_seconds'].update(stage1=7201),
                   lambda c: c['limits'].update(termination_grace_seconds=600),
                   lambda c: c['limits'].update(wall_ledger_dir=''),
                   lambda c: c['limits'].update(profile_command_cap_seconds=7200),
                   lambda c: c['contracts'].pop('runner'),
                   lambda c: c['contracts']['optional_cache'].update(path='docs/other.md'),
                   lambda c: c['stage1'].update(samples=['train']),
                   lambda c: c.update(adoption=True),
                   lambda c: c.update(extra_axis=1),
                   lambda c: c['samples'].update(dev_or_june_cache='allowed')]:
        changed = deepcopy(config); mutate(changed)
        with pytest.raises(ValueError): runner.config_check(changed, real=False)
    frozen = deepcopy(config); frozen['repo_commit'] = 'a'*40; frozen['contracts']['runner']['sha256'] = 'b'*64
    frozen['sources'] = {name: 'c'*64 for name in runner.SOURCES}
    assert runner.config_check(frozen, real=True)


def test_cli_help_and_generated_worker_argv_parse():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and 'stage3' in result.stdout and 'summary' in result.stdout
    tail = [*COMMON, 'stage3', '--arm', 'F4-H0', '--path', 'cached', '--attempt', '2']
    argv = runner.worker_argv(tail)
    assert argv[:2] == [sys.executable, str(SCRIPT.resolve())]
    args = runner.build_parser().parse_args(argv[2:])
    assert args.worker and args.command == 'stage3' and args.path == 'cached' and args.attempt == 2
    with pytest.raises(SystemExit): runner.build_parser().parse_args([*tail, '--worker'])
    with pytest.raises(SystemExit): runner.build_parser().parse_args([*COMMON, 'stage3', '--arm', 'F4-H0', '--attempt', '1'])
    with pytest.raises(SystemExit): runner.build_parser().parse_args([*COMMON, 'prepare'])
    for command, extra in [('prepare', []), ('summary', []), ('stage1', ['--arm', 'F4-32']), ('compare', ['--arm', 'F4-128'])]:
        parsed = runner.build_parser().parse_args(runner.worker_argv([*COMMON, command, *extra, '--attempt', '1'])[2:])
        assert parsed.worker and parsed.command == command


# ------------------------------------------------------------------ supervisor wall ledger

def supervisor_fixture(tmp_path, *, caps=None, prior_seconds=1., prior_mutate=None):
    config = draft()
    root = tmp_path/'artifacts'; (root/'runs'/'ML-MATRIX-20260924').mkdir(parents=True)
    config['limits']['wall_ledger_dir'] = str(root/'runs'/'ML-MATRIX-20260924'/'audit'/'ledger')
    if caps: config['limits']['command_seconds'].update(caps)
    ledger = prior_ledger(tmp_path, config['parent']['preparation_sha256'], seconds=prior_seconds, mutate=prior_mutate)
    config['prior_cost_ledger'] = {'path': str(ledger), 'sha256': hash_file(ledger)}
    local = {'artifact_root': str(root)}
    output = root/'runs'/'ML-MATRIX-20260924'/'EXP-P4-002-v3-cache-audit-v2'
    return config, local, output, Path(config['limits']['wall_ledger_dir'])


def fake_popen(code, seen):
    def popen(argv):
        seen.append(argv)
        return SimpleNamespace(wait=lambda timeout=None: code, terminate=lambda: None, kill=lambda: None)
    return popen


def test_supervisor_records_start_and_one_outcome_with_full_wall_and_parses_child_argv(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    seen = []
    argv = runner.worker_argv([*COMMON, 'stage1', '--arm', 'F4-H0', '--attempt', '1'])
    outcome, code = runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, argv, popen=fake_popen(0, seen))
    assert (outcome, code) == ('completed', 0)
    assert runner.build_parser().parse_args(seen[0][2:]).worker
    state = runner.ledger_state(ledger)
    assert len(state['jobs']) == 1 and state['jobs'][0]['ended'] and state['jobs'][0]['outcome'] == 'completed'
    assert 0 <= state['charged_seconds'] < config['limits']['command_seconds']['stage1'] and state['reserved_seconds'] == 0.
    start = runner.read_json(next((ledger/'jobs').glob('*.json')))
    assert start['cap_seconds'] == 1200 and start['argv'] == argv and start['audit_charged_before_launch_seconds'] == 0.
    with pytest.raises(RuntimeError, match='already launched'):
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, argv, popen=fake_popen(0, seen))
    assert len(runner.ledger_state(ledger)['jobs']) == 1
    with pytest.raises(RuntimeError, match='never overwrite'):
        runner.end_job(ledger, start['job_id'], 'completed', 0, 1., 1200, '')


def test_supervisor_charges_startup_failure_not_equivalent_and_launch_exception(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    argv = runner.worker_argv([*COMMON, 'prepare', '--attempt', '1'])
    assert runner.supervise('prepare', None, None, 1, config, local, output, argv, popen=fake_popen(1, []))[0] == 'failed'
    assert runner.supervise('stage2', 'F4-32', None, 1, config, local, output, argv, popen=fake_popen(runner.EXIT_NOT_EQUIVALENT, []))[0] == 'not_equivalent'
    def broken(argv): raise OSError('cannot launch')
    with pytest.raises(OSError): runner.supervise('summary', None, None, 1, config, local, output, argv, popen=broken)
    state = runner.ledger_state(ledger)
    assert [j['outcome'] for j in state['jobs']] == ['failed', 'not_equivalent', 'failed'] and all(j['ended'] for j in state['jobs'])
    assert state['jobs'][2]['exit_code'] == -1 and state['unresolved_jobs'] == []


@pytest.mark.parametrize('ignore_term', [False, True])
def test_supervisor_hard_deadline_terms_then_kills_and_records_actual_wall(tmp_path, ignore_term):
    config, local, output, ledger = supervisor_fixture(tmp_path, caps={'stage3': 1.2})
    config['limits']['termination_grace_seconds'] = .4
    program = ('import signal, time; ' + ('signal.signal(signal.SIGTERM, signal.SIG_IGN); ' if ignore_term else '') + 'time.sleep(30)')
    popen = lambda argv: subprocess.Popen([sys.executable, '-c', program])
    outcome, code = runner.supervise('stage3', 'F4-H0', 'cached', 1, config, local, output, ['x'], popen=popen)
    assert outcome == 'timeout' and code in (-9, -15) and (code == -9) == ignore_term
    job = runner.ledger_state(ledger)['jobs'][0]
    end = runner.read_json(ledger/'ends'/(job['job_id']+'.json'))
    assert .8 <= end['elapsed_seconds'] < 5. and end['outcome'] == 'timeout' and 'SIG' in end['note']
    assert job['ended'] and job['charged_seconds'] == end['elapsed_seconds']


def test_supervisor_interrupt_after_launch_reaps_child_before_ending_job(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    launched = []

    class InterruptingChild:
        def __init__(self):
            self.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            self.first_wait = True

        def wait(self, timeout=None):
            if self.first_wait:
                self.first_wait = False
                raise KeyboardInterrupt()
            return self.process.wait(timeout=timeout)

        def terminate(self): self.process.terminate()
        def kill(self): self.process.kill()

    def launch(argv):
        child = InterruptingChild(); launched.append(child)
        return child

    with pytest.raises(KeyboardInterrupt):
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, ['x'], popen=launch)
    assert len(launched) == 1 and launched[0].process.poll() is not None
    state = runner.ledger_state(ledger)
    assert len(state['jobs']) == 1 and state['jobs'][0]['ended']
    assert state['jobs'][0]['outcome'] == 'failed' and state['unresolved_jobs'] == []


def test_supervisor_sigterm_after_launch_reaps_child_and_restores_handler(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    launched = []

    class SignalledChild:
        def __init__(self):
            self.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
            self.first_wait = True
        def wait(self, timeout=None):
            if self.first_wait:
                self.first_wait = False
                signal.raise_signal(signal.SIGTERM)
            return self.process.wait(timeout=timeout)
        def terminate(self): self.process.terminate()
        def kill(self): self.process.kill()

    def launch(argv):
        child = SignalledChild(); launched.append(child)
        return child

    with pytest.raises(KeyboardInterrupt, match='Supervisor received signal'):
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, ['x'], popen=launch)
    assert signal.getsignal(signal.SIGTERM) is previous
    assert launched[0].process.poll() is not None
    assert runner.ledger_state(ledger)['jobs'][0]['ended']


def test_supervisor_sigterm_during_launch_defers_until_child_is_known(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    launched = []
    def launch(argv):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        launched.append(child)
        signal.raise_signal(signal.SIGTERM)
        return child
    with pytest.raises(KeyboardInterrupt, match='during launch'):
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, ['x'], popen=launch)
    assert launched[0].poll() is not None
    assert runner.ledger_state(ledger)['jobs'][0]['ended']


def test_termination_and_kill_share_one_grace_interval():
    class SlowChild:
        def __init__(self): self.waits = []; self.killed = False
        def terminate(self): pass
        def kill(self): self.killed = True
        def wait(self, timeout=None):
            self.waits.append(timeout)
            if not self.killed:
                time.sleep(timeout)
                raise subprocess.TimeoutExpired('worker', timeout)
            return -9
    child = SlowChild()
    code, reaped, note = runner.terminate_and_reap(child, .08)
    assert reaped and code == -9 and 'SIGKILL' in note
    assert len(child.waits) == 2 and sum(child.waits) <= .081


def test_supervisor_keeps_start_unresolved_if_child_cannot_be_reaped(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)

    class UnreapableChild:
        def __init__(self): self.signals = []
        def wait(self, timeout=None):
            if not self.signals: raise KeyboardInterrupt()
            raise subprocess.TimeoutExpired('worker', timeout)
        def terminate(self): self.signals.append('TERM')
        def kill(self): self.signals.append('KILL')

    child = UnreapableChild()
    with pytest.raises(KeyboardInterrupt):
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, ['x'], popen=lambda argv: child)
    assert child.signals == ['TERM', 'KILL']
    state = runner.ledger_state(ledger)
    assert len(state['jobs']) == 1 and not state['jobs'][0]['ended']
    assert state['reserved_seconds'] == config['limits']['command_seconds']['stage1']
    assert not list((ledger/'ends').glob('*.json'))


def test_supervisor_cap_includes_prior_validation_and_start_overhead(tmp_path, monkeypatch):
    config, local, output, ledger = supervisor_fixture(tmp_path, caps={'stage3': 1.2})
    config['limits']['termination_grace_seconds'] = .4
    original = runner.prior_seconds
    def slow_prior(config):
        time.sleep(.05)
        return original(config)
    monkeypatch.setattr(runner, 'prior_seconds', slow_prior)
    waits = []
    def launch(argv):
        return SimpleNamespace(wait=lambda timeout=None: waits.append(timeout) or 0,
                               terminate=lambda: None, kill=lambda: None)
    assert runner.supervise('stage3', 'F4-H0', 'cached', 1, config, local, output, ['x'], popen=launch)[0] == 'completed'
    assert len(waits) == 1 and 0 < waits[0] < .75  # 1.2 cap - .4 grace - >=.05 preflight
    assert runner.ledger_state(ledger)['jobs'][0]['charged_seconds'] >= .05


def test_ledger_state_ignores_only_verified_appledouble_json(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    runner.start_job(ledger, config, 1., 'stage1', 'F4-H0', None, 1, ['x'])
    sidecar = ledger/'jobs'/'._metadata.json'
    sidecar.write_bytes(bytes.fromhex('00051607') + b'meta')
    assert len(runner.ledger_state(ledger)['jobs']) == 1
    sidecar.write_text('not a JSON job')
    with pytest.raises(json.JSONDecodeError): runner.ledger_state(ledger)


def test_unresolved_job_reserves_cap_and_blocks_launch_until_resolved(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path)
    job_id, cap = runner.start_job(ledger, config, 5., 'stage1', 'F4-H0', None, 1, ['x'])
    state = runner.ledger_state(ledger)
    assert state['reserved_seconds'] == cap == 1200 and state['charged_seconds'] == 1200 and state['unresolved_jobs'] == [job_id]
    with pytest.raises(RuntimeError, match='Unresolved audit job'):
        runner.supervise('stage2', 'F4-H0', None, 1, config, local, output, ['x'], popen=fake_popen(0, []))
    runner.end_job(ledger, job_id, 'completed', 0, 7.5, cap, 'resolved by root')
    state = runner.ledger_state(ledger)
    assert state['charged_seconds'] == 7.5 and state['reserved_seconds'] == 0.
    assert runner.supervise('stage2', 'F4-H0', None, 1, config, local, output, ['x'], popen=fake_popen(0, []))[0] == 'completed'


def test_family_budget_gate_counts_prior_plus_audit_plus_cap_before_launch(tmp_path):
    config, local, output, ledger = supervisor_fixture(tmp_path, prior_seconds=(28800-600-100)/5)   # 28100 prior
    with pytest.raises(RuntimeError, match='Launch refused'):   # 28100 + 1200 cap > 28800 even with an empty audit ledger
        runner.supervise('stage1', 'F4-H0', None, 1, config, local, output, ['x'], popen=fake_popen(0, []))
    assert runner.ledger_state(ledger)['jobs'] == []
    assert runner.supervise('stage3', 'F4-H0', 'original', 1, config, local, output, ['x'], popen=fake_popen(0, []))[0] == 'completed'
    job_id, _ = runner.start_job(ledger, config, runner.prior_seconds(config), 'stage3', 'F4-H0', 'cached', 1, ['x'])
    runner.end_job(ledger, job_id, 'completed', 0, 150., 600, '')
    with pytest.raises(RuntimeError, match='Launch refused'):   # 28100 + ~0 + 150 + 600 cap > 28800
        runner.supervise('compare', 'F4-H0', None, 1, config, local, output, ['x'], popen=fake_popen(0, []))
    assert len(runner.ledger_state(ledger)['jobs']) == 2
    with pytest.raises(ValueError, match='outside the audit output'):
        runner.ledger_location(local, output, str(output/'ledger'))
    with pytest.raises(ValueError, match='artifact root'):
        runner.ledger_location(local, output, str(tmp_path/'elsewhere'))


# ------------------------------------------------------------------ owner prior-cost ledger

def test_prior_ledger_v2_rejects_negative_duplicate_bad_evidence_missing_category_and_wrong_total(tmp_path):
    preparation = 'b'*64
    good = prior_ledger(tmp_path, preparation, seconds=2.); valid = runner.read_json(good)   # later fixtures reuse the path
    assert audit.validate_prior_ledger(valid, preparation, lambda p: hash_file(Path(p))) == 10.
    with pytest.raises(ValueError, match='another parent'):
        audit.validate_prior_ledger(valid, 'c'*64, lambda p: hash_file(Path(p)))
    for match, mutate in [
            ('nonnegative', lambda l: l['entries'][0].update(seconds=-1.)),
            ('Distinct', lambda l: l['entries'][1].update(attempt=l['entries'][0]['attempt'])),
            ('evidence changed', lambda l: l['entries'][2]['evidence'][0].update(sha256='0'*64)),
            ('needs evidence', lambda l: l['entries'][2].update(evidence=[])),
            ('exactly preparation', lambda l: l['entries'].pop()),
            ('exactly preparation', lambda l: l['entries'].append({**l['entries'][0], 'attempt': 'new', 'category': 'other'})),
            ('total differs', lambda l: l.update(elapsed_seconds_total=9.)),
            ('category totals', lambda l: l['category_seconds'].update(profiles=9.)),
            ('entry count', lambda l: l.update(entry_count=4)),
            ('protocol', lambda l: l.update(protocol='ml_long_owner_budget_ledger_v1')),
            ('entry fields', lambda l: l['entries'][0].update(bonus=1)),
            ('carry protocol', lambda l: l.update(extra=1))]:
        path = prior_ledger(tmp_path, preparation, seconds=2., mutate=mutate)
        with pytest.raises(ValueError, match=match):
            audit.validate_prior_ledger(runner.read_json(path), preparation, lambda p: hash_file(Path(p)))
    nonfinite = deepcopy(valid); nonfinite['entries'][0]['seconds'] = float('nan')   # JSON cannot carry NaN; check in memory
    with pytest.raises(ValueError, match='nonnegative'):
        audit.validate_prior_ledger(nonfinite, preparation, lambda p: hash_file(Path(p)))
    good = prior_ledger(tmp_path, preparation, seconds=2.)
    config = draft(); config['prior_cost_ledger'] = {'path': str(good), 'sha256': 'f'*64}
    with pytest.raises(ValueError, match='ledger changed'): runner.prior_seconds(config)


# ------------------------------------------------------------------ predecessors and seals

def seal_result(output, command, arm, path, attempt, result, dependencies=None):
    directory = runner.fresh(runner.stage_root(output, command, arm, path)/f'attempt{attempt}')
    runner.dump(directory/'results.json', result)
    return runner.seal(directory, runner.stage_identity('reg', command, arm, path, attempt, dependencies))


def test_predecessor_chain_requires_passed_sealed_and_untampered_evidence(tmp_path):
    output = tmp_path
    with pytest.raises(ValueError, match='Exactly one sealed stage1'): runner.predecessors(output, 'reg', 'stage2', 'F4-H0')
    seal_result(output, 'stage1', 'F4-H0', None, 1, {'all_bitwise_identical': True, 'all_guards_rejected': False})
    with pytest.raises(ValueError, match='did not pass'): runner.predecessors(output, 'reg', 'stage2', 'F4-H0')
    one = seal_result(output, 'stage1', 'F4-H0', None, 2, {'all_bitwise_identical': True, 'all_guards_rejected': True})
    with pytest.raises(ValueError, match='found 2'): runner.predecessors(output, 'reg', 'stage2', 'F4-H0')
    (output/'stage1'/'F4-H0'/'attempt1'/'manifest.json').unlink()
    assert runner.predecessors(output, 'reg', 'stage2', 'F4-H0') == {'stage1/F4-H0': one}
    with pytest.raises(ValueError, match='Exactly one sealed stage2'): runner.predecessors(output, 'reg', 'stage3', 'F4-H0')
    seal_result(output, 'stage2', 'F4-H0', None, 1, {'equivalent': False}, {'stage1/F4-H0': one})
    with pytest.raises(ValueError, match='did not pass'): runner.predecessors(output, 'reg', 'stage3', 'F4-H0')
    (output/'stage2'/'F4-H0'/'attempt1'/'manifest.json').unlink()
    two = seal_result(output, 'stage2', 'F4-H0', None, 2, {'equivalent': True}, {'stage1/F4-H0': one})
    chain = runner.predecessors(output, 'reg', 'stage3', 'F4-H0')
    assert chain == {'stage1/F4-H0': one, 'stage2/F4-H0': two}
    with pytest.raises(ValueError, match='Exactly one sealed stage3'): runner.predecessors(output, 'reg', 'compare', 'F4-H0')
    for path in runner.PATHS: seal_result(output, 'stage3', 'F4-H0', path, 1, {'path': path}, chain)
    full = runner.predecessors(output, 'reg', 'compare', 'F4-H0')
    assert set(full) == {'stage1/F4-H0', 'stage2/F4-H0', 'stage3/F4-H0/original', 'stage3/F4-H0/cached'}
    # Stage2 sealed against a different stage1 digest is rejected as a tampered chain.
    seal_result(output, 'stage2', 'F4-32', None, 1, {'equivalent': True}, {'stage1/F4-32': 'x'*64})
    seal_result(output, 'stage1', 'F4-32', None, 1, {'all_bitwise_identical': True, 'all_guards_rejected': True})
    with pytest.raises(ValueError, match='identity/artifact family differs'): runner.predecessors(output, 'reg', 'stage3', 'F4-32')
    # Tampering a sealed predecessor's result after sealing is caught by its own manifest.
    (output/'stage1'/'F4-H0'/'attempt2'/'results.json').write_text('{"all_bitwise_identical": true, "all_guards_rejected": true, "x": 1}')
    with pytest.raises(ValueError, match='Artifact identity changed'): runner.predecessors(output, 'reg', 'stage2', 'F4-H0')
    assert runner.stage_passed('compare', {'equivalent': False}) is False and runner.stage_passed('prepare', {}) is True


def test_all_three_arms_complete_successful_predecessor_chains(tmp_path):
    output = tmp_path
    for arm in runner.CELLS:
        one = seal_result(output, 'stage1', arm, None, 1,
                          {'all_bitwise_identical': True, 'all_guards_rejected': True})
        two = seal_result(output, 'stage2', arm, None, 1, {'equivalent': True}, {f'stage1/{arm}': one})
        fit_chain = {f'stage1/{arm}': one, f'stage2/{arm}': two}
        for path in runner.PATHS:
            seal_result(output, 'stage3', arm, path, 1, {'path': path}, fit_chain)
        compare_chain = runner.predecessors(output, 'reg', 'compare', arm)
        seal_result(output, 'compare', arm, None, 1, {'equivalent': True}, compare_chain)
    for arm in runner.CELLS:
        chain = runner.predecessors(output, 'reg', 'compare', arm)
        found, _ = runner.completed_attempt(output, 'reg', 'compare', arm, None, chain)
        assert runner.read_json(found/'results.json')['equivalent'] is True
        assert set(chain) == {f'stage1/{arm}', f'stage2/{arm}',
                              f'stage3/{arm}/original', f'stage3/{arm}/cached'}


def test_fresh_output_allows_only_appledouble_and_sealing_detects_changes(tmp_path):
    target = tmp_path/'stage1'/'F4-H0'/'attempt1'
    runner.fresh(target)
    (target/'._results.json').write_bytes(bytes.fromhex('00051607')+b'meta')
    runner.fresh(target)
    (target/'results.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh attempt'): runner.fresh(target)
    identity = runner.stage_identity('reg', 'stage1', 'F4-H0', None, 1)
    runner.seal(target, identity)
    assert set(json.loads((target/'manifest.json').read_text())['artifact_hashes']) == {'results.json'}
    assert runner.sealed(target, identity) == hash_file(target/'manifest.json')
    with pytest.raises(ValueError, match='identity'): runner.sealed(target, runner.stage_identity('other', 'stage1', 'F4-H0', None, 1))
    (target/'extra.json').write_text('{}')
    with pytest.raises(ValueError): runner.sealed(target, identity)
    (target/'extra.json').unlink(); (target/'results.json').write_text('{"changed": 1}')
    with pytest.raises(ValueError, match='Artifact identity changed'): runner.sealed(target, identity)


def parent_fixture(tmp_path, config):
    root = tmp_path/'artifacts'; parent = root/'runs'/'ML-MATRIX-20260924'/'EXP-P4-002-v3'; parent.mkdir(parents=True)
    (parent/'source').mkdir(); (parent/'source'/'frozen.py').write_text('x')
    (parent/'features.json').write_text('{}')
    prep = {'identity': {'source_hashes': {}}, 'artifact_hashes': {'features.json': hash_file(parent/'features.json')},
            'features': {'long_stream': {'version': 'batter_dual_stream_v2'}}, 'dev_scores_read': False,
            'samples': {name: {'n': n} for name, n in [('train', 100000), ('earlystop', 5000), ('temperature', 640), ('blend', 1000), ('dev', 2000)]}}
    runner.dump(parent/'preparation.json', prep)
    repo = tmp_path/'repo'; (repo/'configs').mkdir(parents=True); (repo/'configs'/'EXP-P4-002-v3.yaml').write_text('{}')
    config['parent'].update(run=str(parent), config_sha256=hash_file(repo/'configs'/'EXP-P4-002-v3.yaml'),
                            preparation_sha256=hash_file(parent/'preparation.json'))
    for cell, length in runner.CELLS.items():
        directory = parent/'profiles'/cell; directory.mkdir(parents=True)
        profile = {'long_length': length, 'dev_scores_read': False, 'projection': {'load_fit_temperature_prediction_seconds': 5000.},
                   'samples': {name: {'n': config['samples'][name]['n'], 'rows_sha256': config['samples'][name]['rows_sha256']} for name in ('train', 'earlystop', 'temperature')},
                   'warmup': {'samples': {'train': {'n': 8192, 'rows_sha256': 'w'*64},
                                          'evaluation_train': {'n': 2048, 'rows_sha256': config['samples']['train_evaluation']['rows_sha256']}}}}
        runner.dump(directory/'profile.json', profile)
        runner.dump(directory/'manifest.json', {'identity': {'preparation_sha256': config['parent']['preparation_sha256'], 'cell': cell, 'profile_version': 'f4_resource_profile_v2'},
                                               'artifact_hashes': {'profile.json': hash_file(directory/'profile.json')}})
        config['parent']['profiles'][cell] = {'profile_sha256': hash_file(directory/'profile.json'), 'manifest_sha256': hash_file(directory/'manifest.json')}
    return {'artifact_root': str(root)}, repo, parent


def test_verify_parent_binds_preparation_profiles_and_registered_selectors(tmp_path):
    config = draft(); local, repo, parent = parent_fixture(tmp_path, config)
    _, prep, profiles = runner.verify_parent(config, local, repo=repo)
    assert set(profiles) == set(runner.CELLS) and prep['dev_scores_read'] is False
    wrong = deepcopy(config); wrong['parent']['preparation_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='preparation changed'): runner.verify_parent(wrong, local, repo=repo)
    wrong = deepcopy(config); wrong['samples']['train']['rows_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='selectors differ'): runner.verify_parent(wrong, local, repo=repo)
    wrong = deepcopy(config); wrong['parent']['config_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='configuration file changed'): runner.verify_parent(wrong, local, repo=repo)
    (parent/'profiles'/'F4-32'/'profile.json').write_text('{"tampered": true}')
    with pytest.raises(ValueError, match='profile evidence changed'): runner.verify_parent(config, local, repo=repo)


def test_prepare_refuses_existing_output_before_touching_parent(tmp_path, monkeypatch):
    output = tmp_path/'audit'; output.mkdir(); (output/'old.json').write_text('{}')
    monkeypatch.setattr(runner, 'verify_parent', lambda *a, **k: pytest.fail('Existing output must be refused first'))
    with pytest.raises(ValueError, match='Preserve the existing audit directory'): runner.prepare(draft(), {}, output, {})


def test_contract_pins_cover_both_contracts_and_identity_includes_them():
    hashes = runner.contract_hashes()
    assert set(hashes) == {'optional_cache', 'runner'} and hashes['optional_cache'] == draft()['contracts']['optional_cache']['sha256']
    for name in runner.CONTRACTS: assert (PROJECT.parents[1]/runner.CONTRACTS[name]).is_file()


# ------------------------------------------------------------------ stage helpers

def test_selectors_reject_dev_and_track_exact_identity():
    parts = batches(0)
    selected, warm = audit.select_samples(parts, COUNTS, warmup_train=8)
    records = audit.sample_records(selected)
    assert records['train']['n'] == 24 and len(warm) == 8
    np.testing.assert_array_equal(selected['train_evaluation'].rows, warm.rows[np.linspace(0, 7, 6, dtype=np.int64)])
    audit.check_sample_identity(records, records)
    with pytest.raises(ValueError, match='selector identity'):
        audit.check_sample_identity(records, {**records, 'train': {'n': 24, 'rows_sha256': 'x'*64}})
    with pytest.raises(ValueError, match='DEV'):
        audit.select_samples({**parts, 'dev': parts['train']}, COUNTS, warmup_train=8)
    with pytest.raises(ValueError, match='do not shrink'):
        audit.select_samples(parts, {**COUNTS, 'train': 1000}, warmup_train=8)


class Tampered(FrozenObservedContext):
    def transform(self, frame):
        values = super().transform(frame)
        if len(values): values = values.copy(); values[0, 0] += np.float32(1e-7)
        return values


@pytest.mark.parametrize('length', [0, 128])
def test_stage1_checks_both_samples_in_all_orders_guards_timing_and_rejects_unequal_cache(length, monkeypatch):
    selected, _ = audit.select_samples(batches(length), COUNTS, warmup_train=8)
    result = audit.stage1_context(selected, chunk_size=5, repetitions=2, uneven_chunk_sizes=(1, 4), timing_batch_size=4)
    assert result['all_bitwise_identical'] and result['all_guards_rejected'] and result['samples_checked'] == ['train', 'train_evaluation']
    for sample, rows in (('train', 24), ('train_evaluation', 6)):
        checks = result['checks'][sample]
        assert set(checks) == {'rows_sha256', *audit.STAGE1_ORDERS}
        assert checks['original']['rows'] == rows and checks['repeated']['rows'] == 2*rows and checks['reverse']['rows'] == rows
        assert checks['uneven_chunks']['chunk_sizes'] == [1, 4] and checks['uneven_chunks']['chunks'] == {24: 10, 6: 3}[rows]
        assert checks['original']['chunks'] == -(-rows//5) and all(checks[o]['differing_bits'] == 0 for o in audit.STAGE1_ORDERS)
        assert checks['original']['chunk_sha256_chain'] != checks['reverse']['chunk_sha256_chain']
    assert all(result['must_fail'][name]['rejected'] for name in audit.STAGE1_MUST_FAIL)
    assert result['construction']['rng_state_unchanged'] and result['construction']['construction_rows'] == len(np.union1d(selected['train'].rows, selected['train_evaluation'].rows))
    assert [len(v) for kind in result['timing']['seconds'].values() for v in kind.values()] == [2]*4
    assert [row[2] for row in result['timing']['schedule'][:4]] == ['original', 'cached', 'cached', 'original']
    assert result['long_length'] == length and result['dev_scores_read'] is False
    monkeypatch.setattr(audit, 'FrozenObservedContext', Tampered)
    with pytest.raises(ValueError, match='differs bitwise'):
        audit.stage1_context(selected, chunk_size=5, repetitions=2, uneven_chunk_sizes=(1, 4), timing_batch_size=4)


def test_stage2_equivalence_rng_and_unequal_context_rejected(monkeypatch):
    selected, _ = audit.select_samples(batches(32), COUNTS, warmup_train=8)
    torch.set_num_threads(1)
    numpy_state = np.random.get_state()[1].copy()
    result = audit.stage2_backend(selected['train'], device='cpu', minibatches=2, batch_size=4, width=8, chunk_size=3)
    assert result['equivalent'] and result['identical_initialization'] and result['rng_state_unchanged_by_steps']
    assert len(result['steps']) == 2 and result['tolerances'] == audit.STAGE2_TOLERANCES
    assert all(step['updated_weights']['max_absolute_difference'] == 0. for step in result['steps'])
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    with pytest.raises(ValueError, match='exceeds the fixed TRAIN sample'):
        audit.stage2_backend(selected['train'], device='cpu', minibatches=7, batch_size=4, width=8)
    monkeypatch.setattr(audit, 'FrozenObservedContext', Tampered)
    with pytest.raises(ValueError, match='differs bitwise'):
        audit.stage2_backend(selected['train'], device='cpu', minibatches=1, batch_size=4, width=8)


def test_difference_rejects_nonfinite_and_beyond_tolerance():
    assert audit._difference([1., 2.], [1., 2.+5e-7], {'atol': 1e-6, 'rtol': 0.})['pass']
    assert not audit._difference([1., 2.], [1., 2.+5e-6], {'atol': 1e-6, 'rtol': 0.})['pass']
    assert not audit._difference([1., np.nan], [1., np.nan], {'atol': 1e-6, 'rtol': 0.})['pass']
    with pytest.raises(ValueError, match='shape'): audit._difference([1.], [1., 2.], {'atol': 0, 'rtol': 0})


def test_stage3_paired_fit_compare_objective_and_perturbed_outputs_rejected():
    selected, warm = audit.select_samples(batches(32), COUNTS, warmup_train=8)
    torch.set_num_threads(1)
    budget = {'epochs': 2, 'patience': 2, 'batch_size': 4, 'learning_rate': .0005}
    original, oa = audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget)
    rows = np.unique(np.concatenate([warm.rows, *[b.rows for b in selected.values()]]))
    cache, construction = audit.build_cache(selected['train'].context, selected['train'].store.frame, rows, chunk_size=7)
    cached, ca = audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget, cache=cache, path='cached')
    assert construction['construction_rows'] == len(rows) and construction['rng_state_unchanged']
    assert original['profile_weights_reused_as_full_member'] is False and 'log_loss' not in original
    comparison = audit.compare_stage3(original, oa, cached, ca)
    assert comparison['equivalent'] and comparison['selected_state']['max_absolute_difference'] == 0.
    assert comparison['history']['pass'] and comparison['fallback_tiers_equal'] and comparison['calibration_objective']['pass']
    assert comparison['calibration_objective']['original'] == original['calibration_integrated_log_loss']
    with pytest.raises(ValueError, match='in that order'): audit.compare_stage3(cached, ca, original, oa)
    with pytest.raises(ValueError, match='Path label'):
        audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget, cache=cache)
    objective = deepcopy(cached); objective['calibration_integrated_log_loss'] += 1e-5
    verdict = audit.compare_stage3(original, oa, objective, ca)
    assert not verdict['equivalent'] and not verdict['calibration_objective']['pass'] and verdict['calibrated_probability']['pass']
    objective['calibration_integrated_log_loss'] = float('nan')
    assert not audit.compare_stage3(original, oa, objective, ca)['calibration_objective']['finite']
    perturbed = deepcopy(ca); perturbed['probability'][0, 0] += 1e-3
    assert not audit.compare_stage3(original, oa, cached, perturbed)['equivalent']
    perturbed = deepcopy(ca); key = next(iter(perturbed['state'])); perturbed['state'][key] = perturbed['state'][key]+1e-4
    assert not audit.compare_stage3(original, oa, cached, perturbed)['equivalent']
    drifted = deepcopy(cached); drifted['history'][0]['earlystop_conditional_nll'] += 1e-4
    assert not audit.compare_stage3(original, oa, drifted, ca)['equivalent']
    shorter = deepcopy(cached); shorter['best_epoch'] = original['best_epoch']+1
    assert not audit.compare_stage3(original, oa, shorter, ca)['equivalent']


def test_cost_summary_is_a_projection_keeps_full_population_cache_unknown_and_not_adopted():
    def result(fit):
        return {'warmup_seconds': 3., 'fit_seconds': fit, 'optimizer_updates': 1024, 'temperature_seconds': .2,
                'inference_seconds': .3, 'load_seconds': 2.}
    populations = {'train': 100000, 'earlystop': 3000, 'temperature': 640, 'blend': 1000, 'dev': 2000}
    profiles = {cell: {'projection': {'load_fit_temperature_prediction_seconds': 5000.}} for cell in runner.CELLS}
    stage3 = {cell: {'original': result(40.), 'cached': result(20.)} for cell in runner.CELLS}
    constructions = {cell: {'stage3_fit_process_rows': {'construction_seconds': 1.5, 'construction_rows': 77792}} for cell in runner.CELLS}
    summary = audit.cost_summary(profiles, stage3, constructions, populations, prior_seconds=1000., audit_seconds=500.)
    arm = summary['arms']['F4-H0']
    assert arm['cached_member_total_seconds'] is None and arm['full_population_cache_construction_seconds']['fit_process'] == 'unmeasured'
    assert summary['family_projection_seconds'] is None and summary['family_within_budget'] is None and summary['decision'] == 'not_adopted'
    assert 'lower_bound' not in json.dumps(summary) and 'not a guaranteed' in summary['projection_meaning']
    cached = arm['audit_cached_projection_seconds_excluding_cache_construction']
    assert summary['family_projection_excluding_cache_construction_seconds'] == pytest.approx(1500.+9*cached)
    assert summary['family_projection_gate_excluding_cache_construction'] and arm['audit_original_projection_seconds'] > cached
    failed = audit.cost_summary(profiles, stage3, constructions, populations, prior_seconds=28000., audit_seconds=500.)
    assert not failed['family_projection_gate_excluding_cache_construction'] and 'projection gate' in failed['reason'] and failed['decision'] == 'not_adopted'
    with pytest.raises(ValueError, match='update-count'):
        audit.cost_summary(profiles, {c: {p: {**r, 'optimizer_updates': 64} for p, r in v.items()} for c, v in stage3.items()},
                           constructions, populations, prior_seconds=0., audit_seconds=0.)
