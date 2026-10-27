"""EXP-BC-CAL-01 runner: fail-closed registration gate, destination rules, caps and the protection
window (mock clock / RSS), fixed stage order, DEV-label and DEV-metric gates, aborts that keep
outputs, freeze-manifest determinism and tamper detection, Holm / tie rule / exhaustive labels.
The end-to-end synthetic attempt runs in an isolated interpreter (one OpenMP runtime)."""
import copy
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_behavior_calibration as runner  # noqa: E402
from behavior_calibration_synthetic import make_frame, run_isolated  # noqa: E402
from pitchmdp import behavior_calibration_registry as reg  # noqa: E402
from pitchmdp.behavior_features import SEALED  # noqa: E402
from pitchmdp.policy_artifacts import IntegrityError  # noqa: E402

CONFIG = Path(__file__).resolve().parents[3] / 'configs' / 'EXP-BC-CAL-01.json'
GIT_OK = {'head': 'a' * 40, 'dirty': False, 'commit_is_ancestor': True, 'changed_since_commit': [],
          'config_committed': True}
AFTERNOON = datetime(2026, 10, 27, 15, 0, tzinfo=runner.KST)


@pytest.fixture()
def draft():
    return json.loads(CONFIG.read_text())


def registered(config, runner_sha='r' * 64):
    c = copy.deepcopy(config)
    c.update(status='REGISTERED', registered=True, execute=True, real_data_enabled=True)
    c['registration'].update(registered_kst='2026-10-28T14:00+09:00', design_review_reference='design-review',
                             code_review_reference='code-review')
    c['code'].update(branch='research/track-a-a3-1027', commit='a' * 40, runner_sha256=runner_sha)
    c['data']['s6_ledger'].update(ledger_sha256='1' * 64, manifest_sha256='2' * 64)
    c['data']['s6_ledger']['runtime_sha256']['value'] = reg.S6['expected_runtime_sha256']
    c['code']['config_sha256'] = runner.config_content_sha256(c)
    return c


def test_draft_config_is_refused_before_any_data(draft):
    with pytest.raises(runner.Refused, match='not REGISTERED'):
        runner.registration_gate(draft, git=GIT_OK, runner_sha256='r' * 64)


def test_registered_config_passes_and_every_missing_pin_is_refused(draft):
    good = registered(draft)
    assert runner.registration_gate(good, git=GIT_OK, runner_sha256='r' * 64)
    cases = [
        (lambda c: c.update(execute=False), {}, 'not REGISTERED'),
        (lambda c: c['registration'].update(code_review_reference=None), {}, 'code_review_reference'),
        (lambda c: c['data']['s6_ledger'].update(ledger_sha256=None), {}, 'data.s6_ledger.ledger_sha256'),
        (lambda c: c['code'].update(commit=None), {}, 'code.commit'),
        (lambda c: c['runner'].update(output_root='/tmp/elsewhere'), {}, 'config_sha256'),
    ]
    for mutate, git, message in cases:
        changed = copy.deepcopy(good)
        mutate(changed)
        with pytest.raises(runner.Refused, match=message):
            runner.registration_gate(changed, git={**GIT_OK, **git}, runner_sha256='r' * 64)
    with pytest.raises(runner.Refused, match='runner_sha256'):
        runner.registration_gate(good, git=GIT_OK, runner_sha256='x' * 64)
    for git in ({'dirty': True}, {'commit_is_ancestor': False}, {'changed_since_commit': ['experiments/x.py']},
                {'config_committed': False}, {'config_committed': None}):
        with pytest.raises(runner.Refused):
            runner.registration_gate(good, git={**GIT_OK, **git}, runner_sha256='r' * 64)
    deviating = copy.deepcopy(good)
    deviating['candidates']['B3'] = {'mandatory': True}
    with pytest.raises(runner.Refused, match='deviates'):
        runner.registration_gate(deviating, git=GIT_OK, runner_sha256='r' * 64)


def test_config_sha_ignores_only_its_own_field_and_appended_addenda(draft):
    base = runner.config_content_sha256(draft)
    later = copy.deepcopy(draft)
    later['code']['config_sha256'] = 'f' * 64
    later['registration']['addenda_records'] = [{'addendum': 'smoke'}]
    assert runner.config_content_sha256(later) == base
    later['seeds']['bootstrap'] += 1
    assert runner.config_content_sha256(later) != base


def test_main_refuses_a_draft_config_without_touching_data(tmp_path, capsys):
    destination = tmp_path / 'attempt'
    code = runner.main(['--config', str(CONFIG), '--local-config', str(tmp_path / 'missing.json'),
                        '--destination', str(destination), 'smoke'])
    assert code == 2 and not destination.exists()
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
    morning = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: datetime(2026, 10, 28, 6, 30, tzinfo=runner.KST),
                           alarm=False)
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


def make_run(tmp_path, config, source=None, now=AFTERNOON):
    destination = tmp_path / 'EXP-BC-CAL-01' / 'attempt-1'
    destination.mkdir(parents=True)
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: now, alarm=False)
    return runner.Run(config, destination=destination, source=source, guard=guard, commit='a' * 40)


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
    other = runner.Run(changed, destination=run.destination, source=None, guard=run.guard, commit='a' * 40)
    with pytest.raises(runner.Refused, match='config changed since smoke'):
        runner.execute_stage(other, 'census')
    assert runner.execute_stage(run, 'census') == 'PASS'
    assert runner.execute_stage(run, 'fit') == 'FAILED_INTEGRITY'
    final = json.loads((run.destination / 'final.json').read_text())
    assert final['label'] == 'FAILED_INTEGRITY' and final['citable'] is False
    with pytest.raises(runner.Refused):
        runner.execute_stage(run, 'calibrate')
    assert [r['status'] for r in run.attempt.records()] == ['PASS', 'PASS', 'FAILED_INTEGRITY']


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
    times = iter([AFTERNOON] * 4 + [datetime(2026, 10, 28, 6, 31, tzinfo=runner.KST)] * 10)
    destination = tmp_path / 'EXP-BC-CAL-01' / 'attempt-1'
    destination.mkdir(parents=True)
    guard = runner.Guard(reg.CAPS, reg.PROTECTION_WINDOW_KST, now=lambda: next(times), alarm=False)
    run = runner.Run(draft, destination=destination, source=None, guard=guard, commit='a' * 40)

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
    runner.append_chain(run.attempt.root / 'stages.jsonl', {'stage': stage, 'status': status})


def test_dev_labels_and_dev_metrics_are_gated(tmp_path, draft):
    run = make_run(tmp_path, draft)
    frame = make_frame(1, games={'train': 3, 'earlystop': 1, 'temperature': 1, 'dev': 2})
    original = frame.loc[frame.split.eq('dev'), 'pitch_type'].copy()
    sealed = runner.SealedLabels(frame)
    assert frame.loc[frame.split.eq('dev'), 'pitch_type'].eq(SEALED).all()
    with pytest.raises(runner.Refused, match='freeze-primary manifest pinned'):
        sealed.release(frame, run.gate, 'xf-fit')
    with pytest.raises(runner.Refused, match='freeze-primary'):
        runner.stage_s6_integrity(run, None)
    pin_stage(run, 'freeze-primary')
    with pytest.raises(runner.Refused, match='s6-integrity'):
        sealed.release(frame, run.gate, 'xf-fit')
    pin_stage(run, 's6-integrity', manifest=False)
    sealed.release(frame, run.gate, 'xf-fit')
    restored = frame.loc[frame.split.eq('dev'), 'pitch_type']
    assert restored.astype(object).where(restored.notna(), None).tolist() == \
        original.astype(object).where(original.notna(), None).tolist()
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
        assert result in reg.RUN_LABELS and result != 'ABORTED'
        seen.add(result)
    assert seen == set(reg.RUN_LABELS) - {'ABORTED'}


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


def test_end_to_end_synthetic_attempt_isolated():
    """smoke -> decide on synthetic data in a fresh interpreter (isolated_behavior_calibration_runner.py)."""
    done = run_isolated('isolated_behavior_calibration_runner.py')
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert ' passed' in done.stdout and 'failed' not in done.stdout
