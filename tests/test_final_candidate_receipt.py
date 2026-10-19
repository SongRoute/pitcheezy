"""Parser, labelling and refusal checks for scripts/final_candidate_receipt.py (no step is actually run)."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/final_candidate_receipt.py'
spec = importlib.util.spec_from_file_location('final_candidate_receipt', SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module  # dataclasses look the module up
spec.loader.exec_module(module)
KNOWN = SCRIPT.parent / 'final_candidate_known_failures.json'

PYTEST_OUT = """\
........................................................................ [  5%]
=========================== short test summary info ============================
FAILED apps/observer/backend/tests/test_profile_reference.py::test_a_differing_pin_fails_closed_and_importing_leaves_no_trace
FAILED apps/observer/backend/tests/test_rev3_integration_review.py::test_pregame_only_for_revealed_people_at_every_cutoff[849843] - AssertionError: x
ERROR apps/observer/backend/tests/test_x.py::test_y - RuntimeError
5 failed, 1227 passed, 3 skipped, 1 xfailed, 1 error, 2 warnings in 245.85s (0:04:05)
"""


def test_pytest_summary_and_failure_names():
    r = module.parse_pytest(PYTEST_OUT)
    assert (r['passed'], r['failed'], r['skipped'], r['xfailed']) == (1227, 6, 3, 1)
    assert r['failures'] == [
        'apps/observer/backend/tests/test_profile_reference.py::test_a_differing_pin_fails_closed_and_importing_leaves_no_trace',
        'apps/observer/backend/tests/test_rev3_integration_review.py::test_pregame_only_for_revealed_people_at_every_cutoff[849843]',
        'apps/observer/backend/tests/test_x.py::test_y']


def test_pytest_banner_summary_and_missing_summary():
    r = module.parse_pytest('x\n======================== 5 passed, 2 warnings in 3.96s =========================\n')
    assert (r['summary_found'], r['passed'], r['failed']) == (True, 5, 0)
    assert module.parse_pytest('ImportError: boom\n')['summary_found'] is False


@pytest.mark.parametrize('text, passed, failed, names', [
    ('ok   a\nok   b\nFAIL c thing\n2 passed, 1 failed\nstatic publish smoke passed\n', 2, 1, ['c thing']),
    ('FAIL 390x844 live key info [{"bottom":790.8}]\nlog line {"a":1}\n238 ok, 1 FAIL\n', 238, 1, ['390x844 live key info']),
    ('ok   a\nFAIL 390x844 live/latest R strip above the tab bar {"strip":786.4}\nok   b\n1 FAIL\n', 2, 1,
     ['390x844 live/latest R strip above the tab bar']),
    ('ok   a\nok   b\nnote x\n\nall passed\n', 2, 0, []),
])
def test_node_formats(text, passed, failed, names):
    r = module.parse_node(text)
    assert (r['summary_found'], r['passed'], r['failed'], r['failures']) == (True, passed, failed, names)


def test_node_summary_mismatch_is_noted_and_missing_summary_flagged():
    r = module.parse_node('ok   a\nFAIL b\n3 passed, 2 failed\n')
    assert any('2 failed but 1 FAIL' in n for n in r['notes'])
    assert module.parse_node('ok   a\n')['summary_found'] is False


def _result(failures, failed=None, kind='node'):
    return {'kind': kind, 'summary_found': True, 'passed': 1, 'failed': len(failures) if failed is None else failed,
            'skipped': 0, 'failures': failures}


def test_known_failures_are_labelled_not_hidden_and_max_is_enforced():
    known = module.load_known(KNOWN)
    names = [f'390x844 live/{p} R score, count and zone above the fold, strip above the tab bar' for p in
             ('latest', 'pinned', 'latest', 'pinned')]
    c = module.classify('web-compact', _result(names), 1, known)
    assert len(c['known']) == 4 and not c['unexpected']
    c = module.classify('web-compact', _result(names + names[:1]), 1, known)
    assert len(c['known']) == 4 and len(c['unexpected']) == 1
    assert c['unexpected'][0]['reason'] == 'matches a known rule beyond its max'
    c = module.classify('web-compact', _result(['375x667 something new']), 1, known)
    assert c['unexpected'][0]['name'] == '375x667 something new'
    # a rule for another step never labels this step
    c = module.classify('web-static', _result(names[:1]), 1, known)
    assert len(c['unexpected']) == 1


def test_known_rule_that_did_not_fire_is_reported():
    c = module.classify('web-studio', _result([]), 0, module.load_known(KNOWN))
    assert len(c['known_not_seen']) == 1 and not c['unexpected']


def test_nonzero_exit_without_names_and_unnamed_failures_are_unexpected():
    assert module.classify('web-build', _result([], kind='build'), 2, [])['unexpected']
    c = module.classify('web-static', _result(['a'], failed=3), 1, [])
    assert [u['name'] for u in c['unexpected']] == ['a', '2 failure(s) without a parsable name']
    r = _result([]); r['summary_found'] = False
    assert module.classify('web-studio', r, 0, [])['unexpected']


def test_known_file_validation(tmp_path):
    bad = tmp_path / 'k.json'
    bad.write_text(json.dumps([{'step': 'x', 'match': 'y'}]))
    with pytest.raises(ValueError):
        module.load_known(bad)
    bad.write_text(json.dumps({'step': 'x'}))
    with pytest.raises(ValueError):
        module.load_known(bad)


def _repo(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'apps/observer/web').mkdir(parents=True)
    (repo / 'scripts').mkdir()
    (repo / 'a.txt').write_text('a\n')
    for cmd in (['init', '-q'], ['add', '.'], ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'i']):
        subprocess.run(['git', '-C', str(repo), *cmd], check=True)
    return repo


def test_refuses_dirty_tree_but_tolerates_node_modules_symlink(tmp_path, capsys):
    repo = _repo(tmp_path)
    target = tmp_path / 'nm'; target.mkdir()
    (repo / 'apps/observer/web/node_modules').symlink_to(target)
    assert module.dirty_entries(repo) == []
    (repo / 'a.txt').write_text('changed\n')
    assert module.main(['--repo', str(repo), '--out', str(tmp_path / 'out')]) == 2
    assert 'dirty tree' in capsys.readouterr().err
    subprocess.run(['git', '-C', str(repo), 'checkout', '-q', 'a.txt'], check=True)
    (repo / 'new.txt').write_text('x')
    assert module.dirty_entries(repo) == ['?? new.txt']


def test_real_node_modules_directory_is_not_tolerated(tmp_path):
    repo = _repo(tmp_path)
    (repo / 'apps/observer/web/node_modules').mkdir()
    (repo / 'apps/observer/web/node_modules/x.js').write_text('x')
    assert module.dirty_entries(repo)


def test_refuses_out_inside_repo_and_nonempty_out(tmp_path, capsys):
    repo = _repo(tmp_path)
    assert module.main(['--repo', str(repo), '--out', str(repo / 'receipt')]) == 2
    assert 'inside the repo' in capsys.readouterr().err
    target = tmp_path / 'nm'; target.mkdir()
    (repo / 'apps/observer/web/node_modules').symlink_to(target)
    out = tmp_path / 'out'; out.mkdir(); (out / 'old').write_text('x')
    assert module.main(['--repo', str(repo), '--out', str(out)]) == 2
    assert 'not empty' in capsys.readouterr().err


def test_inherited_candidate_env_is_dropped():
    assert set(module.DROPPED_ENV) == {'OBSERVER_E2E_CANDIDATE', 'PUBLIC_E2E_CANDIDATE'}


def test_step_commands_match_the_manual_receipt(tmp_path):
    plan = {s.id: s for s in module.steps('PY', tmp_path)}
    assert list(plan) == ['web-build', 'web-build-public', 'web-static', 'web-viewer-delay', 'web-compact',
                          'web-studio', 'backend-ops', 'public-e2e']
    assert plan['backend-ops'].argv == ['PY', '-m', 'pytest', 'apps/observer/backend/tests', 'apps/observer/ops/tests',
                                        '-q', '-p', 'no:cacheprovider', '-rfE']
    assert plan['backend-ops'].env == {'PYTHONPATH': 'apps/observer/backend'}
    assert plan['web-compact'].argv == ['node', 'tests/compact-rev3-fit.mjs', str(tmp_path / 'compact-rev3-fit')]
    assert plan['web-build-public'].argv == ['npm', 'run', 'build:public']


# ---- REVIEW-001 follow-ups


def test_viewer_delay_failed_summary_is_parsed():
    r = module.parse_node('ok   a\nFAIL switch off: x\n\n1 FAILED\n')
    assert (r['summary_found'], r['passed'], r['failed'], r['failures']) == (True, 1, 1, ['switch off: x'])


@pytest.mark.parametrize('code', [2, 3, 4, 5])
def test_pytest_abnormal_exit_is_unexpected_even_if_failures_are_known(code):
    known = [{'step': 'backend-ops', 'match': 'test_a', 'reason': 'r'}]
    r = _result(['apps/x.py::test_a'], kind='pytest')
    c = module.classify('backend-ops', r, code, known)
    assert len(c['known']) == 1 and any(f'exit code {code}' in u['name'] for u in c['unexpected'])
    assert not module.classify('backend-ops', r, 1, known)['unexpected']


@pytest.mark.parametrize('rule, message', [
    ({'step': 'web-compat', 'match': 'x', 'reason': 'r'}, 'unknown step'),
    ({'step': 'web-compact', 'match': 'x', 'reason': 'r', 'max': 0}, 'max'),
    ({'step': 'web-compact', 'match': 'x', 'reason': 'r', 'max': '4'}, 'max'),
    ({'step': 'web-compact', 'match': 'x', 'reason': 'r', 'max': True}, 'max'),
    ({'step': 'web-compact', 'match': 'x', 'reason': 'r', 'flaky': 'yes'}, 'flaky'),
    ({'step': 'web-compact', 'match': 'x', 'reason': 'r', 'maxx': 1}, 'unknown keys'),
    ({'step': 'web-compact', 'match': 'x', 'reason': ' '}, 'non-empty'),
])
def test_known_rule_format_is_validated(tmp_path, rule, message):
    path = tmp_path / 'k.json'
    path.write_text(json.dumps([rule]))
    with pytest.raises(ValueError, match=message):
        module.load_known(path)


def test_shipped_known_file_has_no_stale_tst01_rules():
    rules = module.load_known(KNOWN)
    assert {r['step'] for r in rules} == {'web-compact', 'web-studio', 'backend-ops'}
    assert [r for r in rules if r['step'] == 'backend-ops'][0]['flaky'] is True
    assert not any('TST-01' in r['reason'] for r in rules)


def _git_repo_with_gitignore(tmp_path):
    repo = _repo(tmp_path)
    (repo / '.gitignore').write_text('node_modules/\n')
    subprocess.run(['git', '-C', str(repo), 'add', '.gitignore'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'g'],
                   check=True)
    return repo


def test_ignored_real_node_modules_directory_is_refused(tmp_path, capsys):
    repo = _git_repo_with_gitignore(tmp_path)
    (repo / 'apps/observer/web/node_modules').mkdir()
    assert module.dirty_entries(repo) == []          # gitignored, so the status check alone would miss it
    assert module.main(['--repo', str(repo), '--out', str(tmp_path / 'out')]) == 2
    assert 'must be a symlink' in capsys.readouterr().err


def test_partial_run_is_marked_and_link_target_recorded(tmp_path, monkeypatch):
    repo = _git_repo_with_gitignore(tmp_path)
    target = tmp_path / 'nm'; target.mkdir()
    (repo / 'apps/observer/web/node_modules').symlink_to(target)
    fake = tmp_path / 'bin'; fake.mkdir()
    (fake / 'npm').write_text('#!/bin/sh\necho built\n'); (fake / 'npm').chmod(0o755)
    monkeypatch.setenv('PATH', f'{fake}:{os.environ["PATH"]}')
    out = tmp_path / 'out'
    assert module.main(['--repo', str(repo), '--out', str(out), '--steps', 'web-build', '--python', 'python3']) == 0
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['partial'] is True and receipt['steps_requested'] == ['web-build']
    assert receipt['verdict'] == 'PARTIAL_PASS'
    assert receipt['node_modules_link'] == str(target.resolve())
    assert 'PARTIAL' in (out / 'receipt.md').read_text()


def _fake_run(outcome):
    calls = []

    def run_step(step, repo, log_dir, base_env):
        calls.append(step)
        return {'command': step.argv, 'start': 's', 'end': 'e', 'log': f'{step.id}.log', **outcome}
    return run_step, calls


def test_flaky_rerun_runs_only_the_flaky_test_and_failing_rerun_is_unexpected(monkeypatch, tmp_path):
    step = module.steps('PY', tmp_path)[6]
    name = 'apps/observer/backend/tests/test_publisher_background.py::test_retry_cursor_passes_large_head_failure_prefix'
    result = {'unexpected': [], 'known': [{'name': name, 'reason': 'r', 'flaky': True}]}
    run_step, calls = _fake_run({'exit_code': 0, 'passed': 1, 'failed': 0, 'failures': []})
    monkeypatch.setattr(module, 'run_step', run_step)
    rerun = module.flaky_rerun(step, result, tmp_path, tmp_path, {})
    assert calls[0].argv == ['PY', '-m', 'pytest', name, '-q', '-p', 'no:cacheprovider', '-rfE']
    assert calls[0].env == {'PYTHONPATH': 'apps/observer/backend'}
    assert rerun['exit_code'] == 0 and not result['unexpected']
    run_step, _ = _fake_run({'exit_code': 1, 'passed': 0, 'failed': 1, 'failures': [name]})
    monkeypatch.setattr(module, 'run_step', run_step)
    module.flaky_rerun(step, result, tmp_path, tmp_path, {})
    assert 'flaky rerun alone did not pass' in result['unexpected'][0]['name']


def test_no_flaky_rerun_when_other_failures_exist(tmp_path):
    step = module.steps('PY', tmp_path)[6]
    mixed = {'unexpected': [], 'known': [{'name': 'a', 'reason': 'r', 'flaky': True},
                                         {'name': 'b', 'reason': 'r', 'flaky': False}]}
    assert module.flaky_rerun(step, mixed, tmp_path, tmp_path, {}) is None
    other = {'unexpected': [{'name': 'c', 'reason': None}], 'known': [{'name': 'a', 'reason': 'r', 'flaky': True}]}
    assert module.flaky_rerun(step, other, tmp_path, tmp_path, {}) is None
    web = module.steps('PY', tmp_path)[4]
    assert module.flaky_rerun(web, {'unexpected': [], 'known': [{'name': 'a', 'reason': 'r', 'flaky': True}]},
                              tmp_path, tmp_path, {}) is None
