"""Git provenance of the June worker on temporary repositories: source checkout C, registration checkout D.

Metadata only: real git commands through the real ``execution_provenance``/``tracked_clean`` path.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_ml_june_calibration as runner  # noqa: E402

OWNED = (runner.SCRIPT_REL, 'experiments/pitchmdp/pitchmdp/matrix_june_calibration.py',
         'experiments/pitchmdp/pitchmdp/matrix_june_calibration_metrics.py')
REGISTERED = ('configs/ML-JUNE-CALIBRATION-EXECUTION-v1.json', 'configs/ML-JUNE-CALIBRATION-v1.json',
              'docs/contracts/ML-JUNE-CALIBRATION-v1.md')


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    return subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True, env=env).stdout.strip()


def repository(root, files):
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 't@example.invalid')
    git(root, 'config', 'user.name', 'provenance test')
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f'synthetic {rel}\n')
    git(root, 'add', '--', *files)
    git(root, 'commit', '-q', '-m', 'fixture')
    return git(root, 'rev-parse', 'HEAD')


@pytest.fixture
def checkouts(tmp_path, monkeypatch):
    c, d = tmp_path / 'source-C', tmp_path / 'registration-D'
    commit_c, commit_d = repository(c, OWNED), repository(d, REGISTERED)
    monkeypatch.setattr(runner, 'REPO', c.resolve())
    monkeypatch.setattr(runner, '__file__', str(c / runner.SCRIPT_REL))
    config = {'code_commit_c': commit_c, 'scientific_config': {'path': str(d / REGISTERED[1])},
              'contract': {'path': str(d / REGISTERED[2])}}
    return {'c': c, 'd': d, 'commit_c': commit_c, 'commit_d': commit_d, 'config': config,
            'config_path': d / REGISTERED[0]}


def provenance(env):
    return runner.execution_provenance(env['config'], env['config_path'])


def test_clean_c_and_external_d_record_both_real_heads(checkouts):
    record = provenance(checkouts)
    assert record['execution_head'] == record['registered_code_commit'] == checkouts['commit_c']
    for name, rel in zip(('execution_config', 'scientific_config', 'contract'), REGISTERED):
        assert record[name]['repository_head'] == checkouts['commit_d'] != checkouts['commit_c']
        assert record[name]['relpath'] == rel
        assert Path(record[name]['repository_root']) == checkouts['d'].resolve()


def test_wrong_registered_commit_is_refused(checkouts):
    checkouts['config']['code_commit_c'] = checkouts['commit_d']
    with pytest.raises(ValueError, match='differs from registered code_commit_c'):
        provenance(checkouts)


@pytest.mark.parametrize('staged', [False, True])
def test_modified_source_in_c_is_refused(checkouts, staged):
    source = checkouts['c'] / OWNED[1]
    source.write_text('changed\n')
    if staged:
        git(checkouts['c'], 'add', '--', OWNED[1])
    with pytest.raises(ValueError, match='tracked changes'):
        provenance(checkouts)


@pytest.mark.parametrize('staged', [False, True])
def test_modified_registered_config_in_d_is_refused(checkouts, staged):
    checkouts['config_path'].write_text('{"changed": true}\n')
    if staged:
        git(checkouts['d'], 'add', '--', REGISTERED[0])
    with pytest.raises(ValueError, match='execution config has staged or unstaged changes'):
        provenance(checkouts)


def test_untracked_or_newly_staged_config_is_refused(checkouts):
    extra = checkouts['d'] / 'configs' / 'untracked.json'
    extra.write_text('{}\n')
    checkouts['config_path'] = extra
    with pytest.raises(ValueError, match='execution config is not tracked'):
        provenance(checkouts)
    git(checkouts['d'], 'add', '--', 'configs/untracked.json')  # staged but absent from HEAD
    with pytest.raises(ValueError, match='staged or unstaged changes'):
        provenance(checkouts)


def test_config_outside_any_checkout_is_refused(checkouts, tmp_path):
    loose = tmp_path / 'loose' / 'execution.json'
    loose.parent.mkdir()
    loose.write_text('{}\n')
    checkouts['config_path'] = loose
    with pytest.raises(ValueError, match='not inside a git checkout'):
        provenance(checkouts)


def fake_git(tmp_path, monkeypatch, condition, message):
    """Real git on PATH except when ``condition`` holds: then exit 128 like a broken repository."""
    real = shutil.which('git')
    fake = tmp_path / 'bin' / 'git'
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\nif {condition}; then echo "fatal: {message}" >&2; exit 128; fi\nexec "{real}" "$@"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f'{fake.parent}{os.pathsep}{os.environ["PATH"]}')


@pytest.mark.parametrize('failing', ['status', 'rev-parse'])
def test_nonzero_git_commands_in_c_fail_closed_with_evidence(checkouts, tmp_path, monkeypatch, failing):
    fake_git(tmp_path, monkeypatch, f'[ "$1" = "{failing}" ]', f'injected {failing} failure')
    with pytest.raises(ValueError, match=f'git {failing} .*failed .*exit 128.*injected {failing} failure'):
        provenance(checkouts)


@pytest.mark.parametrize('failing', ['status', 'rev-parse'])
def test_nonzero_git_commands_in_d_alone_fail_closed(checkouts, tmp_path, monkeypatch, failing):
    d = checkouts['d'].resolve()
    fake_git(tmp_path, monkeypatch, f'[ "$1" = "{failing}" ] && [ "$(pwd -P)" = "{d}" ]', 'index file corrupt')
    with pytest.raises(ValueError, match=f'git {failing} .*failed in {d} .*exit 128.*index file corrupt'):
        provenance(checkouts)
