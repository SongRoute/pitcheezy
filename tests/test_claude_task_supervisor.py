"""Completion must be backed by a successful CLI result, not just exit zero."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('scenario,expected', [
    ('success', 'returned_for_review'),
    ('model_error', 'failed'),
    ('missing_result', 'failed'),
    ('permission_denied', 'failed'),
    ('wrong_model', 'failed'),
    ('timeout', 'timed_out'),
])
def test_recorded_completion_requires_result(tmp_path, scenario, expected):
    fake = tmp_path / 'fake-claude'
    fake.write_text(f'#!{sys.executable}\n' + '''import json, time
scenario = SCENARIO
print(json.dumps({'type': 'assistant', 'message': {'model':
    'unexpected-model' if scenario == 'wrong_model' else 'claude-opus-5-5'}}), flush=True)
if scenario == 'timeout':
    time.sleep(20)
elif scenario != 'missing_result':
    print(json.dumps({'type': 'result', 'subtype': 'success',
        'session_id': 'fixture-session', 'is_error': scenario == 'model_error',
        'permission_denials': ['denied'] if scenario == 'permission_denied' else []}))
'''.replace('SCENARIO', repr(scenario)))
    fake.chmod(0o755)
    request = tmp_path / 'request.md'
    request.write_text('Read-only fixture request.\n')
    job = tmp_path / 'attempt-001'
    command = [sys.executable, str(ROOT / 'scripts/run_claude_task.py'),
               '--request', str(request), '--worktree', str(ROOT),
               '--job-dir', str(job), '--claude-bin', str(fake),
               '--timeout-seconds', '0.2' if scenario == 'timeout' else '5']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    state = json.loads((job / 'state.json').read_text())
    assert state['status'] == expected
    assert state['review_status'] == 'pending'
    assert completed.returncode == (0 if scenario == 'success' else 1)
    assert (job / 'request.md').read_bytes() == request.read_bytes()
    before = (job / 'state.json').read_bytes()
    duplicate = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert duplicate.returncode != 0
    assert (job / 'state.json').read_bytes() == before
