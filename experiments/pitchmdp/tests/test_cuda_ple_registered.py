"""Torch-free checks: registered EXP-P35-001 is the smoke-verified PLE DRAFT (f0ae4e91) with identity/status/prose
and launch-day deadline changes only (P35 timing probe 3b317f2; fixed deadline rule, review MINOR-1 ed02276)."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_ple_registry as reg  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
REGISTERED = REPO / 'configs' / 'EXP-P35-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration'}
# queue.date is in REGISTRATION_PATHS but the DRAFT is already dated 2026-10-05 (the launch day), so it stays equal.
QUEUE_CHANGED = {'batch', 'date_note', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'}


def read(path):
    return json.loads(path.read_text())


def minutes(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def test_draft_pin_matches_the_smoked_and_probed_file():
    pin = read(REGISTERED)['registration']
    assert pin['draft_path'] == 'configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == pin['draft_sha256'] \
        == 'f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'
    assert pin['source_commit'] == '8864389'
    probe = pin['timing_probe']
    assert probe['commit'] == '3b317f2'
    assert probe['summary_sha256'] == '311f0d4f5dff318320a6da0d8944e6d7b50fdc1a1eb8ee4cd7a95d754fb67ff5'
    assert probe['gates'] == {'complete': True, 'exit_code': 0, 'ple_arms_le_1800': True,
                              'every_arm_task_le_1800': True, 'family_le_36000': True}
    assert max(probe['slowest_task_seconds_projected'].values()) <= 1800
    assert probe['family_worker_seconds_projected'] <= 36000
    assert 'Codex_review_pending' in pin['review']
    assert pin['go'].strip()


def test_only_identity_status_prose_and_launch_day_deadlines_differ():
    draft, registered = read(DRAFT), read(REGISTERED)
    assert list(registered)[:len(draft)] == list(draft) and list(registered)[len(draft):] == ['registration']
    assert {k for k in draft.keys() | registered.keys() if draft.get(k) != registered.get(k)} == CHANGED
    assert {k for k in draft['queue'] if draft['queue'][k] != registered['queue'][k]} == QUEUE_CHANGED
    assert set(draft['queue']) == set(registered['queue'])
    for key in ('timezone', 'date', 'task_timeout_seconds', 'family_budget_seconds', 'max_attempts', 'order',
                'start_after', 'stop_rule'):
        assert registered['queue'][key] == draft['queue'][key], key
    for key in ('arms', 'seeds', 'fits', 'training', 'calibration', 'export', 'device', 'comparison', 'arm_notes',
                'source_configs', 'data_rules', 'protocol', 'purpose', 'seed_rule', 'control_reuse', 'exposure'):
        assert registered[key] == draft[key], key
    assert registered['execution'] == {**draft['execution'], 'enabled': True}


def test_deadlines_follow_the_fixed_rule_on_one_date():
    queue = read(REGISTERED)['queue']
    assert queue['date'] == '2026-10-05'
    launch = queue['date_note'].split('(launch ', 1)[1].split(' KST', 1)[0]
    no_new, kill, agg = (minutes(queue[k]) for k in ('no_new_fit_after', 'kill_own_after', 'aggregate_by'))
    assert no_new >= minutes(launch) + 300 and kill >= no_new + 60 and agg >= kill + 30
    assert agg <= 23 * 60  # coordinator deadline 2026-10-05 23:00 KST


def test_registered_family_passes_frozen_checks_and_allows_real_fits():
    registered = read(REGISTERED)
    assert registered['experiment_id'] == 'EXP-P35-001' and registered['status'] == 'REGISTERED'
    assert registered['seeds'] == [29, 30, 31, 32, 33] == reg.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 25 == reg.FITS
    assert rc.uses_ple(registered)
    rc.require_executable(registered, smoke=False)
    assert rc.config_digest(registered) != rc.config_digest(read(DRAFT))
    assert registered['decision'].count('No scientific N/G decision, promotion or selection') == 1
    queue = registered['queue']
    assert queue['task_timeout_seconds'] == 3600 and queue['family_budget_seconds'] == 36000 and queue['max_attempts'] == 2
    assert registered['training']['epochs'] == 60


def test_draft_still_refuses_real_fits():
    import pytest
    with pytest.raises(Exception):
        rc.require_executable(read(DRAFT), smoke=False)
