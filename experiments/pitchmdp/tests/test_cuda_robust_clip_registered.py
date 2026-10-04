"""Torch-free checks: registered EXP-P36-001 is the smoke-verified robust-clip DRAFT (644239f3) with identity/status/prose,
launch-day deadlines and source_configs.ple_registered changes only (P36 probe 07ab288 summary 0edb45bf; fixed L-relative
deadline rule on one date; P35 pattern test_cuda_ple_registered.py)."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_robust_clip_registry as reg  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-ROBUST-CLIP-20261005.json'
REGISTERED = REPO / 'configs' / 'EXP-P36-001.json'
P35 = REPO / 'configs' / 'EXP-P35-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration', 'source_configs'}
# queue.date is in REGISTRATION_PATHS but the DRAFT is already dated 2026-10-05 (the launch day), so it stays equal.
QUEUE_CHANGED = {'batch', 'date_note', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minutes(hhmm):
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def test_draft_probe_and_p35_pins():
    pin = read(REGISTERED)['registration']
    assert pin['draft_path'] == 'configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json'
    assert sha(DRAFT) == pin['draft_sha256'] == '644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'
    assert pin['source_commit'] == '07ab288'
    probe = pin['timing_probe']
    assert probe['summary_sha256'] == '0edb45bf299ef442f0b0679a6b0612ea55e771235a03af68eba57911f0fca837'
    assert probe['gates'] == {'complete': True, 'exit_code': 0, 'every_arm_within': True, 'family_within': True,
                              'probe_exit_zero': True}
    assert max(probe['slowest_task_seconds_projected'].values()) <= 1800
    assert probe['family_worker_seconds_projected'] <= 36000
    assert 'Codex_review_pending' in pin['review']
    assert pin['go'].strip()
    assert sha(P35) == '0a7cefda17ed961a1563d7fa762888d1eea4f97bf267f7e40f31dd81e870833a'
    assert read(REGISTERED)['source_configs']['ple_registered'] == {'path': 'configs/EXP-P35-001.json',
                                                                     'sha256': sha(P35)}


def test_only_identity_status_prose_deadlines_and_p35_pin_differ():
    draft, registered = read(DRAFT), read(REGISTERED)
    assert list(registered)[:len(draft)] == list(draft) and list(registered)[len(draft):] == ['registration']
    assert {k for k in draft.keys() | registered.keys() if draft.get(k) != registered.get(k)} == CHANGED
    assert {k for k in draft['queue'] if draft['queue'][k] != registered['queue'][k]} == QUEUE_CHANGED
    assert set(draft['queue']) == set(registered['queue'])
    for key in ('timezone', 'date', 'task_timeout_seconds', 'family_budget_seconds', 'max_attempts', 'order',
                'start_after', 'stop_rule'):
        assert registered['queue'][key] == draft['queue'][key], key
    for key in ('arms', 'seeds', 'fits', 'training', 'calibration', 'export', 'device', 'comparison', 'arm_notes',
                'data_rules', 'protocol', 'purpose', 'seed_rule', 'control_reuse', 'exposure'):
        assert registered[key] == draft[key], key
    sources = dict(registered['source_configs'])
    sources.pop('ple_registered')
    assert sources == draft['source_configs']
    assert registered['execution'] == {**draft['execution'], 'enabled': True}


def test_shared_arms_equal_the_registered_p35():
    registered, p35 = read(REGISTERED), read(P35)
    for name in ('control', 'periodic_residual_counts'):
        mine = next(a for a in registered['arms'] if a['name'] == name)
        theirs = next(a for a in p35['arms'] if a['name'] == name)
        assert json.dumps(mine, sort_keys=True) == json.dumps(theirs, sort_keys=True), name


def test_deadlines_follow_the_fixed_rule_on_one_date():
    queue = read(REGISTERED)['queue']
    assert queue['date'] == '2026-10-05'
    launch = queue['date_note'].split('(launch ', 1)[1].split(' KST', 1)[0]
    no_new, kill, agg = (minutes(queue[k]) for k in ('no_new_fit_after', 'kill_own_after', 'aggregate_by'))
    assert minutes(launch) <= 16 * 60 + 30  # pre-registration section 5
    assert no_new >= minutes(launch) + 300 and kill >= no_new + 60 and agg >= kill + 30
    assert agg <= 23 * 60  # coordinator deadline 2026-10-05 23:00 KST


def test_registered_family_passes_frozen_checks_and_allows_real_fits():
    registered = read(REGISTERED)
    assert registered['experiment_id'] == 'EXP-P36-001' and registered['status'] == 'REGISTERED'
    assert registered['seeds'] == [34, 35, 36, 37, 38] == reg.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 20 == reg.FITS
    assert rc.uses_robust_clip(registered)
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
