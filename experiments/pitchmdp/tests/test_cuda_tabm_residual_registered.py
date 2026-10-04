"""Torch-free checks: registered EXP-P34-001 is the smoke-verified TabM-residual DRAFT (5e3ce3ad) with identity/status/prose
and launch-day deadline changes only (Amendment 2 timing gate)."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_tabm_residual_registry as reg  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-TABM-RESIDUAL-20261004.json'
REGISTERED = REPO / 'configs' / 'EXP-P34-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration'}
QUEUE_CHANGED = {'batch', 'date_note', 'no_new_fit_after', 'kill_own_after', 'aggregate_by'}


def read(path):
    return json.loads(path.read_text())


def test_draft_pin_matches_the_smoked_and_probed_file():
    pin = read(REGISTERED)['registration']
    assert pin['draft_path'] == 'configs/DRAFT-CUDA-TABM-RESIDUAL-20261004.json'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == pin['draft_sha256'] \
        == '5e3ce3adc7ecc30f94425d9dc8a9234441cee2160644fe1ed4d9d54788b76d34'
    assert pin['source_commit'] == 'b9f28dc'
    assert pin['timing_probe']['commit'] == '9de203a'
    gates = pin['timing_probe']['gates']
    assert gates['complete'] is True and gates['exit_code'] == 0
    assert gates['tabm_residual_task_le_1800'] is True and gates['every_arm_task_le_1800'] is True
    assert gates['family_le_36000'] is True
    assert 'Codex_review_pending' in pin['review']


def test_only_identity_status_prose_and_launch_day_deadlines_differ():
    draft, registered = read(DRAFT), read(REGISTERED)
    assert {k for k in draft.keys() | registered.keys() if draft.get(k) != registered.get(k)} == CHANGED
    assert {k for k in draft['queue'] if draft['queue'][k] != registered['queue'][k]} == QUEUE_CHANGED
    assert set(draft['queue']) == set(registered['queue'])
    for key in ('date', 'task_timeout_seconds', 'family_budget_seconds', 'max_attempts', 'order', 'start_after', 'stop_rule'):
        assert registered['queue'][key] == draft['queue'][key], key
    for key in ('arms', 'seeds', 'fits', 'training', 'calibration', 'export', 'device', 'comparison', 'arm_notes',
                'source_configs', 'data_rules', 'protocol', 'purpose', 'seed_rule', 'control_reuse', 'exposure'):
        assert registered[key] == draft[key], key


def test_registered_family_passes_frozen_checks_and_allows_real_fits():
    registered = read(REGISTERED)
    assert registered['experiment_id'] == 'EXP-P34-001'
    assert registered['seeds'] == [24, 25, 26, 27, 28] == reg.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 25 == reg.FITS
    assert rc.uses_tabm_residual(registered)
    rc.require_executable(registered, smoke=False)
    assert rc.config_digest(registered) != rc.config_digest(read(DRAFT))
    assert registered['decision'].count('No scientific N/G decision, promotion or selection') == 1
    queue = registered['queue']
    assert queue['date'] == '2026-10-04' and queue['no_new_fit_after'] < queue['kill_own_after'] < queue['aggregate_by']
    assert queue['task_timeout_seconds'] == 3600 and queue['family_budget_seconds'] == 36000 and queue['max_attempts'] == 2
    assert registered['training']['epochs'] == 60


def test_draft_still_refuses_real_fits():
    import pytest
    with pytest.raises(Exception):
        rc.require_executable(read(DRAFT), smoke=False)
