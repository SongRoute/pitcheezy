"""Torch-free checks: registered EXP-P33-001 is the smoke-verified EMA DRAFT with identity/status/prose changes only (review R4/R5)."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_ema_registry as reg  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-EMA-20261004.json'
REGISTERED = REPO / 'configs' / 'EXP-P33-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration'}


def read(path):
    return json.loads(path.read_text())


def test_draft_pin_matches_the_smoked_file():
    pin = read(REGISTERED)['registration']
    assert pin['draft_path'] == 'configs/DRAFT-CUDA-EMA-20261004.json'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == pin['draft_sha256'] \
        == 'a6cf64eb2698a47c3a9302bd07abdecbdcf33e6057b14654a1bd947058ce8de7'
    assert pin['source_commit'] == 'b4b9243'
    assert 'Codex_review_pending' in pin['review']


def test_only_identity_status_and_prose_differ():
    draft, registered = read(DRAFT), read(REGISTERED)
    assert {k for k in draft.keys() | registered.keys() if draft.get(k) != registered.get(k)} == CHANGED
    assert {k for k in draft['queue'] if draft['queue'][k] != registered['queue'][k]} == {'batch'}
    assert set(draft['queue']) == set(registered['queue'])
    for key in ('arms', 'seeds', 'fits', 'training', 'calibration', 'export', 'device', 'comparison', 'arm_notes',
                'source_configs', 'data_rules', 'protocol', 'purpose', 'seed_rule', 'control_reuse'):
        assert registered[key] == draft[key], key


def test_registered_family_passes_frozen_checks_and_allows_real_fits():
    registered = read(REGISTERED)
    assert registered['experiment_id'] == 'EXP-P33-001'
    assert registered['seeds'] == [19, 20, 21, 22, 23] == reg.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 20 == reg.FITS
    assert rc.uses_ema(registered)
    rc.require_executable(registered, smoke=False)
    assert rc.config_digest(registered) != rc.config_digest(read(DRAFT))
    assert registered['decision'].count('No scientific N/G decision, promotion or selection') == 1
    assert registered['queue']['no_new_fit_after'] == '19:00' and registered['queue']['date'] == '2026-10-04'
    assert registered['queue']['family_budget_seconds'] == 36000 and registered['queue']['max_attempts'] == 2


def test_draft_still_refuses_real_fits():
    import pytest
    with pytest.raises(Exception):
        rc.require_executable(read(DRAFT), smoke=False)
