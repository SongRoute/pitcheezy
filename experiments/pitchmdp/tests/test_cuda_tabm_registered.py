"""Torch-free checks: registered EXP-P31-001 is the smoke-verified TabM-mini DRAFT with identity/prose changes only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-TABM-MINI-20261004.json'
REGISTERED = REPO / 'configs' / 'EXP-P31-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration'}


def read(path):
    return json.loads(path.read_text())


def test_draft_pin_matches_the_smoked_file():
    reg = read(REGISTERED)['registration']
    assert reg['draft_path'] == 'configs/DRAFT-CUDA-TABM-MINI-20261004.json'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == reg['draft_sha256'] \
        == '334528a5904bd78f9f4a70955114d3434f3085b99a079a699daff07d252eae70'
    assert reg['source_commit'] == '2d98f95'
    assert 'Codex_review_pending' in reg['review']


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
    assert registered['experiment_id'] == 'EXP-P31-001'
    assert registered['seeds'] == [13, 14, 15] == tr.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 15 == tr.FITS
    rc.require_executable(registered, smoke=False)
    assert rc.config_digest(registered) != rc.config_digest(read(DRAFT))
    assert registered['decision'].count('No scientific N/G decision, promotion or selection') == 1
