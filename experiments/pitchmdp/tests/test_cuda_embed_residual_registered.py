"""Torch-free checks: registered EXP-P32-001 is the smoke-verified embed-residual DRAFT with identity/status/prose changes only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_cuda_direct as rc  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402

DRAFT = REPO / 'configs' / 'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json'
REGISTERED = REPO / 'configs' / 'EXP-P32-001.json'
CHANGED = {'experiment_id', 'status', 'execution', 'decision', 'queue', 'registration'}


def read(path):
    return json.loads(path.read_text())


def test_draft_pin_matches_the_smoked_file():
    reg = read(REGISTERED)['registration']
    assert reg['draft_path'] == 'configs/DRAFT-CUDA-EMBED-RESIDUAL-20261004.json'
    assert hashlib.sha256(DRAFT.read_bytes()).hexdigest() == reg['draft_sha256'] \
        == 'bb0ce6209db7b5b6bbc9243cb57e7ba4913dc1cf567756c67feacf2f508ac14e'
    assert reg['source_commit'] == 'ed5ae1a'
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
    assert registered['experiment_id'] == 'EXP-P32-001'
    assert registered['seeds'] == [16, 17, 18] == er.SEEDS
    assert rc.config_check(copy.deepcopy(registered))['fits'] == 18 == er.FITS
    rc.require_executable(registered, smoke=False)
    assert rc.config_digest(registered) != rc.config_digest(read(DRAFT))
    assert registered['decision'].count('No scientific N/G decision, promotion or selection') == 1
    assert registered['queue']['no_new_fit_after'] == '19:00' and registered['queue']['date'] == '2026-10-04'
