"""Synthetic wiring checks for scripts/run_ml_direct_head_confirm.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(PROJECT / 'tests')]
import run_ml_direct_head_confirm as runner  # noqa: E402
from test_run_ml_direct_head import archives, arm  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P16-001.yaml').read_text())
FROZEN = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P15-001.yaml').read_text())


def test_registered_config_matches_runner_and_frozen_settings():
    runner.config_check(CONFIG, smoke=True)
    assert runner.NAME in runner.base.SOURCES and runner.NAME not in runner.BASE_SOURCES
    assert all(CONFIG[key] == FROZEN[key] for key in runner.SAME_AS_FROZEN)
    assert CONFIG['metrics']['N'] == {**FROZEN['metrics']['N'], 'required_negative_seeds': 4}
    assert CONFIG['metrics']['bootstrap'] == FROZEN['metrics']['bootstrap']
    for key, value, match in (('seeds', [0, 1, 2], 'seeds 0-4'), ('new_fit_seeds', [0, 3, 4], 'seeds 0-4'),
                              ('confirmed_arm', runner.DIRECT, 'fixed equal-weight mix'),
                              ('frozen_direct_run', None, 'Registered input missing')):
        changed = copy.deepcopy(CONFIG)
        changed[key] = value
        with pytest.raises(ValueError, match=match):
            runner.config_check(changed, smoke=True)
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    screen = copy.deepcopy(CONFIG)
    screen['metrics']['N']['required_negative_seeds'] = 2
    with pytest.raises(ValueError, match='four of five'):
        runner.config_check(screen, smoke=True)


def test_frozen_run_guard_rejects_changed_settings_or_sources(tmp_path, monkeypatch):
    from run_ml_matrix import artifact_hashes
    from run_ml_benchmark import dump
    from pitchmdp.data import hash_file
    dump(tmp_path / 'registered_config.json', FROZEN)
    sources = {rel: hash_file(PROJECT / rel) for rel in runner.BASE_SOURCES}
    dump(tmp_path / 'preparation.json', {'identity': {'source_hashes': sources},
                                         'artifact_hashes': artifact_hashes(tmp_path, ['registered_config.json'])})
    config = copy.deepcopy(CONFIG)
    config['frozen_direct_run'] = {'path': str(tmp_path), 'preparation_sha256': hash_file(tmp_path / 'preparation.json')}
    assert runner.verify_frozen_direct(config) == tmp_path
    changed = copy.deepcopy(config)
    changed['mix'] = {**config['mix'], 'rule': 'another rule'}
    with pytest.raises(ValueError, match='must not change the frozen setting: mix'):
        runner.verify_frozen_direct(changed)
    wrong = copy.deepcopy(config)
    wrong['frozen_direct_run']['preparation_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='preparation changed'):
        runner.verify_frozen_direct(wrong)
    monkeypatch.setattr(runner, 'hash_file', lambda path: 'f' * 64 if Path(path).name == 'run_ml_direct_head.py' else hash_file(path))
    config['frozen_direct_run']['preparation_sha256'] = hash_file(tmp_path / 'preparation.json')
    with pytest.raises(ValueError, match='implementation changed'):
        runner.verify_frozen_direct(config)


def whole(seed=5):
    mlb = archives(np.random.default_rng(seed))
    g0_mlb = [{'dev': mlb['dev'], 'dev_keys': mlb['dev_keys'], 'dev_y': mlb['dev_y'], 'dev_game_pk': mlb['dev_game_pk'],
               'dev_delivery_level': mlb['dev_delivery_level']}] * 5
    frequency = {'mlb_dev': np.full((len(mlb['dev_y']), 10), .1), 'mlb_dev_keys': mlb['dev_keys']}
    return mlb, g0_mlb, frequency


def test_five_seed_scoring_and_confirmation_rule():
    baseline = archives(np.random.default_rng(0))
    g0 = arm(baseline, (1, 2, 3, 4, 5))
    mlb, g0_mlb, frequency = whole()
    same_mlb = [{'mlb_dev': mlb['dev'], 'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y']}] * 5
    same = runner.score_arrays(g0, g0, baseline, CONFIG, same_mlb, g0_mlb, frequency)
    assert same['confirmation']['status'] == 'not_confirmed' and same['confirmation']['arm'] == runner.MIX
    assert all(len(row['seed_deltas']) == 5 for row in same['cpanel_dev']['comparison'].values())
    sharper = lambda p, y: (p + np.eye(10)[y]) / 2
    better = [{**m, 'dev': sharper(m['dev'], m['dev_y']), 'blend': sharper(m['blend'], m['blend_y'])} for m in g0]
    better_mlb = [{**m, 'mlb_dev': sharper(mlb['dev'], mlb['dev_y'])} for m in same_mlb]
    result = runner.score_arrays(better, g0, baseline, CONFIG, better_mlb, g0_mlb, frequency)
    assert result['confirmation'] == {**result['confirmation'], 'status': 'confirmed',
                                      'co_primary': {'cpanel_dev_after_blend': runner.PASS, 'whole_mlb_dev_blended': runner.PASS}}
    assert result['cpanel_dev']['comparison'][runner.MIX]['N']['criteria']['seed_direction_stable']
    # Passing on the panel alone is not a confirmation: both co-primary comparisons are required.
    half = runner.score_arrays(better, g0, baseline, CONFIG, same_mlb, g0_mlb, frequency)
    assert half['confirmation']['co_primary']['cpanel_dev_after_blend'] == runner.PASS
    assert half['confirmation']['status'] == 'not_confirmed'
    with pytest.raises(ValueError, match='all five seeds'):
        runner.score_arrays(better[:3], g0[:3], baseline, CONFIG, better_mlb[:3], g0_mlb[:3], frequency)
    shuffled = [{**m, 'mlb_dev_keys': m['mlb_dev_keys'][::-1]} for m in better_mlb]
    with pytest.raises(ValueError, match='whole-MLB'):
        runner.score_arrays(better, g0, baseline, CONFIG, shuffled, g0_mlb, frequency)
