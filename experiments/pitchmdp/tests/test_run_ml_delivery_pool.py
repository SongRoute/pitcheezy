"""Synthetic wiring checks for scripts/run_ml_delivery_pool.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import run_ml_delivery_pool as runner  # noqa: E402
from pitchmdp.sequence_delivery import JointDelivery  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P14-001.yaml').read_text())
N_RULE = CONFIG['metrics']['N']


def test_registered_config_matches_runner():
    runner.config_check(CONFIG, smoke=True)
    assert 'scripts/run_ml_delivery_pool.py' in runner.SOURCES
    assert CONFIG['delivery']['max_tier'] == 1 and CONFIG['baseline']['seeds'] == list(runner.SEEDS)
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    other = copy.deepcopy(CONFIG)
    other['baseline']['cell'] = 'G2-feature'
    with pytest.raises(ValueError, match='G0'):
        runner.config_check(other, smoke=True)
    pitcher = copy.deepcopy(CONFIG)
    pitcher['delivery']['max_tier'] = 3
    with pytest.raises(ValueError, match='league tiers'):
        runner.config_check(pitcher, smoke=True)


def test_no_2026_rows():
    runner.guard_dates({'train': pd.DataFrame({'game_date': ['2023-05-15', '2025-09-28']})})
    with pytest.raises(ValueError, match='after 2025'):
        runner.guard_dates({'dev': pd.DataFrame({'game_date': ['2025-06-01', '2026-04-01']})})


def toy_delivery():
    delivery = JointDelivery()
    delivery.draws = 4
    delivery.fallback = np.zeros((4, 8), np.float32)
    row = ('FF', 'R', 'L')
    delivery.pools = {(0, row): np.full((4, 8), 10., np.float32),
                      (1, (*row, 1, 2)): np.full((4, 8), 11., np.float32),
                      (2, (7, *row)): np.full((4, 8), 12., np.float32),
                      (3, (7, *row, 1, 2)): np.full((4, 8), 13., np.float32)}
    return delivery


def test_restriction_keeps_league_draws_and_leaves_parent_untouched():
    parent = toy_delivery()
    frame = pd.DataFrame({'pitch_type': ['FF', 'FF', 'SL'], 'p_throws': ['R'] * 3, 'stand': ['L'] * 3,
                          'balls': [1, 0, 1], 'strikes': [2, 0, 2], 'pitcher': [7, 7, 7]})
    before, levels = parent.sample(frame)
    assert levels.tolist() == [3, 2, -1]
    limited = runner.restrict_delivery(parent, 1)
    points, used = limited.sample(frame)
    assert used.tolist() == [1, 0, -1]
    assert np.array_equal(points[0], parent.pools[(1, ('FF', 'R', 'L', 1, 2))])
    assert np.array_equal(points[1], parent.pools[(0, ('FF', 'R', 'L'))])
    assert len(parent.pools) == 4 and np.array_equal(parent.sample(frame)[0], before)
    assert runner.pool_report(parent) == {'0': 1, '1': 1, '2': 1, '3': 1}
    assert runner.pool_report(limited) == {'0': 1, '1': 1}
    with pytest.raises(ValueError, match='Restriction'):
        runner.restrict_delivery(parent, 3)
    with pytest.raises(ValueError, match='Restriction'):
        runner.restrict_delivery(limited, 1)


def archives(rng, n=600, games=40):
    keys = np.column_stack([np.repeat(np.arange(games), n // games), np.zeros(n, int), np.arange(n)]).astype(np.int64)
    y = rng.integers(0, 10, n)
    out = {}
    for part in ('blend', 'dev'):
        p = rng.dirichlet(np.ones(10), n)
        out.update({part: p, part + '_raw': p, part + '_keys': keys, part + '_y': y,
                    part + '_game_pk': keys[:, 0], part + '_pitcher': np.full(n, 7, np.int64),
                    part + '_delivery_level': np.where(np.arange(n) % 3 == 0, 1, 3)})
    return out


def g0_members(baseline):
    names = ('dev', 'blend', 'dev_raw', 'blend_raw')
    return [{**baseline, **{k: v for k, v in archives(np.random.default_rng(s)).items() if k in names}} for s in (1, 2, 3)]


def test_scoring_pairs_candidate_against_frozen_g0():
    baseline = archives(np.random.default_rng(0))
    g0 = g0_members(baseline)
    same = runner.score_arrays(g0, g0, baseline, N_RULE)['cpanel_dev']
    assert same['comparison']['paired']['nll']['delta'] == 0 and same['comparison']['seed_deltas'] == [0., 0., 0.]
    assert same['comparison']['N']['status'] != 'predictive_improvement'
    assert same['before_blend']['paired']['nll']['delta'] == 0
    assert same['by_g0_tier_primary']['g0_league_pool_tiers_le1']['n'] == 200
    assert same['by_g0_tier_primary']['g0_pitcher_pool_tiers_ge2']['n'] == 400
    better = [{**m, 'dev': (m['dev'] + np.eye(10)[m['dev_y']]) / 2, 'blend': (m['blend'] + np.eye(10)[m['blend_y']]) / 2}
              for m in g0]
    result = runner.score_arrays(better, g0, baseline, N_RULE)['cpanel_dev']
    assert result['comparison']['paired']['nll']['delta'] <= N_RULE['delta_nll_max']
    assert result['comparison']['N']['status'] == 'predictive_improvement'
    assert all(v['delta_nll'] < 0 for v in result['by_g0_tier_primary'].values())
    unpaired = [{**m, 'dev_y': np.roll(m['dev_y'], 1)} for m in better]
    with pytest.raises(ValueError, match='Unpaired'):
        runner.score_arrays(unpaired, g0, baseline, N_RULE)


def test_whole_mlb_secondaries_need_identical_keys_and_use_each_arm_weight():
    baseline = archives(np.random.default_rng(4))
    g0 = g0_members(baseline)
    mlb = archives(np.random.default_rng(5))
    g0_mlb = [{'dev': mlb['dev'], 'dev_keys': mlb['dev_keys'], 'dev_y': mlb['dev_y'], 'dev_game_pk': mlb['dev_game_pk'],
               'dev_delivery_level': mlb['dev_delivery_level']}] * 3
    mine_mlb = [{'mlb_dev': mlb['dev'], 'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y']}] * 3
    frequency = {'mlb_dev': np.full((len(mlb['dev_y']), 10), .1), 'mlb_dev_keys': mlb['dev_keys']}
    result = runner.score_arrays(g0, g0, baseline, N_RULE, mine_mlb, g0_mlb, frequency)
    assert result['mlb_dev']['comparison']['paired']['nll']['delta'] == 0
    blended = result['mlb_dev_blended']
    assert blended['comparison']['paired']['nll']['delta'] == 0
    assert blended['model_weight'][runner.ARM] == blended['model_weight'][runner.CELL]
    assert 0 <= blended['model_weight'][runner.ARM] <= 1
    shuffled = [{**m, 'mlb_dev_keys': m['mlb_dev_keys'][::-1]} for m in mine_mlb]
    with pytest.raises(ValueError, match='whole-MLB'):
        runner.score_arrays(g0, g0, baseline, N_RULE, shuffled, g0_mlb, frequency)
    with pytest.raises(ValueError, match='frequency'):
        runner.score_arrays(g0, g0, baseline, N_RULE, mine_mlb, g0_mlb, {**frequency, 'mlb_dev_keys': mlb['dev_keys'][::-1]})
