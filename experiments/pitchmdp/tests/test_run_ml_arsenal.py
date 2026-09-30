"""Synthetic wiring checks for scripts/run_ml_arsenal.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import run_ml_arsenal as runner  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P13-001.yaml').read_text())
N_RULE = CONFIG['metrics']['N']


def test_registered_config_matches_runner():
    runner.config_check(CONFIG, smoke=True)
    assert CONFIG['arsenal']['module'].endswith('matrix_arsenal.py')
    assert 'scripts/run_ml_arsenal.py' in runner.SOURCES and 'pitchmdp/matrix_arsenal.py' in runner.SOURCES
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    other = copy.deepcopy(CONFIG)
    other['baseline']['cell'] = 'G2-feature'
    with pytest.raises(ValueError, match='G0'):
        runner.config_check(other, smoke=True)


def test_no_2026_rows():
    ok = {'train': pd.DataFrame({'game_date': ['2023-05-15', '2025-09-28']})}
    runner.guard_dates(ok)
    with pytest.raises(ValueError, match='after 2025'):
        runner.guard_dates({'dev': pd.DataFrame({'game_date': ['2025-06-01', '2026-04-01']})})
    with pytest.raises(ValueError, match='after 2025'):
        runner.guard_dates({'dev': pd.DataFrame({'game_date': [None]})})


def archives(rng, n=600, games=40, shift=0.):
    keys = np.column_stack([np.repeat(np.arange(games), n // games), np.zeros(n, int), np.arange(n)]).astype(np.int64)
    y = rng.integers(0, 10, n)
    out = {}
    for part in ('blend', 'dev'):
        p = rng.dirichlet(np.ones(10), n)
        p[np.arange(n), y] += shift
        p /= p.sum(1, keepdims=True)
        out.update({part: p, part + '_raw': p, part + '_keys': keys, part + '_y': y,
                    part + '_game_pk': keys[:, 0], part + '_pitcher': np.full(n, 7, np.int64)})
    return out


def test_scoring_pairs_candidate_against_frozen_g0():
    rng = np.random.default_rng(0)
    baseline = archives(rng)
    g0 = [{**baseline, **{k: v for k, v in archives(np.random.default_rng(s)).items() if k in ('dev', 'blend', 'dev_raw', 'blend_raw')}}
          for s in (1, 2, 3)]
    same = runner.score_arrays(g0, g0, baseline, N_RULE)['cpanel_dev']['comparison']
    assert same['paired']['nll']['delta'] == 0 and same['seed_deltas'] == [0., 0., 0.]
    assert same['N']['status'] != 'predictive_improvement'
    better = [{**m, 'dev': (m['dev'] + np.eye(10)[m['dev_y']]) / 2, 'blend': (m['blend'] + np.eye(10)[m['blend_y']]) / 2}
              for m in g0]
    result = runner.score_arrays(better, g0, baseline, N_RULE)['cpanel_dev']['comparison']
    assert result['paired']['nll']['delta'] <= N_RULE['delta_nll_max'] and all(d < 0 for d in result['seed_deltas'])
    assert result['N']['status'] == 'predictive_improvement'
    assert 'top_label_ece10' in result['candidate']
    unpaired = [{**m, 'dev_y': np.roll(m['dev_y'], 1)} for m in better]
    with pytest.raises(ValueError, match='Unpaired'):
        runner.score_arrays(unpaired, g0, baseline, N_RULE)


def test_whole_mlb_pairing_requires_identical_keys():
    rng = np.random.default_rng(4)
    baseline = archives(rng)
    g0 = [baseline] * 3
    mlb = archives(np.random.default_rng(5))
    g0_mlb = [{'dev': mlb['dev'], 'dev_keys': mlb['dev_keys'], 'dev_y': mlb['dev_y'], 'dev_game_pk': mlb['dev_game_pk']}] * 3
    mine_mlb = [{'mlb_dev': mlb['dev'], 'mlb_dev_keys': mlb['dev_keys'], 'mlb_dev_y': mlb['dev_y']}] * 3
    result = runner.score_arrays(g0, g0, baseline, N_RULE, mine_mlb, g0_mlb)
    assert result['mlb_dev']['comparison']['paired']['nll']['delta'] == 0
    shuffled = [{**m, 'mlb_dev_keys': m['mlb_dev_keys'][::-1]} for m in mine_mlb]
    with pytest.raises(ValueError, match='whole-MLB'):
        runner.score_arrays(g0, g0, baseline, N_RULE, shuffled, g0_mlb)
