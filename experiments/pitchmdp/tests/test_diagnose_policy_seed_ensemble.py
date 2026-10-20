"""Synthetic wiring checks for scripts/diagnose_policy_seed_ensemble.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import diagnose_policy_seed_ensemble as runner  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-STAB-01.yaml').read_text())
P18 = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P18-001.yaml').read_text())


def test_registered_config():
    runner.config_check(CONFIG)
    assert CONFIG['settings'] == P18['settings'] and CONFIG['policy_registration'] == P18['policy_registration']
    assert all((PROJECT / rel).is_file() for rel in runner.SOURCES)
    overlap = copy.deepcopy(CONFIG)
    overlap['seed_sets']['B'][0] = 0
    with pytest.raises(ValueError, match='disjoint'):
        runner.config_check(overlap)
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)


def _law_inputs(rng, noise, n_seeds=10):
    mask = np.array([True, True, True, False])
    ref = np.array([.4, .35, .25, 0.])
    true_q = np.array([.50, .505, .49, np.nan])
    return {s: true_q + np.where(mask, rng.normal(0, noise, 4), np.nan) for s in range(n_seeds)}, ref, mask


def test_ensemble_averaging_reduces_disagreement_and_verdict():
    rng = np.random.default_rng(0)
    sets = CONFIG['seed_sets']
    columns, games = {}, []
    for i in range(3000):
        q, ref, mask = _law_inputs(rng, .01)
        for k, v in runner.compare(q, ref, mask, .1, sets).items():
            columns.setdefault(k, []).append(v)
        games.append(i // 30)
    s = runner.summarize(columns, games, sets, CONFIG['verdict_rule'], draws=300, seed=1)
    by = s['by_K']
    assert by['5']['top1_agreement'] > by['1']['top1_agreement']
    assert by['5']['mean_tv'] < by['1']['mean_tv']
    assert s['verdict']['averaging_improves_agreement']
    assert s['verdict']['status'].startswith(('STABLE_AT_K', 'NOT_STABLE_UP_TO_K'))
    # identical seeds: perfect agreement, stable at K=1
    columns = {}
    for i in range(500):
        q, ref, mask = _law_inputs(rng, 0.)
        for k, v in runner.compare(q, ref, mask, .1, sets).items():
            columns.setdefault(k, []).append(v)
    s = runner.summarize(columns, [i // 10 for i in range(500)], sets, CONFIG['verdict_rule'], draws=200, seed=1)
    assert s['by_K']['1']['top1_agreement'] == 1 and s['verdict']['status'] == 'STABLE_AT_K1'
    assert s['by_K']['3']['mean_tv'] == 0
