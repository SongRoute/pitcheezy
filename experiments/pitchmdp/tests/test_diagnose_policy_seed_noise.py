"""Synthetic wiring checks for scripts/diagnose_policy_seed_noise.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import diagnose_policy_seed_noise as runner  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P18-001.yaml').read_text())
SWAP = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P17-001.yaml').read_text())


def test_registered_config_keeps_the_swap_settings_and_changes_only_the_seed():
    runner.config_check(CONFIG)
    assert all(CONFIG[key] == SWAP[key] for key in runner.SAME_AS_SWAP)
    assert CONFIG['noise_seed'] == 1 != CONFIG['settings']['seed']
    assert CONFIG['verdict_rule']['excess_mean_tv_at_least'] == SWAP['stop_rule']['mean_tv_below']
    assert all((PROJECT / rel).is_file() for rel in runner.SOURCES)
    same_seed = copy.deepcopy(CONFIG)
    same_seed['noise_seed'] = CONFIG['settings']['seed']
    with pytest.raises(ValueError, match='must differ'):
        runner.config_check(same_seed)
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    no_rule = copy.deepcopy(CONFIG)
    no_rule['verdict_rule'] = {}
    with pytest.raises(ValueError, match='verdict rule'):
        runner.config_check(no_rule)


def test_verdict_is_the_paired_excess_over_the_seed_distance():
    rng = np.random.default_rng(0)
    n, games = 4000, np.repeat(np.arange(100), 40)
    seed_tv = rng.uniform(0, .06, n)
    rule = CONFIG['verdict_rule']
    kwargs = dict(games=games, rule=rule, draws=400, seed=1)
    same = rng.random(n) < .93
    noise_only = runner.verdict(seed_tv + rng.normal(0, .002, n), seed_tv, same, same, **kwargs)
    assert noise_only['verdict']['status'] == runner.WITHIN and abs(noise_only['excess_mean_tv']) < .001
    assert noise_only['seed_noise']['mean_tv'] == pytest.approx(seed_tv.mean())
    clear = runner.verdict(seed_tv + .02, seed_tv, same & (rng.random(n) < .9), same, **kwargs)
    assert clear['verdict']['status'] == runner.BEYOND and clear['excess_mean_tv'] == pytest.approx(.02)
    assert clear['excess_mean_tv_ci95'][0] > 0 and clear['excess_top1_disagreement'] > 0
    assert clear['swap_to_seed_tv_ratio'] == pytest.approx((seed_tv + .02).mean() / seed_tv.mean())
    # A real but small excess (below the registered 0.01) is still WITHIN.
    small = runner.verdict(seed_tv + .005, seed_tv, same, same, **kwargs)
    assert small['excess_mean_tv_ci95'][0] > 0 and small['verdict']['status'] == runner.WITHIN
    # A large excess carried by one game cannot pass: the game bootstrap lower bound is not above zero.
    lumpy = seed_tv.copy()
    lumpy[games == 0] += 1.2
    one_game = runner.verdict(lumpy, seed_tv, same, same, **kwargs)
    assert one_game['excess_mean_tv'] >= .01 and one_game['verdict']['status'] == runner.WITHIN
