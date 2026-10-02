"""Synthetic wiring checks for scripts/diagnose_policy_predictor_swap.py (no real data)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import diagnose_policy_predictor_swap as runner  # noqa: E402

CONFIG = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P17-001.yaml').read_text())


def test_registered_config_matches_runner_and_frozen_inputs():
    runner.config_check(CONFIG)
    bundle = PROJECT.parents[1] / CONFIG['mix_bundle']['path']
    assert runner.hash_file(bundle) == CONFIG['mix_bundle']['sha256']
    policy = PROJECT.parents[1] / CONFIG['policy_registration']['config']
    assert runner.hash_file(policy) == CONFIG['policy_registration']['config_sha256']
    reg = runner.rpv.load_registration(policy, [PROJECT.parents[1] / a for a in CONFIG['policy_registration']['addenda']])
    assert {'bc', 'support', 'style_blend', 'bind_identity', 'tau_freeze'} <= set(reg['inputs'])
    assert reg['expected_identity_sha256'] == CONFIG['sealed_g0_run']['final_identity_sha256']
    assert all((PROJECT / rel).is_file() for rel in runner.SOURCES)
    off = copy.deepcopy(CONFIG)
    off['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(off)
    runner.config_check(off, smoke=True)
    weight = copy.deepcopy(CONFIG)
    weight['mix']['direct_weight'] = .7
    with pytest.raises(ValueError, match='one half'):
        runner.config_check(weight)
    rule = copy.deepcopy(CONFIG)
    del rule['stop_rule']['top1_agreement_at_least']
    with pytest.raises(ValueError, match='stop rule'):
        runner.config_check(rule)


class Member:
    """Logits depend on the current token's physics unless the caller zeroed them."""
    def __init__(self, shift): self.shift = shift

    def logits(self, arrays):
        tokens, _, _ = arrays
        base = np.tile(np.arange(10.) / 10 + self.shift, (len(tokens), 1))
        base[:, 0] += tokens[:, -1, :8].sum(1)
        return base


class Inputs:
    def arrays(self, states, actions, physical):
        tokens = np.zeros((len(states), 6, 30), dtype=np.float32)
        tokens[:, -1, :8] = physical
        tokens[:, -1, 9] = 1  # candidate type channel must survive the zeroing
        return tokens, np.ones((len(states), 6), dtype=bool), np.zeros((len(states), 4), dtype=np.float32)


class G0:
    def __init__(self):
        self.inputs, self.models, self.temperatures = Inputs(), [Member(s / 10) for s in range(5)], [1., 1.1, .9, 1., 1.2]


def test_mix_predictor_is_half_direct_half_conditional_g0():
    g0 = G0()
    mix = runner.MixPredictor(g0, [Member(1 + s / 10) for s in range(5)], [1.] * 5, .5)
    states, actions = ['s'] * 3, ['FF'] * 3
    physical = np.arange(24, dtype=np.float32).reshape(3, 8) / 10
    direct, conditional = mix.components(states, actions, physical)
    assert np.allclose(mix(states, actions, physical), .5 * direct + .5 * conditional)
    assert np.allclose(mix(states, actions, physical).sum(1), 1.)
    other = mix.components(states, actions, physical + 5.)
    assert np.array_equal(other[0], direct)             # the direct head never sees the sampled delivery
    assert not np.allclose(other[1], conditional)       # the G0 half does
    assert np.array_equal(mix.components(states, actions, np.zeros((3, 8), np.float32))[0], direct)
    # Linearity: averaging the conditional mix over draws equals mixing the averaged components.
    draws = np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32)
    d, g = mix.components(['s'] * 40, ['FF'] * 40, draws)
    assert np.allclose(mix(['s'] * 40, ['FF'] * 40, draws).mean(0), .5 * d.mean(0) + .5 * g.mean(0))
    with pytest.raises(ValueError, match='Five members'):
        runner.MixPredictor(g0, [Member(0)] * 3, [1.] * 3, .5)


def test_law_metrics_and_stop_rule():
    mask = np.array([True, True, False, True])
    reference = np.array([.5, .3, 0., .2])
    q = np.array([.50, .52, -np.inf, .49])
    same = runner.decision_metrics(runner.law(q, reference, mask, .1), runner.law(q, reference, mask, .1), reference, mask)
    assert same['tv'] == 0 and same['top1_same'] and same['kl_mix_reference'] == pytest.approx(same['kl_g0_reference'])
    moved = runner.law(np.array([.40, .60, -np.inf, .49]), reference, mask, .1)
    base = runner.law(q, reference, mask, .1)
    far = runner.decision_metrics(moved, base, reference, mask)
    assert 0 < far['tv'] <= 1 and far['tv'] == pytest.approx(.5 * np.abs(moved - base).sum()) and moved[2] == 0
    rule = CONFIG['stop_rule']
    rows = [same] * 990 + [far] * 10
    games, strikes = np.repeat(np.arange(50), 20), np.tile([0, 1, 2, 2], 250)
    summary = runner.summarize(rows, games, strikes, rule, draws=200, seed=1)
    assert summary['decisions'] == 1000 and summary['games'] == 50
    assert summary['mean_tv'] == pytest.approx(far['tv'] / 100)
    assert summary['mean_tv_ci95'][0] <= summary['mean_tv'] <= summary['mean_tv_ci95'][1]
    assert summary['by_count']['two_strikes']['decisions'] == 500
    expected = runner.SAME if summary['mean_tv'] < .01 and summary['top1_agreement'] >= .98 else runner.DIFFERENT
    assert summary['stop_rule']['status'] == expected
    flipped = [{**same, 'top1_same': False}] * 30 + [same] * 970   # 97% agreement fails the registered 98%
    assert runner.summarize(flipped, games, strikes, rule, draws=200, seed=1)['stop_rule']['status'] == runner.DIFFERENT
    assert runner.summarize([same] * 1000, games, strikes, rule, draws=200, seed=1)['stop_rule']['status'] == runner.SAME
