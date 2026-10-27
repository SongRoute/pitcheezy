"""Synthetic-only tests for EXP-STAB-02 (track A1). No real data, no T7, no 2026, no outcome reads.

Every fixture is built here: a three-action TRAIN BC, frozen 400-draw pools, deterministic row-wise
conditional predictors, WE tables and a fake runtime. The pass runner is driven through a synthetic
environment with a mocked clock and a fake heavy lock.
"""
from collections import Counter
import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import inspect
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp import planner_variance as pv  # noqa: E402
from pitchmdp import planner_variance_registry as reg  # noqa: E402
from pitchmdp.matrix_data import canonical_hash  # noqa: E402
from pitchmdp.planner import OUTCOMES, TERMINALS, solve_pa  # noqa: E402
from pitchmdp.rollout_policy import (BCRecord, BudgetExceeded, CategoricalBC, DeliveryPool, JointSimulator,  # noqa: E402
                                     PAState, PastPitch, RolloutImprovement, RowBudget, probabilities, rollouts)
import diagnose_planner_variance as runner  # noqa: E402

CONFIG = json.loads((REPO / 'configs' / 'EXP-STAB-02.yaml').read_text())
ACTIONS = ('A', 'B', 'C')
WE = {'walk': .44, 'strikeout': .63, 'out': .61, 'single': .41, 'double': .35, 'triple': .32,
      'home_run': .22, 'hbp': .44, 'double_play': .72}
CUTOFF = .5
KST = timezone(timedelta(hours=9))


# ---------------------------------------------------------------- synthetic world

def make_bc():
    rng = np.random.default_rng(7)
    records = []
    for b in range(4):
        for s in range(3):
            for prev in (None, *ACTIONS):
                history = () if prev is None else (PastPitch(prev, (0.5,), 'ball', 0, 0),)
                state = PAState(b, s, 'p', 'R', history, 'ctx')
                for action, k in zip(ACTIONS, rng.integers(1, 6, 3)):
                    records += [BCRecord(state, action, 'train')] * int(k)
    return CategoricalBC(prior_strength=3.).fit(records)


POOLS = {(a, b, s): DeliveryPool((np.linspace(0, 1, 400) ** (1 + j) + .01 * b - .01 * s)[:, None], 'train',
                                 'synthetic', f'{a}-{b}-{s}')
         for j, a in enumerate(ACTIONS) for b in range(4) for s in range(3)}
_RNG = np.random.default_rng(11)
BASE, SLOPE, BALL, STRIKE, HISTORY = (_RNG.normal(size=(3, 10)) * .6, _RNG.normal(size=10), _RNG.normal(size=10),
                                      _RNG.normal(size=10), _RNG.normal(size=(11, 10)) * .5)


def pool(state, action):
    return POOLS[(action, state.balls, state.strikes)]


def predictor(states, actions, physical):
    """Deterministic row-wise conditional law: action, delivery, count and the previous outcome matter."""
    x = np.asarray(physical, dtype=float)[:, 0]
    out = np.empty((len(states), 10))
    for r, (s, a) in enumerate(zip(states, actions)):
        z = BASE[ACTIONS.index(a)] + 1.5 * x[r] * SLOPE + .25 * s.balls * BALL + .3 * s.strikes * STRIKE
        if s.history:
            last = s.history[-1].outcome
            z = z + HISTORY[OUTCOMES.index(last) if last in OUTCOMES else 10]
        z[3:] += .8  # PA-ending outcomes reasonably likely: short synthetic PAs
        e = np.exp(z - z.max())
        out[r] = e / e.sum()
    return out


def terminal(state, event):
    return WE[event]


def cutoff(state):
    return CUTOFF


BC = make_bc()


def simulator(budget=None, predict=predictor):
    return JointSimulator(pool, predict, terminal, budget or RowBudget(10**9))


def state(b=0, s=0, history=(), key='ctx'):
    return PAState(b, s, 'p', 'R', tuple(history), key)


HISTORIES = [(), (PastPitch('A', (0.3,), 'ball', 0, 0),),
             (PastPitch('B', (0.7,), 'strike', 0, 0), PastPitch('C', (0.2,), 'foul', 0, 1))]
STATES = [state(0, 0), state(1, 0, HISTORIES[1]), state(1, 2, HISTORIES[2]), state(3, 2, HISTORIES[2])]


def mc(samples, seed, budget=None, predict=predictor, cap=12):
    return RolloutImprovement(BC, simulator(budget, predict), cutoff, samples=samples, pitch_cap=cap, seed=seed)


def rb(samples, seed, budget=None, predict=predictor, chunk=6, cap=12):
    return pv.RaoBlackwellFirstStep(BC, simulator(budget, predict), cutoff, samples=samples, pitch_cap=cap, seed=seed,
                                    first_pitch_chunk=chunk)


# ---------------------------------------------------------------- the Rao-Blackwell first step

def toy_predictor(states, actions, physical):
    """From the two-pitch root history: an action-only law; any later pitch is an out with certainty."""
    out = np.zeros((len(states), 10))
    for r, (s, a) in enumerate(zip(states, actions)):
        if len(s.history) > 2:
            out[r, 3] = 1.
        else:
            z = BASE[ACTIONS.index(a)].copy()
            out[r] = np.exp(z) / np.exp(z).sum()
    return out


def test_rb_first_step_is_the_exact_expectation_on_a_toy_mdp_and_matches_mc_in_the_limit():
    root = state(1, 1, (PastPitch('TOY', (0.5,), 'ball', 0, 0), PastPitch('A', (0.5,), 'strike', 1, 0)))
    planner = rb(6, 3, predict=toy_predictor)
    q, diagnostics = planner.q_values(root)
    mask = BC.support(root)
    for r, action in enumerate(BC.actions):
        p = probabilities(toy_predictor([root], [action], [[0.]]), 10)[0]
        exact = sum(p[k] * (WE[pv.terminal_event(root, o)] if pv.terminal_event(root, o) else WE['out'])
                    for k, o in enumerate(OUTCOMES))
        assert mask[r]
        assert q[r] == pytest.approx(exact, abs=1e-15)
        assert np.ptp(diagnostics['sample_values'][r]) < 1e-15  # zero variance: the first step was the only noise
    draws = np.array([mc(6, seed, predict=toy_predictor).q_values(root)[0] for seed in range(400)])
    se = draws.std(axis=0, ddof=1) / np.sqrt(len(draws))
    assert (np.abs(draws.mean(axis=0) - q) <= 4 * se + 1e-12).all()


def forced_uniform(p, outcome):
    cdf = np.cumsum(p)
    low = cdf[outcome - 1] if outcome else 0.
    return (low + cdf[outcome]) / 2


def test_rb_sample_is_the_probability_weighted_sum_of_outcome_forced_registered_rollouts():
    for root in STATES:
        planner = rb(6, 11)
        _, diagnostics = planner.q_values(root)
        names = [a for a, ok in zip(BC.actions, BC.support(root)) if ok]
        common = np.random.default_rng(pv.planning_seed(root, names, 11)).random((6, 12, 3))
        sim = simulator()
        for action in names:
            r = BC.actions.index(action)
            for i in range(6):
                delivery = pool(root, action).values[int(common[i, 0, 1] * 400)]
                p = probabilities(predictor([root], [action], [delivery]), 10)[0]
                w = np.zeros(10)
                for k in range(10):
                    if p[k] > 0:
                        u = common[i].copy()[None]
                        u[0, 0, 2] = forced_uniform(p, k)
                        w[k] = rollouts(sim, [root], planner.baseline, u, cutoff, first_actions=[action]).values[0]
                assert np.sum(p * w) == diagnostics['sample_values'][r, i]


def test_same_seed_selected_paths_reproduce_the_registered_planner_and_nest():
    for root in STATES:
        q6, d6 = mc(6, 5).q_values(root)
        q12, d12 = mc(12, 5).q_values(root)
        r6, rd6 = rb(6, 5).q_values(root)
        r12, rd12 = rb(12, 5).q_values(root)
        mask = BC.support(root)
        np.testing.assert_array_equal(rd6['selected_values'][mask], d6['sample_values'][mask])
        np.testing.assert_array_equal(rd12['selected_values'][mask], d12['sample_values'][mask])  # MC12, all 12
        np.testing.assert_array_equal(rd12['selected_values'][mask, :6], d6['sample_values'][mask])  # first 6 = MC6
        np.testing.assert_array_equal(pv.selected_q(rd6, 6)[mask], q6[mask])
        np.testing.assert_array_equal(pv.selected_q(rd12, 6)[mask], q6[mask])
        np.testing.assert_array_equal(pv.selected_q(rd12, 12)[mask], q12[mask])
        np.testing.assert_array_equal(rd12['sample_values'][:, :6][mask], rd6['sample_values'][mask])  # S12 nests S6
        whole = rb(12, 5, chunk=12).q_values(root)  # first-pitch batch in one chunk instead of 0-5 / 6-11
        np.testing.assert_array_equal(whole[0], r12)
        np.testing.assert_array_equal(whole[1]['sample_values'], rd12['sample_values'])
        again = rb(6, 5).q_values(root)  # fresh planner, same seed: deterministic
        np.testing.assert_array_equal(again[0], r6)
        assert (q6[~mask] == -np.inf).all() and (r6[~mask] == -np.inf).all()


def test_first_pitch_children_and_traced_steps_equal_joint_simulator_step():
    for root in STATES:
        actions = [a for a, ok in zip(BC.actions, BC.support(root)) if ok]
        u = np.linspace(.05, .95, len(actions))
        budget = RowBudget(10**6)
        delivery, p, values, children = pv.expand_first_pitch(simulator(budget), root, actions, u)
        assert budget.conditional_rows == len(actions)
        for r, action in enumerate(actions):
            for k in range(10):
                if p[r, k] <= 0:
                    continue
                step = simulator().step([root], [action], [[u[r], forced_uniform(p[r], k)]])[0]
                if step.state is None:
                    assert children[r][k] is None and values[r, k] == step.value
                else:
                    assert children[r][k] == step.state and np.isnan(values[r, k])
        states = [c for kids in children for c in kids if c is not None]
        draws = np.random.default_rng(3).random((len(states), 2))
        reference, twin = simulator(RowBudget(10**6)), pv._TracingSimulator(simulator())
        twin.begin(states)
        expected, traced = reference.step(states, ['A'] * len(states), draws), twin.step(states, ['A'] * len(states), draws)
        assert expected == traced and twin.budget.conditional_rows == reference.budget.conditional_rows
        assert np.isfinite(twin.margin).all() and (twin.margin >= 0).all()


def test_rb_row_accounting_and_budget_stop():
    root = STATES[1]
    seen = []

    def counting(states, actions, physical):
        seen.append(len(states))
        return predictor(states, actions, physical)
    budget = RowBudget(10**6)
    planner = rb(6, 2, budget, counting)
    _, diagnostics = planner.q_values(root)
    n = int(BC.support(root).sum())
    assert seen[0] == n * 6 and budget.conditional_rows == sum(seen) == diagnostics['rows']
    assert planner.diagnostics['first_pitch_rows'] == n * 6
    with pytest.raises(BudgetExceeded):
        rb(6, 2, RowBudget(n * 6 + 1)).q_values(root)


# ---------------------------------------------------------------- reconstruction (integrity)

def test_reconstruction_passes_on_matching_streams_and_fails_on_tampered_streams():
    tally = pv.ReconstructionTally(1e-5, .001)
    for root in STATES:
        planner = rb(6, 5)
        _, diagnostics = planner.q_values(root)
        q0, d0 = mc(6, 5).q_values(root)
        mask = BC.support(root)
        good = pv.reconstruction_check(diagnostics, q0, mask, 6, a0_sample_values=d0['sample_values'])
        assert good == {'mismatch': False, 'actions': [], 'margin': None, 'first_pitch_flip': False}
        tally.add('ok', {'seed 5': good})
    assert tally.finish()['mismatched_decisions'] == 0
    root = STATES[1]
    planner = rb(6, 5)
    _, diagnostics = planner.q_values(root)
    mask = BC.support(root)
    other = mc(6, 6).q_values(root)[0]  # a different stream
    with pytest.raises(pv.SearchIntegrityError, match='traced'):
        pv.reconstruction_check(diagnostics, other, mask, 6)
    bad = pv.reconstruction_check(diagnostics, other, mask, 6, trace=planner.trace(root)[1])
    assert bad['mismatch'] and bad['margin'] > 1e-5
    with pytest.raises(pv.SearchIntegrityError):
        pv.ReconstructionTally(1e-5, .001).add('tampered', {'seed 5': bad})
    q0, d0 = mc(6, 5).q_values(root)
    samples = d0['sample_values'].copy()
    r = int(np.flatnonzero(mask)[0])
    samples[r, 2] += .1
    tampered = q0.copy()
    tampered[r] = samples[r].mean()
    check = pv.reconstruction_check(diagnostics, tampered, mask, 6, a0_sample_values=samples,
                                    trace=planner.trace(root)[1])
    assert check['actions'] == [r] and check['margin'] > 1e-5
    trace = planner.trace(root)[1]
    tampered_trace = dict(trace, sample_values=trace['sample_values'] + 1e-9)
    with pytest.raises(pv.SearchIntegrityError, match='reproduce'):
        pv.reconstruction_check(diagnostics, other, mask, 6, trace=tampered_trace)


def test_reconstruction_margin_rule_flips_and_share():
    width, S = 3, 6
    base = {'selected_values': np.full((width, S), .5), 'sample_values': np.full((width, S), .5)}
    mask = np.array([True, True, False])
    a0 = np.array([.5, .5 + 1e-3, -np.inf])
    near = {**base, 'first_outcome_terminal': np.zeros((width, S), bool), 'selected_margin': np.full((width, S), 1e-7)}
    check = pv.reconstruction_check(base, a0, mask, S, trace=near)
    assert check['mismatch'] and check['actions'] == [1] and check['margin'] == pytest.approx(1e-7)
    tally = pv.ReconstructionTally(1e-5, .001)
    tally.add('near-boundary', {'seed 0': check})  # a float-rounding flip next to a CDF boundary is tolerated
    for k in range(999):
        tally.add(f'ok-{k}', {'seed 0': {'mismatch': False}})
    assert tally.finish()['mismatched_decisions'] == 1  # 1 / 1000 = 0.1%: allowed
    tally.add('second', {'seed 0': check})
    with pytest.raises(pv.SearchIntegrityError, match='exceed'):
        tally.finish()
    terminal_first = {**near, 'first_outcome_terminal': np.ones((width, S), bool)}
    flip = pv.reconstruction_check(base, a0, mask, S, trace=terminal_first)
    assert flip['first_pitch_flip'] and np.isinf(flip['margin'])
    with pytest.raises(pv.SearchIntegrityError, match='first-pitch'):
        pv.ReconstructionTally(1e-5, .001).add('flip', {'seed 0': flip})
    far = {**near, 'selected_margin': np.full((width, S), .2)}
    with pytest.raises(pv.SearchIntegrityError, match='margin'):
        pv.ReconstructionTally(1e-5, .001).add('far', {'seed 0': pv.reconstruction_check(base, a0, mask, S, trace=far)})


# ---------------------------------------------------------------- the frozen-history DP oracle

P_TABLE = np.random.default_rng(21).dirichlet(np.ones(10), size=(4, 3, 3))
P_TABLE[:, 2, :, 2] = 0  # two-strike foul probability 0: every PA ends within 6 pitches < cap 12
P_TABLE /= P_TABLE.sum(axis=-1, keepdims=True)
P_LOOP = np.random.default_rng(22).dirichlet(np.ones(10), size=(4, 3, 3))
P_LOOP[:, 2, :, 2] = .25 + .1 * np.arange(3)  # two-strike foul loops
P_LOOP[:, 2, :, :2] *= .6
P_LOOP /= P_LOOP.sum(axis=-1, keepdims=True)


def table_predictor(table):
    def predict(states, actions, physical):
        return np.stack([table[s.balls, s.strikes, ACTIONS.index(a)] for s, a in zip(states, actions)])
    return predict


def oracle(root, table):
    mask = BC.support(root)
    names = [a for a, ok in zip(BC.actions, mask) if ok]
    own = root.history[-1].action if root.history else pv.START
    previous = names + ([own] if own not in names else [])
    p = np.stack([np.stack([np.stack([table[b, s, [ACTIONS.index(a) for a in names]]] * len(previous))
                            for s in range(3)]) for b in range(4)])
    base = np.empty((4, 3, len(previous), len(names)))
    for b, s in itertools.product(range(4), range(3)):
        for k, prev in enumerate(previous):
            history = root.history if prev == own else (PastPitch(prev, (0.,), 'unknown', 0, 0),)
            base[b, s, k] = BC.probabilities(state(b, s, history))[mask]
    plan = solve_pa(p, WE, [previous.index(a) for a in names], base)
    q = np.full(len(BC.actions), -np.inf)
    q[mask] = plan.myopic_q_values[root.balls, root.strikes, previous.index(own)]
    return q


def test_frozen_history_dp_equals_solve_pa_and_mc_rb_agree_when_no_pitch_cap_binds():
    predict = table_predictor(P_TABLE)
    for root in (state(0, 0), state(1, 1, HISTORIES[1])):
        exact = oracle(root, P_TABLE)
        budget = RowBudget(10**9)
        dp_q, dp_d = pv.FrozenHistoryDP(BC, simulator(budget, predict), cutoff).q_values(root)
        mask = BC.support(root)
        n, nodes = int(mask.sum()), (4 - root.balls) * (3 - root.strikes)
        np.testing.assert_allclose(dp_q[mask], exact[mask], rtol=0, atol=1e-12)
        assert budget.conditional_rows == dp_d['rows'] == n * (400 + (nodes - 1) * 40)
        a0 = np.array([mc(6, 1000 + k, predict=predict).q_values(root)[0] for k in range(300)])
        rbq = np.array([rb(6, 1000 + k, predict=predict).q_values(root)[0] for k in range(300)])
        for draws in (a0, rbq):
            se = draws[:, mask].std(axis=0, ddof=1) / np.sqrt(len(draws))
            assert (np.abs(draws[:, mask].mean(axis=0) - exact[mask]) <= 4 * se + 1e-12).all()
        anchor = int(np.argmax(np.where(mask, exact, -np.inf)))
        for r in np.flatnonzero(mask):
            if r != anchor:
                assert np.var(rbq[:, r] - rbq[:, anchor]) <= np.var(a0[:, r] - a0[:, anchor])
        _, d = mc(6, 1, predict=predict).q_values(root)
        assert d['truncated'] == 0  # the cap never binds


def test_mc_with_a_large_cap_converges_to_the_dp_with_two_strike_foul_loops():
    predict = table_predictor(P_LOOP)
    root = state(0, 0)
    exact = oracle(root, P_LOOP)
    dp_q, _ = pv.FrozenHistoryDP(BC, simulator(predict=predict), cutoff).q_values(root)
    mask = BC.support(root)
    np.testing.assert_allclose(dp_q[mask], exact[mask], rtol=0, atol=1e-12)
    samples = np.concatenate([mc(400, 7 + k, predict=predict, cap=60).q_values(root)[1]['sample_values']
                              for k in range(6)], axis=1)[mask]
    se = samples.std(axis=1, ddof=1) / np.sqrt(samples.shape[1])
    assert (np.abs(samples.mean(axis=1) - exact[mask]) <= 4 * se + 1e-9).all()


# ---------------------------------------------------------------- metric functions

def test_metric_functions_ties_one_action_support_and_bin_edges():
    edges = reg.GAP_BINS
    assert [pv.gap_bin(g, edges) for g in (0., .0024999, .0025, .005, .01, .02, .05, 3.)] == [0, 0, 1, 2, 3, 4, 5, 5]
    assert pv.gap_bin(None, edges) == -1
    reference = np.array([.5, .3, .2])
    mask = np.array([True, True, True])
    q = np.array([.6, .55, .5])
    p = pv.law(q, reference, mask, .1)
    j1, j2 = pv.top_two(p)
    assert pv.effective_gap(q, reference, mask, .1) == pytest.approx(.1 * np.log(p[j1] / p[j2]), abs=1e-12)
    tie_q = np.array([.6 + .1 * np.log(.3 / .5), .6, .2])
    assert pv.effective_gap(tie_q, reference, mask, .1) == pytest.approx(0, abs=1e-12)
    one = np.array([True, False, False])
    assert pv.effective_gap(q, np.array([1., 0, 0]), one, .1) is None and pv.q_gap(q, one) is None
    assert pv.q_gap(q, mask) == pytest.approx(.05)
    samples = np.array([[.5, .7, .6, .4], [.4, .7, .5, .3], [np.nan] * 4])
    v = pv.paired_difference_variance(samples, np.array([True, True, False]), 0)
    assert v == pytest.approx(np.var(samples[1] - samples[0], ddof=1) / 4)
    assert pv.paired_difference_variance(samples, np.array([True, False, False]), 0) == 0.
    games = np.repeat([3, 1, 2], [4, 3, 5])
    values = np.random.default_rng(0).random(12)
    boot = pv.GameBootstrap(games, draws=500, seed=20260924)
    unique, index = np.unique(games, return_inverse=True)  # the EXP-STAB-01 summarize() table, inline
    picks = np.random.default_rng(20260924).integers(0, 3, (500, 3))
    sums = np.bincount(index, weights=values, minlength=3)
    expected = np.quantile(sums[picks].sum(1) / np.bincount(index, minlength=3)[picks].sum(1), (.025, .975))
    np.testing.assert_allclose(boot.ci(values), expected, rtol=0, atol=1e-15)
    empty = boot.subset(values, np.zeros(12, bool))
    assert empty['decisions'] == 0 and empty['ci95'] is None
    only = boot.subset(values, games == 3)
    assert only['valid_replicates'] < 500 and only['value'] == pytest.approx(values[games == 3].mean())


# ---------------------------------------------------------------- registry: config, gate, verdict, window

def registered_config(**overrides):
    config = copy.deepcopy(CONFIG)
    config['status'], config['execute'] = reg.STATUS_REGISTERED, True
    config['registration'].update(registered_kst='synthetic test', registered_commit='0' * 40,
                                  design_review_reference='REVIEW-A1-ASTRA-003', code_review_reference='synthetic',
                                  code_sha256={rel: '0' * 64 for rel in reg.NEW_CODE})
    for dotted, value in overrides.items():
        node = config
        *head, last = dotted.split('.')
        for part in head:
            node = node[part]
        node[last] = value
    return config


def test_draft_config_matches_the_registry_and_fails_closed():
    assert reg.config_check(CONFIG) is CONFIG
    assert CONFIG['status'] == reg.STATUS_DRAFT and CONFIG['execute'] is False
    with pytest.raises(reg.RegistrationError, match='DRAFT_UNREGISTERED'):
        reg.registration_gate(CONFIG)
    with pytest.raises(reg.RegistrationError):
        runner.PassRunner(CONFIG, object(), Path('/nonexistent/attempt-001'), identity={})
    with pytest.raises(reg.RegistrationError):
        runner.RealEnvironment(CONFIG, {})
    assert reg.registration_gate(registered_config())['status'] == reg.STATUS_REGISTERED
    assert reg.verify_unchanged_sources(REPO) == reg.UNCHANGED_SHA256  # the A0 path is byte-identical
    assert all((REPO / rel).is_file() for rel in reg.NEW_CODE)


def test_fail_closed_cli_refuses_a_draft_before_any_read(tmp_path, capsys):
    attempt = tmp_path / 'EXP-STAB-02' / 'attempt-001'
    code = runner.main(['--config', str(REPO / reg.CONFIG_PATH), '--local-config', str(tmp_path / 'missing.json'),
                        '--attempt', str(attempt), '--through', 'smoke'])
    assert code == runner.EXIT_REFUSED and 'DRAFT_UNREGISTERED' in capsys.readouterr().out
    assert not attempt.exists() and not (tmp_path / 'EXP-STAB-02').exists()
    with pytest.raises(reg.RegistrationError, match='inside the repository'):
        runner.assert_attempt_path(REPO / 'results' / 'attempt-001')
    with pytest.raises(reg.RegistrationError, match='attempt-NNN'):
        runner.assert_attempt_path(tmp_path / 'EXP-STAB-02')
    assert runner.assert_attempt_path(attempt) == attempt.resolve()


@pytest.mark.parametrize('dotted, value, message', [
    ('decision_rules.top1_lower_at_least', .94, 'decision_rules'),
    ('decision_rules.mean_tv_upper_at_most', .03, 'decision_rules'),
    ('decision_rules.r_rows_at_most', 8.0, 'decision_rules'),
    ('seeds.reference', 3, 'disjoint'),
    ('seeds.pair', [0, 0], 'two distinct'),
    ('arms.RB12.row_budget', {'0': 1, '5': 36000000}, 'arms'),
    ('arms.A0-MC6.row_budget', {'0': 9000000, '5': 8000000}, 'arms'),
    ('settings.samples', 12, 'settings'),
    ('games.games_sha256', '0' * 64, 'games'),
    ('games.n_games', 151, 'games'),
    ('cost.unit_U_seconds', 1200.0, 'cost'),
    ('passes.smoke.cap_seconds', 7200, 'passes'),
    ('run_window.protection_window_kst', ['06:30', '12:00'], 'run_window'),
    ('integrity.reconstruction_margin_max', 1e-3, 'integrity'),
    ('bootstrap.draws', 1000, 'bootstrap'),
    ('stab01.decisions_npz_sha256', 'f' * 64, 'stab01'),
    ('first_pitch_chunk', 12, 'first_pitch_chunk'),
    ('registered_policy_identity', 'x', 'registered_policy_identity'),
])
def test_config_check_rejects_every_deviation(dotted, value, message):
    deviating = registered_config(**{dotted: value})
    with pytest.raises(reg.RegistrationError, match=message):
        reg.config_check(deviating)


def test_status_and_registration_rules():
    with pytest.raises(reg.RegistrationError, match='cannot enable'):
        reg.config_check({**copy.deepcopy(CONFIG), 'execute': True})
    with pytest.raises(reg.RegistrationError, match='execute true'):
        reg.config_check(registered_config(execute=False))
    missing = registered_config()
    missing['registration']['code_review_reference'] = None
    with pytest.raises(reg.RegistrationError, match='code_review_reference'):
        reg.registration_gate(missing)
    wrong_code = registered_config()
    wrong_code['registration']['code_sha256'] = {'x.py': '0' * 64}
    with pytest.raises(reg.RegistrationError, match='code_sha256'):
        reg.registration_gate(wrong_code)
    with pytest.raises(reg.RegistrationError, match='code differs'):
        reg.verify_registered_code(registered_config(), REPO)
    draft_with_fields = copy.deepcopy(CONFIG)
    draft_with_fields['registration']['registered_commit'] = 'abc'
    with pytest.raises(reg.RegistrationError, match='stay null'):
        reg.config_check(draft_with_fields)


def test_new_modules_stay_outside_the_s2_bind_identity_source_closure():
    """policy_identity.project_sources hashes the runtime modules and their import closure; the new modules are
    imported in this process (as in a real run) and must not enter it, so the identity stays 5683fb61."""
    from pitchmdp import policy_identity as pid
    import run_ml_g0_whole
    assert 'pitchmdp.planner_variance' in sys.modules and 'pitchmdp.planner_variance_registry' in sys.modules
    closure = pid.project_sources([run_ml_g0_whole.load_member])
    assert closure and not [k for k in closure if 'planner_variance' in k or k.endswith('__init__.py')]


def test_unchanged_source_tampering_is_detected(tmp_path):
    for rel in reg.UNCHANGED_SHA256:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / rel).read_bytes())
    assert reg.verify_unchanged_sources(tmp_path)
    changed = tmp_path / 'experiments/pitchmdp/pitchmdp/rollout_policy.py'
    changed.write_bytes(changed.read_bytes() + b'\n')
    with pytest.raises(reg.RegistrationError, match='rollout_policy.py'):
        reg.verify_unchanged_sources(tmp_path)


def reference_label(smoke_r_b, pass_m, rb6, rb12):
    """Independent restatement of the design section 6 precedence."""
    if smoke_r_b > 4.0:
        return 'COST_EXCEEDED_AT_SMOKE', []
    if pass_m == 'aborted':
        return None, ['PASS_M_ABORTED']
    run = [('RB6', rb6['stable'], rb6['cost_ok'])]
    flags = []
    if rb12 is not None:
        if rb12['state'] == 'completed':
            run.append(('RB12', rb12['stable'], rb12['cost_ok']))
        elif rb12['state'] == 'aborted':
            flags = ['RB12_ABORTED']
        else:
            return None, ['PENDING_PASS_C']
    within = [arm for arm, s, c in run if s and c]
    if within:
        return 'STABLE_WITHIN_COST:' + min(within, key=lambda a: {'RB6': 6, 'RB12': 12}[a]), flags
    over = [arm for arm, s, _ in run if s]
    if over:
        return 'STABLE_OVER_COST:' + over[0], flags
    return 'NOT_STABLE', flags


def test_verdict_partition_is_exhaustive_and_exclusive():
    assert 'h2' not in ''.join(inspect.signature(reg.verdict).parameters).lower()  # H2 never enters the verdict
    reached = Counter()
    bools = (True, False)
    rb12_states = [None, {'state': 'pending'}, {'state': 'aborted'}] + [
        {'state': 'completed', 'stable': s, 'cost_ok': c} for s in bools for c in bools]
    for smoke_r_b, pass_m, rb12 in itertools.product((1.5, 2.0, 3.0, 4.0, 4.5), ('completed', 'aborted', 'not_run'),
                                                     rb12_states):
        rb6_options = ([{'stable': s, 'cost_ok': c} for s in bools for c in bools] if pass_m == 'completed'
                       else [None])
        for rb6 in rb6_options:
            kwargs = {'smoke_r_b': smoke_r_b, 'pass_m': pass_m, 'rb6': rb6, 'rb12': rb12}
            if smoke_r_b > 4:
                consistent = pass_m == 'not_run' and rb12 is None
            elif pass_m == 'aborted':
                consistent = rb12 is None
            elif pass_m == 'completed':
                consistent = (rb12 is None) == (not reg.rb12_required(smoke_r_b, rb6['stable']))
            else:
                consistent = False
            if not consistent:  # a combination the registered run conditions cannot produce
                with pytest.raises(ValueError):
                    reg.verdict(**kwargs)
                continue
            out = reg.verdict(**kwargs)
            assert (out['label'], out['flags']) == reference_label(smoke_r_b, pass_m, rb6, rb12), kwargs
            assert out['final'] == (out['row'] is not None)
            reached[out['row']] += 1
    assert set(reached) == set(reg.VERDICT_ROWS) | {None}  # rows 0-8 and the pending state, each reachable
    assert reg.confirmation_verdict('aborted') == 'CONFIRMATION_ABORTED'
    assert reg.confirmation_verdict('completed', top1_ci95=[.96, .99], mean_tv_ci95=[.01, .019], r_rows=3.) == 'CONFIRMED'
    assert reg.confirmation_verdict('completed', top1_ci95=[.94, .99], mean_tv_ci95=[.01, .019], r_rows=3.) == 'NOT_CONFIRMED'
    assert reg.confirmation_verdict('completed', top1_ci95=[.96, .99], mean_tv_ci95=[.01, .021], r_rows=3.) == 'NOT_CONFIRMED'
    assert reg.confirmation_verdict('completed', top1_ci95=[.96, .99], mean_tv_ci95=[.01, .019], r_rows=4.1) == 'NOT_CONFIRMED'


def test_caps_window_and_deadline_rule():
    U = reg.COST['unit_U_seconds']
    assert reg.pass_cap_seconds('smoke') == 3600 and reg.pass_cap_seconds('M') == 21600
    assert reg.pass_cap_seconds('C', r_b=2.0) == pytest.approx(1.2 * 4 * 2.0 * U)
    assert reg.pass_cap_seconds('D', arm='BF48') == pytest.approx(1.2 * 16 * U * 50 / 150)
    assert reg.pass_cap_seconds('D', arm='FH-DP') == 7200
    assert reg.pass_cap_seconds('confirmation', r_b=3.0, arm='RB6') == pytest.approx(1.2 * (2 + 2 * 3.) * U / 3)
    assert reg.pass_cap_seconds('confirmation', r_b=3.0, arm='RB12') == pytest.approx(1.2 * (2 + 4 * 3.) * U / 3)
    with pytest.raises(ValueError):
        reg.pass_cap_seconds('C')
    at = lambda h, m=0, day=27: datetime(2026, 10, day, h, m, tzinfo=KST).astimezone(timezone.utc)  # noqa: E731
    decision, deadline = reg.start_decision(at(14), 21600)
    assert decision == 'START' and deadline == at(20)
    assert reg.start_decision(at(10), 600)[0] == 'DEFER'  # protection window (06:00-13:30 no start)
    assert reg.start_decision(at(6, 10), 60)[0] == 'DEFER'
    assert reg.start_decision(at(13, 30), 60)[0] == 'START'
    assert reg.start_decision(at(1, day=28), 3600)[0] == 'START'  # 01:00-02:00 KST still before 06:00
    assert reg.start_decision(at(5, 30, day=28), 3600)[0] == 'DEFER'  # would end after 06:00 KST
    assert reg.start_decision(at(0, day=28), 21600)[0] == 'START'  # 00:00 + 6 h ends exactly at 06:00 KST
    assert reg.start_decision(at(0, 1, day=28), 21600)[0] == 'DEFER'
    assert reg.next_deadline_utc(at(23)) == at(6, day=28)
    assert reg.next_deadline_utc(at(3, day=28)) == at(6, day=28)


def test_run_conditions_of_the_optional_arms():
    assert reg.rb12_required(2.0, False) and not reg.rb12_required(2.01, False) and not reg.rb12_required(1.0, True)
    assert reg.descriptive_allowed(False, True, 100., 200.) and reg.descriptive_allowed(True, False, 100., 200.)
    assert not reg.descriptive_allowed(True, True, 100., 200.) and not reg.descriptive_allowed(False, False, 300., 200.)


# ---------------------------------------------------------------- the pass runner on a synthetic environment

@dataclass
class Request:
    request_id: str
    pa_id: str
    state: PAState
    pitcher_hand: str = 'R'


class FakeRuntime:
    """The registered runtime's contract as the runner uses it: submit -> row with the seed-0 planning Q."""
    def __init__(self, env, ledger_path, budget):
        self.reference, self.sha256 = env.reference, 'synthetic-runtime'
        self.improvement = RolloutImprovement(env.reference, simulator(budget, env.predict), cutoff, samples=6,
                                              pitch_cap=12, seed=0)
        Path(ledger_path).write_text('{"kind": "header", "synthetic": true}\n')

    def submit(self, request):
        s = request.state
        mask, reference = self.reference.support(s), self.reference.probabilities(s)
        q, diagnostics = self.improvement.q_values(s)
        off = np.where(mask, 0., np.nan)
        listed = lambda v: [None if not np.isfinite(x) else float(x) for x in v]  # noqa: E731
        se = runner.paired_difference_se(diagnostics['sample_values'], mask)
        return {'status': 'SUPPORTED', 'result': {'mask': mask.tolist(), 'reference': reference.tolist(),
                                                  'q_planning': listed(q + off), 'planning_diff_se': listed(se + off)}}

    def verify_components(self):
        return True


class SyntheticEnv:
    def __init__(self, counts=((0, 0), (1, 1), (3, 2)), games=(101, 102, 103, 104), predict=predictor):
        self.predict, self.reference = predict, BC
        self.games = list(games)
        self.games_record = {'games_sha256': 'synthetic', 'selected': len(self.games)}
        order = runner.subsample_order(self.games)
        self.subsamples = {'BF48': order[:reg.SUBSAMPLES['BF48']], 'FH-DP': order[:reg.SUBSAMPLES['FH-DP']]}
        self.fhdp_smoke = order[0]
        self.confirmation = [201, 202]
        self.confirmation_record = {'games_sha256': canonical_hash(sorted(self.confirmation)), 'selected': 2}
        self.pas = {}
        for g in self.games + self.confirmation:
            blocks = []
            for pa in range(2):
                history, requests = (), []
                for k, (b, s) in enumerate(counts):
                    requests.append(Request(f'{g}:{pa}:{k + 1}', f'{g}:{pa}', state(b, s, history, f'{g}-{pa}')))
                    history = history + (PastPitch(ACTIONS[(g + pa + k) % 3], (.1 + .2 * k,), 'foul', b, s),)
                blocks.append(requests)
            self.pas[g] = blocks
        self.prepared, self.verified, self._stab01 = [], 0, None

    def prepare(self, scope):
        self.prepared.append(scope)

    def smoke_games(self):
        return self.games[:reg.GAMES['smoke_games']]

    def blocks(self, games):
        for g in sorted(set(games)):
            for block in self.pas[g]:
                yield g, block

    def requests(self, block):
        return block

    def build_runtime(self, ledger_path, budget):
        return FakeRuntime(self, ledger_path, budget)

    def planner(self, kind, samples, seed, budget):
        if getattr(self, 'tiny_a0_seed5_budget', False) and kind == 'MC' and seed == 5:
            budget = RowBudget(50, seed_count=5)
        sim = simulator(budget, self.predict)
        if kind == 'MC':
            return RolloutImprovement(BC, sim, cutoff, samples=samples, pitch_cap=12, seed=seed)
        if kind == 'RB':
            return pv.RaoBlackwellFirstStep(BC, sim, cutoff, samples=samples, pitch_cap=12, seed=seed)
        return pv.FrozenHistoryDP(BC, sim, cutoff)

    def check_request(self, request, mask):
        assert request.state.pitcher == 'p'

    def verify(self, planners, runtime=None):
        for planner in planners:
            assert planner.simulator.predictor is self.predict and planner.cutoff is cutoff
        self.verified += 1

    def stab01(self):
        if self._stab01 is None:
            stab = runner._stab()
            sets = {'A': [0, 1, 2, 3, 4], 'B': [5, 6, 7, 8, 9], 'K': [1, 3, 5]}
            planners = {s: mc(6, s, predict=self.predict) for s in range(10)}
            columns = {k: [] for k in reg.STAB01_COLUMNS}
            for g, block in self.blocks(self.games):
                for request in block:
                    s = request.state
                    mask, reference = BC.support(s), BC.probabilities(s)
                    q = {seed: planner.q_values(s)[0] for seed, planner in planners.items()}
                    q[0] = q[0] + np.where(mask, 0., np.nan)  # the runtime row convention (NaN off the support)
                    columns['request_id'].append(request.request_id)
                    columns['game_pk'].append(g)
                    for key, value in stab.compare(q, reference, mask, .1, sets).items():
                        if key in columns:
                            columns[key].append(value)
            self._stab01 = {k: np.array(v) for k, v in columns.items()}
        return self._stab01

    def data_version(self):
        return {'synthetic': True}


class FakeClock:
    def __init__(self, local, tick=0.):
        self.t, self.tick, self.mono = local.astimezone(timezone.utc), tick, 0.

    def now(self):
        self.t += timedelta(seconds=self.tick)
        self.mono += self.tick
        return self.t

    def monotonic(self):
        return self.mono


class FakeLock:
    def __init__(self, held=False):
        self.events, self.held = [], held

    @contextlib.contextmanager
    def __call__(self):
        if self.held:
            raise RuntimeError('Another ML matrix heavy job holds .heavy.lock')
        self.events.append('acquire')
        try:
            yield
        finally:
            self.events.append('release')


def identity():
    return {'config_sha256': 'synthetic-config', 'source_hashes': {'pitchmdp/planner_variance.py': 'synthetic'},
            'pins_sha256': reg.pins_digest()}


def make_runner(config, env, attempt, local=datetime(2026, 10, 27, 14, 0, tzinfo=KST), tick=0., lock=None, rss=None):
    return runner.PassRunner(config, env, attempt, identity=identity(), clock=FakeClock(local, tick),
                             lock=lock or FakeLock(), rss=rss or (lambda: 1 << 30), log=lambda *a: None)


def receipts(attempt):
    return [json.loads(line) for line in (attempt / 'receipts.jsonl').read_text().splitlines()]


def test_smoke_records_cost_and_structure_only(tmp_path):
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv()
    lock = FakeLock()
    work = make_runner(registered_config(), env, attempt, lock=lock)
    assert work.advance('smoke') == runner.EXIT_OK
    assert lock.events == ['acquire', 'release'] and env.prepared == ['smoke']
    folder = attempt / 'smoke'
    results = json.loads((folder / 'results.json').read_text())
    assert reg.smoke_metric_keys(results) == []
    assert not list(folder.glob('*.npz'))
    assert results['structural_ok'] and results['structural']['failure_count'] == 0
    assert results['mechanical_rule']['status'] in ('FULL_RUN_ALLOWED', 'COST_EXCEEDED_AT_SMOKE')
    rows = results['cost']['rows_by_arm_seed']
    assert results['r_b'] == pytest.approx(np.mean([rows[f'RB6 seed {s}'] / rows[f'A0 seed {s}'] for s in (0, 5)]))
    assert {'RB12 seed 0', 'BF48 seed 5', 'REF-RB6 seed 100', 'RB6-rerun seed 0', 'FH-DP'} <= set(rows)
    assert results['decisions'] == 3 * 2 * 3 and results['fhdp_decisions'] >= 6
    text = (folder / 'results.json').read_text()
    assert '"q_' not in text and 'top1' not in text and '"tv' not in text
    runner.verify_sums(folder)
    assert [r['event'] for r in receipts(attempt)] == ['STARTED', 'COMPLETED']
    assert work.advance('smoke') == runner.EXIT_OK  # nothing left: no rerun into a finished folder
    assert [r['event'] for r in receipts(attempt)] == ['STARTED', 'COMPLETED']
    with pytest.raises(runner.IntegrityFailure):  # the smoke key guard refuses metric-looking keys
        runner.require(not reg.smoke_metric_keys({'cost': {'mean_tv': 0.1}}), 'metric key')


def flow_config(monkeypatch, *, stable=False, rb12_cap=10.0):
    """Registry pins moved for a synthetic flow test; the config is rewritten to agree with them."""
    rules = dict(reg.THRESHOLDS)
    if stable:
        rules.update(top1_lower_at_least=-1.0, mean_tv_upper_at_most=10.0, r_rows_at_most=100.0)
    else:
        rules.update(top1_lower_at_least=1.01)  # never STABLE on any data
    rules.update(rb12_smoke_r_b_at_most=rb12_cap)
    for key, value in rules.items():
        monkeypatch.setitem(reg.THRESHOLDS, key, value)
    return registered_config(decision_rules=dict(reg.THRESHOLDS))


def test_mandatory_rb12_deferral_lock_release_and_continuation(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(counts=((3, 2), (2, 2), (3, 1)))
    first_lock = FakeLock()
    assert make_runner(config, env, attempt, lock=first_lock).advance('M') == runner.EXIT_OK
    mandatory = json.loads((attempt / 'mandatory' / 'results.json').read_text())
    assert mandatory['rb12_required'] and mandatory['verdict'] == {'row': None, 'label': None,
                                                                   'flags': ['PENDING_PASS_C'], 'final': False}
    assert mandatory['integrity']['stab01_k1_match_rate'] == 1.0
    assert mandatory['integrity']['reconstruction']['mismatched_decisions'] == 0
    assert mandatory['integrity']['replay']['max_abs_q_difference'] == 0.0
    secondary = mandatory['comparison']['secondary']
    assert set(secondary['gap_bins']['effective_gap']) == {'support_1', '[0.0, 0.0025)', '[0.0025, 0.005)',
                                                            '[0.005, 0.01)', '[0.01, 0.02)', '[0.02, 0.05)', '[0.05, inf)'}
    assert 'rb_improves_over_baseline' in mandatory['comparison']['secondary_flags']
    with np.load(attempt / 'mandatory' / 'decisions.npz') as data:
        assert {'q_A0_0', 'q_A0_5', 'q_RB6_0', 'q_RB6_5', 'q_REF_100', 'rows_A0_0', 'v_RB6_5', 'bin_eff',
                'stab01_K5_top1_same', 'k1_match'} <= set(data.files)
        assert len(data['request_id']) == len(env.stab01()['request_id'])
    assert not (attempt / 'verdict.json').exists()
    late = make_runner(config, env, attempt, local=datetime(2026, 10, 28, 5, 50, tzinfo=KST), lock=FakeLock())
    assert late.advance('D') == runner.EXIT_DEFERRED  # pass C does not fit before 06:00 KST
    assert not (attempt / 'rb12').exists() and receipts(attempt)[-1]['event'] == 'DEFERRED'
    next_lock = FakeLock()
    resumed = make_runner(config, env, attempt, local=datetime(2026, 10, 28, 14, 0, tzinfo=KST), lock=next_lock)
    assert resumed.advance('D') == runner.EXIT_OK
    assert next_lock.events == ['acquire', 'release']
    verdict = json.loads((attempt / 'verdict.json').read_text())['verdict']
    assert verdict['final'] and verdict['row'] == 6 and verdict['label'] == 'NOT_STABLE'
    rb12 = json.loads((attempt / 'rb12' / 'results.json').read_text())
    assert rb12['integrity']['reconstruction']['first_pitch_flips'] == 0
    assert (attempt / 'descriptive' / 'BF48' / 'results.json').exists()
    fhdp = json.loads((attempt / 'descriptive' / 'FH-DP' / 'results.json').read_text())
    assert fhdp['descriptive_only'] and 'dp_minus_rb_agreement' in fhdp['comparison']['rb_RB6_0_vs_RB6_5']
    for folder in ('smoke', 'mandatory', 'rb12', 'descriptive/BF48', 'descriptive/FH-DP'):
        runner.verify_sums(attempt / folder)
    tampered = attempt / 'mandatory' / 'results.json'
    tampered.write_text(tampered.read_text() + ' ')
    with pytest.raises(runner.IntegrityFailure, match='SHA256SUMS'):
        make_runner(config, env, attempt).advance('D')


def test_a_held_lock_starts_nothing(tmp_path):
    attempt = tmp_path / 'attempt-001'
    work = make_runner(registered_config(), SyntheticEnv(), attempt, lock=FakeLock(held=True))
    assert work.advance('smoke') == runner.EXIT_STOPPED
    assert not (attempt / 'smoke').exists() and receipts(attempt)[-1]['event'] == 'LOCK_HELD'


def test_window_deferral_and_deadline_rss_and_budget_stops(tmp_path, monkeypatch):
    early = make_runner(registered_config(), SyntheticEnv(), tmp_path / 'attempt-001',
                        local=datetime(2026, 10, 27, 10, 0, tzinfo=KST))
    assert early.advance('smoke') == runner.EXIT_DEFERRED and not (tmp_path / 'attempt-001' / 'smoke').exists()
    slow = make_runner(registered_config(), SyntheticEnv(), tmp_path / 'attempt-002', tick=1000.)
    assert slow.advance('smoke') == runner.EXIT_STOPPED
    aborted = json.loads((tmp_path / 'attempt-002' / 'smoke' / 'aborted.json').read_text())
    assert aborted['reason'].startswith('DeadlinePassed') and aborted['effect'] == 'NO_FULL_RUN'
    assert not aborted['citable'] and not (tmp_path / 'attempt-002' / 'smoke' / 'results.json').exists()
    runner.verify_sums(tmp_path / 'attempt-002' / 'smoke')
    assert slow.advance('M') == runner.EXIT_STOPPED  # an aborted smoke means no full run
    heavy = make_runner(registered_config(), SyntheticEnv(), tmp_path / 'attempt-003', rss=lambda: 9 << 30)
    assert heavy.advance('smoke') == runner.EXIT_STOPPED
    assert 'RSSExceeded' in json.loads((tmp_path / 'attempt-003' / 'smoke' / 'aborted.json').read_text())['reason']
    config = flow_config(monkeypatch)
    env = SyntheticEnv()
    assert make_runner(config, env, tmp_path / 'attempt-004').advance('smoke') == runner.EXIT_OK
    env.tiny_a0_seed5_budget = True  # the A0 seed-5 row budget is exceeded inside pass M
    budget = make_runner(config, env, tmp_path / 'attempt-004')
    assert budget.advance('M') == runner.EXIT_STOPPED
    aborted = json.loads((tmp_path / 'attempt-004' / 'mandatory' / 'aborted.json').read_text())
    assert aborted['reason'].startswith('BudgetExceeded') and aborted['effect'] == 'EXPERIMENT_ABORTED'
    verdict = json.loads((tmp_path / 'attempt-004' / 'verdict.json').read_text())['verdict']
    assert verdict == {'row': 8, 'label': None, 'flags': ['PASS_M_ABORTED'], 'final': True}
    assert not list((tmp_path / 'attempt-004' / 'mandatory').glob('*.npz'))
    assert budget.advance('D') == runner.EXIT_STOPPED  # nothing after an aborted pass M


def test_pass_m_stops_when_a0_does_not_reproduce_stab01(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    env = SyntheticEnv()
    columns = env.stab01()
    columns['K1_top1_same'] = ~columns['K1_top1_same'].astype(bool)
    work = make_runner(config, env, tmp_path / 'attempt-001')
    assert work.advance('M') == runner.EXIT_STOPPED
    reason = json.loads((tmp_path / 'attempt-001' / 'mandatory' / 'aborted.json').read_text())['reason']
    assert 'does not reproduce the EXP-STAB-01 K1' in reason


def test_confirmation_after_stable_within_cost(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, stable=True)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv()
    work = make_runner(config, env, attempt)
    with pytest.raises(runner.IntegrityFailure, match='no final verdict'):
        work.confirm({'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': 'f' * 64})
    assert work.advance('D') == runner.EXIT_OK
    verdict = json.loads((attempt / 'verdict.json').read_text())['verdict']
    assert verdict['label'] == 'STABLE_WITHIN_COST:RB6' and verdict['row'] == 1
    assert not (attempt / 'rb12').exists() and not (attempt / 'descriptive').exists()  # C and D not required
    wrong = {'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': 'f' * 64}
    assert work.confirm(wrong) == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'confirmation' / 'try-01' / 'aborted.json').read_text())
    assert aborted['effect'] == 'CONFIRMATION_ABORTED' and 'pin' in aborted['reason']
    assert work.confirm_select() == runner.EXIT_OK
    selection = json.loads((attempt / 'confirmation-selection.json').read_text())
    pin = {'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': selection['record']['games_sha256']}
    assert work.confirm(pin) == runner.EXIT_OK  # same-profile retry in try-02
    confirmation = json.loads((attempt / 'confirmation' / 'try-02' / 'results.json').read_text())
    assert confirmation['confirmation_verdict'] == 'CONFIRMED' and confirmation['passing_arm'] == 'RB6'
    assert confirmation['integrity']['reconstruction']['mismatched_decisions'] == 0
    assert 'gap_bins' not in confirmation['comparison']  # not inherited by the confirmation slice
    runner.verify_sums(attempt / 'confirmation' / 'try-02')
    assert work.confirm(pin) == runner.EXIT_OK and not (attempt / 'confirmation' / 'try-03').exists()


def test_cost_exceeded_at_smoke_writes_the_verdict_and_blocks_the_full_run(tmp_path, monkeypatch):
    monkeypatch.setitem(reg.THRESHOLDS, 'smoke_r_b_at_most', 0.5)
    config = registered_config(decision_rules=dict(reg.THRESHOLDS))
    attempt = tmp_path / 'attempt-001'
    assert make_runner(config, SyntheticEnv(), attempt).advance('D') == runner.EXIT_STOPPED
    assert json.loads((attempt / 'verdict.json').read_text())['verdict']['label'] == 'COST_EXCEEDED_AT_SMOKE'
    assert not (attempt / 'mandatory').exists()


def test_a_smoke_structural_failure_blocks_the_full_run(tmp_path, monkeypatch):
    original = pv.RaoBlackwellFirstStep.q_values

    def broken(self, s):
        q, diagnostics = original(self, s)
        q = q.copy()
        q[np.flatnonzero(BC.support(s))[0]] = np.nan
        return q, diagnostics
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, 'q_values', broken)
    attempt = tmp_path / 'attempt-001'
    assert make_runner(registered_config(), SyntheticEnv(), attempt).advance('M') == runner.EXIT_STOPPED
    smoke = json.loads((attempt / 'smoke' / 'results.json').read_text())
    assert not smoke['structural_ok'] and smoke['mechanical_rule']['status'] == 'STRUCTURAL_FAILURE'
    assert smoke['structural']['failure_count'] > 0 and reg.smoke_metric_keys(smoke) == []
    assert not (attempt / 'mandatory').exists() and not (attempt / 'verdict.json').exists()


def test_a_crashed_pass_folder_is_closed_as_an_abort(tmp_path):
    attempt = tmp_path / 'attempt-001'
    (attempt / 'smoke').mkdir(parents=True)
    (attempt / 'smoke' / 'started.json').write_text('{}\n')
    assert make_runner(registered_config(), SyntheticEnv(), attempt).advance('smoke') == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'smoke' / 'aborted.json').read_text())
    assert 'without results' in aborted['reason'] and not aborted['citable']
    runner.verify_sums(attempt / 'smoke')


def test_a_stopped_pass_c_gives_not_stable_with_rb12_aborted_and_keeps_pass_m(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(counts=((3, 2), (2, 2), (3, 1)))
    assert make_runner(config, env, attempt).advance('M') == runner.EXIT_OK
    assert make_runner(config, env, attempt, tick=3000.).advance('C') == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'rb12' / 'aborted.json').read_text())
    assert aborted['effect'] == 'RB12_ABORTED' and aborted['reason'].startswith('DeadlinePassed')
    verdict = json.loads((attempt / 'verdict.json').read_text())['verdict']
    assert verdict == {'row': 7, 'label': 'NOT_STABLE', 'flags': ['RB12_ABORTED'], 'final': True}
    runner.verify_sums(attempt / 'mandatory')  # pass M evidence kept
    assert make_runner(config, env, attempt).advance('D') == runner.EXIT_OK  # pass D may still run
    assert (attempt / 'descriptive' / 'BF48' / 'results.json').exists()
