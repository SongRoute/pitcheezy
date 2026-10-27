"""Synthetic-only tests for EXP-STAB-02 (track A1). No real data, no T7, no 2026, no outcome reads.

Every fixture is built here: a three-action TRAIN BC, frozen 400-draw pools, deterministic row-wise
conditional predictors and WE tables. The pass runner is driven through a synthetic environment whose
runtime is the REAL ``PolicyRuntime`` (pinned BC artifact, support table, hash-chained ledger and its
serialized result rows) around a synthetic candidate wired like ``build_runtime``'s, with a mocked
clock and a fake heavy lock.
"""
import builtins
from collections import Counter
import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
from pitchmdp import planner_variance as pv  # noqa: E402
from pitchmdp import planner_variance_registry as reg  # noqa: E402
from pitchmdp import policy_artifacts as pa  # noqa: E402
from pitchmdp import policy_runtime as prt  # noqa: E402
from pitchmdp.matrix_data import canonical_hash  # noqa: E402
from pitchmdp.planner import OUTCOMES, solve_pa  # noqa: E402
from pitchmdp.rollout_policy import (BCRecord, BudgetExceeded, CategoricalBC, DeliveryPool, JointSimulator,  # noqa: E402
                                     PAState, PastPitch, RolloutImprovement, RowBudget, kl_policy, probabilities,
                                     rollouts)
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
    # pitcher q never threw C in TRAIN: a logged C is a logging-positivity refusal of the real runtime
    records += [BCRecord(PAState(0, 0, 'q', 'R', (), 'ctx'), action, 'train') for action in ('A', 'A', 'B')]
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


def evidence(planner, root):
    """A0 search with its first-step evidence captured during the call (A0Witness)."""
    witness = pv.A0Witness(planner)
    q, diagnostics = planner.q_values(root)
    return q, diagnostics, witness.take(root)


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


def test_same_seed_selected_paths_and_first_outcomes_reproduce_the_registered_planner_and_nest():
    for root in STATES:
        q6, d6, e6 = evidence(mc(6, 5), root)
        q12, d12, e12 = evidence(mc(12, 5), root)
        r6, rd6 = rb(6, 5).q_values(root)
        r12, rd12 = rb(12, 5).q_values(root)
        mask = BC.support(root)
        np.testing.assert_array_equal(rd6['selected_values'][mask], d6['sample_values'][mask])
        np.testing.assert_array_equal(rd12['selected_values'][mask], d12['sample_values'][mask])  # MC12, all 12
        np.testing.assert_array_equal(rd12['selected_values'][mask, :6], d6['sample_values'][mask])  # first 6 = MC6
        np.testing.assert_array_equal(rd6['first_outcome'][mask], e6['first'][mask])  # zero first-pitch flips
        np.testing.assert_array_equal(rd12['first_outcome'][mask], e12['first'][mask])
        np.testing.assert_array_equal(rd12['first_outcome'][mask, :6], e6['first'][mask])
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


def test_the_a0_witness_observes_without_changing_the_search_or_its_wiring():
    for root in STATES:
        plain_q, plain_d = mc(6, 5).q_values(root)
        budget = RowBudget(10**6)
        planner = mc(6, 5, budget)
        sim = planner.simulator
        witness = pv.A0Witness(planner)
        assert (planner.simulator.predictor is sim.predictor and planner.simulator.pool is sim.pool
                and planner.simulator.terminal is sim.terminal and planner.simulator.budget is budget)
        q, d = planner.q_values(root)
        ev = witness.take(root)
        np.testing.assert_array_equal(q, plain_q)
        np.testing.assert_array_equal(d['sample_values'], plain_d['sample_values'])
        np.testing.assert_array_equal(ev['q'], q)
        mask = BC.support(root)
        assert (ev['first'][mask] >= 0).all() and (ev['first'][~mask] == -1).all() and ev['ms'] >= 0
        with pytest.raises(pv.SearchIntegrityError):  # evidence is taken once, for the state searched
            witness.take(root)


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

def test_reconstruction_compares_first_outcomes_and_paths_and_traces_only_differing_paths():
    tally = pv.ReconstructionTally(1e-5, .001)
    for root in STATES:
        _, diagnostics = rb(6, 5).q_values(root)
        q0, _, ev = evidence(mc(6, 5), root)
        mask = BC.support(root)
        check = pv.reconstruction_compare(diagnostics, ev, mask, 6)
        assert not check['mismatch'] and check['first_pitch_flips'] == [] and check['differing_paths'] == []
        tally.add('ok', {'seed 5': check})
    assert tally.finish()['mismatched_decisions'] == 0
    root = STATES[1]
    planner = rb(6, 5)
    _, diagnostics = planner.q_values(root)
    q0, _, ev = evidence(mc(6, 5), root)
    mask = BC.support(root)
    r = int(np.flatnonzero(mask)[0])
    flipped = {**ev, 'first': ev['first'].copy()}
    flipped['first'][r, 2] = (flipped['first'][r, 2] + 1) % 10  # a first-pitch flip that leaves every Q unchanged
    check = pv.reconstruction_compare(diagnostics, flipped, mask, 6)
    assert check['first_pitch_flips'] == [(r, 2)] and check['differing_paths'] == []
    with pytest.raises(pv.SearchIntegrityError, match='first-pitch'):
        pv.ReconstructionTally(1e-5, .001).add('flip', {'seed 5': check})
    i = next(i for i in range(6) if not diagnostics['first_outcome_terminal'][r, i])
    tampered = {**ev, 'samples': ev['samples'].copy(), 'q': ev['q'].copy()}
    tampered['samples'][r, i] += .1
    tampered['q'][r] = np.ascontiguousarray(tampered['samples'][r]).mean()
    check = pv.reconstruction_compare(diagnostics, tampered, mask, 6)
    assert check['differing_paths'] == [(r, i)] and check['margin'] is None
    with pytest.raises(pv.SearchIntegrityError, match='not traced'):  # a differing path needs its traced margin
        pv.ReconstructionTally(1e-5, .001).add('untraced', {'seed 5': check})
    classified = pv.classify_paths(check, diagnostics, planner.trace(root)[1])
    assert len(classified['path_margins']) == 1 and classified['margin'] == classified['path_margins'][0]
    with pytest.raises(pv.SearchIntegrityError, match='margin'):
        pv.ReconstructionTally(1e-5, .001).add('far', {'seed 5': classified})
    trace = planner.trace(root)[1]
    with pytest.raises(pv.SearchIntegrityError, match='reproduce'):
        pv.classify_paths(check, diagnostics, dict(trace, sample_values=trace['sample_values'] + 1e-9))
    with pytest.raises(pv.SearchIntegrityError, match='A0 Q'):
        pv.reconstruction_compare(diagnostics, {**ev, 'q': ev['q'] + 1e-6}, mask, 6)


def test_reconstruction_tally_near_boundary_share_and_far_rules():
    near = {'mismatch': True, 'first_pitch_flips': [], 'differing_paths': [(1, 0)], 'path_margins': [1e-7],
            'margin': 1e-7}
    ok = {'mismatch': False, 'first_pitch_flips': [], 'differing_paths': [], 'path_margins': None, 'margin': None}
    tally = pv.ReconstructionTally(1e-5, .001)
    tally.add('near-boundary', {'seed 0': near})  # a float-rounding flip next to a CDF boundary is tolerated
    for k in range(999):
        tally.add(f'ok-{k}', {'seed 0': ok})
    assert tally.finish()['mismatched_decisions'] == 1  # 1 / 1000 = 0.1%: allowed
    tally.add('second', {'seed 0': near})
    with pytest.raises(pv.SearchIntegrityError, match='exceed'):
        tally.finish()
    far = dict(near, path_margins=[.2], margin=.2)
    with pytest.raises(pv.SearchIntegrityError, match='margin'):
        pv.ReconstructionTally(1e-5, .001).add('far', {'seed 0': far})
    terminal_path = dict(near, path_margins=[np.inf], margin=np.inf)
    with pytest.raises(pv.SearchIntegrityError, match='margin'):
        pv.ReconstructionTally(1e-5, .001).add('terminal', {'seed 0': terminal_path})


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


# ---------------------------------------------------------------- registry: config, gate, provenance, verdict, window

def registered_config(**overrides):
    config = copy.deepcopy(CONFIG)
    config['status'], config['execute'] = reg.STATUS_REGISTERED, True
    config['registration'].update(registered_kst='synthetic test', registered_commit='0' * 40,
                                  design_review_reference='REVIEW-A1-ASTRA-003', code_review_reference='synthetic',
                                  code_sha256={rel: '0' * 64 for rel in reg.NEW_CODE},
                                  digest_manifest_path='/nonexistent/digest-manifest.jsonl')
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
    for field in ('code_review_reference', 'digest_manifest_path'):
        missing = registered_config()
        missing['registration'][field] = None
        with pytest.raises(reg.RegistrationError, match=field):
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


def test_config_digest_ignores_only_the_registration_fields():
    assert reg.config_digest(CONFIG) == reg.config_digest(registered_config())
    changed = registered_config(question='another question')
    assert reg.config_digest(changed) != reg.config_digest(CONFIG)
    obj = copy.deepcopy(CONFIG)
    for dotted in reg.DIGEST_NULLED_PATHS:
        *head, last = dotted.split('.')
        node = obj
        for part in head:
            node = node[part]
        node[last] = None
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    assert reg.config_digest(CONFIG) == hashlib.sha256(text.encode('utf-8')).hexdigest()


def test_the_external_digest_manifest_pins_the_config(tmp_path):
    manifest = tmp_path / 'digest-manifest.jsonl'
    config = registered_config(**{'registration.digest_manifest_path': str(manifest)})
    entry = {'experiment_id': reg.EXPERIMENT_ID, 'kind': 'config_digest', 'digest': reg.config_digest(config),
             'registered_commit': config['registration']['registered_commit'],
             'code_sha256': config['registration']['code_sha256']}
    with pytest.raises(reg.RegistrationError, match='existing file'):
        reg.verify_digest_manifest(config, REPO)
    manifest.write_text(json.dumps(entry) + '\n')
    assert reg.verify_digest_manifest(config, REPO) == manifest.resolve()
    with pytest.raises(reg.RegistrationError, match='digest differs'):
        reg.verify_digest_manifest(registered_config(question='x', **{'registration.digest_manifest_path': str(manifest)}),
                                   REPO)
    manifest.write_text(json.dumps(entry) + '\n' + json.dumps(entry) + '\n')
    with pytest.raises(reg.RegistrationError, match='exactly one'):
        reg.verify_digest_manifest(config, REPO)
    inside = registered_config(**{'registration.digest_manifest_path': str(REPO / 'configs' / 'EXP-STAB-02.yaml')})
    with pytest.raises(reg.RegistrationError, match='outside the repository'):
        reg.verify_digest_manifest(inside, REPO)


def git(repo, *args):
    return subprocess.run(['git', '-c', 'user.name=synthetic', '-c', 'user.email=synthetic@example.invalid',
                           '-c', 'commit.gpgsign=false', *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_the_registered_commit_must_contain_the_reviewed_sources(tmp_path):
    repo = tmp_path / 'repo'
    for rel in (*reg.NEW_CODE, reg.CONFIG_PATH):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, repo / rel)
    git(repo, 'init', '-q')
    git(repo, 'add', '.')
    git(repo, 'commit', '-q', '-m', 'reviewed')
    commit = git(repo, 'rev-parse', 'HEAD')
    pins = {rel: reg.file_sha256(repo / rel) for rel in reg.NEW_CODE}
    config = registered_config(**{'registration.registered_commit': commit, 'registration.code_sha256': pins})
    assert reg.verify_registered_commit(config, repo)['registered_commit'] == commit
    assert runner.git_state(repo)[0] == commit
    with pytest.raises(reg.RegistrationError, match='reviewed sources'):
        reg.verify_registered_commit(registered_config(**{'registration.registered_commit': commit}), repo)
    with pytest.raises(reg.RegistrationError, match='beyond the registration fields'):
        reg.verify_registered_commit(registered_config(question='changed', **{
            'registration.registered_commit': commit, 'registration.code_sha256': pins}), repo)
    with pytest.raises(reg.RegistrationError, match='Git provenance'):
        reg.verify_registered_commit(registered_config(**{'registration.registered_commit': 'f' * 40,
                                                          'registration.code_sha256': pins}), repo)
    branch = git(repo, 'rev-parse', '--abbrev-ref', 'HEAD')
    for moving in ('HEAD', branch, commit[:12], commit.upper()):  # symbolic, short or non-canonical revisions
        symbolic = registered_config(**{'registration.registered_commit': moving, 'registration.code_sha256': pins})
        with pytest.raises(reg.RegistrationError, match='full 40-hex'):
            reg.verify_registered_commit(symbolic, repo)
        with pytest.raises(reg.RegistrationError, match='full 40-hex'):
            reg.registration_gate(symbolic)
    plain_dir = tmp_path / 'not-a-repo'
    plain_dir.mkdir()
    with pytest.raises(reg.RegistrationError, match='Git provenance'):
        runner.git_state(plain_dir)


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
    assert reg.window_id(at(14)) == reg.window_id(at(3, day=28)) != reg.window_id(at(14, day=28))


def test_run_conditions_of_the_optional_arms():
    assert reg.rb12_required(2.0, False) and not reg.rb12_required(2.01, False) and not reg.rb12_required(1.0, True)
    assert reg.descriptive_allowed(False, True, 100., 200.) and reg.descriptive_allowed(True, False, 100., 200.)
    assert not reg.descriptive_allowed(True, True, 100., 200.) and not reg.descriptive_allowed(False, False, 300., 200.)


def test_the_smoke_sink_refuses_anything_but_allowlisted_structure():
    record = {'r_b': 2.1, 'cost': {'rows_by_arm_seed': {'RB6 seed 0': 10}}, 'mechanical_rule': {'status': 'FULL_RUN_ALLOWED'}}
    assert reg.smoke_record_problems(record) == []
    assert reg.smoke_record_problems({**record, 'q_planning': [None, .5]})
    assert reg.smoke_record_problems({**record, 'cost': {'rows_by_arm_seed': {}, 'mean_tv': .1}})
    assert reg.smoke_record_problems({**record, 'statuses': {'SUPPORTED': 3}, 'identity': {'law': [.5, .5]}})
    assert reg.smoke_record_problems({**record, 'data_version': {'scores': [.25, .75]}})  # a float vector


# ---------------------------------------------------------------- the pass runner on a synthetic environment

PROV = {'train_start': '2023-05-15', 'train_end': '2025-04-30', 'dates': 'declared_unverified',
        'train_rows': int(sum(BC.league.values())), 'source_ids': {'synthetic': 'a' * 64}, 'config_sha256': 'b' * 64,
        'code_commit': 'f83caea2d8540992a446b168b498f910111478b0', 'data_version': 'synthetic-v1'}
OUTCOME_PLAN = ('ball', 'strike', 'foul', 'ball')  # decisions at 0-0, 1-0, 1-1 and 1-2


@dataclass
class PlannedPA:
    game: int
    pa: int
    actions: tuple
    pitcher: str = 'p'


class SyntheticEnv:
    """The runner's environment contract on synthetic data around the REAL PolicyRuntime: pinned BC artifact and
    support table files, a hash-chained ledger and serialized result rows; only the candidate (built by
    build_runtime from bound components in a real run) is synthetic, wired exactly like build_runtime's."""
    MASK = np.array([True, True, False])  # C off the intervention support: OUTSIDE_POLICY_SUPPORT rows

    def __init__(self, root, games=(101, 102, 103, 104), predict=predictor, refusals=False):
        self.root = Path(tempfile.mkdtemp(dir=root, prefix='synthetic-artifacts-'))  # frozen files are write-once
        self.art = pa.save_train_bc(BC, self.root / 'bc.json', PROV)
        rows = [(pitcher, 'R', self.MASK & self.art.bc.support(PAState(0, 0, pitcher, 'R', (), 'ctx')))
                for pitcher in ('p', 'q')]
        _, file_sha = pa.save_support_table(self.art, rows, self.root / 'support.json')
        self.support, self.support_identity = pa.load_support_table(self.root / 'support.json', file_sha, self.art)
        self.reference = prt.MaskedReference(self.art.bc, self.support)
        self.predict, self.runtime_sha = predict, 'replay-without-runtime'
        self.games = list(games)
        self.games_record = {'games_sha256': 'synthetic', 'selected': len(self.games)}
        order = runner.subsample_order(self.games)
        self.subsamples = {'BF48': order[:reg.SUBSAMPLES['BF48']], 'FH-DP': order[:reg.SUBSAMPLES['FH-DP']]}
        self.fhdp_smoke = order[0]
        self.confirmation = [201, 202]
        self.confirmation_record = {'games_sha256': canonical_hash(sorted(self.confirmation)), 'selected': 2}
        self.plans = {g: [PlannedPA(g, k, tuple(ACTIONS[(g + k + i) % 3] for i in range(len(OUTCOME_PLAN))))
                          for k in range(2)] for g in self.games + self.confirmation}
        if refusals:  # real runtime refusals: logging positivity (q logs C) and an unknown pitcher, then MID_PA rows
            self.plans[self.games[0]] += [PlannedPA(self.games[0], 2, ('C', 'A', 'B', 'A'), 'q'),
                                          PlannedPA(self.games[0], 3, ('A', 'B', 'A', 'B'), 'z')]
        self.prepared, self.prepare_hook, self.verified, self._stab01 = [], None, 0, None
        self.tiny_a0_seed5_budget = False

    def prepare(self, scope):
        self.prepared.append(scope)
        if self.prepare_hook is not None:
            self.prepare_hook()

    def smoke_games(self):
        return self.games[:reg.GAMES['smoke_games']]

    def blocks(self, games):
        for g in sorted(set(games)):
            for plan in self.plans[g]:
                yield g, plan

    def requests(self, plan):
        """Consistent PA: 0-0 start, each history token is the previous logged pitch and its outcome."""
        out, history, (b, s) = [], (), (0, 0)
        for i, (action, outcome) in enumerate(zip(plan.actions, OUTCOME_PLAN)):
            st = PAState(b, s, plan.pitcher, 'R', history, f'{plan.game}-{plan.pa}')
            out.append(prt.DecisionRequest(f'{plan.game}:{plan.pa}:{i + 1}', f'{plan.game}:{plan.pa}', i, st, action,
                                           self.runtime_sha, 'R'))
            history = history + (PastPitch(action, (.1 + .2 * i + .01 * plan.pa,), outcome, b, s),)
            b, s = prt.next_count(b, s, outcome)
        return out

    def build_runtime(self, ledger_path, budget, *, in_memory=False):
        reference = self.reference
        improvement = RolloutImprovement(reference, simulator(budget, self.predict), cutoff, samples=6, pitch_cap=12,
                                         seed=0)

        def candidate(s):  # build_runtime's candidate (policy_runtime.py), here over synthetic components
            mask = reference.support(s)
            q, diagnostics = improvement.q_values(s)
            p = kl_policy(q, reference.probabilities(s), mask, reg.TAU)
            off = np.where(mask, 0., np.nan)
            return p, {'q': q + off, 'mc_se': diagnostics['mc_se'] + off, 'q_planning': q + off,
                       'planning_diff_se': prt.paired_difference_se(diagnostics['sample_values'], mask) + off,
                       'source': 'planning_reuse'}
        with runner.memory_ledger() if in_memory else contextlib.nullcontext():
            runtime = prt.PolicyRuntime(self.art, self.support, self.support_identity, ledger_path, candidate=candidate,
                                        candidate_identity={'name': 'synthetic P3 (build_runtime wiring)'})
        assert isinstance(runtime.ledger, runner.MemoryLedger) == in_memory and prt.Ledger is not runner.MemoryLedger
        runtime.reference, runtime.improvement = reference, improvement
        self.runtime_sha = runtime.sha256
        return runtime

    def planner(self, kind, samples, seed, budget):
        if self.tiny_a0_seed5_budget and kind == 'MC' and seed == 5:
            budget = RowBudget(50, seed_count=5)
        sim = simulator(budget, self.predict)
        if kind == 'MC':
            return RolloutImprovement(self.reference, sim, cutoff, samples=samples, pitch_cap=12, seed=seed)
        if kind == 'RB':
            return pv.RaoBlackwellFirstStep(self.reference, sim, cutoff, samples=samples, pitch_cap=12, seed=seed)
        return pv.FrozenHistoryDP(self.reference, sim, cutoff)

    def check_request(self, request, mask):
        assert request.state.pitcher == 'p'

    def verify(self, planners, runtime=None):
        for planner in (*planners, *((runtime.improvement,) if runtime is not None else ())):
            sim = planner.simulator
            assert sim.predictor is self.predict and sim.pool is pool and sim.terminal is terminal
            assert planner.cutoff is cutoff
        self.verified += 1

    def stab01(self):
        if self._stab01 is None:
            stab = runner._stab()
            sets = {'A': [0, 1, 2, 3, 4], 'B': [5, 6, 7, 8, 9], 'K': [1, 3, 5]}
            planners = {s: RolloutImprovement(self.reference, simulator(predict=self.predict), cutoff, samples=6,
                                              pitch_cap=12, seed=s) for s in range(10)}
            columns = {k: [] for k in reg.STAB01_COLUMNS}
            with runner.memory_ledger():  # statuses from a scratch real runtime: only evaluated decisions count
                scratch = prt.PolicyRuntime(self.art, self.support, self.support_identity, None,
                                            candidate=self.reference.probabilities, candidate_identity={'name': 'scratch'})
            for g, plan in self.blocks(self.games):
                runtime_sha, self.runtime_sha = self.runtime_sha, scratch.sha256
                requests, self.runtime_sha = self.requests(plan), runtime_sha
                for request in requests:
                    if scratch.submit(request)['status'] not in prt.EVALUATED:
                        continue
                    s = request.state
                    mask, reference = self.reference.support(s), self.reference.probabilities(s)
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

    def jump(self, seconds):
        self.t += timedelta(seconds=seconds)
        self.mono += seconds


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


def identity(config=None):
    return {'config_sha256': 'synthetic-config', 'config_digest': reg.config_digest(config or registered_config()),
            'source_hashes': {'pitchmdp/planner_variance.py': 'synthetic'}, 'pins_sha256': reg.pins_digest()}


def make_runner(config, env, attempt, local=datetime(2026, 10, 27, 14, 0, tzinfo=KST), tick=0., lock=None, rss=None,
                manifest=None, clock=None):
    return runner.PassRunner(config, env, attempt, identity=identity(config), clock=clock or FakeClock(local, tick),
                             lock=lock or FakeLock(), rss=rss or (lambda: 1 << 30), log=lambda *a: None,
                             manifest=manifest)


def receipts(attempt):
    return [json.loads(line) for line in (attempt / 'receipts.jsonl').read_text().splitlines()]


def flow_config(monkeypatch, *, stable=False, rb12_cap=10.0, **integrity):
    """Registry pins moved for a synthetic flow test; the config is rewritten to agree with them."""
    rules = dict(reg.THRESHOLDS)
    if stable:
        rules.update(top1_lower_at_least=-1.0, mean_tv_upper_at_most=10.0, r_rows_at_most=100.0)
    else:
        rules.update(top1_lower_at_least=1.01)  # never STABLE on any data
    rules.update(rb12_smoke_r_b_at_most=rb12_cap)
    for key, value in rules.items():
        monkeypatch.setitem(reg.THRESHOLDS, key, value)
    for key, value in integrity.items():
        monkeypatch.setitem(reg.INTEGRITY, key, value)
    return registered_config(decision_rules=dict(reg.THRESHOLDS), integrity=dict(reg.INTEGRITY))


def test_runtime_rows_are_read_from_the_real_policy_runtime_serialization(tmp_path):
    env = SyntheticEnv(tmp_path)
    work = make_runner(registered_config(), env, tmp_path / 'attempt-001')
    budget = RowBudget(10**6, seed_count=5)
    runtime = env.build_runtime(tmp_path / 'ledger.jsonl', budget)
    witness = pv.A0Witness(runtime.improvement)
    watch = runner.Watch(work.clock, work.clock.now() + timedelta(hours=1), 3600, lambda: 1 << 30)
    statuses = Counter()
    decisions = list(work.runtime_decisions([101], runtime, witness, budget, watch, statuses))
    rows = [json.loads(line) for line in (tmp_path / 'ledger.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r.get('kind') == 'decision']
    assert len(decisions) == len(rows) == 2 * len(OUTCOME_PLAN)
    assert set(statuses) <= set(prt.EVALUATED) and 'OUTSIDE_POLICY_SUPPORT' in statuses
    for decision, row in zip(decisions, rows):
        result = row['result']
        assert 'q_planning_diff_se' in result and 'planning_diff_se' not in result  # the serialized contract
        np.testing.assert_array_equal(decision.runtime['q'], runner.vector(result['q_planning']))
        np.testing.assert_array_equal(decision.runtime['diff_se'], runner.vector(result['q_planning_diff_se']))
        assert decision.runtime['rows'] > 0 and decision.runtime['first'].shape == (3, 6)
        assert decision.runtime['submit_ms'] >= decision.runtime['ms'] >= 0


def walk_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file())


def watch_writes(monkeypatch):
    """Every path opened for writing through builtins.open, Path.open or os.open while active."""
    opened = []
    real_open, real_path_open, real_os_open = builtins.open, Path.open, os.open

    def spy_open(file, mode='r', *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and any(c in mode for c in 'wax+'):
            opened.append(Path(file).resolve())
        return real_open(file, mode, *args, **kwargs)

    def spy_path_open(self, mode='r', *args, **kwargs):
        if any(c in mode for c in 'wax+'):
            opened.append(self.resolve())
        return real_path_open(self, mode, *args, **kwargs)

    def spy_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT):
            opened.append(Path(path).resolve())
        return real_os_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', spy_open)
    monkeypatch.setattr(Path, 'open', spy_path_open)
    monkeypatch.setattr(os, 'open', spy_os_open)
    return opened


def test_smoke_persists_only_allowlisted_structure_in_every_file(tmp_path, monkeypatch):
    attempt, manifest = tmp_path / 'attempt-001', tmp_path / 'digest-manifest.jsonl'
    env = SyntheticEnv(tmp_path)
    lock = FakeLock()
    work = make_runner(registered_config(), env, attempt, lock=lock, manifest=manifest)
    opened = watch_writes(monkeypatch)
    assert work.advance('smoke') == runner.EXIT_OK
    monkeypatch.undo()
    allowed = {attempt / 'receipts.jsonl', manifest, attempt / 'smoke' / 'started.json.tmp',
               attempt / 'smoke' / 'results.json.tmp', attempt / 'smoke' / 'SHA256SUMS'}
    assert opened and set(opened) <= {p.resolve() for p in allowed}, sorted(set(opened))  # no ledger, ever
    assert lock.events == ['acquire', 'release'] and env.prepared == ['smoke']
    files = walk_files(attempt) + [manifest]
    assert {p.relative_to(tmp_path).as_posix() for p in files} == {
        'attempt-001/receipts.jsonl', 'attempt-001/smoke/started.json', 'attempt-001/smoke/results.json',
        'attempt-001/smoke/SHA256SUMS', 'digest-manifest.jsonl'}
    for path in files:
        text = path.read_text()
        assert 'ledger' not in path.name and 'q_planning' not in text and 'q_reference' not in text
        records = ([json.loads(line) for line in text.splitlines()] if path.suffix == '.jsonl' else
                   [json.loads(text)] if path.suffix == '.json' else [])
        for record in records:
            assert reg.smoke_metric_keys(record) == [] and reg.float_vectors(record) == [], path
    results = json.loads((attempt / 'smoke' / 'results.json').read_text())
    assert reg.smoke_record_problems(results) == []
    assert results['structural_ok'] and results['structural']['failure_count'] == 0
    assert results['runtime_ledger']['persisted'] is False and results['runtime_ledger']['rows'] > 0
    assert 'memory' in results['runtime_ledger']['location']
    cost = results['cost']
    rows = cost['rows_by_arm_seed']
    assert results['r_b'] == pytest.approx(np.mean([rows[f'RB6 seed {s}'] / rows[f'A0 seed {s}'] for s in (0, 5)]))
    assert {'RB12 seed 0', 'BF48 seed 5', 'REF-RB6 seed 100', 'RB6-rerun seed 0', 'FH-DP'} <= set(rows)
    assert 'A0 seed 0' in cost['planner_ms_per_decision_median'] and cost['runtime_submit_seconds'] > 0
    assert 'peak_rss_bytes_process_lifetime_pre_sealing' in results and results['decisions'] == 3 * 2 * len(OUTCOME_PLAN)
    runner.verify_sums(attempt / 'smoke')
    assert [r['event'] for r in receipts(attempt)] == ['STARTED', 'COMPLETED']
    assert len(manifest.read_text().splitlines()) == 2  # receipts mirrored to the external digest manifest
    assert work.advance('smoke') == runner.EXIT_OK  # nothing left: no rerun into a finished folder


def test_mandatory_rb12_deferral_lock_release_and_continuation(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt).advance('M') == runner.EXIT_OK
    mandatory = json.loads((attempt / 'mandatory' / 'results.json').read_text())
    assert mandatory['rb12_required'] and mandatory['verdict'] == {'row': None, 'label': None,
                                                                   'flags': ['PENDING_PASS_C'], 'final': False}
    integrity = mandatory['integrity']
    assert integrity['stab01_k1_match_rate'] == 1.0 and integrity['replay']['max_abs_q_difference'] == 0.0
    assert integrity['reconstruction']['mismatched_decisions'] == 0 and integrity['reconstruction']['first_pitch_flips'] == 0
    secondary = mandatory['comparison']['secondary']
    assert set(secondary['gap_bins']['effective_gap']) == {'support_1', '[0.0, 0.0025)', '[0.0025, 0.005)',
                                                            '[0.005, 0.01)', '[0.01, 0.02)', '[0.02, 0.05)', '[0.05, inf)'}
    assert 'rb_improves_over_baseline' in mandatory['comparison']['secondary_flags']
    assert 'runtime_submit_ms_mean' in secondary['resources']['A0_0']
    with np.load(attempt / 'mandatory' / 'decisions.npz') as data:
        assert {'q_A0_0', 'q_A0_5', 'q_RB6_0', 'q_RB6_5', 'q_REF_100', 'rows_A0_0', 'v_RB6_5', 'bin_eff', 'k1_match',
                'stab01_K5_top1_same', 'a0_first_0', 'a0_samples_0', 'a0_first_5', 'a0_samples_5', 'rb_first_RB6_0',
                'submit_ms_A0_0'} <= set(data.files)
        np.testing.assert_array_equal(data['rb_first_RB6_5'], data['a0_first_5'])  # zero first-pitch flips
        assert len(data['request_id']) == len(env.stab01()['request_id'])
    assert (attempt / 'mandatory' / 'ledger-dev-0.jsonl').exists() and not (attempt / 'verdict.json').exists()
    late = make_runner(config, env, attempt, local=datetime(2026, 10, 28, 5, 50, tzinfo=KST))
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


def perturb_rb6_seed5(monkeypatch, margin):
    """Make one RB6 seed-5 path differ from A0 (same first outcome, different value) in the untraced and the traced
    search alike, and give that path the traced boundary margin ``margin``."""
    original = pv.RaoBlackwellFirstStep._search
    target = {}

    def search(self, s, first, continuation, *, count):
        q, d = original(self, s, first, continuation, count=count)
        if self.seed == 5 and self.samples == 6 and target.setdefault('state', s) == s:
            r = int(np.flatnonzero(np.isfinite(q))[0])
            i = next(i for i in range(6) if not d['first_outcome_terminal'][r, i])
            d['selected_values'] = d['selected_values'].copy()
            d['selected_values'][r, i] += .01
            if 'selected_margin' in d:
                d['selected_margin'] = d['selected_margin'].copy()
                d['selected_margin'][r, i] = margin
        return q, d
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, '_search', search)


def test_the_runner_traces_a_differing_path_and_accepts_a_near_boundary_margin(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, reconstruction_mismatch_share_max=1.0)
    perturb_rb6_seed5(monkeypatch, 1e-7)
    traced = []
    trace = pv.RaoBlackwellFirstStep.trace
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, 'trace', lambda self, s: (traced.append(self.seed), trace(self, s))[1])
    attempt = tmp_path / 'attempt-001'
    assert make_runner(config, SyntheticEnv(tmp_path), attempt).advance('M') == runner.EXIT_OK
    reconstruction = json.loads((attempt / 'mandatory' / 'results.json').read_text())['integrity']['reconstruction']
    assert reconstruction['mismatched_decisions'] == 1 and reconstruction['max_margin'] == pytest.approx(1e-7)
    assert traced == [5]  # only the differing decision was traced, only for the seed whose path differs


def test_the_runner_rejects_a_far_boundary_margin_and_never_traces_matching_paths(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    perturb_rb6_seed5(monkeypatch, .2)
    attempt = tmp_path / 'attempt-001'
    assert make_runner(config, SyntheticEnv(tmp_path), attempt).advance('M') == runner.EXIT_STOPPED
    reason = json.loads((attempt / 'mandatory' / 'aborted.json').read_text())['reason']
    assert 'margin' in reason and 'SearchIntegrityError' in reason


def test_the_runner_stops_on_a_first_pitch_flip_that_leaves_every_q_unchanged(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    original = pv.A0Witness.take

    def take(self, s):
        ev = original(self, s)
        if self.improvement.seed == 5:
            r = int(np.flatnonzero(ev['first'][:, 0] >= 0)[0])
            ev['first'][r, 0] = (ev['first'][r, 0] + 1) % 10
        return ev
    monkeypatch.setattr(pv.A0Witness, 'take', take)
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, 'trace', lambda self, s: pytest.fail('a flip is never traced'))
    attempt = tmp_path / 'attempt-001'
    assert make_runner(config, SyntheticEnv(tmp_path), attempt).advance('M') == runner.EXIT_STOPPED
    assert 'first-pitch' in json.loads((attempt / 'mandatory' / 'aborted.json').read_text())['reason']


def test_a_held_lock_starts_nothing(tmp_path):
    attempt = tmp_path / 'attempt-001'
    work = make_runner(registered_config(), SyntheticEnv(tmp_path), attempt, lock=FakeLock(held=True))
    assert work.advance('smoke') == runner.EXIT_STOPPED
    assert not (attempt / 'smoke').exists() and receipts(attempt)[-1]['event'] == 'LOCK_HELD'


def test_window_deferral_deadline_rss_budget_and_preparation_stops(tmp_path, monkeypatch):
    early = make_runner(registered_config(), SyntheticEnv(tmp_path), tmp_path / 'attempt-001',
                        local=datetime(2026, 10, 27, 10, 0, tzinfo=KST))
    assert early.advance('smoke') == runner.EXIT_DEFERRED and not (tmp_path / 'attempt-001' / 'smoke').exists()
    slow = make_runner(registered_config(), SyntheticEnv(tmp_path), tmp_path / 'attempt-002', tick=1000.)
    assert slow.advance('smoke') == runner.EXIT_STOPPED
    aborted = json.loads((tmp_path / 'attempt-002' / 'smoke' / 'aborted.json').read_text())
    assert aborted['reason'].startswith('DeadlinePassed') and aborted['effect'] == 'NO_FULL_RUN'
    assert not aborted['citable'] and not (tmp_path / 'attempt-002' / 'smoke' / 'results.json').exists()
    runner.verify_sums(tmp_path / 'attempt-002' / 'smoke')
    assert slow.advance('M') == runner.EXIT_STOPPED  # an aborted smoke means no full run
    heavy = make_runner(registered_config(), SyntheticEnv(tmp_path), tmp_path / 'attempt-003', rss=lambda: 9 << 30)
    assert heavy.advance('smoke') == runner.EXIT_STOPPED
    assert 'RSSExceeded' in json.loads((tmp_path / 'attempt-003' / 'smoke' / 'aborted.json').read_text())['reason']
    broken = SyntheticEnv(tmp_path)
    broken.prepare_hook = lambda: runner.require(False, 'processed sha differs from its pin')
    assert make_runner(registered_config(), broken, tmp_path / 'attempt-005').advance('smoke') == runner.EXIT_STOPPED
    reason = json.loads((tmp_path / 'attempt-005' / 'smoke' / 'aborted.json').read_text())['reason']
    assert 'processed sha differs' in reason  # a preparation failure leaves its abort record
    config = flow_config(monkeypatch)
    env = SyntheticEnv(tmp_path)
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


def test_finalization_overruns_abort_and_preparation_time_counts(tmp_path, monkeypatch):
    clock = FakeClock(datetime(2026, 10, 27, 14, 0, tzinfo=KST))
    env = SyntheticEnv(tmp_path)
    env.prepare_hook = lambda: clock.jump(120)
    work = make_runner(registered_config(), env, tmp_path / 'attempt-001', clock=clock)
    assert work.advance('smoke') == runner.EXIT_OK
    results = json.loads((tmp_path / 'attempt-001' / 'smoke' / 'results.json').read_text())
    assert results['preparation_seconds'] >= 120 and results['seconds_pre_sealing'] >= 120
    original = runner.PassRunner.pass_smoke
    late = FakeClock(datetime(2026, 10, 27, 14, 0, tzinfo=KST))
    monkeypatch.setattr(runner.PassRunner, 'pass_smoke',
                        lambda self, out, watch, arm=None: (original(self, out, watch, arm), late.jump(4000))[0])
    over = make_runner(registered_config(), SyntheticEnv(tmp_path), tmp_path / 'attempt-002', clock=late)
    assert over.advance('smoke') == runner.EXIT_STOPPED
    reason = json.loads((tmp_path / 'attempt-002' / 'smoke' / 'aborted.json').read_text())['reason']
    assert reason.startswith('DeadlinePassed') and not (tmp_path / 'attempt-002' / 'smoke' / 'results.json').exists()
    calls = []

    def rss():
        calls.append(1)
        return 9 << 30 if getattr(rss, 'finished', False) else 1 << 30
    monkeypatch.setattr(runner.PassRunner, 'pass_smoke',
                        lambda self, out, watch, arm=None: (original(self, out, watch, arm), setattr(rss, 'finished', True))[0])
    heavy = make_runner(registered_config(), SyntheticEnv(tmp_path), tmp_path / 'attempt-003', rss=rss)
    assert heavy.advance('smoke') == runner.EXIT_STOPPED
    reason = json.loads((tmp_path / 'attempt-003' / 'smoke' / 'aborted.json').read_text())['reason']
    assert reason.startswith('RSSExceeded')


def test_an_overrun_while_sealing_aborts_and_no_completion_marker_is_written(tmp_path, monkeypatch):
    clock = FakeClock(datetime(2026, 10, 27, 14, 0, tzinfo=KST))
    seal = runner.write_sums

    def slow_seal(folder):
        clock.jump(4000)  # the cap (3,600 s) and the deadline pass while SHA256SUMS is written
        return seal(folder)
    monkeypatch.setattr(runner, 'write_sums', slow_seal)
    attempt = tmp_path / 'attempt-001'
    assert make_runner(registered_config(), SyntheticEnv(tmp_path), attempt, clock=clock).advance('smoke') == runner.EXIT_STOPPED
    reason = json.loads((attempt / 'smoke' / 'aborted.json').read_text())['reason']
    assert 'final check' in reason and not (attempt / 'smoke' / 'results.json').exists()
    assert [r['event'] for r in receipts(attempt)] == ['STARTED', 'ABORTED']  # never COMPLETED


def test_the_fh_dp_arm_cap_is_enforced(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt).advance('C') == runner.EXIT_OK
    original = runner.PassRunner.pass_descriptive
    clock = FakeClock(datetime(2026, 10, 27, 16, 0, tzinfo=KST))

    def slow_fhdp(self, out, watch, arm):
        if arm == 'FH-DP':
            clock.tick = 2500.
        return original(self, out, watch, arm)
    monkeypatch.setattr(runner.PassRunner, 'pass_descriptive', slow_fhdp)
    assert make_runner(config, env, attempt, clock=clock).advance('D') == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'descriptive' / 'FH-DP' / 'aborted.json').read_text())
    assert aborted['effect'] == 'ARM_DISCARDED' and aborted['reason'].split(':')[0] in ('DeadlinePassed', 'CapExceeded')
    assert (attempt / 'descriptive' / 'BF48' / 'results.json').exists()  # the other arm is kept


def test_pass_m_stops_when_a0_does_not_reproduce_stab01(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    env = SyntheticEnv(tmp_path)
    columns = env.stab01()
    columns['K1_top1_same'] = ~columns['K1_top1_same'].astype(bool)
    work = make_runner(config, env, tmp_path / 'attempt-001')
    assert work.advance('M') == runner.EXIT_STOPPED
    reason = json.loads((tmp_path / 'attempt-001' / 'mandatory' / 'aborted.json').read_text())['reason']
    assert 'does not reproduce the EXP-STAB-01 K1' in reason


def test_a_substituted_verdict_is_refused_on_resume_and_before_confirmation(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, rb12_cap=0.5)  # RB12 not required: the verdict is final after pass M
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt).advance('M') == runner.EXIT_OK
    path = attempt / 'verdict.json'
    assert json.loads(path.read_text())['verdict']['label'] == 'NOT_STABLE'
    forged = json.loads(path.read_text())
    forged['verdict'].update(row=4, label='STABLE_WITHIN_COST:RB12')
    path.write_text(json.dumps(forged))
    with pytest.raises(runner.IntegrityFailure, match='recomputed'):
        make_runner(config, env, attempt).advance('D')
    with pytest.raises(runner.IntegrityFailure, match='recomputed'):
        make_runner(config, env, attempt).confirm({'experiment_id': reg.EXPERIMENT_ID,
                                                   'confirmation_games_sha256': env.confirmation_record['games_sha256']})


def test_confirmation_after_stable_within_cost_with_a_retry_in_the_next_window(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, stable=True)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    work = make_runner(config, env, attempt)
    with pytest.raises(runner.IntegrityFailure, match='no final verdict'):
        work.confirm({'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': 'f' * 64})
    assert work.advance('D') == runner.EXIT_OK
    verdict = json.loads((attempt / 'verdict.json').read_text())['verdict']
    assert verdict['label'] == 'STABLE_WITHIN_COST:RB6' and verdict['row'] == 1
    assert not (attempt / 'rb12').exists() and not (attempt / 'descriptive').exists()  # C and D not required
    with pytest.raises(runner.IntegrityFailure, match='published selection'):  # nothing published yet
        work.confirm({'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': 'f' * 64})
    assert work.confirm_select() == runner.EXIT_OK
    selection = json.loads((attempt / runner.SELECTION_FILE).read_text())
    pin = {'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': selection['record']['games_sha256']}
    with pytest.raises(runner.IntegrityFailure, match='published selection'):  # a pin that is not the publication
        work.confirm({'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': 'f' * 64})
    slow = make_runner(config, env, attempt, tick=3000.)
    assert slow.confirm(pin) == runner.EXIT_STOPPED  # the try overruns its window: CONFIRMATION_ABORTED
    aborted = json.loads((attempt / 'confirmation' / 'try-01' / 'aborted.json').read_text())
    assert aborted['effect'] == 'CONFIRMATION_ABORTED' and not aborted['citable']
    assert work.confirm(pin) == runner.EXIT_DEFERRED  # same window as the aborted try
    assert not (attempt / 'confirmation' / 'try-02').exists()
    tomorrow = make_runner(config, env, attempt, local=datetime(2026, 10, 28, 14, 0, tzinfo=KST))
    assert tomorrow.confirm(pin) == runner.EXIT_OK  # same-profile retry in the next window
    confirmation = json.loads((attempt / 'confirmation' / 'try-02' / 'results.json').read_text())
    assert confirmation['confirmation_verdict'] == 'CONFIRMED' and confirmation['passing_arm'] == 'RB6'
    assert confirmation['integrity']['reconstruction']['mismatched_decisions'] == 0
    assert 'gap_bins' not in confirmation['comparison']  # not inherited by the confirmation slice
    with np.load(attempt / 'confirmation' / 'try-02' / 'decisions.npz') as data:
        assert {'a0_first_0', 'a0_samples_0', 'a0_first_5', 'a0_samples_5'} <= set(data.files)
    runner.verify_sums(attempt / 'confirmation' / 'try-02')
    assert tomorrow.confirm(pin) == runner.EXIT_OK and not (attempt / 'confirmation' / 'try-03').exists()


def test_an_interrupted_confirmation_start_is_recovered_and_retried_in_a_later_window(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, stable=True)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt).advance('D') == runner.EXIT_OK
    assert make_runner(config, env, attempt).confirm_select() == runner.EXIT_OK
    interrupted = attempt / 'confirmation' / 'try-01'
    interrupted.mkdir(parents=True)  # interrupted inside write_json: only the unfinished staging file exists
    (interrupted / 'started.json.tmp').write_text('{"experiment_id": "EXP-STAB-')
    pin = {'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': env.confirmation_record['games_sha256']}
    work = make_runner(config, env, attempt)
    assert work.confirm(pin) == runner.EXIT_DEFERRED  # recovered as an abort; its retry waits for a later window
    recovered = json.loads((attempt / 'confirmation' / 'try-01' / 'aborted.json').read_text())
    assert recovered['window_id'] == reg.window_id(work.clock.now()) and not recovered['citable']
    assert not (interrupted / 'started.json.tmp').exists()
    runner.verify_sums(interrupted)
    assert work.confirm(pin) == runner.EXIT_DEFERRED and not (attempt / 'confirmation' / 'try-02').exists()
    later = make_runner(config, env, attempt, local=datetime(2026, 10, 28, 14, 0, tzinfo=KST))
    assert later.confirm(pin) == runner.EXIT_OK
    assert json.loads((attempt / 'confirmation' / 'try-02' / 'results.json').read_text())['confirmation_verdict'] == 'CONFIRMED'


def test_an_unsuccessful_smoke_stops_at_once_and_keeps_its_evidence(tmp_path, monkeypatch):
    original = pv.RaoBlackwellFirstStep.q_values

    def broken(self, s):
        q, diagnostics = original(self, s)
        q = q.copy()
        q[np.flatnonzero(np.isfinite(q))[0]] = np.nan
        return q, diagnostics
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, 'q_values', broken)
    attempt = tmp_path / 'attempt-001'
    assert make_runner(registered_config(), SyntheticEnv(tmp_path), attempt).advance('smoke') == runner.EXIT_STOPPED
    smoke, stop = (json.loads((attempt / 'smoke' / name).read_text()) for name in ('results.json', 'aborted.json'))
    assert not smoke['structural_ok'] and smoke['structural']['failure_count'] > 0
    assert stop['smoke_status'] == 'STRUCTURAL_FAILURE' and stop['effect'] == 'NO_FULL_RUN'
    assert receipts(attempt)[-1]['event'] == 'ABORTED' and not (attempt / 'verdict.json').exists()
    runner.verify_sums(attempt / 'smoke')
    monkeypatch.setattr(pv.RaoBlackwellFirstStep, 'q_values', original)
    monkeypatch.setitem(reg.THRESHOLDS, 'smoke_r_b_at_most', 0.5)
    config = registered_config(decision_rules=dict(reg.THRESHOLDS))
    second = tmp_path / 'attempt-002'
    assert make_runner(config, SyntheticEnv(tmp_path), second).advance('smoke') == runner.EXIT_STOPPED
    assert json.loads((second / 'smoke' / 'aborted.json').read_text())['smoke_status'] == 'COST_EXCEEDED_AT_SMOKE'
    assert json.loads((second / 'verdict.json').read_text())['verdict']['label'] == 'COST_EXCEEDED_AT_SMOKE'
    assert make_runner(config, SyntheticEnv(tmp_path), second).advance('D') == runner.EXIT_STOPPED
    assert not (second / 'mandatory').exists()


def test_a_crashed_pass_folder_is_closed_as_an_abort(tmp_path):
    attempt = tmp_path / 'attempt-001'
    (attempt / 'smoke').mkdir(parents=True)
    (attempt / 'smoke' / 'started.json').write_text('{}\n')
    assert make_runner(registered_config(), SyntheticEnv(tmp_path), attempt).advance('smoke') == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'smoke' / 'aborted.json').read_text())
    assert 'without a terminal receipt' in aborted['reason'] and not aborted['citable']
    runner.verify_sums(attempt / 'smoke')


def test_pass_contents_must_match_their_terminal_receipt(tmp_path, monkeypatch):
    attempt, manifest = tmp_path / 'attempt-001', tmp_path / 'digest-manifest.jsonl'
    env = SyntheticEnv(tmp_path)
    assert make_runner(registered_config(), env, attempt, manifest=manifest).advance('smoke') == runner.EXIT_OK
    results = attempt / 'smoke' / 'results.json'
    record = json.loads(results.read_text())
    record['r_b'] = 0.01  # a resealed replacement under the same config and code identity
    results.write_text(json.dumps(record))
    runner.write_sums(attempt / 'smoke')
    with pytest.raises(runner.IntegrityFailure, match='differ from its COMPLETED receipt'):
        make_runner(registered_config(), env, attempt, manifest=manifest).advance('M')
    second = tmp_path / 'attempt-002'
    assert make_runner(registered_config(), env, second, manifest=manifest).advance('smoke') == runner.EXIT_OK
    lines = manifest.read_text().splitlines()
    manifest.write_text('\n'.join(line for line in lines if 'attempt-002' not in line or '"COMPLETED"' not in line) + '\n')
    with pytest.raises(runner.IntegrityFailure, match='missing from the digest manifest'):
        make_runner(registered_config(), env, second, manifest=manifest).advance('smoke')
    original = runner.PassRunner.receipt

    def crash_before_marker(self, name, event, *args, **kwargs):
        if event == 'COMPLETED':
            raise KeyboardInterrupt('process killed after sealing')
        return original(self, name, event, *args, **kwargs)
    monkeypatch.setattr(runner.PassRunner, 'receipt', crash_before_marker)
    third = tmp_path / 'attempt-003'
    with pytest.raises(KeyboardInterrupt):
        make_runner(registered_config(), env, third).advance('smoke')
    assert (third / 'smoke' / 'SHA256SUMS').exists() and (third / 'smoke' / 'results.json').exists()
    monkeypatch.setattr(runner.PassRunner, 'receipt', original)
    assert make_runner(registered_config(), env, third).advance('M') == runner.EXIT_STOPPED  # never resumes as complete
    assert json.loads((third / 'smoke' / 'aborted.json').read_text())['effect'] == 'NO_FULL_RUN'
    assert not (third / 'mandatory').exists()


def test_a_stopped_pass_c_gives_not_stable_with_rb12_aborted_and_keeps_pass_m(tmp_path, monkeypatch):
    config = flow_config(monkeypatch)
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt).advance('M') == runner.EXIT_OK
    assert make_runner(config, env, attempt, tick=3000.).advance('C') == runner.EXIT_STOPPED
    aborted = json.loads((attempt / 'rb12' / 'aborted.json').read_text())
    assert aborted['effect'] == 'RB12_ABORTED' and aborted['reason'].startswith('DeadlinePassed')
    verdict = json.loads((attempt / 'verdict.json').read_text())['verdict']
    assert verdict == {'row': 7, 'label': 'NOT_STABLE', 'flags': ['RB12_ABORTED'], 'final': True}
    runner.verify_sums(attempt / 'mandatory')  # pass M evidence kept
    assert make_runner(config, env, attempt).advance('D') == runner.EXIT_OK  # pass D may still run
    assert (attempt / 'descriptive' / 'BF48' / 'results.json').exists()


def test_selection_publication_is_anchored_on_its_receipt_and_recovers_from_interruption(tmp_path, monkeypatch):
    config = flow_config(monkeypatch, stable=True)
    attempt, manifest = tmp_path / 'attempt-001', tmp_path / 'digest-manifest.jsonl'
    env = SyntheticEnv(tmp_path)
    assert make_runner(config, env, attempt, manifest=manifest).advance('D') == runner.EXIT_OK
    original = runner.PassRunner.receipt

    def failing(self, name, event, *args, **kwargs):
        if event == 'SELECTED':
            raise OSError('receipt write failed after the rename')
        return original(self, name, event, *args, **kwargs)
    monkeypatch.setattr(runner.PassRunner, 'receipt', failing)
    assert make_runner(config, env, attempt, manifest=manifest).confirm_select() == runner.EXIT_STOPPED
    assert not (attempt / runner.SELECTION_FILE).exists() and receipts(attempt)[-1]['event'] == 'SELECTION_ABORTED'

    def killed(self, name, event, *args, **kwargs):
        if event == 'SELECTED':
            raise KeyboardInterrupt('process killed after the rename')
        return original(self, name, event, *args, **kwargs)
    monkeypatch.setattr(runner.PassRunner, 'receipt', killed)
    with pytest.raises(KeyboardInterrupt):
        make_runner(config, env, attempt, manifest=manifest).confirm_select()
    assert (attempt / runner.SELECTION_FILE).exists()  # renamed, never receipted
    monkeypatch.setattr(runner.PassRunner, 'receipt', original)
    pin = {'experiment_id': reg.EXPERIMENT_ID, 'confirmation_games_sha256': env.confirmation_record['games_sha256']}
    with pytest.raises(runner.IntegrityFailure, match='published selection'):
        make_runner(config, env, attempt, manifest=manifest).confirm(pin)
    assert not (attempt / runner.SELECTION_FILE).exists() and receipts(attempt)[-1]['event'] == 'SELECTION_RECOVERED'
    work = make_runner(config, env, attempt, manifest=manifest)
    assert work.confirm_select() == runner.EXIT_OK and work.confirm_select() == runner.EXIT_OK  # published once
    assert [r['event'] for r in receipts(attempt)].count('SELECTED') == 1
    published = attempt / runner.SELECTION_FILE
    published.write_text(published.read_text().replace('"selected": 2', '"selected": 3'))
    with pytest.raises(runner.IntegrityFailure, match='SELECTED receipt'):
        make_runner(config, env, attempt, manifest=manifest).confirm(pin)


def test_the_terminal_receipt_carries_post_sealing_measurements(tmp_path, monkeypatch):
    clock = FakeClock(datetime(2026, 10, 27, 14, 0, tzinfo=KST))
    seal = runner.write_sums
    state = {'sealed': False}

    def slow_seal(folder):
        clock.jump(100)  # a successful, sub-cap sealing delay
        state['sealed'] = True
        return seal(folder)
    monkeypatch.setattr(runner, 'write_sums', slow_seal)
    rss = lambda: (3 << 30) if state['sealed'] else (1 << 30)  # noqa: E731  (the peak grows while sealing)
    attempt = tmp_path / 'attempt-001'
    assert make_runner(registered_config(), SyntheticEnv(tmp_path), attempt, clock=clock, rss=rss).advance('smoke') == runner.EXIT_OK
    results = json.loads((attempt / 'smoke' / 'results.json').read_text())
    completed = receipts(attempt)[-1]
    assert completed['event'] == 'COMPLETED'
    final = completed['measurements']
    assert final['seconds_total'] >= results['seconds_pre_sealing'] + 100
    assert final['peak_rss_bytes_process_lifetime'] == 3 << 30 > results['peak_rss_bytes_process_lifetime_pre_sealing']
    assert final['utc'] > results['computed_utc_pre_sealing']


def test_the_memory_ledger_is_guarded_restored_and_hash_chain_equivalent(tmp_path):
    original = prt.Ledger
    with pytest.raises(ValueError, match='inside'):
        with runner.memory_ledger():
            assert prt.Ledger is runner.MemoryLedger
            raise ValueError('failure inside the substitution')
    assert prt.Ledger is original  # restored on an exception
    with runner.memory_ledger():
        with pytest.raises(runner.IntegrityFailure, match='single-threaded'):
            with runner.memory_ledger():  # nested or overlapping substitution refused
                pass
        assert prt.Ledger is runner.MemoryLedger
    assert prt.Ledger is original
    release = threading.Event()
    other = threading.Thread(target=release.wait)
    other.start()
    try:
        with pytest.raises(runner.IntegrityFailure, match='single-threaded'):
            with runner.memory_ledger():
                pass
    finally:
        release.set()
        other.join()
    assert prt.Ledger is original
    header = {'runtime_sha256': 'synthetic', 'pins': {'contract': 'ML-POLICY-RUNTIME-v2'}}
    records = [{'kind': 'decision', 'request_id': f'g:1:{k}', 'result': {'q_planning': [.5 + k / 10, None]}}
               for k in range(3)]
    on_disk, in_memory = prt.Ledger(tmp_path / 'ledger.jsonl', header), runner.MemoryLedger(None, header)
    for record in records:
        assert on_disk._append(copy.deepcopy(record)) == in_memory._append(copy.deepcopy(record))
    assert in_memory.rows == on_disk.rows and (in_memory.count, in_memory.head) == (on_disk.count, on_disk.head)
    first = in_memory.read(1)
    first['result']['q_planning'][0] = -1.  # a caller mutating what it read cannot change the ledger
    assert in_memory.read(1)['result']['q_planning'][0] == .5
    in_memory._rows[2]['result'] = {'q_planning': [0.]}
    with pytest.raises(runner.IntegrityFailure, match='tampered'):
        in_memory.read(2)


def test_the_smoke_record_accepts_exactly_the_runtime_status_vocabulary():
    counts = {status: 1 for status in prt.STATUSES}  # incl. UNSUPPORTED_LOGGING_POSITIVITY ("logging")
    record = {'r_b': 2.1, 'statuses': counts, 'fhdp_only_game_statuses': dict(counts),
              'mechanical_rule': {'status': 'FULL_RUN_ALLOWED'}}
    assert reg.runtime_statuses() == tuple(prt.STATUSES) and reg.smoke_record_problems(record) == []
    assert reg.smoke_metric_keys(record) == []
    assert reg.smoke_record_problems({**record, 'statuses': {**counts, 'mean_tv': 1}})  # not a runtime status
    assert reg.smoke_record_problems({**record, 'statuses': {'SUPPORTED': .5}})  # not a count
    assert reg.smoke_record_problems({**record, 'cost': {'logging_law': 1}})  # metric-like keys elsewhere still refused


def test_a_smoke_with_real_runtime_refusals_completes(tmp_path):
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path, refusals=True)
    assert make_runner(registered_config(), env, attempt).advance('smoke') == runner.EXIT_OK
    statuses = json.loads((attempt / 'smoke' / 'results.json').read_text())['statuses']
    assert {'UNSUPPORTED_LOGGING_POSITIVITY', 'UNSUPPORTED_UNKNOWN_PITCHER', 'UNSUPPORTED_MID_PA', 'SUPPORTED'} <= set(statuses)
    assert set(statuses) <= set(prt.STATUSES) and all(type(v) is int for v in statuses.values())


APPLEDOUBLE = bytes.fromhex('00051607') + b'\x00' * 60  # what macOS writes beside files on a non-APFS volume


def sidecars(folder):
    for path in list(Path(folder).iterdir()):
        if path.is_file() and not path.name.startswith('._'):
            (path.parent / f'._{path.name}').write_bytes(APPLEDOUBLE)


def test_appledouble_sidecars_never_break_a_seal_and_the_abort_path_reseals(tmp_path, monkeypatch):
    attempt = tmp_path / 'attempt-001'
    env = SyntheticEnv(tmp_path)
    assert make_runner(registered_config(), env, attempt).advance('smoke') == runner.EXIT_OK
    sidecars(attempt / 'smoke')  # created after sealing, as on the T7 drive (._SHA256SUMS included)
    assert make_runner(registered_config(), env, attempt).status('smoke')[0] == 'complete'
    (attempt / 'smoke' / '._payload').write_bytes(b'not AppleDouble')  # same prefix, no magic: a real file
    with pytest.raises(runner.IntegrityFailure, match='files differ'):
        make_runner(registered_config(), env, attempt).status('smoke')
    second = tmp_path / 'attempt-002'
    original = reg.smoke_record_problems
    monkeypatch.setattr(reg, 'smoke_record_problems', lambda record: original(record) + ['injected sink refusal'])
    seal = runner.write_sums

    def sealing_with_sidecars(folder):
        sidecars(folder)  # sidecars of started.json/aborted.json exist when the abort path seals
        return seal(folder)
    monkeypatch.setattr(runner, 'write_sums', sealing_with_sidecars)
    assert make_runner(registered_config(), env, second).advance('smoke') == runner.EXIT_STOPPED
    sidecars(second / 'smoke')  # ._SHA256SUMS and ._aborted.json appear after the seal
    monkeypatch.undo()
    state, record = make_runner(registered_config(), env, second).status('smoke')
    assert state == 'aborted' and 'injected sink refusal' in record['reason']
    listed = [line.split('  ')[1] for line in (second / 'smoke' / 'SHA256SUMS').read_text().splitlines()]
    assert listed == ['aborted.json', 'started.json']  # the final runner files, resealed after aborted.json
    assert make_runner(registered_config(), env, second).advance('smoke') == runner.EXIT_STOPPED  # no IntegrityFailure


def test_the_digest_manifest_accepts_a_chained_reregistration(tmp_path):
    manifest = tmp_path / 'digest-manifest.jsonl'
    config = registered_config(**{'registration.digest_manifest_path': str(manifest)})
    void = {'experiment_id': reg.EXPERIMENT_ID, 'kind': 'config_digest', 'digest': '9' * 64,
            'registered_commit': '1' * 40, 'code_sha256': {}}
    active = {'experiment_id': reg.EXPERIMENT_ID, 'kind': 'config_digest_reregistration', 'previous_digest': '9' * 64,
              'reason': 'bug-fix retry 1', 'aborted_attempt': '/runs/EXP-STAB-02/attempt-001',
              'digest': reg.config_digest(config), 'registered_commit': config['registration']['registered_commit'],
              'code_sha256': config['registration']['code_sha256']}
    manifest.write_text(json.dumps(void) + '\n' + json.dumps({'kind': 'receipt'}) + '\n' + json.dumps(active) + '\n')
    assert reg.verify_digest_manifest(config, REPO) == manifest.resolve()
    for broken in ({**active, 'previous_digest': '8' * 64}, {**active, 'reason': ''}, {**active, 'kind': 'config_digest'}):
        manifest.write_text(json.dumps(void) + '\n' + json.dumps(broken) + '\n')
        with pytest.raises(reg.RegistrationError, match='re-registration|exactly one'):
            reg.verify_digest_manifest(config, REPO)
    manifest.write_text(json.dumps(void) + '\n')  # the void registration alone does not match this config
    with pytest.raises(reg.RegistrationError, match='digest differs'):
        reg.verify_digest_manifest(config, REPO)
