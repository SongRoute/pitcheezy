"""EXP-BC-CAL-01 candidates on synthetic data: B0 recipe identity (T1), B1 shrinkage math, support
(T6), BehaviorModel protocol compliance of B1/B2 and the B2 round rule. No real data."""
import math

import numpy as np
import pandas as pd
import pytest

from behavior_calibration_synthetic import bc_states, make_world, run_isolated
from pitchmdp.behavior_calibration_registry import ACTION_CODES, b0_alphas
from pitchmdp.behavior_calibration import Temperature
from pitchmdp.behavior_model import (LEVEL_COMPONENTS, UNSEEN, BackoffBC, BehaviorModel, BehaviorTable, Calibrated,
                                     CountTable, Keys, RepertoireSupport, VectorBC, previous_index, side_code)
from pitchmdp.policy_artifacts import IntegrityError
from pitchmdp.rollout_policy import BCRecord, CategoricalBC, PAState, PastPitch

V = len(ACTION_CODES)
OOV = ('<UNKNOWN>', 'ZZ', 'PITCHOUT?', 'xx')  # previous labels outside the 18 codes (raw strings)


@pytest.fixture(scope='module')
def world():
    return make_world(0)


def random_states(rng, pitchers, n):
    states = []
    for _ in range(n):
        previous = rng.choice([*ACTION_CODES, '<START>', *OOV])
        history = () if previous == '<START>' else (PastPitch(str(previous), (), 'unknown', 0, 0),)
        states.append(PAState(int(rng.integers(4)), int(rng.integers(3)), str(rng.choice(pitchers)),
                              str(rng.choice(['L', 'R'])), history))
    return states


def keys_of(states):
    return Keys([int(s.pitcher) for s in states], side_code([s.batter_side for s in states]),
                [s.balls for s in states], [s.strikes for s in states],
                [s.history[-1].action if s.history else '<START>' for s in states])


def full_probabilities(bc, state):
    """CategoricalBC probabilities placed on the 18 registered codes (0 outside its actions)."""
    return np.array([dict(zip(bc.actions, bc.probabilities(state))).get(a, 0.) for a in ACTION_CODES])


def toy_state(previous, pitcher='1'):
    history = () if previous == '<START>' else (PastPitch(previous, (), 'unknown', 0, 0),)
    return PAState(0, 0, pitcher, 'L', history)


def test_t1_unseen_previous_label_matches_no_cell():
    """Review counterexample 1: an unseen previous label must not borrow a known OOV label's cell
    (CategoricalBC: no cell -> prior (0.5, 0.5); the old collapsed key gave (0.75, 0.25))."""
    records = [BCRecord(toy_state('AA'), 'FF', 'train'), BCRecord(toy_state('<START>'), 'SL', 'train')]
    bc = CategoricalBC(1., 1).fit(records)
    ff, sl = ACTION_CODES.index('FF'), ACTION_CODES.index('SL')
    vector = VectorBC.fit(keys_of([r.state for r in records]), [ACTION_CODES.index(r.action) for r in records],
                          prior_strength=1., minimum_action_count=1)
    probe = toy_state('BB')
    expected = full_probabilities(bc, probe)
    assert np.allclose(expected[[ff, sl]], [.5, .5], rtol=0, atol=1e-15)
    assert np.abs(vector.batch_keys(keys_of([probe]))[0] - expected).max() <= 1e-12
    assert np.abs(vector.batch_keys(keys_of([toy_state('AA')]))[0][[ff, sl]] - [.75, .25]).max() <= 1e-12


def test_t1_distinct_oov_labels_stay_distinct_cells():
    """Review counterexample 2: two distinct OOV labels are two cells (the collapsed key merged them,
    a 0.25 discrepancy)."""
    records = [BCRecord(toy_state('AA'), 'FF', 'train'), BCRecord(toy_state('BB'), 'SL', 'train')]
    bc = CategoricalBC(1., 1).fit(records)
    vector = VectorBC.fit(keys_of([r.state for r in records]), [ACTION_CODES.index(r.action) for r in records],
                          prior_strength=1., minimum_action_count=1)
    probes = [toy_state(p) for p in ('AA', 'BB', 'CC', '<START>', 'FF')]
    expected = np.array([full_probabilities(bc, s) for s in probes])
    assert np.abs(vector.batch_keys(keys_of(probes)) - expected).max() <= 1e-12
    assert vector.cells.vocabulary == ('AA', 'BB') and len(vector.cells.keys) == 2
    b1 = BackoffBC.fit(keys_of([r.state for r in records]), [ACTION_CODES.index(r.action) for r in records],
                       alphas={**b0_alphas(), 'previous_type': 1.})
    assert np.abs(b1.batch_keys(keys_of(probes)) - expected).max() <= 1e-12  # B1 at a B0-shaped setting


def test_t1_vector_bc_equals_categorical_bc(world):
    rng = np.random.default_rng(3)
    bc = world['artifact'].bc
    pitchers = [*bc.pitchers, '123', '999999']  # two unknown pitchers: league fallback
    states = random_states(rng, pitchers, 3000)
    expected = np.array([bc.probabilities(s) for s in states])
    vector = VectorBC.from_artifact(world['artifact'])
    assert np.abs(vector.batch_keys(keys_of(states)) - expected).max() <= 1e-12
    # fitted from rows (OOF / XF path) on random synthetic records: same law as CategoricalBC.fit
    records = bc_states(world['frame'])
    keep = rng.random(len(records)) < .7
    part = [r for r, k in zip(records, keep) if k]
    fitted = CategoricalBC(20., 1).fit(part)
    refit = VectorBC.fit(keys_of([r.state for r in part]), [ACTION_CODES.index(r.action) for r in part])
    probe = [s for s in random_states(rng, list(fitted.pitchers) + ['42'], 2000)]
    assert np.abs(refit.batch_keys(keys_of(probe)) - np.array([full_probabilities(fitted, s) for s in probe])).max() \
        <= 1e-12
    # OOV previous labels inside the fit rows stay distinct raw-string cells
    oov = [BCRecord(PAState(r.state.balls, r.state.strikes, r.state.pitcher, r.state.batter_side,
                            (PastPitch(OOV[i % len(OOV)], (), 'unknown', 0, 0),)), r.action, 'train')
           for i, r in enumerate(part[:300])]
    mixed = CategoricalBC(20., 1).fit(part + oov)
    remix = VectorBC.fit(keys_of([r.state for r in part + oov]), [ACTION_CODES.index(r.action) for r in part + oov])
    assert np.abs(remix.batch_keys(keys_of(probe)) - np.array([full_probabilities(mixed, s) for s in probe])).max() \
        <= 1e-12


def test_t1_b1_at_the_b0_setting_is_b0(world):
    table, b0 = world['tables']['train'], world['b0']
    labels = table.frame.label.to_numpy()
    b1 = BackoffBC.fit(table.keys, labels, repertoire=b0.repertoire)
    assert b1.alphas == b0_alphas()
    for name in ('earlystop', 'temperature', 'train'):
        other = world['tables'][name]
        assert np.abs(b1.batch(other) - b0.batch(other)).max() <= 1e-12
    assert np.array_equal(b1.tables['previous_type'].keys, b0.vector.cells.keys)
    assert np.array_equal(b1.tables['previous_type'].counts, b0.vector.cells.counts)


def test_hierarchical_shrinkage_math_on_a_toy_table():
    # one pitcher (7), two codes in R_j (FF, SL); one league-only code (CH) outside R_j
    ff, sl, ch = ACTION_CODES.index('FF'), ACTION_CODES.index('SL'), ACTION_CODES.index('CH')
    rows = pd.DataFrame({'pitcher': [7, 7, 7, 7, 8], 'batter_side': ['L', 'L', 'R', 'L', 'R'],
                         'balls': [0, 0, 0, 1, 0], 'strikes': [0, 0, 0, 0, 0],
                         'previous': ['<START>'] * 5, 'base_out': [0, 0, 0, 0, 0], 'tto_group': [1] * 5,
                         'batter_cluster': [5] * 5})
    actions = np.array([ff, ff, sl, sl, ch])
    keys = Keys.from_frame(rows)
    alphas = {'pitcher': 0., 'batter_side': 2., 'count': math.inf, 'previous_type': 5., 'base_out': 1.,
              'times_through_order': math.inf, 'batter_cluster': math.inf}
    b1 = BackoffBC.fit(keys, actions, alphas=alphas)
    query = Keys.from_frame(rows.iloc[[0]])  # pitcher 7, L, 0-0, START, base_out 0
    p = b1.batch_keys(query)[0]
    # l1 (alpha 0): pitcher frequency on R_j = {FF: 2, SL: 2} -> (.5, .5)
    p1 = np.array([.5, .5])
    # l2 x side L: n = (FF 2, SL 1), N = 3, alpha 2 -> (2 + 2*.5, 1 + 2*.5) / 5
    p2 = (np.array([2, 1]) + 2 * p1) / 5
    # l3 inf: skip. l4 x previous START within (7, L, 0-0): n = (FF 2, SL 0), N = 2, alpha 5
    p4 = (np.array([2, 0]) + 5 * p2) / 7
    # l5 x base_out 0 inside the same key: same counts, alpha 1
    p5 = (np.array([2, 0]) + 1 * p4) / 3
    assert p[ch] == 0 and abs(p.sum() - 1) < 1e-15
    assert np.allclose(p[[ff, sl]], p5, rtol=0, atol=1e-15)
    # an unseen deeper cell keeps its parent (here: a new base_out code at level 5)
    unseen = Keys.from_frame(rows.iloc[[0]].assign(base_out=8))
    assert np.allclose(b1.batch_keys(unseen)[0][[ff, sl]], p4, rtol=0, atol=1e-15)
    # unknown pitcher: league frequency on the league support (B0 fallback)
    stranger = Keys.from_frame(rows.iloc[[0]].assign(pitcher=99))
    league = np.bincount(actions, minlength=V) / len(actions)
    assert np.allclose(b1.batch_keys(stranger)[0], league, rtol=0, atol=1e-15)
    with pytest.raises(IntegrityError):
        BackoffBC.fit(keys, actions, alphas={**alphas, 'count': 0.})  # alpha 0 only at the pitcher level


def test_t6_support_zero_outside_and_same_as_b0(world):
    b0 = world['b0']
    table = world['tables']['earlystop']
    b1 = BackoffBC.fit(world['tables']['train'].keys, world['tables']['train'].frame.label.to_numpy(),
                       repertoire=b0.repertoire, alphas={**b0_alphas(), 'batter_side': 5., 'base_out': 50.})
    for model in (b0, b1, VectorBC.from_artifact(world['artifact'])):
        p = model.batch(table)
        mask = b0.support_batch(table)
        assert (p[~mask] == 0).all() and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
        assert np.array_equal(model.support_batch(table), mask)
    state = PAState(0, 0, '424242', 'L', ())
    assert b1.fallback(state) and b0.fallback(state)
    assert np.array_equal(b1.support(state), b0.support(state))
    assert np.array_equal(b1.support(state), b0.repertoire.league >= 1)


def test_behavior_model_protocol_compliance(world):
    table = world['tables']['temperature']
    b0 = world['b0']
    b1 = BackoffBC.fit(world['tables']['train'].keys, world['tables']['train'].frame.label.to_numpy(),
                       repertoire=b0.repertoire, alphas={**b0_alphas(), 'base_out': 10., 'batter_side': 2.})
    calibrated = Calibrated(b1, Temperature(1.3), 'B1')
    models = (b0, VectorBC.from_artifact(world['artifact']), b1, calibrated)  # B2: isolated file
    frame = table.frame
    for model in models:
        assert isinstance(model, BehaviorModel)
        assert tuple(model.actions) == ACTION_CODES
        model.bind(table)
        batch = model.batch(table)
        for i in range(0, len(table), max(1, len(table) // 25)):
            row = frame.iloc[i]
            history = () if row.previous == '<START>' else (PastPitch(str(row.previous), (), 'unknown', 0, 0),)
            state = PAState(int(row.balls), int(row.strikes), str(int(row.pitcher)), row.batter_side, history,
                            f'{int(row.game_pk)}:{int(row.at_bat_number)}:{int(row.pitch_number)}')
            p = model.probabilities(state)
            assert p.shape == (V,) and np.allclose(p, batch[i], rtol=0, atol=1e-12)
            assert np.array_equal(model.support(state), b0.support(state))
            assert (p[~model.support(state)] == 0).all() and abs(p.sum() - 1) < 1e-12
            assert model.fallback(state) == b0.fallback(state)
        identity = model.identity()
        assert isinstance(identity, dict) and 'class' in identity
    wrong = PAState(3, 2, '1', 'L', (), frame.iloc[0].pipe(
        lambda r: f'{int(r.game_pk)}:{int(r.at_bat_number)}:{int(r.pitch_number)}'))
    with pytest.raises(IntegrityError):
        b1.probabilities(wrong)  # a state that is not its frozen context row


def test_count_table_and_keys_guards():
    keys = Keys([5, 5, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0], ['<START>'] * 3)
    table = CountTable.from_rows(keys, np.array([0, 1, 1]), 3, LEVEL_COMPONENTS['pitcher'])
    counts, found = table.lookup(Keys([3, 4, 5], [0, 0, 0], [0, 0, 0], [0, 0, 0], ['x', 'y', 'z']))
    assert found.tolist() == [True, False, True] and counts.tolist() == [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
    cells = CountTable.from_rows(Keys([5, 5], [0, 0], [0, 0], [0, 0], ['AA', 'FF']), np.array([0, 1]), 3,
                                 LEVEL_COMPONENTS['previous_type'])
    assert cells.vocabulary == ('AA', 'FF')
    _, found = cells.lookup(Keys([5, 5, 5], [0, 0, 0], [0, 0, 0], [0, 0, 0], ['AA', 'BB', 'FF']))
    assert found.tolist() == [True, False, True]  # an unseen label matches no cell
    assert previous_index(['AA', 'BB'], ('AA',)).tolist() == [0, UNSEEN]
    with pytest.raises(IntegrityError):
        CountTable(cells.keys, cells.counts, LEVEL_COMPONENTS['previous_type'], ('FF', 'AA'))  # unsorted vocabulary
    with pytest.raises(IntegrityError):
        Keys([1 << 25], [0], [0], [0], ['<START>'])
    with pytest.raises(IntegrityError):
        Keys([1], [0], [0], [0], [''])  # previous labels are nonempty strings
    with pytest.raises(IntegrityError):
        side_code(['S'])
    support = RepertoireSupport.from_rows(np.array([1, 1, 2]), np.array([0, 2, 1]), 3)
    assert support.mask(np.array([1, 9])).tolist() == [[True, False, True], [True, True, True]]
    with pytest.raises(IntegrityError):
        BehaviorTable(pd.DataFrame({'game_pk': [1, 1], 'at_bat_number': [1, 1], 'pitch_number': [1, 1],
                                    'pitcher': [1, 1], 'batter_side': ['L', 'L'], 'balls': [0, 0], 'strikes': [0, 0],
                                    'previous': ['<START>'] * 2}))


def test_lightgbm_b2_paths_run_with_one_openmp_runtime():
    """B2 (LightGBM) checks run in a fresh interpreter with torch and LightGBM sharing one OpenMP
    runtime (repository launch rule); see isolated_behavior_calibration_lightgbm.py."""
    done = run_isolated('isolated_behavior_calibration_lightgbm.py')
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert ' passed' in done.stdout and 'failed' not in done.stdout
