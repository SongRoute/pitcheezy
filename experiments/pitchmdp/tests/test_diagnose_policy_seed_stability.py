"""Synthetic wiring checks for scripts/diagnose_policy_seed_stability.py (no real data).

The data path (sealed ledger, bound components, planner) is NOT exercised here; a PM-run smoke on
the registered inputs is required before any registered run.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import diagnose_policy_seed_stability as runner  # noqa: E402
import diagnose_policy_predictor_swap as swap  # noqa: E402

DRAFT_PATH = PROJECT / 'configs' / 'drafts' / 'policy-seed-stability-DRAFT-v1.json'
DRAFT = json.loads(DRAFT_PATH.read_text())
P18 = json.loads((PROJECT.parents[1] / 'configs' / 'EXP-P18-001.yaml').read_text())
SEEDS = {'planning_main': 0, 'planning_v2': [0, 1, 2]}
SETTINGS = {'tau': .1, 'samples': 6, 'pitch_cap': 12, 'seed': 0}


def registered_copy(experiment='EXP-P99-999'):
    config = copy.deepcopy(DRAFT)
    config.update({'experiment_id': experiment, 'status': 'REGISTERED'})
    config['execution']['enabled'] = True
    return config


def test_draft_config_is_execution_disabled_unassigned_and_otherwise_well_formed():
    assert DRAFT['status'] == 'DRAFT' and DRAFT['execution']['enabled'] is False and DRAFT['experiment_id'] == 'UNASSIGNED'
    assert all(DRAFT[key] == P18[key] for key in runner.SAME_AS_P18), 'draft must keep the P18 sealed-side inputs'
    assert 'after' in DRAFT['origin'] and 'P18' in DRAFT['origin'] and 'verdict_rule' not in DRAFT and 'stop_rule' not in DRAFT
    assert all((PROJECT / rel).is_file() for rel in runner.SOURCES)
    with pytest.raises(ValueError, match='not assigned'):
        runner.config_check(DRAFT, SEEDS)
    plan = runner.config_check(DRAFT, SEEDS, smoke=True)       # a draft may be smoked as is (wiring check, no repository result)
    assert [v['label'] for v in plan] == [v['label'] for v in DRAFT['variants']]
    regs = registered_copy()
    assert runner.config_check(regs, SEEDS) == plan
    regs['execution']['enabled'] = False
    with pytest.raises(ValueError, match='not enabled'):
        runner.config_check(regs, SEEDS)
    assert runner.config_check(regs, SEEDS, smoke=True) == plan
    still_draft = registered_copy()
    still_draft['status'] = 'DRAFT'
    with pytest.raises(ValueError, match='REGISTERED'):
        runner.config_check(still_draft, SEEDS)
    odd_status = registered_copy()
    odd_status['status'] = 'PROPOSED'
    with pytest.raises(ValueError, match='DRAFT or REGISTERED'):
        runner.config_check(odd_status, SEEDS, smoke=True)
    # Row budget floor: 1.2 x the measured P18 rows scaled to the largest sample count (12 -> 9,132,084).
    assert runner.row_budget_floor(plan) == 9132084 and DRAFT['row_budget'] >= 9132084
    low = registered_copy()
    low['row_budget'] = 6000000                                 # the P18 cap would overflow for a 12-sample variant
    with pytest.raises(ValueError, match='row_budget must be an integer >= 9132084'):
        runner.config_check(low, SEEDS)
    six_only = registered_copy()
    six_only['variants'] = [v for v in six_only['variants'] if v['samples'] == 6]
    six_only['row_budget'] = 4600000
    assert len(runner.config_check(six_only, SEEDS)) == 3     # floor for 6 samples is 4,566,042
    no_boot = registered_copy()
    no_boot['bootstrap'] = {'draws': 0, 'seed': 1}
    with pytest.raises(ValueError, match='Bootstrap'):
        runner.config_check(no_boot, SEEDS)
    with_rule = registered_copy()
    with_rule['verdict_rule'] = {'excess_mean_tv_at_least': .01}
    with pytest.raises(ValueError, match='descriptive'):
        runner.config_check(with_rule, SEEDS)
    no_origin = registered_copy()
    no_origin['origin'] = 'prospective'
    with pytest.raises(ValueError, match='origin'):
        runner.config_check(no_origin, SEEDS)
    for bad in ('EXP-P18', 'exp-p18-002', 'EXP-P18-002x'):
        malformed = registered_copy(bad)
        with pytest.raises(ValueError, match='malformed'):
            runner.config_check(malformed, SEEDS)


def test_variant_plan_rejects_invalid_seed_and_sample_settings():
    good = [{'label': 'seed2-s6', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2'},
            {'label': 'seed3-s6', 'seed': 3, 'samples': 6, 'seed_basis': 'extension'},
            {'label': 'seed0-s12', 'seed': 0, 'samples': 12, 'seed_basis': 'planning_main'},
            {'label': 'seed1-s12', 'seed': 1, 'samples': 12, 'seed_basis': 'planning_v2'}]
    plan = runner.variant_plan(good, SETTINGS, SEEDS)
    assert [v['kind'] for v in plan] == ['seed', 'seed', 'samples', 'samples']
    assert [v['shares_prefix_draws'] for v in plan] == [False, False, True, False]
    cases = [
        ([], 'At least one'),
        ([{'label': 'x', 'seed': 0, 'samples': 6, 'seed_basis': 'planning_main'}], 'identical to the sealed'),
        ([{'label': 'x', 'seed': True, 'samples': 6, 'seed_basis': 'planning_v2'}], 'non-negative integer'),
        ([{'label': 'x', 'seed': -1, 'samples': 6, 'seed_basis': 'extension'}], 'non-negative integer'),
        ([{'label': 'x', 'seed': 1.0, 'samples': 6, 'seed_basis': 'planning_v2'}], 'non-negative integer'),
        ([{'label': 'x', 'seed': 2, 'samples': 0, 'seed_basis': 'planning_v2'}], 'positive integer'),
        ([{'label': 'x', 'seed': 2, 'samples': 6.0, 'seed_basis': 'planning_v2'}], 'positive integer'),
        ([{'label': 'x', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2'},
          {'label': 'y', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2'}], 'duplicate \\(seed, samples\\)'),
        ([{'label': 'x', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2'},
          {'label': 'x', 'seed': 3, 'samples': 6, 'seed_basis': 'extension'}], 'Duplicate variant label'),
        ([{'label': 'bad label', 'seed': 2, 'samples': 6, 'seed_basis': 'planning_v2'}], 'label invalid'),
        ([{'label': 'x', 'seed': 2, 'samples': 6}], 'seed_basis'),
        ([{'label': 'x', 'seed': 3, 'samples': 6, 'seed_basis': 'planning_v2'}], 'not a registered planning seed'),
        ([{'label': 'x', 'seed': 2, 'samples': 6, 'seed_basis': 'extension'}], 'is a registered planning seed'),
        ([{'label': 'x', 'seed': 1, 'samples': 6, 'seed_basis': 'planning_main'}], 'differs from the registered main seed'),
    ]
    for variants, message in cases:
        with pytest.raises(ValueError, match=message):
            runner.variant_plan(variants, SETTINGS, SEEDS)
    with pytest.raises(ValueError, match='Sealed setting'):
        runner.variant_plan(good, {**SETTINGS, 'samples': 6.0}, SEEDS)


def test_registered_seeds_come_from_the_policy_registration():
    policy = json.loads((PROJECT.parents[1] / 'configs' / 'ML-POLICY-MATERIALIZATION-v1.json').read_text())
    seeds = runner.registered_seeds(policy)
    assert seeds == SEEDS and seeds['planning_main'] == P18['settings']['seed'] and P18['noise_seed'] in seeds['planning_v2']


def test_raising_the_sample_count_with_the_sealed_seed_reuses_the_sealed_draws_as_a_prefix():
    """numpy draws sequentially, so a (12, cap, 3) tensor from the same key starts with the (6, cap, 3) one."""
    key = 987654321
    six = np.random.default_rng(key).random((6, 12, 3))
    twelve = np.random.default_rng(key).random((12, 12, 3))
    assert np.array_equal(twelve[:6], six)
    other = np.random.default_rng(key + 1).random((6, 12, 3))
    assert not np.allclose(other, six)


class FakePlanner:
    def __init__(self, q): self.q = np.asarray(q, dtype=float)

    def q_values(self, state): return self.q.copy(), {}


def test_compare_decision_keeps_laws_and_measures_distance_to_the_sealed_law():
    mask = np.array([True, True, False, True, False, False, False, False, False, False])
    reference = np.array([.5, .3, 0., .2, 0., 0., 0., 0., 0., 0.])
    q_sealed = np.array([.50, .52, -np.inf, .49, *([-np.inf] * 6)])
    q_moved = np.array([.40, .60, -np.inf, .49, *([-np.inf] * 6)])
    planners = {'same': FakePlanner(q_sealed), 'moved': FakePlanner(q_moved)}
    sealed_row, rows = runner.compare_decision(planners, 'state', q_sealed, reference, mask, .1)
    assert np.allclose(sealed_row['law'], swap.law(q_sealed, reference, mask, .1)) and sealed_row['mask'].dtype == bool
    assert rows['same']['tv'] == 0 and rows['same']['top1_same'] and rows['same']['max_abs_q_difference'] == 0
    assert np.array_equal(rows['same']['law'], sealed_row['law']) and rows['same']['top1'] == sealed_row['top1']
    moved = rows['moved']
    assert moved['tv'] == pytest.approx(.5 * np.abs(moved['law'] - sealed_row['law']).sum()) and 0 < moved['tv'] <= 1
    assert moved['max_abs_q_difference'] == pytest.approx(.1) and moved['top1'] == 1 and not moved['top1_same']
    assert moved['law'][~mask].sum() == 0 and set(moved) == {'tv', 'top1_same', 'kl_variant_reference', 'kl_sealed_reference',
                                                             'max_abs_q_difference', 'law', 'top1'}
    assert moved['kl_sealed_reference'] == pytest.approx(rows['same']['kl_variant_reference'])
