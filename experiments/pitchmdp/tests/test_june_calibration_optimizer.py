"""Synthetic optimizer and family-fit gates for ML-JUNE-CALIBRATION-v1; no real data."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.optimize._optimize import _minimize_scalar_bounded

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pitchmdp import matrix_june_calibration as mj  # noqa: E402
from pitchmdp.matrix_g0_whole import frozen_weights  # noqa: E402
from run_sequence_calibration import fit_blend  # noqa: E402


def simplex(rng, n, sharpen=None, y=None):
    p = rng.gamma(1.0, size=(n, 10))
    if sharpen is not None:
        p[np.arange(n), y] += sharpen
    return p / p.sum(axis=1, keepdims=True)


def two_sided(n_model, n_frequency):
    """L(w) = -a log(w) - b log(1-w): model right on a rows, frequency right on b rows."""
    n = n_model + n_frequency
    y = np.zeros(n, dtype=np.int64)
    model, frequency = np.zeros((n, 10)), np.zeros((n, 10))
    model[:n_model, 0], frequency[:n_model, 1] = 1.0, 1.0
    model[n_model:, 1], frequency[n_model:, 0] = 1.0, 1.0
    return y, model, frequency


def test_explicit_options_are_the_pinned_scipy_defaults():
    defaults = {k: v.default for k, v in inspect.signature(_minimize_scalar_bounded).parameters.items()
                if k in ('xatol', 'maxiter', 'disp')}
    assert defaults == {k: mj.OPTIMIZER[k] for k in ('xatol', 'maxiter', 'disp')}


@pytest.mark.parametrize('seed,sharpen', [(0, 0.0), (1, 0.3), (2, 2.0), (3, 8.0)])
def test_wrapper_reproduces_legacy_fit_blend_bitwise(seed, sharpen):
    rng = np.random.default_rng(seed)
    n = 2500
    y = rng.integers(0, 10, n)
    model, frequency = simplex(rng, n, sharpen, y), simplex(rng, n)
    record = mj.fit_scalar_blend(y, model, frequency, label='t', kind='candidate')
    legacy = fit_blend(y, model, frequency, 'log_loss')
    assert record['weight'] == legacy['model_weight']
    assert record['objective'] == legacy['calibration_score']
    assert record['scipy_success'] is True and record['status'] == 'passed'
    assert record['scipy_nfev'] == record['optimizer_objective_evaluations']


def test_both_boundaries_are_exact_endpoints():
    rng = np.random.default_rng(4)
    n = 800
    y = rng.integers(0, 10, n)
    oracle = np.full((n, 10), 1e-3)
    oracle[np.arange(n), y] = 1 - 9e-3
    useless = simplex(rng, n)
    to_frequency = mj.fit_scalar_blend(y, useless, oracle, label='zero', kind='candidate')
    to_model = mj.fit_scalar_blend(y, oracle, useless, label='one', kind='candidate')
    assert (to_frequency['weight'], to_frequency['chosen_candidate']) == (0.0, '0')
    assert (to_model['weight'], to_model['chosen_candidate']) == (1.0, '1')
    assert to_model['weight'] == fit_blend(y, oracle, useless, 'log_loss')['model_weight']


def test_flat_objective_breaks_ties_in_listed_order():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 10, 300)
    same = simplex(rng, 300)
    record = mj.fit_scalar_blend(y, same, same.copy(), label='flat', kind='candidate')
    assert record['candidate_objectives'][0] == record['candidate_objectives'][2]
    assert (record['weight'], record['chosen_candidate']) == (0.0, '0')
    assert record['weight'] == fit_blend(y, same, same.copy(), 'log_loss')['model_weight']


def test_tiny_true_class_probabilities_use_the_floor_and_stay_finite():
    y, model, frequency = two_sided(40, 60)
    model[:5, 0], model[:5, 2] = 0.0, 1.0  # true class exactly zero in both components for five rows
    frequency[:5, 1], frequency[:5, 2] = 0.0, 1.0
    record = mj.fit_scalar_blend(y, model, frequency, label='floor', kind='candidate')
    assert np.isfinite(record['objective']) and record['status'] == 'passed'
    assert record['objective'] >= 5 / 100 * -np.log(1e-12) - 1e-12


def test_high_curvature_converged_solution_worse_than_precise_anchor_stops_family():
    y, model, frequency = two_sided(1, 9999)
    free = mj.fit_scalar_blend(y, model, frequency, label='free', kind='candidate')
    score = mj.blend_objective(y, model, frequency)
    precise = float(minimize_scalar(score, bounds=(0, 1), method='bounded', options={'xatol': 1e-14, 'maxiter': 5000}).x)
    assert free['scipy_success'] is True and free['objective'] - score(precise) > mj.OPTIMIZER['objective_tolerance']
    ledger = mj.CallLedger()
    with pytest.raises(mj.FamilyStop, match='worse_than_anchor') as stop:
        mj.fit_scalar_blend(y, model, frequency, label='B1:x', kind='candidate', anchor=precise, ledger=ledger)
    assert stop.value.diagnostics['anchor_gap'] > 1e-9 and ledger.count('candidate') == 1
    # The anchor is a validation point only: the same data without the anchor still returns the optimizer's choice.
    assert free['weight'] != precise


def _stub(result):
    def minimizer(fun, **kwargs):
        fun(0.5)
        return OptimizeResult(**result)
    return minimizer


@pytest.mark.parametrize('result,failure', [
    ({'x': 0.4, 'fun': 1.0, 'success': False, 'status': 1, 'nfev': 500, 'nit': 500, 'message': 'max iter'}, 'scipy_success_false'),
    ({'x': float('nan'), 'fun': 1.0, 'success': True, 'status': 0, 'nfev': 3, 'nit': 3, 'message': 'ok'}, 'optimum_not_finite_feasible'),
    ({'x': 1.2, 'fun': 1.0, 'success': True, 'status': 0, 'nfev': 3, 'nit': 3, 'message': 'ok'}, 'optimum_not_finite_feasible')])
def test_failed_or_infeasible_optimizer_result_is_refused_not_clipped(result, failure):
    rng = np.random.default_rng(6)
    y = rng.integers(0, 10, 200)
    ledger = mj.CallLedger()
    with pytest.raises(mj.FamilyStop, match=failure):
        mj.fit_scalar_blend(y, simplex(rng, 200), simplex(rng, 200), label='f', kind='candidate', ledger=ledger,
                            minimizer=_stub(result))
    assert ledger.snapshot()['total_calls'] == 1 and ledger.calls[0]['status'] == 'failed'
    assert ledger.calls[0]['optimizer_objective_evaluations'] == 1


def test_optimizer_exception_is_a_recorded_failed_call():
    def broken(fun, **kwargs):
        raise FloatingPointError('boom')
    rng = np.random.default_rng(7)
    ledger = mj.CallLedger()
    with pytest.raises(mj.FamilyStop, match='optimizer raised'):
        mj.fit_scalar_blend(rng.integers(0, 10, 50), simplex(rng, 50), simplex(rng, 50), label='e', kind='replay',
                            ledger=ledger, minimizer=broken)
    assert ledger.count('replay') == 1 and 'FloatingPointError' in ledger.calls[0]['failures'][0]


@pytest.mark.parametrize('mutate,message', [
    (lambda y, p: (y, p * (1 + 2e-6)), 'mass'),
    (lambda y, p: (y, np.where(np.arange(p.size).reshape(p.shape) == 0, -1e-3, p)), 'finite and in'),
    (lambda y, p: (y, np.where(np.arange(p.size).reshape(p.shape) == 0, np.nan, p)), 'finite and in'),
    (lambda y, p: (y, p[:, :9] / p[:, :9].sum(1, keepdims=True)), 'float64'),
    (lambda y, p: (y, p.astype(np.float32)), 'float64'),
    (lambda y, p: (y.astype(float), p), 'integer label'),
    (lambda y, p: (np.where(np.arange(len(y)) == 0, 10, y), p), 'outside')])
def test_malformed_inputs_fail_without_renormalization(mutate, message):
    rng = np.random.default_rng(8)
    y, p = rng.integers(0, 10, 100), simplex(rng, 100)
    y, bad = mutate(y, p)
    with pytest.raises(ValueError, match=message):
        mj.fit_scalar_blend(y, bad, simplex(rng, 100), label='m', kind='candidate')


# ---------------------------------------------------------------- family fit

def correlated_members(rng, n, y, strength):
    base = rng.gamma(1.0, size=(n, 10))
    base[np.arange(n), y] += strength
    members = [base + 0.3 * s * rng.gamma(1.0, size=(n, 10)) for s in mj.SEEDS]
    return np.stack([m / m.sum(axis=1, keepdims=True) for m in members])


def family_inputs(seed=11, n_june=2600, games=40, cpanel=300, small_group=None):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n_june)
    frequency = simplex(rng, n_june, 0.5, y)
    seeds = correlated_members(rng, n_june, y, 0.9)
    game_pk = np.tile(np.arange(games), n_june // games + 1)[:n_june] + 1000
    groups = np.asarray([mj.GROUPS[(i // games) % 4] for i in range(n_june)], dtype='<U6')
    if small_group:
        groups[groups == small_group] = 'high'
        groups[:40] = small_group
    ry = rng.integers(0, 10, cpanel)
    rf = simplex(rng, cpanel, 0.5, ry)
    rs = correlated_members(rng, cpanel, ry, 0.7)
    ensemble = mj.ensemble_mean(rs)
    components = {'ensemble': ensemble, **{f'seed{s}': rs[s] for s in mj.SEEDS}}
    weights = {name: fit_blend(ry, components[name], rf, 'log_loss')['model_weight'] for name in mj.PREDICTORS}
    replay = {'y': ry, 'frequency': rf, 'calibrated': ensemble, 'seed_calibrated': rs,
              'primary': mj.blend(weights['ensemble'], ensemble, rf),
              'seed_primary': np.stack([mj.blend(weights[f'seed{s}'], rs[s], rf) for s in mj.SEEDS])}
    june = {'y': y, 'frequency': frequency, 'seed_calibrated': seeds, 'groups': groups, 'game_pk': game_pk}
    return june, replay, weights


@pytest.fixture
def small_cpanel(monkeypatch):
    monkeypatch.setitem(mj.CPANEL, 'rows', 300)


def test_family_runs_6_replays_and_30_candidates_with_b2_anchored_on_new_b1(small_cpanel):
    june, replay, b0 = family_inputs()
    record = mj.fit_family(june=june, replay=replay, b0_weights=b0)
    assert mj.require_expected_calls(record) == mj.EXPECTED_CALLS
    kinds = [c['kind'] for c in record['call_ledger']['calls']]
    assert kinds == ['replay'] * 6 + ['candidate'] * 30
    for name in mj.PREDICTORS:
        assert abs(record['replay'][name]['weight'] - b0[name]) <= mj.WEIGHT_REPLAY_ATOL
        assert record['B1_fits'][name]['anchor'] == b0[name]
        for g in mj.GROUPS:
            fit = record['B2_fits'][g][name]
            n = record['B2_support'][g]['pitches']
            rho = n / (n + 1000)
            assert fit['record']['anchor'] == record['B1_weights'][name] != b0[name]
            assert record['B2_weights'][g][name] == rho * fit['local_mle'] + (1 - rho) * record['B1_weights'][name]
            assert record['B2_weights'][g][name] != rho * fit['local_mle'] + (1 - rho) * b0[name]
    assert record['B2_fallback_rows'] == 0 and record['fit_losses_are_quality_estimates'] is False


def test_valid_unsupported_group_falls_back_to_exact_b1_without_a_call(small_cpanel):
    june, replay, b0 = family_inputs(small_group='zero')
    record = mj.fit_family(june=june, replay=replay, b0_weights=b0)
    assert record['B2_support']['zero'] == {'pitches': 40, 'games': 40, 'supported': False}
    assert record['B2_fallback_groups'] == ['zero'] and record['B2_fallback_rows'] == 40
    assert record['call_ledger']['candidate_calls'] == 6 * 4
    for name in mj.PREDICTORS:
        assert record['B2_weights']['zero'][name] == record['B1_weights'][name]
        assert record['B2_fits']['zero'][name]['optimizer_call'] is False
    with pytest.raises(ValueError, match='support differs'):
        mj.require_expected_support(record['B2_support'])
    applied = mj.apply_family(record, seed_calibrated=june['seed_calibrated'], frequency=june['frequency'],
                              groups=june['groups'])
    rows = june['groups'] == 'zero'
    assert np.array_equal(applied['B2_primary'][rows], applied['B1_primary'][rows])
    assert np.array_equal(applied['B2_seed_primary'][:, rows], applied['B1_seed_primary'][:, rows])


def test_unknown_group_label_fails_closed(small_cpanel):
    june, replay, b0 = family_inputs()
    june['groups'] = june['groups'].copy()
    june['groups'][7] = 'unseen'
    with pytest.raises(mj.FamilyStop, match='Unknown TRAIN-volume'):
        mj.fit_family(june=june, replay=replay, b0_weights=b0)


def test_absent_from_train_is_zero_but_conflicting_metadata_fails():
    panel = {'volume_thresholds': {'q25': 170, 'q75': 1514},
             'train_players': [{'pitcher': 1, 'train_pitches': 100, 'train_volume': 'low'},
                               {'pitcher': 2, 'train_pitches': 900, 'train_volume': 'middle'},
                               {'pitcher': 3, 'train_pitches': 3000, 'train_volume': 'high'}]}
    lookup = mj.volume_lookup(panel)
    groups = mj.volume_groups(np.asarray([1, 2, 3, 99]), lookup, np.asarray(['low', 'middle', 'high', 'zero']))
    assert groups.tolist() == ['low', 'middle', 'high', 'zero']
    with pytest.raises(ValueError, match='conflicts'):
        mj.volume_groups(np.asarray([1, 99]), lookup, np.asarray(['low', 'low']))
    bad = {**panel, 'train_players': [{'pitcher': 1, 'train_pitches': 171, 'train_volume': 'low'}]}
    with pytest.raises(ValueError, match='Conflicting'):
        mj.volume_lookup(bad)
    with pytest.raises(ValueError, match='170/1514'):
        mj.volume_lookup({**panel, 'volume_thresholds': {'q25': 169, 'q75': 1514}})


def test_seed_order_swap_breaks_the_archived_weight_replay(small_cpanel):
    june, replay, b0 = family_inputs()
    swapped = dict(replay)
    swapped['seed_calibrated'] = replay['seed_calibrated'][[1, 0, 2, 3, 4]]
    swapped['calibrated'] = mj.ensemble_mean(swapped['seed_calibrated'])
    with pytest.raises(mj.FamilyStop, match='replay:seed0: weight differs') as stop:
        mj.fit_family(june=june, replay=swapped, b0_weights=b0)
    assert stop.value.diagnostics['ledger']['candidate_calls'] == 0


def test_three_seed_weights_are_never_substituted(small_cpanel):
    report = {'selection': {'model_weight': 0.7, 'calibration_objective': 'log_loss'},
              'seeds': [{'blend_selection': {'model_weight': 0.7}}] * 3}
    with pytest.raises(ValueError, match='five ordered seed'):
        frozen_weights(report, {str(s): 1.0 for s in mj.SEEDS})
    june, replay, b0 = family_inputs()
    substituted = {**b0, 'seed3': b0['seed0'], 'seed4': b0['seed1']}
    with pytest.raises(mj.FamilyStop, match='replay:seed3'):
        mj.fit_family(june=june, replay=replay, b0_weights=substituted)


def test_archived_b0_prediction_replay_is_checked_to_1e6(small_cpanel):
    june, replay, b0 = family_inputs()
    tampered = dict(replay)
    tampered['primary'] = replay['primary'].copy()
    tampered['primary'][0, :2] += np.asarray([2e-6, -2e-6])
    with pytest.raises(mj.FamilyStop, match='B0 Cpanel prediction replay'):
        mj.fit_family(june=june, replay=tampered, b0_weights=b0)


def test_failure_mid_family_stops_everything_and_keeps_the_call_count(small_cpanel):
    june, replay, b0 = family_inputs()
    calls = []

    def failing(fun, **kwargs):
        calls.append(1)
        result = minimize_scalar(fun, **kwargs)
        if len(calls) == 8:
            result.success = False
        return result
    ledger = mj.CallLedger()
    with pytest.raises(mj.FamilyStop, match='B1:seed0'):
        mj.fit_family(june=june, replay=replay, b0_weights=b0, minimizer=failing, ledger=ledger)
    snapshot = ledger.snapshot()
    assert (snapshot['replay_calls'], snapshot['candidate_calls'], snapshot['total_calls']) == (6, 2, 8)
    assert snapshot['calls'][-1]['status'] == 'failed'


def test_apply_is_label_free_and_rejects_malformed_components(small_cpanel):
    june, replay, b0 = family_inputs()
    record = mj.fit_family(june=june, replay=replay, b0_weights=b0)
    assert 'labels' not in inspect.signature(mj.apply_family).parameters
    out = mj.apply_family(record, seed_calibrated=june['seed_calibrated'], frequency=june['frequency'], groups=june['groups'])
    ensemble = mj.ensemble_mean(june['seed_calibrated'])
    assert np.array_equal(out['B0_primary'], b0['ensemble'] * ensemble + (1 - b0['ensemble']) * june['frequency'])
    assert out['B2_seed_primary'].shape == (5, len(june['y']), 10)
    bad = june['frequency'].copy()
    bad[0] *= 1.01
    with pytest.raises(ValueError, match='mass'):
        mj.apply_family(record, seed_calibrated=june['seed_calibrated'], frequency=bad, groups=june['groups'])
    groups = june['groups'].copy()
    groups[0] = 'unknown'
    with pytest.raises(ValueError, match='Unknown'):
        mj.apply_family(record, seed_calibrated=june['seed_calibrated'], frequency=june['frequency'], groups=groups)


def test_shuffled_or_duplicate_keys_are_rejected():
    keys = np.asarray([[1, 1, 1], [1, 1, 2], [2, 1, 1]])
    assert mj.subset_positions(keys, keys[[2, 0]]).tolist() == [2, 0]
    with pytest.raises(ValueError, match='repeated'):
        mj.key_array(np.vstack([keys, keys[:1]]))
    with pytest.raises(ValueError, match='absent'):
        mj.subset_positions(keys, np.asarray([[3, 1, 1]]))
    assert mj.ordered_key_sha256(keys) != mj.ordered_key_sha256(keys[[1, 0, 2]])
