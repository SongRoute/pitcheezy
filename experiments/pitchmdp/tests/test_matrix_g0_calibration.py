"""Synthetic checks for the two fixed stage-3 G0 calibration corrections. No real data."""
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax

from pitchmdp import matrix_g0_calibration as cal

N, C = 4821, 10
SEEDS = ['ensemble', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4']


def june(seed=0, bias=None):
    """Rows whose labels follow ``truth``; ``pred`` is truth distorted by a class bias."""
    rng = np.random.default_rng(seed)
    i = np.arange(N)
    keys = np.column_stack([600000 + i % 150, i // 150 + 1, np.ones(N, int)]).astype(np.int64)
    truth = softmax(rng.normal(0, .8, (N, C)), axis=1)
    y = np.array([rng.choice(C, p=row) for row in truth], dtype=np.int64)
    pred = truth if bias is None else softmax(np.log(truth) + np.asarray(bias), axis=1)
    return keys, y, truth, pred


def config_for(keys, y):
    config = cal.fixed_spec()
    config['fit_population'] = {'ordered_keys_sha256': cal.array_sha256(keys, np.int64),
                                'labels_sha256': cal.array_sha256(y, np.int64)}
    return config


def nll(y, p):
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def fit_i1(keys, y, p, config, /, **override):
    args = dict(probabilities=p, prediction_keys=keys, labels=y, keys=keys, predictor='ensemble',
                split_token='june_cpanel', config=config)
    args.update(override)
    return cal.fit_class_bias_june(**args)


@pytest.fixture(scope='module')
def weak_bias():
    b = np.array([.3, -.3, .2, -.2, .1, -.1, 0, 0, 0, 0])
    keys, y, truth, pred = june(1, b)
    return b, keys, y, truth, pred, config_for(keys, y)


# ---------------------------------------------------------------- config

def test_config_requires_every_fixed_number_exactly(weak_bias):
    config = weak_bias[-1]
    assert cal.validate_config(config) is config
    mutations = [
        lambda c: c['i1_class_bias'].update(penalty=.02),
        lambda c: c['i1_class_bias'].update(maxiter=1000),
        lambda c: c['i1_class_bias'].update(ftol='1e-10'),
        lambda c: c['i1_class_bias'].pop('optimizer'),
        lambda c: c['i2_volume_blend'].update(min_pitches=500.0),
        lambda c: c['i2_volume_blend'].update(shrinkage_pitches=True),
        lambda c: c['i2_volume_blend'].update(groups=['low', 'middle', 'high']),
        lambda c: c.update(predictors=SEEDS[:4]),
        lambda c: c.update(fit_split_token='mlb_dev'),
        lambda c: c.update(extra=1),
        lambda c: c['fit_population'].update(labels_sha256='abc'),
        lambda c: c.pop('fit_population'),
        lambda c: c['indicators'].update(i2_positive_groups=['zero', 'low', 'middle', 'high']),
    ]
    for mutate in mutations:
        broken = json.loads(json.dumps(config))
        mutate(broken)
        with pytest.raises(ValueError):
            cal.validate_config(broken)


# ---------------------------------------------------------------- I1

def test_i1_analytic_gradient_matches_central_differences():
    rng = np.random.default_rng(3)
    log_p = np.log(softmax(rng.normal(size=(200, C)), axis=1))
    y = rng.integers(0, C, 200)
    b = rng.uniform(-.4, .4, C)
    _, grad = cal.class_bias_objective(b, log_p, y, .01)
    eps = 1e-6
    numeric = np.array([(cal.class_bias_objective(b + eps*e, log_p, y, .01)[0]
                         - cal.class_bias_objective(b - eps*e, log_p, y, .01)[0]) / (2*eps) for e in np.eye(C)])
    np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-9)


def test_i1_objective_equals_documented_formula():
    rng = np.random.default_rng(4)
    p = softmax(rng.normal(size=(50, C)), axis=1)
    p[0, 3] = 0.
    p[0] /= p[0].sum()
    y = rng.integers(0, C, 50)
    b = rng.uniform(-.5, .5, C)
    q = softmax(np.log(np.maximum(p, 1e-12)) + b, axis=1)
    value, _ = cal.class_bias_objective(b, np.log(np.maximum(p, 1e-12)), y, .01)
    assert value == pytest.approx(nll(y, q) + .01*np.mean(b**2), abs=1e-12)


def test_i1_corrects_weak_class_bias_with_constraints(weak_bias):
    b_true, keys, y, truth, pred, config = weak_bias
    fit = fit_i1(keys, y, pred, config)
    b = np.array(fit['parameters']['bias'])
    report = fit['optimizer_report']
    assert fit['fit_success'] and report['feasible'] and report['finite'] and report['not_worse_than_initial']
    assert abs(b.sum()) <= 1e-9 and np.all(np.abs(b) <= .5 + 1e-9)
    assert report['objective_final'] <= report['objective_initial'] + 1e-9
    assert np.corrcoef(b, -b_true)[0, 1] > .9 and np.max(np.abs(b + b_true)) < .15
    corrected = cal.apply_class_bias(probabilities=pred, parameters=fit, config=config)
    assert nll(y, corrected) < nll(y, pred)
    assert report['method'] == 'SLSQP' and report['ftol'] == 1e-10 and report['maxiter'] == 500
    assert fit['fit_population']['data_origin_proof'] == 'hash_match_to_config_pins_only'
    json.dumps(fit, allow_nan=False)


def test_i1_large_bias_is_held_at_bounds():
    keys, y, truth, pred = june(2, [2., -2., 0, 0, 0, 0, 0, 0, 0, 0])
    config = config_for(keys, y)
    fit = fit_i1(keys, y, pred, config)
    b = np.array(fit['parameters']['bias'])
    assert fit['fit_success']
    assert b[0] == pytest.approx(-.5, abs=1e-8) and b[1] == pytest.approx(.5, abs=1e-8)
    assert abs(b.sum()) <= 1e-9 and np.all(np.abs(b) <= .5 + 1e-9)


def test_i1_zero_bias_is_identity_and_apply_has_no_labels(weak_bias):
    _, keys, y, _, pred, config = weak_bias
    fit = fit_i1(keys, y, pred, config)
    zero = json.loads(json.dumps(fit))
    zero['parameters']['bias'] = [0.] * C
    np.testing.assert_allclose(cal.apply_class_bias(probabilities=pred, parameters=zero, config=config),
                               pred, rtol=0, atol=1e-14)
    assert 'labels' not in inspect.signature(cal.apply_class_bias).parameters
    assert 'labels' not in inspect.signature(cal.apply_volume_blend).parameters
    # Any row count (e.g. DEV) is accepted at apply time; output is a simplex and row-wise.
    other = softmax(np.random.default_rng(9).normal(size=(37, C)), axis=1)
    out = cal.apply_class_bias(probabilities=other, parameters=fit, config=config)
    np.testing.assert_allclose(out.sum(1), 1, atol=1e-12)
    np.testing.assert_array_equal(out[5:9], cal.apply_class_bias(probabilities=other[5:9], parameters=fit,
                                                                   config=config))
    reloaded = json.loads(json.dumps(fit))
    np.testing.assert_array_equal(out, cal.apply_class_bias(probabilities=other, parameters=reloaded,
                                                            config=config))


def test_i1_fit_rejects_non_june_or_misaligned_rows(weak_bias):
    _, keys, y, _, pred, config = weak_bias
    dev_keys = keys.copy()
    dev_keys[:, 0] += 1000000
    y_mut = y.copy()
    y_mut[0] = (y_mut[0] + 1) % C
    dup = keys.copy()
    dup[1] = dup[0]
    bad_mass = pred.copy()
    bad_mass[0, 0] += .01
    nan = pred.copy()
    nan[0, 0] = np.nan
    cases = [dict(split_token='mlb_dev'), dict(split_token='june'),
             dict(keys=dev_keys, prediction_keys=dev_keys), dict(labels=y_mut),
             dict(prediction_keys=keys[::-1]), dict(keys=dup, prediction_keys=dup),
             dict(keys=keys[:-1], prediction_keys=keys[:-1], labels=y[:-1], probabilities=pred[:-1]),
             dict(probabilities=bad_mass), dict(probabilities=nan), dict(probabilities=pred[:, :9]),
             dict(labels=np.where(np.arange(N) == 0, C, y)), dict(labels=y.astype(float)),
             dict(predictor='seed5')]
    for override in cases:
        with pytest.raises(ValueError):
            fit_i1(keys, y, pred, config, **override)


def test_i1_optimizer_failure_is_preserved_and_not_applicable(weak_bias, monkeypatch):
    _, keys, y, _, pred, config = weak_bias
    failed = SimpleNamespace(x=np.full(C, np.nan), success=False, status=9, message='Iteration limit',
                             nit=500, nfev=501, njev=500)
    monkeypatch.setattr(cal, 'minimize', lambda *a, **k: failed)
    fit = fit_i1(keys, y, pred, config)
    assert not fit['fit_success'] and fit['parameters']['bias'] is None
    assert fit['optimizer_report']['message'] == 'Iteration limit'
    json.dumps(fit, allow_nan=False)
    with pytest.raises(ValueError, match='Failed fits'):
        cal.apply_class_bias(probabilities=pred, parameters=fit, config=config)
    worse = SimpleNamespace(x=np.array([.5, -.5] + [0.] * 8), success=True, status=0, message='ok',
                            nit=1, nfev=1, njev=1)
    monkeypatch.setattr(cal, 'minimize', lambda *a, **k: worse)
    _, _, _, calibrated = june(5)
    config5 = config_for(keys, june(5)[1])
    fit = fit_i1(keys, june(5)[1], calibrated, config5)
    assert fit['optimizer_report']['feasible'] and not fit['optimizer_report']['not_worse_than_initial']
    assert not fit['fit_success']
    infeasible = SimpleNamespace(x=np.array([.6, -.6] + [0.] * 8), success=True, status=0, message='ok',
                                 nit=1, nfev=1, njev=1)
    monkeypatch.setattr(cal, 'minimize', lambda *a, **k: infeasible)
    assert not fit_i1(keys, y, pred, config)['optimizer_report']['feasible']


def test_i1_apply_rejects_parameters_from_another_config(weak_bias):
    _, keys, y, _, pred, config = weak_bias
    fit = fit_i1(keys, y, pred, config)
    other = json.loads(json.dumps(config))
    other['fit_population']['labels_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='different config'):
        cal.apply_class_bias(probabilities=pred, parameters=fit, config=other)
    tampered = json.loads(json.dumps(fit))
    tampered['parameters']['bias'][0] = .7
    with pytest.raises(ValueError, match='Invalid frozen'):
        cal.apply_class_bias(probabilities=pred, parameters=tampered, config=config)


def test_i1_family_fits_each_of_five_seeds_and_ensemble_separately(weak_bias):
    _, keys, y, truth, _, config = weak_bias
    predictions = {name: softmax(np.log(truth) + np.roll([.2, -.2] + [0] * 8, i), axis=1)
                   for i, name in enumerate(SEEDS)}
    family = cal.fit_class_bias_family(predictions=predictions, prediction_keys=keys, labels=y, keys=keys,
                                       split_token='june_cpanel', config=config)
    assert [f['predictor'] for f in family['fits']] == SEEDS
    biases = np.array([f['parameters']['bias'] for f in family['fits']])
    assert all(f['fit_success'] for f in family['fits'])
    assert np.argmin(biases[0]) == 0 and np.argmin(biases[3]) == 3
    assert len({f['prediction_sha256'] for f in family['fits']}) == 6
    for broken in ({k: predictions[k] for k in SEEDS[:4]}, {k: predictions[k] for k in SEEDS[::-1]}):
        with pytest.raises(ValueError, match='registered order'):
            cal.fit_class_bias_family(predictions=broken, prediction_keys=keys, labels=y, keys=keys,
                                      split_token='june_cpanel', config=config)


# ---------------------------------------------------------------- I2

@pytest.fixture(scope='module')
def optimizer(weak_bias):
    return cal.load_blend_optimizer(weak_bias[-1])


def volume_rows(zero_rows=0):
    """zero absent (or supported), low <500 pitches, middle <30 games, high supported."""
    rng = np.random.default_rng(11)
    i = np.arange(N)
    keys = np.column_stack([600000 + i % 150, i // 150 + 1, np.ones(N, int)]).astype(np.int64)
    game = i % 150
    groups = np.full(N, 'high', dtype=object)
    groups[game < 20] = 'middle'
    low = np.flatnonzero(game >= 20)[:400]
    groups[low] = 'low'
    if zero_rows:
        groups[np.flatnonzero(groups == 'high')[:zero_rows]] = 'zero'
    truth = softmax(rng.normal(0, 1., (N, C)), axis=1)
    y = np.array([rng.choice(C, p=row) for row in truth], dtype=np.int64)
    network = softmax(np.log(truth) * 1.6, axis=1)
    frequency = np.tile(np.bincount(y, minlength=C) / N, (N, 1))
    return keys, y, groups, network, frequency


def fit_i2(keys, y, groups, network, frequency, config, optimizer, /, **override):
    args = dict(calibrated=network, prediction_keys=keys, frequency=frequency, frequency_keys=keys, labels=y,
                keys=keys, groups=groups, global_weight=.55, predictor='ensemble', split_token='june_cpanel',
                config=config, blend_optimizer=optimizer)
    args.update(override)
    return cal.fit_volume_blend_june(**args)


def test_i2_shrinks_supported_groups_and_falls_back_exactly(optimizer):
    keys, y, groups, network, frequency = volume_rows()
    config = config_for(keys, y)
    fit = fit_i2(keys, y, groups, network, frequency, config, optimizer)
    rows = {r['group']: r for r in fit['parameters']['groups']}
    assert [r['group'] for r in fit['parameters']['groups']] == ['zero', 'low', 'middle', 'high']
    assert rows['zero']['june_pitches'] == 0 and rows['zero']['weight'] == .55 and rows['zero']['fallback']
    assert rows['low']['june_pitches'] == 400 and not rows['low']['supported'] and rows['low']['weight'] == .55
    assert rows['middle']['june_games'] == 20 and rows['middle']['june_pitches'] >= 500
    assert not rows['middle']['supported'] and rows['middle']['weight'] == .55
    assert rows['low']['mle_weight'] is None and rows['middle']['mle_report'] is None
    high = rows['high']
    mask = groups == 'high'
    n = int(mask.sum())
    mle = optimizer.function(y[mask], network[mask], frequency[mask], 'log_loss')['model_weight']
    assert high['supported'] and high['mle_weight'] == mle and 0 < mle < 1
    assert high['weight'] == pytest.approx(n/(n+1000)*mle + 1000/(n+1000)*.55, abs=1e-15)
    assert fit['blend_optimizer']['sha256'] == cal.FIXED_SPEC['i2_volume_blend']['blend_optimizer_sha256']
    fit = json.loads(json.dumps(fit, allow_nan=False))
    out = cal.apply_volume_blend(calibrated=network, frequency=frequency, groups=groups, keys=keys,
                                 calibrated_keys=keys, frequency_keys=keys, parameters=fit, config=config)
    unsupported = groups != 'high'
    np.testing.assert_array_equal(out[unsupported], (.55*network + (1-.55)*frequency)[unsupported])
    np.testing.assert_allclose(out[mask], high['weight']*network[mask] + (1-high['weight'])*frequency[mask],
                               rtol=0, atol=1e-15)
    # Zero-TRAIN DEV rows absent from June still apply with exactly the global weight.
    zero = np.full(5, 'zero', dtype=object)
    np.testing.assert_array_equal(
        cal.apply_volume_blend(calibrated=network[:5], frequency=frequency[:5], groups=zero, keys=keys[:5],
                               calibrated_keys=keys[:5], frequency_keys=keys[:5], parameters=fit, config=config),
        .55*network[:5] + (1-.55)*frequency[:5])


def test_i2_fits_supported_zero_group(optimizer):
    keys, y, groups, network, frequency = volume_rows(zero_rows=800)
    config = config_for(keys, y)
    rows = {r['group']: r for r in fit_i2(keys, y, groups, network, frequency, config, optimizer)
            ['parameters']['groups']}
    assert rows['zero']['supported'] and rows['zero']['june_pitches'] == 800
    assert rows['zero']['mle_weight'] is not None and rows['zero']['group_fraction'] == 800/1800


def test_i2_rejects_bad_groups_rows_weights_and_optimizers(optimizer):
    keys, y, groups, network, frequency = volume_rows()
    config = config_for(keys, y)
    unknown = groups.copy()
    unknown[3] = 'unknown'
    missing = groups.copy()
    missing[3] = None
    fake = cal.BlendOptimizer(optimizer.function, optimizer.source, optimizer.function_name, '0' * 64)
    cases = [dict(groups=unknown), dict(groups=missing), dict(groups=groups[:-1]),
             dict(prediction_keys=keys[::-1]), dict(frequency_keys=keys[::-1]),
             dict(global_weight=1.2), dict(global_weight=float('nan')), dict(global_weight=True),
             dict(blend_optimizer=fake), dict(blend_optimizer=optimizer.function),
             dict(split_token='mlb_dev'), dict(labels=y[::-1].copy())]
    for override in cases:
        with pytest.raises(ValueError):
            fit_i2(keys, y, groups, network, frequency, config, optimizer, **override)
    fit = fit_i2(keys, y, groups, network, frequency, config, optimizer)
    apply = dict(calibrated=network, frequency=frequency, groups=groups, keys=keys, calibrated_keys=keys,
                 frequency_keys=keys, parameters=fit, config=config)
    tampered = json.loads(json.dumps(fit))
    tampered['parameters']['groups'][1]['weight'] = .9
    dropped = json.loads(json.dumps(fit))
    dropped['parameters']['groups'].pop(0)
    for override in (dict(groups=unknown), dict(calibrated_keys=keys[::-1]), dict(frequency_keys=keys[::-1]),
                     dict(parameters=tampered), dict(parameters=dropped)):
        with pytest.raises(ValueError):
            cal.apply_volume_blend(**{**apply, **override})


def test_i2_family_uses_each_predictors_archived_weight(optimizer):
    keys, y, groups, network, frequency = volume_rows()
    config = config_for(keys, y)
    calibrated = {name: softmax(np.log(network) * (1 - .05*i), axis=1) for i, name in enumerate(SEEDS)}
    weights = dict(zip(SEEDS, [.5, .4, .45, .6, .35, .55]))
    family = cal.fit_volume_blend_family(calibrated=calibrated, prediction_keys=keys, frequency=frequency,
                                         frequency_keys=keys, labels=y, keys=keys, groups=groups,
                                         global_weights=weights, split_token='june_cpanel', config=config,
                                         blend_optimizer=optimizer)
    assert [f['predictor'] for f in family['fits']] == SEEDS
    for f in family['fits']:
        rows = {r['group']: r for r in f['parameters']['groups']}
        assert f['parameters']['global_weight'] == weights[f['predictor']] == rows['low']['weight']
    assert len({f['parameters']['groups'][3]['mle_weight'] for f in family['fits']}) > 1
    three = {k: weights[k] for k in SEEDS[:4]}
    with pytest.raises(ValueError, match='registered order'):
        cal.fit_volume_blend_family(calibrated=calibrated, prediction_keys=keys, frequency=frequency,
                                    frequency_keys=keys, labels=y, keys=keys, groups=groups,
                                    global_weights=three, split_token='june_cpanel', config=config,
                                    blend_optimizer=optimizer)


def test_blend_optimizer_loader_checks_pinned_source(tmp_path, weak_bias):
    config = weak_bias[-1]
    source = cal.PROJECT / 'scripts/run_sequence_calibration.py'
    (tmp_path / 'scripts').mkdir()
    shutil.copy(source, tmp_path / 'scripts/run_sequence_calibration.py')
    with open(tmp_path / 'scripts/run_sequence_calibration.py', 'a') as handle:
        handle.write('\n# altered\n')
    with pytest.raises(ValueError, match='pinned hash'):
        cal.load_blend_optimizer(config, project_root=tmp_path)
    loaded = cal.load_blend_optimizer(config)
    assert Path(loaded.function.__code__.co_filename).resolve() == source.resolve()


# ---------------------------------------------------------------- indicators

def class_diag(gaps, events=100):
    return [{'class': c, 'events': events if not isinstance(events, list) else events[c],
             'observed_prevalence': .1, 'predicted_prevalence': .1 + gaps[c]} for c in range(C)]


def volume_diag(deltas, june=(40, 600), dev=(40, 600)):
    return [{'group': g, 'june_games': june[0], 'june_pitches': june[1], 'dev_games': dev[0],
             'dev_pitches': dev[1], 'dev_g0_minus_frequency_nll': d} for g, d in deltas.items()]


def test_weakness_indicators_follow_fixed_thresholds(weak_bias):
    config = weak_bias[-1]
    flat = volume_diag({'zero': -.1, 'low': -.01, 'middle': -.011, 'high': -.012})
    run = lambda c, v: cal.weakness_indicators(class_diagnostics=c, volume_diagnostics=v, config=config)
    assert run(class_diag([0]*9 + [.0011]), flat)['I1']['status'] == 'met'
    assert run(class_diag([0]*9 + [.0009]), flat)['I1']['status'] == 'not_met'
    # A large gap in an under-supported class does not qualify.
    assert run(class_diag([0]*9 + [.05], events=[100]*9 + [29]), flat)['I1']['status'] == 'not_met'
    assert run(class_diag([.05]*10, events=29), flat)['I1']['status'] == 'unmeasured'
    # Zero group excluded even though its gap is large; positive range .002 < .003.
    result = run(class_diag([0]*10), flat)['I2']
    assert result['status'] == 'not_met' and result['nll_range'] == pytest.approx(.002)
    wide = volume_diag({'zero': 0., 'low': -.01, 'middle': -.02, 'high': -.0135})
    assert run(class_diag([0]*10), wide)['I2']['status'] == 'met'
    thin = volume_diag({'zero': 0., 'low': -.01, 'middle': -.02, 'high': -.0135})
    thin[2]['dev_games'] = 29
    thin[3]['june_pitches'] = 499
    assert run(class_diag([0]*10), thin)['I2']['status'] == 'unmeasured'
    absent = volume_diag({'zero': None, 'low': -.01, 'middle': -.02, 'high': -.0135})
    absent[0].update(june_games=0, june_pitches=0, dev_games=0, dev_pitches=0)
    assert run(class_diag([0]*10), absent)['I2']['status'] == 'met'
    for c, v in [(class_diag([0]*10)[:-1], flat), (class_diag([0]*10), flat[:-1]),
                 (class_diag([0]*10), flat + flat[:1]), (class_diag([float('nan')]*10), flat),
                 (class_diag([0]*10), volume_diag({'zero': 0., 'low': None, 'middle': 0., 'high': 0.}))]:
        with pytest.raises(ValueError):
            run(c, v)
    json.dumps(run(class_diag([0]*10), flat), allow_nan=False)
