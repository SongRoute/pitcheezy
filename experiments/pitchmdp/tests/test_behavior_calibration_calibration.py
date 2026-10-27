"""EXP-BC-CAL-01 calibration chain on synthetic logits (T7): temperature recovery, vector scaling
inside the support, the CV selection margin, fallbacks and the CALIBRATION_FAILED path."""
import numpy as np
import pytest

from pitchmdp import behavior_calibration as bc

K = 18
SPEC = {'bounds': (0.5, 2.5), 'l2': 1e-3, 'margin': 5e-4, 'k': 5, 'seed': 20260924}


def softmax(z, mask):
    z = np.where(mask, z, -np.inf)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def synthetic(n=12000, temperature=1.6, bias=None, seed=0):
    """Labels drawn from the true law on a random support per row; the model is overconfident by
    ``temperature`` (and optionally shifted by per-class ``bias``)."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, K)) < .45
    mask[np.arange(n), rng.integers(K, size=n)] = True
    z = rng.normal(size=(n, K)) * 1.2
    true = softmax(z, mask)
    y = np.array([rng.choice(K, p=p) for p in true])
    model_logits = z * temperature + (0 if bias is None else bias)
    games = rng.integers(0, 300, size=n)
    return softmax(model_logits, mask), mask, y, games


def test_t7_temperature_is_recovered_and_support_is_kept():
    p, mask, y, _ = synthetic()
    model, info = bc.fit_temperature(p, mask, y)
    assert abs(model.temperature - 1.6) < 0.08 and not info['at_bound']
    q = model.apply(p, mask)
    assert (q[~mask] == 0).all() and np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-12)
    assert bc.heldout_nll(model, p, mask, y) < bc.heldout_nll(bc.Temperature(1.), p, mask, y)


def test_t7_vector_scaling_preserves_the_mask_and_gradient_is_exact():
    p, mask, y, _ = synthetic(n=3000, bias=np.linspace(-.6, .6, K))
    z, finite = bc._finite_logits(p, mask)
    fun = bc.vector_objective(z, finite, y, 1e-3)
    theta = np.r_[np.full(K, .7), np.linspace(-.1, .1, K)]
    f, g = fun(theta)
    step = 1e-6
    numeric = np.array([(fun(theta + step * e)[0] - fun(theta - step * e)[0]) / (2 * step) for e in np.eye(2 * K)])
    assert np.allclose(g, numeric, rtol=1e-5, atol=1e-7)
    model, _ = bc.fit_vector(p, mask, y, l2=1e-3, temperature=1.6)
    q = model.apply(p, mask)
    assert (q[~mask] == 0).all() and np.allclose(q.sum(axis=1), 1, rtol=0, atol=1e-12) and (model.a > 0).all()


def test_cv_selection_vector_only_with_the_registered_margin():
    p, mask, y, games = synthetic(bias=np.linspace(-1., 1., K))
    record = bc.calibrate(p, mask, y, games, **SPEC)
    assert record['status'] == 'OK' and record['cv']['selected'] == 'vector' and record['method'] == 'vector'
    assert record['cv']['mean_temperature_nll'] - record['cv']['mean_vector_nll'] >= 5e-4
    huge = bc.calibrate(p, mask, y, games, **{**SPEC, 'margin': 10.})  # gain below the margin -> temperature
    assert huge['method'] == 'temperature' and huge['cv']['selected'] == 'temperature'
    t_only = bc.calibrate(p, mask, y, games, **SPEC, vector=False)  # B0-T: temperature only
    assert t_only['method'] == 'temperature' and t_only['cv'] is None
    assert bc.calibrator_from_identity(record['identity']).apply(p[:5], mask[:5]).shape == (5, K)
    folds = bc.cv_folds(games, 20260924, 5)
    assert all(len(set(folds[games == g])) == 1 for g in np.unique(games)[:20])


def test_vector_failure_falls_back_to_temperature(monkeypatch):
    p, mask, y, games = synthetic(n=4000, bias=np.linspace(-1., 1., K))

    def failing(*args, **kwargs):
        raise bc.CalibrationFailure('vector optimizer failed: synthetic')
    monkeypatch.setattr(bc, 'fit_vector', failing)
    record = bc.calibrate(p, mask, y, games, **SPEC)
    assert record['status'] == 'OK' and record['method'] == 'temperature'
    assert any('vector_failed_in_cv_fold_0' in f for f in record['fallbacks'])


def test_temperature_failure_is_calibration_failed(monkeypatch):
    p, mask, y, games = synthetic(n=3000)
    import scipy.optimize

    class Result:
        success, x, fun, message, nit = False, 1.0, np.nan, 'synthetic failure', 0
    monkeypatch.setattr(scipy.optimize, 'minimize_scalar', lambda *a, **k: Result())
    record = bc.calibrate(p, mask, y, games, **SPEC)
    assert record['status'] == bc.CALIBRATION_FAILED and record['calibrator'] is None and record['method'] is None
    assert 'temperature optimizer failed' in record['failure']
    assert 'calibrator' not in bc.public(record)


def test_label_outside_the_support_cannot_be_calibrated():
    p, mask, y, games = synthetic(n=500)
    outside = np.flatnonzero(~mask[0])[0]
    y = y.copy()
    y[0] = outside
    assert bc.calibrate(p, mask, y, games, **SPEC)['status'] == bc.CALIBRATION_FAILED
    with pytest.raises(bc.CalibrationFailure):
        bc.fit_temperature(p, mask, y)
