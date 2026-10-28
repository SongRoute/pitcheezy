"""Calibration chain of EXP-BC-CAL-01 (design section 4.1): B1 and B2, temperature-only B0-T.

Fit on the temperature split (2025-05-16..31) S_in rows, inside each row's support R_j:
(a) temperature p ~ exp(log p / T), T in [0.5, 2.5] (run_cuda_direct.py BOUNDS precedent);
(b) vector scaling p ~ exp(a_c log p_c + b_c), a_c > 0, plus 1e-3 * sum((a_c - 1)^2 + b_c^2).
Selection: 5-fold game-grouped CV inside the split (``cv_folds`` ported from
experiments/ws_seq/calibrate_logits.py:324, which is not modified); vector only when its mean CV
NLL is lower than temperature's by >= 0.0005, else temperature; the chosen method is refit on the
whole split. A vector failure (no convergence or a non-finite value) falls back to temperature and
is recorded; a temperature failure makes the candidate CALIBRATION_FAILED (not eligible). The
objective and analytic gradient follow calibrate_logits.vector_objective (:238) restricted to the
support. Probabilities stay 0 outside the support.
"""
from __future__ import annotations

import numpy as np

CALIBRATION_FAILED = 'CALIBRATION_FAILED'


class CalibrationFailure(RuntimeError):
    """An optimizer did not converge or produced a non-finite value."""


def _finite_logits(p, mask):
    p = np.asarray(p, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if p.shape != mask.shape or p.ndim != 2:
        raise ValueError('probabilities and support mask must be [rows, classes]')
    finite = mask & (p > 0)
    with np.errstate(divide='ignore'):
        z = np.where(finite, np.log(np.where(finite, p, 1.)), -np.inf)
    return z, finite


def _softmax(s, finite):
    s = np.where(finite, s, -np.inf)
    m = s.max(axis=1, keepdims=True)
    if not np.isfinite(m).all():
        raise ValueError('a row has no positive supported probability')
    e = np.exp(s - m)
    return e / e.sum(axis=1, keepdims=True)


class Temperature:
    method = 'temperature'

    def __init__(self, temperature):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        self.temperature = float(temperature)

    def apply(self, p, mask):
        z, finite = _finite_logits(p, mask)
        return _softmax(z / self.temperature, finite)

    def identity(self):
        return {'method': self.method, 'temperature': self.temperature}


class VectorScaling:
    method = 'vector'

    def __init__(self, a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all() or (a <= 0).any():
            raise ValueError('vector scaling needs finite a > 0 and finite b of one width')
        self.a, self.b = a, b

    def apply(self, p, mask):
        z, finite = _finite_logits(p, mask)
        return _softmax(np.where(finite, z, 0.) * self.a + self.b, finite)

    def identity(self):
        return {'method': self.method, 'a': self.a.tolist(), 'b': self.b.tolist()}


def calibrator_from_identity(identity):
    if identity['method'] == 'temperature':
        return Temperature(identity['temperature'])
    if identity['method'] == 'vector':
        return VectorScaling(identity['a'], identity['b'])
    raise ValueError(f'unknown calibrator {identity!r}')


def _labels_inside(finite, y):
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or len(y) != len(finite) or not len(y):
        raise CalibrationFailure('calibration needs one label per nonempty row')
    if not finite[np.arange(len(y)), y].all():
        raise CalibrationFailure('a calibration label has zero probability (outside the support)')
    return y


def heldout_nll(calibrator, p, mask, y):
    q = calibrator.apply(p, mask)
    y = np.asarray(y, dtype=np.int64)
    return float(-np.log(q[np.arange(len(y)), y]).mean())


def temperature_objective(z, finite, y):
    rows = np.arange(len(y))
    zf = np.where(finite, z, 0.)

    def fun(temperature):
        s = np.where(finite, zf / temperature, -np.inf)
        m = s.max(axis=1)
        return float(np.mean(m + np.log(np.exp(s - m[:, None]).sum(axis=1)) - s[rows, y]))
    return fun


def fit_temperature(p, mask, y, bounds=(0.5, 2.5)):
    """Bounded scalar NLL minimization (run_cuda_direct.fit_temperature form, inside the support)."""
    from scipy.optimize import minimize_scalar
    z, finite = _finite_logits(p, mask)
    y = _labels_inside(finite, y)
    try:
        result = minimize_scalar(temperature_objective(z, finite, y), bounds=tuple(bounds), method='bounded')
    except (ValueError, FloatingPointError, ArithmeticError) as error:
        raise CalibrationFailure(f'temperature optimizer error: {error}') from error
    if not result.success or not np.isfinite(result.x) or not np.isfinite(result.fun):
        raise CalibrationFailure(f'temperature optimizer failed: {result.message}')
    return Temperature(result.x), {'temperature': float(result.x), 'objective': float(result.fun),
                                   'iterations': int(getattr(result, 'nit', -1)),
                                   'at_bound': bool(np.isclose(result.x, bounds[0]) or np.isclose(result.x, bounds[1]))}


def vector_objective(z, finite, y, l2):
    """Mean NLL of s = a z + b on the support plus l2 * sum((a - 1)^2 + b^2); analytic gradient."""
    n, k = z.shape
    rows = np.arange(n)
    zf = np.where(finite, z, 0.)
    onehot = np.zeros((n, k))
    onehot[rows, y] = 1.

    def fun(theta):
        a, b = theta[:k], theta[k:]
        s = np.where(finite, zf * a + b, -np.inf)
        m = s.max(axis=1, keepdims=True)
        e = np.exp(s - m)
        total = e.sum(axis=1, keepdims=True)
        q = e / total
        f = float(np.mean(m[:, 0] + np.log(total[:, 0]) - s[rows, y])) + l2 * float(((a - 1) ** 2 + b ** 2).sum())
        g = (q - onehot) / n
        return f, np.concatenate([(g * zf).sum(axis=0) + 2 * l2 * (a - 1), g.sum(axis=0) + 2 * l2 * b])
    return fun


def fit_vector(p, mask, y, *, l2, temperature, lower=1e-6):
    """L-BFGS-B from the temperature solution (a = 1/T, b = 0); a_c >= ``lower`` > 0."""
    from scipy.optimize import minimize
    z, finite = _finite_logits(p, mask)
    y = _labels_inside(finite, y)
    k = z.shape[1]
    x0 = np.r_[np.full(k, 1. / temperature), np.zeros(k)]
    try:
        result = minimize(vector_objective(z, finite, y, l2), x0, jac=True, method='L-BFGS-B',
                          bounds=[(lower, None)] * k + [(None, None)] * k)
    except (ValueError, FloatingPointError, ArithmeticError) as error:
        raise CalibrationFailure(f'vector optimizer error: {error}') from error
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise CalibrationFailure(f'vector optimizer failed: {result.message}')
    a, b = result.x[:k], result.x[k:]
    return VectorScaling(a, b), {'objective': float(result.fun), 'iterations': int(result.nit),
                                 'message': str(result.message)}


def cv_folds(games, seed, k):
    """calibrate_logits.cv_folds: permute the sorted unique game_pks with default_rng(seed), split
    into k contiguous chunks (np.array_split); every row of a game is in one fold."""
    games = np.asarray(games)
    perm = np.random.default_rng(seed).permutation(np.unique(games))
    chunks = np.array_split(perm, k)
    fold = np.full(len(games), -1, np.int64)
    for f, chunk in enumerate(chunks):
        fold[np.isin(games, chunk)] = f
    if (fold < 0).any() or any(len(c) == 0 for c in chunks):
        raise ValueError('CV fold assignment needs at least k games')
    return fold


def calibrate(p, mask, y, games, *, bounds, l2, margin, k, seed, lower=1e-6, vector=True):
    """Registered chain for one candidate on the temperature split.

    Returns a JSON-ready record with ``status`` 'OK' or CALIBRATION_FAILED, the chosen ``method``,
    the calibrator ``identity`` and the CV table, plus the calibrator object under ``calibrator``.
    ``vector=False`` is the temperature-only descriptive B0-T."""
    record = {'rows': int(len(y)), 'games': int(len(np.unique(games))), 'bounds': list(bounds), 'l2': l2,
              'margin': margin, 'folds': k, 'seed': seed, 'fallbacks': [], 'cv': None}
    try:
        if vector:
            try:
                fold = cv_folds(games, seed, k)
            except ValueError as error:
                fold = None
                record['fallbacks'].append(f'cv_unavailable: {error}; temperature (vector not selectable)')
            if fold is not None:
                table, vector_ok = [], True
                for f in range(k):
                    train, test = fold != f, fold == f
                    t_model, _ = fit_temperature(p[train], mask[train], y[train], bounds)
                    row = {'fold': f, 'rows': int(test.sum()),
                           'temperature': heldout_nll(t_model, p[test], mask[test], y[test]), 'vector': None}
                    if vector_ok:
                        try:
                            v_model, _ = fit_vector(p[train], mask[train], y[train], l2=l2,
                                                    temperature=t_model.temperature, lower=lower)
                            row['vector'] = heldout_nll(v_model, p[test], mask[test], y[test])
                        except CalibrationFailure as error:
                            vector_ok = False
                            record['fallbacks'].append(f'vector_failed_in_cv_fold_{f}: {error}; temperature')
                    table.append(row)
                mean_t = float(np.mean([r['temperature'] for r in table]))
                mean_v = float(np.mean([r['vector'] for r in table])) if vector_ok else None
                choose_vector = vector_ok and (mean_t - mean_v) >= margin
                record['cv'] = {'folds': table, 'mean_temperature_nll': mean_t, 'mean_vector_nll': mean_v,
                                'vector_minus_temperature': None if mean_v is None else mean_v - mean_t,
                                'rule': f'vector iff mean_temperature - mean_vector >= {margin}; ties -> temperature',
                                'selected': 'vector' if choose_vector else 'temperature'}
            else:
                choose_vector = False
        else:
            choose_vector = False
        t_model, t_info = fit_temperature(p, mask, y, bounds)
        record['temperature_fit'] = t_info
        chosen = t_model
        if choose_vector:
            try:
                chosen, v_info = fit_vector(p, mask, y, l2=l2, temperature=t_model.temperature, lower=lower)
                record['vector_fit'] = v_info
            except CalibrationFailure as error:
                record['fallbacks'].append(f'vector_failed_on_refit: {error}; temperature')
                chosen = t_model
    except CalibrationFailure as error:
        record.update(status=CALIBRATION_FAILED, method=None, identity=None, calibrator=None,
                      failure=f'temperature optimizer failed: {error}')
        return record
    record.update(status='OK', method=chosen.method, identity=chosen.identity(), calibrator=chosen)
    return record


def public(record):
    """The JSON-ready part of a calibration record (without the live calibrator object)."""
    return {k: v for k, v in record.items() if k != 'calibrator'}
