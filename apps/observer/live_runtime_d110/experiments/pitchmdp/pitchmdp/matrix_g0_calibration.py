"""Two fixed stage-3 probability corrections for the frozen G0 predictor.

I1 is a bounded, zero-sum class-bias correction on log probabilities. I2 shrinks
per-TRAIN-volume-group June blend weights toward the archived global weight.
Both are fitted only on the registered June Cpanel rows and applied without labels.

Source code cannot prove where arrays came from. Fits compare hashes of the
supplied keys and labels with config pins; the caller (root) must pin those hashes
to the immutable June Cpanel split before any fit. Weakness indicators consume
precomputed stage-2 diagnostics and never fit or score DEV rows.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Callable, NamedTuple

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import logsumexp

SPEC_VERSION = 'g0-calibration-stage3-v1'
PROJECT = Path(__file__).resolve().parents[1]

# Fixed settings from science-draft-for-opus.md (draft v1) stage 3. A config must
# restate each value exactly; this constant is the reference, not a silent default.
FIXED_SPEC = {
    'spec_version': SPEC_VERSION,
    'n_classes': 10,
    'predictors': ['ensemble', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4'],
    'fit_split_token': 'june_cpanel',
    'fit_rows': 4821,
    'simplex_atol': 1e-6,
    'i1_class_bias': {
        'probability_floor': 1e-12, 'bias_lower': -0.5, 'bias_upper': 0.5, 'penalty': 0.01,
        'initialization': 'zeros', 'optimizer': 'SLSQP', 'ftol': 1e-10, 'maxiter': 500,
        'objective_tolerance': 1e-9, 'feasibility_tolerance': 1e-9},
    'i2_volume_blend': {
        'groups': ['zero', 'low', 'middle', 'high'], 'shrinkage_pitches': 1000,
        'min_games': 30, 'min_pitches': 500, 'blend_objective': 'log_loss',
        'blend_optimizer_source': 'scripts/run_sequence_calibration.py',
        'blend_optimizer_function': 'fit_blend',
        'blend_optimizer_sha256': '951bada57ea33f34cc5440d9420943df7dc9cb782b27e9359c582e55b5cfa6bc'},
    'indicators': {
        'i1_min_class_events': 30, 'i1_min_abs_prevalence_gap': 0.001,
        'i2_positive_groups': ['low', 'middle', 'high'], 'i2_min_supported_groups': 2,
        'i2_min_games': 30, 'i2_min_pitches': 500, 'i2_min_nll_range': 0.003},
}
# Data identity pins are not part of the algorithm spec; root supplies them.
POPULATION_FIELDS = ('ordered_keys_sha256', 'labels_sha256')


def fixed_spec() -> dict:
    return copy.deepcopy(FIXED_SPEC)


def _same(expected, actual, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'Config {path} fields differ from the fixed spec')
        for key in expected:
            _same(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'Config {path} differs from the fixed spec')
        for i, (e, a) in enumerate(zip(expected, actual)):
            _same(e, a, f'{path}[{i}]')
    elif isinstance(expected, str):
        if actual != expected:
            raise ValueError(f'Config {path} differs from the fixed spec')
    else:
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual != expected:
            raise ValueError(f'Config {path} differs from the fixed spec')
        if isinstance(expected, int) and not isinstance(actual, int):
            raise ValueError(f'Config {path} must be an integer')


def _is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_config(config: dict) -> dict:
    """Require every fixed number exactly plus June population hash pins."""
    if not isinstance(config, dict) or set(config) != set(FIXED_SPEC) | {'fit_population'}:
        raise ValueError('Config fields differ from the fixed spec plus fit_population')
    _same(FIXED_SPEC, {k: config[k] for k in FIXED_SPEC}, 'config')
    population = config['fit_population']
    if not isinstance(population, dict) or set(population) != set(POPULATION_FIELDS) or \
            not all(_is_sha256(population[k]) for k in POPULATION_FIELDS):
        raise ValueError('fit_population must pin lowercase sha256 ordered keys and labels')
    return config


def config_sha256(config: dict) -> str:
    validate_config(config)
    return canonical_sha256(config)


def canonical_sha256(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def array_sha256(values, dtype) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


# ---------------------------------------------------------------- validation

def _keys(keys, name='keys'):
    k = np.asarray(keys)
    if k.ndim != 2 or k.shape[1] != 3 or not np.issubdtype(k.dtype, np.integer):
        raise ValueError(f'{name} must be an integer (n, 3) pitch-key array')
    k = k.astype(np.int64, copy=False)
    if len(np.unique(k, axis=0)) != len(k):
        raise ValueError(f'{name} must be unique')
    return k


def _aligned(reference, other, name):
    if other.shape != reference.shape or not np.array_equal(other, reference):
        raise ValueError(f'{name} rows are not aligned with the labelled keys')


def _labels(labels, n, n_classes):
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != n or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Labels must be a 1-D integer array aligned with keys')
    if np.any((y < 0) | (y >= n_classes)):
        raise ValueError('Labels outside the declared class contract')
    return y.astype(np.int64, copy=False)


def _simplex(probabilities, n, config, name='probabilities'):
    p = np.asarray(probabilities)
    if not np.issubdtype(p.dtype, np.floating):
        raise ValueError(f'{name} must be floating point')
    p = p.astype(np.float64, copy=False)
    if p.shape != (n, config['n_classes']):
        raise ValueError(f'{name} shape differs from rows x classes')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError(f'{name} must be finite and in [0,1]')
    if not np.allclose(p.sum(axis=1), 1, atol=config['simplex_atol'], rtol=0):
        raise ValueError(f'{name} mass differs from one')
    return p


def _groups(groups, n, config):
    g = np.asarray(groups, dtype=object)
    allowed = set(config['i2_volume_blend']['groups'])
    if g.ndim != 1 or len(g) != n:
        raise ValueError('Volume groups must be a 1-D array aligned with rows')
    if not all(isinstance(v, str) and v in allowed for v in g):
        raise ValueError('Unknown or missing TRAIN-volume group')
    return g


def _weight(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or \
            not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f'{name} must be a finite weight in [0,1]')
    return float(value)


def _predictor(name, config):
    if name not in config['predictors']:
        raise ValueError('Predictor must be the ensemble or one of the five ordered seeds')
    return name


def _june_population(*, split_token, keys, labels, config):
    """Hash-match the supplied rows to root's June pins. This cannot prove origin."""
    validate_config(config)
    if split_token != config['fit_split_token']:
        raise ValueError('Correction parameters may be fitted only on the registered June split')
    k = _keys(keys)
    y = _labels(labels, len(k), config['n_classes'])
    if len(k) != config['fit_rows']:
        raise ValueError('June fit rows differ from the registered count')
    identity = {'split_token': split_token, 'n_rows': int(len(k)),
                'ordered_keys_sha256': array_sha256(k, np.int64),
                'labels_sha256': array_sha256(y, np.int64)}
    for field in POPULATION_FIELDS:
        if identity[field] != config['fit_population'][field]:
            raise ValueError(f'Supplied rows do not match the pinned June {field}')
    identity['data_origin_proof'] = 'hash_match_to_config_pins_only'
    return k, y, identity


def _mapping(values, config, name):
    if not isinstance(values, Mapping) or list(values) != config['predictors']:
        raise ValueError(f'{name} must map the ensemble and seeds 0-4 in registered order')
    return values


# ---------------------------------------------------------------- I1

def class_bias_objective(bias, log_probabilities, labels, penalty):
    """Mean NLL of softmax(log p + b) plus penalty*mean(b**2), with analytic gradient."""
    b = np.asarray(bias, dtype=np.float64)
    z = log_probabilities + b
    lse = logsumexp(z, axis=1)
    n, c = z.shape
    nll = float(np.mean(lse - z[np.arange(n), labels]))
    value = nll + penalty * float(np.mean(b ** 2))
    q = np.exp(z - lse[:, None])
    grad = (q.sum(axis=0) - np.bincount(labels, minlength=c)) / n + 2 * penalty * b / c
    return value, grad


def _floored_log(p, floor):
    return np.log(np.maximum(p, floor))


def fit_class_bias_june(*, probabilities, prediction_keys, labels, keys, predictor, split_token,
                        config) -> dict:
    """Fit I1 once on June rows; failures are reported, never retried with other settings."""
    k, y, population = _june_population(split_token=split_token, keys=keys, labels=labels, config=config)
    _aligned(k, _keys(prediction_keys, 'prediction_keys'), 'Prediction')
    p = _simplex(probabilities, len(k), config)
    _predictor(predictor, config)
    spec = config['i1_class_bias']
    c, penalty = config['n_classes'], spec['penalty']
    log_p = _floored_log(p, spec['probability_floor'])
    start = np.zeros(c)
    initial, _ = class_bias_objective(start, log_p, y, penalty)
    result = minimize(class_bias_objective, start, args=(log_p, y, penalty), jac=True,
                      method=spec['optimizer'], bounds=[(spec['bias_lower'], spec['bias_upper'])] * c,
                      constraints=[{'type': 'eq', 'fun': lambda b: float(np.sum(b)),
                                    'jac': lambda b: np.ones_like(b)}],
                      options={'ftol': spec['ftol'], 'maxiter': spec['maxiter']})
    b = np.asarray(result.x, dtype=np.float64)
    finite = bool(np.isfinite(b).all())
    tol = spec['feasibility_tolerance']
    residual = float(abs(np.sum(b))) if finite else None
    violation = float(max(0., np.max(spec['bias_lower'] - b), np.max(b - spec['bias_upper']))) if finite else None
    feasible = bool(finite and residual <= tol and violation <= tol)
    final, grad = class_bias_objective(b, log_p, y, penalty) if finite else (None, None)
    not_worse = bool(finite and math.isfinite(final) and final <= initial + spec['objective_tolerance'])
    # An optimizer-reported failure (e.g. iteration limit) is never a completed fit, even at a feasible point.
    scipy_success = bool(result.success)
    return {
        'algorithm': 'I1_class_bias', 'spec_version': SPEC_VERSION, 'config_sha256': config_sha256(config),
        'predictor': predictor, 'fit_population': population,
        'prediction_sha256': array_sha256(p, np.float64),
        'fit_success': bool(scipy_success and finite and feasible and not_worse),
        'parameters': {'bias': [float(v) for v in b] if finite else None},
        'optimizer_report': {
            'method': spec['optimizer'], 'ftol': spec['ftol'], 'maxiter': spec['maxiter'],
            'scipy_version': scipy.__version__, 'numpy_version': np.__version__,
            'scipy_success': scipy_success, 'status': int(result.status),
            'message': str(result.message), 'nit': int(getattr(result, 'nit', -1)),
            'nfev': int(getattr(result, 'nfev', -1)), 'njev': int(getattr(result, 'njev', -1)),
            'objective_initial': float(initial), 'objective_final': float(final) if finite else None,
            'penalty_final': float(penalty * np.mean(b ** 2)) if finite else None,
            'gradient_final': [float(v) for v in grad] if finite else None,
            'sum_residual': residual, 'max_bound_violation': violation,
            'finite': finite, 'feasible': feasible, 'not_worse_than_initial': not_worse}}


def fit_class_bias_family(*, predictions, prediction_keys, labels, keys, split_token, config) -> dict:
    """Fit the same I1 algorithm separately for the ensemble and each of five seeds."""
    validate_config(config)
    _mapping(predictions, config, 'predictions')
    return {'algorithm': 'I1_class_bias', 'predictors': list(config['predictors']),
            'fits': [fit_class_bias_june(probabilities=predictions[name], prediction_keys=prediction_keys,
                                         labels=labels, keys=keys, predictor=name,
                                         split_token=split_token, config=config)
                     for name in config['predictors']]}


def _fitted(parameters, config, algorithm):
    if not isinstance(parameters, dict) or parameters.get('algorithm') != algorithm:
        raise ValueError(f'{algorithm} parameters required')
    if parameters.get('spec_version') != SPEC_VERSION or parameters.get('config_sha256') != config_sha256(config):
        raise ValueError('Parameters were fitted under a different config')
    if parameters.get('fit_success') is not True:
        raise ValueError('Failed fits are preserved but cannot be applied')
    return parameters['parameters']


def apply_class_bias(*, probabilities, parameters, config) -> np.ndarray:
    """Apply frozen I1 bias without labels: softmax(log(max(p, floor)) + b)."""
    b = np.asarray(_fitted(parameters, config, 'I1_class_bias')['bias'], dtype=np.float64)
    spec = config['i1_class_bias']
    if b.shape != (config['n_classes'],) or not np.isfinite(b).all() or \
            np.any(b < spec['bias_lower'] - spec['feasibility_tolerance']) or \
            np.any(b > spec['bias_upper'] + spec['feasibility_tolerance']) or \
            abs(float(np.sum(b))) > spec['feasibility_tolerance']:
        raise ValueError('Invalid frozen class bias')
    p = np.asarray(probabilities)
    p = _simplex(p, p.shape[0] if p.ndim else -1, config)
    z = _floored_log(p, spec['probability_floor']) + b
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


# ---------------------------------------------------------------- I2

class BlendOptimizer(NamedTuple):
    function: Callable
    source: str
    function_name: str
    sha256: str


def load_blend_optimizer(config, project_root=PROJECT) -> BlendOptimizer:
    """Import the existing scalar June blend optimizer after checking its pinned source hash."""
    validate_config(config)
    spec = config['i2_volume_blend']
    path = (Path(project_root) / spec['blend_optimizer_source']).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec['blend_optimizer_sha256']:
        raise ValueError('Existing blend optimizer source differs from its pinned hash')
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = importlib.import_module(path.stem)
    if Path(module.__file__).resolve() != path:
        raise ValueError('Imported blend optimizer is not the pinned source file')
    return BlendOptimizer(getattr(module, spec['blend_optimizer_function']), spec['blend_optimizer_source'],
                          spec['blend_optimizer_function'], digest)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def fit_volume_blend_june(*, calibrated, prediction_keys, frequency, frequency_keys, labels, keys, groups,
                          global_weight, predictor, split_token, config, blend_optimizer) -> dict:
    """Fit I2: shrink each supported group's June MLE blend weight toward the archived global weight.

    Game IDs are the first pitch-key column. Unsupported or absent groups keep exactly
    ``global_weight``; the zero group is fitted like any other when it is supported.
    """
    k, y, population = _june_population(split_token=split_token, keys=keys, labels=labels, config=config)
    _aligned(k, _keys(prediction_keys, 'prediction_keys'), 'Prediction')
    _aligned(k, _keys(frequency_keys, 'frequency_keys'), 'Frequency')
    cal = _simplex(calibrated, len(k), config, 'calibrated')
    freq = _simplex(frequency, len(k), config, 'frequency')
    g = _groups(groups, len(k), config)
    _predictor(predictor, config)
    w_global = _weight(global_weight, 'Archived global weight')
    spec = config['i2_volume_blend']
    if not isinstance(blend_optimizer, BlendOptimizer) or blend_optimizer.sha256 != spec['blend_optimizer_sha256'] \
            or blend_optimizer.source != spec['blend_optimizer_source'] \
            or blend_optimizer.function_name != spec['blend_optimizer_function']:
        raise ValueError('Use the pinned existing scalar blend optimizer')
    games, shrink = k[:, 0], spec['shrinkage_pitches']
    rows = []
    for name in spec['groups']:
        mask = g == name
        n, n_games = int(mask.sum()), int(len(np.unique(games[mask])))
        supported = n_games >= spec['min_games'] and n >= spec['min_pitches']
        row = {'group': name, 'june_pitches': n, 'june_games': n_games, 'supported': supported,
               'fallback': not supported, 'mle_weight': None, 'mle_report': None,
               'group_fraction': None, 'weight': w_global}
        if supported:
            report = _plain(blend_optimizer.function(y[mask], cal[mask], freq[mask], spec['blend_objective']))
            w_mle = _weight(report['model_weight'], 'Group MLE weight')
            row.update(mle_weight=w_mle, mle_report=report, group_fraction=n / (n + shrink),
                       weight=n / (n + shrink) * w_mle + shrink / (n + shrink) * w_global)
        rows.append(row)
    return {'algorithm': 'I2_volume_blend', 'spec_version': SPEC_VERSION, 'config_sha256': config_sha256(config),
            'predictor': predictor, 'fit_population': population,
            'prediction_sha256': array_sha256(cal, np.float64), 'frequency_sha256': array_sha256(freq, np.float64),
            'blend_optimizer': {'source': blend_optimizer.source, 'function': blend_optimizer.function_name,
                                'sha256': blend_optimizer.sha256},
            'fit_success': True,
            'parameters': {'global_weight': w_global, 'groups': rows}}


def fit_volume_blend_family(*, calibrated, prediction_keys, frequency, frequency_keys, labels, keys, groups,
                            global_weights, split_token, config, blend_optimizer) -> dict:
    """Fit the same I2 algorithm separately for the ensemble and each seed with its own archived weight."""
    validate_config(config)
    _mapping(calibrated, config, 'calibrated')
    _mapping(global_weights, config, 'global_weights')
    return {'algorithm': 'I2_volume_blend', 'predictors': list(config['predictors']),
            'fits': [fit_volume_blend_june(calibrated=calibrated[name], prediction_keys=prediction_keys,
                                           frequency=frequency, frequency_keys=frequency_keys, labels=labels,
                                           keys=keys, groups=groups, global_weight=global_weights[name],
                                           predictor=name, split_token=split_token, config=config,
                                           blend_optimizer=blend_optimizer)
                     for name in config['predictors']]}


def apply_volume_blend(*, calibrated, frequency, groups, keys, calibrated_keys, frequency_keys,
                       parameters, config) -> np.ndarray:
    """Apply frozen per-group weights without labels; unsupported groups use the global weight."""
    fitted = _fitted(parameters, config, 'I2_volume_blend')
    names = config['i2_volume_blend']['groups']
    rows = fitted['groups']
    if [r['group'] for r in rows] != names:
        raise ValueError('Frozen I2 parameters must list every registered group in order')
    w_global = _weight(fitted['global_weight'], 'Frozen global weight')
    weights = {}
    for r in rows:
        weights[r['group']] = _weight(r['weight'], 'Frozen group weight')
        if not r['supported'] and weights[r['group']] != w_global:
            raise ValueError('Unsupported groups must keep exactly the global weight')
    k = _keys(keys)
    _aligned(k, _keys(calibrated_keys, 'calibrated_keys'), 'Calibrated')
    _aligned(k, _keys(frequency_keys, 'frequency_keys'), 'Frequency')
    cal = _simplex(calibrated, len(k), config, 'calibrated')
    freq = _simplex(frequency, len(k), config, 'frequency')
    g = _groups(groups, len(k), config)
    w = np.array([weights[v] for v in g], dtype=np.float64)[:, None]
    return w * cal + (1 - w) * freq


# ---------------------------------------------------------------- indicators

def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return int(value)


def _real(value, name, lower=-math.inf, upper=math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)) or \
            not math.isfinite(float(value)) or not lower <= float(value) <= upper:
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def weakness_indicators(*, class_diagnostics, volume_diagnostics, config) -> dict:
    """Evaluate the fixed I1/I2 indicators from precomputed frozen stage-2 diagnostics.

    ``class_diagnostics``: one row per class with ``class``, whole-MLB ``events``,
    ``observed_prevalence`` and ``predicted_prevalence``. ``volume_diagnostics``: one
    row per registered volume group with ``june_games``, ``june_pitches``,
    ``dev_games``, ``dev_pitches`` and ``dev_g0_minus_frequency_nll`` (None if no DEV
    rows). Nothing here fits, predicts, or reads labels.
    """
    validate_config(config)
    spec = config['indicators']
    rows = list(class_diagnostics)
    if sorted(r.get('class') for r in rows) != list(range(config['n_classes'])) or len(rows) != config['n_classes']:
        raise ValueError('Class diagnostics must list every class exactly once')
    classes = []
    for r in sorted(rows, key=lambda r: r['class']):
        events = _count(r['events'], 'Class events')
        gap = abs(_real(r['predicted_prevalence'], 'Predicted prevalence', 0, 1)
                  - _real(r['observed_prevalence'], 'Observed prevalence', 0, 1))
        supported = events >= spec['i1_min_class_events']
        classes.append({'class': int(r['class']), 'events': events, 'abs_prevalence_gap': gap,
                        'supported': supported,
                        'qualifies': supported and gap >= spec['i1_min_abs_prevalence_gap']})
    i1_supported = [c for c in classes if c['supported']]
    i1 = {'status': 'unmeasured' if not i1_supported else
          'met' if any(c['qualifies'] for c in classes) else 'not_met', 'classes': classes}

    vrows = list(volume_diagnostics)
    names = config['i2_volume_blend']['groups']
    if sorted(r.get('group') for r in vrows) != sorted(names) or len(vrows) != len(names):
        raise ValueError('Volume diagnostics must list every registered group exactly once')
    groups = []
    for r in sorted(vrows, key=lambda r: names.index(r['group'])):
        counts = {f: _count(r[f], f) for f in ('june_games', 'june_pitches', 'dev_games', 'dev_pitches')}
        delta = r['dev_g0_minus_frequency_nll']
        delta = None if delta is None else _real(delta, 'DEV NLL difference')
        if delta is None and counts['dev_pitches']:
            raise ValueError('DEV NLL difference missing for a group with DEV rows')
        eligible = (r['group'] in spec['i2_positive_groups'] and delta is not None
                    and counts['june_games'] >= spec['i2_min_games'] and counts['june_pitches'] >= spec['i2_min_pitches']
                    and counts['dev_games'] >= spec['i2_min_games'] and counts['dev_pitches'] >= spec['i2_min_pitches'])
        groups.append({'group': r['group'], **counts, 'dev_g0_minus_frequency_nll': delta, 'eligible': eligible})
    deltas = [g['dev_g0_minus_frequency_nll'] for g in groups if g['eligible']]
    if len(deltas) < spec['i2_min_supported_groups']:
        i2 = {'status': 'unmeasured', 'nll_range': None, 'groups': groups}
    else:
        spread = max(deltas) - min(deltas)
        i2 = {'status': 'met' if spread >= spec['i2_min_nll_range'] else 'not_met',
              'nll_range': spread, 'groups': groups}
    return {'spec_version': SPEC_VERSION, 'config_sha256': config_sha256(config),
            'source': 'precomputed_frozen_stage2_diagnostics', 'I1': i1, 'I2': i2}
