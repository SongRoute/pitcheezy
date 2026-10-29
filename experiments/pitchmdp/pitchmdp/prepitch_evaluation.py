"""Synthetic-only pre-pitch scoring contract; never a real-data authorization."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json

import numpy as np

from .matrix_metrics import paired_game_comparison, pitch_losses, validate_probabilities

# Pinned to model.py OUTCOMES at ca55f47c; importing it would load torch/pandas.
LABELS = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple',
          'home_run', 'hbp', 'double_play')
MODELS = ('synthetic_control', 'synthetic_candidate')
STRATA = {'count': ('0-0', '1-0'), 'handed_matchup': ('R-R', 'R-L'),
          'prior_history': ('none', 'available')}
KEY_FIELDS = ('game_id', 'at_bat_number', 'pitch_number')


def _time(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be an aware ISO string')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid timestamp') from exc
    if result.utcoffset() is None:
        raise ValueError('timestamp must be timezone aware')
    return result


def _key(row):
    values = tuple(row.get(name) for name in KEY_FIELDS)
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('physical pitch identity requires positive integers')
    return values


def _fields(row, expected):
    if not isinstance(row, dict) or set(row) != set(expected):
        raise ValueError('unknown or missing fields')


def _fingerprint(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('input fingerprint must be lowercase sha256')


def validate_config(config):
    _fields(config, ('id', 'mode', 'real_data_gate', 'schema', 'labels', 'primary_pair',
                     'bootstrap', 'synthetic_holdout_start', 'training_cutoffs',
                     'input_contract', 'strata', 'real_data_prerequisites'))
    fixed = {'id': 'EVAL-PREPITCH-001', 'mode': 'SYNTHETIC_ONLY',
             'real_data_gate': 'CLOSED', 'schema': 'pitchmdp-ten-v1',
             'labels': list(LABELS), 'primary_pair': list(MODELS),
             'input_contract': 'synthetic-prepitch-v1',
             'strata': {k: list(v) for k, v in STRATA.items()},
             'real_data_prerequisites': {k: None for k in (
                 'data_manifest', 'exposure_audit', 'fit_select_cal_test_split',
                 'eligibility_and_target_mapping', 'input_availability_evidence',
                 'model_source_and_artifact_pins', 'practical_improvement_rule',
                 'independent_review')}}
    for name, value in fixed.items():
        if config[name] != value:
            raise ValueError(f'unregistered or real-data configuration: {name}')
    _fields(config['bootstrap'], ('seed', 'draws'))
    seed, draws = config['bootstrap']['seed'], config['bootstrap']['draws']
    if type(seed) is not int or not 0 <= seed < 2**32 or type(draws) is not int or not 100 <= draws <= 10000:
        raise ValueError('invalid synthetic bootstrap budget')
    _fields(config['training_cutoffs'], MODELS)
    start = _time(config['synthetic_holdout_start'])
    if any(_time(c) >= start for c in config['training_cutoffs'].values()):
        raise ValueError('training cutoff must precede holdout')


def _metrics(keys, roster, predictions):
    if not keys:
        return {'n': 0, 'nll': None, 'brier_class_sum': None, 'accuracy': None,
                'zero_true_probability': 0}
    y = np.array([LABELS.index(roster[k]['label']) for k in keys])
    p = np.array([predictions[k]['probabilities'] for k in keys])
    loss = pitch_losses(y, p)
    return {'n': len(keys), 'nll': float(loss[:, 0].mean()),
            'brier_class_sum': float(loss[:, 1].mean()),
            'accuracy': float((p.argmax(axis=1) == y).mean()),
            'zero_true_probability': int((p[np.arange(len(y)), y] == 0).sum())}


def evaluate(config, eligible, predictions):
    """Validate declared synthetic rows. Metadata is not proof of input provenance.

    The CLI only constructs synthetic rows internally. This Python helper accepts
    declarations for contract tests; it must not be used to score real payloads.
    """
    validate_config(config)
    roster = {}
    for row in eligible:
        _fields(row, (*KEY_FIELDS, 'label', 'deadline', 'input_fingerprint', 'strata'))
        key = _key(row)
        if key in roster:
            raise ValueError('duplicate physical pitch in eligible roster')
        if row['label'] not in LABELS:
            raise ValueError('unknown outcome label')
        if _time(row['deadline']) < _time(config['synthetic_holdout_start']):
            raise ValueError('pitch before holdout')
        _fingerprint(row['input_fingerprint'])
        _fields(row['strata'], STRATA)
        if any(row['strata'][name] not in (*values, 'unavailable') for name, values in STRATA.items()):
            raise ValueError('unregistered stratum')
        roster[key] = row
    if not roster:
        raise ValueError('empty eligible roster')
    streams = {model: {} for model in MODELS}
    counts = {model: Counter() for model in MODELS}
    seen = set()
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError('prediction must be an object')
        status = row.get('status')
        expected = (*KEY_FIELDS, 'model', 'status')
        if status == 'ok':
            expected += ('labels', 'probabilities', 'input_fingerprint', 'input_available_at', 'generated_at')
        _fields(row, expected)
        model, key = row['model'], _key(row)
        if model not in MODELS or key not in roster:
            raise ValueError('unregistered model or unknown pitch')
        if (model, key) in seen:
            raise ValueError('duplicate model prediction')
        seen.add((model, key))
        if status not in ('ok', 'error', 'abstain'):
            raise ValueError('unknown prediction status')
        if status == 'ok':
            if row['labels'] != list(LABELS):
                raise ValueError('ordered label schema mismatch')
            if row['input_fingerprint'] != roster[key]['input_fingerprint']:
                raise ValueError('input fingerprint mismatch')
            if not isinstance(row['probabilities'], list) or any(
                    type(p) not in (int, float) for p in row['probabilities']):
                raise ValueError('probabilities must be numeric JSON values')
            validate_probabilities(np.array([LABELS.index(roster[key]['label'])]), [row['probabilities']])
            available, generated = _time(row['input_available_at']), _time(row['generated_at'])
            deadline = _time(roster[key]['deadline'])
            if generated <= _time(config['training_cutoffs'][model]):
                raise ValueError('prediction generation must follow model training cutoff')
            if available > generated or available >= deadline:
                raise ValueError('input must be available before deadline and by generation')
            if generated >= deadline:
                status = 'late'
            else:
                streams[model][key] = row
        counts[model][status] += 1
    keys = sorted(roster)
    common = sorted(set(streams[MODELS[0]]) & set(streams[MODELS[1]]))
    coverage = {}
    for model in MODELS:
        counts[model]['missing'] = len(roster) - sum(counts[model].values())
        coverage[model] = {'eligible': len(roster), 'timely_ok': len(streams[model]),
                           'fraction': len(streams[model]) / len(roster),
                           'statuses': {s: counts[model][s] for s in ('ok', 'missing', 'error', 'abstain', 'late')},
                           'own_supported_metrics': _metrics(sorted(streams[model]), roster, streams[model])}
    y = np.array([LABELS.index(roster[k]['label']) for k in common], dtype=int)
    pair = paired_game_comparison(y,
        np.array([streams[MODELS[1]][k]['probabilities'] for k in common]).reshape(-1, 10),
        np.array([streams[MODELS[0]][k]['probabilities'] for k in common]).reshape(-1, 10),
        np.array([k[0] for k in common]), **config['bootstrap'])
    # The reused helper's hypothesis test is not registered for this readiness check.
    for name in ('nll', 'brier'):
        if pair[name] is not None:
            pair[name].pop('p_less', None)
    pair.pop('test', None)
    strata = {}
    for name, values in STRATA.items():
        strata[name] = []
        for value in (*values, 'unavailable'):
            selected = [k for k in common if roster[k]['strata'][name] == value]
            strata[name].append({'value': value,
                'eligible': sum(roster[k]['strata'][name] == value for k in keys),
                'paired': len(selected),
                'metrics': {m: _metrics(selected, roster, streams[m]) for m in MODELS}})
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'mode': 'SYNTHETIC_ONLY', 'real_data_gate': 'CLOSED', 'scientific_winner': None,
            'decision': 'NOT_EVALUATED_SYNTHETIC_ONLY', 'config_sha256': digest,
            'schema': config['schema'], 'labels': list(LABELS), 'primary_pair': list(MODELS),
            'nll_floor': 1e-12, 'accuracy_ties': 'first registered label',
            'provenance': 'declared synthetic timestamps/fingerprints; not verified historical availability',
            'coverage': coverage, 'paired_keys': [list(k) for k in common],
            'paired_metrics': {m: _metrics(common, roster, streams[m]) for m in MODELS},
            'comparison': pair, 'comparison_scope': 'conditional on common timely successful predictions',
            'complete_common_coverage': len(common) == len(roster),
            'strata_descriptive_only': strata}


def synthetic_fixture(config):
    """Four hand-authored pitches; no data loader, filesystem or model access."""
    validate_config(config)
    eligible, predictions = [], []
    for i in range(4):
        key = {'game_id': 1 + i // 2, 'at_bat_number': 1, 'pitch_number': 1 + i % 2}
        fingerprint = hashlib.sha256(f'synthetic context {i}'.encode()).hexdigest()
        eligible.append({**key, 'label': LABELS[i % 2], 'deadline': '2000-01-02T12:00:00Z',
            'input_fingerprint': fingerprint,
            'strata': {name: values[i % 2] for name, values in STRATA.items()}})
        for model, correct in zip(MODELS, (.5, .75)):
            p = [0.] * 10
            p[i % 2], p[1 - i % 2] = correct, 1 - correct
            predictions.append({**key, 'model': model, 'status': 'ok', 'labels': list(LABELS),
                'probabilities': p, 'input_fingerprint': fingerprint,
                'input_available_at': '2000-01-02T11:59:58Z', 'generated_at': '2000-01-02T11:59:59Z'})
    return eligible, predictions
