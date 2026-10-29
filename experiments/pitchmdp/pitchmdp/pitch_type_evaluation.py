"""Frozen frequency baselines and scoring for a synthetic pitch-TYPE contract."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from .matrix_metrics import validate_probabilities

LABELS = ('FF', 'SL', 'CH', 'OTHER')  # Synthetic vocabulary, not a real MLB taxonomy.
KEY = ('game_id', 'at_bat_number', 'pitch_number')
FEATURES = ('pitcher', 'balls', 'strikes', 'batter_side', 'pitcher_hand', 'bases', 'outs', 'score_diff')
TIERS = {'global': (), 'pitcher': FEATURES[:1], 'count': FEATURES[:3],
         'matchup': FEATURES[:5], 'situation': FEATURES}
MODELS = ('pitcher_reference', 'context_control', 'hypothetical_candidate')
PENDING = ('data_manifest', 'label_mapping_and_eligibility', 'exposure_audit',
           'fit_select_cal_test_split', 'availability_evidence', 'tree_model_pin',
           'practical_and_coverage_rule', 'independent_registration_review')


def _fields(row, names):
    if not isinstance(row, dict) or set(row) != set(names):
        raise ValueError('unknown or missing fields')


def _time(value):
    if not isinstance(value, str):
        raise ValueError('timestamp must be an aware ISO string')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid timestamp') from exc
    if parsed.utcoffset() is None:
        raise ValueError('timestamp must be timezone aware')
    return parsed


def _key(row):
    values = tuple(row.get(name) for name in KEY)
    if any(type(v) is not int or v < 1 for v in values):
        raise ValueError('physical pitch identity requires positive integers')
    return values


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _context(context):
    _fields(context, FEATURES)
    bounds = {'pitcher': (1, 2**31 - 1), 'balls': (0, 3), 'strikes': (0, 2),
              'bases': (0, 7), 'outs': (0, 2), 'score_diff': (-100, 100)}
    for name, (lo, hi) in bounds.items():
        if type(context[name]) is not int or not lo <= context[name] <= hi:
            raise ValueError(f'invalid pre-pitch context: {name}')
    if context['batter_side'] not in ('L', 'R') or context['pitcher_hand'] not in ('L', 'R'):
        raise ValueError('handedness must be resolved L/R for this synthetic contract')


def validate_config(config):
    fixed = {'id': 'EVAL-PITCH-TYPE-003', 'mode': 'SYNTHETIC_ONLY', 'real_data_gate': 'CLOSED',
             'schema': 'synthetic-pitch-type-v1', 'fixture_version': 'four-types-v1',
             'labels': list(LABELS), 'label_map': {v: v for v in LABELS},
             'features': list(FEATURES), 'hierarchy': {k: list(v) for k, v in TIERS.items()},
             'primary_pair': [MODELS[1], MODELS[2]],
             'candidate_kind': 'HYPOTHETICAL_NOT_FITTED_TREE',
             'source_audit_base': '3863ba1b80b57ac702a969a976a6b70827586bc7',
             'real_data_prerequisites': {name: None for name in PENDING}}
    _fields(config, (*fixed, 'smoothing', 'bootstrap', 'training_cutoff', 'holdout_start'))
    for name, value in fixed.items():
        if config[name] != value:
            raise ValueError(f'unregistered or real-data configuration: {name}')
    _fields(config['smoothing'], TIERS)
    for alpha in config['smoothing'].values():
        if type(alpha) not in (int, float) or not np.isfinite(alpha) or not 0 < alpha <= 10000:
            raise ValueError('smoothing must be finite positive and bounded')
    _fields(config['bootstrap'], ('seed', 'draws'))
    seed, draws = config['bootstrap']['seed'], config['bootstrap']['draws']
    if type(seed) is not int or not 0 <= seed < 2**32 or type(draws) is not int or not 100 <= draws <= 10000:
        raise ValueError('invalid synthetic bootstrap budget')
    if _time(config['training_cutoff']) >= _time(config['holdout_start']):
        raise ValueError('training cutoff must strictly precede holdout')


def _rosters(config, history, eligible):
    """Validate separate FIT/query roles without filtering rows by predictions."""
    validate_config(config)
    if not isinstance(history, list) or not isinstance(eligible, list):
        raise ValueError('FIT and eligible rows must be JSON arrays')
    fit, query = {}, {}
    for rows, target, extra in ((history, fit, ('observed_at',)),
                                (eligible, query, ('input_available_at', 'generated_at', 'deadline'))):
        for row in rows:
            _fields(row, (*KEY, 'label', 'context', *extra))
            key = _key(row)
            if key in target:
                raise ValueError('duplicate physical pitch in roster')
            if row['label'] not in LABELS:
                raise ValueError('unregistered or missing pitch-type label')
            _context(row['context'])
            if target is fit:
                if _time(row['observed_at']) >= _time(config['training_cutoff']):
                    raise ValueError('FIT label observed at or after frozen cutoff')
            else:
                available, generated, deadline = (_time(row[n]) for n in extra)
                if deadline < _time(config['holdout_start']):
                    raise ValueError('query deadline before holdout')
                if not _time(config['training_cutoff']) < generated < deadline or available > generated:
                    raise ValueError('invalid query generation/input chronology')
            target[key] = row
    if not fit or not query:
        raise ValueError('nonempty FIT and eligible rosters required')
    if set(fit) & set(query):
        raise ValueError('FIT/query physical pitch overlap')
    if {k[0] for k in fit} & {k[0] for k in query}:
        raise ValueError('same game crosses FIT/query roles')
    return fit, query


def _baselines(config, fit, query):
    counts = {tier: defaultdict(lambda: np.zeros(len(LABELS), dtype=int)) for tier in TIERS}
    for key in sorted(fit):
        row = fit[key]
        for tier, names in TIERS.items():
            counts[tier][tuple(row['context'][name] for name in names)][LABELS.index(row['label'])] += 1
    predictions, diagnostics = {m: {} for m in MODELS[:2]}, []
    for key in sorted(query):
        row = query[key]
        p = np.full(len(LABELS), 1 / len(LABELS))
        support, effective = {}, 'global'
        for tier, names in TIERS.items():
            n = counts[tier].get(tuple(row['context'][name] for name in names), np.zeros(len(LABELS), dtype=int))
            support[tier] = int(n.sum())
            if n.sum():
                effective = tier
            alpha = config['smoothing'][tier]
            if support[tier]:
                p = (n + alpha * p) / (n.sum() + alpha)
            if tier == 'pitcher':
                predictions[MODELS[0]][key] = p.tolist()
        predictions[MODELS[1]][key] = p.tolist()
        diagnostics.append({'key': list(key), 'support': support, 'effective_level': effective,
                            'unseen_pitcher': support['pitcher'] == 0,
                            'context_fallback': effective != 'situation'})
    return predictions, diagnostics


def frequency_predictions(config, history, eligible):
    """Contract-test entry point; query labels are validated but never counted."""
    fit, query = _rosters(config, history, eligible)
    return _baselines(config, fit, query)


def probability_scores(labels, probabilities, n_classes):
    """Variable-K metrics; first schema index wins ties, including top-2 ties."""
    y, p = validate_probabilities(labels, probabilities, n_classes=n_classes)
    if n_classes < 2:
        raise ValueError('at least two classes required')
    selected = p[np.arange(len(y)), y]
    losses = np.column_stack((-np.log(np.clip(selected, 1e-12, 1)),
                              (p * p).sum(axis=1) - 2 * selected + 1))
    order = np.argsort(-p, axis=1, kind='stable')
    scores = {'n': len(y), 'nll': None, 'brier_class_sum': None, 'top1_accuracy': None,
              'top2_coverage': None, 'zero_true_probability': int((selected == 0).sum())}
    if len(y):
        scores.update(nll=float(losses[:, 0].mean()), brier_class_sum=float(losses[:, 1].mean()),
                      top1_accuracy=float((order[:, 0] == y).mean()),
                      top2_coverage=float((order[:, :2] == y[:, None]).any(axis=1).mean()))
    return scores, losses


def _paired_interval(delta, game_ids, bootstrap):
    """Sorted pitches, pitch-weighted whole-game percentile CI; no winner/test."""
    games, inverse = np.unique(game_ids, return_inverse=True)
    result = {'n': len(delta), 'games': len(games), **bootstrap,
              'status': 'unmeasured' if not len(delta) else 'insufficient_games',
              'nll': None, 'brier': None}
    if not len(delta):
        return result
    counts = np.bincount(inverse)
    sums = np.column_stack([np.bincount(inverse, weights=delta[:, j]) for j in range(2)])
    observed = sums.sum(axis=0) / counts.sum()
    for j, name in enumerate(('nll', 'brier')):
        result[name] = {'delta': float(observed[j]), 'ci95': None}
    if len(games) < 2:
        return result
    rng = np.random.default_rng(bootstrap['seed'])
    # ponytail: serial bootstrap is sufficient for this tiny synthetic fixture;
    # future large-cohort execution must register a bounded/chunked implementation.
    samples = []
    for _ in range(bootstrap['draws']):
        selected = rng.integers(0, len(games), size=len(games))
        samples.append(sums[selected].sum(axis=0) / counts[selected].sum())
    result['status'] = 'measured'
    ci = np.quantile(samples, [.025, .975], axis=0)
    for j, name in enumerate(('nll', 'brier')):
        result[name]['ci95'] = ci[:, j].tolist()
    return result


def evaluate(config, history, eligible, candidate):
    """Accept declared synthetic rows for tests, not real-data execution authority."""
    if not isinstance(candidate, list):
        raise ValueError('candidate rows must be a JSON array')
    fit, query = _rosters(config, history, eligible)
    predictions, support = _baselines(config, fit, query)
    predictions[MODELS[2]] = {}
    states = Counter()
    seen = set()
    for row in candidate:
        if not isinstance(row, dict):
            raise ValueError('candidate row must be an object')
        status = row.get('status')
        extra = ('labels', 'schema', 'probabilities', 'input_fingerprint', 'input_available_at', 'generated_at') if status == 'ok' else ()
        _fields(row, (*KEY, 'status', *extra))
        key = _key(row)
        if key not in query or key in seen:
            raise ValueError('unknown or duplicate candidate identity')
        seen.add(key)
        if status not in ('ok', 'error', 'abstain'):
            raise ValueError('unknown candidate status')
        if status == 'ok':
            if row['schema'] != config['schema'] or row['labels'] != list(LABELS):
                raise ValueError('ordered pitch-type schema mismatch')
            if row['input_fingerprint'] != _hash(query[key]['context']):
                raise ValueError('shared context fingerprint mismatch')
            if _time(row['input_available_at']) != _time(query[key]['input_available_at']):
                raise ValueError('shared context availability mismatch')
            available, generated = _time(row['input_available_at']), _time(row['generated_at'])
            if generated <= _time(config['training_cutoff']) or available > generated:
                raise ValueError('candidate generation violates input/training chronology')
            p = row['probabilities']
            if not isinstance(p, list) or any(type(v) not in (int, float) for v in p):
                raise ValueError('probabilities must be numeric JSON values')
            validate_probabilities(np.array([LABELS.index(query[key]['label'])]), [p], n_classes=len(LABELS))
            if generated >= _time(query[key]['deadline']):
                status = 'late'
            else:
                predictions[MODELS[2]][key] = p
        states[status] += 1
    states['missing'] = len(query) - len(seen)
    def score(model, keys):
        y = np.array([LABELS.index(query[k]['label']) for k in keys], dtype=int)
        p = np.array([predictions[model][k] for k in keys]).reshape(-1, len(LABELS))
        return probability_scores(y, p, len(LABELS))
    keys, common = sorted(query), sorted(predictions[MODELS[2]])
    coverage = {}
    for model in MODELS:
        counts = states if model == MODELS[2] else Counter(ok=len(query))
        supported = sorted(predictions[model])
        coverage[model] = {'eligible': len(query), 'timely_ok': len(supported),
                           'fraction': len(supported) / len(query),
                           'statuses': {s: counts[s] for s in ('ok', 'missing', 'error', 'abstain', 'late')},
                           'own_supported_metrics': score(model, supported)[0]}
    pair_scores = {model: score(model, common) for model in MODELS[1:]}
    delta = pair_scores[MODELS[2]][1] - pair_scores[MODELS[1]][1]
    pair = _paired_interval(delta, np.array([k[0] for k in common], dtype=int), config['bootstrap'])
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('pitch_type_evaluation.py', 'matrix_metrics.py')}
    return {'mode': 'SYNTHETIC_ONLY', 'real_data_gate': 'CLOSED', 'scientific_winner': None,
            'candidate_kind': config['candidate_kind'], 'schema': config['schema'], 'labels': list(LABELS),
            'source_sha256': hashes, 'config_sha256': _hash(config),
            'synthetic_input_sha256': {'FIT': _hash([fit[k] for k in sorted(fit)]),
                'eligible': _hash([query[k] for k in keys]), 'candidate': _hash(sorted(candidate, key=_key))},
            'training_cutoff': config['training_cutoff'], 'holdout_start': config['holdout_start'],
            'nll_floor': 1e-12, 'ties': 'fixed schema order; no label-based tie breaking',
            'provenance': 'synthetic declarations; timestamps/hash do not prove historical availability',
            'coverage': coverage, 'support': support,
            'fallback_counts': dict(sorted(Counter(r['effective_level'] for r in support).items())),
            'unseen_pitcher_queries': sum(r['unseen_pitcher'] for r in support),
            'baseline_predictions': {model: [{'key': list(k), 'probabilities': predictions[model][k],
                'input_fingerprint': _hash(query[k]['context'])} for k in keys] for model in MODELS[:2]},
            'primary_pair': config['primary_pair'], 'paired_keys': [list(k) for k in common],
            'paired_metrics': {m: pair_scores[m][0] for m in MODELS[1:]}, 'comparison': pair,
            'comparison_scope': 'candidate minus control; pitch-weighted; common timely successes only; fixed-prediction whole-game CI',
            'complete_common_coverage': len(common) == len(query)}


def synthetic_fixture(config):
    """Finite, internal toy history and forecasts; no data loader or fitted tree."""
    validate_config(config)
    context = dict(pitcher=7, balls=0, strikes=0, batter_side='R', pitcher_hand='R', bases=0, outs=0, score_diff=0)
    history = []
    for i, label in enumerate(('FF', 'FF', 'SL', 'CH', 'FF', 'SL')):
        history.append({'game_id': 1 + i // 3, 'at_bat_number': 1, 'pitch_number': 1 + i % 3,
            'label': label, 'observed_at': '2000-01-01T12:00:00Z',
            'context': {**context, 'pitcher': 7 if i < 4 else 8}})
    eligible, candidate = [], []
    for i, label in enumerate(('FF', 'SL', 'CH', 'FF')):
        key = dict(game_id=11 + i // 2, at_bat_number=1, pitch_number=1 + i % 2)
        ctx = {**context, 'pitcher': 99 if i == 3 else 7, 'balls': i % 2}
        eligible.append({**key, 'label': label, 'context': ctx,
            'input_available_at': '2000-01-03T11:59:58Z', 'generated_at': '2000-01-03T11:59:59Z',
            'deadline': '2000-01-03T12:00:00Z'})
        # Fixed hypothetical vector; does not inspect the observed query label.
        candidate.append({**key, 'status': 'ok', 'schema': config['schema'], 'labels': list(LABELS),
            'probabilities': [.4, .3, .2, .1], 'input_fingerprint': _hash(ctx),
            'input_available_at': '2000-01-03T11:59:58Z', 'generated_at': '2000-01-03T11:59:59Z'})
    return history, eligible, candidate
