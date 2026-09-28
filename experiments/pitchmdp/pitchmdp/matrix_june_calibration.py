"""Pure helpers for the additive full-June calibration family ML-JUNE-CALIBRATION-v1.

B0 is the archived five-seed G0 network-frequency blend, B1 refits one global
scalar on all 104,970 eligible June pitches, and B2 fits four TRAIN-volume
scalars shrunk toward the new B1 weight.  This module never reads files: it
holds the frozen constants, the scientific-config cross-check, probability/key
validation, the diagnostic bounded scalar optimizer, the 6 replay + 30
candidate family fit and the label-free application.  Scoring lives in
``matrix_june_calibration_metrics``; data access lives in the runner.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.optimize import minimize_scalar

FAMILY_ID = 'ML-JUNE-CALIBRATION-v1'
EXECUTION_PROTOCOL = 'ml_june_calibration_execution_v1'
SEEDS = (0, 1, 2, 3, 4)
PREDICTORS = ('ensemble', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4')
SEED_PARENT = {0: 'EXP-P4-001', 1: 'EXP-P4-001', 2: 'EXP-P4-001', 3: 'EXP-P10-001', 4: 'EXP-P10-001'}
N_CLASSES = 10
GROUPS = ('zero', 'low', 'middle', 'high')
VARIANTS = ('B0', 'B1', 'B2')
STAGES = ('prepare', 'profile', 'predict', 'fit', 'apply', 'score')
STAGE_CAPS = {'prepare': 300, 'profile': 600, 'predict': 600, 'fit': 300, 'apply': 300, 'score': 300}
JUNE = {'rows': 104970, 'games': 394, 'pitchers': 549, 'date_min': '2025-06-01', 'date_max': '2025-06-30',
        'ordered_key_sha256': 'ddb3a0859240821056ea687713b68ab4d2e2d0d8debb8dbd2d64871796e37f82'}
CPANEL = {'rows': 4821, 'games': 110, 'ordered_key_sha256': '0f66807ba93c185e40416e6543183f389dac595fec310fb5a56cfe21bd9c4674'}
DEV = {'rows': 311721, 'games': 1161, 'cpanel_rows': 12334, 'cpanel_games': 328, 'complement_rows': 299387,
       'months': ('2025-07', '2025-08', '2025-09')}
EXPECTED_SUPPORT = {'zero': {'pitches': 4788, 'games': 127}, 'low': {'pitches': 5338, 'games': 184},
                    'middle': {'pitches': 38546, 'games': 393}, 'high': {'pitches': 56298, 'games': 386}}
VOLUME_THRESHOLDS = (170.0, 1514.0)
DRAWS = 400
PROFILE_ROWS, PROBE_ROWS = 8192, 64
SIMPLEX_ATOL = 1e-6
PROBABILITY_REPLAY_ATOL = 1e-6
WEIGHT_REPLAY_ATOL = 1e-8
OPTIMIZER = {'method': 'bounded', 'bounds': (0.0, 1.0), 'xatol': 1e-5, 'maxiter': 500, 'disp': 0,
             'probability_floor': 1e-12, 'objective_argument': 'log_loss', 'objective_tolerance': 1e-9,
             'choices_in_tie_order': ('0', 'optimum.x', '1')}
SHRINKAGE_PITCHES, MIN_PITCHES, MIN_GAMES = 1000, 500, 30
EXPECTED_CALLS = {'replay': 6, 'candidate': 30, 'total': 36}
RESEARCH_STATUS = {'independent_confirmation': None, 'held_out_confirmation': False,
                   'policy_effect': None, 'service_adoption': None}
HEX = set('0123456789abcdef')


class FamilyStop(RuntimeError):
    """A replay, optimizer or identity gate stopped the whole family; diagnostics are preserved."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def is_commit(value):
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def array_sha256(values, dtype):
    """Hash of explicit C-contiguous bytes; Fortran-ordered archives hash like their C copies."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values), dtype=dtype).tobytes()).hexdigest()


# ---------------------------------------------------------------- scientific config cross-check

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def check_scientific_config(sci):
    """The immutable scientific registration must carry exactly the values this code implements."""
    _require(sci.get('family_id') == FAMILY_ID, 'Scientific family ID differs')
    _require(sci.get('seeds') == list(SEEDS) and sci.get('predictors') == list(PREDICTORS), 'Seed/predictor order differs')
    _require(sci.get('n_classes') == N_CLASSES, 'Class count differs')
    comp = sci['components']
    _require((comp['draws'], comp['neural_fit_count'], comp['refit_temperatures'], comp['refit_group_boundaries'])
             == (DRAWS, 0, False, False), 'Frozen component settings differ')
    _require((comp['input_simplex_atol'], comp['cpanel_probability_replay_atol'], comp['cpanel_weight_replay_atol'],
              comp['replay_rtol'], comp['probability_replay_rtol'])
             == (SIMPLEX_ATOL, PROBABILITY_REPLAY_ATOL, WEIGHT_REPLAY_ATOL, 0, 0), 'Replay tolerances differ')
    fit = sci['populations']['fit']
    _require((fit['rows'], fit['games'], fit['pitchers'], fit['date_min'], fit['date_max'], fit['ordered_key_sha256'])
             == (JUNE['rows'], JUNE['games'], JUNE['pitchers'], JUNE['date_min'], JUNE['date_max'],
                 JUNE['ordered_key_sha256']), 'June fit population differs')
    _require((sci['populations']['cpanel_replay']['rows'], sci['populations']['cpanel_replay']['games'])
             == (CPANEL['rows'], CPANEL['games']), 'Cpanel replay population differs')
    dev = sci['populations']['dev']
    _require((dev['whole_rows'], dev['whole_games'], dev['cpanel_rows'], dev['complement_rows'])
             == (DEV['rows'], DEV['games'], DEV['cpanel_rows'], DEV['complement_rows']), 'DEV population differs')
    opt = sci['optimizer']
    _require((opt['probability_floor'], opt['dtype'], tuple(opt['bounds']), opt['xatol'], opt['maxiter'], opt['disp'],
              tuple(opt['choices_in_tie_order']), opt['require_success'], opt['require_finite_feasible'],
              opt['objective_tolerance'], opt['objective_argument'], opt['anchor_is_additional_optimizer_choice'])
             == (OPTIMIZER['probability_floor'], 'float64', OPTIMIZER['bounds'], OPTIMIZER['xatol'],
                 OPTIMIZER['maxiter'], OPTIMIZER['disp'], OPTIMIZER['choices_in_tie_order'], True, True,
                 OPTIMIZER['objective_tolerance'], 'log_loss', False), 'Optimizer settings differ')
    b2 = sci['variants']['B2']
    _require((tuple(b2['groups']), b2['shrinkage_pitches'], b2['min_games'], b2['min_pitches'], b2['combine'])
             == (GROUPS, SHRINKAGE_PITCHES, MIN_GAMES, MIN_PITCHES, 'AND'), 'B2 group rule differs')
    _require(b2['expected_support'] == EXPECTED_SUPPORT, 'Expected June volume support differs')
    calls = sci['optimization_calls']
    _require((calls['candidate_scalar_fits'], calls['cpanel_identity_replay_scalar_fits'], calls['total_expected'])
             == (EXPECTED_CALLS['candidate'], EXPECTED_CALLS['replay'], EXPECTED_CALLS['total']), 'Call count differs')
    _require([c['id'] for c in sci['contrasts']] == ['B1-B0', 'B2-B0', 'B2-B1'], 'Contrast order differs')
    budget = sci['budget']
    caps = budget['stage_caps']
    _require((caps['prepare'], caps['profile'], caps['predict_each_seed'], caps['fit_and_replay'], caps['apply'],
              caps['score'], budget['family_seconds'])
             == (300, 600, 600, 300, 300, 300, 3600), 'Stage caps differ')
    gate = budget['profile_gate']
    _require((gate['rows'], gate['seed'], gate['draws'], gate['cpanel_probe_rows'], gate['quality_computation'],
              gate['profile_probabilities_reused_as_full_member'])
             == (PROFILE_ROWS, 0, DRAWS, PROBE_ROWS, False, False), 'Profile gate differs')
    _require(sci['research_status']['independent_confirmation'] is None and
             sci['research_status']['held_out_confirmation'] is False, 'Research status differs')
    return sci


# ---------------------------------------------------------------- arrays, keys and probabilities

def validate_labels(labels, n=None, name='labels'):
    y = np.asarray(labels)
    if y.ndim != 1 or (n is not None and len(y) != n) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f'{name}: integer label vector required')
    if ((y < 0) | (y >= N_CLASSES)).any():
        raise ValueError(f'{name}: labels outside the ten-class contract')
    return y.astype(np.int64, copy=False)


def validate_simplex(probabilities, n, name):
    """Finite float64 ten-class rows in [0,1] summing to one (atol 1e-6, rtol 0); never renormalized."""
    p = np.asarray(probabilities)
    if p.shape != (n, N_CLASSES) or p.dtype != np.float64:
        raise ValueError(f'{name}: expected float64 [{n},{N_CLASSES}] probabilities')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f'{name}: probabilities must be finite and in [0,1]')
    if not np.allclose(p.sum(axis=1), 1.0, atol=SIMPLEX_ATOL, rtol=0):
        raise ValueError(f'{name}: probability mass differs from one')
    return p


def key_array(keys, name='keys'):
    k = np.asarray(keys)
    if k.ndim != 2 or k.shape[1] != 3 or not np.issubdtype(k.dtype, np.integer):
        raise ValueError(f'{name}: integer [n,3] pitch keys required')
    k = np.ascontiguousarray(k, dtype=np.int64)
    if len(np.unique(k, axis=0)) != len(k):
        raise ValueError(f'{name}: repeated pitch keys')
    return k


def ordered_key_sha256(keys):
    return array_sha256(key_array(keys), np.int64)


def subset_positions(whole_keys, subset_keys, name='subset'):
    """Positions of exact subset key tuples inside ``whole_keys``, in subset order; no missing rows."""
    whole, subset = key_array(whole_keys, 'whole keys'), key_array(subset_keys, name)
    lookup = {tuple(row): i for i, row in enumerate(whole.tolist())}
    try:
        return np.asarray([lookup[tuple(row)] for row in subset.tolist()], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f'{name}: key absent from the frozen population') from error


def compare_probabilities(actual, reference, *, atol, name):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.dtype != np.float64 or reference.dtype != np.float64:
        raise ValueError(f'{name}: replay shapes/dtypes differ')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError(f'{name}: nonfinite replay values')
    difference = float(np.abs(actual - reference).max()) if actual.size else 0.0
    return {'name': name, 'rows': int(len(actual)), 'maximum_absolute_difference': difference,
            'atol': atol, 'rtol': 0, 'passed': bool(difference <= atol)}


def compare_exact(actual, reference, *, name):
    """Exact value equality; string widths may differ, numeric/string kinds may not."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    same_kind = (actual.dtype.kind in 'US') == (reference.dtype.kind in 'US')
    equal = actual.shape == reference.shape and same_kind and np.array_equal(actual, reference)
    mismatches = int((actual != reference).sum()) if actual.shape == reference.shape else None
    return {'name': name, 'rows': int(len(actual)), 'mismatches': mismatches, 'passed': bool(equal)}


def require_passed(reports, context):
    failed = [r['name'] for r in reports if r['passed'] is not True]
    if failed:
        raise FamilyStop(f'{context}: replay gate failed for {failed}', {'reports': reports})
    return reports


# ---------------------------------------------------------------- groups

def volume_lookup(panel):
    """Frozen P4 D100 TRAIN mapping; labels re-derived from the frozen q25/q75 cutoffs."""
    thresholds = panel['volume_thresholds']
    q25, q75 = float(thresholds['q25']), float(thresholds['q75'])
    if (q25, q75) != VOLUME_THRESHOLDS:
        raise ValueError('Frozen TRAIN-volume cutoffs differ from 170/1514')
    lookup = {}
    for player in panel['train_players']:
        pitcher, n = int(player['pitcher']), player['train_pitches']
        expected = 'low' if n <= q25 else 'middle' if n <= q75 else 'high'
        if player['train_volume'] != expected or pitcher in lookup:
            raise ValueError('Conflicting frozen TRAIN-volume mapping')
        lookup[pitcher] = expected
    return lookup


def volume_groups(pitchers, lookup, recorded=None):
    """Absent-from-TRAIN is the valid zero group; recorded metadata must agree exactly."""
    pitchers = np.asarray(pitchers)
    if pitchers.ndim != 1 or not np.issubdtype(pitchers.dtype, np.integer):
        raise ValueError('Integer pitcher identities required for group assignment')
    groups = np.asarray([lookup.get(int(p), 'zero') for p in pitchers.tolist()], dtype='<U6')
    if recorded is not None:
        recorded = np.asarray(recorded).astype(str)
        if recorded.shape != groups.shape or not np.array_equal(recorded, groups):
            raise ValueError('Recorded TRAIN-volume metadata conflicts with the frozen mapping')
    return validate_groups(groups, len(pitchers))


def validate_groups(groups, n):
    groups = np.asarray(groups)
    if groups.shape != (n,) or groups.dtype.kind != 'U':
        raise ValueError('Unicode group vector aligned with rows required')
    unknown = sorted(set(np.unique(groups).tolist()) - set(GROUPS))
    if unknown:
        raise ValueError(f'Unknown TRAIN-volume group labels: {unknown}')
    return groups


def group_support(groups, games):
    groups, games = validate_groups(groups, len(games)), np.asarray(games)
    support = {}
    for name in GROUPS:
        mask = groups == name
        n, g = int(mask.sum()), int(len(np.unique(games[mask])))
        support[name] = {'pitches': n, 'games': g, 'supported': bool(n >= MIN_PITCHES and g >= MIN_GAMES)}
    return support


def require_expected_support(support, expected=EXPECTED_SUPPORT):
    observed = {g: {'pitches': support[g]['pitches'], 'games': support[g]['games']} for g in GROUPS}
    if observed != expected:
        raise ValueError(f'June TRAIN-volume support differs from registration: {observed}')
    return observed


# ---------------------------------------------------------------- bounded scalar optimizer

def blend_objective(y, model_p, frequency_p):
    """Literal legacy ``fit_blend`` log_loss objective, float64, floor 1e-12."""
    def score(weight):
        p = weight * model_p + (1 - weight) * frequency_p
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    return score


class CallLedger:
    """Every optimizer invocation, successful or not, in call order."""

    def __init__(self):
        self.calls = []

    def count(self, kind):
        return sum(1 for c in self.calls if c['kind'] == kind)

    def snapshot(self):
        return {'calls': list(self.calls), 'replay_calls': self.count('replay'),
                'candidate_calls': self.count('candidate'), 'total_calls': len(self.calls),
                'objective_evaluations_in_optimizer': int(sum(c.get('optimizer_objective_evaluations', 0)
                                                              for c in self.calls))}


def fit_scalar_blend(y, model_p, frequency_p, *, label, kind, anchor=None, ledger=None, minimizer=minimize_scalar):
    """Pinned legacy bounded search plus diagnostics; any failed gate raises FamilyStop, never falls back."""
    y = validate_labels(y, name=label)
    model_p = validate_simplex(model_p, len(y), label + ':model')
    frequency_p = validate_simplex(frequency_p, len(y), label + ':frequency')
    if not len(y):
        raise FamilyStop(f'{label}: empty fit subset', {'label': label})
    score = blend_objective(y, model_p, frequency_p)
    evaluations = [0]

    def counted(weight):
        evaluations[0] += 1
        return score(weight)

    record = {'label': label, 'kind': kind, 'n': int(len(y)), 'anchor': anchor,
              'settings': {k: (list(v) if isinstance(v, tuple) else v) for k, v in OPTIMIZER.items()}}
    if ledger is not None:
        ledger.calls.append(record)
    tol = OPTIMIZER['objective_tolerance']
    try:
        optimum = minimizer(counted, bounds=OPTIMIZER['bounds'], method=OPTIMIZER['method'],
                            options={'xatol': OPTIMIZER['xatol'], 'maxiter': OPTIMIZER['maxiter'],
                                     'disp': OPTIMIZER['disp']})
    except Exception as error:  # an optimizer exception is a failed call, recorded before stopping
        record.update(status='failed', failures=[f'optimizer_exception:{type(error).__name__}:{error}'],
                      optimizer_objective_evaluations=evaluations[0])
        raise FamilyStop(f'{label}: optimizer raised', record) from error
    x = float(np.asarray(optimum.x, dtype=np.float64))
    record.update(scipy_success=bool(optimum.success), scipy_status=int(optimum.status),
                  scipy_nfev=int(optimum.nfev), scipy_nit=int(getattr(optimum, 'nit', -1)),
                  scipy_message=str(optimum.message), optimum_x=x, optimum_fun=float(optimum.fun),
                  optimizer_objective_evaluations=evaluations[0])
    failures = []
    if record['scipy_success'] is not True:
        failures.append('scipy_success_false')
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        failures.append('optimum_not_finite_feasible')
    if not failures:
        choices = [0.0, x, 1.0]
        scores = [score(c) for c in choices]
        index = min(range(3), key=scores.__getitem__)  # Python min: first minimum wins, as legacy
        weight, objective = choices[index], scores[index]
        record.update(weight=weight, objective=objective, chosen_candidate=OPTIMIZER['choices_in_tie_order'][index],
                      candidate_objectives=scores, endpoint_objectives=[scores[0], scores[2]])
        if not math.isfinite(objective) or not all(map(math.isfinite, scores)):
            failures.append('objective_not_finite')
        elif objective > scores[0] + tol or objective > scores[2] + tol:
            failures.append('worse_than_endpoint')
        if anchor is not None:
            if not isinstance(anchor, float) or not math.isfinite(anchor) or not 0.0 <= anchor <= 1.0:
                failures.append('anchor_not_finite_feasible')
            else:
                anchor_objective = score(anchor)
                record.update(anchor_objective=anchor_objective, anchor_gap=objective - anchor_objective)
                if not math.isfinite(anchor_objective) or objective > anchor_objective + tol:
                    failures.append('worse_than_anchor')
    record['status'] = 'failed' if failures else 'passed'
    record['failures'] = failures
    if failures:
        raise FamilyStop(f'{label}: optimizer gate failed {failures}', record)
    return record


# ---------------------------------------------------------------- family fit

def blend(weight, model_p, frequency_p):
    return weight * model_p + (1 - weight) * frequency_p


def ensemble_mean(seed_calibrated):
    stacked = np.asarray(seed_calibrated)
    if stacked.ndim != 3 or stacked.shape[0] != len(SEEDS) or stacked.dtype != np.float64:
        raise ValueError('Five ordered float64 seed arrays required')
    return np.mean(np.stack([stacked[s] for s in range(len(SEEDS))]), axis=0)


def _components(seed_calibrated, ensemble):
    return {'ensemble': ensemble, **{f'seed{s}': seed_calibrated[s] for s in SEEDS}}


def check_b0_weights(weights):
    if not isinstance(weights, dict) or list(weights) != list(PREDICTORS):
        raise ValueError('Six ordered archived B0 weights required')
    for name, value in weights.items():
        if not isinstance(value, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f'Archived B0 weight must be finite in [0,1]: {name}')
    return weights


def _replay(replay, b0_weights, ledger, minimizer):
    y = validate_labels(replay['y'], name='replay labels')
    n = len(y)
    if n != CPANEL['rows']:
        raise FamilyStop(f'Replay population has {n} rows, not {CPANEL["rows"]}')
    frequency = validate_simplex(replay['frequency'], n, 'replay frequency')
    seeds = np.asarray(replay['seed_calibrated'])
    for s in SEEDS:
        validate_simplex(seeds[s], n, f'replay seed{s}')
    ensemble = validate_simplex(replay['calibrated'], n, 'replay ensemble')
    if not np.array_equal(ensemble, ensemble_mean(seeds)):
        raise FamilyStop('Archived Cpanel ensemble differs from the ordered five-seed mean')
    components = _components(seeds, ensemble)
    records, reports = {}, []
    for name in PREDICTORS:
        record = fit_scalar_blend(y, components[name], frequency, label=f'replay:{name}', kind='replay',
                                  ledger=ledger, minimizer=minimizer)
        difference = abs(record['weight'] - b0_weights[name])
        record.update(archived_weight=b0_weights[name], weight_absolute_difference=difference,
                      weight_replay_passed=bool(difference <= WEIGHT_REPLAY_ATOL))
        records[name] = record
        if not record['weight_replay_passed']:
            raise FamilyStop(f'replay:{name}: weight differs by {difference:.3g} > {WEIGHT_REPLAY_ATOL}',
                             {'record': record, 'ledger': ledger.snapshot()})
    primary = blend(b0_weights['ensemble'], ensemble, frequency)
    reports.append(compare_probabilities(primary, replay['primary'], atol=PROBABILITY_REPLAY_ATOL, name='B0_cpanel_primary'))
    for s in SEEDS:
        reports.append(compare_probabilities(blend(b0_weights[f'seed{s}'], seeds[s], frequency), replay['seed_primary'][s],
                                             atol=PROBABILITY_REPLAY_ATOL, name=f'B0_cpanel_seed{s}'))
    require_passed(reports, 'Archived B0 Cpanel prediction replay')
    return records, reports


def fit_family(*, june, replay, b0_weights, minimizer=minimize_scalar, ledger=None):
    """Six identity replays on archived Cpanel arrays, then 6 B1 + 24 B2 candidate fits.

    ``june`` holds y, frequency, seed_calibrated [5,n,10], groups, game_pk and the
    June/DEV identity flags checked by the caller.  Any failure raises FamilyStop
    with the complete call ledger; no fallback or tolerance change is attempted.
    """
    b0_weights = check_b0_weights(dict(b0_weights))
    ledger = CallLedger() if ledger is None else ledger
    try:
        replay_records, replay_reports = _replay(replay, b0_weights, ledger, minimizer)
        y = validate_labels(june['y'], name='June labels')
        n = len(y)
        frequency = validate_simplex(june['frequency'], n, 'June frequency')
        seeds = np.asarray(june['seed_calibrated'])
        if seeds.shape != (len(SEEDS), n, N_CLASSES):
            raise ValueError('June seed arrays must be [5,n,10]')
        for s in SEEDS:
            validate_simplex(seeds[s], n, f'June seed{s}')
        components = _components(seeds, ensemble_mean(seeds))
        groups = validate_groups(june['groups'], n)
        games = np.asarray(june['game_pk'])
        if games.shape != (n,):
            raise ValueError('June game identities must align')
        support = group_support(groups, games)
        b1 = {}
        b1_records = {}
        for name in PREDICTORS:
            record = fit_scalar_blend(y, components[name], frequency, label=f'B1:{name}', kind='candidate',
                                      anchor=b0_weights[name], ledger=ledger, minimizer=minimizer)
            b1[name], b1_records[name] = record['weight'], record
        b2 = {g: {} for g in GROUPS}
        b2_records = {g: {} for g in GROUPS}
        for g in GROUPS:
            mask = groups == g
            count = support[g]['pitches']
            rho = count / (count + SHRINKAGE_PITCHES)
            for name in PREDICTORS:
                if not support[g]['supported']:
                    b2[g][name] = b1[name]
                    b2_records[g][name] = {'fallback': True, 'optimizer_call': False, 'weight': b1[name],
                                           'support': support[g], 'reason': 'valid group below 500 pitches AND 30 games'}
                    continue
                record = fit_scalar_blend(y[mask], components[name][mask], frequency[mask], label=f'B2:{g}:{name}',
                                          kind='candidate', anchor=b1[name], ledger=ledger, minimizer=minimizer)
                local = record['weight']
                shrunk = rho * local + (1 - rho) * b1[name]
                if not math.isfinite(shrunk) or not 0.0 <= shrunk <= 1.0:
                    raise FamilyStop(f'B2:{g}:{name}: shrunk weight infeasible', {'record': record})
                b2[g][name] = shrunk
                b2_records[g][name] = {'fallback': False, 'optimizer_call': True, 'local_mle': local, 'rho': rho,
                                       'anchor_b1': b1[name], 'weight': shrunk, 'record': record}
    except FamilyStop as stop:
        stop.diagnostics = {**stop.diagnostics, 'ledger': ledger.snapshot()}
        raise
    except ValueError as error:
        raise FamilyStop(str(error), {'ledger': ledger.snapshot()}) from error
    counts = ledger.snapshot()
    fallback_rows = int(sum(support[g]['pitches'] for g in GROUPS if not support[g]['supported']))
    return {'predictors': list(PREDICTORS), 'groups': list(GROUPS), 'B0_weights': b0_weights, 'B1_weights': b1,
            'B2_weights': b2, 'B2_support': support, 'B2_fallback_rows': fallback_rows,
            'B2_fallback_groups': [g for g in GROUPS if not support[g]['supported']],
            'shrinkage_pitches': SHRINKAGE_PITCHES, 'replay': replay_records, 'replay_prediction_reports': replay_reports,
            'B1_fits': b1_records, 'B2_fits': b2_records, 'call_ledger': counts,
            'fit_losses_are_quality_estimates': False, 'in_sample_objectives': 'optimizer audit only'}


def require_expected_calls(fit_record):
    ledger = fit_record['call_ledger']
    supported = sum(1 for g in GROUPS if fit_record['B2_support'][g]['supported'])
    expected_candidates = len(PREDICTORS) * (1 + supported)
    if (ledger['replay_calls'], ledger['candidate_calls']) != (EXPECTED_CALLS['replay'], expected_candidates):
        raise FamilyStop('Optimizer call inventory differs from the registered replay/candidate counts', ledger)
    return {'replay': ledger['replay_calls'], 'candidate': ledger['candidate_calls'], 'total': ledger['total_calls']}


# ---------------------------------------------------------------- label-free application

def apply_family(fit_record, *, seed_calibrated, frequency, groups):
    """B0/B1/B2 ensemble and seed predictions from frozen components; no labels are involved."""
    seeds = np.asarray(seed_calibrated)
    n = seeds.shape[1]
    frequency = validate_simplex(frequency, n, 'frequency')
    for s in SEEDS:
        validate_simplex(seeds[s], n, f'seed{s}')
    groups = validate_groups(groups, n)
    components = _components(seeds, ensemble_mean(seeds))
    codes = np.full(n, -1, dtype=np.int64)
    for i, name in enumerate(GROUPS):
        codes[groups == name] = i
    if (codes < 0).any():
        raise ValueError('Group coding failed')
    out = {}
    for variant in VARIANTS:
        arrays = {}
        for name in PREDICTORS:
            if variant == 'B2':
                table = np.asarray([fit_record['B2_weights'][g][name] for g in GROUPS], dtype=np.float64)
                w = table[codes][:, None]
                arrays[name] = w * components[name] + (1 - w) * frequency
            else:
                arrays[name] = blend(fit_record[variant + '_weights'][name], components[name], frequency)
            validate_simplex(arrays[name], n, f'{variant}:{name}')
        out[variant + '_primary'] = arrays['ensemble']
        out[variant + '_seed_primary'] = np.stack([arrays[f'seed{s}'] for s in SEEDS])
    out['ensemble_calibrated'] = components['ensemble']
    return out
