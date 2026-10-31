"""Descriptive frozen-TRAIN-role/pitcher reports; no new inference or selection."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
POLICY_PAIRS = [('P1', 'P0'), ('P2', 'P1'), ('P3', 'P2')]
RL_PAIRS = [('IQL', 'planner'), ('CQL', 'planner'), ('IQL', 'NNBC'), ('CQL', 'NNBC')]
LIMITS = ['Descriptive only: no additional tests, confidence intervals or promotion claims',
          'TRAIN role is a frozen player classification, not actual role in this game',
          'WE summaries are model-internal conditional simulations, not observational OPE or causal effects',
          'Unselected requests are retained in denominators; their policy value is unmeasured']


def keys(frame):
    raw = frame[KEY].to_numpy()
    try: numeric = raw.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc: raise ValueError('Invalid pitch keys') from exc
    if not np.array_equal(raw, numeric) or len(set(map(tuple, numeric))) != len(frame):
        raise ValueError('Noninteger or duplicate pitch keys')
    return numeric


def with_roles(frame, panel):
    result = frame.copy().reset_index(drop=True)
    keys(result)
    players = panel['train_players']
    mapping = {int(row['pitcher']): row['train_role'] for row in players}
    if len(mapping) != len(players): raise ValueError('Duplicate frozen TRAIN pitcher metadata')
    role = result.pitcher.map(mapping)
    if role.isna().any(): raise ValueError('Pitcher missing from frozen TRAIN role metadata')
    if 'train_role' in result and not np.array_equal(result.train_role.to_numpy(), role.to_numpy()):
        raise ValueError('Request role differs from frozen TRAIN role')
    result['train_role'] = role
    return result


def groups(frame):
    yield 'overall', np.ones(len(frame), dtype=bool)
    for column in ('train_role', 'pitcher'):
        for value in sorted(frame[column].unique(), key=str):
            yield column + ':' + str(value), frame[column].eq(value).to_numpy(bool)


def p0_groups(frame, arrays, actions, panel):
    frame = with_roles(frame, panel)
    if not np.array_equal(keys(frame), arrays['keys']): raise ValueError('P0 archive/request key alignment differs')
    n, width = len(frame), len(actions)
    if len(set(actions)) != width: raise ValueError('Duplicate action vocabulary')
    expected_labels = np.array([actions.index(a) if a in actions else -1 for a in frame.observed_action])
    labels = np.asarray(arrays['labels'])
    if not np.array_equal(labels, expected_labels): raise ValueError('P0 observed action labels differ')
    flags = {}
    for name in ('supported', 'observed_supported', 'fallback'):
        value = np.asarray(arrays[name])
        if value.shape != (n,) or value.dtype != np.bool_: raise ValueError('Invalid P0 support flags')
        flags[name] = value
    probabilities = {}
    for name in ('bc', 'frequency'):
        p = np.asarray(arrays[name], dtype=float)
        if p.shape != (n, width) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Invalid P0 probabilities')
        if not np.allclose(p[flags['supported']].sum(1), 1., rtol=0, atol=1e-6):
            raise ValueError('P0 supported probability mass differs')
        probabilities[name] = p
    expected = flags['supported'] & (labels >= 0)
    expected &= probabilities['bc'][np.arange(n), np.maximum(labels, 0)] > 0
    if not np.array_equal(expected, flags['observed_supported']) or (flags['fallback'] & ~flags['supported']).any():
        raise ValueError('P0 observed support or fallback inconsistent')
    result = {}
    for name, mask in groups(frame):
        valid = np.flatnonzero(mask & flags['observed_supported'])
        scores = {}
        for model, p in probabilities.items():
            if len(valid) and (p[valid, labels[valid]] <= 0).any(): raise ValueError('Conditional label lacks common support')
            value = float(-np.log(p[valid, labels[valid]]).mean()) if len(valid) else None
            scores[model] = {'conditional_action_nll': value,
                'full_requested_action_nll': value if len(valid) == int(mask.sum()) else None}
        result[name] = {'requested_pitches': int(mask.sum()), 'games': int(frame.loc[mask, 'game_pk'].nunique()),
            'supported_requests': int((mask & flags['supported']).sum()),
            'supported_observed_labels': len(valid), 'unsupported_observed_labels': int(mask.sum())-len(valid),
            'unknown_pitcher_fallbacks': int((mask & flags['fallback']).sum()), 'models': scores}
    return {'groups': result, 'limits': LIMITS, 'action_agreement_is_not_policy_value': True}


def rollout_groups(requests, pa_keys, games, values, truncated, pairs, selected_limit, panel):
    frame = with_roles(requests, panel)
    if frame.duplicated(KEY[:2]).any(): raise ValueError('PA request denominator contains duplicate starts')
    for name in ('selected', 'supported'):
        if frame[name].dtype != np.bool_: raise ValueError('Invalid PA request flags')
    if (frame.supported & ~frame.selected).any(): raise ValueError('Unselected PA marked supported')
    if type(selected_limit) is not int or selected_limit < 1: raise ValueError('Invalid execution selection limit')
    prepared = frame.selected.to_numpy(bool)
    chosen = np.zeros(len(frame), dtype=bool)
    chosen[np.flatnonzero(prepared)[:selected_limit]] = True
    observed = chosen & frame.supported.to_numpy(bool)
    expected = [':'.join(map(str, row)) for row in keys(frame.loc[observed])]
    if list(pa_keys) != expected or not np.array_equal(games, frame.loc[observed, 'game_pk'].to_numpy()):
        raise ValueError('Exact common supported PA keys/game order differs')
    if not expected or set(values) != set(truncated): raise ValueError('Missing common rollout family')
    shape = None
    for name, array in values.items():
        array, flag = np.asarray(array), np.asarray(truncated[name])
        if array.ndim != 2 or array.shape[0] != len(expected) or array.shape[1] < 1:
            raise ValueError('Invalid PA/MC rollout shape')
        if shape is not None and array.shape != shape: raise ValueError('Unequal common MC rollout shapes')
        shape = array.shape
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any(): raise ValueError('Invalid WE values')
        if flag.dtype != np.bool_ or flag.shape != shape: raise ValueError('Invalid truncation flags')
    if any(a not in values or b not in values for a, b in pairs): raise ValueError('Missing registered descriptive contrast')
    result = {}
    for name, mask in groups(frame):
        common = mask[observed]
        count = int(common.sum())
        report = {'requested_pa_starts': int(mask.sum()), 'requested_games': int(frame.loc[mask, 'game_pk'].nunique()),
            'preparation_selected': int((mask & prepared).sum()), 'execution_selected': int((mask & chosen).sum()),
            'not_execution_selected': int((mask & ~chosen).sum()), 'supported_selected': count,
            'selected_unsupported': int((mask & chosen & ~observed).sum()),
            'supported_games': int(frame.loc[mask & observed, 'game_pk'].nunique()),
            'unsupported_reasons': frame.loc[mask & chosen & ~observed, 'reason'].fillna('unspecified').value_counts().to_dict(),
            'policies': {}, 'paired_delta_we': {}}
        for policy, value in values.items():
            report['policies'][policy] = {'mean_original_defensive_we': float(value[common].mean()) if count else None,
                'truncated_rollout_rate': float(truncated[policy][common].mean()) if count else None}
        for a, b in pairs:
            report['paired_delta_we'][a+'_minus_'+b] = float((values[a][common]-values[b][common]).mean()) if count else None
        result[name] = report
    return {'groups': result, 'common_pa_keys': expected, 'mc_rollouts_per_pa': shape[1],
        'pairs': [list(p) for p in pairs], 'limits': LIMITS,
        'value_population': 'Execution-selected supported starts only; never imputed to unsupported or unselected requests'}
