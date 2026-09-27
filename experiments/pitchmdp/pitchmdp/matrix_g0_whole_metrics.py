"""Standalone five-seed G0 evaluation; all inference and calibration are external."""
from __future__ import annotations

import numpy as np

from .data import KEY
from .matrix_metrics import (validate_probabilities, prediction_metrics, pitch_losses,
                             paired_game_comparison, holm_adjust)
from .matrix_group_metrics import masks, bootstrap_difference
from .matrix_panel import group_reporting_status


def frozen_five_predictions(members, frequency, archived_report):
    """Apply archived five-seed June weights without fitting any parameter."""
    if len(members) != 5 or len(archived_report['seeds']) != 5:
        raise ValueError('Exactly five ordered members and archived seed weights required')
    y = frequency['dev_y']
    validate_probabilities(y, frequency['dev'])
    for member in members:
        for suffix in ('keys', 'y', 'game_pk', 'pitcher'):
            if not np.array_equal(member['dev_' + suffix], frequency['dev_' + suffix]):
                raise ValueError('Member row alignment differs: ' + suffix)
        for name in ('dev', 'dev_raw'):
            validate_probabilities(y, member[name])
    weights = [archived_report['selection']['model_weight'],
               *[r['blend_selection']['model_weight'] for r in archived_report['seeds']]]
    if not all(np.isfinite(w) and 0 <= w <= 1 for w in weights):
        raise ValueError('Invalid archived blend weight')
    calibrated = np.mean([m['dev'] for m in members], axis=0)
    return {'calibrated': calibrated, 'raw': np.mean([m['dev_raw'] for m in members], axis=0),
            'primary': weights[0] * calibrated + (1 - weights[0]) * frequency['dev'],
            'seed_primary': np.stack([w * m['dev'] + (1-w)*frequency['dev']
                                      for w, m in zip(weights[1:], members)])}


def aligned_population(frequency, metadata, panel_frequency):
    """Define complement by exact keys, never by an inferred player label alone."""
    keys = np.asarray(frequency['dev_keys'])
    panel_keys = np.asarray(panel_frequency['dev_keys'])
    if keys.ndim != 2 or keys.shape[1] != len(KEY) or panel_keys.ndim != 2 or panel_keys.shape[1] != len(KEY):
        raise ValueError('Malformed pitch keys')
    whole_map = {tuple(k): i for i, k in enumerate(keys)}
    if len(whole_map) != len(keys) or len(set(map(tuple, panel_keys))) != len(panel_keys):
        raise ValueError('Duplicate whole or panel key')
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), keys):
        raise ValueError('Metadata ordering differs')
    if not np.array_equal(keys[:, 0], frequency['dev_game_pk']):
        raise ValueError('Key game differs')
    try:
        panel_indices = np.asarray([whole_map[tuple(k)] for k in panel_keys], dtype=np.int64)
    except KeyError as error:
        raise ValueError('Missing panel key') from error
    for suffix in ('y', 'game_pk', 'pitcher'):
        if not np.array_equal(np.asarray(frequency['dev_'+suffix])[panel_indices], panel_frequency['dev_'+suffix]):
            raise ValueError('Panel identity differs: '+suffix)
    mask = np.zeros(len(keys), dtype=bool)
    mask[panel_indices] = True
    if not np.array_equal(mask, metadata.in_cpanel.to_numpy(bool)):
        raise ValueError('Exact panel intersection differs from metadata')
    games = np.asarray(frequency['dev_game_pk'])
    return {'whole': np.ones(len(keys), dtype=bool), 'cpanel': mask, 'non_cpanel': ~mask}, panel_indices, {
        'overlap_pitches': int(mask.sum()), 'overlap_games': int(len(np.unique(games[mask]))),
        'complement_pitches': int((~mask).sum()), 'complement_games': int(len(np.unique(games[~mask]))),
        'shared_games_panel_complement': int(len(np.intersect1d(games[mask], games[~mask])))}


def replay_panel(predictions, panel_predictions, indices, *, atol=1e-6):
    errors = {}
    for name in ('raw', 'calibrated', 'primary', 'seed_primary'):
        actual = predictions[name][:, indices] if name == 'seed_primary' else predictions[name][indices]
        expected = panel_predictions[name]
        if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=0):
            raise ValueError('Frozen C1 panel replay failed: '+name)
        errors[name] = float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    return {'passed': True, 'absolute_tolerance': atol, 'max_abs_errors': errors}


def comparison_decision(pair, adjusted_p, seed_deltas):
    """Five seeds are a direction criterion, never five independent game samples."""
    if pair.get('status') in ('unmeasured', 'insufficient_games'):
        return {'status': 'inconclusive', 'negative_seeds': None}
    nll, brier = pair['nll'], pair['brier']
    count = sum(v < 0 for v in seed_deltas)
    passed = (nll['delta'] <= -.003 and nll['ci95'][1] < 0 and adjusted_p <= .05
              and brier['ci95'][1] <= .001 and count >= 4)
    worse = nll['ci95'][0] > 0 or brier['ci95'][0] > .001
    return {'status': 'development_improvement' if passed else 'worse_or_guardrail_failure' if worse else 'inconclusive',
            'negative_seeds': count, 'required_negative_seeds': 4, 'adjusted_p': adjusted_p}


def absolute_interval(y, probabilities, games, *, draws=10000):
    value = bootstrap_difference(pitch_losses(y, probabilities), games, draws=draws)
    return None if value is None else {'nll_ci95': value['ci95'][0], 'brier_ci95': value['ci95'][1],
        'draws': draws, 'seed': 20260924, 'uncertainty': 'Fixed predictor; whole-game pitch-weighted bootstrap'}


def robust_bounds(y, candidate, control, games, named_masks, *, family_size, draws=100000):
    if family_size not in (24, 52):
        raise ValueError('Only preregistered standalone or correction bound inventories supported')
    losses = pitch_losses(y, candidate) - pitch_losses(y, control)
    groups = {}
    for name, mask in named_masks.items():
        gate = group_reporting_status(games, mask)
        measured = bootstrap_difference(losses[mask], np.asarray(games)[mask], draws=draws,
                                        upper_alpha=.05/family_size) if gate['status'] == 'reporting_eligible' else None
        groups[name] = {'reporting': gate, 'paired': measured, 'passed': bool(
            measured['simultaneous_upper'][0] <= .010 and measured['simultaneous_upper'][1] <= .002) if measured else None}
    status = 'failed' if any(v['passed'] is False for v in groups.values()) else 'unconfirmed' if any(
        v['passed'] is None for v in groups.values()) else 'passed'
    return {'status': status, 'groups': groups, 'family_size': family_size,
            'family_alpha': .05, 'per_bound_alpha': .05/family_size, 'draws': draws}


def evaluate_whole(frequency, metadata, predictions, panel_frequency, *, draws=10000, r_draws=100000):
    """Return stage-2 summaries without reading any artifact or mutating parents."""
    population, indices, overlap = aligned_population(frequency, metadata, panel_frequency)
    y, games, base = frequency['dev_y'], frequency['dev_game_pk'], frequency['dev']
    primary = predictions['primary']
    if predictions['seed_primary'].shape != (5, len(y), 10):
        raise ValueError('Five aligned seed primary arrays required')
    pairs, seed_deltas = [], []
    for name in ('whole', 'non_cpanel'):
        mask = population[name]
        pairs.append(paired_game_comparison(y[mask], primary[mask], base[mask], games[mask], draws=draws))
        seed_deltas.append([float((pitch_losses(y[mask], p[mask])-pitch_losses(y[mask], base[mask]))[:, 0].mean())
                            for p in predictions['seed_primary']])
    adjusted = holm_adjust([p['nll']['p_less'] if p.get('nll') else None for p in pairs])
    estimates = {}
    for i, name in enumerate(('whole', 'non_cpanel')):
        mask = population[name]
        decision = comparison_decision(pairs[i], adjusted[i], seed_deltas[i])
        if decision['status'] == 'development_improvement':
            decision['status'] = 'development_advantage_over_frequency'
        estimates[name] = {'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
            'g0': prediction_metrics(y[mask], primary[mask]), 'frequency': prediction_metrics(y[mask], base[mask]),
            'g0_absolute_ci': absolute_interval(y[mask], primary[mask], games[mask], draws=draws),
            'frequency_absolute_ci': absolute_interval(y[mask], base[mask], games[mask], draws=draws),
            'paired': pairs[i], 'seed_nll_deltas': seed_deltas[i], 'N': decision}
    subgroup_masks = masks(metadata)
    descriptive = {'cpanel': population['cpanel'], **subgroup_masks}
    for field in ('seen_pitcher', 'seen_batter'):
        for seen in (True, False):
            descriptive[('seen_' if seen else 'unseen_') + field.removeprefix('seen_')] = metadata[field].eq(seen).to_numpy(bool)
    descriptive.update({'month_'+str(month): metadata.month.eq(month).to_numpy(bool) for month in sorted(metadata.month.unique())})
    slices = {name: {'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))), 'status': 'descriptive',
        'g0': prediction_metrics(y[mask], primary[mask]), 'frequency': prediction_metrics(y[mask], base[mask])}
        for name, mask in descriptive.items()}
    players = {}
    for pitcher in sorted(np.unique(frequency['dev_pitcher'])):
        mask = frequency['dev_pitcher'] == pitcher
        players[str(int(pitcher))] = {'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
            'reporting': group_reporting_status(games, mask),
            **{name: {metric: value for metric, value in prediction_metrics(y[mask], prob[mask]).items()
                      if metric in ('n', 'log_loss', 'brier_multiclass', 'top_label_ece10')}
               for name, prob in (('g0', primary), ('frequency', base))}}
    macro = {model: {metric: float(np.mean([p[model][metric] for p in players.values()]))
                    for metric in ('log_loss', 'brier_multiclass')} for model in ('g0', 'frequency')}
    return {'scope': 'Previously exposed eligible 2025 whole-MLB DEV', 'n': len(y),
        'games': int(len(np.unique(games))), 'estimands': estimates, 'overlap': overlap,
        'reports': {name: prediction_metrics(y, predictions[name]) for name in ('primary', 'calibrated', 'raw')},
        'R': robust_bounds(y, primary, base, games, subgroup_masks, family_size=24, draws=r_draws),
        'multiplicity': {'primary_nll_tests': 2, 'method': 'Holm', 'adjusted_p': adjusted},
        'slices': slices, 'per_pitcher': players, 'pitcher_macro': macro,
        'class_reporting': {'minimum_events': 30, 'eligible': [int((y == c).sum()) >= 30 for c in range(10)]},
        'metadata_missing': {name: int(metadata[name].isna().sum()) for name in metadata.columns},
        'metadata_unknown': {name: int(metadata[name].eq('unknown').sum()) for name in ('game_role', 'throwing_hand')},
        'independent_confirmation': None, 'held_out_confirmation': False, 'policy_effect': None, 'service_adoption': None,
        'limits': ['Cpanel exact-row complement is not independent; games overlap and T4 DEV was exposed.',
                   'Retrospective eligibility is not live pre-pitch availability.',
                   'TRAIN0 may have earlier legal DEV context; not strict zero-shot.',
                   'Bootstrap conditions on fitted models/calibration/selection; local tests do not restore global alpha.']}


def evaluate_candidates(y, games, metadata, baseline, candidates, activation, *, draws=10000, r_draws=100000):
    """Two fixed I1/I2 slots; activation is {'I1': bool, 'I2': bool}.

    Baseline/candidate mappings contain primary (N,10) and seed_primary (5,N,10).
    Metadata must already pass the exact-key stage-2 alignment gate.
    """
    names = ('I1', 'I2')
    if set(activation) != set(names) or any(type(activation[n]) is not bool for n in names):
        raise ValueError('Exactly two explicit boolean activation slots required')
    if set(candidates) - set(names) or any(activation[n] != (n in candidates) for n in names):
        raise ValueError('Candidate inventory differs from sealed activation')
    y, games = np.asarray(y), np.asarray(games)
    if len(metadata) != len(y) or len(games) != len(y):
        raise ValueError('Misaligned evaluation metadata')
    if np.asarray(baseline['seed_primary']).shape != (5, len(y), 10):
        raise ValueError('Five baseline seed predictions required')
    validate_probabilities(y, baseline['primary'])
    group_masks = {**masks(metadata), 'non_cpanel': ~metadata.in_cpanel.to_numpy(bool)}
    pairs, deltas = {}, {}
    for name, candidate in candidates.items():
        if np.asarray(candidate['seed_primary']).shape != (5, len(y), 10):
            raise ValueError('Five candidate seed predictions required')
        validate_probabilities(y, candidate['primary'])
        pairs[name] = paired_game_comparison(y, candidate['primary'], baseline['primary'], games, draws=draws)
        deltas[name] = [float((pitch_losses(y, p)-pitch_losses(y, q))[:, 0].mean())
                       for p, q in zip(candidate['seed_primary'], baseline['seed_primary'])]
    adjusted = holm_adjust([pairs[n]['nll']['p_less'] if n in pairs else 1.0 for n in names])
    results = {}
    for i, name in enumerate(names):
        if name not in candidates:
            results[name] = {'status': 'not_activated', 'N': None, 'adjusted_p': adjusted[i],
                'R': {'status': 'not_activated', 'family_size': 52,
                      'groups': {g: {'passed': None, 'status': 'not_activated'} for g in group_masks}}}
            continue
        candidate = candidates[name]
        sliced = {}
        description = {'cpanel': metadata.in_cpanel.to_numpy(bool), **group_masks}
        for column in ('seen_pitcher', 'seen_batter'):
            for seen in (True, False):
                description[('seen_' if seen else 'unseen_')+column.removeprefix('seen_')] = metadata[column].eq(seen).to_numpy(bool)
        description.update({'month_'+str(month): metadata.month.eq(month).to_numpy(bool) for month in sorted(metadata.month.unique())})
        for slice_name, mask in description.items():
            sliced[slice_name] = {'status': 'descriptive', 'n': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
                'candidate': prediction_metrics(y[mask], candidate['primary'][mask]),
                'baseline': prediction_metrics(y[mask], baseline['primary'][mask])}
            if slice_name in ('cpanel', 'non_cpanel'):
                sliced[slice_name]['paired'] = paired_game_comparison(y[mask], candidate['primary'][mask],
                    baseline['primary'][mask], games[mask], draws=draws)
        results[name] = {'status': 'evaluated', 'metrics': prediction_metrics(y, candidate['primary']),
            'baseline_metrics': prediction_metrics(y, baseline['primary']), 'paired': pairs[name],
            'absolute_ci': absolute_interval(y, candidate['primary'], games, draws=draws),
            'seed_nll_deltas': deltas[name], 'N': comparison_decision(pairs[name], adjusted[i], deltas[name]),
            'R': robust_bounds(y, candidate['primary'], baseline['primary'], games, group_masks,
                               family_size=52, draws=r_draws), 'slices': sliced}
    return {'candidates': results, 'activation': activation,
        'multiplicity': {'primary_nll_tests': 2, 'slot_order': list(names), 'method': 'Holm; inactive p=1',
                         'adjusted_p': adjusted, 'R_family_size': 52},
        'independent_confirmation': None, 'held_out_confirmation': False, 'policy_effect': None,
        'service_adoption': None, 'stage': 'Adaptive diagnostic-driven development comparison',
        'limits': ['Fixed two slots do not restore fresh alpha after DEV-driven activation.',
                   'Cpanel June correction fitting may fail to transfer to other pitchers.',
                   'Inactive and insufficient-support bounds retain the original 52-slot denominator.']}
