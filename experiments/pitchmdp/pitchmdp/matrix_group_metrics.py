"""Predeclared group guardrails and descriptive temporal interaction estimates."""
from __future__ import annotations
import numpy as np
from .matrix_metrics import pitch_losses
from .matrix_panel import group_reporting_status

GROUPS = {'role_starter': ('game_role', 'starter'), 'role_relief': ('game_role', 'relief'),
    'hand_L': ('throwing_hand', 'L'), 'hand_R': ('throwing_hand', 'R'),
    'volume_low': ('train_volume', 'low'), 'volume_middle': ('train_volume', 'middle'),
    'volume_high': ('train_volume', 'high'), 'volume_zero': ('train_volume', 'zero'),
    'two_strikes': ('two_strikes', True), 'less_two_strikes': ('two_strikes', False),
    'runners_on': ('runners_on', True), 'bases_empty': ('runners_on', False)}


def masks(metadata):
    return {name: metadata[column].eq(value).fillna(False).to_numpy(bool)
            for name, (column, value) in GROUPS.items()}


def bootstrap_difference(delta, games, *, draws=10000, seed=20260924, upper_alpha=.025):
    delta, games = np.asarray(delta, float), np.asarray(games)
    if delta.ndim != 2 or delta.shape[1] != 2 or len(games) != len(delta):
        raise ValueError('Aligned two-loss matrix and games required')
    if not np.isfinite(delta).all() or not 0 < upper_alpha < .5:
        raise ValueError('Finite losses and declared upper-bound alpha required')
    unique, inverse = np.unique(games, return_inverse=True)
    if len(unique) < 2:
        return None
    count = np.bincount(inverse).astype(float)
    sums = np.column_stack([np.bincount(inverse, weights=delta[:, j]) for j in range(2)])
    rng = np.random.default_rng(seed)
    replicates = np.empty((draws, 2))
    for begin in range(0, draws, 256):
        ix = rng.integers(len(unique), size=(min(256, draws - begin), len(unique)))
        replicates[begin:begin + len(ix)] = sums[ix].sum(1) / count[ix].sum(1)[:, None]
    return {'delta': delta.mean(0).tolist(), 'ci95': np.quantile(replicates, [.025, .975], axis=0).T.tolist(),
            'simultaneous_upper': np.quantile(replicates, 1 - upper_alpha, axis=0).tolist(),
            'upper_alpha': upper_alpha, 'draws': draws, 'seed': seed}


def guardrails(labels, candidate, control, games, metadata, *, candidate_family_size=4):
    delta = pitch_losses(labels, candidate) - pitch_losses(labels, control)
    if len(metadata) != len(delta) or candidate_family_size != 4:
        raise ValueError('G family fixes four comparisons and aligned metadata')
    alpha = .05 / (len(GROUPS) * 2 * candidate_family_size)
    results = {}
    for name, mask in masks(metadata).items():
        gate = group_reporting_status(games, mask)
        measured = bootstrap_difference(delta[mask], np.asarray(games)[mask], upper_alpha=alpha) if gate['status'] == 'reporting_eligible' else None
        passed = measured['simultaneous_upper'][0] <= .010 and measured['simultaneous_upper'][1] <= .002 if measured else None
        results[name] = {'reporting': gate, 'paired': measured, 'passed': passed}
    return {'groups': results, 'family_size': len(GROUPS) * 2 * candidate_family_size,
            'family_alpha': .05, 'per_bound_alpha': alpha,
            'status': 'unconfirmed' if any(r['passed'] is None for r in results.values()) else
                      'passed' if all(r['passed'] for r in results.values()) else 'failed',
            'note': 'All declared groups retained, including zero-TRAIN players absent from a TRAIN-selected panel.'}


def volume_interaction(labels, candidate, control, games, metadata, *, draws=10000, seed=20260924):
    low = metadata.train_volume.eq('low').to_numpy(bool)
    high = metadata.train_volume.eq('high').to_numpy(bool)
    gates = {name: group_reporting_status(games, mask) for name, mask in [('low', low), ('high', high)]}
    if any(gate['status'] != 'reporting_eligible' for gate in gates.values()):
        return {'status': 'unmeasured', 'reporting': gates, 'delta_low_minus_high': None}
    loss = (pitch_losses(labels, candidate) - pitch_losses(labels, control))[:, 0]
    keep = low | high
    unique, inverse = np.unique(np.asarray(games)[keep], return_inverse=True)
    sums, counts = [], []
    for mask in (low[keep], high[keep]):
        sums.append(np.bincount(inverse, weights=loss[keep] * mask, minlength=len(unique)))
        counts.append(np.bincount(inverse, weights=mask, minlength=len(unique)))
    sums, counts = np.column_stack(sums), np.column_stack(counts)
    rng, samples = np.random.default_rng(seed), []
    for begin in range(0, draws, 256):
        ix = rng.integers(len(unique), size=(min(256, draws - begin), len(unique)))
        denominator = counts[ix].sum(1)
        valid = (denominator > 0).all(1)
        means = sums[ix].sum(1)[valid] / denominator[valid]
        samples.extend((means[:, 0] - means[:, 1]).tolist())
    return {'status': 'descriptive', 'reporting': gates,
        'delta_low_minus_high': float(loss[low].mean() - loss[high].mean()),
        'ci95': np.quantile(samples, [.025, .975]).tolist(), 'valid_bootstrap_draws': len(samples),
        'note': 'Effect heterogeneity across observed groups; not randomized sample-size effect.'}
