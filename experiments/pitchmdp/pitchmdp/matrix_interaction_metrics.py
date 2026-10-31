"""Paired pitch-loss factorial contrasts; probability averaging is not a loss contrast."""
from __future__ import annotations
import numpy as np

ORDER = ('MLP25', 'MLP100', 'TF25', 'TF100')
CONTRASTS = {'interaction': (1, -1, -1, 1), 'data_main': (-.5, .5, -.5, .5),
             'architecture_main': (-.5, -.5, .5, .5), 'mlp_data': (-1, 1, 0, 0),
             'transformer_data': (0, 0, -1, 1), 'architecture_D25': (-1, 0, 1, 0),
             'architecture_D100': (0, -1, 0, 1)}


def factorial_contrasts(losses, games, *, draws=10000, seed=20260924):
    values, games = np.asarray(losses, dtype=float), np.asarray(games)
    if values.ndim != 3 or values.shape[1:] != (4, 2) or games.shape != (len(values),) or not np.isfinite(values).all():
        raise ValueError('Aligned [pitches, four cells, NLL/Brier] losses required')
    unique, inverse = np.unique(games, return_inverse=True)
    if len(unique) < 2 or draws < 1:
        raise ValueError('At least two games and positive draws required')
    coefficients = np.asarray(list(CONTRASTS.values()))
    delta = np.einsum('ncm,kc->nkm', values, coefficients)
    count = np.bincount(inverse).astype(float)
    sums = np.stack([np.column_stack([np.bincount(inverse, weights=delta[:, k, j])
                                     for j in range(2)]) for k in range(len(coefficients))], axis=1)
    rng, samples = np.random.default_rng(seed), np.empty((draws, len(coefficients), 2))
    for begin in range(0, draws, 256):
        ix = rng.integers(len(unique), size=(min(256, draws-begin), len(unique)))
        samples[begin:begin+len(ix)] = sums[ix].sum(1) / count[ix].sum(1)[:, None, None]
    mean = delta.mean(0)
    result = {}
    for k, name in enumerate(CONTRASTS):
        result[name] = {}
        for j, metric in enumerate(('nll', 'brier')):
            null = samples[:, k, j]-mean[k, j]
            p = (1 + int((np.abs(null) >= abs(mean[k, j])).sum())) / (draws+1)
            result[name][metric] = {'delta': float(mean[k, j]),
                'ci95': np.quantile(samples[:, k, j], [.025, .975]).tolist(), 'p_two_sided_centered': p}
    return {'contrasts': result, 'cell_order': list(ORDER), 'coefficients': CONTRASTS,
            'draws': draws, 'seed': seed, 'n': len(values), 'games': len(unique),
            'unit': 'whole game, pitch-weighted paired loss contrast'}


def interaction_decision(contrast, seed_interactions):
    values = np.asarray(seed_interactions, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Three finite seed interactions required')
    delta = contrast['delta']
    direction = 'transformer_benefits_more' if delta < 0 else 'mlp_benefits_more' if delta > 0 else 'none'
    same = int((values*delta > 0).sum())
    checks = {'practical_magnitude': abs(delta) >= .003,
        'ci_excludes_zero': contrast['ci95'][1] < 0 or contrast['ci95'][0] > 0,
        'two_sided_p': contrast['p_two_sided_centered'] <= .05, 'seed_direction': same >= 2}
    return {'status': 'interaction_detected' if all(checks.values()) else 'inconclusive',
            'direction': direction, 'checks': checks, 'same_direction_seeds': same,
            'seed_interactions': values.tolist(), 'policy_effect': None}
