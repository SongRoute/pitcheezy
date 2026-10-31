"""Frozen-calibration stress summaries and predeclared robustness bounds."""
from __future__ import annotations
import numpy as np
from .matrix_metrics import validate_probabilities, pitch_losses
from .matrix_group_metrics import bootstrap_difference
from .matrix_panel import group_reporting_status

FAMILY_SIZE = 270  # 9 * (13 candidate-control groups + 2 within-model) * 2 losses
DRAWS = 200000


def frozen_predictions(members, baseline, archived_report):
    """Use archived ensemble and per-seed June weights; never refit on stress."""
    if len(members) != 3 or len(archived_report['seeds']) != 3:
        raise ValueError('Exactly three ordered seeds required')
    y = baseline['dev_y']
    for member in members:
        for field in ('dev', 'dev_raw'):
            validate_probabilities(y, member[field])
    validate_probabilities(y, baseline['dev'])
    weight = archived_report['selection']['model_weight']
    seed_weights = [r['blend_selection']['model_weight'] for r in archived_report['seeds']]
    if not all(np.isfinite(w) and 0 <= w <= 1 for w in [weight, *seed_weights]):
        raise ValueError('Archived blend weights invalid')
    calibrated = np.mean([m['dev'] for m in members], axis=0)
    return {'primary': weight * calibrated + (1-weight) * baseline['dev'],
        'calibrated': calibrated, 'raw': np.mean([m['dev_raw'] for m in members], axis=0),
        'seed_primary': np.stack([w*m['dev'] + (1-w)*baseline['dev']
                                 for m, w in zip(members, seed_weights)])}


def bounded_comparison(y, candidate, control, games, mask, *, draws=DRAWS):
    mask = np.asarray(mask, dtype=bool)
    gate = group_reporting_status(games, mask)
    result = None
    if gate['status'] == 'reporting_eligible':
        delta = pitch_losses(np.asarray(y)[mask], candidate[mask]) - pitch_losses(np.asarray(y)[mask], control[mask])
        result = bootstrap_difference(delta, np.asarray(games)[mask], draws=draws,
                                      upper_alpha=.05/FAMILY_SIZE)
    return {'reporting': gate, 'paired': result,
            'passed': bool(result['simultaneous_upper'][0] <= .010 and
                           result['simultaneous_upper'][1] <= .002) if result else None}


def combined_status(records):
    values = [r['passed'] for r in records]
    if not values:
        raise ValueError('An empty robustness family cannot pass')
    return 'failed' if any(v is False for v in values) else 'unconfirmed' if any(v is None for v in values) else 'passed'
