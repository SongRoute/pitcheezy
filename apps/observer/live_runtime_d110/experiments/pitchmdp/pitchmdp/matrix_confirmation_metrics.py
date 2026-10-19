"""Fixed five-seed Cpanel confirmation family; exploratory L6 stays separate."""
from __future__ import annotations
import numpy as np
from .matrix_metrics import pitch_losses, paired_game_comparison, holm_adjust, prediction_decision
from .matrix_group_metrics import guardrails, volume_interaction
from .matrix_panel import group_reporting_status

SCORING = {'primary_slots': ['candidate1_overall', 'candidate1_low', 'candidate2_overall', 'candidate2_low'],
    'holm_alpha': .05, 'seeds': [0, 1, 2, 3, 4], 'bootstrap_draws': 10000,
    'bootstrap_seed': 20260924, 'minimum_games': 30, 'minimum_pitches': 500,
    'nll_improvement': .003, 'brier_guardrail': .001, 'required_negative_seeds': 4,
    'R_family_size': 48, 'R_draws': 100000, 'L6': 'fixed 1/3/5 seeds; descriptive only'}


def finish_decisions(comparisons):
    if len(comparisons) > 2:
        raise ValueError('At most two registered confirmation contrasts')
    p_values = []
    for row in comparisons:
        p_values.extend([row['paired']['nll']['p_less'] if row['paired'] else None,
                         row['low_paired']['nll']['p_less'] if row['low_paired'] else None])
    p_values += [None] * (4 - len(p_values))
    adjusted = holm_adjust(p_values)
    for i, row in enumerate(comparisons):
        row['N'] = prediction_decision(row['paired'], adjusted[2*i], row['seed_deltas'], stage='final') if row['paired'] else {
            'status': 'unmeasured', 'reason': 'Overall 30-game/500-pitch reporting gate not met'}
        if not row['low_paired'] or not row['paired']:
            row['G'] = {'status': 'unmeasured', 'reason': 'Low or overall reporting gate not met'}
        else:
            low = prediction_decision(row['low_paired'], adjusted[2*i+1], row['low_seed_deltas'], stage='final')
            overall = row['paired']['nll']['ci95'][1] <= .001 and row['paired']['brier']['ci95'][1] <= .001
            row['G'] = {'low_group': low, 'whole_population_noninferior': overall,
                'status': 'group_improvement' if low['status'] == 'predictive_improvement' and overall else 'inconclusive'}
    return adjusted


def compare_family(labels, predictions, games, metadata, pairs):
    if len(pairs) > 2 or len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError('Unique registered confirmation pairs required')
    y, games = np.asarray(labels), np.asarray(games)
    low = metadata.train_volume.eq('low').to_numpy(bool)
    low_gate = group_reporting_status(games, low)
    whole_gate = group_reporting_status(games, np.ones(len(y), dtype=bool))
    comparisons = []
    for candidate, control in pairs:
        a, b = predictions[candidate], predictions[control]
        if a['seed_primary'].shape != (5, len(y), 10) or b['seed_primary'].shape != (5, len(y), 10):
            raise ValueError('Complete ordered five-seed prediction family required')
        whole = paired_game_comparison(y, a['primary'], b['primary'], games) if whole_gate['status'] == 'reporting_eligible' else None
        low_pair = paired_game_comparison(y[low], a['primary'][low], b['primary'][low], games[low]) if low_gate['status'] == 'reporting_eligible' else None
        losses = [(pitch_losses(y, x) - pitch_losses(y, z))[:, 0]
                  for x, z in zip(a['seed_primary'], b['seed_primary'])]
        comparisons.append({'candidate': candidate, 'control': control, 'paired': whole,
            'overall_reporting': whole_gate, 'low_reporting': low_gate, 'low_paired': low_pair,
            'seed_deltas': [float(d.mean()) for d in losses],
            'low_seed_deltas': [float(d[low].mean()) for d in losses] if low_pair else None,
            'robustness': guardrails(y, a['primary'], b['primary'], games, metadata, candidate_family_size=2, draws=100000),
            'volume_interaction': volume_interaction(y, a['primary'], b['primary'], games, metadata)})
    adjusted = finish_decisions(comparisons)
    return comparisons, {'settings': SCORING, 'unadjusted': [
        value for row in comparisons for value in (row['paired']['nll']['p_less'] if row['paired'] else None,
                                                   row['low_paired']['nll']['p_less'] if row['low_paired'] else None)] +
        [None]*(4-2*len(comparisons)), 'adjusted': adjusted, 'unused_slots_retained': True,
        'independent_confirmation': False}
