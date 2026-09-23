"""Pure accounting for frozen Observer pitch-type × zone choice diagnostics."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def action_accounting(actions, q_values, actual_type):
    """Rank every supported action; never impute an unsupported observed type.

    Ties use the adapter's original action order. Values are defensive-team PA
    win probabilities, so the gap is reported in percentage points.
    """
    q = np.asarray(q_values, dtype=float)
    if q.ndim != 1 or len(q) != len(actions) or not np.isfinite(q).all():
        raise ValueError('One finite Q value is required per supported action')
    pairs = [(a.pitch_type, a.zone_id) for a in actions]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Duplicate supported action')
    if not pairs:
        return {'top_action': None, 'top_type': None, 'actual_type_supported': False,
                'actual_best_rank': None, 'actual_best_zone': None,
                'actual_best_value': None, 'best_value': None, 'gap_pp': None,
                'supported_actions': 0, 'supported_types': 0}
    order = sorted(range(len(q)), key=lambda i: (-q[i], i))
    actual = [i for i in order if actions[i].pitch_type == actual_type]
    best_actual = actual[0] if actual else None
    return {'top_action': {'pitch_type': pairs[order[0]][0], 'zone_id': pairs[order[0]][1]},
            'top_type': pairs[order[0]][0], 'actual_type_supported': bool(actual),
            'actual_best_rank': order.index(best_actual) + 1 if actual else None,
            'actual_best_zone': pairs[best_actual][1] if actual else None,
            'actual_best_value': float(q[best_actual]) if actual else None,
            'best_value': float(q[order[0]]),
            'gap_pp': float(100 * (q[order[0]] - q[best_actual])) if actual else None,
            'supported_actions': len(pairs),
            'supported_types': len({item[0] for item in pairs})}


def concentration(rows, group_fields=('pitcher', 'balls', 'strikes')):
    """Observed and recommended type mixtures on the same evaluable pitches."""
    groups = defaultdict(list)
    for row in rows:
        if row['top_type'] is not None:
            groups[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for key, part in sorted(groups.items()):
        n = len(part)
        observed = Counter(r['actual_type'] for r in part)
        advised = Counter(r['top_type'] for r in part)
        types = sorted(observed.keys() | advised.keys())
        actual_share = {t: observed[t] / n for t in types}
        top_share = {t: advised[t] / n for t in types}
        result.append({'group': dict(zip(group_fields, key)), 'pitches': n,
                       'games': len({r['game_pk'] for r in part}),
                       'pas': len({(r['game_pk'], r['at_bat_number']) for r in part}),
                       'actual_counts': dict(sorted(observed.items())),
                       'top_type_counts': dict(sorted(advised.items())),
                       'actual_share': actual_share, 'top_type_share': top_share,
                       'actual_hhi': sum(v*v for v in actual_share.values()),
                       'top_type_hhi': sum(v*v for v in top_share.values()),
                       'total_variation': .5 * sum(abs(actual_share[t] - top_share[t]) for t in types),
                       'top_type_match_count': sum(r['actual_type'] == r['top_type'] for r in part)})
    return result


def stability(rows_by_seed):
    """Describe rank changes across independent draw resamples for aligned keys."""
    if len(rows_by_seed) < 2:
        raise ValueError('At least two independent sampling seeds required')
    seeds = sorted(rows_by_seed)
    keys = set(rows_by_seed[seeds[0]])
    if any(set(rows_by_seed[s]) != keys for s in seeds):
        raise ValueError('Sampling seeds have different pitch keys')
    rows = []
    for key in sorted(keys):
        values = [rows_by_seed[s][key] for s in seeds]
        ranks = [v['actual_best_rank'] for v in values]
        tops = [v['top_action'] for v in values]
        top_types = [v['top_type'] for v in values]
        rows.append({'pitch_key': list(key), 'top_action_agreement': len({str(v) for v in tops}) == 1,
                     'top_type_agreement': len(set(top_types)) == 1,
                     'actual_type_supported_all': all(v['actual_type_supported'] for v in values),
                     'actual_rank_min': min(ranks) if all(r is not None for r in ranks) else None,
                     'actual_rank_max': max(ranks) if all(r is not None for r in ranks) else None})
    return {'seeds': seeds, 'pitches': len(rows),
            'top_action_agreement_rate': sum(r['top_action_agreement'] for r in rows)/len(rows) if rows else None,
            'top_type_agreement_rate': sum(r['top_type_agreement'] for r in rows)/len(rows) if rows else None,
            'per_pitch': rows}
