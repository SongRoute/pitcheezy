"""Pure, generated-checkable join for a future paired conditional-FF evaluation.

No receipt, forecast ledger, model, profile, or label file is opened here. Callers
must separately prove the supplied forecast rows were frozen before target intake.
"""
from __future__ import annotations

from collections import Counter
import math
from numbers import Real

import numpy as np

from .fullseason import LEAVES
from .summary import paired_bootstrap, pitch_statistics


CLIP = 1e-12
BINS = 10
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261017
QUANTILES = (0.025, 0.975)
_KEY = ('game_pk', 'at_bat_number', 'pitch_number')
_SAME = ('decision_id', *_KEY, 'action', 'receipt_seq', 'query_sha256', 'context_sha256')


def _key(row):
    return tuple(row.get(name) for name in _KEY)


def _positive_int(value):
    return type(value) is int and value > 0


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _arm(row, arm, order):
    if not isinstance(arm, dict) or any(arm.get(name) != row[name] for name in _SAME):
        raise ValueError('arm decision/query/clock binding mismatch')
    if arm.get('leaf_order') != order:
        raise ValueError('arm leaf order mismatch')
    status, reason, q = arm.get('status'), arm.get('reason'), arm.get('probabilities')
    if status == 'abstain':
        if not isinstance(reason, str) or not reason or q is not None:
            raise ValueError('abstain needs reason and no probability')
        return None
    if status != 'scored' or reason is not None or not isinstance(q, list) or len(q) != 7:
        raise ValueError('invalid scored arm')
    if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or v < 0 or v > 1 for v in q):
        raise ValueError('invalid seven-vector value')
    if abs(math.fsum(q)-1) > 1e-10:
        raise ValueError('seven-vector incoherent')
    return np.asarray(q, dtype=np.float64)


def evaluate(attempts):
    """Return aggregate-only, fixed-FF denominators and paired game-cluster metrics."""
    if not isinstance(attempts, list):
        raise ValueError('attempts must be a list')
    order = list(LEAVES)
    seen_keys, seen_ids = set(), set()
    excluded, realized, arm_abstain = Counter(), Counter(), {'native': Counter(), 'step5': Counter()}
    scored_ff = Counter()
    native, step5, labels, games = [], [], [], []
    eligible_ff = 0
    for row in attempts:
        if not isinstance(row, dict) or any(not _positive_int(row.get(k)) for k in _KEY):
            raise ValueError('invalid target key')
        key = _key(row)
        if key in seen_keys:
            raise ValueError('duplicate physical target key')
        seen_keys.add(key)
        decision_id = row.get('decision_id')
        if not isinstance(decision_id, str) or not decision_id or decision_id in seen_ids:
            raise ValueError('missing or duplicate decision ID')
        seen_ids.add(decision_id)
        if (row.get('action') != 'FF' or not _positive_int(row.get('receipt_seq'))
                or not _digest(row.get('query_sha256')) or not _digest(row.get('context_sha256'))
                or row.get('leaf_order') != order):
            raise ValueError('invalid FF decision binding')
        q = {name: _arm(row, row.get(name), order) for name in ('native', 'step5')}
        for name in q:
            if q[name] is None:
                arm_abstain[name][row[name]['reason']] += 1
        target = row.get('target')
        if target is None:
            excluded['target_absent'] += 1
            continue
        if not isinstance(target, dict) or _key(target) != key or not _positive_int(target.get('first_receipt_seq')):
            raise ValueError('target identity or first receipt invalid')
        if row['receipt_seq'] >= target['first_receipt_seq']:
            raise ValueError('target not locally after forecast')
        pitch_type, leaf, reason = (target.get(k) for k in ('realized_pitch_type', 'leaf', 'leaf_reason'))
        if pitch_type is not None and (not isinstance(pitch_type, str) or not pitch_type):
            raise ValueError('realized type malformed')
        if leaf is not None and (leaf not in LEAVES or reason is not None):
            raise ValueError('label malformed')
        if leaf is None and (not isinstance(reason, str) or not reason):
            raise ValueError('unresolved label needs reason')
        if pitch_type in (None, 'UNKNOWN', 'UN') and leaf is not None:
            raise ValueError('unknown realized type has label')
        unknown_type = pitch_type in (None, 'UNKNOWN', 'UN')
        realized['UNKNOWN' if unknown_type else pitch_type] += 1
        if unknown_type:
            excluded['realized_type_unknown'] += 1
        elif pitch_type != 'FF':
            excluded['realized_non_ff'] += 1
        elif leaf is None:
            excluded['ff_label_unresolved'] += 1
        else:
            eligible_ff += 1
            for name in q:
                scored_ff[name] += int(q[name] is not None)
            if q['native'] is None:
                excluded['native_abstain'] += 1
            elif q['step5'] is None:
                excluded['step5_abstain'] += 1
            else:
                native.append(q['native']); step5.append(q['step5'])
                labels.append(LEAVES.index(leaf)); games.append(row['game_pk'])
    counts = {'attempts': len(attempts), 'observed_targets': sum(realized.values()),
              'realized_types': dict(sorted(realized.items())), 'excluded_in_order': dict(sorted(excluded.items())),
              'arm_abstention_reasons_all_attempts': {k: dict(sorted(v.items())) for k, v in arm_abstain.items()},
              'eligible_resolved_ff': eligible_ff,
              'arm_scored_resolved_ff': dict(scored_ff), 'paired_scored_ff': len(labels),
              'paired_games': len(set(games))}
    if not labels:
        return {'contract': 'CONDITIONAL_FF_GENERATED_EVALUATOR_ONLY', 'leaf_order': order,
                'counts': counts, 'nll': None, 'brier': None, 'paired_difference_native_minus_step5': None,
                'game_cluster_ci95': None, 'superiority_claim': False}
    y = np.asarray(labels, dtype=np.int64)
    a, b = (pitch_statistics(np.stack(v), y, nll_clip=CLIP, bins=BINS) for v in (native, step5))
    nll = {'native': float(a['nll'].mean()), 'step5': float(b['nll'].mean())}
    brier = {'native': float(a['brier'].mean()), 'step5': float(b['brier'].mean())}
    difference = {'nll': nll['native']-nll['step5'], 'brier': brier['native']-brier['step5']}
    ci = None
    if len(set(games)) >= 2:
        boot = paired_bootstrap(a, b, games, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
                                chunk=100, quantiles=QUANTILES)
        ci = {k: boot['ci'][k] for k in ('nll_mean', 'brier_mean')}
    return {'contract': 'CONDITIONAL_FF_GENERATED_EVALUATOR_ONLY', 'leaf_order': order,
            'counts': counts, 'nll': nll, 'brier': brier,
            'paired_difference_native_minus_step5': difference, 'game_cluster_ci95': ci,
            'superiority_claim': False}
