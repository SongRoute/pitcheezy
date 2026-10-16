"""Pure fixed-six corrected-retrospective quality, context, and score adapter.

No source, model, profile or label file is opened here. The caller must verify
collector bytes and model/profile bindings under a separate actual grant.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import math
from numbers import Real

import numpy as np
import pandas as pd

from experiments.ws_fullseason.data import closure, key_digest
from .fullseason import LEAVES
from .fullseason_stream import CONTEXT, KEY, StreamState
from .ledger import sha
from .summary import pitch_statistics


SIX = {849812: '2026-10-12', 849820: '2026-10-12', 849813: '2026-10-13',
       849811: '2026-10-14', 849810: '2026-10-15', 849818: '2026-10-15'}
FIT_COUNTS = (258626, 132039, 2144, 78550, 124789, 51974, 72383)
FIT_TOTAL = 720505
STOP_SHARE = {'excluded_duplicate_key': .001, 'quarantined_multi_date_game': .01,
              'excluded_invalid_state': .001, 'excluded_missing_handedness': .001,
              'excluded_unmapped_label': .01, 'excluded_bip_missing_bb_type': .001,
              'excluded_nonbip_with_bb_type': .001, 'excluded_bip_unknown_bb_type': .001}


def _int(value, low=1):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) \
        and low <= value < 2**53 and value == math.floor(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _key(row):
    vals = [row[k] for k in KEY]
    return tuple(int(v) for v in vals) if all(_int(v) for v in vals) else None


def _query(row):
    # Exact pre-pitch schema. Current realization/physics/labels never pass here.
    q = {}
    for name in CONTEXT:
        value = row[name]
        if name == 'game_date':
            q[name] = pd.Timestamp(value).strftime('%Y-%m-%d')
        elif name in ('on_1b', 'on_2b', 'on_3b'):
            q[name] = None if pd.isna(value) else int(value)
        elif name in (*KEY, 'pitcher', 'batter', 'balls', 'strikes', 'outs_when_up',
                      'inning', 'home_score', 'away_score'):
            q[name] = int(value)
        else:
            q[name] = value
    return q


def _policy(M):
    return {'row_policy': {'game_type': 'L', 'start': '2026-10-12', 'end': '2026-10-15',
            'segregation_game_pks': [], 'types': list(M.fsf.TYPES),
            'bb_types': list(M.safe.BB_TYPES), 'stop_share': STOP_SHARE}}


def prepare(raw, source_sha256, M=None):
    """Project exact six L games and build retrospective, never live, snapshots.

    ``source_sha256`` must be the caller-verified collector projection SHA.
    Cases hold labels separately from immutable model snapshots.
    """
    M = M or closure()
    if not _sha(source_sha256) or not isinstance(raw, pd.DataFrame) or list(raw.columns) != M.cp.PROJECT:
        raise ValueError('verified source digest and exact 29-column projection required')
    if len(raw) > 12_000 or raw.empty:
        raise ValueError('fixed-six source row bound')
    source = raw.reset_index(drop=True)
    game = source.game_pk
    if game.isna().any() or not all(_int(v) and int(v) in SIX for v in game):
        raise ValueError('unattributed or nonselected game row')
    days = pd.to_datetime(source.game_date, errors='coerce')
    if (days.isna().any() or any(d.tz is not None or d != d.normalize()
                                or str(d.date()) != SIX[int(g)] for g, d in zip(game, days))
            or not source.game_type.eq('L').all()):
        raise ValueError('original game date/type mismatch')
    physical = [_key(row) for _, row in source.iterrows()]
    by_game_keys = defaultdict(set); unkeyed = Counter(); frequencies = Counter(k for k in physical if k)
    duplicate = {k for k, n in frequencies.items() if n > 1}
    if duplicate:
        raise ValueError('duplicate physical key; no copy may enter retrospective context')
    for (_, row), g, key in zip(source.iterrows(), game, physical):
        if key is None or not _int(row.pitcher):
            unkeyed[int(g)] += 1
        else:
            by_game_keys[int(g)].add(key)
    accepted, quality = M.cp.project(source, _policy(M), M.safe.outcome_labels, M.safe.head_labels)
    labels = M.safe.core_fate(accepted)
    if not np.isin(labels, M.safe.CORE).all():
        raise ValueError('accepted non-core label')
    accepted_rows = {_key(row): row for _, row in accepted.iterrows()}
    if None in accepted_rows or len(accepted_rows) != len(accepted):
        raise ValueError('accepted physical key collision')
    normalizer = sha({'source_sha256': source_sha256, 'adapter': 'fixed-six-retrospective-v1'})
    cases = []
    census = {'source_rows': len(source), 'accepted_quality_rows': len(accepted),
              'quality': quality, 'duplicate_physical_keys': len(duplicate),
              'unkeyed_rows_by_game': {str(g): unkeyed[g] for g in SIX},
              'games': {str(g): {'official_date': day, 'source_rows': int((game == g).sum()),
                                 'accepted_quality_rows': int((accepted.game_pk == g).sum()),
                                 'query_ready': 0, 'query_abstain': 0}
                        for g, day in SIX.items()},
              'source_sha256': source_sha256,
              'role': '2026 EXPOSED DEVELOPMENT corrected retrospective; arrival UNKNOWN'}
    for g, day in SIX.items():
        keys = sorted(by_game_keys[g]); valid = set(keys)
        by_pa = defaultdict(set)
        for _, ab, pitch in keys:
            by_pa[ab].add(pitch)
        gap_pas = {ab for ab, pitches in by_pa.items() if pitches != set(range(1, max(pitches)+1))}
        state = StreamState(':memory:', game_pk=g, game_date=day,
                            normalizer_sha256=normalizer, evidence_kind='retrospective')
        try:
            # Logical ordinals are internal SQL ordering only, never captured arrival.
            for ordinal, key in enumerate(keys, 1):
                row = accepted_rows.get(key) if key not in duplicate else None
                # The training vocabulary contains UNKNOWN/UN as sentinels;
                # they are not known realized actions for a native context token.
                if row is not None and row.pitch_type in ('UNKNOWN', 'UN'):
                    row = None
                token = ({k: int(row[k]) if k in ('pitcher', 'batter', 'outcome10', 'balls', 'strikes')
                          else row[k] for k in ('pitcher', 'batter', 'pitch_type', 'outcome10', 'balls', 'strikes')}
                         if row is not None else None)
                state.ingest(f'retrospective:{g}:{key[1]}:{key[2]}', key, 1,
                             {'status': 'accepted' if token else 'unknown',
                              'reason': 'projected_quality' if token else 'source_quality_or_duplicate_unknown',
                              'accepted': token}, received_ns=ordinal)
            for i, row in accepted.loc[accepted.game_pk == g].iterrows():
                key = _key(row)
                if key in duplicate:
                    cases.append({'key': key, 'status': 'abstain', 'reason': 'duplicate_physical_key', 'label': None,
                                  'snapshot': None})
                    continue
                if row.pitch_type in ('UNKNOWN', 'UN'):
                    cases.append({'key': key, 'status': 'abstain', 'reason': 'observed_type_unknown', 'label': None,
                                  'snapshot': None})
                    continue
                previous = (g, key[1], key[2]-1)
                has_prefix = (not unkeyed[g] and not any(ab < key[1] for ab in gap_pas)
                              and all((g, key[1], n) in valid for n in range(1, key[2])))
                # A missing/unknown immediate predecessor is never defaulted to no previous pitch.
                if key[2] > 1 and (previous not in valid or previous not in accepted_rows or previous in duplicate):
                    has_prefix = False
                q = _query(row)
                snapshot = state.decide(f'retrospective:{g}:{key[1]}:{key[2]}', q,
                                        pitch_type_basis='observed_actual',
                                        coverage={'complete': bool(has_prefix), 'before_key': list(key),
                                                  'source_sha256': source_sha256},
                                        received_ns=len(keys)+1+len(cases))
                cases.append({'key': key, 'status': snapshot['status'], 'reason': snapshot['reason'],
                              'label': int(labels[i][-1]), 'snapshot': snapshot})
                census['games'][str(g)]['query_ready' if snapshot['status'] == 'ready' else 'query_abstain'] += 1
        finally:
            state.close()
    # Explicit denominators include duplicate/unknown-action cases and missing games.
    for case in cases:
        if case['snapshot'] is None:
            census['games'][str(case['key'][0])]['query_abstain'] += 1
    census['query_ready'] = sum(v['query_ready'] for v in census['games'].values())
    census['query_abstain'] = sum(v['query_abstain'] for v in census['games'].values())
    if census['query_ready'] + census['query_abstain'] != len(accepted):
        raise ValueError('query fate census mismatch')
    census['accepted_keys_sha256'] = key_digest(accepted[list(KEY)].to_numpy(dtype=np.int64))
    return {'census': census, 'cases': cases}


def evaluate(prepared, predictor):
    """Aggregate-only observed-type metrics; predictor loads no assets in this module."""
    if not isinstance(prepared, dict) or set(prepared) != {'census', 'cases'}:
        raise ValueError('prepared population required')
    if (not isinstance(prepared['cases'], list)
            or len(prepared['cases']) != prepared['census']['accepted_quality_rows']):
        raise ValueError('prepared case census mismatch')
    baseline = np.asarray(FIT_COUNTS, dtype=np.float64) / FIT_TOTAL
    if sum(FIT_COUNTS) != FIT_TOTAL or len(baseline) != len(LEAVES):
        raise ValueError('fixed FIT marginal invalid')
    rows, reasons = defaultdict(list), Counter()
    for case in prepared['cases']:
        if (case['status'] not in ('ready', 'abstain') or not isinstance(case['reason'], (str, type(None)))
                or (case['status'] == 'ready' and
                    (case['reason'] is not None or not isinstance(case['snapshot'], dict)
                     or case['snapshot'].get('status') != 'ready'
                     or type(case['label']) is not int or not 0 <= case['label'] < 7))
                or (case['status'] == 'abstain' and not case['reason'])):
            raise ValueError('prepared case status/label mismatch')
        if case['status'] != 'ready':
            reasons[case['reason']] += 1
            continue
        result = predictor(deepcopy(case['snapshot']))
        if not isinstance(result, dict):
            raise ValueError('predictor result mapping required')
        q = result.get('probabilities')
        if result.get('status') != 'scored' or result.get('leaf_order') != list(LEAVES) or not isinstance(q, list) or len(q) != 7:
            raise ValueError('scored fixed seven-leaf predictor required')
        if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or v < 0 or v > 1 for v in q) \
                or abs(math.fsum(q)-1) > 1e-10:
            raise ValueError('invalid predicted seven-vector')
        rows[case['key'][0]].append((q, case['label']))
    metrics = {}
    for game in SIX:
        pair = rows[game]
        if pair:
            y = np.asarray([v[1] for v in pair], np.int64)
            model = pitch_statistics(np.asarray([v[0] for v in pair]), y, nll_clip=1e-12, bins=10)
            fixed = pitch_statistics(np.tile(baseline, (len(pair), 1)), y, nll_clip=1e-12, bins=10)
            metrics[str(game)] = {'n': len(pair), 'model_nll': float(model['nll'].mean()),
                'model_brier': float(model['brier'].mean()), 'fixed_fit_marginal_nll': float(fixed['nll'].mean()),
                'fixed_fit_marginal_brier': float(fixed['brier'].mean())}
        else:
            metrics[str(game)] = {'n': 0, 'model_nll': None, 'model_brier': None,
                'fixed_fit_marginal_nll': None, 'fixed_fit_marginal_brier': None}
    scored = sum(v['n'] for v in metrics.values())
    pooled = [item for game in SIX for item in rows[game]]
    if pooled:
        y = np.asarray([v[1] for v in pooled], np.int64)
        model = pitch_statistics(np.asarray([v[0] for v in pooled]), y, nll_clip=1e-12, bins=10)
        fixed = pitch_statistics(np.tile(baseline, (len(pooled), 1)), y, nll_clip=1e-12, bins=10)
        overall = {'n': len(pooled), 'model_nll': float(model['nll'].mean()),
                   'model_brier': float(model['brier'].mean()),
                   'fixed_fit_marginal_nll': float(fixed['nll'].mean()),
                   'fixed_fit_marginal_brier': float(fixed['brier'].mean())}
    else:
        overall = {'n': 0, 'model_nll': None, 'model_brier': None,
                   'fixed_fit_marginal_nll': None, 'fixed_fit_marginal_brier': None}
    return {'role': prepared['census']['role'], 'leaf_order': list(LEAVES),
            'source_rows': prepared['census']['source_rows'],
            'accepted_quality_rows': prepared['census']['accepted_quality_rows'],
            'query_ready': prepared['census']['query_ready'], 'scored': scored,
            'query_abstain_reasons': dict(sorted(reasons.items())), 'per_game': metrics,
            'overall': overall,
            'baseline': {'kind': 'fixed FIT seven-leaf marginal', 'counts': list(FIT_COUNTS), 'total': FIT_TOTAL},
            'prospective_claim': False, 'superiority_claim': False}
