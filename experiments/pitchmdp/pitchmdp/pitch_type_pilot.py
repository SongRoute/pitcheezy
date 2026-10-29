"""One preregistered retrospective pitch-type comparison; no timing claims."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .pitch_type_evaluation import probability_scores, _paired_interval

KEY = ['game_pk', 'at_bat_number', 'pitch_number']
FEATURES = ['pitcher', 'balls', 'strikes', 'batter_side', 'pitcher_hand', 'bases', 'outs', 'score_diff']
TIERS = [FEATURES[:1], FEATURES[:3], FEATURES[:5], FEATURES]
TIER_NAMES = ['pitcher', 'count', 'matchup', 'situation']
MODELS = ['pitcher_reference', 'context_control', 'tree_candidate']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_config(c):
    if c['id'] != 'EVAL-PITCH-TYPE-004' or c['mode'] != 'HISTORICAL_DEVELOPMENT_REPLAY':
        raise ValueError('unregistered mode')
    if c['features'] != FEATURES or c['alpha'] != 4 or c['calibration'] != 'identity':
        raise ValueError('unregistered inputs/smoothing')
    if c['labels'] != ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT']:
        raise ValueError('unregistered label order')
    if c['tree'] != dict(criterion='log_loss', max_depth=8, min_samples_leaf=100, random_state=0):
        raise ValueError('unregistered tree')
    if set(c['label_map'].values()) != set(c['labels']):
        raise ValueError('incomplete label mapping')
    spans = [tuple(pd.Timestamp(v) for v in c['roles'][role]) for role in ('FIT', 'SELECT', 'CAL', 'EVAL')]
    if any(lo >= hi for lo, hi in spans) or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('role overlap or nonchronological dates')


def prepare(frame, c):
    """Decode allowlist only; SELECT/CAL and outside rows are never modeled/scored."""
    validate_config(c)
    if set(frame.columns) != set(c['columns']):
        raise ValueError('column allowlist mismatch')
    if len(frame) > c['budget']['max_source_rows']:
        raise ValueError('source row budget exceeded')
    frame = frame.copy()
    dates = pd.to_datetime(frame.game_date, errors='raise')
    if dates.isna().any() or dates.dt.tz is not None or (dates != dates.dt.normalize()).any():
        raise ValueError('game_date must be naive nonmissing calendar midnight')
    # Physical identity errors abort instead of silently changing the denominator.
    for col in KEY:
        if frame[col].isna().any() or not np.isfinite(frame[col]).all() or (frame[col] < 1).any() or (frame[col] % 1 != 0).any():
            raise ValueError('invalid physical identity')
    if frame.duplicated(KEY).any():
        raise ValueError('duplicate physical identity')
    if pd.DataFrame({'game': frame.game_pk, 'date': dates}).groupby('game').date.nunique().max() > 1:
        raise ValueError('same game has conflicting dates/roles')
    frame['role'] = 'UNUSED'
    for role in ('FIT', 'EVAL'):
        lo, hi = c['roles'][role]
        frame.loc[(dates >= lo) & (dates < hi), 'role'] = role
    selected = frame.loc[frame.role != 'UNUSED'].sort_values(KEY).copy()
    if len(selected[selected.role == 'FIT']) > c['budget']['max_fit_rows'] or len(selected[selected.role == 'EVAL']) > c['budget']['max_eval_rows']:
        raise ValueError('role row budget exceeded; no truncation')
    ledger, rosters, populations = [], {}, {}
    for role in ('FIT', 'EVAL'):
        part = selected.loc[selected.role == role].copy()
        mapped = part.pitch_type.map(c['label_map'])
        reasons = pd.Series('', index=part.index, dtype=object)
        reasons.loc[part.game_type != 'R'] = 'non_regular_game'
        reasons.loc[(reasons == '') & mapped.isna()] = 'unregistered_or_missing_pitch_type'
        label_eligible = reasons == ''
        ctx = pd.DataFrame(index=part.index)
        ctx['pitcher'] = part.pitcher
        ctx['balls'], ctx['strikes'] = part.balls, part.strikes
        ctx['batter_side'], ctx['pitcher_hand'] = part.stand, part.p_throws
        ctx['bases'] = sum(part[col].notna().astype(int) * (1 << j) for j, col in enumerate(('on_1b', 'on_2b', 'on_3b')))
        ctx['outs'], ctx['score_diff'] = part.outs_when_up, part.fld_score - part.bat_score
        valid = pd.Series(True, index=part.index)
        for col, lo, hi in (('pitcher', 1, 2**31-1), ('balls', 0, 3), ('strikes', 0, 2), ('outs', 0, 2), ('score_diff', -100, 100)):
            v = ctx[col]
            valid &= v.notna() & np.isfinite(v) & (v % 1 == 0) & v.between(lo, hi)
        for col in ('bat_score', 'fld_score'):
            v = part[col]
            valid &= v.notna() & np.isfinite(v) & (v % 1 == 0) & v.between(0, 100)
        for col in ('on_1b', 'on_2b', 'on_3b'):
            v = part[col]
            valid &= v.isna() | (np.isfinite(v) & (v % 1 == 0) & (v > 0))
        valid &= ctx.batter_side.isin(['L', 'R']) & ctx.pitcher_hand.isin(['L', 'R'])
        reasons.loc[label_eligible & ~valid] = 'invalid_pre_pitch_input'
        for idx in part.index[reasons != '']:
            ledger.append({'role': role, 'key': [int(part.at[idx, k]) for k in KEY],
                           'reason': reasons.at[idx], 'raw_pitch_type': None if pd.isna(part.at[idx, 'pitch_type']) else str(part.at[idx, 'pitch_type'])})
        good = label_eligible & valid
        roster = part.loc[good, KEY].astype('int64').copy()
        for col in FEATURES:
            roster[col] = ctx.loc[good, col]
        for col in set(FEATURES) - {'batter_side', 'pitcher_hand'}:
            roster[col] = roster[col].astype('int64')
        roster['label'] = mapped.loc[good].map({v: i for i, v in enumerate(c['labels'])}).astype('int64')
        rosters[role] = roster.reset_index(drop=True)
        populations[role] = {'source_period_rows': len(part), 'label_eligible': int(label_eligible.sum()),
            'valid_input': int(good.sum()), 'input_error': int((label_eligible & ~valid).sum()),
            'exclusions': dict(sorted(Counter(reasons[reasons != '']).items()))}
    if any(not len(rosters[role]) for role in rosters):
        raise ValueError('empty usable role')
    return rosters['FIT'], rosters['EVAL'], populations, ledger


def predict(fit, query, c):
    """Only FIT labels enter priors, encoder or the single tree fit."""
    k, alpha = len(c['labels']), c['alpha']
    y = fit.label.to_numpy(dtype=int)
    prior = (np.bincount(y, minlength=k) + alpha / k) / (len(y) + alpha)
    tables = []
    for fields in TIERS:
        table = {}
        for values, label in zip(fit[fields].itertuples(index=False, name=None), y):
            table.setdefault(values, np.zeros(k, dtype=int))[label] += 1
        tables.append(table)
    reference, control, supports = [], [], []
    for values in query[FEATURES].itertuples(index=False, name=None):
        row = dict(zip(FEATURES, values)); p = prior.copy(); support = []; level = 'global'
        for j, (fields, table) in enumerate(zip(TIERS, tables)):
            counts = table.get(tuple(row[f] for f in fields), np.zeros(k, dtype=int))
            n = int(counts.sum()); support.append(n)
            p = (counts + alpha * p) / (n + alpha)
            if n: level = TIER_NAMES[j]
            if j == 0: reference.append(p.copy())
        control.append(p); supports.append(dict(zip(TIER_NAMES, support), effective_level=level))
    encoder = ColumnTransformer([
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32), ['pitcher', 'batter_side', 'pitcher_hand']),
        ('numeric', 'passthrough', ['balls', 'strikes', 'bases', 'outs', 'score_diff'])], sparse_threshold=1.0)
    xfit = encoder.fit_transform(fit[FEATURES])
    xquery = encoder.transform(query[FEATURES])
    tree = DecisionTreeClassifier(**c['tree']).fit(xfit, y)
    leaves = tree.apply(xfit)
    leaf_counts = {int(leaf): np.bincount(y[leaves == leaf], minlength=k) for leaf in np.unique(leaves)}
    candidate = np.array([(leaf_counts[int(leaf)] + alpha * prior) / (leaf_counts[int(leaf)].sum() + alpha)
                          for leaf in tree.apply(xquery)])
    return dict(zip(MODELS, (np.asarray(reference), np.asarray(control), candidate))), supports, {
        'node_count': tree.tree_.node_count, 'depth': tree.get_depth(), 'leaves': tree.get_n_leaves(),
        'fit_class_counts': np.bincount(y, minlength=k).tolist(), 'global_prior': prior.tolist()}


def evaluate_frame(frame, c):
    fit, query, populations, ledger = prepare(frame, c)
    probabilities, supports, tree_info = predict(fit, query, c)
    metrics, losses = {}, {}
    for model, p in probabilities.items():
        metrics[model], losses[model] = probability_scores(query.label.to_numpy(dtype=int), p, len(c['labels']))
    comparison = _paired_interval(losses['tree_candidate'] - losses['context_control'], query.game_pk.to_numpy(), c['bootstrap'])
    denominator = populations['EVAL']['label_eligible']
    result = {'id': c['id'], 'mode': c['mode'], 'scientific_winner': None, 'availability': c['availability'],
        'exposure': c['exposure'], 'labels': c['labels'], 'populations': populations,
        'coverage': {m: {'eligible': denominator, 'ok': len(query), 'error': denominator-len(query),
            'missing': 0, 'abstain': 0, 'late': None, 'fraction': len(query)/denominator} for m in MODELS},
        'metrics_common_valid_inputs': metrics, 'primary_pair': c['primary_pair'], 'comparison': comparison,
        'tree': tree_info, 'fallback_counts': dict(sorted(Counter(s['effective_level'] for s in supports).items())),
        'unseen_pitcher_queries': sum(s['pitcher'] == 0 for s in supports),
        'nll_floor': 1e-12, 'timeliness': 'unmeasured_retrospective', 'cuda_used': False,
        'fit_games': int(fit.game_pk.nunique()), 'eval_games': int(query.game_pk.nunique())}
    audit = query[KEY + FEATURES + ['label']].copy()
    for model, p in probabilities.items():
        for i, label in enumerate(c['labels']): audit[f'{model}_{label}'] = p[:, i]
    for tier in TIER_NAMES: audit[f'support_{tier}'] = [s[tier] for s in supports]
    return result, ledger, audit


def load_registered(path, c):
    path = Path(path)
    if path.name != c['source']['file'] or path.stat().st_size != c['source']['bytes'] or digest(path) != c['source']['sha256']:
        raise ValueError('source identity mismatch; stop without decode')
    frame = pd.read_parquet(path, columns=c['columns'], use_threads=False)
    if len(frame) != c['source']['rows']: raise ValueError('source row count mismatch')
    return frame
