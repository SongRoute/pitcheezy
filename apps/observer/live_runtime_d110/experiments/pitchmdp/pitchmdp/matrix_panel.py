"""TRAIN-only development panel selection and evaluation grouping metadata.

No candidate predictions or DEV outcomes are consumed by panel selection. This
module is additive: it never changes the frozen C6/ML1/ML2 input contracts.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from .data import KEY
from .matrix_data import ordered_key_hash

ROLES = ('starter', 'relief')
HANDS = ('L', 'R')
VOLUMES = ('low', 'middle', 'high')
SELECTOR_VERSION = 'cpanel_train_stratified_v1'


def _rank(pitcher, seed):
    return hashlib.sha256(f'{seed}|{int(pitcher)}'.encode('ascii')).hexdigest()


def verify_starters(train, starter_source):
    """Reconstruct game/team starters from the unfiltered approved TRAIN log."""
    needed = {*KEY, 'game_date', 'game_type', 'split', 'inning', 'inning_topbot', 'outs_when_up', 'pitcher'}
    if starter_source is None or not needed <= set(starter_source.columns):
        raise ValueError('Unfiltered TRAIN source metadata is required to verify starters')
    dates = pd.to_datetime(starter_source.game_date)
    if (not starter_source.split.eq('train').all() or not starter_source.game_type.eq('R').all() or
            dates.isna().any() or not dates.between('2023-05-15', '2025-04-30').all()):
        raise ValueError('Starter source must be the same regular-season TRAIN window')
    source_hash = ordered_key_hash(starter_source.sort_values(KEY))
    ordered = starter_source.sort_values(['game_date', *KEY])
    first = ordered.groupby(['game_pk', 'inning_topbot'], sort=False).head(1)
    if (not first.inning.eq(1).all() or not first.outs_when_up.eq(0).all() or
            first.pitcher.isna().any() or not first.inning_topbot.isin(['Top', 'Bot']).all()):
        raise ValueError('Incomplete/ambiguous first defensive pitches in starter source')
    lookup = first.set_index(['game_pk', 'inning_topbot']).pitcher
    expected = lookup.reindex(pd.MultiIndex.from_frame(train[['game_pk', 'inning_topbot']]))
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), train.starter_pitcher.to_numpy()):
        raise ValueError('Recorded TRAIN starters differ from first defensive pitcher')
    if not pd.MultiIndex.from_frame(train[KEY]).isin(pd.MultiIndex.from_frame(starter_source[KEY])).all():
        raise ValueError('Eligible TRAIN rows absent from starter source')
    return {'method': 'first chronological pitch for each game/defensive half; inning 1 and zero outs required',
            'source_rows': len(starter_source), 'canonical_source_keys_sha256': source_hash,
            'recorded_starter_column_verified': True}


def select_panel(train, *, starter_source=None, per_cell=4, seed=20260924, c6_ids=()):
    """Select at most 48 players from an already frozen eligible D100 frame.

    The caller supplies exactly the eligible TRAIN pool, including both roles.
    Role is the majority of those pitches thrown as the recorded game starter.
    Quantile cutoffs are global over positive TRAIN pitcher counts, not fitted
    separately in each hand/role group. Boundary ties stay in one volume group.
    """
    required = {*KEY, 'game_date', 'game_type', 'split', 'pitcher', 'starter_pitcher', 'p_throws', 'batter', 'inning_topbot'}
    if not required <= set(train.columns) or train.empty:
        raise ValueError('Nonempty eligible D100 TRAIN metadata is required')
    if per_cell != 4 or not isinstance(seed, int):
        raise ValueError('Version 1 fixes four per cell and an integer sampling seed')
    dates = pd.to_datetime(train.game_date)
    if not train.split.eq('train').all() or not train.game_type.eq('R').all():
        raise ValueError('Panel selector requires only regular-season TRAIN rows')
    if dates.isna().any() or not dates.between('2023-05-15', '2025-04-30').all():
        raise ValueError('Panel TRAIN dates fall outside the frozen population')
    if train.pitcher.isna().any() or train.batter.isna().any() or train.starter_pitcher.isna().any():
        raise ValueError('TRAIN player and starter identities must be available')
    # Canonical key hashing also rejects duplicate/missing pitch identities.
    starter_validation = verify_starters(train, starter_source)
    ordered_hash = ordered_key_hash(train)
    canonical_hash = ordered_key_hash(train.sort_values(KEY))
    work = train[[*KEY, 'pitcher', 'starter_pitcher', 'p_throws']].copy()
    work['starter'] = work.pitcher.eq(work.starter_pitcher)
    rows = []
    for pitcher, group in work.groupby('pitcher', sort=True):
        hands = sorted(set(group.p_throws.dropna().astype(str)))
        hand = hands[0] if len(hands) == 1 and hands[0] in HANDS and group.p_throws.notna().all() else 'unknown'
        starter_pitches = int(group.starter.sum())
        rows.append({'pitcher': int(pitcher), 'train_pitches': len(group),
                     'train_games': int(group.game_pk.nunique()), 'starter_pitches': starter_pitches,
                     'starter_share': starter_pitches / len(group),
                     'train_role': 'starter' if starter_pitches * 2 >= len(group) else 'relief',
                     'train_hand': hand, 'observed_hands': hands,
                     'selection_hash': _rank(pitcher, seed)})
    counts = np.array([row['train_pitches'] for row in rows])
    q25, q75 = np.quantile(counts, [.25, .75], method='linear')
    for row in rows:
        n = row['train_pitches']
        row['train_volume'] = 'low' if n <= q25 else 'middle' if n <= q75 else 'high'
    selected, strata = [], []
    for role, hand, volume in itertools.product(ROLES, HANDS, VOLUMES):
        candidates = sorted((r for r in rows if (r['train_role'], r['train_hand'], r['train_volume']) ==
                             (role, hand, volume)), key=lambda r: (r['selection_hash'], r['pitcher']))
        chosen = candidates[:per_cell]
        ids = [r['pitcher'] for r in chosen]
        selected.extend(ids)
        strata.append({'role': role, 'hand': hand, 'volume': volume,
                       'available_players': len(candidates), 'selected_players': len(ids),
                       'pitcher_ids': ids, 'status': 'full' if len(ids) == per_cell else 'empty' if not ids else 'underfilled'})
    return {'version': SELECTOR_VERSION, 'sampling_seed': seed, 'per_cell': per_cell,
            'requested_panel_size': 48, 'panel_size': len(selected), 'pitcher_ids': selected,
            'train_rows': len(train), 'train_games': int(train.game_pk.nunique()), 'train_pitchers': len(rows),
            'ordered_train_keys_sha256': ordered_hash, 'canonical_train_keys_sha256': canonical_hash,
            'train_batter_ids': sorted(int(x) for x in train.batter.unique()),
            'train_players': rows, 'strata': strata,
            'volume_thresholds': {'q25': float(q25), 'q75': float(q75),
                                  'method': 'global positive TRAIN pitcher-count linear quantiles; low<=q25,middle<=q75,high>q75'},
            'starter_validation': starter_validation,
            'role_rule': 'majority of eligible TRAIN pitches as game starter; exact 50% assigned starter',
            'hand_rule': 'all TRAIN observations must have the same L/R; ambiguous or missing hand excluded from panel',
            'excluded_hand_pitcher_ids': [r['pitcher'] for r in rows if r['train_hand'] == 'unknown'],
            'c6_overlap_ids': sorted(set(selected) & {int(x) for x in c6_ids}),
            'selection_information': 'TRAIN identity, role, hand, count and deterministic hash only; no DEV coverage or scores',
            'replacement': 'none for empty strata, absent DEV players, weak performance or unavailable future appearances'}


def evaluation_metadata(frame, panel):
    """Attach grouping columns to requested pitches without filtering any row.

    Called on either the full requested denominator or its explicit eligible
    subset. This function must never be used to fit or select the panel.
    """
    if panel.get('version') != SELECTOR_VERSION:
        raise ValueError('Unknown frozen panel version')
    needed = {*KEY, 'game_date', 'pitcher', 'batter', 'starter_pitcher', 'p_throws', 'strikes', 'bases'}
    if not needed <= set(frame.columns):
        raise ValueError('Missing evaluation grouping metadata')
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('Evaluation keys must be complete and unique')
    dates = pd.to_datetime(frame.game_date)
    if dates.isna().any() or dates.dt.year.ge(2026).any():
        raise ValueError('Invalid or forbidden evaluation dates')
    if frame.pitcher.isna().any() or frame.batter.isna().any():
        raise ValueError('Requested player identities must be available')
    lookup = {row['pitcher']: row for row in panel['train_players']}
    result = frame[KEY].copy()
    result['pitcher'] = frame.pitcher.to_numpy()
    result['batter'] = frame.batter.to_numpy()
    result['in_cpanel'] = frame.pitcher.isin(panel['pitcher_ids']).to_numpy()
    result['seen_pitcher'] = frame.pitcher.isin(lookup).to_numpy()
    result['seen_batter'] = frame.batter.isin(panel['train_batter_ids']).to_numpy()
    result['train_pitches'] = [lookup.get(int(pid), {}).get('train_pitches', 0) for pid in frame.pitcher]
    result['train_role'] = [lookup.get(int(pid), {}).get('train_role', 'unseen') for pid in frame.pitcher]
    result['train_volume'] = [lookup.get(int(pid), {}).get('train_volume', 'zero') for pid in frame.pitcher]
    known_role = frame.pitcher.notna() & frame.starter_pitcher.notna()
    result['game_role'] = np.where(known_role, np.where(frame.pitcher.eq(frame.starter_pitcher), 'starter', 'relief'), 'unknown')
    result['throwing_hand'] = frame.p_throws.where(frame.p_throws.isin(HANDS), 'unknown').to_numpy()
    result['two_strikes'] = pd.array([bool(value == 2) if pd.notna(value) and value in (0, 1, 2) else None
                                      for value in frame.strikes], dtype='boolean')
    result['runners_on'] = pd.array([bool(value > 0) if pd.notna(value) and value in range(8) else None
                                    for value in frame.bases], dtype='boolean')
    result['month'] = dates.dt.to_period('M').astype(str).to_numpy()
    return result


def group_reporting_status(game_ids, mask, *, event_count=None):
    """Minimum counts are reporting gates, never evidence of adequate power."""
    games, selected = np.asarray(game_ids), np.asarray(mask)
    if games.ndim != 1 or selected.shape != games.shape or selected.dtype != np.bool_:
        raise ValueError('Aligned game IDs and boolean selection required')
    if pd.isna(games).any():
        raise ValueError('Game IDs must be available for all requested rows')
    pitches = int(selected.sum())
    n_games = len(np.unique(games[selected]))
    reasons = []
    if n_games < 30:
        reasons.append('fewer_than_30_games')
    if pitches < 500:
        reasons.append('fewer_than_500_pitches')
    if event_count is not None:
        if not isinstance(event_count, (int, np.integer)) or not 0 <= event_count <= pitches:
            raise ValueError('Invalid event count')
        if event_count < 30:
            reasons.append('fewer_than_30_events')
    return {'n': pitches, 'games': n_games, 'events': event_count,
            'status': 'reporting_eligible' if not reasons else 'unmeasured_for_guardrail',
            'reasons': reasons, 'power_guarantee': False}


def guardrail_protocol(group_names):
    """Conservative simultaneous upper-bound contract for parent-owned scoring."""
    names = list(group_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('A nonempty frozen set of unique guardrail groups is required')
    return {'groups': names, 'metrics': ['nll', 'brier'], 'family_size': 2 * len(names),
            'method': 'Bonferroni one-sided upper confidence limits over all frozen groups and both metrics',
            'family_alpha': .05, 'per_bound_alpha': .05 / (2 * len(names)),
            'nll_margin': .010, 'brier_margin': .002, 'minimum_games': 30, 'minimum_pitches': 500,
            'missing_group': 'retain family slot; overall R remains unconfirmed if any required group is unmeasured',
            'bootstrap': 'paired whole-game resampling; pitch-weighted within each group; fixed predictions',
            'game_role_timing': 'starter identity known from first defensive pitch; later entrants are relief before their pitch',
            'warning': 'do not substitute ordinary two-sided 95% intervals for these simultaneous upper bounds'}
