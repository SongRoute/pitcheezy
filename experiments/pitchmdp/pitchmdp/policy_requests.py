"""Decision requests, PA outcomes and pre-decision profile snapshots from processed rows (COOP-018).

The request generator turns each regular-season PA into one ``DecisionRequest`` per recorded row
(R2) whose state holds only pre-decision information (R4): the current row's SAFE_COLUMNS
identity/count and the FULL list of strictly previous rows of the same PA (type label, normalized
physics, 10-class outcome, pre-pitch count). ``matrix_policy.state_from_row`` keeps only the last
five rows, which would break the ledger's ``len(history) == decision_index`` check for longer PAs,
so it is not reused here. A row without a pitcher choice (automatic ball/strike, missing type)
gets the ``NO_PITCH`` label as both its logged action and its later history action (R3c).
Outcomes (terminal event, next state, final result) are read only by ``pa_outcome`` for the
estimator's reward (R5) and never enter a request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .archetypes import HISTORY_COLUMNS, add_batter_style_history
from .data import KEY
from .game import GameState
from .matrix_policy import context_key
from .planner import OUTCOMES
from .policy_artifacts import _require
from .policy_runtime import NO_PITCH, DecisionRequest
from .rollout_policy import PAState, PastPitch

AUTOMATIC = frozenset({'automatic_ball', 'automatic_strike'})
PA_KEY = ('game_pk', 'at_bat_number')


def logged_label(pitch_type, description):
    """Observed action label: the raw Statcast code, or NO_PITCH when there was no pitcher choice."""
    if description in AUTOMATIC or pd.isna(pitch_type) or not str(pitch_type):
        return NO_PITCH
    return str(pitch_type)


def pa_blocks(frame, rows=None):
    """[(pa_id, positions)] for contiguous PAs of ``frame`` (optionally only PAs touching ``rows``)."""
    game, ab = frame.game_pk.to_numpy(), frame.at_bat_number.to_numpy()
    start = np.r_[True, (game[1:] != game[:-1]) | (ab[1:] != ab[:-1])]
    bounds = np.r_[np.flatnonzero(start), len(frame)]
    keep = None if rows is None else set(np.asarray(rows).tolist())
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if keep is None or lo in keep:
            out.append((f'{int(game[lo])}:{int(ab[lo])}', np.arange(lo, hi)))
    _require(len({pa for pa, _ in out}) == len(out), 'plate appearances must be contiguous in frame order')
    return out


def pa_requests(store, positions, runtime_sha256):
    """(requests, problem): one request per row, full strictly-previous history; ``problem`` names a
    structural defect that makes the PA unsubmittable (it stays in the reported PA denominator)."""
    frame = store.frame
    rows = frame.iloc[positions]
    if (len(positions) == 0 or (np.diff(positions) != 1).any() or rows.game_pk.nunique() != 1
            or rows.at_bat_number.nunique() != 1 or (np.diff(rows.pitch_number.to_numpy()) <= 0).any()):
        return [], 'ordering'
    if not (rows.balls.between(0, 3).all() and rows.strikes.between(0, 2).all()):
        return [], 'illegal_count'
    if rows.pitcher.isna().any() or rows.stand.isna().any():
        return [], 'missing_identity'
    game, ab = int(rows.game_pk.iloc[0]), int(rows.at_bat_number.iloc[0])
    history, requests = [], []
    for index, (position, row) in enumerate(zip(positions, rows.to_dict('records'))):
        label = logged_label(row['pitch_type'], row['description'])
        state = PAState(int(row['balls']), int(row['strikes']), str(int(row['pitcher'])), str(row['stand']),
                        tuple(history), context_key(row))
        requests.append(DecisionRequest(f'{game}:{ab}:{int(row["pitch_number"])}', f'{game}:{ab}', index, state,
                                        label, runtime_sha256))
        outcome = int(store.outcome_channels[position].argmax())
        history.append(PastPitch(label, tuple(float(v) for v in store.physical[position]),
                                 OUTCOMES[outcome] if outcome < len(OUTCOMES) else 'unknown',
                                 int(row['balls']), int(row['strikes'])))
    return requests, None


def pa_outcome(frame, positions, defense_we):
    """Terminal reward of one PA for the initial defender (R5), or None with a reason (R6).

    Terminal = last row's ``events`` present and not ``truncated_pa``. The end state is the next
    recorded row's pre-pitch state in the same game; for a game's last PA it is the final result.
    """
    first, last = frame.iloc[positions[0]], frame.iloc[positions[-1]]
    game, defender_is_home = int(first.game_pk), str(first.inning_topbot) == 'Top'
    out = {'game': game, 'reward': None, 'reason': None, 'end': None, 'flags': []}
    if pd.isna(last.events) or last.events == 'truncated_pa':
        return {**out, 'reason': 'no_terminal_event'}
    following = positions[-1] + 1
    if following < len(frame) and int(frame.game_pk.iloc[following]) == game:
        nxt = frame.iloc[following]
        if int(nxt.at_bat_number) <= int(last.at_bat_number):
            return {**out, 'reason': 'unordered_next_row'}
        try:
            state = GameState.from_row(nxt)
        except (ValueError, TypeError, KeyError):
            return {**out, 'reason': 'invalid_next_state'}
        if {'post_home_score', 'post_away_score'} <= set(frame.columns) and (
                last.post_home_score != nxt.home_score or last.post_away_score != nxt.away_score):
            out['flags'].append('post_pitch_score_disagrees_next_row')
        value = float(defense_we(state, defender_is_home))
        _require(np.isfinite(value) and 0 <= value <= 1, 'defensive WE outside [0, 1]')
        return {**out, 'reward': value, 'end': 'next_row_state'}
    if not {'post_home_score', 'post_away_score'} <= set(frame.columns):
        return {**out, 'reason': 'final_score_columns_missing'}
    if 'complete_game' in frame.columns and not bool(last.complete_game):
        return {**out, 'reason': 'incomplete_game'}
    home, away = last.post_home_score, last.post_away_score
    if pd.isna(home) or pd.isna(away) or home == away:
        return {**out, 'reason': 'final_result_undetermined'}
    return {**out, 'reward': float((home > away) == defender_is_home), 'end': 'final_result'}


def style_snapshot(frame, as_of):
    """Batter style priors frozen at ``as_of`` (R4b): each batter's C0 prior computed from rows dated
    strictly before ``as_of``; no update from rows inside the evaluation window."""
    as_of = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    columns = [c for c in ('batter', 'events', 'description', 'is_pa_terminal', 'launch_angle') if c in frame.columns]
    past = frame.loc[dates < as_of, columns].assign(game_date=dates[dates < as_of])
    batters = pd.unique(frame.loc[dates >= as_of, 'batter'].dropna())
    probe = pd.DataFrame({'batter': batters, 'game_date': as_of, 'events': None, 'description': '',
                          'is_pa_terminal': False})
    if 'launch_angle' in columns:
        probe['launch_angle'] = np.nan
    combined = add_batter_style_history(pd.concat([past, probe], ignore_index=True))
    return combined.iloc[len(past):].set_index('batter')[list(HISTORY_COLUMNS)]


def apply_style_snapshot(rows, snapshot, as_of):
    """Replace the style columns of rows on/after ``as_of`` by the frozen snapshot (every batter required)."""
    rows = rows.copy()
    late = pd.to_datetime(rows.game_date).dt.normalize() >= pd.Timestamp(as_of).normalize()
    _require(rows.loc[late, 'batter'].isin(snapshot.index).all(), 'batter missing from the style snapshot')
    for column in HISTORY_COLUMNS:
        rows.loc[late, column] = rows.loc[late, 'batter'].map(snapshot[column]).to_numpy()
    return rows


def census(frame, vocabulary):
    """Label-blind pre-run census (S0): codes, no-pitch rows and PA structure per split.

    Reads the action codes, whether a description is an automatic call, and whether a PA has a
    terminal event; no outcome distribution is computed.
    """
    vocabulary, out = set(vocabulary), {}
    for split, part in frame.groupby('split', sort=True):
        codes = part.pitch_type.fillna('<NA>').astype(str).value_counts().to_dict()
        blocks = pa_blocks(part.reset_index(drop=True))
        first = part.groupby(list(PA_KEY), sort=False).head(1)
        last = part.groupby(list(PA_KEY), sort=False).tail(1)
        out[str(split)] = {
            'rows': int(len(part)), 'pas': len(blocks), 'games': int(part.game_pk.nunique()),
            'pitch_type_codes': {k: int(v) for k, v in sorted(codes.items())},
            'codes_outside_vocabulary': sorted(c for c in codes if c != '<NA>' and c not in vocabulary),
            'missing_pitch_type_rows': int(part.pitch_type.isna().sum()),
            'automatic_call_rows': int(part.description.isin(AUTOMATIC).sum()),
            'illegal_count_rows': int((~(part.balls.between(0, 3) & part.strikes.between(0, 2))).sum()),
            'pa_first_row_not_0_0': int(((first.balls != 0) | (first.strikes != 0)).sum()),
            'pa_without_terminal_event': int((last.events.isna() | last.events.eq('truncated_pa')).sum()),
            'pa_with_pitcher_change': int((part.groupby(list(PA_KEY)).pitcher.nunique() > 1).sum()),
            'pa_with_batter_change': int((part.groupby(list(PA_KEY)).batter.nunique() > 1).sum())}
    train = frame.loc[frame.split.eq('train')]
    hands = train.groupby('pitcher').p_throws.nunique()
    out['train_pitchers_with_two_hands'] = int((hands > 1).sum())
    known = set(zip(train.pitcher, train.p_throws))
    for split in sorted(set(frame.split) - {'train'}):
        part = frame.loc[frame.split.eq(split)]
        pairs = set(zip(part.pitcher, part.p_throws))
        out[str(split)]['pitcher_hand_pairs_unseen_in_train'] = len(pairs - known)
        out[str(split)]['pitchers_unseen_in_train'] = len(set(part.pitcher) - set(train.pitcher))
    out['key'] = list(KEY)
    return out
