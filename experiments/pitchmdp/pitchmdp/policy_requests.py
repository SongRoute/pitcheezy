"""Decision requests, PA outcomes, pre-decision profile snapshots and the S0 census (COOP-018, D93).

The request generator turns each regular-season PA into one ``DecisionRequest`` per recorded row
(R2) whose state holds only pre-decision information (R4): the current row's SAFE_COLUMNS
identity/count and the FULL list of strictly previous rows of the same PA (type label, normalized
physics, 10-class outcome, pre-pitch count). ``matrix_policy.state_from_row`` keeps only the last
five rows, which would break the ledger's ``len(history) == decision_index`` check for longer PAs,
so it is not reused here. A row without a usable choice gets a sentinel as both its logged action
and its later history action (D-3): ``NO_PITCH`` for a registered no-pitch call (automatic
ball/strike; the description wins over a present type), ``MISSING`` for a real pitch without a
type label. A structural defect at row k (gap, illegal count, missing identity) ends the request
list at k and is reported with its index, so earlier decisions are still submitted (R1/R4c) and
the PA stays in the denominator. A PA's last row that carries the terminal event but no pitch
(a registered no-pitch call such as ball four by the clock, or a type- and description-less
runner-event row) is a state transition, not a decision: no request is made for it (D-6 critic 2).
Outcomes (terminal event, next state, final result) are read only by ``pa_outcome`` for the
estimator's reward (R5) and never enter a request.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .archetypes import HISTORY_COLUMNS, add_batter_style_history
from .data import KEY
from .game import GameState
from .matrix_policy import context_key
from .model import outcome_labels
from .planner import OUTCOMES
from .policy_artifacts import IntegrityError, _require, normalize_hand, single_hand
from .policy_runtime import MISSING, NO_PITCH, SENTINELS, DecisionRequest, next_count
from .rollout_policy import PAState, PastPitch

AUTOMATIC = frozenset({'automatic_ball', 'automatic_strike'})  # default registered no-pitch descriptions
PA_KEY = ('game_pk', 'at_bat_number')
TERMINAL_RULES = ('structural-end-v1', 'r5-events-v1')
SCORE_SOURCES = ('next_row', 'post_pitch')
NO_TERMINAL, TERMINAL_VALUE_MISSING = 'NO_TERMINAL', 'TERMINAL_VALUE_MISSING'
BATTER_EVENTS = frozenset({
    'field_out', 'strikeout', 'force_out', 'fielders_choice_out', 'fielders_choice', 'sac_fly', 'sac_bunt',
    'grounded_into_double_play', 'double_play', 'sac_fly_double_play', 'sac_bunt_double_play',
    'strikeout_double_play', 'triple_play', 'single', 'double', 'triple', 'home_run', 'walk', 'intent_walk',
    'hit_by_pitch', 'field_error', 'catcher_interf'})
LEAGUE = 'league'  # style snapshot row for batters unseen before the as-of date (reliability 0)


def _missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value) or not str(value)


def logged_label(pitch_type, description, no_pitch=AUTOMATIC):
    """Observed action label (D-3): NO_PITCH for a registered no-pitch description (priority over a
    present type), MISSING for a pitch without a type label, else the raw Statcast code."""
    if description in no_pitch:
        return NO_PITCH
    if _missing(pitch_type):
        return MISSING
    return str(pitch_type)


def bc_population_mask(frame, no_pitch=AUTOMATIC):
    """D-1 BC-P: TRAIN rows with a logged pitch-type label and a legal pre-pitch count. Reads only
    split, pitch_type, balls, strikes and whether the description is a registered no-pitch call."""
    labelled = frame.pitch_type.notna() & frame.pitch_type.astype(str).ne('') & ~frame.description.isin(no_pitch)
    legal = frame.balls.between(0, 3) & frame.strikes.between(0, 2)
    return (frame.split.eq('train') & labelled & legal).to_numpy()


def _block_bounds(frame):
    game, ab = frame.game_pk.to_numpy(), frame.at_bat_number.to_numpy()
    start = np.r_[True, (game[1:] != game[:-1]) | (ab[1:] != ab[:-1])]
    return np.r_[np.flatnonzero(start), len(frame)]


def pa_blocks(frame, rows=None, *, strict=True):
    """[(pa_id, positions)] for contiguous PAs of ``frame`` (optionally only PAs touching ``rows``).

    strict: plate appearances and games must each be one contiguous block in frame order
    (IntegrityError otherwise); the S0 census counts violations instead.
    """
    bounds = _block_bounds(frame)
    game, ab = frame.game_pk.to_numpy(), frame.at_bat_number.to_numpy()
    keep = None if rows is None else set(np.asarray(rows).tolist())
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if keep is None or keep.intersection(range(lo, hi)):
            out.append((f'{int(game[lo])}:{int(ab[lo])}', np.arange(lo, hi)))
    if strict:
        _require(len({pa for pa, _ in out}) == len(out), 'plate appearances must be contiguous in frame order')
        changes = np.r_[True, game[1:] != game[:-1]]
        _require(changes.sum() == len(pd.unique(game)), 'games must be contiguous in frame order')
    return out


def terminal_placeholder(row, no_pitch=AUTOMATIC):
    """A last row with a terminal event and no pitcher choice: a registered no-pitch call, or a row
    without both type and description (runner-event / truncation record)."""
    if _missing(row.get('events')):
        return False
    return row['description'] in no_pitch or (_missing(row.get('description')) and _missing(row.get('pitch_type')))


def pa_requests(store, positions, runtime_sha256, no_pitch=AUTOMATIC):
    """(requests, problem, index): one request per decision row with the full strictly-previous history.

    A structural defect at row ``index`` ('ordering' at 0 for the whole PA; 'missing_row' for a
    pitch_number gap or a first row after pitch 1; 'illegal_count'; 'missing_identity';
    'invalid_history') truncates the list at ``index``: rows before it are submitted, the PA is
    censored there (or excluded from the start population when ``index == 0``). A terminal
    placeholder last row gets no request; a PA made only of one is 'no_decision'.
    """
    frame = store.frame
    rows = frame.iloc[positions]
    if (len(positions) == 0 or (np.diff(positions) != 1).any() or rows.game_pk.nunique() != 1
            or rows.at_bat_number.nunique() != 1 or (np.diff(rows.pitch_number.to_numpy()) <= 0).any()):
        return [], 'ordering', 0
    records = rows.to_dict('records')
    if terminal_placeholder(records[-1], no_pitch):
        if len(records) == 1:
            return [], 'no_decision', 0
        records, positions = records[:-1], positions[:-1]
    game, ab = int(rows.game_pk.iloc[0]), int(rows.at_bat_number.iloc[0])
    history, requests, previous_number = [], [], 0
    for index, (position, row) in enumerate(zip(positions, records)):
        if int(row['pitch_number']) != previous_number + 1:
            return requests, 'missing_row', index
        previous_number = int(row['pitch_number'])
        if not (0 <= row['balls'] <= 3 and 0 <= row['strikes'] <= 2):
            return requests, 'illegal_count', index
        if _missing(row['pitcher']) or _missing(row['stand']):
            return requests, 'missing_identity', index
        label = logged_label(row['pitch_type'], row['description'], no_pitch)
        hand = normalize_hand(row.get('p_throws'))
        state = PAState(int(row['balls']), int(row['strikes']), str(int(row['pitcher'])), str(row['stand']),
                        tuple(history), context_key(row))
        requests.append(DecisionRequest(f'{game}:{ab}:{int(row["pitch_number"])}', f'{game}:{ab}', index, state,
                                        label, runtime_sha256, hand))
        outcome = int(store.outcome_channels[position].argmax())
        try:
            history.append(PastPitch(label, tuple(float(v) for v in store.physical[position]),
                                     OUTCOMES[outcome] if outcome < len(OUTCOMES) else 'unknown',
                                     int(row['balls']), int(row['strikes'])))
        except ValueError:
            if index + 1 < len(positions):
                return requests, 'invalid_history', index + 1
    return requests, None, None


def pa_manifest(frame, positions):
    """Pre-decision PA facts outside the request fingerprint (D-4, M-13): the first pitcher/batter
    change index, stand change, start pitcher and date. The ledger's own pitcher field is the
    authority for the pitcher change; this manifest is the only source for a same-side batter swap."""
    rows = frame.iloc[positions]
    pitcher, batter, stand = rows.pitcher.to_numpy(), rows.batter.to_numpy(), rows.stand.astype(str).to_numpy()

    def first_change(values):
        changed = np.flatnonzero(values != values[0])
        return int(changed[0]) if len(changed) else None
    return {'game': int(rows.game_pk.iloc[0]), 'start_pitcher': str(int(pitcher[0])) if pd.notna(pitcher[0]) else None,
            'first_pitcher_change_index': first_change(pitcher), 'batter_change_index': first_change(batter),
            'stand_change': first_change(stand) is not None, 'date': str(pd.Timestamp(rows.game_date.iloc[0]).date())}


# ---------------------------------------------------------------- PA end and reward (D-6)

def game_table(frame):
    """Per game: frame position range, last at-bat number and the game_final_v1 verdict.

    Games must be contiguous and ordered by (at_bat_number, pitch_number); post-pitch scores are
    required. game_final_v1 (D-6 critic 3; replaces data.py's complete_game heuristic): the game's
    last recorded row carries a terminal event (any code, truncated_pa included), its post-pitch
    scores are present and not tied, and the inning is at least 5 (MLB regulation game). Since
    2020 an unfinished game is suspended and resumed under the same game_pk, so its rows stay in
    one block; a tie or a shorter record is not final.
    """
    _require({'post_home_score', 'post_away_score'} <= set(frame.columns), 'post-pitch score columns are required')
    game = frame.game_pk.to_numpy()
    starts = np.flatnonzero(np.r_[True, game[1:] != game[:-1]])
    _require(len(starts) == len(pd.unique(game)), 'games must be contiguous in frame order')
    table = {}
    for lo, hi in zip(starts, np.r_[starts[1:], len(frame)]):
        part = frame.iloc[lo:hi]
        order = part.at_bat_number.to_numpy() * 10 ** 4 + part.pitch_number.to_numpy()
        _require((np.diff(order) > 0).all(), f'game rows out of (at_bat, pitch) order: {int(game[lo])}')
        last = part.iloc[-1]
        home, away = last.post_home_score, last.post_away_score
        scored = not (pd.isna(home) or pd.isna(away))
        final = (not _missing(last.events)) and scored and home != away and int(last.inning) >= 5
        table[int(game[lo])] = {'lo': int(lo), 'hi': int(hi), 'max_ab': int(last.at_bat_number), 'final': bool(final),
                                'home_won': bool(final and home > away),
                                'complete_game_flag': (None if 'complete_game' not in part or pd.isna(last.complete_game)
                                                       else bool(last.complete_game))}
    return table


def _legal_transition(before, after):
    """Next PA's first row after ``before`` (this PA's first row): same half-inning with outs and
    scores not decreasing, or a legal half-inning change (game._advance: outs 0, bases empty or
    the extra-inning runner on second)."""
    same = int(after.inning) == int(before.inning) and after.inning_topbot == before.inning_topbot
    if same:
        return (int(after.outs_when_up) >= int(before.outs_when_up) and int(after.home_score) >= int(before.home_score)
                and int(after.away_score) >= int(before.away_score))
    flips = ((before.inning_topbot == 'Top' and after.inning_topbot == 'Bot' and int(after.inning) == int(before.inning))
             or (before.inning_topbot == 'Bot' and after.inning_topbot == 'Top' and int(after.inning) == int(before.inning) + 1))
    bases = GameState.from_row(after).bases
    return flips and int(after.outs_when_up) == 0 and bases == (2 if int(after.inning) >= 10 else 0)


def pa_outcome(frame, positions, defense_we, games, *, rule='structural-end-v1', score_source='next_row',
               batter_events=BATTER_EVENTS):
    """Terminal reward of one PA for the initial defender (R5/D-6), or None with a reason and kind.

    structural-end-v1: the PA ended iff its last recorded row has an ``events`` code (any code,
    ``truncated_pa`` and runner events included) AND the adjacent next PA's first row is observed
    (same game, at_bat + 1, pitch 1 at 0-0, legal transition) or this is the game's last PA of an
    officially complete game. The reward is the frozen C0 WE of that next pre-pitch state, or the
    final result. ``kind``: NO_TERMINAL (the PA's end is not observed; policy-specific unknown
    continuation) or TERMINAL_VALUE_MISSING (the PA ended but its end value is unobservable; one
    unknown shared by both policies). r5-events-v1 (registered sensitivity) also treats
    ``truncated_pa`` as NO_TERMINAL.
    """
    _require(rule in TERMINAL_RULES and score_source in SCORE_SOURCES, 'registered terminal rule required')
    first, last = frame.iloc[positions[0]], frame.iloc[positions[-1]]
    game, defender_is_home = int(first.game_pk), str(first.inning_topbot) == 'Top'
    out = {'game': game, 'reward': None, 'reason': None, 'kind': None, 'end': None, 'end_kind': None, 'flags': []}
    if pd.isna(last.events) or not str(last.events):
        return {**out, 'reason': 'no_terminal_event', 'kind': NO_TERMINAL}
    if rule == 'r5-events-v1' and last.events == 'truncated_pa':
        return {**out, 'reason': 'truncated_pa_r5', 'kind': NO_TERMINAL}
    end_kind = 'batter_event' if last.events in batter_events else 'non_batter_end'
    info = games[game]
    if int(last.at_bat_number) == info['max_ab']:
        _require(positions[-1] == info['hi'] - 1, 'the last PA of a game must end its frame block')
        if not info['final']:
            return {**out, 'reason': 'game_not_final_v1', 'kind': TERMINAL_VALUE_MISSING, 'end_kind': 'game_final'}
        return {**out, 'reward': float(info['home_won'] == defender_is_home), 'end': 'final_result',
                'end_kind': 'game_final'}
    following = positions[-1] + 1
    _require(following < info['hi'], 'a non-final PA must be followed by a row of the same game')
    nxt = frame.iloc[following]
    if (int(nxt.at_bat_number) != int(last.at_bat_number) + 1 or int(nxt.pitch_number) != 1
            or (int(nxt.balls), int(nxt.strikes)) != (0, 0)):
        return {**out, 'reason': 'end_state_gap', 'kind': TERMINAL_VALUE_MISSING, 'end_kind': end_kind}
    try:
        legal = _legal_transition(first, nxt)
        state = GameState.from_row(nxt)
    except (ValueError, TypeError, KeyError):
        return {**out, 'reason': 'invalid_next_state', 'kind': TERMINAL_VALUE_MISSING, 'end_kind': end_kind}
    if not legal:
        return {**out, 'reason': 'end_state_gap', 'kind': TERMINAL_VALUE_MISSING, 'end_kind': end_kind}
    if last.post_home_score != nxt.home_score or last.post_away_score != nxt.away_score:
        out['flags'].append('post_pitch_score_disagrees_next_row')
    if score_source == 'post_pitch':
        if pd.isna(last.post_home_score) or pd.isna(last.post_away_score):
            return {**out, 'reason': 'post_pitch_score_missing', 'kind': TERMINAL_VALUE_MISSING, 'end_kind': end_kind}
        state = GameState(state.inning, state.half, state.outs, state.bases, int(last.post_home_score),
                          int(last.post_away_score))
    value = float(defense_we(state, defender_is_home))
    _require(np.isfinite(value) and 0 <= value <= 1, 'defensive WE outside [0, 1]')
    return {**out, 'reward': value, 'end': 'next_row_state', 'end_kind': end_kind}


# ---------------------------------------------------------------- batter style snapshot (D-7)

def style_snapshot(frame, as_of):
    """Batter style priors frozen at ``as_of`` (exclusive; R4b/D-7): every batter seen in rows dated
    strictly before ``as_of`` gets its C0 prior from those rows only, plus a ``league`` row
    (reliability 0) for batters first seen later. The evaluation-window roster is never read.
    Returns a float32 frame indexed by batter id string (``league`` last)."""
    as_of = pd.Timestamp(as_of).normalize()
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    columns = [c for c in ('batter', 'events', 'description', 'is_pa_terminal', 'launch_angle') if c in frame.columns]
    past = frame.loc[dates < as_of, columns].assign(game_date=dates[dates < as_of])
    _require(len(past) and not past.batter.isin([-1]).any(), 'style snapshot needs earlier rows (and no batter id -1)')
    # add_batter_style_history rejects 2026 dates; with no rows in between, the day after the last
    # earlier row gives the same strictly-prior values.
    probe_date = as_of if as_of.year < 2026 else past.game_date.max() + pd.Timedelta(days=1)
    batters = np.sort(pd.unique(past.batter))
    probe = pd.DataFrame({'batter': np.r_[batters, -1], 'game_date': probe_date, 'events': None, 'description': '',
                          'is_pa_terminal': False})
    if 'launch_angle' in columns:
        probe['launch_angle'] = np.nan
    combined = add_batter_style_history(pd.concat([past, probe], ignore_index=True))
    values = combined.iloc[len(past):][list(HISTORY_COLUMNS)].astype(np.float32)
    values.index = [str(int(b)) for b in batters] + [LEAGUE]
    return values


def apply_style_snapshot(rows, snapshot, as_of):
    """Replace the style columns of evaluation rows by the frozen snapshot; batters absent from it
    take the ``league`` row. Every row must be dated on/after ``as_of`` (no mixing). Returns
    (rows, report)."""
    rows = rows.copy()
    dates = pd.to_datetime(rows.game_date).dt.normalize()
    _require((dates >= pd.Timestamp(as_of).normalize()).all(), 'evaluation rows dated before the style as-of date')
    keys = rows.batter.map(lambda b: str(int(b)))
    unknown = ~keys.isin(snapshot.index) | keys.eq(LEAGUE)
    lookup = keys.where(~unknown, LEAGUE)
    for column in HISTORY_COLUMNS:
        rows[column] = lookup.map(snapshot[column]).to_numpy(np.float32)
    return rows, {'unknown_batters': int(keys[unknown].nunique()), 'unknown_batter_rows': int(unknown.sum())}


def snapshot_rolling_mismatches(rows, snapshot, as_of):
    """Rows dated exactly ``as_of``: their rolling C0 prior uses only earlier dates, so it must equal
    the snapshot (known batters) or the league row (first seen that day) bit for bit."""
    day = pd.to_datetime(rows.game_date).dt.normalize().eq(pd.Timestamp(as_of).normalize())
    if not day.any():
        return {'rows_on_as_of': 0, 'mismatches': 0}
    applied, _ = apply_style_snapshot(rows.loc[day], snapshot, as_of)
    rolling = rows.loc[day, list(HISTORY_COLUMNS)].to_numpy(np.float32)
    return {'rows_on_as_of': int(day.sum()),
            'mismatches': int((applied[list(HISTORY_COLUMNS)].to_numpy(np.float32) != rolling).any(axis=1).sum())}


# ---------------------------------------------------------------- roles, strata (M-13)

def pitcher_roles(frame):
    """D87 role per row: 'SP' if the pitcher is the first pitcher of its fielding side in the game
    (known before the PA), else 'RP'."""
    side = np.where(frame.inning_topbot.astype(str).eq('Top'), 'home', 'away')
    keyed = pd.DataFrame({'game': frame.game_pk.to_numpy(), 'side': side, 'pitcher': frame.pitcher.to_numpy()})
    starters = keyed.groupby(['game', 'side'], sort=False).pitcher.transform('first')
    return pd.Series(np.where(keyed.pitcher.to_numpy() == starters.to_numpy(), 'SP', 'RP'), index=frame.index)


def volume_edges(train_rows, quantiles):
    """Pitcher TRAIN pitch-count quantile edges (registered levels) for descriptive volume strata."""
    counts = train_rows.groupby('pitcher').size().to_numpy(np.float64)
    return [float(np.quantile(counts, q)) for q in quantiles]


# ---------------------------------------------------------------- S0 census

def _crosstab(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def census(frame, vocabulary, *, no_pitch=AUTOMATIC, outcome_adjacent_splits=('train',)):
    """Label-blind pre-run census (S0): codes, no-action rows, PA/game structure, hands, changes and
    count-path breaks per split. Outcome-adjacent items (terminal event codes, truncation, score
    mismatches, game ends) are computed only for ``outcome_adjacent_splits`` (TRAIN by default);
    no outcome distribution or value is computed for any evaluation split.
    """
    vocabulary, out = set(vocabulary), {}
    label = pd.Series([logged_label(t, d, no_pitch) for t, d in zip(frame.pitch_type, frame.description)],
                      index=frame.index)
    real = ~label.isin(SENTINELS)
    train = frame.loc[frame.split.eq('train')]
    train_codes = set(label[frame.split.eq('train') & real])
    encoded = pd.Series(outcome_labels(frame), index=frame.index)
    for split, part in frame.groupby('split', sort=True):
        blocks = pa_blocks(part.reset_index(drop=True), strict=False)
        ids = [pa for pa, _ in blocks]
        lab = label.loc[part.index].to_numpy()
        first_no_action, zero_decisions, count_breaks, changes = Counter(), 0, 0, Counter()
        unknown_nonterminal, gaps, first_not_1 = 0, 0, 0
        pitcher_change, batter_change, stand_change, first_last_pitcher, change_unseen = 0, 0, 0, 0, 0
        values = part.reset_index(drop=True)
        enc = encoded.loc[part.index].to_numpy()
        train_pitchers = set(train.pitcher)
        for _, positions in blocks:
            rows = values.iloc[positions]
            sentinel = np.isin(lab[positions], SENTINELS)
            if sentinel.any():
                k = int(np.flatnonzero(sentinel)[0])
                first_no_action['first' if k == 0 else 'last' if k == len(positions) - 1 else 'middle'] += 1
                zero_decisions += int(sentinel.all())
            numbers = rows.pitch_number.to_numpy()
            first_not_1 += int(numbers[0] != 1)
            gaps += int((np.diff(numbers) != 1).any())
            for j in range(1, len(positions)):
                outcome = OUTCOMES[enc[positions[j - 1]]] if enc[positions[j - 1]] >= 0 else 'unknown'
                try:
                    follows = next_count(int(rows.balls.iloc[j - 1]), int(rows.strikes.iloc[j - 1]), outcome)
                except (IntegrityError, ValueError):
                    follows = 'unknown'
                if follows == 'unknown':
                    unknown_nonterminal += int(not sentinel[j - 1])
                elif follows != (int(rows.balls.iloc[j]), int(rows.strikes.iloc[j])):
                    count_breaks += 1
            p = rows.pitcher.to_numpy()
            n_changes = int((p[1:] != p[:-1]).sum())
            changes[n_changes] += 1
            pitcher_change += int(n_changes > 0)
            first_last_pitcher += int(p[0] != p[-1])
            change_unseen += int(n_changes > 0 and not set(p[1:][p[1:] != p[0]]) <= train_pitchers)
            batter_change += int(rows.batter.nunique() > 1)
            stand_change += int(rows.stand.astype(str).nunique() > 1)
        game = values.game_pk.to_numpy()
        codes = part.pitch_type.fillna('<NA>').astype(str).replace('', '<EMPTY>')
        missing_type = part.pitch_type.isna() | part.pitch_type.astype(str).eq('')
        hands = part.p_throws.astype(object).where(part.p_throws.notna(), '<NA>').astype(str)
        at_bats = values.drop_duplicates(list(PA_KEY))
        ab_gaps = int(sum((np.diff(g.at_bat_number.to_numpy()) != 1).sum() for _, g in at_bats.groupby('game_pk')))
        item = {
            'rows': int(len(part)), 'pas': len(blocks), 'games': int(part.game_pk.nunique()),
            'noncontiguous_pas': len(ids) - len(set(ids)),
            'noncontiguous_games': int(np.r_[True, game[1:] != game[:-1]].sum() - len(pd.unique(game))),
            'pitch_type_codes': {k: int(v) for k, v in sorted(codes.value_counts().items())},
            'codes_outside_vocabulary': sorted(set(codes) - vocabulary - {'<NA>', '<EMPTY>'}),
            'codes_outside_train_labels': sorted(set(lab[~np.isin(lab, SENTINELS)]) - train_codes),
            'missing_label_rows': int((lab == MISSING).sum()), 'no_pitch_rows': int((lab == NO_PITCH).sum()),
            'no_pitch_rows_with_type': int((part.description.isin(no_pitch) & ~missing_type).sum()),
            'missing_type_by_description': _crosstab(part.description.fillna('<NA>')[missing_type]),
            'pas_by_first_no_action_position': dict(first_no_action), 'pas_without_real_decision': zero_decisions,
            'illegal_count_rows': int((~(part.balls.between(0, 3) & part.strikes.between(0, 2))).sum()),
            'pa_first_row_not_0_0': int(sum(values.iloc[pos[0]][['balls', 'strikes']].tolist() != [0, 0]
                                             for _, pos in blocks)),
            'pa_first_pitch_number_not_1': first_not_1, 'pa_with_pitch_number_gap': gaps,
            'at_bat_number_gaps': ab_gaps, 'count_path_breaks': count_breaks,
            'unknown_outcome_nonterminal_real_pitches': unknown_nonterminal,
            'pa_with_pitcher_change': pitcher_change, 'pitcher_changes_per_pa': _crosstab(changes.elements()),
            'pa_first_last_pitcher_differ': first_last_pitcher, 'pa_change_to_pitcher_unseen_in_train': change_unseen,
            'pa_with_batter_change': batter_change, 'pa_with_stand_change': stand_change,
            'hand_values': _crosstab(hands), 'hand_missing_or_invalid_rows': int((~hands.isin(['L', 'R'])).sum()),
            'stand_missing_or_invalid_rows': int((~part.stand.astype(object).map(normalize_hand).isin(['L', 'R'])).sum()),
            'games_with_several_dates': int((values.groupby('game_pk').game_date.nunique() > 1).sum()),
            'pas_by_game': {str(int(g)): int(n) for g, n in
                            values.drop_duplicates(list(PA_KEY)).groupby('game_pk').size().items()}}
        if split in outcome_adjacent_splits:
            last = values.iloc[[pos[-1] for _, pos in blocks]]
            nonlast = values.drop(index=[pos[-1] for _, pos in blocks])
            ends = values.groupby('game_pk', sort=False).tail(1)
            placeholder = [terminal_placeholder(r, no_pitch) for r in last.to_dict('records')]
            following = values.shift(-1)
            same_game = following.game_pk.eq(values.game_pk)
            last_mask = values.index.isin(last.index) & same_game
            mismatch = last_mask & ((values.post_home_score != following.home_score)
                                    | (values.post_away_score != following.away_score))
            try:
                games = game_table(values)
            except IntegrityError as error:  # counted here; the reward stages fail closed on it
                games, item['game_table_error'] = {}, str(error)
            item['outcome_adjacent'] = {
                'last_row_event_codes': _crosstab(last.events.fillna('<NA>')),
                'last_row_events_by_type_missing_and_description': _crosstab(
                    f"{e}|{'type_missing' if _missing(t) else 'typed'}|{d}" for e, t, d in
                    zip(last.events.fillna('<NA>'), last.pitch_type, last.description.fillna('<NA>'))),
                'terminal_placeholder_last_rows': int(sum(placeholder)),
                'nonlast_rows_with_events': int(nonlast.events.notna().sum()),
                'truncated_pa': int(last.events.eq('truncated_pa').sum()),
                'post_score_disagrees_next_row': int(mismatch.sum()),
                'games_last_row_without_events': int(ends.events.isna().sum()),
                'games_ending_before_inning_9': int((ends.inning < 9).sum()),
                'games_final_v1': int(sum(g['final'] for g in games.values())),
                'games_final_v1_disagrees_complete_game': int(sum(
                    g['complete_game_flag'] is not None and g['final'] != g['complete_game_flag'] for g in games.values())),
                'missing_label_by_count': _crosstab(f'{b}-{k}' for b, k, lb in zip(part.balls, part.strikes, lab)
                                                    if lb == MISSING),
                'pitchers_with_missing_labels': int(part.pitcher[lab == MISSING].nunique())}
        out[str(split)] = item
    hands = train.groupby('pitcher').p_throws.agg(lambda v: single_hand(v.tolist()) == 'AMBIGUOUS')
    out['train_pitchers_ambiguous_hand'] = int(hands.sum())
    ambiguous = set(hands.index[hands])
    splits_per_game = frame.groupby('game_pk').split.nunique()
    out['games_spanning_splits'] = sorted(int(g) for g in splits_per_game.index[splits_per_game > 1])
    out['train_rows_of_ambiguous_hand_pitchers'] = int(train.pitcher.isin(ambiguous).sum())
    known = set(zip(train.pitcher, train.p_throws))
    for split in sorted(set(frame.split) - {'train'}):
        part = frame.loc[frame.split.eq(split)]
        pairs = list(zip(part.pitcher, part.p_throws))
        out[str(split)]['rows_with_pitcher_hand_unseen_in_train'] = int(sum(p not in known for p in pairs))
        out[str(split)]['pitchers_unseen_in_train'] = len(set(part.pitcher) - set(train.pitcher))
    bc_rows = bc_population_mask(frame, no_pitch)
    tr = frame.split.eq('train')
    out['bc_p_train'] = {'rows': int(bc_rows.sum()), 'train_rows': int(tr.sum()),
                         'excluded_missing_label': int((tr & (label == MISSING)).sum()),
                         'excluded_no_pitch': int((tr & (label == NO_PITCH)).sum()),
                         'excluded_illegal_count': int((tr & real & ~(frame.balls.between(0, 3)
                                                                      & frame.strikes.between(0, 2))).sum())}
    out['key'] = list(KEY)
    out['no_pitch_descriptions'] = sorted(no_pitch)
    out['reads'] = ('pitch_type, description (no-pitch flag, 10-class encoding for count paths), pre-pitch fields, '
                    'identity, at_bat/pitch numbers; events/scores/completion only for outcome_adjacent_splits')
    return out
