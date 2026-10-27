"""Pre-pitch features, populations, folds and A3 batter-style inputs of EXP-BC-CAL-01 (sections 5, 6, 8).

Every feature of a row uses only (a) strictly earlier rows of the same PA or game, (b) frozen TRAIN
artifacts (the registered BC_P counts, TRAIN volumes, the frozen G0 archetypes), or (c) the
registered A3 batter-style history: strictly-prior-date values on TRAIN rows and window-start
snapshots (earlystop 2025-05-01, temperature 2025-05-16, DEV 2025-07-01), all built only from
TRAIN, earlystop and temperature rows (2023-05-15..2025-05-31; never June, DEV, 2023-03-30..05-14
or 2026). The current pitch's type, physics and outcome never enter: history is a within-PA shift
of earlier rows only. DEV pitch-type labels are replaced by ``SEALED`` until the runner releases
them; any feature or population built over a sealed label raises ``SealedLabelError``.

The census works on ``label_free_view`` (pre-pitch columns only), so it cannot read a pitch-type
label, description or event by construction. Nothing here reads files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .archetypes import HISTORY_COLUMNS, RELIABILITY_COLUMNS, add_batter_style_history
from .behavior_calibration_registry import (ACTION_CODES, B2_FEATURES, LAST_DATE, SPLIT_DATES, STYLE_AS_OF,
                                            STYLE_INPUT_FILTER, USED_SPLITS)
from .behavior_model import START, UNKNOWN, BehaviorTable, Keys, VectorBC
from .policy_artifacts import IntegrityError, normalize_hand

KEY = ('game_pk', 'at_bat_number', 'pitch_number')
SEALED = '<SEALED>'
TYPE_UNKNOWN, TYPE_NONE = len(ACTION_CODES), len(ACTION_CODES) + 1
OUTCOME_UNKNOWN, OUTCOME_NONE = 10, 11
HISTORY = 5
AUTOMATIC = frozenset({'automatic_ball', 'automatic_strike'})  # policy_requests.AUTOMATIC
CENSUS_COLUMNS = ('game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'split', 'pitcher', 'p_throws', 'stand',
                  'balls', 'strikes')
FORBIDDEN_INPUTS = ('pitch_type', 'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'plate_x',
                    'plate_z', 'effective_speed', 'sz_top', 'sz_bot', 'description', 'events', 'type',
                    'post_home_score', 'post_away_score', 'post_bat_score', 'post_fld_score', 'launch_speed',
                    'launch_angle', 'supported_pa', 'pitcher_days_until_next_game', 'batter_days_until_next_game',
                    'n_thruorder_pitcher', 'pitcher_days_since_prev_game', 'game_date', 'month', 'season',
                    'game_year', 'pitcher', 'batter', 'is_pa_terminal')
STYLE_SOURCE_COLUMNS = ('batter', 'game_date', 'events', 'description', 'is_pa_terminal', 'launch_angle')
WORKLOAD_RAW_COLUMNS = ['game_date', *KEY, 'pitcher']
WORKLOAD_FORBIDDEN = ('pitcher_days_until_next_game', 'batter_days_until_next_game')


class SealedLabelError(IntegrityError):
    """A DEV pitch-type label was needed before the runner released it (stage-order violation)."""


# ---------------------------------------------------------------- folds, dates, allowlists

def game_fold(games, salt, k):
    """fold(g) = int(sha256(salt + ':' + str(game_pk)).hexdigest(), 16) mod k; one fold per game."""
    games = np.asarray(games)
    unique, inverse = np.unique(games.astype(np.int64), return_inverse=True)
    folds = np.array([int(hashlib.sha256(f'{salt}:{int(g)}'.encode()).hexdigest(), 16) % k for g in unique],
                     dtype=np.int64)
    return folds[inverse] if len(games) else np.zeros(0, dtype=np.int64)


def guard_dates(frame):
    """Refuse any 2026 row or any date after 2025-09-30 (stop rule: stop immediately)."""
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    if dates.isna().any() or dates.dt.year.ge(2026).any() or (dates > pd.Timestamp(LAST_DATE)).any():
        raise IntegrityError('a 2026 row or a date after 2025-09-30 reached EXP-BC-CAL-01; stop')
    return frame


def check_splits(frame):
    """Only the used splits, each inside its registered dates."""
    guard_dates(frame)
    if not frame.split.isin(USED_SPLITS).all():
        raise IntegrityError(f'only the splits {USED_SPLITS} may be loaded')
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    for split in USED_SPLITS:
        inside = frame.split.eq(split)
        start, end = (pd.Timestamp(d) for d in SPLIT_DATES[split])
        if not dates[inside].between(start, end).all():
            raise IntegrityError(f'{split} rows outside the registered dates {SPLIT_DATES[split]}')
    return frame


def assert_pre_pitch(columns):
    """Model-input columns must be registered pre-pitch features (B2 allowlist); else raise."""
    columns = list(columns)
    outside = [c for c in columns if c not in B2_FEATURES]
    forbidden = [c for c in columns if c in FORBIDDEN_INPUTS]
    if outside or forbidden or len(set(columns)) != len(columns):
        raise IntegrityError(f'non-registered or forbidden model inputs: {sorted(set(outside + forbidden))}')
    return columns


def label_free_view(frame):
    """The census input: pre-pitch columns only (no type label, description, event or physics)."""
    return frame.loc[:, list(CENSUS_COLUMNS)].copy()


# ---------------------------------------------------------------- populations

def label_index(pitch_type, vocabulary=ACTION_CODES):
    """Vocabulary index of each logged type; -1 for missing/out-of-vocabulary; raises on SEALED."""
    index = {code: i for i, code in enumerate(vocabulary)}
    values = list(pitch_type)
    if any(isinstance(v, str) and v == SEALED for v in values):
        raise SealedLabelError('a sealed DEV pitch-type label was read before its release')
    return np.array([index.get(v, -1) if isinstance(v, str) else -1 for v in values], dtype=np.int64)


def bc_rule(frame, no_pitch=AUTOMATIC):
    """policy_requests.bc_population_mask without the split condition: logged type label, not a
    registered no-pitch description, legal pre-pitch count (the same D-1 rule as BC_P)."""
    from .policy_requests import bc_population_mask
    part = frame.loc[:, ['pitch_type', 'description', 'balls', 'strikes']].assign(split='train')
    return np.asarray(bc_population_mask(part, frozenset(no_pitch)), dtype=bool)


def population_masks(frame, *, hands, repertoire, vocabulary=ACTION_CODES, no_pitch=AUTOMATIC):
    """P, S_in, S_off, P_plus and TRAIN0 (design section 6) over the rows of ``frame``.

    P_plus: BC rule + 18-code label + batter L/R. P: P_plus + pitcher in the TRAIN single-hand
    registry whose row hand equals the registered hand (policy_runtime._hand). S_in / S_off: P and
    the logged code inside / outside R_j. TRAIN0: P_plus rows whose pitcher has no TRAIN rows."""
    y = label_index(frame.pitch_type, vocabulary)
    rule = bc_rule(frame, no_pitch)
    side = frame.stand.isin(['L', 'R']).to_numpy()
    plus = rule & (y >= 0) & side
    pitchers = [str(int(p)) for p in frame.pitcher]
    registered = np.array([hands.get(p) in ('L', 'R') for p in pitchers], dtype=bool)
    agrees = np.array([normalize_hand(h) == hands.get(p) for h, p in zip(frame.p_throws, pitchers)], dtype=bool)
    p_mask = plus & registered & agrees
    ids = frame.pitcher.to_numpy(np.int64)
    support = repertoire.mask(ids)
    inside = support[np.arange(len(y)), np.maximum(y, 0)] & (y >= 0)
    known = repertoire.known(ids)
    return {'y': y, 'bc_rule': rule, 'P_plus': plus, 'P': p_mask, 'S_in': p_mask & inside, 'S_off': p_mask & ~inside,
            'TRAIN0': plus & ~known, 'out_of_vocabulary': rule & (y < 0), 'batter_side_unknown': rule & (y >= 0) & ~side,
            'P_plus_minus_P': plus & ~p_mask}


def population_counts(masks, frame):
    games = frame.game_pk.to_numpy()
    out = {}
    for name in ('bc_rule', 'P_plus', 'P', 'S_in', 'S_off', 'TRAIN0', 'P_plus_minus_P', 'out_of_vocabulary',
                 'batter_side_unknown'):
        mask = masks[name]
        out[name] = {'rows': int(mask.sum()), 'games': int(len(np.unique(games[mask])))}
    return out


def census_counts(view, *, hands, repertoire):
    """Label-free census (pre-pitch fields only); refuses any label/outcome column by construction."""
    present = [c for c in view.columns if c not in CENSUS_COLUMNS]
    if present:
        raise IntegrityError(f'census view carries non-pre-pitch columns {present}')
    guard_dates(view)
    legal = (view.balls.between(0, 3) & view.strikes.between(0, 2)).to_numpy()
    side = view.stand.isin(['L', 'R']).to_numpy()
    pitchers = [str(int(p)) for p in view.pitcher]
    registered = np.array([hands.get(p) in ('L', 'R') for p in pitchers], dtype=bool)
    agrees = np.array([normalize_hand(h) == hands.get(p) for h, p in zip(view.p_throws, pitchers)], dtype=bool)
    unknown = ~repertoire.known(view.pitcher.to_numpy(np.int64))
    out = {}
    for split in USED_SPLITS:
        inside = view.split.eq(split).to_numpy()
        part = view.loc[inside]
        out[split] = {'rows': int(inside.sum()), 'games': int(part.game_pk.nunique()),
                      'pas': int(part[['game_pk', 'at_bat_number']].drop_duplicates().shape[0]),
                      'legal_count': int((inside & legal).sum()), 'batter_lr': int((inside & side).sum()),
                      'registered_single_hand_pitcher': int((inside & registered).sum()),
                      'hand_agrees': int((inside & registered & agrees).sum()),
                      'pre_label_P_candidates': int((inside & legal & side & registered & agrees).sum()),
                      'pre_label_P_plus_candidates': int((inside & legal & side).sum()),
                      'pre_label_TRAIN0_candidates': int((inside & legal & side & unknown).sum())}
    return {'label_free': True, 'columns_read': list(CENSUS_COLUMNS), 'splits': out}


# ---------------------------------------------------------------- history (earlier rows of the PA only)

def pa_positions(frame):
    """0-based position of each row inside its PA; PAs contiguous, pitch numbers strictly increasing."""
    game, ab, pitch = (frame[c].to_numpy(np.int64) for c in KEY)
    n = len(frame)
    if not n:
        return np.zeros(0, dtype=np.int64)
    first = np.r_[True, (game[1:] != game[:-1]) | (ab[1:] != ab[:-1])]
    if len(set(zip(game[first].tolist(), ab[first].tolist()))) != int(first.sum()):
        raise IntegrityError('plate appearances must be contiguous in frame order')
    if np.any((~first[1:]) & (pitch[1:] <= pitch[:-1])):
        raise IntegrityError('pitch order inside each PA must be strictly increasing')
    positions = np.arange(n, dtype=np.int64)
    start = np.maximum.accumulate(np.where(first, positions, 0))
    return positions - start


def history_features(frame, vocabulary=ACTION_CODES):
    """prev_type_1..5, prev_outcome_1..5, pa_pitch_index and the B0 ``previous`` key, from strictly
    earlier rows of the same PA (the current row's own type/description/events are never read for
    itself). Codes: types 0..17, UNKNOWN 18 (missing / out of vocabulary), NONE 19; outcomes 0..9
    (model.outcome_labels), unknown 10, NONE 11. ``previous`` = '<START>' or the raw earlier label
    (fit_bc: '<UNKNOWN>' when missing)."""
    from .model import outcome_labels
    values = frame.pitch_type.tolist()
    if any(isinstance(v, str) and v == SEALED for v in values):
        raise SealedLabelError('a sealed DEV pitch-type label was read before its release')
    pos = pa_positions(frame)
    index = {code: i for i, code in enumerate(vocabulary)}
    missing = pd.isna(pd.Series(values, dtype=object)).to_numpy()
    types = np.array([TYPE_UNKNOWN if m else index.get(v, TYPE_UNKNOWN) for v, m in zip(values, missing)],
                     dtype=np.int64)
    raw = np.array([UNKNOWN if m else str(v) for v, m in zip(values, missing)], dtype=object)
    outcomes = outcome_labels(frame)
    outcomes = np.where(outcomes >= 0, outcomes, OUTCOME_UNKNOWN).astype(np.int64)
    n = len(frame)
    out = {'pa_pitch_index': pos}
    for k in range(1, HISTORY + 1):
        valid = pos >= k
        source = np.maximum(np.arange(n) - k, 0)
        out[f'prev_type_{k}'] = np.where(valid, types[source], TYPE_NONE)
        out[f'prev_outcome_{k}'] = np.where(valid, outcomes[source], OUTCOME_NONE)
    out['previous'] = np.where(pos >= 1, raw[np.maximum(np.arange(n) - 1, 0)], START)
    return pd.DataFrame(out, index=frame.index)


# ---------------------------------------------------------------- game state, hands, workload

def game_state_features(frame):
    outs = frame.outs_when_up.to_numpy(np.int64)
    bases = frame.bases.to_numpy(np.int64)
    if ((outs < 0) | (outs > 2)).any() or ((bases < 0) | (bases > 7)).any():
        raise IntegrityError('outs must be 0..2 and bases a 3-bit code before a pitch')
    top = frame.inning_topbot.eq('Top').to_numpy()
    home, away = frame.home_score.to_numpy(np.int64), frame.away_score.to_numpy(np.int64)
    runners = np.where(bases == 0, 0, np.where(bases == 1, 1, 2))  # empty / 1B only / RISP
    return pd.DataFrame({'outs': outs, 'first_base': bases & 1, 'second_base': (bases >> 1) & 1,
                         'third_base': (bases >> 2) & 1, 'inning': np.minimum(frame.inning.to_numpy(np.int64), 10),
                         'defensive_score_lead': np.where(top, home - away, away - home),
                         'defender_is_home': top.astype(np.int64), 'base_out': outs * 3 + runners},
                        index=frame.index)


def hand_features(frame):
    stand, throws = frame.stand.astype(object), frame.p_throws.astype(object)
    return pd.DataFrame({'batter_left': stand.eq('L').to_numpy(np.int64),
                         'pitcher_left': throws.eq('L').to_numpy(np.int64),
                         'same_hand': (stand.eq(throws) & stand.isin(['L', 'R'])).to_numpy(np.int64)},
                        index=frame.index)


def rebuild_workload(raw):
    """workload_features.rebuild (branch research/direct-head-confirm-1026, :33-45), counts only:
    earlier Statcast records of the pitcher in this game and the rebuilt times through the order."""
    if set(WORKLOAD_FORBIDDEN).intersection(raw.columns):
        raise IntegrityError('future-looking workload columns must not be loaded')
    if raw[list(KEY)].isna().any().any() or raw.duplicated(list(KEY)).any():
        raise IntegrityError('workload pitch keys must be complete and unique')
    f = raw.copy()
    f['game_date'] = pd.to_datetime(f.game_date).dt.normalize()
    if f.groupby('game_pk').game_date.nunique().gt(1).any():
        raise IntegrityError('a game_pk spans several dates')
    f = f.sort_values(['game_date', *KEY], kind='stable')
    prior = f.groupby(['game_pk', 'pitcher'], sort=False).cumcount()
    new_pa = ~f.duplicated(['game_pk', 'pitcher', 'at_bat_number'])
    faced_before = new_pa.astype(np.int64).groupby([f.game_pk, f.pitcher], sort=False).cumsum() - 1
    out = pd.DataFrame({'prior_pitches_game': prior.astype(np.int64),
                        'rebuilt_thruorder': (faced_before // 9 + 1).astype(np.int64)}, index=f.index)
    return out.reindex(raw.index)


def workload_features(frame, raw):
    """prior_pitches_game, times_through_order (rebuilt), tto_group (1/2/3+) and is_game_starter
    (pitcher = first pitcher of (game_pk, inning_topbot) in frame order) aligned to ``frame``."""
    if list(raw.columns) != WORKLOAD_RAW_COLUMNS:
        raise IntegrityError('workload source must contain only keys, date and pitcher')
    built = pd.concat([raw, rebuild_workload(raw)], axis=1)
    merged = frame[[*KEY, 'game_date', 'pitcher']].reset_index(drop=True).merge(
        built, on=list(KEY), how='left', validate='one_to_one', suffixes=('', '_raw'))
    if merged.prior_pitches_game.isna().any():
        raise IntegrityError('frame rows without an exact raw Statcast key')
    same_date = pd.to_datetime(merged.game_date).dt.normalize().eq(pd.to_datetime(merged.game_date_raw).dt.normalize())
    if not (same_date.all() and merged.pitcher.astype(np.int64).eq(merged.pitcher_raw.astype(np.int64)).all()):
        raise IntegrityError('frame and raw records disagree on date or pitcher')
    first = frame.groupby(['game_pk', 'inning_topbot'], sort=False).pitcher.transform('first')
    thru = merged.rebuilt_thruorder.to_numpy(np.int64)
    return pd.DataFrame({'prior_pitches_game': merged.prior_pitches_game.to_numpy(np.int64),
                         'times_through_order': thru, 'tto_group': np.minimum(thru, 3),
                         'is_game_starter': frame.pitcher.eq(first).to_numpy(np.int64)}, index=frame.index)


# ---------------------------------------------------------------- A3 batter-style inputs (section 5 E)

def style_input_rows(frame):
    """The registered input filter: 2023-05-15 <= game_date <= 2025-05-31 and split in TRAIN /
    earlystop / temperature (no June, DEV, 2023-03-30..05-14 or 2026 row; a 2026 row raises)."""
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    if dates.isna().any() or dates.dt.year.ge(2026).any():
        raise IntegrityError('2026 rows never enter the A3 batter-style inputs')
    start, end = (pd.Timestamp(STYLE_INPUT_FILTER[k]) for k in ('start', 'end'))
    keep = dates.between(start, end) & frame.split.isin(STYLE_INPUT_FILTER['splits'])
    columns = [c for c in STYLE_SOURCE_COLUMNS if c in frame.columns]
    return frame.loc[keep, columns]


@dataclass
class StyleTables:
    rolling: pd.DataFrame  # HISTORY_COLUMNS for every filtered row (index = frame index)
    snapshots: dict  # split -> snapshot frame (batter id string -> values, 'league' last)
    input_rows: int
    input_dates: list


def style_tables(frame):
    from .policy_requests import style_snapshot
    inputs = style_input_rows(frame)
    if not len(inputs):
        raise IntegrityError('no A3 batter-style input rows')
    rolling = add_batter_style_history(inputs.copy())[list(HISTORY_COLUMNS)].astype(np.float32)
    snapshots = {split: style_snapshot(inputs, as_of) for split, as_of in STYLE_AS_OF.items()}
    dates = pd.to_datetime(inputs.game_date).dt.normalize()
    return StyleTables(rolling, snapshots, int(len(inputs)), [str(dates.min().date()), str(dates.max().date())])


def style_values(rows, split, style):
    """Style priors of ``rows`` (one split): TRAIN rolling values, else the split's snapshot."""
    if split == 'train':
        missing = ~rows.index.isin(style.rolling.index)
        if missing.any():
            raise IntegrityError('TRAIN rows without a strictly-prior-date A3 style value')
        return style.rolling.loc[rows.index], {'source': 'rolling_strictly_prior_date'}
    from .policy_requests import apply_style_snapshot
    as_of = STYLE_AS_OF[split]
    applied, report = apply_style_snapshot(rows[['game_date', 'batter', 'game_pk', 'at_bat_number']],
                                           style.snapshots[split], as_of)
    return applied[list(HISTORY_COLUMNS)], {'source': f'snapshot_as_of_{as_of}', **report}


def batter_features(style_frame, archetypes):
    """The 17 frozen-G0 batter channels (z-scored rates, reliabilities, soft memberships) and the
    B1 batter cluster (argmax membership; 5 = unknown when the mean reliability is 0)."""
    numeric = np.asarray(archetypes.numeric_features(style_frame), dtype=np.float32)
    if numeric.shape != (len(style_frame), 17):
        raise IntegrityError('frozen archetypes must give 17 batter channels')
    confidence = style_frame[list(RELIABILITY_COLUMNS)].to_numpy(np.float64).mean(axis=1)
    cluster = np.where(confidence > 0, numeric[:, 12:17].argmax(axis=1), 5).astype(np.int64)
    names = ([f'style_rate_{n}' for n in ('contact', 'swing', 'walk', 'strikeout', 'isolated_power', 'groundball')]
             + [f'style_reliability_{n}' for n in ('contact', 'swing', 'walk', 'strikeout', 'isolated_power',
                                                    'groundball')] + [f'membership_{k}' for k in range(5)])
    out = pd.DataFrame(numeric, columns=names, index=style_frame.index)
    out['batter_cluster'] = cluster
    return out


# ---------------------------------------------------------------- OOF B0 encodings and tables

def oof_b0(keys, actions, games, *, salt, k, prior_strength, minimum_action_count, vocabulary=ACTION_CODES):
    """TRAIN rows: B0-recipe probabilities from VectorBC fits on the other k-1 TRAIN game folds."""
    folds = game_fold(games, salt, k)
    actions = np.asarray(actions, dtype=np.int64)
    encoded = np.zeros((len(actions), len(vocabulary)))
    models = []
    for f in range(k):
        fit, rows = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        model = VectorBC.fit(keys.take(fit), actions[fit], vocabulary, prior_strength, minimum_action_count,
                             {'fold': f, 'salt': salt, 'fit_rows': int(len(fit))})
        if len(rows):
            encoded[rows] = model.batch_keys(keys.take(rows))
        models.append(model)
    return encoded, folds, models


@dataclass
class FeatureContext:
    vocabulary: tuple
    archetypes: object
    b0: VectorBC  # registered BC_P recipe (encoding for every non-TRAIN row)
    train_volume: dict  # pitcher id (int) -> all TRAIN rows of the pitcher
    volume_edges: tuple  # (q25, q75) of the TRAIN volumes
    style: StyleTables
    workload: pd.DataFrame  # aligned to the loaded frame index
    hands: dict
    repertoire: object
    no_pitch: frozenset = AUTOMATIC


def volume_group(pitchers, train_volume, edges):
    q25, q75 = edges
    out = []
    for p in pitchers:
        n = train_volume.get(int(p), 0)
        out.append('zero' if n == 0 else 'low' if n <= q25 else 'middle' if n <= q75 else 'high')
    return np.array(out, dtype=object)


def split_table(frame, split, ctx, *, oof=None):
    """The behavior table of one split's P_plus rows: B0/B1 keys, the 63 B2 features, labels,
    populations and descriptive strata. TRAIN needs ``oof`` = {'salt', 'k', 'prior_strength',
    'minimum_action_count'}: its B0 features are out-of-fold (VectorBC on the other TRAIN game
    folds); every other split uses the registered BC_P recipe ``ctx.b0``. Returns (table, info,
    masks, oof_models)."""
    if not frame.split.eq(split).all():
        raise IntegrityError('split_table takes the rows of exactly one split')
    if (split == 'train') != (oof is not None):
        raise IntegrityError('TRAIN rows need out-of-fold B0 encodings; other splits must not have them')
    history = history_features(frame, ctx.vocabulary)
    masks = population_masks(frame, hands=ctx.hands, repertoire=ctx.repertoire, vocabulary=ctx.vocabulary,
                             no_pitch=ctx.no_pitch)
    keep = masks['P_plus']
    rows = frame.loc[keep]
    hist = history.loc[keep]
    state = game_state_features(rows)
    hands = hand_features(rows)
    style, style_report = style_values(rows, split, ctx.style)
    batter = batter_features(rows[[]].join(style), ctx.archetypes)
    work = ctx.workload.loc[rows.index]
    table = pd.DataFrame({'game_pk': rows.game_pk.to_numpy(np.int64), 'at_bat_number': rows.at_bat_number.to_numpy(np.int64),
                          'pitch_number': rows.pitch_number.to_numpy(np.int64), 'split': split,
                          'game_date': pd.to_datetime(rows.game_date).dt.normalize().to_numpy(),
                          'pitcher': rows.pitcher.to_numpy(np.int64), 'batter_side': rows.stand.astype(object).to_numpy(),
                          'balls': rows.balls.to_numpy(np.int64), 'strikes': rows.strikes.to_numpy(np.int64),
                          'previous': hist.previous.to_numpy(object), 'base_out': state.base_out.to_numpy(),
                          'tto_group': work.tto_group.to_numpy(), 'batter_cluster': batter.batter_cluster.to_numpy(),
                          'label': masks['y'][keep]})
    keys = Keys.from_frame(table)
    models, folds = None, None
    if oof is not None:
        b0, folds, models = oof_b0(keys, table.label.to_numpy(), table.game_pk.to_numpy(), salt=oof['salt'],
                                   k=oof['k'], prior_strength=oof['prior_strength'],
                                   minimum_action_count=oof['minimum_action_count'], vocabulary=ctx.vocabulary)
        table['oof_fold'] = folds
    else:
        b0 = ctx.b0.batch_keys(keys)
    for i, code in enumerate(ctx.vocabulary):
        table[f'b0_p_{code}'] = b0[:, i].astype(np.float32)  # model inputs (B0 scoring uses VectorBC itself)
    table['pitcher_train_log_volume'] = np.log1p([ctx.train_volume.get(int(p), 0) for p in table.pitcher])
    table['count_id'] = table.balls * 3 + table.strikes
    for name in ['pa_pitch_index', *(f'prev_type_{k}' for k in range(1, 6)), *(f'prev_outcome_{k}' for k in range(1, 6))]:
        table[name] = hist[name].to_numpy()
    for name in ('outs', 'first_base', 'second_base', 'third_base', 'inning', 'defensive_score_lead', 'defender_is_home'):
        table[name] = state[name].to_numpy()
    for name in ('batter_left', 'pitcher_left', 'same_hand'):
        table[name] = hands[name].to_numpy()
    for name in batter.columns.drop('batter_cluster'):
        table[name] = batter[name].to_numpy()
    for name in ('prior_pitches_game', 'times_through_order', 'is_game_starter'):
        table[name] = work[name].to_numpy()
    for name in ('P', 'S_in', 'S_off', 'TRAIN0'):
        table[name] = masks[name][keep]
    table['volume_group'] = volume_group(table.pitcher, ctx.train_volume, ctx.volume_edges)
    table['count_label'] = [f'{b}-{s}' for b, s in zip(table.balls, table.strikes)]
    table['role'] = np.where(table.is_game_starter.eq(1), 'SP', 'RP')
    table['month'] = pd.to_datetime(table.game_date).dt.month.to_numpy()
    table['two_strikes'] = table.strikes.eq(2).to_numpy()
    table['runners_on'] = (table.first_base + table.second_base + table.third_base).gt(0).to_numpy()
    table['platoon'] = np.where(table.same_hand.eq(1), 'same', 'opposite')
    assert_pre_pitch(B2_FEATURES)
    known_previous = {*ctx.vocabulary, START, UNKNOWN}
    info = {'split': split, 'rows': int(len(frame)), 'P_plus_rows': int(keep.sum()), 'style': style_report,
            'previous_out_of_vocabulary': int(sum(1 for v in table.previous if v not in known_previous)),
            'counts': population_counts(masks, frame)}
    return BehaviorTable(table), info, masks, models
