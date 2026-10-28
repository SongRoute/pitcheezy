"""EXP-BC-CAL-01 features and populations on synthetic data: leakage perturbation (T2), current
pitch blocked (T3), OOF encodings (T4), folds (T5), 2026 guard (T11), style input filter (T14),
label-free census, sealed DEV labels. No real data."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from behavior_calibration_synthetic import PHYSICS, make_context, make_world
from pitchmdp.archetypes import HISTORY_COLUMNS
from pitchmdp.behavior_calibration_registry import ACTION_CODES, B2_FEATURES
from pitchmdp.behavior_features import (CENSUS_COLUMNS, SEALED, SealedLabelError, assert_pre_pitch, census_counts,
                                        check_splits, game_fold, guard_dates, history_features, label_free_view,
                                        population_masks, split_table, style_input_rows, style_tables,
                                        workload_features)
from pitchmdp.behavior_model import BackoffBC, pitch_key
from pitchmdp.policy_artifacts import IntegrityError

OOF = {'salt': 'EXP-BC-CAL-01/te', 'k': 5, 'prior_strength': 20., 'minimum_action_count': 1}
KEY = ['game_pk', 'at_bat_number', 'pitch_number']


@pytest.fixture(scope='module')
def world():
    return make_world(0)


def table_of(frame, split, world):
    ctx = make_context(frame, world)
    part = frame.loc[frame.split.eq(split)]
    return split_table(part, split, ctx, oof=OOF if split == 'train' else None)[0]


def features_at(table, key, b1):
    position = int(table.positions([key])[0])
    assert position >= 0
    row = table.frame.iloc[position]
    values = {c: row[c] for c in [*B2_FEATURES, 'previous', 'base_out', 'tto_group', 'batter_cluster', 'pitcher',
                                  'balls', 'strikes', 'batter_side']}
    values['b1'] = tuple(b1.batch(table, [position])[0])
    values['b0'] = tuple(table.frame.iloc[[position]][[f'b0_p_{c}' for c in ACTION_CODES]].to_numpy()[0])
    return values


def pick_row(world, split):
    """A row in the middle of a PA (>= 2 earlier pitches) whose game continues afterwards."""
    frame = world['frame']
    part = frame.loc[frame.split.eq(split)]
    candidates = part.loc[part.pitch_number.ge(3)]
    for index in candidates.index:
        row = frame.loc[index]
        later = frame.index[(frame.game_pk == row.game_pk) & (frame.index > index)]
        if row.pitch_type in ACTION_CODES and row.description not in ('automatic_ball',) and len(later) > 5:
            return index, later
    raise AssertionError('no suitable row')


def perturb(frame, index, later, rng, own=True):
    out = frame.copy()
    if own:
        current = out.at[index, 'pitch_type']
        out.at[index, 'pitch_type'] = next(c for c in ('FF', 'SL', 'CH') if c != current)
        out.loc[index, list(PHYSICS)] = rng.normal(size=len(PHYSICS)) * 10
        out.at[index, 'description'] = 'swinging_strike' if out.at[index, 'description'] != 'swinging_strike' else 'ball'
        out.at[index, 'launch_angle'] = 77.
    for j in later:
        if out.at[j, 'description'] != 'automatic_ball':
            out.at[j, 'pitch_type'] = str(rng.choice(['CU', 'FC', 'SI']))
        out.at[j, 'description'] = str(rng.choice(['ball', 'foul', 'called_strike']))
        out.at[j, 'events'] = None
        out.loc[j, list(PHYSICS)] = rng.normal(size=len(PHYSICS))
    return out


@pytest.mark.parametrize('split', ['dev', 'train'])
def test_t2_t3_own_and_later_rows_never_change_a_rows_features(world, split):
    rng = np.random.default_rng(11)
    frame = world['frame']
    index, later = pick_row(world, split)
    key = int(pitch_key(*(frame.loc[[index], c] for c in KEY))[0])
    train = world['tables']['train']
    b1 = BackoffBC.fit(train.keys, train.frame.label.to_numpy(), repertoire=world['b0'].repertoire,
                       alphas={'pitcher': 0., 'batter_side': 2., 'count': 5., 'previous_type': 20., 'base_out': 10.,
                               'times_through_order': 50., 'batter_cluster': 100.})
    before = features_at(table_of(frame, split, world), key, b1)
    after = features_at(table_of(perturb(frame, index, later, rng), split, world), key, b1)
    assert before == after
    current_only = features_at(table_of(perturb(frame, index, [], rng), split, world), key, b1)  # T3
    assert before == current_only
    # sanity: the earlier row of the same PA does enter (prev_type_1 / previous)
    earlier = frame.copy()
    previous = index - 1
    earlier.at[previous, 'pitch_type'] = next(c for c in ('KN', 'EP') if c != earlier.at[previous, 'pitch_type'])
    changed = features_at(table_of(earlier, split, world), key, b1)
    assert changed['previous'] != before['previous'] and changed['prev_type_1'] != before['prev_type_1']


def test_t4_oof_encoding_ignores_the_rows_own_label(world):
    frame = world['frame']
    train = frame.loc[frame.split.eq('train')]
    index = train.index[train.pitch_type.isin(['FF', 'SL']) & ~train.description.eq('automatic_ball')][40]
    key = int(pitch_key(*(frame.loc[[index], c] for c in KEY))[0])
    columns = [f'b0_p_{c}' for c in ACTION_CODES]
    before = table_of(frame, 'train', world)
    changed = frame.copy()
    changed.at[index, 'pitch_type'] = 'CH' if frame.at[index, 'pitch_type'] != 'CH' else 'FF'
    after = table_of(changed, 'train', world)
    own = int(before.positions([key])[0])
    assert np.array_equal(before.frame.iloc[own][columns].to_numpy(float), after.frame.iloc[own][columns].to_numpy(float))
    pitcher = before.frame.pitcher.iloc[own]
    others = np.flatnonzero((before.frame.pitcher == pitcher).to_numpy() & (before.frame.oof_fold != before.frame.oof_fold.iloc[own]).to_numpy())
    assert len(others) and not np.array_equal(before.frame.iloc[others][columns].to_numpy(float),
                                              after.frame.iloc[others][columns].to_numpy(float))


def test_t5_game_fold_rule():
    games = np.array([700001, 700001, 700002, 700003, 700002, 700010])
    folds = game_fold(games, 'EXP-BC-CAL-01/xf', 5)
    assert np.array_equal(folds, game_fold(games, 'EXP-BC-CAL-01/xf', 5))
    assert folds[0] == folds[1] and folds[2] == folds[4]
    expected = [int(hashlib.sha256(f'EXP-BC-CAL-01/xf:{g}'.encode()).hexdigest(), 16) % 5 for g in games]
    assert folds.tolist() == expected
    many = np.arange(700000, 700200)
    assert not np.array_equal(game_fold(many, 'EXP-BC-CAL-01/xf', 5), game_fold(many, 'EXP-BC-CAL-01/te', 5))
    assert set(game_fold(many, 'EXP-BC-CAL-01/te', 5).tolist()) == set(range(5))


def test_t11_2026_and_post_september_rows_raise(world):
    frame = world['frame'].head(50).copy()
    guard_dates(frame)
    for date in ('2026-04-01', '2025-10-01'):
        bad = frame.copy()
        bad.loc[bad.index[0], 'game_date'] = pd.Timestamp(date)
        with pytest.raises(IntegrityError):
            guard_dates(bad)
        with pytest.raises(IntegrityError):
            check_splits(bad)
    june = frame.copy()
    june['split'] = 'blend'
    with pytest.raises(IntegrityError):
        check_splits(june)
    with pytest.raises(IntegrityError):
        style_input_rows(frame.assign(game_date=pd.Timestamp('2026-04-01')))


def test_t14_style_inputs_exclude_june_dev_early_2023_and_raise_on_2026(world):
    frame = world['frame']
    train_game = frame.loc[frame.split.eq('train')].game_pk.iloc[0]
    june = frame.loc[frame.game_pk.eq(train_game)].assign(game_date=pd.Timestamp('2025-06-10'), split='blend',
                                                          game_pk=880001)
    early = frame.loc[frame.game_pk.eq(train_game)].assign(game_date=pd.Timestamp('2023-04-20'), split='unused',
                                                           game_pk=880002)
    mixed = pd.concat([frame, june, early]).sort_values(['game_date', *KEY], kind='stable').reset_index(drop=True)
    inputs = style_input_rows(mixed)
    kept = mixed.loc[inputs.index]
    assert set(kept.split) == {'train', 'earlystop', 'temperature'}
    assert kept.game_date.min() >= pd.Timestamp('2023-05-15') and kept.game_date.max() <= pd.Timestamp('2025-05-31')
    base = style_tables(mixed)
    scrambled = mixed.copy()
    excluded = scrambled.split.isin(['blend', 'unused', 'dev'])
    scrambled.loc[excluded, 'events'] = 'home_run'
    scrambled.loc[excluded, 'description'] = 'hit_into_play'
    scrambled.loc[excluded, 'is_pa_terminal'] = True
    again = style_tables(scrambled)
    for split in base.snapshots:
        pd.testing.assert_frame_equal(base.snapshots[split], again.snapshots[split])
    pd.testing.assert_frame_equal(base.rolling, again.rolling)
    assert list(base.rolling.columns) == list(HISTORY_COLUMNS)
    with pytest.raises(IntegrityError):
        style_input_rows(pd.concat([mixed, mixed.head(3).assign(game_date=pd.Timestamp('2026-04-02'))]))


def test_census_is_label_free_by_construction(world):
    frame = world['frame']
    view = label_free_view(frame)
    assert list(view.columns) == list(CENSUS_COLUMNS)
    kwargs = {'hands': world['hands'], 'repertoire': world['b0'].repertoire}
    counts = census_counts(view, **kwargs)
    scrambled = frame.copy()
    scrambled['pitch_type'] = 'UN'
    scrambled['description'] = 'automatic_strike'
    scrambled['events'] = 'home_run'
    assert census_counts(label_free_view(scrambled), **kwargs) == counts
    for column in ('pitch_type', 'description', 'events'):
        with pytest.raises(IntegrityError, match='non-pre-pitch'):
            census_counts(view.assign(**{column: frame[column]}), **kwargs)
    assert counts['label_free'] is True and counts['splits']['dev']['pre_label_TRAIN0_candidates'] > 0


def test_sealed_dev_labels_raise_before_release(world):
    frame = world['frame'].copy()
    dev = frame.split.eq('dev')
    frame['pitch_type'] = frame.pitch_type.astype(object)
    frame.loc[dev, 'pitch_type'] = SEALED
    part = frame.loc[dev]
    with pytest.raises(SealedLabelError):
        history_features(part)
    with pytest.raises(SealedLabelError):
        population_masks(part, hands=world['hands'], repertoire=world['b0'].repertoire)
    with pytest.raises(SealedLabelError):
        split_table(part, 'dev', world['ctx'])
    # TRAIN rows never read DEV rows: still buildable while DEV is sealed
    split_table(frame.loc[frame.split.eq('earlystop')], 'earlystop', world['ctx'])


def test_population_partition(world):
    frame = world['frame']
    dev = frame.loc[frame.split.eq('dev')]
    masks = population_masks(dev, hands=world['hands'], repertoire=world['b0'].repertoire)
    assert not (masks['S_in'] & masks['S_off']).any() and np.array_equal(masks['S_in'] | masks['S_off'], masks['P'])
    assert (masks['P'] <= masks['P_plus']).all() and not (masks['TRAIN0'] & masks['P']).any()
    assert masks['S_off'].sum() > 0 and masks['TRAIN0'].sum() > 0
    ambiguous = [p for p, h in world['hands'].items() if h == 'AMBIGUOUS']
    rows = dev.pitcher.astype(str).isin(ambiguous).to_numpy()
    assert rows.any() and not masks['P'][rows].any() and masks['P_plus'][rows].any()
    automatic = dev.description.eq('automatic_ball').to_numpy()
    assert not masks['P_plus'][automatic].any()


def test_assert_pre_pitch_allowlist():
    assert assert_pre_pitch(B2_FEATURES) == list(B2_FEATURES)
    for bad in (['plate_x'], ['description'], ['pitcher'], ['game_date'], ['release_speed'], ['foo'],
                ['balls', 'balls']):
        with pytest.raises(IntegrityError):
            assert_pre_pitch(bad)


def test_history_and_workload_match_brute_force(world):
    frame = world['frame']
    part = frame.loc[frame.split.eq('temperature')]
    history = history_features(part)
    rows = part.reset_index()
    for i in range(0, len(rows), 7):
        row = rows.iloc[i]
        same = rows.loc[(rows.game_pk == row.game_pk) & (rows.at_bat_number == row.at_bat_number)
                        & (rows.pitch_number < row.pitch_number)]
        assert history.pa_pitch_index.iloc[i] == len(same)
        if len(same):
            previous = same.iloc[-1].pitch_type
            assert history.previous.iloc[i] == (previous if isinstance(previous, str) else '<UNKNOWN>')
        else:
            assert history.previous.iloc[i] == '<START>' and history.prev_type_1.iloc[i] == len(ACTION_CODES) + 1
    work = workload_features(frame, frame[['game_date', *KEY, 'pitcher']])
    ordered = frame.reset_index()
    for i in range(0, len(ordered), 53):
        row = ordered.iloc[i]
        earlier = ordered.loc[(ordered.game_pk == row.game_pk) & (ordered.pitcher == row.pitcher)
                              & ((ordered.at_bat_number < row.at_bat_number)
                                 | ((ordered.at_bat_number == row.at_bat_number)
                                    & (ordered.pitch_number < row.pitch_number)))]
        faced = earlier.loc[earlier.at_bat_number < row.at_bat_number].at_bat_number.nunique()
        assert work.prior_pitches_game.iloc[i] == len(earlier)
        assert work.times_through_order.iloc[i] == faced // 9 + 1
    with pytest.raises(IntegrityError):
        workload_features(frame, frame[['game_date', *KEY, 'pitcher']].assign(pitcher_days_until_next_game=1))
