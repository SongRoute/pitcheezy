"""Synthetic TRAIN-only cohort and evaluation metadata contracts."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.matrix_panel import (select_panel, evaluation_metadata, verify_starters,
                                   group_reporting_status, guardrail_protocol)


def training():
    eligible, source = [], []
    for pid in range(1, 65):
        role = 'starter' if pid % 2 else 'relief'
        starter = pid if role == 'starter' else 10000 + pid
        count = [2, 4, 8, 12][(pid // 4) % 4]
        common = {'game_date': pd.Timestamp('2024-04-01'), 'game_type': 'R', 'split': 'train',
                  'game_pk': pid, 'inning': 1, 'inning_topbot': 'Top', 'outs_when_up': 0,
                  'starter_pitcher': starter, 'p_throws': 'L' if pid % 4 < 2 else 'R', 'batter': 999,
                  'strikes': 0, 'bases': 0}
        if role == 'relief':
            source.append({**common, 'pitcher': starter, 'at_bat_number': 1, 'pitch_number': 1})
        for j in range(count):
            row = {**common, 'pitcher': pid, 'at_bat_number': 1 if role == 'starter' else 2, 'pitch_number': j + 1}
            eligible.append(row)
            source.append(row)
    return pd.DataFrame(eligible), pd.DataFrame(source)


def test_panel_deterministic_by_player_hash_not_input_order_or_outcomes():
    train, source = training()
    panel = select_panel(train, starter_source=source, c6_ids=[1, 2])
    shuffled = train.sample(frac=1, random_state=9)
    shuffled['description'] = 'home_run'
    other = select_panel(shuffled, starter_source=source.sample(frac=1, random_state=8))
    assert panel['pitcher_ids'] == other['pitcher_ids']
    assert panel['strata'] == other['strata']
    assert panel['volume_thresholds'] == other['volume_thresholds']
    assert panel['panel_size'] <= 48
    assert all(row['selected_players'] <= 4 for row in panel['strata'])
    assert panel['starter_validation']['recorded_starter_column_verified']
    assert panel['canonical_train_keys_sha256'] == other['canonical_train_keys_sha256']


def test_starter_source_is_required_and_verified():
    train, source = training()
    with pytest.raises(ValueError, match='Unfiltered TRAIN'):
        select_panel(train)
    changed = train.copy()
    changed.loc[0, 'starter_pitcher'] = 77777
    with pytest.raises(ValueError, match='first defensive'):
        verify_starters(changed, source)
    invalid = source.copy()
    invalid['inning'] = 2
    with pytest.raises(ValueError, match='Incomplete'):
        verify_starters(train, invalid)


def test_ties_and_empty_strata_are_retained_without_replacement():
    train, source = training()
    train = train.loc[train.pitcher.eq(1)].copy()
    panel = select_panel(train, starter_source=source)
    assert panel['panel_size'] == 1
    assert panel['volume_thresholds']['q25'] == panel['volume_thresholds']['q75'] == 2
    assert panel['train_players'][0]['train_volume'] == 'low'
    assert sum(row['status'] == 'empty' for row in panel['strata']) == 11
    assert panel['strata'][0]['status'] == 'underfilled'


def test_dev_rows_never_select_panel_and_ambiguous_hand_recorded():
    train, source = training()
    bad = train.copy()
    bad.loc[0, 'split'] = 'dev'
    with pytest.raises(ValueError, match='TRAIN'):
        select_panel(bad, starter_source=source)
    train.loc[train.pitcher.eq(1), 'p_throws'] = None
    panel = select_panel(train, starter_source=source)
    assert 1 not in panel['pitcher_ids']
    assert 1 in panel['excluded_hand_pitcher_ids']


def test_evaluation_groups_preserve_requested_denominator_and_novelty():
    train, source = training()
    panel = select_panel(train, starter_source=source)
    requested = train.iloc[[0, 1]].copy()
    requested['game_date'] = pd.Timestamp('2025-07-01')
    requested.iloc[1, requested.columns.get_loc('pitcher')] = 55555
    requested.iloc[1, requested.columns.get_loc('batter')] = 55555
    requested.iloc[1, requested.columns.get_loc('bases')] = 1
    requested.iloc[1, requested.columns.get_loc('strikes')] = 2
    metadata = evaluation_metadata(requested, panel)
    assert len(metadata) == len(requested)
    assert metadata['seen_pitcher'].tolist() == [True, False]
    assert metadata['seen_batter'].tolist() == [True, False]
    assert metadata.train_volume.tolist()[1] == 'zero'
    assert metadata.game_role.tolist() == ['starter', 'relief']
    assert metadata.runners_on.tolist() == [False, True]
    assert metadata.two_strikes.tolist() == [False, True]
    assert metadata.month.tolist() == ['2025-07', '2025-07']


def test_reporting_counts_and_fixed_simultaneous_guardrail_family():
    games = np.repeat(np.arange(30), 20)
    mask = np.ones(600, dtype=bool)
    assert group_reporting_status(games, mask)['status'] == 'reporting_eligible'
    assert group_reporting_status(games, mask, event_count=29)['status'] == 'unmeasured_for_guardrail'
    assert group_reporting_status(games, np.zeros(600, dtype=bool))['games'] == 0
    protocol = guardrail_protocol(['starter', 'relief', 'L', 'R'])
    assert protocol['family_size'] == 8
    assert protocol['per_bound_alpha'] == .05 / 8
    with pytest.raises(ValueError):
        guardrail_protocol(['starter', 'starter'])


def test_missing_state_groups_remain_unknown():
    train, source = training()
    panel = select_panel(train, starter_source=source)
    requested = train.iloc[[0]].copy()
    requested['strikes'] = np.nan
    requested['bases'] = np.nan
    metadata = evaluation_metadata(requested, panel)
    assert pd.isna(metadata.two_strikes.iloc[0])
    assert pd.isna(metadata.runners_on.iloc[0])
