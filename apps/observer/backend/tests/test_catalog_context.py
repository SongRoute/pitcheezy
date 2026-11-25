"""Small-frame v2 context/catalogue tests; no real data, downloads or model inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2]/'scripts')]
from observer_app.context import ObservedContext, context_notes
from observer_app.player_metadata import load_player_names, player_label
from build_catalog import build_catalog, pa_exclusions


def sample_frame():
    rows = []
    for game, date, pa_count in [(1, '2025-08-01', 4), (2, '2025-08-16', 11),
                                  (3, '2025-08-20', 2), (4, '2025-08-24', 2), (5, '2025-08-28', 2)]:
        for pa in range(1, pa_count+1):
            for pitch in (1, 2):
                rows.append(dict(game_pk=game, game_date=date, at_bat_number=pa, pitch_number=pitch,
                    pitcher=11, batter=500+(pa % 3), pitch_type='FF' if pitch == 1 else 'SL',
                    release_speed=90.+pa+pitch, supported_pa=not (game == 2 and pa == 3),
                    is_pa_terminal=pitch == 2, events='field_out' if pitch == 2 else None,
                    description='hit_into_play' if pitch == 2 else 'called_strike', launch_angle=15.,
                    balls=0, strikes=pitch-1, inning=1, inning_topbot='Top', outs_when_up=0, bases=0,
                    home_score=0, away_score=0, next_inning=1, next_half='Top', next_outs=1,
                    next_bases=0, next_home_score=0, next_away_score=0, stand='R',
                    sz_bot=1.5, sz_top=3.5, plate_x=.2, plate_z=2., home_team='SF', away_team='TB'))
    return pd.DataFrame(rows)


def build(frame):
    return build_catalog(frame, {'11': {'pitch_types': ['FF', 'SL']}},
                         {'11': 'Pitcher Name', '501': 'Batter Name'}, minimum_repertoire=1)


def test_all_supported_pas_and_chronological_per_pitcher_cap():
    data, coverage = build(sample_frame().sample(frac=1, random_state=4))
    assert [game['id'] for game in data['games']] == [2, 3, 4]
    assert len(data['games'][0]['plate_appearances']) == 10  # original eight-PA cap removed
    assert coverage['included_pa_total'] == 14
    assert coverage['excluded_pa_total'] == 1
    assert coverage['support_rate'] == 14/15
    assert coverage['excluded_pas'][0]['reasons'] == ['unsupported_transition']
    assert coverage['per_pitcher'][0]['window_games'] == 4
    assert coverage['per_pitcher'][0]['selected_games'] == 3
    first = data['games'][0]['plate_appearances'][0]
    assert first['pitcher_id'] == 11 and first['batter_id'] == 501
    assert first['pitcher_label'] == 'Pitcher Name' and first['batter_label'] == 'Batter Name'
    assert first['pitches'][0]['request']['batter_profile']['as_of'] == '2025-08-15'
    assert 'context' not in first['pitches'][0]['request']


def test_future_actual_and_same_date_changes_do_not_change_pre_pitch_request_or_context():
    original = sample_frame()
    first, _ = build(original)
    changed = original.copy()
    # The target is game2 PA4 pitch2. Mutate current and all future realized physics/results,
    # retaining structural support flags, current count and prior observed records.
    future = changed.game_pk.gt(2) | (changed.game_pk.eq(2) &
        (changed.at_bat_number.gt(4) | (changed.at_bat_number.eq(4) & changed.pitch_number.ge(2))))
    changed.loc[future, ['release_speed', 'plate_x', 'plate_z', 'sz_bot', 'sz_top']] = [140., 8., 9., 8., 9.]
    changed.loc[future, 'pitch_type'] = 'CU'
    changed.loc[future, 'events'] = 'home_run'
    changed.loc[future, 'description'] = 'hit_into_play'
    second, _ = build(changed)
    def target(data):
        pa = next(pa for pa in data['games'][0]['plate_appearances'] if pa['id'] == 4)
        return pa, pa['pitches'][1]
    first_pa, before = target(first)
    second_pa, after = target(second)
    assert before['request'] == after['request']
    assert before['context'] == after['context']
    assert before['context_notes'] == after['context_notes']
    assert first_pa['zone_bounds'] == second_pa['zone_bounds']
    assert first_pa['repertoire_counts'] == second_pa['repertoire_counts']
    assert before['actual'] != after['actual']


def test_same_date_earlier_results_do_not_enter_profile_or_repertoire():
    frame = sample_frame()
    original, _ = build(frame)
    earlier = frame.game_pk.eq(2) & frame.at_bat_number.lt(4)
    frame.loc[earlier, 'events'] = 'walk'
    frame.loc[earlier, 'sz_bot'] = 8.
    frame.loc[earlier, 'sz_top'] = 9.
    frame.loc[earlier, 'pitch_type'] = 'CU'
    modified, _ = build(frame)
    pa1 = next(pa for pa in original['games'][0]['plate_appearances'] if pa['id'] == 4)
    pa2 = next(pa for pa in modified['games'][0]['plate_appearances'] if pa['id'] == 4)
    assert pa1['pitches'][0]['request'] == pa2['pitches'][0]['request']
    assert pa1['zone_bounds'] == pa2['zone_bounds']
    assert pa1['repertoire_counts'] == pa2['repertoire_counts']
    # Display-only prior game observations correctly respond to changed PRIOR pitches.
    assert pa1['pitches'][0]['context'] != pa2['pitches'][0]['context']


def test_context_counts_prior_distinct_pas_and_excludes_current_pa_from_encounter_count():
    frame = sample_frame()
    context = ObservedContext(frame)
    current = frame.loc[frame.game_pk.eq(2) & frame.at_bat_number.eq(4) & frame.pitch_number.eq(2)].iloc[0]
    result = context.before(current, ['FF', 'SL'])
    assert result['prior_pitch_count'] == 7
    assert result['times_facing_batter'] == 2  # PA1 already faced; current PA4 not counted twice
    ff = next(item for item in result['speed_by_pitch_type'] if item['pitch_type'] == 'FF')
    assert ff['recent_pitch_count'] == 4 and ff['prior90_measured_count'] == 4
    assert ff['recent_mean_mph'] == np.mean([92., 93., 94., 95.])
    assert result['condition_inference'] is False
    assert '판정한 값이 아닙니다' in context_notes(result)[-1]


def test_last_five_is_pitch_window_not_five_nonmissing_measurements():
    frame = sample_frame()
    frame.loc[frame.game_pk.eq(2) & frame.pitch_type.eq('FF') & frame.at_bat_number.eq(6), 'release_speed'] = np.nan
    current = frame.loc[frame.game_pk.eq(2) & frame.at_bat_number.eq(8) & frame.pitch_number.eq(1)].iloc[0]
    result = ObservedContext(frame).before(current, ['FF'])['speed_by_pitch_type'][0]
    assert result['recent_pitch_count'] == 5
    assert result['recent_measured_count'] == 4
    assert result['recent_mean_mph'] == np.mean([94., 95., 96., 98.])


def test_no_velocity_samples_are_unavailable_not_zero_or_diagnosis():
    frame = sample_frame()
    current = frame.loc[frame.game_pk.eq(2)].iloc[0]
    result = ObservedContext(frame).before(current, ['CU'])['speed_by_pitch_type'][0]
    assert result['recent_mean_mph'] is None and result['prior90_mean_mph'] is None
    assert result['delta_mph'] is None
    assert result['recent_measured_count'] == 0


def test_player_names_are_optional_local_and_validate_ids(tmp_path):
    assert load_player_names() == {}
    assert player_label(500, '타자', {}) == '타자 #500'
    path = tmp_path/'names.json'
    path.write_text(json.dumps({'500': '  Player Name  '}))
    names = load_player_names(path)
    assert player_label(500, '타자', names) == 'Player Name'
    path.write_text(json.dumps({'not-an-id': 'Name'}))
    with pytest.raises(ValueError):
        load_player_names(path)


def test_mixed_pitcher_pa_is_excluded_using_full_game_rows():
    frame = sample_frame()
    rows = frame.loc[frame.game_pk.eq(2) & frame.at_bat_number.eq(1)].copy()
    rows.loc[rows.pitch_number.eq(2), 'pitcher'] = 99
    assert 'pitcher_change_within_pa' in pa_exclusions(rows)
