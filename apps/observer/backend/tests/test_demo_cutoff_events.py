"""Demo/archive cutoff semantics for events BETWEEN pitches (integration, review-integration-002).

A steal, pickoff, wild-pitch run or pitcher/batter substitution after the last released pitch has no mapped
release time of its own, so the view at that cutoff must hold the state right after the released pitch and must
not show the next pitch's pre-state or its recommendation (they are built from the unreleased event).
A finished game's current state is the state after the final event, not before the last pitch.
"""
import gzip
import json
from pathlib import Path

import pytest

from observer_app import game_history, service_api
from test_game_history_state import PK, action, build, move, pa, pitch, single_to_first

DEMO = {'kind': 'demo', 'session_id': 'x'}


def steal_game():
    return build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), action('stolen_base_2b', 1, 0, 0), pitch(2, 'S', 1, 1, 0),
                                pitch(3, 'X', 1, 1, 1)],
           [move(11, 1, '1B', '2B', kind='stolen_base_2b'), move(12, 3, None, None, out=True)], outs=1))[1]


def test_steal_after_released_pitch_is_held_back_with_next_rec():
    record = steal_game()
    index = [p['key'] for p in record['pitches']].index(f'{PK}:2:1')
    view = service_api.cut_view(record, index, DEMO)
    text = json.dumps(view)
    assert 'gap_after' not in text and 'situation_after' not in text  # the flag itself would announce a later event
    cur = view['current']
    assert cur['phase'] == 'resolving' and cur['situation']['bases'] == 1 and cur['situation']['runners']['second'] is False
    assert (cur['situation']['balls'], cur['situation']['strikes']) == (1, 0)
    assert view['next'] is None  # its inputs (runner on 2nd) are not released
    assert all(p['index'] <= index for p in view['pitches'])
    # once pitch 2 is released, the steal is part of its pre-pitch state
    view2 = service_api.cut_view(record, index+1, DEMO)
    assert view2['pitches'][-1]['situation_before']['bases'] == 2


def test_no_event_between_pitches_keeps_next_pitch_view():
    record = steal_game()
    index = [p['key'] for p in record['pitches']].index(f'{PK}:2:2')
    view = service_api.cut_view(record, index, DEMO)
    assert view['current']['phase'] == 'pre_pitch' and view['next'] is not None
    assert view['current']['situation'] == {k: v for k, v in record['pitches'][index+1]['situation_before'].items()}


def test_new_pitcher_after_pa_end_is_not_announced_early():
    record = build(
        single_to_first(1, 11),
        pa(2, 1, True, 12, 91, [action('pitching_substitution', 0, 0, 0), pitch(1, 'X', 0, 0, 1)],
           [move(12, 0, None, None, out=True)], outs=1))[1]
    view = service_api.cut_view(record, 0, DEMO)
    assert view['current']['pitcher']['id'] == 90 and view['current']['batter'] is None and view['next'] is None
    assert [pa['at_bat_number'] for pa in view['pas']] == [1] and view['pas'][0]['complete'] is True


def test_final_current_is_after_the_last_event():
    record = build(
        pa(1, 9, False, 21, 95, [pitch(1, 'B', 1, 0, 2), pitch(2, 'X', 1, 0, 2)],
           [move(21, 1, None, None, out=True)], event='field_out', home=0, away=0, outs=3))[1]
    record['final'] = {'away': 0, 'home': 0}
    view = service_api.cut_view(record, len(record['pitches'])-1, {'kind': 'archive', 'session_id': None})
    cur = view['current']
    assert cur['phase'] == 'final' and cur['situation']['outs'] == 3 and cur['situation']['bases'] == 0
    assert (cur['situation']['balls'], cur['situation']['strikes']) == (0, 0)


FEEDS = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds')


@pytest.mark.skipif(not (FEEDS/'849843_final.json.gz').is_file(), reason='T7 archive not mounted')
def test_real_849843_every_cutoff_never_shows_unreleased_between_pitch_state():
    record = game_history.build(json.load(gzip.open(FEEDS/'849843_final.json.gz')))
    pitches = record['pitches']
    held = 0
    for cutoff in range(len(pitches)-1):
        view = service_api.cut_view(record, cutoff, DEMO)
        after = pitches[cutoff]['situation_after']
        nxt = pitches[cutoff+1]['situation_before']
        shown = view['current']['situation']
        if pitches[cutoff]['gap_after']:
            held += 1
            assert shown == after and view['next'] is None
        else:  # nothing happens in between: the after-state IS the next pre-state (inning flip aside)
            assert after['outs'] >= 3 or all(after[k] == nxt[k] for k in ('outs', 'balls', 'strikes', 'bases', 'home_score', 'away_score'))
        assert (view['game']['away_score'], view['game']['home_score']) == (shown['away_score'], shown['home_score'])
    assert held == 12  # AB19 steal, AB42 passed ball, AB57 pickoff, pitching changes, ...
    final = service_api.cut_view(record, len(pitches)-1, {'kind': 'archive', 'session_id': None})['current']['situation']
    assert final['outs'] == 3 and (final['away_score'], final['home_score']) == (0, 8)


@pytest.mark.skipif(not FEEDS.is_dir(), reason='T7 archive not mounted')
def test_state_after_all_events_matches_feed_linescore_on_archived_games():
    games = sorted(p for p in FEEDS.iterdir() if p.name.endswith('_final.json.gz') and p.name[0].isdigit())  # skip macOS ._ files
    assert games
    for path in games:
        feed = json.load(gzip.open(path))
        now, line = game_history.build(feed)['state_now'], feed['liveData']['linescore']
        offense = line.get('offense') or {}
        assert (now['away_score'], now['home_score'], now['outs']) == (line['teams']['away']['runs'], line['teams']['home']['runs'], line.get('outs')), path.name
        assert now['bases'] == ('first' in offense) + 2*('second' in offense) + 4*('third' in offense), path.name
