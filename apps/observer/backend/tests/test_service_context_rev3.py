"""Contract rev3 additions (api-rev3-031): venue, game jersey, as-of linescore, pre-game stats cache.

The linescore is built from revealed events only and must agree with ``current.situation`` at every cutoff; the
pre-game cache is served only for the exact game, strictly before its official date, and only for shown players.
"""
import copy
import gzip
import json
from pathlib import Path

import pytest

from observer_app import game_history, player_metadata, service_api
from test_game_history_state import PK, action, feed_of, move, pa, pitch

FEEDS = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/feeds')
DEMO = {'kind': 'demo', 'session_id': 'x'}
ARCHIVE = {'kind': 'archive', 'session_id': None}


def err(rid, index, start, end, fielder=7):
    runner = move(rid, index, start, end)
    runner['credits'] = [{'player': {'id': fielder}, 'position': {'code': '6'}, 'credit': 'f_throwing_error'}]
    return runner


def game_feed():
    feed = feed_of(
        # top 1: walk, then HBP, then a throwing error lets both runners advance (one error, credited on two runners)
        pa(1, 1, True, 11, 90, [pitch(1, 'B', 1, 0, 0), pitch(2, 'B', 2, 0, 0), pitch(3, 'B', 3, 0, 0), pitch(4, 'B', 4, 0, 0)],
           [move(11, 3, None, '1B')], event='walk'),
        pa(2, 1, True, 12, 90, [pitch(1, 'H', 0, 0, 0)], [move(12, 0, None, '1B'), move(11, 0, '1B', '2B')], event='hit_by_pitch'),
        pa(3, 1, True, 13, 90, [pitch(1, 'X', 0, 0, 0)],
           [err(13, 0, None, '2B'), err(12, 0, '1B', '3B'), err(11, 0, '2B', 'score')], event='field_error', away=1),
        pa(4, 1, True, 14, 90, [pitch(1, 'X', 0, 0, 3)], [move(14, 0, None, None, out=True)], event='triple_play', away=1, outs=3),
        # bottom 1: a home run
        pa(5, 1, False, 21, 80, [pitch(1, 'X', 0, 0, 0)], [move(21, 0, None, 'score')], event='home_run', away=1, home=1),
        pa(6, 1, False, 22, 80, [pitch(1, 'X', 0, 0, 3)], [move(22, 0, None, None, out=True)], away=1, home=1, outs=3),
        # top 10 (extra innings), single
        pa(7, 10, True, 15, 90, [pitch(1, 'X', 0, 0, 0)], [move(15, 0, None, '1B')], event='single', away=1, home=1),
        pa(8, 10, True, 16, 90, [pitch(1, 'S', 0, 1, 0)], [], complete=False, away=1, home=1))
    feed['gameData']['venue'] = {'name': 'Dodger Stadium'}
    feed['gameData']['players'] = {'ID11': {'fullName': 'B11', 'primaryNumber': '99'}}
    feed['liveData']['boxscore'] = {'teams': {'away': {'players': {'ID11': {'jerseyNumber': '50'}}},
                                              'home': {'players': {'ID80': {'jerseyNumber': '7'}}}}}
    return feed


def test_venue_and_game_jersey_not_primary_number():
    record = game_history.build(game_feed())
    assert record['card']['venue'] == 'Dodger Stadium'
    view = service_api.cut_view(record, 0, DEMO)
    assert view['game']['venue'] == 'Dodger Stadium'
    assert view['pitches'][0]['batter']['jersey'] == '50' and view['pitches'][0]['batter']['side'] == 'R'
    assert view['pitches'][0]['pitcher']['jersey'] is None  # not in the boxscore: unknown, never guessed
    assert '"99"' not in json.dumps(view)


def test_linescore_walk_hbp_error_dedup_and_extra_innings():
    record = game_history.build(game_feed())
    keys = [p['key'] for p in record['pitches']]
    # before the error pitch: walk + HBP shown, no error, no run
    view = service_api.cut_view(record, keys.index(f'{PK}:2:1'), DEMO)
    line = view['linescore']
    assert line['teams']['away'] == {'runs': 0, 'hits': 0, 'errors': 0, 'bb_hbp': 2}
    assert line['innings'][0] == {'inning': 1, 'away': 0, 'home': None} and len(line['innings']) == 9
    assert all(row['away'] is None and row['home'] is None for row in line['innings'][1:])
    # after the error: one error for the fielding (home) team though three runners carry the credit
    view = service_api.cut_view(record, keys.index(f'{PK}:3:1'), DEMO)
    assert view['linescore']['teams']['home']['errors'] == 1 and view['linescore']['teams']['away']['runs'] == 1
    # final state point: extra inning row appears, hits counted, runs agree with the header
    last = service_api.cut_view(record, len(keys)-2, DEMO)  # pre-pitch of the last pitch (top 10)
    line, sit = last['linescore'], last['current']['situation']
    assert len(line['innings']) == 10 and line['innings'][9] == {'inning': 10, 'away': 0, 'home': None}
    assert line['innings'][0] == {'inning': 1, 'away': 1, 'home': 1} and line['innings'][1] == {'inning': 2, 'away': None, 'home': None}
    assert line['teams']['away']['hits'] == 1 and line['teams']['home']['hits'] == 1
    assert (line['teams']['away']['runs'], line['teams']['home']['runs']) == (sit['away_score'], sit['home_score'])
    assert 'line_before' not in json.dumps(last) and 'line_after' not in json.dumps(last)


def test_trailing_non_pitch_event_is_not_counted_before_it_is_released():
    feed = feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'B', 1, 0, 0), action('stolen_base_2b', 1, 0, 0, details={'awayScore': 0}),
                                pitch(2, 'X', 1, 0, 0)],
           [err(10, 1, '1B', '3B'), move(11, 2, None, '1B')], event='single'),
        pa(2, 1, True, 12, 90, [pitch(1, 'S', 0, 1, 0)], [], complete=False))
    record = game_history.build(feed)
    view = service_api.cut_view(record, 0, DEMO)  # the throwing error on the steal is not released yet: held
    assert view['next'] is None and view['linescore']['teams']['home']['errors'] == 0
    view = service_api.cut_view(record, 1, DEMO)
    assert view['linescore']['teams']['home']['errors'] == 1 and view['linescore']['teams']['away']['hits'] == 1


def test_catcher_interference_is_an_error_on_the_fielding_team():
    """The scorer charges catcher's interference as an error on the catcher (the feed's final linescore counts it:
    849819 Braves E=2 = one fielding error + one catcher interference)."""
    ci = move(12, 0, None, '1B')
    ci['credits'] = [{'player': {'id': 80}, 'position': {'code': '2'}, 'credit': 'c_catcher_interf'}]
    feed = feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'B', 1, 0, 0)], [], complete=False),
        pa(2, 1, True, 12, 90, [pitch(1, 'S', 0, 1, 0)], [ci], event='catcher_interf'),
        pa(3, 1, True, 13, 90, [pitch(1, 'S', 0, 1, 0)], [], complete=False))
    feed['liveData']['plays']['allPlays'][0]['about']['isComplete'] = True
    record = game_history.build(feed)
    keys = [p['key'] for p in record['pitches']]
    assert service_api.cut_view(record, keys.index(f'{PK}:1:1'), DEMO)['linescore']['teams']['home']['errors'] == 0
    line = service_api.cut_view(record, keys.index(f'{PK}:2:1'), DEMO)['linescore']
    assert line['teams']['home']['errors'] == 1 and line['teams']['away']['errors'] == 0
    assert line['teams']['away']['hits'] == 0 and line['teams']['away']['bb_hbp'] == 0  # not a hit, not a walk


def snapshot(tmp_path, **over):
    raw = {'schema': 'pitcheezy-player-pregame-v1', 'game_pk': PK, 'as_of': '2026-09-28',
           'date_basis': 'regular_season_through_previous_day',
           'players': {'11': {'batting': {'avg': '.285', 'ops': '.912'}, 'pitching': None, 'source': 'statsapi?date=2026-09-28',
                              'as_of': '2026-09-28'},
                       '90': {'batting': None, 'pitching': {'era': 3.21}, 'source': 's', 'as_of': '2026-09-28'},
                       '16': {'batting': {'avg': '.300', 'ops': '.800'}, 'pitching': None, 'source': 's', 'as_of': '2026-09-28'}}}
    raw |= over
    (tmp_path/f'{PK}.json').write_text(json.dumps(raw))
    return tmp_path


def test_pregame_cache_served_only_for_shown_players(tmp_path):
    record = game_history.build(game_feed())
    record['pregame'] = player_metadata.load_pregame(snapshot(tmp_path), PK, record['official_date'])
    view = service_api.cut_view(record, 0, DEMO)
    pre = view['pregame']
    assert pre['basis'] == 'regular_season_through_previous_day' and pre['as_of'] == '2026-09-28'
    assert pre['players']['11'] == {'as_of': '2026-09-28', 'source': 'statsapi?date=2026-09-28', 'avg': '.285', 'ops': '.912'}
    assert pre['players']['90'] == {'as_of': '2026-09-28', 'source': 's', 'era': '3.21'}
    assert '16' not in pre['players'] and '.800' not in json.dumps(view)  # batter 16 comes up later


@pytest.mark.parametrize('over', [{'as_of': '2026-09-29'}, {'game_pk': PK+1}, {'date_basis': 'season_to_date'},
                                  {'schema': 'x'}])
def test_pregame_cache_rejected_unless_exact_game_and_strictly_before_game_day(tmp_path, over):
    assert player_metadata.load_pregame(snapshot(tmp_path, **over), PK, '2026-09-29') is None


def test_pregame_missing_cache_is_null_and_api_works(tmp_path):
    assert player_metadata.load_pregame(None, PK, '2026-09-29') is None
    assert player_metadata.load_pregame(tmp_path, PK, '2026-09-29') is None
    record = game_history.build(game_feed())
    assert service_api.cut_view(record, 0, DEMO)['pregame'] is None


def archived():
    return sorted(p for p in FEEDS.iterdir() if p.name.endswith('_final.json.gz') and p.name[0].isdigit())


@pytest.mark.skipif(not FEEDS.is_dir(), reason='T7 archive not mounted')
def test_every_cutoff_linescore_agrees_with_situation_and_final_matches_feed():
    for path in archived():
        feed = json.load(gzip.open(path))
        record = game_history.build(feed)
        for cutoff in range(-1, len(record['pitches'])):
            view = service_api.cut_view(record, cutoff, DEMO)
            line, sit = view['linescore'], view['current']['situation']
            if sit is None:
                continue
            teams = line['teams']
            assert (teams['away']['runs'], teams['home']['runs']) == (sit['away_score'], sit['home_score']), (path.name, cutoff)
            for side in ('away', 'home'):
                assert sum(r[side] or 0 for r in line['innings']) == teams[side]['runs'], (path.name, cutoff)
            cur = line['current']
            for row in line['innings']:  # nothing after the current half-inning
                if row['inning'] > cur['inning'] or (row['inning'] == cur['inning'] and cur['half'] == 'Top'):
                    assert row['home'] is None, (path.name, cutoff)
                if row['inning'] > cur['inning']:
                    assert row['away'] is None, (path.name, cutoff)
        final = service_api.cut_view(record, len(record['pitches'])-1, ARCHIVE)['linescore']
        fl = feed['liveData']['linescore']
        for side in ('away', 'home'):
            assert all(final['teams'][side][k] == fl['teams'][side][k] for k in ('runs', 'hits', 'errors')), path.name
            batting = feed['liveData']['boxscore']['teams'][side]['teamStats']['batting']
            assert final['teams'][side]['bb_hbp'] == batting['baseOnBalls']+batting['hitByPitch'], path.name
        for row in fl['innings']:
            mine = final['innings'][row['num']-1]
            assert all(mine[s] == (row.get(s) or {}).get('runs') for s in ('away', 'home')), (path.name, row['num'])


@pytest.mark.skipif(not FEEDS.is_dir(), reason='T7 archive not mounted')
def test_linescore_and_people_do_not_change_when_later_plays_change():
    """Future mutation: drop every play after the next pitch's PA; the demo view's rev3 fields must not move."""
    for path in archived()[:6]:
        feed = json.load(gzip.open(path))
        record = game_history.build(feed)
        pitches = record['pitches']
        for cutoff in range(0, len(pitches)-2, 23):
            ab = pitches[cutoff+1]['at_bat_number']
            cut = copy.deepcopy(feed)
            cut['liveData']['plays']['allPlays'] = [p for p in feed['liveData']['plays']['allPlays']
                                                    if p['about']['atBatIndex']+1 <= ab]
            cut['gameData']['status'] = {'abstractGameState': 'Live', 'detailedState': 'In Progress'}
            short = game_history.build(cut)
            if short['pitches'][cutoff].get('gap_after') != pitches[cutoff].get('gap_after'):
                continue  # the held/next decision itself depends on the next PA's tail; covered by the leak sweep
            a, b = service_api.cut_view(record, cutoff, DEMO), service_api.cut_view(short, cutoff, DEMO)
            assert a['linescore'] == b['linescore'], (path.name, cutoff)
            assert a['current'] == b['current'], (path.name, cutoff)
