"""Revealed-prefix watch-along summary on small synthetic precomputed games (no T7, no model)."""
import copy
import json
import math
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.main import create_app
from observer_app.watch_along import SCHEMA, WatchAlong
from observer_app.watch_summary import summarize

GAME = 900002


def pitch(index, inning, half, ab, number, away, home, before, after, event=None):
    return {'index': index, 'pa_id': f'{GAME}:{ab}', 'at_bat_number': ab, 'pitch_number': number,
            'situation': {'inning': inning, 'half': half, 'outs': 0, 'balls': 0, 'strikes': 0, 'bases': 0,
                          'away_score': away, 'home_score': home},
            'pitcher': {'id': 10 + ab, 'name': f'P{ab}', 'hand': 'R'}, 'batter': {'id': 20 + ab, 'name': f'B{ab}', 'side': 'L'},
            'status': 'SUPPORTED', 'root_status': None, 'path': 'ledger', 'pre': {'status': 'ready', 'reason': None},
            'actual': {'pitch_type': 'FF', 'pitch_label': '포심', 'result_label': event or '볼', 'event_label': event,
                       'play_text': f'play {index}' if event else None, 'speed_mph': 95.0, 'x': 0., 'z': 2.5},
            'we': {'home_before': before, 'home_after': after,
                   'home_delta': None if before is None or after is None else after - before}}


def game(decisions=None, final=None):
    """1st: away HR (1-0), home scores 2 on a double (1-2); top 2nd ends the game, bottom 2nd not played."""
    if decisions is None:
        decisions = [pitch(0, 1, 'Top', 1, 1, 0, 0, .50, .50),
                     pitch(1, 1, 'Top', 1, 2, 0, 0, .50, .40, '홈런'),
                     pitch(2, 1, 'Top', 2, 1, 1, 0, .40, .42, '삼진'),
                     pitch(3, 1, 'Bot', 3, 1, 1, 0, .42, .45, '안타'),
                     pitch(4, 1, 'Bot', 4, 1, 1, 0, .45, .70, '2루타'),
                     pitch(5, 2, 'Top', 5, 1, 1, 2, .70, 1.0, '땅볼 아웃')]
        final = {'away': 1, 'home': 2} if final is None else final
    record = {'game_pk': GAME, 'date': '2026-09-29', 'game_type': 'F', 'away_team': 'Away Club', 'home_team': 'Home Club'}
    if final is not None:
        record['final'] = final
    return {'schema': SCHEMA, 'badge': '검증 전', 'game': record, 'policy': {}, 'location': {},
            'we': {'note': '표시용'}, 'coverage': {'pitch_decisions': len(decisions), 'ready': 0, 'ready_share_of_pitches': 0},
            'decisions': decisions}


def write(tmp_path, data):
    (tmp_path/f'{GAME}.json').write_text(json.dumps(data, ensure_ascii=False))  # NaN stays NaN, as json.dump writes it
    return tmp_path


def cells(view, team):
    return [(row[team]['runs'], row[team]['status']) for row in view['innings']]


def test_nothing_revealed_shows_only_the_first_pre_pitch_score():
    view = summarize(game(), -1)
    assert view['revealed_pitches'] == 0 and not view['complete']
    assert view['innings'] == [] and view['we_curve'] == [] and view['events'] == [] and view['largest_we_changes'] == []
    assert view['score'] == {'away': 0, 'home': 0, 'source': 'pre_pitch_situation', 'after_index': None}


def test_empty_game_and_bounds():
    empty = game(decisions=[])
    view = summarize(empty, -1)
    assert view['score'] == {'away': None, 'home': None, 'source': 'unavailable', 'after_index': None}
    assert not view['complete'] and view['innings'] == []
    for through in (0, -2, True, 1.0):
        with pytest.raises(IndexError):
            summarize(empty, through)
    with pytest.raises(IndexError):
        summarize(game(), 6)
    with pytest.raises(IndexError):
        summarize(game(), 0, top=21)


def test_line_score_advances_with_the_prefix():
    data = game()
    at = lambda k: summarize(data, k)
    assert cells(at(0), 'away') == [(0, 'in_progress')] and cells(at(0), 'home') == [(None, 'not_reached')]
    # The home run's run is known from the next pitch's pre-pitch situation; the half is not over yet.
    assert cells(at(1), 'away') == [(1, 'in_progress')]
    assert at(1)['score'] == {'away': 1, 'home': 0, 'source': 'next_pre_pitch_situation', 'after_index': 1}
    # The strikeout ends the top half: the next pre-pitch view is in the bottom of the 1st.
    assert cells(at(2), 'away') == [(1, 'complete')] and cells(at(2), 'home') == [(None, 'not_reached')]
    assert cells(at(3), 'home') == [(0, 'in_progress')]
    four = at(4)
    assert cells(four, 'away') == [(1, 'complete')] and cells(four, 'home') == [(2, 'complete')]
    assert four['score'] == {'away': 1, 'home': 2, 'source': 'next_pre_pitch_situation', 'after_index': 4}
    assert four['issues'] == []


def test_completed_game_uses_the_final_record_and_marks_unplayed_half():
    view = summarize(game(), 5)
    assert view['complete'] and view['issues'] == []
    assert cells(view, 'away') == [(1, 'complete'), (0, 'complete')]
    assert cells(view, 'home') == [(2, 'complete'), (None, 'not_played')]
    assert view['score'] == {'away': 1, 'home': 2, 'source': 'final_record', 'after_index': 5}
    assert view['we_curve'][-1] == {'index': 5, 'inning': 2, 'half': 'Top', 'at_bat_number': 5, 'pitch_number': 1,
                                    'home_before': .7, 'home_after': 1.0}
    assert [e['index'] for e in view['events']] == [1, 2, 3, 4, 5]
    hr = view['events'][0]
    assert hr['runs_scored'] == {'away': 1, 'home': 0} and hr['score_after'] == {'away': 1, 'home': 0}
    assert hr['event_label'] == '홈런' and hr['play_text'] == 'play 1' and hr['inning'] == 1 and hr['half'] == 'Top'
    assert hr['we']['kind'] == 'observed_model_change' and math.isclose(hr['we']['batting_delta'], .1)
    assert view['notes']['attribution'] is None and '인과' in view['notes']['we']


def test_last_pitch_runs_are_unknown_without_a_final_record():
    data = game()
    del data['game']['final']
    view = summarize(data, 5)
    assert view['complete']
    assert view['score'] == {'away': None, 'home': None, 'source': 'unavailable', 'after_index': 5}
    assert cells(view, 'away') == [(1, 'complete'), (None, 'complete')]  # played, runs unknown
    assert cells(view, 'home') == [(2, 'complete'), (None, 'not_played')]
    assert view['events'][-1]['runs_scored'] == {'away': None, 'home': None}


def test_final_runs_on_the_last_pitch_are_counted_and_checked():
    data = game()
    data['game']['final'] = {'away': 3, 'home': 2}  # a 2-run play on the last pitch
    view = summarize(data, 5)
    assert cells(view, 'away') == [(1, 'complete'), (2, 'complete')] and view['issues'] == []
    assert view['events'][-1]['runs_scored'] == {'away': 2, 'home': 0}
    data['game']['final'] = {'away': 1, 'home': 5}  # home cannot score in the top half
    view = summarize(data, 5)
    assert 'home_runs_outside_batting_half_after_index_5' in view['issues']
    assert 'home_line_score_differs_from_score' in view['issues']
    data['game']['final'] = {'away': 0, 'home': 2}  # a score can only grow
    view = summarize(data, 5)
    assert 'away_score_decreased_after_index_5' in view['issues'] and cells(view, 'away')[1] == (None, 'complete')


@pytest.mark.parametrize('through', [-1, 0, 1, 2, 3, 4])
def test_prefix_never_reads_the_future(through):
    data = game()
    expected = summarize(data, through)
    text = json.dumps(expected, ensure_ascii=False)
    assert 'final_record' not in text and expected['complete'] is False
    future = copy.deepcopy(data)
    future['game']['final'] = {'away': 99, 'home': 98}
    nxt = through + 1
    future['decisions'][nxt]['actual'] = {'event_label': 'LEAK', 'result_label': 'LEAK', 'play_text': 'LEAK'}
    future['decisions'][nxt]['we'] = {'home_before': .01, 'home_after': .99}
    future['decisions'][nxt]['pitcher'] = {'id': 999, 'name': 'LEAK'}
    for d in future['decisions'][nxt + 1:]:
        d['situation'].update(away_score=50, home_score=60)
        d['actual'] = {'event_label': 'LEAK'}
        d['we'] = {'home_before': 0., 'home_after': 1.}
    assert summarize(future, through) == expected


def test_unknown_we_values_stay_null_and_valid_json(tmp_path):
    data = game()
    data['decisions'][0]['we'] = {'home_before': float('nan'), 'home_after': .5}
    data['decisions'][1]['we'] = {'home_before': None, 'home_after': .4}
    data['decisions'][2]['we'] = {'home_before': 1.5, 'home_after': True}
    data['decisions'][3]['we'] = {'home_before': '0.4', 'home_after': float('inf')}
    data['decisions'][4]['we'] = {}
    view = summarize(data, 5)
    assert [(p['home_before'], p['home_after']) for p in view['we_curve'][:5]] == [
        (None, .5), (None, .4), (None, None), (None, None), (None, None)]
    assert all(e['we']['home_delta'] is None and e['we']['batting_delta'] is None for e in view['events'][:4])
    assert [e['index'] for e in view['largest_we_changes']] == [5]
    write(tmp_path, data)
    with TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path)) as client:
        response = client.get(f'/api/watch/{GAME}/summary?through=5')
        assert response.status_code == 200 and 'NaN' not in response.text and 'Infinity' not in response.text


def test_largest_we_changes_are_ordered_and_limited():
    view = summarize(game(), 5, top=3)
    assert [e['index'] for e in view['largest_we_changes']] == [5, 4, 1]  # .30, .25, .10
    assert math.isclose(view['largest_we_changes'][0]['we']['home_delta'], .3)
    assert view['largest_we_changes'][1]['runs_scored'] == {'away': 0, 'home': 2}
    tie = game([pitch(0, 1, 'Top', 1, 1, 0, 0, .5, .4), pitch(1, 1, 'Top', 1, 2, 0, 0, .4, .5)], final={'away': 0, 'home': 0})
    assert [e['index'] for e in summarize(tie, 1, top=5)['largest_we_changes']] == [0, 1]
    assert summarize(game(), 5, top=0)['largest_we_changes'] == []


def test_missing_scores_make_runs_unknown_not_zero():
    data = game()
    del data['decisions'][2]['situation']['away_score']
    view = summarize(data, 4)
    assert cells(view, 'away') == [(None, 'complete')] and cells(view, 'home') == [(2, 'complete')]
    assert view['events'][0]['runs_scored'] == {'away': None, 'home': 0}


def test_half_without_pitches_is_unknown_and_order_is_enforced():
    gap = game([pitch(0, 1, 'Top', 1, 1, 0, 0, .5, .5), pitch(1, 2, 'Top', 2, 1, 0, 0, .5, .5)], final={'away': 0, 'home': 0})
    assert cells(summarize(gap, 1), 'home') == [(None, 'unknown'), (None, 'not_played')]
    assert cells(summarize(gap, 0), 'home') == [(None, 'not_reached')]
    assert cells(summarize(gap, 0), 'away') == [(0, 'complete')]  # next pre-pitch view is already in the 2nd
    backwards = game([pitch(0, 2, 'Top', 1, 1, 0, 0, .5, .5), pitch(1, 1, 'Bot', 2, 1, 0, 0, .5, .5)], final={'away': 0, 'home': 0})
    with pytest.raises(ValueError):
        summarize(backwards, 1)
    bad = game()
    bad['decisions'][0]['situation']['half'] = 'Middle'
    with pytest.raises(ValueError):
        summarize(bad, 0)


def test_run_before_the_first_pitch_of_a_half_goes_to_that_half():
    data = game([pitch(0, 10, 'Top', 1, 1, 3, 3, .5, .5, '삼진'), pitch(1, 10, 'Bot', 2, 1, 3, 4, .5, 1., '볼넷')],
                final={'away': 3, 'home': 4})
    view = summarize(data, 0)
    assert view['innings'][9] == {'inning': 10, 'away': {'runs': 0, 'status': 'complete'},
                                  'home': {'runs': 1, 'status': 'in_progress'}}
    assert view['issues'] == []  # the 1st-9th are unknown here, so the line score is not checked against the score
    assert cells(view, 'away')[:9] == [(None, 'unknown')] * 9


def test_api_summary_routes_and_privacy(tmp_path):
    write(tmp_path, game())
    with TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path)) as client:
        body = client.get(f'/api/watch/{GAME}/summary?through=2').json()
        assert body['schema'] == 'pitcheezy-watch-summary-v1' and body['through'] == 2
        assert body['teams'] == {'away': 'Away Club', 'home': 'Home Club'}
        assert client.get(f'/api/watch/{GAME}/summary?through=5&top=1').json()['score']['source'] == 'final_record'
        assert client.get(f'/api/watch/{GAME}/summary').status_code == 400  # no default: the client says what is revealed
        assert client.get(f'/api/watch/{GAME}/summary?through=-2').status_code == 400
        assert client.get(f'/api/watch/{GAME}/summary?through=x').status_code == 400
        assert client.get(f'/api/watch/{GAME}/summary?through=0&top=21').status_code == 400
        assert client.get(f'/api/watch/{GAME}/summary?through=6').status_code == 404
        assert client.get('/api/watch/123/summary?through=0').status_code == 404
        # The pre-pitch timeline and the picker still never carry the final score.
        assert '"final"' not in client.get(f'/api/watch/{GAME}').text
        assert 'final' not in client.get('/api/watch/games').text
    bad = game()
    bad['decisions'][2]['situation']['inning'] = 0
    write(tmp_path, bad)
    with TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path)) as client:
        assert client.get(f'/api/watch/{GAME}/summary?through=1').status_code == 503  # reads the next pre-pitch view
        assert client.get(f'/api/watch/{GAME}/summary?through=0').status_code == 200  # never reads beyond it


def test_watch_along_summary_reads_the_dataset(tmp_path):
    view = WatchAlong(write(tmp_path, game())).summary(GAME, 4)
    assert view['game_pk'] == GAME and view['score']['home'] == 2
