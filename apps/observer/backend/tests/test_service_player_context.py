"""Selected-pitch context: terminal PA accounting, scorer ownership, substitutions and release boundaries."""
import copy
import json

from observer_app import game_history, service_api
from test_game_history_state import action, feed_of, move, pa, pitch

DEMO = {'kind': 'demo', 'session_id': 'synthetic'}


def lineup(feed):
    feed['liveData']['boxscore'] = {'teams': {side: {'players': {
        f'ID{base+n}': {'person': {'id': base+n, 'fullName': f'{side}{n}'}, 'battingOrder': str(n*100)}
        for n in range(1, 10)}, 'battingOrder': [999]*9} for side, base in [('away', 10), ('home', 20)]}}
    return feed


def next_pa(ab, batter=11, pitcher=90):
    return pa(ab, 1, True, batter, pitcher, [pitch(1, 'B', 1, 0, 0)], [], complete=False)


def test_today_before_pitch_counts_terminal_hits_walks_hbp_and_no_ab_sacrifices():
    feed = lineup(feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'X', 0, 0, 0)], [move(11, 0, None, '1B')], event='single'),
        pa(2, 1, True, 11, 90, [action('intent_walk', 4, 0, 0)], [], event='intent_walk'),
        pa(3, 1, True, 11, 90, [pitch(1, 'H', 0, 0, 0)], [], event='hit_by_pitch'),
        pa(4, 1, True, 11, 90, [pitch(1, 'X', 0, 0, 1)], [], event='sac_fly', outs=1),
        next_pa(5)))
    rec = game_history.build(feed)
    first, last = rec['pitches'][0], rec['pitches'][-1]
    assert first['context_before']['batter'] == {'at_bats': 0, 'hits': 0, 'on_base': 0}
    assert last['context_before']['batter'] == {'at_bats': 1, 'hits': 1, 'on_base': 3}
    assert last['context_before']['pitcher'] == {'pitches': 3, 'strikeouts': 0, 'earned_runs': 0}
    # The non-pitch intentional walk is withheld at the last pitch boundary.
    held = service_api.cut_view(rec, 0, DEMO)
    assert held['current']['phase'] == 'resolving' and held['next'] is None
    assert held['current']['context']['pitcher']['pitches'] == 1
    assert held['current']['context']['batter']['on_base'] is None  # batter is absent while held
    assert 'context_after' not in json.dumps(held) and 'context_now' not in json.dumps(held)
    latest = service_api.cut_view(rec, len(rec['pitches'])-2, DEMO, after=99)
    assert latest['pitches'] == [] and latest['current']['context'] == last['context_before']


def test_zero_event_intentional_walk_and_nonpitch_inning_end_do_not_create_at_bat():
    rec = game_history.build(feed_of(
        pa(1, 1, True, 11, 90, [], [], event='intent_walk'),
        pa(2, 1, True, 11, 90, [pitch(1, 'B', 1, 0, 0), action('caught_stealing_2b', 1, 0, 3)],
           [move(12, 1, '1B', None, out=True)], event='caught_stealing_2b', outs=3),
        pa(3, 1, False, 21, 80, [pitch(1, 'B', 1, 0, 0)], [], complete=False),
        next_pa(4)))
    assert rec['pitches'][-1]['context_before']['batter'] == {'at_bats': 0, 'hits': 0, 'on_base': 1}
    held = service_api.cut_view(rec, 0, DEMO)
    assert held['current']['context']['pitcher']['pitches'] == 1
    assert held['current']['situation']['half'] == 'Top'


def test_earned_run_credited_to_responsible_pitcher_only_and_missing_metadata_unknown():
    scored = move(11, 0, '3B', 'score')
    scored['details'] |= {'earned': True, 'responsiblePitcher': {'id': 90}, 'isScoringEvent': True}
    feed = feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'X', 0, 0, 0)], [], event='single'),
        pa(2, 1, True, 12, 91, [pitch(1, 'X', 0, 0, 0)], [scored], event='single', away=1),
        next_pa(3, pitcher=90))
    rec = game_history.build(feed)
    assert rec['pitches'][1]['context_before']['pitcher']['earned_runs'] == 0
    assert rec['pitches'][1]['context_after']['pitcher']['earned_runs'] == 0
    assert rec['pitches'][2]['context_before']['pitcher']['earned_runs'] == 1
    scored['details'].pop('earned')
    rec = game_history.build(feed)
    assert rec['pitches'][2]['context_before']['pitcher']['earned_runs'] is None
    scored['details'].pop('responsiblePitcher')
    assert game_history.build(feed)['pitches'][1]['context_after']['pitcher']['earned_runs'] is None


def test_strikeout_counts_and_inherited_two_strikes_are_unknown():
    rec = game_history.build(feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'S', 0, 3, 1)], [], event='strikeout', outs=1),
        pa(2, 1, True, 12, 91, [pitch(1, 'S', 0, 2, 1),
            action('pitching_substitution', 0, 2, 1, player={'id': 91}, replacedPlayer={'id': 90}),
            pitch(2, 'S', 0, 3, 2)], [], event='strikeout', outs=2), next_pa(3, pitcher=90)))
    assert rec['pitches'][1]['context_before']['pitcher']['strikeouts'] == 1
    assert rec['pitches'][2]['context_after']['pitcher']['strikeouts'] is None
    assert rec['pitches'][3]['context_before']['pitcher']['strikeouts'] is None
    assert rec['pitches'][3]['context_before']['pitcher']['pitches'] == 2


def test_lineup_starter_order_rotation_and_reached_substitutions_only():
    change = action('offensive_substitution', 1, 0, 0, player={'id': 99, 'fullName': 'Replacement'},
                    replacedPlayer={'id': 11}, battingOrder='101', position={'code': '11'})
    feed = lineup(feed_of(pa(1, 1, True, 19, 90, [pitch(1, 'B', 1, 0, 0), change,
                                               pitch(2, 'B', 2, 0, 0)], [], complete=False)))
    feed['liveData']['boxscore']['teams']['away']['players']['ID99'] = {
        'person': {'id': 99, 'fullName': 'Replacement'}, 'battingOrder': '101'}
    rec = game_history.build(feed)
    assert rec['pitches'][0]['context_before']['next_batters'] == [
        {'id': 11, 'name': 'away1', 'order': 1}, {'id': 12, 'name': 'away2', 'order': 2}]
    assert rec['pitches'][1]['context_before']['next_batters'][0]['id'] == 99
    held = service_api.cut_view(rec, 0, DEMO)
    assert held['current']['context']['next_batters'][0]['id'] == 11
    assert 'Replacement' not in json.dumps(held)
    changed = copy.deepcopy(feed)
    changed['liveData']['plays']['allPlays'][0]['playEvents'][1]['player'] = {'id': 98}
    assert game_history.build(changed)['pitches'][0]['context_before'] == rec['pitches'][0]['context_before']


def test_incomplete_prefix_and_unknown_lineup_never_default_to_zero_or_final_order():
    rec = game_history.build(feed_of(next_pa(4)))
    ctx = rec['pitches'][0]['context_before']
    assert set(ctx['batter'].values()) == {None} and set(ctx['pitcher'].values()) == {None}
    assert ctx['next_batters'] == []
    rec = game_history.build(feed_of(pa(1, 1, True, 11, 90, [pitch(4, 'B', 1, 0, 0)], [], complete=False)))
    assert rec['pitches'][0]['context_before']['pitcher']['pitches'] is None


def test_live_context_matches_current_identity_and_never_relabels_previous_players():
    rec = game_history.build(lineup(feed_of(next_pa(1))))
    live = {'phase': 'pre_pitch', 'situation': rec['state_now'], 'batter': {'id': 11}, 'pitcher': {'id': 90}}
    view = service_api.cut_view(rec, 0, {'kind': 'live'}, live_current=live)
    assert view['current']['context']['pitcher']['pitches'] == 1
    live['pitcher'] = {'id': 99}
    assert service_api.cut_view(rec, 0, {'kind': 'live'}, live_current=live)['current']['context']['pitcher']['pitches'] is None


def test_nonpitch_terminal_result_and_missing_scoring_metadata_are_not_released_early():
    rec = game_history.build(feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'B', 1, 0, 0), action('intent_walk', 4, 0, 0)], [], event='intent_walk'),
        next_pa(2)))
    assert rec['pitches'][0]['context_after']['batter']['on_base'] == 0
    assert rec['pitches'][1]['context_before']['batter']['on_base'] == 1
    assert service_api.cut_view(rec, 0, DEMO)['next'] is None
    rec = game_history.build(feed_of(pa(1, 1, True, 11, 90, [], [], event='intent_walk', away=1), next_pa(2)))
    assert rec['pitches'][0]['context_before']['pitcher']['earned_runs'] is None


def test_defensive_substitution_and_pa_batting_slot():
    rec = game_history.build(lineup(feed_of(
        pa(1, 1, True, 11, 90, [action('defensive_substitution', 0, 0, 0,
            replacedPlayer={'id': 22}, player={'id': 88, 'fullName': 'New Home'}, battingOrder='201'),
            pitch(1, 'X', 0, 0, 3)], [], outs=3),
        pa(2, 1, False, 21, 80, [pitch(1, 'B', 1, 0, 0)], [], complete=False))))
    assert rec['pas'][0]['batting_order'] == rec['pas'][1]['batting_order'] == 1
    assert rec['pitches'][1]['context_before']['next_batters'][0] == {'id': 88, 'name': 'New Home', 'order': 2}


def test_existing_record_without_context_is_backward_compatible():
    rec = game_history.build(feed_of(next_pa(1)))
    rec.pop('context_now'); rec.pop('context_people_now')
    for p in rec['pitches']:
        p.pop('context_before'); p.pop('context_after')
    assert service_api.cut_view(rec, -1, DEMO)['current']['context'] is None


def test_demo_source_exposes_existing_clock_status_without_operator_details(tmp_path):
    from fastapi.testclient import TestClient
    from test_service_demo_api import env, start_demo, control, GAME
    make, _, directory = env.__wrapped__(tmp_path)
    with TestClient(make()) as client:
        session, headers = start_demo(client, directory)
        sid = session['session_id']
        view = client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()
        assert view['source'] == {'kind': 'demo', 'session_id': sid, 'session_status': session['status']}
        assert 'operator' not in json.dumps(view) and 'rows' not in view['source']
        response = control(client, headers, sid, action='start')
        assert response.status_code == 200
        assert client.get(f'/api/service/games/{GAME}', params={'demo': sid}).json()['source']['session_status'] == 'running'
