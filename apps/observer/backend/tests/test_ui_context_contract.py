"""UI additions use synthetic feeds, event-prefix accounting and existing pregame caches only."""
import copy
import json
from pathlib import Path

from observer_app import game_history, service_api
from observer_app.live_feed import LiveFeed
from test_game_history_state import action, feed_of, pa, pitch
from test_service_player_context import lineup


def test_full_lineup_and_boxscore_do_not_leak_later_replacement_or_totals():
    change = action('offensive_substitution', 0, 0, 0, player={'id': 99, 'fullName': 'Later'},
                    replacedPlayer={'id': 11}, battingOrder='101')
    feed = lineup(feed_of(
        pa(1, 1, True, 11, 90, [pitch(1, 'X', 0, 0, 0)], [], event='single'),
        pa(2, 1, True, 12, 90, [pitch(1, 'B', 1, 0, 0), change, pitch(2, 'B', 2, 0, 0)], [], complete=False)))
    feed['liveData']['boxscore']['teams']['away']['players']['ID99'] = {
        'person': {'id': 99, 'fullName': 'Later'}, 'battingOrder': '101', 'stats': {'batting': {'hits': 99}}}
    record = game_history.build(feed)
    before = record['pitches'][0]['context_before']
    assert len(before['lineups']['away']) == 9
    assert before['boxscore']['away']['batters'][0]['hits'] == 0
    held = service_api.cut_view(record, 1, {'kind': 'demo'})
    assert 'Later' not in json.dumps(held)
    assert held['current']['context']['lineups']['away'][0]['id'] == 11
    assert all('boxscore' not in p['context_before'] and 'lineups' not in p['context_before'] for p in held['pitches'])
    after = record['context_now']
    assert after['lineups']['away'][0]['id'] == 99
    assert next(p for p in after['boxscore']['away']['batters'] if p['id'] == 11)['hits'] == 1
    assert next(p for p in after['boxscore']['away']['batters'] if p['id'] == 99)['hits'] == 0
    assert after['boxscore']['home']['pitchers'][0]['pitches'] == 3
    feed['liveData']['boxscore']['teams']['away']['players']['ID99']['stats']['batting']['hits'] = 500
    assert game_history.build(feed)['pitches'][0]['context_before'] == before


def test_incomplete_prefix_has_unknown_counts_and_tracking_is_observed_only():
    event = pitch(1, 'B', 1, 0, 0)
    event['pitchData'] = {'extension': 6.4, 'breaks': {'spinRate': 2400}}
    event['hitData'] = {'launchSpeed': 98.1, 'launchAngle': -5.0, 'totalDistance': 123}
    feed = lineup(feed_of(pa(4, 1, True, 11, 90, [event], [], complete=False)))
    record = game_history.build(feed)
    ctx = record['context_now']
    assert ctx['lineups']['away'] == []
    assert all(p['hits'] is None for p in ctx['boxscore']['away']['batters'])
    actual = record['pitches'][0]['actual']
    assert actual['tracking'] == {'extension_feet': 6.4, 'spin_rate_rpm': 2400,
                                 'exit_velocity_mph': 98.1, 'launch_angle_deg': -5.0, 'distance_feet': 123}
    view = service_api.cut_view(record, 0, {'kind': 'live'})
    service_api._apply_receipts(view['pitches'], {}, view['pas'])
    assert view['pitches'][0]['actual']['tracking'] is None
    assert set(game_history.tracking({'hitData': {'launchSpeed': float('nan')}, 'pitchData': {'extension': True}}).values()) == {None}


def test_live_pregame_cache_is_asof_and_exact_game(tmp_path, monkeypatch):
    from publish_fixture import World, LIVE_GAME
    cache = tmp_path/'pregame'
    cache.mkdir()
    monkeypatch.setenv('PITCHEEZY_PLAYER_SNAPSHOT_DIR', str(cache))
    raw = {'schema': 'pitcheezy-player-pregame-v1', 'game_pk': LIVE_GAME, 'as_of': '2026-10-04',
           'date_basis': 'regular_season_through_previous_day',
           'players': {'32': {'as_of': '2026-10-04', 'batting': {'avg': '.250', 'ops': '.700'}}}}
    path = cache/f'{LIVE_GAME}.json'
    path.write_text(json.dumps(raw))
    world = World(tmp_path/'world')
    try:
        view = world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert view['pregame']['players']['32']['ops'] == '.700'
        for invalid in (raw | {'game_pk': LIVE_GAME+1}, raw | {'as_of': '2026-10-05'}):
            path.write_text(json.dumps(invalid))
            assert world.client.get(f'/api/service/games/{LIVE_GAME}').json()['pregame'] is None
    finally:
        world.close()


def test_schedule_metadata_passes_only_observed_fields():
    row = {'gamePk': 900001, 'gameType': 'W', 'gameDate': '2026-10-27T00:00:00Z',
           'seriesDescription': 'World Series', 'seriesGameNumber': 3, 'gamesInSeries': 7,
           'status': {'abstractGameState': 'Preview', 'detailedState': 'Scheduled'},
           'teams': {'away': {'team': {'name': 'Away'}, 'probablePitcher': {'id': 7, 'fullName': 'Starter', 'secret': 'omit'}},
                     'home': {'team': {'name': 'Home'}}}}
    upstream = lambda *_: {'dates': [{'games': [copy.deepcopy(row)]}]}
    result = LiveFeed(upstream).schedule('2026-10-27')[0]
    assert result['game_type'] == 'W' and result['series_game_number'] == 3
    assert result['probable_pitchers'] == {'away': {'id': 7, 'name': 'Starter'}, 'home': None}


def test_win_expectancy_read_is_gated_by_recorded_recommendation(tmp_path, monkeypatch):
    from publish_fixture import World, LIVE_GAME, snap
    world = World(tmp_path/'world')
    calls = []
    estimate = {'status': 'available', 'home_probability': .6}
    monkeypatch.setattr(world.armb, 'win_expectancy', lambda *args, **kwargs: calls.append(kwargs) or estimate)
    try:
        world.cycle()
        view = world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert view['display_state']['state'] == 'recommended'
        assert view['win_expectancy'] == estimate and calls[-1]['snapshot_revision']
        assert calls[-1]['pending_pitch_key'] == view['display_state']['pitch_key']
        calls.clear()
        world.upstream.body = ConnectionError('synthetic offline')
        world.now[0] += 20
        view = world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert 'win_expectancy' not in view and calls == []
        world.cycle(snap(half='final'))
        assert 'win_expectancy' not in world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert calls == []
    finally:
        world.close()


def test_public_schema_accepts_new_values_and_old_files_but_rejects_wrong_types():
    from observer_app import publish_schema
    folder = Path(__file__).resolve().parents[2]/'publish_contract/v1'
    body = json.loads((folder/'examples/game-live.json').read_text())
    schema = json.loads((folder/'schema/game.schema.json').read_text())
    body['game'] |= {'game_type': 'W', 'series_game_number': 3, 'games_in_series': 7,
                     'probable_pitchers': {'away': {'id': 7, 'name': None}, 'home': None}}
    context = body['current']['context']
    context['lineups']['away'] = [{'id': 11, 'name': 'Starter', 'order': 1}]
    context['boxscore']['away']['batters'] = [{'id': 11, 'name': 'Starter', 'at_bats': None, 'hits': 0, 'on_base': 0}]
    body['pitches'][0]['actual']['tracking'] = game_history.tracking({'pitchData': {'extension': 6.4}})
    assert publish_schema.validate(body, schema) == []

    body['pitches'][0]['actual']['tracking']['extension_feet'] = True
    assert publish_schema.validate(body, schema)
    body['pitches'][0]['actual'].pop('tracking')
    for key in ('game_type', 'series_description', 'series_game_number', 'games_in_series', 'probable_pitchers'):
        body['game'].pop(key, None)
    for ctx in [context, *(p['context_before'] for p in body['pitches'])]:
        ctx.pop('lineups', None); ctx.pop('boxscore', None)
    body.pop('win_expectancy', None)
    assert publish_schema.validate(body, schema) == []


def test_unconfirmed_terminal_result_cannot_leak_through_later_statistics(tmp_path, monkeypatch):
    from publish_fixture import World, LIVE_GAME, snap
    body = snap()
    terminal = body['liveData']['plays']['allPlays'][0]['playEvents'][-1]
    terminal.pop('playId')
    world = World(tmp_path/'world', first=body)
    calls = []
    monkeypatch.setattr(world.armb, 'win_expectancy', lambda *args, **kwargs: calls.append(kwargs) or {'status': 'available', 'home_probability': .6})
    try:
        world.cycle()
        view = world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert view['receipts']['unconfirmed'] > 0
        assert view['current']['context'] is None
        assert 'win_expectancy' not in view and calls == []
        index = next(p['index'] for p in view['pitches'] if p['receipt']['status'] == 'unconfirmed')
        assert all(p['context_before'] is None for p in view['pitches'] if p['index'] >= index)
        assert view['pitches'][0]['context_before'] is not None
        # A newly admitted version restores the context without keeping the suppressed values cached.
        world.cycle(snap('20261005_100100'))
        repaired = world.client.get(f'/api/service/games/{LIVE_GAME}').json()
        assert repaired['receipts']['unconfirmed'] == 0
        assert repaired['current']['context']['boxscore']
    finally:
        world.close()
