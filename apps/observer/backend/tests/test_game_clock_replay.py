"""Real-pitch-time rehearsal (PACE=game): the schedule from the replay folder, game bodies from the timed replay,
and the env knobs (game, start pitch, T0, lead, lag, speed) handed to it unchanged."""
import json

from observer_app.live_feed import GameClockReplay

GAME = 777002


def schedule_folder(tmp_path):
    (tmp_path/'schedule_2026-10-10.json').write_text(json.dumps({'dates': [{'games': [{
        'gamePk': GAME, 'gameType': 'F', 'gameDate': '2026-10-10T00:00:00Z',
        'status': {'abstractGameState': 'Live', 'detailedState': 'In Progress (rehearsal)'},
        'teams': {side: {'team': {'name': side}} for side in ('away', 'home')}}]}]}))
    return tmp_path


class FakeTimed:
    def __init__(self, **options):
        self.options = options

    def position(self):
        return {'elapsed_s': 0.0, 'pitches': 3, 'index': self.options.get('start', 0)}

    def __call__(self, url, timeout):
        return {'gameData': {'game': {'pk': GAME}}, 'from': 'timed'}


def test_schedule_from_the_folder_and_game_bodies_from_the_timed_clock(tmp_path):
    player = GameClockReplay(schedule_folder(tmp_path), FakeTimed(start=4))
    assert player('https://x/api/v1/schedule?sportId=1&date=2026-10-10', 1)['dates'][0]['games'][0]['gamePk'] == GAME
    assert player(f'https://x/api/v1.1/game/{GAME}/feed/live', 1)['from'] == 'timed'
    assert player.position()['index'] == 4


def test_the_env_switch_passes_every_knob_to_the_timed_replay(tmp_path, monkeypatch):
    from observer_app.main import create_app
    from starlette.testclient import TestClient
    built = []

    def timed_replay(game_pk, **options):
        built.append((game_pk, options))
        return FakeTimed(**options), {'pitches': 3, 'halves': []}
    for name, value in {'DIR': str(schedule_folder(tmp_path)), 'PACE': 'game', 'GAME': str(GAME), 'START': '2',
                        'T0': '1800000000.5', 'LEAD_S': '40', 'LAG_S': '3', 'SPEED': '1'}.items():
        monkeypatch.setenv(f'PITCHEEZY_OBSERVER_LIVE_REPLAY_{name}', value)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'store.sqlite3'))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    with TestClient(create_app(object(), timed_replay=timed_replay)) as client:
        client.get('/api/live/games', params={'date': '2026-10-10'})
        health = client.get('/api/health').json()['live_store']
    assert built == [(GAME, {'start': 2, 't0': 1800000000.5, 'speed': 1.0, 'lag_s': 3.0, 'lead_s': 40.0})]
    assert health['source'] == 'replay_poll' and health['replay_position']['index'] == 2
