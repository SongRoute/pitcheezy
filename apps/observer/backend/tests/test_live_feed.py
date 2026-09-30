"""Live GUMBO parser/poller on trimmed real snapshots (PHI @ ATL, gamePk 849845, 2026-09-29 Wild Card)."""
import copy
import json
from pathlib import Path
import sys
import urllib.error

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_feed import LIVE_CONFIG, LiveFeed, LiveFeedError, ReplayTransport, parse_state
from observer_app.main import create_app

FIXTURES = Path(__file__).parent/'fixtures/live'
PROFILE = {'rates': [.8, .4, .1, .2, .14, .41], 'reliabilities': [.9]*6, 'as_of': '2025-04-30'}
PINS = {'pitchers': {'554430': {'p_throws': 'R', 'pitch_types': ['FF', 'SI'], 'name': 'Wheeler, Zack'}},
        'profiles': {'645277': PROFILE},
        'default_profile': {'rates': [.76, .47, .085, .225, .165, .43], 'reliabilities': [0.]*6, 'as_of': '2023-01-01'},
        'repertoire_counts': {'554430': {'FF': 2517, 'SI': 1061}}}


def feed(name):
    return json.loads((FIXTURES/name).read_text())


def with_pitcher(snapshot, pitcher_id):
    snapshot = copy.deepcopy(snapshot)
    snapshot['liveData']['plays']['currentPlay']['matchup']['pitcher']['id'] = pitcher_id
    return snapshot


def test_mid_pa_state_count_runners_and_unknown_pitcher():
    state = parse_state(feed('849845_20260929_195940.json'), PINS)
    assert state['game']['game_type'] == 'F' and state['game']['feed_timestamp'] == '20260929_195940'
    assert state['situation'] == {'inning': 7, 'half': 'Top', 'outs': 1, 'bases': 1,
                                  'runners': {'first': True, 'second': False, 'third': False},
                                  'home_score': 2, 'away_score': 1, 'balls': 1, 'strikes': 1}
    assert [(p['pitch_type'], p['description']) for p in state['pitch_sequence']] == [('FF', 'Foul Bunt'), ('SL', 'Ball')]
    assert state['pitch_sequence'][-1]['count_after'] == {'balls': 1, 'strikes': 1}
    assert state['batter'] == {'id': 681082, 'name': 'Bryson Stott', 'side': 'L', 'profile_source': 'league_default'}
    # Chris Sale is not pinned in the frozen bundle: explicit status, no fallback guess.
    assert state['status'] == 'unsupported_pitcher' and state['pre_pitch'] is None
    assert state['pitcher'] == {'id': 519242, 'name': 'Chris Sale', 'hand': 'L', 'supported': False}


def test_supported_pitcher_builds_frozen_pre_pitch_input():
    state = parse_state(with_pitcher(feed('849845_20260929_203124.json'), 554430), PINS)
    assert state['status'] == 'ready' and state['batter']['profile_source'] == 'frozen_snapshot'
    request = state['pre_pitch'].request
    assert request == {'inning': 8, 'outs': 2, 'bases': 1, 'home_score': 2, 'away_score': 3, 'balls': 2, 'strikes': 2,
                       'topbot': 'Bot', 'pitcher_id': 554430, 'batter_stand': 'L', 'date': '2026-09-29', 'top_k': 3,
                       'batter_profile': PROFILE}
    assert state['pre_pitch'].zone_bounds == LIVE_CONFIG['zone_bounds']  # <=2025 league pin, not the feed's 2026 zone
    assert state['pre_pitch'].repertoire_counts == PINS['repertoire_counts']['554430']
    leaked = copy.deepcopy(PINS)
    leaked['profiles']['645277'] = PROFILE | {'as_of': '2026-06-01'}
    with pytest.raises(ValueError):
        parse_state(with_pitcher(feed('849845_20260929_203124.json'), 554430), leaked)


def test_between_batters_and_out_of_scope_game_type():
    snapshot = feed('849845_20260929_195940.json')
    done = copy.deepcopy(snapshot)
    done['liveData']['plays']['currentPlay']['about']['isComplete'] = True
    state = parse_state(done, PINS)
    assert state['situation']['balls'] == 0 and state['batter']['id'] == 681082 and state['pitch_sequence'] == []
    done['liveData']['linescore']['outs'] = 3
    assert parse_state(done, PINS)['status'] == 'between_innings'
    spring = copy.deepcopy(snapshot)
    spring['gameData']['game']['type'] = 'S'
    assert parse_state(spring, PINS)['status'] == 'unsupported_game_type'


class FakeTransport:
    def __init__(self, failures, payload):
        self.failures, self.payload, self.calls = failures, payload, 0

    def __call__(self, url, timeout):
        self.calls += 1
        if self.calls <= self.failures:
            raise urllib.error.URLError('synthetic outage')
        return self.payload


def test_poller_backoff_retries_then_gives_up():
    sleeps = []
    ok = LiveFeed(FakeTransport(2, {'dates': []}), sleep=sleeps.append)
    assert ok.schedule('2026-09-29') == [] and sleeps == [1.0, 2.0]
    sleeps.clear()
    down = LiveFeed(FakeTransport(99, None), sleep=sleeps.append)
    with pytest.raises(LiveFeedError):
        down.snapshot(849845)
    assert sleeps == [1.0, 2.0, 4.0]


def test_delay_buffer_serves_only_snapshots_older_than_delay():
    now = [0.0]
    live = LiveFeed(ReplayTransport(FIXTURES), delay_s=30, clock=lambda: now[0], sleep=lambda _: None)
    assert live.snapshot(849845) == (None, 30.0)
    now[0] = 15.0
    live.snapshot(849845)  # within poll interval: no second fetch yet
    now[0] = 31.0
    first, _ = live.snapshot(849845)  # fetches the second recording, serves the first
    assert first['metaData']['timeStamp'] == '20260929_195940'
    now[0] = 62.0
    assert live.snapshot(849845)[0]['metaData']['timeStamp'] == '20260929_203124'


class FakeService:
    def __init__(self):
        self.calls = []

    def _recommend(self, pitch, pa):
        self.calls.append((pitch, pa))
        return {'id': 'rec', 'status': 'ready', 'mode': 'experimental_location_proxy', 'candidates': []}


def test_live_routes_replay_offline_with_demo_badge():
    service = FakeService()
    app = create_app(service, live_feed=LiveFeed(ReplayTransport(FIXTURES), delay_s=0), live_pins=PINS)
    with TestClient(app) as client:
        games = client.get('/api/live/games', params={'date': '2026-09-29'}).json()
        assert games['demo_status'] == '검증 전 실험 버전'
        assert {g['game_pk'] for g in games['games']} >= {849845} and all(g['game_type'] == 'F' for g in games['games'])
        state = client.get('/api/live/849845/state').json()
        assert state['demo_status'] == '검증 전 실험 버전'
        assert state['status'] == 'unsupported_pitcher' and state['recommendation'] is None and not service.calls
        assert client.get('/api/live/games', params={'date': 'bad'}).status_code == 400
    pinned = copy.deepcopy(PINS)
    pinned['pitchers']['519242'] = {'p_throws': 'L', 'pitch_types': ['FF'], 'name': 'test pin'}
    pinned['repertoire_counts']['519242'] = {'FF': 100}
    app = create_app(service, live_feed=LiveFeed(ReplayTransport(FIXTURES), delay_s=0), live_pins=pinned)
    with TestClient(app) as client:
        state = client.get('/api/live/849845/state').json()
    assert state['status'] == 'ready' and state['recommendation']['id'] == 'rec'
    pitch, pa = service.calls[-1]
    assert pitch['id'] == 'live:849845:20260929_195940:1-1' and pitch['request']['balls'] == 1
    assert pa == {'zone_bounds': LIVE_CONFIG['zone_bounds'], 'repertoire_counts': {'FF': 100}}
