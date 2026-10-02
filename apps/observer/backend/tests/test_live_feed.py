"""Live GUMBO parser/poller on trimmed real snapshots (PHI @ ATL, gamePk 849845, 2026-09-29 Wild Card)."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.error

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.live_armb import LiveArmB, state_key
from observer_app.live_feed import LIVE_CONFIG, LiveFeed, LiveFeedError, ReplayTransport, due_up, event_wall, last_pitch, parse_state, pitcher_today, tap_delay
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


def at(snapshot, wall):
    """The snapshot, with its newest game event stated at wall time ``wall`` (as a replay transport does)."""
    snapshot = copy.deepcopy(snapshot)
    snapshot['metaData']['eventWall'] = wall
    return snapshot


def test_event_time_is_the_newest_pitch_end_with_the_feed_stamp_as_floor():
    snapshot = feed('849845_20260929_195940.json')  # trimmed fixture: no event times, so the feed stamp (UTC) stands
    stamp = datetime(2026, 9, 29, 19, 59, 40, tzinfo=timezone.utc).timestamp()
    assert event_wall(snapshot) == stamp
    timed = copy.deepcopy(snapshot)
    events = timed['liveData']['plays']['currentPlay']['playEvents']
    events[-1] |= {'startTime': '2026-09-29T19:59:40.100Z', 'endTime': '2026-09-29T19:59:44.600Z'}
    assert event_wall(timed) == stamp + 4.6  # the pitch counts from its end: never revealed before the ball arrives
    assert event_wall(at(timed, 12.5)) == 12.5 and event_wall({'metaData': {}}) is None


def test_each_viewer_sees_the_game_as_it_stood_their_own_delay_ago():
    now = [1000.0]
    before, pitch = at(feed('849845_20260929_195940.json'), 990.0), at(feed('849845_20260929_203124.json'), 1012.0)
    live = LiveFeed(lambda url, timeout: before if now[0] < 1020 else pitch, delay_s=30,
                    clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    assert live.snapshot(849845) == (None, 20.0)  # the only snapshot is 10 s old: a 30 s broadcast has not shown it yet
    entry, wait = live.snapshot(849845, 5)
    assert entry['feed'] is before and wait == 0.0 and (entry['event'], entry['fetched']) == (990.0, 1000.0)
    now[0] = 1020.0  # the pitch thrown at 1012 reaches us 8 s later
    assert live.snapshot(849845, 5)[0]['feed'] is pitch       # a 5 s broadcast already showed it: served as soon as fetched
    assert live.snapshot(849845, 5)[0]['fetched'] - live.snapshot(849845, 5)[0]['event'] == 8.0
    assert live.snapshot(849845)[0]['feed'] is before         # a 30 s broadcast is still before the pitch
    assert live.snapshot(849845, 0)[0]['feed'] is pitch and live.latest(849845) is pitch
    now[0] = 1042.0
    assert live.snapshot(849845)[0]['feed'] is pitch          # 1012 + 30
    assert live.snapshot(849845, 9999) == (None, 68.0)        # clamped to max_delay_s (120): nothing that old was seen
    now[0] = 1200.0
    live.snapshot(849845)
    assert [entry['feed'] is pitch for entry in live.buffers[849845]] == [True]  # older than every allowed delay: dropped


class FakePolicy:
    """Stands in for demo_precompute.LivePolicy: one result per feed, counting calls."""
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot):
        self.calls.append(snapshot['metaData']['timeStamp'])
        pre = {'status': 'ready', 'reason': None, 'recommendation': {'candidates': [{'rank': 1, 'pitch_type': 'SL'}]}}
        return {'key': '849845:61:3', 'pitch': {'key': '849845:61:3', 'pre': pre}, 'pa_pitches': [],
                'previous': None, 'home_we_now': .6, 'policy_identity': 'a6dffaea' + '0' * 56, 'seconds': {'bind': 5.0}}


def wait_computed(armb, count):
    deadline = time.time() + 10
    while armb.status()['computed'] < count and time.time() < deadline:
        time.sleep(.01)


def test_live_route_serves_the_frozen_armb_view_computed_in_the_background():
    policy = FakePolicy()
    armb = LiveArmB(lambda: policy)
    app = create_app(object(), live_feed=LiveFeed(ReplayTransport(FIXTURES), delay_s=0), live_policy=armb)
    with TestClient(app) as client:
        games = client.get('/api/live/games', params={'date': '2026-09-29'}).json()
        assert games['demo_status'] == '검증 전 실험 버전' and games['badge'].endswith('실제 투구 분포 근사')
        assert {g['game_pk'] for g in games['games']} >= {849845} and all(g['game_type'] == 'F' for g in games['games'])
        first = client.get('/api/live/849845/state').json()
        # Chris Sale is outside the legacy 6-pitcher bundle; the ARM-B view does not use those pins.
        assert first['status'] == 'ready' and first['pitcher']['name'] == 'Chris Sale'
        assert first['recommendation']['status'] in ('computing', 'ready')
        wait_computed(armb, 1)
        state = client.get('/api/live/849845/state').json()
        assert state['recommendation']['status'] == 'ready' and state['recommendation']['policy_identity'] == 'a6dffaea'
        assert state['recommendation']['recommendation']['candidates'][0]['pitch_type'] == 'SL'
        assert client.get('/api/live/games', params={'date': 'bad'}).status_code == 400
        assert client.get('/api/health').json()['demo']['live_policy']['state'] == 'ready'


def test_live_policy_failures_are_explicit_not_fatal():
    def broken():
        raise FileNotFoundError('/Volumes/T7 Shield not mounted')
    armb = LiveArmB(broken)
    snapshot = feed('849845_20260929_195940.json')
    assert armb.lookup(snapshot) is None
    deadline = time.time() + 5
    while armb.status()['state'] == 'loading' and time.time() < deadline:
        time.sleep(.01)
    assert armb.status()['state'] == 'unavailable' and 'not mounted' in armb.status()['reason']
    app = create_app(object(), live_feed=LiveFeed(ReplayTransport(FIXTURES), delay_s=0), live_policy=armb)
    with TestClient(app) as client:
        state = client.get('/api/live/849845/state').json()
    assert state['status'] == 'ready' and state['recommendation']['status'] == 'unavailable'


def test_state_key_changes_with_each_pre_pitch_input_and_cache_is_reused():
    snapshot = feed('849845_20260929_195940.json')
    moved = copy.deepcopy(snapshot)
    moved['liveData']['linescore']['offense']['second'] = moved['liveData']['linescore']['offense'].pop('first')
    assert state_key(snapshot) != state_key(moved)
    assert state_key(snapshot) == state_key(copy.deepcopy(snapshot))
    # The feed revises thrown pitches without adding an event: each revision is a new state.
    events = lambda s: s['liveData']['plays']['currentPlay']['playEvents']
    overturned = copy.deepcopy(snapshot)  # a challenged call: ball -> strike, and the count with it
    events(overturned)[-1]['details']['call'] = {'code': 'C'}
    overturned['liveData']['plays']['currentPlay']['count'] = {'balls': 0, 'strikes': 2, 'outs': 1}
    completed = copy.deepcopy(snapshot)   # spin arrives a few seconds after the pitch itself
    events(completed)[-1].setdefault('pitchData', {}).setdefault('breaks', {})['spinRate'] = 2511
    retyped = copy.deepcopy(snapshot)
    events(retyped)[-1]['details']['type'] = {'code': 'ST', 'description': 'Sweeper'}
    assert len({state_key(s) for s in (snapshot, overturned, completed, retyped)}) == 4
    policy = FakePolicy()
    armb = LiveArmB(lambda: policy)
    armb.lookup(snapshot)
    wait_computed(armb, 1)
    assert armb.lookup(snapshot)['result']['key'] == '849845:61:3' and len(policy.calls) == 1


def test_feed_outage_is_a_503_with_a_korean_message():
    down = LiveFeed(FakeTransport(99, None), sleep=lambda _: None)
    app = create_app(object(), live_feed=down, live_policy=LiveArmB(lambda: FakePolicy()))
    with TestClient(app) as client:
        response = client.get('/api/live/849845/state')
    assert response.status_code == 503 and 'MLB 경기 정보를' in response.json()['detail']


class FakeTimedReplay:
    """Stands in for demo_precompute.TimedReplay: one snapshot, a clock position, counted constructions."""
    built = []

    def __init__(self, game_pk, start, t0, speed, lag_s):
        self.t0, self.lag_s, self.options = t0, lag_s, (game_pk, start, speed, lag_s)
        FakeTimedReplay.built.append(self)

    def position(self):
        return {'elapsed_s': 12.0, 'pitches': 250, 'index': self.options[1]}

    def __call__(self, url, timeout):
        return at(feed('849845_20260929_195940.json'), self.t0)


def fake_timed_replay(game_pk, **options):
    if game_pk != 849845:
        raise KeyError(game_pk)
    if options.get('start', 0) >= 250:
        raise IndexError(options['start'])
    outline = {'game': {'game_pk': game_pk}, 'pitches': 250, 'halves': [{'label': '1초', 'first': 0}]}
    return (FakeTimedReplay(game_pk, **options) if options else None), outline


def test_replay_route_pins_one_clock_per_link_and_takes_a_viewer_delay():
    FakeTimedReplay.built.clear()
    armb = LiveArmB(lambda: FakePolicy())
    app = create_app(object(), live_feed=LiveFeed(ReplayTransport(FIXTURES), delay_s=0), live_policy=armb,
                     timed_replay=fake_timed_replay)
    with TestClient(app) as client:
        assert client.get('/api/replay/849845/outline').json()['halves'] == [{'label': '1초', 'first': 0}]
        assert client.get('/api/replay/1/outline').status_code == 404
        first = client.get('/api/live/849845/state', params={'replay': 40}).json()
        clock = first['replay']
        assert first['status'] == 'ready' and (clock['start'], clock['speed'], clock['index']) == (40, 1.0, 40)
        assert abs(clock['t0'] - time.time()) < 5 and first['delay_s'] == 0
        again = client.get('/api/live/849845/state', params={'replay': 40, 't0': clock['t0']}).json()
        assert again['replay']['t0'] == clock['t0'] and len(FakeTimedReplay.built) == 1  # same link, same clock
        rehearsal = client.get('/api/live/849845/state', params={'replay': 40, 't0': clock['t0'], 'lag': 16}).json()
        assert rehearsal['replay']['lag_s'] == 16 and again['replay']['lag_s'] == 0 and len(FakeTimedReplay.built) == 2
        assert again['sync']['event_age_s'] >= 0 and again['sync']['max_delay_s'] == 120
        late = client.get('/api/live/849845/state', params={'replay': 40, 't0': clock['t0'], 'delay': 45}).json()
        assert late['status'] == 'buffering' and late['delay_s'] == 45  # a viewer 45 s behind has not reached it yet
        assert client.get('/api/live/849845/state', params={'replay': 40, 't0': time.time()+3600}).status_code == 400
        assert client.get('/api/live/849845/state', params={'replay': 999}).status_code == 404
        assert client.get('/api/live/1/state', params={'replay': 0}).status_code == 404
        live = client.get('/api/live/849845/state', params={'delay': 20}).json()  # the real-feed path is untouched
        assert live['replay'] is None and live['delay_s'] == 20 and live['status'] == 'ready'


def test_a_request_waits_briefly_for_a_just_queued_state():
    class SlowPolicy(FakePolicy):
        def __call__(self, snapshot):
            time.sleep(.15)
            return super().__call__(snapshot)
    armb = LiveArmB(lambda: SlowPolicy())
    snapshot = feed('849845_20260929_195940.json')
    armb.lookup(feed('849845_20260929_203124.json'))  # loads the policy and computes another state first
    wait_computed(armb, 1)
    assert armb.lookup(snapshot) is None                        # no wait: queued, answered on a later poll
    assert armb.lookup(snapshot, wait_s=5)['result']['key'] == '849845:61:3'  # waits for the worker instead
    started = time.monotonic()
    moved = copy.deepcopy(snapshot)
    moved['liveData']['linescore']['outs'] = 2
    assert armb.lookup(moved, queue=False, wait_s=5) is None and time.monotonic()-started < 1  # nothing to wait for


def test_pitcher_today_counts_only_this_pitchers_pitches_in_the_feed():
    pitch = lambda index, code, speed: {'index': index, 'isPitch': True, 'details': {'type': {'code': code, 'description': code}},
                                        'pitchData': {'startSpeed': speed}}
    change = {'index': 1, 'isPitch': False, 'details': {'eventType': 'pitching_substitution'}, 'player': {'id': 11}}
    plays = [{'matchup': {'pitcher': {'id': 10}}, 'playEvents': [pitch(0, 'FF', 95.0), pitch(1, 'SL', 86.0)]},
             # 10 threw the first pitch of this PA, then 11 came in: only the pitches after the change are 11's
             {'matchup': {'pitcher': {'id': 11}}, 'playEvents': [pitch(0, 'FF', 96.0), change, pitch(2, 'FF', 97.0), pitch(3, 'XX', None)]},
             {'matchup': {'pitcher': {'id': 11}}, 'playEvents': [pitch(0, 'FF', 98.0), {'index': 1, 'isPitch': False, 'details': {}}]}]
    snapshot = {'liveData': {'plays': {'allPlays': plays}}}
    assert pitcher_today(snapshot, 11) == {'pitches': 3, 'mix': [
        {'pitch_type': 'FF', 'pitch_label': '포심', 'count': 2, 'share': 0.6667, 'avg_mph': 97.5},
        {'pitch_type': 'XX', 'pitch_label': 'XX', 'count': 1, 'share': 0.3333, 'avg_mph': None}]}
    assert pitcher_today(snapshot, 10)['pitches'] == 2 and pitcher_today(snapshot, 99) == {'pitches': 0, 'mix': []}
    state = parse_state(feed('849845_20260929_195940.json'), None)
    assert state['pitcher_today']['pitches'] == len(state['pitch_sequence']) == 2  # the trimmed fixture holds only currentPlay


def test_tap_measures_the_viewers_delay_from_the_pitch_they_just_saw():
    text = lambda wall: datetime.fromtimestamp(wall, timezone.utc).isoformat().replace('+00:00', 'Z')
    pitch = lambda arrival: {'isPitch': True, 'startTime': text(arrival-.5), 'endTime': text(arrival+20)}  # endTime: rewritten
    snapshot = {'liveData': {'plays': {'allPlays': [{'playEvents': [pitch(1000.0), pitch(1023.0)]}],
                                       'currentPlay': {'playEvents': [pitch(1046.5), {'isPitch': False}]}}}}
    assert tap_delay(snapshot, 1056.0, 30.0) == 33.0   # set to 30 s: the pitch that ended at 1023, not the newer or older one
    assert tap_delay(snapshot, 1056.0, 12.0) == 9.5    # set to 12 s: the newest pitch
    assert tap_delay(snapshot, 1056.0, 20.0) is None   # between two pitches by more than the tolerance: not guessed
    assert tap_delay({'liveData': {'plays': {}}}, 1056.0, 30.0) is None
    live = LiveFeed(lambda url, timeout: snapshot | {'metaData': {'eventWall': 0.0}}, delay_s=0)
    app = create_app(object(), live_feed=live, live_policy=LiveArmB(lambda: FakePolicy()))
    with TestClient(app) as client:
        assert client.get('/api/live/7/tap', params={'delay': 30}).json()['status'] == 'unclear'  # nothing fetched yet
        live.snapshot(7)
        assert client.get('/api/live/7/tap', params={'delay': 30}).json()['status'] == 'unclear'  # those pitches are decades old
        assert client.get('/api/live/7/tap').status_code == 400


def test_the_pitch_that_ended_an_inning_is_still_shown_during_the_break():
    pitch = lambda index, code, call: {'index': index, 'isPitch': True, 'pitchData': {'startSpeed': 88.5},
                                       'details': {'type': {'code': code, 'description': code}, 'description': call}}
    auto = {'index': 1, 'isPitch': False, 'type': 'no_pitch', 'details': {'call': {'code': 'VP'}}}
    plays = [{'about': {'atBatIndex': 40, 'inning': 5, 'isTopInning': False, 'isComplete': True},
              'result': {'eventType': 'strikeout', 'description': 'A strikes out swinging.'},
              'playEvents': [pitch(0, 'FF', 'Called Strike'), auto, pitch(2, 'SL', 'Swinging Strike'), {'index': 3, 'isPitch': False, 'details': {}}]}]
    snapshot = {'gameData': {'game': {'pk': 7}}, 'liveData': {'plays': {'allPlays': plays}}}
    assert last_pitch(snapshot) == {'key': '7:41:3', 'pitch_type': 'SL', 'pitch_label': '슬라이더', 'speed_mph': 88.5,
                                    'result_label': '삼진', 'play_text': 'A strikes out swinging.', 'inning': 5, 'half': 'Bot'}
    plays[0]['about']['isComplete'] = False  # mid-PA: the call of the pitch, no plate-appearance result yet
    assert last_pitch(snapshot)['result_label'] == 'Swinging Strike' and last_pitch(snapshot)['play_text'] is None
    assert last_pitch({'gameData': {'game': {'pk': 7}}, 'liveData': {'plays': {'allPlays': []}}}) is None
    done = feed('849845_20260929_195940.json')
    done['liveData']['plays']['currentPlay']['about']['isComplete'] = True
    done['liveData']['linescore']['outs'] = 3
    assert parse_state(done, None)['status'] == 'between_innings' and 'last_pitch' in parse_state(done, None)


def test_between_batters_the_real_feed_still_names_the_finished_batter():
    """849844 (2026-10-02): for about ten seconds after a PA completes ``offense.batter`` is the batter who
    just finished and ``onDeck`` the next one; the new play then moves ``batter`` on."""
    finished = {'matchup': {'batter': {'id': 607208}}}
    stale = {'offense': {'batter': {'id': 607208}, 'onDeck': {'id': 656941}}}
    assert due_up(stale, finished) == 656941
    assert due_up({'offense': {'batter': {'id': 656941}, 'onDeck': {'id': 547180}}}, finished) == 656941  # already moved on
    assert due_up(stale, None) == 607208  # before the first pitch nobody has finished
    done = feed('849845_20260929_195940.json')
    done['liveData']['plays']['currentPlay']['about']['isComplete'] = True
    offense = done['liveData']['linescore']['offense']
    offense['onDeck'] = {'id': 592518}
    done['gameData']['players']['ID592518'] = {'fullName': 'Next Batter', 'batSide': {'code': 'S'}}
    state = parse_state(done, None)
    assert state['batter'] == {'id': 592518, 'name': 'Next Batter', 'side': 'R'} and state['situation']['balls'] == 0  # vs LHP Sale


def test_a_pitch_that_decided_the_pa_is_not_followed_by_a_pending_pitch():
    snapshot = feed('849845_20260929_195940.json')
    play = snapshot['liveData']['plays']['currentPlay']
    assert parse_state(snapshot, None)['status'] == 'ready'
    in_play = copy.deepcopy(snapshot)
    in_play['liveData']['plays']['currentPlay']['playEvents'][-1]['details']['isInPlay'] = True
    state = parse_state(in_play, None)
    assert state['status'] == 'resolving' and state['situation'] is not None and state['pitcher']['name'] == 'Chris Sale'
    for count in ({'strikes': 3}, {'balls': 4}):
        ended = copy.deepcopy(snapshot)
        ended['liveData']['plays']['currentPlay']['count'] = play['count'] | count
        assert parse_state(ended, None)['status'] == 'resolving'
    hit = copy.deepcopy(snapshot)
    hit['liveData']['plays']['currentPlay']['playEvents'][-1]['details']['call'] = {'code': 'H'}
    assert parse_state(hit, None)['status'] == 'resolving'
    armb = LiveArmB(lambda: FakePolicy())
    app = create_app(object(), live_feed=LiveFeed(lambda url, timeout: in_play, delay_s=0), live_policy=armb)
    with TestClient(app) as client:
        body = client.get('/api/live/849845/state').json()
    assert body['status'] == 'resolving' and body['recommendation'] is None and armb.status()['pending'] == 0


def test_feed_is_polled_faster_only_while_a_low_delay_viewer_is_watching():
    now, calls = [0.0], []
    snapshot = at(feed('849845_20260929_195940.json'), -100.0)
    config = LIVE_CONFIG | {'poll_interval_s': 10, 'fast_poll_s': 5, 'fast_poll_below_delay_s': 25}
    live = LiveFeed(lambda url, timeout: calls.append(now[0]) or snapshot, config=config, delay_s=30,
                    clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    for second in range(0, 31, 2):  # a 30 s viewer: the feed's own 10 s pace
        now[0] = float(second)
        live.snapshot(849845)
    assert calls == [0.0, 10.0, 20.0, 30.0]
    calls.clear()
    for second in range(32, 51, 2):  # a 10 s viewer joins: 5 s pace (first fetch at the next poll)
        now[0] = float(second)
        live.snapshot(849845, 10)
        live.snapshot(849845)
    assert calls == [36.0, 42.0, 48.0]
    calls.clear()
    for second in range(52, 111, 2):  # they leave: back to 10 s once their last visit is 30 s old
        now[0] = float(second)
        live.snapshot(849845)
    assert calls == [54.0, 60.0, 66.0, 72.0, 78.0, 88.0, 98.0, 108.0]  # the last low-delay visit was at 50


def test_game_tally_counts_only_pitches_before_the_served_one():
    pre = lambda code: {'status': 'ready' if code else 'unsupported', 'reason': None,
                        'recommendation': {'candidates': [{'rank': 1, 'pitch_type': code}]} if code else None}
    pitch = lambda key, code, thrown=None: {'key': key, 'pre': pre(code), 'actual': thrown and {'pitch_type': thrown}}
    armb = LiveArmB(lambda: None)
    armb._note({'key': '7:1:2', 'pitch': pitch('7:1:2', 'SL'), 'pa_pitches': [pitch('7:1:1', 'FF', 'FF')], 'previous': None})
    armb._note({'key': '7:2:1', 'pitch': pitch('7:2:1', None), 'pa_pitches': [],
                'previous': {'key': '7:1:2', 'actual': {'pitch_type': 'CH'}}})
    armb._note({'key': '7:10:1', 'pitch': pitch('7:10:1', 'FF'), 'pa_pitches': [],
                'previous': {'key': '7:2:1', 'actual': {'pitch_type': 'FF'}}})
    assert armb.tally('7:1:2') == {'pitches': 1, 'recommended': 1, 'same_type': 1}
    assert armb.tally('7:2:1') == {'pitches': 2, 'recommended': 2, 'same_type': 1}   # a later viewer state sees more
    assert armb.tally('7:10:1') == {'pitches': 3, 'recommended': 2, 'same_type': 1}  # 7:2:1 had no recommendation
    assert armb.tally('8:1:1') == {'pitches': 0, 'recommended': 0, 'same_type': 0}
