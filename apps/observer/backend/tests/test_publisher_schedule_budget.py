"""CL-03c (review R-pubfix non-blocking #3, #6): a hanging schedule cannot hold a publish cycle for minutes, and
the schedule-outage carry holds for a doubleheader and for games on two live dates with one game-body request
per carried game per cycle. Fake upstreams and fake builders only; no network, no real waiting (a hanging
request is simulated by moving the publisher's monotonic clock by the feed's timeout before it fails)."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_fixture
from observer_app import publisher
from observer_app.live_feed import LIVE_CONFIG
from publish_fixture import LIVE_DATE, LIVE_GAME
from test_publisher import make, world  # noqa: F401 - fixtures

AFTER = 25.0  # past the feed's poll interval and the 20 s schedule cache of the game list
TIMEOUT = LIVE_CONFIG['timeout_s']
ATTEMPTS = LIVE_CONFIG['retries']+1


# ---------- a hanging schedule (the real list builder and live feed, fake transport) ----------
class Hang:
    """``mode``: 'ok' (answers), 'hang' (each schedule request takes ``timeout_s`` and then fails, as a
    black-holed proxy does), 'slow' (answers after ``slow_s``)."""

    def __init__(self):
        self.mode, self.slow_s, self.clock = 'ok', 0.0, [0.0]

    def install(self, world, monkeypatch):
        world.publisher.monotonic = lambda: self.clock[0]
        original = publish_fixture.Upstream.__call__
        hang = self

        def call(upstream, url, timeout):
            if 'schedule' in url and hang.mode == 'hang':
                upstream.calls.append(url)
                hang.clock[0] += timeout
                raise TimeoutError('timed out')
            if 'schedule' in url and hang.mode == 'slow':
                hang.clock[0] += hang.slow_s
            return original(upstream, url, timeout)
        monkeypatch.setattr(publish_fixture.Upstream, '__call__', call)
        return self


def schedule_requests(world):
    return sum('schedule' in url for url in world.upstream.calls)


def codes(health, code='schedule_unavailable'):
    return sorted(item['game_pk'] for item in health['unavailable_games'] if item['code'] == code)


def outage_cycle(world, hang):
    asked, started = schedule_requests(world), hang.clock[0]
    health = world.cycle(after=AFTER, settle=False)
    return health, schedule_requests(world)-asked, hang.clock[0]-started


def test_the_budget_is_on_by_default_and_validated():
    assert publisher.load_config(env={})['schedule_budget_s'] == publisher.SCHEDULE_BUDGET_S
    for wrong in (0, -1, 601, 'x', True, float('nan')):
        with pytest.raises(ValueError, match='schedule_budget_s'):
            publisher.load_config(env={}, schedule_budget_s=wrong)


def test_a_hanging_schedule_costs_one_retried_request_then_one_single_attempt_probe_per_cycle(world, monkeypatch):
    hang = Hang().install(world, monkeypatch)
    world.cycle(after=AFTER)
    hang.mode = 'hang'
    first, asked, spent = outage_cycle(world, hang)
    # the request in flight when the outage starts runs with the feed's retries; nothing after it is asked
    assert asked == ATTEMPTS and spent == ATTEMPTS*TIMEOUT
    assert first['game_checks'][str(LIVE_GAME)]['checked_at'] == first['published_at'] and codes(first) == [LIVE_GAME]
    for _ in range(3):
        health, asked, spent = outage_cycle(world, hang)
        assert asked == 1 and spent == TIMEOUT  # one probe, one attempt
        assert health['game_checks'][str(LIVE_GAME)]['checked_at'] == health['published_at']
        assert codes(health) == [LIVE_GAME]
    # the schedule answers again: the probe ends the down state and the same cycle asks every live list
    hang.mode = 'ok'
    health, asked, _ = outage_cycle(world, hang)
    assert asked == 2*len(world.publisher.live_dates(world.now[0]))
    assert codes(health) == [] and world.read(f'v1/games/{LIVE_DATE}.json')['live_ok'] is True


def test_without_the_budget_a_hanging_schedule_holds_the_cycle_for_every_request(world, monkeypatch):
    hang = Hang().install(world, monkeypatch)
    world.cycle(after=AFTER)
    world.publisher.config['schedule_budget_s'] = None  # the behaviour before 2026-10-19
    hang.mode = 'hang'
    _, asked, spent = outage_cycle(world, hang)
    live_dates = len(world.publisher.live_dates(world.now[0]))
    assert asked == live_dates*2*ATTEMPTS and spent == asked*TIMEOUT  # minutes of one cycle (the review's case)


def test_a_slow_schedule_stops_being_asked_once_the_cycle_budget_is_spent(world, monkeypatch):
    hang = Hang().install(world, monkeypatch)
    world.cycle(after=AFTER)
    hang.mode, hang.slow_s = 'slow', 5.0
    health, asked, spent = outage_cycle(world, hang)
    budget = publisher.SCHEDULE_BUDGET_S
    assert asked == 2 and spent == 10.0  # started at 0 and 5 (< 8); the third would start at 10 and is not asked
    assert spent < budget+hang.slow_s
    # the live date with the remembered game was asked first: its list is good and the game is not flagged
    assert world.read(f'v1/games/{LIVE_DATE}.json')['live_ok'] is True and codes(health) == []
    assert health['game_checks'][str(LIVE_GAME)]['checked_at'] == health['published_at']


def test_a_restarted_publisher_rewraps_the_feed_schedule_once(world, monkeypatch):
    hang = Hang().install(world, monkeypatch)
    world.cycle(after=AFTER)
    world.start()
    world.start()
    assert world.feed.schedule.__closure__ is not None
    original = world.feed._unguarded_schedule
    assert getattr(original, '__self__', None) is world.feed  # the bound method, not an older wrapper
    hang.install(world, monkeypatch)
    hang.mode = 'hang'
    _, asked, _ = outage_cycle(world, hang)
    assert asked == ATTEMPTS


# ---------- doubleheader, two dates, request count (fake builders) ----------
D1, D2 = '2026-10-19', '2026-10-20'
GAMES = {D1: [1001, 1002], D2: [2001]}  # 1001/1002: a doubleheader


class Bodies:
    """``service_api.ServiceBodies`` as far as the publisher uses it."""

    def __init__(self):
        self.schedule_ok, self.feed_ok, self.final, self.calls = True, True, set(), []

    def dates(self):
        return {'schema': 'pitcheezy-service-dates-v2', 'source': {'kind': 'archive', 'session_id': None}, 'dates': []}

    def games(self, date):
        cards = [{'game_pk': pk, 'status': 'final' if pk in self.final else 'live', 'embargoed': False,
                  'game_number': n+1, 'double_header': len(GAMES.get(date, [])) > 1}
                 for n, pk in enumerate(GAMES.get(date, [])) if self.schedule_ok or pk in self.final]
        return {'schema': 'pitcheezy-service-games-v2', 'source': {'kind': 'archive', 'session_id': None},
                'date': date, 'games': cards, 'live_ok': self.schedule_ok}

    def game(self, game_pk):
        self.calls.append(game_pk)
        final = game_pk in self.final
        state = 'final' if final else 'recommended'
        return {'schema': 'pitcheezy-service-game-v2', 'source': {'kind': 'live', 'session_id': None},
                'game': {'game_pk': game_pk, 'status': 'final' if final else 'live'},
                'feed': {'ok': self.feed_ok}, 'display_state': {'state': state, 'feed': {'ok': self.feed_ok}},
                'current': {'phase': 'final' if final else 'pre_pitch'}, 'receipts': {'unconfirmed': 0},
                'pitches': [], 'next': None}

    def archived(self):
        return []

    def embargoed(self):
        return []


@pytest.fixture
def fake(tmp_path):
    bodies, now = Bodies(), [1_792_400_000.0]
    config = publisher.load_config(env={}, out_dir=str(tmp_path/'out'))
    made = publisher.Publisher(bodies, publisher.LocalDirSink(tmp_path/'out'), config=config,
                               state_path=tmp_path/'state.json', wall=lambda: now[0], dates=[D1, D2])

    def cycle():
        now[0] += AFTER
        bodies.calls.clear()
        return made.cycle()
    yield bodies, made, cycle
    made.close()


ALL = [1001, 1002, 2001]


def test_a_doubleheader_and_two_live_dates_are_all_carried_with_one_request_each(fake):
    bodies, made, cycle = fake
    good = cycle()
    assert sorted(bodies.calls) == ALL and good['feed_failing_games'] == []
    assert made.state['schedule_live'] == {D1: [1001, 1002], D2: [2001]}
    bodies.schedule_ok = bodies.feed_ok = False
    for _ in range(3):
        health = cycle()
        assert sorted(bodies.calls) == ALL  # every carried game asked exactly once per cycle: no more
        assert health['feed_failing_games'] == ALL and codes(health) == ALL
        for pk in ALL:
            check = health['game_checks'][str(pk)]
            assert check['feed_ok'] is False and check['checked_at'] == health['published_at']
        assert set(health['game_checks']) == {str(pk) for pk in ALL}  # nothing invented
    bodies.schedule_ok = bodies.feed_ok = True
    health = cycle()
    assert health['feed_failing_games'] == [] and codes(health) == [] and sorted(bodies.calls) == ALL
    assert all(health['game_checks'][str(pk)]['feed_ok'] for pk in ALL)


def test_the_finished_game_of_a_doubleheader_is_neither_polled_nor_flagged_during_the_outage(fake):
    bodies, made, cycle = fake
    bodies.final = {1001}
    cycle()
    assert json.loads(made.sink.read('v1/game/1001.json'))['publish']['final'] is True
    bodies.schedule_ok = bodies.feed_ok = False
    for _ in range(3):
        health = cycle()
        assert sorted(bodies.calls) == [1002, 2001]  # the settled game waits for its settled recheck
        assert codes(health) == [1002, 2001] and health['feed_failing_games'] == [1002, 2001]
        assert health['game_checks']['1001']['feed_ok'] is True


def test_schedule_down_on_one_date_only_carries_only_that_date(fake):
    bodies, made, cycle = fake
    cycle()
    original = bodies.games

    def games(date):
        body = original(date)
        return body | {'games': [], 'live_ok': False} if date == D2 else body
    bodies.games = games
    health = cycle()
    assert codes(health) == [2001] and sorted(bodies.calls) == ALL
    assert made.state['schedule_live'][D1] == [1001, 1002] and made.state['schedule_live'][D2] == [2001]
