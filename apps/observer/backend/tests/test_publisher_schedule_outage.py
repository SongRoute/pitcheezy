"""A schedule outage (CL-03a, rehearsal 849817): when the schedule cannot be fetched the live game's list card
disappears (``live_ok`` false), and before this fix the game was no longer checked, so health's game_checks
stood still at the pre-outage values (feed_ok true) even while the feed was down too. Games the last good list
showed are now still checked every cycle; a feed that is down is reported as such, ``schedule_unavailable``
names the schedule fault, nothing new is invented, and normal behaviour resumes when the schedule answers.
Synthetic games only (``publish_fixture``); no network."""
from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish_fixture
from publish_fixture import ARCHIVE_GAME, LIVE_DATE, LIVE_GAME, snap
from test_publisher import make, world  # noqa: F401 - fixtures

LIVE_FILE = f'v1/game/{LIVE_GAME}.json'
AFTER = 25.0  # past the feed's poll interval and the 20 s schedule cache of the game list


@pytest.fixture
def schedule(monkeypatch):
    """Switch for the fake upstream's schedule answers: ``schedule.down = True`` makes them fail (dead proxy)."""
    switch = type('Switch', (), {'down': False})()
    original = publish_fixture.Upstream.__call__

    def call(self, url, timeout):
        if switch.down and 'schedule' in url:
            self.calls.append(url)
            raise OSError('schedule down')
        return original(self, url, timeout)
    monkeypatch.setattr(publish_fixture.Upstream, '__call__', call)
    return switch


def polls(world):
    return sum('/feed/live' in url for url in world.upstream.calls)


def codes(health, code='schedule_unavailable'):
    return sorted(item['game_pk'] for item in health['unavailable_games'] if item['code'] == code)


def epoch(stamp):
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp()


def test_schedule_and_feed_down_the_live_game_is_reported_failing_and_its_check_advances(world, schedule):
    good = world.cycle(after=AFTER)
    before = good['game_checks'][str(LIVE_GAME)]
    assert before['feed_ok'] is True and good['feed_failing_games'] == []
    count = polls(world)
    schedule.down = True
    world.upstream.body = OSError('upstream down')
    checks = []
    for _ in range(3):
        health = world.cycle(after=AFTER, settle=False)
        assert world.read(f'v1/games/{LIVE_DATE}.json')['live_ok'] is False  # the list itself says the schedule is out
        check = health['game_checks'][str(LIVE_GAME)]
        checks.append(check)
        assert check['feed_ok'] is False and LIVE_GAME in health['feed_failing_games']
        assert check['sequence_checked'] == health['sequence'] and check['checked_at'] == health['published_at']
        assert codes(health) == [LIVE_GAME]
        assert ARCHIVE_GAME not in health['feed_failing_games'] and ARCHIVE_GAME not in codes(health)
        assert set(health['game_checks']) == set(good['game_checks'])  # nothing invented, nothing dropped
    assert polls(world) > count  # the feed was really asked during the outage
    assert len({c['checked_at'] for c in checks}) == 3 and checks[0]['checked_at'] > before['checked_at']
    # the game file shows the failure the way a feed failure is shown today (top-level feed and display state)
    body = world.game()
    assert body['feed']['ok'] is False and body['display_state']['feed']['ok'] is False
    assert body['game']['status'] == 'live' and body['publish']['final'] is False


def test_schedule_down_alone_the_live_game_keeps_being_checked_and_stays_ok(world, schedule):
    good = world.cycle(after=AFTER)
    written = world.path(LIVE_FILE).read_bytes()
    file_sequence = good['game_checks'][str(LIVE_GAME)]['file_sequence']
    schedule.down = True
    count, stamps = polls(world), []
    for _ in range(3):
        health = world.cycle(after=AFTER, settle=False)
        check = health['game_checks'][str(LIVE_GAME)]
        stamps.append(check['checked_at'])
        assert check['feed_ok'] is True and health['feed_failing_games'] == []  # the game's own feed is fine
        assert check['checked_at'] == health['published_at'] and check['file_sequence'] == file_sequence
        assert codes(health) == [LIVE_GAME]  # ...but the schedule fault is named
        assert world.path(LIVE_FILE).read_bytes() == written  # skip_unchanged_live: nothing new, no write
    assert polls(world) == count+3 and len(set(stamps)) == 3
    # a new pitch during the schedule outage is still written at once
    health = world.cycle(snap('20261005_100100', second=True, wall=world.now[0]+AFTER-5), after=AFTER, settle=False)
    assert world.game()['publish']['sequence'] == health['sequence'] == health['game_checks'][str(LIVE_GAME)]['file_sequence']


def test_recovery_resumes_normal_behaviour(world, schedule):
    world.cycle(after=AFTER)
    schedule.down = True
    world.upstream.body = OSError('upstream down')
    down = world.cycle(after=AFTER, settle=False)
    assert down['game_checks'][str(LIVE_GAME)]['feed_ok'] is False
    schedule.down = False
    world.upstream.body = snap('20261005_100100', second=True, wall=world.now[0]+AFTER-5)
    health = world.cycle(after=AFTER, settle=False)
    check = health['game_checks'][str(LIVE_GAME)]
    assert world.read(f'v1/games/{LIVE_DATE}.json')['live_ok'] is True
    assert [g['game_pk'] for g in world.read(f'v1/games/{LIVE_DATE}.json')['games']] == [LIVE_GAME]
    assert check['feed_ok'] is True and health['feed_failing_games'] == [] and health['unavailable_games'] == []
    assert check['file_sequence'] == health['sequence'] == world.game()['publish']['sequence']
    assert world.game()['feed']['ok'] is True
    assert world.publisher.state['schedule_live'][LIVE_DATE] == [LIVE_GAME]


def test_the_last_good_list_survives_a_restart_during_the_outage(world, schedule):
    world.cycle(after=AFTER)
    schedule.down = True
    world.upstream.body = OSError('upstream down')
    world.cycle(after=AFTER, settle=False)
    world.start()  # a publisher restart in the middle of the outage
    health = world.cycle(after=AFTER, settle=False)
    assert health['game_checks'][str(LIVE_GAME)]['feed_ok'] is False and LIVE_GAME in health['feed_failing_games']
    assert health['game_checks'][str(LIVE_GAME)]['checked_at'] == health['published_at']


def test_no_good_list_yet_nothing_is_invented(make, schedule):
    schedule.down = True
    world = make()
    health = world.cycle(after=AFTER, settle=False)
    assert str(LIVE_GAME) not in health['game_checks'] and codes(health) == []
    assert not world.path(LIVE_FILE).exists() and polls(world) == 0


def test_a_finished_game_carried_over_the_outage_is_not_flagged(world, schedule):
    world.cycle(after=AFTER)
    world.cycle(snap('20261005_100100', half='final', wall=world.now[0]+AFTER-5), after=AFTER)
    assert world.game()['publish']['final'] is True
    schedule.down = True
    health = world.cycle(after=AFTER, settle=False)
    assert codes(health) == [] and health['feed_failing_games'] == []
    assert health['game_checks'][str(LIVE_GAME)]['feed_ok'] is True


def test_a_corrupt_remembered_list_is_ignored(world, schedule):
    world.cycle(after=AFTER)
    world.publisher.state['schedule_live'] = {LIVE_DATE: ['x']}
    schedule.down = True
    health = world.cycle(after=AFTER, settle=False)
    assert codes(health) == []  # dropped, not trusted; the old freeze is the worst case
    world.publisher.state['schedule_live'] = 'garbage'
    schedule.down = False
    world.cycle(after=AFTER, settle=False)
    assert world.publisher.state['schedule_live'][LIVE_DATE] == [LIVE_GAME]
