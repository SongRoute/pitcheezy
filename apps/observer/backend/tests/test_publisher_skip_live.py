"""``skip_unchanged_live``: an in-progress game file is rewritten only when what a viewer reads in it changed, while
health keeps advancing and its ``game_checks`` say the game was checked. Synthetic games only (``publish_fixture``)."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app import publisher
from publish_fixture import LIVE_GAME, snap
from test_publisher import make, world, candidates, unconfirmed  # noqa: F401 - fixtures
from test_publisher_upload import endpoint, attach  # noqa: F401 - fixtures

LIVE_FILE = f'v1/game/{LIVE_GAME}.json'
CONFIG = Path(publisher.__file__).resolve().parents[2]/'publish_config.json'


def check_of(health, game=LIVE_GAME):
    return health['game_checks'][str(game)]


def body_of(world):
    return {key: value for key, value in world.game().items() if key != 'publish'}


# ---------- configuration ----------
def test_the_switch_is_on_in_the_committed_config_and_on_when_absent(tmp_path):
    assert json.loads(CONFIG.read_text())['skip_unchanged_live'] is True
    assert publisher.load_config(env={})['skip_unchanged_live'] is True
    bare = {key: value for key, value in json.loads(CONFIG.read_text()).items() if not key.startswith('skip_unchanged_live')}
    (tmp_path/'c.json').write_text(json.dumps(bare))
    assert publisher.load_config(tmp_path/'c.json', env={})['skip_unchanged_live'] is True
    assert publisher.load_config(env={}, skip_unchanged_live=False)['skip_unchanged_live'] is False
    for wrong in ('true', 1, 0, 'no'):
        with pytest.raises(ValueError, match='skip_unchanged_live'):
            publisher.load_config(env={}, skip_unchanged_live=wrong)


# ---------- the publisher ----------
def test_an_unchanged_live_game_is_not_rewritten_while_health_and_its_check_advance(world):
    world.cycle()
    written = world.path(LIVE_FILE).read_bytes()
    sequence = json.loads(written)['publish']['sequence']
    assert json.loads(written)['publish']['final'] is False and world.game()['game']['status'] == 'live'
    polls = lambda: sum('/feed/live' in url for url in world.upstream.calls)
    before, checks = polls(), []
    for _ in range(4):
        health = world.cycle(settle=False)
        checks.append(check_of(health))
        assert world.path(LIVE_FILE).read_bytes() == written  # same bytes: no write at all
        assert world.read('v1/health.json')['sequence'] == health['sequence'] > sequence
        assert LIVE_FILE not in world.publisher._written
    assert polls() == before+4  # still built (and upstream still asked) every cycle: only the write is skipped
    assert [c['file_sequence'] for c in checks] == [sequence]*4 and all(c['feed_ok'] for c in checks)
    assert [c['sequence_checked'] for c in checks] == sorted({c['sequence_checked'] for c in checks})  # strictly rising
    assert len({c['checked_at'] for c in checks}) == 4  # the check time moves every cycle: that is the freshness


def test_a_new_pitch_is_written_at_once_with_the_current_sequence(world):
    world.cycle()
    old = world.game()
    health = world.cycle(snap('20261005_100100', second=True, wall=world.now[0]+7), settle=False)
    new = world.game()
    assert len(new['pitches']) == len(old['pitches'])+1
    assert new['publish']['sequence'] == health['sequence'] == check_of(health)['file_sequence']


def test_switched_off_the_live_game_is_rewritten_every_cycle_as_before(make):
    world = make(skip_unchanged_live=False)
    world.cycle()
    for _ in range(3):
        health = world.cycle(settle=False)
        assert world.game()['publish']['sequence'] == health['sequence'] == check_of(health)['file_sequence']


def test_a_display_state_change_alone_is_written(world):
    """Same snapshot, same pitches: only the display state moves (the feed stays down long enough to disconnect)."""
    world.cycle()
    before = world.game()
    health = world.cycle(OSError('upstream down'), after=400.0, settle=False)
    after = world.game()
    assert after['pitches'] == before['pitches'] and after['pas'] == before['pas']
    assert (before['display_state']['state'], after['display_state']['state']) == ('recommended', 'disconnected')
    assert after['publish']['sequence'] == health['sequence'] == check_of(health)['file_sequence']
    assert check_of(health)['feed_ok'] is False and LIVE_GAME in health['feed_failing_games']


def test_a_correction_of_a_past_call_is_written(world):
    world.cycle()
    before = world.game()
    health = world.cycle(snap('20261005_100000', call='S'), settle=False)
    after = world.game()
    assert [p['actual'] for p in after['pitches']] != [p['actual'] for p in before['pitches']]
    assert after['publish']['sequence'] == health['sequence'] == check_of(health)['file_sequence']


def test_a_short_feed_outage_is_named_in_health_even_when_the_file_is_not_rewritten(world):
    world.cycle()
    health = world.cycle(OSError('upstream down'), after=12.0, settle=False)
    assert check_of(health)['feed_ok'] is False and LIVE_GAME in health['feed_failing_games']
    # whatever the file says, its recorded digest is the one of what is on disk now
    assert publisher.live_digest(body_of(world)) == world.publisher.state['files'][LIVE_FILE]['digest']


def test_a_missing_or_foreign_local_file_is_written_again(world):
    world.cycle()
    world.path(LIVE_FILE).unlink()
    health = world.cycle(settle=False)
    assert world.game()['publish']['sequence'] == health['sequence']
    # a file that is not the one the state recorded (another sequence in its publish block) is replaced as well
    text = world.path(LIVE_FILE).read_text()
    world.path(LIVE_FILE).write_text(text.replace(f'"sequence":{health["sequence"]}', '"sequence":1'))
    health = world.cycle(settle=False)
    assert world.game()['publish']['sequence'] == health['sequence']
    health = world.cycle(settle=False)
    assert world.game()['publish']['sequence'] == health['sequence']-1  # and then skipped again


def test_a_final_game_still_uses_the_final_digest_and_is_written_once(world):
    world.cycle()
    world.cycle(snap('20261005_100100', half='final'), settle=False)
    final = world.path(LIVE_FILE).read_bytes()
    assert json.loads(final)['publish']['final'] is True
    assert world.publisher.state['files'][LIVE_FILE]['digest'] == publisher.game_digest(
        {k: v for k, v in json.loads(final).items() if k != 'publish'})
    for _ in range(3):
        world.cycle(settle=False)
    assert world.path(LIVE_FILE).read_bytes() == final


def test_an_unconfirmed_pitch_and_its_admission_are_both_written(world):
    world.cycle()
    health = world.cycle(unconfirmed(), settle=False)
    assert world.game()['display_state']['state'] == 'unconfirmed' and world.game()['publish']['sequence'] == health['sequence']
    health = world.cycle(snap('20261005_100100', second=True, wall=world.now[0]-5), settle=False)
    assert world.game()['receipts']['unconfirmed'] == 0 and world.game()['publish']['sequence'] == health['sequence']


# ---------- uploads ----------
def test_a_skipped_write_queues_no_upload_and_keeps_the_conditional_state(world, endpoint, monkeypatch):
    sink = attach(world, endpoint, monkeypatch)
    world.cycle()
    puts = lambda path: [r for r in endpoint.requests if r[:2] == ('PUT', path)]
    sent, tag = len(puts(LIVE_FILE)), endpoint.tags[LIVE_FILE]
    assert endpoint.objects[LIVE_FILE] == world.path(LIVE_FILE).read_bytes() and world.publisher.state['upload_pending'] == {}
    revision = world.publisher.state['upload_revision']
    for _ in range(3):
        world.cycle(settle=False)
        assert world.publisher.state['upload_pending'] == {} and LIVE_FILE not in world.publisher.state['upload_intents']
        assert LIVE_FILE not in world.publisher.state['upload_attempts']
    assert len(puts(LIVE_FILE)) == sent and not [r for r in endpoint.requests[-12:] if r[1] == LIVE_FILE]
    assert endpoint.tags[LIVE_FILE] == tag  # the remote copy and its ETag are untouched
    assert len(puts('v1/health.json')) >= 4  # health still goes out every cycle
    assert world.publisher.state['upload_revision'] > revision  # ...as intents of other files, not of the game
    # the next real change goes out as a conditional PUT on the ETag the remote still has
    world.cycle(snap('20261005_100100', second=True, wall=world.now[0]+7), settle=False)
    last = puts(LIVE_FILE)[-1]
    assert last[3].get('If-Match') == tag and endpoint.objects[LIVE_FILE] == world.path(LIVE_FILE).read_bytes()
    assert sink.pending == {}


# ---------- what the live digest counts ----------
# Everything not listed here must change the digest when it changes (checked leaf by leaf on real published bodies).
LIVE_EXEMPT = {'/feed/as_of', '/feed/stale_s'} | {f'/display_state/feed/{key}' for key in (
    'lag_max_s', 'lag_measured', 'last_ok_at', 'failing_since', 'failing_for_s', 'received_at')}
FINAL_EXEMPT = {'/feed'} | {f'/display_state/feed/{key}' for key in (
    'ok', 'delayed', 'lag_s', 'lag_max_s', 'lag_measured', 'last_ok_at', 'failing_since', 'failing_for_s', 'received_at')}


def leaves(value, pointer=''):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from leaves(item, f'{pointer}/{key}')
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from leaves(item, f'{pointer}/{index}')
    else:
        yield pointer


def changed(body, pointer):
    out = copy.deepcopy(body)
    *parents, last = pointer.strip('/').split('/')
    holder = out
    for part in parents:
        holder = holder[int(part)] if isinstance(holder, list) else holder[part]
    key = int(last) if isinstance(holder, list) else last
    old = holder[key]
    holder[key] = ('other' if old != 'other' else 'another') if not isinstance(old, bool) else not old
    if isinstance(old, (int, float)) and not isinstance(old, bool):
        holder[key] = old+1
    elif old is None:
        holder[key] = 'set'
    return out


def exempt(pointer, rules):
    return any(pointer == rule or pointer.startswith(rule+'/') for rule in rules)


@pytest.fixture
def bodies(make):
    """Published live bodies in several states: recommended, unconfirmed, disconnected (and a delayed variant)."""
    world = make()
    world.cycle()
    out = [body_of(world)]
    world.cycle(unconfirmed())
    out.append(body_of(world))
    world.cycle(OSError('upstream down'), after=400.0)
    out.append(body_of(world))
    return out


def test_every_viewer_visible_field_of_a_live_body_changes_the_live_digest(bodies):
    tried = 0
    for body in bodies:
        base = publisher.live_digest(body)
        for pointer in leaves(body):
            if exempt(pointer, LIVE_EXEMPT) or (pointer == '/display_state/feed/lag_s' and not body['display_state']['feed']['delayed']):
                assert publisher.live_digest(changed(body, pointer)) == base, pointer
            else:
                assert publisher.live_digest(changed(body, pointer)) != base, pointer
                tried += 1
    assert tried > 300


def test_the_final_digest_is_unchanged_by_this_switch(bodies):
    for body in bodies:
        base = publisher.game_digest(body)
        for pointer in leaves(body):
            same = publisher.game_digest(changed(body, pointer)) == base
            assert same == exempt(pointer, FINAL_EXEMPT), pointer


def test_the_connection_facts_a_viewer_reads_count_and_lag_only_while_delayed(bodies):
    body = bodies[0]
    base = publisher.live_digest(body)
    assert body['display_state']['feed']['delayed'] is False
    for patch in ({'feed': body['feed'] | {'ok': not body['feed']['ok']}},
                  {'display_state': body['display_state'] | {'feed': body['display_state']['feed'] | {'ok': False}}},
                  {'display_state': body['display_state'] | {'feed': body['display_state']['feed'] | {'delayed': True}}}):
        assert publisher.live_digest(body | patch) != base
    quiet = body | {'display_state': body['display_state'] | {'feed': body['display_state']['feed'] | {'lag_s': 99.9}}}
    assert publisher.live_digest(quiet) == base  # not delayed: the lag is not shown, so it does not count
    delayed = body | {'display_state': body['display_state'] | {'feed': body['display_state']['feed'] | {'delayed': True, 'lag_s': 40.0}}}
    later = delayed | {'display_state': delayed['display_state'] | {'feed': delayed['display_state']['feed'] | {'lag_s': 41.0}}}
    assert publisher.live_digest(later) != publisher.live_digest(delayed)  # "N초 넘게 늦음" changes: written
    # the clock alone never counts
    ticking = body | {'feed': body['feed'] | {'as_of': '2030-01-01T00:00:00Z', 'stale_s': 1234.5},
                      'display_state': body['display_state'] | {'feed': body['display_state']['feed'] | {
                          'received_at': '2030-01-01T00:00:00Z', 'last_ok_at': '2030-01-01T00:00:00Z', 'lag_max_s': 3.0}}}
    assert publisher.live_digest(ticking) == base
