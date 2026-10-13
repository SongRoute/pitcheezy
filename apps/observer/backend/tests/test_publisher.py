"""The public publisher: every published file is the body the service route answers (plus one ``publish`` key),
the display gate and the ledger-before-display rule hold in the files, files are replaced atomically, the sequence
never goes back, a final game is written once, and demo/operator data never reaches the output.
Synthetic games and a fake policy only (``publish_fixture``); no network, no T7, no model, no port."""
import json
import os
from pathlib import Path
import re
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app import publish_schema, publisher
from observer_app.demo_clock import DemoClock
from observer_app.publisher import LocalDirSink
import publish_fixture
from publish_fixture import ARCHIVE_DATE, ARCHIVE_GAME, LIVE_DATE, LIVE_GAME, World, pregame, snap, strip, unidentified
from live_fixture import listed_dir

LIVE_FILE, ARCHIVE_FILE = f'v1/game/{LIVE_GAME}.json', f'v1/game/{ARCHIVE_GAME}.json'
CONTRACT_DIR = Path(__file__).resolve().parents[2]/'publish_contract/v1'
ALLOWED = re.compile(r'v1/(manifest\.json|health\.json|dates\.json|games/\d{4}-\d{2}-\d{2}\.json|game/\d+\.json|setup/\d+\.json)')


@pytest.fixture
def make(tmp_path):
    worlds = []

    def build(**options):
        folder = tmp_path/f'world{len(worlds)}'
        folder.mkdir()
        worlds.append(World(folder, **options))
        return worlds[-1]
    yield build
    for world in worlds:
        world.close()


@pytest.fixture
def world(make):
    return make()


def unconfirmed(ts='20261005_100100', **options):
    """The open at-bat's second pitch arrives without a provider identity: the store does not admit it."""
    return unidentified(snap(ts, second=True, **options))


def candidates(body):
    return [c['pitch_type'] for c in ((body.get('next') or {}).get('rec') or {}).get('candidates') or []]


# ---------- the published body is the route's body ----------
def test_health_checks_track_unchanged_final_and_advancing_live_file(world):
    first = world.cycle()
    for game in (LIVE_GAME, ARCHIVE_GAME):
        check = first['game_checks'][str(game)]
        assert check['file_sequence'] == world.game(game)['publish']['sequence'] and check['feed_ok'] is True
        assert check['sequence_checked'] <= first['sequence']
    assert first['game_checks'][str(LIVE_GAME)]['sequence_checked'] == first['sequence']
    second = world.cycle(after=30)
    assert second['game_checks'][str(LIVE_GAME)]['file_sequence'] > first['game_checks'][str(LIVE_GAME)]['file_sequence']
    assert second['game_checks'][str(ARCHIVE_GAME)] == first['game_checks'][str(ARCHIVE_GAME)]  # no archive recheck
    failed = world.cycle(OSError('upstream down'), after=30, settle=False)
    check = failed['game_checks'][str(LIVE_GAME)]
    assert check['sequence_checked'] == failed['sequence'] and check['feed_ok'] is False
    assert check['file_sequence'] == world.game(LIVE_GAME)['publish']['sequence']


@pytest.mark.parametrize('stage', ['pregame', 'live', 'unconfirmed', 'between', 'final'])
def test_a_published_game_file_is_the_body_the_route_answers(make, stage):
    first = {'pregame': pregame, 'live': snap, 'unconfirmed': unconfirmed,
             'between': lambda: snap(half='between'), 'final': lambda: snap(half='final')}[stage]()
    world = make(first=first)
    world.cycle()
    published = world.game()
    assert strip(published) == world.served(f'/api/service/games/{LIVE_GAME}')  # same moment: the clock did not move
    assert published['publish'] == {'contract': 'v1', 'sequence': published['publish']['sequence'],
                                    'published_at': '2026-10-05T10:20:12Z', 'final': stage == 'final'}
    assert published['source'] == {'kind': 'live', 'session_id': None}
    assert published['display_state']['state'] == {'pregame': 'pregame', 'live': 'recommended', 'unconfirmed': 'unconfirmed',
                                                   'between': 'waiting', 'final': 'final'}[stage]


def test_dates_game_lists_and_the_archive_game_are_the_route_bodies(world):
    world.cycle()
    assert strip(world.read('v1/dates.json')) == world.served('/api/service/dates')
    for date in (ARCHIVE_DATE, LIVE_DATE, '2026-10-04', '2026-10-06'):
        assert strip(world.read(f'v1/games/{date}.json')) == world.served('/api/service/games', date=date)
    assert [g['game_pk'] for g in world.read(f'v1/games/{LIVE_DATE}.json')['games']] == [LIVE_GAME]
    archive = world.game(ARCHIVE_GAME)
    assert strip(archive) == world.served(f'/api/service/games/{ARCHIVE_GAME}')
    assert archive['publish']['final'] is True and len(archive['pitches']) == 10
    assert world.files() == sorted(['v1/dates.json', 'v1/health.json', 'v1/manifest.json', ARCHIVE_FILE, LIVE_FILE,
                                    f'v1/setup/{ARCHIVE_GAME}.json',
                                    *(f'v1/games/{d}.json' for d in (ARCHIVE_DATE, '2026-10-04', LIVE_DATE, '2026-10-06'))])


def test_the_publisher_drives_the_upstream_poll_without_any_viewer(world):
    assert world.upstream.calls == []
    world.cycle(settle=False)
    assert sum('/feed/live' in url for url in world.upstream.calls) == 1  # nobody asked the routes
    world.cycle(settle=False)
    assert sum('/feed/live' in url for url in world.upstream.calls) == 2


def test_manifest_and_health_state_the_contract(world):
    health = world.cycle()
    manifest, on_disk = world.read('v1/manifest.json'), world.read('v1/health.json')
    assert strip(on_disk) == health and health['sequence'] == on_disk['publish']['sequence'] == 2
    assert (manifest['contract'], manifest['commit'], manifest['commit_dirty']) == ('v1', '0'*40, False)
    assert (manifest['publish_interval_s'], manifest['fixed_delay_s']) == (world.config['publish_interval_s'], 0)
    assert manifest['published_at'] == health['published_at'] == '2026-10-05T10:20:12Z'
    assert manifest['live_dates'] == ['2026-10-04', '2026-10-05', '2026-10-06']
    assert health['unavailable_games'] == []


# ---------- the gate holds in the files ----------
def test_an_unconfirmed_pitch_is_published_blank_and_no_candidate_leaves(world):
    world.cycle()
    assert candidates(world.game()) == ['SL']
    world.cycle(unconfirmed())
    body = world.game()
    last = body['pitches'][-1]
    assert last['receipt']['status'] == 'unconfirmed' and body['receipts']['unconfirmed'] == 1
    assert {k: v for k, v in last['actual'].items() if v is not None} == {'call_label': '확인 중', 'result_label': '확인 중'}
    assert body['display_state']['state'] == 'unconfirmed' and body['display_state']['recommendation_shown'] is False
    assert candidates(body) == [] and 'SL' not in json.dumps(body['next'])
    assert body['pas'][-1]['complete'] is False and body['pas'][-1]['event'] is None  # open at-bat: no outcome


def test_no_candidate_is_published_in_any_state_that_shows_none(make, monkeypatch):
    for first in (pregame(), snap(half='between'), snap(half='final')):
        world = make(first=first)
        world.cycle()
        assert candidates(world.game()) == [] and world.game()['display_state']['recommendation_shown'] is False
    world = make()
    world.policy.status = 'UNSUPPORTED_PITCHER'
    world.cycle()
    assert candidates(world.game()) == [] and world.game()['display_state']['state'] == 'unsupported'
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    off = make()
    off.cycle()
    assert candidates(off.game()) == [] and off.game()['display_state']['reason'] == 'recommender_off'


def test_one_fixed_delay_holds_back_every_pitch_newer_than_it(make):
    world = make(fixed_delay_s=30)
    health = world.cycle(settle=False)  # the only snapshot's event is 17 s old: nothing may be shown yet
    assert not world.path(LIVE_FILE).exists() and health['unavailable_games'] == [{'game_pk': LIVE_GAME, 'status': 503, 'code': 'refused'}]
    world.cycle(after=20.0)
    assert len(world.game()['pitches']) == 9 and world.read('v1/manifest.json')['fixed_delay_s'] == 30
    world.cycle(snap('20261005_100100', second=True, wall=world.now[0]+7), after=12.0)  # a new pitch, 5 s old
    held = world.game()
    # the newer pitch is nowhere among the thrown ones: for this viewer it is still the pitch to come
    assert len(held['pitches']) == 9 and held['next']['pitch_key'] == '900200:5:2' and held['pas'][-1]['pitch_count'] == 1
    assert held['current']['situation']['balls']+held['current']['situation']['strikes'] == 1
    assert strip(held) == world.served(f'/api/service/games/{LIVE_GAME}')  # the private app's route has the same delay
    world.cycle(after=30.0)
    assert [p['key'] for p in world.game()['pitches']][-1] == '900200:5:2'


def test_one_fixed_delay_holds_back_the_game_list_as_it_holds_back_the_game(make, monkeypatch):
    """M1: the published list is the as-of point of the same fixed delay as the game file, with the optional viewer
    feature OFF (``?delay=`` stays ignored). The schedule row is the ballpark's present, so a LIVE card shows no
    score, inning or status words while the game is buffering or held back, and a card the schedule already calls
    Final stays presented as live without a score until the delayed game page could have reached the end. Delay 0
    is unchanged. Only lead information is withheld: the card's other fields are not rebuilt to a past state."""
    monkeypatch.delenv('PITCHEEZY_OBSERVER_VIEWER_DELAY', raising=False)
    world = make(fixed_delay_s=30)
    lead = lambda card: (card['away_score'], card['home_score'], card['inning'], card['half'], card['status_text'], card.get('score_hidden'))
    listed = lambda w=world: {c['game_pk']: c for c in w.read(f'v1/games/{LIVE_DATE}.json')['games']}[LIVE_GAME]
    world.cycle(settle=False)  # buffering: no game file yet, so the list may not run ahead of it
    assert not world.path(LIVE_FILE).exists()
    assert listed()['status'] == 'live' and lead(listed()) == (None, None, None, None, None, True)
    world.cycle(after=20.0)  # the game file is 30 s behind; the list still says nothing the game file does not
    assert len(world.game()['pitches']) == 9 and lead(listed()) == (None, None, None, None, None, True)
    assert strip(world.read(f'v1/games/{LIVE_DATE}.json')) == world.served('/api/service/games', date=LIVE_DATE)  # the private route: same delay
    assert world.served('/api/service/games', date=LIVE_DATE, delay='0') == world.served('/api/service/games', date=LIVE_DATE)  # feature off: ignored
    world.cycle(snap('20261005_100100', half='final', wall=world.now[0]+7), after=25.0)  # the schedule says Final now (its cache is 20 s)
    assert listed()['status'] == 'live' and lead(listed()) == (None, None, None, None, None, True)  # the delayed game page has not reached the end
    world.cycle(after=30.0)
    assert listed()['status'] == 'final' and listed()['home_score'] == 1 and not listed().get('score_hidden')
    plain = make()  # delay 0: the schedule row as it is
    plain.cycle()
    assert listed(plain)['status'] == 'live' and listed(plain)['home_score'] == 1 and not listed(plain).get('score_hidden')


def test_a_published_game_never_holds_a_pitch_the_feed_has_not_delivered(world):
    world.cycle()
    keys = [p['key'] for p in world.game()['pitches']]
    assert keys == [f'{LIVE_GAME}:{ab}:{n}' for ab in range(1, 5) for n in (1, 2)]+[f'{LIVE_GAME}:5:1']


# ---------- ledger first, fail closed ----------
class Watching(LocalDirSink):
    """Records, for every game file holding candidates, which ledger rows existed when the write began."""

    def __init__(self, root, store):
        super().__init__(root)
        self.store, self.seen = store, []

    def write(self, path, data, *, final=False):
        body = json.loads(data)
        if path == LIVE_FILE and candidates(body):
            self.seen.append((body['next']['rec']['ledger']['decision_id'],
                              [row['decision_id'] for row in self.store.served_recommendations(LIVE_GAME)]))
        super().write(path, data, final=final)


def test_a_recommendation_is_in_the_ledger_before_the_file_that_carries_it_is_written(world):
    sink = Watching(world.out, world.feed.store)
    world.start(sink=sink)
    world.cycle()
    assert sink.seen and all(decision in rows for decision, rows in sink.seen)
    rec = world.game()['next']['rec']
    assert rec['provenance'] == 'captured_live' and rec['ledger']['status'] == 'recorded'


def test_a_recommendation_the_ledger_could_not_record_is_not_published(world, monkeypatch):
    def broken(**_):
        raise sqlite3.OperationalError('disk I/O error')
    monkeypatch.setattr(world.feed.store, 'append_recommendation', broken)
    world.cycle()
    body = world.game()
    assert candidates(body) == [] and 'SL' not in json.dumps(body['next'])
    assert (body['display_state']['state'], body['display_state']['recommendation_shown']) == ('ledger_unavailable', False)
    assert world.feed.store.served_recommendations(LIVE_GAME) == []
    assert len(body['pitches']) == 9  # the game itself is still published
    monkeypatch.undo()
    world.cycle()
    assert candidates(world.game()) == ['SL']




# ---------- atomic replacement ----------
def test_a_file_is_replaced_whole_and_a_failed_write_leaves_the_old_one(world, monkeypatch):
    world.cycle()
    before, real, seen = world.path(LIVE_FILE).read_bytes(), os.replace, []

    def watching(source, target):
        if str(target).endswith(LIVE_FILE):  # at the swap: the published name still holds the complete old file,
            assert Path(target).read_bytes() == before  # and the new one is complete outside the output folder
            assert not Path(source).is_relative_to(world.out) and json.loads(Path(source).read_bytes())['publish']['sequence'] == 3
            seen.append(source)
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', watching)
    world.cycle(settle=False)
    assert len(seen) == 1 and world.game()['publish']['sequence'] == 3
    monkeypatch.undo()

    kept = world.path(LIVE_FILE).read_bytes()

    def failing(source, target):
        if str(target).endswith('v1/dates.json'):
            raise OSError('disk full')
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', failing)
    with pytest.raises(OSError):  # a file every reader needs could not be replaced: the cycle is not complete
        world.cycle(settle=False)
    assert world.read('v1/dates.json')['publish']['sequence'] == 3 and world.read('v1/health.json')['sequence'] == 3  # health did not move
    assert all(ALLOWED.fullmatch(name) for name in world.files())  # no temporary file is left behind,
    assert list(world.publisher.sink.staging.iterdir()) == []  # in the output or beside it
    monkeypatch.undo()
    world.cycle(settle=False)
    assert world.game()['publish']['sequence'] == 5 and world.read('v1/health.json')['sequence'] == 5  # 4 was taken, never reused


def test_the_run_loop_survives_a_failed_cycle(world, monkeypatch):
    calls, real = [], world.publisher.cycle

    def cycle():
        calls.append(len(calls))
        if len(calls) == 1:
            raise OSError('disk full')
        return real()

    def sleep(seconds):
        assert 0 <= seconds <= world.config['publish_interval_s']
        if len(calls) == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(world.publisher, 'cycle', cycle)
    with pytest.raises(KeyboardInterrupt):
        world.publisher.run(sleep=sleep)
    assert calls == [0, 1] and world.read('v1/health.json')['sequence'] == 1


def test_the_local_sink_refuses_paths_outside_its_folder(tmp_path):
    sink = LocalDirSink(tmp_path/'out')
    for path in ('../x.json', '/etc/x.json', 'v1/../../x.json', 'v1/.hidden.json', ''):
        with pytest.raises(ValueError):
            sink.write(path, b'{}')
    sink.write('v1/a.json', b'{}', final=True)
    assert sink.read('v1/a.json') == b'{}' and sink.read('v1/b.json') is None
    sink.delete('v1/a.json')
    assert sink.read('v1/a.json') is None


# ---------- sequence ----------
def test_the_sequence_only_goes_up_across_cycles_restarts_and_a_lost_state_file(world):
    seen = []
    for _ in range(3):
        world.cycle(settle=False)
        seen.append(world.read('v1/health.json')['sequence'])
        assert world.game()['publish']['sequence'] == world.read('v1/dates.json')['publish']['sequence'] == seen[-1]
    world.start()  # a new process over the same folders
    world.cycle(settle=False)
    seen.append(world.read('v1/health.json')['sequence'])
    Path(world.publisher.state_path).unlink()  # the bookkeeping is lost: the published health file still knows
    world.start()
    world.cycle(settle=False)
    seen.append(world.read('v1/health.json')['sequence'])
    assert seen == [1, 2, 3, 4, 5]
    assert json.loads(Path(world.publisher.state_path).read_text())['sequence'] == 5
    assert Path(world.publisher.state_path).parent == world.out.parent  # the state file is not inside the output


# ---------- final games ----------
def test_a_final_game_is_written_once_and_a_correction_takes_a_new_sequence(world):
    world.cycle()
    assert world.game()['publish']['final'] is False
    world.cycle(snap('20261005_100100', half='final'), settle=False)
    final = world.path(LIVE_FILE).read_bytes()
    assert json.loads(final)['publish'] == {'contract': 'v1', 'sequence': 3, 'published_at': '2026-10-05T10:20:24Z', 'final': True}
    polls = lambda: sum('/feed/live' in url for url in world.upstream.calls)
    asked = polls()
    for _ in range(5):
        world.cycle(settle=False)
    assert world.path(LIVE_FILE).read_bytes() == final and polls() == asked  # not rewritten, not even rebuilt
    assert world.read('v1/health.json')['sequence'] == 8
    archive = world.path(ARCHIVE_FILE).read_bytes()
    assert json.loads(archive)['publish']['sequence'] == 1
    # the provider corrects the last pitch after the game: found at the settled recheck, written with a new sequence
    world.cycle(snap('20261005_100200', half='final', call='C'), after=world.config['settled_recheck_s'], settle=False)
    corrected = world.game()
    assert corrected['publish'] == {'contract': 'v1', 'sequence': 9, 'published_at': corrected['publish']['published_at'], 'final': True}
    assert corrected['pitches'][8]['actual'] != json.loads(final)['pitches'][8]['actual']
    assert world.path(ARCHIVE_FILE).read_bytes() == archive  # rebuilt at the recheck, unchanged, so not rewritten


def test_a_final_game_with_a_pitch_still_unconfirmed_is_not_marked_final_yet(world):
    body = snap(half='final')
    del body['liveData']['plays']['allPlays'][3]['playEvents'][1]['playId']
    world.cycle(body)
    assert world.game()['game']['status'] == 'final' and world.game()['publish']['final'] is False
    world.cycle(snap('20261005_100100', half='final'))
    assert world.game()['publish']['final'] is True and world.game()['receipts']['unconfirmed'] == 0


def test_never_rechecking_settled_files_is_a_configuration(make):
    world = make(first=snap(half='final'))
    world.config |= {'settled_recheck_s': None, 'final_slow_recheck_s': None}
    world.cycle(settle=False)
    final = world.path(LIVE_FILE).read_bytes()
    world.cycle(snap('20261005_100200', half='final', call='C'), after=86400.0, settle=False)
    assert world.path(LIVE_FILE).read_bytes() == final


# ---------- setup estimates ----------
def test_setup_estimates_are_published_only_for_a_finished_game_that_passes_validation(make):
    world = make()
    world.write_setup(game=LIVE_GAME, lines=[])  # an (empty but pinned) file for the game in progress
    world.cycle()
    setup = world.read(f'v1/setup/{ARCHIVE_GAME}.json')
    assert strip(setup)['pitches'] == world.served(f'/api/watch/{ARCHIVE_GAME}/setup')['pitches']
    assert setup['publish']['final'] is True and setup['coverage']['with_plate_x'] == 3
    assert not world.path(f'v1/setup/{LIVE_GAME}.json').exists()  # in progress: never
    written = world.path(f'v1/setup/{ARCHIVE_GAME}.json').read_bytes()
    world.cycle(after=world.config['settled_recheck_s'])
    assert world.path(f'v1/setup/{ARCHIVE_GAME}.json').read_bytes() == written
    world.write_setup(sha256='0'*64)  # the file no longer matches its pin: withdrawn at the next recheck
    world.cycle(after=world.config['settled_recheck_s'])
    assert not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()

    bad = make(setup=False)
    bad.cycle()
    assert not bad.path(f'v1/setup/{ARCHIVE_GAME}.json').exists() and bad.path(ARCHIVE_FILE).exists()
    bad.write_setup()  # the video module's file arrives after the game: picked up at the settled recheck
    bad.cycle(after=bad.config['settled_recheck_s'])
    assert bad.read(f'v1/setup/{ARCHIVE_GAME}.json')['publish']['sequence'] == bad.read('v1/health.json')['sequence']-1


# ---------- demo sessions and the operator API ----------
def test_demo_sessions_and_operator_data_are_never_published(world):
    token = DemoClock(world.demo).token()
    head = {'X-Pitcheezy-Operator': token}
    made = world.client.post('/api/service/operator/sessions', json={'game_pk': ARCHIVE_GAME, 'mapping_id': 'fx', 'start_index': 4}, headers=head)
    assert made.status_code == 201, made.text
    sid = made.json()['session_id']
    assert world.client.get(f'/api/service/games/{ARCHIVE_GAME}', params={'demo': sid}).json()['source']['kind'] == 'demo'
    health = world.cycle()
    assert not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()  # embargoed: 423, no file
    assert health['unavailable_games'] == []  # and a demo is not reported as a fault
    assert health['game_checks'][str(ARCHIVE_GAME)]['file_sequence'] is None
    assert health['game_checks'][str(ARCHIVE_GAME)]['feed_ok'] is False
    schema = json.loads((CONTRACT_DIR/'schema/health.schema.json').read_text())
    assert publish_schema.validate(world.read('v1/health.json'), schema) == []
    card = world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]
    assert card['embargoed'] is True and card['away_score'] is None
    assert strip(world.read(f'v1/games/{ARCHIVE_DATE}.json')) == world.served('/api/service/games', date=ARCHIVE_DATE)
    assert all(ALLOWED.fullmatch(name) for name in world.files())
    for name in world.files():
        text = world.path(name).read_text()
        assert sid not in text and token not in text and '"demo"' not in text and 'operator' not in text
        body = json.loads(text)
        assert (body.get('source') or {'kind': 'archive'})['kind'] in ('archive', 'live')
    assert world.game()['source'] == {'kind': 'live', 'session_id': None}


def test_a_body_that_is_not_the_public_view_is_refused(world):
    for body in ({'source': {'kind': 'demo', 'session_id': 'abc'}}, {'source': {'kind': 'archive', 'session_id': 'abc'}},
                 {'source': {'kind': 'live', 'session_id': None}, 'publish': {}}, {}):
        with pytest.raises(publisher.NotPublic):
            publisher.check_public(body)
    demo = world.app.state.service_bodies.game
    world.publisher.bodies.game = lambda pk: demo(pk) | {'source': {'kind': 'demo', 'session_id': 'abc'}}
    health = world.cycle(settle=False)
    assert not world.path(LIVE_FILE).exists() and not world.path(ARCHIVE_FILE).exists()
    assert codes(health) == [(ARCHIVE_GAME, 'not_public'), (LIVE_GAME, 'not_public')]


# ---------- upstream failure ----------
def test_a_game_whose_first_poll_fails_has_no_file_and_health_says_so(make):
    world = make(first=OSError('upstream down'))
    health = world.cycle(settle=False)
    assert not world.path(LIVE_FILE).exists() and health['unavailable_games'] == [{'game_pk': LIVE_GAME, 'status': 503, 'code': 'refused'}]
    assert world.path(ARCHIVE_FILE).exists()  # one game never stops the rest
    world.cycle(snap(), settle=False)
    assert world.game()['feed']['ok'] is True and world.read('v1/health.json')['unavailable_games'] == []


def test_a_disconnected_feed_is_published_as_the_route_states_it(world):
    world.cycle()
    world.cycle(OSError('upstream down'))
    body = world.game()
    assert body['feed']['ok'] is False and body['display_state']['state'] == 'disconnected'
    assert strip(body) == world.served(f'/api/service/games/{LIVE_GAME}')


# ---------- configuration ----------
def test_configuration_comes_from_the_file_the_environment_and_overrides(tmp_path):
    base = publisher.load_config(env={})
    assert base['fixed_delay_s'] == 0 and base['contract_version'] == 'v1' and base['out_dir'] is None
    env = {'PITCHEEZY_PUBLISH_OUT_DIR': str(tmp_path), 'PITCHEEZY_PUBLISH_INTERVAL_S': '4', 'PITCHEEZY_PUBLISH_DELAY_S': '15'}
    config = publisher.load_config(env=env)
    assert (config['out_dir'], config['publish_interval_s'], config['fixed_delay_s']) == (str(tmp_path), 4.0, 15.0)
    assert publisher.load_config(env=env, fixed_delay_s=0, out_dir=None)['fixed_delay_s'] == 0
    for bad in ({'publish_interval_s': 0}, {'publish_interval_s': float('nan')}, {'fixed_delay_s': -1}, {'live_days_back': 1.5}):
        with pytest.raises(ValueError):
            publisher.load_config(env={}, **bad)
    with pytest.raises(ValueError):
        publisher.build(base)  # no output directory anywhere


def test_explicit_dates_replace_the_live_window(world):
    world.start(dates=[ARCHIVE_DATE])
    world.cycle(settle=False)
    assert world.read('v1/manifest.json')['live_dates'] == [ARCHIVE_DATE]
    assert not world.path(LIVE_FILE).exists() and world.path(ARCHIVE_FILE).exists()
    with pytest.raises(ValueError):
        world.start(dates=['10/05'])


def test_the_command_line_publishes_recorded_fixtures_once(tmp_path, monkeypatch):
    """``python -m observer_app.publisher --once`` end to end over the recorded 849845 fixtures (replay transport)."""
    for name in ('PITCHEEZY_OBSERVER_LIVE_STORE', 'PITCHEEZY_OBSERVER_LIVE_DELAY_S', 'PITCHEEZY_OBSERVER_LIVE_REPLAY_PACE',
                 'PITCHEEZY_PUBLISH_OUT_DIR', 'PITCHEEZY_PUBLISH_STATE', 'PITCHEEZY_PUBLISH_INTERVAL_S', 'PITCHEEZY_PUBLISH_DELAY_S'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(listed_dir(tmp_path/'replay')))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_POLL_S', '0')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    monkeypatch.setenv('PITCHEEZY_PUBLISH_COMMIT', 'fixture-commit')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(tmp_path/'store.sqlite3'))
    empty = tmp_path/'empty'
    empty.mkdir()
    out = tmp_path/'out'
    arguments = ['--out', str(out), '--once', '--date', '2026-09-30', '--feed-dir', str(empty), '--watch-dir', str(empty),
                 '--intent-dir', str(empty), '--demo-dir', str(tmp_path/'demo')]
    assert publisher.main(arguments) == 0
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    assert all(ALLOWED.fullmatch(name) for name in files)
    assert {'v1/manifest.json', 'v1/health.json', 'v1/dates.json', 'v1/games/2026-09-30.json', 'v1/game/849845.json'} <= set(files)
    game = json.loads((out/'v1/game/849845.json').read_text())
    assert game['source']['kind'] == 'live' and game['publish']['sequence'] == 1 and candidates(game) == []
    assert json.loads((out/'v1/manifest.json').read_text())['commit'] == 'fixture-commit'
    assert publisher.main(arguments) == 0  # a second process continues the sequence
    assert json.loads((out/'v1/health.json').read_text())['sequence'] == 2


# ---------- the contract folder handed to the other team ----------
def test_the_committed_examples_and_schemas_are_what_the_publisher_writes_now(tmp_path):
    rendered = publish_fixture.render(tmp_path)
    committed = {str(path.relative_to(CONTRACT_DIR)): path.read_text() for path in CONTRACT_DIR.rglob('*.json')}
    assert sorted(committed) == sorted(rendered)
    for name, text in rendered.items():  # regenerate with scripts/export_publish_contract.py after a deliberate change
        assert committed[name] == text, name


def test_v1_files_without_optional_rc_metadata_remain_valid(world):
    world.cycle()
    for kind, body in (('game', world.game()), ('games', world.read(f'v1/games/{LIVE_DATE}.json'))):
        schema = json.loads((CONTRACT_DIR/f'schema/{kind}.schema.json').read_text())
        body.pop('features')
        body.pop('viewer', None)
        assert publish_schema.validate(body, schema) == []


def test_every_example_and_every_fixture_body_validates_against_its_schema(tmp_path):
    schemas = {kind: json.loads((CONTRACT_DIR/f'schema/{kind}.schema.json').read_text()) for kind in publish_fixture.KINDS}
    for kind, names in publish_fixture.EXAMPLES.items():
        for name in names:
            assert publish_schema.validate(json.loads((CONTRACT_DIR/'examples'/name).read_text()), schemas[kind]) == [], name
    for kind, bodies in publish_fixture.collect(tmp_path).items():
        assert bodies
        for name, body in bodies:
            assert publish_schema.validate(body, schemas[kind]) == [], name


def test_recommendation_schema_accepts_real_candidate_types_and_unsupported_example():
    schema = json.loads((CONTRACT_DIR/'schema/game.schema.json').read_text())
    supported = json.loads((CONTRACT_DIR/'examples/game-live.json').read_text())
    candidate = supported['next']['rec']['candidates'][0]
    candidate['detail']['probability'] = .42
    candidate['target'] = {'x': .3, 'z': 2.1}
    assert publish_schema.validate(supported, schema) == []
    unsupported = json.loads((CONTRACT_DIR/'examples/game-live-unsupported.json').read_text())
    assert unsupported['display_state']['state'] == 'unsupported'
    assert unsupported['next']['rec']['status'] == 'unsupported'
    assert unsupported['next']['rec']['candidates'] == []
    assert publish_schema.validate(unsupported, schema) == []


def test_bodies_built_outside_the_export_validate_too(world):
    """The schema is not only true of the bodies it was inferred from: states the export does not walk through."""
    schemas = {kind: json.loads((CONTRACT_DIR/f'schema/{kind}.schema.json').read_text()) for kind in ('game', 'health', 'games')}
    world.cycle()
    world.cycle(snap('20261005_100100', call='S'))  # another first call: the fake policy now recommends FF, a correction
    assert world.game()['display_state']['correction'] is not None
    assert publish_schema.validate(world.game(), schemas['game']) == []
    world.cycle(snap(None, second=True, wall=world.now[0]+10))  # a newer event without the provider's stamp: order not established
    assert world.game()['display_state']['state'] == 'unconfirmed'
    assert publish_schema.validate(world.game(), schemas['game']) == []
    world.cycle(OSError('upstream down'), after=400.0)  # disconnected for a long while
    assert world.game()['display_state']['state'] == 'disconnected'
    assert publish_schema.validate(world.game(), schemas['game']) == []
    assert publish_schema.validate(world.read('v1/health.json'), schemas['health']) == []
    assert publish_schema.validate(world.read(f'v1/games/{LIVE_DATE}.json'), schemas['games']) == []


def test_the_schemas_refuse_what_the_contract_forbids():
    schema = json.loads((CONTRACT_DIR/'schema/game.schema.json').read_text())
    good = json.loads((CONTRACT_DIR/'examples/game-live.json').read_text())
    assert publish_schema.validate(good, schema) == []
    for change in (lambda b: b.pop('publish'), lambda b: b['publish'].pop('sequence'), lambda b: b['publish'].update(sequence=0),
                   lambda b: b['publish'].update(sequence='4'), lambda b: b['publish'].update(contract='v2'),
                   lambda b: b.update(schema='something-else'), lambda b: b.update(pitches={}), lambda b: b.pop('pitches'),
                   lambda b: b['pitches'][0].pop('actual')):
        broken = json.loads(json.dumps(good))
        change(broken)
        assert publish_schema.validate(broken, schema), change


def test_schema_inference_and_validation_basics():
    schema = publish_schema.infer([{'a': 1, 'b': None, 'm': {'12': {'x': 1.5}}}, {'a': 2, 'b': 'x', 'c': [1, 2], 'm': {'7:1': {'x': 2}}}])
    assert schema == {'type': 'object', 'required': ['a', 'b', 'm'], 'properties': {
        'a': {'type': 'integer'}, 'b': {'type': ['string', 'null']}, 'c': {'type': 'array', 'items': {'type': 'integer'}},
        'm': {'type': 'object', 'additionalProperties': {'type': 'object', 'properties': {'x': {'type': 'number'}}, 'required': ['x']}}}}
    assert publish_schema.validate({'a': 1, 'b': None, 'm': {'99': {'x': 3}}, 'new': True}, schema) == []  # unknown keys are fine
    assert publish_schema.validate({'a': True, 'b': 1, 'm': {'99': {}}}, schema) == [
        '$.a: boolean is not integer', '$.b: integer is not string or null', "$.m.99: missing 'x'"]
    assert publish_schema.infer([]) == {}


# ---------- review fixes (2026-10-11) ----------
def polls(world):
    return sum('/feed/live' in url for url in world.upstream.calls)


def start_demo(world):
    head = {'X-Pitcheezy-Operator': DemoClock(world.demo).token()}
    made = world.client.post('/api/service/operator/sessions', json={'game_pk': ARCHIVE_GAME, 'mapping_id': 'fx', 'start_index': 4}, headers=head)
    assert made.status_code == 201, made.text
    return made.json()['session_id'], head


def test_a_final_live_game_rechecked_with_no_upstream_change_keeps_its_bytes_and_sequence(world):
    world.cycle()
    world.cycle(snap('20261005_100100', half='final'), settle=False)
    final, asked = world.path(LIVE_FILE).read_bytes(), polls(world)
    assert json.loads(final)['publish']['final'] is True
    for _ in range(3):  # each recheck polls upstream again and gets the same game; only the clock has moved
        world.cycle(after=world.config['settled_recheck_s']+1, settle=False)
    assert polls(world) == asked+3
    assert world.path(LIVE_FILE).read_bytes() == final and world.game()['publish']['sequence'] == 3
    world.cycle(OSError('upstream down'), after=world.config['settled_recheck_s']+1, settle=False)  # nor does a failed recheck
    assert world.path(LIVE_FILE).read_bytes() == final
    assert world.read('v1/health.json')['sequence'] == 7


def test_the_change_digest_ignores_exactly_the_clock_driven_fields(world):
    world.cycle()
    body = strip(world.game())
    moved = json.loads(json.dumps(body))
    moved['feed'] = {'as_of': 'later', 'ok': False, 'stale_s': 99.0}
    moved['display_state']['feed'] |= {'last_ok_at': 'later', 'received_at': 'later', 'lag_s': 9.9, 'lag_max_s': 9.9,
                                       'failing_since': 'later', 'failing_for_s': 9.9, 'ok': False}
    assert publisher.game_digest(moved) == publisher.game_digest(body)
    for change in (lambda b: b['display_state'].update(state='final'), lambda b: b['pitches'][0]['actual'].update(pitch_type='XX'),
                   lambda b: b['receipts'].update(version='x'), lambda b: b['game'].update(home_score=9),
                   lambda b: b['display_state']['feed'].update(provider_order='stale')):
        other = json.loads(json.dumps(body))
        change(other)
        assert publisher.game_digest(other) != publisher.game_digest(body), change


def test_a_demo_that_starts_after_a_game_was_published_withdraws_its_files_at_once(world):
    world.cycle()
    assert world.path(ARCHIVE_FILE).exists() and world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['home_score'] == 1
    sid, head = start_demo(world)
    world.cycle(settle=False)  # the very next cycle, long before any settled recheck is due
    assert not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    card = world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]
    assert card['embargoed'] is True and card['home_score'] is None
    assert world.read('v1/health.json')['unavailable_games'] == [] and world.path(LIVE_FILE).exists()
    ended = world.client.post(f'/api/service/operator/sessions/{sid}/control', json={'action': 'end'}, headers=head)
    assert ended.status_code == 200, ended.text
    world.cycle(settle=False)  # the demo is over: the game is public again, with a new sequence
    again = world.game(ARCHIVE_GAME)
    assert again['publish']['final'] is True and again['publish']['sequence'] == world.read('v1/health.json')['sequence']
    assert world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['home_score'] == 1


def test_a_second_publisher_over_the_same_output_fails_fast(world, monkeypatch):
    world.cycle(settle=False)
    with pytest.raises(publisher.AlreadyRunning, match='another publisher'):
        publisher.Publisher(world.app.state.service_bodies, LocalDirSink(world.out), config=world.config,
                            state_path=world.publisher.state_path)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(listed_dir(world.tmp/'replay')))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_STORE', str(world.tmp/'cli-store.sqlite3'))
    monkeypatch.delenv('PITCHEEZY_PUBLISH_STATE', raising=False)
    arguments = ['--out', str(world.out), '--once', '--feed-dir', str(world.feeds), '--watch-dir', str(world.watch),
                 '--intent-dir', str(world.intent), '--demo-dir', str(world.demo)]
    with pytest.raises(SystemExit, match='another publisher'):
        publisher.main(arguments)
    assert world.read('v1/health.json')['sequence'] == 1  # the second one wrote nothing
    world.publisher.close()
    other = publisher.Publisher(world.app.state.service_bodies, LocalDirSink(world.out), config=world.config,
                                state_path=world.publisher.state_path)  # the lock is free once the first one is gone
    other.close()


def test_a_final_live_game_is_no_longer_polled_after_the_recheck_period_and_its_file_stays(world):
    world.config['final_recheck_for_s'] = 1500
    world.cycle(snap(half='final'), settle=False)
    final, asked = world.path(LIVE_FILE).read_bytes(), polls(world)
    world.cycle(after=601.0, settle=False)
    world.cycle(after=601.0, settle=False)
    assert polls(world) == asked+2  # still inside the period: rechecked
    for _ in range(3):
        world.cycle(snap('20261005_100900', half='final', call='C'), after=601.0, settle=False)
    assert polls(world) == asked+2 and world.path(LIVE_FILE).read_bytes() == final  # past it: left alone, kept
    assert world.path(ARCHIVE_FILE).exists()
    world.config['final_recheck_for_s'] = None  # null: rechecked for as long as the publisher knows the game
    world.cycle(after=601.0, settle=False)
    assert polls(world) == asked+3 and world.game()['publish']['sequence'] == world.read('v1/health.json')['sequence']


def test_a_setup_file_that_arrives_after_the_recheck_period_is_still_published(make):
    world = make(first=snap(half='final'))
    world.config['final_recheck_for_s'] = 0
    world.cycle(settle=False)
    assert not world.path(f'v1/setup/{LIVE_GAME}.json').exists()
    world.write_setup(game=LIVE_GAME)
    asked = polls(world)
    world.cycle(after=601.0, settle=False)
    assert world.read(f'v1/setup/{LIVE_GAME}.json')['publish']['final'] is True and polls(world) == asked  # a local read only


def test_a_game_list_that_could_not_be_built_is_named_in_health(world):
    world.cycle(settle=False)
    real = world.publisher.bodies.games

    def games(date):
        if date == LIVE_DATE:
            raise RuntimeError('secret detail /private/path')
        return real(date)
    world.publisher.bodies.games = games
    kept = world.path(f'v1/games/{LIVE_DATE}.json').read_bytes()
    health = world.cycle(settle=False)
    assert health['unavailable_dates'] == [LIVE_DATE] and world.path(f'v1/games/{LIVE_DATE}.json').read_bytes() == kept
    assert 'secret' not in world.path('v1/health.json').read_text() and 'private' not in world.path('v1/health.json').read_text()
    world.publisher.bodies.games = real
    assert world.cycle(settle=False)['unavailable_dates'] == []


@pytest.mark.parametrize('garbage', ['{"sequence": 3, "files"', '[1, 2]', '{"sequence": "x"}', ''])
def test_a_corrupted_state_file_does_not_stop_the_publisher_and_the_sequence_continues(world, garbage):
    world.cycle(settle=False)
    world.cycle(settle=False)
    world.publisher.close()
    Path(world.publisher.state_path).write_text(garbage)
    world.start()
    world.cycle(settle=False)
    assert world.read('v1/health.json')['sequence'] == 3 and world.game()['publish']['sequence'] == 3
    assert json.loads(Path(world.publisher.state_path).read_text())['sequence'] == 3


def test_temporary_files_are_never_inside_the_output_folder(world, monkeypatch):
    real, sources = os.replace, []

    def watching(source, target):
        if Path(target).is_relative_to(world.out):
            sources.append(Path(source))
            assert not Path(source).is_relative_to(world.out)
            assert os.stat(Path(source).parent).st_dev == os.stat(Path(target).parent).st_dev  # same filesystem: a rename
            assert sorted(str(p.relative_to(world.out)) for p in world.out.rglob('*') if p.is_file()) == world.files()
            assert all(ALLOWED.fullmatch(name) for name in world.files())
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', watching)
    world.cycle(settle=False)
    assert len(sources) >= 6 and not any(source.exists() for source in sources)
    assert all(ALLOWED.fullmatch(str(p.relative_to(world.out))) for p in world.out.rglob('*') if p.is_file())


def test_the_file_prefix_and_the_contract_version_are_one_value(make):
    world = make()
    assert world.read_after_cycle('v1/manifest.json')['files']['game'] == 'v1/game/{game_pk}.json'
    other = make(contract_version='v2')
    manifest = other.read_after_cycle('v2/manifest.json')
    assert manifest['contract'] == manifest['publish']['contract'] == 'v2' and manifest['files']['health'] == 'v2/health.json'
    assert all(name.startswith('v2/') for name in other.files())
    assert other.read(f'v2/game/{LIVE_GAME}.json')['publish']['contract'] == 'v2'
    for bad in ('', 'v1/x', '../v1', 'V1', 1):
        with pytest.raises(ValueError):
            publisher.load_config(env={}, contract_version=bad)


# ---------- second review (2026-10-11, A-I) ----------
def tree(world):
    return {name: world.path(name).read_text() for name in world.files()}


def codes(health):
    return [(item['game_pk'], item['code']) for item in health['unavailable_games']]


# A: store and ledger are mandatory
def test_a_publisher_without_a_store_refuses_to_start(tmp_path):
    with pytest.raises(publisher.Unsafe, match='live store'):
        World(tmp_path, store=False)
    assert not (tmp_path/'out').exists()


def test_with_the_ledger_switched_off_nothing_is_published(world, monkeypatch):
    world.cycle()
    before = tree(world)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_REC_LEDGER', '0')
    with pytest.raises(publisher.Unsafe, match='ledger'):
        world.cycle(snap('20261005_100100', second=True), settle=False)
    assert tree(world) == before  # not one file moved, health included


def test_a_live_body_without_receipts_or_with_an_unrecorded_recommendation_is_never_written(world):
    world.cycle()
    kept, real = world.path(LIVE_FILE).read_bytes(), world.publisher.bodies.game
    assert candidates(world.game()) == ['SL']

    def unmasked(pk):
        body = real(pk)
        return body | {'receipts': None} if pk == LIVE_GAME else body

    def unrecorded(pk):
        body = real(pk)
        if pk == LIVE_GAME:
            body['next']['rec'] = body['next']['rec'] | {'ledger': None}
        return body
    for fake in (unmasked, unrecorded):
        world.publisher.bodies.game = fake
        health = world.cycle(settle=False)
        assert world.path(LIVE_FILE).read_bytes() == kept and codes(health) == [(LIVE_GAME, 'not_public')]
    world.publisher.bodies.game = real
    assert codes(world.cycle(settle=False)) == []


def test_the_command_line_refuses_to_run_without_a_store(tmp_path, monkeypatch):
    monkeypatch.delenv('PITCHEEZY_OBSERVER_LIVE_STORE', raising=False)
    monkeypatch.delenv('PITCHEEZY_PUBLISH_STATE', raising=False)
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_REPLAY_DIR', str(listed_dir(tmp_path/'replay')))
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '0')
    empty = tmp_path/'empty'
    empty.mkdir()
    arguments = ['--out', str(tmp_path/'out'), '--once', '--date', '2026-09-30', '--feed-dir', str(empty), '--watch-dir', str(empty),
                 '--intent-dir', str(empty), '--demo-dir', str(tmp_path/'demo')]
    with pytest.raises(SystemExit, match='PITCHEEZY_OBSERVER_LIVE_STORE'):
        publisher.main(arguments)
    assert not (tmp_path/'out').exists()


# B: withdrawal does not depend on the bookkeeping
@pytest.mark.parametrize('damage', ['lost', 'corrupted'])
def test_an_embargoed_game_is_withdrawn_even_when_the_state_file_does_not_know_its_files(world, damage):
    world.cycle()
    assert world.path(ARCHIVE_FILE).exists() and world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    start_demo(world)
    world.publisher.close()
    state = Path(world.publisher.state_path)
    state.unlink() if damage == 'lost' else state.write_text('{"sequence": 1, "files"')
    world.start()
    world.cycle(settle=False)
    assert not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['embargoed'] is True


# C: a demo that starts in the middle of a cycle
def test_a_demo_that_starts_between_the_build_and_the_write_is_not_published(world):
    real, started = world.publisher.bodies.game, []

    def game(pk):
        body = real(pk)  # built while the game was still free...
        if pk == ARCHIVE_GAME and not started:
            started.append(start_demo(world))  # ...and the demo takes it before the file is written
        return body
    world.publisher.bodies.game = game
    world.cycle(settle=False)
    assert started and not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    card = world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]  # the list written earlier in this cycle is redone
    assert card['embargoed'] is True and card['home_score'] is None


def test_a_demo_that_starts_while_the_file_is_being_written_is_withdrawn_in_the_same_cycle(world):
    real, started = world.publisher.sink.write, []

    def write(path, data, *, final=False):
        real(path, data, final=final)
        if path == ARCHIVE_FILE and not started:
            started.append(start_demo(world))
    world.publisher.sink.write = write
    world.cycle(settle=False)
    assert started and not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['embargoed'] is True


def test_when_a_list_cannot_be_rebuilt_during_a_demo_its_old_scores_do_not_stay(world):
    world.cycle()
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['home_score'] == 1
    start_demo(world)
    real = world.publisher.bodies.games

    def games(date):
        if date == ARCHIVE_DATE:
            raise RuntimeError('boom')
        return real(date)
    world.publisher.bodies.games = games
    health = world.cycle(settle=False)
    assert not world.path(f'v1/games/{ARCHIVE_DATE}.json').exists() and health['unavailable_dates'] == [ARCHIVE_DATE]
    world.publisher.bodies.games = real
    world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['embargoed'] is True


# D: the sequence after lost bookkeeping
def test_the_sequence_recovers_from_every_published_file_not_only_health(world, monkeypatch):
    world.cycle(settle=False)
    real = os.replace

    def failing(source, target):
        if str(target).endswith('v1/health.json'):
            raise OSError('disk full')
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', failing)
    for _ in range(3):  # three cycles write the game file but never reach health
        with pytest.raises(OSError):
            world.cycle(settle=False)
    monkeypatch.undo()
    assert world.read('v1/health.json')['sequence'] == 1 and world.game()['publish']['sequence'] == 4
    world.publisher.close()
    Path(world.publisher.state_path).unlink()
    world.start()
    world.cycle(settle=False)
    assert world.game()['publish']['sequence'] == world.read('v1/health.json')['sequence'] == 5  # above every file, not health+1


def test_the_sequence_is_on_disk_before_any_file_carries_it(world, monkeypatch):
    seen, real = [], world.publisher.sink.write

    def write(path, data, *, final=False):
        seen.append((json.loads(Path(world.publisher.state_path).read_text())['sequence'], json.loads(data)['publish']['sequence']))
        real(path, data, final=final)
    world.publisher.sink.write = write
    synced, fsync = [], os.fsync
    monkeypatch.setattr(publisher.os, 'fsync', lambda fd: (synced.append(fd), fsync(fd))[1])
    world.cycle(settle=False)
    assert seen and all(saved == written for saved, written in seen)
    world.publisher.sink.write = lambda *a, **k: None
    synced.clear()
    world.publisher._save_state()
    assert len(synced) >= 2  # the state file and its folder


# E: one lock per output folder
def test_the_lock_belongs_to_the_output_folder_not_to_the_state_file(world, tmp_path):
    link = tmp_path/'link'
    link.symlink_to(world.out.parent, target_is_directory=True)
    for out in (world.out, link/'out', Path(str(world.out)+'/')):
        with pytest.raises(publisher.AlreadyRunning, match='another publisher'):
            publisher.Publisher(None, LocalDirSink(out), config=world.config, state_path=tmp_path/'other.state.json')
    free = publisher.Publisher(None, LocalDirSink(tmp_path/'elsewhere'), config=world.config, state_path=tmp_path/'other.state.json')
    free.close()


def test_a_second_process_cannot_publish_and_a_killed_holder_frees_the_lock(tmp_path):
    import subprocess
    out, backend = tmp_path/'out', Path(__file__).resolve().parents[1]
    script = ('import sys, time\n'
              'from observer_app import publisher\n'
              'try:\n'
              '    held = publisher.Publisher(None, publisher.LocalDirSink(sys.argv[1]), config=publisher.load_config(env={}), state_path=sys.argv[2])\n'
              'except publisher.AlreadyRunning as error:\n'
              '    print("REFUSED", error, flush=True)\n'
              '    sys.exit(3)\n'
              'print("HELD", flush=True)\n'
              'time.sleep(60)\n')
    spawn = lambda state: subprocess.Popen([sys.executable, '-c', script, str(out), str(tmp_path/state)], cwd=backend,
                                           env=os.environ | {'PYTHONPATH': str(backend)}, stdout=subprocess.PIPE, text=True)
    children = []  # only processes this test started are ever signalled, and every one of them is reaped

    def child(state):
        children.append(spawn(state))
        return children[-1]
    try:
        holder = child('a.state.json')
        assert holder.stdout.readline().strip() == 'HELD'
        second = child('b.state.json')  # another process, another state file, the same output folder
        assert second.stdout.readline().startswith('REFUSED another publisher') and second.wait(timeout=20) == 3
        with pytest.raises(publisher.AlreadyRunning):
            publisher.Publisher(None, LocalDirSink(out), config=publisher.load_config(env={}), state_path=tmp_path/'c.state.json')
        holder.kill()
        holder.wait(timeout=20)
        third = child('b.state.json')
        assert third.stdout.readline().strip() == 'HELD'  # the dead holder's lock is gone
    finally:
        for process in children:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
            process.stdout.close()


# F: the public setup file carries no internal source field
def test_the_public_setup_file_keeps_only_a_commit_and_a_content_hash_of_its_source(world):
    world.write_setup(path='/Volumes/T7 Shield/secret/game.jsonl', repository='org/private-repo', branch='feature/secret-branch',
                      video='/Users/song/Movies/secret.mp4')
    world.cycle()
    setup = world.read(f'v1/setup/{ARCHIVE_GAME}.json')
    served = world.served(f'/api/watch/{ARCHIVE_GAME}/setup')
    assert served['source']['path'] == '/Volumes/T7 Shield/secret/game.jsonl'  # the private route is unchanged
    assert setup['source'] == {'commit': served['source']['commit'], 'sha256': served['source']['sha256']}
    assert {k: v for k, v in strip(setup).items() if k != 'source'} == {k: v for k, v in served.items() if k != 'source'}
    text = world.path(f'v1/setup/{ARCHIVE_GAME}.json').read_text()
    assert all(word not in text for word in ('secret', 'private-repo', 'T7 Shield', '/Users/'))


def test_no_published_file_holds_a_path_key_or_an_absolute_path(world):
    world.write_setup(path='/Volumes/T7 Shield/secret/game.jsonl')
    world.cycle()
    world.cycle(snap('20261005_100100', half='final'))

    def walk(value, where):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key != 'path', where
                walk(item, f'{where}.{key}')
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f'{where}[{index}]')
        elif isinstance(value, str):
            assert not value.startswith(('/', '~', 'file:')) and re.search(r'/(Users|Volumes|private|tmp|var|home)/', value) is None, (where, value)
            assert str(world.tmp) not in value, where
    assert len(world.files()) >= 9
    for name in world.files():
        walk(world.read(name), name)
    for path in CONTRACT_DIR.rglob('*.json'):
        if path.parent.name == 'examples':
            walk(json.loads(path.read_text()), path.name)


# G: after the recheck period
def test_after_the_recheck_period_a_late_correction_is_still_found_at_the_slow_pace(world):
    world.config |= {'final_recheck_for_s': 1000, 'final_slow_recheck_s': 5000}
    world.cycle(snap(half='final'), settle=False)
    final, asked = world.path(LIVE_FILE).read_bytes(), polls(world)
    world.upstream.body = snap('20261005_100900', half='final', call='C')  # an official correction, long after the game
    world.now[0] += 1100
    for _ in range(4):
        world.cycle(after=601.0, settle=False)  # four settled rechecks pass after the period: no poll storm
    assert polls(world) == asked and world.path(LIVE_FILE).read_bytes() == final
    world.cycle(after=5000.0, settle=False)  # the slow check comes due
    corrected = world.game()
    assert polls(world) == asked+1 and corrected['publish']['final'] is True
    assert corrected['pitches'][8]['actual'] != json.loads(final)['pitches'][8]['actual']
    assert corrected['publish']['sequence'] == world.read('v1/health.json')['sequence']
    world.cycle(after=601.0, settle=False)
    assert polls(world) == asked+1


def test_a_final_archive_that_arrives_locally_replaces_the_live_file_without_any_poll(world):
    import gzip
    world.config |= {'final_recheck_for_s': 0, 'final_slow_recheck_s': None}
    world.cycle(snap(half='final'), settle=False)
    assert world.game()['source']['kind'] == 'live'
    asked = polls(world)
    with gzip.open(world.feeds/f'{LIVE_GAME}_final.json.gz', 'wt') as handle:  # the nightly sync stores the final feed
        json.dump(publish_fixture.final_feed(LIVE_GAME, '2026-10-05T10:00:00Z'), handle)
    world.cycle(settle=False)  # the very next cycle, whatever the recheck timers say
    after = world.game()
    assert after['source']['kind'] == 'archive' and after['publish']['final'] is True and polls(world) == asked
    assert after['publish']['sequence'] == world.read('v1/health.json')['sequence']
    written = world.path(LIVE_FILE).read_bytes()
    for _ in range(3):
        world.cycle(settle=False)
    assert world.path(LIVE_FILE).read_bytes() == written and polls(world) == asked


# H: one game's failure stays its own
@pytest.mark.parametrize('error', [OSError('unreadable'), TypeError('malformed'), RuntimeError('anything')])
def test_a_setup_file_that_cannot_be_read_does_not_stop_the_cycle(world, error):
    world.cycle()
    assert world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()

    def view(game_pk):
        raise error
    world.publisher.setups.view = view
    health = world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert codes(health) == [(ARCHIVE_GAME, 'setup_failed')] and health['sequence'] == world.game()['publish']['sequence']
    assert not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()  # what cannot be validated is not public
    assert str(error) not in world.path('v1/health.json').read_text()


def test_a_game_file_that_cannot_be_written_does_not_stop_the_cycle(world, monkeypatch):
    world.cycle()
    kept, real = world.path(LIVE_FILE).read_bytes(), os.replace

    def failing(source, target):
        if str(target).endswith(LIVE_FILE):
            raise OSError('disk full')
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', failing)
    health = world.cycle(settle=False)
    assert codes(health) == [(LIVE_GAME, 'write_failed')] and world.read('v1/health.json')['sequence'] == 3
    assert world.path(LIVE_FILE).read_bytes() == kept and 'disk full' not in world.path('v1/health.json').read_text()
    monkeypatch.undo()
    assert codes(world.cycle(settle=False)) == [] and world.game()['publish']['sequence'] == 4


# I: a failed re-fetch is visible in health
def test_a_failed_refetch_is_counted_in_health_although_the_final_file_does_not_change(world):
    world.cycle()
    assert world.read('v1/health.json')['feed_failing_games'] == []
    world.cycle(snap('20261005_100100', half='final'), settle=False)
    final = world.path(LIVE_FILE).read_bytes()
    health = world.cycle(OSError('upstream down'), after=world.config['settled_recheck_s']+1, settle=False)
    assert health['feed_failing_games'] == [LIVE_GAME] and health['unavailable_games'] == []
    assert world.path(LIVE_FILE).read_bytes() == final
    assert world.cycle(settle=False)['feed_failing_games'] == []  # not asked this cycle: nothing to report


def test_a_disconnected_live_game_is_counted_in_health(world):
    world.cycle()
    assert world.cycle(OSError('upstream down'))['feed_failing_games'] == [LIVE_GAME]


# ---------- third review (2026-10-11) ----------
def test_an_output_folder_with_a_symlink_inside_is_refused_and_nothing_outside_is_deleted(world, tmp_path):
    world.cycle()
    world.publisher.close()
    outside = tmp_path/'outside'
    outside.mkdir()
    precious = outside/f'{ARCHIVE_GAME}.json'
    precious.write_text('{"not": "ours"}')
    game_dir = world.out/'v1/game'
    for path in game_dir.iterdir():
        path.unlink()
    game_dir.rmdir()
    game_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(publisher.UnsafeOutput, match='symbolic link'):
        world.start()  # at start-up
    assert precious.read_text() == '{"not": "ours"}'


def test_a_symlink_that_appears_while_running_stops_writes_and_deletes_through_it(world, tmp_path):
    world.cycle()
    outside = tmp_path/'outside'
    outside.mkdir()
    precious = outside/f'{ARCHIVE_GAME}.json'
    precious.write_text('{"not": "ours"}')
    game_dir = world.out/'v1/game'
    for path in game_dir.iterdir():
        path.unlink()
    game_dir.rmdir()
    game_dir.symlink_to(outside, target_is_directory=True)
    start_demo(world)  # a forced withdrawal of ARCHIVE_GAME: v1/game/<pk>.json now points outside
    health = world.cycle(settle=False)
    assert precious.read_text() == '{"not": "ours"}' and sorted(p.name for p in outside.iterdir()) == [precious.name]
    assert (ARCHIVE_GAME, 'withdraw_failed') in codes(health) and (LIVE_GAME, 'write_failed') in codes(health)
    for sink_call in (lambda: world.publisher.sink.delete(ARCHIVE_FILE), lambda: world.publisher.sink.write(ARCHIVE_FILE, b'{}'),
                      lambda: world.publisher.sink.read(ARCHIVE_FILE)):
        with pytest.raises(publisher.UnsafeOutput):
            sink_call()


def test_the_output_root_is_resolved_once_so_a_moved_link_cannot_move_the_writes(tmp_path):
    first, second, link = tmp_path/'first', tmp_path/'second', tmp_path/'out'
    first.mkdir()
    second.mkdir()
    link.symlink_to(first, target_is_directory=True)
    sink = LocalDirSink(link)
    held = publisher.Publisher(None, sink, config=publisher.load_config(env={}), state_path=tmp_path/'s.json')
    try:
        link.unlink()
        link.symlink_to(second, target_is_directory=True)  # someone repoints the link while the publisher runs
        sink.write('v1/health.json', b'{}')
        assert (first/'v1/health.json').exists() and not (second/'v1').exists()
        assert sink.root == first.resolve() and sink.lock_path.name == '.first.lock' and sink.staging.name == '.first.tmp'
        assert sink.read('v1/health.json') == b'{}' and sink.paths() == ['v1/health.json']
        with pytest.raises(publisher.AlreadyRunning):  # the folder that is locked is the folder that is written
            publisher.Publisher(None, LocalDirSink(first), config=publisher.load_config(env={}), state_path=tmp_path/'t.json')
    finally:
        held.close()


def store_archive(world, text=None):
    import gzip
    target = world.feeds/f'{LIVE_GAME}_final.json.gz'
    if text is not None:
        target.write_bytes(text)
        return
    with gzip.open(target, 'wt') as handle:
        json.dump(publish_fixture.final_feed(LIVE_GAME, '2026-10-05T10:00:00Z'), handle)


def test_an_arrived_archive_that_cannot_be_read_stays_in_health_and_is_taken_once_it_can(world):
    world.config |= {'final_recheck_for_s': 0, 'final_slow_recheck_s': None}
    world.cycle(snap(half='final'), settle=False)
    final, asked = world.path(LIVE_FILE).read_bytes(), polls(world)
    store_archive(world, b'not gzip at all')  # the sync left a broken file
    health = world.cycle(settle=False)
    assert (LIVE_GAME, 'archive_unread') in codes(health) and world.path(LIVE_FILE).read_bytes() == final
    for _ in range(3):  # between retries the problem does not vanish from health
        assert (LIVE_GAME, 'archive_unread') in codes(world.cycle(settle=False))
    assert polls(world) == asked  # and nobody was polled for it
    store_archive(world)  # the next sync writes a good one
    health = world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert codes(health) == [] and world.game()['source']['kind'] == 'archive' and world.game()['publish']['final'] is True
    assert codes(world.cycle(settle=False)) == [] and polls(world) == asked


def test_a_malformed_setup_file_on_the_very_first_cycle_does_not_stop_it(make):
    world = make(setup=False)
    lines = b''
    (world.intent/f'{ARCHIVE_GAME}.jsonl').write_bytes(lines)
    (world.intent/f'{ARCHIVE_GAME}.source.json').write_text('5')  # a pin that is not even an object
    health = world.cycle()
    assert health['sequence'] == 2 and strip(world.read('v1/dates.json'))['dates'] == [ARCHIVE_DATE]
    assert world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.served('/api/service/dates')['dates'] == [ARCHIVE_DATE]  # the route survives it too
    assert world.client.get(f'/api/watch/{ARCHIVE_GAME}/setup').status_code == 503


def test_a_delete_that_fails_does_not_stop_the_cycle_and_is_reported(world):
    world.cycle()
    start_demo(world)
    real = world.publisher.sink.delete

    def delete(path):
        raise OSError('read-only file system /secret/place')
    world.publisher.sink.delete = delete
    health = world.cycle(settle=False)
    assert (ARCHIVE_GAME, 'withdraw_failed') in codes(health) and health['sequence'] == world.game()['publish']['sequence']
    assert 'secret' not in world.path('v1/health.json').read_text()
    world.publisher.sink.delete = real
    health = world.cycle(settle=False)  # tried again in the next cycle: gone, and no longer reported
    assert codes(health) == [] and not world.path(ARCHIVE_FILE).exists() and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()


def test_recovery_reads_only_our_own_files_and_survives_hostile_json(world):
    world.cycle(settle=False)
    world.cycle(settle=False)
    world.publisher.close()
    Path(world.publisher.state_path).unlink()
    (world.out/'v1/notes.json').write_text('{"publish": {"contract": "v1", "sequence": 900}}')  # not a published name
    (world.out/'stray.json').write_text('{"publish": {"contract": "v1", "sequence": 901}}')
    (world.out/'v1/game/1.json').write_text('{"x": 1, "publish": {"contract": "v9", "sequence": 902}}')  # another contract
    (world.out/'v1/game/2.json').write_text('['*200000)  # would blow the parser's stack
    (world.out/'v1/game/3.json').write_text('{"publish":{"contract":"v1","sequence":"903"}}')
    (world.out/'v1/game/4.json').write_bytes(b'\xff\xfe'*50)
    (world.out/'v1/game/5.json').write_text('{"publish":{"contract":"v1","sequence":true}}')
    world.start()
    assert world.publisher.state['sequence'] == 2
    world.cycle(settle=False)
    assert world.read('v1/health.json')['sequence'] == 3


def test_recovery_never_reads_more_than_the_end_of_a_file(world, monkeypatch):
    world.cycle(settle=False)
    world.publisher.close()
    read = []
    monkeypatch.setattr(LocalDirSink, 'read', lambda self, path: read.append(path) or b'{}')
    world.start()
    assert read == [] and world.publisher.state['sequence'] == 1  # the tail reader was used, never a whole file
    assert len(world.publisher.sink.read_tail(LIVE_FILE, 64)) == 64


def test_a_setup_source_without_a_hash_like_commit_is_published_as_null_and_the_schema_allows_it(world):
    world.write_setup(commit='main (dirty) /Users/song/checkout')
    world.cycle()
    setup = world.read(f'v1/setup/{ARCHIVE_GAME}.json')
    assert setup['source']['commit'] is None and re.fullmatch(r'[0-9a-f]{64}', setup['source']['sha256'])
    schema = json.loads((CONTRACT_DIR/'schema/setup.schema.json').read_text())
    assert publish_schema.validate(setup, schema) == []
    assert schema['properties']['source']['properties']['commit']['type'] == ['string', 'null']


def test_the_last_embargo_check_comes_right_before_health(world):
    world.cycle()
    real, order, started = world.publisher.sink.write, [], []

    def write(path, data, *, final=False):
        order.append(path)
        real(path, data, final=final)
        if path == 'v1/manifest.json' and not started:  # everything but health is out; a demo begins now
            started.append(start_demo(world))
    world.publisher.sink.write = write
    world.cycle(settle=False)
    assert order[-1] == 'v1/health.json' and started
    assert not world.path(ARCHIVE_FILE).exists() and world.read(f'v1/games/{ARCHIVE_DATE}.json')['games'][0]['embargoed'] is True


# ---------- final review (2026-10-11) ----------
def test_a_deeply_nested_state_file_does_not_block_start_up(world):
    world.cycle(settle=False)
    world.publisher.close()
    Path(world.publisher.state_path).write_text('['*200000)
    world.start()
    world.cycle(settle=False)
    assert world.read('v1/health.json')['sequence'] == 2


def test_an_unreadable_setup_file_is_reported_but_a_failed_validation_is_not(world):
    world.cycle()
    assert world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    (world.intent/f'{ARCHIVE_GAME}.source.json').write_text('5')  # corrupt: not a pin at all
    health = world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert codes(health) == [(ARCHIVE_GAME, 'setup_failed')] and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.client.get(f'/api/watch/{ARCHIVE_GAME}/setup').status_code == 503
    world.write_setup(sha256='0'*64)  # readable, but it fails validation: simply not published, no fault
    health = world.cycle(after=world.config['settled_recheck_s'], settle=False)
    assert codes(health) == [] and not world.path(f'v1/setup/{ARCHIVE_GAME}.json').exists()
    assert world.client.get(f'/api/watch/{ARCHIVE_GAME}/setup').status_code == 503
    world.write_setup()
    assert codes(world.cycle(after=world.config['settled_recheck_s'], settle=False)) == []
    assert world.client.get(f'/api/watch/{ARCHIVE_GAME}/setup').status_code == 200
    assert world.client.get(f'/api/watch/{LIVE_GAME}/setup').status_code == 404


def test_temporary_files_left_by_a_crash_are_removed_at_start_up_and_nothing_else_is(world):
    world.cycle(settle=False)
    world.publisher.close()
    staging = world.publisher.sink.staging
    left = [staging/'v1_game_900200.json.12345.tmp', staging/'v1_health.json.7.tmp']
    kept = [staging/'notes.txt', staging/'v1_health.json.tmp', staging/'sub']
    for path in left:
        path.write_text('{"half": ')
    kept[0].write_text('mine')
    kept[1].write_text('not our pattern')
    kept[2].mkdir()
    (kept[2]/'x.1.tmp').write_text('nested: not ours to touch')
    before = tree(world)
    world.start()
    assert not any(path.exists() for path in left) and all(path.exists() for path in kept) and (kept[2]/'x.1.tmp').exists()
    assert tree(world) == before


def test_running_without_an_output_folder_is_a_clear_refusal(monkeypatch):
    monkeypatch.delenv('PITCHEEZY_PUBLISH_OUT_DIR', raising=False)
    with pytest.raises(SystemExit, match='--out'):
        publisher.main(['--once'])


def test_the_schemas_accept_the_shapes_the_builders_can_produce_beyond_the_fixtures():
    """Types decided from the builder code (publish_schema.CONTRACT_TYPES), first seen on a real replayed game."""
    schema = json.loads((CONTRACT_DIR/'schema/game.schema.json').read_text())
    body = json.loads((CONTRACT_DIR/'examples/game-live.json').read_text())
    assert publish_schema.validate(body, schema) == []
    body['game']['venue'] = 'A Ballpark'
    for holder in (body['current'], *body['pas'], *body['pitches']):
        for role in ('batter', 'pitcher'):
            if isinstance(holder.get(role), dict):
                holder[role]['jersey'] = '27'
    body['current']['context']['pitcher']['earned_runs'] = 2
    body['pas'][0]['batting_order'] = 3
    body['pas'][1].pop('batting_order', None)  # not set before the plate appearance's first pitch
    body['pas'][-1] |= {'batter': None, 'pitcher': None, 'first_index': None, 'last_index': None}
    for stat in body['current']['context']['batter']:
        body['current']['context']['batter'][stat] = None
    assert publish_schema.validate(body, schema) == []
    for change in (lambda b: b['game'].update(venue=5), lambda b: b['pitches'][0]['batter'].update(jersey=27),
                   lambda b: b['pas'][0].update(batting_order='3'), lambda b: b['pas'][0].update(first_index='0'),
                   lambda b: b['pas'][0].update(batter='x'), lambda b: b['current']['context']['pitcher'].update(earned_runs='2')):
        broken = json.loads(json.dumps(body))
        change(broken)
        assert publish_schema.validate(broken, schema), change  # widened to the builder's types, not to anything
    manifest_schema = json.loads((CONTRACT_DIR/'schema/manifest.schema.json').read_text())
    manifest = json.loads((CONTRACT_DIR/'examples/manifest-live.json').read_text())
    assert publish_schema.validate(manifest | {'commit': None, 'commit_dirty': None}, manifest_schema) == []
    assert publish_schema.validate(manifest | {'commit_dirty': 'no'}, manifest_schema)
    widened = {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                                                                     'c': {'type': 'array', 'items': {'type': 'object', 'properties': {'y': {'type': 'integer'}}}}}}
    publish_schema.widen(widened, ('a',), ('null',), optional=True)
    publish_schema.widen(widened, ('b', '*'), ('null',))
    publish_schema.widen(widened, ('c', '*', 'y'), ('number',))
    publish_schema.widen(widened, ('missing', 'path'), ('null',))  # a path the schema does not have: nothing is invented
    assert widened == {'type': 'object', 'required': ['b'], 'properties': {'a': {'type': ['integer', 'null']}, 'b': {'type': 'object', 'properties': {'x': {'type': ['string', 'null']}}},
                                                                    'c': {'type': 'array', 'items': {'type': 'object', 'properties': {'y': {'type': 'number'}}}}}}
