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
    assert not world.path(LIVE_FILE).exists() and health['unavailable_games'] == [{'game_pk': LIVE_GAME, 'status': 503}]
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


def test_without_a_store_the_file_says_the_recommendation_is_unrecorded_as_the_route_does(make):
    world = make(store=False)
    world.cycle()
    assert strip(world.game()) == world.served(f'/api/service/games/{LIVE_GAME}')
    assert world.game()['receipts'] is None


# ---------- atomic replacement ----------
def test_a_file_is_replaced_whole_and_a_failed_write_leaves_the_old_one(world, monkeypatch):
    world.cycle()
    before, real, seen = world.path(LIVE_FILE).read_bytes(), os.replace, []

    def watching(source, target):
        if str(target).endswith(LIVE_FILE):  # at the swap: the published name still holds the complete old file,
            assert Path(target).read_bytes() == before  # and the new one is complete under a hidden name
            assert Path(source).name.startswith('.') and json.loads(Path(source).read_bytes())['publish']['sequence'] == 3
            seen.append(source)
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', watching)
    world.cycle(settle=False)
    assert len(seen) == 1 and world.game()['publish']['sequence'] == 3
    monkeypatch.undo()

    kept = world.path(LIVE_FILE).read_bytes()

    def failing(source, target):
        if str(target).endswith(LIVE_FILE):
            raise OSError('disk full')
        return real(source, target)
    monkeypatch.setattr(publisher.os, 'replace', failing)
    with pytest.raises(OSError):
        world.cycle(settle=False)
    assert world.path(LIVE_FILE).read_bytes() == kept and world.read('v1/health.json')['sequence'] == 3  # health did not move
    assert all(not Path(name).name.startswith('.') for name in world.files())  # no temporary file is left behind
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
    world.config['settled_recheck_s'] = None
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
    assert strip(setup) == world.served(f'/api/watch/{ARCHIVE_GAME}/setup')
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
    with pytest.raises(publisher.NotPublic):
        world.cycle(settle=False)
    assert not world.path(LIVE_FILE).exists() and not world.path('v1/health.json').exists()


# ---------- upstream failure ----------
def test_a_game_whose_first_poll_fails_has_no_file_and_health_says_so(make):
    world = make(first=OSError('upstream down'))
    health = world.cycle(settle=False)
    assert not world.path(LIVE_FILE).exists() and health['unavailable_games'] == [{'game_pk': LIVE_GAME, 'status': 503}]
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


def test_every_example_and_every_fixture_body_validates_against_its_schema(tmp_path):
    schemas = {kind: json.loads((CONTRACT_DIR/f'schema/{kind}.schema.json').read_text()) for kind in publish_fixture.KINDS}
    for kind, names in publish_fixture.EXAMPLES.items():
        for name in names:
            assert publish_schema.validate(json.loads((CONTRACT_DIR/'examples'/name).read_text()), schemas[kind]) == [], name
    for kind, bodies in publish_fixture.collect(tmp_path).items():
        assert bodies
        for name, body in bodies:
            assert publish_schema.validate(body, schemas[kind]) == [], name


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
