"""capture_to_replay.py on synthetic capture databases (the pinned capture writer's ``snapshots`` DDL), then the
exported folder through ``ReplayTransport``/``LiveFeed``. No real capture, network or server is touched."""
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'apps/observer/backend')]
import capture_to_replay as c2r  # noqa: E402
from observer_app.live_feed import LiveFeed, ReplayTransport, event_wall  # noqa: E402
from observer_app.live_store import LiveSnapshotStore  # noqa: E402

GAME, T0 = 849831, 1791700000.0
# snapshots DDL of the pinned writer (20261009-ws-service-002 live_store.py, sha dc13e792...): no ``requested`` column.
PINNED_DDL = '''
CREATE TABLE IF NOT EXISTS snapshots (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    game_pk INTEGER NOT NULL,
    event_wall REAL NOT NULL,
    version INTEGER NOT NULL,
    feed_ts TEXT,
    sha TEXT NOT NULL,
    first_fetched REAL NOT NULL,
    fetched REAL NOT NULL,
    source TEXT NOT NULL,
    body BLOB NOT NULL,
    UNIQUE(game_pk, event_wall, version));
CREATE INDEX IF NOT EXISTS snapshots_game_event ON snapshots(game_pk, event_wall, version);
CREATE TRIGGER IF NOT EXISTS snapshots_append_only
    BEFORE UPDATE ON snapshots BEGIN SELECT RAISE(ABORT, 'append-only snapshot'); END;
'''


def feed(stamp, marker, state='Live', game=GAME):
    return {'gameData': {'game': {'pk': game, 'type': 'L', 'gameNumber': 1, 'doubleHeader': 'N'},
                         'datetime': {'officialDate': '2026-10-10', 'dateTime': '2026-10-11T00:08:00Z'},
                         'status': {'abstractGameState': state, 'detailedState': 'In Progress' if state == 'Live' else state},
                         'teams': {'away': {'id': 1, 'name': 'Away Club'}, 'home': {'id': 2, 'name': 'Home Club é'}},
                         'players': {}},
            'liveData': {'plays': {'allPlays': [], 'currentPlay': {'marker': marker}},
                         'linescore': {'teams': {'away': {'runs': marker}, 'home': {'runs': 0}}}},
            'metaData': {'timeStamp': stamp}}


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def capture_db(tmp_path, bodies, *, sha=None):
    """Rows in the given order; ``bodies`` items are (feed, slot offset s, version). ``sha`` overrides one row."""
    run = tmp_path/'run'
    run.mkdir()
    path = run/f'live-{GAME}.sqlite3'
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript(PINNED_DDL)
        for index, (body, slot, version) in enumerate(bodies):
            raw = canonical(body)
            db.execute('INSERT INTO snapshots(game_pk, event_wall, version, feed_ts, sha, first_fetched, fetched, source, body) '
                       'VALUES (?,?,?,?,?,?,?,?,?)',
                       (body['gameData']['game']['pk'], T0+slot, version, body['metaData']['timeStamp'],
                        (sha or {}).get(index, hashlib.sha256(raw).hexdigest()), T0+slot+12, T0+10*index+12, 'live_poll',
                        gzip.compress(raw, compresslevel=5)))
    assert not Path(str(path)+'-wal').exists()  # the last connection checkpointed and removed it
    return path


def receipts(db, **terminal):
    """The run-dir receipts of a cleanly finished capture whose processes are gone."""
    run = db.parent
    child = {'terminal_state': 'CLEAN', 'wal_state': 'CHECKPOINTED_TRUNCATED', 'wal_bytes': 0, 'quick_check': 'ok',
             'db': str(db), 'db_sha256': c2r.file_sha256(db), 'pid': 4000001, 'reason': 'FINAL_PLUS_ONE',
             'outcome_class': 'COMPLETED', 'game_pk': GAME} | terminal
    (run/'TERMINAL.json').write_text(json.dumps(child))
    (run/'SUPERVISOR_TERMINAL.json').write_text(json.dumps(
        {'state': 'CLEAN', 'owned_children_remaining': 0, 'nonempty_wal_preserved': []}))
    (run/'heartbeat.json').write_text(json.dumps({'phase': 'TERMINAL'}))
    (run/'owner.json').write_text(json.dumps({'pid': 4000001}))
    (run/'supervisor.json').write_text(json.dumps({'pid': 4000000}))


GONE = lambda pid: ''  # noqa: E731  no capture process: the process table is not consulted in these tests
A, B, C = feed('20261011_000100', 1, 'Preview'), feed('20261011_000200', 2), feed('20261011_000200', 3)
D, E = feed('20261011_000300', 4), feed('20261011_000400', 5, 'Final')
# seq 1 A, 2 B, 3 C (same timeStamp as B, different content: a correction), 4 B again (revert), 5 D, 6 E
ROWS = [(A, 0, 1), (B, 60, 1), (C, 60, 2), (B, 60, 3), (D, 120, 1), (E, 180, 1)]


def listing(path):
    return sorted(p.name for p in path.iterdir())


def test_export_keeps_bodies_byte_identical_in_receipt_order_and_replays(tmp_path):
    db = capture_db(tmp_path, ROWS)
    receipts(db)
    source, run_listing, out = db.read_bytes(), listing(db.parent), tmp_path/'replay'
    code, manifest = c2r.export(db, out, command=GONE)
    assert code == 0 and manifest['status'] == 'OK' and manifest['mode']['bodies_modified'] is False
    assert manifest['counts'] | {'bytes_written': 0, 'inflated_bytes': 0} == {
        'rows_read': 6, 'files_written': 5, 'duplicates_skipped': 1, 'sha_mismatches': 0, 'receipt_order_violations': 0,
        'bytes_written': 0, 'inflated_bytes': 0, 'rows_in_plan': 6, 'skipped_seqs': [4]}
    names = [f'{GAME}_20261011_000100_000001.json.gz', f'{GAME}_20261011_000200_000002.json.gz',
             f'{GAME}_20261011_000200_000003.json.gz', f'{GAME}_20261011_000300_000005.json.gz',
             f'{GAME}_20261011_000400_000006.json.gz']
    assert listing(out) == sorted(names+['MANIFEST.json', 'schedule_2026-10-10.json'])
    assert [entry['file'] for entry in manifest['files']] == names and manifest['mode']['naming'] == 'stamp-seq'
    for entry, body in zip(manifest['files'], (A, B, C, D, E)):  # unchanged: the inflated sha is the stored sha
        raw = gzip.decompress((out/entry['file']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == entry['sha256'] and raw == canonical(body)
    assert (manifest['receipt']['first_unix'], manifest['receipt']['last_unix']) == (T0+12, T0+62)
    assert manifest['source']['unchanged'] and manifest['source']['sha256'] == hashlib.sha256(source).hexdigest()
    assert manifest['source']['before']['bytes'] == len(source) and manifest['mode']['terminal_gate'] == 'PASSED'
    assert db.read_bytes() == source and listing(db.parent) == run_listing  # no -wal/-shm/journal beside the original
    assert json.loads((out/'MANIFEST.json').read_text()) == manifest

    transport = ReplayTransport(out)
    url = f'https://x/api/v1.1/game/{GAME}/feed/live'
    assert [transport(url, 1) for _ in range(7)] == [A, B, C, D, E, E, E]  # in order, then holds the last one
    games = LiveFeed(ReplayTransport(out), delay_s=0).schedule('2026-10-10')
    assert [(g['game_pk'], g['game_type'], g['live'], g['away_team'], g['home_team'], g['away_score']) for g in games] == [
        (GAME, 'L', True, 'Away Club', 'Home Club é', 2)]  # first Live body (seq 2), marked synthesized
    assert manifest['schedule'] == manifest['schedule'] | {'synthesized': True, 'from_seq': 2, 'file': 'schedule_2026-10-10.json'}
    assert LiveFeed(ReplayTransport(out), delay_s=0).schedule('2026-10-11') == []


def test_export_is_deterministic_and_refuses_a_used_output_dir(tmp_path):
    db = capture_db(tmp_path, ROWS)
    one, two = tmp_path/'one', tmp_path/'two'
    c2r.export(db, one, assume_terminal=True)
    c2r.export(db, two, assume_terminal=True)
    assert listing(one) == listing(two) and all((one/name).read_bytes() == (two/name).read_bytes() for name in listing(one))
    before = {name: (one/name).read_bytes() for name in listing(one)}
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, one, assume_terminal=True)
    assert refused.value.reason == 'OUTPUT_DIR_NOT_EMPTY' and {name: (one/name).read_bytes() for name in listing(one)} == before
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, db.parent/'replay', assume_terminal=True)
    assert refused.value.reason == 'OUTPUT_INSIDE_CAPTURE_RUN_DIR' and not (db.parent/'replay').exists()


def test_reverted_body_can_be_kept_and_backward_timestamps_switch_the_naming(tmp_path):
    late = feed('20261011_000150', 9)  # received last but stamped before B: stamp order is not receipt order
    db = capture_db(tmp_path, ROWS+[(late, 240, 1)])
    out = tmp_path/'replay'
    code, manifest = c2r.export(db, out, assume_terminal=True, keep_repeats=True)
    assert code == 0 and manifest['counts']['files_written'] == 7 and manifest['counts']['duplicates_skipped'] == 0
    assert manifest['mode']['naming'] == 'seq-stamp' and manifest['mode']['timestamp_order_is_receipt_order'] is False
    assert manifest['files'][3]['file'] == f'{GAME}_000004_20261011_000200.json.gz'
    transport = ReplayTransport(out)
    assert [transport(f'/game/{GAME}/feed/live', 1) for _ in range(7)] == [A, B, C, B, D, E, late]
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, tmp_path/'forced', assume_terminal=True, naming='stamp-seq')
    assert refused.value.reason == 'STAMP_ORDER_IS_NOT_RECEIPT_ORDER' and not (tmp_path/'forced').exists()


@pytest.mark.parametrize('breakage, failed', [
    ('no_receipts', 'TERMINAL.json terminal_state'), ('recovery', 'RECOVERY.json absent'),
    ('capturing', 'heartbeat.json phase'), ('not_clean', 'TERMINAL.json terminal_state'),
    ('supervisor', 'SUPERVISOR_TERMINAL.json state'), ('alive', 'owner.pid 4000001 is not a running capture process')])
def test_non_terminal_capture_is_refused_before_the_database_is_opened(tmp_path, monkeypatch, breakage, failed):
    db = capture_db(tmp_path, ROWS)
    command = GONE
    if breakage != 'no_receipts':
        receipts(db, **({'terminal_state': 'RECOVERY_REQUIRED'} if breakage == 'not_clean' else {}))
    if breakage == 'recovery':
        (db.parent/'RECOVERY.json').write_text('{}')
    elif breakage == 'capturing':
        (db.parent/'heartbeat.json').write_text(json.dumps({'phase': 'CAPTURING'}))
    elif breakage == 'supervisor':
        (db.parent/'SUPERVISOR_TERMINAL.json').unlink()
    elif breakage == 'alive':
        command = lambda pid: '/usr/bin/python3 /x/capture_one_game.py --game 849831' if pid == 4000001 else ''  # noqa: E731
    monkeypatch.setattr(c2r, 'open_source', lambda path: pytest.fail('database opened'))
    monkeypatch.setattr(c2r, 'file_sha256', lambda path: pytest.fail('database read'))
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, tmp_path/'replay', command=command)
    assert refused.value.reason == 'CAPTURE_NOT_TERMINAL' and not (tmp_path/'replay').exists()
    assert failed in [item['check'] for item in refused.value.detail['failed']]


def test_cli_refuses_a_non_terminal_capture_with_exit_2(tmp_path, capsys):
    db = capture_db(tmp_path, ROWS)
    assert c2r.main([str(db), str(tmp_path/'replay')]) == 2
    assert json.loads(capsys.readouterr().out)['refused'] == 'CAPTURE_NOT_TERMINAL' and not (tmp_path/'replay').exists()
    assert c2r.main([str(db), str(tmp_path/'replay'), '--assume-terminal']) == 0
    assert json.loads(capsys.readouterr().out)['mode']['terminal_gate'] == 'SKIPPED_ASSUME_TERMINAL'


def test_changed_database_or_pending_wal_is_refused(tmp_path):
    db = capture_db(tmp_path, ROWS)
    receipts(db, db_sha256='0'*64)
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, tmp_path/'replay', command=GONE)
    assert refused.value.reason == 'DB_SHA_IS_NOT_TERMINAL_RECEIPT'
    Path(str(db)+'-wal').write_bytes(b'x'*32)  # committed rows could live only here: never opened, even for tests
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(db, tmp_path/'replay', assume_terminal=True)
    assert refused.value.reason == 'WAL_NOT_EMPTY' and not (tmp_path/'replay').exists()


def test_sha_mismatch_stops_and_is_reported(tmp_path):
    db = capture_db(tmp_path, ROWS, sha={2: 'f'*64})
    out = tmp_path/'replay'
    code, manifest = c2r.export(db, out, assume_terminal=True)
    assert code == 3 and manifest['status'] == 'FAILED_BODY_CHECK' and manifest['counts']['sha_mismatches'] == 1
    assert manifest['problem']['seq'] == 3 and manifest['counts']['rows_read'] == 3 and manifest['counts']['files_written'] == 2
    assert listing(out) == [f'{GAME}_20261011_000100_000001.json.gz', f'{GAME}_20261011_000200_000002.json.gz', 'MANIFEST.json']
    assert c2r.main([str(db), str(tmp_path/'again'), '--assume-terminal']) == 3


def test_truncated_body_or_wrong_game_stops(tmp_path):
    db = capture_db(tmp_path, ROWS)
    code, manifest = c2r.export(db, tmp_path/'capped', assume_terminal=True, max_body_bytes=16)
    assert code == 3 and manifest['problem']['seq'] == 1 and manifest['counts']['files_written'] == 0
    other = tmp_path/'other'
    other.mkdir()
    foreign = capture_db(other, [(A, 0, 1), (feed('20261011_000200', 2, game=1), 60, 1)])
    with pytest.raises(c2r.Refused) as refused:
        c2r.export(foreign, tmp_path/'two-games', assume_terminal=True)
    assert refused.value.reason == 'SOURCE_NOT_SINGLE_GAME'
    code, manifest = c2r.export(foreign, tmp_path/'one-game', assume_terminal=True, game=GAME)
    assert code == 0 and manifest['counts']['rows_read'] == 1 and manifest['counts']['files_written'] == 1


def test_rebase_moves_event_wall_to_now_and_marks_bodies_modified(tmp_path):
    db = capture_db(tmp_path, ROWS)
    out, start = tmp_path/'replay', 2000000000.0
    code, manifest = c2r.export(db, out, assume_terminal=True, rebase=True, rebase_start=start)
    assert code == 0 and manifest['mode']['bodies'] == 'MODIFIED_EVENT_WALL_REBASED' and manifest['mode']['bodies_modified']
    assert manifest['mode']['rebase_shift_s'] == start-T0
    transport, served = ReplayTransport(out), []
    for _ in range(5):
        served.append(transport(f'/game/{GAME}/feed/live', 1))
    assert [event_wall(body) for body in served] == [start, start+60, start+60, start+120, start+180]  # original spacing
    for entry, body, original in zip(manifest['files'], served, (A, B, C, D, E)):
        assert entry['sha256'] == hashlib.sha256(canonical(original)).hexdigest() != entry['sha256_written']
        assert entry['sha256_written'] == hashlib.sha256(canonical(body)).hexdigest()
        assert {**body, 'metaData': original['metaData']} == original  # nothing else changed
    again = tmp_path/'again'
    c2r.export(db, again, assume_terminal=True, rebase=True, rebase_start=start)
    assert all((out/name).read_bytes() == (again/name).read_bytes() for name in listing(out))


def test_database_written_by_the_current_store_exports_and_replays(tmp_path):
    """The live writer path (WAL journal, canonical gzip bodies, slots/corrections, no stored duplicates)."""
    run = tmp_path/'run'
    store = LiveSnapshotStore(run/f'live-{GAME}.sqlite3')
    outcomes = [store.ingest(GAME, body, fetched=T0+500+10*index, source='live_poll')[0]
                for index, body in enumerate((A, B, B, C, D, E))]
    assert outcomes == ['new', 'new', 'duplicate', 'correction', 'new', 'new']
    assert not Path(str(store.path)+'-wal').exists() or not Path(str(store.path)+'-wal').stat().st_size
    code, manifest = c2r.export(store.path, tmp_path/'replay', assume_terminal=True)
    assert code == 0 and manifest['counts']['rows_read'] == 5 and manifest['counts']['files_written'] == 5
    transport = ReplayTransport(tmp_path/'replay')
    assert [transport(f'/game/{GAME}/feed/live', 1) for _ in range(5)] == [A, B, C, D, E]
    replayed = LiveSnapshotStore(tmp_path/'replayed.sqlite3')
    transport = ReplayTransport(tmp_path/'replay')
    again = [replayed.ingest(GAME, transport(f'/game/{GAME}/feed/live', 1), fetched=T0+900+index, source='replay_poll')[0]
             for index in range(5)]
    assert again == ['new', 'new', 'correction', 'new', 'new']
    with closing(sqlite3.connect(replayed.path)) as db:
        assert [row[0] for row in db.execute('SELECT sha FROM snapshots ORDER BY seq')] == [e['sha256'] for e in manifest['files']]
