"""Offline replay of a capture through the publisher (``capture_replay``, ``publish_audit``,
``scripts/publisher_replay.py``) on SYNTHETIC captures only: no real capture, network, T7, model or port."""
from contextlib import closing
import re
import threading
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parents[2]
sys.path[:0] = [str(BACKEND), str(Path(__file__).resolve().parent), str(REPO/'scripts')]
from observer_app import publish_audit, publish_schema, publisher
from observer_app.capture_replay import CaptureReplayTransport, SimClock
from observer_app.live_feed import LiveFeedError
import publisher_replay
from publish_fixture import BASE, LIVE_GAME, pregame, snap, unidentified
from test_display_state import Policy

CONTRACT = REPO/'apps/observer/publish_contract/v1'
# the capture writer's table (pinned DDL of the real capture: no ``requested`` column)
DDL = '''CREATE TABLE snapshots (seq INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL, event_wall REAL NOT NULL,
    version INTEGER NOT NULL, feed_ts TEXT, sha TEXT NOT NULL, first_fetched REAL NOT NULL, fetched REAL NOT NULL,
    source TEXT NOT NULL, body BLOB NOT NULL, UNIQUE(game_pk, event_wall, version))'''
LIMITS = {'publish_interval_s': 10, 'fixed_delay_s': 0, 'tail_cycles': 2, 'max_rss_bytes': 1 << 40, 'min_free_bytes': 0,
          'max_seconds': 600, 'settle_cap_s': 10, 'max_body_bytes': 1 << 26, 'policy_load_cap_s': 30,
          'min_free_bytes_running': 0, 'free_check_every_cycles': 1, 'policy_device': None, 'style_sha256': {},
          'capture_original_dir': None, 'policy_dirs': [], 'listing_time_limit_s': 30, 'free_floor_consecutive': 1, 'max_measure_failures': 2}


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def game_script():
    """(seconds after BASE the capture received it, body): before the game, one pitch, a second one that is not
    identified yet, the same pitch identified, the game over."""
    at = lambda seconds: BASE+seconds
    return [(0, pregame('20261005_100000', wall=at(-5))),
            (35, snap('20261005_100100', wall=at(30))),
            (62, unidentified(snap('20261005_100200', wall=at(60), second=True))),
            (71, snap('20261005_100300', wall=at(60), second=True)),
            (128, snap('20261005_100400', wall=at(125), half='final'))]


def capture_db(folder, script=None):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'capture.sqlite3'
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute(DDL)
        for version, (offset, body) in enumerate(game_script() if script is None else script, 1):
            raw = canonical(body)
            db.execute('INSERT INTO snapshots(game_pk, event_wall, version, feed_ts, sha, first_fetched, fetched, source, body) '
                       'VALUES (?,?,?,?,?,?,?,?,?)', (LIVE_GAME, BASE+offset, 1, body['metaData'].get('timeStamp'),
                                                      hashlib.sha256(raw).hexdigest(), BASE+offset, BASE+offset, 'live_poll',
                                                      gzip.compress(raw)))
    return path


# ---------- the replay input ----------
def rows_of(times, log):
    def loader(index):
        def load():
            log.append(index)
            return {'gameData': {'game': {'pk': 7}, 'datetime': {'officialDate': '2026-10-05'}, 'status': {'abstractGameState': f's{index}'}},
                    'index': index}
        return load
    return [(moment, loader(index)) for index, moment in enumerate(times)]


def transport(times, start, log):
    clock = SimClock(start)
    schedule = lambda body: {'gamePk': 7, 'officialDate': body['gameData']['datetime']['officialDate'], 'status': body['gameData']['status']}
    return clock, CaptureReplayTransport(rows_of(times, log), clock, game_pk=7, schedule_game=schedule)


GAME_URL, DAY_URL = 'https://x/api/v1.1/game/7/feed/live', 'https://x/api/v1/schedule?sportId=1&date=2026-10-05'


def test_a_request_is_answered_only_with_what_had_been_received_by_then():
    log = []
    clock, replay = transport([100.0, 130.0, 131.0, 200.0], 99.0, log)
    with pytest.raises(LiveFeedError):
        replay(GAME_URL, 10)  # nothing had been received yet
    assert replay(DAY_URL, 10) == {'dates': []} and log == []  # ...so not even the schedule knows the game
    clock.set(100.0)  # the receipt time itself counts as received
    assert replay(GAME_URL, 10)['index'] == 0
    clock.set(129.999)
    assert replay(GAME_URL, 10)['index'] == 0
    clock.set(130.5)
    assert replay(GAME_URL, 10)['index'] == 1
    clock.set(199.0)
    assert replay(GAME_URL, 10)['index'] == 2
    assert max(log) == 2 and 3 not in log  # the last body was never even read from the capture
    assert replay.index() == 2 and not replay.exhausted()


def test_bodies_come_in_receipt_order_and_a_skipped_one_is_never_served_later():
    log = []
    clock, replay = transport([100.0, 110.0, 120.0, 130.0], 100.0, log)
    served = []
    for _ in range(5):  # a poll every 15 s jumps over a version the capture received in between
        served.append(replay(GAME_URL, 10)['index'])
        clock.advance(15)
    assert served == [0, 1, 3, 3, 3] and served == sorted(served)
    assert replay.requests == 5 and replay.highest == 3


def test_the_clock_only_moves_forward_and_only_when_told():
    clock = SimClock(50.0)
    assert clock() == 50.0 and clock() == 50.0
    assert clock.advance(10) == 60.0 and clock.set(75.5) == 75.5
    for bad in (-1, float('nan')):
        with pytest.raises(ValueError):
            clock.advance(bad)
    with pytest.raises(ValueError):
        clock.set(10.0)
    assert clock() == 75.5


def test_after_the_last_receipt_the_last_body_keeps_being_the_answer():
    log = []
    clock, replay = transport([100.0, 110.0], 100.0, log)
    assert not replay.exhausted()
    clock.set(110.0)
    assert replay.exhausted() and replay(GAME_URL, 10)['index'] == 1
    clock.advance(86400)
    assert replay(GAME_URL, 10)['index'] == 1 and replay(GAME_URL, 10) is not replay(GAME_URL, 10)  # a fresh object each time
    assert replay.position() == {'served': 2, 'bodies': 2, 'requests': 4, 'duration_s': 10.0, 'elapsed_s': 86410.0}


def test_the_schedule_answer_knows_no_later_status_and_only_the_game_day():
    log = []
    clock, replay = transport([100.0, 200.0], 100.0, log)
    assert replay(DAY_URL, 10)['dates'][0]['games'][0]['status'] == {'abstractGameState': 's0'}
    assert replay(DAY_URL.replace('10-05', '10-04'), 10) == {'dates': []}
    clock.set(200.0)
    assert replay(DAY_URL, 10)['dates'][0]['games'][0]['status'] == {'abstractGameState': 's1'}


def test_the_replay_never_answers_for_anything_else():
    log = []
    _, replay = transport([100.0], 100.0, log)
    for url in ('https://x/api/v1.1/game/8/feed/live', 'https://statsapi.mlb.com/api/v1/people/1', 'https://x/'):
        with pytest.raises(LiveFeedError):
            replay(url, 10)
    for times in ([], [5.0, 4.0]):
        with pytest.raises(ValueError):
            CaptureReplayTransport(rows_of(times, log), SimClock(0), game_pk=7, schedule_game=lambda body: None)


def test_capture_rows_are_read_in_order_checked_against_their_sha_and_never_written(tmp_path):
    path = capture_db(tmp_path/'job')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    assert [fetched-BASE for fetched, _ in rows] == [0, 35, 62, 71, 128] and info['versions'] == 5 and info['loads'] == 0
    # every stored version was read and checked once, before anything runs; the first Final one is known from that
    assert (info['final_fetched'], info['final_index'], info['final_seq']) == (BASE+128, 4, 5) and info['integrity_error'] is None
    assert rows[1][1]()['metaData']['timeStamp'] == '20261005_100100' and info['loads'] == 1
    with pytest.raises(sqlite3.OperationalError):
        info['db'].execute('DELETE FROM snapshots')  # the reader's connection cannot write
    info['db'].close()
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("UPDATE snapshots SET sha='0' WHERE seq=2")
    with pytest.raises(publisher_replay.Refused, match='stored version 2'):  # a corrupted body refuses the run up front
        publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    with pytest.raises(publisher_replay.Refused):
        publisher_replay.capture_rows(path, 1, 1 << 26)
    with pytest.raises(publisher_replay.Refused, match='body limit'):
        publisher_replay.capture_rows(capture_db(tmp_path/'big'), LIVE_GAME, 100)
    assert before != hashlib.sha256(path.read_bytes()).hexdigest()  # only this test's own UPDATE changed the file
    assert sorted(p.name for p in path.parent.iterdir()) == ['capture.sqlite3']  # no -wal / -shm appeared beside it


# ---------- the harness on a synthetic capture ----------
@pytest.fixture
def replayed(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '1')
    monkeypatch.delenv('PITCHEEZY_OBSERVER_REC_LEDGER', raising=False)
    path = capture_db(tmp_path/'job'/'input')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    schemas = {p.name.split('.')[0]: json.loads(p.read_text()) for p in (CONTRACT/'schema').glob('*.schema.json')}

    def go(name='run-001', prepare=None, **over):
        rows, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
        if prepare:
            prepare(rows, info)
        options = dict(rows=rows, info=info, run_dir=tmp_path/'job'/name, game_pk=LIVE_GAME, commit='c'*40, limits=LIMITS,
                       policy_factory=lambda: Policy(), schemas=schemas, log=lambda text: None) | over
        (tmp_path/'job'/name).mkdir()
        try:
            return publisher_replay.run(**options)
        finally:
            info['db'].close()
    go.job, go.path, go.before = tmp_path/'job', path, before
    return go


def test_a_synthetic_capture_replays_to_a_final_file_with_every_invariant_held(replayed):
    summary = replayed()
    run = replayed.job/'run-001'
    assert summary['status'] == 'COMPLETED' and summary['stop'] is None
    assert summary['invariants']['held'] == {name: True for name in publish_audit.INVARIANTS} | {'nothing_outside_the_job_folder': None}
    assert summary['cycles']['by_label'] == {'replay': 13, 'tail': 2, 'settled_recheck': 1} and summary['cycles']['n'] == 16
    rows = [json.loads(line) for line in (run/'CYCLES.jsonl').read_text().splitlines()]
    assert [row['sequence'] for row in rows] == list(range(1, 17))
    # no look-ahead, cycle by cycle: the file holds exactly the pitches the capture had received by that moment
    held = lambda t: 0 if t < 35 else 9 if t < 62 else 10
    assert [row['game']['pitches'] for row in rows[:13]] == [held(10*i) if 10*i < 128 else 9 for i in range(13)]
    assert [row['capture_index'] for row in rows[:13]] == [0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 3]
    assert rows[7]['game']['unconfirmed'] == 1 and rows[7]['game']['recommendation'] == 'none'  # the unidentified pitch
    assert rows[8]['game']['unconfirmed'] == 0
    assert summary['game_file']['cycles_with_an_unconfirmed_pitch'] == 1
    assert summary['game_file']['recommendation_by_cycle']['published'] >= 2 and summary['ledger']['rows'] >= 2
    assert summary['final']['feed_final_first_version_fetched'] == BASE+128 and summary['final']['first_final_version_was_served'] is True
    assert summary['final']['publish_final']['simulated'] == BASE+130 and summary['final']['simulated_seconds_from_feed_final_to_publish_final'] == 2.0
    final_writes = [row['cycle'] for row in rows if f'v1/game/{LIVE_GAME}.json' in row['written'] and row['game']['final']]
    assert final_writes == [14]  # written once: the tail and the settled recheck left its bytes alone
    assert summary['capture']['versions_never_served'] == 0 and summary['capture']['bodies_loaded'] > 0
    assert summary['capture']['served_version_indices'] == [0, 1, 2, 3, 4] and summary['capture']['versions_never_served_indices'] == []
    assert summary['incomplete'] is False and summary['invariants']['all_held'] is True and summary['invariants']['not_applicable'] == []
    assert summary['environment']['versions']['python'] and 'torch' in summary['environment']['versions'] and summary['environment']['threads']['cpu_count']
    assert summary['memory']['free_samples'] == 16 and summary['memory']['free_bytes_min_sampled'] > 0
    assert summary['schema_validation']['validated'] and set(summary['schema_validation']['kinds_seen']) == {'manifest', 'health', 'dates', 'games', 'game'}
    assert summary['schema_validation']['kinds_never_seen'] == ['setup']
    assert all(set(problem) == {'kind', 'pointer', 'keyword', 'seen', 'count'} for problem in summary['schema_validation']['problems'])
    assert summary['bytes_by_kind']['game']['raw_max'] > summary['bytes_by_kind']['game']['gzip_max'] > 0
    assert summary['memory']['peak_rss_bytes'] > 0 and summary['data_role'] == 'SERVICE_EXPOSED_DEV'
    assert json.loads((run/'SUMMARY.json').read_text())['status'] == 'COMPLETED' and '지켜짐' in (run/'SUMMARY.md').read_text()
    # everything the run made is in its own folder; the capture copy is untouched and is not the publisher's store
    assert hashlib.sha256(replayed.path.read_bytes()).hexdigest() == replayed.before
    assert sorted(p.name for p in replayed.path.parent.iterdir()) == ['capture.sqlite3']
    assert (run/'publisher-store.sqlite3').is_file() and (run/'out/v1/health.json').is_file()
    assert sorted(p.name for p in replayed.job.iterdir()) == ['input', 'run-001']
    assert sorted(p.name for p in run.iterdir()) == ['.out.lock', '.out.tmp', 'CYCLES.jsonl', 'SUMMARY.json', 'SUMMARY.md', 'demo', 'empty',
                                                    'out', 'publisher-store.sqlite3', 'publisher.state.json', 'tmp']


def test_the_run_stops_at_the_first_broken_invariant_and_keeps_what_it_has(replayed, monkeypatch):
    real, seen = publish_audit.PublishAudit.check, []

    def check(self, path, data, sha):
        body = real(self, path, data, sha)
        seen.append(path)
        if path.endswith('health.json') and json.loads(data)['sequence'] == 3:
            self._fail('unconfirmed_pitch_blank', path, '/pitches/0/actual', 'injected by the test')
        return body
    monkeypatch.setattr(publish_audit.PublishAudit, 'check', check)
    summary = replayed()
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'invariant', 3)
    assert summary['invariants']['held']['unconfirmed_pitch_blank'] is False and summary['invariants']['violation_count'] == 1
    assert len((replayed.job/'run-001/CYCLES.jsonl').read_text().splitlines()) == 3 and '**깨짐**' in (replayed.job/'run-001/SUMMARY.md').read_text()


def test_the_run_stops_on_an_outside_write_a_memory_cap_and_a_time_cap(replayed):
    class Leaky:
        violations, read = [{'kind': 'write_outside', 'event': 'open', 'target': '/somewhere/else'}], {}
    summary = replayed('run-a', fs_audit=Leaky())
    assert summary['stop']['reason'] == 'write_process_or_network_outside_the_job_folder' and summary['cycles']['n'] == 0
    assert summary['invariants']['held']['nothing_outside_the_job_folder'] is False
    summary = replayed('run-b', rss=lambda: LIMITS['max_rss_bytes']+1)
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'memory_cap', 0)
    ticks = iter(range(0, 10**9, 400))
    summary = replayed('run-c', perf=lambda: next(ticks))
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'runtime_cap')


def test_a_policy_that_cannot_load_stops_the_run_before_any_cycle(replayed):
    def broken():
        raise RuntimeError('no model here')
    summary = replayed(policy_factory=broken)
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'policy_unavailable', 0)


def test_without_a_policy_the_replay_still_completes_and_publishes_no_recommendation(replayed):
    summary = replayed(policy_factory=None)
    assert summary['status'] == 'COMPLETED' and summary['ledger']['rows'] == 0
    assert 'published' not in summary['game_file']['recommendation_by_cycle'] and summary['policy']['requested'] is False


def test_the_command_refuses_before_writing_anything(tmp_path, capsys):
    job = tmp_path/'job'
    path = capture_db(job/'input')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    config, original = job/'replay_config.json', tmp_path/'original'
    original.mkdir()
    limits = LIMITS | {'capture_original_dir': str(original)}
    config.write_text(json.dumps(limits))
    commit = publisher_replay.head_commit()
    assert commit and len(commit) == 40
    digest = lambda: hashlib.sha256(config.read_bytes()).hexdigest()
    make = lambda: ['--job-dir', str(job), '--run-name', 'run-001', '--game', str(LIVE_GAME), '--capture-db', str(path),
                    '--capture-sha256', sha, '--commit', commit, '--config', str(config), '--config-sha256', digest(),
                    '--policy', 'off', '--allow-dirty']
    base = make()
    swap = lambda flag, value: [value if base[i-1] == flag else item for i, item in enumerate(base)]
    outside = capture_db(tmp_path/'elsewhere')
    cases = {'registered sha256': swap('--capture-sha256', '0'*64), 'registered commit': swap('--commit', '0'*40),
             'COPY inside the job folder': swap('--capture-db', str(outside)), 'plain name': swap('--run-name', '../x'),
             'plain name ': swap('--run-name', '...'), 'plain name  ': swap('--run-name', '.'), 'plain name   ': swap('--run-name', 'a..b'),
             'config does not have the registered sha256': swap('--config-sha256', '0'*64)}
    for words, argv in cases.items():
        assert publisher_replay.main(argv) == 2
        assert words.strip() in capsys.readouterr().err
    assert 'observer_app.main' in sys.modules  # this test process imported the app long ago, with its default folders:
    assert publisher_replay.main(make()) == 2 and 'before its folders could be pinned' in capsys.readouterr().err  # so a full run is refused here
    (job/'run-001').mkdir()
    assert publisher_replay.main(base) == 2 and 'already exists' in capsys.readouterr().err
    (job/'run-001').rmdir()
    config.write_text(json.dumps(limits | {'fixed_delay_s': 5}))
    assert publisher_replay.main(make()) == 2 and 'fixed delay 0' in capsys.readouterr().err
    config.write_text(json.dumps(limits | {'min_free_bytes': 1 << 60}))
    assert publisher_replay.main(make()) == 2 and 'below the registered minimum' in capsys.readouterr().err
    config.write_text(json.dumps(limits | {'capture_original_dir': str(tmp_path/'not-there')}))
    assert publisher_replay.main(make()) == 2 and 'registered as unchanged' in capsys.readouterr().err
    config.write_text(json.dumps({k: v for k, v in limits.items() if k != 'policy_device'}))
    assert publisher_replay.main(make()) == 2 and 'policy_device' in capsys.readouterr().err
    config.write_text(json.dumps(limits))
    for broken_reader in ('free_memory_vm_stat', 'free_memory_bytes'):  # could not measure is its own refusal, not "zero"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(publisher_replay, broken_reader, lambda: None)
            assert publisher_replay.main(make()) == 2 and 'could not be measured' in capsys.readouterr().err
    # a policy run needs its pins: the style files must be there with the registered sha256
    style = tmp_path/'style'
    style.mkdir()
    (style/'style_2026_index.json').write_text('{"file": "s.json"}')
    (style/'s.json').write_text('{}')
    pins = {name: hashlib.sha256((style/name).read_bytes()).hexdigest() for name in ('style_2026_index.json', 's.json')}
    config.write_text(json.dumps(limits | {'policy_device': 'mps', 'style_sha256': pins}))
    armb = [('armb' if base[i-1] == '--policy' else digest() if base[i-1] == '--config-sha256' else item) for i, item in enumerate(base)]
    (style/'s.json').write_text('{"tampered": true}')  # one byte of the style snapshot differs from its pin
    assert publisher_replay.execute(armb, style_dir=style, volume='/') == 2 and 's.json is missing or does not have its registered sha256' in capsys.readouterr().err
    assert publisher_replay.execute(armb, style_dir=style, volume=tmp_path) == 2 and 'T7 volume' in capsys.readouterr().err
    config.write_text(json.dumps(limits | {'style_sha256': pins}))
    armb[armb.index('--config-sha256')+1] = digest()
    assert publisher_replay.execute(armb, style_dir=style, volume='/') == 2 and 'registered device' in capsys.readouterr().err
    config.write_text(json.dumps(limits | {'policy_device': 'mps', 'style_sha256': pins, 'policy_dirs': [str(tmp_path/'not-there')]}))
    armb[armb.index('--config-sha256')+1] = digest()
    (style/'s.json').write_text('{}')
    assert publisher_replay.execute(armb, style_dir=style, volume='/') == 2 and 'registered as unchanged' in capsys.readouterr().err  # only a policy run needs them
    config.write_text(json.dumps(limits))
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("UPDATE snapshots SET sha='0' WHERE seq=3")
    broken = swap('--capture-sha256', hashlib.sha256(path.read_bytes()).hexdigest())
    assert publisher_replay.main(broken) == 2 and 'stored version 3' in capsys.readouterr().err  # a corrupt body: refused, not replayed
    assert sorted(p.name for p in job.iterdir()) == ['input', 'replay_config.json']  # nothing was written


def test_a_tampered_style_file_never_reaches_the_policy(tmp_path):
    style, run = tmp_path/'style', tmp_path/'run'
    style.mkdir()
    run.mkdir()
    (style/'style_2026_index.json').write_text('{"file": "s.json"}')
    (style/'s.json').write_text('{}')
    pins = {name: hashlib.sha256((style/name).read_bytes()).hexdigest() for name in ('style_2026_index.json', 's.json')}
    made = []
    import tempfile
    kept = tempfile.tempdir
    try:
        factory = publisher_replay.armb_factory(run, pins, style, make_policy=lambda root, games: made.append(root) or 'policy')
        assert factory() == 'policy' and made == [run/'policy-root'] and tempfile.tempdir == str(run/'tmp')
        assert sorted(p.name for p in (run/'policy-root/style').iterdir()) == ['s.json', 'style_2026_index.json']
        (style/'s.json').write_text('{"tampered": true}')  # changed after the start-up check, before the copy
        with pytest.raises(RuntimeError, match='registered sha256'):
            factory()
        assert len(made) == 1 and (run/'policy-root/style/s.json').read_text() == '{}'  # the tampered bytes were not copied
    finally:
        tempfile.tempdir = kept


def test_the_whole_command_runs_in_its_own_process_under_the_audit(tmp_path):
    """End to end as registered (policy off, synthetic capture): the audit hook is live for every import and
    cycle, so this also shows the publisher itself writes nowhere else, starts nothing and opens no socket."""
    job = tmp_path/'job'
    path = capture_db(job/'input')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    watched, original = tmp_path/'watched', tmp_path/'original'
    watched.mkdir()
    original.mkdir()
    (watched/'model.bin').write_bytes(b'x')
    (original/'live.sqlite3').write_bytes(b'the original capture')
    (job/'replay_config.json').write_text(json.dumps(LIMITS | {'policy_dirs': [str(watched)], 'capture_original_dir': str(original)}))
    argv = [sys.executable, '-B', str(REPO/'scripts/publisher_replay.py'), '--job-dir', str(job), '--run-name', 'run-001',
            '--game', str(LIVE_GAME), '--capture-db', str(path), '--capture-sha256', sha, '--commit', publisher_replay.head_commit(),
            '--config', str(job/'replay_config.json'), '--config-sha256', hashlib.sha256((job/'replay_config.json').read_bytes()).hexdigest(),
            '--policy', 'off', '--allow-dirty']
    env = {key: value for key, value in os.environ.items() if not key.startswith('PITCHEEZY_')}
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads((job/'run-001/SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETED' and summary['fs_audit']['violations'] == [] and summary['fs_audit']['files_read_under_watch'] == {}
    assert 'tripwire' in summary['fs_audit']['kind']
    # policy off: the policy folders named in the config are not part of the check at all
    assert summary['outside_check'] | {'method': 0} == {'method': 0, 'capture': 'passed', 'worktree': 'passed', 'policy': 'not_applicable',
                                                        'caches': 'not_applicable', 'overall': 'passed', 'changes': {}}
    assert 'transient' in summary['outside_check']['method'] and summary['invariants']['all_held'] is True
    assert summary['invariants']['not_applicable'] and summary['incomplete'] is False
    assert sorted(p.name for p in (job/'run-001').iterdir() if 'SUMMARY' in p.name or 'STATUS' in p.name or 'INCOMPLETE' in p.name) == ['SUMMARY.json', 'SUMMARY.md']
    assert json.loads((job/'run-001/RUN.json').read_text())['versions']['python']
    assert summary['invariants']['held']['nothing_outside_the_job_folder'] is True
    assert summary['capture_copy'] == {'sha256_before': sha, 'sha256_after': sha, 'unchanged': True}
    assert sorted(p.name for p in job.iterdir()) == ['input', 'replay_config.json', 'run-001']
    assert sorted(p.name for p in (job/'input').iterdir()) == ['capture.sqlite3']
    assert not (REPO/'apps/observer/live_snapshots').exists() or not any(
        p.stat().st_mtime > (job/'run-001/RUN.json').stat().st_mtime for p in (REPO/'apps/observer/live_snapshots').iterdir())


# ---------- each invariant fails on a deliberately bad body ----------
def example(name):
    return json.loads((CONTRACT/'examples'/name).read_text())


def audit_of(**options):
    return publish_audit.PublishAudit(**options)


def ledger_of(body):
    """The ledger rows a body's recommendations would have come from (what the store would hold)."""
    rows = {}
    for _, rec, pitch_key in publish_audit.recommendations(body):
        ledger = rec.get('ledger') or {}
        if ledger.get('decision_id'):
            rows[ledger['decision_id']] = {
                'decision_id': ledger['decision_id'], 'outcome': 'first', 'pitch_key': pitch_key, 'decided': ledger.get('decided'),
                'policy_identity': str((rec.get('policy') or {}).get('identity'))+'0'*56,
                'displayed': {'status': 'ready', 'recommendation': {'candidates': json.loads(json.dumps(rec['candidates']))}}}
    return rows


def check(audit, path, body):
    data = json.dumps(body, ensure_ascii=False).encode()
    audit.check(path, data, hashlib.sha256(data).hexdigest())
    return [v['invariant'] for v in audit.violations]


GAME_PATH = 'v1/game/900200.json'


def test_the_committed_examples_break_no_invariant():
    audit = audit_of(ledger_rows=lambda: ledger_of(example('game-live.json')), forbidden=['/Users/song'])
    names = {'game-live.json': GAME_PATH, 'games-live.json': 'v1/games/2026-10-05.json', 'dates.json': 'v1/dates.json',
             'health-live.json': 'v1/health.json', 'manifest-live.json': 'v1/manifest.json', 'setup.json': 'v1/setup/900101.json',
             'game-final-archive.json': 'v1/game/900101.json'}
    for name, path in names.items():
        assert check(audit, path, example(name)) == [], name
    assert check(audit_of(), 'v1/game/900200.json', example('game-live-unconfirmed.json')) == []
    assert audit.checked == 7


def test_an_unconfirmed_pitch_with_any_observed_detail_fails():
    for key in ('pitch_type', 'speed_mph', 'x', 'z', 'event', 'pitch_label', 'zone_label', 'play_text', 'description'):
        body = example('game-live-unconfirmed.json')
        assert body['pitches'][-1]['receipt']['status'] == 'unconfirmed'
        body['pitches'][-1]['actual'][key] = 1.5
        audit = audit_of()
        assert check(audit, GAME_PATH, body) == ['unconfirmed_pitch_blank']
        assert audit.violations[0]['pointer'] == f"/pitches/{len(body['pitches'])-1}/actual" and key in audit.violations[0]['detail']
        assert '1.5' not in json.dumps(audit.violations)  # the report names the key, never the value
    body = example('game-live.json')
    del body['pitches'][0]['receipt']
    assert check(audit_of(ledger_rows=lambda: ledger_of(body)), GAME_PATH, body) == ['unconfirmed_pitch_blank']


def test_an_unconfirmed_pitch_whose_labels_do_not_read_the_fixed_words_fails():
    for key in ('call_label', 'result_label'):
        for value in ('Ball', None, '확인중'):
            body = example('game-live-unconfirmed.json')
            body['pitches'][-1]['actual'][key] = value
            audit = audit_of()
            assert check(audit, GAME_PATH, body) == ['unconfirmed_pitch_blank'], (key, value)
            assert key in audit.violations[0]['detail'] and 'Ball' not in json.dumps(audit.violations, ensure_ascii=False)


def test_a_recommendation_without_a_ledger_row_fails():
    body = example('game-live.json')
    assert publish_audit.recommendations(body)[0][0] == '/next/rec'
    held = lambda: ledger_of(body)
    assert check(audit_of(ledger_rows=held), GAME_PATH, body) == []
    assert check(audit_of(ledger_rows=lambda: {}), GAME_PATH, body) == ['recommendation_in_ledger']
    unrecorded = example('game-live.json')
    unrecorded['next']['rec']['ledger'] = None
    assert check(audit_of(ledger_rows=held), GAME_PATH, unrecorded) == ['recommendation_in_ledger']
    hidden = example('game-live.json')
    hidden['display_state']['recommendation_shown'] = False
    assert check(audit_of(ledger_rows=held), GAME_PATH, hidden) == ['recommendation_in_ledger']
    wrong_state = example('game-live.json')
    wrong_state['display_state']['state'] = 'unconfirmed'
    assert check(audit_of(ledger_rows=held), GAME_PATH, wrong_state) == ['recommendation_in_ledger']
    past = example('game-live.json')
    past['pitches'][0]['rec'] = dict(past['next']['rec'], ledger={'status': 'recorded', 'decision_id': 'not-in-the-ledger'})
    assert check(audit_of(ledger_rows=held), GAME_PATH, past) == ['recommendation_in_ledger']


def test_a_recommendation_that_differs_from_its_ledger_row_fails():
    """The id exists, but what is published is not what the ledger recorded for it."""
    good = example('game-live.json')
    held = lambda: ledger_of(good)
    changes = {'candidates': [lambda b: b['next']['rec']['candidates'][0].update(pitch_type='XX'),
                              lambda b: b['next']['rec']['candidates'][0].update(zone_id='middle'),
                              lambda b: b['next']['rec']['candidates'].append(dict(b['next']['rec']['candidates'][0], rank=2)),
                              lambda b: b['next']['rec']['candidates'][0]['detail'].update(probability=0.99)],
               'pitch_key': [lambda b: b['next'].update(pitch_key='900200:9:9')],
               'decided': [lambda b: b['next']['rec']['ledger'].update(decided=1.0)],
               'policy': [lambda b: b['next']['rec']['policy'].update(identity='ffffffff'),
                          lambda b: b['next']['rec']['policy'].update(identity=b['next']['rec']['policy']['identity'][:4]),  # a shorter prefix
                          lambda b: b['next']['rec']['policy'].update(identity=''),
                          lambda b: b['next']['rec']['policy'].update(identity=b['next']['rec']['policy']['identity']+'0'),
                          lambda b: b['next']['rec']['policy'].update(identity=None)]}
    for name, mutations in changes.items():
        for change in mutations:
            body = example('game-live.json')
            change(body)
            audit = audit_of(ledger_rows=held)
            assert check(audit, GAME_PATH, body) == ['recommendation_in_ledger'], name
            assert name in audit.violations[0]['detail'] and audit.violations[0]['pointer'] == '/next/rec'
            assert 'XX' not in json.dumps(audit.violations) and '0.99' not in json.dumps(audit.violations)
    # the same recommendation hung on a past pitch it was not recorded for, or under another provenance
    moved = example('game-live.json')
    moved['pitches'][0]['rec'] = json.loads(json.dumps(moved['next']['rec']))
    audit = audit_of(ledger_rows=held)
    assert check(audit, GAME_PATH, moved) == ['recommendation_in_ledger'] and audit.violations[0]['pointer'] == '/pitches/0/rec'
    relabelled = example('game-live.json')
    relabelled['pitches'][0]['rec'] = dict(json.loads(json.dumps(relabelled['next']['rec'])), provenance='stored')
    assert check(audit_of(ledger_rows=held), GAME_PATH, relabelled) == ['recommendation_in_ledger']  # not skipped for its label
    conflict = ledger_of(good)
    next(iter(conflict.values()))['outcome'] = 'conflict'
    assert check(audit_of(ledger_rows=lambda: conflict), GAME_PATH, example('game-live.json')) == ['recommendation_in_ledger']


def test_a_sequence_that_goes_down_or_stays_while_the_bytes_change_fails():
    body = example('dates.json')
    audit = audit_of()
    assert check(audit, 'v1/dates.json', body) == []
    assert check(audit, 'v1/dates.json', body) == []  # the same bytes again are fine
    lower = json.loads(json.dumps(body))
    lower['publish']['sequence'] -= 1
    assert check(audit, 'v1/dates.json', lower) == ['sequence_monotonic']
    audit = audit_of()
    check(audit, 'v1/dates.json', body)
    same = json.loads(json.dumps(body))
    same['dates'] = same['dates']+['2026-10-01']
    assert check(audit, 'v1/dates.json', same) == ['sequence_monotonic']
    assert check(audit_of(), 'v1/dates.json', {'dates': []}) == ['sequence_monotonic']  # no publish block at all


def test_a_final_file_rewritten_without_a_content_change_fails():
    body = example('game-final.json')
    assert body['publish']['final'] is True
    ledger = lambda: ledger_of(body)
    audit = audit_of(ledger_rows=ledger)
    assert check(audit, GAME_PATH, body) == []
    moved = json.loads(json.dumps(body))
    moved['publish']['sequence'] += 5
    moved['feed']['as_of'] = '2026-10-05T11:00:00Z'  # only the clock moved
    moved['display_state']['feed']['last_ok_at'] = '2026-10-05T11:00:00Z'
    assert check(audit, GAME_PATH, moved) == ['final_stable']
    audit = audit_of(ledger_rows=ledger)
    check(audit, GAME_PATH, body)
    corrected = json.loads(json.dumps(body))
    corrected['publish']['sequence'] += 5
    corrected['pitches'][0]['actual']['pitch_type'] = 'CH'  # a real correction: allowed
    assert check(audit, GAME_PATH, corrected) == []


def test_a_path_a_credential_a_demo_marker_or_a_forbidden_string_fails():
    cases = [lambda b: b['game'].update(path='feeds/x.json'), lambda b: b['game'].update(venue='/Volumes/T7 Shield/x'),
             lambda b: b['game'].update(venue='see /Users/song/notes'), lambda b: b.update(token='abc'),
             lambda b: b['source'].update(kind='demo'), lambda b: b['source'].update(session_id='abc'),
             lambda b: b['game'].update(venue='the-secret-word')]
    for change in cases:
        body = example('game-final-archive.json')
        change(body)
        audit = audit_of(forbidden=['the-secret-word', ''])
        assert check(audit, 'v1/game/900101.json', body) == ['no_private_strings'], change
        assert 'the-secret-word' not in json.dumps(audit.violations) and '/Users/song' not in json.dumps(audit.violations)
    setup = example('setup.json')
    setup['source']['path'] = 'docs/x.jsonl'
    assert check(audit_of(), 'v1/setup/900101.json', setup) == ['no_private_strings']


def test_a_file_that_is_not_one_of_the_published_names_fails():
    for path in ('v1/notes.json', 'v1/game/abc.json', 'v2/health.json', 'health.json', 'v1/games/2026-10-05.json.bak'):
        assert check(audit_of(), path, example('health-live.json')) == ['only_public_names']
    assert [publish_audit.kind_of(p) for p in ('v1/health.json', 'v1/games/2026-10-05.json', 'v1/setup/5.json')] == ['health', 'games', 'setup']
    audit = audit_of()
    check(audit, 'v1/dates.json', example('dates.json'))
    audit.gone('v1/dates.json')  # withdrawn... and back: the sequence history is not forgotten
    for sequence, expected in ((1, ['sequence_monotonic']), (example('dates.json')['publish']['sequence'], ['sequence_monotonic'])):
        fresh = audit_of()
        check(fresh, 'v1/dates.json', example('dates.json'))
        fresh.gone('v1/dates.json')
        back = example('dates.json')
        back['publish']['sequence'] = sequence
        assert check(fresh, 'v1/dates.json', back) == expected
    higher = example('dates.json')
    higher['publish']['sequence'] += 1
    assert check(audit, 'v1/dates.json', higher) == []


def test_schema_problems_name_the_place_and_the_type_never_the_value():
    schema = json.loads((CONTRACT/'schema/game.schema.json').read_text())
    body = example('game-live.json')
    assert publish_schema.problems(body, schema) == []
    body['pitches'][0]['actual']['speed_mph'] = 'secret-value'
    body['pitches'][1].pop('actual')
    body['publish']['contract'] = 'v9'
    body['game']['new/key'] = 1
    body['pregame'] = {'basis': 'x', 'as_of': 'x', 'players': {'secret-player-id': {'era': {}}}}
    found = publish_schema.problems(body, schema)
    assert 'secret-player-id' not in json.dumps(found)  # a key from the data is folded, never reported
    assert {'pointer': '/pitches/0/actual/speed_mph', 'keyword': 'type', 'seen': 'string'} in found
    assert {'pointer': '/pitches/1/actual', 'keyword': 'required', 'seen': 'missing'} in found
    assert {'pointer': '/publish/contract', 'keyword': 'const', 'seen': 'string'} in found
    assert 'secret-value' not in json.dumps(found) and 'v9' not in json.dumps(found)
    assert publisher_replay.general('/pitches/12/actual/x') == '/pitches/*/actual/x'
    assert publisher_replay.general('/pregame/players/660271/era') == '/pregame/players/*/era'


# ---------- the filesystem / process / network tripwire ----------
def test_writes_outside_the_allowed_folder_are_refused_and_reads_under_a_watched_one_are_listed(tmp_path):
    inside, outside, watched = tmp_path/'run', tmp_path/'other', tmp_path/'volume'
    for folder in (inside, outside, watched):
        folder.mkdir()
    audit = publish_audit.FsAudit([inside], watch=[watched])
    audit.event('open', (str(inside/'a.json'), 'w', os.O_WRONLY | os.O_CREAT))
    audit.event('open', (str(outside/'a.json'), 'r', os.O_RDONLY))
    audit.event('open', ('/dev/null', 'w', os.O_WRONLY))
    audit.event('open', (1, 'w', os.O_WRONLY))  # stdout
    audit.event('os.rename', (str(inside/'a'), str(inside/'b'), -1, -1))
    audit.event('sqlite3.connect', (str(inside/'store.sqlite3'),))
    audit.event('sqlite3.connect', (f'file:{outside}/c.sqlite3?mode=ro&immutable=1',))
    audit.event('sqlite3.connect', (':memory:',))
    audit.event('sqlite3.connect', ('file::memory:?cache=shared',))
    audit.event('open', (str(watched/'model.pt'), 'rb', os.O_RDONLY))
    audit.event('open', (str(watched/'model.pt'), 'rb', os.O_RDONLY))
    audit.event('sqlite3.connect', (f'file:{watched}/w.sqlite3?mode=ro',))
    audit.event('some.other.event', ())
    assert audit.violations == []
    assert audit.read == {os.path.realpath(watched/'model.pt'): 2, os.path.realpath(watched/'w.sqlite3'): 1}
    refused = [('open', (str(outside/'a.json'), 'w', os.O_WRONLY | os.O_CREAT)), ('open', (str(outside/'a.json'), 'a', None)),
               ('open', (str(watched/'x'), 'r+', os.O_RDWR)), ('os.remove', (str(outside/'a.json'), -1)),
               ('os.rename', (str(inside/'a'), str(outside/'a'), -1, -1)), ('os.rename', (str(outside/'a'), str(inside/'a'), -1, -1)),
               ('os.mkdir', (str(outside/'new'), 0o777, -1)), ('shutil.rmtree', (str(watched), None)),
               ('sqlite3.connect', (str(outside/'c.sqlite3'),)), ('sqlite3.connect', (f'file:{outside}/c.sqlite3?mode=rwc',)),
               ('open', (str(inside/'..'/'other'/'sneaky'), 'w', os.O_WRONLY))]
    for name, args in refused:
        with pytest.raises(publish_audit.Forbidden):
            audit.event(name, args)
    assert [v['kind'] for v in audit.violations] == ['write_outside']*len(refused)
    link = inside/'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(publish_audit.Forbidden):  # a link out of the folder does not make the target allowed
        audit.event('open', (str(link/'a.json'), 'w', os.O_WRONLY))


def test_sqlite_is_read_only_only_when_its_own_mode_parameter_says_so(tmp_path):
    target = publish_audit.sqlite_target
    assert target(':memory:') == (None, False) and target('file::memory:') == (None, False)
    assert target('/a/b.sqlite3') == ('/a/b.sqlite3', False)
    assert target('file:/a/b.sqlite3?mode=ro&immutable=1') == ('/a/b.sqlite3', True)
    assert target('file:/a/b%20c.sqlite3?mode=ro') == ('/a/b c.sqlite3', True)
    # the words "mode=ro" somewhere in the text are not a read-only open
    for tricky in ('file:/a/mode=ro.sqlite3', 'file:/a/b.sqlite3?mode=rwc&x=mode=ro', 'file:/a/b.sqlite3?vfs=mode=ro',
                   'file:/a/b.sqlite3?mode=rw', 'file:/a/b.sqlite3?immutable=1', 'file:/a/b.sqlite3?mode=ro&mode=rwc',
                   '/a/b.sqlite3?mode=ro'):
        assert target(tricky)[1] is False, tricky
    audit = publish_audit.FsAudit([tmp_path/'run'])
    for tricky in (f'file:{tmp_path}/x/mode=ro.sqlite3', f'file:{tmp_path}/x.sqlite3?mode=rwc&x=mode=ro'):
        with pytest.raises(publish_audit.Forbidden):
            audit.event('sqlite3.connect', (tricky,))


def test_descriptors_and_descriptor_relative_paths_are_judged_by_where_they_really_are(tmp_path):
    inside, outside = tmp_path/'run', tmp_path/'other'
    for folder in (inside, outside):
        folder.mkdir()
    (outside/'victim').write_text('x')
    audit = publish_audit.FsAudit([inside])
    in_fd, out_fd = os.open(inside, os.O_RDONLY), os.open(outside, os.O_RDONLY)
    file_fd = os.open(outside/'victim', os.O_RDONLY)
    try:
        assert publish_audit.fd_path(in_fd) == os.path.realpath(inside)
        # what tempfile's cleanup does: a relative name against a directory descriptor inside the allowed folder
        audit.event('os.remove', ('leftover', in_fd))
        audit.event('os.rmdir', ('sub', in_fd))
        audit.event('os.mkdir', ('sub', 0o700, in_fd))
        audit.event('os.rename', ('a', 'b', in_fd, in_fd))
        assert audit.violations == []
        for name, args in (('os.remove', ('victim', out_fd)), ('os.rmdir', ('sub', out_fd)), ('os.mkdir', ('sub', 0o700, out_fd)),
                           ('os.rename', ('a', 'b', in_fd, out_fd)), ('os.rename', ('a', 'b', out_fd, in_fd)),
                           ('open', (file_fd, 'w', os.O_WRONLY))):  # reopening for writing a descriptor that points outside
            with pytest.raises(publish_audit.Forbidden):
                audit.event(name, args)
    finally:
        for fd in (in_fd, out_fd, file_fd):
            os.close(fd)
    blind = publish_audit.FsAudit([inside], fd_path=lambda fd: None)  # a descriptor nobody can name is not trusted
    for name, args in (('os.remove', ('x', 7)), ('open', (7, 'w', os.O_WRONLY))):
        with pytest.raises(publish_audit.Forbidden):
            blind.event(name, args)
    blind.event('os.remove', (str(inside/'x'), 7))  # an absolute path does not need the descriptor
    assert (outside/'victim').read_text() == 'x'


def test_under_a_live_hook_temp_cleanup_inside_works_and_a_write_outside_does_not(tmp_path):
    script = ('import os, sys, tempfile, json\n'
              'from observer_app import publish_audit\n'
              'run, other = sys.argv[1], sys.argv[2]\n'
              'os.chdir(run)\n'
              'audit = publish_audit.FsAudit([run])\n'
              'sys.addaudithook(audit.event)\n'
              'tempfile.tempdir = os.path.join(run, "tmp"); os.mkdir(tempfile.tempdir)\n'
              'scratch = tempfile.TemporaryDirectory(prefix="pitcheezy-live-")\n'
              'open(os.path.join(scratch.name, "ledger.jsonl"), "w").write("x")\n'
              'os.mkdir(os.path.join(scratch.name, "deep")); open(os.path.join(scratch.name, "deep", "f"), "w").write("y")\n'
              'scratch.cleanup()\n'
              'open("relative.txt", "w").write("in the run folder")\n'
              'clean = list(audit.violations)\n'
              'blocked = []\n'
              'fd = os.open(other, os.O_RDONLY)\n'
              'for attempt in (lambda: open(os.path.join(other, "a"), "w"), lambda: os.remove(os.path.join(other, "victim")),\n'
              '                lambda: os.unlink("victim", dir_fd=fd), lambda: os.mkdir("d", dir_fd=fd),\n'
              '                lambda: __import__("subprocess").run(["true"]),\n'
              '                lambda: __import__("socket").create_connection(("127.0.0.1", 9), timeout=0.2)):\n'
              '    try:\n'
              '        attempt(); blocked.append(False)\n'
              '    except publish_audit.Forbidden:\n'
              '        blocked.append(True)\n'
              'print(json.dumps({"clean": clean, "blocked": blocked, "left": os.listdir(tempfile.tempdir), "kinds": [v["kind"] for v in audit.violations]}))\n')
    run, other = tmp_path/'run', tmp_path/'other'
    run.mkdir()
    other.mkdir()
    (other/'victim').write_text('x')
    done = subprocess.run([sys.executable, '-B', '-c', script, str(run), str(other)], capture_output=True, text=True, timeout=60,
                          env=os.environ | {'PYTHONPATH': str(BACKEND)}, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout)
    assert result['clean'] == [] and result['left'] == []  # the scratch folder was removed, nothing was refused
    assert result['blocked'] == [True]*6 and result['kinds'] == ['write_outside']*4+['process', 'network']
    assert (other/'victim').read_text() == 'x' and sorted(p.name for p in other.iterdir()) == ['victim']
    assert (run/'relative.txt').read_text() == 'in the run folder'


def test_a_new_process_or_a_socket_is_refused():
    audit = publish_audit.FsAudit(['/nowhere'])
    for name, args in (('subprocess.Popen', ('git', ['git', 'status'], None, None)), ('os.system', (b'ls',)), ('os.fork', ()),
                       ('os.posix_spawn', ('/bin/ls', [], {})), ('os.exec', ('/bin/ls', [], {}))):
        with pytest.raises(publish_audit.Forbidden):
            audit.event(name, args)
    for name, args in (('socket.connect', (None, ('statsapi.mlb.com', 443))), ('socket.getaddrinfo', ('statsapi.mlb.com', 443, 0, 0, 0)),
                       ('socket.bind', (None, ('127.0.0.1', 8766)))):
        with pytest.raises(publish_audit.Forbidden):
            audit.event(name, args)
    assert [v['kind'] for v in audit.violations] == ['process']*5+['network']*3
    quiet = publish_audit.FsAudit(['/nowhere'], block=False)
    quiet.event('socket.connect', (None, ('x', 1)))
    assert quiet.violations[0]['kind'] == 'network'  # recorded even when it is not blocking
    quiet.armed = False
    quiet.event('os.remove', ('/etc/passwd', -1))
    assert len(quiet.violations) == 1


def test_the_before_and_after_listing_sees_what_the_tripwire_cannot(tmp_path):
    root = tmp_path/'watched'
    (root/'sub').mkdir(parents=True)
    (root/'a.bin').write_bytes(b'12345')
    (root/'sub/b.bin').write_bytes(b'1')
    (root/'skipped').mkdir()
    before, reason = publish_audit.tree_snapshot(root, skip=('skipped',))
    assert reason is None
    assert set(before) == {'a.bin', 'sub', os.path.join('sub', 'b.bin'), 'skipped'} and before['a.bin'][0] == 5
    assert publish_audit.tree_changes(before, publish_audit.tree_snapshot(root, skip=('skipped',))[0]) == []
    (root/'skipped/x').write_text('not looked at')
    os.utime(root/'a.bin', ns=(1, 1))  # same size, another mtime
    (root/'sub/b.bin').unlink()
    (root/'sub/c.bin').write_bytes(b'new')
    changes = publish_audit.tree_changes(before, publish_audit.tree_snapshot(root, skip=('skipped',))[0])
    assert ('changed', 'a.bin') in changes and ('removed', os.path.join('sub', 'b.bin')) in changes
    assert ('added', os.path.join('sub', 'c.bin')) in changes and not any('x' == os.path.basename(path) for _, path in changes)
    ticks = iter(range(100))
    assert publish_audit.tree_snapshot(root, deadline=0, clock=lambda: next(ticks)) == (None, 'timeout')  # not a listing at all
    assert publish_audit.tree_snapshot(root, deadline=1000, clock=lambda: next(ticks))[1] is None
    assert publish_audit.tree_snapshot(tmp_path/'not-there') == (None, 'walk_error')
    with pytest.MonkeyPatch.context() as patch:  # an entry that cannot be examined is a reason, never a silent gap
        real = os.lstat
        patch.setattr(publish_audit.os, 'lstat', lambda path: (_ for _ in ()).throw(OSError('gone')) if str(path).endswith('a.bin') else real(path))
        assert publish_audit.tree_snapshot(root) == (None, 'lstat_error')
    locked = root/'locked'
    locked.mkdir()
    locked.chmod(0)
    try:
        assert publish_audit.tree_snapshot(root) == (None, 'walk_error')  # a folder that cannot be read
    finally:
        locked.chmod(0o700)


def test_the_in_process_memory_reading_agrees_with_vm_stat():
    if sys.platform != 'darwin':
        pytest.skip('Mach host statistics')
    seen, tool = publisher_replay.free_memory_bytes(), publisher_replay.free_memory_vm_stat()
    assert seen and tool and abs(seen-tool) < 3 << 30  # two instants on a busy machine: the same quantity, not the same number


# ---------- stop conditions reach the runner; every exit leaves a record ----------
def test_a_capture_body_that_fails_its_check_mid_run_stops_the_run(replayed):
    def corrupt(rows, info):
        fetched, load = rows[2]

        def bad():
            info['integrity_error'] = {'seq': 3, 'error': 'ValueError'}
            raise ValueError('stored version 3 does not match its sha256')
        rows[2] = (fetched, bad)
    summary = replayed(prepare=corrupt)
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'capture_integrity') and summary['stop']['detail']['seq'] == 3
    assert summary['capture']['integrity_error'] == {'seq': 3, 'error': 'ValueError'} and summary['incomplete'] is True
    assert summary['cycles']['n'] == 8  # the cycle that first asked for that version, not the end of the game
    assert json.loads((replayed.job/'run-001/SUMMARY.json').read_text())['incomplete'] is True and summary['invariants']['all_held'] is False


def test_a_policy_computation_that_fails_stops_the_run(replayed):
    class Failing(Policy):
        def __call__(self, snapshot):
            raise RuntimeError('bound components differ from the S2-certified identity; refusing to run')
    summary = replayed(policy_factory=lambda: Failing())
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'policy_compute_failed')
    assert summary['stop']['detail'] == {'states': 1, 'errors': ['RuntimeError']} and summary['policy']['failed_states'] == 1
    assert summary['cycles']['n'] == 5 and summary['ledger']['rows'] == 0  # the first cycle with a pitch to recommend for


class Bound(Policy):
    """A fake policy that exposes its members' device the way ``demo_precompute.LivePolicy`` holds them."""
    device = 'mps'

    def __init__(self):
        super().__init__()
        from types import SimpleNamespace as N
        member = N(model=N(global_model=N(device=self.device)))
        self.bindings, self.binds = {900200: {'components': N(g0=N(models=[member, member]))}}, 1


def test_the_policy_device_is_recorded_and_a_different_one_stops_the_run(replayed):
    summary = replayed(limits=LIMITS | {'policy_device': 'mps'}, policy_factory=lambda: Bound())
    assert summary['status'] == 'COMPLETED' and summary['policy']['devices_seen'] == ['mps'] and summary['policy']['registered_device'] == 'mps'

    class OnCpu(Bound):
        device = 'cpu'
    summary = replayed('run-b', limits=LIMITS | {'policy_device': 'mps'}, policy_factory=lambda: OnCpu())
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'policy_device')
    assert summary['stop']['detail'] == {'registered': 'mps', 'seen': ['cpu']} and summary['cycles']['n'] == 1

    class Unreadable(Policy):
        binds = 1
    summary = replayed('run-c', limits=LIMITS | {'policy_device': 'mps'}, policy_factory=lambda: Unreadable())
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'policy_device_unknown')
    assert publisher_replay.policy_devices(object()) is None


def test_low_free_memory_during_the_run_stops_it(replayed):
    readings = iter([8 << 30, 8 << 30, 1 << 30])
    summary = replayed(limits=LIMITS | {'min_free_bytes_running': 2 << 30}, free_memory=lambda: next(readings))
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'free_memory_low', 3)
    assert summary['memory']['free_bytes_min_sampled'] == 1 << 30


def test_one_momentary_dip_does_not_stop_a_healthy_run_but_a_lasting_one_does(replayed):
    floor = LIMITS | {'min_free_bytes_running': 2 << 30, 'free_floor_consecutive': 3}
    dips = iter([8 << 30, 1 << 30, 8 << 30, 1 << 30, 1 << 30, 8 << 30]+[8 << 30]*20)  # never three in a row
    summary = replayed(limits=floor, free_memory=lambda: next(dips))
    assert summary['status'] == 'COMPLETED' and summary['memory']['measurement']['free_below_floor_checks'] == 3
    lasting = iter([8 << 30, 1 << 30, 1 << 30, 1 << 30])
    summary = replayed('run-b', limits=floor, free_memory=lambda: next(lasting))
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'free_memory_low', 4)
    assert summary['stop']['detail']['checks_in_a_row'] == 3


def test_a_reading_that_cannot_be_taken_is_not_a_zero(replayed):
    # free memory: one failure is recorded and tolerated, N in a row stop the run with their own reason
    flaky = iter([8 << 30, None, 8 << 30]+[8 << 30]*20)
    summary = replayed(free_memory=lambda: next(flaky))
    assert summary['status'] == 'COMPLETED' and summary['memory']['measurement']['free_failures'] == 1
    rows = [json.loads(line) for line in (replayed.job/'run-001/CYCLES.jsonl').read_text().splitlines()]
    assert rows[1]['free_checked'] is True and rows[1]['free_bytes'] is None
    summary = replayed('run-b', free_memory=lambda: None)
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'measurement_failed', 2)
    assert summary['stop']['detail'] == {'what': 'free_memory', 'checks_in_a_row': 2}

    def broken():
        raise publisher_replay.MeasureError('MPS memory: RuntimeError')
    summary = replayed('run-c', mps=broken)
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'measurement_failed') and summary['stop']['detail']['what'] == 'mps_memory'
    assert summary['stop']['detail'] == {'what': 'mps_memory', 'cycles_in_a_row': 2} and summary['cycles']['n'] == 2  # cycles, not calls
    assert summary['memory']['mps_driver_allocated_peak_bytes'] is None and summary['memory']['measurement']['mps_unknown_cycles'] == 2
    assert summary['memory']['measurement']['mps_failed_readings'] > 2  # several readings per cycle failed: counted once per cycle
    rows = [json.loads(line) for line in (replayed.job/'run-c/CYCLES.jsonl').read_text().splitlines()]
    assert [(row['mps_bytes'], row['mps_known']) for row in rows] == [(None, False)]*2  # unknown, not zero
    summary = replayed('run-d', mps=lambda: 3 << 20)  # a real reading counts against the cap and is reported
    assert summary['status'] == 'COMPLETED' and summary['memory']['mps_driver_allocated_peak_bytes'] == 3 << 20
    summary = replayed('run-e', mps=lambda: LIMITS['max_rss_bytes'])
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'memory_cap')
    summary = replayed('run-f', limits=LIMITS | {'min_free_bytes_running': 2 << 30, 'free_check_every_cycles': 5}, free_memory=lambda: 8 << 30)
    assert summary['status'] == 'COMPLETED' and summary['memory']['free_samples'] == 4


@pytest.mark.parametrize('error, status, reason', [(RuntimeError('unexpected'), 'FAILED', 'exception'),
                                                   (publisher_replay.Interrupted('SIGTERM'), 'INTERRUPTED', 'signal')])
def test_an_unexpected_exception_or_a_signal_still_leaves_a_summary_and_an_incomplete_mark(replayed, monkeypatch, error, status, reason):
    real, calls = publisher.Publisher.cycle, []

    def cycle(self):
        calls.append(1)
        if len(calls) == 3:
            raise error
        return real(self)
    monkeypatch.setattr(publisher.Publisher, 'cycle', cycle)
    summary = replayed()
    run = replayed.job/'run-001'
    assert (summary['status'], summary['stop']['reason'], summary['incomplete'], summary['cycles']['n']) == (status, reason, True, 2)
    on_disk = json.loads((run/'SUMMARY.json').read_text())
    assert on_disk['status'] == status and '미완료' in (run/'SUMMARY.md').read_text()
    assert on_disk['incomplete'] is True and on_disk['cycles']['n'] == 2 and on_disk['invariants']['all_held'] is False
    # the lock was released on the way out: the folder is not left held
    publisher.Publisher(None, publisher.LocalDirSink(run/'out'), config=publisher.load_config(env={}), state_path=run/'x.json').close()


def test_a_failure_before_the_first_cycle_is_recorded_too(replayed, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError('no space left on device')
    monkeypatch.setattr(publisher, 'build', broken)
    summary = replayed()
    assert (summary['status'], summary['stop']['detail']['type'], summary['cycles']['n']) == ('FAILED', 'OSError', 0)
    assert summary['invariants']['held']['sequence_monotonic'] is None  # nothing was checked: not claimed as held
    assert json.loads((replayed.job/'run-001/SUMMARY.json').read_text())['status'] == 'FAILED'


def test_one_record_says_how_the_run_stands_and_its_final_form_is_the_last_write(tmp_path, monkeypatch):
    publisher_replay.mark(tmp_path, 'RUNNING', phase='starting')
    first = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert (first['status'], first['incomplete'], first['phase']) == ('RUNNING', True, 'starting')
    with pytest.raises(AssertionError):
        publisher_replay.mark(tmp_path, 'COMPLETED')  # a progress record can never claim completion
    done = {'status': 'COMPLETED', 'incomplete': False, 'stop': None, 'game_pk': 1, 'cycles': {'n': 3}}
    writes, real = [], Path.write_text

    def write_text(self, *args, **kwargs):
        writes.append(self.name)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', write_text)
    publisher_replay.finish(tmp_path, done)
    assert len(writes) == 2 and writes[0] == 'SUMMARY.md' and re.fullmatch(r'\.SUMMARY\.json\.\d+\.tmp', writes[1])
    assert json.loads((tmp_path/'SUMMARY.json').read_text()) == done  # the record last
    assert sorted(p.name for p in tmp_path.iterdir()) == ['SUMMARY.json', 'SUMMARY.md']  # no second status file to disagree
    # a write that fails on the way: nothing is written after it, and the record still says what it said before
    publisher_replay.mark(tmp_path, 'POST_RUN_CHECKS_PENDING')
    writes.clear()

    def failing(self, *args, **kwargs):
        writes.append(self.name)
        if self.name == 'SUMMARY.md':
            raise OSError('disk full')
        return real(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', failing)
    with pytest.raises(OSError):
        publisher_replay.finish(tmp_path, done)
    after = json.loads((tmp_path/'SUMMARY.json').read_text())
    assert writes == ['SUMMARY.md'] and (after['status'], after['incomplete']) == ('POST_RUN_CHECKS_PENDING', True)


def test_a_version_between_two_polls_is_reported_as_never_served_and_the_first_final_counts_even_then(tmp_path, monkeypatch):
    monkeypatch.setenv('PITCHEEZY_OBSERVER_LIVE_ARMB', '1')
    at = lambda seconds: BASE+seconds
    script = [(0, pregame('20261005_100000', wall=at(-5))), (35, snap('20261005_100100', wall=at(30))),
              (42, snap('20261005_100200', wall=at(41), second=True)),  # received at 42, replaced at 44: no poll sees it
              (44, snap('20261005_100300', wall=at(43), second=True, call='C')),
              (121, snap('20261005_100400', wall=at(120), half='final')),  # the first Final version: also skipped
              (126, snap('20261005_100500', wall=at(125), half='final', call='C'))]
    path = capture_db(tmp_path/'job'/'input', script)
    rows, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    (tmp_path/'job/run-001').mkdir()
    try:
        summary = publisher_replay.run(rows=rows, info=info, run_dir=tmp_path/'job/run-001', game_pk=LIVE_GAME, commit='c'*40, limits=LIMITS,
                                       policy_factory=lambda: Policy(), log=lambda text: None)
    finally:
        info['db'].close()
    assert summary['status'] == 'COMPLETED'
    assert summary['capture']['served_version_indices'] == [0, 1, 3, 5] and summary['capture']['versions_never_served_indices'] == [2, 4]
    assert summary['capture']['versions_never_served'] == 2 and summary['capture']['versions_served'] == 4
    final = summary['final']
    assert (final['feed_final_first_version_fetched'], final['feed_final_first_version_index'], final['first_final_version_was_served']) == (BASE+121, 4, False)
    assert final['publish_final']['simulated'] == BASE+130 and final['simulated_seconds_from_feed_final_to_publish_final'] == 9.0


def test_the_schedule_cache_follows_the_injected_clock(tmp_path, monkeypatch):
    """``service_wall``: without it a fast replay would keep one cached game list for 20 real seconds."""
    from observer_app.main import create_app
    now, calls = [1000.0], []

    class Feed:
        failing = {}

        def schedule(self, day):
            calls.append(day)
            return []
    app = create_app(object(), start_worker=False, live_feed=Feed(), watch_dir=tmp_path, feed_dir=tmp_path, demo_dir=tmp_path/'demo',
                     intent_dir=tmp_path, service_wall=lambda: now[0])
    bodies = app.state.service_bodies
    bodies.games('2026-10-05')
    bodies.games('2026-10-05')
    assert len(calls) == 2  # two US dates, asked once
    now[0] += 21
    bodies.games('2026-10-05')
    assert len(calls) == 4


# ---------- the whole command with the policy on, and its last steps ----------
DRIVER = '''
import json, os, sys, tempfile
from types import SimpleNamespace as N
sys.path[:0] = [{tests!r}, {backend!r}, {scripts!r}]
import publisher_replay
mode, style_dir, leak = sys.argv[1], sys.argv[2], sys.argv[3]
fd = os.open(leak, os.O_WRONLY | os.O_APPEND) if mode == 'leak' else None  # opened before the tripwire exists
if mode == 'post_fail':
    def broken(*args, **kwargs):
        raise RuntimeError('the post-run check itself broke')
    publisher_replay.outside_check = broken


def Fake(output_root, max_games):
    """Stands in for demo_precompute.LivePolicy: reads its style cache where it was told to, makes its scratch
    folder through tempfile like the real bind, and holds its members on a device. Built only when the run asks
    for the policy, like the real one (the app is not imported before its folders are pinned)."""
    from test_display_state import Policy
    policy = Policy()
    index = json.loads((output_root/'style'/'style_2026_index.json').read_text())
    assert (output_root/'style'/index['file']).is_file() and max_games >= 1
    policy.scratch = tempfile.TemporaryDirectory(prefix='pitcheezy-live-')
    open(os.path.join(policy.scratch.name, 'ledger.jsonl'), 'w').write('header')
    (output_root/'scratch-path.txt').write_text(policy.scratch.name)
    (output_root/'env.json').write_text(json.dumps({{name: os.environ.get(name) for name in ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'PYTORCH_KERNEL_CACHE_PATH', 'TMPDIR')}}))
    member = N(model=N(global_model=N(device='mps')))
    policy.bindings, policy.binds = {{900200: {{'components': N(g0=N(models=[member, member]))}}}}, 1
    if fd is not None:
        os.write(fd, b'x')  # a write no audit event reports: only the after-the-fact check can see it
    return policy


sys.exit(publisher_replay.execute(sys.argv[4:], make_policy=Fake, style_dir=style_dir, volume='/dev'))
'''


@pytest.fixture
def command(tmp_path):
    job, style, policy_dir, original = tmp_path/'job', tmp_path/'style', tmp_path/'policy', tmp_path/'original'
    for folder in (style, policy_dir, original):
        folder.mkdir()
    path = capture_db(job/'input')
    (style/'style_2026_index.json').write_text('{"file": "style_2026-03-25.json"}')
    (style/'style_2026-03-25.json').write_text('{"snapshot": 1}')
    (policy_dir/'model.pt').write_bytes(b'weights')
    (original/'live.sqlite3').write_bytes(b'the original capture')
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in style.iterdir()}
    config = job/'replay_config.json'
    config.write_text(json.dumps(LIMITS | {'policy_device': 'mps', 'style_sha256': pins, 'policy_dirs': [str(policy_dir)],
                                           'capture_original_dir': str(original)}))
    driver = tmp_path/'driver.py'
    driver.write_text(DRIVER.format(tests=str(Path(__file__).resolve().parent), backend=str(BACKEND), scripts=str(REPO/'scripts')))
    env = {key: value for key, value in os.environ.items() if not key.startswith('PITCHEEZY_') and key != 'TMPDIR'}

    def go(mode, policy='armb'):
        argv = [sys.executable, '-B', str(driver), mode, str(style), str(policy_dir/'model.pt'), '--job-dir', str(job), '--run-name', 'run-001',
                '--game', str(LIVE_GAME), '--capture-db', str(path), '--capture-sha256', hashlib.sha256(path.read_bytes()).hexdigest(),
                '--commit', publisher_replay.head_commit(), '--config', str(config),
                '--config-sha256', hashlib.sha256(config.read_bytes()).hexdigest(), '--policy', policy, '--allow-dirty']
        return subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    go.run, go.job = job/'run-001', job
    return go


def test_the_whole_command_with_the_policy_on_runs_to_a_completed_record(command):
    """The registered code path end to end with a stand-in policy: start-up pins, the tripwire, the temp-folder
    redirection, the style copy, the device check, the replay and the before/after check, in one process."""
    done = command('clean')
    assert done.returncode == 0, done.stderr[-3000:]
    summary, run = json.loads((command.run/'SUMMARY.json').read_text()), command.run
    assert (summary['status'], summary['incomplete'], summary['stop']) == ('COMPLETED', False, None)
    assert summary['policy']['devices_seen'] == ['mps'] and summary['policy']['requested'] is True and summary['policy']['final_status']['state'] == 'ready'
    assert summary['game_file']['recommendation_by_cycle']['published'] >= 2 and summary['ledger']['rows'] >= 2
    assert summary['invariants']['held'] == {name: True for name in publish_audit.INVARIANTS} | {'nothing_outside_the_job_folder': True}
    assert summary['fs_audit']['violations'] == []
    assert summary['outside_check']['overall'] == 'passed' and summary['outside_check']['capture'] == summary['outside_check']['worktree'] == 'passed'
    assert list(summary['outside_check']['policy'].values()) == ['passed'] and summary['invariants']['all_held'] is True
    assert summary['capture_copy']['unchanged'] is True
    assert sorted(p.name for p in (run/'policy-root/style').iterdir()) == ['style_2026-03-25.json', 'style_2026_index.json']
    scratch = Path((run/'policy-root/scratch-path.txt').read_text())  # where tempfile put the policy's scratch ledger
    assert scratch.parent == (run/'tmp').resolve() and scratch.name.startswith('pitcheezy-live-')
    seen_env = json.loads((run/'policy-root/env.json').read_text())  # what the policy saw: every cache points into the run folder
    assert all(value and Path(value).resolve().is_relative_to(run.resolve()) for value in seen_env.values()), seen_env
    assert summary['outside_check']['caches'].keys() == {'~/.cache', '~/Library/Caches'}
    assert summary['ledger']['cycles_adding_rows_without_a_published_recommendation'] == []
    assert summary['policy']['computations'] >= 2 and summary['policy']['compute_s']['n'] == summary['policy']['computations']
    assert summary['policy']['live_wait_s'] == 1.0 and summary['policy']['computations_over_the_live_wait'] == 0
    assert sorted(p.name for p in run.iterdir() if p.is_file()) == ['.out.lock', 'CYCLES.jsonl', 'RUN.json', 'SUMMARY.json', 'SUMMARY.md',
                                                                    'publisher-store.sqlite3', 'publisher.state.json']
    assert sorted(p.name for p in command.job.iterdir()) == ['input', 'replay_config.json', 'run-001']


def test_a_change_outside_that_the_tripwire_cannot_see_fails_the_run_at_the_very_end(command):
    done = command('leak')
    assert done.returncode == 3, done.stderr[-3000:]
    summary, run = json.loads((command.run/'SUMMARY.json').read_text()), command.run
    assert summary['status'] == 'FAILED' and summary['incomplete'] is True
    assert summary['stop']['reason'] == 'something_outside_the_job_folder_changed_or_could_not_be_checked'
    assert summary['fs_audit']['violations'] == []  # the tripwire saw nothing: the listing did
    assert list(summary['outside_check']['changes'].values()) == [[['changed', 'model.pt']]] and summary['outside_check']['overall'] == 'failed'
    assert list(summary['outside_check']['policy'].values()) == ['failed']
    assert summary['invariants']['held']['nothing_outside_the_job_folder'] is False and summary['invariants']['all_held'] is False
    assert summary['cycles']['n'] == 16  # the replay itself ran to its end; COMPLETED was never written
    assert '미완료' in (run/'SUMMARY.md').read_text() and not (run/'STATUS.json').exists()


def test_an_exception_after_the_replay_leaves_a_failed_record_that_does_not_say_all_held(command):
    done = command('post_fail')
    assert done.returncode == 3, done.stderr[-3000:]
    summary = json.loads((command.run/'SUMMARY.json').read_text())
    assert (summary['status'], summary['incomplete'], summary['stop']['reason']) == ('FAILED', True, 'exception_outside_the_replay')
    assert summary['cycles']['n'] == 16 and summary['invariants']['all_held'] is False  # the replay had completed with every invariant held
    assert all(value is True for value in summary['invariants']['held'].values())


def test_summary_writes_are_serialised_and_each_writer_has_its_own_temporary_name(tmp_path, monkeypatch):
    names, inside, overlap, real = set(), [0], [], publisher_replay._replace_record

    def watched(run_dir, record):
        inside[0] += 1
        overlap.append(inside[0])
        names.add(threading.get_ident())
        try:
            return real(run_dir, record)
        finally:
            inside[0] -= 1
    monkeypatch.setattr(publisher_replay, '_replace_record', watched)
    workers = [threading.Thread(target=lambda n=n: [publisher_replay.mark(tmp_path, 'RUNNING', writer=n, i=i) for i in range(40)]) for n in range(6)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert max(overlap) == 1 and len(names) == 6  # never two writers at once
    assert json.loads((tmp_path/'SUMMARY.json').read_text())['status'] == 'RUNNING'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['SUMMARY.json']  # no temporary file left
    seen = []
    monkeypatch.setattr(Path, 'write_text', lambda self, *a, **k: seen.append(self.name))
    monkeypatch.setattr(publisher_replay.os, 'replace', lambda *a: None)
    monkeypatch.undo() if False else None
    real(tmp_path, {'x': 1})
    assert seen == [f'.SUMMARY.json.{threading.get_ident()}.tmp']


def test_the_replay_record_is_not_called_completed_before_the_last_check(replayed):
    summary = replayed(finalize=False)
    run = replayed.job/'run-001'
    assert summary['status'] == 'COMPLETED'  # what the replay itself found...
    on_disk = json.loads((run/'SUMMARY.json').read_text())
    assert (on_disk['status'], on_disk['incomplete'], on_disk['replay_status']) == ('POST_RUN_CHECKS_PENDING', True, 'COMPLETED')
    assert not (run/'SUMMARY.md').exists()
    publisher_replay.finish(run, summary)
    assert json.loads((run/'SUMMARY.json').read_text())['status'] == 'COMPLETED' and (run/'SUMMARY.md').exists()


def test_the_outside_check_compares_exactly_what_it_says(tmp_path, monkeypatch):
    original, policy_dir = tmp_path/'original', tmp_path/'policy'
    original.mkdir()
    policy_dir.mkdir()
    (original/'live.sqlite3').write_bytes(b'12345')
    (policy_dir/'model.pt').write_bytes(b'weights')
    limits = {'capture_original_dir': str(original), 'policy_dirs': [str(policy_dir)], 'listing_time_limit_s': 30}
    caches = {'~/.cache': {'a': 1}, '~/Library/Caches': {'com.example': 5}}
    monkeypatch.setattr(publisher_replay, 'cache_entries', lambda folder: dict(caches['~/'+folder.split(os.path.expanduser('~')+os.sep, 1)[1]]))
    porcelain = ['?? new.txt']
    monkeypatch.setattr(publisher_replay, 'git_porcelain', lambda repo=None: list(porcelain))
    before = publisher_replay.outside_state(limits, True)
    assert publisher_replay.outside_check(before, publisher_replay.outside_state(limits, True), True) | {'method': 0} == {
        'method': 0, 'capture': 'passed', 'worktree': 'passed', 'policy': {str(policy_dir): 'passed'}, 'overall': 'passed', 'changes': {},
        'caches': {'~/.cache': 'unchanged', '~/Library/Caches': 'unchanged'}, 'cache_entries_changed': {}}
    stamp = (original/'live.sqlite3').stat()
    (original/'live.sqlite3').write_bytes(b'12346')  # same size, and the old mtime put back: the sha256 still sees it
    os.utime(original/'live.sqlite3', ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    porcelain.append(' M scripts/x.py')
    changed = publisher_replay.outside_check(before, publisher_replay.outside_state(limits, True), True)
    assert (changed['capture'], changed['worktree'], changed['overall']) == ('failed', 'failed', 'failed')
    assert changed['changes'] == {'capture': ['live.sqlite3'], 'worktree': [' M scripts/x.py']}
    (original/'live.sqlite3').write_bytes(b'12345')
    porcelain.pop()
    # a policy folder that could not be listed in time is "not checked" (allowed), with its reason
    late = publisher_replay.outside_check(before, publisher_replay.outside_state(limits | {'listing_time_limit_s': -1}, True), True)
    assert late['policy'] == {str(policy_dir): 'not_checked: timeout'} and late['overall'] == 'not_checked' and late['changes'] == {}
    # git status that cannot be read is a failure, not "not checked"
    monkeypatch.setattr(publisher_replay, 'git_porcelain', lambda repo=None: None)
    no_git = publisher_replay.outside_check(before, publisher_replay.outside_state(limits, True), True)
    assert no_git['worktree'].startswith('failed') and no_git['overall'] == 'failed'
    gone = publisher_replay.outside_check(before, publisher_replay.outside_state(limits | {'capture_original_dir': str(tmp_path/'x')}, True), True)
    assert gone['capture'].startswith('failed')
    monkeypatch.undo()
    assert publisher_replay.git_porcelain(REPO) is not None
    # a home cache entry that appears or is re-stamped is recorded and never fails the run (the OS writes there too)
    monkeypatch.setattr(publisher_replay, 'git_porcelain', lambda repo=None: [])
    monkeypatch.setattr(publisher_replay, 'cache_entries', lambda folder: dict(caches['~/'+folder.split(os.path.expanduser('~')+os.sep, 1)[1]]))
    clean = publisher_replay.outside_state(limits, True)
    caches['~/Library/Caches']['com.apple.metal'] = 9
    caches['~/.cache']['a'] = 2
    touched = publisher_replay.outside_check(clean, publisher_replay.outside_state(limits, True), True)
    assert touched['caches'] == {'~/.cache': 'changed_outside: caches', '~/Library/Caches': 'changed_outside: caches'}
    assert touched['cache_entries_changed'] == {'~/.cache': ['a'], '~/Library/Caches': ['com.apple.metal']} and touched['overall'] == 'passed'
    monkeypatch.undo()
    folder = tmp_path/'cachedir'
    (folder/'sub').mkdir(parents=True)
    (folder/'sub/deep').write_text('never read')
    (folder/'x').write_text('1')
    assert set(publisher_replay.cache_entries(folder)) == {'sub', 'x'} and publisher_replay.cache_entries(tmp_path/'none') is None
    monkeypatch.setattr(publisher_replay, 'CACHE_ENTRY_LIMIT', 1)
    assert publisher_replay.cache_entries(folder) is None  # too many to list: not checked, not a partial listing
    monkeypatch.undo()
    monkeypatch.setattr(publisher_replay, 'cache_entries', lambda folder: {})
    # without the policy the policy folders are not looked at at all
    monkeypatch.setattr(publisher_replay, 'git_porcelain', lambda repo=None: [])
    walked = []
    monkeypatch.setattr(publish_audit, 'tree_snapshot', lambda *args, **kwargs: walked.append(args) or ({}, None))
    off = publisher_replay.outside_state(limits, False)
    assert off['policy'] == {} and walked == []
    assert off['caches'] is None and publisher_replay.outside_check(off, off, False)['policy'] == 'not_applicable'
    assert publisher_replay.outside_check(off, off, False)['caches'] == 'not_applicable'


WATCHDOG = '''
import sys, time
sys.path[:0] = [{backend!r}, {scripts!r}]
import publisher_replay
mode, run_dir = sys.argv[1], sys.argv[2]
if mode == 'record_fails':
    def broken(*args, **kwargs):
        raise OSError('disk full')
    publisher_replay._replace_record = broken
peak = (lambda: 1/0) if mode == 'cannot_measure' else None
limits = {{'max_rss_bytes': 1, 'max_seconds': 3600}} if mode != 'time' else {{'max_rss_bytes': 1 << 60, 'max_seconds': -60}}
publisher_replay.start_watchdog(run_dir, limits, peak=peak, interval=.05)
if mode == 'lock_held':  # someone else holds the record lock for longer than the watchdog will wait
    publisher_replay.RECORD_LOCK_TIMEOUT_S = .1
    publisher_replay.mark(run_dir, 'RUNNING', phase='replay')
    publisher_replay.RECORD_LOCK.acquire()
if mode == 'racing':  # the main flow keeps writing its own progress record as fast as it can
    while True:
        publisher_replay.mark(run_dir, 'RUNNING', phase='replay')
time.sleep(30)  # a main thread that never gets to its own checks
sys.exit(0)
'''


@pytest.mark.parametrize('mode, why', [('memory', 'memory_cap'), ('time', 'runtime_cap_hard'), ('cannot_measure', 'watchdog_could_not_measure'),
                                       ('racing', 'memory_cap'), ('record_fails', None)])
def test_the_watchdog_ends_its_own_process_whatever_happens_while_it_records(tmp_path, mode, why):
    script = tmp_path/'watchdog.py'
    script.write_text(WATCHDOG.format(backend=str(BACKEND), scripts=str(REPO/'scripts')))
    child = subprocess.Popen([sys.executable, '-B', str(script), mode, str(tmp_path)], cwd=tmp_path)
    try:
        assert child.wait(timeout=20) == 87  # this test's own child, ended by its own watchdog
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=20)
    if why is None:
        assert not (tmp_path/'SUMMARY.json').exists()  # the record could not be written; the process still ended
    else:
        assert json.loads((tmp_path/'ABORTED.json').read_text())['why'] == why
        state = json.loads((tmp_path/'SUMMARY.json').read_text())
        assert (state['status'], state['incomplete'], state['stop']['reason']) == ('ABORTED', True, why)


# ---------- final round: policy OFF touches nothing of the policy; the registered K / N values ----------
OFF_DRIVER = '''
import json, os, sys
sys.path[:0] = [{backend!r}, {scripts!r}]
volume, log = os.path.realpath(sys.argv[1]), sys.argv[2]
touched = []


def under(path):
    try:
        text = os.path.abspath(os.fsdecode(path))  # no system call: this runs inside the wrapped ones
    except (TypeError, ValueError):
        return False
    return text == volume or text.startswith(volume+os.sep) or text == '/Volumes' or text.startswith('/Volumes/')  # the fake volume AND the real one


def wrap(name):
    real = getattr(os, name)

    def call(path, *args, **kwargs):
        if not isinstance(path, int) and under(path):
            touched.append([name, os.fsdecode(path)])
        return real(path, *args, **kwargs)
    setattr(os, name, call)


for name in ('stat', 'lstat', 'scandir', 'listdir', 'open'):
    wrap(name)
sys.addaudithook(lambda event, args: touched.append([event, str(args[0])]) if event in ('open', 'os.scandir', 'os.listdir') and args
                 and isinstance(args[0], (str, bytes)) and under(args[0]) else None)
import publisher_replay
code = publisher_replay.execute(sys.argv[3:], style_dir=os.path.join(volume, 'style'), volume=volume)
open(log, 'w').write(json.dumps(touched))
sys.exit(code)
'''


def test_a_policy_off_run_never_touches_the_external_volume_or_any_policy_folder(tmp_path):
    """The volume, the style folder and the policy folders all exist and are even named in the config: an OFF run
    must not stat, list or open any of them."""
    job, volume, original = tmp_path/'job', tmp_path/'volume', tmp_path/'original'
    (volume/'style').mkdir(parents=True)
    (volume/'policy').mkdir()
    (volume/'policy/model.pt').write_bytes(b'weights')
    (volume/'style/style_2026_index.json').write_text('{}')
    original.mkdir()
    (original/'live.sqlite3').write_bytes(b'the original capture')
    path = capture_db(job/'input')
    config = job/'replay_config.json'
    config.write_text(json.dumps(LIMITS | {'capture_original_dir': str(original), 'policy_dirs': [str(volume/'policy'), str(volume/'missing')],
                                           'policy_device': 'mps', 'style_sha256': {'style_2026_index.json': '0'*64}}))
    driver, log = tmp_path/'driver.py', tmp_path/'touched.json'
    driver.write_text(OFF_DRIVER.format(backend=str(BACKEND), scripts=str(REPO/'scripts')))
    argv = [sys.executable, '-B', str(driver), str(volume), str(log), '--job-dir', str(job), '--run-name', 'run-off-001', '--game', str(LIVE_GAME),
            '--capture-db', str(path), '--capture-sha256', hashlib.sha256(path.read_bytes()).hexdigest(), '--commit', publisher_replay.head_commit(),
            '--config', str(config), '--config-sha256', hashlib.sha256(config.read_bytes()).hexdigest(), '--policy', 'off', '--allow-dirty']
    env = {key: value for key, value in os.environ.items() if not key.startswith('PITCHEEZY_')}
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    assert json.loads(log.read_text()) == []  # nothing under the fake volume or under /Volumes/ was stat'ed, listed or opened
    summary = json.loads((job/'run-off-001/SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETED' and summary['policy']['requested'] is False and summary['outside_check']['policy'] == 'not_applicable'
    assert summary['fs_audit']['files_read_under_watch'] == {} and summary['memory']['mps_driver_allocated_peak_bytes'] is None
    assert summary['invariants']['all_held'] is True and summary['ledger']['rows'] == 0
    assert 'torch' not in done.stderr and not (job/'run-off-001/policy-root').exists()


def test_the_registered_memory_rules_k3_every_10_cycles_n3(replayed):
    registered = LIMITS | {'tail_cycles': 50, 'min_free_bytes_running': 2 << 30, 'free_check_every_cycles': 10,
                           'free_floor_consecutive': 3, 'max_measure_failures': 3}
    low, fine = 1 << 30, 8 << 30
    # checks happen at cycles 1, 11, 21, 31, 41, 51: two low ones, a good one, then three low in a row
    readings = iter([low, low, fine, low, low, low])
    summary = replayed(limits=registered, free_memory=lambda: next(readings))
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'free_memory_low', 51)
    assert summary['stop']['detail']['checks_in_a_row'] == 3 and summary['memory']['measurement']['free_below_floor_checks'] == 5
    rows = [json.loads(line) for line in (replayed.job/'run-001/CYCLES.jsonl').read_text().splitlines()]
    assert [row['cycle'] for row in rows if row['free_checked']] == [1, 11, 21, 31, 41, 51]
    # an unreadable check between two low ones neither breaks the streak nor extends it
    readings = iter([low, None, low, None, low])
    summary = replayed('run-b', limits=registered, free_memory=lambda: next(readings))
    assert (summary['stop']['reason'], summary['cycles']['n'], summary['stop']['detail']['checks_in_a_row']) == ('free_memory_low', 41, 3)
    # three unreadable checks in a row are their own stop
    summary = replayed('run-c', limits=registered, free_memory=lambda: None)
    assert (summary['stop']['reason'], summary['cycles']['n']) == ('measurement_failed', 21)
    # MPS unreadable for three CYCLES in a row (whatever the number of readings inside a cycle)
    calls = []

    def flaky():
        calls.append(1)
        raise publisher_replay.MeasureError('MPS memory: RuntimeError')
    summary = replayed('run-d', limits=registered, mps=flaky)
    assert (summary['stop']['reason'], summary['cycles']['n'], summary['stop']['detail']) == ('measurement_failed', 3, {'what': 'mps_memory', 'cycles_in_a_row': 3})
    assert len(calls) > 3 and summary['memory']['measurement']['mps_unknown_cycles'] == 3


def test_a_writer_that_cannot_take_the_record_lock_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher_replay, 'RECORD_LOCK_TIMEOUT_S', .05)
    publisher_replay.mark(tmp_path, 'RUNNING', phase='replay')
    before = (tmp_path/'SUMMARY.json').read_bytes()
    done = {'status': 'COMPLETED', 'incomplete': False, 'stop': None, 'game_pk': 1, 'cycles': {'n': 3}}
    publisher_replay.RECORD_LOCK.acquire()  # held longer than the timeout, as the watchdog holds it while it aborts
    try:
        with pytest.raises(publisher_replay.RecordBusy):
            publisher_replay.write_record(tmp_path, done)
        with pytest.raises(publisher_replay.RecordBusy):  # so the main flow cannot end the run as COMPLETED
            publisher_replay.finish(tmp_path, done)
        with pytest.raises(publisher_replay.RecordBusy):
            publisher_replay.mark(tmp_path, 'POST_RUN_CHECKS_PENDING')
        assert (tmp_path/'SUMMARY.json').read_bytes() == before
        assert not any(p.name.startswith('.SUMMARY') for p in tmp_path.iterdir())
    finally:
        publisher_replay.RECORD_LOCK.release()
    publisher_replay.write_record(tmp_path, done)  # free again: written
    assert json.loads((tmp_path/'SUMMARY.json').read_text()) == done
    # the watchdog itself: it cannot take the lock either, writes nothing, and still ends its process
    child_dir = tmp_path/'child'
    child_dir.mkdir()
    script = tmp_path/'watchdog.py'
    script.write_text(WATCHDOG.format(backend=str(BACKEND), scripts=str(REPO/'scripts')))
    child = subprocess.Popen([sys.executable, '-B', str(script), 'lock_held', str(child_dir)], cwd=tmp_path)
    try:
        assert child.wait(timeout=20) == 87
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=20)
    left = json.loads((child_dir/'SUMMARY.json').read_text())
    assert (left['status'], left['incomplete']) == ('RUNNING', True) and not (child_dir/'ABORTED.json').exists()


def test_the_replay_publishes_the_same_bodies_however_slow_the_policy_is(replayed):
    """Live, a computation slower than the wait shows "computing" for a poll; the replay waits for it, so its
    files do not depend on the machine's speed. How many computations were over the live wait is counted."""
    import time as real_time

    class Slow(Policy):
        delay = 0.0

        def __call__(self, snapshot):
            real_time.sleep(self.delay)
            return super().__call__(snapshot)

    def out_of(name):
        root = replayed.job/name/'out'
        return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}

    fast = replayed('run-fast', policy_factory=lambda: Slow(), live_wait_s=.05)

    class Slower(Slow):
        delay = 0.15  # three times the live wait used here
    slow = replayed('run-slow', policy_factory=lambda: Slower(), live_wait_s=.05)
    assert fast['status'] == slow['status'] == 'COMPLETED'
    assert out_of('run-fast') == out_of('run-slow')  # byte for byte
    strip = lambda rows: [(row['sequence'], row['written'], row['game']) for row in rows]
    cycles = lambda name: [json.loads(line) for line in (replayed.job/name/'CYCLES.jsonl').read_text().splitlines()]
    assert strip(cycles('run-fast')) == strip(cycles('run-slow'))
    assert fast['ledger'] == slow['ledger'] and fast['game_file']['recommendation_by_cycle'] == slow['game_file']['recommendation_by_cycle']
    assert 'none|waiting|computing' not in str(slow['game_file']['recommendation_by_cycle'])  # never published as still computing
    n = slow['policy']['computations']
    assert n == fast['policy']['computations'] >= 2 and slow['policy']['compute_s']['n'] == n
    assert (fast['policy']['computations_over_the_live_wait'], slow['policy']['computations_over_the_live_wait']) == (0, n)
    assert slow['policy']['compute_s']['min'] >= .15 and slow['policy']['live_wait_s'] == .05
    assert cycles('run-slow')[-1]['policy_computations'] == n and cycles('run-slow')[-1]['ledger_rows'] == slow['ledger']['rows']
    import observer_app.main as app_main
    assert app_main.LIVE_WAIT_S == 1.0  # the server's own wait is put back


def test_a_computation_that_outlasts_the_cap_stops_the_run(replayed):
    import time as real_time

    class Stuck(Policy):
        def __call__(self, snapshot):
            real_time.sleep(1.0)
            return super().__call__(snapshot)
    summary = replayed(limits=LIMITS | {'settle_cap_s': .2}, policy_factory=lambda: Stuck())
    assert (summary['status'], summary['stop']['reason']) == ('STOPPED', 'policy_did_not_settle')
    assert summary['ledger']['rows'] == 0
