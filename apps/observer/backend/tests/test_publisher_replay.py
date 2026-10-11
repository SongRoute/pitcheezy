"""Offline replay of a capture through the publisher (``capture_replay``, ``publish_audit``,
``scripts/publisher_replay.py``) on SYNTHETIC captures only: no real capture, network, T7, model or port."""
from contextlib import closing
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
          'max_seconds': 600, 'settle_cap_s': 10, 'max_body_bytes': 1 << 26, 'policy_load_cap_s': 30}


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
    assert rows[1][1]()['metaData']['timeStamp'] == '20261005_100100' and info['final_fetched'] is None
    rows[4][1]()
    assert info['final_fetched'] == BASE+128 and info['loads'] == 2
    with pytest.raises(sqlite3.OperationalError):
        info['db'].execute('DELETE FROM snapshots')  # the reader's connection cannot write
    info['db'].close()
    with closing(sqlite3.connect(path, isolation_level=None)) as db:
        db.execute("UPDATE snapshots SET sha='0' WHERE seq=2")
    rows, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
    with pytest.raises(ValueError):
        rows[1][1]()
    info['db'].close()
    with pytest.raises(publisher_replay.Refused):
        publisher_replay.capture_rows(path, 1, 1 << 26)
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

    def go(name='run-001', **over):
        rows, info = publisher_replay.capture_rows(path, LIVE_GAME, 1 << 26)
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
    assert summary['final']['feed_final_first_served_version_fetched'] == BASE+128
    assert summary['final']['publish_final']['simulated'] == BASE+130 and summary['final']['simulated_seconds_from_feed_final_to_publish_final'] == 2.0
    final_writes = [row['cycle'] for row in rows if f'v1/game/{LIVE_GAME}.json' in row['written'] and row['game']['final']]
    assert final_writes == [14]  # written once: the tail and the settled recheck left its bytes alone
    assert summary['capture']['versions_never_served'] == 0 and summary['capture']['bodies_loaded'] > 0
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
    assert (summary['status'], summary['stop']['reason'], summary['cycles']['n']) == ('STOPPED', 'rss_cap', 0)
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
    config = job/'replay_config.json'
    config.write_text(json.dumps(LIMITS))
    commit = publisher_replay.head_commit()
    assert commit and len(commit) == 40
    base = ['--job-dir', str(job), '--run-name', 'run-001', '--game', str(LIVE_GAME), '--capture-db', str(path),
            '--capture-sha256', sha, '--commit', commit, '--config', str(config), '--policy', 'off', '--allow-dirty']
    swap = lambda flag, value: [value if base[i-1] == flag else item for i, item in enumerate(base)]
    outside = capture_db(tmp_path/'elsewhere')
    cases = {'registered sha256': swap('--capture-sha256', '0'*64), 'registered commit': swap('--commit', '0'*40),
             'COPY inside the job folder': swap('--capture-db', str(outside)), 'plain name': swap('--run-name', '../x')}
    for words, argv in cases.items():
        assert publisher_replay.main(argv) == 2
        assert words in capsys.readouterr().err
    (job/'run-001').mkdir()
    assert publisher_replay.main(base) == 2 and 'already exists' in capsys.readouterr().err
    (job/'run-001').rmdir()
    config.write_text(json.dumps(LIMITS | {'fixed_delay_s': 5}))
    assert publisher_replay.main(base) == 2 and 'fixed delay 0' in capsys.readouterr().err
    config.write_text(json.dumps(LIMITS | {'min_free_bytes': 1 << 60}))
    assert publisher_replay.main(base) == 2 and 'free memory' in capsys.readouterr().err
    assert sorted(p.name for p in job.iterdir()) == ['input', 'replay_config.json']  # nothing was written


def test_the_whole_command_runs_in_its_own_process_under_the_audit(tmp_path):
    """End to end as registered (policy off, synthetic capture): the audit hook is live for every import and
    cycle, so this also shows the publisher itself writes nowhere else, starts nothing and opens no socket."""
    job = tmp_path/'job'
    path = capture_db(job/'input')
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (job/'replay_config.json').write_text(json.dumps(LIMITS))
    argv = [sys.executable, '-B', str(REPO/'scripts/publisher_replay.py'), '--job-dir', str(job), '--run-name', 'run-001',
            '--game', str(LIVE_GAME), '--capture-db', str(path), '--capture-sha256', sha, '--commit', publisher_replay.head_commit(),
            '--config', str(job/'replay_config.json'), '--policy', 'off', '--allow-dirty']
    env = {key: value for key, value in os.environ.items() if not key.startswith('PITCHEEZY_')}
    done = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env, cwd=tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads((job/'run-001/SUMMARY.json').read_text())
    assert summary['status'] == 'COMPLETED' and summary['fs_audit'] == {'violations': [], 'violation_count': 0, 'files_read_under_watch': {}}
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


def check(audit, path, body):
    data = json.dumps(body, ensure_ascii=False).encode()
    audit.check(path, data, hashlib.sha256(data).hexdigest())
    return [v['invariant'] for v in audit.violations]


GAME_PATH = 'v1/game/900200.json'


def ledger_of(body):
    return {rec['ledger']['decision_id'] for _, rec in publish_audit.recommendations(body)}


def test_the_committed_examples_break_no_invariant():
    audit = audit_of(ledger_ids=lambda: ledger_of(example('game-live.json')), forbidden=['/Users/song'])
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
    assert check(audit_of(ledger_ids=lambda: ledger_of(body)), GAME_PATH, body) == ['unconfirmed_pitch_blank']


def test_a_recommendation_without_a_ledger_row_fails():
    body = example('game-live.json')
    assert publish_audit.recommendations(body)[0][0] == '/next/rec'
    assert check(audit_of(ledger_ids=lambda: ledger_of(body)), GAME_PATH, body) == []
    assert check(audit_of(ledger_ids=lambda: set()), GAME_PATH, body) == ['recommendation_in_ledger']
    unrecorded = example('game-live.json')
    unrecorded['next']['rec']['ledger'] = None
    assert check(audit_of(ledger_ids=lambda: ledger_of(body)), GAME_PATH, unrecorded) == ['recommendation_in_ledger']
    hidden = example('game-live.json')
    hidden['display_state']['recommendation_shown'] = False
    assert check(audit_of(ledger_ids=lambda: ledger_of(body)), GAME_PATH, hidden) == ['recommendation_in_ledger']
    past = example('game-live.json')
    past['pitches'][0]['rec'] = dict(past['next']['rec'], ledger={'status': 'recorded', 'decision_id': 'not-in-the-ledger'})
    assert check(audit_of(ledger_ids=lambda: ledger_of(body)), GAME_PATH, past) == ['recommendation_in_ledger']


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
    audit = audit_of(ledger_ids=ledger)
    assert check(audit, GAME_PATH, body) == []
    moved = json.loads(json.dumps(body))
    moved['publish']['sequence'] += 5
    moved['feed']['as_of'] = '2026-10-05T11:00:00Z'  # only the clock moved
    moved['display_state']['feed']['last_ok_at'] = '2026-10-05T11:00:00Z'
    assert check(audit, GAME_PATH, moved) == ['final_stable']
    audit = audit_of(ledger_ids=ledger)
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
    audit.gone('v1/dates.json')
    lower = example('dates.json')
    lower['publish']['sequence'] = 1
    assert check(audit, 'v1/dates.json', lower) == []  # a withdrawn file starts over when it returns


def test_schema_problems_name_the_place_and_the_type_never_the_value():
    schema = json.loads((CONTRACT/'schema/game.schema.json').read_text())
    body = example('game-live.json')
    assert publish_schema.problems(body, schema) == []
    body['pitches'][0]['actual']['speed_mph'] = 'secret-value'
    body['pitches'][1].pop('actual')
    body['publish']['contract'] = 'v9'
    body['game']['new/key'] = 1
    found = publish_schema.problems(body, schema)
    assert {'pointer': '/pitches/0/actual/speed_mph', 'keyword': 'type', 'seen': 'string'} in found
    assert {'pointer': '/pitches/1/actual', 'keyword': 'required', 'seen': 'missing'} in found
    assert {'pointer': '/publish/contract', 'keyword': 'const', 'seen': 'string'} in found and len(found) == 3
    assert 'secret-value' not in json.dumps(found) and 'v9' not in json.dumps(found)
    assert publisher_replay.general('/pitches/12/actual/x') == '/pitches/*/actual/x'
    assert publisher_replay.general('/pregame/players/660271/era') == '/pregame/players/*/era'


# ---------- the filesystem / process / network audit ----------
def test_writes_outside_the_allowed_folder_are_refused_and_reads_under_a_watched_one_are_listed(tmp_path):
    inside, outside, watched = tmp_path/'run', tmp_path/'other', tmp_path/'volume'
    for folder in (inside, outside, watched):
        folder.mkdir()
    audit = publish_audit.FsAudit([inside], watch=[watched])
    audit.event('open', (str(inside/'a.json'), 'w', os.O_WRONLY | os.O_CREAT))
    audit.event('open', (str(outside/'a.json'), 'r', os.O_RDONLY))
    audit.event('open', ('/dev/null', 'w', os.O_WRONLY))
    audit.event('open', (3, 'w', os.O_WRONLY))  # an already open descriptor
    audit.event('os.rename', (str(inside/'a'), str(inside/'b')))
    audit.event('sqlite3.connect', (str(inside/'store.sqlite3'),))
    audit.event('sqlite3.connect', (f'file:{outside}/c.sqlite3?mode=ro&immutable=1',))
    audit.event('sqlite3.connect', (':memory:',))
    audit.event('open', (str(watched/'model.pt'), 'rb', os.O_RDONLY))
    audit.event('open', (str(watched/'model.pt'), 'rb', os.O_RDONLY))
    audit.event('some.other.event', ())
    assert audit.violations == [] and audit.read == {os.path.realpath(watched/'model.pt'): 2}
    refused = [('open', (str(outside/'a.json'), 'w', os.O_WRONLY | os.O_CREAT)), ('open', (str(outside/'a.json'), 'a', None)),
               ('open', (str(watched/'x'), 'r+', os.O_RDWR)), ('os.remove', (str(outside/'a.json'),)),
               ('os.rename', (str(inside/'a'), str(outside/'a'))), ('os.rename', (str(outside/'a'), str(inside/'a'))),
               ('os.mkdir', (str(outside/'new'),)), ('shutil.rmtree', (str(watched),)),
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
    quiet.event('os.remove', ('/etc/passwd',))
    assert len(quiet.violations) == 1


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
