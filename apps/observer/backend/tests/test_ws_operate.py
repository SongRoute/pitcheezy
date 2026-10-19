"""ws_operate.py (World Series operations launcher) with a synthetic fake publisher: no MLB, no network, no launchd.

The real process table is never consulted for publisher detection here (a rehearsal supervisor may be running on
this machine): ``process_table`` and ``protected_processes`` are replaced per test."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

OPS = Path(__file__).resolve().parents[2]/'ops'
FAKE = Path(__file__).resolve().parent/'fixtures'/'fake_ws_publisher.py'
REAL_PUBLISH_CONFIG = Path(__file__).resolve().parents[2]/'publish_config.json'
KST = timezone(timedelta(hours=9))
TOKEN = 'T'*20+'secretvalue'+'x'*20  # 51 URL-safe characters, never to appear in any file or output
PK = 990001

spec = importlib.util.spec_from_file_location('ws_operate', OPS/'ws_operate.py')
ws = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ws)


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', *args],
                   check=True, capture_output=True)


@pytest.fixture
def env(tmp_path, monkeypatch):
    code = tmp_path/'code'
    (code/'apps/observer/backend').mkdir(parents=True)
    shutil.copy(REAL_PUBLISH_CONFIG, code/'apps/observer/publish_config.json')
    git(code, 'init', '-q')
    git(code, 'add', '-A')
    git(code, 'commit', '-q', '-m', 'init')
    start = time.time()+60
    kst_date = datetime.fromtimestamp(start, KST).date().isoformat()
    schedule = tmp_path/'schedule.json'
    schedule.write_text(json.dumps({'schema': 'pitcheezy-schedule-query-v1', 'rows': [
        {'game_pk': PK, 'game_type': 'W', 'away': 'A', 'home': 'H', 'abstract': 'Preview', 'detailed': 'Scheduled',
         'game_date_utc': datetime.fromtimestamp(start, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}]}))
    cfg = json.loads((OPS/'ws_ops_config.json').read_text())
    cfg |= {'ops_root': str(tmp_path/'ops'), 'code_dir': str(code), 'required_mounts': [], 'caffeinate': False,
            'childwatch': False, 'lead_min': 5, 'hard_end_min': 2, 'settle_s': 0.4, 'tick_s': 0.05, 'sample_s': 0.1,
            'status_every_s': 0.1, 'waiting_status_every_s': 0.1, 'profile_sources': None}
    cfg['restart'] = {'max_restarts': 2, 'backoff_s': [0.3, 0.6], 'health_stall_s': 0.8, 'startup_grace_s': 0.5}
    cfg['stop'] = {'graceful_s': 5, 'term_s': 2}
    cfg['guards'] = dict(cfg['guards'], start_min_free_disk_bytes=0, stop_free_disk_bytes=0)
    cfg['upload_read_s'] = 0.1
    cfg['prestart_retry_s'] = 0.2
    cfg_path = tmp_path/'ws_ops_config.json'
    cfg_path.write_text(json.dumps(cfg))
    monkeypatch.setenv(ws.FAKE_ENV, str(FAKE))
    for name in (ws.TOKEN_ENV, ws.TOKEN_FILE_ENV, ws.BASE_URL_ENV, ws.ROOT_ENV, ws.PROFILE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ws, 'process_table', lambda: [])
    monkeypatch.setattr(ws, 'protected_processes', lambda: [])
    monkeypatch.setattr(ws, 'available_bytes', lambda: 16*ws.GIB)

    class Env:
        pass
    e = Env()
    e.tmp, e.code, e.date, e.schedule, e.cfg_path = tmp_path, code, kst_date, schedule, cfg_path
    e.cfg = lambda: ws.load_config(cfg_path)
    e.run = tmp_path/'ops'/'runs'/kst_date
    return e


def control(e, **body):
    e.run.mkdir(parents=True, exist_ok=True)
    (e.run/'fake_control.json').write_text(json.dumps(body))


def supervise(e, upload=False, cfg=None):
    sup = ws.Supervisor(cfg or e.cfg(), e.date, upload=upload, schedule_file=str(e.schedule))
    return sup, sup.main()


def fake_events(e):
    path = e.run/'fake_events.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def segments(e):
    return [json.loads(x) for x in (e.run/'segments.jsonl').read_text().splitlines()]


def assert_never_two(events):
    """Each start comes after the previous publisher's end: never two publishers at once."""
    starts = [x for x in events if 'start' in x]
    ends = {x['pid']: x['end'] for x in events if 'end' in x}
    for prev, nxt in zip(starts, starts[1:]):
        assert prev['pid'] in ends and ends[prev['pid']] <= nxt['start']


# ------------------------------------------------------------------------------------------------ restart policy

def test_restart_backoff_then_limit(env):
    control(env, crash_first=99, exit_after=2, exit_code=3)
    sup, reason = supervise(env)
    assert reason == 'restart_limit:publisher_exited:3'
    events = fake_events(env)
    starts = [x for x in events if 'start' in x]
    assert len(starts) == 3  # first start + max_restarts 2
    assert_never_two(events)
    ends = [x['end'] for x in events if 'end' in x]
    for i, backoff in enumerate([0.3, 0.6]):
        assert starts[i+1]['start']-ends[i] >= backoff-0.05
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['state'] == 'failed' and status['restarts']['count'] == 2
    assert [h['backoff_s'] for h in status['restarts']['history']] == [0.3, 0.6]


def test_one_crash_is_restarted_and_date_ends_after_final(env):
    control(env, crash_first=1, exit_after=2, final_after=6)
    sup, reason = supervise(env)
    assert reason == 'final_settled'
    assert sup.restarts == 1 and str(PK) in sup.finals
    assert_never_two(fake_events(env))
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['state'] == 'stopped' and status['game_checks'][str(PK)]['final'] is True
    assert [x['code'] for x in fake_events(env) if 'end' in x] == [3, 0]  # the last one ended by SIGINT


def test_hung_publisher_is_stopped_and_restarted(env):
    control(env, hang_after=2)
    sup, reason = supervise(env)
    assert reason == 'restart_limit:health_stalled'
    stops = [s for s in segments(env) if s['phase'] == 'stop']
    assert [s['why'] for s in stops][:3] == ['health_stalled']*3
    assert all(s['method'] == 'SIGINT' and s['exit_code'] == 0 for s in stops)
    assert_never_two(fake_events(env))


# ------------------------------------------------------------------------------------------------ deploy guard

def test_restart_onto_a_different_commit_is_refused(env, monkeypatch):
    real = ws.code_commit
    calls = []

    def moving(code_dir):
        head, dirty = real(code_dir)
        calls.append(head)
        return (head if len(calls) == 1 else 'b'*40), dirty
    monkeypatch.setattr(ws, 'code_commit', moving)
    control(env, crash_first=99, exit_after=2)
    sup, reason = supervise(env)
    assert reason.startswith('deploy_guard:')
    assert len([x for x in fake_events(env) if 'start' in x]) == 1
    assert json.loads((env.tmp/'ops/status.json').read_text())['state'] == 'failed'


def test_new_start_on_other_commit_refused_while_game_not_final(env):
    control(env, final_after=None)
    (env.run/'segments.jsonl').write_text(json.dumps({'phase': 'start', 'segment': 1, 'commit': 'c'*40})+'\n')
    ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
    assert not ok and any(p.startswith('deploy guard') for p in report['problems'])
    (env.run/'finals.json').write_text(json.dumps({str(PK): 'x'}))  # between games: allowed (here: all final)
    ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
    assert not any(p.startswith('deploy guard') for p in report['problems'])


def test_dirty_code_refused(env):
    (env.code/'apps/observer/publish_config.json').write_text('{}')
    ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
    assert not ok and any('tracked changes' in p for p in report['problems'])


# ------------------------------------------------------------------------------------------------ single instance

def test_other_publisher_blocks_start(env, monkeypatch):
    monkeypatch.setattr(ws, 'process_table', lambda: [
        (99991, 1, '/Users/x/.venv/bin/python -B -m observer_app.publisher --out /elsewhere'),
        (99992, 1, "/bin/zsh -c pgrep -f 'rehearse.py supervise' ; python -m observer_app.publisher")])
    ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
    assert not ok and [o['pid'] for o in report['other_publishers']] == [99991]  # the shell text is not a publisher
    control(env)
    cfg = env.cfg()
    cfg['prestart_retry_s'] = 120  # a retry would land after first pitch: refuse at once (retries: separate tests)
    sup, reason = supervise(env, cfg=cfg)
    assert reason == 'prestart_refused' and fake_events(env) == []


def test_publisher_kind_ignores_command_text():
    assert ws.publisher_kind('/a/python3.12 -B -m observer_app.publisher --out x') == 'publisher'
    assert ws.publisher_kind('/a/python -I /j/rehearse.py supervise --game 1') == 'rehearsal_supervisor'
    assert ws.publisher_kind('/a/python -I -B /r/ws_operate.py supervise --date 2026-10-24') == 'ops_supervisor'
    assert ws.publisher_kind('/bin/zsh -c "pgrep -f rehearse.py supervise"') is None
    assert ws.publisher_kind('/a/python /r/ws_operate.py status') is None


def test_second_supervisor_refused_by_lock(env):
    control(env)
    (env.tmp/'ops').mkdir(parents=True, exist_ok=True)
    with open(env.tmp/'ops/supervisor.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ws.lock_held(env.tmp/'ops/supervisor.lock')
        ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
        assert not ok and report['supervisor_lock_held']
        sup, reason = supervise(env)
    assert reason == 'supervisor_lock_held' and fake_events(env) == []
    assert not (env.tmp/'ops/status.json').exists()  # the shared status file belongs to the lock holder


# ------------------------------------------------------------------------------------------------ stop safety

def spawn_fake(e, extra=()):
    e.run.mkdir(parents=True, exist_ok=True)
    control(e, interval=0.1)
    (e.run/'publish_config.runtime.json').write_text(json.dumps({'upload': {'enabled': False}, 'fixed_delay_s': 0}))
    argv = [sys.executable, '-B', str(FAKE), '--out', str(e.run/'out'), '--date', e.date,
            '--config', str(e.run/'publish_config.runtime.json'), *extra]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        if (e.run/'out/v1/health.json').exists():
            break
        time.sleep(0.05)
    return proc


def test_stop_signals_only_a_matching_publisher(env, capsys):
    proc = spawn_fake(env)
    (env.run/'RUN.json').write_text('{}')
    ident = ws.ps_identity(proc.pid)
    try:
        (env.run/'publisher.identity.json').write_text(json.dumps({'pid': proc.pid, 'identity': dict(ident, lstart='Mon Jan  1 00:00:00 2001')}))
        assert ws.main(['--config', str(env.cfg_path), 'stop', '--date', env.date]) == 0
        assert 'NOT signalled' in capsys.readouterr().out
        time.sleep(0.3)
        assert proc.poll() is None  # start time differs: a reused pid is never signalled
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True)
        try:  # right start time, wrong program (not our run's publisher)
            (env.run/'publisher.identity.json').write_text(json.dumps({'pid': other.pid, 'identity': ws.ps_identity(other.pid)}))
            ws.main(['--config', str(env.cfg_path), 'stop', '--date', env.date, '--publisher-only'])
            time.sleep(0.3)
            assert other.poll() is None
        finally:
            other.kill()
            other.wait()
        (env.run/'publisher.identity.json').write_text(json.dumps({'pid': proc.pid, 'identity': ident}))
        assert ws.main(['--config', str(env.cfg_path), 'stop', '--date', env.date, '--publisher-only']) == 0
        assert proc.wait(timeout=10) == 0  # SIGINT -> graceful exit
    finally:
        if proc.poll() is None:
            proc.kill()


def test_graceful_stop_via_stop_file(env):
    control(env)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault('r', supervise(env)[1]))
    thread.start()
    for _ in range(200):
        if fake_events(env):
            break
        time.sleep(0.05)
    time.sleep(0.3)
    (env.run/'STOP').write_text('now')
    thread.join(timeout=20)
    assert result['r'] == 'operator_stop'
    assert [x['code'] for x in fake_events(env) if 'end' in x] == [0]
    assert json.loads((env.tmp/'ops/status.json').read_text())['state'] == 'stopped'


def test_childwatch_stops_orphaned_publisher(env):
    proc = spawn_fake(env)
    try:
        ident = ws.ps_identity(proc.pid)
        ws.main(['childwatch', '--pid', str(proc.pid), '--lstart', ident['lstart'], '--run-dir', str(env.run),
                 '--supervisor', '1', '--supervisor-lstart', 'not-launchd', '--deadline', str(time.time()+600),
                 '--graceful-s', '5', '--term-s', '2', '--every-s', '0.1'])
        assert proc.wait(timeout=10) == 0
        rows = [json.loads(x) for x in (env.run/'childwatch.jsonl').read_text().splitlines()]
        assert any(r.get('why') == 'supervisor_gone' and r.get('signal') == 'SIGINT' for r in rows)
    finally:
        if proc.poll() is None:
            proc.kill()


# ------------------------------------------------------------------------------------------------ status contract

def test_status_file_schema(env):
    control(env, final_after=4)
    sup, reason = supervise(env)
    assert reason == 'final_settled'
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert set(ws.STATUS_KEYS) <= set(status)
    assert status['schema'] == ws.STATUS_SCHEMA and status['state'] in ws.STATES
    assert status == json.loads((env.run/'status.json').read_text())
    assert isinstance(status['written_epoch'], float) and status['kst_date'] == env.date
    assert status['health']['sequence'] >= 4 and isinstance(status['health']['age_s'], float)
    assert status['feed_ok'] is True and status['game_checks'][str(PK)]['feed_ok'] is True
    assert {'count', 'max', 'backoff_s', 'last_exit', 'next_restart_at', 'history'} <= set(status['restarts'])
    assert {'available_bytes', 'free_disk_bytes', 'publisher_rss_bytes', 'run_dir_bytes'} <= set(status['resources'])
    assert status['publisher']['running'] is False and status['supervisor']['pid'] == os.getpid()
    assert status['schema_version'] == '1.1' and status['updated_at'] == status['written_at']
    assert status['out_dir'] == str(env.run/'out') and status['state_path'] == str(env.run/'out')+'.state.json'
    game = status['games'][0]
    assert game['game_pk'] == PK and game['status'] == 'final' and game['start'].endswith('+00:00')
    assert game['window_start'] and game['window_end'] and game['window_start'] < game['window_end']
    assert status['problems'] == [] or all(not p['active'] for p in status['problems'])
    # while running, the publisher block carries identity and commit
    rows = [json.loads(x) for x in (env.run/'segments.jsonl').read_text().splitlines() if '"start"' in x]
    assert rows[0]['commit'] and len(rows[0]['commit']) == 40
    states = [json.loads(x).get('state') for x in (env.run/'events.jsonl').read_text().splitlines()]
    assert [x for x in states if x][:4] == ['waiting', 'prestart', 'running', 'stopping']


def test_status_during_run_has_publisher_identity(env):
    control(env)
    result = {}
    thread = threading.Thread(target=lambda: result.setdefault('r', supervise(env)[1]))
    thread.start()
    try:
        status = None
        for _ in range(200):
            status = json.loads((env.tmp/'ops/status.json').read_text()) if (env.tmp/'ops/status.json').exists() else None
            if status and status['publisher'] and status['health']:
                break
            time.sleep(0.05)
        pub = status['publisher']
        assert status['state'] == 'running' and pub['running'] is True and pub['pid'] and pub['lstart']
        assert len(pub['commit']) == 40 and status['games'][0]['status'] == 'scheduled'
        assert pub['fixed_delay_s'] == 0 and pub['upload_enabled'] is False and pub['fake'] is True
    finally:
        (env.run/'STOP').write_text('now')
        thread.join(timeout=20)


# ------------------------------------------------------------------------------------------------ upload gating

def test_upload_off_by_default_and_gated(env, monkeypatch):
    cfg = env.cfg()
    assert ws.upload_decision(cfg, False)[0] is False
    enabled, problems, _ = ws.upload_decision(cfg, True)
    assert not enabled and any('upload.allowed' in p for p in problems) and any(ws.TOKEN_ENV in p for p in problems)
    cfg['upload']['allowed'] = True
    monkeypatch.setenv(ws.TOKEN_ENV, 'short')
    monkeypatch.setenv(ws.BASE_URL_ENV, 'http://host/_publish')
    enabled, problems, presence = ws.upload_decision(cfg, True)
    assert not enabled and len(problems) == 2 and presence['token_env_present']
    assert 'short' not in json.dumps(problems) and 'host' not in json.dumps(problems)
    monkeypatch.setenv(ws.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(ws.BASE_URL_ENV, 'https://example.invalid/x/_publish')
    assert ws.upload_decision(cfg, True)[0] is True
    body = ws.runtime_publish_config(cfg, False)
    assert body['upload']['enabled'] is False and body['upload']['base_url'] is None


def test_upload_run_never_writes_the_token(env, monkeypatch, capsys):
    raw = json.loads(env.cfg_path.read_text())
    raw['upload']['allowed'] = True
    env.cfg_path.write_text(json.dumps(raw))
    monkeypatch.setenv(ws.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(ws.BASE_URL_ENV, 'https://example.invalid/x/_publish')
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--upload', '--dry-run',
                    '--schedule-file', str(env.schedule)]) == 0
    printed = capsys.readouterr().out
    assert TOKEN not in printed and 'example.invalid' not in printed and '<present>' in printed
    control(env, final_after=5, upload_blocked=True)
    sup, reason = supervise(env, upload=True)
    assert reason == 'final_settled'
    first = [x for x in fake_events(env) if 'start' in x][0]
    assert first['token_seen'] is True and first['upload'] is True and first['delay'] == 0
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['upload']['enabled'] is True and status['upload']['blocked'] == 1
    assert status['upload']['last_error_class'] in ('upload_failed', 'conflict_412')
    assert status['upload']['error_classes'] == {'conflict_412': 1}
    assert oct((env.run/'publish_config.runtime.json').stat().st_mode & 0o777) == '0o600'
    for path in [*env.tmp.joinpath('ops').rglob('*')]:
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes(), path


# ------------------------------------------------------------------------------------------------ dry run / start / plan

def tree(root):
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*')) if Path(root).exists() else None


def test_dry_run_writes_nothing(env, capsys):
    before = tree(env.tmp)
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--dry-run',
                    '--schedule-file', str(env.schedule)]) == 0
    assert 'nothing was written, started or signalled' in capsys.readouterr().out
    assert ws.main(['--config', str(env.cfg_path), 'plan', '--schedule-file', str(env.schedule), '--dry-run']) == 0
    assert ws.main(['--config', str(env.cfg_path), 'status', '--dry-run']) == 1  # no status yet
    assert tree(env.tmp) == before and not (env.tmp/'ops').exists()
    env.run.mkdir(parents=True)
    (env.run/'RUN.json').write_text('{}')
    before = tree(env.tmp)
    assert ws.main(['--config', str(env.cfg_path), 'stop', '--date', env.date, '--dry-run']) == 0
    assert tree(env.tmp) == before and not (env.run/'STOP').exists()


def test_start_spawns_a_detached_supervisor(env, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(ws, 'spawn_supervisor', lambda argv, run_dir: seen.setdefault('argv', argv) and 4242)
    (env.run).mkdir(parents=True)
    (env.run/'STOP').write_text('old')
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 0
    argv = seen['argv']
    assert argv[argv.index('supervise')-1].endswith('ws_operate.py') and argv[argv.index('--date')+1] == env.date
    run = json.loads((env.run/'RUN.json').read_text())
    assert run['games'][0]['game_pk'] == PK and (env.run/'RUN-001.json').exists()
    assert not (env.run/'STOP').exists()
    assert json.loads((env.run/'supervisor.identity.json').read_text())['pid'] == 4242


def test_start_refuses_without_a_ws_game(env, capsys):
    other = env.tmp/'empty_schedule.json'
    other.write_text(json.dumps({'dates': []}))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(other)]) == 2
    assert 'no W game scheduled' in capsys.readouterr().out
    assert not env.run.exists()


def test_plan_lists_ws_dates_with_timeline(env, capsys):
    assert ws.main(['--config', str(env.cfg_path), 'plan', '--schedule-file', str(env.schedule)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['dates'][0]['kst_date'] == env.date and out['dates'][0]['games'][0]['game_pk'] == PK
    assert 'publisher_start' in out['dates'][0]['timeline'] and out['fixed_delay_s'] == 0


def test_plan_malformed_schedule_prints_one_line_not_traceback(env, capsys):
    bad = env.tmp/'bad_schedule.json'
    bad.write_text(json.dumps({'totalGames': 3, 'dates': [{'games': [{'gamePk': 1, 'gameType': 'W', 'gameDate': '2026-10-24T00:08:00Z'}]}]}))
    assert ws.main(['--config', str(env.cfg_path), 'plan', '--schedule-file', str(bad)]) == 2
    cap = capsys.readouterr()
    lines = [l for l in cap.err.splitlines() if l.strip()]
    assert len(lines) == 1 and lines[0].startswith('schedule check failed (ValueError)') and 'totalGames' in lines[0]
    assert 'Traceback' not in cap.err+cap.out


def test_statsapi_schedule_rows_and_final_tail(env, tmp_path):
    rows = ws.schedule_rows({'dates': [{'games': [{'gamePk': 1, 'gameType': 'W', 'gameDate': '2026-10-23T00:08:00Z',
                                                    'status': {'abstractGameState': 'Preview', 'detailedState': 'Scheduled'},
                                                    'teams': {'away': {'team': {'name': 'A'}}, 'home': {'team': {'name': 'H'}}}}]}]})
    assert rows[0]['kst_date'] == '2026-10-23' and rows[0]['label'] == 'A @ H'
    sup = ws.Supervisor(env.cfg(), env.date)
    (sup.out/'v1/game').mkdir(parents=True)
    (sup.out/'v1/game/7.json').write_text(json.dumps({'x': 'y'*5000, 'publish': {'sequence': 3, 'final': True}}))
    (sup.out/'v1/game/8.json').write_text(json.dumps({'publish': {'sequence': 3, 'final': False}}))
    assert sup.game_final('7') is True and sup.game_final('8') is False and sup.game_final('9') is False


# ------------------------------------------------------------------------------------------------ review 001 fixes

def problem(status, key):
    rows = [p for p in status['problems'] if p['key'] == key]
    assert len(rows) <= 1
    return rows[0] if rows else None


def test_problems_for_restart_limit(env):
    control(env, crash_first=99, exit_after=2)
    supervise(env)
    status = json.loads((env.tmp/'ops/status.json').read_text())
    row = problem(status, 'restart_limit')
    assert row == dict(row, severity='urgent', active=True) and 'publisher_exited:3' in row['message']
    assert problem(status, 'publisher_restarting')['active'] is False  # it was active during the backoffs
    assert {'key', 'severity', 'message', 'active'} <= set(row)


def test_problem_for_deploy_guard(env, monkeypatch):
    real, calls = ws.code_commit, []

    def moving(code_dir):
        head, dirty = real(code_dir)
        calls.append(head)
        return (head if len(calls) == 1 else 'b'*40), dirty
    monkeypatch.setattr(ws, 'code_commit', moving)
    control(env, crash_first=99, exit_after=2)
    supervise(env)
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert problem(status, 'deploy_guard')['severity'] == 'urgent' and problem(status, 'deploy_guard')['active']


def test_exit_after_all_final_ends_normally_even_on_other_commit(env, monkeypatch):
    real, calls = ws.code_commit, []

    def moving(code_dir):
        head, dirty = real(code_dir)
        calls.append(head)
        return (head if len(calls) == 1 else 'b'*40), dirty
    monkeypatch.setattr(ws, 'code_commit', moving)
    raw = json.loads(env.cfg_path.read_text())
    raw['settle_s'] = 30
    env.cfg_path.write_text(json.dumps(raw))
    control(env, crash_first=99, final_after=2, exit_after=4)  # final, then the publisher dies during settle
    sup, reason = supervise(env)
    assert reason == 'final_settled' and sup.restarts == 0
    assert json.loads((env.tmp/'ops/status.json').read_text())['state'] == 'stopped'


def test_backoff_honors_hard_end(env, monkeypatch):
    raw = json.loads(env.cfg_path.read_text())
    raw['restart']['backoff_s'] = [600]
    env.cfg_path.write_text(json.dumps(raw))
    control(env, crash_first=99, exit_after=2)
    sup = ws.Supervisor(env.cfg(), env.date, schedule_file=str(env.schedule))
    real_prestart = sup.prestart

    def prestart():
        out = real_prestart()
        sup.timeline['hard_end_epoch'] = time.time()+1.5  # the hard end falls inside the 600 s backoff
        return out
    sup.prestart = prestart
    t0 = time.time()
    assert sup.main() == 'hard_end' and time.time()-t0 < 10


def test_prestart_retries_transient_memory_then_starts(env, monkeypatch):
    seq = iter([ws.GIB, ws.GIB])
    monkeypatch.setattr(ws, 'available_bytes', lambda: next(seq, 16*ws.GIB))
    control(env, final_after=3)
    sup, reason = supervise(env)
    assert reason == 'final_settled' and sup.prestart_attempt == 3
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert problem(status, 'prestart_retrying')['active'] is False
    assert len(list(env.run.glob('PRESTART-*.json'))) == 3


def test_prestart_gives_up_at_first_pitch_with_urgent_problem(env, monkeypatch):
    monkeypatch.setattr(ws, 'available_bytes', lambda: ws.GIB)
    start = time.time()+1.0
    env.schedule.write_text(json.dumps({'schema': 'pitcheezy-schedule-query-v1', 'rows': [
        {'game_pk': PK, 'game_type': 'W', 'abstract': 'Preview', 'detailed': 'Scheduled',
         'game_date_utc': datetime.fromtimestamp(start, timezone.utc).isoformat()}]}))
    env.date = datetime.fromtimestamp(start, KST).date().isoformat()
    env.run = env.tmp/'ops'/'runs'/env.date
    control(env)
    sup, reason = supervise(env)
    assert reason == 'prestart_refused' and sup.prestart_attempt >= 2 and fake_events(env) == []
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['state'] == 'refused'
    row = problem(status, 'prestart_refused')
    assert row['severity'] == 'urgent' and row['active'] and 'available memory' in row['message']


def test_permanent_prestart_problem_is_not_retried(env):
    (env.code/'apps/observer/publish_config.json').write_text('{}')
    control(env)
    sup, reason = supervise(env)
    assert reason == 'prestart_refused' and sup.prestart_attempt == 1


def test_schedule_failure_falls_back_to_run_json(env):
    control(env, final_after=3)
    rows = ws.schedule_rows(json.loads(env.schedule.read_text()))
    (env.run/'RUN.json').write_text(json.dumps({'games': rows}))
    env.schedule.unlink()  # the schedule source fails at prestart
    sup, reason = supervise(env)
    assert reason == 'final_settled'
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert problem(status, 'schedule_fallback')['active'] and problem(status, 'schedule_fallback')['severity'] == 'warn'


def test_upload_blocked_is_an_urgent_problem(env, monkeypatch):
    raw = json.loads(env.cfg_path.read_text())
    raw['upload']['allowed'] = True
    env.cfg_path.write_text(json.dumps(raw))
    monkeypatch.setenv(ws.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(ws.BASE_URL_ENV, 'https://example.invalid/x/_publish')
    control(env, final_after=8, upload_blocked=True)
    supervise(env, upload=True)
    rows = [json.loads(x) for x in (env.run/'events.jsonl').read_text().splitlines()]
    assert any(r['event'] == 'problem' and r['key'] == 'upload_blocked' for r in rows)
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert problem(status, 'upload_blocked')['severity'] == 'urgent'


def test_token_file_must_be_private(env, monkeypatch, tmp_path_factory):
    cfg = env.cfg()
    cfg['upload']['allowed'] = True
    token = tmp_path_factory.mktemp('secret')/'token'
    token.write_text(TOKEN)
    os.chmod(token, 0o644)
    monkeypatch.setenv(ws.TOKEN_FILE_ENV, str(token))
    monkeypatch.setenv(ws.BASE_URL_ENV, 'https://example.invalid/x/_publish')
    enabled, problems, _ = ws.upload_decision(cfg, True)
    assert not enabled and any('chmod 600' in p for p in problems) and str(token) not in json.dumps(problems)
    os.chmod(token, 0o600)
    assert ws.upload_decision(cfg, True)[0] is True


def test_statsapi_query_has_all_game_types(env, monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"dates": []}'
    monkeypatch.setattr(ws.urllib.request, 'urlopen', lambda req, timeout: seen.setdefault('url', req.full_url) and Resp())
    ws.fetch_schedule(env.cfg(), '2026-10-24', '2026-10-24')
    assert 'gameType' not in seen['url']  # never sent anywhere: urlopen is replaced


# ------------------------------------------------------------------------------------------------ review 002 D1 / D3

def fast_start_retries(env, **more):
    raw = json.loads(env.cfg_path.read_text())
    raw |= {'start_retry_s': 0.1, 'start_retry_window_s': 1.0, **more}
    env.cfg_path.write_text(json.dumps(raw))


def test_start_retries_a_failed_schedule_request_then_starts(env, monkeypatch, capsys):
    fast_start_retries(env, start_retry_window_s=10)
    good = env.schedule.read_text()
    env.schedule.unlink()
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 2:
            env.schedule.write_text(good)  # the schedule source comes back on the third attempt
    monkeypatch.setattr(ws, '_sleep', sleep)
    monkeypatch.setattr(ws, 'spawn_supervisor', lambda argv, run_dir: 4242)
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 0
    assert calls == [0.1, 0.1] and (env.run/'RUN.json').exists()
    assert 'transient problems' in capsys.readouterr().out
    assert not (env.tmp/'ops/status.json').exists()  # nothing refused: the supervisor will write it


def test_start_refusal_after_retries_is_written_for_the_watcher(env, monkeypatch, capsys):
    fast_start_retries(env)
    env.schedule.unlink()
    monkeypatch.setattr(ws, '_sleep', lambda s: time.sleep(s))
    t0 = time.time()
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    assert 0.8 <= time.time()-t0 < 5  # retried for about the window, then gave up
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert set(ws.STATUS_KEYS) <= set(status) and status['schema_version'] == '1.1'
    assert status['state'] == 'refused' and status['kst_date'] == env.date and status['updated_at']
    assert status['publisher'] == dict(status['publisher'], running=False)
    row = [p for p in status['problems'] if p['key'] == 'start_refused'][0]
    assert row['severity'] == 'urgent' and row['active'] and 'schedule check failed' in row['message']
    assert 'refused written' in capsys.readouterr().out


def test_start_permanent_refusal_is_immediate_and_written(env, monkeypatch):
    fast_start_retries(env, start_retry_window_s=600)
    monkeypatch.setattr(ws, '_sleep', lambda s: pytest.fail('a permanent problem must not be retried'))
    (env.code/'apps/observer/publish_config.json').write_text('{}')
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['state'] == 'refused' and status['games'][0]['game_pk'] == PK and status['games'][0]['window_start']
    assert 'tracked changes' in status['state_reason']


def test_start_refusal_does_not_touch_status_owned_by_a_supervisor(env, monkeypatch, capsys):
    fast_start_retries(env)
    (env.tmp/'ops').mkdir(parents=True)
    (env.tmp/'ops/status.json').write_text(json.dumps({'state': 'running', 'kst_date': env.date}))
    with open(env.tmp/'ops/supervisor.lock', 'a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date,
                        '--schedule-file', str(env.schedule)]) == 2
    assert json.loads((env.tmp/'ops/status.json').read_text())['state'] == 'running'
    assert 'not written' in capsys.readouterr().out


def test_dry_run_refusal_neither_retries_nor_writes(env, monkeypatch):
    fast_start_retries(env, start_retry_window_s=600)
    monkeypatch.setattr(ws, '_sleep', lambda s: pytest.fail('dry-run must not retry'))
    env.schedule.unlink()
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--dry-run',
                    '--schedule-file', str(env.schedule)]) == 2
    assert not (env.tmp/'ops').exists()


def test_new_supervisor_clears_previous_refused_and_failed_problems(env, monkeypatch):
    (env.tmp/'ops').mkdir(parents=True)
    yesterday = (datetime.fromisoformat(env.date)-timedelta(days=1)).date().isoformat()
    (env.tmp/'ops/status.json').write_text(json.dumps({
        'state': 'failed', 'kst_date': yesterday, 'problems': [
            {'key': 'restart_limit', 'severity': 'urgent', 'message': 'old', 'active': True},
            {'key': 'start_refused', 'severity': 'urgent', 'message': 'older', 'active': True},
            {'key': 'memory_low', 'severity': 'warn', 'message': 'x', 'active': False}]}))
    control(env, final_after=3)
    sup, reason = supervise(env)
    assert reason == 'final_settled'
    status = json.loads((env.tmp/'ops/status.json').read_text())
    rows = {p['key']: p for p in status['problems']}
    assert rows['restart_limit']['active'] is False and rows['start_refused']['active'] is False
    assert 'cleared_by' in rows['restart_limit'] and 'memory_low' not in rows
    assert status['state'] == 'stopped' and status['kst_date'] == env.date
    events = [json.loads(x) for x in (env.run/'events.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'problems_carried_inactive' for e in events)


# ------------------------------------------------------------------------------------------------ review 003 B3

def test_off_day_start_writes_nothing_and_keeps_yesterday(env, capsys):
    (env.tmp/'ops').mkdir(parents=True)
    old = json.dumps({'state': 'stopped', 'state_reason': 'final_settled', 'kst_date': 'yesterday'})
    (env.tmp/'ops/status.json').write_text(old)
    empty = env.tmp/'off_day.json'
    empty.write_text(json.dumps({'dates': []}))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(empty)]) == 2
    assert (env.tmp/'ops/status.json').read_text() == old
    assert 'nothing to operate on' in capsys.readouterr().out


def test_all_final_start_writes_nothing(env, capsys):
    env.run.mkdir(parents=True)
    (env.run/'finals.json').write_text(json.dumps({str(PK): 'x'}))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    assert not (env.tmp/'ops/status.json').exists()
    assert "['all_final']" in capsys.readouterr().out


def test_benign_plus_real_failure_is_still_urgent(env):
    empty = env.tmp/'off_day.json'
    empty.write_text(json.dumps({'dates': []}))
    (env.code/'apps/observer/publish_config.json').write_text('{}')  # dirty code is a real fault
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(empty)]) == 2
    status = json.loads((env.tmp/'ops/status.json').read_text())
    assert status['state'] == 'refused' and [p['key'] for p in status['problems'] if p['active']] == ['start_refused']


def test_other_publisher_message_is_short(env, monkeypatch):
    monkeypatch.setattr(ws, 'process_table', lambda: [
        (pid, 1, f'/a/python -m observer_app.publisher --out /x{pid}') for pid in (91001, 91002, 91003)])
    ok, report, _ = ws.preflight(env.cfg(), env.date, upload_requested=False, schedule_file=str(env.schedule))
    message = [p for k, p in zip(report['problem_keys'], report['problems']) if k == 'other_publishers'][0]
    assert '(3: first pid 91001 publisher)' in message and '91002' not in message


# ------------------------------------------------------------------------------------------------ review 004 B4 / F1

MALFORMED = [{'messageNumber': 10, 'message': 'Object not found', 'timestamp': '2026-10-24T00:00:00Z'},
             [], {'rows': None}, {'dates': None}, {'dates': [{'date': '2026-10-24'}]},
             {'dates': [], 'totalGames': 2}]


@pytest.mark.parametrize('body', MALFORMED, ids=['error_body', 'list', 'rows_null', 'dates_null', 'day_no_games',
                                                 'total_mismatch'])
def test_malformed_schedule_is_retried_then_refused_urgently(env, monkeypatch, body):
    fast_start_retries(env, start_retry_window_s=0.35)
    env.schedule.write_text(json.dumps(body))
    sleeps = []
    monkeypatch.setattr(ws, '_sleep', lambda s: sleeps.append(s) or time.sleep(s))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    assert len(sleeps) >= 2  # retried as a failed schedule check, not taken as "no game"
    status = json.loads((env.tmp/'ops/status.json').read_text())
    row = [p for p in status['problems'] if p['key'] == 'start_refused'][0]
    assert status['state'] == 'refused' and row['active'] and 'schedule check failed' in row['message']


def test_empty_but_well_formed_schedule_is_a_quiet_off_day(env, monkeypatch):
    monkeypatch.setattr(ws, '_sleep', lambda s: pytest.fail('an off day must not be retried'))
    env.schedule.write_text(json.dumps({'dates': [], 'totalGames': 0}))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    assert not (env.tmp/'ops/status.json').exists()


def test_supervisor_prestart_treats_malformed_schedule_as_failure(env):
    control(env, final_after=3)
    env.schedule.write_text(json.dumps({'rows': None}))
    cfg = env.cfg()
    cfg['prestart_retry_s'] = 600  # no RUN.json and no retry room: refused at once, as a schedule failure
    sup, reason = supervise(env, cfg=cfg)
    assert reason == 'prestart_refused'
    report = json.loads(sorted(env.run.glob('PRESTART-*.json'))[0].read_text())
    assert report['problem_keys'] == ['schedule_failed'] and fake_events(env) == []
    rows = ws.schedule_rows(json.loads(json.dumps({'schema': 'pitcheezy-schedule-query-v1', 'rows': [
        {'game_pk': PK, 'game_type': 'W', 'game_date_utc': '2026-10-24T00:00:00Z'}]})))
    (env.run/'RUN.json').write_text(json.dumps({'games': rows}))
    (env.run/'STOP').unlink(missing_ok=True)
    sup = ws.Supervisor(cfg, env.date, schedule_file=str(env.schedule))
    sup.run_games = rows  # what main() reads from RUN.json
    problems, keys = sup.prestart()
    assert 'schedule_failed' not in keys and sup.problems['schedule_fallback']['active']
    assert not any('schedule' in p for p in problems)  # with RUN.json: fallback, not a failure


def test_called_off_game_is_named_in_the_no_game_reason(env, capsys):
    start = datetime.fromisoformat(env.date+'T10:00:00+09:00').astimezone(timezone.utc)
    env.schedule.write_text(json.dumps({'dates': [{'games': [
        {'gamePk': PK, 'gameType': 'W', 'gameDate': start.strftime('%Y-%m-%dT%H:%M:%SZ'),
         'status': {'abstractGameState': 'Preview', 'detailedState': 'Postponed'}}]}], 'totalGames': 1}))
    assert ws.main(['--config', str(env.cfg_path), 'start', '--date', env.date, '--schedule-file', str(env.schedule)]) == 2
    out = capsys.readouterr().out
    assert f'called off: {PK} Postponed' in out and not (env.tmp/'ops/status.json').exists()


def test_preflight_mount_check_uses_the_shared_helper(env, monkeypatch):
    """OPS-03: preflight and the running supervisor judge required_mounts with the same helper."""
    monkeypatch.setattr(ws, 'is_mounted', lambda path: False)
    cfg = dict(env.cfg(), required_mounts=['/nonexistent/ops03-fake-T7'])
    assert ws.missing_mounts(cfg) == ['/nonexistent/ops03-fake-T7']
    ok, report, _ = ws.preflight(cfg, env.date, upload_requested=False, schedule_file=str(env.schedule))
    assert not ok and 'mount' in report['problem_keys'] and 'mount' in ws.TRANSIENT
