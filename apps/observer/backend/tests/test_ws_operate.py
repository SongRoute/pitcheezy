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
    sup, reason = supervise(env)
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
    assert status['publisher'] is None and status['supervisor']['pid'] == os.getpid()
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
        assert status['state'] == 'running' and pub['pid'] and pub['lstart'] and len(pub['commit']) == 40
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
