"""End to end: ws_operate.py supervisor (fake publisher) -> real status.json v1.1 -> ws_alert.py --once (fake notifier).

No MLB, no network, no launchd, no real notification. The supervisor runs in a thread against a temp ops root and a
synthetic schedule file; the publisher is apps/observer/backend/tests/fixtures/fake_ws_publisher.py (the launcher's
test hook), and the process table is injected so a rehearsal running on this machine is never looked at. The watcher
is given only the ops status path: it must find the day's out_dir/state_path in status.json itself.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[1]
FAKE = OPS.parent/'backend'/'tests'/'fixtures'/'fake_ws_publisher.py'
REAL_PUBLISH_CONFIG = OPS.parent/'publish_config.json'
KST = timezone(timedelta(hours=9))
PK = 990001
TOKEN = 'T'*20+'secretvalue'+'x'*20
GIB = 1024**3


def _load(name):
    spec = importlib.util.spec_from_file_location(name, OPS/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ws = _load('ws_operate')
wa = _load('ws_alert')


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.email=t@example.invalid', '-c', 'user.name=t', *args],
                   check=True, capture_output=True)


class Rig:
    """Temp ops root + supervisor thread + watcher pointed only at <ops_root>/status.json."""

    def __init__(self, tmp, monkeypatch):
        self.tmp, self.mp = tmp, monkeypatch
        code = tmp/'code'
        (code/'apps/observer/backend').mkdir(parents=True)
        shutil.copy(REAL_PUBLISH_CONFIG, code/'apps/observer/publish_config.json')
        git(code, 'init', '-q')
        git(code, 'add', '-A')
        git(code, 'commit', '-q', '-m', 'init')
        start = time.time()+90  # first pitch 90 s away: game status 'scheduled', window already open (lead 5 min)
        self.date = datetime.fromtimestamp(start, KST).date().isoformat()
        self.schedule = tmp/'schedule.json'
        self.schedule.write_text(json.dumps({'schema': 'pitcheezy-schedule-query-v1', 'rows': [
            {'game_pk': PK, 'game_type': 'W', 'away': 'A', 'home': 'H', 'abstract': 'Preview', 'detailed': 'Scheduled',
             'game_date_utc': datetime.fromtimestamp(start, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}]}))
        cfg = json.loads((OPS/'ws_ops_config.json').read_text())
        cfg |= {'ops_root': str(tmp/'ops'), 'code_dir': str(code), 'required_mounts': [], 'caffeinate': False,
                'childwatch': False, 'lead_min': 5, 'hard_end_min': 3, 'settle_s': 0.4, 'tick_s': 0.05,
                'sample_s': 0.1, 'status_every_s': 0.1, 'waiting_status_every_s': 0.1, 'profile_sources': None,
                'upload_read_s': 0.1, 'prestart_retry_s': 0.2}
        cfg['restart'] = {'max_restarts': 1, 'backoff_s': [2.0], 'health_stall_s': 30, 'startup_grace_s': 10}
        cfg['stop'] = {'graceful_s': 5, 'term_s': 2}
        cfg['guards'] = dict(cfg['guards'], start_min_free_disk_bytes=0, stop_free_disk_bytes=0)
        self.cfg_path = tmp/'ws_ops_config.json'
        self.cfg_path.write_text(json.dumps(cfg))
        monkeypatch.setenv(ws.FAKE_ENV, str(FAKE))
        for name in (ws.TOKEN_ENV, ws.TOKEN_FILE_ENV, ws.BASE_URL_ENV, ws.ROOT_ENV, ws.PROFILE_ENV):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(ws, 'process_table', lambda: [])
        monkeypatch.setattr(ws, 'protected_processes', lambda: [])
        monkeypatch.setattr(ws, 'available_bytes', lambda: 16*GIB)
        self.run_dir = tmp/'ops'/'runs'/self.date
        self.status_path = tmp/'ops'/'status.json'
        # watcher: only the ops status path; profile log says ok for today so the 21:20 KST rule stays quiet
        self.profile_log = tmp/'RESULTS.log'
        self.fake = wa.FakeNotifier()
        self.wcfg = wa.load_config(None, ops_status=str(self.status_path), alert_dir=str(tmp/'alert'), env={})
        self.wcfg['paths'].update(profile_log=str(self.profile_log), disk_paths=[str(tmp)])
        assert self.wcfg['paths']['out_dir'] is None and self.wcfg['cli_out'] is False
        self.thread, self.result = None, {}

    def control(self, **body):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir/'fake_control.json').write_text(json.dumps(body))

    def config(self, **changes):
        raw = json.loads(self.cfg_path.read_text())
        for key, value in changes.items():
            if isinstance(value, dict):
                raw[key] = dict(raw.get(key) or {}, **value)
            else:
                raw[key] = value
        self.cfg_path.write_text(json.dumps(raw))

    def start(self, upload=False):
        sup = ws.Supervisor(ws.load_config(self.cfg_path), self.date, upload=upload, schedule_file=str(self.schedule))

        def body():
            self.result['reason'] = sup.main()
        self.thread = threading.Thread(target=body, daemon=True)
        self.thread.start()
        return sup

    def status(self):
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            return None

    def until(self, check, timeout=30):
        end = time.time()+timeout
        while time.time() < end:
            status = self.status()
            if status is not None and check(status):
                return status
            time.sleep(0.05)
        raise AssertionError(f'timed out; last status state={(self.status() or {}).get("state")}')

    def stop(self):
        if self.thread and self.thread.is_alive():
            (self.run_dir/'STOP').write_text('test')
            self.thread.join(30)

    def join(self, timeout=30):
        self.thread.join(timeout)
        assert not self.thread.is_alive()
        return self.result['reason']

    def watch(self):
        """ws_alert --once in process (FakeNotifier; memory/disk injected so the real machine never decides)."""
        now = time.time()
        day = datetime.fromtimestamp(now, KST).date()
        self.profile_log.write_text(f'2026 ok target={day} cutoff={day - timedelta(days=1)}\n')
        collect = lambda cfg, t: wa.collect(cfg, t, memory=lambda: 8*GIB,  # noqa: E731
                                            disk=lambda paths: {p: 100*GIB for p in paths})
        self.events = wa.run_once(self.wcfg, now=now, senders=[self.fake], collect_fn=collect)
        self.obs = collect(self.wcfg, now)
        return self.events

    def fired(self, kind='fire'):
        return {e['key']: e['severity'] for e in self.events if e['kind'] == kind}

    def notices(self):
        return [e['key'] for e in self.events if e['kind'] == 'notice']


@pytest.fixture
def rig(tmp_path, monkeypatch):
    r = Rig(tmp_path, monkeypatch)
    yield r
    r.stop()


def running_with_health(status):
    return (status['state'] == 'running' and status['publisher'].get('running') is True
            and status.get('health') is not None and status['health'].get('sequence', 0) >= 2)


def test_contract_problem_keys_and_watcher_dedup_agree():
    """Every key the watcher skips as 'judged by itself' is a launcher problem key; urgent maps to critical."""
    assert set(wa.OPS_ALERTS_COVERED) <= set(ws.PROBLEM_SEVERITY)
    assert set(wa.OPS_ALERTS_COVERED) <= set(ws.ALERT_KEYS)
    assert set(ws.PROBLEM_SEVERITY.values()) == {'urgent', 'warn'}
    # protected_process_changed is the one launcher alert the watcher cannot see itself: it must pass through
    assert 'protected_process_changed' not in wa.OPS_ALERTS_COVERED
    # heartbeat (launcher) fits inside staleness (watcher) with room for several missed writes
    th, cfg = wa.BUILTIN['thresholds'], json.loads((OPS/'ws_ops_config.json').read_text())
    assert th['ops_stale_running_s'] >= 3*cfg['status_every_s']
    assert th['ops_stale_other_s'] >= 3*cfg['waiting_status_every_s']
    shipped = json.loads((OPS/'ws_alert_config.json').read_text())['thresholds']
    assert (shipped['ops_stale_running_s'], shipped['ops_stale_other_s']) == (90, 300)


def test_normal_running_is_quiet_with_one_watch_start_notice(rig):
    rig.control(interval=0.1)
    rig.start()
    status = rig.until(running_with_health)
    assert status['schema_version'] == '1.1' and status['games'][0]['status'] == 'scheduled'
    rig.watch()
    # the watcher followed status.json to the day's out folder and state file
    assert rig.obs['out_dir'] == status['out_dir'] == str(rig.run_dir/'out')
    assert rig.obs['state_path'] == status['state_path'] and rig.obs['state'] is not None
    assert rig.obs['health']['schema'] == 'pitcheezy-publish-health-v1'
    assert rig.fired() == {} and rig.notices() == [f'watch_start:{PK}']
    assert len(rig.fake.sent) == 1 and '감시 시작' in rig.fake.sent[0]['body']
    time.sleep(0.5)
    rig.watch()
    assert [e for e in rig.events if e['kind'] != 'closed'] == [] and len(rig.fake.sent) == 1
    rig.stop()
    assert rig.result['reason'] == 'operator_stop' and rig.status()['state'] == 'stopped'


def test_normal_running_cli_once_with_fake_notifier(rig):
    """The real CLI (``--once --notifiers fake``) against the same status.json; machine families are ignored here
    because the CLI reads this Mac's real memory and disk."""
    rig.control(interval=0.1)
    rig.start()
    rig.until(running_with_health)
    day = datetime.now(KST).date()
    rig.profile_log.write_text(f'ok target={day} cutoff={day - timedelta(days=1)}\n')
    cfg = rig.tmp/'alert_config.json'
    cfg.write_text(json.dumps({'paths': {'profile_log': str(rig.profile_log), 'disk_paths': [str(rig.tmp)]}}))
    env = {k: v for k, v in os.environ.items() if not k.startswith('PITCHEEZY_')}
    done = subprocess.run([sys.executable, str(OPS/'ws_alert.py'), '--once', '--notifiers', 'fake', '--config', str(cfg),
                           '--ops-status', str(rig.status_path), '--alert-dir', str(rig.tmp/'cli-alert')],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    lines = [json.loads(x) for x in (rig.tmp/'cli-alert'/'alerts.jsonl').read_text().splitlines()]
    fired = {x['key'] for x in lines if x['kind'] == 'fire'} - {'memory_low', 'disk_low'}
    assert fired == set() and [x['key'] for x in lines if x['kind'] == 'notice'] == [f'watch_start:{PK}']
    sent = [json.loads(x) for x in (rig.tmp/'cli-alert'/'fake_notify.jsonl').read_text().splitlines()]
    assert len(sent) == 1 and '감시 시작' in sent[0]['body']


def test_crash_backoff_then_restart_limit(rig):
    rig.control(crash_first=99, exit_after=4, exit_code=3, interval=0.1)
    rig.start()
    status = rig.until(lambda s: s['state'] == 'backoff')
    assert status['publisher']['running'] is False
    rig.watch()
    fired = rig.fired()
    assert fired['ops_state:backoff'] == 'critical' and fired['publisher_down'] == 'critical'
    assert fired.get('ops_problem:publisher_restarting') == 'warn'
    assert '긴급' in rig.fake.sent[-1]['title'] and rig.fake.sent[-1]['critical']
    assert 'publisher_exited:3' in rig.fake.sent[-1]['body']
    # one restart allowed, the second crash ends the day: failed + restart_limit (urgent)
    assert rig.join() == 'restart_limit:publisher_exited:3'
    status = rig.status()
    assert status['state'] == 'failed'
    assert {p['key']: p['active'] for p in status['problems']}['restart_limit'] is True
    rig.watch()
    fired = rig.fired()
    assert fired == {'ops_state:failed': 'critical', 'ops_problem:restart_limit': 'critical'}
    assert set(rig.fired('recover')) == {'ops_state:backoff', 'ops_problem:publisher_restarting'}
    assert 'restart_limit' in rig.fake.sent[-1]['body'] and rig.fake.sent[-1]['critical']


def test_prestart_retrying_then_refused(rig, monkeypatch):
    other = [(99991, 1, '/Users/x/.venv/bin/python -B -m observer_app.publisher --out /elsewhere')]
    monkeypatch.setattr(ws, 'process_table', lambda: other)
    rig.control()
    rig.start()
    status = rig.until(lambda s: any(p['key'] == 'prestart_retrying' and p['active'] for p in s['problems']))
    assert status['state'] == 'prestart'
    rig.watch()
    fired = rig.fired()
    assert fired['ops_problem:prestart_retrying'] == 'warn'
    assert fired['publisher_down'] == 'critical'  # in window, no health.json yet
    assert 'ops_status_stale' not in fired
    rig.stop()
    assert rig.result['reason'] == 'operator_stop_before_start'


def test_prestart_refused_is_urgent(rig, monkeypatch):
    other = [(99991, 1, '/Users/x/.venv/bin/python -B -m observer_app.publisher --out /elsewhere')]
    monkeypatch.setattr(ws, 'process_table', lambda: other)
    rig.config(prestart_retry_s=600)  # a retry would land after first pitch: refused at once
    rig.control()
    rig.start()
    assert rig.join() == 'prestart_refused'
    status = rig.status()
    assert status['state'] == 'refused' and status['publisher']['running'] is False
    rig.watch()
    fired = rig.fired()
    assert fired['ops_state:refused'] == 'critical' and fired['ops_problem:prestart_refused'] == 'critical'
    assert fired['publisher_down'] == 'critical'
    assert 'health.json 없음' in rig.fake.sent[-1]['body'] and '시작 전 점검 거부' in rig.fake.sent[-1]['body']
    assert rig.fake.sent[-1]['critical']


def test_upload_blocked_is_urgent_once(rig, monkeypatch):
    rig.config(upload={'allowed': True})
    monkeypatch.setenv(ws.TOKEN_ENV, TOKEN)
    monkeypatch.setenv(ws.BASE_URL_ENV, 'https://example.invalid/x/_publish')
    rig.control(upload_blocked=True, interval=0.1)
    rig.start(upload=True)
    rig.until(lambda s: running_with_health(s) and any(
        p['key'] == 'upload_blocked' and p['active'] for p in s['problems']))
    rig.watch()
    # the watcher's own upload rule (state file phase=blocked) fires; the launcher's same problem is not repeated
    assert rig.fired() == {'upload_blocked': 'critical'}
    assert not any(e['key'] == 'ops_problem:upload_blocked' for e in rig.events)
    assert 'conflict_412' in rig.fake.sent[-1]['body']
    for path in rig.tmp.joinpath('alert').rglob('*'):
        assert TOKEN.encode() not in path.read_bytes() and b'example.invalid' not in path.read_bytes()
    rig.stop()


def test_all_final_is_quiet_and_says_watch_end(rig):
    rig.control(final_after=25, interval=0.2)
    rig.start()
    rig.until(running_with_health)
    rig.watch()
    assert rig.fired() == {} and rig.notices() == [f'watch_start:{PK}']
    assert rig.join() == 'final_settled'
    status = rig.status()
    assert status['state'] == 'stopped' and status['games'][0]['status'] == 'final'
    rig.watch()
    assert rig.fired() == {} and rig.fired('recover') == {}
    assert rig.notices() == [f'watch_end:{PK}']
    assert '감시 끝' in rig.fake.sent[-1]['body'] and not rig.fake.sent[-1]['critical']
    rig.watch()
    assert rig.events == [] and len(rig.fake.sent) == 2


def test_start_refusal_alerts_then_next_supervisor_clears_it(rig, monkeypatch):
    """REVIEW-002 D1/D3: a refused `start` (schedule source down for the whole retry window) reaches the watcher as
    critical; the next supervisor's status carries start_refused as inactive and the alert resolves."""
    rig.config(start_retry_s=0.1, start_retry_window_s=0.5)
    good = rig.schedule.read_text()
    rig.schedule.unlink()
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date,
                    '--schedule-file', str(rig.schedule)]) == 2
    status = rig.status()
    assert status['state'] == 'refused' and status['refused_by'] == 'start'
    rig.watch()
    fired = rig.fired()
    assert fired['ops_state:refused'] == 'critical' and fired['ops_problem:start_refused'] == 'critical'
    rig.schedule.write_text(good)
    rig.control()
    rig.start()
    status = rig.until(running_with_health)
    assert [p for p in status['problems'] if p['key'] == 'start_refused'][0]['active'] is False
    rig.watch()
    assert 'ops_state:refused' not in rig.fired() and 'ops_problem:start_refused' not in rig.fired()
    closed = {e['key'] for e in rig.events if e['kind'] in ('recover', 'closed')}
    assert {'ops_state:refused', 'ops_problem:start_refused'} <= closed


def test_off_day_and_all_final_starts_leave_the_watcher_quiet(rig, tmp_path):
    """REVIEW-003 B3: the daily start on a date with nothing to run (all games final, or no WS game at all) must not
    page: status.json keeps the clean stop and the watcher stays quiet."""
    rig.control(final_after=6, interval=0.1)
    rig.start()
    assert rig.join() == 'final_settled'
    rig.watch()
    rig.watch()
    assert rig.fired() == {} and rig.events == []
    before, sent = rig.status_path.read_bytes(), len(rig.fake.sent)
    # same date again: every game final -> benign refusal, nothing written
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date, '--schedule-file', str(rig.schedule)]) == 2
    # an off day: no WS game on the date
    empty = tmp_path/'off_day.json'
    empty.write_text(json.dumps({'dates': []}))
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date, '--schedule-file', str(empty)]) == 2
    assert rig.status_path.read_bytes() == before and rig.status()['state'] == 'stopped'
    rig.watch()
    assert rig.fired() == {} and rig.events == [] and len(rig.fake.sent) == sent  # nothing new sent


def test_statsapi_error_body_at_start_pages_instead_of_silence(rig):
    """REVIEW-004 B4: an error body is a failed schedule check (retried, then urgent), never a quiet off day."""
    rig.config(start_retry_s=0.1, start_retry_window_s=0.3)
    rig.schedule.write_text(json.dumps({'messageNumber': 10, 'message': 'Object not found'}))
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date,
                    '--schedule-file', str(rig.schedule)]) == 2
    rig.watch()
    fired = rig.fired()
    assert fired['ops_state:refused'] == 'critical' and fired['ops_problem:start_refused'] == 'critical'


# ---- OPS-03: the T7 artifact drive goes away while the publisher runs (a fake mount path; the real T7 is never read)
FAKE_MOUNT = '/nonexistent/ops03-fake-T7'


def active(status, key):
    return any(p['key'] == key and p['active'] for p in status['problems'])


def test_drive_lost_mid_run_is_one_warning_and_clears_when_it_returns(rig, monkeypatch):
    up = {'now': True}
    monkeypatch.setattr(ws, 'is_mounted', lambda path: path == FAKE_MOUNT and up['now'])
    rig.config(required_mounts=[FAKE_MOUNT])
    rig.control(interval=0.1)
    rig.start()
    rig.until(running_with_health)
    up['now'] = False
    status = rig.until(lambda s: active(s, 'mount_lost'))
    # the publisher is not stopped or restarted for it: only the recommendation goes away
    assert status['state'] == 'running' and status['publisher']['running'] is True and status['restarts']['count'] == 0
    assert status['resources']['mounts_missing'] == [FAKE_MOUNT]
    rig.watch()
    assert rig.fired() == {'ops_problem:mount_lost': 'warn'}
    body = rig.fake.sent[-1]['body']
    assert 'not mounted' in body and FAKE_MOUNT not in body and not rig.fake.sent[-1]['critical']
    up['now'] = True
    status = rig.until(lambda s: not active(s, 'mount_lost'))
    assert not active(status, 'mount_restart_needed')  # this segment loaded with the drive: it recovers by itself
    rig.watch()
    assert set(rig.fired('recover')) == {'ops_problem:mount_lost'} and rig.fired() == {}
    rig.stop()
    assert rig.result['reason'] == 'operator_stop'


def test_segment_started_without_the_drive_asks_for_a_restart_after_it_returns(rig, monkeypatch):
    calls, up = [], {'now': False}

    def mounted(path):
        calls.append(path)
        return len(calls) == 1 or up['now']  # prestart sees it; it is gone when the publisher starts
    monkeypatch.setattr(ws, 'is_mounted', mounted)
    rig.config(required_mounts=[FAKE_MOUNT])
    rig.control(interval=0.1)
    rig.start()
    rig.until(lambda s: running_with_health(s) and active(s, 'mount_lost'))
    up['now'] = True
    status = rig.until(lambda s: active(s, 'mount_restart_needed') and not active(s, 'mount_lost'))
    assert status['state'] == 'running' and status['restarts']['count'] == 0
    rig.watch()
    assert rig.fired().get('ops_problem:mount_restart_needed') == 'warn'
    assert 'stop + start' in rig.fake.sent[-1]['body']
    rig.stop()
    status = rig.status()
    assert not active(status, 'mount_restart_needed') and not active(status, 'mount_lost')


def test_mount_problem_keys_are_warnings_that_pass_through_the_watcher():
    """OPS-03 keys: launcher warnings the watcher forwards as ops_problem:<key> (it cannot see the mount itself)."""
    assert 'mount_lost' in ws.ALERT_KEYS and 'mount_restart_needed' in ws.ALERT_KEYS
    assert ws.PROBLEM_SEVERITY['mount_lost'] == ws.PROBLEM_SEVERITY['mount_restart_needed'] == 'warn'
    assert 'mount_lost' not in wa.OPS_ALERTS_COVERED  # the watcher cannot see the mount itself: it passes through


# ---------------------------------------------------------------------------------------------- OPS-02 drill paths

def test_drill_process_table_hook_only_with_the_fake(tmp_path, monkeypatch):
    """FAKE_TABLE_ENV replaces the guard's process table only while the fake publisher is in use."""
    table = tmp_path/'table.json'
    table.write_text('[]')
    rehearsal = [(57237, 1, '/opt/x/Python -I -B /p/rehearse.py supervise --game 1')]
    monkeypatch.setattr(ws, 'process_table', lambda: rehearsal)
    monkeypatch.setenv(ws.FAKE_TABLE_ENV, str(table))
    monkeypatch.delenv(ws.FAKE_ENV, raising=False)
    assert ws.injected_process_table() is None
    assert [o['pid'] for o in ws.other_publishers()] == [57237]  # without the fake: the real guard
    monkeypatch.setenv(ws.FAKE_ENV, str(FAKE))
    assert ws.injected_process_table() == [] and ws.other_publishers() == []
    table.write_text(json.dumps([[4242, 1, '/v/bin/python -B -m observer_app.publisher --out /x']]))
    assert [o['pid'] for o in ws.other_publishers()] == [4242]
    table.write_text('not json')
    with pytest.raises(ValueError):
        ws.other_publishers()


def test_external_kill_of_our_child_restarts_and_alerts_clear(rig):
    """A publisher killed from outside (SIGKILL of the pid the supervisor recorded) -> backoff (critical) -> restart ->
    running (recovered) -> operator stop -> stopped/operator_stop, quiet."""
    import signal
    rig.config(restart={'max_restarts': 2, 'backoff_s': [1.5], 'health_stall_s': 30, 'startup_grace_s': 10})
    rig.control(interval=0.1, live=True)
    sup = rig.start()
    rig.until(running_with_health)
    rig.watch()
    assert rig.fired() == {} and rig.notices() == [f'watch_start:{PK}']
    ident = json.loads((rig.run_dir/'publisher.identity.json').read_text())
    assert ident['pid'] == sup.proc.pid and ws.is_our_publisher(ws.ps_identity(ident['pid']),
                                                                 ident['identity']['lstart'], rig.run_dir)
    os.kill(ident['pid'], signal.SIGKILL)
    status = rig.until(lambda s: s['state'] == 'backoff')
    assert status['restarts']['last_exit']['why'] == 'publisher_exited:-9'
    rig.watch()
    fired = rig.fired()
    assert fired['ops_state:backoff'] == 'critical' and fired['publisher_down'] == 'critical'
    assert fired['ops_problem:publisher_restarting'] == 'warn'
    status = rig.until(lambda s: running_with_health(s) and s['publisher']['segment'] == 2)
    assert status['restarts']['count'] == 1 and status['publisher']['pid'] != ident['pid']
    time.sleep(0.3)
    rig.watch()
    assert {'ops_state:backoff', 'publisher_down', 'ops_problem:publisher_restarting'} <= set(rig.fired('recover'))
    assert rig.fired() == {}
    rig.stop()
    assert rig.result['reason'] == 'operator_stop'
    status = rig.status()
    assert status['state'] == 'stopped' and status['state_reason'] == 'operator_stop'
    assert not any(p['active'] for p in status['problems'])


def test_file_stuck_then_hang_publisher_down_and_stall_restart(rig):
    """Watcher rules on a live game with short drill thresholds: file_stuck (checks advance, file does not), then a
    hung publisher (health_stalled warn + publisher_down critical) that the launcher restarts (health_stalled)."""
    rig.config(restart={'max_restarts': 1, 'backoff_s': [2.0], 'health_stall_s': 8, 'startup_grace_s': 0.5})
    # published_at / checked_at have 1 s resolution: every age below may read up to 1 s high
    rig.wcfg['thresholds'].update(file_stuck_s=1.0, health_stale_s=1.5, publisher_down_s=4.0, game_check_stale_s=1.5)
    rig.control(interval=0.1, live=True)
    rig.start()
    rig.until(running_with_health)
    rig.watch()
    assert rig.fired() == {}
    rig.control(interval=0.1, live=True, freeze_file=True)
    time.sleep(0.4)
    rig.watch()  # tracker starts the clock on the frozen file_sequence
    time.sleep(1.4)
    rig.watch()
    assert rig.fired() == {f'file_stuck:{PK}': 'warn'}
    rig.control(interval=0.1, live=True)
    time.sleep(0.4)
    rig.watch()
    assert set(rig.fired('recover')) == {f'file_stuck:{PK}'}
    rig.control(interval=0.1, live=True, hang=True)
    time.sleep(2.6)
    rig.watch()
    # the launcher's own health_stale problem is not repeated by the watcher
    assert rig.fired() == {'health_stalled': 'warn', f'game_check_stalled:{PK}': 'warn'}
    time.sleep(2.4)
    rig.watch()
    assert rig.fired() == {'publisher_down': 'critical'} and rig.status()['state'] == 'running'
    status = rig.until(lambda s: (s['restarts'].get('last_exit') or {}).get('why') == 'health_stalled', timeout=20)
    assert status['state'] in ('stopping', 'backoff') and status['publisher']['running'] is False
    rig.control(interval=0.1, live=True)  # the restarted segment publishes normally
    status = rig.until(lambda s: s['publisher'].get('segment') == 2 and running_with_health(s), timeout=20)
    assert status['restarts']['history'][0]['why'] == 'health_stalled'
    time.sleep(0.3)
    rig.watch()
    assert {'health_stalled', 'publisher_down', f'game_check_stalled:{PK}'} <= set(rig.fired('recover'))
    rig.stop()
    assert rig.result['reason'] == 'operator_stop'


def test_cli_start_supervisor_really_runs_then_cli_stop(rig, monkeypatch):
    """OPS-02: the real `start` spawns a detached supervisor process (no thread, no injected spawn). Before the fix its
    argv put --config after 'supervise', the child died on an argparse error and start still printed success."""
    table = rig.tmp/'table.json'
    table.write_text('[]')
    monkeypatch.setenv(ws.FAKE_TABLE_ENV, str(table))  # the child cannot see this test's monkeypatched process table
    rig.control(interval=0.1, live=True)
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date,
                    '--schedule-file', str(rig.schedule)]) == 0
    sup = json.loads((rig.run_dir/'supervisor.identity.json').read_text())
    assert ws.is_our_supervisor(ws.ps_identity(sup['pid']), sup, rig.run_dir)
    status = rig.until(running_with_health)
    assert status['supervisor']['pid'] == sup['pid'] and status['publisher']['fake_process_table'] is True
    rig.watch()
    assert rig.fired() == {} and rig.notices() == [f'watch_start:{PK}']
    assert ws.main(['--config', str(rig.cfg_path), 'stop', '--date', rig.date]) == 0
    status = rig.until(lambda s: s['state'] == 'stopped')
    assert status['state_reason'] == 'operator_stop'
    end = time.time()+10
    while time.time() < end and ws.ps_identity(sup['pid']):
        time.sleep(0.1)
    assert not ws.is_our_supervisor(ws.ps_identity(sup['pid']), sup, rig.run_dir)


def test_cli_start_reports_a_supervisor_that_dies_at_once(rig, monkeypatch, capsys):
    """If the detached supervisor never records supervisor_start, start fails loudly and the watcher pages."""
    rig.config(start_confirm_s=1)
    monkeypatch.setattr(ws, 'spawn_supervisor', lambda argv, run_dir: 999999)
    assert ws.main(['--config', str(rig.cfg_path), 'start', '--date', rig.date,
                    '--schedule-file', str(rig.schedule)]) == 2
    assert 'did not start' in capsys.readouterr().out
    status = rig.status()
    assert status['state'] == 'refused' and status['refused_by'] == 'start'
    rig.watch()
    assert rig.fired()['ops_problem:start_refused'] == 'critical'
