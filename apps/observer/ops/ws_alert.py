#!/usr/bin/env python3
"""World Series publish anomaly watcher: tells Song (reading on a phone) when live publishing goes wrong.

Standard library only. It only READS: the publisher's output folder (``v1/health.json``, ``v1/games/<date>.json``,
``v1/game/<pk>.json``), the publisher state file (``<out>.state.json``), an optional operations status file, the
daily profile refresh ``RESULTS.log``, ``vm_stat`` and free disk space. It never signals or lists processes, never
writes into the publisher's folders, and keeps its own state in ``alert_dir`` (default ``~/.pitcheezy/ws-alert``):
``alert_state.json`` and the append-only ``alerts.jsonl``.

    python3 apps/observer/ops/ws_alert.py --once --dry-run       # evaluate, print, send/write nothing
    python3 apps/observer/ops/ws_alert.py --once                 # evaluate, notify, remember
    python3 apps/observer/ops/ws_alert.py --loop                 # every loop_interval_s
    python3 apps/observer/ops/ws_alert.py --test-notify          # local notifiers only (jsonl + macOS)
    python3 apps/observer/ops/ws_alert.py --test-notify --push   # also the configured phone push

Thresholds live in ``ws_alert_config.json`` (all provisional). Rules, notifier setup and the ops status interface:
docs/pm/service/ws-operations-runbook.md section 7.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
GIB = 1024 ** 3
DEFAULT_CONFIG = Path(__file__).with_name('ws_alert_config.json')
CRITICAL, WARN = 'critical', 'warn'
UNKNOWN_REASONS = ('upload_unknown', 'interrupted_unknown', 'legacy_unknown')
# MLB detailedState values in which the game is actually being played (the file must keep changing). Warmup,
# Delayed*, Suspended*, Manager challenge and the like are 'live' in the service but may sit still for minutes.
IN_PLAY_TEXT = ('In Progress',)
OPS_DOWN_STATES = ('backoff', 'failed', 'refused', 'stopped')
# Supervisor alert strings the watcher already judges itself from the same sources: not repeated.
OPS_ALERTS_COVERED = ('memory_low', 'health_stale', 'feed_failing', 'upload_blocked')
# Inputs whose shape is checked (collect() names them 'health', 'games/<date>', 'game/<pk>', 'state', 'ops').
INPUT_ROLES = ('health', 'games', 'game', 'state', 'ops')

BUILTIN = {
    'paths': {'out_dir': None, 'state_path': None, 'ops_status': '~/Projects/pitcheezy-pm-jobs/ws-ops/status.json',
              'profile_log': '~/Projects/pitcheezy-pm-jobs/profile-refresh-daily/logs/RESULTS.log',
              'alert_dir': '~/.pitcheezy/ws-alert', 'disk_paths': ['~']},
    'contract_version': 'v1', 'loop_interval_s': 15,
    'game_window': {'before_start_s': 1800, 'max_game_s': 21600}, 'schedule': [],
    'thresholds': {'health_stale_s': 30, 'publisher_down_s': 120, 'publisher_down_hold_s': 30,
                   'game_check_stale_s': 30,
                   'game_check_missing_hold_s': 60, 'feed_down_hold_s': 60, 'feed_lag_s': 45,
                   'feed_delayed_hold_s': 20, 'file_stuck_s': 450, 'file_stuck_upstream_s': 60,
                   'game_unavailable_hold_s': 30,
                   'state_unreadable_hold_s': 30, 'memory_available_gib': 2.0,
                   'memory_hold_s': 60, 'disk_free_gib': 12.5, 'profile_deadline_kst': '21:20',
                   'ops_stale_running_s': 90, 'ops_stale_other_s': 300, 'upload_pending_floor': 5,
                   'upload_pending_cycles': 4, 'upload_oldest_s': 60},
    'notify': {'cooldown_s': 300, 'renotify_s': {CRITICAL: 900, WARN: 1800}, 'renotify_outside_game_s': 21600},
    'game_gated': ['health_stalled', 'publisher_down', 'game_check_stalled', 'feed_down', 'feed_delayed',
                   'file_stuck', 'game_unavailable', 'upload_pending', 'upload_stalled', 'memory_low'],
}


# ----------------------------------------------------------------------------- config and small helpers

def _merge(base, over):
    out = dict(base)
    for key, value in (over or {}).items():
        out[key] = _merge(base[key], value) if isinstance(base.get(key), dict) and isinstance(value, dict) else value
    return out


def load_config(path=None, *, out_dir=None, ops_status=None, alert_dir=None, env=None):
    """Built-in defaults <- config file <- env <- CLI. Unknown keys (notes) are kept and ignored."""
    env = os.environ if env is None else env
    cfg = _merge(BUILTIN, {})
    path = Path(path) if path else DEFAULT_CONFIG
    if path.exists():
        cfg = _merge(cfg, json.loads(path.read_text()))
    paths = cfg['paths']
    if env.get('PITCHEEZY_WS_OPS_ROOT'):
        paths['ops_status'] = str(Path(env['PITCHEEZY_WS_OPS_ROOT']) / 'status.json')
    for key, name in (('out_dir', 'PITCHEEZY_PUBLISH_OUT_DIR'), ('state_path', 'PITCHEEZY_PUBLISH_STATE'),
                      ('ops_status', 'PITCHEEZY_WS_OPS_STATUS'), ('profile_log', 'PITCHEEZY_PROFILE_REFRESH_LOG'),
                      ('alert_dir', 'PITCHEEZY_ALERT_DIR')):
        if env.get(name):
            paths[key] = env[name]
    cfg['cli_out'] = bool(out_dir)  # an explicit --out wins over the ops status file's out_dir
    for key, value in (('out_dir', out_dir), ('ops_status', ops_status), ('alert_dir', alert_dir)):
        if value:
            paths[key] = value
    for key in ('out_dir', 'state_path', 'ops_status', 'profile_log', 'alert_dir'):
        if paths.get(key):
            paths[key] = str(Path(paths[key]).expanduser())
    if paths.get('out_dir') and not paths.get('state_path'):
        out = Path(paths['out_dir'])
        paths['state_path'] = str(out.parent / f'{out.name}.state.json')
    return cfg


def parse_time(value):
    """ISO string (Z or offset) or epoch number -> epoch seconds; None when unreadable."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _read_json(path):
    """(body, error). error is None, 'missing', 'unreadable' or 'malformed'. One retry covers a file caught mid-replace.

    Every input the watcher reads is a JSON object. Valid JSON of another shape (a list, a number, a string) is
    'malformed' with body None, so one bad input never raises inside collect/evaluate and silences the independent
    rules (OPS-ALERT-JSON01)."""
    if not path:
        return None, 'missing'
    path = Path(path)
    for attempt in range(2):
        try:
            body = json.loads(path.read_text())
            return (body, None) if isinstance(body, dict) else (None, 'malformed')
        except FileNotFoundError:
            return None, 'missing'
        except (OSError, ValueError):
            if attempt == 0:
                time.sleep(0.2)
    return None, 'unreadable'


def _last_line(path):
    try:
        with open(path, 'rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 8192))
            lines = [line for line in handle.read().decode('utf-8', 'replace').splitlines() if line.strip()]
    except OSError:
        return None
    return lines[-1] if lines else ''


def memory_available_bytes():
    """macOS ``vm_stat``: free + inactive + speculative pages. An approximation of 'available', not pressure."""
    try:
        text = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = re.search(r'page size of (\d+) bytes', text)
    pages = {m[1].strip().lower(): int(m[2]) for m in re.finditer(r'^(Pages [^:]+):\s+(\d+)\.', text, re.M)}
    if not page or 'pages free' not in pages:
        return None
    return int(page[1]) * sum(pages.get(k, 0) for k in ('pages free', 'pages inactive', 'pages speculative'))


def disk_free_bytes(paths):
    out = {}
    for raw in paths:
        path = Path(raw).expanduser()
        while not path.exists() and path != path.parent:
            path = path.parent
        try:
            out[str(raw)] = shutil.disk_usage(path).free
        except OSError:
            out[str(raw)] = None
    return out


# ----------------------------------------------------------------------------- observation

def collect(cfg, now, *, memory=memory_available_bytes, disk=disk_free_bytes):
    """Everything the rules look at, read once. Missing inputs are recorded, never raised."""
    paths, version = cfg['paths'], cfg['contract_version']
    # reads: input name -> read result (None = an object was read, else missing/unreadable/malformed), for every input
    # attempted this run; malformed: the names whose JSON was not an object (OPS-ALERT-JSON01).
    obs = {'now': now, 'games': [], 'game_files': {}, 'malformed': [], 'reads': {}}

    def note(name, error):
        obs['reads'][name] = error
        if error == 'malformed':
            obs['malformed'].append(name)

    obs['ops'], obs['ops_error'] = _read_json(paths['ops_status']) if paths.get('ops_status') else (None, None)
    if paths.get('ops_status'):
        note('ops', obs['ops_error'])
    out_dir, state_path = paths.get('out_dir'), paths.get('state_path')
    ops = obs['ops']
    # The ops status (v1.1) names the current day's publisher output and state file: follow it (--out still wins).
    if isinstance(ops, dict) and ops_is_current(ops, now) and not cfg.get('cli_out'):
        if isinstance(ops.get('out_dir'), str) and ops['out_dir']:
            out_dir = str(Path(ops['out_dir']).expanduser())
            state_path = None
        if isinstance(ops.get('state_path'), str) and ops['state_path']:
            state_path = str(Path(ops['state_path']).expanduser())
        if out_dir and not state_path:
            state_path = str(Path(out_dir).parent / f'{Path(out_dir).name}.state.json')
    obs['out_dir'], obs['state_path'] = out_dir, state_path
    root = Path(out_dir) / version if out_dir else None
    if root is not None:
        obs['health'], obs['health_error'] = _read_json(root / 'health.json')
        note('health', obs['health_error'])
        try:
            obs['health_mtime'] = (root / 'health.json').stat().st_mtime
        except OSError:
            obs['health_mtime'] = None
        today = datetime.fromtimestamp(now, KST).date()
        for delta in (-1, 0, 1):
            day = today + timedelta(days=delta)
            body, error = _read_json(root / 'games' / f'{day}.json')
            games = (body or {}).get('games') or []
            if error is None and not isinstance(games, list):
                error = 'malformed'
            note(f'games/{day}', error)
            if error is not None:
                continue
            for game in games:
                if isinstance(game, dict) and isinstance(game.get('game_pk'), int):
                    obs['games'].append(game)
    obs['state'], obs['state_error'] = _read_json(state_path)
    note('state', obs['state_error'])
    pks = {g['game_pk'] for g in obs['games'] if g.get('status') == 'live' or g.get('live')}
    checks = (obs.get('health') or {}).get('game_checks')
    pks |= {int(k) for k in (checks if isinstance(checks, dict) else {}) if str(k).isdigit()}
    for item in _ops_games(obs['ops']):
        pks.add(item['game_pk'])
    if root is not None:
        for pk in sorted(pks):
            body, error = _read_json(root / 'game' / f'{pk}.json')
            note(f'game/{pk}', error)
            if isinstance(body, dict):
                obs['game_files'][pk] = body
    obs['memory'] = memory()
    disk_paths = list(paths.get('disk_paths') or [])
    if out_dir:
        disk_paths.append(str(Path(out_dir).parent))
    obs['disk'] = disk(disk_paths)
    obs['profile_line'] = _last_line(Path(paths['profile_log']).expanduser()) if paths.get('profile_log') else None
    return obs


def ops_is_current(ops, now):
    """A status file from an older operating day says nothing about today (the supervisor writes kst_date)."""
    day = datetime.fromtimestamp(now, KST).date()
    return not isinstance(ops.get('kst_date'), str) or ops['kst_date'] in (
        day.isoformat(), (day - timedelta(days=1)).isoformat())


def _ops_games(ops):
    """Ops status games, tolerant: ``games`` or ``schedule``; entries without an integer game_pk are skipped."""
    if not isinstance(ops, dict):
        return []
    items = ops.get('games') if isinstance(ops.get('games'), list) else ops.get('schedule')
    out = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            try:
                out.append(dict(item, game_pk=int(item.get('game_pk'))))
            except (TypeError, ValueError):
                continue
    return out


def game_windows(cfg, obs):
    """{game_pk: info} for games whose window contains now. Sources merged; a live status wins over the clock."""
    now, gw = obs['now'], cfg['game_window']
    cands = {}
    for source, items in (('schedule', cfg.get('schedule') or []), ('publisher', obs['games']),
                          ('ops', _ops_games(obs.get('ops')) if isinstance(obs.get('ops'), dict)
                           and ops_is_current(obs['ops'], now) else [])):
        for item in items:
            try:
                pk = int(item.get('game_pk'))
            except (TypeError, ValueError):
                continue
            entry = cands.setdefault(pk, {'game_pk': pk, 'start': None, 'status': None, 'sources': [],
                                          'window_start': None, 'window_end': None, 'live_from': set()})
            entry['sources'].append(source)
            if item.get('status') == 'live' or (item.get('live') is True and source == 'publisher'):
                entry['live_from'].add(source)  # 'ops' alone = the launcher's clock (start passed), not play seen
            entry['start'] = parse_time(item.get('start') or item.get('start_utc')) or entry['start']
            entry['status'] = item.get('status') or entry['status']
            entry['status_text'] = item.get('status_text') or entry.get('status_text')
            if source == 'ops' and item.get('final_seen_at'):
                entry['status'] = 'final'
            if item.get('live') is True and source == 'publisher' and entry['status'] != 'final':
                entry['status'] = 'live'
            entry['window_start'] = parse_time(item.get('window_start')) or entry['window_start']
            entry['window_end'] = parse_time(item.get('window_end')) or entry['window_end']
    for pk, body in obs['game_files'].items():  # the game file is the publisher's newest word on its status
        game = body.get('game') or {}
        if pk in cands and game.get('status'):
            cands[pk]['status'] = game['status']
            cands[pk]['status_text'] = game.get('status_text')
            if game['status'] == 'live':
                cands[pk]['live_from'].add('game_file')
    out = {}
    for pk, entry in cands.items():
        if entry['status'] in ('final', 'other', 'postponed'):
            continue
        start = entry['window_start'] or (entry['start'] - gw['before_start_s'] if entry['start'] else None)
        end = entry['window_end'] or (entry['start'] + gw['max_game_s'] if entry['start'] else None)
        in_clock = start is not None and start <= now <= (end if end is not None else start + gw['max_game_s'])
        live_bounded = entry['status'] == 'live' and (end is None or now <= end)
        if in_clock or live_bounded:
            out[pk] = entry
    return out


# ----------------------------------------------------------------------------- rules

def _cond(key, severity, active, message, hold_s=0, **detail):
    return {'key': key, 'family': key.split(':')[0], 'severity': severity, 'active': bool(active),
            'message': message, 'hold_s': hold_s, 'detail': detail}


def _fmt_s(seconds):
    seconds = int(seconds)
    return f'{seconds}초' if seconds < 120 else f'{seconds // 60}분'


def evaluate(cfg, obs, tracker):
    """Conditions for this run. ``tracker`` (persisted) carries per-game file_sequence history and upload pending.

    Only evaluated keys are returned; a key that is absent (no game in window, game gone, before the profile
    deadline) is closed silently by the engine, never announced as recovered."""
    th, now = cfg['thresholds'], obs['now']
    windows = game_windows(cfg, obs)
    in_game = bool(windows)
    tracker['windows_now'] = sorted(windows)
    gated = set(cfg['game_gated'])
    conds = []

    if not obs.get('out_dir'):
        conds.append(_cond('watch_config', WARN, True, '감시 설정: 발행 출력 폴더가 지정되지 않음(PITCHEEZY_PUBLISH_OUT_DIR)'))
        health = None
    else:
        conds.append(_cond('watch_config', WARN, False, ''))
        health = obs.get('health') if isinstance(obs.get('health'), dict) else None

    ops = obs.get('ops') if isinstance(obs.get('ops'), dict) else None
    ops_pub = (ops or {}).get('publisher') if isinstance((ops or {}).get('publisher'), dict) else {}
    ops_current = ops is not None and ops_is_current(ops, now)
    ops_state = ops.get('state') if ops_current and isinstance(ops.get('state'), str) else None

    # --- publisher liveness (game-gated)
    if in_game and obs.get('out_dir'):
        names = ', '.join(str(pk) for pk in sorted(windows))
        published = parse_time((health or {}).get('published_at')) or obs.get('health_mtime')
        age = None if published is None else max(0.0, now - published)
        ops_says_down = ops_pub.get('running') is False or ops_state in OPS_DOWN_STATES
        stalled = age is not None and age > th['publisher_down_s']
        down = health is None or age is None or stalled or ops_says_down
        # OPS-02 (PM 2026-10-19): the window opens when the launcher starts the publisher, so 'no health.json yet' /
        # 'publisher.running false' (prestart) is normal for a few seconds: hold those for publisher_down_hold_s.
        # A launcher that says backoff/failed/refused/stopped, or a health stalled > publisher_down_s, fires at once.
        hold = 0 if (ops_state in OPS_DOWN_STATES or stalled) else th['publisher_down_hold_s']
        why = ('health.json 없음' if obs.get('health_error') == 'missing' else
               'health.json 읽기 실패' if health is None else
               f'health {_fmt_s(age)} 멈춤' if age is not None and age > th['publisher_down_s'] else
               f'운영 상태 {ops_state or "publisher.running=false"}')
        conds.append(_cond('publisher_down', CRITICAL, down,
                           f'발행기 꺼진 듯: {why} (경기 {names} 시간대)', hold, age_s=age))
        conds.append(_cond('health_stalled', WARN, age is not None and age > th['health_stale_s'],
                           f'발행 정체: health {_fmt_s(age or 0)} 안 바뀜(기준 {th["health_stale_s"]}초)', age_s=age))

    # --- per game (game-gated)
    files = tracker.setdefault('files', {})
    ops_by_pk = {item['game_pk']: item for item in _ops_games(ops)}
    if in_game and health is not None:
        checks = health.get('game_checks') if isinstance(health.get('game_checks'), dict) else {}
        failing = set(health.get('feed_failing_games') or [])
        unavailable = {}
        for item in health.get('unavailable_games') or []:
            if isinstance(item, dict) and 'game_pk' in item and item.get('code') != 'upload_failed':
                unavailable.setdefault(item['game_pk'], []).append(str(item.get('code')))
        for pk, info in sorted(windows.items()):
            body = obs['game_files'].get(pk) or {}
            live = info['status'] == 'live' or (body.get('game') or {}).get('status') == 'live'
            gc = checks.get(str(pk))
            codes = unavailable.get(pk, [])
            conds.append(_cond(f'game_unavailable:{pk}', WARN, bool(codes),
                               f'경기 {pk} 발행 실패 코드: {", ".join(sorted(set(codes)))}',
                               th['game_unavailable_hold_s']))
            if not live:
                continue
            if not isinstance(gc, dict):
                conds.append(_cond(f'game_check_stalled:{pk}', WARN, True,
                                   f'경기 {pk} 진행 중인데 health에 확인 기록(game_checks) 없음',
                                   th['game_check_missing_hold_s']))
                continue
            checked = parse_time(gc.get('checked_at'))
            check_age = None if checked is None else max(0.0, now - checked)
            conds.append(_cond(f'game_check_stalled:{pk}', WARN,
                               check_age is None or check_age > th['game_check_stale_s'],
                               f'경기 {pk} 확인 정체: checked_at {_fmt_s(check_age or 0)} 멈춤', age_s=check_age))
            feed_bad = gc.get('feed_ok') is False or pk in failing
            conds.append(_cond(f'feed_down:{pk}', CRITICAL, feed_bad,
                               f'경기 {pk} 피드 끊김(feed_ok false) — 화면 "연결 끊김"', th['feed_down_hold_s']))
            feed = ((body.get('display_state') or {}).get('feed') or {}) if body else {}
            lag = feed.get('lag_s') if isinstance(feed.get('lag_s'), (int, float)) else None
            delayed = feed.get('delayed') is True  # the publisher's own judgement (lag > max(45, viewer delay))
            limit = feed.get('lag_threshold_s', th['feed_lag_s'])
            conds.append(_cond(f'feed_delayed:{pk}', WARN, delayed and not feed_bad,
                               f'경기 {pk} 피드 지연 {lag if lag is not None else "?"}초(기준 {limit}초)',
                               th['feed_delayed_hold_s'], lag_s=lag))
            # file_sequence must move while the game moves: checks happen, the feed is fine, yet no new file.
            seq = gc.get('file_sequence')
            text = (body.get('game') or {}).get('status_text') or info.get('status_text')
            # No status_text: trust a live status only when the publisher said it. The launcher's games[].status
            # 'live' means "scheduled start has passed" (clock), so on its own it never makes a game 'in play'.
            clock_only = info.get('live_from', set()) <= {'ops'} and (body.get('game') or {}).get('status') != 'live'
            in_play = (any(str(text).startswith(t) for t in IN_PLAY_TEXT) if text is not None else not clock_only)
            track = files.get(str(pk))
            if track is None or track.get('file_sequence') != seq or not in_play:
                # Warmup / delay / review: the clock restarts, so the first minutes of play are not 'stuck'.
                track = files[str(pk)] = {'file_sequence': seq, 'since': now}
            unchanged = now - track['since']
            upstream = parse_time((ops_by_pk.get(pk) or {}).get('upstream_changed_at'))
            healthy_checks = check_age is not None and check_age <= th['game_check_stale_s'] and not feed_bad
            stuck_long = unchanged > th['file_stuck_s']
            stuck_upstream = (upstream is not None and upstream > track['since']
                              and now - upstream > th['file_stuck_upstream_s'])
            conds.append(_cond(f'file_stuck:{pk}', WARN, in_play and healthy_checks and (stuck_long or stuck_upstream),
                               f'경기 {pk} 파일 갱신 멈춤: 확인은 도는데 file_sequence {seq}가 {_fmt_s(unchanged)} 그대로'
                               + (' (운영 상태는 경기 진행을 보고)' if stuck_upstream else ''), unchanged_s=unchanged))
    for key in list(files):
        if not in_game or int(key) not in windows:
            files.pop(key)

    # --- upload. With background upload the publisher queues dates/manifest/changed game files every cycle and
    # samples pending right after, so pending >= 2 and upload.code='upload_failed' are its normal steady state.
    # Judge progress instead; never use upload.code. A --dry-run-upload publisher never drains: skip it all.
    upload = (health or {}).get('upload') if isinstance((health or {}).get('upload'), dict) else None
    dry_upload = bool(upload and upload.get('dry_run'))
    up = tracker.setdefault('upload', {})
    if upload is not None and in_game and not dry_upload:
        pending = upload.get('pending') if isinstance(upload.get('pending'), int) else 0
        samples = up.setdefault('samples', [])
        seq = (health or {}).get('sequence')
        if not samples or samples[-1][0] != seq:
            samples.append([seq, pending])
        del samples[:-th['upload_pending_cycles']]
        values = [v for _, v in samples]
        growing = (len(values) >= th['upload_pending_cycles'] and min(values) > th['upload_pending_floor']
                   and all(b >= a for a, b in zip(values, values[1:])))
        conds.append(_cond('upload_pending', WARN, growing,
                           f'업로드 밀림: pending {" → ".join(map(str, values))}건, {len(values)}주기째 줄지 않음'
                           f'(기준 {th["upload_pending_floor"]}건 초과)', pending=pending))
    else:
        up.pop('samples', None)
    state = obs.get('state') if isinstance(obs.get('state'), dict) else None
    if obs.get('state_error') == 'unreadable':
        conds.append(_cond('state_unreadable', WARN, True, '발행기 상태 파일 읽기 실패', th['state_unreadable_hold_s']))
    elif obs.get('state_error') is None:
        conds.append(_cond('state_unreadable', WARN, False, ''))
    if state is not None and not dry_upload:
        attempts = state.get('upload_attempts') if isinstance(state.get('upload_attempts'), dict) else {}
        by_phase, stuck_paths = {}, set()
        for path, attempt in attempts.items():
            if isinstance(attempt, dict):
                by_phase.setdefault(attempt.get('phase'), []).append(str(attempt.get('reason')))
                if attempt.get('phase') in ('blocked', 'rejected'):
                    stuck_paths.add(path)
        blocked = by_phase.get('blocked', [])
        conflict = [r for r in blocked if r == 'upload_conflict']
        unknown = [r for r in blocked if r in UNKNOWN_REASONS]
        other = [r for r in blocked if r not in UNKNOWN_REASONS and r != 'upload_conflict']
        if state.get('upload_recovery_blocked'):
            other.append('upload_recovery_blocked')
        rejected = by_phase.get('rejected', [])
        conds.append(_cond('upload_conflict', CRITICAL, bool(conflict),
                           f'업로드 412 충돌 {len(conflict)}건: 다른 발행기가 같은 원격 경로에 쓰는지 확인(런북 4.4)'))
        conds.append(_cond('upload_unknown', CRITICAL, bool(unknown),
                           f'업로드 결과 모름 {len(unknown)}건({", ".join(sorted(set(unknown)))}) — 자동 재전송 없음, 런북 4.4'))
        conds.append(_cond('upload_blocked', CRITICAL, bool(other),
                           f'업로드 막힘 {len(other)}건({", ".join(sorted(set(other)))}) — 런북 4.4'))
        conds.append(_cond('upload_rejected', CRITICAL, bool(rejected),
                           f'업로드 거절(영구 4xx) {len(rejected)}건({", ".join(sorted(set(rejected)))}) — 런북 4.4'))
        # The oldest queued intent (blocked/rejected ones are announced above) must be replaced or sent soon.
        if in_game:
            intents = state.get('upload_intents') if isinstance(state.get('upload_intents'), dict) else {}
            pend = state.get('upload_pending') if isinstance(state.get('upload_pending'), dict) else {}
            revs = [i.get('revision') for p, i in intents.items()
                    if p in pend and p not in stuck_paths and isinstance(i, dict) and isinstance(i.get('revision'), int)]
            oldest = min(revs) if revs else None
            if oldest is None or up.get('oldest') != oldest:
                up['oldest'], up['oldest_since'] = oldest, now
            waited = now - up['oldest_since']
            conds.append(_cond('upload_stalled', WARN, oldest is not None and waited > th['upload_oldest_s'],
                               f'업로드 멈춤: 가장 오래된 전송 대기가 {_fmt_s(waited)}째 그대로(기준 {th["upload_oldest_s"]}초)'))

    # --- machine
    if in_game:
        mem = obs.get('memory')
        if mem is not None:
            conds.append(_cond('memory_low', WARN, mem < th['memory_available_gib'] * GIB,
                               f'맥 메모리 여유 {mem / GIB:.1f} GiB(기준 {th["memory_available_gib"]} GiB) — 런북 4.5',
                               th['memory_hold_s'], bytes=mem))
    low = {p: b for p, b in (obs.get('disk') or {}).items() if b is not None and b < th['disk_free_gib'] * GIB}
    conds.append(_cond('disk_low', WARN, bool(low),
                       '디스크 여유 ' + ', '.join(f'{b / GIB:.1f} GiB' for b in sorted(low.values()))
                       + f'(기준 {th["disk_free_gib"]} GiB, 프로필 갱신이 건너뛰어질 수 있음)'))

    # --- daily profile refresh
    kst = datetime.fromtimestamp(now, KST)
    hh, mm = (int(x) for x in th['profile_deadline_kst'].split(':'))
    if (kst.hour, kst.minute) >= (hh, mm):
        today, yesterday = kst.date().isoformat(), (kst.date() - timedelta(days=1)).isoformat()
        line = obs.get('profile_line')
        ok = bool(line) and re.search(rf'\bok target={today} cutoff={yesterday}\b', line) is not None
        if line is None:
            head = '로그 없음'
        else:
            head = ' '.join(t for t in line.split() if '/' not in t and not t.startswith(('log=', 'version=')))
            head = head[:90] or '빈 로그'
        conds.append(_cond(f'profile_refresh:{today}', WARN, not ok,
                           f'프로필 일일 갱신이 {th["profile_deadline_kst"]}까지 ok 아님: {head}'))

    # --- operations status file (pitcheezy-ws-ops-status-v1 from ws_operate.py; tolerant, runbook 7.3)
    if ops is not None and ops_current:
        reason = ops.get('state_reason') or ops.get('guard') or ''
        last_exit = ((ops.get('restarts') or {}).get('last_exit') or {}) if isinstance(ops.get('restarts'), dict) else {}
        state_msgs = {
            'backoff': (CRITICAL, f'운영: 발행기가 끝나 재시작 대기({last_exit.get("why") or reason or "이유 없음"}) — 공개 화면 갱신 멈춤'),
            'refused': (CRITICAL, f'운영: 시작 전 점검 거부({reason or "이유 없음"})'),
            'failed': (CRITICAL, f'운영: 비정상 종료({reason or "이유 없음"})'),
        }
        for name, (sev, msg) in state_msgs.items():
            conds.append(_cond(f'ops_state:{name}', sev, ops_state == name, msg))
        conds.append(_cond('ops_state:hard_end', WARN, ops_state == 'stopped' and reason == 'hard_end',
                           '운영: 마감 시각(hard_end)에 발행기를 멈춤 — 경기가 끝났는지 확인'))
        items = ops.get('problems') if isinstance(ops.get('problems'), list) else ops.get('alerts')
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str) and item and item not in OPS_ALERTS_COVERED:
                sev = CRITICAL if item == 'protected_process_changed' else WARN
                text = '보호 프로세스(기록기·운영 서버) 변화 감지 — 신호 보내지 말고 확인' if sev == CRITICAL else item
                conds.append(_cond(f'ops_problem:{item}', sev, True, f'운영: {text[:120]}'))
            elif isinstance(item, dict) and item.get('key') and str(item['key']) not in OPS_ALERTS_COVERED:
                sev = CRITICAL if item.get('severity') in (CRITICAL, 'urgent') else WARN
                conds.append(_cond(f'ops_problem:{item["key"]}', sev, item.get('active', True) is not False,
                                   f'운영: {str(item.get("message") or item["key"])[:120]}'))
        if ops_state in (None, 'waiting', 'prestart', 'running', 'backoff', 'stopping'):
            written = parse_time(ops.get('written_epoch')) or parse_time(ops.get('written_at')) \
                or parse_time(ops.get('updated_at'))
            limit = th['ops_stale_running_s'] if ops_state in ('running', 'backoff') else th['ops_stale_other_s']
            age = None if written is None else now - written
            conds.append(_cond('ops_status_stale', WARN, age is None or age > limit,
                               f'운영 감독이 멈춘 듯: 상태 파일 {_fmt_s(age) if age is not None else "시각 없음"} 갱신 안 됨'
                               f'(기준 {limit}초)'))
    elif obs.get('ops_error') == 'unreadable':
        conds.append(_cond('ops_problem:unreadable', WARN, True, '운영 상태 파일 읽기 실패', 30))

    # --- input shape (OPS-ALERT-JSON01). Valid JSON that is not an object: that input is set aside (treated as
    # absent above) and the other rules keep running. One always-on key per input kind, announced once (dedup/renotify
    # as any other key). An input stays 'malformed' (tracker) until it is read again AS AN OBJECT: missing, unreadable
    # or unparsable afterwards never counts as recovery (those have their own rules). An input no longer read at all
    # (date rolled over, game no longer watched) is dropped; if that empties its kind, the key closes silently.
    reads = obs.get('reads') or {}
    known = tracker.setdefault('malformed', {})
    cleared = {}
    for name in list(known):
        if name not in reads:
            known.pop(name)
            cleared.setdefault(name.split('/')[0], set()).add('dropped')
        elif reads[name] is None:
            known.pop(name)
            cleared.setdefault(name.split('/')[0], set()).add('ok')
    for name, error in reads.items():
        if error == 'malformed':
            known[name] = True
    for role in INPUT_ROLES:
        names = sorted(name for name in known if name.split('/')[0] == role)
        if not names and 'dropped' in cleared.get(role, ()):
            continue
        conds.append(_cond(f'input_malformed:{role}', WARN, bool(names),
                           f'감시 입력 형식 오류: {", ".join(names)} 가 JSON 객체가 아님(객체로 다시 읽힐 때까지 유지) '
                           '— 이 입력만 건너뛰고 나머지 감시는 계속'))

    return [c for c in conds if in_game or c['family'] not in gated], in_game


# ----------------------------------------------------------------------------- engine: hold, dedup, cooldown, recovery

def process(cfg, alerts, conds, now, in_game):
    """Update ``alerts`` (persisted dict key -> record) and return events in order.

    Event kinds: fire / renotify / recover (notified), closed / suppressed (log only)."""
    note = cfg['notify']
    events, seen = [], set()
    for c in conds:
        key = c['key']
        seen.add(key)
        rec = alerts.get(key)
        if c['active']:
            if rec is None or rec['status'] == 'resolved':
                rec = alerts[key] = {'status': 'pending', 'first_seen': now, 'severity': c['severity'],
                                     'resolved_at': (rec or {}).get('resolved_at')}
            rec.update(message=c['message'], severity=c['severity'], last_seen=now)
            if rec['status'] == 'pending':
                held = now - rec['first_seen'] >= c['hold_s']
                cooled = rec.get('resolved_at') is None or now - rec['resolved_at'] >= note['cooldown_s']
                if held and cooled:
                    rec.update(status='firing', notified_at=now, last_notified=now)
                    events.append(_event('fire', key, c, now))
                elif held and not rec.get('suppressed_logged'):
                    rec['suppressed_logged'] = True
                    events.append(_event('suppressed', key, c, now))
            elif rec['status'] == 'firing':
                every = note['renotify_s'].get(c['severity'], 1800) if in_game else note['renotify_outside_game_s']
                if now - rec['last_notified'] >= every:
                    rec['last_notified'] = now
                    events.append(_event('renotify', key, c, now, since=rec['notified_at']))
        elif rec is not None:
            if rec['status'] == 'firing':
                rec.update(status='resolved', resolved_at=now)
                rec.pop('suppressed_logged', None)
                events.append(_event('recover', key, c | {'message': rec.get('message', '')}, now,
                                     since=rec.get('notified_at')))
            elif rec['status'] == 'pending':
                if rec.get('resolved_at') is None:
                    alerts.pop(key)
                else:
                    rec.update(status='resolved')
                    rec.pop('suppressed_logged', None)
    for key in list(alerts):
        rec = alerts[key]
        if key in seen:
            continue
        if rec['status'] == 'firing':
            events.append({'kind': 'closed', 'key': key, 'severity': rec['severity'], 'at': _iso(now),
                           'message': rec.get('message', ''), 'why': 'not evaluated any more (no game in window, '
                           'game over, or a new day)'})
        if rec['status'] != 'resolved' or now - (rec.get('resolved_at') or now) > 86400:
            alerts.pop(key)
    return events


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _event(kind, key, cond, now, since=None):
    out = {'kind': kind, 'key': key, 'severity': cond['severity'], 'at': _iso(now), 'message': cond['message']}
    if since is not None:
        out['since'] = _iso(since)
    return out


def window_notices(tracker, now):
    """One notice when a game window opens and one when it closes: a dead-man signal (no notice = no watcher)."""
    current = tracker.pop('windows_now', [])
    previous = tracker.get('window_pks', [])
    tracker['window_pks'] = current
    out = []
    for pk in current:
        if pk not in previous:
            out.append({'kind': 'notice', 'key': f'watch_start:{pk}', 'severity': 'info', 'at': _iso(now),
                        'message': f'감시 시작: 경기 {pk} 시간대(이 알림이 안 오면 감시기가 꺼진 것)'})
    for pk in previous:
        if pk not in current:
            out.append({'kind': 'notice', 'key': f'watch_end:{pk}', 'severity': 'info', 'at': _iso(now),
                        'message': f'감시 끝: 경기 {pk} 시간대 닫힘'})
    return out


def render(events, now):
    """(title, body) for one batched notification, or None when nothing is to be sent."""
    sent = [e for e in events if e['kind'] in ('fire', 'renotify', 'recover', 'notice')]
    if not sent:
        return None
    bad = [e for e in sent if e['kind'] in ('fire', 'renotify')]
    crit = any(e['severity'] == CRITICAL for e in bad)
    recovered = any(e['kind'] == 'recover' for e in sent)
    title = (f'Pitcheezy {"긴급" if crit else "경고"} {len(bad)}건' if bad else
             'Pitcheezy 복구' if recovered else 'Pitcheezy 감시') + \
        f' {datetime.fromtimestamp(now, KST):%H:%M} KST'
    lines = []
    order = {'fire': 0, 'renotify': 1, 'recover': 2, 'notice': 3}
    for e in sorted(sent, key=lambda e: (order[e['kind']], e['severity'] != CRITICAL)):
        tag = {'fire': '긴급' if e['severity'] == CRITICAL else '경고', 'renotify': '계속', 'recover': '복구',
               'notice': '알림'}[e['kind']]
        lines.append(f'[{tag}] {e["message"]}')
    return title, '\n'.join(lines)


# ----------------------------------------------------------------------------- notifiers

class JsonlNotifier:
    """Always on: every event (also closed/suppressed) as one JSON line, append-only."""
    name = 'jsonl'

    def __init__(self, path):
        self.path = Path(path)

    def log(self, record):
        """Never raises: a full disk (exactly when disk_low matters) must not stop the phone/mac sends."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        except OSError as error:
            print(f'ws_alert: alerts.jsonl write failed: {error}', file=sys.stderr)


class MacNotifier:
    """Local macOS banner via osascript. Message text goes in as argv, never spliced into the script."""
    name = 'macos'
    SCRIPT = ['-e', 'on run argv', '-e', 'display notification (item 2 of argv) with title (item 1 of argv)',
              '-e', 'end run']

    def send(self, title, body, *, critical=False):
        subprocess.run(['osascript', *self.SCRIPT, title, body[:600]], check=True, timeout=10,
                       capture_output=True)


class NtfyNotifier:
    """ntfy.sh (or a self-hosted ntfy) JSON publish. The topic name is the secret: env only, never the repo."""
    name = 'ntfy'

    def __init__(self, topic, server='https://ntfy.sh', token=None, opener=urllib.request.urlopen):
        self.topic, self.server, self.token, self.opener = topic, server.rstrip('/'), token, opener

    def send(self, title, body, *, critical=False):
        data = json.dumps({'topic': self.topic, 'title': title, 'message': body[:3500],
                           'priority': 5 if critical else 3, 'tags': ['rotating_light' if critical else 'baseball']},
                          ensure_ascii=False).encode('utf-8')
        request = urllib.request.Request(self.server + '/', data=data, method='POST',
                                         headers={'Content-Type': 'application/json'})
        if self.token:
            request.add_header('Authorization', f'Bearer {self.token}')
        with self.opener(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise OSError(f'ntfy HTTP {response.status}')


class IMessageNotifier:
    """Messages.app on this Mac sends an iMessage to the address in env (phone number or Apple ID email)."""
    name = 'imessage'
    SCRIPT = ['-e', 'on run argv', '-e', 'tell application "Messages"',
              '-e', 'set svc to 1st account whose service type = iMessage',
              '-e', 'send (item 1 of argv) to participant (item 2 of argv) of svc',
              '-e', 'end tell', '-e', 'end run']

    def __init__(self, to):
        self.to = to

    def send(self, title, body, *, critical=False):
        subprocess.run(['osascript', *self.SCRIPT, f'{title}\n{body}'[:1500], self.to], check=True, timeout=20,
                       capture_output=True)


class FakeNotifier:
    """Development/tests: records messages in memory and, given a path, in a local file. Sends nothing."""
    name = 'fake'

    def __init__(self, path=None):
        self.sent, self.path = [], Path(path) if path else None

    def send(self, title, body, *, critical=False):
        self.sent.append({'title': title, 'body': body, 'critical': critical})
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(self.sent[-1], ensure_ascii=False) + '\n')


def build_notifiers(cfg, *, push, names=None, env=None):
    """Senders (jsonl is separate). Default: macOS banner. Phone push only when ``push`` and env configures it.

    ``names`` (CLI --notifiers) replaces the choice, e.g. ``fake`` for a local dry exercise."""
    env = os.environ if env is None else env
    alert_dir = Path(cfg['paths']['alert_dir'])
    if names is not None:
        chosen = [n.strip() for n in names.split(',') if n.strip()]
    else:
        chosen = [] if env.get('PITCHEEZY_ALERT_MACOS') == '0' else ['macos']
        if push:
            chosen += [n.strip() for n in env.get('PITCHEEZY_ALERT_PUSH', '').split(',') if n.strip()]
    out, problems = [], []
    for name in chosen:
        if name == 'macos':
            out.append(MacNotifier())
        elif name == 'fake':
            out.append(FakeNotifier(alert_dir / 'fake_notify.jsonl'))
        elif name == 'ntfy':
            topic = env.get('PITCHEEZY_ALERT_NTFY_TOPIC')
            token = None
            if env.get('PITCHEEZY_ALERT_NTFY_TOKEN_FILE'):
                try:
                    token = Path(env['PITCHEEZY_ALERT_NTFY_TOKEN_FILE']).expanduser().read_text().strip()
                except OSError:
                    problems.append('ntfy token file unreadable')
            if topic:
                out.append(NtfyNotifier(topic, env.get('PITCHEEZY_ALERT_NTFY_SERVER') or 'https://ntfy.sh', token))
            else:
                problems.append('ntfy chosen but PITCHEEZY_ALERT_NTFY_TOPIC is empty')
        elif name == 'imessage':
            if env.get('PITCHEEZY_ALERT_IMESSAGE_TO'):
                out.append(IMessageNotifier(env['PITCHEEZY_ALERT_IMESSAGE_TO']))
            else:
                problems.append('imessage chosen but PITCHEEZY_ALERT_IMESSAGE_TO is empty')
        else:
            problems.append(f'unknown notifier {name!r}')
    return out, problems


def dispatch(events, now, senders, journal):
    """Log every event, then send one batched message to each sender. A failing sender is logged, not raised."""
    for e in events:
        journal.log(e)
    message = render(events, now)
    if message is None:
        return None
    critical = any(e['kind'] in ('fire', 'renotify') and e['severity'] == CRITICAL for e in events)
    for sender in senders:
        try:
            sender.send(*message, critical=critical)
            journal.log({'kind': 'notified', 'via': sender.name, 'at': _iso(now), 'title': message[0]})
        except Exception as error:  # noqa: BLE001 - a notifier must never stop the watcher
            journal.log({'kind': 'notify_error', 'via': sender.name, 'at': _iso(now),
                         'error': f'{type(error).__name__}: {error}'[:300]})
    return message


# ----------------------------------------------------------------------------- run

def run_once(cfg, *, now=None, senders=(), journal=None, dry_run=False, collect_fn=collect, out=None):
    now = time.time() if now is None else now
    alert_dir = Path(cfg['paths']['alert_dir'])
    state_path = alert_dir / 'alert_state.json'
    saved, _ = _read_json(state_path)
    saved = saved if isinstance(saved, dict) else {}
    alerts, tracker = saved.get('alerts') or {}, saved.get('tracker') or {}
    obs = collect_fn(cfg, now)
    conds, in_game = evaluate(cfg, obs, tracker)
    events = window_notices(tracker, now) + process(cfg, alerts, conds, now, in_game)
    message = render(events, now)
    if dry_run:
        active = [c for c in conds if c['active']]
        print(json.dumps({'at': _iso(now), 'in_game': in_game, 'active': [
            {'key': c['key'], 'severity': c['severity'], 'message': c['message']} for c in active],
            'events': events, 'would_send': message}, ensure_ascii=False, indent=1), file=out or sys.stdout)
        return events
    journal = journal or JsonlNotifier(alert_dir / 'alerts.jsonl')
    dispatch(events, now, senders, journal)
    try:
        alert_dir.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'alerts': alerts, 'tracker': tracker, 'last_run': _iso(now), 'in_game': in_game},
                                  ensure_ascii=False, indent=1))
        os.replace(tmp, state_path)
    except OSError as error:  # sends already happened; the next run may repeat them, which beats silence
        print(f'ws_alert: alert_state.json write failed: {error}', file=sys.stderr)
    return events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true', help='evaluate once')
    mode.add_argument('--loop', action='store_true', help='evaluate every loop_interval_s until stopped')
    mode.add_argument('--test-notify', action='store_true', help='send one test message (local only unless --push)')
    parser.add_argument('--dry-run', action='store_true', help='print what would be sent; write and send nothing')
    parser.add_argument('--push', action='store_true', help='also use the phone push notifiers configured in env')
    parser.add_argument('--notifiers', help='replace the sender list, e.g. "fake" (local file only)')
    parser.add_argument('--config', help=f'config file (default {DEFAULT_CONFIG.name} beside this script)')
    parser.add_argument('--out', help='publisher output folder (else env PITCHEEZY_PUBLISH_OUT_DIR)')
    parser.add_argument('--ops-status', help='operations status file (else env PITCHEEZY_WS_OPS_STATUS)')
    parser.add_argument('--alert-dir', help='watcher state + alerts.jsonl (else env PITCHEEZY_ALERT_DIR)')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, out_dir=args.out, ops_status=args.ops_status, alert_dir=args.alert_dir)
    senders, problems = build_notifiers(cfg, push=args.push or args.loop or args.once, names=args.notifiers)
    for problem in problems:
        print(f'ws_alert: notifier setup: {problem}', file=sys.stderr)
    journal = JsonlNotifier(Path(cfg['paths']['alert_dir']) / 'alerts.jsonl')
    if args.test_notify:
        if not args.push and args.notifiers is None:
            senders = [s for s in senders if s.name in ('macos', 'fake')]
        now = time.time()
        title, body = f'Pitcheezy 알림 시험 {datetime.fromtimestamp(now, KST):%H:%M} KST', '이 알림이 보이면 경로가 동작합니다.'
        if args.dry_run:
            print(json.dumps({'would_send': [title, body], 'via': [s.name for s in senders]}, ensure_ascii=False))
            return 0
        dispatch([{'kind': 'fire', 'key': 'test_notify', 'severity': WARN, 'at': _iso(now), 'message': body}],
                 now, senders, journal)
        print(f'ws_alert: test sent via {[s.name for s in senders] or "jsonl only"}; see alerts.jsonl')
        return 0
    if args.once:
        run_once(cfg, senders=senders, journal=journal, dry_run=args.dry_run)
        return 0
    while True:  # --loop
        started = time.time()
        try:
            run_once(cfg, senders=senders, journal=journal, dry_run=args.dry_run)
        except Exception as error:  # noqa: BLE001 - keep watching; the error is in alerts.jsonl
            print(f'ws_alert: run failed: {type(error).__name__}: {error}', file=sys.stderr)
            if not args.dry_run:
                journal.log({'kind': 'watcher_error', 'at': _iso(time.time()), 'error': f'{type(error).__name__}: {error}'[:300]})
        time.sleep(max(1.0, cfg['loop_interval_s'] - (time.time() - started)))


if __name__ == '__main__':
    sys.exit(main())
