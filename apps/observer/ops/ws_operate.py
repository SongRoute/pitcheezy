"""World Series operations launcher: runs the public publisher for one KST date on the Mac mini (stdlib only).

    ws_operate.py plan   [--days N] [--schedule-file F]          WS games from schedule metadata + each date's timeline
    ws_operate.py start  --date YYYY-MM-DD [--upload] [--dry-run] preflight; then a detached supervisor for that date
    ws_operate.py status [--date D]                               status.json + whether the recorded processes still match
    ws_operate.py stop   --date D [--publisher-only [--term]] [--dry-run]

The supervisor (``supervise``, started by ``start`` in its own session, kept awake by caffeinate) sleeps until the
earliest scheduled first pitch of the date minus ``lead_min``, re-checks the schedule (metadata only), memory, disk,
other publishers and the code commit, then runs ``python -m observer_app.publisher --date D`` from ``code_dir``. An
unexpected exit (or a hung publisher) is restarted at most ``restart.max_restarts`` times after ``restart.backoff_s``,
never onto a different commit (deploy only between games) and never while another publisher runs. It stops
``settle_s`` after every operated game is final, at the hard end, on a stop guard, or when ``stop`` writes STOP.
An independent childwatch stops the publisher if the supervisor dies or the deadline passes.

It never signals a process other than the publisher it started (identity: pid + start time + argv), its caffeinate and
its childwatch. It writes ``<ops_root>/status.json`` (contract: docs/pm/service/ws-ops-status-contract.md).
``--dry-run`` writes, starts and signals nothing. Upload credentials are read from the environment and never printed.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import date as Date, datetime, timedelta, timezone

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE/'ws_ops_config.json'
DEFAULT_CODE_DIR = HERE.parents[2]
KST = timezone(timedelta(hours=9))
GIB = 2**30
STATUS_SCHEMA = 'pitcheezy-ws-ops-status-v1'
STATUS_VERSION = '1.1'  # 1.1 (2026-10-18 review B1): updated_at, publisher.running, problems[], per-game windows, out_dir
# problems[] keys the alert watcher reads (severity is fixed per key)
PROBLEM_SEVERITY = {
    'prestart_refused': 'urgent', 'restart_limit': 'urgent', 'deploy_guard': 'urgent', 'resource_guard': 'urgent',
    'upload_blocked': 'urgent', 'another_publisher': 'urgent', 'supervisor_error': 'urgent', 'start_refused': 'urgent',
    'protected_process_changed': 'warn', 'prestart_retrying': 'warn', 'publisher_restarting': 'warn',
    'memory_low': 'warn', 'health_stale': 'warn', 'feed_failing': 'warn', 'schedule_fallback': 'warn',
    'identity_unreadable': 'warn',
}
ALERT_KEYS = ('memory_low', 'health_stale', 'feed_failing', 'upload_blocked', 'protected_process_changed')
STATUS_KEYS = ('schema', 'schema_version', 'updated_at', 'problems', 'out_dir', 'state_path', 'written_at', 'written_epoch', 'state', 'state_reason', 'kst_date', 'run_dir', 'games', 'timeline',
               'supervisor', 'publisher', 'health', 'game_checks', 'feed_ok', 'upload', 'resources', 'restarts',
               'protected', 'alerts', 'guard')
STATES = ('waiting', 'prestart', 'running', 'backoff', 'stopping', 'stopped', 'refused', 'failed')
NORMAL_ENDS = ('final_settled', 'operator_stop', 'operator_stop_before_start', 'hard_end')
SAFE_ENV = ('HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR', 'SHELL')
ROOT_ENV = 'PITCHEEZY_WS_OPS_ROOT'
TOKEN_ENV = 'PITCHEEZY_PUBLISH_TOKEN'
TOKEN_FILE_ENV = 'PITCHEEZY_WS_OPS_TOKEN_FILE'
BASE_URL_ENV = 'PITCHEEZY_WS_OPS_UPLOAD_BASE_URL'
PROFILE_ENV = 'PITCHEEZY_OBSERVER_PROFILE_SOURCES'
FAKE_ENV = 'PITCHEEZY_WS_OPS_FAKE_PUBLISHER'  # tests only: run this script instead of the real publisher module
PUBLISHER_MODULE = 'observer_app.publisher'
PROTECTED_PORTS = tuple(range(8766, 8773))
DEMO_PORT = 8772
PYTHON_EXE = re.compile(r'(^|/)(python[0-9.]*|Python)$')
PUBLISH_TAIL = re.compile(rb'"publish"\s*:\s*(\{[^{}]*\})\s*\}\s*$')
DATE_RE = re.compile(r'\d{4}-\d{2}-\d{2}')


# ----------------------------------------------------------------------------------------------- small helpers

def now_iso(epoch=None):
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).isoformat(timespec='seconds')


def kst_text(epoch):
    return datetime.fromtimestamp(epoch, KST).strftime('%Y-%m-%d %H:%M KST')


def parse_utc(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def run(argv, timeout=15, cwd=None):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return done.returncode, done.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return None, type(exc).__name__


def write_json_atomic(path, body, mode=None):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(body, indent=1, ensure_ascii=False, sort_keys=True)+'\n')
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_config(path=None):
    cfg = json.loads(Path(path or DEFAULT_CONFIG).read_text())
    cfg['_config_path'] = str(Path(path or DEFAULT_CONFIG).resolve())
    root = os.environ.get(ROOT_ENV) or cfg['ops_root']
    cfg['ops_root'] = str(Path(os.path.expanduser(root)).resolve())
    cfg['code_dir'] = str(Path(os.path.expanduser(cfg.get('code_dir') or DEFAULT_CODE_DIR)).resolve())
    restart = cfg['restart']
    if type(restart['max_restarts']) is not int or not 0 <= restart['max_restarts'] <= 10:
        raise SystemExit('restart.max_restarts must be an integer 0..10')
    if not restart['backoff_s'] or any(not isinstance(x, (int, float)) or x < 0 for x in restart['backoff_s']):
        raise SystemExit('restart.backoff_s must be a non-empty list of seconds >= 0')
    if not (isinstance(cfg['fixed_delay_s'], (int, float)) and 0 <= cfg['fixed_delay_s'] <= 120):
        raise SystemExit('fixed_delay_s must be between 0 and 120 seconds')
    if cfg.get('armb') not in ('auto', '0', '1'):
        raise SystemExit("armb must be 'auto', '0' or '1'")
    return cfg


def ops_paths(cfg, kst_date=None):
    root = Path(cfg['ops_root'])
    paths = {'root': root, 'status': root/'status.json', 'lock': root/'supervisor.lock', 'runs': root/'runs'}
    if kst_date:
        paths['run'] = root/'runs'/kst_date
    return paths


def check_date(text):
    if not DATE_RE.fullmatch(text or ''):
        raise SystemExit('--date must look like YYYY-MM-DD (a KST date)')
    Date.fromisoformat(text)
    return text


# ----------------------------------------------------------------------------------------------- machine facts

def available_bytes():
    """vm_stat free + inactive + speculative pages (the measure the 10/12-13 live trials used)."""
    code, out = run(['/usr/bin/vm_stat'])
    if code != 0:
        return None
    match = re.search(r'page size of (\d+) bytes', out)
    if not match:
        return None
    total = 0
    for name in ('free', 'inactive', 'speculative'):
        row = re.search(rf'Pages {name}:\s+(\d+)\.', out)
        total += int(row[1]) if row else 0
    return total*int(match[1])


def free_disk(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def ps_identity(pid):
    if not pid:
        return None
    code, out = run(['/bin/ps', '-o', 'lstart=,command=', '-p', str(int(pid))])
    if code != 0 or not out.strip():
        return None
    line = out.strip()
    return {'lstart': line[:24].strip(), 'command': line[24:].strip()}


def rss_bytes(pid):
    code, out = run(['/bin/ps', '-o', 'rss=', '-p', str(pid)])
    return int(out.strip())*1024 if code == 0 and out.strip().isdigit() else None


def tree_bytes(path):
    total = 0
    for item in Path(path).rglob('*') if Path(path).is_dir() else ():
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def process_table():
    code, out = run(['/bin/ps', '-axo', 'pid=,ppid=,command='])
    rows = []
    for line in (out or '').splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def publisher_kind(command):
    """'publisher', 'rehearsal_supervisor', 'ops_supervisor' or None. Only a Python interpreter whose argv has
    ``-m observer_app.publisher`` / ``rehearse.py supervise`` / ``ws_operate.py supervise`` counts: a shell or pgrep whose
    command TEXT mentions those words does not (the 849816 rehearsal refusal, 2026-10-18)."""
    tokens = command.split()
    if not tokens or not PYTHON_EXE.search(tokens[0]):
        return None
    for a, b in zip(tokens, tokens[1:]):
        if a == '-m' and b == PUBLISHER_MODULE:
            return 'publisher'
        if (a == 'rehearse.py' or a.endswith('/rehearse.py')) and b == 'supervise':
            return 'rehearsal_supervisor'
        if (a == 'ws_operate.py' or a.endswith('/ws_operate.py')) and b == 'supervise':
            return 'ops_supervisor'
    return None


def other_publishers(table=None, me=None, extra_mine=()):
    """Publishers and supervisors other than this process, its ancestors, its children and ``extra_mine``."""
    table = process_table() if table is None else table
    me = os.getpid() if me is None else me
    parent = {pid: ppid for pid, ppid, _ in table}
    mine, cursor = {me, *extra_mine}, me
    while cursor in parent and parent[cursor] not in mine and parent[cursor] > 1:
        cursor = parent[cursor]
        mine.add(cursor)
    mine |= {pid for pid, ppid, _ in table if ppid == me}
    return [{'pid': pid, 'kind': publisher_kind(command), 'command': command[:200]} for pid, ppid, command in table
            if pid not in mine and publisher_kind(command)]


def protected_processes():
    """Listeners on 8766-8772, the demo loop and the observer worker: recorded, never signalled (rehearsal v3 roles)."""
    found = {}
    code, out = run(['/usr/sbin/lsof', '-nP', '-iTCP', '-sTCP:LISTEN', '-Fpn'])
    pid = None
    for line in (out or '').splitlines():
        if line.startswith('p'):
            pid = int(line[1:])
        elif line.startswith('n') and pid is not None:
            port = line.rsplit(':', 1)[-1]
            if port.isdigit() and int(port) in PROTECTED_PORTS:
                found.setdefault(pid, set()).add(line[1:])
    for pid, _, command in process_table():
        if 'demo_loop.sh' in command or 'observer_app.worker' in command:
            found.setdefault(pid, set()).add('process')
    rows = []
    for pid, what in sorted(found.items()):
        ident = ps_identity(pid)
        demo_child = any(w.endswith(f':{DEMO_PORT}') for w in what) and not any(
            w.endswith(f':{port}') for w in what for port in PROTECTED_PORTS if port != DEMO_PORT)
        rows.append({'pid': pid, 'what': sorted(what), 'lstart': ident and ident['lstart'],
                     'role': 'demo_child_8772' if demo_child else 'pinned'})
    return rows


def protected_check(recorded, current):
    if not recorded:
        return {'ok': True, 'not_measured': 'nothing recorded at start'}
    now = {(r['pid'], r['lstart']) for r in current}
    gone = [r['pid'] for r in recorded if r.get('role') == 'pinned' and (r['pid'], r['lstart']) not in now]
    listeners = sum(1 for r in current if r.get('role') == 'demo_child_8772')
    return {'ok': not gone and listeners <= 1, 'pinned_gone': gone, 'listeners_8772': listeners}


def code_commit(code_dir):
    """(HEAD sha or None, dirty tracked files: bool or None)."""
    code, head = run(['/usr/bin/git', '-C', str(code_dir), 'rev-parse', 'HEAD'])
    if code != 0 or not re.fullmatch(r'[0-9a-f]{40}', (head or '').strip()):
        return None, None
    code, dirty = run(['/usr/bin/git', '-C', str(code_dir), 'status', '--porcelain', '--untracked-files=no'])
    return head.strip(), (None if code != 0 else bool(dirty.strip()))


def lock_held(path):
    """True if another process holds the supervisor lock (checked by trying it, released at once)."""
    if not Path(path).exists():
        return False
    with open(path, 'a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


# ----------------------------------------------------------------------------------------------- schedule

def schedule_rows(data):
    """Rows {game_pk, game_type, start_utc, kst_date, abstract, detailed, label} from a statsapi schedule body or a
    ``pitcheezy-schedule-query-v1`` file."""
    rows = []
    if isinstance(data, dict) and isinstance(data.get('rows'), list):
        for r in data['rows']:
            rows.append({'game_pk': int(r['game_pk']), 'game_type': r.get('game_type'), 'start_utc': r['game_date_utc'],
                         'abstract': r.get('abstract'), 'detailed': r.get('detailed'),
                         'label': f"{r.get('away', '?')} @ {r.get('home', '?')}"})
    else:
        for day in (data or {}).get('dates', []):
            for g in day.get('games', []):
                teams = g.get('teams') or {}
                name = lambda side: ((teams.get(side) or {}).get('team') or {}).get('name', '?')
                rows.append({'game_pk': int(g['gamePk']), 'game_type': g.get('gameType'), 'start_utc': g['gameDate'],
                             'abstract': (g.get('status') or {}).get('abstractGameState'),
                             'detailed': (g.get('status') or {}).get('detailedState'),
                             'label': f"{name('away')} @ {name('home')}"})
    for row in rows:
        row['kst_date'] = datetime.fromtimestamp(parse_utc(row['start_utc']), KST).date().isoformat()
    return sorted(rows, key=lambda r: (r['start_utc'], r['game_pk']))


def fetch_schedule(cfg, first_kst, last_kst):
    """ONE statsapi schedule request (metadata only) covering the US dates of ``first_kst`` .. ``last_kst``."""
    start = (Date.fromisoformat(first_kst)-timedelta(days=1)).isoformat()
    # all game types: the 'other unfinished games on the date' check needs the non-WS rows too
    url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={last_kst}'
    request = urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-ws-ops-schedule-check'})
    with urllib.request.urlopen(request, timeout=cfg['schedule']['timeout_s']) as response:
        return json.load(response)


def load_schedule(cfg, schedule_file, first_kst, last_kst):
    if schedule_file:
        data = read_json(schedule_file)
        if data is None:
            raise OSError('schedule file unreadable')
        return schedule_rows(data), {'source': 'file', 'path': str(schedule_file)}
    return schedule_rows(fetch_schedule(cfg, first_kst, last_kst)), {'source': 'statsapi', 'requests': 1, 'at': now_iso()}


def eligible(rows, cfg, kst_date):
    """Games of ``kst_date`` the launcher operates: configured types, not postponed/cancelled."""
    return [r for r in rows if r['kst_date'] == kst_date and r['game_type'] in cfg['game_types']
            and r.get('detailed') not in ('Postponed', 'Cancelled')]


def timeline(games, cfg):
    starts = [parse_utc(g['start_utc']) for g in games]
    publisher_start = min(starts)-cfg['lead_min']*60
    hard_end = max(starts)+cfg['hard_end_min']*60
    return {'first_pitch_epoch': min(starts), 'publisher_start_epoch': publisher_start, 'hard_end_epoch': hard_end,
            'childwatch_deadline_epoch': hard_end+cfg['childwatch_slack_s'],
            'first_pitch': kst_text(min(starts)), 'publisher_start': kst_text(publisher_start),
            'hard_end': kst_text(hard_end), 'stop_rule': f'every operated game final + {cfg["settle_s"]} s, or the hard end'}


# ----------------------------------------------------------------------------------------------- upload / env

def upload_decision(cfg, requested, env=None):
    """(enabled, problems, presence). Values of credentials never leave this function."""
    env = os.environ if env is None else env
    presence = {'token_env_present': bool(env.get(TOKEN_ENV)), 'token_file_env_present': bool(env.get(TOKEN_FILE_ENV)),
                'base_url_env_present': bool(env.get(BASE_URL_ENV))}
    if not requested:
        return False, [], presence
    problems = []
    if cfg['upload'].get('allowed') is not True:
        problems.append('--upload given but ws_ops_config.json upload.allowed is not true')
    token, token_file = env.get(TOKEN_ENV), env.get(TOKEN_FILE_ENV)
    if token:
        if re.fullmatch(r'[A-Za-z0-9_-]{43,128}', token) is None:
            problems.append(f'{TOKEN_ENV} is present but not 43-128 URL-safe characters (value not shown)')
    elif token_file:
        path = Path(token_file).expanduser()
        if not path.is_file():
            problems.append(f'{TOKEN_FILE_ENV} is present but the file does not exist (path not shown)')
        elif any((parent/'.git').exists() for parent in path.resolve().parents):
            problems.append(f'{TOKEN_FILE_ENV} points inside a git checkout: refused')
        elif path.stat().st_mode & 0o077:
            problems.append(f'{TOKEN_FILE_ENV} file is readable by group/other: chmod 600 it (path not shown)')
    else:
        problems.append(f'--upload needs {TOKEN_ENV} or {TOKEN_FILE_ENV} in the environment')
    base = env.get(BASE_URL_ENV)
    if not base:
        problems.append(f'--upload needs {BASE_URL_ENV} in the environment')
    elif not base.startswith('https://') or base.count('/_publish') != 1:
        problems.append(f'{BASE_URL_ENV} must be https and contain /_publish exactly once (value not shown)')
    return not problems, problems, presence


def runtime_publish_config(cfg, upload, env=None):
    """The repository publish_config with the launcher's delay and upload decision (never written back)."""
    env = os.environ if env is None else env
    body = json.loads((Path(cfg['code_dir'])/cfg['publish_config']).read_text())
    body['fixed_delay_s'] = cfg['fixed_delay_s']
    body.setdefault('upload', {})['enabled'] = bool(upload)
    if upload:
        body['upload']['base_url'] = env[BASE_URL_ENV]
        body['upload']['token_env'] = TOKEN_ENV
        body['upload']['token_file'] = (None if env.get(TOKEN_ENV)
                                        else str(Path(env[TOKEN_FILE_ENV]).expanduser().resolve()))
    else:
        body['upload']['base_url'] = None
    return body


def profile_sources(cfg, env=None):
    env = os.environ if env is None else env
    value = cfg.get('profile_sources') or env.get(PROFILE_ENV)
    return os.path.expanduser(value) if value else None


def child_env(cfg, run_dir, armb, upload, env=None):
    """Minimal environment: locale/home basics plus what the publisher needs. Mirrors the rehearsal's set, except the
    reference card (on for the World Series, Song 2026-10-18) and, with upload, the token variable."""
    env = os.environ if env is None else env
    code_dir, empty = Path(cfg['code_dir']), Path(run_dir)/'empty'
    child = {k: env[k] for k in SAFE_ENV if k in env}
    child |= {
        'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONPATH': str(code_dir/'apps/observer/backend'), 'PYTHONUNBUFFERED': '1',
        'PITCHEEZY_OBSERVER_LIVE_STORE': str(Path(run_dir)/'store.sqlite3'),
        'PITCHEEZY_OBSERVER_LIVE_ARMB': '1' if armb else '0',
        'PITCHEEZY_OBSERVER_LIVE_DELAY_S': '0',
        'PITCHEEZY_OBSERVER_REC_LEDGER': '1', 'PITCHEEZY_OBSERVER_LEGACY': '0',
        'PITCHEEZY_OBSERVER_VIEWER_DELAY': '0', 'PITCHEEZY_OBSERVER_STEP5_EXPECTATION': '0',
        'PITCHEEZY_FEED_DIR': str(empty/'feeds'), 'PITCHEEZY_WATCH_DIR': str(empty/'watch'),
        'PITCHEEZY_INTENT_DIR': str(empty/'intent'), 'PITCHEEZY_DEMO_DIR': str(empty/'demo'),
    }
    sources = profile_sources(cfg, env)
    if sources:
        child[PROFILE_ENV] = sources
    if upload and env.get(TOKEN_ENV):
        child[TOKEN_ENV] = env[TOKEN_ENV]
    return child


def shown_env(child):
    """What may be logged: never the token value."""
    return {k: ('<present>' if k == TOKEN_ENV else v) for k, v in child.items() if k.startswith(('PITCHEEZY_', 'PYTHON'))}


def publisher_argv(cfg, run_dir, kst_date, env=None):
    env = os.environ if env is None else env
    run_dir, empty = Path(run_dir), Path(run_dir)/'empty'
    tail = ['--out', str(run_dir/'out'), '--date', kst_date, '--config', str(run_dir/'publish_config.runtime.json'),
            '--feed-dir', str(empty/'feeds'), '--watch-dir', str(empty/'watch'),
            '--intent-dir', str(empty/'intent'), '--demo-dir', str(empty/'demo')]
    if env.get(FAKE_ENV):
        return [sys.executable, '-B', str(Path(env[FAKE_ENV]).resolve()), *tail]
    return [cfg['python'], '-B', '-m', PUBLISHER_MODULE, *tail]


def is_our_publisher(ident, recorded_lstart, run_dir, env=None):
    """pid identity check: same start time, our run's out folder in argv, and the publisher module (or the test fake)."""
    env = os.environ if env is None else env
    if not ident or ident['lstart'] != (recorded_lstart or '').strip():
        return False
    command = ident['command']
    marker = f'-m {PUBLISHER_MODULE}' in command or (env.get(FAKE_ENV) and str(Path(env[FAKE_ENV]).resolve()) in command)
    return bool(marker) and str(Path(run_dir)/'out') in command


def previous_run(run_dir):
    """Facts of an earlier supervisor of the same date (for the deploy guard and segment numbering)."""
    segs = []
    path = Path(run_dir)/'segments.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                segs.append(json.loads(line))
            except ValueError:
                continue
    starts = [s for s in segs if s.get('phase') == 'start']
    finals = read_json(Path(run_dir)/'finals.json', {}) or {}
    return {'segments': len(starts), 'commit': starts[-1].get('commit') if starts else None, 'finals': finals}


# ----------------------------------------------------------------------------------------------- preflight

TRANSIENT = ('schedule_failed', 'other_games', 'other_publishers', 'mount', 'disk', 'memory')


def preflight(cfg, kst_date, *, upload_requested, schedule_file=None, expect_commit=None, now=None, final=False,
              fallback_games=None):
    """Read-only checks for ``kst_date``. Returns (ok, report, games). ``report['problem_keys']`` names each problem;
    keys in TRANSIENT may clear by themselves (the supervisor retries them). With ``fallback_games`` (the RUN.json
    list) a failed schedule request is not a problem: those games are used and a warning is recorded."""
    now = time.time() if now is None else now
    paths = ops_paths(cfg, kst_date)
    problems, keys, warnings, report = [], [], [], {'kst_date': kst_date}

    def add(key, message):
        problems.append(message)
        keys.append(key)
    try:
        rows, source = load_schedule(cfg, schedule_file, kst_date, kst_date)
        report['schedule'] = source
        games = eligible(rows, cfg, kst_date)
    except Exception as exc:  # noqa: BLE001
        rows, report['schedule'] = [], {'error': type(exc).__name__}
        if fallback_games:
            games = list(fallback_games)
            report['schedule']['fallback'] = 'RUN.json games'
            warnings.append(f'schedule check failed ({type(exc).__name__}): using the game list recorded at start')
        else:
            games = []
            add('schedule_failed', f'schedule check failed: {type(exc).__name__}')
    finals = previous_run(paths['run'])['finals'] if paths['run'].exists() else {}
    open_games = [g for g in games if g.get('abstract') != 'Final' and str(g['game_pk']) not in finals]
    report['games'] = games
    if not games and 'schedule_failed' not in keys:
        add('no_game', f'no {"/".join(cfg["game_types"])} game scheduled on KST {kst_date}')
    elif games and not open_games:
        add('all_final', 'every operated game of this date is already final')
    if games:
        report['timeline'] = timeline(games, cfg)
        if now >= report['timeline']['hard_end_epoch']:
            add('hard_end_passed', 'the hard end of this date has passed')
    others = [r for r in rows if r['kst_date'] == kst_date and r['game_type'] not in cfg['game_types']
              and r.get('abstract') != 'Final' and r['game_type'] in ('R', 'F', 'D', 'L', 'W')]
    if others:
        add('other_games', f'other unfinished games on {kst_date} that the publisher would also poll: '
                           f'{[o["game_pk"] for o in others]}')
    # code / deploy guard
    head, dirty = code_commit(cfg['code_dir'])
    report['code'] = {'dir': cfg['code_dir'], 'head': head, 'dirty': dirty}
    want = expect_commit or cfg.get('expect_commit')
    if head is None:
        add('code', f'code_dir {cfg["code_dir"]} is not a git checkout')
    if cfg.get('require_clean_code', True) and dirty is not False:
        add('code', 'code_dir has tracked changes (or git status failed): refused')
    if want and head != want:
        add('code', f'code_dir HEAD {head} != expected {want}')
    prev = previous_run(paths['run']) if paths['run'].exists() else {'segments': 0, 'commit': None}
    report['previous_run'] = {'segments': prev['segments'], 'commit': prev['commit']}
    if prev['commit'] and head and prev['commit'] != head and open_games:
        add('deploy_guard', f'deploy guard: this date already ran commit {prev["commit"][:12]} and a game is not final; '
                            f'starting {head[:12]} now would change decision IDs mid-game (deploy only between games)')
    # processes and locks
    report['supervisor_lock_held'] = lock_held(paths['lock'])
    if report['supervisor_lock_held']:
        add('supervisor_lock', 'another ws_operate supervisor holds the lock (status / stop it first)')
    report['other_publishers'] = other_publishers()
    if report['other_publishers']:
        add('other_publishers', 'another publisher or supervisor is running: '
                                f'{[(o["pid"], o["kind"]) for o in report["other_publishers"]]}')
    # machine
    if not os.access(cfg['python'], os.X_OK) and not os.environ.get(FAKE_ENV):
        add('python', f'python {cfg["python"]} missing')
    for mount in cfg.get('required_mounts', []):
        if not os.path.ismount(mount):
            add('mount', f'{mount} is not mounted')
    report['available_bytes'] = available_bytes()
    report['free_disk_bytes'] = free_disk(paths['root'])
    guards = cfg['guards']
    if report['free_disk_bytes'] < guards['start_min_free_disk_bytes']:
        add('disk', f'free disk {report["free_disk_bytes"]} < {guards["start_min_free_disk_bytes"]}')
    if final and (report['available_bytes'] or 0) < guards['start_min_available_bytes']:
        add('memory', f'available memory {report["available_bytes"]} < {guards["start_min_available_bytes"]}')
    # upload / reference card
    upload, upload_problems, presence = upload_decision(cfg, upload_requested)
    for message in upload_problems:
        add('upload_config', message)
    report['upload'] = {'requested': upload_requested, 'enabled': upload, **presence}
    report['profile_sources_present'] = bool(profile_sources(cfg))
    if not report['profile_sources_present']:
        warnings.append(f'{PROFILE_ENV} is not set: the reference card will be absent')
    report['fixed_delay_s'] = cfg['fixed_delay_s']
    report['problems'], report['problem_keys'], report['warnings'] = problems, keys, warnings
    return not problems, report, games


def game_rows(games, tl, finals, all_final_at, settle_s, now):
    """status.json ``games[]`` (contract 1.1): start, status, window_start, window_end per operated game."""
    win_start = tl and now_iso(tl['publisher_start_epoch'])
    hard_end = tl and tl['hard_end_epoch']
    rows = []
    for g in games:
        pk, start = str(g['game_pk']), parse_utc(g['start_utc'])
        if pk in finals or g.get('abstract') == 'Final':
            status = 'final'
        elif g.get('detailed') in ('Postponed', 'Cancelled', 'Suspended'):
            status = 'postponed'
        elif now >= start:
            status = 'live'
        else:
            status = 'scheduled'
        end = hard_end
        if hard_end and all_final_at is not None:
            end = min(hard_end, all_final_at+settle_s)
        rows.append({'game_pk': g['game_pk'], 'label': g.get('label'), 'start': now_iso(start),
                     'start_utc': g['start_utc'], 'status': status, 'window_start': win_start,
                     'window_end': end and now_iso(end), 'final_seen_at': finals.get(pk)})
    return rows


def carried_problems(previous, now):
    """Problems still active in an earlier status file (yesterday's refused/failed, an earlier start refusal), carried
    as ``active: false`` so the watcher sees them cleared by this new start (review REVIEW-002 D3)."""
    out = {}
    for item in (previous or {}).get('problems') or []:
        if isinstance(item, dict) and item.get('active') is True and item.get('key') in PROBLEM_SEVERITY:
            out[item['key']] = dict(item, active=False, cleared_at=now_iso(now),
                                    cleared_by=f'new start ({previous.get("kst_date")} -> current)')
    return out


def write_start_refused(cfg, kst_date, report, games, now=None):
    """``start`` refused: make it visible to the alert watcher in the shared status.json (state 'refused', urgent
    'start_refused'), instead of leaving yesterday's file there (REVIEW-002 D1). Not written while a supervisor holds
    the lock (that supervisor owns status.json). Returns True if written."""
    now = time.time() if now is None else now
    paths = ops_paths(cfg, kst_date)
    paths['root'].mkdir(parents=True, exist_ok=True)
    with open(paths['lock'], 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        try:
            message = ('start refused: '+'; '.join(report.get('problems') or ['unknown']))[:500]
            problems = carried_problems(read_json(paths['status']), now)
            problems['start_refused'] = {'key': 'start_refused', 'severity': 'urgent', 'message': message,
                                         'active': True, 'since': now_iso(now)}
            tl = report.get('timeline')
            out = paths['run']/'out'
            body = {
                'schema': STATUS_SCHEMA, 'schema_version': STATUS_VERSION, 'updated_at': now_iso(now),
                'written_at': now_iso(now), 'written_epoch': round(now, 3), 'out_dir': str(out),
                'state_path': f'{out}.state.json',
                'problems': sorted(problems.values(), key=lambda p: (not p['active'], p['key'])),
                'state': 'refused', 'state_reason': message, 'kst_date': kst_date, 'run_dir': str(paths['run']),
                'games': game_rows(games, tl, {}, None, cfg['settle_s'], now),
                'timeline': tl and {k: v for k, v in tl.items() if not k.endswith('_epoch')} | {
                    'publisher_start_epoch': tl['publisher_start_epoch'], 'hard_end_epoch': tl['hard_end_epoch']},
                'supervisor': None, 'publisher': {'running': False, 'segment': 0, 'commit': None, 'last_exit': None},
                'health': None, 'game_checks': {}, 'feed_ok': None,
                'upload': {'enabled': bool((report.get('upload') or {}).get('enabled'))}, 'resources': {},
                'restarts': {'count': 0, 'max': cfg['restart']['max_restarts'], 'backoff_s': cfg['restart']['backoff_s'],
                             'last_exit': None, 'next_restart_at': None, 'history': []},
                'protected': {}, 'alerts': [], 'guard': None, 'refused_by': 'start',
            }
            write_json_atomic(paths['status'], body)
            return True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------------------------- supervisor

class Supervisor:
    def __init__(self, cfg, kst_date, *, upload=False, schedule_file=None, expect_commit=None, clock=time.time,
                 sleep=time.sleep):
        self.cfg, self.date, self.upload = cfg, kst_date, upload
        self.schedule_file, self.expect_commit = schedule_file, expect_commit
        self.clock, self.sleep = clock, sleep
        self.paths = ops_paths(cfg, kst_date)
        self.run_dir = self.paths['run']
        self.out = self.run_dir/'out'
        self.state, self.state_reason = None, None  # set_state('waiting') records the first transition
        self.games, self.timeline = [], None
        self.proc, self.ident, self.segment, self.segment_started = None, None, 0, None
        self.commit, self.commit_dirty, self.armb = None, None, None
        self.restarts, self.restart_history, self.next_restart_at, self.last_exit = 0, [], None, None
        self.health, self.health_stat, self.health_seen_at, self.health_seq = None, None, None, None
        self.finals = {}
        self.all_final_at = None
        self.upload_view = {'enabled': upload}
        self.resources = {}
        self.streak = {'rss': 0}
        self.protected, self.protected_now = [], {'ok': True, 'not_measured': 'not started'}
        self.alerts, self.guard = [], None
        self.last_sample, self.last_status, self.last_walk, self.last_upload_read = 0.0, 0.0, 0.0, 0.0
        self.run_bytes = 0
        self.childwatch_pid = None
        self.started_at = self.clock()
        self.me = None
        self.problems = {}
        self.run_games = []

    # -- records
    def log(self, name, row):
        with open(self.run_dir/name, 'a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')

    def event(self, name, **more):
        self.log('events.jsonl', {'at': now_iso(self.clock()), 'event': name, **more})

    def set_problem(self, key, active, message=None):
        """Keep one entry per key: active while the condition holds, then active false (so the watcher can clear)."""
        now = now_iso(self.clock())
        current = self.problems.get(key)
        if active:
            since = current['since'] if current and current['active'] else now
            self.problems[key] = {'key': key, 'severity': PROBLEM_SEVERITY[key], 'message': (message or key)[:500],
                                  'active': True, 'since': since}
            if not (current and current['active']):
                self.event('problem', key=key, message=(message or key)[:500])
        elif current and current['active']:
            current |= {'active': False, 'cleared_at': now}
            self.event('problem_cleared', key=key)

    def game_rows(self, now):
        return game_rows(self.games, self.timeline, self.finals, self.all_final_at, self.cfg['settle_s'], now)

    def set_state(self, state, reason=None):
        assert state in STATES, state
        if (state, reason) != (self.state, self.state_reason):
            self.state, self.state_reason = state, reason
            self.event('state', state=state, reason=reason)
        self.write_status()

    def status_body(self):
        now = self.clock()
        if self.me is None:
            self.me = ps_identity(os.getpid()) or {}
        me = self.me
        health = self.health or {}
        checks = health.get('game_checks') or {}
        game_checks = {}
        for pk in sorted({str(g['game_pk']) for g in self.games} | set(checks)):
            row = checks.get(pk) or {}
            game_checks[pk] = {'checked_at': row.get('checked_at'), 'feed_ok': row.get('feed_ok'),
                               'file_sequence': row.get('file_sequence'), 'sequence_checked': row.get('sequence_checked'),
                               'final': pk in self.finals}
        failing = health.get('feed_failing_games') or []
        feed_ok = None if not checks and not failing else (not failing and all(
            game_checks[pk]['feed_ok'] is not False for pk in game_checks if pk in checks))
        unavailable = health.get('unavailable_games') or []
        return {
            'schema': STATUS_SCHEMA, 'schema_version': STATUS_VERSION, 'updated_at': now_iso(now),
            'written_at': now_iso(now), 'written_epoch': round(now, 3),
            'out_dir': str(self.out), 'state_path': f'{self.out}.state.json',
            'problems': sorted(self.problems.values(), key=lambda p: (not p['active'], p['key'])),
            'state': self.state, 'state_reason': self.state_reason, 'kst_date': self.date, 'run_dir': str(self.run_dir),
            'games': self.game_rows(now),
            'timeline': self.timeline and {k: v for k, v in self.timeline.items() if not k.endswith('_epoch')} | {
                'publisher_start_epoch': self.timeline['publisher_start_epoch'],
                'hard_end_epoch': self.timeline['hard_end_epoch'],
                'all_final_at': self.all_final_at and now_iso(self.all_final_at),
                'settle_until': self.all_final_at and now_iso(self.all_final_at+self.cfg['settle_s'])},
            'supervisor': {'pid': os.getpid(), 'lstart': me and me['lstart'], 'started_at': now_iso(self.started_at)},
            'publisher': {'running': False, 'segment': self.segment, 'commit': self.commit,
                          'last_exit': self.last_exit} if self.proc is None else {
                'running': True, 'pid': self.proc.pid, 'lstart': self.ident and self.ident['lstart'], 'segment': self.segment,
                'started_at': now_iso(self.segment_started), 'commit': self.commit, 'commit_dirty': self.commit_dirty,
                'armb': self.armb, 'fixed_delay_s': self.cfg['fixed_delay_s'], 'upload_enabled': self.upload,
                'childwatch_pid': self.childwatch_pid, 'fake': bool(os.environ.get(FAKE_ENV))},
            'health': None if self.health is None else {
                'sequence': health.get('sequence'), 'published_at': health.get('published_at'),
                'observed_at': now_iso(self.health_seen_at), 'age_s': round(now-self.health_seen_at, 1),
                'files_written': health.get('files_written'), 'feed_failing_games': failing,
                'unavailable_count': len(unavailable),
                'unavailable_codes': sorted({str(u.get('code')) for u in unavailable if isinstance(u, dict)})},
            'game_checks': game_checks, 'feed_ok': feed_ok,
            'upload': self.upload_view,
            'resources': self.resources,
            'restarts': {'count': self.restarts, 'max': self.cfg['restart']['max_restarts'],
                         'backoff_s': self.cfg['restart']['backoff_s'], 'last_exit': self.last_exit,
                         'next_restart_at': self.next_restart_at and now_iso(self.next_restart_at),
                         'history': self.restart_history[-10:]},
            'protected': self.protected_now, 'alerts': self.alerts, 'guard': self.guard,
        }

    def write_status(self):
        self.last_status = self.clock()
        body = self.status_body()
        write_json_atomic(self.run_dir/'status.json', body)
        write_json_atomic(self.paths['status'], body)

    # -- publisher segments
    def runtime_config(self):
        path = self.run_dir/'publish_config.runtime.json'
        write_json_atomic(path, runtime_publish_config(self.cfg, self.upload), mode=0o600)
        return path

    def start_segment(self, kind):
        others = other_publishers(extra_mine=[p for p in (self.childwatch_pid,) if p])
        if others:
            raise RuntimeError(f'another publisher appeared: {[(o["pid"], o["kind"]) for o in others]}')
        self.segment += 1
        self.runtime_config()
        env = child_env(self.cfg, self.run_dir, self.armb, self.upload)
        argv = publisher_argv(self.cfg, self.run_dir, self.date)
        with open(self.run_dir/'publisher.log', 'a') as logf:
            logf.write(f'\n===== segment {self.segment} ({kind}) {now_iso(self.clock())} =====\n')
            logf.flush()
            self.proc = subprocess.Popen(argv, cwd=Path(self.cfg['code_dir'])/'apps/observer/backend', env=env,
                                         stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.segment_started = self.clock()
        self.ident = ps_identity(self.proc.pid)
        if self.ident is None:  # childwatch could not match it: say so instead of running without a watchdog silently
            self.event('identity_unreadable', pid=self.proc.pid)
            self.set_problem('identity_unreadable', True, f'ps identity of publisher pid {self.proc.pid} unreadable: '
                                                          'no childwatch for this segment')
        else:
            self.set_problem('identity_unreadable', False)
        self.childwatch_pid = (self.spawn_childwatch() if self.cfg.get('childwatch', True) and self.ident else None)
        row = {'segment': self.segment, 'kind': kind, 'pid': self.proc.pid, 'identity': self.ident, 'argv': argv,
               'commit': self.commit, 'commit_dirty': self.commit_dirty, 'armb': self.armb, 'upload': self.upload,
               'childwatch_pid': self.childwatch_pid, 'env': shown_env(env), 'started_at': now_iso(self.segment_started)}
        self.log('segments.jsonl', row | {'phase': 'start'})
        write_json_atomic(self.run_dir/'publisher.identity.json', row)
        self.event('segment_start', segment=self.segment, kind=kind, pid=self.proc.pid)

    def spawn_childwatch(self):
        me = ps_identity(os.getpid())
        argv = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), 'childwatch', '--pid', str(self.proc.pid),
                '--lstart', (self.ident or {}).get('lstart') or '', '--run-dir', str(self.run_dir),
                '--supervisor', str(os.getpid()), '--supervisor-lstart', (me or {}).get('lstart') or '',
                '--deadline', str(self.timeline['childwatch_deadline_epoch']),
                '--graceful-s', str(self.cfg['stop']['graceful_s']), '--term-s', str(self.cfg['stop']['term_s']),
                '--every-s', str(self.cfg.get('childwatch_s', 5))]
        watch = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        return watch.pid

    def stop_segment(self, why):
        if self.proc is None:
            return None
        proc, method = self.proc, None
        if proc.poll() is None:  # our unreaped child: the pid is still this process
            self.set_state('stopping', why)
            for sig, wait in ((signal.SIGINT, self.cfg['stop']['graceful_s']), (signal.SIGTERM, self.cfg['stop']['term_s']),
                              (signal.SIGKILL, 10)):
                proc.send_signal(sig)
                method = sig.name
                try:
                    proc.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        code = proc.poll()
        self.last_exit = {'segment': self.segment, 'exit_code': code, 'why': why, 'method': method,
                          'at': now_iso(self.clock())}
        self.log('segments.jsonl', {'segment': self.segment, 'phase': 'stop', 'why': why, 'method': method,
                                    'exit_code': code, 'stopped_at': now_iso(self.clock())})
        self.event('segment_stop', segment=self.segment, why=why, method=method, exit_code=code)
        self.proc, self.ident = None, None
        return code

    # -- what the publisher wrote
    def observe(self):
        health = self.out/'v1/health.json'
        try:
            stat = health.stat()
        except FileNotFoundError:
            return
        key = (stat.st_mtime_ns, stat.st_size)
        if key == self.health_stat:
            return
        try:
            body = json.loads(health.read_bytes())
        except (OSError, ValueError):
            return  # mid-replace: read again next tick
        self.health_stat = key
        seq = body.get('sequence') if isinstance(body, dict) else None
        if type(seq) is not int or seq == self.health_seq:
            return
        self.health, self.health_seq, self.health_seen_at = body, seq, self.clock()
        if self.upload and isinstance(body.get('upload'), dict):
            self.upload_view |= {'pending': body['upload'].get('pending'), 'code': body['upload'].get('code')}
        for game in self.games:
            pk = str(game['game_pk'])
            if pk not in self.finals and self.game_final(pk):
                self.finals[pk] = now_iso(self.clock())
                write_json_atomic(self.run_dir/'finals.json', self.finals)
                self.event('final_seen', game_pk=game['game_pk'])
        if self.games and self.all_final_at is None and all(str(g['game_pk']) in self.finals for g in self.games):
            self.all_final_at = self.clock()
            self.event('all_final')

    def game_final(self, pk):
        """The publish block (last key of the file) says final: read only the file's tail."""
        path = self.out/f'v1/game/{pk}.json'
        try:
            with open(path, 'rb') as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell()-1024))
                tail = handle.read()
        except OSError:
            return False
        match = PUBLISH_TAIL.search(tail)
        try:
            return bool(match and json.loads(match[1]).get('final') is True)
        except ValueError:
            return False

    def read_upload_state(self):
        """Upload bookkeeping from the publisher's state file: counts and reason classes only (no paths, no host)."""
        state = read_json(Path(f'{self.out}.state.json'), {}) or {}
        attempts = state.get('upload_attempts') or {}
        phases, reasons = {}, {}
        for attempt in attempts.values():
            if isinstance(attempt, dict):
                phases[str(attempt.get('phase'))] = phases.get(str(attempt.get('phase')), 0)+1
                if attempt.get('phase') in ('blocked', 'rejected') and attempt.get('reason'):
                    reasons[str(attempt['reason'])] = reasons.get(str(attempt['reason']), 0)+1
        last = self.upload_view.get('code') or (max(reasons, key=reasons.get) if reasons else None)
        self.upload_view |= {'attempts_by_phase': phases, 'blocked': phases.get('blocked', 0),
                             'error_classes': reasons, 'last_error_class': last,
                             'pending_in_state': len(state.get('upload_pending') or {}),
                             'recovery_blocked': bool(state.get('upload_recovery_blocked'))}

    def sample(self):
        """Resources and guards. Returns a stop reason, 'health_stalled', or None."""
        now, guards = self.clock(), self.cfg['guards']
        rss = rss_bytes(self.proc.pid) if self.proc is not None and self.proc.poll() is None else None
        avail, disk = available_bytes(), free_disk(self.run_dir)
        if now-self.last_walk >= 60:
            self.last_walk = now
            self.run_bytes = tree_bytes(self.run_dir)
            self.protected_now = protected_check(self.protected, protected_processes())
        if self.upload and now-self.last_upload_read >= self.cfg.get('upload_read_s', 30):
            self.last_upload_read = now
            self.read_upload_state()
        self.resources = {'publisher_rss_bytes': rss, 'available_bytes': avail, 'free_disk_bytes': disk,
                          'run_dir_bytes': self.run_bytes, 'sampled_at': now_iso(now)}
        self.log('resources.jsonl', self.resources | {'segment': self.segment})
        alerts = []
        if avail is not None and avail < guards['warn_available_bytes']:
            alerts.append('memory_low')
        if self.health_seen_at and now-self.health_seen_at > guards['health_stale_warn_s'] and self.proc is not None:
            alerts.append('health_stale')
        if (self.health or {}).get('feed_failing_games'):
            alerts.append('feed_failing')
        if self.upload_view.get('blocked') or self.upload_view.get('recovery_blocked'):
            alerts.append('upload_blocked')
        if not self.protected_now.get('ok', True):
            alerts.append('protected_process_changed')
        self.alerts = alerts
        messages = {'memory_low': f'available memory {avail} < {guards["warn_available_bytes"]}',
                    'health_stale': f'health.json not advanced for > {guards["health_stale_warn_s"]} s',
                    'feed_failing': f'feed failing: {(self.health or {}).get("feed_failing_games")}',
                    'upload_blocked': f'upload blocked: {self.upload_view.get("error_classes")}',
                    'protected_process_changed': f'protected processes changed: {self.protected_now}'}
        for key in ALERT_KEYS:
            self.set_problem(key, key in alerts, messages[key])
        self.streak['rss'] = self.streak['rss']+1 if rss is not None and rss > guards['stop_rss_bytes'] else 0
        if self.streak['rss'] >= guards['stop_rss_samples']:
            return 'guard:publisher_rss_over_limit'
        if disk < guards['stop_free_disk_bytes']:
            return 'guard:disk_low'
        if self.run_bytes > guards['stop_run_dir_bytes']:
            return 'guard:run_dir_over_limit'
        restart = self.cfg['restart']
        if self.proc is not None and now-self.segment_started > restart['startup_grace_s']:
            last = max(self.health_seen_at or 0, self.segment_started)
            if now-last > restart['health_stall_s']:
                return 'health_stalled'
        return None

    # -- deploy guard
    def deploy_check(self):
        head, dirty = code_commit(self.cfg['code_dir'])
        if head != self.commit:
            return False, f'code_dir HEAD is now {head} but this date started on {self.commit}'
        if self.cfg.get('require_clean_code', True) and dirty is not False:
            return False, 'code_dir has tracked changes now'
        return True, None

    # -- flow
    def stop_requested(self):
        return (self.run_dir/'STOP').exists()

    def wait(self, seconds, *, observe=True):
        """Sleep ``seconds`` in ticks, observing and heartbeating. False if STOP appeared."""
        end = self.clock()+seconds
        while self.clock() < end:
            if self.stop_requested():
                return False
            if observe:
                self.observe()
            every = self.cfg['status_every_s'] if self.state != 'waiting' else self.cfg['waiting_status_every_s']
            if self.clock()-self.last_status >= every:
                self.write_status()
            self.sleep(max(0.0, min(self.cfg['tick_s'], end-self.clock())))
        return True

    def prestart(self):
        self.set_state('prestart')
        ok, report, games = preflight(self.cfg, self.date, upload_requested=self.upload, schedule_file=self.schedule_file,
                                      expect_commit=self.expect_commit, now=self.clock(), final=True,
                                      fallback_games=self.run_games)
        pairs = [(k, p) for k, p in zip(report['problem_keys'], report['problems']) if k != 'supervisor_lock']  # ours
        report['problem_keys'], report['problems'] = [k for k, _ in pairs], [p for _, p in pairs]
        fallback = report['schedule'].get('fallback')
        self.set_problem('schedule_fallback', bool(fallback),
                         f'schedule request failed ({report["schedule"].get("error")}): using the RUN.json game list')
        self.games = [g for g in games if g.get('abstract') != 'Final' and str(g['game_pk']) not in self.finals] or games
        if games:
            self.timeline = timeline(games, self.cfg)
        avail = report.get('available_bytes') or 0
        mode = self.cfg['armb']
        self.armb = avail >= self.cfg['armb_min_available_bytes'] if mode == 'auto' else mode == '1'
        if mode == '1' and avail < self.cfg['armb_min_available_bytes']:
            report['problems'].append(f'armb forced on but available memory {avail} is below the floor')
            report['problem_keys'].append('memory')
        self.commit, self.commit_dirty = report['code']['head'], report['code']['dirty']
        self.protected = protected_processes()
        self.protected_now = protected_check(self.protected, self.protected)
        report['armb'] = self.armb
        self.prestart_attempt = getattr(self, 'prestart_attempt', 0)+1
        write_json_atomic(self.run_dir/f'PRESTART-{self.segment+1:03d}-{self.prestart_attempt:02d}.json', report)
        return report['problems'], report['problem_keys']

    def prestart_with_retries(self):
        """prestart until it passes. Transient problems (schedule request, memory, disk, mount, another publisher, other
        games) are retried every prestart_retry_s until first pitch (or prestart_late_window_s when started late);
        any other problem refuses at once. Returns (problems or [], stop_requested)."""
        retry_s, began, deadline = self.cfg.get('prestart_retry_s', 60), self.clock(), None
        while True:
            problems, keys = self.prestart()
            if deadline is None:  # after the first attempt the timeline is known (RUN.json or the schedule)
                first = self.timeline['first_pitch_epoch'] if self.timeline else began
                deadline = first if first > began else began+self.cfg.get('prestart_late_window_s', 600)
                if self.timeline:
                    deadline = min(deadline, self.timeline['hard_end_epoch'])
            if not problems:
                self.set_problem('prestart_retrying', False)
                return [], False
            permanent = [k for k in keys if k not in TRANSIENT]
            if permanent or self.clock()+retry_s > deadline:
                self.set_problem('prestart_retrying', False)
                return problems, False
            self.set_problem('prestart_retrying', True, f'attempt {self.prestart_attempt}: ' + '; '.join(problems))
            self.set_state('prestart', 'retrying')
            if not self.wait(retry_s, observe=False):
                return problems, True

    def handle_exit(self, why):
        """After the publisher ended unexpectedly: restart within the policy, or a reason to end."""
        restart = self.cfg['restart']
        if self.all_final_at is not None:  # every game already final: nothing left to publish, end normally
            return 'final_settled'
        if self.restarts >= restart['max_restarts']:
            return f'restart_limit:{why}'
        delay = restart['backoff_s'][min(self.restarts, len(restart['backoff_s'])-1)]
        delay = max(0.0, min(delay, self.timeline['hard_end_epoch']-self.clock()))
        self.next_restart_at = self.clock()+delay
        self.set_problem('publisher_restarting', True, f'{why}; restart {self.restarts+1} in {delay:.0f} s')
        self.set_state('backoff', why)
        if not self.wait(delay):
            self.next_restart_at = None
            return 'operator_stop'
        self.next_restart_at = None
        if self.clock() >= self.timeline['hard_end_epoch']:
            return 'hard_end'
        if self.all_final_at is not None:
            return 'final_settled'
        ok, detail = self.deploy_check()
        if not ok:
            self.event('deploy_guard_refused', detail=detail)
            return f'deploy_guard:{detail}'
        others = other_publishers(extra_mine=[p for p in (self.childwatch_pid,) if p])
        if others:
            return f'another_publisher:{[(o["pid"], o["kind"]) for o in others]}'
        self.restarts += 1
        self.restart_history.append({'n': self.restarts, 'why': why, 'backoff_s': delay, 'at': now_iso(self.clock())})
        self.start_segment('restart')
        self.set_problem('publisher_restarting', False)
        self.set_state('running')
        return None

    def loop(self):
        while True:
            now = self.clock()
            if self.stop_requested():
                return 'operator_stop'
            if now >= self.timeline['hard_end_epoch']:
                return 'hard_end'
            if self.all_final_at is not None and now >= self.all_final_at+self.cfg['settle_s']:
                return 'final_settled'
            self.observe()
            if self.proc is not None and self.proc.poll() is not None:
                code = self.proc.returncode
                self.stop_segment(f'publisher_exited:{code}')
                reason = self.handle_exit(f'publisher_exited:{code}')
                if reason:
                    return reason
                continue
            if now-self.last_sample >= self.cfg['sample_s']:
                self.last_sample = now
                verdict = self.sample()
                if verdict == 'health_stalled':
                    self.stop_segment('health_stalled')
                    reason = self.handle_exit('health_stalled')
                    if reason:
                        return reason
                    continue
                if verdict:
                    self.guard = verdict
                    return verdict
            if now-self.last_status >= self.cfg['status_every_s']:
                self.write_status()
            self.sleep(self.cfg['tick_s'])

    def main(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        lock = open(self.paths['lock'], 'a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.event('refused', reason='supervisor lock held by another process')
            self.state, self.state_reason = 'refused', 'supervisor_lock_held'
            write_json_atomic(self.run_dir/'status.json', self.status_body())  # never the shared status: not ours
            return 'supervisor_lock_held'
        prev = previous_run(self.run_dir)
        self.segment, self.finals = prev['segments'], dict(prev['finals'])
        caffeinate = None
        if self.cfg.get('caffeinate', True):
            caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())], stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.event('supervisor_start', pid=os.getpid(), caffeinate_pid=caffeinate and caffeinate.pid, upload=self.upload)
        reason, final_state = None, 'stopped'
        try:
            run = read_json(self.run_dir/'RUN.json', {}) or {}
            self.games = run.get('games') or []
            self.run_games = list(self.games)
            if self.games:
                self.timeline = timeline(self.games, self.cfg)
            previous = read_json(self.paths['status'])
            if previous:  # we hold the lock: whoever wrote it is gone; its active problems end with this start
                self.problems = carried_problems(previous, self.clock()) | self.problems
                if self.problems:
                    self.event('problems_carried_inactive', keys=sorted(self.problems))
            self.set_state('waiting')
            start_at = self.timeline['publisher_start_epoch'] if self.timeline else self.clock()
            if not self.wait(max(0.0, start_at-self.clock()), observe=False):
                reason = 'operator_stop_before_start'
            else:
                problems, stopped = self.prestart_with_retries()
                if stopped:
                    reason = 'operator_stop_before_start'
                elif problems:
                    reason, final_state = 'prestart_refused', 'refused'
                    self.event('prestart_refused', problems=problems)
                    self.set_problem('prestart_refused', True, '; '.join(problems))
                else:
                    self.start_segment('initial')
                    self.set_state('running')
                    reason = self.loop()
        except Exception as exc:  # noqa: BLE001
            reason = f'supervisor_error:{type(exc).__name__}:{exc}'
        finally:
            self.stop_segment(reason or 'end')
            self.observe()
            if self.upload:
                self.read_upload_state()
            if final_state != 'refused' and reason not in NORMAL_ENDS:
                final_state = 'failed'
            for prefix, key in (('restart_limit', 'restart_limit'), ('deploy_guard', 'deploy_guard'),
                                ('guard:', 'resource_guard'), ('supervisor_error', 'supervisor_error'),
                                ('another_publisher', 'another_publisher')):
                if reason and reason.startswith(prefix):
                    self.set_problem(key, True, reason)
            for key in ('publisher_restarting', 'health_stale', 'feed_failing', 'memory_low'):
                self.set_problem(key, False)  # the publisher is gone: these no longer describe anything
            end = {'at': now_iso(self.clock()), 'reason': reason, 'state': final_state, 'kst_date': self.date,
                   'segments': self.segment, 'restarts': self.restarts, 'finals': self.finals, 'commit': self.commit}
            write_json_atomic(self.run_dir/f'END-{self.segment:03d}.json', end)
            write_json_atomic(self.run_dir/'END.json', end)
            self.set_state(final_state, reason)
            self.event('supervisor_end', reason=reason)
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        return reason


# ----------------------------------------------------------------------------------------------- commands

def cmd_plan(args, cfg):
    today = datetime.now(KST).date()
    last = today+timedelta(days=args.days)
    rows, source = load_schedule(cfg, args.schedule_file, today.isoformat(), last.isoformat())
    by_date = {}
    for row in rows:
        if row['game_type'] in cfg['game_types'] and today.isoformat() <= row['kst_date'] <= last.isoformat():
            by_date.setdefault(row['kst_date'], []).append(row)
    out = {'today_kst': today.isoformat(), 'schedule': source, 'fixed_delay_s': cfg['fixed_delay_s'],
           'dates': [{'kst_date': d, 'games': [{k: g[k] for k in ('game_pk', 'label', 'start_utc', 'detailed')} for g in gs],
                      'timeline': {k: v for k, v in timeline(eligible(gs, cfg, d) or gs, cfg).items()
                                   if not k.endswith('_epoch')},
                      'command': f'ws_operate.py start --date {d}'} for d, gs in sorted(by_date.items())]}
    if not by_date:
        out['note'] = f'no {"/".join(cfg["game_types"])} game in the schedule for {today} .. {last}'
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


_sleep = time.sleep  # tests replace it


def spawn_supervisor(argv, run_dir):
    with open(Path(run_dir)/'supervisor.log', 'a') as logf:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, cwd=run_dir,
                                start_new_session=True)
    return proc.pid


def cmd_start(args, cfg):
    kst_date = check_date(args.date)
    ok, report, games = preflight(cfg, kst_date, upload_requested=args.upload, schedule_file=args.schedule_file,
                                  expect_commit=args.expect_commit)
    paths = ops_paths(cfg, kst_date)
    run_dir = paths['run']
    upload = report['upload']['enabled']
    armb_preview = cfg['armb'] if cfg['armb'] != 'auto' else (
        'auto: on' if (report['available_bytes'] or 0) >= cfg['armb_min_available_bytes'] else 'auto: off (memory now)')
    plan = {'kst_date': kst_date, 'run_dir': str(run_dir), 'status_file': str(paths['status']),
            'timeline': report.get('timeline') and {k: v for k, v in report['timeline'].items() if not k.endswith('_epoch')},
            'games': games, 'armb': armb_preview, 'restart': cfg['restart'], 'fixed_delay_s': cfg['fixed_delay_s'],
            'publisher_argv': publisher_argv(cfg, run_dir, kst_date),
            'publisher_cwd': str(Path(cfg['code_dir'])/'apps/observer/backend'),
            'publisher_env': shown_env(child_env(cfg, run_dir, True, upload)), 'preflight': report}
    if args.dry_run:
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        print(f'--dry-run: preflight {"PASS" if ok else "REFUSED"}; nothing was written, started or signalled.')
        return 0 if ok else 2
    retry_s, began, attempt = cfg.get('start_retry_s', 60), time.time(), 1
    while not ok:  # transient problems (schedule request, disk, mount, another publisher, other games): retry
        keys = report.get('problem_keys') or []
        deadline = began+cfg.get('start_retry_window_s', 1800)
        if report.get('timeline'):
            deadline = min(deadline, report['timeline']['publisher_start_epoch'])
        if not keys or any(k not in TRANSIENT for k in keys) or time.time()+retry_s > deadline:
            break
        print(f'start attempt {attempt}: transient problems {keys}; retrying in {retry_s} s', flush=True)
        _sleep(retry_s)
        attempt += 1
        ok, report, games = preflight(cfg, kst_date, upload_requested=args.upload, schedule_file=args.schedule_file,
                                      expect_commit=args.expect_commit)
    if not ok:
        print(json.dumps(report, indent=1, ensure_ascii=False))
        print('REFUSED: '+'; '.join(report['problems']))
        wrote = write_start_refused(cfg, kst_date, report, games)
        print(f'status.json: {"refused written" if wrote else "not written (a supervisor holds the lock and owns it)"}')
        return 2
    for sub in ('empty/feeds', 'empty/watch', 'empty/intent', 'empty/demo'):
        (run_dir/sub).mkdir(parents=True, exist_ok=True)
    prior = sorted(run_dir.glob('RUN-*.json'))
    run_body = plan | {'games': games, 'config_path': cfg['_config_path'], 'started_by_pid': os.getpid(),
                       'written_at': now_iso()}
    write_json_atomic(run_dir/f'RUN-{len(prior)+1:03d}.json', run_body)
    write_json_atomic(run_dir/'RUN.json', run_body)
    (run_dir/'STOP').unlink(missing_ok=True)  # an earlier stop of this date must not stop the new supervisor
    argv = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), 'supervise', '--date', kst_date,
            '--config', cfg['_config_path']]
    if args.upload:
        argv.append('--upload')
    if args.schedule_file:
        argv += ['--schedule-file', str(Path(args.schedule_file).resolve())]
    if args.expect_commit:
        argv += ['--expect-commit', args.expect_commit]
    pid = spawn_supervisor(argv, run_dir)
    ident = {'pid': pid, 'identity': ps_identity(pid), 'started_at': now_iso(), 'argv': argv}
    write_json_atomic(run_dir/'supervisor.identity.json', ident)
    print(json.dumps({'run_dir': str(run_dir), 'supervisor': ident, 'timeline': plan['timeline'],
                      'upload': report['upload'], 'warnings': report['warnings']}, indent=1, ensure_ascii=False))
    return 0


def cmd_supervise(args, cfg):
    sup = Supervisor(cfg, check_date(args.date), upload=args.upload, schedule_file=args.schedule_file,
                     expect_commit=args.expect_commit)
    reason = sup.main()
    return 0 if reason in NORMAL_ENDS else 1


def is_our_supervisor(ident, recorded, run_dir):
    return bool(ident and recorded and ident['lstart'] == ((recorded.get('identity') or {}).get('lstart') or '').strip()
                and publisher_kind(ident['command']) == 'ops_supervisor' and f'--date {Path(run_dir).name}' in ident['command'])


def cmd_status(args, cfg):
    paths = ops_paths(cfg, args.date)
    path = paths['run']/'status.json' if args.date else paths['status']
    body = read_json(path)
    if body is None:
        print(json.dumps({'status_file': str(path), 'present': False}, indent=1))
        return 1
    run_dir = Path(body['run_dir'])
    sup = read_json(run_dir/'supervisor.identity.json') or {}
    pub = read_json(run_dir/'publisher.identity.json') or {}
    live = {'supervisor_alive': is_our_supervisor(ps_identity(sup.get('pid')), sup, run_dir),
            'publisher_alive': is_our_publisher(ps_identity(pub.get('pid')), (pub.get('identity') or {}).get('lstart'), run_dir),
            'status_age_s': round(time.time()-body.get('written_epoch', 0), 1)}
    print(json.dumps({'status_file': str(path), 'live_check': live, 'status': body}, indent=1, ensure_ascii=False))
    return 0


def cmd_stop(args, cfg):
    kst_date = check_date(args.date)
    run_dir = ops_paths(cfg, kst_date)['run']
    if not (run_dir/'RUN.json').exists():
        raise SystemExit(f'{run_dir} is not a ws_operate run folder')
    sup = read_json(run_dir/'supervisor.identity.json') or {}
    pub = read_json(run_dir/'publisher.identity.json') or {}
    sup_alive = is_our_supervisor(ps_identity(sup.get('pid')), sup, run_dir)
    pub_ident = ps_identity(pub.get('pid'))
    pub_ours = is_our_publisher(pub_ident, (pub.get('identity') or {}).get('lstart'), run_dir)
    direct = args.publisher_only or not sup_alive
    plan = {'run_dir': str(run_dir), 'supervisor_alive': sup_alive, 'publisher_pid': pub.get('pid'),
            'publisher_identity_matches': pub_ours,
            'action': ('write STOP (the supervisor stops its publisher gracefully)' if not direct else
                       ('SIGINT to the recorded publisher' if pub_ours else 'nothing to signal (no matching publisher)'))}
    if args.dry_run:
        print(json.dumps(plan, indent=1))
        print('--dry-run: nothing was written or signalled.')
        return 0
    (run_dir/'STOP').write_text(now_iso()+'\n')
    if not direct:
        print(json.dumps(plan, indent=1))
        return 0
    if not pub_ours:
        print(json.dumps(plan, indent=1))
        if pub_ident:
            print(f'pid {pub.get("pid")} is running but its identity does not match the recorded publisher: NOT signalled')
        return 0
    os.kill(pub['pid'], signal.SIGINT)
    print(json.dumps(plan, indent=1))
    if args.term:
        deadline = time.time()+cfg['stop']['graceful_s']
        while time.time() < deadline and is_our_publisher(ps_identity(pub['pid']), pub['identity']['lstart'], run_dir):
            time.sleep(1)
        if is_our_publisher(ps_identity(pub['pid']), pub['identity']['lstart'], run_dir):
            os.kill(pub['pid'], signal.SIGTERM)
            print('SIGTERM sent (identity matched again)')
    return 0


def cmd_childwatch(args, cfg):
    """Independent hard stop for ONE publisher child: if the supervisor is gone (pid + start time no longer match) or
    the deadline passed, SIGINT (then SIGTERM) the child, only while its pid, start time and argv still match."""
    logpath = Path(args.run_dir)/'childwatch.jsonl'

    def log(**row):
        with open(logpath, 'a') as handle:
            handle.write(json.dumps({'at': now_iso(), 'child': args.pid} | row)+'\n')

    def child_matches():
        return is_our_publisher(ps_identity(args.pid), args.lstart, args.run_dir)
    log(event='childwatch_start', deadline=now_iso(args.deadline))
    if not child_matches():  # nothing to watch: say why instead of exiting silently
        log(event='child_identity_unreadable_or_mismatch', identity=ps_identity(args.pid), lstart=args.lstart)
    while child_matches():
        sup = ps_identity(args.supervisor)
        orphaned = not (sup and sup['lstart'] == args.supervisor_lstart.strip())
        if orphaned or time.time() > args.deadline:
            why = 'supervisor_gone' if orphaned else 'deadline'
            for sig, wait in ((signal.SIGINT, args.graceful_s), (signal.SIGTERM, args.term_s)):
                if not child_matches():
                    break
                os.kill(args.pid, sig)
                log(event='signal', signal=sig.name, why=why)
                end = time.time()+wait
                while time.time() < end and child_matches():
                    time.sleep(0.5)
            break
        time.sleep(args.every_s)
    log(event='childwatch_end', child_alive=child_matches())
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='World Series operations launcher (see module docstring)')
    parser.add_argument('--config', default=None, help='ws_ops_config.json (default: next to this file)')
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--schedule-file')
    p.add_argument('--dry-run', action='store_true', help='accepted for symmetry: plan never writes')
    for name in ('start', 'supervise'):
        p = sub.add_parser(name)
        p.add_argument('--date', required=True)
        p.add_argument('--upload', action='store_true')
        p.add_argument('--schedule-file')
        p.add_argument('--expect-commit')
        if name == 'start':
            p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('status')
    p.add_argument('--date')
    p.add_argument('--dry-run', action='store_true', help='accepted for symmetry: status never writes')
    p = sub.add_parser('stop')
    p.add_argument('--date', required=True)
    p.add_argument('--publisher-only', action='store_true')
    p.add_argument('--term', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('childwatch')
    for name, kind in (('--pid', int), ('--lstart', str), ('--run-dir', str), ('--supervisor', int),
                       ('--supervisor-lstart', str), ('--deadline', float), ('--graceful-s', float),
                       ('--term-s', float), ('--every-s', float)):
        p.add_argument(name, type=kind, required=True)
    args = parser.parse_args(argv)
    if args.cmd == 'childwatch':
        return cmd_childwatch(args, None)
    cfg = load_config(args.config)
    if args.cmd == 'supervise' or (args.cmd == 'start' and not args.dry_run):
        Path(cfg['ops_root']).mkdir(parents=True, exist_ok=True)
    return {'plan': cmd_plan, 'start': cmd_start, 'supervise': cmd_supervise, 'status': cmd_status,
            'stop': cmd_stop}[args.cmd](args, cfg)


if __name__ == '__main__':
    sys.exit(main())
