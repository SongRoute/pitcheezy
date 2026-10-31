"""World Series schedule sync: confirmed first-pitch times -> alert config starts + per-game launchd plists.

    ws_schedule_sync.py --from-file SCHEDULE.json --out-dir DIR [--write]
    ws_schedule_sync.py --fetch [--save-raw FILE] --out-dir DIR [--write]

Input: a saved statsapi schedule body (``{"dates": [{"games": [...]}]}``) or ONE live statsapi schedule request for
the seven known World Series gamePks (``--fetch``; one request, never retried or looped, same User-Agent as
ws_operate.py's schedule check).

Default is a dry run. It prints
  - the diff of ws_alert_config.json ``schedule`` starts (placeholder -> real UTC, with the KST time),
  - flags per game: TBD time, midnight placeholder, postponed/suspended/cancelled, if-necessary, missing,
  - one launchd plist per game written to ``--out-dir`` (derived from com.pitcheezy.ws-operate.plist.example) that
    runs ``ws_operate.py start --date <KST game date>`` once, at first pitch - (lead_min + start_retry_window_s)
    from ws_ops_config.json (30 + 30 min = T-60 min with the current config): the supervisor then sleeps until
    first pitch - lead_min, and start's own retry window (start_retry_window_s) fits before the publisher start.
  - the exact cp / plutil / launchctl bootstrap / bootout commands for Song. It never installs or loads anything.

``--write`` replaces only the ``"start"`` strings inside ws_alert_config.json's ``schedule`` entries (text edit, the
rest of the file is byte-identical), and only when no game is refused.

Refuses (exit 2, nothing written, no plists): a World Series (gameType W) gamePk outside the seven known ones, a
known gamePk missing from the input, a time that is still the 00:00:00Z placeholder or marked startTimeTBD (unless
the game is postponed/cancelled), two games on one KST date, a template missing a required key or the date
expression, an alert config whose schedule gamePks differ from the seven, or an --out-dir inside a git checkout or
~/Library/LaunchAgents. Non-W rows in a saved file (e.g. a date-range body with ALCS games) are ignored and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ws_operate  # noqa: E402  (same folder; import has no side effects)

KNOWN_PKS = (849805, 849802, 849804, 849803, 849799, 849801, 849800)
SEOUL = ZoneInfo('Asia/Seoul')
DEFAULT_ALERT_CONFIG = HERE/'ws_alert_config.json'
DEFAULT_TEMPLATE = HERE/'com.pitcheezy.ws-operate.plist.example'
USER_AGENT = 'pitcheezy-ws-ops-schedule-check'  # ws_operate.fetch_schedule uses the same one
STATSAPI = 'https://statsapi.mlb.com/api/v1/schedule'
TEMPLATE_KEYS = ('Label', 'ProgramArguments', 'EnvironmentVariables', 'StartCalendarInterval', 'RunAtLoad',
                 'AbandonProcessGroup', 'StandardOutPath', 'StandardErrorPath')
DATE_EXPR = re.compile(r'--date "\$\(TZ=Asia/Seoul /bin/date \+%Y-%m-%d\)"')
OFF_STATES = ('Postponed', 'Suspended', 'Cancelled')
ENTRY_RE = re.compile(r'(\{\s*"game_pk"\s*:\s*(\d+)\s*,\s*"start"\s*:\s*")([^"]*)(")')


class Refused(Exception):
    pass


class RefusedWithRows(Refused):
    def __init__(self, reasons, rows, ignored):
        super().__init__('; '.join(reasons))
        self.reasons, self.rows, self.ignored = reasons, rows, ignored


def fetch(timeout_s):
    """ONE statsapi schedule request for the seven gamePks. No retry, no loop."""
    url = f'{STATSAPI}?sportId=1&gamePks={",".join(str(p) for p in KNOWN_PKS)}'
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def parse_games(body):
    """{game_pk: row} for the known World Series games, plus the count of ignored non-W rows."""
    if not isinstance(body, dict) or not isinstance(body.get('dates'), list):
        raise Refused('schedule body has no dates list (not a statsapi schedule response)')
    seen, ignored = {}, 0
    for day in body['dates']:
        if not isinstance(day, dict) or not isinstance(day.get('games'), list):
            raise Refused('schedule date entry without a games list')
        for g in day['games']:
            pk = int(g['gamePk'])
            if g.get('gameType') != 'W':
                if pk in KNOWN_PKS:
                    raise Refused(f'gamePk {pk} is not gameType W ({g.get("gameType")!r})')
                ignored += 1
                continue
            if pk not in KNOWN_PKS:
                raise Refused(f'World Series gamePk {pk} is not one of the seven known ones {KNOWN_PKS}')
            status = g.get('status') or {}
            row = {'game_pk': pk, 'start_utc': g['gameDate'], 'detailed': status.get('detailedState'),
                   'tbd': bool(status.get('startTimeTBD')), 'if_necessary': g.get('ifNecessary') == 'Y',
                   'if_necessary_text': g.get('ifNecessaryDescription'), 'series_game': g.get('seriesGameNumber'),
                   'reschedule': g.get('rescheduleDate') or g.get('rescheduleGameDate')}
            seen.setdefault(pk, []).append(row)
    games = {}
    for pk, rows in seen.items():
        live = [r for r in rows if r['detailed'] not in OFF_STATES]
        if len(live) > 1:
            raise Refused(f'gamePk {pk} listed {len(live)} times as not postponed')
        games[pk] = live[0] if live else rows[0]
    return games, ignored


def utc_text(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def seoul_text(epoch):
    return datetime.fromtimestamp(epoch, SEOUL).strftime('%Y-%m-%d %H:%M KST')


def is_midnight_placeholder(start_utc):
    return datetime.fromtimestamp(ws_operate.parse_utc(start_utc), timezone.utc).strftime('%H:%M:%S') == '00:00:00'


def read_alert_schedule(path):
    text = Path(path).read_text()
    cfg = json.loads(text)
    current = {int(e['game_pk']): e['start'] for e in cfg.get('schedule') or []}
    if sorted(current) != sorted(KNOWN_PKS):
        raise Refused(f'alert config schedule gamePks {sorted(current)} != the seven known {sorted(KNOWN_PKS)}')
    return text, current


def load_template(path):
    raw = Path(path).read_bytes()
    try:
        tpl = plistlib.loads(raw)
    except Exception as exc:  # noqa: BLE001  (plistlib raises several types)
        raise Refused(f'template not a valid plist: {type(exc).__name__}')
    missing = [k for k in TEMPLATE_KEYS if k not in tpl]
    if missing:
        raise Refused(f'template missing keys: {missing}')
    args = tpl['ProgramArguments']
    if not isinstance(args, list) or not args or not DATE_EXPR.search(args[-1]) or 'ws_operate.py start' not in args[-1]:
        raise Refused('template ProgramArguments last item lacks "ws_operate.py start --date "$(TZ=Asia/Seoul ...)""')
    return tpl


def make_plist(tpl, pk, kst_date, job_epoch, ws_operate_path=None):
    job = datetime.fromtimestamp(job_epoch, SEOUL)
    body = json.loads(json.dumps(tpl))  # deep copy (plain types only)
    body['Label'] = f'{tpl["Label"]}.{pk}'
    cmd = DATE_EXPR.sub(f'--date {kst_date}', body['ProgramArguments'][-1])
    if ws_operate_path:
        cmd = re.sub(r'\S*/apps/observer/ops/ws_operate\.py', str(ws_operate_path), cmd)
    body['ProgramArguments'][-1] = cmd
    body['StartCalendarInterval'] = {'Month': job.month, 'Day': job.day, 'Hour': job.hour, 'Minute': job.minute}
    body['RunAtLoad'] = False
    return body


def inside_checkout(path):
    for p in [path, *path.parents]:
        if (p/'.git').exists():
            return p
    return None


def plan(body, alert_text, current, tpl, ops_cfg):
    """Pure: everything the dry run shows. Raises Refused on any refusal."""
    games, ignored = parse_games(body)
    offset_s = ops_cfg['lead_min']*60+ops_cfg['start_retry_window_s']
    rows, refusals, by_date = [], [], {}
    for pk in KNOWN_PKS:
        g = games.get(pk)
        if g is None:
            refusals.append(f'{pk}: not in the schedule input')
            rows.append({'game_pk': pk, 'old': current[pk], 'new': None, 'flags': ['missing']})
            continue
        flags = []
        off = g['detailed'] in OFF_STATES
        if off:
            flags.append(g['detailed'].lower())
        if g['tbd']:
            flags.append('tbd')
        if is_midnight_placeholder(g['start_utc']):
            flags.append('midnight_placeholder')
        if g['if_necessary']:
            flags.append('if_necessary')
        if not off and ('tbd' in flags or 'midnight_placeholder' in flags):
            refusals.append(f'{pk}: time not confirmed ({", ".join(f for f in flags if f in ("tbd", "midnight_placeholder"))})')
        epoch = ws_operate.parse_utc(g['start_utc'])
        kst_date = datetime.fromtimestamp(epoch, ws_operate.KST).date().isoformat()  # the launcher's own date rule
        row = {'game_pk': pk, 'old': current[pk], 'new': current[pk] if off else utc_text(epoch), 'kst': seoul_text(epoch), 'kst_date': kst_date,
               'detailed': g['detailed'], 'series_game': g['series_game'], 'flags': flags}
        if not off:
            if kst_date in by_date:
                refusals.append(f'{pk}: same KST date {kst_date} as {by_date[kst_date]}')
            by_date[kst_date] = pk
            job = epoch-offset_s
            row['job_kst'], row['offset_min'] = seoul_text(job), offset_s//60
            row['plist'] = make_plist(tpl, pk, kst_date, job, ops_cfg.get('_ws_operate_path'))
        rows.append(row)
    if refusals:
        raise RefusedWithRows(refusals, rows, ignored)
    # a postponed/cancelled row with no live replacement keeps its old start (flagged, no plist)
    new_text = replace_starts(alert_text, {r['game_pk']: r['new'] for r in rows if 'plist' in r})
    return {'rows': rows, 'ignored_non_w': ignored, 'offset_s': offset_s, 'new_alert_text': new_text}


def replace_starts(text, new_starts):
    """Replace only the "start" values of the schedule entries; every other byte stays."""
    head, sep, tail = text.partition('"schedule": [')
    if not sep:
        raise Refused('alert config has no "schedule": [ list')
    block, close, rest = tail.partition(']')
    done = set()

    def sub(m):
        pk = int(m.group(2))
        done.add(pk)
        return m.group(1)+new_starts.get(pk, m.group(3))+m.group(4)
    block = ENTRY_RE.sub(sub, block)
    if done != set(KNOWN_PKS):
        raise Refused(f'alert config schedule entries not in the expected one-line form (found {sorted(done)})')
    return head+sep+block+close+rest


def show(rows, ignored):
    print(f'World Series schedule ({len(KNOWN_PKS)} known gamePks; ignored non-W rows: {ignored})')
    for r in rows:
        flags = f' [{", ".join(r["flags"])}]' if r['flags'] else ''
        if r.get('new') is None:
            print(f'  {r["game_pk"]}: {r["old"]} -> (missing){flags}')
            continue
        same = ' (unchanged)' if r['new'] == r['old'] else ''
        g = f' G{r["series_game"]}' if r.get('series_game') else ''
        print(f'  {r["game_pk"]}{g}: {r["old"]} -> {r["new"]}{same}  = {r["kst"]}  {r.get("detailed")}{flags}')
        if r.get('job_kst'):
            print(f'      launchd: ws_operate.py start --date {r["kst_date"]} at {r["job_kst"]}'
                  f' (first pitch - {r["offset_min"]} min)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--from-file', type=Path, help='saved statsapi schedule body (JSON)')
    src.add_argument('--fetch', action='store_true', help='ONE statsapi schedule request for the seven gamePks')
    ap.add_argument('--save-raw', type=Path, help='with --fetch: save the raw response here (outside the repo)')
    ap.add_argument('--out-dir', type=Path, required=True, help='folder for the generated plists (not LaunchAgents)')
    ap.add_argument('--write', action='store_true', help='apply the start times to the alert config')
    ap.add_argument('--alert-config', type=Path, default=DEFAULT_ALERT_CONFIG)
    ap.add_argument('--ops-config', type=Path, default=ws_operate.DEFAULT_CONFIG)
    ap.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    ap.add_argument('--ws-operate', type=Path, help='ws_operate.py path to put in the plists (default: template)')
    args = ap.parse_args(argv)
    try:
        out_dir = args.out_dir.expanduser().resolve()
        agents = (Path.home()/'Library'/'LaunchAgents').resolve()
        if out_dir == agents or agents in out_dir.parents:
            raise Refused('--out-dir must not be ~/Library/LaunchAgents (this tool never installs)')
        repo = inside_checkout(out_dir)
        if repo:
            raise Refused(f'--out-dir {out_dir} is inside the git checkout {repo}')
        ops_cfg = ws_operate.load_config(args.ops_config)
        ops_cfg['_ws_operate_path'] = args.ws_operate and args.ws_operate.expanduser().resolve()
        tpl = load_template(args.template)
        alert_text, current = read_alert_schedule(args.alert_config)
        if args.fetch:
            raw = fetch(ops_cfg['schedule']['timeout_s'])
            if args.save_raw:
                args.save_raw.parent.mkdir(parents=True, exist_ok=True)
                args.save_raw.write_bytes(raw)
                print(f'raw response saved: {args.save_raw}')
            body = json.loads(raw)
        else:
            body = json.loads(args.from_file.read_text())
        result = plan(body, alert_text, current, tpl, ops_cfg)
    except RefusedWithRows as exc:
        show(exc.rows, exc.ignored)
        print(f'REFUSED (nothing written): {exc}', file=sys.stderr)
        return 2
    except Refused as exc:
        print(f'REFUSED (nothing written): {exc}', file=sys.stderr)
        return 2
    show(result['rows'], result['ignored_non_w'])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nplists (not installed) in {out_dir}:')
    files = []
    for r in result['rows']:
        if r.get('plist'):
            name = f'{r["plist"]["Label"]}.plist'
            (out_dir/name).write_bytes(plistlib.dumps(r['plist']))
            files.append((name, r['plist']['Label'], r))
            print(f'  {name}')
    print('\nfor Song, after review and approval (not run by this tool):')
    print('  # the daily 07:00 job (com.pitcheezy.ws-operate) must NOT also be installed')
    for name, label, _ in files:
        print(f'  cp {out_dir/name} ~/Library/LaunchAgents/{name} && plutil -lint ~/Library/LaunchAgents/{name}'
              f' && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{name}')
    print('  # remove one (after its game, or if the game is not played):')
    for name, label, _ in files:
        print(f'  launchctl bootout gui/$(id -u)/{label} && rm ~/Library/LaunchAgents/{name}')
    flagged = [r for r in result['rows'] if r['flags']]
    if flagged:
        print('\nflagged: ' + '; '.join(f'{r["game_pk"]} {",".join(r["flags"])}' for r in flagged))
    if result['new_alert_text'] == alert_text:
        print('\nalert config: no change')
    elif args.write:
        tmp = args.alert_config.with_name(f'.{args.alert_config.name}.{os.getpid()}.tmp')
        tmp.write_text(result['new_alert_text'])
        os.replace(tmp, args.alert_config)
        print(f'\nalert config WRITTEN: {args.alert_config} (only schedule starts; update schedule_provisional_note '
              'by hand, then restart the watcher)')
    else:
        print('\nalert config: dry run (pass --write to apply the starts above)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
