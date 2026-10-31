"""Export a FINISHED one-game capture SQLite into the folder ``observer_app.live_feed.ReplayTransport`` replays.

Source: the ``snapshots`` table written by ``observer_app.live_store.LiveSnapshotStore`` (the SVC06 capture
wrapper ``capture_one_game.py`` uses it): one row per stored version, ``body`` = one gzip member of the canonical
JSON ``feed/live`` response, ``sha`` = sha256 of the inflated bytes, ``seq`` = commit (receipt) order. The store
never keeps a ``duplicate`` or ``stale`` response, so this folder holds the stored versions, not every HTTP reply.

Output (``ReplayTransport`` hands out ``{gamePk}_*.json*`` in file-name order, one per upstream request):
  {pk}_{timeStamp}_{seq:06d}.json.gz   when ``metaData.timeStamp`` never goes backwards in receipt order, else
  {pk}_{seq:06d}_{timeStamp}.json.gz   so that name order is always receipt order (``--naming`` forces either)
  schedule_{officialDate}.json         synthesized from one body's gameData (the capture stores no schedule reply)
  MANIFEST.json                        counts, source identity, mode, one line per file; written last

Default mode writes each stored gzip member byte for byte, so the inflated sha256 is the stored ``sha``.
``--rebase-event-wall`` rewrites every body with ``metaData.eventWall`` moved to today's clock (first event at
``--rebase-start``, original spacing kept); those bodies are MODIFIED and the manifest says so.

Safety: refuses unless the capture is terminal (TERMINAL.json CLEAN, SUPERVISOR_TERMINAL.json CLEAN, child
heartbeat TERMINAL, no RECOVERY.json, owner processes gone, empty WAL). The source is opened with
``mode=ro&immutable=1`` + ``query_only``: no lock, no -wal/-shm, no checkpoint, no copy. A non-empty WAL is always
refused (immutable would miss its content and a plain read-only open could create -shm beside the original).
``--assume-terminal`` skips only the receipt/process checks and exists for synthetic tests.

    python scripts/capture_to_replay.py <run-dir>/live-<pk>.sqlite3 <new-empty-out-dir> [--rebase-event-wall]

Never writes ``{pk}_final.json.gz``: the fan page would then show the archive instead of the replay.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
import zlib

MAX_BODY_BYTES = 64*1024*1024
STAMP = re.compile(r'\d{8}_\d{6}')
CAPTURE_PROCESS_WORDS = ('capture_one_game.py', 'capture_supervisor.py')


class Refused(Exception):
    def __init__(self, reason, **detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _command(pid):
    """Full command line of ``pid`` ('' if no such process). Reads the process table; signals nothing."""
    if type(pid) is not int or pid <= 0:
        return ''
    return subprocess.run(['/bin/ps', '-o', 'command=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()


def terminal_gate(db_path, command=_command):
    """Metadata only (no database byte is read). Returns the child terminal receipt or raises Refused with every
    failed check. Mirrors the literals ``postcapture_replay.py`` requires, minus its registration pins; a capture
    that ended cleanly without reaching Final (e.g. hard end) is still exportable and is reported as such."""
    run, bad = Path(db_path).parent, []
    child, top = _json(run/'TERMINAL.json') or {}, _json(run/'SUPERVISOR_TERMINAL.json') or {}
    beat, own, sup = _json(run/'heartbeat.json') or {}, _json(run/'owner.json') or {}, _json(run/'supervisor.json') or {}

    def need(name, got, want):
        if got != want:
            bad.append({'check': name, 'got': got, 'want': want})
    need('RECOVERY.json absent', (run/'RECOVERY.json').exists(), False)
    need('TERMINAL.json terminal_state', child.get('terminal_state'), 'CLEAN')
    need('TERMINAL.json wal_state', child.get('wal_state'), 'CHECKPOINTED_TRUNCATED')
    need('TERMINAL.json wal_bytes', child.get('wal_bytes'), 0)
    need('TERMINAL.json quick_check', child.get('quick_check'), 'ok')
    need('TERMINAL.json db', child.get('db') and os.path.realpath(child['db']), os.path.realpath(db_path))
    need('SUPERVISOR_TERMINAL.json state', top.get('state'), 'CLEAN')
    need('SUPERVISOR_TERMINAL.json owned_children_remaining', top.get('owned_children_remaining'), 0)
    need('SUPERVISOR_TERMINAL.json nonempty_wal_preserved', top.get('nonempty_wal_preserved'), [])
    need('heartbeat.json phase', beat.get('phase'), 'TERMINAL')
    for name, pid in (('owner.pid', own.get('pid')), ('supervisor.pid', sup.get('pid')), ('terminal.pid', child.get('pid'))):
        line = command(pid)
        need(f'{name} {pid} is not a running capture process', any(word in line for word in CAPTURE_PROCESS_WORDS), False)
    if bad:
        raise Refused('CAPTURE_NOT_TERMINAL', failed=bad, note='metadata only; the database was not opened')
    return child


def open_source(path):
    db = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro&immutable=1', uri=True, isolation_level=None)
    db.execute('PRAGMA query_only=ON')
    return db


def inflate(blob, limit):
    """The stored gzip member, never inflated past ``limit``; None if it is not exactly one complete member."""
    try:
        inflater = zlib.decompressobj(31)
        body = inflater.decompress(blob, limit+1)
    except zlib.error:
        return None
    return body if len(body) <= limit and inflater.eof and not inflater.unused_data else None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(path):
    path = Path(path)
    side = {suffix: (Path(str(path)+suffix).stat().st_size if Path(str(path)+suffix).exists() else None)
            for suffix in ('-wal', '-shm')}
    return {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns, 'wal_bytes': side['-wal'],
            'shm_bytes': side['-shm']}


def _iso(epoch):
    return None if epoch is None else datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec='milliseconds')


def _epoch(text):
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def _plan(db, game, keep_repeats, naming):
    """Body-free pass over (seq, feed_ts, sha): which rows become files and under which name."""
    rows = db.execute('SELECT seq, feed_ts, sha FROM snapshots WHERE game_pk=? ORDER BY seq', (game,)).fetchall()
    seen, emit, skipped = set(), [], []
    for seq, feed_ts, sha in rows:
        if sha in seen and not keep_repeats:
            skipped.append(seq)
            continue
        seen.add(sha)
        emit.append((seq, feed_ts if isinstance(feed_ts, str) and STAMP.fullmatch(feed_ts) else None))
    if rows and rows[-1][0] > 999999:
        raise Refused('SEQ_OVER_SIX_DIGITS', seq=rows[-1][0])
    stamps = [stamp for _, stamp in emit]
    ordered = all(stamps) and stamps == sorted(stamps)
    if naming == 'auto':
        naming = 'stamp-seq' if ordered else 'seq-stamp'
    if naming == 'stamp-seq' and not ordered:
        raise Refused('STAMP_ORDER_IS_NOT_RECEIPT_ORDER', note='metaData.timeStamp is missing or goes backwards in '
                      'receipt order; use --naming seq-stamp (or auto)')
    names = {seq: (f'{game}_{stamp}_{seq:06d}' if naming == 'stamp-seq' else f'{game}_{seq:06d}_{stamp or "nostamp"}')
             + '.json.gz' for seq, stamp in emit}
    return len(rows), names, skipped, naming, ordered


def _schedule_game(feed):
    """One schedule ``games`` item with exactly the fields ``LiveFeed.schedule`` reads, from a feed body."""
    data, live = feed.get('gameData') or {}, feed.get('liveData') or {}
    game, when, status = data.get('game') or {}, data.get('datetime') or {}, data.get('status') or {}
    runs = (live.get('linescore') or {}).get('teams') or {}
    sides = {side: {'team': {key: ((data.get('teams') or {}).get(side) or {}).get(key) for key in ('id', 'name')},
                    'score': (runs.get(side) or {}).get('runs')} for side in ('away', 'home')}
    return {'gamePk': game.get('pk'), 'gameType': game.get('type'), 'gameDate': when.get('dateTime'),
            'officialDate': when.get('officialDate'), 'gameNumber': game.get('gameNumber'),
            'doubleHeader': game.get('doubleHeader'),
            'status': {key: status.get(key) for key in ('abstractGameState', 'detailedState')}, 'teams': sides}


def export(db_path, out_dir, *, game=None, assume_terminal=False, rebase=False, rebase_start=None,
           keep_repeats=False, naming='auto', schedule_from='first-live', max_body_bytes=MAX_BODY_BYTES,
           wall=time.time, command=_command):
    """Returns (exit code, manifest). Raises Refused before anything is written to ``out_dir``."""
    db_path, out = Path(db_path).resolve(), Path(out_dir).resolve()
    if not db_path.is_file():
        raise Refused('SOURCE_NOT_A_FILE', path=str(db_path))
    if out == db_path.parent or db_path.parent in out.parents:
        raise Refused('OUTPUT_INSIDE_CAPTURE_RUN_DIR', out=str(out))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refused('OUTPUT_DIR_NOT_EMPTY', out=str(out))
    terminal = None if assume_terminal else terminal_gate(db_path, command)
    before = _stat(db_path)
    if before['wal_bytes']:
        raise Refused('WAL_NOT_EMPTY', **before, note='committed rows may live only in the WAL; not opened')
    sha_before = file_sha256(db_path)
    if terminal and terminal.get('db_sha256') != sha_before:
        raise Refused('DB_SHA_IS_NOT_TERMINAL_RECEIPT', got=sha_before, want=terminal.get('db_sha256'))

    counts = {'rows_read': 0, 'files_written': 0, 'duplicates_skipped': 0, 'sha_mismatches': 0,
              'receipt_order_violations': 0, 'bytes_written': 0, 'inflated_bytes': 0}
    files, receipts, schedule, problem, shift = [], [None, None], {}, None, None
    with closing(open_source(db_path)) as db:
        games = [row[0] for row in db.execute('SELECT DISTINCT game_pk FROM snapshots ORDER BY game_pk')]
        if game is None and len(games) != 1:
            raise Refused('SOURCE_NOT_SINGLE_GAME', games=games, note='pass --game')
        game = games[0] if game is None else int(game)
        if game not in games:
            raise Refused('GAME_NOT_IN_SOURCE', game=game, games=games)
        total, names, skipped, naming, ordered = _plan(db, game, keep_repeats, naming)
        out.mkdir(parents=True, exist_ok=True)
        previous = None
        cursor = db.execute('SELECT seq, event_wall, version, feed_ts, sha, first_fetched, fetched, source, body '
                            'FROM snapshots WHERE game_pk=? ORDER BY seq', (game,))
        for seq, event, version, feed_ts, sha, first, fetched, source, blob in cursor:  # one body in memory at a time
            counts['rows_read'] += 1
            receipts = [fetched if receipts[0] is None else receipts[0], fetched]
            if previous is not None and fetched < previous:
                counts['receipt_order_violations'] += 1
            previous = fetched
            body = inflate(bytes(blob), max_body_bytes)
            if body is None or hashlib.sha256(body).hexdigest() != sha:
                counts['sha_mismatches'] += 1
                problem = {'seq': seq, 'problem': 'stored body is not one complete gzip member within the cap'
                           if body is None else 'sha256 of the stored body is not snapshots.sha'}
                break
            if seq not in names:
                counts['duplicates_skipped'] += 1
                continue
            feed = json.loads(body)
            meta, data = feed.get('metaData') or {}, feed.get('gameData') or {}
            if (data.get('game') or {}).get('pk') != game or meta.get('timeStamp') != feed_ts:
                problem = {'seq': seq, 'problem': 'body gamePk/timeStamp disagree with the row',
                           'body': [(data.get('game') or {}).get('pk'), meta.get('timeStamp')], 'row': [game, feed_ts]}
                break
            state = (data.get('status') or {}).get('abstractGameState')
            if ('first' not in schedule) or schedule_from == 'last' or (
                    schedule_from == 'first-live' and state == 'Live' and schedule.get('state') != 'Live'):
                schedule = {'first': True, 'state': state, 'seq': seq, 'game': _schedule_game(feed)}
            entry = {'file': names[seq], 'seq': seq, 'version': version, 'sha256': sha, 'first_fetched': first,
                     'fetched': fetched, 'event_wall': event, 'source': source}
            if rebase:
                shift = (wall() if rebase_start is None else rebase_start)-event if shift is None else shift
                feed['metaData'] = meta | {'eventWall': round(event+shift, 3)}
                body = json.dumps(feed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
                payload = gzip.compress(body, compresslevel=5, mtime=0)
                entry |= {'event_wall_written': feed['metaData']['eventWall'], 'sha256_written': hashlib.sha256(body).hexdigest()}
            else:
                payload = bytes(blob)
            with open(out/names[seq], 'xb') as handle:
                handle.write(payload)
            counts['files_written'] += 1
            counts['bytes_written'] += len(payload)
            counts['inflated_bytes'] += len(body)
            files.append(entry)
            del feed, body, payload, blob
    after, sha_after = _stat(db_path), file_sha256(db_path)
    official = (schedule.get('game') or {}).get('officialDate')
    schedule_file = None
    if not problem and official and re.fullmatch(r'\d{4}-\d{2}-\d{2}', official):
        schedule_file = f'schedule_{official}.json'
        (out/schedule_file).write_text(json.dumps({
            'synthesized_by': 'capture_to_replay.py', 'synthesized_from_seq': schedule['seq'],
            'note': 'NOT a provider schedule response; built from one stored feed body',
            'totalGames': 1, 'dates': [{'date': official, 'games': [schedule['game']]}]}, indent=1, ensure_ascii=False))
    unchanged = before == after and sha_before == sha_after
    status = ('FAILED_BODY_CHECK' if problem else 'FAILED_SOURCE_CHANGED' if not unchanged else
              'FAILED_ROW_COUNT' if counts['rows_read'] != total else 'OK')
    manifest = {
        'status': status, 'game_pk': game, 'problem': problem,
        'mode': {'bodies': 'MODIFIED_EVENT_WALL_REBASED' if rebase else 'UNCHANGED_STORED_GZIP_MEMBER',
                 'bodies_modified': bool(rebase), 'rebase_shift_s': shift,
                 'rebase_first_event_wall': _iso(files[0]['event_wall_written']) if rebase and files else None,
                 'rebase_rule': 'metaData.eventWall = snapshots.event_wall + shift; canonical JSON re-encoded' if rebase else None,
                 'repeated_bodies': 'kept' if keep_repeats else 'first occurrence only',
                 'naming': naming, 'timestamp_order_is_receipt_order': ordered,
                 'terminal_gate': 'SKIPPED_ASSUME_TERMINAL' if assume_terminal else 'PASSED'},
        'counts': counts | {'rows_in_plan': total, 'skipped_seqs': skipped},
        'receipt': {'clock': 'snapshots.fetched: local availability upper bound on the capture host, not provider time',
                    'first_unix': receipts[0], 'last_unix': receipts[1], 'first': _iso(receipts[0]), 'last': _iso(receipts[1])},
        'source': {'path': str(db_path), 'opened': 'mode=ro&immutable=1, query_only', 'sha256': sha_before,
                   'before': before, 'after': after, 'unchanged': unchanged,
                   'terminal': terminal and {key: terminal.get(key) for key in (
                       'reason', 'outcome_class', 'terminal_state', 'requests', 'started', 'ended', 'db_sha256',
                       'last_state', 'snapshots_by_game', 'data_role')}},
        'schedule': {'file': schedule_file, 'synthesized': True, 'from_seq': schedule.get('seq'),
                     'status_from': schedule_from, 'reason': 'the capture requests only feed/live; no schedule reply is stored'},
        'not_included': ['duplicate and stale HTTP replies (never stored)', 'request/poll timing', '{pk}_final.json.gz'],
        'files': files}
    (out/'MANIFEST.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    return (0 if status == 'OK' else 3), manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('db')
    parser.add_argument('out_dir')
    parser.add_argument('--game', type=int)
    parser.add_argument('--rebase-event-wall', action='store_true', help='MODIFIES bodies: adds metaData.eventWall on today\'s clock')
    parser.add_argument('--rebase-start', type=_epoch, help='epoch seconds or ISO time of the first event (default: now)')
    parser.add_argument('--keep-repeated-bodies', action='store_true',
                        help='also emit a row whose content equals an earlier row (a provider revert), keeping every stored version')
    parser.add_argument('--naming', choices=['auto', 'stamp-seq', 'seq-stamp'], default='auto')
    parser.add_argument('--schedule-status-from', choices=['first-live', 'first', 'last'], default='first-live')
    parser.add_argument('--max-body-bytes', type=int, default=MAX_BODY_BYTES)
    parser.add_argument('--assume-terminal', action='store_true',
                        help='SYNTHETIC TESTS ONLY: skip the terminal receipt and process checks')
    args = parser.parse_args(argv)
    try:
        code, manifest = export(args.db, args.out_dir, game=args.game, assume_terminal=args.assume_terminal,
                                rebase=args.rebase_event_wall, rebase_start=args.rebase_start,
                                keep_repeats=args.keep_repeated_bodies, naming=args.naming,
                                schedule_from=args.schedule_status_from, max_body_bytes=args.max_body_bytes)
    except Refused as exc:
        print(json.dumps({'refused': exc.reason} | exc.detail, indent=1, ensure_ascii=False))
        return 2
    print(json.dumps({key: manifest[key] for key in ('status', 'game_pk', 'problem', 'mode', 'counts', 'receipt', 'schedule')},
                     indent=1, ensure_ascii=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
