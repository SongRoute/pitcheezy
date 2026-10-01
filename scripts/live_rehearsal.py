"""Real-game rehearsal of the delayed-live observer (DEMO-WS-2026; a service measurement, not a model evaluation).

``record`` polls a running observer's ``/api/live/{gamePk}/state`` like a phone would, from before first
pitch until the game is over, and writes one JSON line per poll. The observer itself fetches the public
feed (one upstream fetcher) and stores every new raw snapshot under ``apps/observer/live_snapshots``; the
file's mtime is the wall time the snapshot was first fetched.

``report`` joins the poll log with those snapshots: how long after a pitch the feed shows it (publication
lag, including our poll phase), how much time is left before the NEXT pitch once its recommendation can be
computed (the margin a viewer has at each broadcast delay), recommendation coverage and server timings.
Nothing here reads or writes model inputs; stdlib only.

    python scripts/live_rehearsal.py record --game-pk 849844 --server http://100.108.252.111:8766 --output DIR
    python scripts/live_rehearsal.py report --game-pk 849844 --output DIR --snapshots apps/observer/live_snapshots
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

OVER = ('Final', 'Game Over', 'Completed Early')
BROADCAST_DELAYS = (0, 10, 20, 30, 45, 60)
READ_S = 5  # seconds a viewer needs to read a recommendation before the pitch


def _get(url, timeout):
    request = urllib.request.Request(url, headers={'User-Agent': 'pitcheezy-live-rehearsal'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def poll_line(server, game_pk, timeout=20, query=''):
    """One viewer poll, reduced to what the report needs (no raw feed, no candidate numbers)."""
    started, wall = time.perf_counter(), time.time()
    line = {'wall': round(wall, 3), 'utc': datetime.fromtimestamp(wall, timezone.utc).isoformat(timespec='seconds')}
    try:
        status, body = _get(f'{server}/api/live/{game_pk}/state{query and "?" + query}', timeout)
    except urllib.error.HTTPError as exc:
        return line | {'http': exc.code, 'ms': round((time.perf_counter()-started)*1000, 1), 'error': 'http'}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return line | {'http': None, 'ms': round((time.perf_counter()-started)*1000, 1), 'error': type(exc).__name__}
    rec, game, situation = body.get('recommendation') or {}, body.get('game') or {}, body.get('situation') or {}
    top = ((rec.get('recommendation') or {}).get('candidates') or [None])[0]
    previous = rec.get('previous') or {}
    return line | {
        'http': status, 'ms': round((time.perf_counter()-started)*1000, 1), 'status': body.get('status'),
        'reason': body.get('reason'), 'delay_s': body.get('delay_s'), 'sync': body.get('sync'),
        'game_status': game.get('status'),
        'feed_timestamp': game.get('feed_timestamp'),
        'situation': {k: situation.get(k) for k in ('inning', 'half', 'outs', 'balls', 'strikes', 'bases')} if situation else None,
        'pitcher': (body.get('pitcher') or {}).get('id'), 'batter': (body.get('batter') or {}).get('id'),
        'rec_status': rec.get('status'), 'rec_reason': rec.get('reason'), 'rec_key': rec.get('key'),
        'rec_top': None if top is None else {'pitch_type': top.get('pitch_type'), 'zone_id': top.get('zone_id')},
        'rec_seconds': rec.get('seconds'), 'previous_key': previous.get('key'),
        'previous_type': (previous.get('actual') or {}).get('pitch_type')}


def record(server, game_pk, output, *, interval=5.0, idle_interval=30.0, max_hours=7.0, sleep=time.sleep, poll=poll_line,
           clock=time.time):
    """Polls until the game has been live and is over (three consecutive polls), or ``max_hours`` passed."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    log, deadline = output/f'{int(game_pk)}_polls.jsonl', clock()+max_hours*3600
    seen_live, over, polls = False, 0, 0
    with log.open('a') as handle:
        while clock() < deadline:
            line = poll(server, game_pk)
            handle.write(json.dumps(line, ensure_ascii=False)+'\n')
            handle.flush()
            polls += 1
            live = line.get('status') in ('ready', 'between_innings', 'buffering')
            seen_live = seen_live or line.get('status') in ('ready', 'between_innings')
            over = over+1 if line.get('status') == 'not_live' and line.get('game_status') in OVER else 0
            if over >= 3:
                break
            sleep(interval if live or seen_live else idle_interval)
    return {'log': str(log), 'polls': polls, 'seen_live': seen_live, 'finished': over >= 3}


def _time(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()


def pitches_of(feed):
    """{(atBatIndex, event index): (start, end)} wall times of every pitch a snapshot already contains."""
    out = {}
    for play in feed['liveData']['plays'].get('allPlays') or []:
        for event in play.get('playEvents') or []:
            if event.get('isPitch') and event.get('startTime') and event.get('endTime'):
                out[(play['about']['atBatIndex'], event['index'])] = (_time(event['startTime']), _time(event['endTime']))
    return out


def _quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda p: round(ordered[min(len(ordered)-1, int(len(ordered)*p))], 2)
    return {'n': len(ordered), 'p10': pick(.1), 'median': round(statistics.median(ordered), 2), 'p90': pick(.9),
            'min': round(ordered[0], 2), 'max': round(ordered[-1], 2)}


def report(game_pk, output, snapshots, *, poll_interval_s=10.0):
    output, game_pk = Path(output), int(game_pk)
    files = sorted(Path(snapshots).glob(f'{game_pk}_*.json.gz'), key=lambda p: p.stat().st_mtime)
    first_seen, times = {}, {}
    for path in files:
        with gzip.open(path, 'rt') as handle:
            found = pitches_of(json.load(handle))
        for key, span in found.items():
            times[key] = span
            first_seen.setdefault(key, path.stat().st_mtime)
    order = sorted(times, key=lambda key: times[key][0])
    # A pitch already in the first snapshot was thrown before recording started: its first-seen time is not a lag.
    opened = files[0].stat().st_mtime if files else None
    fresh = [key for key in order if first_seen[key] > opened and times[key][1] > opened-poll_interval_s]
    lag = [first_seen[key]-times[key][1] for key in fresh]
    # Recommendation for pitch k can be computed once the snapshot showing pitch k-1 arrived.
    margin = [times[key][0]-first_seen[before] for before, key in zip(order, order[1:])
              if before in fresh and times[key][0]-times[before][1] < 120]
    polls = [json.loads(line) for line in (output/f'{game_pk}_polls.jsonl').read_text().splitlines() if line.strip()]
    live = [p for p in polls if p.get('status') == 'ready']
    by_key, compute = {}, []
    for p in live:
        if p.get('rec_key') and p.get('rec_status') in ('ready', 'unsupported'):
            if p['rec_key'] not in by_key:
                by_key[p['rec_key']] = p['rec_status']
                if p.get('rec_seconds'):
                    compute.append(p['rec_seconds'].get('total'))
    count = lambda rows, field: {str(k): sum(r.get(field) == k for r in rows) for k in sorted({str(r.get(field)) for r in rows})}
    result = {
        'schema': 'pitcheezy.live-rehearsal.v1', 'game_pk': game_pk,
        'note': 'Service timing/coverage measurement on DEMO-WS-2026 data; not a model or policy evaluation.',
        'snapshots': len(files), 'pitches_in_snapshots': len(order), 'pitches_first_seen_live': len(fresh),
        'publication_lag_s': _quantiles(lag),
        'publication_lag_note': f'first fetch of a snapshot containing the pitch minus the pitch endTime; includes our poll phase (0..{poll_interval_s:g} s)',
        'next_pitch_margin_s': _quantiles(margin),
        'next_pitch_margin_note': 'next pitch startTime minus first fetch of the snapshot containing the previous pitch (gaps under 120 s); compute time not subtracted',
        'share_with_read_time_by_broadcast_delay': {
            str(delay): None if not margin else round(sum(m+delay >= READ_S for m in margin)/len(margin), 4)
            for delay in BROADCAST_DELAYS},
        'read_time_s': READ_S,
        'polls': {'total': len(polls), 'errors': sum('error' in p for p in polls), 'status': count(polls, 'status'),
                  'response_ms': _quantiles([p['ms'] for p in polls if 'error' not in p])},
        'recommendation': {'poll_status_when_ready_state': count(live, 'rec_status'), 'decisions_seen': len(by_key),
                           'decisions_ready': sum(v == 'ready' for v in by_key.values()),
                           'decisions_unsupported': sum(v == 'unsupported' for v in by_key.values()),
                           'compute_total_s': _quantiles([c for c in compute if c is not None])}}
    (output/f'{game_pk}_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    rec = commands.add_parser('record')
    rec.add_argument('--game-pk', type=int, required=True)
    rec.add_argument('--server', default='http://127.0.0.1:8766')
    rec.add_argument('--output', type=Path, required=True)
    rec.add_argument('--interval', type=float, default=5.0)
    rec.add_argument('--max-hours', type=float, default=7.0)
    rec.add_argument('--query', default='', help='extra query of the state route, e.g. delay=0')
    rep = commands.add_parser('report')
    rep.add_argument('--game-pk', type=int, required=True)
    rep.add_argument('--output', type=Path, required=True)
    rep.add_argument('--snapshots', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'record':
        poll = lambda server, game_pk: poll_line(server, game_pk, query=args.query)
        print(json.dumps(record(args.server.rstrip('/'), args.game_pk, args.output, interval=args.interval,
                                max_hours=args.max_hours, poll=poll), ensure_ascii=False), flush=True)
    else:
        print(json.dumps(report(args.game_pk, args.output, args.snapshots), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
