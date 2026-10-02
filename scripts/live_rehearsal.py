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
    python scripts/live_rehearsal.py simulate --feeds DEMO-WS-2026/feeds --output OUT.json --lags 5 10 15

``simulate`` answers "which poll interval is fast enough" from completed games' recorded pitch times.
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
SEEN_S = 1.0  # release to the pitch having arrived (the live feed's first endTime is about start + 0.7..1.2 s)


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
    fresh = [key for key in order if first_seen[key] > opened and times[key][0] > opened-poll_interval_s]
    # Times count from a pitch's startTime (release): the live feed first gives endTime = start + about 1 s,
    # then rewrites it to the next event's start, and the final feed re-stamps both (849844, 2026-10-02).
    lag = [first_seen[key]-times[key][0] for key in fresh]
    # Recommendation for pitch k can be computed once the snapshot showing pitch k-1 arrived.
    margin = [times[key][0]-first_seen[before] for before, key in zip(order, order[1:])
              if before in fresh and times[key][0]-times[before][0] < 120]
    polls = [json.loads(line) for line in (output/f'{game_pk}_polls.jsonl').read_text().splitlines() if line.strip()]
    live = [p for p in polls if p.get('status') == 'ready']
    by_key, compute = {}, []
    for p in live:
        if p.get('rec_key') and p.get('rec_status') in ('ready', 'unsupported'):
            if p['rec_key'] not in by_key:
                by_key[p['rec_key']] = p['rec_status']
                if p.get('rec_seconds'):
                    compute.append(p['rec_seconds'].get('total'))
    count = lambda rows, field: {k: sum(str(r.get(field)) == k for r in rows) for k in sorted({str(r.get(field)) for r in rows})}
    result = {
        'schema': 'pitcheezy.live-rehearsal.v1', 'game_pk': game_pk,
        'note': 'Service timing/coverage measurement on DEMO-WS-2026 data; not a model or policy evaluation.',
        'snapshots': len(files), 'pitches_in_snapshots': len(order), 'pitches_first_seen_live': len(fresh),
        'publication_lag_s': _quantiles(lag),
        'publication_lag_note': f'first fetch of a snapshot containing the pitch minus the pitch startTime (release); includes our poll phase (0..{poll_interval_s:g} s)',
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


def lead_shares(feeds, *, lag_s, poll_s, client_s=2.0, compute_s=0.15, delays=BROADCAST_DELAYS, phases=20):
    """From completed games' recorded pitch times: the share of pitches whose recommendation is on screen at
    least ``READ_S`` before the pitch, for a viewer whose broadcast runs ``delay`` behind the ballpark.

    Times count from each pitch's release (``startTime``, the same in the live and the final feed). The
    state after pitch k-1 reaches the screen ``lag_s + U(0, poll_s) + compute_s + U(0, client_s)`` after its
    release (feed publication, our poll phase, the model, the page's refresh), but never before the viewer's
    own broadcast has shown pitch k-1 (``SEEN_S`` after its release, plus ``delay``). Pitch k appears on that
    broadcast at its release plus ``delay``. Poll phases are averaged over an even grid. Only consecutive
    pitches under 120 s apart count.
    """
    pairs = []
    for feed in feeds:
        starts = sorted(start for start, _ in pitches_of(feed).values())
        pairs += [(start, before) for before, start in zip(starts, starts[1:]) if start-before < 120]
    grid = [(i+.5)/phases for i in range(phases)]
    shares = {}
    for delay in delays:
        hits = 0
        for start, before in pairs:
            for a in grid:
                for b in grid:
                    shown = max(before+SEEN_S+delay, before+lag_s+a*poll_s+compute_s+b*client_s)
                    hits += start+delay-shown >= READ_S
        shares[str(delay)] = round(hits/(len(pairs)*phases*phases), 4) if pairs else None
    return {'pairs': len(pairs), 'lag_s': lag_s, 'poll_s': poll_s, 'client_s': client_s, 'compute_s': compute_s,
            'read_time_s': READ_S, 'share_by_broadcast_delay': shares}


def simulate(feeds_dir, output, *, lags, polls):
    feeds = []
    for path in sorted(Path(feeds_dir).glob('[0-9]*_final.json.gz')):  # skips exFAT '._*' sidecar files
        with gzip.open(path, 'rt') as handle:
            feeds.append(json.load(handle))
    result = {'schema': 'pitcheezy.live-lead-simulation.v1', 'games': len(feeds),
              'note': 'Timing arithmetic on recorded pitch times of completed DEMO-WS-2026 games; not a model evaluation.',
              'grid': [lead_shares(feeds, lag_s=lag, poll_s=poll) for lag in lags for poll in polls]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    return result


def pitch_keys(feed):
    """{policy pitch key 'gamePk:at_bat_number:pitch_number': (start, end)} for every pitch in a feed."""
    out, game_pk = {}, feed['gameData']['game']['pk']
    for play in feed['liveData']['plays'].get('allPlays') or []:
        number = 0
        for event in play.get('playEvents') or []:
            if event.get('isPitch') or (event.get('type') == 'no_pitch' and ((event.get('details') or {}).get('call') or {}).get('code')):
                number += 1
                if event.get('isPitch') and event.get('startTime') and event.get('endTime'):
                    out[f"{game_pk}:{play['about']['atBatIndex']+1}:{number}"] = (_time(event['startTime']), _time(event['endTime']))
    return out


def measured_leads(polls, feed, *, since=None, until=None, delays=BROADCAST_DELAYS):
    """Measured on a real game: for each pitch, when a zero-delay viewer first had its recommendation
    (first poll answering that pitch key with a ready/unsupported recommendation) against when the pitch
    was really thrown. A viewer ``delay`` behind sees the pitch at ``start + delay`` and the recommendation
    no earlier than the previous pitch on their own broadcast (``previous end + delay``)."""
    times = pitch_keys(feed)
    order = sorted(times, key=lambda key: times[key][0])
    before = dict(zip(order[1:], order))
    first = {}
    for poll in polls:
        key = poll.get('rec_key')
        if key and poll.get('rec_status') in ('ready', 'unsupported') and key not in first:
            first[key] = poll['wall']
    # (pitch start, first shown, previous pitch at the plate): startTime is the release and stays fixed in
    # the live feed; endTime is rewritten, so the previous pitch counts as seen SEEN_S after its release.
    rows = [(times[key][0], first[key], times[before[key]][0]+SEEN_S) for key in order
            if key in first and key in before and (since is None or first[key] >= since) and (until is None or first[key] < until)
            and times[key][0]-times[before[key]][0] < 120]
    lead0 = [start-shown for start, shown, _ in rows]
    shares = {str(delay): round(sum(start+delay-max(shown, previous_seen+delay) >= READ_S for start, shown, previous_seen in rows)/len(rows), 4)
              if rows else None for delay in delays}
    return {'pitches': len(rows), 'lead_at_zero_delay_s': _quantiles(lead0),
            'arrival_after_previous_pitch_s': _quantiles([shown-previous_seen for _, shown, previous_seen in rows]),
            'read_time_s': READ_S, 'share_with_read_time_by_broadcast_delay': shares}


# Fixed before the division series recordings were read (plan D149): what a recorded game must show.
CRITERIA = {'poll_errors_max': 0, 'compute_errors_max': 0, 'share_at_30s_min': 0.95, 'same_top_type_first_view_min': 0.95}


def verdict(report, check):
    """[(criterion, passed or None when its file is missing, measured value)] for one recorded game."""
    share = (report or {}).get('share_with_read_time_by_broadcast_delay', {}).get('30')
    compared = None if check is None else check['pitches_seen_pending_live']-len(check['seen_live_not_in_final'])
    same = None if not compared else check['first_view_same_top_type']/compared
    errors = None if report is None else report['polls']['errors']
    broken = None if check is None else len(check['errors'])
    return [('조회 오류 0', None if errors is None else errors <= CRITERIA['poll_errors_max'], errors),
            ('계산 오류 0', None if broken is None else broken <= CRITERIA['compute_errors_max'], broken),
            ('지연 30초 "5초 전 추천" 95% 이상', None if share is None else share >= CRITERIA['share_at_30s_min'], share),
            ('1순위 구종 일치(처음 본 화면) 95% 이상', None if same is None else same >= CRITERIA['same_top_type_first_view_min'],
             None if same is None else round(same, 4))]


def summarize(directory, games, checks=None):
    """One phone-readable Markdown page over several recorded games: their ``report`` files and, when
    present, the live-versus-final checks (``demo_precompute.py live-check``)."""
    directory = Path(directory)
    lines = ['# 실경기 기록 요약', '', '서비스 시간·지원 범위 측정이다(모델·정책 평가 아님). 수치는 각 경기의 `_report.json`·`live_check` 파일에서 옮겼다.',
             '조회 오류가 있으면 서버 재시작 때문인지 `_polls.jsonl`의 오류 시각으로 확인한다(기준은 재시작과 무관한 오류 0).', '']
    for game_pk in games:
        path = directory/f'{int(game_pk)}_report.json'
        lines.append(f'## {int(game_pk)}')
        if not path.is_file():
            lines += ['- 보고서 없음(기록이 끝나지 않았거나 실패). `_record.log`를 확인한다.', '']
            continue
        report = json.loads(path.read_text())
        polls, rec, lag = report['polls'], report['recommendation'], report['publication_lag_s'] or {}
        shares = report['share_with_read_time_by_broadcast_delay']
        lines += [f"- 조회 {polls['total']}회, 오류 {polls['errors']}회, 상태 {polls['status']}",
                  f"- 투구 {report['pitches_first_seen_live']}개를 실시간으로 받음. 공을 놓은 뒤 처음 받기까지: 중앙값 {lag.get('median')}초 (10% {lag.get('p10')}, 90% {lag.get('p90')})",
                  '- 공 5초 전에 추천을 낼 수 있던 비율(받은 시각 기준, 계산 시간 제외): ' + ', '.join(
                      f"지연 {delay}초 {'미측정' if share is None else format(share, '.1%')}" for delay, share in shares.items()),
                  f"- 추천 {rec['decisions_seen']}개: 가능 {rec['decisions_ready']}, 불가 {rec['decisions_unsupported']}. 계산 시간 {rec['compute_total_s']}"]
        check = Path(checks)/f'{int(game_pk)}.json' if checks else None
        if check is not None and check.is_file():
            c = json.loads(check.read_text())
            compared = c['pitches_seen_pending_live']-len(c['seen_live_not_in_final'])
            lines.append(f"- 실시간 화면 대 최종 기록: 비교 {compared}구, 1순위 구종 같음 처음 {c['first_view_same_top_type']}·마지막 {c['last_view_same_top_type']}, "
                         f"완전히 같음 처음 {c['first_view_equals_final']}·마지막 {c['last_view_equals_final']}, 계산 오류 {len(c['errors'])}, 모델 준비 {c['binds']}회")
        else:
            c = None
            lines.append('- 실시간 화면 대 최종 기록 대조: 아직 없음')
        lines.append('- 기준 대조: ' + '; '.join(
            f"{name} → {'미정' if passed is None else '통과' if passed else '미달'}({value})" for name, passed, value in verdict(report, c)))
        lines.append('')
    text = '\n'.join(lines)
    (directory/'summary.md').write_text(text)
    return text


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
    rec.add_argument('--not-before', help='UTC ISO time to start polling at (e.g. ten minutes before first pitch)')
    rep = commands.add_parser('report')
    rep.add_argument('--game-pk', type=int, required=True)
    rep.add_argument('--output', type=Path, required=True)
    rep.add_argument('--snapshots', type=Path, required=True)
    lead = commands.add_parser('leads', help='measured lead times of a recorded real game')
    lead.add_argument('--polls', type=Path, required=True)
    lead.add_argument('--feed', type=Path, required=True, help='a late or final snapshot of the game (.json.gz)')
    lead.add_argument('--since', help='UTC ISO: only recommendations first seen from this time')
    lead.add_argument('--until', help='UTC ISO: only recommendations first seen before this time')
    summary = commands.add_parser('summary', help='one Markdown page over several recorded games')
    summary.add_argument('--dir', type=Path, required=True)
    summary.add_argument('--games', type=int, nargs='+', required=True)
    summary.add_argument('--checks', type=Path, help='directory of live-check {gamePk}.json files')
    sim = commands.add_parser('simulate')
    sim.add_argument('--feeds', type=Path, required=True, help='directory of {gamePk}_final.json.gz feeds')
    sim.add_argument('--output', type=Path, required=True)
    sim.add_argument('--lags', type=float, nargs='+', required=True, help='feed publication lags to try (s)')
    sim.add_argument('--polls', type=float, nargs='+', default=[10, 5, 3, 2])
    args = parser.parse_args(argv)
    if args.command == 'summary':
        print(summarize(args.dir, args.games, args.checks))
        return
    if args.command == 'leads':
        polls = [json.loads(line) for line in args.polls.read_text().splitlines() if line.strip()]
        with gzip.open(args.feed, 'rt') as handle:
            feed = json.load(handle)
        print(json.dumps(measured_leads(polls, feed, since=args.since and _time(args.since), until=args.until and _time(args.until)),
                         ensure_ascii=False, indent=1))
        return
    if args.command == 'simulate':
        result = simulate(args.feeds, args.output, lags=args.lags, polls=args.polls)
        for row in result['grid']:
            print(row['lag_s'], row['poll_s'], row['share_by_broadcast_delay'], flush=True)
        return
    if args.command == 'record':
        poll = lambda server, game_pk: poll_line(server, game_pk, query=args.query)
        if args.not_before:  # no polling (and so no upstream fetches) while the game is hours away
            time.sleep(max(0.0, _time(args.not_before) - time.time()))
        print(json.dumps(record(args.server.rstrip('/'), args.game_pk, args.output, interval=args.interval,
                                max_hours=args.max_hours, poll=poll), ensure_ascii=False), flush=True)
    else:
        print(json.dumps(report(args.game_pk, args.output, args.snapshots), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
