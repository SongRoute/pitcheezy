"""Freeze the exact /api responses the showcase needs into files (GET only, bounded).

Fallback for the 10/6 demo when the Mac mini or the tailnet is unreachable: serve_offline.py replays these
bytes on the MacBook next to the same static build. Nothing is recomputed or invented; a request outside the
snapshot gets a clear 404 there. Stdlib only.

    python3 capture_snapshot.py --base http://100.108.252.111:8771 --scenes ../src/showcase-scenes.json --out DIR
    python3 capture_snapshot.py ... --all-decisions   # every pitch of the game, so watch-page ←/→ never runs out
"""
import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOP = 5  # WatchSummaryPanel.tsx `TOP`
MAX_REQUESTS = 400
MAX_REQUESTS_ALL = 800  # --all-decisions: 1 timeline + reveal and summary per decision (849843: 262 -> 525)


def needed_paths(timeline, scenes, top=TOP):
    """API paths the showcase + embedded watch request for every pinned scene.

    Embedded watch at decision i fetches reveal j for the earlier pitches of the same plate appearance
    (paStart..i-1), reveal i on explicit reveal, then summary?through=i&top=TOP.
    """
    game_pk = timeline['game']['game_pk']
    decisions = timeline['decisions']
    by_key = {(d['at_bat_number'], d['pitch_number']): d['index'] for d in decisions}
    reveals, summaries, missing = set(), set(), []
    for scene in scenes:
        i = by_key.get((scene['atBat'], scene['pitchNumber']))
        if i is None:
            missing.append(scene['pitchId'])
            continue
        start = i
        while start > 0 and decisions[start - 1]['pa_id'] == decisions[i]['pa_id']:
            start -= 1
        reveals.update(range(start, i + 1))
        summaries.add(i)
    paths = [f'/api/watch/{game_pk}/reveal/{j}' for j in sorted(reveals)]
    paths += [f'/api/watch/{game_pk}/summary?through={i}&top={top}' for i in sorted(summaries)]
    return paths, missing


def all_decision_paths(timeline, top=TOP):
    """Every reveal and every summary?through=i the watch page can request after an explicit reveal."""
    game_pk = timeline['game']['game_pk']
    indices = [d['index'] for d in timeline['decisions']]
    if indices != list(range(len(indices))):
        raise SystemExit('timeline decision indices are not 0..N-1')
    return ([f'/api/watch/{game_pk}/reveal/{i}' for i in indices]
            + [f'/api/watch/{game_pk}/summary?through={i}&top={top}' for i in indices])


def snapshot_file(path):
    """Relative file for an API path; serve_offline.py uses the same mapping."""
    route, _, query = path.partition('?')
    name = route.strip('/').replace('/', '__')
    if query:
        name += '__' + query.replace('&', '__').replace('=', '-')
    return f'api/{name}.json'


def fetch(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--scenes', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--all-decisions', action='store_true',
                        help='save every reveal and summary of the game, not only the scene paths')
    args = parser.parse_args()
    cap = MAX_REQUESTS_ALL if args.all_decisions else MAX_REQUESTS
    base = args.base.rstrip('/')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'refusing to overwrite non-empty {out}')
    source = json.loads(Path(args.scenes).read_text())
    started = datetime.now(timezone.utc)
    entries = []

    def save(path):
        if len(entries) >= cap:
            raise SystemExit(f'request cap {cap} reached')
        status, body = fetch(base + path)
        if status != 200:
            raise SystemExit(f'{path}: HTTP {status}')
        json.loads(body)  # must be JSON
        rel = snapshot_file(path)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        entries.append({'path': path, 'file': rel, 'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
        time.sleep(0.02)
        return body

    timeline = json.loads(save(f'/api/watch/{source["gamePk"]}'))
    if timeline['game']['game_pk'] != source['gamePk']:
        raise SystemExit('timeline game does not match scenes')
    games = json.loads(fetch(base + '/api/watch/games')[1])
    paths, missing = needed_paths(timeline, source['scenes'])
    if missing:
        raise SystemExit(f'scenes without a decision: {missing}')
    if args.all_decisions:
        scene_paths, paths = set(paths), all_decision_paths(timeline)
        if not scene_paths <= set(paths):
            raise SystemExit('scene paths not covered by all decisions')
        if len(paths) + 1 > cap:
            raise SystemExit(f'{len(paths) + 1} requests exceed cap {cap}')
    for path in paths:
        save(path)
    # Picker list: only the snapshot game, so no link leads to a game that is not in the bundle.
    listed = [g for g in games['games'] if g['game_pk'] == source['gamePk']]
    body = json.dumps({'games': listed}, ensure_ascii=False).encode()
    (out / 'api').mkdir(parents=True, exist_ok=True)
    (out / snapshot_file('/api/watch/games')).write_bytes(body)
    entries.append({'path': '/api/watch/games', 'file': snapshot_file('/api/watch/games'), 'bytes': len(body),
                    'sha256': hashlib.sha256(body).hexdigest(), 'note': f'filtered to game {source["gamePk"]}'})
    finished = datetime.now(timezone.utc)
    kst = timezone(timedelta(hours=9))
    manifest = {'schema': 'pitcheezy.offline-snapshot.v1', 'base': base, 'game_pk': source['gamePk'],
                'scenes': len(source['scenes']), 'top': TOP,
                'coverage': 'all_decisions' if args.all_decisions else 'scenes',
                'decisions': len(timeline['decisions']),
                'captured_utc': started.isoformat(timespec='seconds'),
                'captured_kst': started.astimezone(kst).strftime('%Y-%m-%d %H:%M KST'),
                'finished_utc': finished.isoformat(timespec='seconds'),
                'scenes_sha256': hashlib.sha256(Path(args.scenes).read_bytes()).hexdigest(),
                'files': entries}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(e['bytes'] for e in entries),
                      'captured_kst': manifest['captured_kst']}))


if __name__ == '__main__':
    main()
