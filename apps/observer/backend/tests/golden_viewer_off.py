"""Golden bodies of the fan routes for a fixed sequence of synthetic feed states (deterministic clock, fake policy).

Why: with the viewer-delay switch OFF (the default) ``/api/service/games/{pk}`` and ``/api/service/games`` must
answer exactly what the delay-0-fixed code of commit a72f1fa answered. ``fixtures/viewer_off/golden-a72f1fa.json.gz``
holds what a72f1fa answered; ``test_viewer_off_golden.py`` compares the current code against it on every run.

How the fixture was made (2026-10-11), out of process against the OLD tree:

    git archive a72f1fa apps/observer | tar -x -C <tmp>
    PYTHONPATH= .venv-observer/bin/python apps/observer/backend/tests/golden_viewer_off.py \
        --backend <tmp>/apps/observer/backend --out apps/observer/backend/tests/fixtures/viewer_off/golden-a72f1fa.json.gz

``--backend`` is imported first, so ``observer_app`` is the old code; the feed states and the harness (``snap``, ``Run``
of ``test_display_state.py``, unchanged since a72f1fa) are this tree's. ``--replay-dir`` runs real recorded bodies
instead (one request per body; not committed: used once to compare both trees on the 849831 prefix)."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile


def scenario(tmp_path, extra_params=None):
    """``{step name: response body}`` of the fan routes over the fixed sequence. ``extra_params`` are added to every
    game and list request (the current code's ``?delay=``, which must change nothing while the switch is off)."""
    from test_display_state import GAME, Run, snap, with_pickoff
    run, out, extra = Run(Path(tmp_path)), {}, dict(extra_params or {})
    row = lambda pk, abstract, status: {
        'game_pk': pk, 'game_type': 'W', 'start': '2026-10-05T10:00:00Z', 'status': status, 'live': abstract == 'Live',
        'away_team': 'A', 'home_team': 'B', 'abstract': abstract, 'official_date': '2026-10-05', 'away_score': 2, 'home_score': 5,
        'game_number': 1, 'double_header': 'N', 'inning': 7, 'half': 'Top'}
    run.feed.schedule = lambda day: ([row(1, 'Live', 'In Progress'), row(2, 'Final', 'Final'), row(3, 'Preview', 'Scheduled')]
                                     if day == '2026-10-05' else [])
    game = f'/api/service/games/{GAME}'

    def ask(name, body=None, advance=12.0, **params):
        if body is not None:
            run.upstream.body = body
        run.now[0] += advance
        answer = run.client.get(game, params=params | extra)
        import time
        deadline = time.time()+5
        while time.time() < deadline and (run.armb.pending or run.armb.running is not None or run.armb.loading):
            time.sleep(.01)
        out[name+':first'] = {'status': answer.status_code, 'body': answer.json()}
        answer = run.client.get(game, params=params | extra)
        out[name] = {'status': answer.status_code, 'body': answer.json()}

    ask('01-pending')
    ask('02-pitch-thrown', snap(1015.0, '20261005_100100', second=True))
    ask('03-cursor', advance=0.0, after=8)
    ask('04-first-pitch-corrected', snap(1015.0, '20261005_100200', second=True, call='C'))
    ask('05-same-stamp-conflict', snap(1040.0, '20261005_100200', second=True))
    ask('06-pickoff', with_pickoff(1055.0, '20261005_100400'))
    ask('07-stale-stamp', snap(1070.0, '20261005_095900', second=True))
    ask('08-between-innings', snap(1085.0, '20261005_100600', half='between'))
    ask('09-final', snap(1100.0, '20261005_100700', half='final'))
    listed = run.client.get('/api/service/games', params={'date': '2026-10-05'} | extra)
    out['10-list'] = {'status': listed.status_code, 'body': listed.json()}
    return out


def replayed(tmp_path, directory, extra_params=None):
    """The same over a folder of recorded bodies (``ReplayTransport`` order), one poll per body."""
    from observer_app.live_armb import LiveArmB
    from observer_app.live_feed import ReplayTransport
    from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
    from observer_app.main import create_app
    from fastapi.testclient import TestClient
    from test_display_state import Policy
    import re
    import time
    files = sorted(Path(directory).glob('[0-9]*_*.json*'))
    pk, now, tmp = int(re.match(r'(\d+)_', files[0].name)[1]), [1_800_000_000.0], Path(tmp_path)

    class SafePolicy(Policy):
        def __call__(self, snapshot):
            try:
                return super().__call__(snapshot)
            except (KeyError, IndexError, TypeError):
                return None
    feed = PersistentLiveFeed(ReplayTransport(directory), store=LiveSnapshotStore(tmp/'live.sqlite3'), source='replay_poll',
                              delay_s=0, clock=lambda: now[0], wall=lambda: now[0], sleep=lambda _: None)
    armb = LiveArmB(lambda: SafePolicy())
    client = TestClient(create_app(object(), start_worker=False, live_feed=feed, live_policy=armb, watch_dir=tmp, feed_dir=tmp,
                                   demo_dir=tmp/'demo', intent_dir=tmp))
    out = {}
    for index, path in enumerate(files):
        now[0] += 12.0
        client.get(f'/api/service/games/{pk}', params=extra_params or {})
        deadline = time.time()+5
        while time.time() < deadline and (armb.pending or armb.running is not None or armb.loading):
            time.sleep(.01)
        answer = client.get(f'/api/service/games/{pk}', params=extra_params or {})
        out[f'{index:03d}-{path.name}'] = {'status': answer.status_code, 'body': answer.json()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--backend', required=True, help='backend directory whose observer_app is run')
    parser.add_argument('--out', required=True)
    parser.add_argument('--replay-dir')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.backend).resolve()))
    import observer_app.main  # noqa: F401 - bind observer_app to --backend before the harness is imported
    assert Path(observer_app.main.__file__).resolve().is_relative_to(Path(args.backend).resolve())
    sys.path.append(str(Path(__file__).resolve().parent))
    with tempfile.TemporaryDirectory() as tmp:
        bodies = replayed(tmp, args.replay_dir) if args.replay_dir else scenario(tmp)
    with gzip.GzipFile(args.out, 'wb', mtime=0) as handle:
        handle.write(json.dumps(bodies, ensure_ascii=False, sort_keys=True, indent=0).encode())
    print(f'{len(bodies)} bodies -> {args.out}')


if __name__ == '__main__':
    main()
