"""Golden bodies of the fan routes for a fixed sequence of synthetic feed states (deterministic clock, fake policy).

Why: with the viewer-delay switch OFF (the default) ``/api/service/games/{pk}`` and ``/api/service/games`` must
answer exactly the baseline bodies. ``fixtures/viewer_off/golden-505932e.json.gz`` holds what commit 505932e answered;
``test_viewer_off_golden.py`` compares the current code against it on every run (its docstring says why the baseline
moved from a72f1fa: intended commits 0fe66af9, a52012eb, 75e25189). ``golden-a72f1fa.json.gz`` is kept as history.

How the current fixture was made (2026-10-19, BLK-06), from the 505932e tree itself:

    env -u PITCHEEZY_OBSERVER_VIEWER_DELAY -u PITCHEEZY_OBSERVER_PROFILE_REFERENCE -u PITCHEEZY_OBSERVER_STEP5_EXPECTATION \
    PYTHONPATH= .venv-observer/bin/python apps/observer/backend/tests/golden_viewer_off.py \
        --backend apps/observer/backend --out apps/observer/backend/tests/fixtures/viewer_off/golden-505932e.json.gz

The JSON is deterministic; the gzip header carries the output file's base name (mtime 0), so the same base name gives
the same bytes. The a72f1fa fixture (2026-10-11) was made out of process against the OLD tree:

    git archive a72f1fa apps/observer | tar -x -C <tmp>
    PYTHONPATH= .venv-observer/bin/python apps/observer/backend/tests/golden_viewer_off.py \
        --backend <tmp>/apps/observer/backend --out apps/observer/backend/tests/fixtures/viewer_off/golden-a72f1fa.json.gz

``--backend`` is imported first, so ``observer_app`` is the code of that tree; the feed states and the harness (``snap``,
``Run`` of ``test_display_state.py``) are this tree's. ``--replay-dir`` runs real recorded bodies
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
    out |= stored(Path(tmp_path)/'stored', extra)
    return out


def stored(tmp_path, extra):
    """Archive and demo bodies (stored final games; no live feed): the same routes, which must not notice ``?delay=``.
    The demo session id is random and an archive body stamps ``feed.as_of`` with the wall clock: both are replaced
    by fixed words before comparing (they are not content)."""
    import gzip as gz
    from fastapi.testclient import TestClient
    from observer_app.main import create_app
    from test_service_demo_api import GAME, Wall, feed, start_demo, watch
    feeds, wdir, demo, intent = (tmp_path/name for name in ('feeds', 'watch', 'demo', 'intent'))
    for folder in (feeds, wdir, demo/'mappings', intent):
        folder.mkdir(parents=True)
    with gz.open(feeds/f'{GAME}_final.json.gz', 'wt') as handle:
        json.dump(feed(GAME, '2026-09-30T02:00:00Z'), handle)
    (wdir/f'{GAME}.json').write_text(json.dumps(watch(GAME)))
    rows = [{'index': i, 'pitch_key': key, 'reveal_at_s': 10.0*i} for i, key in
            enumerate(f'{GAME}:{ab}:{n}' for ab in range(1, 6) for n in (1, 2))]
    (demo/'mappings'/'fx.json').write_text(json.dumps({'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': 'fx', 'game_pk': GAME,
                                                       'kind': 'test_fixture', 'verified': False, 'video': None, 'rows': rows}))
    out = {}
    with TestClient(create_app(object(), start_worker=False, watch_dir=wdir, feed_dir=feeds, demo_dir=demo, intent_dir=intent,
                               demo_wall=Wall(), live_feed=False)) as client:
        def keep(name, path, **params):
            answer = client.get(path, params=params | extra)
            text = json.dumps(answer.json(), ensure_ascii=False, sort_keys=True)
            body = json.loads(text.replace(sid, 'SID') if sid else text)
            if isinstance(body.get('feed'), dict) and body['source']['kind'] != 'live':
                body['feed']['as_of'] = 'WALL-CLOCK'
            out[name] = {'status': answer.status_code, 'body': body}
        sid = None
        keep('20-archive-game', f'/api/service/games/{GAME}')
        keep('21-archive-cursor', f'/api/service/games/{GAME}', after=6)
        keep('22-archive-list', '/api/service/games', date='2026-09-30')
        keep('23-dates', '/api/service/dates')
        keep('24-unknown-game', f'/api/service/games/{GAME+50}')
        keep('25-bad-date', '/api/service/games', date='9/30')
        sid = start_demo(client, demo)[0]['session_id']
        keep('30-demo-game', f'/api/service/games/{GAME}', demo=sid)
        keep('31-demo-list', '/api/service/games', date='2026-09-30', demo=sid)
        keep('32-embargoed-archive-game', f'/api/service/games/{GAME}')
        keep('33-embargoed-list', '/api/service/games', date='2026-09-30')
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
