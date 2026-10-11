"""A synthetic world for the public publisher: one finished archive game (with stored recommendations and a
validated setup-estimate file), one live game behind a store-backed feed and a fake policy, an injected clock and
a schedule that lists the live game. No network, no T7, no model. ``export`` writes the contract examples and
schemas from it (``scripts/export_publish_contract.py``; ``test_publisher.py`` checks the committed copies)."""
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from observer_app import publish_schema, publisher, service_api
from observer_app.live_armb import LiveArmB
from observer_app.live_feed import LiveFeed
from observer_app.live_store import LiveSnapshotStore, PersistentLiveFeed
from test_display_state import GAME as LIVE_GAME, Policy, snap as _snap
from test_service_demo_api import feed as final_feed, watch
import test_setup_estimates as setup_fixture

ARCHIVE_GAME = 900101
ARCHIVE_DATE, LIVE_DATE = '2026-09-30', '2026-10-05'  # KST dates of the two games
BASE = datetime(2026, 10, 5, 10, 20, tzinfo=timezone.utc).timestamp()  # "now": 19:20 KST on the live game's day
COMMIT = ('0'*40, False)  # the examples never carry a real commit


def snap(ts='20261005_100000', *, wall=BASE-5, **options):
    return _snap(wall, ts, **options)


class Upstream:
    """The MLB API as the feed sees it: the schedule of the live game's day and the game's current body."""

    def __init__(self, body):
        self.body, self.calls = body, []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if 'schedule' in url:
            if f'date={LIVE_DATE}' not in url:
                return {'dates': []}
            status = self.body['gameData']['status'] if isinstance(self.body, dict) else {'abstractGameState': 'Live', 'detailedState': 'In Progress'}
            return {'dates': [{'games': [{'gamePk': LIVE_GAME, 'gameType': 'F', 'gameDate': '2026-10-05T10:00:00Z',
                                          'officialDate': LIVE_DATE, 'status': status, 'gameNumber': 1, 'doubleHeader': 'N',
                                          'teams': {'away': {'team': {'name': 'Away Club'}, 'score': 0},
                                                    'home': {'team': {'name': 'Home Club'}, 'score': 1}}}]}]}
        if isinstance(self.body, Exception):
            raise self.body
        return copy.deepcopy(self.body)


class _ServiceTime:
    """``service_api``'s clock: the world's injected time instead of the machine's (bodies become repeatable)."""
    strftime = staticmethod(time.strftime)

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now[0]

    def gmtime(self, seconds=None):
        return time.gmtime(self.now[0] if seconds is None else seconds)


class World:
    def __init__(self, tmp, *, store=True, first=None, setup=True, **config):
        self.tmp, self.now, self.policy = Path(tmp), [BASE], Policy()
        self.upstream = Upstream(snap() if first is None else first)
        self.feeds, self.watch, self.demo, self.intent, self.out = (self.tmp/name for name in ('feeds', 'watch', 'demo', 'intent', 'out'))
        for folder in (self.feeds, self.watch, self.demo/'mappings', self.intent):
            folder.mkdir(parents=True)
        with gzip.open(self.feeds/f'{ARCHIVE_GAME}_final.json.gz', 'wt') as handle:
            json.dump(final_feed(ARCHIVE_GAME, '2026-09-30T02:00:00Z'), handle)
        (self.watch/f'{ARCHIVE_GAME}.json').write_text(json.dumps(watch(ARCHIVE_GAME)))
        rows = [{'index': i, 'pitch_key': key, 'reveal_at_s': 10.0*i} for i, key in
                enumerate(f'{ARCHIVE_GAME}:{ab}:{n}' for ab in range(1, 6) for n in (1, 2))]
        (self.demo/'mappings'/'fx.json').write_text(json.dumps({
            'schema': 'pitcheezy-demo-mapping-v1', 'mapping_id': 'fx', 'game_pk': ARCHIVE_GAME, 'kind': 'test_fixture',
            'verified': False, 'video': None, 'rows': rows}))
        if setup:
            self.write_setup()
        wall = lambda: self.now[0]
        options = dict(delay_s=0, clock=wall, wall=wall, sleep=lambda _: None)
        self.feed = (PersistentLiveFeed(self.upstream, store=LiveSnapshotStore(self.tmp/'live.sqlite3'), source='live_poll', **options)
                     if store else LiveFeed(self.upstream, **options))
        self.armb = LiveArmB(lambda: self.policy)
        self.config = publisher.load_config(env={}, out_dir=str(self.out), **config)
        self._time, service_api.time = service_api.time, _ServiceTime(self.now)
        self.publisher = None
        try:
            self.start()
        except BaseException:
            service_api.time = self._time
            raise

    def start(self, **extra):
        """(Re)start the publisher process over the same output folder, state file, store and clock."""
        if getattr(self, 'publisher', None) is not None:
            self.publisher.close()  # the old process is gone: its lock with it
        self.app, self.publisher = publisher.build(
            self.config, wall=lambda: self.now[0], commit=COMMIT, live_feed=self.feed, live_policy=self.armb,
            watch_dir=self.watch, feed_dir=self.feeds, demo_dir=self.demo, intent_dir=self.intent,
            demo_wall=lambda: self.now[0], **extra)
        self.client = TestClient(self.app)

    def close(self):
        service_api.time = self._time
        if self.publisher is not None:
            self.publisher.close()

    def write_setup(self, lines=None, game=ARCHIVE_GAME, **pin):
        est = lambda ab, n, x: setup_fixture.estimate(ab, n, x) | {'pitch_id': f'{game}:{ab}:{n}'}
        lines = [est(1, 1, .1), est(2, 2, -.5), est(5, 2, 1.2)] if lines is None else lines
        raw = ''.join(json.dumps(line)+'\n' for line in lines).encode()
        (self.intent/f'{game}.jsonl').write_bytes(raw)
        (self.intent/f'{game}.source.json').write_text(json.dumps(
            setup_fixture.SOURCE | {'sha256': hashlib.sha256(raw).hexdigest()} | pin))

    def settle(self):
        deadline = time.time()+5
        while time.time() < deadline and (self.armb.pending or self.armb.running is not None or self.armb.loading):
            time.sleep(.01)

    def cycle(self, body=None, *, after=12.0, settle=True):
        """Advance past the poll interval, optionally change what upstream answers, and publish. With ``settle``
        the fake policy gets its (instant) turn and a second cycle at the same moment publishes the settled view,
        as two successive real cycles would."""
        if body is not None:
            self.upstream.body = body
        self.now[0] += after
        health = self.publisher.cycle()
        if settle:
            self.settle()
            health = self.publisher.cycle()
        return health

    def read_after_cycle(self, name):
        self.cycle(settle=False)
        return self.read(name)

    def path(self, name):
        return self.out/name

    def read(self, name):
        return json.loads(self.path(name).read_text())

    def game(self, game_pk=LIVE_GAME):
        return self.read(f'v1/game/{game_pk}.json')

    def files(self):
        return sorted(str(path.relative_to(self.out)) for path in self.out.rglob('*') if path.is_file())

    def served(self, path, **params):
        answer = self.client.get(path, params=params)
        assert answer.status_code == 200, answer.text
        return answer.json()


def strip(body):
    """A published file without the key the publisher added: what the route answers."""
    return {key: value for key, value in body.items() if key != 'publish'}


# ---- contract examples and schemas -------------------------------------------------------------------------
KINDS = {'manifest': False, 'health': False, 'dates': False, 'games': False, 'game': True, 'setup': True}  # kind -> has final mark
def pregame(ts='20261005_100000', *, wall=BASE-5):
    """The live game before its first pitch: no play yet."""
    body = snap(ts, wall=wall, half='preview')
    body['liveData']['plays'] = {'allPlays': []}
    body['liveData']['linescore'] = {'teams': {'away': {'runs': 0}, 'home': {'runs': 0}}}
    return body


def unidentified(body):
    """``body`` with its newest pitch lacking the provider identity: the store does not admit it (확인 중)."""
    del body['liveData']['plays']['currentPlay']['playEvents'][-1]['playId']
    return body


def _broken_ledger(**_):
    raise sqlite3.OperationalError('disk I/O error')


def collect(tmp):
    """``{kind: [(example file name, published body), ...]}``: one live game walked from before its first pitch to
    its end through the states the fan screen has, and the archive game. Every body goes
    into the schemas; ``EXAMPLES`` names the ones written out as example files."""
    tmp = Path(tmp)
    out = {kind: [] for kind in KINDS}
    world = World(tmp/'main', first=pregame())
    stamps = iter(f'20261005_10{minute:02d}00' for minute in range(1, 60))
    # 30 s apart: past the feed's poll interval and the 20 s schedule cache of the game list
    body = lambda **options: snap(next(stamps), wall=world.now[0]+25, **options)

    def take(name, upstream=None):
        world.cycle(upstream, after=30.0)
        out['game'].append((f'game-{name}.json', world.game()))
        out['games'].append((f'games-{name}.json', world.read(f'v1/games/{LIVE_DATE}.json')))
        out['health'].append((f'health-{name}.json', world.read('v1/health.json')))
        out['manifest'].append((f'manifest-{name}.json', world.read('v1/manifest.json')))
    try:
        take('pregame')
        take('live', body())
        take('live-corrected', body(call='C'))  # the provider changes the first pitch: a visible correction
        take('live-second-pitch', body(call='C', second=True))
        take('live-unconfirmed', unidentified(body(call='C', second=True, pid='g5p1b')))
        world.policy.status = 'UNSUPPORTED_PITCHER'
        take('live-unsupported', body(call='C', second=True))
        world.policy.status = None
        real, world.feed.store.append_recommendation = world.feed.store.append_recommendation, _broken_ledger
        take('live-ledger-unavailable', body(second=True))
        world.feed.store.append_recommendation = real
        take('disconnected', OSError('upstream down'))
        take('between-innings', body(half='between'))
        take('final', body(half='final'))
        out['game'].append(('game-final-archive.json', world.game(ARCHIVE_GAME)))
        out['games'].append(('games-archive.json', world.read(f'v1/games/{ARCHIVE_DATE}.json')))
        out['dates'].append(('dates.json', world.read('v1/dates.json')))
        out['setup'].append(('setup.json', world.read(f'v1/setup/{ARCHIVE_GAME}.json')))
    finally:
        world.close()
    failing = World(tmp/'failing', first=OSError('upstream down'))  # the first poll fails: health names the game
    try:
        failing.cycle(settle=False)
        out['health'].append(('health-game-unavailable.json', failing.read('v1/health.json')))
    finally:
        failing.close()
    return out


EXAMPLES = {'manifest': ('manifest-live.json',), 'health': ('health-live.json',), 'dates': ('dates.json',),
            'games': ('games-live.json', 'games-archive.json'),
            'game': ('game-pregame.json', 'game-live.json', 'game-live-unconfirmed.json', 'game-disconnected.json',
                     'game-between-innings.json', 'game-final.json', 'game-final-archive.json'),
            'setup': ('setup.json',)}


def render(tmp):
    """``{relative file name: text}`` of the contract folder: ``examples/*.json`` and ``schema/*.schema.json``."""
    collected, files = collect(tmp), {}
    dump = lambda value: json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True)+'\n'
    for kind, final in KINDS.items():
        bodies = dict(collected[kind])
        for name in EXAMPLES[kind]:
            files[f'examples/{name}'] = dump(bodies[name])
        files[f'schema/{kind}.schema.json'] = dump(publish_schema.document(
            kind, list(bodies.values()), final=final, source='apps/observer/backend/tests/publish_fixture.py'))
    return files


def export(target, tmp):
    target = Path(target)
    for name, text in render(tmp).items():
        (target/name).parent.mkdir(parents=True, exist_ok=True)
        (target/name).write_text(text)
    return target
