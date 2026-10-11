"""Public publisher: writes, with no viewer request, the bodies a viewer would have fetched, as static files.

    python -m observer_app.publisher --out <dir> [--once] [--date YYYY-MM-DD ...]

Each cycle the publisher asks the SAME builders the ``/api/service/**`` routes call (``service_api.ServiceBodies``)
for the dates, each date's game list and each game's full body, and hands them to a sink:

    v1/manifest.json   contract version, code commit, published_at, publish interval, fixed delay
    v1/health.json     last publish time and sequence (written last: a cycle is complete when it changes)
    v1/dates.json      GET /api/service/dates
    v1/games/<YYYY-MM-DD>.json   GET /api/service/games?date=
    v1/game/<game_pk>.json       GET /api/service/games/{pk}   (the full body, never ``after``)
    v1/setup/<game_pk>.json      GET /api/watch/{pk}/setup, finished games whose estimates pass validation only

What is shown is decided by those builders alone (display gate, unconfirmed pitches, the recommendation ledger
written before the body is returned): nothing here selects or changes a field. A body the route would refuse
(423 demo embargo, 503 feed) is not written. Every file gets one added top-level key and nothing else changes:

    "publish": {"contract": "v1", "sequence": <int>, "published_at": "<UTC ISO>"[, "final": <bool>]}

``sequence`` is one counter per publisher, raised once per cycle; a file carries the sequence of the cycle that
last wrote it. ``final`` exists on game files: once a game is written with ``final: true`` it is not rewritten
unless its content changes (a correction), which gets the then-current, higher sequence.

Calling the game builder is what drives the upstream poll (``LiveFeed.snapshot`` fetches when its interval has
passed), so the publisher is the "someone is watching". Demo sessions, the operator API and replay clocks are
never asked for. Uploading is not done here: ``Sink`` is the seam; only ``LocalDirSink`` exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

from .service_api import Refused

LOGGER = logging.getLogger(__name__)
APP = Path(__file__).resolve().parents[2]
CONFIG_PATH = APP/'publish_config.json'
KST = timezone(timedelta(hours=9))
PREFIX = 'v1'
LAYOUT = {'manifest': f'{PREFIX}/manifest.json', 'health': f'{PREFIX}/health.json', 'dates': f'{PREFIX}/dates.json',
          'games': f'{PREFIX}/games/{{date}}.json', 'game': f'{PREFIX}/game/{{game_pk}}.json',
          'setup': f'{PREFIX}/setup/{{game_pk}}.json'}
ENV = {'out_dir': ('PITCHEEZY_PUBLISH_OUT_DIR', str), 'state_path': ('PITCHEEZY_PUBLISH_STATE', str),
       'publish_interval_s': ('PITCHEEZY_PUBLISH_INTERVAL_S', float), 'fixed_delay_s': ('PITCHEEZY_PUBLISH_DELAY_S', float)}
PROVISIONAL = ('publish_interval_s', 'settled_recheck_s', 'live_days_back', 'live_days_ahead')


def load_config(path=None, env=None, **overrides):
    """``publish_config.json`` <- environment <- explicit overrides (None = not given)."""
    config = json.loads(Path(path or CONFIG_PATH).read_text())
    env = os.environ if env is None else env
    for key, (name, kind) in ENV.items():
        if env.get(name) not in (None, ''):
            config[key] = kind(env[name])
    config |= {key: value for key, value in overrides.items() if value is not None}
    interval, delay, recheck = config['publish_interval_s'], config['fixed_delay_s'], config['settled_recheck_s']
    if not (isinstance(interval, (int, float)) and 0 < interval <= 3600):  # also refuses NaN
        raise ValueError('publish_interval_s must be between 0 (exclusive) and 3600 seconds')
    if not (isinstance(delay, (int, float)) and 0 <= delay <= 3600):
        raise ValueError('fixed_delay_s must be between 0 and 3600 seconds')
    if recheck is not None and not (isinstance(recheck, (int, float)) and recheck >= 0):
        raise ValueError('settled_recheck_s must be null or a number of seconds')
    for key in ('live_days_back', 'live_days_ahead'):
        if type(config[key]) is not int or not 0 <= config[key] <= 7:
            raise ValueError(f'{key} must be an integer between 0 and 7')
    return config


class Sink:
    """Where published files go. ``path`` is always a relative POSIX path such as ``v1/game/123.json``.

    ``write`` must replace the whole file at once (a reader sees the old or the new content, never a part).
    ``final`` says the content is not expected to change again (an uploader may cache it longer). An uploader to a
    public host implements these three calls; the publisher's build logic does not change."""

    def write(self, path, data, *, final=False):
        raise NotImplementedError

    def read(self, path):
        """The bytes last written at ``path``, or None if there are none or this sink cannot read back."""
        return None

    def delete(self, path):
        raise NotImplementedError


class LocalDirSink(Sink):
    """Files under one local directory, each replaced atomically (temporary file in the same folder + rename)."""

    def __init__(self, root):
        self.root = Path(root)

    def _target(self, path):
        parts = PurePosixPath(path).parts
        if not parts or PurePosixPath(path).is_absolute() or any(part in ('..', '.') or part.startswith('.') for part in parts):
            raise ValueError(f'not a relative publish path: {path}')
        return self.root.joinpath(*parts)

    def write(self, path, data, *, final=False):
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')  # a dot file: never one of the published names
        try:
            with open(temporary, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def read(self, path):
        target = self._target(path)
        return target.read_bytes() if target.is_file() else None

    def delete(self, path):
        self._target(path).unlink(missing_ok=True)


def code_commit(repo=None, env=None):
    """(commit sha or None, dirty flag or None) of the code that publishes. Never guessed: None when unknown."""
    env = os.environ if env is None else env
    if env.get('PITCHEEZY_PUBLISH_COMMIT'):
        return env['PITCHEEZY_PUBLISH_COMMIT'], None
    repo = repo or APP.parents[1]
    try:
        head = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10, check=True)
        dirty = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None, None
    return head.stdout.strip() or None, bool(dirty.stdout.strip())


def _iso(epoch):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(epoch))


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def is_final(body):
    """True when a game body is finished and settled: the builder calls the game final and no pitch of it is
    still unconfirmed (a body with a pitch in 확인 중 keeps being rewritten until the store admits it)."""
    state, receipts = body.get('display_state'), body.get('receipts')
    return bool(body['game'].get('status') == 'final' and (body.get('current') or {}).get('phase') == 'final'
                and (state is None or state.get('state') == 'final')
                and (receipts is None or receipts.get('unconfirmed') == 0))


class NotPublic(RuntimeError):
    """A body that is not the plain public view reached the publisher (it is never written)."""


def check_public(body):
    """Last line of defence, not the gate: the builders are only ever asked for the non-demo view, and this
    refuses to write anything that says otherwise."""
    source = body.get('source') or {}
    if source.get('kind') not in ('archive', 'live') or source.get('session_id') is not None or 'publish' in body:
        raise NotPublic(f"refusing to publish a body with source {source!r}")
    return body


class Publisher:
    """One publish cycle at a time. ``bodies`` is ``service_api.ServiceBodies`` of the app whose routes would
    otherwise answer; ``setups`` is its ``SetupEstimates``; ``dates`` (KST ``YYYY-MM-DD``) replaces the live
    window. ``state_path`` keeps the sequence and what was written across restarts."""

    def __init__(self, bodies, sink, *, config, setups=None, state_path=None, wall=time.time, commit=(None, None), dates=None):
        self.bodies, self.sink, self.config, self.setups, self.wall = bodies, sink, config, setups, wall
        self.commit, self.dates = commit, None if dates is None else sorted(set(dates))
        for date in self.dates or []:
            if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date) is None:
                raise ValueError('date must be YYYY-MM-DD')
        self.state_path = None if state_path is None else Path(state_path)
        self.state = self._load_state()

    # ---- bookkeeping --------------------------------------------------------------------------------
    def _load_state(self):
        state = {'sequence': 0, 'files': {}, 'checked': {}}
        if self.state_path is not None and self.state_path.is_file():
            state |= json.loads(self.state_path.read_text())
        try:  # the sequence never goes back, even if the state file was lost: the last health file knows it too
            health = self.sink.read(LAYOUT['health'])
            state['sequence'] = max(state['sequence'], int(json.loads(health)['sequence'])) if health else state['sequence']
        except (ValueError, KeyError, TypeError, OSError):
            LOGGER.warning('published health file unreadable; sequence continues from the state file only', exc_info=True)
        return state

    def _save_state(self):
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f'.{self.state_path.name}.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(self.state, sort_keys=True))
        os.replace(temporary, self.state_path)

    def _due(self, path, now):
        """A settled file is rebuilt when it was never tried or ``settled_recheck_s`` has passed since."""
        checked, every = self.state['checked'].get(path), self.config['settled_recheck_s']
        return checked is None or (every is not None and now-checked >= every)

    def live_dates(self, now):
        if self.dates is not None:
            return list(self.dates)
        today = datetime.fromtimestamp(now, KST).date()
        return [(today+timedelta(days=offset)).isoformat()
                for offset in range(-self.config['live_days_back'], self.config['live_days_ahead']+1)]

    def _put(self, path, body, *, always, final=None, volatile=()):
        """Write ``body`` + the ``publish`` key at ``path``. ``always`` False: only if the content (without the
        ``volatile`` keys, which change on every build) or its final mark differs from what is published there."""
        digest = _digest({key: value for key, value in body.items() if key not in volatile})
        known = self.state['files'].get(path)
        self.state['checked'][path] = self._now
        if not always and known and known['digest'] == digest and known.get('final') == final:
            return False
        if 'publish' in body:
            raise NotPublic(f'{path}: the body already has a publish key')
        publish = {'contract': self.config['contract_version'], 'sequence': self._sequence, 'published_at': _iso(self._now)}
        if final is not None:
            publish['final'] = final
        data = json.dumps(body | {'publish': publish}, ensure_ascii=False, separators=(',', ':')).encode()
        self.sink.write(path, data, final=bool(final))
        self.state['files'][path] = {'digest': digest, 'sequence': self._sequence, 'final': final}
        self._written.append(path)
        return True

    def _drop(self, path):
        if path in self.state['files']:
            self.sink.delete(path)
            del self.state['files'][path]

    # ---- one cycle ----------------------------------------------------------------------------------
    def cycle(self):
        """Build and write everything that is due. Returns the health body of this cycle."""
        self._now, self._written = self.wall(), []
        self._sequence = self.state['sequence'] = self.state['sequence']+1
        self._save_state()  # the number is taken before any file carries it: a crash never reuses it
        now, unavailable = self._now, []
        dates_body = check_public(self.bodies.dates())
        live_dates = self.live_dates(now)
        fast, listed = set(), set()
        for date in sorted(set(dates_body['dates']) | set(live_dates)):
            path, live = LAYOUT['games'].format(date=date), date in live_dates
            if not live and not self._due(path, now):
                continue
            try:
                body = check_public(self.bodies.games(date))
            except Exception:  # noqa: BLE001 - one date never stops the cycle; its last file stays
                LOGGER.warning('game list of %s not built; the published one is kept', date, exc_info=True)
                self.state['checked'][path] = now
                continue
            self._put(path, body, always=live)
            for card in body['games']:
                if card.get('embargoed'):  # a demo is running on it: its game body is refused anyway
                    continue
                listed.add(int(card['game_pk']))
                if live:
                    fast.add(int(card['game_pk']))
        known = {int(PurePosixPath(path).stem) for path in self.state['files'] if path.startswith(f'{PREFIX}/game/')}
        for game_pk in sorted(listed | known):
            path = LAYOUT['game'].format(game_pk=game_pk)
            settled = (self.state['files'].get(path) or {}).get('final') is True
            if (settled or game_pk not in fast) and not self._due(path, now):
                continue
            self.state['checked'][path] = now
            try:
                body = check_public(self.bodies.game(game_pk))  # polls upstream; ledger is written in here
            except Refused as refused:
                if refused.status != 423:  # a demo embargo is not a fault and is not reported
                    unavailable.append({'game_pk': game_pk, 'status': refused.status})
                continue
            except NotPublic:
                raise
            except Exception:  # noqa: BLE001 - one game never stops the cycle; its last file stays
                LOGGER.warning('game %s not built; the published file is kept', game_pk, exc_info=True)
                unavailable.append({'game_pk': game_pk, 'status': None})
                continue
            final = is_final(body)
            # ``feed`` (as_of / stale_s) moves with the clock, not with the game: it never makes a final file new
            self._put(path, body, always=not final, final=final, volatile=('feed',))
            if final:
                self._setup(game_pk)
        self._put(LAYOUT['dates'], dates_body, always=True)
        commit, dirty = self.commit
        self._put(LAYOUT['manifest'], {
            'schema': 'pitcheezy-publish-manifest-v1', 'contract': self.config['contract_version'],
            'commit': commit, 'commit_dirty': dirty, 'published_at': _iso(now),
            'publish_interval_s': self.config['publish_interval_s'], 'fixed_delay_s': self.config['fixed_delay_s'],
            'provisional': list(PROVISIONAL), 'live_dates': live_dates, 'files': LAYOUT}, always=True)
        health = {'schema': 'pitcheezy-publish-health-v1', 'sequence': self._sequence, 'published_at': _iso(now),
                  'publish_interval_s': self.config['publish_interval_s'], 'files_written': len(self._written)+1,
                  'unavailable_games': unavailable}
        self._put(LAYOUT['health'], health, always=True)
        self._save_state()
        return health

    def _setup(self, game_pk):
        """A finished game's catcher-setup file: only what ``SetupEstimates.view`` accepts (sha256 pin and every
        line validated); a game without estimates has no file, and one that stops validating loses its file."""
        if self.setups is None:
            return
        path = LAYOUT['setup'].format(game_pk=game_pk)
        try:
            view = self.setups.view(game_pk)
        except KeyError:
            view = None
        except ValueError:
            LOGGER.warning('setup estimates of %s refused by validation; not published', game_pk, exc_info=True)
            view = None
        if view is None:
            self._drop(path)
        else:
            self._put(path, view, always=False, final=True)

    def run(self, *, once=False, sleep=time.sleep, clock=time.monotonic):
        """Cycles forever at ``publish_interval_s`` (start to start). A failed cycle is logged and the next one
        runs: the health file then simply stops moving, which is what viewers and monitors read."""
        while True:
            started = clock()
            try:
                self.cycle()
            except Exception:  # noqa: BLE001
                if once:
                    raise
                LOGGER.exception('publish cycle failed')
            if once:
                return
            sleep(max(0.0, self.config['publish_interval_s']-(clock()-started)))


def build(config, *, sink=None, dates=None, commit=None, wall=None, **app_options):
    """A ``Publisher`` over a private app (never served: no port, no worker) built like the server's. The app's
    fan route delay is the publisher's fixed delay. ``app_options`` go to ``main.create_app``."""
    from .main import create_app
    if not config.get('out_dir') and sink is None:
        raise ValueError('an output directory is required (--out, PITCHEEZY_PUBLISH_OUT_DIR or publish_config.json out_dir)')
    app_options.setdefault('service', object())  # the legacy replay service and its database are not opened
    app = create_app(start_worker=False, service_delay_s=config['fixed_delay_s'], **app_options)
    state_path = config.get('state_path') or (config.get('out_dir') and f"{str(config['out_dir']).rstrip('/')}.state.json")
    return app, Publisher(app.state.service_bodies, sink or LocalDirSink(config['out_dir']), config=config,
                          setups=app.state.setups, state_path=state_path, dates=dates,
                          commit=code_commit() if commit is None else commit, **({} if wall is None else {'wall': wall}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', help='output directory (or PITCHEEZY_PUBLISH_OUT_DIR)')
    parser.add_argument('--once', action='store_true', help='one cycle, then exit')
    parser.add_argument('--date', action='append', help='KST date whose games are rebuilt every cycle (repeatable); default: the window in publish_config.json')
    parser.add_argument('--config', help='publish_config.json to use')
    parser.add_argument('--interval', type=float, help='publish interval in seconds')
    parser.add_argument('--delay', type=float, help='fixed delay in seconds')
    parser.add_argument('--feed-dir', help='finished-game feeds (default PITCHEEZY_FEED_DIR)')
    parser.add_argument('--watch-dir', help='watch datasets (default PITCHEEZY_WATCH_DIR)')
    parser.add_argument('--intent-dir', help='setup estimates (default PITCHEEZY_INTENT_DIR)')
    parser.add_argument('--demo-dir', help='demo clock folder (default PITCHEEZY_DEMO_DIR)')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    config = load_config(args.config, out_dir=args.out, publish_interval_s=args.interval, fixed_delay_s=args.delay)
    app, publisher = build(config, dates=args.date, feed_dir=args.feed_dir, watch_dir=args.watch_dir,
                           intent_dir=args.intent_dir, demo_dir=args.demo_dir)
    feed = app.state.live_parts()
    if config['fixed_delay_s'] > getattr(feed, 'max_delay_s', config['fixed_delay_s']):
        raise SystemExit(f"fixed_delay_s {config['fixed_delay_s']} exceeds the live feed's max_delay_s {feed.max_delay_s}")
    if os.environ.get('PITCHEEZY_OBSERVER_LIVE_ARMB', '1') != '0':
        app.state.live_policy.start()  # warm the frozen policy in the background, as the server does at startup
    LOGGER.info('publishing to %s every %s s (fixed delay %s s)', config['out_dir'], config['publish_interval_s'], config['fixed_delay_s'])
    try:
        publisher.run(once=args.once)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
