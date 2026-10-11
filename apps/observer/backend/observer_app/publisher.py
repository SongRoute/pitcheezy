"""Public publisher: writes, with no viewer request, the bodies a viewer would have fetched, as static files.

    python -m observer_app.publisher --out <dir> [--once] [--date YYYY-MM-DD ...]

Each cycle the publisher asks the SAME builders the ``/api/service/**`` routes call (``service_api.ServiceBodies``)
for the dates, each date's game list and each game's full body, and hands them to a sink:

    v1/manifest.json   contract version, code commit, published_at, publish interval, fixed delay
    v1/health.json     last publish time and sequence (written last: a cycle is complete when it changes),
                       the games and dates that could not be built this cycle
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
never asked for; a game a demo session takes is withdrawn (its game and setup files are deleted) in the next
cycle and published again when the demo ends. Uploading is not done here: ``Sink`` is the seam; only
``LocalDirSink`` exists. One publisher per output: a second one over the same state file refuses to start.
"""
from __future__ import annotations

import argparse
import fcntl
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
FILES = {'manifest': 'manifest.json', 'health': 'health.json', 'dates': 'dates.json', 'games': 'games/{date}.json',
         'game': 'game/{game_pk}.json', 'setup': 'setup/{game_pk}.json'}
# ``display_state.feed`` describes the connection at the moment a body was built (last good poll, receipt time,
# measured lag, outage): it moves with the clock and with every poll. Only these keys say something about the
# game's content (the admitted order of the served snapshot) or are configuration.
STATE_FEED_CONTENT = ('provider_order', 'lag_threshold_s', 'thresholds_provisional')


def layout(contract):
    """Published path of each file kind. The contract version IS the prefix (``v1/...``): one value, never two."""
    return {kind: f'{contract}/{path}' for kind, path in FILES.items()}
ENV = {'out_dir': ('PITCHEEZY_PUBLISH_OUT_DIR', str), 'state_path': ('PITCHEEZY_PUBLISH_STATE', str),
       'publish_interval_s': ('PITCHEEZY_PUBLISH_INTERVAL_S', float), 'fixed_delay_s': ('PITCHEEZY_PUBLISH_DELAY_S', float)}
PROVISIONAL = ('publish_interval_s', 'settled_recheck_s', 'final_recheck_for_s', 'live_days_back', 'live_days_ahead')


def load_config(path=None, env=None, **overrides):
    """``publish_config.json`` <- environment <- explicit overrides (None = not given)."""
    config = json.loads(Path(path or CONFIG_PATH).read_text())
    env = os.environ if env is None else env
    for key, (name, kind) in ENV.items():
        if env.get(name) not in (None, ''):
            config[key] = kind(env[name])
    config |= {key: value for key, value in overrides.items() if value is not None}
    interval, delay, recheck = config['publish_interval_s'], config['fixed_delay_s'], config['settled_recheck_s']
    if not isinstance(config['contract_version'], str) or re.fullmatch(r'v[1-9]\d*', config['contract_version']) is None:
        raise ValueError('contract_version must look like v1, v2, ... (it is also the path prefix)')
    period = config.get('final_recheck_for_s')
    if period is not None and not (isinstance(period, (int, float)) and period >= 0):
        raise ValueError('final_recheck_for_s must be null or a number of seconds')
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
    public host implements these three calls; the publisher's build logic does not change.

    Temporary names: a sink must never let a half-written or temporary object be readable under the public
    prefix. ``LocalDirSink`` keeps its temporary files in a sibling folder outside the output directory, so the
    output holds published files only. An uploader that mirrors a local folder must still skip dot-files and
    anything that is not one of the published names (``publisher.FILES``), and one that writes remotely must
    upload under a private name and switch, or rely on the host's atomic object replace."""

    def write(self, path, data, *, final=False):
        raise NotImplementedError

    def read(self, path):
        """The bytes last written at ``path``, or None if there are none or this sink cannot read back."""
        return None

    def delete(self, path):
        raise NotImplementedError


class LocalDirSink(Sink):
    """Files under one local directory, each replaced atomically: the content is written completely to a
    temporary file in ``<parent>/.<name>.tmp/`` (a sibling of the output directory, so on the same filesystem and
    never inside what gets uploaded) and then renamed onto the published name."""

    def __init__(self, root):
        self.root = Path(root)
        self.staging = self.root.parent/f'.{self.root.name}.tmp'

    def _target(self, path):
        parts = PurePosixPath(path).parts
        if not parts or PurePosixPath(path).is_absolute() or any(part in ('..', '.') or part.startswith('.') for part in parts):
            raise ValueError(f'not a relative publish path: {path}')
        return self.root.joinpath(*parts)

    def write(self, path, data, *, final=False):
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        temporary = self.staging/f"{'_'.join(PurePosixPath(path).parts)}.{os.getpid()}.tmp"
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


def game_digest(body):
    """What makes a settled game file new: its content, not the moment it was built. Left out are the fields that
    move with the clock or with the connection while the game itself is unchanged: the whole top-level ``feed``
    (``as_of``, ``ok``, ``stale_s``) and, inside ``display_state.feed``, everything but ``STATE_FEED_CONTENT``
    (``ok``, ``delayed``, ``lag_s``, ``lag_max_s``, ``lag_measured``, ``last_ok_at``, ``failing_since``,
    ``failing_for_s``, ``received_at``). Pitches, plate appearances, receipts, the display state itself, the
    recommendation and its ledger record all count."""
    stable = {key: value for key, value in body.items() if key != 'feed'}
    state = stable.get('display_state')
    if isinstance(state, dict) and isinstance(state.get('feed'), dict):
        stable['display_state'] = state | {'feed': {key: state['feed'].get(key) for key in STATE_FEED_CONTENT}}
    return _digest(stable)


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


class AlreadyRunning(RuntimeError):
    """Another publisher holds this output's lock."""


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
        self.layout = layout(config['contract_version'])
        self.state_path = None if state_path is None else Path(state_path)
        self._lock = self._acquire()
        try:
            self.state = self._load_state()
        except BaseException:
            self.close()
            raise

    def _acquire(self):
        """Exclusive, non-blocking lock on ``<state file>.lock`` for this object's life: two publishers over one
        output would hand out the same sequence twice and overwrite each other's files with older ones."""
        if self.state_path is None:
            return None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(f'{self.state_path}.lock', 'a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise AlreadyRunning(f'another publisher is already running on {self.state_path} (lock {self.state_path}.lock is held)') from None
        return handle

    def close(self):
        """Release the lock (the process exiting does the same)."""
        if getattr(self, '_lock', None) is not None:
            self._lock.close()
            self._lock = None

    # ---- bookkeeping --------------------------------------------------------------------------------
    def _load_state(self):
        state = {'sequence': 0, 'files': {}, 'checked': {}, 'embargoed': []}
        if self.state_path is not None and self.state_path.is_file():
            try:
                saved = json.loads(self.state_path.read_text())
                if not (isinstance(saved, dict) and type(saved.get('sequence')) is int and isinstance(saved.get('files'), dict)
                        and isinstance(saved.get('checked'), dict) and isinstance(saved.get('embargoed', []), list)):
                    raise ValueError('not a publisher state')
                state |= saved
            except (ValueError, OSError):  # unreadable bookkeeping: start as if it were lost (everything is rebuilt)
                LOGGER.warning('state file %s unreadable; continuing from the published health file', self.state_path, exc_info=True)
        try:  # the sequence never goes back, even if the state file was lost: the last health file knows it too
            health = self.sink.read(self.layout['health'])
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

    def _put(self, path, body, *, always, final=None, digest=None):
        """Write ``body`` + the ``publish`` key at ``path``. ``always`` False: only if the content (``digest``,
        default the whole body) or its final mark differs from what is published there."""
        digest = _digest(body) if digest is None else digest
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

    def _withdraw(self, game_pk):
        """A demo session holds this game: nothing of it stays public (its game and setup files are deleted) and
        it is tried again in every cycle, so it returns as soon as the demo is over."""
        for kind in ('game', 'setup'):
            path = self.layout[kind].format(game_pk=game_pk)
            self._drop(path)
            self.state['checked'].pop(path, None)

    # ---- one cycle ----------------------------------------------------------------------------------
    def cycle(self):
        """Build and write everything that is due. Returns the health body of this cycle."""
        self._now, self._written = self.wall(), []
        self._sequence = self.state['sequence'] = self.state['sequence']+1
        self._save_state()  # the number is taken before any file carries it: a crash never reuses it
        now, unavailable, unavailable_dates = self._now, [], []
        layout_ = self.layout
        embargoed = {int(pk) for pk in self.bodies.embargoed()}
        if sorted(embargoed) != self.state['embargoed']:
            # a demo started or ended: every game list is rebuilt now (its card is hidden or shown again), not at
            # the next settled recheck
            prefix = layout_['games'].split('{')[0]
            for path in [path for path in self.state['checked'] if path.startswith(prefix)]:
                del self.state['checked'][path]
            self.state['embargoed'] = sorted(embargoed)
        for game_pk in embargoed:
            self._withdraw(game_pk)
        dates_body = check_public(self.bodies.dates())
        live_dates = self.live_dates(now)
        fast, listed = set(), set()
        for date in sorted(set(dates_body['dates']) | set(live_dates)):
            path, live = layout_['games'].format(date=date), date in live_dates
            if not live and not self._due(path, now):
                continue
            try:
                body = check_public(self.bodies.games(date))
            except Exception:  # noqa: BLE001 - one date never stops the cycle; its last file stays
                LOGGER.warning('game list of %s not built; the published one is kept', date, exc_info=True)
                self.state['checked'][path] = now
                unavailable_dates.append(date)
                continue
            self._put(path, body, always=live)
            for card in body['games']:
                if card.get('embargoed'):  # a demo is running on it: its game body is refused anyway
                    continue
                listed.add(int(card['game_pk']))
                if live:
                    fast.add(int(card['game_pk']))
        prefix = layout_['game'].split('{')[0]
        known = {int(PurePosixPath(path).stem) for path in self.state['files'] if path.startswith(prefix)}
        period = self.config.get('final_recheck_for_s')
        for game_pk in sorted((listed | known)-embargoed):
            path = layout_['game'].format(game_pk=game_pk)
            entry = self.state['files'].get(path) or {}
            settled = entry.get('final') is True
            if (settled or game_pk not in fast) and not self._due(path, now):
                continue
            self.state['checked'][path] = now
            if settled and entry.get('kind') == 'live' and period is not None and now-entry['final_at'] >= period:
                # long final: the upstream feed is not asked again and the file stays; a setup file that arrives
                # late is still found (a local read)
                self._setup(game_pk)
                continue
            try:
                body = check_public(self.bodies.game(game_pk))  # polls upstream; ledger is written in here
            except Refused as refused:
                if refused.status == 423:  # a demo took the game between the check above and now: not a fault
                    self._withdraw(game_pk)
                else:
                    unavailable.append({'game_pk': game_pk, 'status': refused.status})
                continue
            except NotPublic:
                raise
            except Exception:  # noqa: BLE001 - one game never stops the cycle; its last file stays
                LOGGER.warning('game %s not built; the published file is kept', game_pk, exc_info=True)
                unavailable.append({'game_pk': game_pk, 'status': None})
                continue
            final = is_final(body)
            self._put(path, body, always=not final, final=final, digest=game_digest(body))
            written = self.state['files'][path]
            written['kind'] = body['source']['kind']
            written['final_at'] = (entry.get('final_at') if settled else now) if final else None
            if final:
                self._setup(game_pk)
            else:
                self._drop(layout_['setup'].format(game_pk=game_pk))
        self._put(layout_['dates'], dates_body, always=True)
        commit, dirty = self.commit
        self._put(layout_['manifest'], {
            'schema': 'pitcheezy-publish-manifest-v1', 'contract': self.config['contract_version'],
            'commit': commit, 'commit_dirty': dirty, 'published_at': _iso(now),
            'publish_interval_s': self.config['publish_interval_s'], 'fixed_delay_s': self.config['fixed_delay_s'],
            'provisional': list(PROVISIONAL), 'live_dates': live_dates, 'files': layout_}, always=True)
        health = {'schema': 'pitcheezy-publish-health-v1', 'sequence': self._sequence, 'published_at': _iso(now),
                  'publish_interval_s': self.config['publish_interval_s'], 'files_written': len(self._written)+1,
                  'unavailable_games': unavailable, 'unavailable_dates': unavailable_dates}
        self._put(layout_['health'], health, always=True)
        self._save_state()
        return health

    def _setup(self, game_pk):
        """A finished game's catcher-setup file: only what ``SetupEstimates.view`` accepts (sha256 pin and every
        line validated); a game without estimates has no file, and one that stops validating loses its file."""
        if self.setups is None:
            return
        path = self.layout['setup'].format(game_pk=game_pk)
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
    config, publisher = _start(args)
    LOGGER.info('publishing to %s every %s s (fixed delay %s s)', config['out_dir'], config['publish_interval_s'], config['fixed_delay_s'])
    try:
        publisher.run(once=args.once)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()
    return 0


def _start(args):
    config = load_config(args.config, out_dir=args.out, publish_interval_s=args.interval, fixed_delay_s=args.delay)
    try:
        app, publisher = build(config, dates=args.date, feed_dir=args.feed_dir, watch_dir=args.watch_dir,
                               intent_dir=args.intent_dir, demo_dir=args.demo_dir)
    except AlreadyRunning as held:
        raise SystemExit(str(held)) from None
    feed = app.state.live_parts()
    if config['fixed_delay_s'] > getattr(feed, 'max_delay_s', config['fixed_delay_s']):
        publisher.close()
        raise SystemExit(f"fixed_delay_s {config['fixed_delay_s']} exceeds the live feed's max_delay_s {feed.max_delay_s}")
    if os.environ.get('PITCHEEZY_OBSERVER_LIVE_ARMB', '1') != '0':
        app.state.live_policy.start()  # warm the frozen policy in the background, as the server does at startup
    return config, publisher


if __name__ == '__main__':
    sys.exit(main())
