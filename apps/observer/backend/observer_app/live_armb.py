"""Delayed-live ARM-B recommendations for the observer (DEMO-WS-2026; display only, not an evaluation).

``scripts/demo_precompute.LivePolicy`` binds the frozen <=2025 ARM-B components once per game (about
5-8 s; again only when an already-bound row changes) and then evaluates each pending pitch exactly like
the watch-along precompute in about 0.15 s. The first state of a game is too slow for a request, so one
background thread computes each new game state once and the route serves the
cached result: the delay buffer (``delay_s``) gives the thread a head start, because the newest raw
snapshot is queued as soon as it is fetched, before it is served.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import logging
import sys
import threading
import time

from .settings import REPO

LOGGER = logging.getLogger(__name__)


def state_key(feed):
    """Identity of the pending decision in a feed: every pre-pitch input that can change between polls."""
    data, live = feed['gameData'], feed['liveData']
    plays = live['plays'].get('allPlays') or []
    line = live.get('linescore') or {}
    last = live['plays'].get('currentPlay') or (plays[-1] if plays else {})
    offense = line.get('offense') or {}
    return json.dumps([data['game']['pk'], len(plays), (last.get('about') or {}).get('atBatIndex'),
                       bool(last.get('about', {}).get('isComplete', True)),
                       [base for base in ('first', 'second', 'third') if base in offense],
                       len(last.get('playEvents') or []), len(last.get('runners') or []),
                       (last.get('matchup') or {}).get('pitcher', {}).get('id'),
                       line.get('outs'), line.get('inningState'), line.get('currentInning'),
                       offense.get('batter', {}).get('id'),
                       (line.get('defense') or {}).get('pitcher', {}).get('id')])


def _default_factory():
    scripts = str(REPO/'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import demo_precompute
    from .live_feed import LIVE_CONFIG
    return demo_precompute.LivePolicy(max_games=LIVE_CONFIG.get('max_bound_games', 2))


class LiveArmB:
    """One worker thread: loads the policy once, then computes queued game states newest-priority first."""

    def __init__(self, factory=_default_factory, *, cache_size=256, max_pending=4):
        self.factory, self.cache_size, self.max_pending = factory, cache_size, max_pending
        self.cache, self.pending = OrderedDict(), OrderedDict()
        self.failed, self.running = {}, None
        self.policy, self.error, self.loading = None, None, False
        self.computed, self.last_seconds = 0, None
        self.games = OrderedDict()  # gamePk -> {'pre': {pitch key: top type or None}, 'actual': {pitch key: type}}
        self._cond = threading.Condition()  # its default lock is an RLock
        self._thread = None

    def start(self):
        with self._cond:  # reentrant: ``lookup`` starts the worker while holding it
            if self._thread is None:
                self.loading = True
                self._thread = threading.Thread(target=self._run, name='live-armb', daemon=True)
                self._thread.start()

    def status(self):
        state = ('ready' if self.policy is not None else 'loading' if self.loading else
                 'unavailable' if self.error else 'idle')
        return {'state': state, 'reason': self.error, 'computed': self.computed, 'pending': len(self.pending),
                'last_seconds': self.last_seconds}

    def lookup(self, feed, *, queue=True, first=False, wait_s=0.0):
        """Cached result for this feed's state; else queue it (``first``: ahead of other states) and return
        None. ``wait_s`` waits that long for the worker first: an ordinary state takes about 0.15 s, so the
        viewer gets the recommendation with the pitch instead of one poll later."""
        key = state_key(feed)
        deadline = time.monotonic() + wait_s
        with self._cond:
            if key not in self.cache and key not in self.failed and queue and self.error is None and key != self.running:
                self.pending[key] = feed
                self.pending.move_to_end(key, last=not first)
                while len(self.pending) > self.max_pending:  # stale intermediate states are dropped
                    self.pending.popitem(last=True)
                self._cond.notify_all()
                self.start()
            while True:
                if key in self.cache:
                    self.cache.move_to_end(key)
                    return self.cache[key]
                if key in self.failed:
                    return {'error': self.failed[key]}
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.policy is None or (key != self.running and key not in self.pending):
                    return None
                self._cond.wait(remaining)

    def _note(self, result):
        """Remembers, per pitch of a game, the top recommended type before it and the type actually thrown."""
        game = self.games.setdefault(result['key'].split(':')[0], {'pre': {}, 'actual': {}})
        self.games.move_to_end(result['key'].split(':')[0])
        while len(self.games) > 4:
            self.games.popitem(last=False)
        for pitch in [*result['pa_pitches'], result['pitch']]:
            candidates = (pitch['pre'].get('recommendation') or {}).get('candidates') or []
            game['pre'][pitch['key']] = candidates[0]['pitch_type'] if pitch['pre']['status'] == 'ready' and candidates else None
            if (pitch.get('actual') or {}).get('pitch_type'):
                game['actual'][pitch['key']] = pitch['actual']['pitch_type']
        previous = result.get('previous')
        if previous and (previous.get('actual') or {}).get('pitch_type'):
            game['actual'][previous['key']] = previous['actual']['pitch_type']

    def tally(self, key):
        """Of the pitches before pending pitch ``key`` that this server recommended on and then saw thrown:
        how many had a recommendation and how many were the top recommended type. A description of what
        the pitcher chose, not a measure of the recommendation's quality."""
        order = lambda k: tuple(int(part) for part in k.split(':')[1:])
        with self._cond:
            game = self.games.get(key.split(':')[0], {'pre': {}, 'actual': {}})
            seen = [(game['pre'][k], thrown) for k, thrown in game['actual'].items() if k in game['pre'] and order(k) < order(key)]
        return {'pitches': len(seen), 'recommended': sum(top is not None for top, _ in seen),
                'same_type': sum(top == thrown for top, thrown in seen)}

    def _run(self):
        try:
            self.policy = self.factory()
        except Exception as exc:  # noqa: BLE001 - reported in health and to the viewer, never fatal
            LOGGER.exception('ARM-B live policy unavailable')
            self.error = f'{type(exc).__name__}: {exc}'
            return
        finally:
            self.loading = False
        while True:
            with self._cond:
                while not self.pending:
                    self._cond.wait()
                key, feed = self.pending.popitem(last=False)
                self.running = key
            started = time.perf_counter()
            try:
                result = self.policy(feed)
            except Exception as exc:  # noqa: BLE001 - one bad state must not stop the worker
                LOGGER.exception('ARM-B live computation failed')
                with self._cond:
                    self.running = None
                    self.failed[key] = f'{type(exc).__name__}'
                    self._cond.notify_all()
                continue
            seconds = round(time.perf_counter() - started, 2)
            with self._cond:
                self.running = None
                self.cache[key] = {'result': result}
                if result is not None:
                    self._note(result)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                self.computed, self.last_seconds = self.computed + 1, seconds
                self._cond.notify_all()
            LOGGER.info('live state computed', extra={'fields': {'key': key, 'seconds': seconds}})
