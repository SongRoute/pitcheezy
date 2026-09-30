"""Delayed-live ARM-B recommendations for the observer (DEMO-WS-2026; display only, not an evaluation).

``scripts/demo_precompute.LivePolicy`` binds the frozen <=2025 ARM-B components over the game so far
(about 5-8 s) and evaluates the pending pitch exactly like the watch-along precompute. That is too slow
for a request, so one background thread computes each new game state once and the route serves the
cached result: the delay buffer (``delay_s``) gives the thread a head start, because the newest raw
snapshot is queued as soon as it is fetched, before it is served.
"""
from __future__ import annotations

from collections import OrderedDict
import gc
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
    return demo_precompute.LivePolicy()


class LiveArmB:
    """One worker thread: loads the policy once, then computes queued game states newest-priority first."""

    def __init__(self, factory=_default_factory, *, cache_size=256, max_pending=4):
        self.factory, self.cache_size, self.max_pending = factory, cache_size, max_pending
        self.cache, self.pending = OrderedDict(), OrderedDict()
        self.failed, self.running = {}, None
        self.policy, self.error, self.loading = None, None, False
        self.computed, self.last_seconds = 0, None
        self._cond = threading.Condition()
        self._thread = None

    def start(self):
        with self._cond:
            if self._thread is None:
                self.loading = True
                self._thread = threading.Thread(target=self._run, name='live-armb', daemon=True)
                self._thread.start()

    def status(self):
        state = ('ready' if self.policy is not None else 'loading' if self.loading else
                 'unavailable' if self.error else 'idle')
        return {'state': state, 'reason': self.error, 'computed': self.computed, 'pending': len(self.pending),
                'last_seconds': self.last_seconds}

    def lookup(self, feed, *, queue=True, first=False):
        """Cached result for this feed's state; else queue it (``first``: ahead of other states) and return None."""
        key = state_key(feed)
        with self._cond:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            if key in self.failed:
                return {'error': self.failed[key]}
            if queue and self.error is None and key != self.running:
                self.pending[key] = feed
                self.pending.move_to_end(key, last=not first)
                while len(self.pending) > self.max_pending:  # stale intermediate states are dropped
                    self.pending.popitem(last=True)
                self._cond.notify()
        self.start()
        return None

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
                continue
            gc.collect()  # each bind loads fresh frozen components; release the previous ones promptly
            seconds = round(time.perf_counter() - started, 2)
            with self._cond:
                self.running = None
                self.cache[key] = {'result': result}
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                self.computed, self.last_seconds = self.computed + 1, seconds
            LOGGER.info('live state computed', extra={'fields': {'key': key, 'seconds': seconds}})
