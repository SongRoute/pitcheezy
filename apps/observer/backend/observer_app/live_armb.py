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
import hashlib
import json
import logging
import math
import sys
import threading
import time

from .settings import REPO
from .live_store import _canonical

LOGGER = logging.getLogger(__name__)


def unavailable_win_expectancy(reason='not_available'):
    return {'status': 'unavailable', 'home_probability': None, 'perspective': 'home',
            'basis': 'count_independent_game_state', 'source': 'frozen_c0_we',
            'policy_identity': None, 'pitch_key': None, 'reason': reason}


def _we_input_sha256(feed):
    # LivePolicy.live_rows/statcast_rows read gameData/liveData, never this poll stamp.
    # Preserve every other byte-level JSON value, including eventWall and earlier corrections.
    metadata = feed.get('metaData')
    if isinstance(metadata, dict):
        feed = feed | {'metaData': {key: value for key, value in metadata.items() if key != 'timeStamp'}}
    return hashlib.sha256(_canonical(feed)).hexdigest()


def _pitch_data(play):
    """What the feed currently says about a play's pitches: the model's inputs for the next pitch."""
    out = []
    for event in play.get('playEvents') or []:
        if event.get('isPitch'):
            details, data = event.get('details') or {}, event.get('pitchData') or {}
            breaks, where = data.get('breaks') or {}, data.get('coordinates') or {}
            out.append([event.get('index'), (details.get('type') or {}).get('code'), (details.get('call') or {}).get('code'),
                        data.get('startSpeed'), data.get('extension'), breaks.get('spinRate'), breaks.get('spinDirection'),
                        breaks.get('breakHorizontal'), breaks.get('breakVerticalInduced'), where.get('pX'), where.get('pZ'),
                        data.get('strikeZoneTop'), data.get('strikeZoneBottom')])
    return out


def _get(holder, *path):
    for key in path:
        holder = holder.get(key) if isinstance(holder, dict) else None
    return holder


def _context_inputs(feed):
    """What ``demo_precompute.live_rows``/``statcast_rows`` derive the next pitch's CONTEXT row from, beyond the
    fields ``state_key`` lists directly (REVIEW-001 C1): score, bases and outs are replayed from the runner
    movements and events of EVERY play, the count from every event's count, the pitcher from substitutions, and
    ``stand``/``p_throws`` from the matchup (or ``gameData.players`` for the due-up batter and a replaced pitcher).
    Pitch physics of earlier at-bats are not here (they are not in the decision's at-bat history); the recent
    plays' pitch bodies are, through ``_pitch_data``."""
    data, plays = feed['gameData'], feed['liveData']['plays'].get('allPlays') or []
    line = feed['liveData'].get('linescore') or {}
    current = feed['liveData']['plays'].get('currentPlay')
    every = [*plays, *([current] if isinstance(current, dict) and current and (not plays or plays[-1] != current) else [])]
    ids = {_get(p, 'matchup', 'pitcher', 'id') for p in every} | {_get(p, 'matchup', 'batter', 'id') for p in every}
    ids |= {_get(line, 'offense', 'batter', 'id'), _get(line, 'offense', 'onDeck', 'id'), _get(line, 'defense', 'pitcher', 'id')}
    ids |= {_get(e, 'player', 'id') for p in every for e in p.get('playEvents') or [] if isinstance(e, dict)}
    players = data.get('players') or {}
    hands = sorted([str(i), _get(players, f'ID{i}', 'pitchHand', 'code'), _get(players, f'ID{i}', 'batSide', 'code')]
                   for i in ids if i is not None)
    return [_get(data, 'game', 'type'), _get(data, 'datetime', 'officialDate'),
            [_get(line, 'teams', side, 'runs') for side in ('home', 'away')], line.get('isTopInning'), hands,
            [[_get(p, 'about', 'atBatIndex'), _get(p, 'about', 'inning'), _get(p, 'about', 'isTopInning'),
              _get(p, 'about', 'isComplete'), _get(p, 'matchup', 'pitcher', 'id'), _get(p, 'matchup', 'pitchHand', 'code'),
              _get(p, 'matchup', 'batter', 'id'), _get(p, 'matchup', 'batSide', 'code'), _get(p, 'result', 'eventType'),
              [[_get(r, 'details', 'playIndex'), _get(r, 'details', 'runner', 'id'), _get(r, 'movement', 'start'),
                _get(r, 'movement', 'end'), _get(r, 'movement', 'isOut')] for r in p.get('runners') or []],
              [[_get(e, 'index'), _get(e, 'isPitch'), _get(e, 'type'), _get(e, 'details', 'eventType'),
                _get(e, 'details', 'call', 'code'), _get(e, 'count', 'balls'), _get(e, 'count', 'strikes'),
                _get(e, 'count', 'outs'), _get(e, 'player', 'id'), _get(e, 'pitchNumber')]
               for e in p.get('playEvents') or [] if isinstance(e, dict)]] for p in every if isinstance(p, dict)]]


def state_key(feed):
    """Identity of the pending decision in a feed: every pre-pitch input that can change between polls.

    That includes what the feed says about the pitches already thrown, because it revises them without
    adding an event (seen on 849844): a pitch is first published without spin and extension, a challenged
    call flips ball/strike and the count, and a pitch type can be reclassified. A changed revision is a
    new state, so the recommendation is computed again from the data as it now stands. It also carries what the
    policy's context row is replayed from (``_context_inputs``: score, runner movements, per-event counts,
    stand/p_throws), so the result cache and ``LiveSnapshotStore.same_stamp_scope`` miss on any of them."""
    data, live = feed['gameData'], feed['liveData']
    plays = live['plays'].get('allPlays') or []
    line = live.get('linescore') or {}
    last = live['plays'].get('currentPlay') or (plays[-1] if plays else {})
    offense = line.get('offense') or {}
    count = last.get('count') or {}
    return json.dumps([data['game']['pk'], len(plays), (last.get('about') or {}).get('atBatIndex'),
                       bool(last.get('about', {}).get('isComplete', True)),
                       [base for base in ('first', 'second', 'third') if base in offense],
                       len(last.get('playEvents') or []), len(last.get('runners') or []),
                       (last.get('matchup') or {}).get('pitcher', {}).get('id'),
                       line.get('outs'), line.get('inningState'), line.get('currentInning'),
                       offense.get('batter', {}).get('id'), offense.get('onDeck', {}).get('id'),
                       (line.get('defense') or {}).get('pitcher', {}).get('id'),
                       count.get('balls'), count.get('strikes'),
                       [_pitch_data(play) for play in [*plays[-3:], last][-3:]],
                       _context_inputs(feed)])


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

    def win_expectancy(self, feed, *, pending_pitch_key, snapshot_revision=None):
        """Read existing WE for the served input and pending pitch; never queue or infer.

        C0 was fitted on PA-start states and ignores count. This is a home-team game-state
        estimate, not pitch WPA. Only metaData.timeStamp may differ from the computed body.
        The supplied receipt SHA must still match the exact, unmodified served body.
        The caller additionally enforces its receipt, display and viewer-delay gates.
        """
        try:
            digest = hashlib.sha256(_canonical(feed)).hexdigest()
            input_sha256 = _we_input_sha256(feed)
            key = state_key(feed)
        except (KeyError, TypeError, ValueError):
            return unavailable_win_expectancy('snapshot_unverifiable')
        if not isinstance(pending_pitch_key, str) or not pending_pitch_key:
            return unavailable_win_expectancy('pending_pitch_unavailable')
        if snapshot_revision is not None and snapshot_revision != digest:
            return unavailable_win_expectancy('snapshot_revision_mismatch')
        with self._cond:
            cached = self.cache.get(key)
            if (not isinstance(cached, dict) or not cached.get('snapshot_sha256')
                    or cached.get('we_input_sha256') != input_sha256):
                return unavailable_win_expectancy('matching_snapshot_not_cached')
            result = cached.get('result')
            if not isinstance(result, dict):
                return unavailable_win_expectancy('estimate_unavailable')
            pitch = result.get('pitch')
            if (result.get('key') != pending_pitch_key or not isinstance(pitch, dict)
                    or pitch.get('key') != pending_pitch_key or pitch.get('actual') is not None
                    or not isinstance(pitch.get('pre'), dict) or pitch['pre'].get('status') != 'ready'):
                return unavailable_win_expectancy('pending_pitch_mismatch')
            value, identity = result.get('home_we_now'), result.get('policy_identity')
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1
                    or not isinstance(identity, str) or len(identity) != 64
                    or any(c not in '0123456789abcdef' for c in identity)):
                return unavailable_win_expectancy('estimate_invalid')
            return unavailable_win_expectancy() | {'status': 'available', 'home_probability': float(value),
                'policy_identity': identity, 'pitch_key': pending_pitch_key, 'reason': None}

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
                snapshot_sha256 = hashlib.sha256(_canonical(feed)).hexdigest()
                we_input_sha256 = _we_input_sha256(feed)
                result = self.policy(feed)
                if hashlib.sha256(_canonical(feed)).hexdigest() != snapshot_sha256:
                    snapshot_sha256 = None  # never attribute a result to a body changed during computation
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
                self.cache[key] = {'result': result, 'snapshot_sha256': snapshot_sha256,
                                   'we_input_sha256': we_input_sha256}
                if result is not None:
                    self._note(result)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                self.computed, self.last_seconds = self.computed + 1, seconds
                self._cond.notify_all()
            LOGGER.info('live state computed', extra={'fields': {'key': key, 'seconds': seconds}})
