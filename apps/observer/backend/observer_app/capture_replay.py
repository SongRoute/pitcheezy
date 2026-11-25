"""Offline replay input for the publisher: a finished capture's stored upstream responses, handed out by a
simulated clock so that a request made at simulated time ``t`` is answered with what the capture had received
by ``t`` and nothing later.

``rows`` is the capture's stored versions in receipt order: ``(fetched, load)`` where ``fetched`` is the wall time
the capture had that version completely (``snapshots.fetched``) and ``load()`` returns its body (a fresh object
each call). The capture keeps one row per stored version, not per HTTP reply: a poll that got the same body
again, or a stale one, left no row. So the answer at ``t`` is the newest stored version with ``fetched <= t``:
exactly what a poller reading the same upstream would hold, as far as the capture can tell.

No look-ahead is structural: bodies are loaded by index, and the index is chosen from ``fetched`` alone.
The transport never reaches a network: any URL it does not know is refused.
"""
from __future__ import annotations

from bisect import bisect_right
import re

from .live_feed import LiveFeedError


class SimClock:
    """The one clock of a replay (epoch seconds). It moves only when the driver moves it, and never back."""

    def __init__(self, start):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        if not seconds >= 0:  # also refuses NaN
            raise ValueError('a replay clock does not go back')
        self.now += seconds
        return self.now

    def set(self, moment):
        return self.advance(float(moment)-self.now)


class CaptureReplayTransport:
    """``LiveFeed`` transport over ``rows`` (see the module text). ``schedule_game(body)`` turns a feed body into
    the schedule item ``LiveFeed.schedule`` reads (the capture stores no schedule reply); the schedule answer at
    ``t`` is built from the body held at ``t``, so it knows no later status either."""

    def __init__(self, rows, clock, *, game_pk, schedule_game):
        self.fetched = [float(fetched) for fetched, _ in rows]
        if not rows or any(later < earlier for earlier, later in zip(self.fetched, self.fetched[1:])):
            raise ValueError('a replay needs at least one stored version, in receipt order')
        self.loaders, self.clock, self.game_pk, self.schedule_game = [load for _, load in rows], clock, int(game_pk), schedule_game
        self.requests, self.highest = 0, -1  # game-feed requests answered; newest row index ever handed out
        self.served = set()  # every row index ever handed out (a row between two polls is never in it)

    def index(self):
        """Row held at the clock's time, or -1 before the first receipt."""
        return bisect_right(self.fetched, self.clock())-1

    def exhausted(self):
        return self.clock() >= self.fetched[-1]

    def position(self):
        """Where the replay is (also marks this transport as a replay for ``display_state._is_real_poll``)."""
        return {'served': self.highest+1, 'bodies': len(self.fetched), 'requests': self.requests,
                'duration_s': round(self.fetched[-1]-self.fetched[0], 1), 'elapsed_s': round(self.clock()-self.fetched[0], 1)}

    def _held(self):
        index = self.index()
        if index < 0:
            return None
        self.highest = max(self.highest, index)
        self.served.add(index)
        return self.loaders[index]()

    def __call__(self, url, timeout):
        if (match := re.search(r'/api/v1/schedule\?.*\bdate=(\d{4}-\d{2}-\d{2})', url)):
            body = self._held()
            game = None if body is None else self.schedule_game(body)
            return {'dates': [{'date': match[1], 'games': [game]}] if game and game.get('officialDate') == match[1] else []}
        match = re.search(r'/game/(\d+)/feed/live', url)
        if match is None or int(match[1]) != self.game_pk:
            raise LiveFeedError(f'the replay has no answer for {url}')
        body = self._held()
        if body is None:
            raise LiveFeedError('the capture had received nothing yet at this time')
        self.requests += 1
        return body
