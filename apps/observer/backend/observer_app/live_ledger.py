"""Seam between the serving live path and the append-only store: a pre-pitch recommendation is written to the
``served_recommendations`` ledger (``live_store.LiveSnapshotStore.append_recommendation``) before it is displayed.

``admit`` is the only call the routes make. It never computes or changes a recommendation:

* no store on the feed (plain ``LiveFeed``, the default run) -> the view is returned untouched (``ledger`` absent);
* a store, and the write (or an identical earlier one) is committed -> the *stored* recommendation is returned
  with ``ledger = {'status': 'recorded', ...}``;
* a store, but the snapshot has no committed identity (store outage: memory-only entry) or the write fails ->
  the recommendation is withheld (``status: 'unavailable'``) rather than shown unrecorded;
* the same decision was stored earlier with a different content -> the first stored one is shown and the new
  one is kept as a ``conflict`` row (no silent flip).

``PITCHEEZY_OBSERVER_REC_LEDGER=0`` turns the ledger off (recommendations are then shown unrecorded, as before).
"""
from __future__ import annotations

import hashlib
import logging
import os
import sqlite3

from .live_armb import state_key
from .live_store import _canonical

LOGGER = logging.getLogger(__name__)
# What a viewer is shown of a recommendation and what the ledger therefore stores. Per-request extras
# (``tally``, ``seconds``, the pitches already thrown) are facts of the snapshot, not of the decision.
DISPLAYED = ('status', 'reason', 'key', 'recommendation', 'policy_identity')
WITHHELD = '추천 기록을 저장하지 못해 이번 추천은 표시하지 않습니다.'


def enabled():
    return os.environ.get('PITCHEEZY_OBSERVER_REC_LEDGER', '1') != '0'


def _withheld(why):
    return {'status': 'unavailable', 'reason': WITHHELD, 'recommendation': None,
            'ledger': {'status': 'withheld', 'why': why}}


def admit(feed, entry, view, *, game_pk):
    """``view`` (``main.armb_view`` output for ``entry``'s snapshot) as it may be displayed. ``feed`` is the
    ``LiveFeed`` that served ``entry``; ``entry`` is its buffer entry (None when the caller has none)."""
    store = getattr(feed, 'store', None)
    if store is None or not enabled() or not view or not view.get('recommendation'):
        return view
    if not entry or entry.get('seq') is None or entry.get('sha') is None:
        return _withheld('snapshot_not_committed')
    if entry.get('_verified') is not entry['feed']:  # once per served body: it is the committed version's bytes
        if hashlib.sha256(_canonical(entry['feed'])).hexdigest() != entry['sha']:
            return _withheld('snapshot_differs_from_committed_version')
        entry['_verified'] = entry['feed']
    try:
        outcome, record = store.append_recommendation(
            game_pk=game_pk, state_sha=hashlib.sha256(state_key(entry['feed']).encode()).hexdigest(),
            policy_identity=view.get('policy_identity'), pitch_key=view.get('key'), snapshot=entry,
            displayed={key: view.get(key) for key in DISPLAYED}, decided=feed.wall(),
            feed_ok=int(game_pk) not in feed.failing)
    except (sqlite3.Error, OSError, ValueError) as exc:
        LOGGER.warning('recommendation ledger write failed; recommendation withheld', exc_info=True)
        return _withheld(type(exc).__name__)
    if outcome == 'conflict':
        LOGGER.warning('recommendation for decision %s differs from the stored one; showing the stored one',
                       record['decision_id'])
    return view | record['displayed'] | {'ledger': {'status': 'recorded', 'outcome': outcome,
                                                    'decision_id': record['decision_id'], 'decided': record['decided'],
                                                    'decision_clock': record['decision_clock']}}
