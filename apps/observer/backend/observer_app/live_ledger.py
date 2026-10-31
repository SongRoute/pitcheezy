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

from collections import OrderedDict
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


def _is_committed_version(feed, entry):
    """True if ``entry['feed']`` hashes to the committed ``entry['sha']``. Remembered per served body (held by
    reference, so never confused with another object) on the ``LiveFeed``: one hash per poll, not per request."""
    seen = feed.__dict__.setdefault('_ledger_verified', OrderedDict())
    held = seen.get(entry['seq'])
    if held is not None and held[0] is entry['feed'] and held[1] == entry['sha']:
        return True
    if hashlib.sha256(_canonical(entry['feed'])).hexdigest() != entry['sha']:
        return False
    seen[entry['seq']] = (entry['feed'], entry['sha'])
    while len(seen) > 32:
        seen.popitem(last=False)
    return True


def admit(feed, entry, view, *, game_pk, snapshot=None, policy_identity=None, record_new=True, inputs=None):
    """``view`` (``main.armb_view`` output for ``entry``'s snapshot) as it may be displayed. ``feed`` is the
    ``LiveFeed`` that served ``entry``; ``entry`` is the copy ``LiveFeed.snapshot`` returned (None when the
    caller has none). ``snapshot`` is the body the view was computed from: it must be ``entry['feed']`` itself,
    so the row can never cite another version than the one the recommendation was made for.
    ``policy_identity`` is the policy's full identity (the view only shows a short prefix).
    ``record_new=False`` (the feed is disconnected: ``entry`` is only the last snapshot) never writes: the
    recommendation is returned only if the ledger already holds it for this decision, else it is withheld.
    ``inputs`` (``rec_pairing.decision_inputs`` of the same snapshot) is stored with a new row: who was pitching
    and batting and the count the decision was computed for, so the pitch thrown next can be reconciled with it."""
    store = getattr(feed, 'store', None)
    if store is None or not enabled() or not view or not view.get('recommendation'):
        return view
    if not entry or entry.get('seq') is None or entry.get('sha') is None:
        return _withheld('snapshot_not_committed')
    if snapshot is not None and snapshot is not entry['feed']:
        return _withheld('view_not_computed_from_this_snapshot')
    if not _is_committed_version(feed, entry):
        return _withheld('snapshot_differs_from_committed_version')
    state_sha = hashlib.sha256(state_key(entry['feed']).encode()).hexdigest()
    if not record_new:
        try:
            record = store.stored_recommendation(game_pk, state_sha, policy_identity or view.get('policy_identity'))
        except (sqlite3.Error, OSError, ValueError) as exc:
            return _withheld(type(exc).__name__)
        if record is None:
            return _withheld('feed_disconnected_nothing_recorded')
        return view | record['displayed'] | {'ledger': {'status': 'recorded', 'outcome': 'held',
                                                        'decision_id': record['decision_id'], 'decided': record['decided'],
                                                        'decision_clock': record['decision_clock']}}
    try:
        outcome, record = store.append_recommendation(
            game_pk=game_pk, state_sha=state_sha,
            policy_identity=policy_identity or view.get('policy_identity'), pitch_key=view.get('key'), snapshot=entry,
            displayed={key: view.get(key) for key in DISPLAYED}, decided=feed.wall(),
            feed_ok=int(game_pk) not in feed.failing, inputs=inputs)
    except (sqlite3.Error, OSError, ValueError) as exc:
        LOGGER.warning('recommendation ledger write failed; recommendation withheld', exc_info=True)
        return _withheld(type(exc).__name__)
    if outcome == 'conflict':
        LOGGER.warning('recommendation for decision %s differs from the stored one; showing the stored one',
                       record['decision_id'])
    return view | record['displayed'] | {'ledger': {'status': 'recorded', 'outcome': outcome,
                                                    'decision_id': record['decision_id'], 'decided': record['decided'],
                                                    'decision_clock': record['decision_clock']}}
