"""Pairs each PAST pitch of a live game with the pre-pitch recommendation the ledger recorded for it.

The ledger (``served_recommendations``) records decisions, not what stayed on a screen and not which pitch
followed (``target`` is UNKNOWN there). This module is the read-only reconciliation the fan route uses to show
"투구 전에 기록된 추천" next to a pitch that was then thrown. Principle: an ambiguous join shows NOTHING rather than a
wrong pairing. A pitch gets a recommendation only if every rule holds (``pair``):

1. the pitch itself is ``received`` (SVC-WS-03 receipt of the body on screen) and its first appearance is known:
   the pitch ledger's ``new`` row for its official key, no later than the served store version;
2. a ``first`` ledger row names the same key ``gamePk:at_bat_number:pitch_number`` (a ``conflict`` row is never
   used: the first stored content of a decision is the one that was displayable);
3. that row was decided BEFORE the pitch existed here: its snapshot's ``store_seq`` is lower than the ``store_seq``
   of the first body that held the pitch, and its ``decided`` time is before that body's receipt (a viewer on a
   long delay can write a row from an old snapshot after the pitch arrived: that one is not "before the pitch");
4. of several such rows the LAST state before the pitch is taken (highest snapshot ``store_seq``, then newest
   row). If that one fails a later rule nothing is shown: an older row is not what was current when the pitch
   was thrown;
5. the policy's key counts automatic balls/strikes while the official ``pitchNumber`` might not, so the key alone
   could point one pitch off: the previous pitch of the same at-bat must already be in the decision's snapshot
   (and the pitch itself is not, by rule 3), which pins the pending pitch to this one;
6. the decision's own inputs equal the pitch as thrown: pitcher, batter, inning, half, outs, balls and strikes.
   A recommendation computed for the pitcher who was then replaced, or for a count that was later corrected,
   is not this pitch's recommendation.

A recommendation that left the screen before the pitch (the state went to 확인 중, or was being recomputed) is still
paired when rules 1-6 hold. Reason: the ledger cannot say what stayed on a screen, so the honest claim is the one
the label makes, "recorded before the pitch", and rule 6 already drops a decision whose inputs turned out not to
be the real pre-pitch situation. Nothing here claims the recommendation was being shown at the pitch.

Nothing is written. Ledger rows are append-only, so they are read incrementally (``LedgerCache``): one small
query per request, not one per pitch.
"""
from __future__ import annotations

from collections import OrderedDict
import threading
import time

from .display_state import RECEIPT_ALL

ABSENT = '이 공은 기록된 추천이 없어요.'  # the one calm sentence a fan reads when nothing is paired
IDENTITY = ('pitcher_id', 'batter_id')
SITUATION = ('inning', 'half', 'outs', 'balls', 'strikes')
# Refusal codes (machine-readable, never shown to a fan): which rule left a pitch without a recommendation.
RULES = ('no_store', 'ledger_unreadable', 'snapshot_not_committed', 'pitch_unconfirmed', 'pitch_first_seen_unknown',
         'no_recorded_recommendation', 'recorded_after_pitch_appeared', 'previous_pitch_not_in_decision',
         'decision_inputs_unknown', 'different_pitcher', 'different_batter', 'different_situation')
# Refusals that can pass without any pitch receipt changing (a read failed, the served version is not committed):
# the fan route marks ``receipts.version`` while one holds, so an open screen takes the full history again after.
TRANSIENT = ('ledger_unreadable', 'snapshot_not_committed', 'pitch_first_seen_unknown')
# Refusals a later provider correction can turn into a pairing or back (the count before a pitch, or who threw
# it, as the feed now tells it): the fan route folds which pitches hold one into ``receipts.version`` as well.
REVISABLE = ('different_pitcher', 'different_batter', 'different_situation')


def absent(rule):
    return {'provenance': 'absent', 'status': 'missing', 'reason': ABSENT, 'recorded_at': None, 'policy': None,
            'candidates': [], 'pairing': {'status': 'none', 'rule': rule}}


def decision_inputs(parsed):
    """The pre-pitch situation a decision was computed for (``live_feed.parse_state`` of its snapshot), as the
    ledger keeps it; None when the snapshot has no pending pitch."""
    if not isinstance(parsed, dict) or parsed.get('status') != 'ready' or not parsed.get('situation'):
        return None
    situation = parsed['situation']
    try:
        return {'pitcher_id': int(parsed['pitcher']['id']), 'batter_id': int(parsed['batter']['id'])} | {
            key: situation.get(key) for key in (*SITUATION, 'bases', 'home_score', 'away_score')}
    except (KeyError, TypeError, ValueError):
        return None


def _thrown(pitch):
    """The same facts of the pitch as thrown (``game_history`` row)."""
    situation = pitch.get('situation_before') or {}
    return {'pitcher_id': (pitch.get('pitcher') or {}).get('id'), 'batter_id': (pitch.get('batter') or {}).get('id')} | {
        key: situation.get(key) for key in SITUATION}


def slim(row):
    """What pairing needs of one ledger row (``LiveSnapshotStore.served_recommendations`` item)."""
    snapshot, displayed = row.get('snapshot') or {}, row.get('displayed') or {}
    return {'id': row['id'], 'outcome': row['outcome'], 'pitch_key': row.get('pitch_key'), 'decided': row.get('decided'),
            'decision_id': row.get('decision_id'), 'decision_clock': row.get('decision_clock'),
            'store_seq': snapshot.get('store_seq'), 'feed_sha256': snapshot.get('feed_sha256'),
            'policy_identity': displayed.get('policy_identity'),
            'candidates': ((displayed.get('recommendation') or {}).get('candidates') or []) if displayed.get('status') == 'ready' else [],
            'inputs': row.get('inputs'), 'derived': False}


def _rec(row):
    """A ledger row in the shape ``main.service_rec`` gives the pending pitch."""
    return {'provenance': 'captured_live', 'status': 'ready', 'reason': None,
            'ledger': {'status': 'recorded', 'outcome': row['outcome'], 'decision_id': row['decision_id'],
                       'decided': row['decided'], 'decision_clock': row['decision_clock']},
            'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(row['decided'])),
            'policy': {'name': 'ARM-B', 'identity': row['policy_identity']},
            'candidates': [{'rank': c.get('rank', i+1), 'pitch_type': c.get('pitch_type'), 'pitch_label': c.get('pitch_label'),
                            'zone_id': c.get('zone_id'), 'zone_label': c.get('zone_label'), 'target': c.get('target'),
                            'detail': {'probability': (c.get('detail') or {}).get('probability')}}
                           for i, c in enumerate(row['candidates'])],
            'pairing': {'status': 'paired', 'rule': 'recorded_before_pitch_same_inputs'}}


def pair(pitches, receipts, rows, served_seq, inputs_of=lambda row: row.get('inputs')):
    """``{pitch key: rec}`` for the visible ``pitches`` (``game_history`` rows, in game order).

    ``receipts`` are ``display_state.pitch_receipts`` of the served snapshot (as of that snapshot), ``rows`` the
    slim ledger rows of the game in append order, ``served_seq`` the store version being served, and
    ``inputs_of(row)`` the decision inputs of a row (None = unknown). Every pitch gets an entry: a paired
    recommendation or ``absent(rule)``."""
    by_key = {}
    for row in rows:
        if row['outcome'] == 'first' and row['candidates'] and type(row['store_seq']) is int and (
                served_seq is not None and row['store_seq'] <= served_seq):
            by_key.setdefault(row['pitch_key'], []).append(row)
    out, previous = {}, {}
    for pitch in pitches:
        key, at_bat = pitch['key'], pitch['at_bat_number']
        before, previous[at_bat] = previous.get(at_bat), (at_bat, pitch['pitch_number'])
        verdict = _refusal(pitch, before, receipts, by_key.get(key, ()), served_seq, inputs_of)
        out[key] = _rec(verdict) if isinstance(verdict, dict) else absent(verdict)
    return out


def _refusal(pitch, before, receipts, rows, served_seq, inputs_of):  # noqa: C901 - one rule per line, in order
    """The rule that refuses ``pitch`` a recommendation, or the ledger row (a dict) that is paired with it."""
    if served_seq is None:
        return 'snapshot_not_committed'
    item = receipts.get((pitch['at_bat_number'], pitch['pitch_number']))
    if not item or item['status'] != 'received' or RECEIPT_ALL in receipts:
        return 'pitch_unconfirmed'
    seen, received = item.get('first_seq'), item.get('first_received')
    if type(seen) is not int or received is None or seen > served_seq:
        return 'pitch_first_seen_unknown'
    if not rows:
        return 'no_recorded_recommendation'
    earlier = [row for row in rows if row['store_seq'] < seen and row['decided'] is not None and row['decided'] < received]
    if not earlier:
        return 'recorded_after_pitch_appeared'
    row = max(earlier, key=lambda r: (r['store_seq'], r['id']))
    if before is not None:  # the previous pitch of this at-bat was already in the decision's snapshot
        first = (receipts.get(before) or {}).get('first_seq')
        if type(first) is not int or first > row['store_seq']:
            return 'previous_pitch_not_in_decision'
    inputs, thrown = inputs_of(row), _thrown(pitch)
    if not isinstance(inputs, dict) or any(inputs.get(key) is None for key in (*IDENTITY, *SITUATION)):
        return 'decision_inputs_unknown'
    if inputs['pitcher_id'] != thrown['pitcher_id']:
        return 'different_pitcher'
    if inputs['batter_id'] != thrown['batter_id']:
        return 'different_batter'
    if any(inputs[key] != thrown[key] for key in SITUATION):
        return 'different_situation'
    return row


class LedgerCache:
    """Slim ledger rows per game, read incrementally. The table is append-only (update and delete abort), so a row
    read once never changes and each request only asks for rows newer than the last one it holds."""

    def __init__(self, games=8):
        self.games, self.held, self.lock = games, OrderedDict(), threading.Lock()

    def rows(self, store, game_pk):
        key = (str(getattr(store, 'path', id(store))), int(game_pk))
        with self.lock:
            held = self.held.setdefault(key, [])
            self.held.move_to_end(key)
            while len(self.held) > self.games:
                self.held.popitem(last=False)
            held.extend(slim(row) for row in store.served_recommendations(game_pk, after=held[-1]['id'] if held else 0))
            return list(held)

    def inputs(self, store, game_pk, row):
        """Decision inputs of ``row``: the ones the ledger row carries, else (a row written before inputs were
        recorded) the pending situation of the committed snapshot the row cites, read back from the store. That
        derived answer is kept (about 3 ms per row, once per process); a failed read is not: it raises to the
        caller, which shows nothing this time."""
        if row['inputs'] is not None or row['derived']:
            return row['inputs']
        from .live_feed import parse_state
        held = store.snapshot_feed(game_pk, row['store_seq'])
        inputs = None
        if held is not None and held[0] == row['feed_sha256']:
            try:
                inputs = decision_inputs(parse_state(held[1], None))
            except (KeyError, TypeError, AttributeError):
                inputs = None
        with self.lock:
            row['inputs'], row['derived'] = inputs, True
        return inputs
