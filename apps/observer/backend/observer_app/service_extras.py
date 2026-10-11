"""Optional additions to the finished live fan view (``/api/service/games/{pk}``), each behind an owner switch
(``features``; all OFF by default) and each from a standalone module that never raises into a route:

* A2 ``next.reference``: the reference card of ``profile_reference`` (past pitch frequencies of a pitcher the live
  recommender does not support). Attached only while the pending pitch's display state is ``unsupported`` for a
  pitcher-level reason and the phase is ``pre_pitch``. The honest state stays as it is: "추천 없음" with its reason;
  the card is a separate, labelled reference, not a recommendation and not a prediction of the next pitch.
* B2 ``pitches[i].expectation``: the one sentence of ``shadow_expectation`` ("던지기 전 예상: 헛스윙 x% · 땅볼 y%"),
  only on a PAST pitch that ended a plate appearance in a key moment and whose body is ``received``. Asked again
  on every request: a line the ledger later disowns disappears.

Both read nothing but the view being served and the snapshot it was built from (``parsed`` and ``receipts`` of
that snapshot), so a viewer on a delay gets what belongs to THEIR moment: a card for the count they are shown, an
expectation only for a pitch they have been served. With a switch off its field is absent and the body is exactly
what it was before this module (``tests/test_viewer_off_golden.py``).

``attach`` is the single call the route makes. Any failure here is logged and leaves the view untouched."""
from __future__ import annotations

import logging
import threading
import time

from . import features as feature_flags

LOGGER = logging.getLogger(__name__)
# Display-state reasons that say "this PITCHER is not covered" (``display_state.POLICY_REASONS``). An at-bat or
# game-level refusal is not a statement about the pitcher, so no card is offered for it.
PITCHER_REASONS = ('unknown_pitcher', 'no_supported_pitch_types')
# What of each module's answer goes on the fan payload (a published body is a public contract: only what a reader
# of it needs). Operator facts (profile hash and thresholds, attempt key, model hashes, omitted leaves) stay in the
# modules' own ``health()`` / ``status()`` under ``/api/health``.
REFERENCE_PUBLIC = ('kind', 'reason_code', 'chip', 'title', 'items', 'other', 'sample', 'as_of', 'note')
REFERENCE_ITEM, REFERENCE_SAMPLE = ('pitch_type', 'pitch_label', 'share_pct'), ('pitches', 'batter_side', 'balls', 'strikes')
EXPECTATION_PUBLIC = ('kind', 'lead', 'pitch_type', 'pitch_label', 'items', 'text', 'note', 'explain', 'basis', 'recorded')
RECENT_S = 12*3600  # the list guard only concerns games that started this recently


class Extras:
    def __init__(self, features=None, *, clock=time.monotonic):
        self._features, self._clock, self._lock = features or feature_flags.features, clock, threading.Lock()
        self._reference = self._reference_key = None
        self._shadow = self._shadow_key = None
        self._first_final = {}  # gamePk -> wall time the schedule first said Final (list guard)

    # ---- the modules, built on first use and again when the configuration changes -----------------------------
    def reference(self, block):
        """``ProfileReference`` for the configured sources, or None (off, or no source)."""
        sources = tuple(block.get('profile_sources') or ())
        if not block.get('enabled') or not sources:
            return None
        with self._lock:
            if self._reference is None or self._reference_key != sources:
                from .profile_reference import ProfileReference
                self._reference, self._reference_key = ProfileReference(list(sources)), sources
            return self._reference

    def shadow(self, block):
        """``ShadowExpectations`` for the configured ledger, or None (off, or no ledger)."""
        key = (block.get('ledger'), bool(block.get('allow_replayed')), block.get('ground_ball_cap'), block.get('whiff_cap'))
        if not block.get('enabled') or not key[0]:
            return None
        with self._lock:
            if self._shadow is None or self._shadow_key != key:
                from .shadow_expectation import ShadowExpectations
                if self._shadow is not None:
                    self._shadow.close()
                caps = {leaf: cap for leaf, cap in (('ground_ball_in_play', key[2]), ('narrow_whiff', key[3])) if cap is not None}
                self._shadow, self._shadow_key = ShadowExpectations(key[0], enabled=True, allow_replayed=key[1], caps=caps), key
            return self._shadow

    # ---- the one call of the route -----------------------------------------------------------------------------
    def attach(self, view, feed, parsed, receipts, entry=None):
        """Adds ``next.reference`` and ``pitches[i].expectation`` to the live ``view`` where the rules allow.
        ``feed``/``parsed``/``receipts`` are the served snapshot, its ``parse_state`` and its pitch receipts."""
        try:
            flags = self._features()
        except Exception:  # noqa: BLE001
            LOGGER.warning('feature switches unreadable; no optional additions', exc_info=True)
            return view
        for part, block in ((self._attach_reference, 'profile_reference'), (self._attach_expectations, 'step5_expectation')):
            try:
                if (flags.get(block) or {}).get('enabled'):
                    part(view, parsed, receipts, flags[block])
            except Exception:  # noqa: BLE001 - an optional addition never fails or changes the fan's game view
                LOGGER.warning('optional addition %s failed; the view is served without it', block, exc_info=True)
        return view

    def _attach_reference(self, view, parsed, receipts, block):
        state, current, nxt = view.get('display_state') or {}, view.get('current') or {}, view.get('next')
        if (state.get('state') != 'unsupported' or state.get('reason') not in PITCHER_REASONS
                or current.get('phase') != 'pre_pitch' or not isinstance(nxt, dict)):
            return
        reference = self.reference(block)
        if reference is None:
            return
        situation, pitcher, batter = parsed.get('situation') or {}, parsed.get('pitcher') or {}, parsed.get('batter') or {}
        hand = pitcher.get('hand') if pitcher.get('hand') in ('L', 'R') else None
        card = reference.card(pitcher.get('id'), batter.get('side'), situation.get('balls'), situation.get('strikes'),
                              (parsed.get('game') or {}).get('date'), pitcher_hand=hand)
        if card is not None:
            nxt['reference'] = self._public_card(card)

    @staticmethod
    def _public_card(card):
        slim = lambda item, keys: {key: item[key] for key in keys if key in item} if isinstance(item, dict) else None
        out = {key: card[key] for key in REFERENCE_PUBLIC if key in card}
        out['items'] = [slim(item, REFERENCE_ITEM) for item in card.get('items') or []]
        out['other'] = slim(card.get('other'), ('share_pct',))
        out['sample'] = slim(card.get('sample'), REFERENCE_SAMPLE)
        return out

    def _attach_expectations(self, view, parsed, receipts, block):
        from .shadow_expectation import is_key_moment
        if receipts is None:  # no store: nothing says when a pitch was first received, or that its body is admitted
            return
        shadow = None
        for pitch in view.get('pitches') or ():
            actual = pitch.get('actual') or {}
            if not is_key_moment(actual.get('event')) or (pitch.get('receipt') or {}).get('status') != 'received':
                continue
            item = receipts.get((pitch.get('at_bat_number'), pitch.get('pitch_number'))) or {}
            if item.get('status') != 'received':
                continue
            shadow = shadow or self.shadow(block)
            if shadow is None:
                return
            # Asked on every request and never kept here: the ledger can disown a forecast later (a quarantine, a
            # ``late`` order reconciliation, an unlinking at the end of the game, another player seen in the plate
            # appearance), and the line must then go. The module keeps its own per-attempt index, so this is cheap.
            found = shadow.for_pitch(pitch, pitch_first_received=item.get('first_received'))
            if found is not None:
                pitch['expectation'] = {key: found[key] for key in EXPECTATION_PUBLIC if key in found}

    # ---- games list for a delayed viewer (viewer delay on, ``?delay=`` > 0) -----------------------------------------
    def list_card(self, card, delay, feed=None):
        """A schedule card as a viewer ``delay`` seconds behind may see it. The schedule is the ballpark's present:
        a LIVE game loses score, inning, half and status words; a game the schedule calls FINAL (or suspended,
        postponed, cancelled with a score: anything that says the play stopped) stays presented as
        live (no score) until that viewer's own game page could have reached the end: the state the live buffer
        serves at that delay is Final, or ``delay`` seconds have passed since this server first saw the schedule
        say Final (fail closed: after a restart that clock starts again). Never polls and registers no viewer."""
        hidden = {'away_score': None, 'home_score': None, 'inning': None, 'half': None, 'status_text': None, 'score_hidden': True}
        try:
            if card.get('embargoed'):
                return card
            if card.get('status') == 'live' or card.get('live'):
                return card | hidden
            # every other card that is past its start: final, and suspended / postponed / cancelled with a score
            scored = card.get('away_score') is not None or card.get('home_score') is not None
            if card.get('status') == 'scheduled' or (card.get('status') != 'final' and not scored):
                return card
            from datetime import datetime
            now = feed.wall() if feed is not None and hasattr(feed, 'wall') else time.time()
            started = datetime.fromisoformat(str(card.get('start')).replace('Z', '+00:00')).timestamp()
            if not 0 <= now-started <= RECENT_S:
                return card
            game_pk = int(card['game_pk'])
            delay = min(float(delay), float(getattr(feed, 'max_delay_s', delay)))
            with self._lock:
                first = self._first_final.setdefault(game_pk, now)
                while len(self._first_final) > 256:
                    self._first_final.pop(next(iter(self._first_final)))
            if now-first >= delay:
                return card
            served = None
            if feed is not None and hasattr(feed, '_lock'):
                with feed._lock:  # a read of the buffer as it is: no fetch, no viewer registered
                    ready = [entry for entry in feed.buffers.get(game_pk, ()) if entry['event'] <= now-delay]
                    served = ready[-1]['feed'] if ready else None
            if served is not None and (served['gameData'].get('status') or {}).get('abstractGameState') == 'Final':
                return card
            return card | hidden | {'status': 'live', 'live': True}
        except Exception:  # noqa: BLE001 - when unsure the card says nothing about the score
            LOGGER.warning('list card for a delayed viewer could not be judged; shown without score', exc_info=True)
            return card | hidden | ({'status': 'live', 'live': True} if card.get('status') != 'scheduled' else {})

    # ---- operator view (``/api/health``), never on a fan payload ------------------------------------------------
    def health(self):
        try:
            flags = self._features()
            out = {'profile_reference': {'enabled': bool(flags['profile_reference']['enabled']),
                                         'sources': len(flags['profile_reference'].get('profile_sources') or ())},
                   'step5_expectation': {'enabled': bool(flags['step5_expectation']['enabled']),
                                         'ledger_configured': bool(flags['step5_expectation'].get('ledger')),
                                         'allow_replayed': bool(flags['step5_expectation'].get('allow_replayed')),
                                         'ground_ball_cap': flags['step5_expectation'].get('ground_ball_cap'),
                                         'whiff_cap': flags['step5_expectation'].get('whiff_cap')}}
            if self._reference is not None:
                out['profile_reference']['module'] = self._reference.health()
            if self._shadow is not None:
                out['step5_expectation']['module'] = self._shadow.status()
            return out
        except Exception as exc:  # noqa: BLE001
            return {'error': type(exc).__name__}
