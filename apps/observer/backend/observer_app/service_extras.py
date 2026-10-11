"""Optional additions to the finished live fan view (``/api/service/games/{pk}``), each behind an owner switch
(``features``; all OFF by default) and each from a standalone module that never raises into a route:

* A2 ``next.reference``: the reference card of ``profile_reference`` (past pitch frequencies of a pitcher the live
  recommender does not support). Attached only while the pending pitch's display state is ``unsupported`` for a
  pitcher-level reason and the phase is ``pre_pitch``. The honest state stays as it is: "추천 없음" with its reason;
  the card is a separate, labelled reference, not a recommendation and not a prediction of the next pitch.
* B2 ``pitches[i].expectation``: the one sentence of ``shadow_expectation`` ("던지기 전 예상: 헛스윙 x% · 땅볼 y%"),
  only on a PAST pitch that ended a plate appearance in a key moment and whose body is ``received``.

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
RETRY_S = 5.0  # a pitch without an expectation is asked again at most this often (the ledger may still be filling)


class Extras:
    def __init__(self, features=None, *, clock=time.monotonic):
        self._features, self._clock, self._lock = features or feature_flags.features, clock, threading.Lock()
        self._reference = self._reference_key = None
        self._shadow = self._shadow_key = None
        self._found, self._missed = {}, {}  # expectation cache per (ledger, pitch): a past pitch does not change

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
        key = (block.get('ledger'), bool(block.get('allow_replayed')))
        if not block.get('enabled') or not key[0]:
            return None
        with self._lock:
            if self._shadow is None or self._shadow_key != key:
                from .shadow_expectation import ShadowExpectations
                if self._shadow is not None:
                    self._shadow.close()
                self._shadow, self._shadow_key = ShadowExpectations(key[0], enabled=True, allow_replayed=key[1]), key
                self._found, self._missed = {}, {}
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
            nxt['reference'] = card

    def _attach_expectations(self, view, parsed, receipts, block):
        from .shadow_expectation import is_key_moment
        if receipts is None:  # no store: nothing says when a pitch was first received, or that its body is admitted
            return
        shadow, now = None, self._clock()
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
            mark = (pitch.get('key'), actual.get('pitch_type'), actual.get('event'), item.get('sha'), item.get('first_received'))
            found = self._found.get(mark)
            if found is None and now-self._missed.get(mark, -1e18) >= RETRY_S:
                found = shadow.for_pitch(pitch, pitch_first_received=item.get('first_received'))
                with self._lock:
                    if found is None:
                        self._missed[mark] = now
                    else:
                        self._found[mark] = found
                    for held in (self._found, self._missed):
                        while len(held) > 2048:
                            held.pop(next(iter(held)))
            if found is not None:
                pitch['expectation'] = found

    # ---- operator view (``/api/health``), never on a fan payload ------------------------------------------------
    def health(self):
        try:
            flags = self._features()
            out = {'profile_reference': {'enabled': bool(flags['profile_reference']['enabled']),
                                         'sources': len(flags['profile_reference'].get('profile_sources') or ())},
                   'step5_expectation': {'enabled': bool(flags['step5_expectation']['enabled']),
                                         'ledger_configured': bool(flags['step5_expectation'].get('ledger')),
                                         'allow_replayed': bool(flags['step5_expectation'].get('allow_replayed'))}}
            if self._reference is not None:
                out['profile_reference']['module'] = self._reference.health()
            if self._shadow is not None:
                out['step5_expectation']['module'] = self._shadow.status()
            return out
        except Exception as exc:  # noqa: BLE001
            return {'error': type(exc).__name__}
