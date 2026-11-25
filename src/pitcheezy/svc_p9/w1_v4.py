"""W1 v4 token reader (EXP-SVC-R4-003, demo build only): W1 v3 + MISSREPL.

MISSREPL  A structured pitching substitution (type action, isSubstitution True, details.eventType
          pitching_substitution, player.id present) whose provider body has NO ``replacedPlayer`` key breaks the RS-01
          reverse matchup chain (seq_shadow/captured_native_identity.py:_reverse_action, outgoing None -> None), so
          every earlier pitch of that plate appearance gets ``pitch_time_matchup_unverified``; because
          live_inputs.tokens_for refuses the query when ANY earlier pitch of the game is unverified, every later pitch
          of the game is omitted. 849799 PA 49 (mid-PA change Misiorowski -> Romero after pitch 1) is such a case.

          v4 fills the missing outgoing pitcher, on a private copy of the body, with the end-view matchup pitcher of
          the nearest earlier plate appearance of the SAME half (same fielding team): the pitcher who was active when
          that PA ended. It does so only when ALL of these hold (otherwise the event is left as is = v3 behaviour):
            * the event is the only pitching_substitution in its play and the play's matchup pitcher equals player.id;
            * every PA between the game start and this one is present in the body (no gap or conflicting view), and
              the nearest earlier same-half PA exists, has a pitch, a positive matchup pitcher that differs from
              player.id, and no pitching_substitution event after its last pitch;
            * no PA between that PA and this one (either half) holds any pitching_substitution event.
          The RS-01 reverse function then runs unchanged on the filled event (incoming == pitcher, outgoing present,
          incoming != outgoing). Filled events carry ``w1v4_replaced_player_inferred: True`` (not read by RS-01).
          ASSUMPTION (provider data): pitcher changes are recorded as substitution events, so the pitcher active at
          the end of the previous same-half PA stays active until the next pitching substitution.

With ``fill_missing_replaced=False`` this returns exactly ``w1_v3.inspect_tokens`` (tests check it).
"""
from __future__ import annotations

import copy

from pitcheezy.seq_shadow.captured_diagnostic import _get
from pitcheezy.seq_shadow.captured_native_identity import _positive, _views
from pitcheezy.svc_p9 import w1_v3

VERSION = 'v4'
EXCLUDED = w1_v3.EXCLUDED
INFERRED_KEY = 'w1v4_replaced_player_inferred'


def _is_pitching_sub(event):
    return (isinstance(event, dict) and event.get('isPitch') is False and event.get('isSubstitution') is True
            and _get(event, 'details', 'eventType') == 'pitching_substitution')


def _prior_same_half_pitcher(by_pa, conflict, pa, half):
    """End-view pitcher of the nearest earlier PA of the same half, or None when anything is uncertain."""
    if any(p not in by_pa or p in conflict for p in range(pa)):
        return None
    for p in range(pa - 1, -1, -1):
        play = by_pa[p]
        if _get(play, 'about', 'halfInning') != half:
            continue
        events = _get(play, 'playEvents')
        if not isinstance(events, list):
            return None
        last_pitch = max((i for i, e in enumerate(events) if isinstance(e, dict) and e.get('isPitch') is True), default=None)
        if last_pitch is None or any(_is_pitching_sub(e) for e in events[last_pitch + 1:]):
            return None
        # any pitching substitution recorded in a PA between that PA and this one (opposite half, e.g. a change for
        # the team about to take the field) makes the previous end-view pitcher uncertain (review R4-003 F1)
        for q in range(p + 1, pa):
            if any(_is_pitching_sub(e) for e in (_get(by_pa[q], 'playEvents') or [])):
                return None
        # source = the earlier PA's own matchup (end-view) pitcher, never a value filled earlier in this pass, so the
        # result does not depend on the fill order
        return _positive(_get(play, 'matchup', 'pitcher', 'id'))
    return None


def fill_missing_replaced(feed):
    """(copy of ``feed`` with MISSREPL outgoing pitchers filled, list of (pa, event index, incoming, outgoing))."""
    out = copy.deepcopy(feed)
    by_pa, conflict = _views(out)
    filled = []
    for pa, play in sorted(by_pa.items()):
        if pa in conflict:
            continue
        events = _get(play, 'playEvents')
        if not isinstance(events, list):
            continue
        subs = [e for e in events if _is_pitching_sub(e)]
        if len(subs) != 1 or 'replacedPlayer' in subs[0]:
            continue
        event = subs[0]
        incoming = _positive(_get(event, 'player', 'id'))
        if incoming is None or incoming != _positive(_get(play, 'matchup', 'pitcher', 'id')):
            continue
        half = _get(play, 'about', 'halfInning')
        if half not in ('top', 'bottom'):
            continue
        outgoing = _prior_same_half_pitcher(by_pa, conflict, pa, half)
        if outgoing is None or outgoing == incoming:
            continue
        event['replacedPlayer'] = {'id': outgoing}
        event[INFERRED_KEY] = True
        filled.append((pa, event.get('index'), incoming, outgoing))
    # apply each fill to EVERY view of that PA (allPlays entry and currentPlay) so the two views stay equal
    plays = _get(out, 'liveData', 'plays') or {}
    views = list(plays.get('allPlays') or []) + [plays.get('currentPlay')]
    for pa, index, incoming, outgoing in filled:
        for view in views:
            if not isinstance(view, dict) or _get(view, 'about', 'atBatIndex') != pa:
                continue
            for e in view.get('playEvents') or []:
                if _is_pitching_sub(e) and e.get('index') == index and _positive(_get(e, 'player', 'id')) == incoming:
                    e['replacedPlayer'] = {'id': outgoing}
                    e[INFERRED_KEY] = True
    return out, filled


def inspect_tokens(feed, *, game_pk, game_date, receipt_seq, source_sha256, fill_missing=True):
    """W1 v3 inspect_tokens on the MISSREPL-filled copy (``fill_missing=False``: exactly W1 v3)."""
    body = fill_missing_replaced(feed)[0] if fill_missing else feed
    return w1_v3.inspect_tokens(body, game_pk=game_pk, game_date=game_date, receipt_seq=receipt_seq,
                                source_sha256=source_sha256)
