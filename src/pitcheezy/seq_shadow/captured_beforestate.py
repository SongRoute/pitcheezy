"""Receipt-native before-outs/score hypotheses for a contiguous PA pitch prefix.

Only the immediately prior completed PA supplies the initial state. Subsequent
nonterminal, non-BIP pitches may carry that state if their direct event counts
show unchanged outs. Play-level runner movements cannot be timed to each
pitch; absent direct movement is not proof of no movement. Scores remain
hypotheses for a separately registered paired audit. No row is accepted.
"""
from __future__ import annotations

from collections import Counter

from .captured_diagnostic import BIP_CALLS, CALL_DESCRIPTION, inspect_body


def _get(obj, *path):
    for key in path:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def _int(value, low, high=(1 << 53)-1):
    return value if type(value) is int and low <= value <= high else None


def _half(play):
    inning = _int(_get(play, 'about', 'inning'), 1, 30)
    top = _get(play, 'about', 'isTopInning')
    return (inning, top) if inning is not None and type(top) is bool else None


def _seed(prior, play):
    if prior is None:
        return None, 'immediate_prior_completed_pa_absent'
    if _get(prior, 'about', 'isComplete') is not True:
        return None, 'prior_pa_completion_unverified'
    old_half, new_half = _half(prior), _half(play)
    old_outs = _int(_get(prior, 'count', 'outs'), 0, 3)
    score = (_int(_get(prior, 'result', 'homeScore'), 0),
             _int(_get(prior, 'result', 'awayScore'), 0))
    if old_half is None or new_half is None or old_outs is None or None in score:
        return None, 'prior_final_state_unverified'
    if new_half == old_half and old_outs <= 2:
        return (old_outs, *score), None
    if old_outs == 3 and ((old_half[1] and new_half == (old_half[0], False))
                          or (not old_half[1] and new_half == (old_half[0] + 1, True))):
        return (0, *score), None
    return None, 'half_transition_unverified'


def _can_carry(event, state):
    details = _get(event, 'details')
    count = _get(event, 'count')
    if not isinstance(details, dict) or not isinstance(count, dict):
        return False
    call = _get(details, 'call', 'code')
    if (not isinstance(call, str) or call not in CALL_DESCRIPTION or call in BIP_CALLS or call == 'H'
            or details.get('isInPlay') is not False or details.get('isOut') is not False
            or details.get('eventType') or _get(event, 'hitData')):
        return False
    balls, strikes = _int(count.get('balls'), 0, 3), _int(count.get('strikes'), 0, 2)
    outs = _int(count.get('outs'), 0, 3)
    if None in (balls, strikes, outs) or outs != state[0]:
        return False
    # Explicit pitch-level movement/score may veto the carry, never seed it.
    # Play-level runners/result are final PA fields and cannot time the change.
    for holder in (event, details, _get(event, 'result'), _get(details, 'result')):
        if not isinstance(holder, dict):
            continue
        if holder.get('runners') or holder.get('runner') or holder.get('eventType'):
            return False
        for field, value in (('homeScore', state[1]), ('awayScore', state[2])):
            supplied = holder.get(field)
            if supplied is not None and supplied != value:
                return False
    return True


def prefix_candidates(feed, *, game_pk, game_date):
    """Return physical-keyed diagnostic hypotheses with explicit abstentions."""
    rows = inspect_body(feed, game_pk=game_pk, game_date=game_date)
    plays = _get(feed, 'liveData', 'plays')
    listed = _get(plays, 'allPlays')
    current = _get(plays, 'currentPlay')
    by_pa, conflicting = {}, set()
    for play in listed:
        index = _int(_get(play, 'about', 'atBatIndex'), 0, 10000)
        if index is None:
            continue
        if index in by_pa and by_pa[index] != play:
            conflicting.add(index)
        by_pa.setdefault(index, play)
    current_index = _int(_get(current, 'about', 'atBatIndex'), 0, 10000)
    if current_index is not None:
        by_pa[current_index] = current  # one expected currentPlay override
    keyed = {}
    for row in rows:
        if row['key'] is not None:
            keyed.setdefault(row['key'], []).append(row)
    provider_frequency = Counter(
        event.get('playId') for play in by_pa.values()
        for event in (_get(play, 'playEvents') or [])
        if isinstance(event, dict) and event.get('isPitch') is True
        and isinstance(event.get('playId'), str) and event.get('playId'))
    result = []
    for index, play in sorted(by_pa.items()):
        events = _get(play, 'playEvents')
        if not isinstance(events, list):
            continue
        state, seed_reason = _seed(by_pa.get(index-1), play)
        if index in conflicting or index-1 in conflicting:
            state, seed_reason = None, 'conflicting_listed_pa_views'
        stopped = False
        next_pitch = 1
        for event in events:
            if not isinstance(event, dict) or event.get('isPitch') is not True:
                stopped = True
                continue
            number = event.get('pitchNumber')
            key = (game_pk, index+1, number) if type(number) is int and 1 <= number <= 99 else None
            if key is None:
                stopped = True
                continue
            matches = keyed.get(key, [])
            provider = matches[0]['provider_id'] if len(matches) == 1 else None
            identity_ok = len(matches) == 1 and provider is not None and provider_frequency[provider] == 1
            if not identity_ok:
                reason = 'first_pitch_identity_unverified' if number == 1 else 'pitch_identity_unverified'
            elif number != next_pitch:
                reason = 'pitch_prefix_unverified'
            elif stopped:
                reason = 'intervening_action_or_terminal_pitch'
            elif state is None:
                reason = seed_reason or 'prior_pitch_transition_unverified'
            elif matches[0]['pre_count'] is None:
                reason = 'before_count_unverified'
            else:
                reason = 'paired_beforestate_unverified'
            before = state if reason == 'paired_beforestate_unverified' else None
            result.append({'key': key, 'before_outs_score_hypothesis': before,
                           'source': ('immediately_prior_completed_pa_final_then_direct_pitch_counts'
                                      if before is not None else None),
                           'accepted': False, 'reason': reason,
                           'proof_missing': ('pitch_level_score_and_runner_transition',
                                             'paired_statcast_beforestate')})
            if (reason != 'paired_beforestate_unverified' or not _can_carry(event, state)):
                stopped = True
            next_pitch += 1
    return result
