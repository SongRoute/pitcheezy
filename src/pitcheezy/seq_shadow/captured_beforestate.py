"""Generated-only hypotheses for a first pitch's before-outs and score.

The prior completed PA's final state is the sole source. This is paired-audit
material, not accepted normalization evidence or an inference query.
"""
from __future__ import annotations

from .captured_diagnostic import inspect_body


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


def first_pitch_candidates(feed, *, game_pk, game_date):
    """Return keyed hypotheses with explicit failure reasons; never accepted rows."""
    rows = inspect_body(feed, game_pk=game_pk, game_date=game_date)
    plays = _get(feed, 'liveData', 'plays')
    by_pa = {}
    for play in [*_get(plays, 'allPlays'), _get(plays, 'currentPlay')]:
        index = _int(_get(play, 'about', 'atBatIndex'), 0, 10000)
        if index is not None:
            by_pa[index] = play
    result = []
    for row in rows:
        key = row['key']
        if key is None or key[2] != 1:
            continue
        index = key[1] - 1
        play, prior = by_pa.get(index), by_pa.get(index - 1)
        reason = None
        before = None
        if prior is None or play is None:
            reason = 'immediate_prior_completed_pa_absent'
        elif _get(prior, 'about', 'isComplete') is not True:
            reason = 'prior_pa_completion_unverified'
        elif (not isinstance(_get(play, 'playEvents'), list) or not _get(play, 'playEvents')
              or not isinstance(_get(play, 'playEvents')[0], dict)
              or _get(play, 'playEvents')[0].get('isPitch') is not True):
            reason = 'intervening_action_unverified'
        else:
            old_half, new_half = _half(prior), _half(play)
            old_outs = _int(_get(prior, 'count', 'outs'), 0, 3)
            score = (_int(_get(prior, 'result', 'homeScore'), 0),
                     _int(_get(prior, 'result', 'awayScore'), 0))
            if old_half is None or new_half is None or old_outs is None or None in score:
                reason = 'prior_final_state_unverified'
            elif new_half == old_half and old_outs <= 2:
                before = (old_outs, *score)
            elif old_outs == 3 and ((old_half[1] and new_half == (old_half[0], False))
                                      or (not old_half[1] and new_half == (old_half[0] + 1, True))):
                before = (0, *score)
            else:
                reason = 'half_transition_unverified'
        result.append({'key': key, 'before_outs_score_hypothesis': before,
                       'source': 'immediately_prior_completed_pa_final' if before is not None else None,
                       'accepted': False, 'reason': reason or 'paired_beforestate_unverified'})
    return result
