"""Strict GUMBO receipt bridge for a conditional next-pitch shadow query.

An observed GUMBO pitch is not a verified Statcast training row. Its fate stays
unknown until a separately validated normalizer can prove the full acceptance
contract. This bridge only derives a possible *next* pitch from the current
post-last-pitch state of one captured body. It never labels an observed pitch
with that later state or claims upstream first availability.
"""
from __future__ import annotations

from .fullseason_stream import CONTEXT
from experiments.ws_campaign.source.first_small_features import TYPES


def _get(obj, *path):
    for key in path:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def _int(value, lo, hi):
    return value if type(value) is int and lo <= value <= hi else None


def _missing(reason, events=()):
    return {'events': list(events), 'candidate': None, 'abstain_reason': reason}


def step(feed, *, game_pk, game_date, receipt_seq, source_sha256, candidate_pitch_type='FF'):
    """Return receipt events and at most one explicit hypothetical next query.

    All source rows are UNKNOWN because GUMBO alone cannot prove paired
    Statcast description/bb_type and historical prestate quality. Revision is
    the receipt ordinal: replaying a body with a newer ordinal is an append-only
    observation; the ledger preserves its original first local receipt.
    """
    if (_int(game_pk, 1, (1 << 53)-1) is None or _int(receipt_seq, 1, (1 << 53)-1) is None
            or candidate_pitch_type not in TYPES or not isinstance(source_sha256, str)
            or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256)):
        raise ValueError('invalid receipt identity or frozen candidate type')
    if _get(feed, 'gameData', 'game', 'pk') != game_pk:
        return _missing('game_identity_unverified')
    if _get(feed, 'gameData', 'datetime', 'officialDate') != game_date:
        return _missing('official_date_unverified')
    plays = _get(feed, 'liveData', 'plays')
    listed, current = _get(plays, 'allPlays'), _get(plays, 'currentPlay')
    if not isinstance(listed, list) or not isinstance(current, dict):
        return _missing('play_prefix_absent')

    # Preserve each physical pitch once per receipt. Conflicting duplicates or
    # unkeyed pitches make stored-prefix completeness unknowable.
    physical, provider_keys, complete, identity_conflict, seen_plays = {}, {}, True, False, {}
    for play in [*listed, current]:
        index, pitch_events = _get(play, 'about', 'atBatIndex'), _get(play, 'playEvents')
        if _int(index, 0, 10000) is None or not isinstance(pitch_events, list):
            complete = False
            continue
        seen_plays[index] = play
        local_numbers = []
        for event in pitch_events:
            if not isinstance(event, dict):
                complete = False
                continue
            if event.get('isPitch') is not True:
                continue
            number = _int(event.get('pitchNumber'), 1, 99)
            if number is None:
                complete = False
                continue
            local_numbers.append(number)
            key = (game_pk, index + 1, number)
            provider_id = event.get('playId')
            if not isinstance(provider_id, str) or not provider_id:
                complete = False
                continue
            old = physical.get(key)
            if old is not None and old != provider_id:
                complete = False
                identity_conflict = True
            if provider_id in provider_keys and provider_keys[provider_id] != key:
                complete = False
                identity_conflict = True
            provider_keys[provider_id] = key
            physical[key] = provider_id
        if local_numbers != list(range(1, len(local_numbers)+1)):
            complete = False
    events = [dict(event_id=physical[key], key=key, revision=receipt_seq,
                   row={'status': 'unknown', 'reason': 'statcast_acceptance_unverified', 'accepted': None})
              for key in sorted(physical)]

    if identity_conflict:
        return _missing('provider_identity_conflict', ())

    current_index = _get(current, 'about', 'atBatIndex')
    if _int(current_index, 0, 10000) is None or set(seen_plays) != set(range(current_index+1)):
        complete = False
    if not complete:
        return _missing('stored_prefix_unkeyed_or_conflicting', events)
    if _get(feed, 'gameData', 'status', 'abstractGameState') != 'Live':
        return _missing('game_not_live', events)
    about, count, matchup = (_get(current, k) for k in ('about', 'count', 'matchup'))
    index, pitch_events = _get(about, 'atBatIndex'), _get(current, 'playEvents')
    if _int(index, 0, 10000) is None or not isinstance(pitch_events, list):
        return _missing('current_play_unkeyed', events)
    if _get(about, 'isComplete') is not False:
        return _missing('pa_ended_or_completion_unknown', events)
    if any(not isinstance(e, dict) for e in pitch_events):
        return _missing('current_event_unknown', events)
    # A non-pitch action may change count, runners, score, outs or matchup. The
    # narrow source bridge does not infer which; its conservative ceiling is an
    # abstention until an event-aware reducer is independently validated.
    if any(e.get('isPitch') is not True for e in pitch_events):
        return _missing('nonpitch_action_in_current_pa', events)
    numbers = [e.get('pitchNumber') for e in pitch_events]
    if numbers != list(range(1, len(numbers)+1)):
        return _missing('pitch_sequence_unkeyed', events)
    last = pitch_events[-1] if pitch_events else None
    if last is not None:
        details = _get(last, 'details')
        if not isinstance(details, dict) or type(details.get('isInPlay')) is not bool:
            return _missing('last_pitch_fate_unknown', events)
        if (details['isInPlay'] or details.get('isOut') is True
                or _get(details, 'call', 'code') == 'H'):
            return _missing('last_pitch_ended_pa', events)
        if (_get(last, 'count', 'balls'), _get(last, 'count', 'strikes'), _get(last, 'count', 'outs')) != (
                _get(count, 'balls'), _get(count, 'strikes'), _get(count, 'outs')):
            return _missing('count_sources_disagree', events)
    balls, strikes, outs = (_int(_get(count, k), 0, hi) for k, hi in
                            (('balls', 3), ('strikes', 2), ('outs', 2)))
    inning = _int(_get(about, 'inning'), 1, 30)
    top = _get(about, 'isTopInning')
    pitcher = _int(_get(matchup, 'pitcher', 'id'), 1, (1 << 53)-1)
    batter = _int(_get(matchup, 'batter', 'id'), 1, (1 << 53)-1)
    stand, throws = _get(matchup, 'batSide', 'code'), _get(matchup, 'pitchHand', 'code')
    score = tuple(_int(_get(current, 'result', k), 0, (1 << 53)-1)
                  for k in ('homeScore', 'awayScore'))
    line = _get(feed, 'liveData', 'linescore')
    offense = _get(line, 'offense')
    if (None in (balls, strikes, outs, inning, pitcher, batter, *score)
            or type(top) is not bool or stand not in ('L', 'R') or throws not in ('L', 'R')
            or not isinstance(offense, dict)
            or _get(line, 'currentInning') != inning or _get(line, 'isTopInning') is not top
            or _get(line, 'outs') != outs
            or _get(line, 'teams', 'home', 'runs') != score[0]
            or _get(line, 'teams', 'away', 'runs') != score[1]):
        return _missing('next_prestate_unverified', events)
    bases = []
    for base in ('first', 'second', 'third'):
        item = offense.get(base)
        if item is None:
            bases.append(None)
        else:
            runner = _int(_get(item, 'id'), 1, (1 << 53)-1)
            if runner is None:
                return _missing('runner_identity_unverified', events)
            bases.append(runner)
    key = (game_pk, index + 1, len(numbers) + 1)
    if key[2] > 99:
        return _missing('next_pitch_number_out_of_range', events)
    if key in physical:
        return _missing('next_key_already_observed', events)
    query = dict(zip(CONTEXT, (*key, game_date, pitcher, batter, candidate_pitch_type,
                               balls, strikes, outs, inning, 'Top' if top else 'Bot',
                               *score, *bases, stand, throws)))
    candidate = dict(decision_id=f'{game_pk}:{index+1}:{key[2]}:{source_sha256}',
                     query=query, coverage={'complete': True, 'before_key': list(key),
                                            'source_sha256': source_sha256},
                     pitch_type_basis='hypothetical_candidate')
    return {'events': events, 'candidate': candidate, 'abstain_reason': None}
