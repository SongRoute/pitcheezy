"""Unaccepted receipt-local pitch matchup hypotheses across ordered PA actions.

The play matchup is a receipt-view end anchor. Reverse action history can
reconstruct earlier pitch IDs only under explicit provider-field assumptions;
it is not pitch-time ground truth and never supplies substituted hands.
"""
from __future__ import annotations

from collections import Counter

from .captured_diagnostic import _get, _int

BENIGN_NONPITCH = frozenset({'mound_visit'})


def _positive(value):
    return _int(value, 1)


def _views(feed):
    plays = _get(feed, 'liveData', 'plays')
    listed, current = _get(plays, 'allPlays'), _get(plays, 'currentPlay')
    if not isinstance(listed, list) or not isinstance(current, dict):
        raise ValueError('play prefix absent')
    by_pa, conflict = {}, set()
    for play in listed:
        pa = _get(play, 'about', 'atBatIndex')
        if type(pa) is int and 0 <= pa <= 10000:
            if pa in by_pa and by_pa[pa] != play:
                conflict.add(pa)
            by_pa.setdefault(pa, play)
    pa = _get(current, 'about', 'atBatIndex')
    if type(pa) is int and 0 <= pa <= 10000:
        if pa in by_pa and by_pa[pa] != current:
            conflict.add(pa)
        by_pa[pa] = current
    return by_pa, conflict


def _ordered(events):
    if not isinstance(events, list) or not events:
        return False
    prior, pitch = -1, 0
    for event in events:
        if not isinstance(event, dict) or type(event.get('index')) is not int:
            return False
        index = event['index']
        if not prior < index <= 10000:
            return False
        prior = index
        if event.get('isPitch') is True:
            pitch += 1
            if type(event.get('pitchNumber')) is not int or event['pitchNumber'] != pitch:
                return False
        elif event.get('isPitch') is not False:
            return False
    return pitch > 0


def _reverse_action(event, pitcher, batter):
    """Return prior IDs and assumption, or None when the active chain breaks."""
    kind = _get(event, 'details', 'eventType')
    flag = event.get('isSubstitution')
    if flag is True and kind == 'pitching_substitution':
        incoming, outgoing = _positive(_get(event, 'player', 'id')), _positive(_get(event, 'replacedPlayer', 'id'))
        if incoming == pitcher and outgoing is not None and incoming != outgoing:
            return outgoing, batter, 'structured_pitching_substitution'
    elif (flag is False and isinstance(kind, str) and kind in BENIGN_NONPITCH
          and isinstance(event.get('details'), dict)
          and set(event['details']) == {'eventType'}
          and all(name not in event for name in ('player', 'replacedPlayer', 'battingOrder',
                                               'position', 'pitcher', 'batter', 'matchup'))):
        return pitcher, batter, 'provider_flag_benign_mound_visit_assumption'
    return None


def _mixed_timeline(play, game_pk, pa):
    events = _get(play, 'playEvents')
    if not _ordered(events):
        return {}
    pitcher = _positive(_get(play, 'matchup', 'pitcher', 'id'))
    batter = _positive(_get(play, 'matchup', 'batter', 'id'))
    if pitcher is None or batter is None:
        return {}
    valid, transitions, result = True, [], {}
    for event in reversed(events):
        if event['isPitch'] is True:
            if valid:
                key = (game_pk, pa + 1, event['pitchNumber'])
                result[key] = {'identity_hypothesis': {'pitcher': pitcher, 'batter': batter,
                                                       'stand': None, 'throws': None},
                               'identity_provenance': {
                                   'rule': 'receipt_local_reverse_matchup',
                                   'anchor': 'same_receipt_play_matchup_end_view',
                                   'pitch_event_index': event['index'],
                                   'later_action_chain': list(reversed(transitions)),
                                   'hands': 'unverified_across_nonpitch_action',
                                   'accepted': False}}
            continue
        changed = _reverse_action(event, pitcher, batter) if valid else None
        if changed is None:
            valid = False  # earlier segment unknown; later pitch hypotheses remain.
        else:
            pitcher, batter, assumption = changed
            transitions.append({'index': event['index'], 'assumption': assumption})
    return result


def inspect_receipt_identity(feed, *, game_pk, game_date, all_pitch_rows):
    """Return per-key unaccepted identity/provenance, using no other receipt."""
    if (_get(feed, 'gameData', 'game', 'pk') != game_pk
            or _get(feed, 'gameData', 'datetime', 'officialDate') != game_date):
        raise ValueError('game/date mismatch')
    by_pa, conflict = _views(feed)
    baseline = {tuple(row['key']): row for row in all_pitch_rows}
    result = {}
    for pa, play in by_pa.items():
        if pa in conflict:
            continue
        events = _get(play, 'playEvents')
        if not isinstance(events, list) or not events:
            continue
        if all(isinstance(event, dict) and event.get('isPitch') is True for event in events):
            for event in events:
                number = event.get('pitchNumber')
                key = (game_pk, pa + 1, number) if type(number) is int else None
                row = baseline.get(key)
                if row and row['identity_hypothesis'] is not None:
                    result[key] = {'identity_hypothesis': row['identity_hypothesis'],
                                   'identity_provenance': {
                                       'rule': 'pitch_only_play_matchup',
                                       'anchor': 'same_receipt_play_matchup_view',
                                       'accepted': False}}
        else:
            result.update(_mixed_timeline(play, game_pk, pa))
    providers = Counter(
        event.get('playId') for play in by_pa.values()
        for event in (_get(play, 'playEvents') or [])
        if isinstance(event, dict) and event.get('isPitch') is True
        and isinstance(event.get('playId'), str) and event.get('playId'))
    for key in list(result):
        play = by_pa[key[1] - 1]
        events = _get(play, 'playEvents')
        event = next((e for e in events if isinstance(e, dict) and e.get('isPitch') is True
                      and e.get('pitchNumber') == key[2]), None)
        provider = event.get('playId') if event is not None else None
        if not isinstance(provider, str) or not provider or providers[provider] != 1:
            del result[key]
    return result
