"""Unaccepted GUMBO matchup and BIP hypotheses for a paired field audit.

Play matchup and result are final-play views, so even a clean pitch-only PA
does not prove pitch-level identity or outcome at first receipt. A direct
trajectory is compared only when it already uses the frozen bb_type vocabulary.
"""
from __future__ import annotations

from collections import Counter

from experiments.ws_fullseason.data import closure
from .captured_diagnostic import BIP_CALLS, _get, inspect_body
from .tokens import outcome10

BB_TYPES = closure().safe.BB_TYPES


def _int(value):
    return value if type(value) is int and 1 <= value < (1 << 53) else None


def inspect_hypotheses(feed, *, game_pk, game_date):
    """One diagnostic record per physical key, with ambiguous keys unfilled."""
    rows = inspect_body(feed, game_pk=game_pk, game_date=game_date)
    plays = _get(feed, 'liveData', 'plays')
    by_pa, conflicting = {}, set()
    for play in _get(plays, 'allPlays'):
        index = _get(play, 'about', 'atBatIndex')
        if type(index) is not int or not 0 <= index <= 10000:
            continue
        if index in by_pa and by_pa[index] != play:
            conflicting.add(index)
        by_pa.setdefault(index, play)
    current = _get(plays, 'currentPlay')
    index = _get(current, 'about', 'atBatIndex')
    if type(index) is int and 0 <= index <= 10000:
        by_pa[index] = current  # expected currentPlay replacement
    keyed = {}
    for row in rows:
        if row['key'] is not None:
            keyed.setdefault(tuple(row['key']), []).append(row)
    provider_frequency = Counter(
        event.get('playId') for play in by_pa.values()
        for event in (_get(play, 'playEvents') or [])
        if isinstance(event, dict) and event.get('isPitch') is True
        and isinstance(event.get('playId'), str) and event.get('playId'))
    result = []
    for key, matches in sorted(keyed.items()):
        pa = key[1] - 1
        play = by_pa.get(pa)
        events = _get(play, 'playEvents')
        row = matches[0]
        provider = row['provider_id'] if len(matches) == 1 else None
        unique = (len(matches) == 1 and provider is not None
                  and provider_frequency[provider] == 1 and pa not in conflicting)
        clean_pa = (unique and isinstance(events, list) and bool(events)
                    and all(isinstance(e, dict) and e.get('isPitch') is True for e in events))
        identity = None
        if clean_pa:
            observed = row['observed']
            identity = {name: observed['play_final_' + name] for name in
                        ('pitcher', 'batter', 'stand', 'throws')}
            if (any(identity[name] is None for name in identity)
                    or _int(identity['pitcher']) is None or _int(identity['batter']) is None):
                identity = None
        identity_reason = ('play_matchup_hypothesis_unpaired' if identity is not None
                           else 'provider_or_pa_identity_unverified')

        bip_description, bip_events, bip_outcome, bb_type = None, None, None, None
        bip_reason = 'not_final_completed_bip'
        if unique and isinstance(events, list) and _get(play, 'about', 'isComplete') is True:
            pitch_events = [e for e in events if isinstance(e, dict) and e.get('isPitch') is True]
            final = pitch_events[-1] if pitch_events else None
            if (final is not None and final.get('pitchNumber') == key[2]
                    and _get(final, 'details', 'isInPlay') is True
                    and row['observed']['event_call_code'] in BIP_CALLS):
                event_type = _get(play, 'result', 'eventType')
                trajectory = row['observed']['event_trajectory']
                bip_description = 'hit_into_play'
                if isinstance(event_type, str) and event_type and row['pre_count'] is not None:
                    bip_events = event_type
                    label, _ = outcome10('hit_into_play', event_type, row['pre_count'][1])
                    if label >= 0:
                        bip_outcome = label
                if trajectory in BB_TYPES:
                    bb_type = trajectory
                bip_reason = ('bip_result_and_trajectory_hypotheses_unpaired'
                              if bip_outcome is not None and bb_type is not None
                              else 'bip_result_or_trajectory_unavailable')
        result.append({'key': key, 'identity_hypothesis': identity,
                       'description_bip_hypothesis': bip_description,
                       'events_bip_hypothesis': bip_events,
                       'outcome10_bip_hypothesis': bip_outcome,
                       'bb_type_hypothesis': bb_type,
                       'accepted': False,
                       'reasons': {'identity': identity_reason, 'bip': bip_reason},
                       'proof_missing': ('paired_pitch_level_identity',
                                         'paired_bip_event_and_bb_type')})
    return result
