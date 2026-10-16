"""Receipt-native field audit for future paired validation; never emits accepted tokens.

Every returned observation is a diagnostic candidate only. GUMBO call mapping,
play-level matchup, and post-event state are not paired Statcast row evidence.
The caller may retain this in memory; `summarize` exports counts only.
"""
from __future__ import annotations

from collections import Counter

from experiments.ws_campaign.source.first_small_features import TYPES
from .tokens import outcome10

# Non-BIP subset of the historical GUMBO call-code map in
# 20261011-rc-features-001/live/shadow-849809.proposed/shadow_runner.py.
# It is a comparison hypothesis, not a validated Statcast description map.
CALL_DESCRIPTION = {
    'B': 'ball', '*B': 'blocked_ball', 'C': 'called_strike',
    'S': 'swinging_strike', 'W': 'swinging_strike_blocked',
    'T': 'foul_tip', 'F': 'foul', 'L': 'foul_bunt',
    'M': 'missed_bunt', 'O': 'bunt_foul_tip', 'H': 'hit_by_pitch',
    'P': 'pitchout', 'I': 'intent_ball',
}
BIP_CALLS = frozenset(('X', 'D', 'E', 'Y', 'J', 'Z'))


def _get(obj, *path):
    for key in path:
        obj = obj.get(key) if isinstance(obj, dict) else None
    return obj


def _count(event):
    balls, strikes = _get(event, 'count', 'balls'), _get(event, 'count', 'strikes')
    return (balls, strikes) if type(balls) is type(strikes) is int and 0 <= balls <= 4 and 0 <= strikes <= 3 else None


def _int(value, low, high=(1 << 53)-1):
    return value if type(value) is int and low <= value <= high else None


def inspect_body(feed, *, game_pk, game_date):
    """Source-derived observations, keyed to one body; no row gets accepted status.

    Pre-count is reconstructed only from an unbroken ordered PA event prefix.
    All observed-pitch identity, before-outs/score, description and bb-type
    equivalence remain separate proofs for a later paired audit.
    """
    if (_get(feed, 'gameData', 'game', 'pk') != game_pk
            or _get(feed, 'gameData', 'datetime', 'officialDate') != game_date):
        raise ValueError('game/date mismatch')
    plays = _get(feed, 'liveData', 'plays')
    listed, current = _get(plays, 'allPlays'), _get(plays, 'currentPlay')
    if not isinstance(listed, list) or not isinstance(current, dict):
        raise ValueError('play prefix absent')
    by_pa = {}
    for play in [*listed, current]:
        index = _get(play, 'about', 'atBatIndex')
        if type(index) is not int or not 0 <= index <= 10000:
            continue
        by_pa[index] = play  # currentPlay is the latest source view of its PA.
    rows = []
    provider_keys = {}
    for index, play in sorted(by_pa.items()):
        events = _get(play, 'playEvents')
        if not isinstance(events, list):
            continue
        prior_count, next_pitch, prefix_complete = (0, 0), 1, True
        for event in events:
            if not isinstance(event, dict):
                prefix_complete = False
                prior_count = None
                continue
            if event.get('isPitch') is True:
                number, provider_id = event.get('pitchNumber'), event.get('playId')
                key = (game_pk, index + 1, number) if type(number) is int and 1 <= number <= 99 else None
                keyed = prefix_complete and key is not None and number == next_pitch
                if not keyed:
                    prefix_complete = False
                    prior_count = None
                next_pitch = number + 1 if type(number) is int and number >= next_pitch else next_pitch
                provider_ok = isinstance(provider_id, str) and 0 < len(provider_id) <= 256
                if provider_ok and provider_id in provider_keys and provider_keys[provider_id] != key:
                    provider_ok = False
                if provider_ok:
                    provider_keys[provider_id] = key
                code = _get(event, 'details', 'type', 'code')
                call = _get(event, 'details', 'call', 'code')
                description = CALL_DESCRIPTION.get(call) if isinstance(call, str) else None
                label = outcome10(description, '', prior_count[1])[0] if description and prior_count else None
                # Keep direct event fields distinct from play-final fields.
                # Neither the latter nor post-event count may label this
                # observed pitch's own before-state or pitcher identity.
                observed = {
                    'event_type_code': code if isinstance(code, str) else None,
                    'event_call_code': call if isinstance(call, str) else None,
                    'event_post_count': _count(event),
                    'event_post_outs': _int(_get(event, 'count', 'outs'), 0, 3),
                    'event_trajectory': _get(event, 'hitData', 'trajectory') if isinstance(_get(event, 'hitData', 'trajectory'), str) else None,
                    'play_final_pitcher': _int(_get(play, 'matchup', 'pitcher', 'id'), 1),
                    'play_final_batter': _int(_get(play, 'matchup', 'batter', 'id'), 1),
                    'play_final_stand': _get(play, 'matchup', 'batSide', 'code') if _get(play, 'matchup', 'batSide', 'code') in ('L', 'R') else None,
                    'play_final_throws': _get(play, 'matchup', 'pitchHand', 'code') if _get(play, 'matchup', 'pitchHand', 'code') in ('L', 'R') else None,
                    'play_final_outs': _int(_get(play, 'count', 'outs'), 0, 3),
                    'play_final_home_score': _int(_get(play, 'result', 'homeScore'), 0),
                    'play_final_away_score': _int(_get(play, 'result', 'awayScore'), 0),
                }
                if not keyed:
                    reason = 'physical_key_or_prefix_unverified'
                elif not provider_ok:
                    reason = 'provider_identity_unverified'
                elif prior_count is None:
                    reason = 'before_count_unverified'
                elif prior_count[0] > 3 or prior_count[1] > 2:
                    reason = 'before_count_out_of_training_bounds'
                elif isinstance(call, str) and call in BIP_CALLS:
                    reason = 'bip_event_and_bbtype_unverified'
                elif label is None or label < 0:
                    reason = 'call_mapping_unavailable'
                else:
                    reason = 'paired_row_quality_unverified'
                rows.append({'key': key, 'provider_id': provider_id if provider_ok else None,
                             'pre_count': prior_count, 'type_state': (
                                 'known' if code in TYPES and code != 'UNKNOWN' else
                                 'unknown_code' if isinstance(code, str) and code else 'absent'),
                             'description_hypothesis': description, 'outcome10_hypothesis': label,
                             'observed': observed,
                             'native_nonbip_candidate': reason == 'paired_row_quality_unverified',
                             'accepted': False, 'reason': reason,
                             'proof_missing': ('pitch_level_identity', 'before_outs_score',
                                               'statcast_description', 'statcast_bb_type')})
            if 'count' in event:
                prior_count = _count(event)
            else:
                prior_count = None
    return rows


def summarize(receipts):
    """Aggregate first-seen and latest distinct observations without row export.

    `receipts` is ordered `(receipt_seq, observations)`; caller verifies body
    custody/receipt times. Corrections never rewrite first-seen diagnostics.
    """
    first, latest, prior_seq, receipt_count = {}, {}, 0, 0
    for seq, rows in receipts:
        if type(seq) is not int or seq <= prior_seq:
            raise ValueError('receipt sequence must increase')
        prior_seq = seq
        receipt_count += 1
        for row in rows:
            key = row['key']
            if key is None:
                continue
            first.setdefault(key, (seq, row))
            latest[key] = (seq, row)
    def census(values):
        reasons = Counter(row['reason'] for _, row in values.values())
        return {'pitches': len(values), 'candidate_nonbip': sum(row['native_nonbip_candidate'] for _, row in values.values()),
                'reasons': dict(sorted(reasons.items()))}
    return {'receipt_count': receipt_count, 'last_receipt_seq': prior_seq,
            'first_seen': census(first), 'latest_seen': census(latest),
            'changed_keys': sum(first[key][1] != latest[key][1] for key in first)}
