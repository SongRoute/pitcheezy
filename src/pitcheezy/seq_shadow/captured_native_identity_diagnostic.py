"""Pure, unaccepted explanation of receipt-local native identity omissions.

This mirrors the guards in captured_native_identity; it does not supply tokens
or alter that helper's result. Callers may aggregate these in a separate audit.
"""
from __future__ import annotations

import re
from collections import Counter

from .captured_diagnostic import _get, inspect_body
from .captured_native_identity import (
    _ordered, _positive, _reverse_action, _views, inspect_receipt_identity,
)

MAX_DISTINCT_LABELS = 32
MAX_LABEL_LENGTH = 64
_LABEL = re.compile(r'[A-Za-z][A-Za-z_]*\Z')
_ACTION_FIELDS = frozenset({'index', 'isPitch', 'isSubstitution', 'count', 'details',
                            'player', 'replacedPlayer', 'battingOrder', 'position',
                            'pitcher', 'batter', 'matchup'})


def _shape(value, *, present):
    if not present:
        return 'absent'
    if value is None:
        return 'null'
    if isinstance(value, dict):
        return 'dict'
    if isinstance(value, str):
        return 'string' if value else 'empty_string'
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, (int, float)):
        return 'number'
    if isinstance(value, list):
        return 'list'
    return 'other'


def _label(value):
    if value is None:
        return 'null'
    if not isinstance(value, str):
        return 'non_string'
    if not value:
        return 'empty_string'
    if len(value) > MAX_LABEL_LENGTH or not _LABEL.fullmatch(value):
        return 'malformed_string'
    return value


def _id_state(event, field):
    if field not in event:
        return 'container_absent'
    container = event[field]
    if not isinstance(container, dict):
        return 'container_' + _shape(container, present=True)
    if 'id' not in container:
        return 'id_absent'
    return 'valid' if _positive(container['id']) is not None else 'id_malformed'


def _action_audit(event, pitcher):
    details = event.get('details')
    details_map = details if isinstance(details, dict) else {}
    extra_action = set(event) - _ACTION_FIELDS
    extra_details = set(details_map) - {'eventType'}
    if len(extra_action) > MAX_DISTINCT_LABELS or len(extra_details) > MAX_DISTINCT_LABELS:
        raise ValueError('diagnostic action metadata-key cap exceeded')
    flag = event.get('isSubstitution')
    kind = details_map.get('eventType')
    incoming = _positive(_get(event, 'player', 'id'))
    outgoing = _positive(_get(event, 'replacedPlayer', 'id'))
    if flag is True and kind == 'pitching_substitution':
        issue = ('incoming_id_invalid' if incoming is None else
                 'outgoing_id_invalid' if outgoing is None else
                 'incoming_pitcher_mismatch' if incoming != pitcher else
                 'same_incoming_outgoing' if incoming == outgoing else 'none')
    elif flag is False and kind == 'mound_visit':
        issue = ('details_extra_keys' if set(details_map) != {'eventType'} else
                 'action_extra_fields' if any(name in event for name in (
                     'player', 'replacedPlayer', 'battingOrder', 'position',
                     'pitcher', 'batter', 'matchup')) else 'none')
    elif flag is None and 'isSubstitution' not in event:
        issue = 'substitution_flag_absent'
    elif type(flag) is not bool:
        issue = 'substitution_flag_malformed'
    else:
        issue = 'action_kind_not_reversible'
    return {
        'event_type': 'absent' if 'eventType' not in details_map else _label(kind),
        'details_shape': _shape(details, present='details' in event),
        'substitution_flag': ('absent' if 'isSubstitution' not in event else
                              str(flag).lower() if type(flag) is bool else 'malformed'),
        'player_id': _id_state(event, 'player'),
        'replaced_player_id': _id_state(event, 'replacedPlayer'),
        'issue': issue,
        'extra_action_keys': tuple(sorted(_label(name) for name in extra_action)),
        'extra_details_keys': tuple(sorted(_label(name) for name in extra_details)),
    }


def inspect_identity_guards(feed, *, game_pk, game_date, all_pitch_rows):
    """Return one first-failing guard per keyed pitch in this exact body."""
    if (_get(feed, 'gameData', 'game', 'pk') != game_pk or
            _get(feed, 'gameData', 'datetime', 'officialDate') != game_date):
        raise ValueError('game/date mismatch')
    by_pa, conflict = _views(feed)  # Missing global prefix remains a hard failure.
    baseline = {tuple(row['key']): row for row in all_pitch_rows}
    keyed = {tuple(row['key']) for row in inspect_body(feed, game_pk=game_pk,
                                                       game_date=game_date) if row['key'] is not None}
    result = {}
    providers = Counter(
        event.get('playId') for play in by_pa.values()
        for event in (_get(play, 'playEvents') or [])
        if isinstance(event, dict) and event.get('isPitch') is True
        and isinstance(event.get('playId'), str) and event.get('playId'))
    for key in sorted(keyed):
        pa = key[1] - 1
        play = by_pa.get(pa)
        events = _get(play, 'playEvents')
        guard, blocking = None, None
        if pa in conflict:
            guard = 'view_conflict'
        elif not isinstance(events, list) or not events:
            guard = 'play_or_events_absent'
        elif all(isinstance(e, dict) and e.get('isPitch') is True for e in events):
            row = baseline.get(key)
            if not row or row['identity_hypothesis'] is None:
                guard = 'pitch_only_baseline_unavailable'
        else:
            if not _ordered(events):
                guard = 'mixed_timeline_order_invalid'
            else:
                pitcher = _positive(_get(play, 'matchup', 'pitcher', 'id'))
                batter = _positive(_get(play, 'matchup', 'batter', 'id'))
                if pitcher is None or batter is None:
                    guard = 'end_matchup_anchor_invalid'
                else:
                    valid = True
                    for event in reversed(events):
                        if event['isPitch'] is True:
                            if event['pitchNumber'] == key[2]:
                                if not valid:
                                    guard = 'reverse_action_chain_broken'
                                break
                        elif valid:
                            changed = _reverse_action(event, pitcher, batter)
                            if changed is None:
                                valid = False
                                blocking = _action_audit(event, pitcher)
                            else:
                                pitcher, batter, _ = changed
                    else:
                        guard = 'mixed_timeline_pitch_absent'
        if guard is None:
            event = next((e for e in events if isinstance(e, dict) and e.get('isPitch') is True
                          and e.get('pitchNumber') == key[2]), None)
            provider = event.get('playId') if event is not None else None
            guard = ('identity_hypothesis_present' if isinstance(provider, str) and provider
                     and providers[provider] == 1 else 'provider_id_missing_or_reused')
        result[key] = {'guard': guard, 'blocking_action': blocking, 'accepted': False}
    actual = inspect_receipt_identity(feed, game_pk=game_pk, game_date=game_date,
                                      all_pitch_rows=all_pitch_rows)
    present = {key for key, row in result.items() if row['guard'] == 'identity_hypothesis_present'}
    if present != set(actual):
        raise ValueError('diagnostic/helper identity-key divergence')
    return result


def summarize_guard_census(receipt_rows):
    """Aggregate (receipt_seq, diagnostic mapping) without exporting keys or IDs."""
    guards, events, issues, shapes, extra = (Counter() for _ in range(5))
    first, last = {}, {}
    for seq, rows in receipt_rows:
        for key, row in rows.items():
            first.setdefault(key, row['guard'])
            last[key] = row['guard']
            guards[row['guard']] += 1
            action = row['blocking_action']
            if action is None:
                continue
            events[action['event_type']] += 1
            issues[action['issue']] += 1
            for field in ('details_shape', 'substitution_flag', 'player_id', 'replaced_player_id'):
                shapes[f'{field}:{action[field]}'] += 1
            for field in ('extra_action_keys', 'extra_details_keys'):
                for label in action[field]:
                    extra[f'{field}:{label}'] += 1
    if len(events) > MAX_DISTINCT_LABELS or len(extra) > MAX_DISTINCT_LABELS:
        raise ValueError('diagnostic distinct-label cap exceeded')
    return {'receipt_views': sum(guards.values()), 'distinct_keys': len(first),
            'first_seen_guard': dict(sorted(Counter(first.values()).items())),
            'latest_seen_guard': dict(sorted(Counter(last.values()).items())),
            'guard': dict(sorted(guards.items())), 'event_type': dict(sorted(events.items())),
            'blocking_issue': dict(sorted(issues.items())),
            'field_shape': dict(sorted(shapes.items())),
            'extra_key': dict(sorted(extra.items()))}
