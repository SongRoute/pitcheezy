"""Experimental six-field source-native context, never a Statcast accepted row.

One stored receipt supplies each token hypothesis. Corrected retrospective I3
is not consulted. Query prestate is a separate next-hypothetical bridge input.
"""
from __future__ import annotations

from collections import Counter

from .captured_diagnostic import inspect_body
from .captured_identity_bip import inspect_hypotheses

CONTRACT = 'CAPTURE_NATIVE_SHADOW'
TOKEN_FIELDS = ('pitcher', 'batter', 'pitch_type', 'outcome10', 'balls', 'strikes')


def _hex(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def inspect_tokens(feed, *, game_pk, game_date, receipt_seq, source_sha256):
    """One unaccepted local-receipt token fate per physical pitch key."""
    if type(receipt_seq) is not int or receipt_seq <= 0 or not _hex(source_sha256):
        raise ValueError('receipt identity required')
    base_rows = inspect_body(feed, game_pk=game_pk, game_date=game_date)
    identity_rows = inspect_hypotheses(feed, game_pk=game_pk, game_date=game_date)
    by_key = {}
    for row in base_rows:
        if row['key'] is not None:
            by_key.setdefault(tuple(row['key']), []).append(row)
    identity = {tuple(row['key']): row for row in identity_rows}
    provider_counts = Counter(row['provider_id'] for row in base_rows if row['provider_id'] is not None)
    result = []
    for key, observed in sorted(by_key.items()):
        row = observed[0]
        helper = identity.get(key)
        native, reason = None, None
        if len(observed) != 1 or row['provider_id'] is None or provider_counts[row['provider_id']] != 1:
            reason = 'physical_or_provider_identity_unverified'
        elif helper is None or helper['identity_hypothesis'] is None:
            reason = 'pitch_time_matchup_unverified'
        elif row['type_state'] != 'known':
            reason = 'direct_type_unavailable'
        elif row['pre_count'] is None or not (0 <= row['pre_count'][0] <= 3 and 0 <= row['pre_count'][1] <= 2):
            reason = 'source_pre_count_unverified'
        else:
            nonbip = row['native_nonbip_candidate'] and row['outcome10_hypothesis'] is not None
            bip = (helper['description_bip_hypothesis'] == 'hit_into_play'
                   and helper['events_bip_hypothesis'] is not None
                   and helper['outcome10_bip_hypothesis'] is not None)
            if not (nonbip or bip):
                reason = 'source_outcome_unverified'
            else:
                matchup = helper['identity_hypothesis']
                native = dict(zip(TOKEN_FIELDS, (
                    matchup['pitcher'], matchup['batter'], row['observed']['event_type_code'],
                    row['outcome10_hypothesis'] if nonbip else helper['outcome10_bip_hypothesis'],
                    *row['pre_count'])))
                if not (type(native['outcome10']) is int and 0 <= native['outcome10'] <= 9):
                    native, reason = None, 'source_outcome_unverified'
                else:
                    reason = 'source_native_six_fields_unpaired'
        result.append({'key': key, 'provider_id': row['provider_id'] if len(observed) == 1 else None,
                       'receipt_seq': receipt_seq, 'source_sha256': source_sha256,
                       'input_contract': CONTRACT,
                       'status': 'native_candidate' if native is not None else 'native_unknown',
                       'reason': reason, 'native_token': native,
                       'statcast_training_equivalent': False,
                       'provider_identity_scope': 'single_local_receipt',
                       'accepted': False})
    return result
