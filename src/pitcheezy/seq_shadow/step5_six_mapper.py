"""Typed corrected-Statcast FF prestate hypothesis for an optional Step 5 comparison.

This is retrospective and T-ready-only. It does not emulate a live GUMBO feed,
establish physical pre-pitch availability, load a Step 5 profile, or score.
"""
from __future__ import annotations

from .fullseason_stream import SCHEMA, TYPES
from .ledger import sha
from .postcutoff_offline import SIX

CONTRACT = 'CORRECTED_STATCAST_FF_T_READY_RETROSPECTIVE'


def map_snapshot(snapshot, *, source_sha256):
    """Map one immutable T-ready observed FF snapshot to Step 5 typed raw context.

    Result state has only fields consumed by Step 5 ``context_raw``,
    ``build_candidates`` and ``ProfileStore.support``. The fixed Step 5
    support guard and historical profile remain external future gates.
    """
    if not isinstance(snapshot, dict) or 'snapshot_sha256' not in snapshot:
        raise ValueError('immutable T snapshot required')
    body = dict(snapshot)
    digest = body.pop('snapshot_sha256')
    if sha(body) != digest:
        raise ValueError('T snapshot changed')
    query = snapshot.get('query'); identity = snapshot.get('identity'); evidence = snapshot.get('evidence')
    if (snapshot.get('status') != 'ready' or snapshot.get('reason') is not None
            or snapshot.get('pitch_type_basis') != 'observed_actual'
            or snapshot.get('prospective_claim') is not False
            or not isinstance(query, dict) or set(query) != set(SCHEMA)
            or not isinstance(identity, dict) or identity.get('evidence_kind') != 'retrospective'
            or not isinstance(evidence, dict) or evidence.get('complete') is not True
            or not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in source_sha256)
            or evidence.get('source_sha256') != source_sha256):
        raise ValueError('reviewed corrected-retrospective T-ready contract required')
    game, ab, pitch = (query[k] for k in ('game_pk', 'at_bat_number', 'pitch_number'))
    if (type(game) is not int or game not in SIX or query['game_date'] != SIX[game]
            or type(ab) is not int or type(pitch) is not int or not 0 < ab <= 10_000
            or not 0 < pitch <= 99 or evidence.get('before_key') != [game, ab, pitch]
            or identity.get('game_pk') != game or identity.get('game_date') != SIX[game]):
        raise ValueError('fixed six game/date/physical key changed')
    if query['pitch_type'] != 'FF':
        return {'status': 'excluded_non_FF', 'key': [game, ab, pitch], 'contract': CONTRACT,
                'source_snapshot_sha256': digest, 'state': None}
    if (query['previous_valid'] is not False and query['previous_valid'] is not True):
        raise ValueError('previous-pitch validity')
    if (query['previous_valid'] and (pitch == 1 or query['previous_type'] not in TYPES
                                     or query['previous_type'] in ('UNKNOWN', 'UN'))
            or not query['previous_valid'] and (pitch != 1 or query['previous_type'] != 'UNKNOWN')):
        raise ValueError('exact same-PA predecessor required')
    if (query['inning_topbot'] not in ('Top', 'Bot') or query['stand'] not in ('L', 'R')
            or query['p_throws'] not in ('L', 'R')
            or any(type(query[k]) is not int or query[k] <= 0 for k in ('pitcher', 'batter', 'inning'))
            or any(type(query[k]) is not int or query[k] < 0 for k in ('home_score', 'away_score'))
            or type(query['balls']) is not int or not 0 <= query['balls'] <= 3
            or type(query['strikes']) is not int or not 0 <= query['strikes'] <= 2
            or type(query['outs_when_up']) is not int or not 0 <= query['outs_when_up'] <= 2
            or any(v is not None and (type(v) is not int or v <= 0)
                   for v in (query['on_1b'], query['on_2b'], query['on_3b']))):
        raise ValueError('complete corrected prestate required')
    state = {'official_date': query['game_date'], 'pitcher': query['pitcher'],
             'batter': query['batter'], 'balls': query['balls'], 'strikes': query['strikes'],
             'outs': query['outs_when_up'], 'inning': query['inning'],
             'top': query['inning_topbot'] == 'Top',
             'score': (query['home_score'], query['away_score']),
             'bases': tuple(query[k] is not None for k in ('on_1b', 'on_2b', 'on_3b')),
             'stand': query['stand'], 'p_throws': query['p_throws'],
             'previous_type': query['previous_type'],
             'previous_valid': query['previous_valid']}
    return {'status': 'typed_ff_candidate', 'key': [game, ab, pitch], 'contract': CONTRACT,
            'source_snapshot_sha256': digest, 'state': state,
            'limitations': 'corrected_retrospective;prior_exposure_UNKNOWN;no_pitch_time_availability'}
