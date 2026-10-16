"""Experimental native sidecar on the unchanged strict UNKNOWN receipt ledger.

The strict StreamState still sees every captured pitch as UNKNOWN. This module
stores separate six-field source hypotheses and immutable native decisions.
Its runtime-compatible inner snapshot is never a strict accepted-row decision.
"""
from __future__ import annotations

import json

import numpy as np

from experiments.ws_campaign.source.first_small_features import TYPES
from .captured_native import CONTRACT, TOKEN_FIELDS
from .fullseason_stream import StreamState, FIELDS, _key
from .ledger import canonical, sha

NATIVE_KIND = 'capture_native_experimental'
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS native_receipts (
 receipt_seq INTEGER PRIMARY KEY, source_sha256 TEXT NOT NULL, received_ns INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS native_evidence (
 event_seq INTEGER PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL,
 FOREIGN KEY(event_seq) REFERENCES events(seq));
CREATE TABLE IF NOT EXISTS native_decisions (
 decision_id TEXT PRIMARY KEY, request_sha TEXT NOT NULL, body TEXT NOT NULL);
''' + ''.join(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{verb}
 BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'append-only native shadow'); END;
''' for table in ('native_receipts', 'native_evidence', 'native_decisions') for verb in ('UPDATE', 'DELETE'))


def _valid_token(token):
    if not isinstance(token, dict) or set(token) != set(TOKEN_FIELDS):
        return False
    return (all(type(token[k]) is int and 1 <= token[k] < (1 << 53) for k in ('pitcher', 'batter'))
            and token['pitch_type'] in TYPES and token['pitch_type'] != 'UNKNOWN'
            and all(type(token[k]) is int and 0 <= token[k] <= hi
                    for k, hi in (('outcome10', 9), ('balls', 3), ('strikes', 2))))


def _native_row(event, native_row, source_sha256):
    key = tuple(event['key'])
    if native_row is None:
        return {'key': key, 'provider_id': event['event_id'], 'receipt_seq': event['revision'],
                'source_sha256': source_sha256, 'input_contract': CONTRACT,
                'status': 'native_unknown', 'reason': 'source_row_absent', 'native_token': None,
                'statcast_training_equivalent': False,
                'provider_identity_scope': 'single_local_receipt', 'accepted': False}
    if (not isinstance(native_row, dict) or set(native_row) !=
            {'key', 'provider_id', 'receipt_seq', 'source_sha256', 'input_contract', 'status',
             'reason', 'native_token', 'statcast_training_equivalent',
             'provider_identity_scope', 'accepted'}
            or not isinstance(native_row['key'], (tuple, list)) or tuple(native_row['key']) != key
            or (native_row['status'] == 'native_candidate'
                and native_row['provider_id'] != event['event_id'])
            or (native_row['status'] == 'native_unknown'
                and native_row['provider_id'] not in (None, event['event_id']))
            or native_row['receipt_seq'] != event['revision']
            or native_row['source_sha256'] != source_sha256
            or native_row['input_contract'] != CONTRACT or native_row['accepted'] is not False
            or native_row['statcast_training_equivalent'] is not False
            or native_row['provider_identity_scope'] != 'single_local_receipt'
            or native_row['status'] not in ('native_candidate', 'native_unknown')
            or not isinstance(native_row['reason'], str) or not native_row['reason']):
        raise ValueError('mixed or invalid native evidence contract')
    if native_row['status'] == 'native_candidate' and native_row['reason'] != 'source_native_six_fields_unpaired':
        raise ValueError('native candidate proof reason')
    if (native_row['status'] == 'native_candidate') != _valid_token(native_row['native_token']):
        raise ValueError('native status/token mismatch')
    if native_row['status'] == 'native_unknown' and native_row['native_token'] is not None:
        raise ValueError('native unknown must not carry a token')
    return native_row


def _semantic(row):
    """Receipt proof stays first-seen when source-native fields are unchanged."""
    return (row['provider_id'], row['status'], row['reason'], row['native_token'])


class NativeShadow:
    """Single-owner sidecar; strict receipt and decision semantics remain intact."""

    def __init__(self, path, *, game_pk, game_date, normalizer_sha256):
        self.strict = StreamState(path, game_pk=game_pk, game_date=game_date,
                                  normalizer_sha256=normalizer_sha256, evidence_kind='captured')
        self.db = self.strict.db
        self.db.executescript(_SCHEMA)
        if self.db.execute("SELECT 1 FROM events WHERE status!='unknown' LIMIT 1").fetchone():
            self.strict.close()
            raise ValueError('strict accepted/rejected events cannot mix with native sidecar')
        if (self.db.execute('SELECT 1 FROM events LIMIT 1').fetchone()
                and not self.db.execute('SELECT 1 FROM native_receipts LIMIT 1').fetchone()):
            self.strict.close()
            raise ValueError('preexisting strict-only events cannot mix with native sidecar')

    def close(self):
        self.strict.close()

    def observe_receipt(self, receipt_seq, source_sha256, received_ns):
        """Bind one verified canonical body SHA to its local ordinal and clock."""
        if (type(receipt_seq) is not int or receipt_seq <= 0 or not isinstance(source_sha256, str)
                or len(source_sha256) != 64 or any(c not in '0123456789abcdef' for c in source_sha256)):
            raise ValueError('native receipt identity')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            self.strict._clock(received_ns)
            old = self.db.execute('SELECT source_sha256,received_ns FROM native_receipts WHERE receipt_seq=?',
                                  (receipt_seq,)).fetchone()
            if old:
                if (old['source_sha256'], old['received_ns']) != (source_sha256, received_ns):
                    raise ValueError('native receipt ordinal reused')
            else:
                last = self.db.execute('SELECT MAX(receipt_seq) FROM native_receipts').fetchone()[0]
                if (last is None and receipt_seq != 1) or (last is not None and receipt_seq != last + 1):
                    raise ValueError('native receipt sequence gap or regression')
                self.db.execute('INSERT INTO native_receipts VALUES(?,?,?)',
                                (receipt_seq, source_sha256, received_ns))
            self.strict._advance_clock(received_ns)

    def ingest(self, event, native_row, *, received_ns, source_sha256):
        if (not isinstance(event, dict) or set(event) != {'event_id', 'key', 'revision', 'row'}
                or event['row'] != {'status': 'unknown', 'reason': 'statcast_acceptance_unverified',
                                    'accepted': None}
                or not isinstance(source_sha256, str) or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)):
            raise ValueError('strict UNKNOWN event and source binding required')
        key = _key(event['key'])
        if key[0] != self.strict.identity['game_pk']:
            raise ValueError('native event game identity')
        row = _native_row(event, native_row, source_sha256)
        self.observe_receipt(event['revision'], source_sha256, received_ns)
        latest = self.db.execute('''SELECT e.seq,e.event_id,e.revision,e.first_received_ns,n.body
            FROM events e LEFT JOIN native_evidence n ON n.event_seq=e.seq
            WHERE e.ab=? AND e.pitch=? ORDER BY e.seq DESC LIMIT 1''', key[1:]).fetchone()
        if latest and event['revision'] <= latest['revision']:
            if (event['revision'] != latest['revision'] or latest['body'] is None
                    or json.loads(latest['body']) != row):
                raise ValueError('stale or conflicting native receipt revision')
        if (latest and latest['event_id'] == event['event_id'] and latest['body']
                and _semantic(json.loads(latest['body'])) == _semantic(row)):
            return {'strict_status': 'unchanged_observation', 'native_status': row['status'],
                    'event_seq': latest['seq'], 'first_received_ns': latest['first_received_ns']}
        observation = self.strict.ingest(**event, received_ns=received_ns)
        seq, digest = observation['seq'], sha(row)
        old = self.db.execute('SELECT digest FROM native_evidence WHERE event_seq=?', (seq,)).fetchone()
        if old:
            if old['digest'] != digest:
                raise ValueError('native evidence changed at same strict event revision')
        else:
            with self.db:
                self.db.execute('INSERT INTO native_evidence VALUES(?,?,?)',
                                (seq, canonical(row), digest))
        return {'strict_status': observation['status'], 'native_status': row['status'],
                'event_seq': seq, 'first_received_ns': observation['first_received_ns']}

    def decide(self, candidate, *, received_ns):
        if not isinstance(candidate, dict) or set(candidate) != {
                'decision_id', 'query', 'coverage', 'pitch_type_basis'}:
            raise ValueError('exact native decision request required')
        if candidate['pitch_type_basis'] != 'hypothetical_candidate':
            raise ValueError('native contract requires a hypothetical next pitch')
        request_sha = sha(candidate)
        old = self.db.execute('SELECT request_sha,body FROM native_decisions WHERE decision_id=?',
                              (candidate['decision_id'],)).fetchone()
        if old:
            if old['request_sha'] != request_sha:
                raise ValueError('native decision identity reused')
            strict = self.strict.decide(**candidate, received_ns=received_ns)
            prior = json.loads(old['body'])
            if strict['snapshot_sha256'] != prior['strict_snapshot_sha256']:
                raise ValueError('strict/native decision custody mismatch')
            return prior
        latest_receipt = self.db.execute('''SELECT source_sha256 FROM native_receipts
            ORDER BY receipt_seq DESC LIMIT 1''').fetchone()
        if (latest_receipt is None or not isinstance(candidate['coverage'], dict)
                or candidate['coverage'].get('source_sha256') != latest_receipt['source_sha256']):
            raise ValueError('native query must bind current stored receipt')
        raw_query = candidate['query']
        if (isinstance(raw_query, dict) and all(name in raw_query for name in
                ('at_bat_number', 'pitch_number')) and self.db.execute(
                'SELECT 1 FROM events WHERE ab=? AND pitch=? LIMIT 1',
                (raw_query['at_bat_number'], raw_query['pitch_number'])).fetchone()):
            raise ValueError('native query key already observed')
        strict = self.strict.decide(**candidate, received_ns=received_ns)
        query, coverage = candidate['query'], candidate['coverage']
        key = (query['game_pk'], query['at_bat_number'], query['pitch_number'])
        rows = self.db.execute('''SELECT e.ab,e.pitch,e.seq,n.body FROM events e
            LEFT JOIN native_evidence n ON n.event_seq=e.seq
            WHERE (e.ab,e.pitch)<(?,?) AND e.seq=(SELECT MAX(v.seq) FROM events v
                WHERE v.ab=e.ab AND v.pitch=e.pitch)
            ORDER BY e.ab DESC,e.pitch DESC''', key[1:]).fetchall()
        known, unknown = [], []
        for event in rows:
            evidence = json.loads(event['body']) if event['body'] else None
            if evidence is not None and evidence['status'] == 'native_candidate':
                if evidence['native_token']['pitcher'] == query['pitcher']:
                    known.append((event, evidence['native_token']))
            else:
                unknown.append(event)
        known_total = len(known)
        known = known[:32]
        floor = (known[-1][0]['ab'], known[-1][0]['pitch']) if len(known) == 32 else None
        relevant_unknown = sum(
            floor is None or (event['ab'], event['pitch']) >= floor
            or (event['ab'], event['pitch']) == (key[1], key[2]-1)
            for event in unknown)
        predecessor = next((event for event in rows if (event['ab'], event['pitch']) == (key[1], key[2]-1)), None)
        previous_native = ('missing' if predecessor is None else
                           'known' if predecessor['body'] and json.loads(predecessor['body'])['status'] == 'native_candidate'
                           else 'unknown')
        context_counts = {'prior_known_same_pitcher_total': known_total,
                          'prior_unknown_relevant': relevant_unknown,
                          'selected_known_tokens': len(known),
                          'previous_pitch_native_state': previous_native}
        reason = ('stored_prefix_incomplete' if not coverage['complete'] else
                  'native_prior_quality_unknown' if relevant_unknown else
                  'native_previous_pitch_missing' if key[2] > 1 and predecessor is None else None)
        native_snapshot = None
        if reason is None:
            previous = None
            if predecessor and predecessor['body']:
                previous = json.loads(predecessor['body'])
            valid_previous = previous is not None and previous['status'] == 'native_candidate'
            expanded = dict(query, previous_type=(previous['native_token']['pitch_type']
                                                  if valid_previous else 'UNKNOWN'),
                            previous_valid=valid_previous)
            tokens = {name: np.zeros((1, 32), dtype=np.int64) for name in FIELDS}
            tokens['pos'][0] = np.arange(1, 33)
            tokens['valid'] = np.zeros((1, 32), bool)
            tokens['context_keys'] = np.zeros((1, 32, 3), dtype=np.int64)
            tokens['context_pitcher'] = np.zeros((1, 32), dtype=np.int64)
            tokens['query_pitcher'] = np.array([query['pitcher']], dtype=np.int64)
            for slot, (event, token) in enumerate(known):
                for name, value in zip(FIELDS, (TYPES.index(token['pitch_type']), token['outcome10'],
                                                token['balls'], token['strikes'],
                                                int(token['batter'] == query['batter']), slot+1)):
                    tokens[name][0, slot] = value
                tokens['valid'][0, slot] = True
                tokens['context_keys'][0, slot] = (key[0], event['ab'], event['pitch'])
                tokens['context_pitcher'][0, slot] = token['pitcher']
            native_snapshot = {'status': 'ready', 'reason': None,
                'identity': dict(self.strict.identity, evidence_kind=NATIVE_KIND,
                                 input_contract=CONTRACT), 'evidence': dict(coverage),
                'pitch_type_basis': candidate['pitch_type_basis'], 'prospective_claim': False,
                'received_ns': received_ns, 'event_seq': strict['event_seq'],
                'context_dependency_floor': [key[0], *floor] if floor else None,
                'query': expanded, 'tokens': {name: values.tolist() for name, values in tokens.items()},
                'input_contract': CONTRACT, 'statcast_training_equivalent': False}
            native_snapshot['snapshot_sha256'] = sha(native_snapshot)
        result = {'status': 'native_abstain' if reason else 'native_ready', 'reason': reason,
                  'input_contract': CONTRACT, 'evidence_kind': NATIVE_KIND,
                  'context_counts': context_counts,
                  'strict_status': strict['status'],
                  'strict_snapshot_sha256': strict['snapshot_sha256'],
                  'native_snapshot': native_snapshot,
                  'prospective_claim': False, 'prediction_ready': False}
        result['decision_sha256'] = sha(result)
        with self.db:
            self.db.execute('INSERT INTO native_decisions VALUES(?,?,?)',
                            (candidate['decision_id'], request_sha, canonical(result)))
        return result


class NativeRuntime:
    """Explicit opt-in around the unchanged seven-leaf Runtime scorer."""

    def __init__(self, runtime, *, allow_native_experimental=False):
        if allow_native_experimental is not True:
            raise ValueError('explicit native experimental opt-in required')
        self.runtime = runtime

    def predict(self, decision):
        if not isinstance(decision, dict) or decision.get('input_contract') != CONTRACT:
            raise ValueError('native contract decision required')
        body = dict(decision); digest = body.pop('decision_sha256', None)
        if digest is None or sha(body) != digest:
            raise ValueError('native decision changed')
        if decision['status'] == 'native_abstain':
            return {'status': 'experimental_abstain', 'reason': decision['reason'],
                    'input_contract': CONTRACT, 'native_decision_sha256': digest,
                    'probabilities': None, 'prospective_claim': False,
                    'prediction_ready': False}
        snapshot = decision.get('native_snapshot')
        if (decision['status'] != 'native_ready' or not isinstance(snapshot, dict)
                or snapshot.get('input_contract') != CONTRACT
                or snapshot.get('identity', {}).get('evidence_kind') != NATIVE_KIND
                or snapshot.get('statcast_training_equivalent') is not False):
            raise ValueError('native snapshot identity required')
        prediction = self.runtime.predict(snapshot)
        if (prediction['status'] != 'scored' or prediction['stream_evidence_kind'] != NATIVE_KIND
                or prediction['prospective_claim'] is not False):
            raise ValueError('native runtime result identity changed')
        return {'status': 'experimental_conditional_score', 'reason': None,
                'input_contract': CONTRACT, 'native_decision_sha256': digest,
                'underlying_call_sha256': prediction['call_sha256'],
                'probabilities': prediction['probabilities'], 'prospective_claim': False,
                'prediction_ready': False}
