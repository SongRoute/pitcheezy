"""Append-only experimental seven-leaf forecasts for new native shadow runs.

One owner calls forecast before reading the next receipt. A committed intent
without a result is terminal: inference is never resumed after a crash.
"""
from __future__ import annotations

import copy
import json
import math
import time

from . import fullseason
from .captured_native import CONTRACT
from .captured_native_state import NATIVE_KIND, NativeRuntime, NativeShadow
from .ledger import canonical, sha

_SCHEMA = '''
CREATE TABLE IF NOT EXISTS native_forecast_run (
 id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS native_forecast_intents (
 decision_id TEXT PRIMARY KEY, request_sha TEXT NOT NULL,
 binding_sha TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS native_forecast_results (
 decision_id TEXT PRIMARY KEY, body TEXT NOT NULL, digest TEXT NOT NULL,
 FOREIGN KEY(decision_id) REFERENCES native_forecast_intents(decision_id));
''' + ''.join(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{verb}
 BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'append-only native forecast'); END;
''' for table in ('native_forecast_run', 'native_forecast_intents', 'native_forecast_results') for verb in ('UPDATE', 'DELETE'))


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _binding(scorer, config, manifest_sha):
    if not isinstance(scorer, NativeRuntime) or not isinstance(config, dict) or not _digest(manifest_sha):
        raise ValueError('loaded native runtime binding required')
    runtime = scorer.runtime
    identity = getattr(runtime, 'identity', None)
    if (not isinstance(identity, dict) or identity.get('adapter_config_sha256') != sha(config)
            or identity.get('profile_manifest_sha256') != manifest_sha
            or getattr(runtime, 'bundle_sha256', None) != sha(identity)
            or config.get('leaf_order') != list(fullseason.LEAVES)):
        raise ValueError('loaded runtime identity differs from artifact bindings')
    try:
        states = {seed: config['state'][seed]['sha256'] for seed in ('0', '1', '2')}
        artifacts = {name: config[name]['sha256'] for name in ('encoder', 'calibrator', 'recipe')}
    except (KeyError, TypeError) as exc:
        raise ValueError('exact loaded artifact bindings required') from exc
    if any(not _digest(value) for value in (*states.values(), *artifacts.values())):
        raise ValueError('artifact digest required')
    return {'runtime_bundle_sha256': runtime.bundle_sha256,
            'adapter_config_sha256': identity['adapter_config_sha256'],
            'profile_manifest_sha256': manifest_sha,
            'state_sha256': states, **{name + '_sha256': value for name, value in artifacts.items()},
            'leaf_order': list(fullseason.LEAVES), 'input_contract': CONTRACT,
            'evidence_kind': NATIVE_KIND}


def _probabilities(values):
    return (isinstance(values, list) and len(values) == len(fullseason.LEAVES)
            and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            and math.isclose(sum(values), 1, rel_tol=0, abs_tol=1e-6))


class NativeForecast:
    """Single-owner writer; external supervisor holds the existing heavy lock."""

    def __init__(self, shadow: NativeShadow, scorer: NativeRuntime, adapter_config,
                 profile_manifest_sha256, registration_sha256):
        if not isinstance(shadow, NativeShadow):
            raise ValueError('native shadow required')
        if not _digest(registration_sha256):
            raise ValueError('forecast registration digest required')
        self.shadow, self.scorer, self.db = shadow, scorer, shadow.db
        self._runtime = scorer.runtime
        self._config = copy.deepcopy(adapter_config)
        self._manifest_sha = profile_manifest_sha256
        self.binding = _binding(scorer, self._config, profile_manifest_sha256)
        self.binding['normalizer_sha256'] = shadow.strict.identity['normalizer_sha256']
        self.binding['registration_sha256'] = registration_sha256
        self.binding_sha = sha(self.binding)
        existed = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_forecast_intents'").fetchone()
        if not existed and self.db.execute('SELECT 1 FROM native_decisions LIMIT 1').fetchone():
            raise ValueError('cannot attach forecast ledger to prior native decisions')
        self.db.executescript(_SCHEMA)
        run = canonical({'stream_identity': shadow.strict.identity, 'binding_sha256': self.binding_sha})
        old = self.db.execute('SELECT body FROM native_forecast_run WHERE id=1').fetchone()
        if old and old['body'] != run:
            raise ValueError('forecast run identity changed')
        if not old:
            with self.db:
                self.db.execute('INSERT INTO native_forecast_run VALUES(1,?)', (run,))

    def _stored(self, decision_id, request_sha):
        row = self.db.execute('SELECT * FROM native_forecast_intents WHERE decision_id=?',
                              (decision_id,)).fetchone()
        if row is None:
            return None
        if row['request_sha'] != request_sha or row['binding_sha'] != self.binding_sha:
            raise ValueError('forecast decision identity or artifact binding reused')
        intent = json.loads(row['body'])
        if row['digest'] != sha(intent) or intent['decision_id'] != decision_id:
            raise ValueError('forecast intent digest changed')
        result = self.db.execute('SELECT body,digest FROM native_forecast_results WHERE decision_id=?',
                                 (decision_id,)).fetchone()
        if result is None:
            raise ValueError('incomplete forecast intent cannot be backfilled')
        body = json.loads(result['body'])
        digest = body.pop('forecast_sha256', None)
        if digest != result['digest'] or digest != sha(body) or body['intent_sha256'] != row['digest']:
            raise ValueError('forecast result digest changed')
        body['forecast_sha256'] = digest
        return body

    def _receipt(self):
        row = self.db.execute('SELECT receipt_seq,source_sha256,received_ns FROM native_receipts '
                              'ORDER BY receipt_seq DESC LIMIT 1').fetchone()
        if row is None:
            raise ValueError('forecast requires current stored receipt')
        return dict(row)

    def forecast(self, candidate, *, received_ns):
        if (not isinstance(candidate, dict) or set(candidate) !=
                {'decision_id', 'query', 'coverage', 'pitch_type_basis'}
                or not isinstance(candidate['decision_id'], str) or not candidate['decision_id']
                or candidate['pitch_type_basis'] != 'hypothetical_candidate'
                or not isinstance(candidate['query'], dict)
                or candidate['query'].get('pitch_type') != 'FF'):
            raise ValueError('exact forecast decision request required')
        request_sha = sha(candidate)
        if (self.scorer.runtime is not self._runtime or sha(_binding(
                self.scorer, self._config, self._manifest_sha) |
                {'normalizer_sha256': self.shadow.strict.identity['normalizer_sha256'],
                 'registration_sha256': self.binding['registration_sha256']}) != self.binding_sha):
            raise ValueError('loaded runtime binding changed')
        stored = self._stored(candidate['decision_id'], request_sha)
        if stored is not None:
            return stored
        if self.db.execute('SELECT 1 FROM native_decisions WHERE decision_id=?',
                           (candidate['decision_id'],)).fetchone():
            raise ValueError('native decision exists without forecast intent; cannot backfill')
        decision = self.shadow.decide(candidate, received_ns=received_ns)
        receipt = self._receipt()
        if candidate['coverage']['source_sha256'] != receipt['source_sha256']:
            raise ValueError('forecast receipt source changed')
        key = [candidate['query'][name] for name in ('game_pk', 'at_bat_number', 'pitch_number')]
        event_seq = self.db.execute('SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0]
        intent = {'decision_id': candidate['decision_id'], 'request_sha256': request_sha,
                  'native_decision_sha256': decision['decision_sha256'],
                  'strict_snapshot_sha256': decision['strict_snapshot_sha256'],
                  'native_snapshot_sha256': (decision['native_snapshot']['snapshot_sha256']
                                             if decision['native_snapshot'] else None),
                  'key': key, 'action': candidate['query']['pitch_type'],
                  'pitch_type_basis': candidate['pitch_type_basis'],
                  'receipt': receipt, 'decision_received_ns': received_ns,
                  'event_seq': event_seq, 'binding': self.binding,
                  'binding_sha256': self.binding_sha,
                  'intent_written_wall_ns': time.time_ns(),
                  'intent_written_monotonic_ns': time.monotonic_ns()}
        digest = sha(intent)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if (self._receipt() != receipt or self.db.execute(
                    'SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0] != event_seq):
                raise ValueError('receipt changed before forecast intent')
            self.db.execute('INSERT INTO native_forecast_intents VALUES(?,?,?,?,?)',
                            (candidate['decision_id'], request_sha, self.binding_sha,
                             canonical(intent), digest))
        score_start_ns = time.monotonic_ns() if decision['status'] == 'native_ready' else None
        prediction = self.scorer.predict(decision)  # exactly one call; failures leave the intent incomplete.
        score_compute_ns = (time.monotonic_ns() - score_start_ns
                            if score_start_ns is not None else None)
        if (prediction.get('input_contract') != CONTRACT
                or prediction.get('native_decision_sha256') != decision['decision_sha256']
                or prediction.get('prediction_ready') is not False
                or prediction.get('prospective_claim') is not False):
            raise ValueError('native runtime result identity changed')
        if decision['status'] == 'native_abstain':
            if (prediction.get('status') != 'experimental_abstain'
                    or prediction.get('reason') != decision['reason']
                    or prediction.get('probabilities') is not None):
                raise ValueError('native abstention forecast mismatch')
        elif (decision['status'] != 'native_ready'
              or prediction.get('status') != 'experimental_conditional_score'
              or prediction.get('reason') is not None
              or prediction.get('leaf_order') != list(fullseason.LEAVES)
              or not _probabilities(prediction.get('probabilities'))
              or prediction.get('underlying_call_sha256') != sha({
                  'bundle': self.scorer.runtime.bundle_sha256,
                  'snapshot': decision['native_snapshot']['snapshot_sha256']})):
            raise ValueError('invalid experimental seven-leaf score')
        result = {'decision_id': candidate['decision_id'], 'request_sha256': request_sha,
                  'intent_sha256': digest, 'native_decision_sha256': decision['decision_sha256'],
                  'binding_sha256': self.binding_sha, 'key': key, 'action': intent['action'],
                  'leaf_order': list(fullseason.LEAVES), 'status': prediction['status'],
                  'reason': prediction['reason'], 'probabilities': prediction['probabilities'],
                  'underlying_call_sha256': prediction.get('underlying_call_sha256'),
                  'input_contract': CONTRACT, 'evidence_kind': NATIVE_KIND,
                  'prediction_ready': False, 'prospective_claim': False,
                  'local_pre_observation_only': True,
                  'score_compute_ns': score_compute_ns,
                  'result_written_wall_ns': time.time_ns(),
                  'result_written_monotonic_ns': time.monotonic_ns()}
        result['forecast_sha256'] = sha(result)
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if (self._receipt() != receipt or self.db.execute(
                    'SELECT COALESCE(MAX(seq),0) FROM events').fetchone()[0] != event_seq
                    or self.db.execute('SELECT 1 FROM events WHERE ab=? AND pitch=? LIMIT 1',
                                       key[1:]).fetchone()
                    or self.scorer.runtime is not self._runtime
                    or self.binding_sha != sha(_binding(self.scorer, self._config, self._manifest_sha) |
                                              {'normalizer_sha256': self.shadow.strict.identity['normalizer_sha256'],
                                               'registration_sha256': self.binding['registration_sha256']})):
                raise ValueError('receipt, target or artifact changed before forecast commit')
            self.db.execute('INSERT INTO native_forecast_results VALUES(?,?,?)',
                            (candidate['decision_id'], canonical(result), result['forecast_sha256']))
        return result
