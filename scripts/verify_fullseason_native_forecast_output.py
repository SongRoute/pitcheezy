"""Read-only bounded aggregate audit of one *new* native forecast SQLite.

The verifier reads forecast tables and schema only; it never opens capture,
model, profile, corrected-reference, or previous-attempt SQLite files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import stat
from urllib.parse import quote

MAX_DB = 16 << 20
EXPECTED = 100
TABLES = ('native_forecast_run', 'native_forecast_intents', 'native_forecast_results')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify(db_path, receipt):
    db_path = Path(db_path)
    info = db_path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_DB:
        raise ValueError('bounded regular forecast SQLite required')
    if any(Path(str(db_path) + suffix).exists() for suffix in ('-wal', '-shm')):
        raise ValueError('forecast SQLite still has WAL sidecar')
    before = file_sha(db_path)
    uri = 'file:' + quote(str(db_path)) + '?mode=ro&immutable=1'
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA query_only=ON')
        found = {row['name']: row['sql'] for row in db.execute(
            "SELECT name,sql FROM sqlite_master WHERE type='table' AND name IN (?,?,?)", TABLES)}
        if set(found) != set(TABLES):
            raise ValueError('forecast table schema missing')
        for table in TABLES:
            triggers = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,))}
            if not {f'{table}_no_UPDATE', f'{table}_no_DELETE'} <= triggers:
                raise ValueError('forecast append-only trigger missing')
        runs = list(db.execute('SELECT id,body FROM native_forecast_run LIMIT 2'))
        if len(runs) != 1 or runs[0]['id'] != 1:
            raise ValueError('forecast run identity count')
        run = json.loads(runs[0]['body'])
        if runs[0]['body'] != canonical(run) or not isinstance(run.get('binding_sha256'), str):
            raise ValueError('forecast run identity body')
        intents = list(db.execute('SELECT decision_id,request_sha,binding_sha,body,digest '
                                  'FROM native_forecast_intents ORDER BY rowid LIMIT ?', (EXPECTED + 1,)))
        results = list(db.execute('SELECT decision_id,body,digest '
                                  'FROM native_forecast_results ORDER BY rowid LIMIT ?', (EXPECTED + 1,)))
        if len(intents) != EXPECTED or len(results) != EXPECTED:
            raise ValueError('forecast count differs from fixed denominator')
        by_id = {row['decision_id']: row for row in results}
        if len(by_id) != EXPECTED:
            raise ValueError('duplicate forecast result id')
        keys, result_digests = [], []
        scores = abstains = 0
        for row in intents:
            intent = json.loads(row['body'])
            if (row['body'] != canonical(intent) or row['digest'] != digest(intent)
                    or row['decision_id'] != intent['decision_id']
                    or row['request_sha'] != intent['request_sha256']
                    or row['binding_sha'] != intent['binding_sha256']
                    or row['binding_sha'] != run['binding_sha256']
                    or digest(intent['binding']) != row['binding_sha']
                    or intent['binding']['registration_sha256'] != receipt['registration_sha256']
                    or intent['binding']['runtime_bundle_sha256'] != receipt['bundle_sha256']
                    or intent['action'] != 'FF'):
                raise ValueError('forecast intent custody mismatch')
            key = intent['key']
            if (not isinstance(key, list) or len(key) != 3
                    or any(type(value) is not int or value <= 0 for value in key)):
                raise ValueError('forecast physical key')
            keys.append(key)
            result_row = by_id.get(row['decision_id'])
            if result_row is None:
                raise ValueError('pending forecast intent')
            result = json.loads(result_row['body'])
            saved_digest = result.pop('forecast_sha256', None)
            if (result_row['body'] != canonical(result | {'forecast_sha256': saved_digest})
                    or result_row['digest'] != saved_digest or saved_digest != digest(result)
                    or result['intent_sha256'] != row['digest']
                    or result['decision_id'] != row['decision_id']
                    or result['request_sha256'] != row['request_sha']
                    or result['native_decision_sha256'] != intent['native_decision_sha256']
                    or result['binding_sha256'] != row['binding_sha']
                    or result['key'] != key or result['action'] != 'FF'
                    or result['leaf_order'] != intent['binding']['leaf_order']
                    or result['local_pre_observation_only'] is not True
                    or result['prediction_ready'] is not False
                    or result['prospective_claim'] is not False):
                raise ValueError('forecast result custody mismatch')
            probs = result['probabilities']
            if result['status'] == 'experimental_conditional_score':
                if (not isinstance(probs, list) or len(probs) != 7
                        or any(type(p) not in (int, float) or not math.isfinite(p)
                               or p < 0 or p > 1 for p in probs)
                        or not math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-6)
                        or type(result['score_compute_ns']) is not int
                        or result['score_compute_ns'] < 0):
                    raise ValueError('forecast seven-probability coherence')
                scores += 1
            elif result['status'] == 'experimental_abstain':
                if probs is not None or result['score_compute_ns'] is not None:
                    raise ValueError('abstention probability or latency')
                abstains += 1
            else:
                raise ValueError('forecast status')
            result_digests.append(saved_digest)
        if len({tuple(key) for key in keys}) != EXPECTED:
            raise ValueError('duplicate physical forecast key')
        if (digest(keys) != receipt['selected_keys_sha256']
                or digest(result_digests) != receipt['forecast_results_sha256']
                or receipt['forecast_intents'] != EXPECTED
                or receipt['forecast_terminal_results'] != EXPECTED
                or (scores, abstains) != (
                    receipt['counts']['experimental_experimental_conditional_score'],
                    receipt['counts']['experimental_experimental_abstain'])
                or receipt['forecast_pending_gaps'] != 0
                or receipt['score_compute_latency']['count'] != scores
                or receipt['individual_probabilities_exported'] is not False
                or receipt['prediction_ready'] is not False):
            raise ValueError('forecast aggregate receipt mismatch')
    finally:
        db.close()
    if file_sha(db_path) != before or any(Path(str(db_path) + suffix).exists()
                                         for suffix in ('-wal', '-shm')):
        raise ValueError('forecast SQLite changed during read-only audit')
    return {'status': 'PASS_FORECAST_OUTPUT_AGGREGATE_ONLY', 'sqlite_sha256': before,
            'intents': EXPECTED, 'results': EXPECTED, 'pending_gaps': 0,
            'scores': scores, 'abstains': abstains,
            'selected_keys_sha256': digest(keys), 'forecast_results_sha256': digest(result_digests),
            'individual_probabilities_exported': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text())
    db_path = Path(registration['output']) / 'native_stream.sqlite3'
    receipt_path = Path(registration['output']) / 'receipt.json'
    if args.receipt.resolve() != receipt_path.resolve():
        raise ValueError('only registered new attempt receipt allowed')
    receipt = json.loads(args.receipt.read_text())
    if receipt['registration_sha256'] != file_sha(args.registration):
        raise ValueError('receipt registration identity')
    print(canonical(verify(db_path, receipt)))
