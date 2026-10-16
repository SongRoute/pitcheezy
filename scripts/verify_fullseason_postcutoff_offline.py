"""Read only a new owned score artifact/receipt and recompute aggregate arithmetic."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import stat
import struct

GAMES = (849812, 849820, 849813, 849811, 849810, 849818)
LEAVES = ('ball', 'foul', 'hit_by_pitch', 'narrow_whiff', 'other_strike',
          'ground_ball_in_play', 'other_in_play')
COUNTS = (258626, 132039, 2144, 78550, 124789, 51974, 72383)
TOTAL = 720505
MAX_ROWS = 12_000
MAX_BYTES = 16 << 20


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, cap):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or not 0 < path.stat().st_size <= cap:
        raise ValueError('bounded regular output required')
    return path.read_bytes()


def _near(a, b):
    return type(a) in (int, float) and math.isfinite(a) and abs(a-b) <= 1e-10


def _point(rows):
    if not rows:
        return {'n': 0, 'model_nll': None, 'model_brier': None,
                'fixed_fit_marginal_nll': None, 'fixed_fit_marginal_brier': None}
    base = [c/TOTAL for c in COUNTS]
    vals = [[], [], [], []]
    for row in rows:
        y = row['label']; q = row['probabilities']
        vals[0].append(-math.log(max(q[y], 1e-12)))
        vals[1].append(math.fsum((q[k]-(k == y))**2 for k in range(7)))
        vals[2].append(-math.log(max(base[y], 1e-12)))
        vals[3].append(math.fsum((base[k]-(k == y))**2 for k in range(7)))
    return dict(zip(('n', 'model_nll', 'model_brier', 'fixed_fit_marginal_nll',
                     'fixed_fit_marginal_brier'),
                    (len(rows), *(math.fsum(v)/len(rows) for v in vals))))


def verify(receipt_path, registration_path):
    receipt_path = Path(receipt_path)
    receipt = json.loads(_read(receipt_path, 1 << 20))
    reg_path = Path(registration_path)
    reg_raw = _read(reg_path, 1 << 20)
    reg = json.loads(reg_raw)
    source_rows = reg['bounds']['source_rows']
    if (type(source_rows) is not int or not 0 < source_rows <= MAX_ROWS
            or receipt.get('registration_sha256') != _sha(reg_raw)
            or receipt.get('collector_rows') != source_rows
            or reg.get('output') != str(receipt_path.parent)):
        raise ValueError('reviewed registration/row bound changed')
    if (receipt.get('status') != 'COMPLETED' or receipt.get('score_rows_file') != 'score_rows.json'
            or receipt.get('prospective_claim') is not False
            or receipt.get('superiority_claim') is not False):
        raise ValueError('completed bounded retrospective receipt required')
    body_path = receipt_path.parent / 'score_rows.json'
    raw = _read(body_path, MAX_BYTES)
    if len(raw) != receipt.get('score_rows_bytes') or _sha(raw) != receipt.get('score_rows_sha256'):
        raise ValueError('private score artifact binding changed')
    body = json.loads(raw)
    if raw != (json.dumps(body, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False) + '\n').encode():
        raise ValueError('noncanonical private score artifact')
    if (set(body) != {'schema', 'source_sha256', 'bundle_sha256', 'leaf_order',
                      'baseline_counts', 'baseline_total', 'rows'}
            or body['schema'] != 'postcutoff-six-score-rows-v1'
            or body['source_sha256'] != receipt.get('projected_sha256')
            or body['bundle_sha256'] != receipt.get('runtime_bundle_sha256')
            or body['leaf_order'] != list(LEAVES)
            or body['baseline_counts'] != list(COUNTS) or body['baseline_total'] != TOTAL
            or not isinstance(body['rows'], list) or len(body['rows']) > source_rows):
        raise ValueError('private score artifact schema/identity')
    rows = body['rows']; keys = []; scored = defaultdict(list); reasons = Counter(); per_game = Counter(); abstained = Counter()
    for row in rows:
        if set(row) != {'key', 'status', 'reason', 'observed_pitch_type', 'label', 'probabilities'}:
            raise ValueError('private row schema')
        key = row['key']
        if (not isinstance(key, list) or len(key) != 3 or any(type(v) is not int or v <= 0 or v >= 2**53 for v in key)
                or key[0] not in GAMES):
            raise ValueError('private row physical key')
        keys.append(tuple(key))
        per_game[key[0]] += 1
        if row['status'] == 'abstain':
            if (not isinstance(row['reason'], str) or not row['reason']
                    or row['observed_pitch_type'] is not None or row['label'] is not None
                    or row['probabilities'] is not None):
                raise ValueError('abstain must have reason and no score')
            reasons[row['reason']] += 1
            abstained[key[0]] += 1
        elif row['status'] == 'scored':
            q = row['probabilities']
            if (row['reason'] is not None or type(row['label']) is not int or not 0 <= row['label'] < 7
                    or not isinstance(row['observed_pitch_type'], str)
                    or row['observed_pitch_type'] in ('UNKNOWN', 'UN', '')
                    or not isinstance(q, list) or len(q) != 7
                    or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in q)
                    or abs(math.fsum(q)-1) > 1e-10):
                raise ValueError('scored action/label/simplex')
            scored[key[0]].append(row)
        else:
            raise ValueError('private row status')
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate private physical key')
    digest = hashlib.sha256(b''.join(struct.pack('<qqq', *key) for key in sorted(keys))).hexdigest()
    census = receipt['census']; metrics = receipt['metrics']
    if (digest != census.get('accepted_keys_sha256')
            or len(keys) != census.get('accepted_quality_rows')
            or len(keys) != metrics.get('accepted_quality_rows')
            or sum(len(v) for v in scored.values()) != metrics.get('scored')
            or len(keys) - metrics['scored'] != census.get('query_abstain')
            or sum(len(v) for v in scored.values()) != census.get('query_ready')
            or dict(sorted(reasons.items())) != metrics.get('query_abstain_reasons')):
        raise ValueError('accepted population/score denominator mismatch')
    if any((census['games'][str(game)]['accepted_quality_rows'] != per_game[game]
            or census['games'][str(game)]['query_ready'] != len(scored[game])
            or census['games'][str(game)]['query_abstain'] != abstained[game]) for game in GAMES):
        raise ValueError('per-game accepted/query census mismatch')
    actual = {str(game): _point(scored[game]) for game in GAMES}
    actual['overall'] = _point([row for game in GAMES for row in scored[game]])
    expected = dict(metrics['per_game'], overall=metrics['overall'])
    if set(actual) != set(expected):
        raise ValueError('metric game slots')
    for game, point in actual.items():
        if set(point) != set(expected[game]):
            raise ValueError('metric fields')
        for field, value in point.items():
            prior = expected[game][field]
            if (value is None and prior is not None) or (value is not None and not _near(prior, value)):
                raise ValueError('stored metric arithmetic mismatch')
    if _sha(_read(body_path, MAX_BYTES)) != receipt['score_rows_sha256']:
        raise ValueError('score artifact changed during verification')
    return {'verdict': 'PASS_STORED_OUTPUT_ARITHMETIC_ONLY', 'rows': len(rows),
            'scored': metrics['scored'], 'abstain': len(rows)-metrics['scored'],
            'accepted_keys_sha256': digest, 'score_rows_sha256': receipt['score_rows_sha256'],
            'prospective_claim': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--receipt', required=True, type=Path)
    p.add_argument('--registration', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.receipt, a.registration), sort_keys=True))
