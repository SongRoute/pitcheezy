"""Registered fixed-six retrospective scoring; never acquires source data."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import checked_json, gates, sha as file_sha
from experiments.ws_fullseason.data import closure
from pitcheezy.seq_shadow.fullseason import LEAVES
from pitcheezy.seq_shadow.fullseason_runtime import Runtime
from pitcheezy.seq_shadow.ledger import canonical, sha as object_sha
from pitcheezy.seq_shadow.postcutoff_offline import FIT_COUNTS, FIT_TOTAL, SIX, evaluate, prepare


PLAN_SHA = '0cc1e1498b176c0f5461d4384ea454a6017830f9553502ca5a4a9cc5191a4fe7'
ADAPTER_COMMIT = 'bec072fae6ee950b17f56f40929f1383663ec4a9'
CONFIG_SHA = '3abaa2c39420d7f7d2dd7e865406070826487bcaec9018b1958eee46fea02a29'
PROFILE_SHA = '3455dd1af73944c407f1fea88af807f001809dc28fbb4362d37c3b8c4d3d11cb'
MAX_ROWS = 12_000
MAX_OUTPUT = 16 << 20


def score_population(prepared, predictor, *, source_sha256, bundle_sha256):
    """One call per ready key; return aggregate plus private, checkable row records."""
    answers = {}
    def capture(snapshot):
        query = snapshot['query']
        key = (query['game_pk'], query['at_bat_number'], query['pitch_number'])
        if key in answers:
            raise ValueError('duplicate score request')
        result = deepcopy(predictor(snapshot))
        answers[key] = result
        return deepcopy(result)
    metrics = evaluate(prepared, capture)
    records = []
    for case in prepared['cases']:
        key = tuple(case['key'])
        if case['status'] == 'ready':
            result = answers.pop(key)
            if (result.get('status') != 'scored' or result.get('leaf_order') != list(LEAVES)
                    or result.get('bundle_sha256') != bundle_sha256
                    or result.get('snapshot_sha256') != case['snapshot']['snapshot_sha256']
                    or result.get('stream_evidence_kind') != 'retrospective'
                    or result.get('pitch_type_basis') != 'observed_actual'
                    or result.get('prospective_claim') is not False):
                raise ValueError('loaded runtime/result binding mismatch')
            record = {'key': list(key), 'status': 'scored', 'reason': None,
                      'observed_pitch_type': case['snapshot']['query']['pitch_type'],
                      'label': case['label'], 'probabilities': result['probabilities']}
        else:
            record = {'key': list(key), 'status': 'abstain', 'reason': case['reason'],
                      'observed_pitch_type': None, 'label': None, 'probabilities': None}
        records.append(record)
    if answers or len(records) != prepared['census']['accepted_quality_rows']:
        raise ValueError('score/case count mismatch')
    body = {'schema': 'postcutoff-six-score-rows-v1', 'source_sha256': source_sha256,
            'bundle_sha256': bundle_sha256, 'leaf_order': list(LEAVES),
            'baseline_counts': list(FIT_COUNTS), 'baseline_total': FIT_TOTAL,
            'rows': records}
    if len(records) > MAX_ROWS or len(canonical(body).encode()) > MAX_OUTPUT:
        raise ValueError('private score row artifact cap')
    return metrics, body


def _regular_pinned(path, size, digest, cap):
    p = Path(path)
    if (type(size) is not int or not 0 < size <= cap or not stat.S_ISREG(p.lstat().st_mode)
            or p.stat().st_size != size or file_sha(p) != digest):
        raise ValueError('registered regular input bytes changed')
    return p


def _source(reg, M):
    """Verify final exact-six manifest and projected bytes before Parquet decode."""
    import pyarrow.parquet as pq
    src = reg['source']
    manifest_path = _regular_pinned(src['manifest']['path'], src['manifest']['bytes'],
                                    src['manifest']['sha256'], 1 << 20)
    manifest = checked_json(manifest_path)
    if (manifest.get('schema') != 'postseason-six-exact-game-v1' or manifest.get('status') != 'COMPLETE'
            or len(manifest.get('games', [])) != len(SIX)):
        raise ValueError('collector manifest contract')
    for item, (game, day) in zip(manifest['games'], SIX.items()):
        if (item.get('game_pk') != game or item.get('officialDate') != day
                or item.get('gameType') != 'L' or type(item.get('rows')) is not int
                or item['rows'] <= 0 or item['rows'] > 2000):
            raise ValueError('fixed game/date/type/source row census')
    projected = manifest['projected']
    if (projected.get('file') != 'postseason_six.parquet'
            or projected.get('columns') != M.cp.PROJECT
            or projected.get('rows') != sum(item['rows'] for item in manifest['games'])
            or projected.get('rows') > MAX_ROWS
            or any(projected.get(k) != src['projected'][k] for k in ('bytes', 'sha256'))):
        raise ValueError('projected source schema/count/binding')
    parquet_path = _regular_pinned(src['projected']['path'], projected['bytes'],
                                   projected['sha256'], 100 << 20)
    if parquet_path.name != projected['file'] or parquet_path.parent != manifest_path.parent:
        raise ValueError('collector output locality')
    pf = pq.ParquetFile(parquet_path)
    if pf.metadata.num_rows != projected['rows'] or pf.schema_arrow.names != M.cp.PROJECT:
        raise ValueError('Parquet footer projection')
    frame = pq.read_table(parquet_path, columns=M.cp.PROJECT, use_threads=False).to_pandas()
    if len(frame) != projected['rows'] or list(frame.columns) != M.cp.PROJECT:
        raise ValueError('decoded projection')
    counts = Counter(int(v) for v in frame.game_pk)
    if any(counts[g] != item['rows'] for g, item in zip(SIX, manifest['games'])):
        raise ValueError('per-game decoded row census')
    return frame, projected['sha256'], manifest


def run(regpath, grantpath):
    reg, _ = gates(regpath, grantpath)
    if (reg.get('stage') != 'POSTCUTOFF_FIXED_SIX_RETROSPECTIVE_SCORE'
            or reg.get('analysis_plan_sha256') != PLAN_SHA
            or reg.get('adapter_commit') != ADAPTER_COMMIT
            or reg.get('data_role') != '2026 EXPOSED DEVELOPMENT'
            or reg.get('leaf_order') != list(LEAVES)
            or reg['adapter_config']['sha256'] != CONFIG_SHA
            or reg['profile_bundle']['manifest_sha256'] != PROFILE_SHA):
        raise ValueError('fixed analytical contract changed')
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    M = closure()
    frame, source_sha, manifest = _source(reg, M)
    prepared = prepare(frame, source_sha, M)
    config_path = _regular_pinned(reg['adapter_config']['path'], reg['adapter_config']['bytes'],
                                  CONFIG_SHA, 1 << 20)
    config = checked_json(config_path)
    if (config.get('leaf_order') != list(LEAVES) or config.get('threads') != 1
            or config.get('schema') != 'ws-fullseason-shadow-v1'):
        raise ValueError('frozen adapter config')
    runtime = Runtime(config, reg['profile_bundle']['directory'], PROFILE_SHA)
    if runtime.identity['adapter_config_sha256'] != object_sha(config):
        raise ValueError('actual Runtime loaded config mismatch')
    metrics, internal = score_population(prepared, runtime.predict,
                                         source_sha256=source_sha, bundle_sha256=runtime.bundle_sha256)
    output = Path(reg['output'])
    raw = (canonical(internal) + '\n').encode()
    with (output / 'score_rows.json').open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    receipt = {'status': 'COMPLETED', 'stage': reg['stage'], 'registration_sha256': file_sha(regpath),
               'analysis_plan_sha256': PLAN_SHA, 'collector_manifest_sha256': reg['source']['manifest']['sha256'],
               'projected_sha256': source_sha, 'runtime_bundle_sha256': runtime.bundle_sha256,
               'profile_manifest_sha256': PROFILE_SHA,
               'score_rows_file': 'score_rows.json', 'score_rows_bytes': len(raw),
               'score_rows_sha256': file_sha(output / 'score_rows.json'),
               'collector_rows': manifest['projected']['rows'], 'census': prepared['census'],
               'metrics': metrics, 'prospective_claim': False, 'superiority_claim': False}
    with (output / 'receipt.pending.json').open('x') as f:
        json.dump(receipt, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--registration', required=True, type=Path)
    p.add_argument('--grant', required=True, type=Path)
    a = p.parse_args()
    run(a.registration, a.grant)
