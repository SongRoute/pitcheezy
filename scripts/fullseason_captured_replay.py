"""Gated, bounded receipt-order reconstruction of one captured conditional shadow.

The capture is retrospective engineering evidence. It is never a prospective
forecast or performance evaluation. No per-pitch probabilities are exported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import checked_json, sha
from pitcheezy.seq_shadow import captured_bridge, fullseason_stream
from pitcheezy.seq_shadow.fullseason_runtime import Runtime

GAME = 849809
DATE = '2026-10-11'
TYPE = 'FF'
MAX_FILES = 427
MAX_COMPRESSED = 32 << 20
MAX_INFLATED = 205 << 20
MAX_MEMBER = 64 << 20
MAX_QUERIES = 100


def gated(regpath, grantpath):
    reg = checked_json(regpath, 1 << 20)
    grant = checked_json(grantpath, 1 << 20)
    if not (grant.get('execute') is True and grant.get('verdict') == 'PASS'
            and grant.get('authorized_by') == 'root'
            and grant.get('registration_sha256') == sha(regpath)
            and grant.get('reviewer') and grant['reviewer'] != reg.get('author')):
        raise ValueError('independent Root execution grant required')
    for path, digest in reg['source_pins'].items():
        if sha(path) != digest: raise ValueError('source pin mismatch')
    if (reg['game_pk'], reg['game_date'], reg['candidate_pitch_type']) != (GAME, DATE, TYPE):
        raise ValueError('fixed game/date/hypothetical type')
    if reg['query_rule'] != 'first 100 distinct bridge-produced next-hypothetical keys in manifest receipt order; bridge abstention skips; stream abstention consumes; all 427 bodies ingested':
        raise ValueError('fixed query selection')
    if reg['bounds'] != {'max_files': MAX_FILES, 'compressed_total_bytes': MAX_COMPRESSED,
                          'inflated_total_bytes': MAX_INFLATED, 'member_inflated_bytes': MAX_MEMBER,
                          'max_decision_queries': MAX_QUERIES, 'wall_seconds': 300,
                          'rss_bytes': 2 << 30, 'free_memory_bytes': 4 << 30,
                          'free_disk_bytes': 12 << 30, 'output_bytes': 16 << 20,
                          'threads': 1}:
        raise ValueError('registered bounds changed')
    return reg


def exact_file(path, size_cap):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= size_cap:
        raise ValueError('regular bounded file required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        opened = os.fstat(handle.fileno())
        if (not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino, opened.st_size)
                != (info.st_dev, info.st_ino, info.st_size)):
            raise ValueError('file changed during open')
        body = handle.read(size_cap + 1)
        if len(body) != info.st_size or len(body) > size_cap:
            raise ValueError('file changed during bounded read')
    return body


def inflate_one(blob, cap):
    """Match capture_to_replay's exactly-one complete gzip member contract."""
    try:
        inflater = zlib.decompressobj(31)
        body = inflater.decompress(blob, cap + 1)
    except zlib.error as exc:
        raise ValueError('invalid gzip member') from exc
    if len(body) > cap or not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError('oversize, incomplete or trailing gzip member')
    return body


def feed_identity(body):
    feed = json.loads(body)
    if (not isinstance(feed, dict) or not isinstance(feed.get('gameData'), dict)
            or not isinstance(feed['gameData'].get('game'), dict)
            or not isinstance(feed['gameData'].get('datetime'), dict)
            or feed['gameData']['game'].get('pk') != GAME
            or feed['gameData']['datetime'].get('officialDate') != DATE):
        raise ValueError('captured body game/officialDate mismatch')
    return feed


def entries(reg):
    if sha(reg['allowlist']['path']) != reg['allowlist']['sha256']:
        raise ValueError('allowlist changed')
    allow = checked_json(reg['allowlist']['path'], 1 << 20)
    if allow['manifest'] != reg['manifest'] or allow['root'] != reg['capture_root']:
        raise ValueError('capture metadata binding')
    raw = exact_file(reg['manifest']['path'], 1 << 20)
    if hashlib.sha256(raw).hexdigest() != reg['manifest']['sha256']:
        raise ValueError('manifest changed')
    manifest = json.loads(raw)
    if (manifest.get('game_pk') != GAME or manifest.get('status') != 'OK'
            or manifest.get('files') != allow['files']):
        raise ValueError('manifest game/list order mismatch')
    files = allow['files']
    if len(files) != MAX_FILES or [v['seq'] for v in files] != list(range(1, MAX_FILES + 1)):
        raise ValueError('receipt sequence mismatch')
    prior_receipt = 0
    for item in files:
        name = item['file']
        if (not isinstance(name, str) or '/' in name or name in ('', '.', '..')
                or not name.startswith(f'{GAME}_') or not name.endswith('.json.gz')):
            raise ValueError('unsafe body name')
        if (not isinstance(item['sha256'], str) or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('body digest')
        if (not isinstance(item['fetched'], (int, float)) or not math.isfinite(item['fetched'])
                or item['fetched'] < prior_receipt):
            raise ValueError('receipt clock order')
        prior_receipt = item['fetched']
    return files


def run(regpath, grantpath):
    reg = gated(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    files = entries(reg)
    out = Path(reg['output'])
    config = checked_json(reg['adapter_config']['path'], 1 << 20)
    if sha(reg['adapter_config']['path']) != reg['adapter_config']['sha256']:
        raise ValueError('adapter config changed')
    if (config['schema'] != 'ws-fullseason-shadow-v1' or config['seeds'] != [0, 1, 2]
            or config['leaf_order'] != reg['leaf_order'] or config['threads'] != 1):
        raise ValueError('registered adapter identity changed')
    if sha(reg['normalizer_source']) != reg['normalizer_sha256']:
        raise ValueError('normalizer source changed')
    runtime = Runtime(config, reg['profile_bundle']['directory'], reg['profile_bundle']['manifest_sha256'])
    state = fullseason_stream.StreamState(out / 'stream.sqlite3', game_pk=GAME, game_date=DATE,
        normalizer_sha256=reg['normalizer_sha256'], evidence_kind='captured')
    counts = Counter(); reasons = Counter(); seen = set(); selected = []
    latest_event = {}
    compressed = inflated = 0
    try:
        for item in files:
            path = Path(reg['capture_root']) / item['file']
            blob = exact_file(path, MAX_COMPRESSED)
            compressed += len(blob)
            if compressed > MAX_COMPRESSED: raise ValueError('compressed total cap')
            body = inflate_one(blob, MAX_MEMBER)
            inflated += len(body)
            if inflated > MAX_INFLATED: raise ValueError('inflated total cap')
            if hashlib.sha256(body).hexdigest() != item['sha256']:
                raise ValueError('inflated canonical body digest mismatch')
            feed = feed_identity(body)
            receipt_ns = int(item['fetched'] * 1_000_000_000)
            step = captured_bridge.step(feed, game_pk=GAME, game_date=DATE,
                receipt_seq=item['seq'], source_sha256=item['sha256'], candidate_pitch_type=TYPE)
            if step['candidate'] is not None and step['candidate']['coverage']['source_sha256'] != item['sha256']:
                raise ValueError('candidate does not bind this receipt body')
            for event in step['events']:
                event_key = tuple(event['key'])
                event_value = (event['event_id'], json.dumps(event['row'], sort_keys=True, separators=(',', ':')))
                if latest_event.get(event_key) == event_value:
                    counts['unchanged_observation'] += 1
                    continue
                result = state.ingest(event['event_id'], event['key'], event['revision'], event['row'],
                                      received_ns=receipt_ns)
                latest_event[event_key] = event_value
                counts['event_' + result['status']] += 1
                counts['fate_' + event['row']['status']] += 1
                reasons['fate_' + event['row']['reason']] += 1
            counts['bodies'] += 1
            candidate = step['candidate']
            if candidate is None:
                counts['bridge_abstain'] += 1
                reasons['bridge_' + str(step['abstain_reason'])] += 1
                continue
            key = tuple(candidate['query'][k] for k in fullseason_stream.KEY)
            if key in seen:
                counts['repeated_candidate_key'] += 1
                continue
            if len(seen) >= MAX_QUERIES:
                counts['query_cap_skips'] += 1
                continue
            seen.add(key)
            if candidate['pitch_type_basis'] != 'hypothetical_candidate' or candidate['query']['pitch_type'] != TYPE:
                raise ValueError('candidate basis/type changed')
            snapshot = state.decide(candidate['decision_id'], candidate['query'],
                pitch_type_basis=candidate['pitch_type_basis'], coverage=candidate['coverage'],
                received_ns=receipt_ns)
            scored = runtime.predict(snapshot)
            if scored['status'] == 'scored':
                probs = scored['probabilities']
                if (len(probs) != 7 or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs)
                        or not math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-6)):
                    raise ValueError('non-finite seven-leaf output')
            counts['decision_' + scored['status']] += 1
            if scored['reason']: reasons['decision_' + scored['reason']] += 1
            selected.append({'decision_id': candidate['decision_id'], 'query': candidate['query'],
                'basis': candidate['pitch_type_basis'], 'coverage': candidate['coverage'],
                'received_ns': receipt_ns, 'snapshot_sha256': snapshot['snapshot_sha256'],
                'call_sha256': scored['call_sha256']})
        # The same owned ledger must return every original decision after a
        # genuine connection restart, including after later corrections.
        state.close()
        state = fullseason_stream.StreamState(out / 'stream.sqlite3', game_pk=GAME, game_date=DATE,
            normalizer_sha256=reg['normalizer_sha256'], evidence_kind='captured')
        for d in selected:
            same = state.decide(d['decision_id'], d['query'], pitch_type_basis=d['basis'],
                coverage=d['coverage'], received_ns=d['received_ns'])
            if same['snapshot_sha256'] != d['snapshot_sha256']:
                raise ValueError('decision changed after later receipt/correction')
    finally:
        state.close()
    if counts['bodies'] != MAX_FILES:
        raise ValueError('incomplete receipt replay')
    if (compressed != reg['capture_compressed_bytes_reported']
            or inflated != reg['capture_inflated_bytes_reported']):
        raise ValueError('capture byte totals changed')
    decision_hash = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    report = {'status': 'COMPLETED', 'stage': 'CAPTURED_RETROSPECTIVE_ENGINEERING_REPLAY',
        'registration_sha256': sha(regpath), 'grant_sha256': sha(grantpath),
        'source_commit': reg['source_commit'], 'game_pk': GAME, 'official_date': DATE,
        'candidate_type': TYPE, 'candidate_basis': 'hypothetical_candidate',
        'receipt_order': 'manifest seq 1..427; local fetched time availability upper bound',
        'counts': {k: counts[k] for k in ('bodies', 'event_new', 'event_correction',
            'unchanged_observation', 'fate_accepted', 'fate_rejected', 'fate_unknown',
            'bridge_abstain', 'decision_scored', 'decision_abstain',
            'repeated_candidate_key', 'query_cap_skips')},
        'reasons': dict(reasons), 'compressed_bytes': compressed,
        'inflated_bytes': inflated, 'selected_decisions_sha256': decision_hash,
        'selected_query_count': len(selected), 'prediction_ready': counts['decision_scored'] > 0,
        'immutable_restart_checks': len(selected), 'bundle_sha256': runtime.bundle_sha256,
        'profile_origin': DATE, 'data_role': '2026 EXPOSED DEVELOPMENT',
        'prospective_claim': False, 'upstream_first_availability': 'UNKNOWN',
        'stored_prefix_only': True, 'fits': 0, 'calibration_fits': 0,
        'performance_metrics': False, 'raw_probabilities_written': False,
        'we_conversion': False, 'operational_switch': False}
    (out / 'receipt.pending.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    args = parser.parse_args()
    run(args.registration, args.grant)
