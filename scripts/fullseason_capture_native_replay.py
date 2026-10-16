"""Gated finite capture-native experimental shadow, separate from strict rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import checked_json, sha as file_sha
from pitcheezy.seq_shadow import captured_bridge, captured_native, captured_native_state, fullseason_stream
from pitcheezy.seq_shadow.fullseason_runtime import Runtime
from pitcheezy.seq_shadow.ledger import sha as object_sha
from scripts import fullseason_captured_replay as replay

CONTRACT = captured_native.CONTRACT
NORMALIZER_SHA_SEMANTICS = 'sha256(canonical_json(native_source_pins))'
ORACLE_ABSENCE = ('No corrected I3 or other row reference is read, joined, or overlaid during native inference; '
                  'source-native six-field hypotheses remain non-Statcast-equivalent, strict observed rows UNKNOWN.')


def gated(regpath, grantpath):
    reg = replay.gated(regpath, grantpath)
    if (reg['input_contract'] != CONTRACT or reg['oracle_absence'] != ORACLE_ABSENCE
            or reg['normalizer_sha_semantics'] != NORMALIZER_SHA_SEMANTICS
            or reg['normalizer_sha256'] != object_sha(reg['native_source_pins'])
            or any(reg['source_pins'].get(str(ROOT / name)) != digest
                   for name, digest in reg['native_source_pins'].items())
            or 'normalizer_source' in reg
            or reg['data_role'] != '2026 EXPOSED DEVELOPMENT'
            or any(name in reg for name in ('i3', 'reference', 'corrected_rows'))):
        raise ValueError('capture-native source/oracle contract changed')
    return reg


def _native_map(rows, *, receipt_seq, source_sha256):
    mapped = {}
    for row in rows:
        if (row['input_contract'] != CONTRACT or row['accepted'] is not False
                or row['statcast_training_equivalent'] is not False
                or row['receipt_seq'] != receipt_seq or row['source_sha256'] != source_sha256):
            raise ValueError('native source row custody or acceptance changed')
        key = tuple(row['key'])
        if key in mapped:
            raise ValueError('duplicate native physical key in one body')
        mapped[key] = row
    return mapped


def _probabilities(result):
    if result['status'] != 'experimental_conditional_score':
        if result.get('probabilities') is not None:
            raise ValueError('abstention has probabilities')
        return
    probs = result['probabilities']
    if (not isinstance(probs, list) or len(probs) != 7
            or any(type(p) not in (int, float) or not math.isfinite(p) or p < 0 or p > 1 for p in probs)
            or not math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-6)):
        raise ValueError('invalid experimental seven-leaf output')


def run(regpath, grantpath):
    reg = gated(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    files = replay.entries(reg)
    config = checked_json(reg['adapter_config']['path'], 1 << 20)
    if (file_sha(reg['adapter_config']['path']) != reg['adapter_config']['sha256']
            or config['schema'] != 'ws-fullseason-shadow-v1'
            or config['seeds'] != [0, 1, 2]
            or config['leaf_order'] != reg['leaf_order'] or config['threads'] != 1):
        raise ValueError('adapter identity changed')
    runtime = Runtime(config, reg['profile_bundle']['directory'], reg['profile_bundle']['manifest_sha256'])
    scorer = captured_native_state.NativeRuntime(runtime, allow_native_experimental=True)
    state = captured_native_state.NativeShadow(Path(reg['output']) / 'native_stream.sqlite3',
        game_pk=replay.GAME, game_date=replay.DATE, normalizer_sha256=reg['normalizer_sha256'])
    counts = Counter({name: 0 for name in (
        'bodies', 'strict_new', 'strict_correction', 'strict_unchanged_observation',
        'native_native_candidate', 'native_native_unknown', 'native_orphan_rows',
        'bridge_abstain', 'repeated_candidate_key', 'query_cap_skips',
        'strict_decision_ready', 'strict_decision_abstain',
        'native_decision_native_ready', 'native_decision_native_abstain',
        'experimental_experimental_conditional_score', 'experimental_experimental_abstain')})
    reasons = Counter()
    context = Counter({name: 0 for name in (
        'prior_known_same_pitcher_total_sum', 'prior_known_same_pitcher_total_positive_decisions',
        'prior_unknown_relevant_sum', 'prior_unknown_relevant_positive_decisions',
        'selected_known_tokens_sum', 'selected_known_tokens_positive_decisions',
        'previous_pitch_known', 'previous_pitch_unknown', 'previous_pitch_missing')})
    selected, seen = [], set()
    compressed = inflated = 0
    try:
        for item in files:
            blob = replay.exact_file(Path(reg['capture_root']) / item['file'], replay.MAX_COMPRESSED)
            compressed += len(blob)
            if compressed > replay.MAX_COMPRESSED:
                raise ValueError('compressed total cap')
            body = replay.inflate_one(blob, replay.MAX_MEMBER)
            inflated += len(body)
            verified_sha = hashlib.sha256(body).hexdigest()
            if inflated > replay.MAX_INFLATED or verified_sha != item['sha256']:
                raise ValueError('inflated total cap or canonical body digest mismatch')
            feed = replay.feed_identity(body)
            received_ns = int(item['fetched'] * 1_000_000_000)
            step = captured_bridge.step(feed, game_pk=replay.GAME, game_date=replay.DATE,
                receipt_seq=item['seq'], source_sha256=verified_sha, candidate_pitch_type=replay.TYPE)
            natives = _native_map(captured_native.inspect_tokens(feed, game_pk=replay.GAME,
                game_date=replay.DATE, receipt_seq=item['seq'], source_sha256=verified_sha),
                receipt_seq=item['seq'], source_sha256=verified_sha)
            bridge_keys = set()
            for event in step['events']:
                key = tuple(event['key'])
                bridge_keys.add(key)
                result = state.ingest(event, natives.get(key), received_ns=received_ns,
                    source_sha256=verified_sha)
                counts['strict_' + result['strict_status']] += 1
                counts['native_' + result['native_status']] += 1
                if key in natives:
                    reasons['native_' + natives[key]['reason']] += 1
                else:
                    reasons['native_source_row_absent'] += 1
            counts['native_orphan_rows'] += len(set(natives) - bridge_keys)
            state.observe_receipt(item['seq'], verified_sha, received_ns)
            counts['bodies'] += 1
            candidate = step['candidate']
            if candidate is None:
                counts['bridge_abstain'] += 1
                reasons['bridge_' + str(step['abstain_reason'])] += 1
                continue
            if candidate['coverage']['source_sha256'] != verified_sha:
                raise ValueError('candidate receipt proof changed')
            key = tuple(candidate['query'][field] for field in fullseason_stream.KEY)
            if any(event_key >= key for event_key in bridge_keys):
                raise ValueError('observed event is not before hypothetical next key')
            if key in seen:
                counts['repeated_candidate_key'] += 1
                continue
            if len(seen) >= replay.MAX_QUERIES:
                counts['query_cap_skips'] += 1
                continue
            seen.add(key)
            if candidate['pitch_type_basis'] != 'hypothetical_candidate' or candidate['query']['pitch_type'] != replay.TYPE:
                raise ValueError('conditional FF query identity changed')
            decision = state.decide(candidate, received_ns=received_ns)
            result = scorer.predict(decision)
            _probabilities(result)
            if result.get('prediction_ready') is not False or result.get('prospective_claim') is not False:
                raise ValueError('experimental result claimed readiness or prospectivity')
            counts['strict_decision_' + decision['strict_status']] += 1
            counts['native_decision_' + decision['status']] += 1
            counts['experimental_' + result['status']] += 1
            if decision['reason']:
                reasons['decision_' + decision['reason']] += 1
            for field in ('prior_known_same_pitcher_total', 'prior_unknown_relevant', 'selected_known_tokens'):
                value = decision['context_counts'][field]
                if type(value) is not int or value < 0:
                    raise ValueError('native context count')
                context[field + '_sum'] += value
                context[field + '_positive_decisions'] += int(value > 0)
            previous = decision['context_counts']['previous_pitch_native_state']
            if previous not in ('known', 'unknown', 'missing'):
                raise ValueError('native predecessor state')
            context['previous_pitch_' + previous] += 1
            selected.append({'candidate': candidate, 'received_ns': received_ns,
                'native_decision_sha256': decision['decision_sha256'],
                'strict_snapshot_sha256': decision['strict_snapshot_sha256'],
                'status': decision['status'],
                'underlying_call_sha256': result.get('underlying_call_sha256')})
        state.close()
        state = captured_native_state.NativeShadow(Path(reg['output']) / 'native_stream.sqlite3',
            game_pk=replay.GAME, game_date=replay.DATE, normalizer_sha256=reg['normalizer_sha256'])
        for item in selected:
            same = state.decide(item['candidate'], received_ns=item['received_ns'])
            if (same['decision_sha256'], same['strict_snapshot_sha256'], same['status']) != (
                    item['native_decision_sha256'], item['strict_snapshot_sha256'], item['status']):
                raise ValueError('native decision changed after later receipts/restart')
        if state.db.execute("SELECT 1 FROM events WHERE status!='unknown' LIMIT 1").fetchone():
            raise ValueError('strict event promotion detected')
    finally:
        state.close()
    if (counts['bodies'] != replay.MAX_FILES or compressed != reg['capture_compressed_bytes_reported']
            or inflated != reg['capture_inflated_bytes_reported']):
        raise ValueError('capture total changed')
    export = [{k: value for k, value in item.items() if k != 'candidate'} for item in selected]
    report = {'status': 'COMPLETED', 'stage': 'CAPTURE_NATIVE_EXPERIMENTAL_CONDITIONAL_SHADOW',
        'registration_sha256': file_sha(regpath), 'grant_sha256': file_sha(grantpath),
        'source_commit': reg['source_commit'], 'game_pk': replay.GAME, 'official_date': replay.DATE,
        'input_contract': CONTRACT, 'oracle_absence': ORACLE_ABSENCE,
        'candidate_type': replay.TYPE, 'candidate_basis': 'hypothetical_candidate',
        'receipt_order': 'manifest seq 1..427; local fetched time availability upper bound',
        'counts': dict(sorted(counts.items())), 'reasons': dict(sorted(reasons.items())),
        'context': dict(sorted(context.items())), 'compressed_bytes': compressed,
        'inflated_bytes': inflated, 'selected_query_count': len(selected),
        'selected_decisions_sha256': object_sha(export), 'immutable_restart_checks': len(selected),
        'bundle_sha256': runtime.bundle_sha256, 'profile_origin': replay.DATE,
        'data_role': '2026 EXPOSED DEVELOPMENT', 'model_training_includes_game': True,
        'strict_accepted_rows': 0, 'statcast_training_equivalent_rows': 0,
        'prediction_ready': False, 'prospective_claim': False,
        'upstream_first_availability': 'UNKNOWN', 'stored_prefix_only': True,
        'fits': 0, 'calibration_fits': 0, 'performance_metrics': False,
        'raw_probabilities_written': False, 'we_conversion': False,
        'operational_switch': False, 'corrected_reference_reads': 0}
    (Path(reg['output']) / 'receipt.pending.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    args = parser.parse_args()
    run(args.registration, args.grant)
