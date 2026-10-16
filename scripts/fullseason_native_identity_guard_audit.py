"""Bounded, model-free identity-guard audit of registered captured receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import checked_json, sha as file_sha
from pitcheezy.seq_shadow import captured_bridge, captured_native
from pitcheezy.seq_shadow.captured_identity_bip import inspect_hypotheses
from pitcheezy.seq_shadow.captured_native_identity_diagnostic import (
    inspect_identity_guards, summarize_guard_census,
)
from pitcheezy.seq_shadow.captured_native_state import NativeShadow
from pitcheezy.seq_shadow.fullseason_stream import KEY
from pitcheezy.seq_shadow.ledger import sha as object_sha
from scripts import fullseason_captured_replay as replay


def gated(regpath, grantpath):
    reg = replay.gated(regpath, grantpath)
    if (reg['stage'] != 'CAPTURE_NATIVE_IDENTITY_GUARD_AUDIT'
            or reg['input_contract'] != captured_native.CONTRACT
            or reg['actual_input_scope'] != '427_capture_bodies_and_named_accepted_receipt_metadata_only'
            or any(name in reg for name in ('adapter_config', 'profile_bundle', 'model_artifacts',
                                            'model', 'i3', 'recipe'))
            or reg['normalizer_sha_semantics'] != 'sha256(canonical_json(native_source_pins))'
            or reg['normalizer_sha256'] != object_sha(reg['native_source_pins'])
            or any(reg['source_pins'].get(str(ROOT / name)) != digest
                   for name, digest in reg['native_source_pins'].items())
            or reg['reference_receipt']['sha256'] != file_sha(reg['reference_receipt']['path'])
            or reg['guard_event_type_cap'] != 32 or reg['guard_extra_key_cap'] != 32
            or reg['guard_label_length_cap'] != 64):
        raise ValueError('identity guard audit registration changed')
    prior = checked_json(reg['reference_receipt']['path'], 1 << 20)
    if (prior['status'] != 'COMPLETED' or prior['selected_query_count'] != replay.MAX_QUERIES
            or prior['counts']['native_decision_native_ready'] != 10
            or prior['counts']['native_decision_native_abstain'] != 90
            or prior['reasons']['decision_native_prior_quality_unknown'] != 90
            or prior['selected_keys_sha256'] != reg['expected_selected_keys_sha256']):
        raise ValueError('accepted comparison receipt changed')
    return reg, prior


def _native_map(rows, *, seq, digest):
    mapped = {}
    for row in rows:
        if (row['receipt_seq'] != seq or row['source_sha256'] != digest
                or row['accepted'] is not False or row['input_contract'] != captured_native.CONTRACT):
            raise ValueError('native source custody changed')
        key = tuple(row['key'])
        if key in mapped:
            raise ValueError('duplicate native key')
        mapped[key] = row
    return mapped


def relevant_unknown_attribution(latest, candidate, decision):
    """Mirror NativeShadow's 32-known-token floor and verify its count."""
    query = candidate['query']
    key = tuple(query[field] for field in KEY)
    rows = sorted(((item_key, record) for item_key, record in latest.items()
                   if item_key[1:] < key[1:]), key=lambda item: item[0][1:], reverse=True)
    known = [(item_key, record) for item_key, record in rows
             if record['status'] == 'native_candidate'
             and record['native_token']['pitcher'] == query['pitcher']]
    known_total = len(known)
    floor = known[31][0][1:] if len(known) >= 32 else None
    relevant = [(item_key, record) for item_key, record in rows
                if record['status'] != 'native_candidate'
                and (floor is None or item_key[1:] >= floor
                     or item_key[1:] == (key[1], key[2] - 1))]
    context = decision['context_counts']
    if (context['prior_known_same_pitcher_total'] != known_total
            or context['selected_known_tokens'] != min(known_total, 32)
            or context['prior_unknown_relevant'] != len(relevant)):
        raise ValueError('diagnostic/native prefix dependency divergence')
    reasons = Counter(record['guard'] for _, record in relevant)
    return reasons, {item_key for item_key, _ in relevant}


def remember_native(latest, key, native, guard, strict_status):
    """Keep the ledger's first proof on dedup; replace it on a correction."""
    if strict_status == 'unchanged_observation':
        if key not in latest:
            raise ValueError('unchanged observation without earlier proof')
        return
    if native is None:
        latest[key] = {'status': 'native_unknown', 'native_token': None,
                       'guard': 'source_row_absent'}
    else:
        latest[key] = {'status': native['status'], 'native_token': native['native_token'],
                       'guard': (guard if native['reason'] == 'pitch_time_matchup_unverified'
                                 else native['reason'])}


def run(regpath, grantpath):
    reg, prior = gated(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    files = replay.entries(reg)
    output = Path(reg['output'])
    if any(output.iterdir()):
        raise ValueError('fresh output directory required')
    receipt_views, selected, seen = [], [], set()
    latest = {}
    counts, dependencies, decision_causes, native_reason_guard = (
        Counter() for _ in range(4))
    distinct_dependency_keys = set()
    compressed = inflated = 0
    with tempfile.TemporaryDirectory(prefix='guard-ledger-', dir=output) as tmp:
        ledger = Path(tmp) / 'native_stream.sqlite3'
        state = NativeShadow(ledger, game_pk=replay.GAME, game_date=replay.DATE,
                             normalizer_sha256=reg['normalizer_sha256'])
        try:
            for item in files:
                blob = replay.exact_file(Path(reg['capture_root']) / item['file'], replay.MAX_COMPRESSED)
                compressed += len(blob)
                if compressed > replay.MAX_COMPRESSED:
                    raise ValueError('compressed total cap')
                body = replay.inflate_one(blob, replay.MAX_MEMBER)
                inflated += len(body)
                digest = hashlib.sha256(body).hexdigest()
                if inflated > replay.MAX_INFLATED or digest != item['sha256']:
                    raise ValueError('canonical body custody changed')
                feed = replay.feed_identity(body)
                ns = int(item['fetched'] * 1_000_000_000)
                step = captured_bridge.step(feed, game_pk=replay.GAME, game_date=replay.DATE,
                    receipt_seq=item['seq'], source_sha256=digest, candidate_pitch_type=replay.TYPE)
                natives = _native_map(captured_native.inspect_tokens(feed, game_pk=replay.GAME,
                    game_date=replay.DATE, receipt_seq=item['seq'], source_sha256=digest),
                    seq=item['seq'], digest=digest)
                base = inspect_hypotheses(feed, game_pk=replay.GAME, game_date=replay.DATE)
                guards = inspect_identity_guards(feed, game_pk=replay.GAME,
                    game_date=replay.DATE, all_pitch_rows=base)
                receipt_views.append((item['seq'], guards))
                event_keys = set()
                for event in step['events']:
                    key = tuple(event['key'])
                    event_keys.add(key)
                    native = natives.get(key)
                    observation = state.ingest(event, native, received_ns=ns, source_sha256=digest)
                    remember_native(latest, key, native, guards[key]['guard'] if key in guards else None,
                                    observation['strict_status'])
                    counts['strict_' + observation['strict_status']] += 1
                    counts['native_' + (native['status'] if native else 'native_unknown')] += 1
                    if native and native['reason'] == 'pitch_time_matchup_unverified':
                        guard = guards[key]['guard']
                        if guard == 'identity_hypothesis_present':
                            raise ValueError('native/guard identity divergence')
                        native_reason_guard[guard] += 1
                counts['native_orphan_rows'] += len(set(natives) - event_keys)
                state.observe_receipt(item['seq'], digest, ns)
                counts['bodies'] += 1
                candidate = step['candidate']
                if candidate is None:
                    counts['bridge_abstain'] += 1
                    continue
                key = tuple(candidate['query'][field] for field in KEY)
                if candidate['coverage']['source_sha256'] != digest or any(k >= key for k in event_keys):
                    raise ValueError('query receipt or pre-observation guard changed')
                if key in seen:
                    counts['repeated_candidate_key'] += 1
                    continue
                if len(seen) >= replay.MAX_QUERIES:
                    counts['query_cap_skips'] += 1
                    continue
                if (candidate['pitch_type_basis'] != 'hypothetical_candidate'
                        or candidate['query']['pitch_type'] != replay.TYPE):
                    raise ValueError('fixed hypothetical FF query changed')
                seen.add(key)
                decision = state.decide(candidate, received_ns=ns)
                causes, relevant_keys = relevant_unknown_attribution(latest, candidate, decision)
                for cause, number in causes.items():
                    dependencies[cause] += number
                    decision_causes[cause] += 1
                distinct_dependency_keys.update(relevant_keys)
                counts['native_' + decision['status']] += 1
                counts['reason_' + str(decision['reason'])] += 1
                selected.append({'key': list(key), 'candidate': candidate, 'received_ns': ns,
                                 'decision_sha256': decision['decision_sha256']})
            state.close()
            state = NativeShadow(ledger, game_pk=replay.GAME, game_date=replay.DATE,
                                 normalizer_sha256=reg['normalizer_sha256'])
            for item in selected:
                again = state.decide(item['candidate'], received_ns=item['received_ns'])
                if again['decision_sha256'] != item['decision_sha256']:
                    raise ValueError('diagnostic native decision changed after reopen')
        finally:
            state.close()
    selected_digest = object_sha([item['key'] for item in selected])
    if (counts['bodies'] != replay.MAX_FILES or len(selected) != replay.MAX_QUERIES
            or selected_digest != prior['selected_keys_sha256']
            or counts['native_native_ready'] != 10 or counts['native_native_abstain'] != 90
            or counts['reason_native_prior_quality_unknown'] != 90
            or sum(native_reason_guard.values()) != prior['reasons']['native_pitch_time_matchup_unverified']
            or counts['native_native_candidate'] != prior['counts']['native_native_candidate']
            or counts['native_native_unknown'] != prior['counts']['native_native_unknown']
            or compressed != prior['compressed_bytes'] or inflated != prior['inflated_bytes']):
        raise ValueError('accepted capture/selection/native partition not reproduced')
    census = summarize_guard_census(receipt_views)
    final_keys = set(receipt_views[-1][1])
    census['final_body_guard'] = dict(sorted(Counter(
        row['guard'] for row in receipt_views[-1][1].values()).items()))
    census['distinct_keys_absent_from_final_body'] = census['distinct_keys'] - len(final_keys)
    report = {'status': 'COMPLETED', 'stage': reg['stage'], 'registration_sha256': file_sha(regpath),
              'grant_sha256': file_sha(grantpath), 'source_commit': reg['source_commit'],
              'game_pk': replay.GAME, 'official_date': replay.DATE,
              'selected_keys_sha256': selected_digest, 'selected_queries': len(selected),
              'native_ready': counts['native_native_ready'],
              'native_abstain': counts['native_native_abstain'],
              'native_prior_quality_unknown': counts['reason_native_prior_quality_unknown'],
              'counts': dict(sorted(counts.items())), 'census': census,
              'pitch_time_matchup_unverified_by_guard': dict(sorted(native_reason_guard.items())),
              'relevant_unknown_dependencies': dict(sorted(dependencies.items())),
              'decisions_with_relevant_unknown_cause': dict(sorted(decision_causes.items())),
              'distinct_relevant_unknown_keys': len(distinct_dependency_keys),
              'compressed_bytes': compressed, 'inflated_bytes': inflated,
              'model_reads': 0, 'profile_reads': 0, 'i3_reads': 0, 'old_sqlite_reads': 0,
              'scores': 0, 'probabilities_exported': False, 'accepted_rows': 0,
              'performance_metrics': False, 'prospective_claim': False}
    (output / 'receipt.pending.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    args = parser.parse_args()
    run(args.registration, args.grant)
