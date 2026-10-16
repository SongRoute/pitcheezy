"""Aggregate source-native hypothesis comparisons; no accepted live row."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import gates, sha
from experiments.ws_fullseason.final_worker import read_source
from pitcheezy.seq_shadow import captured_beforestate, captured_diagnostic, captured_identity_bip
from scripts import fullseason_capture_diagnostic as v1

FIELDS = ('event_type_code', 'event_call_code', 'description_hypothesis', 'outcome10_hypothesis')
NEW_FIELDS = ('before_outs', 'before_home_score', 'before_away_score',
              'pitcher', 'batter', 'stand', 'p_throws', 'bip_description',
              'bip_events', 'bip_outcome10', 'bb_type')
SELECTION = ('all distinct physical keys at each key\'s first local receipt versus keys in final body 427; '
             'exclude cross-receipt changed/reused provider IDs before pairing to exact I3 game849809/officialOct11/L')
COMPARISON = ('exact equality for source-derived field hypotheses against corrected I3 on first-seen and final-body views; '
              'missing hypothesis/reference is unavailable, never defaulted or accepted; compare first-to-final type/call/description/outcome '
              'as raw value changes and reference-relative status transitions; reference-only rows get I3 field-presence metadata only')


def _unique(rows, counts, label):
    grouped, ambiguous = {}, set()
    for row in rows:
        key = row.get('key')
        if key is None:
            counts[label + '_unkeyed'] += 1
            continue
        key = tuple(key)
        if key in grouped:
            ambiguous.add(key)
        else:
            grouped[key] = row
    for key in ambiguous:
        grouped.pop(key)
    counts[label + '_ambiguous_keys'] += len(ambiguous)
    return grouped


def collect(reg):
    """Validate each registered body once, retaining only first/final keyed views."""
    first, final = {}, {}
    first_provider, provider_key, conflicts = {}, {}, set()
    counts = Counter(); compressed = inflated = 0
    for item in v1.replay.entries(reg):
        blob = v1.replay.exact_file(Path(reg['capture_root']) / item['file'], v1.replay.MAX_COMPRESSED)
        compressed += len(blob)
        if compressed > v1.replay.MAX_COMPRESSED:
            raise ValueError('compressed total cap')
        body = v1.replay.inflate_one(blob, v1.replay.MAX_MEMBER)
        inflated += len(body)
        if inflated > v1.replay.MAX_INFLATED or hashlib.sha256(body).hexdigest() != item['sha256']:
            raise ValueError('inflated total cap or canonical digest mismatch')
        feed = v1.replay.feed_identity(body)
        base = _unique(captured_diagnostic.inspect_body(feed, game_pk=v1.GAME, game_date=v1.DATE), counts, 'base')
        before = _unique(captured_beforestate.prefix_candidates(feed, game_pk=v1.GAME, game_date=v1.DATE), counts, 'before')
        identity = _unique(captured_identity_bip.inspect_hypotheses(feed, game_pk=v1.GAME, game_date=v1.DATE), counts, 'identity_bip')
        group = {}
        for key, row in base.items():
            b, i = before.get(key), identity.get(key)
            if row['accepted'] is not False or (b is not None and b['accepted'] is not False) or (i is not None and i['accepted'] is not False):
                raise ValueError('source diagnostic promoted a row')
            group[key] = {'base': row, 'before': b, 'identity_bip': i}
            provider = row['provider_id']
            if provider is not None:
                earlier, other = first_provider.get(key), provider_key.get(provider)
                if earlier is not None and earlier != provider:
                    conflicts.add(key)
                if other is not None and other != key:
                    conflicts.update((key, other))
                first_provider.setdefault(key, provider)
                provider_key.setdefault(provider, key)
            first.setdefault(key, group[key])
        if item['seq'] == v1.replay.MAX_FILES:
            final = group
        counts['bodies'] += 1
    if (counts['bodies'] != v1.replay.MAX_FILES or compressed != reg['capture_compressed_bytes_reported']
            or inflated != reg['capture_inflated_bytes_reported']):
        raise ValueError('capture total/receipt mismatch')
    counts['distinct_physical_keys_observed'] = len(first)
    counts['provider_conflict_keys_excluded'] = len(conflicts)
    first = {k: v for k, v in first.items() if k not in conflicts}
    final = {k: v for k, v in final.items() if k not in conflicts}
    counts['distinct_first_seen_keys'] = len(first)
    counts['final_body_keys'] = len(final)
    counts['first_seen_missing_from_final_body'] = len(set(first) - set(final))
    return first, final, dict(sorted(counts.items())), compressed, inflated


def _field_value(record, field):
    row = record['base']
    if field == 'event_type_code':
        return row['observed'][field] if row['type_state'] == 'known' else None
    return row['observed'][field] if field == 'event_call_code' else row[field]


def _expected(target, field):
    if field == 'event_type_code': return target.pitch_type
    if field == 'description_hypothesis': return target.description
    if field == 'outcome10_hypothesis':
        result = v1._int(target.source_outcome10)
        return result if result is not None and result >= 0 else None
    return None  # No I3 GUMBO call-code reference exists.


def transitions(first, final, refs):
    counts = Counter()
    for key in first.keys() & final.keys():
        target = refs.get(key)
        if target is None:
            counts['capture_only_key'] += 1
            continue
        counts['paired_both_views'] += 1
        for field in FIELDS:
            old, new = _field_value(first[key], field), _field_value(final[key], field)
            raw = ('both_unavailable' if old is None and new is None else
                   'became_available' if old is None else
                   'became_unavailable' if new is None else
                   'same' if old == new else 'changed')
            counts[field + '_raw_' + raw] += 1
            if field == 'event_call_code':
                counts['event_call_code_reference_unavailable'] += 1
                continue
            expected = _expected(target, field)
            def status(value):
                return ('unavailable' if value is None else 'reference_unavailable' if expected is None
                        else 'exact' if value == expected else 'mismatch')
            counts[field + '_reference_' + status(old) + '_to_' + status(new)] += 1
    return dict(sorted(counts.items()))


def _compare(counts, name, observed, expected):
    v1._compare(counts, name, observed, expected)


def paired(view, refs):
    counts = Counter()
    for key, record in view.items():
        target = refs.get(key)
        if target is None:
            counts['capture_only_key'] += 1
            continue
        counts['paired_key'] += 1
        base, before, identity = record['base'], record['before'], record['identity_bip']
        if (base['accepted'] is not False or (before is not None and before['accepted'] is not False)
                or (identity is not None and identity['accepted'] is not False)):
            raise ValueError('diagnostic promoted a row')
        counts['base_reason_' + base['reason']] += 1
        counts['before_reason_' + (before['reason'] if before else 'module_row_absent')] += 1
        counts['identity_reason_' + (identity['reasons']['identity'] if identity else 'module_row_absent')] += 1
        counts['bip_reason_' + (identity['reasons']['bip'] if identity else 'module_row_absent')] += 1
        state = before['before_outs_score_hypothesis'] if before else None
        matchup = identity['identity_hypothesis'] if identity else None
        for name, observed, expected in (
            ('before_outs', state[0] if state else None, v1._int(target.outs_when_up)),
            ('before_home_score', state[1] if state else None, v1._int(target.home_score)),
            ('before_away_score', state[2] if state else None, v1._int(target.away_score)),
            ('pitcher', matchup['pitcher'] if matchup else None, v1._int(target.pitcher)),
            ('batter', matchup['batter'] if matchup else None, v1._int(target.batter)),
            ('stand', matchup['stand'] if matchup else None, target.stand),
            ('p_throws', matchup['throws'] if matchup else None, target.p_throws),
            ('bip_description', identity['description_bip_hypothesis'] if identity else None, target.description),
            ('bip_events', identity['events_bip_hypothesis'] if identity else None, target.events),
            ('bip_outcome10', identity['outcome10_bip_hypothesis'] if identity else None,
             _expected(target, 'outcome10_hypothesis')),
            ('bb_type', identity['bb_type_hypothesis'] if identity else None, target.bb_type),
        ):
            _compare(counts, name, observed, expected)
            reason = (before['reason'] if before else 'module_row_absent') if name.startswith('before_') else (
                (identity['reasons']['identity'] if identity else 'module_row_absent')
                if name in ('pitcher', 'batter', 'stand', 'p_throws') else
                (identity['reasons']['bip'] if identity else 'module_row_absent'))
            counts[name + '_eligibility_' + ('hypothesis_unpaired' if observed is not None else reason)] += 1
        counts['accepted_false'] += 1
    counts['reference_only_key'] = len(set(refs) - set(view))
    return dict(sorted(counts.items()))


def reference_only(final, refs):
    """I3-only classification using registered columns; no inferred feed events."""
    counts = Counter()
    for key in set(refs) - set(final):
        row = refs[key]
        counts['capture_absent_statcast_row'] += 1
        for field in ('pitch_type', 'description', 'events', 'bb_type', 'pitcher',
                      'batter', 'balls', 'strikes', 'outs_when_up', 'home_score', 'away_score',
                      'stand', 'p_throws'):
            counts[field + ('_absent' if v1.pd.isna(getattr(row, field)) else '_present')] += 1
    return dict(sorted(counts.items()))


def run(regpath, grantpath):
    reg, recipe = gates(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    if (reg['game_pk'], reg['game_date'], reg['columns'], reg['i3']['rows']) != (v1.GAME, v1.DATE, v1.COLUMNS, v1.ROWS):
        raise ValueError('fixed paired diagnostic scope')
    if (reg['bounds'] != v1.BOUNDS or reg['capture_body_count'] != v1.replay.MAX_FILES
            or reg['i3']['parquet_sha256'] != v1.I3_SHA
            or reg['i3']['manifest_sha256'] != v1.I3_MANIFEST_SHA
            or reg['i3']['version'] != 'd20261012-i2026-e20261011'
            or reg['selection_rule'] != SELECTION or reg['comparison_rule'] != COMPARISON
            or recipe['data_role'] != '2026 EXPOSED DEVELOPMENT'
            or reg['capture_root'] != reg['manifest']['path'].rsplit('/', 1)[0]):
        raise ValueError('fixed source/resource/rules')
    source = read_source('I3', reg['i3'], v1.COLUMNS, reg['bounds'])
    if (len(source) != v1.ROWS or sha(reg['i3']['parquet']) != reg['i3']['parquet_sha256']
            or sha(reg['i3']['manifest']) != reg['i3']['manifest_sha256']):
        raise ValueError('I3 changed after read')
    refs = v1.reference(source)  # hard game/date/type/duplicate-key check before capture read
    first, final, capture_counts, compressed, inflated = collect(reg)
    comparison = {'first_seen': paired(first, refs), 'final_body': paired(final, refs)}
    result = {'status': 'COMPLETED', 'stage': 'CAPTURED_I3_FIELD_HYPOTHESIS_DIAGNOSTIC_V2',
              'registration_sha256': sha(regpath), 'grant_sha256': sha(grantpath),
              'source_commit': reg['source_commit'], 'game_pk': v1.GAME, 'official_date': v1.DATE,
              'capture_receipts': capture_counts, 'compressed_bytes': compressed,
              'inflated_bytes': inflated, 'i3_source_rows_decoded': v1.ROWS,
              'i3_game_rows': len(refs), 'i3_columns_decoded': v1.COLUMNS,
              'comparison': comparison, 'first_to_final': transitions(first, final, refs),
              'reference_only_final_body': reference_only(final, refs),
              'comparison_semantics': COMPARISON,
              'source_role': 'corrected retrospective I3, arrival UNKNOWN',
              'capture_role': 'local stored receipt order; upstream arrival UNKNOWN',
              'data_role': '2026 EXPOSED DEVELOPMENT', 'accepted_rows_emitted': 0,
              'model_or_profile_reads': 0, 'fits': 0, 'calibration_fits': 0,
              'performance_metrics': False, 'per_key_rows_exported': False,
              'inference_overlay': False, 'operational_switch': False}
    (Path(reg['output']) / 'receipt.pending.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    args = parser.parse_args()
    run(args.registration, args.grant)
