"""Gated aggregate field comparison of captured GUMBO against exact I3 rows.

This is paired engineering evidence, never an accepted live row, model score,
performance metric, or permission to backfill a previous replay decision.
"""
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
import pandas as pd
from experiments.ws_fullseason.audit import gates, sha
from experiments.ws_fullseason.final_worker import read_source
from experiments.ws_fullseason.data import closure
from pitcheezy.seq_shadow import captured_diagnostic
from scripts import fullseason_captured_replay as replay

GAME = 849809
DATE = '2026-10-11'
ROWS = 7534
I3_SHA = 'd045ab62a1694a87e383a8d30a90c4006975697a2d2a6eb294e398e422e350c0'
I3_MANIFEST_SHA = '8f91af339de0aabaca881f9083518016b0005c340b90799cebfb8dd76f88de8a'
COLUMNS = ['game_pk', 'at_bat_number', 'pitch_number', 'game_date', 'game_type',
           'pitcher', 'batter', 'pitch_type', 'description', 'events', 'bb_type',
           'balls', 'strikes', 'outs_when_up', 'home_score', 'away_score',
           'stand', 'p_throws']
VIEWS = ('first_seen', 'final_body')
DIRECT_FIELDS = ('pre_count', 'event_type_code', 'description_hypothesis', 'outcome10_hypothesis')
FINAL_FIELDS = ('play_final_pitcher', 'play_final_batter', 'play_final_stand',
                'play_final_throws', 'play_final_outs', 'play_final_home_score',
                'play_final_away_score')
BOUNDS = {'wall_seconds': 300, 'rss_bytes': 2 << 30, 'free_memory_bytes': 4 << 30,
          'free_disk_bytes': 12 << 30, 'output_bytes': 16 << 20,
          'parquet_bytes': 1 << 30, 'manifest_bytes': 1 << 20,
          'capture_files': 427, 'compressed_total_bytes': 32 << 20,
          'inflated_total_bytes': 205 << 20, 'member_inflated_bytes': 64 << 20,
          'threads': 1}


def _int(value):
    if pd.isna(value): return None
    result = int(value)
    if result != value: raise ValueError('nonintegral reference value')
    return result


def reference(frame):
    game = frame.loc[frame.game_pk == GAME].copy()
    if (game.empty or len(game) > ROWS
            or not (pd.to_datetime(game.game_date) == pd.Timestamp(DATE)).all()
            or not game.game_type.eq('L').all()):
        raise ValueError('registered I3 game/date/type absent or conflicting')
    if (game[['at_bat_number', 'pitch_number']].isna().any().any()
            or game.duplicated(['at_bat_number', 'pitch_number']).any()):
        raise ValueError('ambiguous I3 physical keys')
    m = closure().safe
    game['source_outcome10'] = m.outcome_labels(game)
    rows = {}
    for r in game.itertuples(index=False):
        key = (GAME, _int(r.at_bat_number), _int(r.pitch_number))
        if key[1] is None or key[2] is None or min(key) <= 0:
            raise ValueError('invalid I3 physical key')
        rows[key] = r
    return rows


def _compare(counts, name, observed, expected):
    if observed is None:
        counts[name + '_unavailable'] += 1
    elif expected is None or (not isinstance(expected, tuple) and pd.isna(expected)):
        counts[name + '_reference_unavailable'] += 1
    else:
        counts[name + '_available'] += 1
        counts[name + ('_exact' if observed == expected else '_mismatch')] += 1


def paired(view, refs):
    """Exact equality only; unavailable values never become a mismatch or default."""
    counts = Counter()
    for key, row in view.items():
        target = refs.get(key)
        if target is None:
            counts['capture_only_key'] += 1
            continue
        counts['paired_key'] += 1
        direct = row['observed']
        count = row['pre_count']
        expected_count = (_int(target.balls), _int(target.strikes))
        _compare(counts, 'pre_count', tuple(count) if count is not None else None,
                 expected_count if None not in expected_count else None)
        counts['type_state_' + row['type_state']] += 1
        _compare(counts, 'event_type_code', direct['event_type_code'] if row['type_state'] == 'known' else None,
                 target.pitch_type)
        _compare(counts, 'description_hypothesis', row['description_hypothesis'], target.description)
        source_outcome = _int(target.source_outcome10)
        _compare(counts, 'outcome10_hypothesis', row['outcome10_hypothesis'],
                 source_outcome if source_outcome is not None and source_outcome >= 0 else None)
        counts['provider_id_available' if row['provider_id'] is not None else 'provider_id_unavailable'] += 1
        counts['call_code_available' if direct['event_call_code'] is not None else 'call_code_unavailable'] += 1
        counts['event_post_count_available' if direct['event_post_count'] is not None else 'event_post_count_unavailable'] += 1
        counts['event_post_outs_available' if direct['event_post_outs'] is not None else 'event_post_outs_unavailable'] += 1
        counts['trajectory_available' if direct['event_trajectory'] is not None else 'trajectory_unavailable'] += 1
        counts['reference_bb_type_available' if isinstance(target.bb_type, str) and target.bb_type not in ('', 'None', 'nan') else 'reference_bb_type_unavailable'] += 1
        counts['reference_pitcher_available' if _int(target.pitcher) is not None else 'reference_pitcher_unavailable'] += 1
        counts['reference_batter_available' if _int(target.batter) is not None else 'reference_batter_unavailable'] += 1
        for field in ('outs_when_up', 'home_score', 'away_score', 'stand', 'p_throws'):
            value = getattr(target, field)
            counts['reference_' + field + ('_unavailable' if pd.isna(value) else '_available')] += 1
        counts['pitch_level_identity_unavailable'] += 1
        counts['before_outs_score_unavailable'] += 1
        counts['bb_type_equivalence_unverified'] += 1
        # These values describe a completed play, not pitch-before state. Their
        # agreement is deliberately not compared against Statcast row values.
        for field in FINAL_FIELDS:
            counts[field + ('_available' if direct[field] is not None else '_unavailable')] += 1
        counts['accepted_false' if row['accepted'] is False else 'invalid_accepted_true'] += 1
        counts['reason_' + row['reason']] += 1
    counts['reference_only_key'] = len(set(refs) - set(view))
    return dict(sorted(counts.items()))


def receipts(reg):
    files = replay.entries(reg)
    first, latest, final = {}, {}, {}
    first_provider, provider_key, conflicting = {}, {}, set()
    counts = Counter(); compressed = inflated = 0
    for item in files:
        blob = replay.exact_file(Path(reg['capture_root']) / item['file'], replay.MAX_COMPRESSED)
        compressed += len(blob)
        if compressed > replay.MAX_COMPRESSED: raise ValueError('compressed total cap')
        body = replay.inflate_one(blob, replay.MAX_MEMBER)
        inflated += len(body)
        if inflated > replay.MAX_INFLATED: raise ValueError('inflated total cap')
        if hashlib.sha256(body).hexdigest() != item['sha256']:
            raise ValueError('canonical inflated body digest mismatch')
        feed = replay.feed_identity(body)
        observed = captured_diagnostic.inspect_body(feed, game_pk=GAME, game_date=DATE)
        group = {}
        duplicate = set()
        for row in observed:
            key = row['key']
            if key is None:
                counts['unkeyed_observation'] += 1
            elif key in group:
                duplicate.add(key)
            else:
                group[key] = row
        for key in duplicate:
            group.pop(key)
        counts['duplicate_key_in_body'] += len(duplicate)
        for key, row in group.items():
            provider = row['provider_id']
            if provider is not None:
                earlier = first_provider.get(key)
                other_key = provider_key.get(provider)
                if earlier is not None and earlier != provider:
                    conflicting.add(key)
                if other_key is not None and other_key != key:
                    conflicting.update((key, other_key))
                first_provider.setdefault(key, provider)
                provider_key.setdefault(provider, key)
            first.setdefault(key, row)
            latest[key] = row
        if item['seq'] == replay.MAX_FILES:
            final = group
        counts['bodies'] += 1
    if (counts['bodies'] != replay.MAX_FILES or compressed != reg['capture_compressed_bytes_reported']
            or inflated != reg['capture_inflated_bytes_reported']):
        raise ValueError('capture total/receipt mismatch')
    counts['distinct_physical_keys_observed'] = len(first)
    counts['provider_conflict_keys_excluded'] = len(conflicting)
    first = {k: v for k, v in first.items() if k not in conflicting}
    latest = {k: v for k, v in latest.items() if k not in conflicting}
    final = {k: v for k, v in final.items() if k not in conflicting}
    counts['distinct_first_seen_keys'] = len(first)
    counts['distinct_latest_seen_keys'] = len(latest)
    counts['final_body_keys'] = len(final)
    counts['first_seen_missing_from_final_body'] = len(set(first) - set(final))
    counts['first_to_latest_changed_keys'] = sum(first[k] != latest[k] for k in first)
    counts['first_to_final_changed_keys'] = sum(first[k] != final[k] for k in first.keys() & final.keys())
    return first, final, dict(sorted(counts.items())), compressed, inflated


def run(regpath, grantpath):
    reg, recipe = gates(regpath, grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.alarm(reg['bounds']['wall_seconds'] + 15)
    if (reg['game_pk'], reg['game_date'], reg['columns'], reg['i3']['rows']) != (GAME, DATE, COLUMNS, ROWS):
        raise ValueError('fixed paired diagnostic scope')
    if (reg['bounds'] != BOUNDS or reg['capture_body_count'] != replay.MAX_FILES
            or reg['i3']['parquet_sha256'] != I3_SHA
            or reg['i3']['manifest_sha256'] != I3_MANIFEST_SHA
            or reg['i3']['version'] != 'd20261012-i2026-e20261011'):
        raise ValueError('fixed source/resource identity')
    if (reg['selection_rule'] != 'all distinct physical keys first seen in 427 receipt bodies versus keys in final body 427; cross-receipt provider-ID conflicts excluded and counted before pairing both to exact I3 game849809/officialOct11/L rows'
            or reg['comparison_rule'] != 'exact equality only for reconstructed pre_count, known direct event type code, source-derived nonBIP description and outcome10 hypotheses; missing is unavailable; play-final identity/outs/score/hands and trajectory/bb_type are availability-only; no tolerance widening'):
        raise ValueError('fixed pre-result selection/comparison rule')
    if (recipe['data_role'] != '2026 EXPOSED DEVELOPMENT' or reg['capture_root'] != reg['manifest']['path'].rsplit('/', 1)[0]):
        raise ValueError('role/capture root')
    source = read_source('I3', reg['i3'], COLUMNS, reg['bounds'])
    if (len(source) != ROWS or sha(reg['i3']['parquet']) != reg['i3']['parquet_sha256']
            or sha(reg['i3']['manifest']) != reg['i3']['manifest_sha256']):
        raise ValueError('I3 changed after read')
    refs = reference(source)
    first, final, capture_counts, compressed, inflated = receipts(reg)
    comparison = {'first_seen': paired(first, refs), 'final_body': paired(final, refs)}
    if any(c.get('invalid_accepted_true', 0) for c in comparison.values()):
        raise ValueError('diagnostic promoted an unverified row')
    result = {'status': 'COMPLETED', 'stage': 'CAPTURED_I3_PAIRED_FIELD_DIAGNOSTIC',
        'registration_sha256': sha(regpath), 'grant_sha256': sha(grantpath),
        'source_commit': reg['source_commit'], 'game_pk': GAME, 'official_date': DATE,
        'capture_receipts': capture_counts, 'compressed_bytes': compressed,
        'inflated_bytes': inflated, 'i3_source_rows_decoded': ROWS,
        'i3_game_rows': len(refs), 'i3_columns_decoded': COLUMNS,
        'comparison': comparison, 'comparison_semantics': reg['comparison_rule'],
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
