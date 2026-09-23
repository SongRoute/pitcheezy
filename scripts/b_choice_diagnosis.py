"""Frozen July Observer choice diagnosis; execute only after A's preregistration.

The checkpoint is PA keyed and immutable after creation. A partial run can be
resumed, but neither a partial checkpoint nor a displayed recommendation is a
diagnostic result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'experiments/pitchmdp'),
                str(ROOT/'experiments/pitchmdp/scripts'), str(ROOT/'apps/observer/backend')]

from a_small_eval import PA_KEY, cohort_pa_rows, prior_zone_inputs
from b_choice_metrics import action_accounting, concentration, stability


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def pitch_key(row):
    return (int(row.game_pk), int(row.at_bat_number), int(row.pitch_number))


def pa_key(rows):
    row = rows.iloc[0]
    return (int(row.game_pk), int(row.at_bat_number))


def request_for(rows, metadata, train_end):
    start = rows.iloc[0]
    batter = int(start.batter)
    snapshot = metadata['profiles'].get(str(batter), metadata['default_profile'])
    return {'inning': int(start.inning), 'topbot': str(start.inning_topbot),
            'outs': int(start.outs_when_up), 'bases': int(start.bases),
            'home_score': int(start.home_score), 'away_score': int(start.away_score),
            'balls': 0, 'strikes': 0, 'pitcher_id': int(start.pitcher),
            'batter_stand': str(start.stand), 'batter_id': batter,
            'date': str(pd.Timestamp(start.game_date).date()), 'top_k': 3,
            'batter_profile': {'rates': snapshot['rates'],
                               'reliabilities': snapshot['reliabilities'],
                               'as_of': train_end}}


def compact_evaluation(ev, rows, variant):
    result = {'variant': variant, 'status': ev.status, 'reason': ev.reason,
              'model_identity': ev.model_identity, 'model_sha256': ev.model_sha256,
              'value_spec_version': ev.value_spec_version,
              'baseline_policy_id': ev.baseline_policy_id,
              'actions': [{'pitch_type': a.pitch_type, 'zone_id': a.zone_id} for a in ev.actions],
              'pitches': []}
    for row in rows.itertuples(index=False):
        b, s = int(row.balls), int(row.strikes)
        q = None if ev.candidate_values is None else ev.candidate_values[b, s, 0]
        accounting = action_accounting(ev.actions, q if q is not None else [], row.pitch_type)
        result['pitches'].append({'pitch_key': list(pitch_key(row)),
            'pitcher': int(row.pitcher), 'balls': b, 'strikes': s,
            'actual_type': row.pitch_type, 'q_values': q.tolist() if q is not None else [],
            'q_by_action': {f'{a.pitch_type}|{a.zone_id}': float(q[i]) for i, a in enumerate(ev.actions)} if q is not None else {},
            **accounting})
    if ev.support_ess is not None:
        result['action_support'] = [
            {'pitch_type': a.pitch_type, 'zone_id': a.zone_id,
             'minimum_ess': float(ev.support_ess[:, :, i].min()),
             'minimum_mass': float(ev.kernel_mass[:, :, i].min())}
            for i, a in enumerate(ev.actions)]
    else:
        result['action_support'] = []
    return result


def target_cohort(frame, pitcher_ids):
    groups = [(key, rows) for key, rows in frame.groupby(PA_KEY, sort=False)
              if rows.pitcher.isin(pitcher_ids).any()]
    rows = pd.concat([g for _, g in groups], ignore_index=True) if groups else frame.iloc[:0]
    return rows, groups


def aggregate(baseline, full, eligible, pitcher_ids, repertoire):
    target, target_pas = target_cohort(full, pitcher_ids)
    eligible_keys = {pitch_key(row) for row in eligible.itertuples(index=False)}
    excluded = target.loc[[pitch_key(row) not in eligible_keys for row in target.itertuples(index=False)]]
    base_pas = [item for item in baseline if item['status'] == 'ready']
    rows = []
    action_support = defaultdict(lambda: [0, 0])
    for item in baseline:
        pitcher = item['pitches'][0]['pitcher']
        for typ in repertoire[str(pitcher)]['pitch_types']:
            for zone in ('low_left', 'low_middle', 'low_right', 'middle_left', 'middle_middle',
                         'middle_right', 'high_left', 'high_middle', 'high_right'):
                action_support[(pitcher, typ, zone)][1] += 1
        for action in item['action_support']:
            action_support[(pitcher, action['pitch_type'], action['zone_id'])][0] += 1
        for row in item['pitches']:
            rows.append(row | {'game_pk': row['pitch_key'][0],
                               'at_bat_number': row['pitch_key'][1]})
    supported_actual = [r for r in rows if r['actual_type_supported']]
    gaps = [r['gap_pp'] for r in supported_actual]
    return {'coverage': {'selected_games': int(full.game_pk.nunique()),
             'selected_full_game_pitches': len(full), 'target_pitcher_pas': len(target_pas),
             'target_pitcher_pitches': len(target), 'eligible_pas': int(eligible.groupby(PA_KEY).ngroups),
             'eligible_pitches': len(eligible), 'excluded_pas': len(target_pas)-int(eligible.groupby(PA_KEY).ngroups),
             'excluded_pitches': len(excluded), 'ready_pas': len(base_pas),
             'ready_pitches': sum(len(item['pitches']) for item in base_pas),
             'actual_type_supported_pitches': len(supported_actual),
             'conditional_actual_type_support_rate': len(supported_actual)/len(rows) if rows else None,
             'whole_target_actual_type_support_rate': len(supported_actual)/len(target) if len(target) else None},
            'actual_mix': {'target': dict(sorted(Counter(target.pitch_type.dropna()).items())),
                           'eligible': dict(sorted(Counter(eligible.pitch_type.dropna()).items())),
                           'excluded': dict(sorted(Counter(excluded.pitch_type.dropna()).items()))},
            'concentration': concentration(rows),
            'action_rank': {'supported_actual_count': len(supported_actual),
                            'rank_counts': dict(sorted(Counter(r['actual_best_rank'] for r in supported_actual).items())),
                            'mean_gap_pp': float(np.mean(gaps)) if gaps else None,
                            'median_gap_pp': float(np.median(gaps)) if gaps else None,
                            'unsupported_actual_count': len(rows)-len(supported_actual)},
            'type_zone_support': [{'pitcher': p, 'pitch_type': typ, 'zone_id': zone,
                                   'supported_pas': yes, 'eligible_pas': total,
                                   'rate': yes/total if total else None}
                                  for (p, typ, zone), (yes, total) in sorted(action_support.items())],
            'per_pitch': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    relative_cfg = args.config.resolve().relative_to(ROOT)
    committed = subprocess.run(['git', 'show', f'HEAD:{relative_cfg}'], cwd=ROOT,
                               capture_output=True, check=True).stdout
    if args.config.read_bytes() != committed:
        raise ValueError('Diagnosis config must match committed preregistration')
    manifest_path = ROOT / cfg['parent_manifest']
    manifest = json.loads(manifest_path.read_text())
    dest = Path(cfg['artifact_dir'])
    if not dest.is_relative_to(Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs')):
        raise ValueError('Diagnostic output must be isolated on T7 Shield')
    paths = {k: Path(manifest['dataset_paths'][k]) for k in ('train', 'dev')}
    for k, path in paths.items():
        if sha(path) != manifest['sha256'][f'{k}_full_games.parquet']:
            raise ValueError(f'Frozen {k} parquet hash mismatch')
    train, dev = pd.read_parquet(paths['train']), pd.read_parquet(paths['dev'])
    metadata = json.loads((Path(cfg['bundle_dir'])/'metadata.json').read_text())
    pitchers = list(map(int, cfg['pitchers']))
    eligible, reasons, cohort_pa_count = cohort_pa_rows(dev, pitchers, metadata['pitchers'])
    if len(eligible) != 563 or eligible.groupby(PA_KEY).ngroups != 151:
        raise ValueError('Frozen eligible row count mismatch')
    os.environ['PITCHEEZY_OBSERVER_RUN'] = str(dest)
    os.environ['PITCHEEZY_OBSERVER_RUNTIME'] = 'research'
    from observer_app.recommender import Recommender
    model = Recommender()
    checkpoint = dest/'checkpoint'
    checkpoint.mkdir(parents=True, exist_ok=True)
    for key, rows in eligible.groupby(PA_KEY, sort=True):
        rows = rows.sort_values('pitch_number')
        game, pa = pa_key(rows)
        output = checkpoint/f'{game}-{pa}.json'
        if output.exists():
            saved = json.loads(output.read_text())
            if saved['config_sha256'] != sha(args.config):
                raise ValueError('Checkpoint belongs to another config')
            continue
        pitcher = int(rows.iloc[0].pitcher)
        bounds, repertoire_counts = prior_zone_inputs(train, pitcher,
            metadata['pitchers'][str(pitcher)]['pitch_types'])
        pitch = {'request': request_for(rows, metadata, cfg['train_period'][1])}
        pitch['request']['balls'] = int(rows.iloc[0].balls)
        pitch['request']['strikes'] = int(rows.iloc[0].strikes)
        pa_input = {'zone_bounds': bounds, 'repertoire_counts': repertoire_counts}
        evaluations = []
        evaluations.append(compact_evaluation(model.evaluate_pre_pitch(pitch, pa_input), rows, 'ensemble'))
        original_models = model.engine.models
        try:
            for member in original_models:
                model.engine.models = [member]
                evaluations.append(compact_evaluation(model.evaluate_pre_pitch(pitch, pa_input), rows,
                    f'member_seed_{member.seed}'))
        finally:
            model.engine.models = original_models
        write_json(output, {'config_sha256': sha(args.config), 'pa_key': [game, pa],
                            'bounds': bounds, 'repertoire_counts': repertoire_counts,
                            'evaluations': evaluations})
        print(json.dumps({'completed_pa': [game, pa], 'count': len(list(checkpoint.glob('*.json')))}), flush=True)
    checkpoints = [json.loads(path.read_text()) for path in sorted(checkpoint.glob('*.json'))]
    if len(checkpoints) != eligible.groupby(PA_KEY).ngroups:
        raise ValueError('Incomplete PA checkpoints')
    baseline = [item['evaluations'][0] for item in checkpoints]
    aggregate_result = aggregate(baseline, dev, eligible, pitchers, metadata['pitchers'])
    aggregate_result['coverage']['cohort_pa_count_check'] = cohort_pa_count
    aggregate_result['coverage']['excluded_pa_reasons'] = reasons
    variants = sorted({v['variant'] for item in checkpoints for v in item['evaluations']} - {'ensemble'})
    if variants:
        by_variant = {v: {tuple(row['pitch_key']): row
                          for item in checkpoints for e in item['evaluations'] if e['variant'] == v
                          for row in e['pitches']} for v in variants}
        ensemble_rows = {tuple(row['pitch_key']): row for row in aggregate_result['per_pitch']}
        aggregate_result['member_sensitivity'] = stability(by_variant, ensemble_rows)
        aggregate_result['member_sensitivity']['scope'] = 'existing frozen trained-member sensitivity; not independent sampling seeds or full-pipeline retraining'
    aggregate_result['experiment_id'] = cfg['experiment_id']
    aggregate_result['interpretation'] = {'prediction_quality': 'separate existing A S1 NLL/Brier; no refit',
        'internal_we': 'frozen model Q values only, observational zone proxy',
        'policy_ope': None, 'causal_policy_value': None}
    aggregate_result['sha256'] = {'config': sha(args.config), 'parent_manifest': sha(manifest_path),
                                  'train_parquet': sha(paths['train']), 'dev_parquet': sha(paths['dev']),
                                  'bundle_manifest': sha(Path(cfg['bundle_dir'])/'bundle_manifest.json'),
                                  'script': sha(__file__)}
    write_json(dest/'results.json', aggregate_result)
    print(json.dumps({'result': str(dest/'results.json')}))


if __name__ == '__main__':
    main()
