"""Score the complete F1 batter-representation bridge (full - masked) on Cpanel.

One primary hypothesis (NLL full - masked), a 24-bound R guardrail family and
descriptive batter-volume summaries. Scores open only after all three masked
members exist; preserved G0 predictions are re-verified before use.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_bridge import BATTER_GROUPS, batter_groups
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_group_metrics import GROUPS, masks, bootstrap_difference
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison
from pitchmdp.matrix_panel import group_reporting_status
from run_ml_matrix import assert_hashes, check_location, heavy_lock
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_bridge import (SEEDS, PARENT_CELL, DECISION, ROBUSTNESS, BOOTSTRAP, LIMITS, config_check,
                           identity, verify, member_dir, member_identity, ledger_attempts, ledger_total,
                           active_elapsed, ledger_stage, check_registered_output,
                           verify_full_reconstruction)
from score_ml_matrix import archive, assert_aligned, summarize_cell

METRICS = ('nll', 'brier')


def bridge_decision(comparison, seed_deltas, rule=DECISION):
    """Single-hypothesis screen rule; full - masked, negative favors batter channels."""
    if len(seed_deltas) != len(SEEDS) or not np.isfinite(np.asarray(seed_deltas, float)).all():
        raise ValueError('Three finite paired seed differences required')
    if comparison.get('status') != 'measured' or comparison['nll']['p_less'] is None:
        return {'status': 'unmeasured', 'reason': 'paired inference unavailable'}
    nll, brier = comparison['nll'], comparison['brier']
    criteria = {'practical_improvement': nll['delta'] <= rule['delta_nll_max'],
                'paired_ci_below_zero': nll['ci95'][1] < rule['nll_ci95_upper_max'],
                'one_sided_p': nll['p_less'] <= rule['one_sided_p_max'],
                'brier_noninferior': brier['ci95'][1] <= rule['brier_ci95_upper_max'],
                'seed_direction_stable': sum(x < 0 for x in seed_deltas) >= rule['required_negative_seeds']}
    if all(criteria.values()):
        status = 'predictive_improvement'
    elif nll['ci95'][0] > 0 or brier['ci95'][0] > rule['brier_ci95_upper_max']:
        status = 'worse_or_guardrail_failure'
    else:
        status = 'inconclusive'
    reverse = nll['delta'] > 0
    return {'status': status, 'criteria': criteria, 'p_one_sided': nll['p_less'],
            'multiplicity': 'single primary hypothesis; no adjustment', 'seed_deltas': list(seed_deltas),
            'reverse_direction_point_estimate': reverse,
            'reverse_direction_note': ('Masked point estimate is better; no automatic promotion. '
                                       'Register any follow-up as a new development hypothesis.') if reverse else None,
            'stage': 'three-seed bridge screen; confirmation not run', 'policy_effect': None}


def robustness(labels, full, masked, games, metadata, rule=ROBUSTNESS):
    """Twelve registered groups x two metrics = 24 simultaneous upper bounds."""
    if len(GROUPS) != rule['groups']:
        raise ValueError('Registered R group list changed')
    delta = pitch_losses(labels, full) - pitch_losses(labels, masked)
    if len(metadata) != len(delta):
        raise ValueError('Aligned R metadata required')
    alpha = rule['family_alpha'] / (rule['groups'] * rule['metrics'] * rule['comparisons'])
    if rule['groups'] * rule['metrics'] * rule['comparisons'] != rule['family_size']:
        raise ValueError('R family size differs from registration')
    games = np.asarray(games)
    groups, slots = {}, []
    for name, mask in masks(metadata).items():
        gate = group_reporting_status(games, mask)
        measured = (bootstrap_difference(delta[mask], games[mask], upper_alpha=alpha, draws=rule['draws'],
                                         seed=rule['seed'])
                    if gate['status'] == 'reporting_eligible' else None)
        structural = name == 'volume_zero' and gate['n'] == 0
        entry = {'reporting': gate, 'paired': measured,
                 'structural_missing': structural,
                 'missing_reason': ('TRAIN-selected Cpanel contains no zero-TRAIN pitcher; structurally unobserved, '
                                    'not a pass') if structural else
                                   (None if measured else 'reporting gate not met; unconfirmed')}
        for j, metric in enumerate(METRICS):
            margin = rule['nll_margin'] if metric == 'nll' else rule['brier_margin']
            passed = None if measured is None else bool(measured['simultaneous_upper'][j] <= margin)
            entry[metric] = {'margin': margin, 'upper': None if measured is None else measured['simultaneous_upper'][j],
                             'passed': passed}
            slots.append({'group': name, 'metric': metric, 'passed': passed, 'structural_missing': structural})
        groups[name] = entry
    outcomes = [slot['passed'] for slot in slots]
    return {'groups': groups, 'slots': slots, 'family_size': len(slots), 'family_alpha': rule['family_alpha'],
            'per_bound_alpha': alpha, 'draws': rule['draws'], 'seed': rule['seed'],
            'status': 'failed' if any(v is False for v in outcomes) else
                      'unconfirmed' if any(v is None for v in outcomes) else 'passed',
            'direction': 'full - masked upper bound; full may not be worse than masked by more than the margin',
            'note': 'All 24 registered slots retained; unobserved groups remain unconfirmed.'}


def batter_descriptive(labels, full, masked, games, dev_batters, volume, bootstrap=BOOTSTRAP):
    groups, counts = batter_groups(dev_batters, volume['counts'], volume['q25'])
    delta = pitch_losses(labels, full) - pitch_losses(labels, masked)
    games = np.asarray(games)
    summary = {}
    for name in BATTER_GROUPS:
        mask = groups == name
        entry = {'n': int(mask.sum()), 'games': len(np.unique(games[mask])),
                 'batters': len(np.unique(np.asarray(dev_batters)[mask]))}
        if mask.any():
            entry['delta'] = {'nll': float(delta[mask, 0].mean()), 'brier': float(delta[mask, 1].mean())}
            entry['paired'] = paired_game_comparison(labels[mask], full[mask], masked[mask], games[mask],
                                                     draws=bootstrap['draws'], seed=bootstrap['seed'])
        else:
            entry['delta'], entry['paired'] = None, None
        summary[name] = entry
    per_batter = {}
    frame = pd.DataFrame({'batter': np.asarray(dev_batters, dtype=np.int64), 'game': games,
                          'nll': delta[:, 0], 'brier': delta[:, 1], 'group': groups, 'train': counts})
    for batter, part in frame.groupby('batter', sort=True):
        per_batter[str(int(batter))] = {'n': len(part), 'games': int(part.game.nunique()),
            'd100_train_pitches_as_batter': int(part.train.iloc[0]), 'volume_group': part.group.iloc[0],
            'delta_nll': float(part.nll.mean()), 'delta_brier': float(part.brier.mean())}
    return {'status': 'descriptive only; no hypothesis test', 'q25': volume['q25'],
            'q25_method': volume['method'], 'boundary': volume['boundary'],
            'positive_train_batters': volume['positive_batters'],
            'count_definition': 'D100 TRAIN supervised pitches per batter, not post-cutoff cumulative observations',
            'groups': summary, 'per_batter': per_batter}


def seed_deltas(labels, full_seeds, masked_seeds):
    if len(full_seeds) != len(SEEDS) or len(masked_seeds) != len(SEEDS):
        raise ValueError('Three seed-paired primary predictions required')
    return [float((pitch_losses(labels, a) - pitch_losses(labels, b))[:, 0].mean())
            for a, b in zip(full_seeds, masked_seeds)]


def analyze(full_members, masked_members, baseline, metadata, volume, stored, stored_report):
    """Pure scoring core used by the CLI and synthetic tests."""
    full_report, full = verify_full_reconstruction(full_members, baseline, stored, stored_report)
    for member in masked_members:
        assert_aligned(member, baseline)
    masked_report, masked = summarize_cell(masked_members, baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), baseline['dev_keys']):
        raise ValueError('Unpaired DEV metadata')
    primary = paired_game_comparison(y, full['primary'], masked['primary'], games,
                                     draws=BOOTSTRAP['draws'], seed=BOOTSTRAP['seed'])
    deltas = seed_deltas(y, full['seed_primary'], masked['seed_primary'])
    return {'reports': {'full': full_report, 'masked': masked_report},
            'primary': {'candidate': 'full', 'control': 'masked', 'paired': primary,
                        'decision': bridge_decision(primary, deltas)},
            'robustness': robustness(y, full['primary'], masked['primary'], games, metadata),
            'batter_volume': batter_descriptive(y, full['primary'], masked['primary'], games,
                                                metadata.batter.to_numpy(), volume)}, full, masked


def costs(prep, output, fits):
    """Preserved logical cost vs actual new ledger spending (score run itself excluded)."""
    attempts = ledger_attempts(output)
    _, active = active_elapsed(output)
    closed = [a for a in attempts if a['id'] != active]
    return {'preserved_full_logical': {'fit_seconds': [m['logical_fit_seconds'] for m in prep['full_reuse']['members']],
                                       'prediction_seconds': [m['logical_prediction_seconds'] for m in prep['full_reuse']['members']],
                                       'spent_in_this_family': 0,
                                       'note': 'Logical training cost of reused G0 members, not new spending'},
            'new_spending': {'ledger_seconds_total': ledger_total(output, exclude=active),
                             'charging_rule': 'unterminated attempts charged up to the next start, capped at the stage timeout',
                             'by_stage_terminated_seconds': {stage: float(sum(a['seconds'] or 0. for a in closed if a['stage'] == stage))
                                          for stage in sorted({a['stage'] for a in closed})},
                             'non_completed_attempts': [a for a in closed if a['status'] != 'completed'],
                             'masked_fit_seconds': fits,
                             'family_budget_seconds': LIMITS['family_wall_budget_seconds'],
                             'new_fits': len(SEEDS), 'preserved_fits': len(SEEDS)}}


def _freeze(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def score(config, local_path, output):
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    missing = [seed for seed in SEEDS if not (member_dir(output, seed) / 'prediction_state.json').is_file()]
    if missing:
        raise ValueError('F1 masked family incomplete; scores remain unopened: seeds ' + str(missing))
    destination = output / 'analysis' / 'bridge'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted F1 scoring output')
    started = time.perf_counter()
    inputs = {str(output / name): hash_file(output / name) for name in
              ('preparation.json', 'baseline_predictions.npz', 'dev_metadata.parquet', 'batter_train_volume.json')}
    inputs.update(prep['external_hashes'])
    masked_members, fits = [], []
    for seed in SEEDS:
        dest = member_dir(output, seed)
        fitted = read_json(dest / 'fit_state.json')
        if fitted['identity'] != member_identity(prep, seed):
            raise ValueError('F1 fit identity differs')
        assert_hashes(dest, fitted['artifact_hashes'])
        state = read_json(dest / 'prediction_state.json')
        if state['identity'] != member_identity(prep, seed) or state['fit_state_sha256'] != hash_file(dest / 'fit_state.json'):
            raise ValueError('F1 prediction identity differs')
        assert_hashes(dest, state['artifact_hashes'])
        for name in ('fit_state.json', 'prediction_state.json', *fitted['artifact_hashes'], *state['artifact_hashes']):
            inputs[str(dest / name)] = hash_file(dest / name)
        masked_members.append(archive(dest / 'predictions.npz'))
        fits.append(read_json(dest / 'fit.json')['seconds_total'])
    full_members = [archive(Path(m['member_dir']) / 'predictions.npz') for m in prep['full_reuse']['members']]
    parent = Path(prep['parent_run'])
    stored = archive(parent / 'analysis' / 'panel' / 'predictions.npz')
    stored_report = read_json(parent / 'analysis' / 'panel' / 'results.json')['reports'][PARENT_CELL]
    baseline = archive(output / 'baseline_predictions.npz')
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    volume = read_json(output / 'batter_train_volume.json')
    core, full, masked = analyze(full_members, masked_members, baseline, metadata, volume, stored, stored_report)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    result = {'experiment_id': config['experiment_id'], 'scope': 'Cpanel', 'n': len(y),
              'games': len(np.unique(games)), 'comparison': 'full G0-global minus batter-masked G0-global',
              'frequency': prediction_metrics(y, baseline['dev']), **core,
              'costs': costs(prep, output, fits), 'input_hashes': inputs,
              'config_sha256': canonical_hash(config),
              'policy_effect': None, 'whole_mlb_robustness': None, 'independent_confirmation': None,
              'f4_transfer_claim': None,
              'limits': ['Three-seed bridge screen on previously exposed Cpanel DEV; not a confirmation',
                         'Masked arm removes the 17 existing batter channels only; H5 history retains batter-related signal',
                         'Effective input capacity differs by design (information-removal control)',
                         'Conditional bootstrap excludes training/calibration/selection uncertainty',
                         'Unobserved R groups remain unconfirmed; no direct target-location or policy-value claim'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat(),
              'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in
                  (Path(__file__).resolve(), PROJECT / 'scripts/score_ml_matrix.py', PROJECT / 'pitchmdp/matrix_metrics.py',
                   PROJECT / 'pitchmdp/matrix_group_metrics.py', PROJECT / 'pitchmdp/matrix_bridge.py')}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{arm + '_' + name: value for arm, values in (('full', full), ('masked', masked)) for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
         'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    for name in ('predictions.npz', 'results.json', 'manifest.json'):
        _freeze(destination / name)
    print({'full': core['reports']['full']['primary']['log_loss'],
           'masked': core['reports']['masked']['primary']['log_loss'],
           'decision': core['primary']['decision']['status'], 'R': core['robustness']['status']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config, output = config_check(read_json(args.config)), args.output.resolve()
    check_registered_output(config, output)
    root = check_location(read_json(args.local_config), output)
    with heavy_lock(root):
        with ledger_stage(output, 'score'):
            score(config, args.local_config, output)


if __name__ == '__main__':
    main()
