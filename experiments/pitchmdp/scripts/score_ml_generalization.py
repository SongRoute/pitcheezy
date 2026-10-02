"""Complete-family T2 analysis; six fixed hypotheses and four R96 contrasts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import numpy as np
import pandas as pd
from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_metrics import (validate_probabilities, prediction_metrics, pitch_losses,
    paired_game_comparison, holm_adjust, prediction_decision)
from pitchmdp.matrix_group_metrics import GROUPS, guardrails
from pitchmdp.matrix_panel import group_reporting_status
from run_ml_generalization import (AXES, REGIMES, SEEDS, MODES, config_check, identity, verify,
    chosen_comparison)
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, check_location
from run_sequence_calibration import fit_blend
from score_ml_matrix import archive, assert_aligned

TESTS = ['natural_candidate_control', 'batter_Z_candidate_control', 'batter_W_candidate_control',
         'batter_O_candidate_control', 'pitcher_W_minus_Z', 'pitcher_O_minus_Z']
SCORING = {'primary_tests': TESTS, 'holm_alpha': .05, 'bootstrap_draws': 10000,
    'bootstrap_seed': 20260924, 'minimum_games': 30, 'minimum_pitches': 500,
    'robustness_family_size': 96, 'robustness_draws': 100000}


def require_complete(output, prep, config):
    """No quality metric may run until all active folds, cells and seeds exist."""
    if set(prep['folds']) != set(AXES):
        raise ValueError('T2 fold family differs')
    missing = []
    for axis in AXES:
        spec = prep['folds'][axis]
        if not spec['active']:
            continue
        expected = ['G0-global'] if axis == 'pitcher' else [config['candidate'], config['control']]
        if spec['cells'] != expected:
            raise ValueError('T2 cell family differs')
        for seed in SEEDS:
            for unit in spec['units']:
                path = output / axis / 'fits' / f'seed{seed}' / unit / 'state.json'
                if not path.is_file(): missing.append(str(path))
            for cell in expected:
                path = output / axis / 'members' / cell / f'seed{seed}' / 'prediction_state.json'
                if not path.is_file(): missing.append(str(path))
    if missing:
        raise ValueError('T2 complete family required before scoring: ' + ', '.join(missing))


def validate_metadata(metadata, values, prefix='dev_'):
    keys = values[prefix + 'keys']
    if (keys.ndim != 2 or keys.shape[1] != 3 or len(np.unique(keys, axis=0)) != len(keys)
            or not np.array_equal(metadata[KEY].to_numpy(np.int64), keys)):
        raise ValueError('T2 metadata keys differ or repeat')
    for name in ('game_pk', 'pitcher', 'batter'):
        if not np.array_equal(metadata[name].to_numpy(), values[prefix + name]):
            raise ValueError('T2 metadata differs: ' + name)
    if not np.array_equal(keys[:, 0], values[prefix + 'game_pk']):
        raise ValueError('T2 game keys differ')
    for name in ('train_pitches', 'train_role', 'train_volume', 'seen_pitcher', 'seen_batter'):
        if 'selection_' + name not in metadata:
            raise ValueError('T2 selection and actual-TRAIN metadata must be separate')
    if not np.array_equal(metadata.seen_pitcher.to_numpy(), metadata.train_pitches.gt(0).to_numpy()):
        raise ValueError('T2 actual TRAIN counts and seen status disagree')
    if not np.array_equal(metadata.train_volume.eq('zero').to_numpy(), metadata.train_pitches.eq(0).to_numpy()):
        raise ValueError('T2 zero-TRAIN stratum differs')


def load_family(output, prep, config):
    """Check all identities, checkpoint dependencies and raw probabilities first."""
    require_complete(output, prep, config)
    inputs = {str(output / 'preparation.json'): hash_file(output / 'preparation.json')}
    datasets, costs = {}, {}
    prep_hash = inputs[str(output / 'preparation.json')]
    for axis in AXES:
        spec = prep['folds'][axis]
        if not spec['active']: continue
        directory = output / axis
        baseline = archive(directory / 'baseline_predictions.npz')
        assert_aligned(baseline, baseline)
        metadata = pd.read_parquet(directory / 'dev_metadata.parquet')
        validate_metadata(metadata, baseline)
        for split in ('blend', 'dev'):
            keys = pd.read_parquet(directory / spec['samples'][split]['path'])
            if not np.array_equal(keys[KEY].to_numpy(np.int64), baseline[split + '_keys']):
                raise ValueError('T2 sample keys differ from archived denominator')
        for name in ('baseline_predictions.npz', 'dev_metadata.parquet'):
            inputs[str(directory / name)] = hash_file(directory / name)
        fit_dependencies, costs[axis] = {}, {}
        for seed in SEEDS:
            fit_dependencies[seed] = {}
            for unit, record in spec['units'].items():
                folder = directory / 'fits' / f'seed{seed}' / unit
                state = read_json(folder / 'state.json')
                expected = {'preparation_sha256': prep_hash, 'axis': axis, 'seed': seed, 'unit': unit, 'spec': record}
                if state['identity'] != expected or set(state['artifact_hashes']) != {'model.pt', 'fit.json'}:
                    raise ValueError('T2 fit member identity/artifact family differs')
                assert_hashes(folder, state['artifact_hashes'])
                deps = {str(folder / 'state.json'): hash_file(folder / 'state.json'),
                        **{str(folder / name): digest for name, digest in state['artifact_hashes'].items()}}
                fit_dependencies[seed][unit] = deps
                inputs.update(deps)
                costs[axis][f'{unit}/seed{seed}'] = read_json(folder / 'fit.json')
        members = {}
        for cell in spec['cells']:
            members[cell] = []
            for seed in SEEDS:
                folder = directory / 'members' / cell / f'seed{seed}'
                state = read_json(folder / 'prediction_state.json')
                expected = {'preparation_sha256': prep_hash, 'axis': axis, 'cell': cell, 'seed': seed}
                dependencies = {name: digest for unit, record in spec['units'].items()
                    if record['mode'] in MODES[cell] for name, digest in fit_dependencies[seed][unit].items()}
                if state['identity'] != expected or state['dependencies'] != dependencies:
                    raise ValueError('T2 prediction identity or exact checkpoint dependencies differ')
                if set(state['artifact_hashes']) != {'predictions.npz', 'calibration.json', 'runtime.json'}:
                    raise ValueError('T2 prediction artifact family differs')
                assert_hashes(folder, state['artifact_hashes'])
                member = archive(folder / 'predictions.npz')
                for regime in REGIMES:
                    assert_aligned({**member, 'dev': member[regime], 'dev_raw': member[regime + '_raw']}, baseline)
                    for split in ('blend', 'dev'):
                        if not np.array_equal(member[split + '_batter'], baseline[split + '_batter']):
                            raise ValueError('T2 batter denominator differs')
                    if not np.array_equal(member[regime + '_delivery_level'], member['Z_delivery_level']):
                        raise ValueError('T2 adaptation changed delivery tiers')
                members[cell].append(member)
                for name in ('prediction_state.json', *state['artifact_hashes']):
                    inputs[str(folder / name)] = hash_file(folder / name)
        datasets[axis] = {'baseline': baseline, 'metadata': metadata, 'members': members}
    natural = archive(output / 'natural_matchup_predictions.npz')
    metadata = pd.read_parquet(output / 'natural_matchup_metadata.parquet')
    validate_metadata(metadata, natural, prefix='')
    validate_probabilities(natural['y'], natural['frequency'])
    for cell in (config['candidate'], config['control']):
        for kind in ('primary', 'calibrated', 'raw'):
            validate_probabilities(natural['y'], natural[cell + '_' + kind])
        seeds = natural[cell + '_seed_primary']
        if seeds.shape != (3, len(natural['y']), 10):
            raise ValueError('Natural-matchup seed family differs')
        for p in seeds: validate_probabilities(natural['y'], p)
    for name in ('natural_matchup_predictions.npz', 'natural_matchup_metadata.parquet', 'natural_matchup_audit.json'):
        inputs[str(output / name)] = hash_file(output / name)
    return datasets, natural, metadata, inputs, costs


def summarize_regimes(members, baseline):
    """Fit June weights once per model ensemble/seed and reuse in Z/W/O."""
    if len(members) != 3: raise ValueError('Exactly three ordered seeds required')
    blend = np.mean([m['blend'] for m in members], axis=0)
    selection = fit_blend(baseline['blend_y'], blend, baseline['blend'], 'log_loss')
    selections = [fit_blend(baseline['blend_y'], m['blend'], baseline['blend'], 'log_loss') for m in members]
    reports, predictions = {}, {}
    for regime in REGIMES:
        calibrated = np.mean([m[regime] for m in members], axis=0)
        raw = np.mean([m[regime + '_raw'] for m in members], axis=0)
        w = selection['model_weight']
        primary = w * calibrated + (1-w) * baseline['dev']
        seed_primary = np.stack([s['model_weight'] * m[regime] + (1-s['model_weight']) * baseline['dev']
                                 for s, m in zip(selections, members)])
        predictions[regime] = {'primary': primary, 'calibrated': calibrated, 'raw': raw, 'seed_primary': seed_primary}
        reports[regime] = {kind: prediction_metrics(baseline['dev_y'], p) for kind, p in predictions[regime].items() if kind != 'seed_primary'}
    return {'selection': selection, 'seed_selections': selections, 'regimes': reports}, predictions


def contrast(test, y=None, candidate=None, control=None, games=None, metadata=None, robustness=False):
    if y is None:
        return {'test': test, 'reporting': {'status': 'inactive_fold'}, 'paired': None, 'seed_deltas': None,
                'robustness': {'status': 'unconfirmed', 'family_size': 96, 'family_alpha': .05,
                    'per_bound_alpha': .05/96, 'reason': 'inactive fold',
                    'groups': {name: {'reporting': {'status': 'inactive_fold'}, 'paired': None, 'passed': None}
                               for name in GROUPS}} if robustness else None}
    gate = group_reporting_status(games, np.ones(len(y), dtype=bool))
    paired = paired_game_comparison(y, candidate['primary'], control['primary'], games) if gate['status'] == 'reporting_eligible' else None
    seeds = [float((pitch_losses(y, a) - pitch_losses(y, b))[:, 0].mean())
             for a, b in zip(candidate['seed_primary'], control['seed_primary'])] if paired else None
    return {'test': test, 'reporting': gate, 'paired': paired, 'seed_deltas': seeds,
        'robustness': guardrails(y, candidate['primary'], control['primary'], games, metadata,
                                 candidate_family_size=4, draws=100000) if robustness else None}


def finish_decisions(comparisons):
    if [c['test'] for c in comparisons] != TESTS: raise ValueError('T2 ordered six-test family differs')
    adjusted = holm_adjust([c['paired']['nll']['p_less'] if c['paired'] else None for c in comparisons])
    for i, (c, p) in enumerate(zip(comparisons, adjusted)):
        if not c['paired']:
            c['N'] = {'status': 'unmeasured', 'reason': 'Inactive or below 30-game/500-pitch reporting gate'}
        else:
            c['N'] = prediction_decision(c['paired'], p, c['seed_deltas'])
            if i >= 4:
                c['N']['status'] = 'history_adaptation_improvement' if c['N']['status'] == 'predictive_improvement' else c['N']['status']
                c['N']['architecture_improvement_claim'] = False
        c['holm_p_adjusted'] = p
    return adjusted


def score(config, local_path, output):
    if config.get('registration', {}).get('t2_scoring') != SCORING:
        raise ValueError('Exact T2 scoring preregistration required')
    check_location(read_json(local_path), output)
    validate_native_runtime()
    prep = verify(output, identity(config, local_path))
    if chosen_comparison(read_json(output / 'selection_analysis.json')) != prep['selection']:
        raise ValueError('Frozen G selection differs')
    destination = output / 'analysis'
    if destination.exists(): raise ValueError('Preserve previous T2 analysis')
    started = time.perf_counter()
    datasets, natural, natural_metadata, inputs, costs = load_family(output, prep, config)
    # All archives and identities have passed before the first DEV metric.
    reports, predictions, saved = {}, {}, {}
    for axis, dataset in datasets.items():
        reports[axis], predictions[axis] = {}, {}
        for cell, members in dataset['members'].items():
            reports[axis][cell], predictions[axis][cell] = summarize_regimes(members, dataset['baseline'])
            for regime, values in predictions[axis][cell].items():
                for kind, p in values.items(): saved[f'{axis}_{cell}_{regime}_{kind}'] = p
        for name in ('keys', 'y', 'game_pk', 'pitcher', 'batter'):
            saved[f'{axis}_{name}'] = dataset['baseline']['dev_' + name]
    natural_predictions = {cell: {kind: natural[cell + '_' + kind] for kind in ('primary', 'calibrated', 'raw', 'seed_primary')}
                           for cell in (config['candidate'], config['control'])}
    reports['natural'] = {cell: {kind: prediction_metrics(natural['y'], p) for kind, p in values.items() if kind != 'seed_primary'}
                          for cell, values in natural_predictions.items()}
    for name, value in natural.items(): saved['natural_' + name] = value
    comparisons = [contrast(TESTS[0], natural['y'], natural_predictions[config['candidate']], natural_predictions[config['control']],
                            natural['game_pk'], natural_metadata, robustness=True)]
    for regime, test in zip(REGIMES, TESTS[1:4]):
        if 'batter' in datasets:
            d = datasets['batter']; p = predictions['batter']
            comparisons.append(contrast(test, d['baseline']['dev_y'], p[config['candidate']][regime], p[config['control']][regime],
                                        d['baseline']['dev_game_pk'], d['metadata'], robustness=True))
        else: comparisons.append(contrast(test, robustness=True))
    for regime, test in zip(('W', 'O'), TESTS[4:]):
        if 'pitcher' in datasets:
            d = datasets['pitcher']; p = predictions['pitcher']['G0-global']
            comparisons.append(contrast(test, d['baseline']['dev_y'], p[regime], p['Z'], d['baseline']['dev_game_pk']))
        else: comparisons.append(contrast(test))
    adjusted = finish_decisions(comparisons)
    descriptive = {}
    if 'batter' in datasets:
        d = datasets['batter']
        for cell, regimes in predictions['batter'].items():
            descriptive[cell] = {regime: {'status': 'descriptive_only', **contrast(
                'descriptive_' + cell + '_' + regime, d['baseline']['dev_y'], regimes[regime], regimes['Z'],
                d['baseline']['dev_game_pk'])} for regime in ('W', 'O')}
    result = {'experiment_id': config['experiment_id'], 'selection': prep['selection'], 'selection_status': config['selection_status'],
        'reports': reports, 'comparisons': comparisons, 'batter_adaptation_descriptive': descriptive,
        'multiplicity': {**SCORING, 'p_adjusted': adjusted, 'missing_slots_retained': True},
        'folds': prep['folds'], 'natural_matchup': prep['natural_matchup'], 'costs': costs,
        'limits': ['Conditional bootstrap excludes fitting/calibration/selection uncertainty',
            'Same previously exposed Cpanel DEV: conditional followup, not independent confirmation',
            'Natural new matchups use frozen known-player predictors and are not strict zero-shot',
            'Pitcher common-global adaptation is not an architecture comparison; O retains unknown profile/routing',
            'O uses fixed first-two-game outcomes only; later scored outcomes never update heldout history statistics',
            'Missing R groups retain their bounds and leave robustness unconfirmed'],
        'policy_effect': None, 'independent_confirmation': None, 'input_hashes': inputs,
        'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat(),
        'scoring_sources': {str(p.relative_to(PROJECT)): hash_file(p) for p in (Path(__file__),
            PROJECT / 'pitchmdp/matrix_metrics.py', PROJECT / 'pitchmdp/matrix_group_metrics.py',
            PROJECT / 'scripts/score_ml_matrix.py', PROJECT / 'scripts/run_sequence_calibration.py')}}
    destination.mkdir()
    np.savez_compressed(destination / 'predictions.npz', **saved)
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
        'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--local-config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(config_check(read_json(a.config)), a.local_config, a.output.resolve())


if __name__ == '__main__': main()
