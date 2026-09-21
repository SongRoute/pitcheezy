"""Independent CPU audit of representation/history data, calibration and contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from audit_temporal_blend import (KEY, PROCESSED_COLUMNS, compare_scores, digest, independent_labels,
    key_digest, probabilities, read_json, real_partitions, require, scores, verify_hashes)

SEEDS = [42, 43, 44, 45, 46]
BATTER = ['hand', 'id', 'continuous', 'clusters_3', 'clusters_5', 'clusters_10', 'clusters_20']
HISTORY = ['history_1', 'history_2', 'history_3', 'history_4']
ALL_VARIANTS = BATTER+HISTORY
EXTRA_COLUMNS = ['batter', 'outs_when_up', 'inning', 'bases', 'home_score', 'away_score',
                 'inning_topbot', 'stand', 'p_throws']


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def expected_configs():
    result = [dict(name=mode, mode=mode, history=5) for mode in ('hand', 'id', 'continuous')]
    result += [dict(name=f'clusters_{k}', mode='clusters', clusters=k, history=5) for k in (3, 5, 10, 20)]
    result += [dict(name=f'history_{h}', mode='reference', history=h) for h in range(1, 5)]
    return result


def delivery_levels(full, query):
    tiers = [('pitch_type', 'p_throws', 'stand'), ('pitch_type', 'p_throws', 'stand', 'balls', 'strikes'),
             ('pitcher', 'pitch_type', 'p_throws', 'stand'),
             ('pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes')]
    levels = np.full(len(query), -1, dtype=np.int64)
    for level, keys in enumerate(tiers):
        counts = full.groupby(list(keys), observed=True).size()
        qualifying = counts[counts >= (20 if 'balls' in keys else 50)].index
        matches = pd.MultiIndex.from_frame(query[list(keys)]).isin(qualifying)
        levels[matches] = level
    return levels


def row_scores(y, probability):
    p = np.asarray(probability, dtype=np.float64)
    return {'log_loss': -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)),
            'brier_multiclass': np.square(p-np.eye(10)[y]).sum(1)}


def check_temperature(root, seed, labels, expected_levels, report):
    logits = np.load(root/f'seed{seed}_temperature_logits.npy', allow_pickle=False, mmap_mode='r')
    require(logits.shape == (len(labels), 400, 10) and np.isfinite(logits).all(),
            'Temperature logits have wrong row/draw/class dimensions or invalid values')
    np.testing.assert_array_equal(np.load(root/f'seed{seed}_temperature_levels.npy', allow_pickle=False), expected_levels)
    def objective(temperature):
        # SciPy passes NumPy float64 scalars while optimizing. Preserve that
        # arithmetic when evaluating the JSON's Python-float temperature too.
        integrated = softmax(logits/np.float64(temperature), axis=-1).mean(axis=1)
        return float(-np.log(np.clip(integrated[np.arange(len(labels)), labels], 1e-12, 1)).mean())
    fitted = minimize_scalar(objective, bounds=(.5, 2.5), method='bounded')
    temperature = report['delivery_temperature']
    require(.5 <= temperature <= 2.5, 'Temperature outside frozen bounds')
    np.testing.assert_allclose(temperature, float(fitted.x), rtol=0, atol=5e-5,
                               err_msg='Temperature does not reproduce from archived CAL logits')
    np.testing.assert_allclose(report['calibration_integrated_log_loss'], objective(temperature), rtol=1e-7, atol=1e-8)
    require(objective(temperature) <= float(fitted.fun)+2e-7, 'Temperature CAL objective exceeds optimum')


def check_variant(run, base, year, name, full, parts, subgroup_masks, expected_levels):
    root = run/str(year)/name
    result = read_json(root/'results.json')
    cfg = next(item for item in expected_configs() if item['name'] == name)
    require(result['config'] == cfg, 'Variant configuration differs')
    context_report = read_json(root/'context.json')
    require(context_report == result['context'], 'Context result differs from context manifest')
    require(context_report['mode'] == cfg['mode'] and context_report['base_channels'] == 11 and
            context_report['game_count_handedness_preserved'], 'Mandatory context channels differ')
    require(context_report['fit_rows'] == len(full), 'Context TRAIN row count differs')
    require(pd.Timestamp(context_report['fit_date_max']) <= pd.Timestamp(f'{year}-04-30'), 'Future context fit date')
    with (root/'context.pkl').open('rb') as stream:
        context = pickle.load(stream)
    require(json.loads(json.dumps(context.report())) == context_report, 'Serialized context differs from report')
    if name == 'id':
        expected_ids = sorted(int(value) for value in parts['train'].batter.dropna().unique())
        expected_map = {value: index+1 for index, value in enumerate(expected_ids)}
        require(context.id_map == expected_map and context.vocab_size == len(expected_ids)+1,
                'ID vocabulary differs from shared neural TRAIN only')
        require(context_report['unknown_index'] == 0 and context_report['embedding_width'] == 16,
                'Unknown ID or embedding dimensions differ')
        query = parts['dev'].iloc[:2].copy()
        query['batter'] = max(expected_ids)+1000000
        np.testing.assert_array_equal(context.transform(query)[:, -1], np.zeros(len(query)))
    if cfg['mode'] == 'clusters':
        geometry = context_report['archetypes']
        require(geometry['n_clusters'] == cfg['clusters'] and geometry['seed'] == 42,
                'Cluster granularity or fit seed differs')
        require(pd.Timestamp(geometry['fit_date_max']) <= pd.Timestamp(f'{year}-04-30'), 'Future cluster geometry')
        require(context_report['n_context'] == 11+cfg['clusters'], 'Cluster-only feature width differs')
    elif cfg['mode'] != 'id':
        require(context_report['n_context'] == {'hand': 11, 'continuous': 23, 'reference': 28}[cfg['mode']],
                'Representation feature width differs')

    baseline = load_arrays(base/str(year)/'predictions.npz')['frequency']
    baseline_cal = load_arrays(base/str(year)/'calibration_predictions.npz')['baseline']
    y = {part: independent_labels(parts[part]) for part in ('temperature', 'blend', 'dev')}
    seed_predictions = []
    parameter_counts = []
    for i, seed in enumerate(SEEDS):
        identity = read_json(root/f'seed{seed}_hashes.json')
        for key, filename in [('model', f'seed{seed}.pt'), ('predictions', f'seed{seed}_predictions.npz'),
                              ('temperature_logits', f'seed{seed}_temperature_logits.npy')]:
            require(digest(root/filename) == identity[key], f'Member identity differs: {name}, {seed}, {key}')
        report = read_json(root/f'seed{seed}_fit.json')
        require(report['seed'] == seed and report['training_rows'] == len(parts['train']) and
                report['calibration_rows'] == len(parts['earlystop']) and
                report['delivery_calibration_rows'] == len(parts['temperature']), 'Member fit/sample identity differs')
        network = report['network']
        require(network['width'] == 128 and network['length'] == 6 and network['n_physical'] == 8 and
                network['n_classes'] == 10, 'Physical model shape differs')
        checkpoint = torch.load(root/f'seed{seed}.pt', map_location='cpu', weights_only=False)
        require(checkpoint['network_config'] == network and checkpoint['report'] == report,
                'Checkpoint configuration or fit report differs')
        require(sum(value.numel() for value in checkpoint['state_dict'].values()) == report['parameter_count'],
                'Reported parameter count differs from saved MLP/embedding tensors')
        physical_key = ('backbone.' if name == 'id' else '')+'sequence_path.0.weight'
        require(tuple(checkpoint['state_dict'][physical_key].shape) == (256, 54),
                'Saved physical MLP no longer consumes the same six token/mask slots')
        np.testing.assert_allclose(checkpoint['delivery_temperature'], report['delivery_temperature'], rtol=0, atol=0)
        parameter_counts.append(report['parameter_count'])
        if cfg['mode'] == 'reference':
            original = read_json(base/str(year)/f'flatten_mlp_{seed}_fit.json')
            require(network == original['network'] and report['parameter_count'] == original['parameter_count'],
                    'History intervention changed architecture or parameter count')
        elif name == 'id':
            require(network['kind'] == 'id_mlp' and network['vocab_size'] == context.vocab_size and network['embedding_dim'] == 16,
                    'ID network embedding identity differs')
            embedding = checkpoint['state_dict']['embedding.weight']
            require(tuple(embedding.shape) == (context.vocab_size, 16), 'ID embedding table shape differs')
            np.testing.assert_array_equal(embedding[0].numpy(), np.zeros(16))
        else:
            require(network['kind'] == 'flatten_mlp' and network['n_context'] == context_report['n_context'],
                    'Neural input differs from declared numeric representation')
        del checkpoint
        check_temperature(root, seed, y['temperature'], expected_levels, report)
        prediction = load_arrays(root/f'seed{seed}_predictions.npz')
        for part in ('blend', 'dev'):
            np.testing.assert_array_equal(prediction[part+'_keys'], parts[part][KEY].to_numpy())
            probabilities(prediction[part], len(parts[part]))
        compare_scores(y['dev'], prediction['dev'], result['seed_metrics'][i])
        seed_predictions.append(prediction)
    require(len(set(parameter_counts)) == 1, 'Parameter count varies across seeds')
    ensemble = {part: np.stack([p[part] for p in seed_predictions]).mean(0) for part in ('blend', 'dev')}
    saved = load_arrays(root/'ensemble_predictions.npz')
    for part in ('blend', 'dev'):
        np.testing.assert_array_equal(saved[part], ensemble[part])
    selection = result['selection']
    weight = selection['model_weight']
    require(0 <= weight <= 1 and selection['calibration_objective'] == 'log_loss', 'Invalid blend weight/objective')
    np.testing.assert_allclose(selection['baseline_weight'], 1-weight, rtol=0, atol=1e-12)
    def objective(w):
        probability = w*ensemble['blend'].astype(float)+(1-w)*baseline_cal.astype(float)
        return scores(y['blend'], probability)['log_loss']
    optimum = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    require(objective(weight) <= min(objective(0), objective(1), float(optimum.fun))+1e-9,
            'Blend is not optimal on June calibration data')
    np.testing.assert_allclose(selection['calibration_score'], objective(weight), rtol=1e-8, atol=1e-10)
    mixed = weight*ensemble['dev']+(1-weight)*baseline
    np.testing.assert_array_equal(saved['mixed_dev'], mixed)
    compare_scores(y['dev'], mixed, result['metrics'])
    compare_scores(y['dev'], ensemble['dev'], result['ensemble_metrics'])
    for group, mask in subgroup_masks.items():
        compare_scores(y['dev'][mask], mixed[mask], result['subgroups'][group])
    return result, mixed, ensemble['dev'], parameter_counts[0]


def check_family(y, predicted, reference, games, names, actual):
    require(actual['contrasts'] == names and actual['replicates'] == 2000, 'Contrast family differs')
    _, inverse = np.unique(games, return_inverse=True)
    counts = np.bincount(inverse)
    require(actual['games'] == len(counts), 'Family game count differs')
    draws = np.random.default_rng(42).integers(len(counts), size=(2000, len(counts)))
    denominator = counts[draws].sum(axis=1)
    ref = row_scores(y, reference)
    for metric in ('log_loss', 'brier_multiclass'):
        estimates, means = [], []
        for name in names:
            delta = row_scores(y, predicted[name])[metric]-ref[metric]
            sums = np.bincount(inverse, weights=delta)
            estimates.append(sums[draws].sum(axis=1)/denominator)
            means.append(delta.mean())
        estimates, means = np.asarray(estimates), np.asarray(means)
        radius = np.quantile(np.max(np.abs(estimates-means[:, None]), axis=0), .95)
        for index, name in enumerate(names):
            expected = {'variant_minus_reference': means[index],
                        'pointwise95': np.quantile(estimates[index], [.025, .975]),
                        'simultaneous95': [means[index]-radius, means[index]+radius]}
            for key, value in expected.items():
                np.testing.assert_allclose(actual['metrics'][metric][name][key], value, rtol=1e-7, atol=1e-10)


def audit_year(run, base, year, processed, only_variant):
    root = run/str(year)
    selected, full, parts = real_partitions(processed, year)
    data = read_json(root/'data.json')
    require([r['pitcher'] for r in data['cohort']['members']] == selected.pitcher.astype(int).tolist(),
            'Cohort differs from independent TRAIN ranking')
    require(data['samples'] == read_json(base/str(year)/'samples.json'), 'Frozen sample manifest differs')
    for part, frame in parts.items():
        saved = pd.read_parquet(root/(part+'_keys.parquet'))
        np.testing.assert_array_equal(saved[KEY+['pitcher', 'batter']].to_numpy(), frame[KEY+['pitcher', 'batter']].to_numpy())
        require(key_digest(frame[KEY]) == data['samples'][part]['rows_sha256'], 'Real eligible key coverage differs')
    train_counts = parts['train'].groupby('batter').size()
    exposure = parts['dev'].batter.map(train_counts).fillna(0).to_numpy(int)
    unknown = exposure == 0
    for part, frame in parts.items():
        require(data['unseen'][part] == {'n': int((~frame.batter.isin(train_counts.index)).sum()), 'total': len(frame)},
                'Unseen-ID coverage differs')
    # In the complete pitch log, observed history is bounded by PA row position.
    history_frame = processed.loc[processed.game_date.le(f'{year}-09-30')].sort_values(['game_date', *KEY]).reset_index(drop=True)
    available = history_frame.groupby(['game_pk', 'at_bat_number'], sort=False).cumcount().clip(upper=5)
    history = available.iloc[parts['dev'].index].to_numpy()
    masks = {'known_id': ~unknown, 'unseen_id': unknown, 'history_ge5': history >= 5,
             'train_exposure_1_99': (exposure > 0) & (exposure < 100), 'train_exposure_ge100': exposure >= 100}
    require(data['subgroup_sizes'] == {key: int(mask.sum()) for key, mask in masks.items()}, 'Subgroup sizes differ')
    require(data['history_available_counts'] == {str(h): int((history == h).sum()) for h in range(6)},
            'Observed within-PA history lengths differ')
    levels = delivery_levels(full, parts['temperature'])
    variants = [only_variant] if only_variant else ALL_VARIANTS
    results, predictions, ensembles, counts = {}, {}, {}, {}
    for name in variants:
        print(f'AUDIT {year} {name}', file=sys.stderr, flush=True)
        results[name], predictions[name], ensembles[name], counts[name] = check_variant(run, base, year, name, full, parts, masks, levels)
    if not only_variant:
        final = read_json(root/'results.json')
        require(final['year'] == year and final['variants'] == results, 'Fold summary differs from variant results')
        archive = load_arrays(root/'predictions.npz')
        old = load_arrays(base/str(year)/'predictions.npz')
        y = independent_labels(parts['dev'])
        np.testing.assert_array_equal(archive['y'], y)
        np.testing.assert_array_equal(archive['pitch_keys'], parts['dev'][KEY].to_numpy())
        np.testing.assert_array_equal(archive['game_pk'], parts['dev'].game_pk.to_numpy())
        for field, expected in [('batter', parts['dev'].batter.to_numpy()), ('unknown_id', unknown),
                                ('history_available', history), ('train_exposure', exposure),
                                ('reference', old['flatten_mlp_blend']), ('reference_ensemble', old['flatten_mlp_ensemble']),
                                ('frequency', old['frequency'])]:
            np.testing.assert_array_equal(archive[field], expected)
        for name in ALL_VARIANTS:
            np.testing.assert_array_equal(archive[name], predictions[name])
            np.testing.assert_array_equal(archive[name+'_ensemble'], ensembles[name])
        for name in ('reference', 'reference_ensemble', 'frequency'):
            compare_scores(y, archive[name], final['reference_metrics'][name])
        for group, mask in masks.items():
            compare_scores(y[mask], archive['reference'][mask], final['reference_subgroups'][group])
        for family, names in [('batter', BATTER), ('history', HISTORY)]:
            check_family(y, predictions, archive['reference'], archive['game_pk'], names, final['comparisons'][family])
    return {'year': year, 'status': 'passed', 'variants': variants, 'models': 5*len(variants),
            'parameter_counts': counts, 'family_intervals_checked': not bool(only_variant)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--year', type=int, choices=(2024, 2025))
    parser.add_argument('--variant', choices=ALL_VARIANTS)
    args = parser.parse_args()
    require(not args.variant or args.year is not None, '--variant requires --year')
    run = args.run.resolve()
    config = read_json(run/'config.json')
    require(config['experiment'] == 'representation_history' and config['variants'] == expected_configs() and
            config['seeds'] == SEEDS and config['years'] == [2024, 2025], 'Experiment protocol differs')
    base = Path(config['base'])
    require(digest(base/'artifact_hashes.json') == config['base_artifact_manifest_sha256'], 'Reference manifest changed')
    verify_hashes(base, read_json(base/'artifact_hashes.json'))
    for origin in (base, run):
        hashes = read_json(origin/'source_hashes.json')
        verify_hashes(origin/'source', hashes)
        verify_hashes(PROJECT, hashes)
    require(config['protocol_sha256'] == digest(run/'source/docs/REPRESENTATION_HISTORY_PROTOCOL.md'),
            'Representation protocol digest differs')
    identity = read_json(run/'data_identity.json')
    require(identity == read_json(base/'data_identity.json'), 'Data identity differs from reference')
    local = read_json(run/'source/configs/local.json')
    source = Path(local['artifact_root'])/'processed/pitches.parquet'
    require(digest(source) == identity['processed_sha256'], 'Processed data digest differs')
    processed = pd.read_parquet(source, columns=list(dict.fromkeys(PROCESSED_COLUMNS+EXTRA_COLUMNS)))
    processed['game_date'] = pd.to_datetime(processed.game_date)
    require(len(processed) == identity['rows'] and processed.game_date.dt.year.isin([2023, 2024, 2025]).all(),
            'Processed row count or years differ')
    global_manifest = run/'artifact_hashes.json'
    if global_manifest.exists():
        hashes = read_json(global_manifest)
        for year in (2024, 2025):
            require(f'{year}/results.json' in hashes, 'Completed fold result lacks final digest')
        verify_hashes(run, hashes)
    else:
        require(args.year is not None, 'Final hashes absent; use --year/--variant for partial audit')
    years = [args.year] if args.year else [2024, 2025]
    folds = [audit_year(run, base, year, processed, args.variant) for year in years]
    if args.year is None:
        merged = read_json(run/'results.json')['folds']
        for year in years:
            require(merged[str(year)] == read_json(run/str(year)/'results.json'), 'Combined summary differs')
    print(json.dumps({'status': 'passed', 'folds': folds, 'final_artifact_hashes_verified': global_manifest.exists(),
        'auditor_sha256': digest(Path(__file__)),
        'supporting_auditor_sha256': digest(Path(__file__).with_name('audit_temporal_blend.py')),
        'scope': 'Independent processed-data membership/coverage, TRAIN-only ID vocabulary and zero UNK, parameter dimensions, archived CAL-logit temperature optima, ensembles/June blends/metrics/family bootstrap; CPU only, no raw files or model inference.'}, indent=2))


if __name__ == '__main__':
    main()
