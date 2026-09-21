"""Independently audit saved temporal-blend predictions; no raw reads or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT = Path(__file__).resolve().parents[1]
KEY = ['game_pk', 'at_bat_number', 'pitch_number']
KINDS = ['flatten_mlp', 'transformer']
SEEDS = [42, 43, 44, 45, 46]
PROCESSED_COLUMNS = list(dict.fromkeys([*KEY, 'game_date', 'pitcher', 'starter_pitcher',
    'outs_recorded', 'outs_ambiguous_extra', 'description', 'events', 'supported_pa',
    'pitch_type', 'plate_x', 'plate_z', 'balls', 'strikes']))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def key_digest(keys):
    return hashlib.sha256(np.asarray(keys, dtype=np.int64).tobytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def probabilities(p, n):
    require(p.shape == (n, 10), 'Unexpected ten-class probability shape')
    require(np.isfinite(p).all() and (p >= 0).all(), 'Invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1., rtol=0, atol=1e-5)


def scores(y, p):
    if not len(y):
        return {'n': 0}
    p = np.asarray(p, dtype=np.float64)
    probabilities(p, len(y))
    return {'n': len(y),
            'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'brier_multiclass': float(np.square(p-np.eye(10)[y]).sum(1).mean()),
            'accuracy': float((p.argmax(1) == y).mean())}


def compare_scores(y, p, recorded):
    for name, value in scores(y, p).items():
        np.testing.assert_allclose(recorded[name], value, rtol=1e-7, atol=1e-9,
                                   err_msg='Metric differs: '+name)


def verify_hashes(root, manifest):
    for relative, expected in manifest.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()), 'Hash path escapes run directory')
        require(digest(path) == expected, 'Artifact digest differs: '+relative)


def independent_labels(frame):
    labels = np.full(len(frame), -1, dtype=np.int64)
    descriptions = {0: ('ball', 'blocked_ball', 'pitchout', 'intent_ball'),
                    1: ('called_strike', 'swinging_strike', 'swinging_strike_blocked',
                        'missed_bunt', 'foul_tip', 'bunt_foul_tip'),
                    2: ('foul', 'foul_bunt'), 8: ('hit_by_pitch',)}
    for label, names in descriptions.items():
        labels[frame.description.isin(names)] = label
    labels[frame.description.eq('foul_bunt') & frame.strikes.eq(2)] = 1
    inplay = {3: ('field_out', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'),
              4: ('single',), 5: ('double',), 6: ('triple',), 7: ('home_run',),
              9: ('double_play', 'grounded_into_double_play', 'sac_fly_double_play', 'sac_bunt_double_play')}
    for label, events in inplay.items():
        labels[frame.description.eq('hit_into_play') & frame.events.isin(events)] = label
    return labels


def real_partitions(processed, year):
    frame = processed.loc[processed.game_date.le(f'{year}-09-30')].copy()
    frame.sort_values(['game_date', *KEY], inplace=True, ignore_index=True)
    train_period = frame.game_date.between('2023-05-15', f'{year}-04-30')
    starts = frame.loc[train_period & frame.pitcher.eq(frame.starter_pitcher)]
    ranking = starts.groupby('pitcher').agg(outs=('outs_recorded', 'sum'),
        starts=('game_pk', 'nunique'), ambiguous=('outs_ambiguous_extra', 'sum')).reset_index()
    ranking = ranking.sort_values(['outs', 'starts', 'pitcher'], ascending=[False, False, True])
    require(len(ranking) >= 6, 'Too few observed TRAIN starters')
    selected = ranking.head(6)
    ids = selected.pitcher.astype(int).tolist()
    ok = ((independent_labels(frame) >= 0) & frame.supported_pa.fillna(False) &
          frame.pitch_type.notna() & frame.plate_x.notna() & frame.plate_z.notna() &
          frame.balls.between(0, 3) & frame.strikes.between(0, 2))
    target = frame.pitcher.isin(ids) & frame.pitcher.eq(frame.starter_pitcher)
    full = frame.loc[train_period & ok]
    parts = {'train': pd.concat([full.sample(min(250000, len(full)), random_state=42),
                                frame.loc[train_period & ok & target]]).drop_duplicates(KEY).sort_index()}
    early = frame.loc[frame.game_date.between(f'{year}-05-01', f'{year}-05-15') & ok]
    parts['earlystop'] = early.sample(min(16000, len(early)), random_state=42).sort_index()
    for name, start, end in (('temperature', '05-16', '05-31'), ('blend', '06-01', '06-30'),
                             ('dev', '07-01', '09-30')):
        parts[name] = frame.loc[frame.game_date.between(f'{year}-{start}', f'{year}-{end}') & ok & target]
    return selected, full, parts


def audit_fold(run, year, processed):
    root = run / str(year)
    result = read_json(root / 'results.json')
    require(result['year'] == year, 'Result year differs')
    sample = read_json(root / 'samples.json')
    require(result['samples'] == sample, 'Result sample manifest differs')
    cohort = read_json(root / 'cohort.json')
    require(result['cohort'] == cohort, 'Result cohort manifest differs')
    ids = [member['pitcher'] for member in cohort['members']]
    require(len(ids) == len(set(ids)) == 6, 'Expected six distinct TRAIN-selected IDs')
    real_cohort, real_full, real_parts = real_partitions(processed, year)
    require(ids == real_cohort.pitcher.astype(int).tolist(), 'Cohort differs from actual TRAIN-only ranking')
    require(len(real_full) == sample['full_train']['n'] and
            key_digest(real_full[KEY]) == sample['full_train']['rows_sha256'],
            'Full TRAIN pool differs from processed data')
    for declared, observed in zip(cohort['members'], real_cohort.itertuples(index=False)):
        require(declared['train_outs'] == observed.outs and declared['train_starts'] == observed.starts and
                declared['ambiguous_outs'] == observed.ambiguous, 'TRAIN workload differs from processed data')
    periods = {'train': ('2023-05-15', f'{year}-04-30'),
               'earlystop': (f'{year}-05-01', f'{year}-05-15'),
               'temperature': (f'{year}-05-16', f'{year}-05-31'),
               'blend': (f'{year}-06-01', f'{year}-06-30'),
               'dev': (f'{year}-07-01', f'{year}-09-30')}
    frames = {}
    for name, (start, end) in periods.items():
        frame = pd.read_parquet(root / (name+'_keys.parquet'))
        require(len(frame) > 0 and not frame.duplicated(KEY).any(), 'Empty or repeated keys: '+name)
        require(pd.to_datetime(frame.game_date).between(start, end).all(), 'Dates escaped partition: '+name)
        require(len(frame) == sample[name]['n'], 'Sample size differs: '+name)
        require(frame.game_pk.nunique() == sample[name]['games'], 'Game count differs: '+name)
        require(key_digest(frame[KEY]) == sample[name]['rows_sha256'], 'Ordered key digest differs: '+name)
        np.testing.assert_array_equal(frame[KEY].to_numpy(), real_parts[name][KEY].to_numpy(),
                                      err_msg='Actual eligible key coverage differs: '+name)
        np.testing.assert_array_equal(frame.pitcher.to_numpy(), real_parts[name].pitcher.to_numpy())
        if name in ('temperature', 'blend', 'dev'):
            require(frame.pitcher.isin(ids).all(), 'Noncohort target rows: '+name)
        for other in frames.values():
            require(set(frame.game_pk).isdisjoint(other.game_pk), 'Partition games overlap')
        frames[name] = frame

    encoder = read_json(root / 'encoders.json')
    for item in (encoder['normalizer'], encoder['context']['archetypes'],
                 encoder['baseline'], encoder['baseline']['parent_report']):
        require(pd.Timestamp(item['fit_date_max']) <= pd.Timestamp(f'{year}-04-30'),
                'Fitted preprocessing contains future dates')
    require(encoder['delivery']['draws'] == 400 and encoder['delivery']['seed'] == 42,
            'Shared delivery budget/seed differs')
    require(encoder['delivery']['training_rows'] == sample['full_train']['n'], 'Delivery TRAIN size differs')
    require(encoder['baseline']['training_rows'] == sample['full_train']['n'], 'Baseline TRAIN size differs')

    dev = arrays(root / 'predictions.npz')
    cal = arrays(root / 'calibration_predictions.npz')
    for saved, name in ((dev, 'dev'), (cal, 'blend')):
        np.testing.assert_array_equal(saved['pitch_keys'], frames[name][KEY].to_numpy())
        np.testing.assert_array_equal(saved['game_pk'], frames[name].game_pk.to_numpy())
        require(saved['y'].shape == (len(frames[name]),), 'Labels differ from key count')
        require(np.issubdtype(saved['y'].dtype, np.integer) and np.isin(saved['y'], range(10)).all(),
                'Labels outside ten-class contract')
        np.testing.assert_array_equal(saved['y'], independent_labels(real_parts[name]),
                                      err_msg='Labels differ from processed data')
    np.testing.assert_array_equal(dev['pitcher'], frames['dev'].pitcher.to_numpy())
    probabilities(cal['baseline'], len(cal['y']))
    probabilities(dev['frequency'], len(dev['y']))
    selections = read_json(root / 'blend_selection.json')
    require(selections == result['selections'], 'Saved blend selection differs from result')
    require(read_json(root / 'baseline_calibration.json') == result['baseline_calibration'],
            'Saved baseline temperature differs from result')
    for kind in KINDS:
        cal_seeds, dev_seeds = [], []
        for i, seed in enumerate(SEEDS):
            name = f'{kind}_{seed}'
            hashes = read_json(root / (name+'_hashes.json'))
            require(digest(root / (name+'.pt')) == hashes['model'], 'Model digest differs: '+name)
            require(digest(root / (name+'_predictions.npz')) == hashes['predictions'],
                    'Seed prediction digest differs: '+name)
            archive = arrays(root / (name+'_predictions.npz'))
            for split in ('blend', 'dev'):
                np.testing.assert_array_equal(archive[split+'_keys'], frames[split][KEY].to_numpy())
                probabilities(archive[split], len(frames[split]))
            cal_seeds.append(archive['blend'])
            dev_seeds.append(archive['dev'])
            compare_scores(dev['y'], archive['dev'], result['seed_metrics'][kind][i])
            fit = read_json(root / (name+'_fit.json'))
            require(fit['seed'] == seed and fit['kind'] == kind and fit['network']['width'] == 128,
                    'Model identity differs: '+name)
            require(fit['training_rows'] == sample['train']['n'] and
                    fit['calibration_rows'] == sample['earlystop']['n'] and
                    fit['delivery_calibration_rows'] == sample['temperature']['n'],
                    'Model fit sample counts differ: '+name)
        np.testing.assert_array_equal(cal[kind], np.stack(cal_seeds))
        cal_ensemble, dev_ensemble = np.mean(cal_seeds, axis=0), np.mean(dev_seeds, axis=0)
        np.testing.assert_allclose(dev[kind+'_ensemble'], dev_ensemble, rtol=1e-7, atol=1e-8)
        selection = selections[kind]
        weight = selection['model_weight']
        require(0 <= weight <= 1 and selection['calibration_objective'] == 'log_loss',
                'Invalid declared blend selection')
        np.testing.assert_allclose(selection['baseline_weight'], 1-weight, rtol=0, atol=1e-12)
        def loss(w):
            p = w*cal_ensemble.astype(np.float64)+(1-w)*cal['baseline'].astype(np.float64)
            return scores(cal['y'], p)['log_loss']
        optimum = minimize_scalar(loss, bounds=(0, 1), method='bounded')
        best_loss = min(loss(0), loss(1), loss(float(optimum.x)))
        require(loss(weight) <= best_loss+1e-9, 'June blend weight is not a CAL optimum')
        np.testing.assert_allclose(selection['calibration_score'], loss(weight), rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dev[kind+'_blend'], weight*dev_ensemble+(1-weight)*dev['frequency'],
                                   rtol=1e-7, atol=1e-8)
    for name in ('frequency', 'flatten_mlp_ensemble', 'flatten_mlp_blend',
                 'transformer_ensemble', 'transformer_blend'):
        compare_scores(dev['y'], dev[name], result['metrics'][name])
        for pid in ids:
            mask = dev['pitcher'] == pid
            compare_scores(dev['y'][mask], dev[name][mask], result['per_pitcher'][str(pid)][name])
    _, game_index = np.unique(dev['game_pk'], return_inverse=True)
    counts = np.bincount(game_index)
    draws = np.random.default_rng(42).integers(len(counts), size=(2000, len(counts)))
    comparisons = [('flatten_mlp_blend', 'frequency'), ('transformer_blend', 'frequency'),
                   ('transformer_blend', 'flatten_mlp_blend'), ('transformer_ensemble', 'flatten_mlp_ensemble')]
    for left, right in comparisons:
        entry = result['paired'][left+'_minus_'+right]
        require(entry['games'] == len(counts) and entry['replicates'] == 2000,
                'Paired bootstrap design differs')
        p, q, y = dev[left].astype(np.float64), dev[right].astype(np.float64), dev['y']
        row = np.arange(len(y))
        losses = {'log_loss': (-np.log(np.clip(p[row, y], 1e-12, 1)),
                               -np.log(np.clip(q[row, y], 1e-12, 1))),
                  'brier_multiclass': (np.square(p-np.eye(10)[y]).sum(1),
                                      np.square(q-np.eye(10)[y]).sum(1))}
        for metric, (a, b) in losses.items():
            delta = a-b
            sums = np.bincount(game_index, weights=delta)
            interval = np.quantile(sums[draws].sum(1)/counts[draws].sum(1), [.025, .975])
            expected = {'mean_model': a.mean(), 'mean_reference': b.mean(),
                        'model_minus_reference': delta.mean(), 'game_only_bootstrap95': interval}
            for key, value in expected.items():
                np.testing.assert_allclose(entry['metrics'][metric][key], value, rtol=1e-7, atol=1e-10)
    return {'year': year, 'status': 'passed', 'eval_rows': len(dev['y']),
            'models': len(KINDS)*len(SEEDS), 'cohort_ids': ids}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--year', type=int, choices=(2024, 2025),
                        help='Audit a completed fold while the other is still running')
    args = parser.parse_args()
    run = args.run.resolve()
    config = read_json(run / 'config.json')
    require(config['years'] == [2024, 2025] and config['seeds'] == SEEDS and
            config['kinds'] == KINDS and config['width'] == 128 and config['draws'] == 400,
            'Run configuration differs from frozen experiment')
    source_hashes = read_json(run / 'source_hashes.json')
    require('scripts/run_temporal_blend.py' in source_hashes, 'Not a temporal run')
    verify_hashes(run / 'source', source_hashes)
    verify_hashes(PROJECT, source_hashes)
    require(source_hashes['docs/TEMPORAL_BLEND_PROTOCOL.md'] == config['protocol_sha256'],
            'Protocol digest differs')
    artifact_path = run / 'artifact_hashes.json'
    if artifact_path.exists():
        artifact_hashes = read_json(artifact_path)
        for year in (2024, 2025):
            require(f'{year}/results.json' in artifact_hashes, 'Completed-fold result lacks final digest')
        verify_hashes(run, artifact_hashes)
    else:
        require(args.year is not None, 'Final artifact digests absent; use --year for partial audit')
    years = [args.year] if args.year is not None else [2024, 2025]
    local = read_json(run / 'source/configs/local.json')
    identity = read_json(run / 'data_identity.json')
    processed_path = Path(local['artifact_root']) / 'processed/pitches.parquet'
    require(digest(processed_path) == identity['processed_sha256'], 'Processed data identity differs')
    processed = pd.read_parquet(processed_path, columns=PROCESSED_COLUMNS)
    processed['game_date'] = pd.to_datetime(processed.game_date)
    require(len(processed) == identity['rows'] and processed.game_date.dt.year.isin([2023, 2024, 2025]).all(),
            'Processed row count or permitted seasons differ')
    folds = [audit_fold(run, year, processed) for year in years]
    if args.year is None:
        combined = read_json(run / 'results.json')['folds']
        for year in years:
            require(combined[str(year)] == read_json(run / str(year) / 'results.json'),
                    'Top-level and completed-fold results differ')
    print(json.dumps({'status': 'passed', 'folds': folds,
        'final_artifact_hashes_verified': artifact_path.exists(),
        'scope': 'Processed-data TRAIN ranking/eligible coverage/labels, saved-array arithmetic, identities, partitions and fit reports; no raw data, GPU, retraining, or causal validation.'}, indent=2))


if __name__ == '__main__':
    main()
