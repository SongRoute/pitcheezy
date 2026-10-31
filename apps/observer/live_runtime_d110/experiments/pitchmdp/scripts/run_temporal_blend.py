"""Fresh rolling-origin, train-selected cohorts and matched neural/frequency blends."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
from pitchmdp.data import KEY, hash_file
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import prepare_frame, HistoryStore
from pitchmdp.sequence_model import SequenceContext, SequenceModel, classification_metrics
from pitchmdp.sequence_delivery import JointDelivery
from run_sequence_pilot import dump, arrays
from run_sequence_ablations import rows_hash
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, fit_temperature, temperature_predictions
from run_sequence_context_frequency import ContextFrequencyBaseline
from run_sequence_calibration import fit_blend
from run_sequence_frequency_blend import paired_fixed_predictors

SEEDS = [42, 43, 44, 45, 46]
KINDS = ['flatten_mlp', 'transformer']


def assign_fold(frame, year):
    if year not in (2024, 2025):
        raise ValueError('Only frozen 2024/2025 folds allowed')
    dates = pd.to_datetime(frame.game_date)
    result = frame.loc[dates.le(f'{year}-09-30')].copy().reset_index(drop=True)
    dates = pd.to_datetime(result.game_date)
    result['split'] = 'unused'
    periods = {'train': ('2023-05-15', f'{year}-04-30'),
               'earlystop': (f'{year}-05-01', f'{year}-05-15'),
               'temperature': (f'{year}-05-16', f'{year}-05-31'),
               'blend': (f'{year}-06-01', f'{year}-06-30'),
               'dev': (f'{year}-07-01', f'{year}-09-30')}
    for name, (start, end) in periods.items():
        result.loc[dates.between(start, end), 'split'] = name
    return result


def select_cohort(frame, count=6):
    train = frame.loc[frame.split.eq('train') & frame.pitcher.eq(frame.starter_pitcher)]
    ranking = train.groupby('pitcher').agg(outs=('outs_recorded', 'sum'),
                    starts=('game_pk', 'nunique'), ambiguous=('outs_ambiguous_extra', 'sum')).reset_index()
    ranking = ranking.sort_values(['outs', 'starts', 'pitcher'], ascending=[False, False, True])
    chosen = ranking.head(count)
    if len(chosen) != count:
        raise ValueError('Insufficient TRAIN starters')
    ids = chosen.pitcher.astype(int).tolist()
    records = []
    for row in chosen.itertuples(index=False):
        names = train.loc[train.pitcher.eq(row.pitcher), 'player_name']
        records.append({'pitcher': int(row.pitcher), 'name': str(names.iloc[-1]),
                        'train_outs': int(row.outs), 'train_starts': int(row.starts),
                        'ambiguous_outs': int(row.ambiguous)})
    outsiders = ranking.iloc[count:]
    stable = not len(outsiders) or int((chosen.outs-chosen.ambiguous).min()) > int(outsiders.outs.max())
    return ids, {'selection': 'TRAIN starter outs descending, TRAIN starts descending, ID ascending; no future filter',
                 'members': records, 'membership_stable_under_out_ambiguity': bool(stable)}


def samples_for(frame, ids):
    ok = eligible(frame)
    target = frame.pitcher.isin(ids) & frame.pitcher.eq(frame.starter_pitcher)
    full = frame.loc[frame.split.eq('train') & ok]
    train = pd.concat([full.sample(min(250000, len(full)), random_state=42),
                       full.loc[target.reindex(full.index)]]).drop_duplicates(KEY).sort_index()
    early = frame.loc[frame.split.eq('earlystop') & ok]
    early = early.sample(min(16000, len(early)), random_state=42).sort_index()
    groups = {'train': train, 'earlystop': early}
    for name in ('temperature', 'blend', 'dev'):
        groups[name] = frame.loc[frame.split.eq(name) & ok & target]
    if any(not len(part) for part in groups.values()):
        raise ValueError('Empty aggregate fold sample; no replacement cohort permitted')
    game_sets = [set(part.game_pk) for part in groups.values()]
    if any(left & right for i, left in enumerate(game_sets) for right in game_sets[i+1:]):
        raise ValueError('Fold fit/calibration/evaluation games overlap')
    return full, groups


def check_sources(output):
    frozen = json.loads((output/'source_hashes.json').read_text())
    for rel, digest in frozen.items():
        if hash_file(PROJECT/rel) != digest or hash_file(output/'source'/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)


def run_fold(raw, year, output):
    dest = output/str(year)
    dest.mkdir(exist_ok=True)
    if (dest/'results.json').exists():
        return json.loads((dest/'results.json').read_text())
    start = time.perf_counter()
    frame = assign_fold(raw, year)
    ids, cohort = select_cohort(frame)
    full, parts = samples_for(frame, ids)
    assert pd.to_datetime(full.game_date).max() <= pd.Timestamp(f'{year}-04-30')
    for member in cohort['members']:
        pid = member['pitcher']
        member['coverage'] = {}
        for name in ('train', 'earlystop', 'temperature', 'blend', 'dev'):
            available = frame.loc[frame.split.eq(name) & frame.pitcher.eq(pid) & frame.pitcher.eq(frame.starter_pitcher)]
            supported = available.loc[eligible(available)]
            member['coverage'][name] = {'raw_pitches': len(available), 'eligible_pitches': len(supported),
                'starts': int(available.game_pk.nunique()), 'eligible_starts': int(supported.game_pk.nunique()),
                'outs': int(available.outs_recorded.sum())}
    dump(dest/'cohort.json', cohort)
    data = {name: {'n': len(part), 'games': int(part.game_pk.nunique()), 'rows_sha256': rows_hash(part),
                    'date_min': str(part.game_date.min()), 'date_max': str(part.game_date.max())}
            for name, part in parts.items()}
    data['full_train'] = {'n': len(full), 'rows_sha256': rows_hash(full)}
    dump(dest/'samples.json', data)
    for name, part in parts.items():
        part[KEY+['game_date', 'pitcher']].to_parquet(dest/(name+'_keys.parquet'), index=False)
    store = HistoryStore.from_frame(frame)
    context = SequenceContext().fit(full)
    delivery = JointDelivery().fit(full, store.normalizer, draws=400, seed=42)
    baseline = ContextFrequencyBaseline(HierarchicalFrequencyBaseline().fit(full)).fit(full)
    dump(dest/'encoders.json', {'normalizer': store.normalizer.report(), 'context': context.report(),
                               'delivery': delivery.report, 'baseline': baseline.report})
    with (dest/'fitted_preprocessors.pkl').open('wb') as stream:
        pickle.dump({'context': context, 'delivery': delivery, 'baseline': baseline,
                     'normalizer': store.normalizer}, stream)
    ys = {name: outcome_labels(part) for name, part in parts.items()}
    btemp = fit_temperature(baseline.predict(parts['temperature']), ys['temperature'])
    bp = {name: temperature_predictions(baseline.predict(parts[name]), btemp['temperature']) for name in ('blend', 'dev')}
    dump(dest/'baseline_calibration.json', btemp)
    training, early = arrays(store, context, parts['train'].index.to_numpy()), arrays(store, context, parts['earlystop'].index.to_numpy())
    predictions = {}
    for kind in KINDS:
        per_seed = []
        for seed in SEEDS:
            name = f'{kind}_{seed}'
            archive = dest/(name+'_predictions.npz')
            modelpath = dest/(name+'.pt')
            identity_file = dest/(name+'_hashes.json')
            saved_identity = json.loads(identity_file.read_text()) if identity_file.exists() else {}
            if archive.exists():
                if saved_identity.get('predictions') != hash_file(archive) or saved_identity.get('model') != hash_file(modelpath):
                    raise ValueError('Completed model/prediction identity mismatch: '+name)
                with np.load(archive) as saved:
                    for part in ('blend', 'dev'):
                        np.testing.assert_array_equal(saved[part+'_keys'], parts[part][KEY].to_numpy())
                    per_seed.append({part: saved[part].copy() for part in ('blend', 'dev')})
                continue
            print('MODEL_START', year, name, flush=True)
            if modelpath.exists():
                if saved_identity.get('model') != hash_file(modelpath):
                    raise ValueError('Completed model identity mismatch: '+name)
                model = SequenceModel.load(modelpath)
            else:
                model = SequenceModel(kind, seed=seed, width=128).fit(training, ys['train'], early, ys['earlystop'],
                    epochs=30, patience=5, batch_size=1024, learning_rate=.0005,
                    checkpoint=dest/(name+'_checkpoint.pt'))
                delivery.calibrate(model, store, context, parts['temperature'].index.to_numpy(), ys['temperature'])
                model.save(modelpath)
                dump(identity_file, {'model': hash_file(modelpath)})
                dump(dest/(name+'_fit.json'), model.report)
            predicted = {}
            for part in ('blend', 'dev'):
                predicted[part] = delivery.predict(model, store, context, parts[part].index.to_numpy())
            np.savez_compressed(archive, **predicted,
                **{part+'_keys': parts[part][KEY].to_numpy() for part in ('blend', 'dev')})
            dump(identity_file, {'model': hash_file(modelpath), 'predictions': hash_file(archive)})
            per_seed.append(predicted)
            print('MODEL_DONE', year, name, model.report['seconds'], flush=True)
            del model
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        predictions[kind] = {part: np.stack([entry[part] for entry in per_seed]) for part in ('blend', 'dev')}
    # All architectures and seeds are completed before any evaluation metrics are inspected.
    selections, devp = {}, {'frequency': bp['dev']}
    for kind in KINDS:
        calensemble = predictions[kind]['blend'].mean(0)
        selections[kind] = fit_blend(ys['blend'], calensemble, bp['blend'], 'log_loss')
        w = selections[kind]['model_weight']
        devp[kind+'_ensemble'] = predictions[kind]['dev'].mean(0)
        devp[kind+'_blend'] = w*devp[kind+'_ensemble']+(1-w)*bp['dev']
    dump(dest/'blend_selection.json', selections)
    y, games = ys['dev'], parts['dev'].game_pk.to_numpy()
    metrics = {name: classification_metrics(y, p) for name, p in devp.items()}
    comparisons = [('flatten_mlp_blend', 'frequency'), ('transformer_blend', 'frequency'),
                   ('transformer_blend', 'flatten_mlp_blend'), ('transformer_ensemble', 'flatten_mlp_ensemble')]
    paired = {a+'_minus_'+b: paired_fixed_predictors(y, devp[a], devp[b], games) for a,b in comparisons}
    per_pitcher = {str(pid): {name: classification_metrics(y[parts['dev'].pitcher.eq(pid)], p[parts['dev'].pitcher.eq(pid)])
                             for name, p in devp.items()} for pid in ids}
    result = {'year': year, 'cohort': cohort, 'samples': data, 'baseline_calibration': btemp,
              'selections': selections, 'metrics': metrics, 'paired': paired, 'per_pitcher': per_pitcher,
              'seed_metrics': {kind: [classification_metrics(y, p) for p in predictions[kind]['dev']] for kind in KINDS},
              'seconds': time.perf_counter()-start}
    np.savez_compressed(dest/'predictions.npz', y=y, game_pk=games, pitch_keys=parts['dev'][KEY].to_numpy(),
                        pitcher=parts['dev'].pitcher.to_numpy(), **devp)
    np.savez_compressed(dest/'calibration_predictions.npz', y=ys['blend'], baseline=bp['blend'], pitch_keys=parts['blend'][KEY].to_numpy(),
                        game_pk=parts['blend'].game_pk.to_numpy(),
                        **{kind: predictions[kind]['blend'] for kind in KINDS})
    check_sources(output)
    dump(dest/'results.json', result)
    print('FOLD_DONE', year, {name: p['log_loss'] for name,p in metrics.items()}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root, output = Path(local['artifact_root']).resolve(), args.output.resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not output.is_relative_to(root) or output == root:
        raise ValueError('Mounted configured SSD required')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise ValueError('Use configured existing Python')
    if output.exists() and any(output.iterdir()) and not (output/'source_hashes.json').exists():
        raise ValueError('Existing nonempty output is not a temporal run')
    output.mkdir(parents=True, exist_ok=True)
    if not (output/'source_hashes.json').exists():
        sources = list((PROJECT/'pitchmdp').glob('*.py'))+list((PROJECT/'scripts').glob('*.py'))+[PROJECT/'docs/TEMPORAL_BLEND_PROTOCOL.md', PROJECT/'configs/local.json']
        hashes = {str(path.relative_to(PROJECT)): hash_file(path) for path in sources}
        for path in sources:
            target = output/'source'/path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        dump(output/'source_hashes.json', hashes)
        dump(output/'config.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'years': [2024, 2025],
             'seeds': SEEDS, 'kinds': KINDS, 'draws': 400, 'width': 128, 'sample_seed': 42,
             'protocol_sha256': hash_file(PROJECT/'docs/TEMPORAL_BLEND_PROTOCOL.md')})
    check_sources(output)
    print('RUN', output, flush=True)
    raw = add_batter_style_history(prepare_frame(local))
    identity = raw.attrs.get('sequence_data_identity', {})
    identity_path = output/'data_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Data identity differs from original temporal run')
    dump(identity_path, identity)
    results = {str(year): run_fold(raw, year, output) for year in (2024, 2025)}
    check_sources(output)
    dump(output/'results.json', {'folds': results, 'finished_utc': datetime.now(timezone.utc).isoformat()})
    hashes = {str(path.relative_to(output)): hash_file(path) for path in output.rglob('*')
              if path.is_file() and 'source' not in path.relative_to(output).parts and path.name != 'artifact_hashes.json'}
    dump(output/'artifact_hashes.json', hashes)

if __name__ == '__main__':
    main()
