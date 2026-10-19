"""Run matched sequence experiments, then save bounded sequence/WE planning demos."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.model import CountBaseline, eligible, outcome_labels
from pitchmdp.sequence_data import prepare_frame, HistoryStore
from pitchmdp.sequence_model import SequenceContext, SequenceModel, classification_metrics
from pitchmdp.sequence_delivery import JointDelivery


def dump(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str)+'\n')
    temporary.replace(path)


def arrays(store, context, rows):
    tokens, valid = store.gather(np.asarray(rows))
    return tokens, valid, context.transform(store.frame.iloc[rows])


def paired(y, p, q, game_ids, replicates=2000):
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))+np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1))
    games, inverse = np.unique(game_ids, return_inverse=True)
    n, sums = np.bincount(inverse), np.bincount(inverse, weights=loss)
    ix = np.random.default_rng(42).integers(0, len(games), (replicates, len(games)))
    bootstrap = sums[ix].sum(1)/n[ix].sum(1)
    return {'model_minus_reference_log_loss': float(loss.mean()), 'bootstrap95': np.quantile(bootstrap, [.025,.975]).tolist(),
            'games': len(games), 'replicates': replicates, 'negative_favors_model': True,
            'limits': 'Conditional on fitted one-seed models; game sampling only, no causal or training uncertainty.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/sequence_pilot.json')
    parser.add_argument('--run', type=Path, help='Reuse finished model checkpoints with unchanged code/config/data')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root'])
    if not Path('/Volumes/T7 Shield').is_mount() or not root.resolve().is_relative_to(Path('/Volumes/T7 Shield').resolve()):
        raise SystemExit('Mounted configured SSD required')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise SystemExit('Use existing configured Python')
    run = args.run or root/'runs'/datetime.now(timezone.utc).strftime('sequence-%Y%m%dT%H%M%SZ')
    if not run.resolve().is_relative_to(root.resolve()):
        raise SystemExit('Run must be inside SSD artifact root')
    start = time.perf_counter()
    sources = sorted((PROJECT/'pitchmdp').glob('*.py'))+[Path(__file__).resolve()]
    hashes = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if args.run:
        if json.loads((run/'config.json').read_text()) != cfg or json.loads((run/'source_hashes.json').read_text()) != hashes:
            raise SystemExit('Code or config changed: start a new run')
    else:
        run.mkdir(parents=True, exist_ok=False)
        dump(run/'config.json', cfg)
        dump(run/'source_hashes.json', hashes)
        for rel in hashes:
            destination = run/'source'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/rel, destination)
    print('RUN_DIR='+str(run), flush=True)
    shutil.copyfile(cfg['cohort_manifest'], run/'cohort_manifest.json')
    cohort = json.loads((run/'cohort_manifest.json').read_text())
    frame = prepare_frame(local)
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame)
    assert frame.index.equals(pd.RangeIndex(len(frame)))
    mask = eligible(frame)
    target = frame.pitcher.isin(cohort['pitcher_ids']) & frame.pitcher.eq(frame.starter_pitcher)
    train_all = frame[(frame.split == 'train') & mask]
    train = pd.concat([train_all.sample(min(len(train_all), cfg['train_random_rows']), random_state=cfg['seed']),
                       train_all[target.reindex(train_all.index)]]).drop_duplicates(['game_pk','at_bat_number','pitch_number']).sort_index()
    cal_all = frame[(frame.split == 'calibration') & mask]
    cal = cal_all.sample(min(len(cal_all), cfg['calibration_rows']), random_state=cfg['seed']).sort_index()
    dev = frame[(frame.split == 'dev') & mask & target].copy()
    assert train.game_date.max() < cal.game_date.min() < dev.game_date.min()
    assert set(train.game_pk).isdisjoint(dev.game_pk)
    context = SequenceContext().fit(train)
    delivery = JointDelivery().fit(train_all, store.normalizer, cfg['delivery_draws'], cfg['seed'])
    with (run/'encoders.pkl').open('wb') as stream:
        pickle.dump({'context': context, 'normalizer': store.normalizer, 'delivery': delivery}, stream)
    samples = {'all_train_eligible': len(train_all), 'train_used': len(train), 'calibration_used': len(cal),
               'dev_rows': len(dev), 'dev_pa': dev.groupby(['game_pk','at_bat_number']).ngroups,
               'dev_games': dev.game_pk.nunique(), 'dev_starts': dev.groupby(['game_pk','pitcher']).ngroups,
               'dev_batters': dev.batter.nunique(), 'dev_dates': [str(dev.game_date.min()),str(dev.game_date.max())],
               'context': context.report(), 'delivery': delivery.report,
               'rows_hash': {name: hashlib.sha256(part[['game_pk','at_bat_number','pitch_number']].to_numpy(np.int64).tobytes()).hexdigest()
                             for name,part in [('train',train),('calibration',cal),('dev',dev)]}}
    dump(run/'samples.json', samples)
    print('SAMPLES', {k:v for k,v in samples.items() if k not in ['context','delivery','rows_hash']}, flush=True)
    # Binary task selects terminal two-strike swing-outs or fair in-play outcomes.
    # This is conditional retrospective discrimination, not all-pitch forecasting.
    binary = mask & frame.is_pa_terminal & frame.strikes.eq(2) & (
        frame.description.eq('hit_into_play') |
        (frame.events.eq('strikeout') & frame.description.isin(['swinging_strike','swinging_strike_blocked'])))
    btrain = frame[binary & frame.split.eq('train')]
    bcal = frame[binary & frame.split.eq('calibration')]
    bdev = frame[binary & frame.split.eq('dev') & target]
    btrain = btrain.sample(min(len(btrain), cfg['train_random_rows']), random_state=cfg['seed']).sort_index()
    bcal = bcal.sample(min(len(bcal), cfg['calibration_rows']), random_state=cfg['seed']).sort_index()
    binary_arrays = [arrays(store, context, part.index.to_numpy()) for part in (btrain,bcal,bdev)]
    binary_y = [part.description.eq('hit_into_play').to_numpy(np.int64) for part in (btrain,bcal,bdev)]
    binary_results, binary_predictions = {}, {}
    for kind in cfg['binary_models']:
        checkpoint = run/f'binary_{kind}.pt'
        if checkpoint.exists():
            model = SequenceModel.load(checkpoint)
        else:
            model = SequenceModel(kind, cfg['seed'], cfg['width'], n_classes=2).fit(
                binary_arrays[0], binary_y[0], binary_arrays[1], binary_y[1], epochs=cfg['binary_epochs'],
                patience=cfg['patience'], batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'],
                checkpoint=run/f'binary_{kind}_best_training.pt')
            model.save(checkpoint)
        p = model.predict(binary_arrays[2])
        binary_predictions[kind] = p
        binary_results[kind] = {'metrics_conditional_current_physics': classification_metrics(binary_y[2],p),
                                'training': model.report}
        dump(run/'binary_results.json', binary_results)
        print('BINARY',kind,binary_results[kind]['metrics_conditional_current_physics'], flush=True)
    binary_results['paired_transformer_minus_flatten'] = paired(binary_y[2],binary_predictions['transformer'],binary_predictions['flatten_mlp'],bdev.game_pk)
    dump(run/'binary_results.json', binary_results)
    del binary_arrays, binary_predictions
    # All-count experiment: same samples, context, histories, execution pools/budget.
    train_arrays = arrays(store, context, train.index.to_numpy())
    cal_arrays = arrays(store, context, cal.index.to_numpy())
    dev_arrays = arrays(store, context, dev.index.to_numpy())
    labels, cy, dy = outcome_labels(train), outcome_labels(cal), outcome_labels(dev)
    mixture_cal = cal.sample(min(len(cal),cfg['delivery_calibration_rows']),random_state=cfg['seed']).sort_index()
    mixture_rows = mixture_cal.index.to_numpy()
    output, predictions = {}, {}
    for kind in cfg['all_count_models']:
        checkpoint = run/f'all_{kind}.pt'
        if checkpoint.exists():
            model = SequenceModel.load(checkpoint)
        else:
            model = SequenceModel(kind,cfg['seed'],cfg['width'],n_classes=10).fit(
                train_arrays,labels,cal_arrays,cy,epochs=cfg['epochs'],patience=cfg['patience'],
                batch_size=cfg['batch_size'],learning_rate=cfg['learning_rate'],
                checkpoint=run/f'all_{kind}_best_training.pt')
            delivery.calibrate(model,store,context,mixture_rows,outcome_labels(mixture_cal))
            model.save(checkpoint)
        conditional = model.predict(dev_arrays)
        p = delivery.predict(model,store,context,dev.index.to_numpy())
        predictions[kind] = p
        output[kind] = {'primary_delivery_integrated': classification_metrics(dy,p),
                        'conditional_current_physics_diagnostic': classification_metrics(dy,conditional),
                        'training':model.report,
                        'per_pitcher':{str(int(pid)): classification_metrics(dy[dev.pitcher.to_numpy()==pid],p[dev.pitcher.to_numpy()==pid])
                                       for pid in cohort['pitcher_ids']}}
        dump(run/'all_count_results.json',output)
        print('ALLCOUNT',kind,output[kind]['primary_delivery_integrated'],flush=True)
    base = CountBaseline().fit(train_all)
    base_p = base.predict(dev)
    output['count_hand_baseline'] = classification_metrics(dy,base_p)
    output['paired_comparisons'] = {
        kind:paired(dy,predictions['transformer'],p,dev.game_pk,cfg['bootstrap_replicates'])
        for kind,p in [('flatten_mlp',predictions['flatten_mlp']),('current_only',predictions['current_only']),('count_hand_baseline',base_p)]}
    dump(run/'all_count_results.json',output)
    np.savez_compressed(run/'heldout_predictions.npz',y=dy,game_pk=dev.game_pk.to_numpy(),
                        pitch_keys=dev[['game_pk','at_bat_number','pitch_number']].to_numpy(),**predictions,count_hand=base_p)
    print('PAIRED',output['paired_comparisons'],flush=True)
    # Independent entrypoint for exact saved-data planning/replay.
    representatives = dev[(dev.pitch_number==1)&(dev.inning<=8)].groupby('pitcher').head(1)
    from pitchmdp.game import WinExpectancy, EmpiricalAdvancement
    # Reuse fixed train-only W/advancement from previous experiment, no extra fitting.
    with (root/'runs/first-20260921T014425Z/game_models.pkl').open('rb') as stream:
        we,advancement=pickle.load(stream)
    from pitchmdp.recommend import supported_actions
    with (run/'planning_context.pkl').open('wb') as stream:
        pickle.dump({'rows':representatives,'we':we,'advancement':advancement,
                     'baseline':base,'actions':{(int(r.pitcher),str(r.stand)):supported_actions(train_all,int(r.pitcher),str(r.stand))
                                                for r in representatives.itertuples()}},stream)
    runtime={'finished_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start,
             'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'source_hashes_end':{str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    dump(run/'runtime.json',runtime)
    print('COMPLETE '+str(run),flush=True)


if __name__=='__main__':
    main()
