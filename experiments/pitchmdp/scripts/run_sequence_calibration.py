"""Fixed five-seed ensemble and calibration-only shrinkage to count frequencies."""
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
from scipy.optimize import minimize_scalar

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_pilot import dump, paired
from run_sequence_ablations import reconstruct_samples, rows_hash


SEEDS = [42, 43, 44, 45, 46]


def fit_blend(y, model_p, baseline_p, objective):
    """Select a convex weight using only explicitly supplied calibration rows."""
    y = np.asarray(y, int)
    model_p, baseline_p = np.asarray(model_p, float), np.asarray(baseline_p, float)
    classification_metrics(y, model_p)
    classification_metrics(y, baseline_p)
    if model_p.shape != baseline_p.shape:
        raise ValueError('Model and baseline probabilities must align')
    if objective not in ('log_loss', 'brier_multiclass'):
        raise ValueError('Unknown fixed calibration objective')
    def score(weight):
        p = weight*model_p+(1-weight)*baseline_p
        if objective == 'log_loss':
            return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        return float(np.square(p-np.eye(p.shape[1])[y]).sum(1).mean())
    optimum = minimize_scalar(score, bounds=(0., 1.), method='bounded')
    choices = [0., float(optimum.x), 1.]
    weight = min(choices, key=score)
    return {'model_weight': weight, 'baseline_weight': 1-weight,
            'calibration_objective': objective, 'calibration_score': score(weight),
            'selection': 'Calibration only; fixed candidates include exact endpoints.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to(Path('/Volumes/T7 Shield').resolve()):
        raise SystemExit('Mounted configured SSD required')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise SystemExit('Use existing configured Python')
    robustness = args.robustness.resolve()
    if not robustness.is_relative_to(root):
        raise SystemExit('Robustness run must be inside configured SSD')
    manifest = json.loads((robustness/'config.json').read_text())
    base = Path(manifest['base_run'])
    cfg = json.loads((base/'config.json').read_text())
    samples = json.loads((base/'samples.json').read_text())
    cohort = json.loads((base/'cohort_manifest.json').read_text())
    output = (args.output or robustness/datetime.now(timezone.utc).strftime('calibration-%Y%m%dT%H%M%SZ')).resolve()
    if not output.is_relative_to(root) or output in (root, base, robustness) or robustness.is_relative_to(output):
        raise SystemExit('Calibration output needs a distinct SSD subdirectory')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    results_paths = [robustness/f'models/seed{seed}/full_transformer/result.json' for seed in SEEDS]
    records = [json.loads(path.read_text()) for path in results_paths]
    checkpoints = [Path(record['checkpoint_path']) for record in records]
    prediction_paths = [path.parent/'predictions.npz' for path in results_paths]
    for path,record in zip(results_paths,records):
        for name,digest in record['artifact_hashes'].items():
            if hash_file(path.parent/name) != digest:
                raise ValueError('Robustness member artifact changed: '+name)
    references = [*results_paths, *checkpoints, *prediction_paths, base/'encoders.pkl', base/'planning_context.pkl',
                  base/'heldout_predictions.npz', base/'config.json', base/'samples.json',
                  base/'cohort_manifest.json', base/'audit_supplemental/sequence_physics.json', robustness/'config.json']
    original = json.loads((base/'source_hashes.json').read_text())
    for rel, digest in original.items():
        if hash_file(PROJECT/rel) != digest:
            raise SystemExit('Original implementation changed: '+rel)
    sources = {**original, 'scripts/run_sequence_ablations.py': hash_file(PROJECT/'scripts/run_sequence_ablations.py'),
               str(Path(__file__).resolve().relative_to(PROJECT)): hash_file(Path(__file__))}
    identities = {str(path): hash_file(path) for path in references}
    config = {'base_run': str(base), 'robustness_run': str(robustness), 'seeds': SEEDS,
              'source_hashes': sources, 'reference_hashes': identities,
              'weight_objectives': ['log_loss', 'brier_multiclass'],
              'primary_blend': 'log_loss', 'ensemble': 'arithmetic probability mean of all five fixed full models',
              'calibration_rows': 'Original16000 minus original4000 temperature-fitting rows',
              'scope': 'Exploratory after DEV inspection; no DEV-based weight or seed selection; no policy claims.'}
    dump(output/'config.json', config)
    for rel in sources:
        target = output/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    print('CALIBRATION_DIR='+str(output), flush=True)
    with (base/'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    with (base/'planning_context.pkl').open('rb') as stream:
        baseline = pickle.load(stream)['baseline']
    frame = prepare_frame(local)
    expected_identity = json.loads((base/'audit_supplemental/sequence_physics.json').read_text())['identity']
    if frame.attrs['sequence_data_identity'] != expected_identity:
        raise ValueError('Physical/processed data identity differs from original run')
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=encoders['normalizer'])
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    hashes = {name: rows_hash(part) for name, part in [('train', train), ('calibration', cal), ('dev', dev)]}
    if hashes != samples['rows_hash']:
        raise ValueError('Ordered sample identity differs from original')
    temperature_rows = cal.sample(min(len(cal),cfg['delivery_calibration_rows']), random_state=cfg['seed']).index
    blend_cal = cal.loc[~cal.index.isin(temperature_rows)]
    if len(blend_cal) != 12000 or not set(blend_cal.index).isdisjoint(temperature_rows):
        raise ValueError('Expected disjoint12000 calibration blend rows')
    cy, dy = outcome_labels(blend_cal), outcome_labels(dev)
    cal_base, dev_base = baseline.predict(blend_cal), baseline.predict(dev)
    with np.load(base/'heldout_predictions.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['pitch_keys'],dev[KEY].to_numpy())
        np.testing.assert_array_equal(saved['y'],dy)
        np.testing.assert_array_equal(saved['count_hand'],dev_base)
    cal_predictions, dev_predictions = [], []
    for seed, checkpoint in zip(SEEDS, checkpoints):
        model = SequenceModel.load(checkpoint)
        if model.seed != seed or model.kind != 'transformer':
            raise ValueError('Wrong fixed ensemble member')
        cal_p = encoders['delivery'].predict(model, store, encoders['context'], blend_cal.index.to_numpy())
        # Replaying DEV independently also checks the exact checkpoint used.
        dev_p = encoders['delivery'].predict(model, store, encoders['context'], dev.index.to_numpy())
        with np.load(robustness/f'models/seed{seed}/full_transformer/predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['pitch_keys'], dev[KEY].to_numpy())
            np.testing.assert_array_equal(saved['y'],dy)
            np.testing.assert_allclose(dev_p, saved['delivery_integrated_calibrated'], atol=1e-7, rtol=1e-7)
        cal_predictions.append(cal_p); dev_predictions.append(dev_p)
        print('REPLAY',seed,classification_metrics(dy,dev_p)['log_loss'],flush=True)
        del model
        gc.collect()
    ensemble_cal = np.mean(cal_predictions, axis=0)
    ensemble_dev = np.mean(dev_predictions, axis=0)
    predictions = {'ensemble': ensemble_dev, 'count_hand': dev_base}
    results = {'seeds': SEEDS, 'blend_calibration_n': len(blend_cal),
               'calibration_row_hash': rows_hash(blend_cal), 'dev_row_hash': rows_hash(dev),
               'calibration_use_note': 'Disjoint from temperature rows; all16000 were used for early stopping, so not independent validation.',
               'interval_scope': 'Game resampling conditional on fitted ensemble and blend weights; not calibration-fitting or seed-fitting uncertainty.',
               'count_hand': classification_metrics(dy,dev_base),
               'ensemble': classification_metrics(dy,ensemble_dev), 'blends': {}}
    for objective in ('log_loss','brier_multiclass'):
        fitted = fit_blend(cy,ensemble_cal,cal_base,objective)
        p = fitted['model_weight']*ensemble_dev+fitted['baseline_weight']*dev_base
        key = 'blend_'+objective
        predictions[key] = p
        results['blends'][objective] = {**fitted, 'dev_metrics': classification_metrics(dy,p),
            'paired_vs_ensemble': paired(dy,p,ensemble_dev,dev.game_pk),
            'paired_vs_baseline': paired(dy,p,dev_base,dev.game_pk)}
    results['ensemble_paired_vs_baseline'] = paired(dy,ensemble_dev,dev_base,dev.game_pk)
    dump(output/'results.json',results)
    np.savez_compressed(output/'predictions.npz',y=dy,game_pk=dev.game_pk.to_numpy(),
                        pitch_keys=dev[KEY].to_numpy(),seed_predictions=np.asarray(dev_predictions),**predictions)
    np.savez_compressed(output/'calibration_predictions.npz',y=cy,pitch_keys=blend_cal[KEY].to_numpy(),
                        game_pk=blend_cal.game_pk.to_numpy(),seed_predictions=np.asarray(cal_predictions),
                        ensemble=ensemble_cal,count_hand=cal_base)
    if {str(path):hash_file(path) for path in references} != identities or {rel:hash_file(PROJECT/rel) for rel in sources} != sources:
        raise RuntimeError('Reference or source changed during analysis')
    dump(output/'runtime.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-start})
    print('COMPLETE '+str(output),flush=True)


if __name__ == '__main__':
    main()
