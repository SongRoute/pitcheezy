"""Reevaluate fixed checkpoints with shared 100-draw delivery integration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
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

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_delivery_adaptation import calibrate_temperature
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_calibration import fit_blend
from run_sequence_pilot import dump, paired
from run_sequence_robustness import (SEEDS, MASKS, FixedMaskedContext, aggregate_comparison,
                                     _save_model_result, update_summary, _verified_predictions)

PROTOCOL = PROJECT / 'docs/INTEGRATION_REEVALUATION_PROTOCOL.md'


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.net.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def probability_pair(logits, temperature):
    return {'delivery_integrated_uncalibrated': softmax(logits, axis=-1).mean(axis=1),
            'delivery_integrated_calibrated': softmax(logits / temperature, axis=-1).mean(axis=1)}


def ensemble_calibration(output, base, original_robustness, context, cy, dy, cal_keys, dev_keys,
                         cal_games, dev_games, cal_hash, dev_hash, baseline_cal, baseline_dev,
                         config, cal_predictions, dev_predictions):
    """Write original-compatible 100-draw archives without changing old runs."""
    directory = output / 'calibration100'
    directory.mkdir(exist_ok=True)
    identity = {'base_run': str(base), 'robustness_run': str(output),
                'original_robustness_run': str(original_robustness), 'seeds': list(SEEDS),
                'source_hashes': config['source_hashes'], 'reference_hashes': config['reference_hashes'],
                'delivery_draws': 100, 'delivery_pool_path': config['delivery100_path'],
                'delivery_pool_sha256': config['reference_hashes'][config['delivery100_path']],
                'delivery_pool_note': config['pool_randomness'],
                'weight_objectives': ['log_loss', 'brier_multiclass'],
                'primary_blend': 'log_loss', 'ensemble': 'arithmetic probability mean of five fixed full models',
                'calibration_rows': 'Original16000 minus original4000 temperature rows',
                'scope': 'Exploratory fixed-checkpoint integration reevaluation; no retraining or DEV selection'}
    dump(directory / 'config.json', identity)
    ensemble_cal = np.mean(cal_predictions, axis=0)
    ensemble_dev = np.mean(dev_predictions, axis=0)
    fitted_blends = {objective: fit_blend(cy, ensemble_cal, baseline_cal, objective)
                    for objective in ('log_loss', 'brier_multiclass')}
    dump(directory / 'selection.json', {'fitted_blends': fitted_blends,
         'criterion': 'CAL-only convex weights; primary log loss, secondary Brier; no DEV selection',
         'calibration_row_hash': cal_hash, 'temperature_row_hash': config['temperature_row_hash']})
    results = {'seeds': list(SEEDS), 'delivery_draws': 100, 'blend_calibration_n': len(cy),
               'calibration_row_hash': cal_hash, 'dev_row_hash': dev_hash,
               'temperature_row_hash': config['temperature_row_hash'],
               'calibration_use_note': 'Disjoint from temperature rows; all16000 influenced original early stopping; not independent validation.',
               'interval_scope': 'Game resampling conditional on fixed ensemble, integration pool and CAL weights; not fitting/pool uncertainty.',
               'count_hand': classification_metrics(dy, baseline_dev),
               'ensemble': classification_metrics(dy, ensemble_dev), 'blends': {}}
    probabilities = {'ensemble': ensemble_dev, 'count_hand': baseline_dev}
    def compare(p, q):
        comparison = paired(dy, p, q, dev_games)
        comparison['limits'] = results['interval_scope']
        return comparison
    for objective in ('log_loss', 'brier_multiclass'):
        fit = fitted_blends[objective]
        p = fit['model_weight'] * ensemble_dev + fit['baseline_weight'] * baseline_dev
        probabilities['blend_' + objective] = p
        results['blends'][objective] = {**fit, 'dev_metrics': classification_metrics(dy, p),
            'paired_vs_ensemble': compare(p, ensemble_dev),
            'paired_vs_baseline': compare(p, baseline_dev)}
    results['ensemble_paired_vs_baseline'] = compare(ensemble_dev, baseline_dev)
    dump(directory / 'results.json', results)
    np.savez_compressed(directory / 'predictions.npz', y=dy, pitch_keys=dev_keys, game_pk=dev_games,
                        seed_predictions=np.asarray(dev_predictions), **probabilities)
    np.savez_compressed(directory / 'calibration_predictions.npz', y=cy, pitch_keys=cal_keys, game_pk=cal_games,
                        seed_predictions=np.asarray(cal_predictions), ensemble=ensemble_cal, count_hand=baseline_cal)
    dump(directory / 'runtime.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                                     'complete': True, 'parent_integration_run_continues': True})
    print('CALIBRATION100_COMPLETE=' + str(directory), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness', required=True, type=Path)
    parser.add_argument('--delivery100', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    local = json.loads((PROJECT / 'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise SystemExit('Mounted configured SSD and existing Python required')
    robustness, delivery_path = args.robustness.resolve(), args.delivery100.resolve()
    original_config = json.loads((robustness / 'config.json').read_text())
    base = Path(original_config['base_run'])
    output = (args.output or robustness / datetime.now(timezone.utc).strftime('integration100-%Y%m%dT%H%M%SZ')).resolve()
    if not all(p.is_relative_to(root) for p in (robustness, delivery_path, base, output)) or output in (base, robustness, root):
        raise SystemExit('All input/output paths must be distinct configured SSD locations')
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((base / 'source_hashes.json').read_text())
    for rel, digest in original.items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Original source changed: ' + rel)
    extra = ['scripts/run_sequence_robustness.py', 'scripts/run_sequence_ablations.py',
             'scripts/run_sequence_calibration.py', 'scripts/run_delivery_adaptation.py',
             str(Path(__file__).resolve().relative_to(PROJECT)), str(PROTOCOL.relative_to(PROJECT))]
    sources = {**original, **{rel: hash_file(PROJECT / rel) for rel in extra}}
    references = [delivery_path, robustness / 'config.json', robustness / 'data.json',
                  *[base / n for n in ['config.json', 'samples.json', 'cohort_manifest.json', 'encoders.pkl',
                                       'planning_context.pkl', 'heldout_predictions.npz']]]
    records = {}
    for variant in MASKS:
        for seed in SEEDS:
            directory = robustness / 'models' / f'seed{seed}' / variant
            record = json.loads((directory / 'result.json').read_text())
            if record['seed'] != seed or record['variant'] != variant or record['zero_context_indices'] != list(MASKS[variant]):
                raise ValueError('Original checkpoint mask/identity mismatch')
            for name, digest in record['artifact_hashes'].items():
                if hash_file(directory / name) != digest:
                    raise ValueError('Original artifact changed: ' + str(directory / name))
            references.extend([directory / n for n in ['result.json', 'model.pt', 'predictions.npz']])
            records[(variant, seed)] = record
    reference_hashes = {str(p): hash_file(p) for p in references}
    config = {'base_run': str(base), 'robustness_run': str(robustness), 'seeds': list(SEEDS),
              'variants': list(MASKS), 'delivery100_path': str(delivery_path), 'draws': 100,
              'calibration_temperature_rows': 4000, 'calibration_blend_rows': 12000,
              'neural_training': False, 'source_hashes': sources, 'reference_hashes': reference_hashes,
              'protocol_sha256': hash_file(PROTOCOL), 'sensitivity_draws': [25, 100, 400],
              'sensitivity_model': 'full_transformer seed42 only; no DEV draw-count selection',
              'pool_randomness': 'Shared100 empirical TRAIN pool; 25/100/400 pools not nested; no pool-uncertainty bootstrap'}
    dump(output / 'config.json', config)
    for rel in sources:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    shutil.copyfile(PROTOCOL, output / PROTOCOL.name)
    print('INTEGRATION_DIR=' + str(output), flush=True)
    with (base / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    with delivery_path.open('rb') as stream:
        delivery = pickle.load(stream)
    if delivery.draws != 100:
        raise ValueError('Expected the fixed100-draw pool')
    for channel in ('mean', 'scale', 'fill'):
        np.testing.assert_array_equal(getattr(delivery.normalizer, channel), getattr(encoders['normalizer'], channel))
    with (base / 'planning_context.pkl').open('rb') as stream:
        baseline = pickle.load(stream)['baseline']
    cfg = json.loads((base / 'config.json').read_text())
    cohort = json.loads((base / 'cohort_manifest.json').read_text())
    samples = json.loads((base / 'samples.json').read_text())
    frame = prepare_frame(local)
    identity = frame.attrs['sequence_data_identity']
    if identity != json.loads((robustness / 'data.json').read_text())['data_identity']:
        raise ValueError('Data identity differs from frozen robustness run')
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=encoders['normalizer'])
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    hashes = {n: rows_hash(part) for n, part in [('train', train), ('calibration', cal), ('dev', dev)]}
    if hashes != samples['rows_hash']:
        raise ValueError('Ordered sample identities changed')
    mixture = cal.sample(cfg['delivery_calibration_rows'], random_state=42).sort_index()
    blend_cal = cal.loc[~cal.index.isin(mixture.index)]
    if len(mixture) != 4000 or len(blend_cal) != 12000:
        raise ValueError('Original temperature/blend calibration split changed')
    config['temperature_row_hash'] = rows_hash(mixture)
    config['blend_calibration_row_hash'] = rows_hash(blend_cal)
    dump(output / 'config.json', config)
    cy, by, dy = outcome_labels(mixture), outcome_labels(blend_cal), outcome_labels(dev)
    keys, games = dev[KEY].to_numpy(), dev.game_pk.to_numpy()
    months = pd.to_datetime(dev.game_date).dt.strftime('%Y-%m').to_numpy(dtype='U7')
    dump(output / 'data.json', {'data_identity': identity, 'ordered_rows_hash': hashes,
         'delivery_calibration_rows_hash': rows_hash(mixture), 'blend_calibration_rows_hash': rows_hash(blend_cal),
         'dev_rows': len(dev), 'dev_games': len(np.unique(games)), 'pool_report': delivery.report})
    dev_predictions, blend_predictions = [], []
    # Complete the full ensemble first, exposing CPU-consumable blend archives early.
    for variant in MASKS:
        for seed in SEEDS:
            start = time.perf_counter()
            source = robustness / 'models' / f'seed{seed}' / variant
            directory = output / 'models' / f'seed{seed}' / variant
            directory.mkdir(parents=True, exist_ok=True)
            model = SequenceModel.load(source / 'model.pt')
            if model.seed != seed:
                raise ValueError('Loaded checkpoint seed mismatch')
            weights_before = tensor_hash(model)
            context = FixedMaskedContext(encoders['context'], variant)
            logits, _ = delivery.logits(model, store, context, mixture.index.to_numpy())
            fit = calibrate_temperature(logits, cy)
            dump(directory / 'calibration100.json', fit)
            del logits
            logits, _ = delivery.logits(model, store, context, dev.index.to_numpy())
            probabilities = probability_pair(logits, fit['temperature'])
            del logits
            old = _verified_predictions(source / 'predictions.npz', dy, keys, games)
            probabilities.update({name: old[name] for name in ('conditional_calibrated', 'conditional_uncalibrated')})
            model.delivery_temperature = fit['temperature']
            model.report = {**model.report, 'integration100_recalibration': fit, 'neural_training_in_this_run': False}
            model.save(directory / 'model.pt')
            if tensor_hash(model) != weights_before:
                raise ValueError('Neural weights changed during fixed-model inference')
            result = _save_model_result(directory, directory / 'model.pt', probabilities, model.report,
                {'mode': 'fixed_neural_weights_shared100draw_pool', 'original_checkpoint': str(source / 'model.pt'),
                 'original_checkpoint_sha256': reference_hashes[str(source / 'model.pt')],
                 'neural_tensor_sha256': weights_before, 'temperature_fit': fit}, dy, keys, games, months, seed, variant)
            result['artifact_hashes']['calibration100.json'] = hash_file(directory / 'calibration100.json')
            if variant == 'full_transformer':
                pcal = delivery.predict(model, store, context, blend_cal.index.to_numpy())
                np.savez_compressed(directory / 'blend_calibration_predictions.npz', y=by, pitch_keys=blend_cal[KEY].to_numpy(),
                                    game_pk=blend_cal.game_pk.to_numpy(), probabilities=pcal)
                blend_predictions.append(pcal)
                dev_predictions.append(probabilities['delivery_integrated_calibrated'])
                result['artifact_hashes']['blend_calibration_predictions.npz'] = hash_file(directory / 'blend_calibration_predictions.npz')
            dump(directory / 'result.json', result)
            print('MODEL100', seed, variant, result['metrics']['delivery_integrated_calibrated']['log_loss'],
                  result['metrics']['delivery_integrated_calibrated']['brier_multiclass'], 'seconds', time.perf_counter()-start, flush=True)
            del model, old, probabilities
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        update_summary(output, dy, keys, games, months)
        if variant == 'full_transformer':
            ensemble_calibration(output, base, robustness, encoders['context'], by, dy,
                blend_cal[KEY].to_numpy(), keys, blend_cal.game_pk.to_numpy(), games,
                rows_hash(blend_cal), rows_hash(dev), baseline.predict(blend_cal), baseline.predict(dev),
                config, blend_predictions, dev_predictions)
    summary = update_summary(output, dy, keys, games, months)
    summary['exploratory_status'] = 'Fixed-checkpoint exploratory100draw reevaluation on inspected DEV; no retraining or new validation'
    summary['integration_draws'] = 100
    summary['comparisons100_minus25'] = {}
    for variant in MASKS:
        current = np.stack([_verified_predictions(output/'models'/f'seed{s}'/variant/'predictions.npz', dy, keys, games)['delivery_integrated_calibrated'] for s in SEEDS])
        original_p = np.stack([_verified_predictions(robustness/'models'/f'seed{s}'/variant/'predictions.npz', dy, keys, games)['delivery_integrated_calibrated'] for s in SEEDS])
        summary['comparisons100_minus25'][variant] = aggregate_comparison(dy, current, original_p, games, SEEDS)
    dump(output / 'summary.json', summary)
    # Final prespecified convergence sensitivity, without architecture/draw selection.
    train_all = frame.loc[frame.split.eq('train') & eligible(frame)]
    delivery400 = JointDelivery().fit(train_all, encoders['normalizer'], draws=400, seed=42)
    with (output / 'delivery_draw400.pkl').open('wb') as stream:
        pickle.dump(delivery400, stream)
    model = SequenceModel.load(robustness / 'models/seed42/full_transformer/model.pt')
    context = FixedMaskedContext(encoders['context'], 'full_transformer')
    logits, _ = delivery400.logits(model, store, context, mixture.index.to_numpy())
    fit400 = calibrate_temperature(logits, cy)
    dump(output / 'calibration400.json', fit400)
    del logits
    logits, _ = delivery400.logits(model, store, context, dev.index.to_numpy())
    p400 = probability_pair(logits, fit400['temperature'])
    np.savez_compressed(output / 'predictions400.npz', y=dy, pitch_keys=keys, game_pk=games, **p400)
    draws = {'400': {regime: classification_metrics(dy, p) for regime, p in p400.items()}}
    for count, parent in ((25, robustness), (100, output)):
        prior = _verified_predictions(parent / 'models/seed42/full_transformer/predictions.npz', dy, keys, games)
        draws[str(count)] = {regime: classification_metrics(dy, prior[regime]) for regime in p400}
    p100 = _verified_predictions(output / 'models/seed42/full_transformer/predictions.npz', dy, keys, games)
    dump(output / 'convergence.json', {'draws': draws, 'temperature400': fit400,
         'pool400_sha256': hash_file(output / 'delivery_draw400.pkl'),
         'paired400_minus100': aggregate_comparison(dy, p400['delivery_integrated_calibrated'],
             p100['delivery_integrated_calibrated'], games, [42]),
         'selection': 'None; fixed25/100/400 comparison, not nested pools or formal convergence proof'})
    sources_end = {rel: hash_file(PROJECT / rel) for rel in sources}
    references_end = {path: hash_file(Path(path)) for path in reference_hashes}
    valid = sources_end == sources and references_end == reference_hashes
    dump(output / 'runtime.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
         'seconds': time.perf_counter()-started, 'checkpoints': 30, 'neural_training': False,
         'integrity_ok': valid, 'source_hashes_end': sources_end, 'reference_hashes_end': references_end})
    if not valid:
        raise RuntimeError('Sources or fixed references changed during inference')
    print('COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
