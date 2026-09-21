"""Fixed-T42-temperature DEV-only sampled-versus-exact empirical integration."""
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
from scipy.special import softmax
import torch

from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import KEY, hash_file
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_data import HistoryStore, prepare_frame
from pitchmdp.sequence_delivery import JointDelivery
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_ablations import reconstruct_samples, rows_hash
from run_sequence_integration_reevaluation import tensor_hash
from run_sequence_pilot import dump
from run_sequence_robustness import aggregate_comparison

PROTOCOL = PROJECT / 'docs/EXACT_DELIVERY_DIAGNOSTIC_PROTOCOL.md'


def exact_memberships(train, queries):
    """Return every TRAIN frame position from the highest-supported original tier."""
    if not len(train) or not train.split.eq('train').all() or not train.index.is_unique:
        raise ValueError('Exact delivery pool requires nonempty uniquely indexed TRAIN rows only')
    positions = train.index.to_numpy(np.int64)
    pools = {}
    for level, keys in enumerate(JointDelivery.TIERS):
        minimum = 20 if 'balls' in keys else 50
        for key, local in train.groupby(list(keys), observed=True).indices.items():
            if len(local) >= minimum:
                pools[(level, key)] = positions[local]
    columns = list(dict.fromkeys(k for tier in JointDelivery.TIERS for k in tier))
    members, tiers, identities = [], [], []
    for values in queries[columns].itertuples(index=False, name=None):
        row = dict(zip(columns, values))
        selected, tier, key = positions, -1, ()
        for level in reversed(range(len(JointDelivery.TIERS))):
            candidate = tuple(row[k] for k in JointDelivery.TIERS[level])
            if (level, candidate) in pools:
                selected, tier, key = pools[(level, candidate)], level, candidate
                break
        members.append(selected)
        tiers.append(tier)
        identities.append({'tier': tier, 'keys': list(JointDelivery.TIERS[tier]) if tier >= 0 else [],
                           'values': [v.item() if isinstance(v, np.generic) else v for v in key]})
    sizes = np.array([len(p) for p in members], dtype=np.int64)
    offsets = np.r_[np.int64(0), sizes.cumsum()]
    return np.concatenate(members), offsets, np.array(tiers, dtype=np.int8), identities


def ragged_probabilities(logits, offsets, temperature):
    logits, offsets = np.asarray(logits), np.asarray(offsets)
    if (logits.ndim != 2 or not np.isfinite(logits).all() or offsets.ndim != 1 or len(offsets) < 2 or
            not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or offsets[-1] != len(logits) or
            np.any(np.diff(offsets) <= 0) or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Expected finite candidate logits, positive segment lengths and temperature')
    probabilities = softmax(logits.astype(np.float64) / temperature, axis=-1)
    return np.add.reduceat(probabilities, offsets[:-1], axis=0) / np.diff(offsets)[:, None]


def l1_diagnostic(probabilities, exact):
    distance = np.abs(np.asarray(probabilities) - np.asarray(exact)).sum(axis=1)
    return {'mean': float(distance.mean()),
            'quantiles': {str(q): float(np.quantile(distance, q)) for q in [0, .25, .5, .75, .9, .95, .99, 1]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--integration', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    local = json.loads((PROJECT / 'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise SystemExit('Use mounted configured SSD and existing Python')
    base, integration, profile = args.run.resolve(), args.integration.resolve(), args.profile.resolve()
    integrated = json.loads((integration / 'config.json').read_text())
    if Path(integrated['base_run']) != base:
        raise ValueError('Integration and base run differ')
    pool100 = Path(integrated['delivery100_path'])
    temperature = float(json.loads((integration / 'calibration400.json').read_text())['temperature'])
    output = (args.output or base / datetime.now(timezone.utc).strftime('exact-dev-diagnostic-%Y%m%dT%H%M%SZ')).resolve()
    if not all(p.is_relative_to(root) for p in [base, integration, profile, output, pool100]) or output in (base, integration, profile, root):
        raise SystemExit('Use distinct paths on configured SSD')
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((base / 'source_hashes.json').read_text())
    for rel, digest in original.items():
        if hash_file(PROJECT / rel) != digest:
            raise ValueError('Original source changed: ' + rel)
    extra = ['scripts/run_sequence_ablations.py', 'scripts/run_sequence_robustness.py',
             'scripts/run_sequence_integration_reevaluation.py', 'scripts/run_sequence_calibration.py',
             'scripts/run_delivery_adaptation.py', str(Path(__file__).resolve().relative_to(PROJECT)),
             str(PROTOCOL.relative_to(PROJECT))]
    sources = {**original, **{rel: hash_file(PROJECT / rel) for rel in extra}}
    references = [*[base / n for n in ['all_transformer.pt', 'encoders.pkl', 'config.json', 'samples.json', 'cohort_manifest.json', 'heldout_predictions.npz']],
                  integration / 'config.json', integration / 'data.json', integration / 'calibration400.json', integration / 'delivery_draw400.pkl',
                  pool100, profile / 'profile.json', profile / 'row_costs.npz']
    identities = {str(p): hash_file(p) for p in references}
    config = {'base_run': str(base), 'integration_run': str(integration), 'profile_run': str(profile),
              'neural_checkpoint': str(base / 'all_transformer.pt'), 'model_seed': 42,
              'common_temperature': temperature, 'temperature_source': str(integration / 'calibration400.json'),
              'regimes': ['draw25', 'draw100', 'draw400', 'exact'], 'calibration_inference': False,
              'temperature_refitting': False, 'neural_training': False, 'selection': 'none',
              'source_hashes': sources, 'reference_hashes': identities, 'protocol_sha256': hash_file(PROTOCOL),
              'scope': 'Numerical DEV diagnostic of one fixed model/temperature against finite empirical TRAIN pools, not population-true integration'}
    dump(output / 'config.json', config)
    for rel in sources:
        target = output / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / rel, target)
    shutil.copyfile(PROTOCOL, output / PROTOCOL.name)
    print('EXACT_DIAGNOSTIC_DIR=' + str(output), flush=True)
    with (base / 'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    pools = {'draw25': encoders['delivery']}
    for name, path in [('draw100', pool100), ('draw400', integration / 'delivery_draw400.pkl')]:
        with path.open('rb') as stream:
            pools[name] = pickle.load(stream)
    for name, pool in pools.items():
        if pool.draws != int(name[4:]):
            raise ValueError('Saved pool draw count mismatch')
        for channel in ('mean', 'scale', 'fill'):
            np.testing.assert_array_equal(getattr(pool.normalizer, channel), getattr(encoders['normalizer'], channel))
    frame = prepare_frame(local)
    data_identity = frame.attrs['sequence_data_identity']
    if data_identity != json.loads((integration / 'data.json').read_text())['data_identity']:
        raise ValueError('Verified physical data identity differs from frozen integration run')
    add_batter_style_history(frame)
    store = HistoryStore.from_frame(frame, normalizer=encoders['normalizer'])
    cfg = json.loads((base / 'config.json').read_text())
    cohort = json.loads((base / 'cohort_manifest.json').read_text())
    train, cal, dev = reconstruct_samples(frame, cfg, cohort)
    hashes = {name: rows_hash(part) for name, part in [('train', train), ('calibration', cal), ('dev', dev)]}
    if hashes != json.loads((base / 'samples.json').read_text())['rows_hash']:
        raise ValueError('Original sample hashes changed')
    full_train = frame.loc[frame.split.eq('train') & eligible(frame)]
    support, offsets, tiers, pool_keys = exact_memberships(full_train, dev)
    with np.load(profile / 'row_costs.npz', allow_pickle=False) as checked:
        np.testing.assert_array_equal(checked['dev_pitch_keys'], dev[KEY].to_numpy())
        np.testing.assert_array_equal(checked['dev_pool_rows'], np.diff(offsets))
        np.testing.assert_array_equal(checked['dev_tier'], tiers)
    if len(support) != 1330692 or np.any(tiers < 0) or not np.all(frame.split.to_numpy()[support] == 'train'):
        raise ValueError('Expected all1,330,692 exact TRAIN pairs and no DEV global fallback')
    y, keys, games = outcome_labels(dev), dev[KEY].to_numpy(np.int64), dev.game_pk.to_numpy()
    dump(output / 'data.json', {'data_identity': data_identity, 'sample_hashes': hashes,
         'dev_rows': len(dev), 'dev_games': int(dev.game_pk.nunique()), 'full_eligible_train_rows': len(full_train),
         'exact_candidate_pairs': len(support), 'global_fallback_rows': 0,
         'pool_keys_per_query': pool_keys, 'exact_weights': 'uniform1/pool_rows for every retained TRAIN member; no caps or subsampling'})
    model = SequenceModel.load(base / 'all_transformer.pt')
    if model.seed != 42 or model.kind != 'transformer' or model.n_classes != 10:
        raise ValueError('Expected original full ten-outcome Transformer42')
    before = tensor_hash(model)
    owner = np.repeat(np.arange(len(dev), dtype=np.int64), np.diff(offsets))
    context = encoders['context'].transform(dev)
    dev_positions = dev.index.to_numpy()
    exact_logits = np.empty((len(support), 10), dtype=np.float32)
    for begin in range(0, len(support), 4096):
        end = min(begin + 4096, len(support))
        query = owner[begin:end]
        tokens, valid = store.gather(dev_positions[query], current=store.physical[support[begin:end]])
        exact_logits[begin:end] = model.logits((tokens, valid, context[query]))
    np.savez_compressed(output / 'exact_dev_logits.npz', logits=exact_logits, row_offsets=offsets,
                        y=y, pitch_keys=keys, game_pk=games, pool_tier=tiers,
                        training_frame_positions=support,
                        training_pitch_keys=frame[KEY].to_numpy(np.int64)[support],
                        uniform_weight_per_query=1. / np.diff(offsets))
    probabilities = {'exact_raw': ragged_probabilities(exact_logits, offsets, 1.),
                     'exact_fixed_temperature': ragged_probabilities(exact_logits, offsets, temperature)}
    del exact_logits
    gc.collect()
    print('EXACT_DEV_COMPLETE', len(support), 'candidate pairs', flush=True)
    for name, delivery in pools.items():
        logits, _ = delivery.logits(model, store, encoders['context'], dev.index.to_numpy())
        sampled_offsets = np.arange(0, len(dev) * delivery.draws + 1, delivery.draws, dtype=np.int64)
        flat = logits.reshape(-1, 10)
        probabilities[name + '_raw'] = ragged_probabilities(flat, sampled_offsets, 1.)
        probabilities[name + '_fixed_temperature'] = ragged_probabilities(flat, sampled_offsets, temperature)
        print('SAMPLED_DEV_COMPLETE', name, classification_metrics(y, probabilities[name + '_fixed_temperature'])['log_loss'], flush=True)
        del logits, flat
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if tensor_hash(model) != before:
        raise ValueError('Neural tensors changed during inference')
    np.savez_compressed(output / 'predictions.npz', y=y, pitch_keys=keys, game_pk=games, **probabilities)
    results = {'common_temperature': temperature, 'neural_tensor_sha256': before,
               'n': len(dev), 'games': int(dev.game_pk.nunique()), 'metrics': {}, 'sampled_minus_exact': {},
               'probability_l1_to_exact': {}, 'regime_selection': 'none; fixed-temperature diagnostic only'}
    for name in ['draw25', 'draw100', 'draw400', 'exact']:
        results['metrics'][name] = {regime: classification_metrics(y, probabilities[name + '_' + regime])
                                   for regime in ['raw', 'fixed_temperature']}
        if name != 'exact':
            results['sampled_minus_exact'][name] = aggregate_comparison(y, probabilities[name + '_fixed_temperature'],
                                                                       probabilities['exact_fixed_temperature'], games, [42])
            results['probability_l1_to_exact'][name] = {regime: l1_diagnostic(probabilities[name + '_' + regime], probabilities['exact_' + regime])
                                                      for regime in ['raw', 'fixed_temperature']}
    results['artifact_hashes'] = {name: hash_file(output / name) for name in ['exact_dev_logits.npz', 'predictions.npz', 'data.json']}
    dump(output / 'results.json', results)
    sources_end = {rel: hash_file(PROJECT / rel) for rel in sources}
    references_end = {path: hash_file(Path(path)) for path in identities}
    valid = sources_end == sources and references_end == identities
    dump(output / 'runtime.json', {'finished_at_utc': datetime.now(timezone.utc).isoformat(),
         'seconds': time.perf_counter()-started, 'device': model.device, 'integrity_ok': valid,
         'neural_tensor_sha256_end': tensor_hash(model), 'calibration_inference': False,
         'source_hashes_end': sources_end, 'reference_hashes_end': references_end})
    if not valid:
        raise RuntimeError('Source/reference identity changed during diagnostic')
    print('COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
