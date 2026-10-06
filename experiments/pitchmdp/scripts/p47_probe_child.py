"""P47 pre-registration fit probe. Reads and hashes TRAIN/earlystop payloads only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import gpu3_bench_child as bench
import run_cuda_direct as runner
from pitchmdp import batter_sidecar as bs
from pitchmdp import direct_pitcher_id_registry as pid

PROTOCOL = 'p47_train_earlystop_probe_v1'


def load_train_only(spec, config):
    """Metadata may describe all splits; payload access is restricted to these two."""
    export = Path(spec['export'])
    manifest = bench.verify_train_only(export, config, spec['pins']['export_manifest_sha256'])
    directory, sidecar = runner.batter_sidecar(export, config, manifest, hash_arrays=False)
    for split in ('train', 'earlystop'):
        item = sidecar['splits'][split]
        if runner.sha256_file(directory / item['path']) != item['sha256']:
            raise ValueError(f'{split}: sidecar payload hash differs')
    return manifest, directory, sidecar


def fit_task(spec, arm_name, seed, full):
    if spec.get('protocol') != PROTOCOL or spec['gpu_uuid'] != pid.RETAINED_GPU:
        raise ValueError('P47 GPU3 probe protocol required')
    expected = [(a, s, False) for s in pid.P47_SEEDS for a in pid.P47_ARM_ORDER]
    expected.append((pid.PITCHER_BATTER_ZERO_UNK, pid.P47_SEEDS[0], True))
    if (arm_name, seed, full) not in expected:
        raise ValueError('Task is not in the 16-mode plus full-TRAIN probe plan')
    config_path = Path(spec['config'])
    if runner.sha256_file(config_path) != spec['pins']['config_sha256']:
        raise ValueError('Draft config pin differs')
    config = runner.config_check(runner.read_json(config_path))
    if config['status'] != 'DRAFT' or config[pid.DESIGN_KEY] != pid.P47_DESIGN:
        raise ValueError('Only the P47 DRAFT may be probed')
    if config['device']['allowed_gpu_uuids'] != [pid.RETAINED_GPU] or config['device']['max_parallel'] != 1:
        raise ValueError('Single GPU3 contract differs')
    sources = runner.own_hashes()
    if sources != spec['pins']['source_sha256']:
        raise ValueError('Source pin differs')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != pid.RETAINED_GPU:
        raise ValueError('Supervisor did not bind GPU3')
    runtime = runner.bind_gpu(pid.RETAINED_GPU, config['device']['allowed_gpu_uuids'])
    started = time.perf_counter()
    manifest, directory, sidecar = load_train_only(spec, config)
    verified_payloads = {
        split: {'export': {name: meta['sha256'] for name, meta in manifest['splits'][split]['files'].items()},
                'batter': sidecar['splits'][split]['sha256']}
        for split in ('train', 'earlystop')}
    arm = runner.arm_spec(config, arm_name)
    limit_train = None if full else spec['workload']['train_rows']
    limit_early = None if full else spec['workload']['earlystop_rows']

    def load(split, limit):
        arrays = runner.load_split(spec['export'], manifest, split, limit)
        batter = bs.load(directory, sidecar, split, limit)
        if len(batter) != len(arrays['y']):
            raise ValueError(f'{split}: sidecar length mismatch')
        return {**arrays, 'batter': batter}

    train, subset = runner.train_subset(config, arm, load('train', limit_train))
    early = load('earlystop', limit_early)
    vocab = runner.pitcher_vocab(arm, manifest, train)
    shares = {}

    def inputs(split, arrays):
        base = runner.arm_inputs(arrays, arm, config['export']['workload_columns'])
        if vocab is None:
            return base
        from pitchmdp.direct_pitcher_batter_id_model import append_indices
        indices, shares[split] = runner.p43_indices(arrays, vocab, split)
        return append_indices(base, indices)

    names = runner.verified_context_names(config, manifest) if runner.needs_context_names(arm) else None
    model = runner.build_model(arm, seed, 'cuda', names, None if vocab is None else runner.vocab_ids_of(vocab))
    x, ex = inputs('train', train), inputs('earlystop', early)
    fit = bench.fit_and_record(model, x, np.asarray(train['y']), ex, np.asarray(early['y']), seed=seed,
                               epochs=2, batch_size=config['training']['batch_size'],
                               learning_rate=config['training']['learning_rate'])
    if runner.own_hashes() != sources:
        raise RuntimeError('Source changed during probe')
    identity = runner.p43_fit_record(model, vocab, shares) if vocab is not None else None
    return {**fit, 'protocol': PROTOCOL, 'arm': arm_name, 'full_train': full, 'train_subset': subset,
            'selected_rows_expected': {'train': int(len(train['y'])), 'earlystop': int(len(early['y']))},
            'id_record': identity, 'source_sha256': sources, 'runtime': runtime, 'train_only': True,
            'splits_loaded': ['train', 'earlystop'], 'export_manifest_sha256': spec['pins']['export_manifest_sha256'],
            'sidecar_sha256': config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'],
            'verified_payload_sha256': verified_payloads,
            'train_keys_sha256': hashlib.sha256(np.ascontiguousarray(train['keys']).tobytes()).hexdigest(),
            'earlystop_keys_sha256': hashlib.sha256(np.ascontiguousarray(early['keys']).tobytes()).hexdigest(),
            'peak_device_bytes': runner.peak_device_bytes('cuda'), 'total_seconds': time.perf_counter() - started,
            'dev_used_for_fit_or_selection': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--full', action='store_true')
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args()
    result = fit_task(json.loads(args.spec.read_text()), args.arm, args.seed, args.full)
    with args.result.open('x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
