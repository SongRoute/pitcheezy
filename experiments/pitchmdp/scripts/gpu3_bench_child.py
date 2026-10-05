"""One TRAIN-only benchmark fit for gpu3_two_slot.py (AUTHOR ONLY; launched only by that supervisor).

Runs the SAME frozen reviewed route as run_cuda_direct.run_task up to and including model.fit — bind_gpu,
config_check, export manifest check, load_split('train'/'earlystop'), train_subset, arm_inputs, build_model and the
model's own fit — with the predeclared epochs and patience == epochs (fixed optimizer steps).  It never loads the
temperature/blend/dev/mlb_dev splits (their arrays are not even hashed), never scores, saves no model, and does not
touch the GPU lock (the parent supervisor owns it).  The only addition is read-only evidence: input hashes, the first
epoch permutation hash (separate RNG instance), a one-shot hash of the initial parameters taken by a global optimizer
pre-step hook before the first update, and per-tensor state_dict hashes after fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import run_cuda_direct as runner  # frozen reviewed runner; imported, never copied


def array_sha256(arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(f'{a.dtype}{a.shape}'.encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def tensor_hashes(state_dict):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() + f':{v.dtype}{tuple(v.shape)}'
            for k, v in state_dict.items()}


def fit_and_record(model, train_inputs, y, early_inputs, ey, *, seed, epochs, batch_size, learning_rate):
    """Call the model's own fit with patience == epochs; return comparable evidence (timing kept separate)."""
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    initial = {}

    def first_step(optimizer, args, kwargs):  # read-only, once: parameters before the first update
        if not initial:
            params = [p for group in optimizer.param_groups for p in group['params']]
            initial['sha256'] = hashlib.sha256(b''.join(p.detach().cpu().contiguous().numpy().tobytes()
                                                        for p in params)).hexdigest()
    handle = register_optimizer_step_pre_hook(first_step)
    try:
        start = time.perf_counter()
        model.fit(train_inputs, y, early_inputs, ey, epochs=epochs, patience=epochs, batch_size=batch_size,
                  learning_rate=learning_rate)
        seconds = time.perf_counter() - start
    finally:
        handle.remove()
    tensors = tensor_hashes(model.net.state_dict())
    report = model.report
    return {'seed': seed, 'initial_param_sha256': initial.get('sha256'),
            'first_permutation_sha256': hashlib.sha256(np.random.default_rng(seed).permutation(len(y)).tobytes()).hexdigest(),
            'tensor_sha256': tensors,
            'state_dict_sha256': hashlib.sha256(json.dumps(tensors, sort_keys=True).encode()).hexdigest(),
            'history': report['history'], 'best_epoch': report['best_epoch'], 'epochs_run': report['epochs_run'],
            'optimizer_updates': report['optimizer_updates'], 'training_rows': report['training_rows'],
            'earlystop_rows': report['earlystop_rows'],
            'train_inputs_sha256': array_sha256([*train_inputs, y]), 'earlystop_inputs_sha256': array_sha256([*early_inputs, ey]),
            'fit_seconds': seconds, 'report_seconds': report['seconds'], 'torch_threads': torch.get_num_threads()}


def verify_train_only(export, config, manifest_sha256):
    """Manifest checks of the frozen runner without hashing other splits, then hash only train/earlystop arrays."""
    manifest = runner.verify_export(export, config, manifest_sha256, hash_arrays=False)
    for split in ('train', 'earlystop'):
        for name, meta in manifest['splits'][split]['files'].items():
            if runner.sha256_file(Path(export) / meta['path']) != meta['sha256']:
                raise ValueError(f'{split}/{name}: array hash differs from the manifest')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--result', type=Path, required=True)
    args = p.parse_args(argv)
    started = time.perf_counter()
    if args.result.exists():
        raise FileExistsError(f'{args.result} exists; results are never overwritten')
    spec = json.loads(args.spec.read_text())
    if {'arm': args.arm, 'seed': args.seed} not in spec['tasks']:
        raise ValueError('Task is not in the predeclared benchmark')
    config_path = Path(spec['config'])
    if runner.sha256_file(config_path) != spec['pins']['config_sha256']:
        raise ValueError('Config differs from the pinned SHA256')
    config = runner.config_check(runner.read_json(config_path))
    sources = runner.own_hashes()
    if sources != spec['pins']['source_sha256']:
        raise ValueError('Frozen runner/model sources differ from the pinned map')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu_uuid']:
        raise ValueError('Supervisor did not bind GPU3 through CUDA_VISIBLE_DEVICES')
    runtime = runner.bind_gpu(spec['gpu_uuid'], config['device']['allowed_gpu_uuids'])
    manifest = verify_train_only(spec['export'], config, spec['pins']['export_manifest_sha256'])
    arm = runner.arm_spec(config, args.arm)
    columns = config['export']['workload_columns']
    w = spec['workload']
    train, _ = runner.train_subset(config, arm, runner.load_split(spec['export'], manifest, 'train', w['train_rows']))
    early = runner.load_split(spec['export'], manifest, 'earlystop', w['earlystop_rows'])
    names = runner.verified_context_names(config, manifest) if runner.needs_context_names(arm) else None
    model = runner.build_model(arm, args.seed, 'cuda', names)
    train_inputs, early_inputs = runner.arm_inputs(train, arm, columns), runner.arm_inputs(early, arm, columns)
    load_seconds = time.perf_counter() - started
    spec_training = config['training']
    result = fit_and_record(model, train_inputs, np.array(train['y']), early_inputs, np.array(early['y']),
                            seed=args.seed, epochs=w['epochs'], batch_size=spec_training['batch_size'],
                            learning_rate=spec_training['learning_rate'])
    if runner.own_hashes() != sources:
        raise RuntimeError('Sources changed during the benchmark fit')
    result.update(arm=args.arm, source_sha256=sources, runtime=runtime, train_only=True,
                  splits_loaded=['train', 'earlystop'], load_seconds=load_seconds,
                  peak_device_bytes=runner.peak_device_bytes('cuda'), total_seconds=time.perf_counter() - started)
    with open(args.result, 'x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print('GPU3_BENCH_CHILD_COMPLETE', args.arm, args.seed, f'{result["total_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
