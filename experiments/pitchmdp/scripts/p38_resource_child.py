"""One TRAIN-only fixed-step P38 resource-measurement fit (protocol p38_train_fixed_step_resource_v2; launched only by
p38_resource_measure.py under the common GPU3 lock).

Same frozen route as run_cuda_direct.run_task up to and including model.fit, including the pitcher-ID TRAIN vocabulary
(`pitcher_vocab`, `id_inputs`, `build_model(..., vocab ids)`), with the predeclared epochs and patience == epochs (fixed
optimizer steps).  Reuses gpu3_bench_child (train/earlystop-only manifest check, fit_and_record) unchanged.  Never loads
temperature/blend/dev/mlb_dev, never scores, saves no model, never touches the GPU lock.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import gpu3_bench_child as bench  # reviewed TRAIN-only helpers (2ffe51f/7566828/da74997), imported, not copied
import run_cuda_direct as runner  # frozen reviewed runner; imported, never copied

PROTOCOL = 'p38_train_fixed_step_resource_v2'


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
    if spec.get('protocol') != PROTOCOL or {'arm': args.arm, 'seed': args.seed} not in spec['tasks']:
        raise ValueError('Task is not in the predeclared P38 resource measurement')
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
    manifest = bench.verify_train_only(spec['export'], config, spec['pins']['export_manifest_sha256'])
    arm = runner.arm_spec(config, args.arm)
    columns = config['export']['workload_columns']
    w = spec['workload']
    train, _ = runner.train_subset(config, arm, runner.load_split(spec['export'], manifest, 'train', w['train_rows']))
    early = runner.load_split(spec['export'], manifest, 'earlystop', w['earlystop_rows'])
    vocab = runner.pitcher_vocab(arm, manifest, train)
    unk_share = {}

    def inputs(split, arrays):  # exactly run_task's rule: pitcher-ID arms append the TRAIN-vocabulary index column
        if vocab is None:
            return runner.arm_inputs(arrays, arm, columns)
        from pitchmdp.direct_pitcher_id_model import id_inputs
        arm_arrays, unk_share[split] = id_inputs(runner.arm_inputs(arrays, arm, columns), arrays['pitcher'], vocab['ids'])
        return arm_arrays
    names = runner.verified_context_names(config, manifest) if runner.needs_context_names(arm) else None
    model = runner.build_model(arm, args.seed, 'cuda', names, None if vocab is None else vocab['ids'])
    train_inputs, early_inputs = inputs('train', train), inputs('earlystop', early)
    load_seconds = time.perf_counter() - started
    training = config['training']
    result = bench.fit_and_record(model, train_inputs, np.array(train['y']), early_inputs, np.array(early['y']),
                                  seed=args.seed, epochs=w['epochs'], batch_size=training['batch_size'],
                                  learning_rate=training['learning_rate'])
    if runner.own_hashes() != sources:
        raise RuntimeError('Sources changed during the measurement fit')
    vocab_out = None
    if vocab is not None:
        from pitchmdp.direct_pitcher_id_registry import vocab_record
        vocab_out = {**vocab_record(vocab), 'unk_share': unk_share}
    result.update(protocol=PROTOCOL, arm=args.arm, source_sha256=sources, runtime=runtime, train_only=True,
                  splits_loaded=['train', 'earlystop'], load_seconds=load_seconds, pitcher_id=vocab_out,
                  parameter_count=int(sum(p.numel() for p in model.net.parameters())),
                  peak_device_bytes=runner.peak_device_bytes('cuda'), total_seconds=time.perf_counter() - started,
                  dev_used_for_fit_or_selection=False)
    with open(args.result, 'x') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print('P38_RESOURCE_CHILD_COMPLETE', args.arm, args.seed, f'{result["total_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
