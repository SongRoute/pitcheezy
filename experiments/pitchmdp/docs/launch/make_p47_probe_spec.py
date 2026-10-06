"""Write a fresh, reviewable P47 TRAIN-only probe spec; this command never reads data payloads."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts')]
import p47_probe
import run_cuda_direct as runner
from pitchmdp import direct_pitcher_id_registry as pid


def build(config_path, export, out, python, ledger_review, independent_review):
    config_path, export, out = Path(config_path), Path(export), Path(out)
    config = runner.config_check(runner.read_json(config_path))
    if config['status'] != 'DRAFT' or config[pid.DESIGN_KEY] != pid.P47_DESIGN:
        raise ValueError('The P47 DRAFT is required')
    if not ledger_review or not independent_review:
        raise ValueError('Review evidence paths are required')
    return {'protocol': p47_probe.PROTOCOL, 'gpu_uuid': pid.RETAINED_GPU, 'concurrent_slots': 1,
            'lock_dir': p47_probe.LOCK_DIR, 'python': str(python), 'config': str(config_path),
            'export': str(export), 'out': str(out), 'tasks': p47_probe.tasks(),
            'workload': {'train_rows': 8192, 'earlystop_rows': 2048, 'epochs': 2},
            'budget': {'task_timeout_seconds': 600, 'total_wall_cap_seconds': 3600},
            'resources': {'host_bytes_per_child': 10737418240, 'host_margin_bytes': 8589934592,
                          'vram_bytes_per_child': 4294967296, 'vram_margin_bytes': 2147483648},
            'wait_for_terminal': [],
            'absent_before': [str(out.parent / 'EXP-P47-001.json'), str(out.parent / 'queue-p47-001'),
                              str(out.parent / 'runs/EXP-P47-001'), str(out.parent / 'score-p47-state.json')],
            'config_export_manifest_sha256': config['export']['manifest_sha256'],
            'pins': {'config_sha256': runner.sha256_file(config_path),
                     'export_manifest_sha256': config['export']['manifest_sha256'],
                     'sidecar_sha256': config[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'],
                     'source_sha256': runner.own_hashes(), 'probe_code_sha256': p47_probe.code_hashes(),
                     'seed_id_ledger_review': ledger_review, 'independent_review': independent_review}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--ledger-review', required=True)
    p.add_argument('--independent-review', required=True)
    p.add_argument('--spec', type=Path, required=True)
    args = p.parse_args()
    value = build(args.config, args.export, args.out, args.python, args.ledger_review, args.independent_review)
    with args.spec.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(args.spec)


if __name__ == '__main__':
    main()
