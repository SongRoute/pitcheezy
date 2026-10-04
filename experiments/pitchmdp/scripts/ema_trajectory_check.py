"""Raw-trajectory identity check of a finished EMA family (DRAFT-CUDA-EMA-20261004; review F6).

The EMA arms consume no randomness, so per seed their RAW trajectory should equal the
base arm's: ema vs control, dropout_ema vs dropout.  Over the overlapping epochs of
fit.json report.history this checks

  candidate.train_weighted_nll            == base.train_weighted_nll
  candidate.raw_earlystop_conditional_nll == base.earlystop_conditional_nll

(in EMA arms earlystop_conditional_nll is the EMA score, so it is never compared).
Every task is first verified against its expected identity (runner verify_task) and
all tasks must share one source map.  Reads identity/state and fit.json only (TRAIN
and early-stop losses): never predictions, DEV or mlb_dev.  Prints one JSON report
(exact flag and max absolute difference per pair and seed, device UUIDs); exit 1 when
any pair differs, in which case the design-note claim becomes "same on CPU tests;
CUDA within X".  Not a score, ranking or verdict.

  PYTHONPATH=. python scripts/ema_trajectory_check.py --config <cfg> \
      --export-manifest-sha256 <sha> --family <runner --output> [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402  numpy only at import

FORMAT = 'ema_trajectory_check_v1'
PAIRS = (('ema', 'control'), ('dropout_ema', 'dropout'))
COMPARED = (('train_weighted_nll', 'train_weighted_nll'), ('raw_earlystop_conditional_nll', 'earlystop_conditional_nll'))


def compare(candidate, base):
    """Exactness and max |difference| of the raw trajectory over the overlapping epochs."""
    n = min(len(candidate), len(base))
    if n < 1:
        raise ValueError('No overlapping epochs')
    for i in range(n):
        if candidate[i]['epoch'] != i + 1 or base[i]['epoch'] != i + 1:
            raise ValueError('History epochs are not 1..n')
    diffs = {c: max(abs(float(candidate[i][c]) - float(base[i][b])) for i in range(n)) for c, b in COMPARED}
    exact = all(float(candidate[i][c]) == float(base[i][b]) for i in range(n) for c, b in COMPARED)
    return {'epochs_compared': n, 'epochs_run': [len(candidate), len(base)], 'exact': exact, 'max_abs_diff': diffs}


def check(config, family, manifest_sha256, smoke, device='cuda', seeds=None):
    config = rc.config_check(config)
    if not rc.uses_ema(config):
        raise ValueError('Not the EMA family')
    seeds = config['seeds'] if seeds is None else list(seeds)
    if not seeds or any(s not in config['seeds'] for s in seeds):
        raise ValueError('Unregistered seed')
    sources, pairs = None, {}
    for candidate, base in PAIRS:
        for seed in seeds:
            fits = {}
            for arm in (candidate, base):
                dest = rc.task_dir(family, arm, seed, smoke)
                state = rc.verify_task(dest, rc.expected_identity(config, manifest_sha256, arm, seed, smoke, device))
                if sources is None:
                    sources = state['identity']['source_sha256']
                elif state['identity']['source_sha256'] != sources:
                    raise ValueError(f'{arm}/{seed}: source map differs within the family')
                fits[arm] = rc.read_json(Path(dest) / 'fit.json')
            if fits[candidate]['report'].get('weights') != 'ema' or fits[base]['report'].get('weights') == 'ema':
                raise ValueError(f'{candidate}/{seed}: pair is not EMA arm vs raw base arm')
            pairs[f'{candidate}_vs_{base}/{seed}'] = {
                **compare(fits[candidate]['report']['history'], fits[base]['report']['history']),
                'device_uuid': [(fits[a].get('runtime') or {}).get('device_uuid') for a in (candidate, base)]}
    return {'format': FORMAT, 'experiment_id': config['experiment_id'], 'smoke': smoke, 'device': device,
            'config_sha256': rc.config_digest(config), 'source_sha256': sources,
            'all_exact': all(p['exact'] for p in pairs.values()), 'pairs': pairs,
            'reads': 'identity/state and fit.json TRAIN / early-stop losses only; no predictions, DEV or mlb_dev'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--export-manifest-sha256', required=True)
    p.add_argument('--family', type=Path, required=True, help='runner --output directory (arm[-smoke]/seed/...)')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if args.smoke != args.family.resolve().name.endswith('-smoke'):
        raise ValueError('Smoke families (only) end -smoke')
    report = check(rc.read_json(args.config), args.family.resolve(), args.export_manifest_sha256, args.smoke)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report['all_exact'] else 1


if __name__ == '__main__':
    sys.exit(main())
