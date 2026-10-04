"""Fake probe worker: writes worker-<uuid>.json with timing-only records; exits PZ_FAKE_EXIT_<uuid> (default 0)."""
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
out, uuid = Path(argv[argv.index('--output') + 1]), argv[-1]
arms = ['control', 'periodic_residual_counts', 'ple_residual_counts', 'tabm_residual_counts', 'ple_tabm_residual_counts']
records = []
for arm in arms:
    r = {'arm': arm, 'seed': 34, 'prepare_seconds': 2., 'predict_seconds': 3., 'measured_seconds_per_epoch': 5.}
    if arm.startswith('ple'):
        r['ple'] = {'bins_seconds': 1.}
    records.append(r)
(out / f'worker-{uuid}.json').write_text(json.dumps({'gpu': uuid, 'startup_seconds': 10., 'arms': records}))
sys.exit(int(os.environ.get('PZ_FAKE_EXIT_' + uuid.replace('-', '_'), '0')))
