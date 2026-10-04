"""Fake robust-clip probe worker: writes worker-<uuid>.json with timing-only records; exits PZ_FAKE_EXIT_<uuid> (default 0)."""
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
out, uuid = Path(argv[argv.index('--output') + 1]), argv[-1]
records = []
for arm in ['residual_counts', 'robust_clip_residual_counts']:
    r = {'arm': arm, 'seed': 39, 'prepare_seconds': 2., 'predict_seconds': 3., 'measured_seconds_per_epoch': 5.}
    if arm.startswith('robust_clip'):
        r['rc'] = {'compute_stats_seconds': 1.}
    records.append(r)
(out / f'worker-{uuid}.json').write_text(json.dumps({'gpu': uuid, 'startup_seconds': 10., 'arms': records}))
sys.exit(int(os.environ.get('PZ_FAKE_EXIT_' + uuid.replace('-', '_'), '0')))
