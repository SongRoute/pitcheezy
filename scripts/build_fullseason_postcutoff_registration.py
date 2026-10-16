"""Build an inactive fixed-six score registration from accepted collector metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from experiments.ws_fullseason.audit import sha as file_sha
from scripts.fullseason_postcutoff_offline import (ADAPTER_COMMIT, CONFIG_SHA, PLAN_SHA,
                                                  PROFILE_SHA, SIX)

SOURCE = (
    'experiments/ws_campaign/__init__.py', 'experiments/ws_campaign/models.py',
    'experiments/ws_campaign/source/abc_corpus.py',
    'experiments/ws_campaign/source/first_small_features.py',
    'experiments/ws_campaign/source/history_transfer.py',
    'experiments/ws_campaign/source/outcome_transfer.py',
    'experiments/ws_campaign/source/scale_runner.py',
    'experiments/ws_campaign/source/scale_safe.py',
    'experiments/ws_fullseason/audit.py', 'experiments/ws_fullseason/calibrate.py',
    'experiments/ws_fullseason/data.py', 'experiments/ws_fullseason/supervise_audit.py',
    'experiments/ws_seq/tm002.py', 'experiments/ws_seq/train_seq.py',
    'src/pitcheezy/__init__.py', 'src/pitcheezy/seq_shadow/__init__.py',
    'src/pitcheezy/seq_shadow/adapter.py', 'src/pitcheezy/seq_shadow/calibration.py',
    'src/pitcheezy/seq_shadow/fullseason.py',
    'src/pitcheezy/seq_shadow/fullseason_profiles.py',
    'src/pitcheezy/seq_shadow/fullseason_runtime.py',
    'src/pitcheezy/seq_shadow/fullseason_stream.py',
    'src/pitcheezy/seq_shadow/ledger.py',
    'src/pitcheezy/seq_shadow/postcutoff_offline.py',
    'src/pitcheezy/seq_shadow/summary.py',
    'src/pitcheezy/seq_shadow/tokens.py',
    'scripts/fullseason_postcutoff_offline.py',
    'scripts/verify_fullseason_postcutoff_offline.py',
)
OLD_REG = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261016-autonomous-profile-stream-005/forecast_persistence/FORECAST-PERSISTENCE-REGISTRATION.json')
SUPERVISOR = '/Users/song/Projects/pitcheezy-worktrees/codex-ws-metadata-supervisor-1013/scripts/supervise_2026_metadata.py'
HEAVY_LOCK = '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/.heavy.lock'


def build(manifest_path, output, *, root=ROOT, prior=OLD_REG):
    """Metadata only; never opens projected Parquet, CSV, model or profile payload."""
    root = Path(root).resolve(); manifest_path = Path(manifest_path).resolve(); output = Path(output)
    prior = json.loads(Path(prior).read_text())
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'postseason-six-exact-game-v1'
            or manifest.get('status') != 'COMPLETE'
            or [(r.get('game_pk'), r.get('officialDate')) for r in manifest.get('games', [])] != list(SIX.items())
            or any(r.get('gameType') != 'L' for r in manifest['games'])
            or manifest.get('projected', {}).get('file') != 'postseason_six.parquet'
            or prior['adapter_config']['sha256'] != CONFIG_SHA
            or prior['profile_bundle']['manifest_sha256'] != PROFILE_SHA):
        raise ValueError('fixed six/accepted artifact metadata required')
    proj = manifest['projected']; parquet = manifest_path.with_name(proj['file'])
    if (not parquet.is_file() or parquet.stat().st_size != proj['bytes']
            or type(proj['rows']) is not int or not 0 < proj['rows'] <= 12_000):
        raise ValueError('collector projection metadata/size')
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    pins = {str(root / name): file_sha(root / name) for name in SOURCE}
    pins[str(root / 'scripts/build_fullseason_postcutoff_registration.py')] = file_sha(
        root / 'scripts/build_fullseason_postcutoff_registration.py')
    reg = {'id': 'FS-POSTCUTOFF-FIXED-SIX-OFFLINE-SCORE-001', 'execute': False,
           'author': 'sol_service', 'source_commit': commit, 'source_pins': pins,
           'stage': 'POSTCUTOFF_FIXED_SIX_RETROSPECTIVE_SCORE',
           'analysis_plan_sha256': PLAN_SHA, 'adapter_commit': ADAPTER_COMMIT,
           'leaf_order': list(prior['leaf_order']), 'data_role': '2026 EXPOSED DEVELOPMENT',
           'asof': 'Corrected retrospective Statcast; source arrival UNKNOWN; prior exposure UNKNOWN; no prospective claim.',
           'source': {'manifest': {'path': str(manifest_path), 'bytes': manifest_path.stat().st_size,
                                  'sha256': file_sha(manifest_path)},
                      'projected': {'path': str(parquet), 'bytes': proj['bytes'], 'sha256': proj['sha256']}},
           'adapter_config': dict(prior['adapter_config'], bytes=Path(prior['adapter_config']['path']).stat().st_size),
           'recipe': prior['recipe'], 'model_artifacts': prior['model_artifacts'],
           'profile_bundle': dict(prior['profile_bundle'], serving_origin='2026-10-12',
                                  note='Runtime loads all four accepted archives; Oct12 final donor-cutoff snapshot answers Oct12-15 queries.'),
           'supervisor_helper': SUPERVISOR, 'heavy_lock': HEAVY_LOCK,
           'output': str(output), 'fence': str(output.parent / '.postcutoff-score-fence.json'),
           'bounds': {'wall_seconds': 1800, 'rss_bytes': 3 << 30,
                      'free_memory_bytes': 4 << 30, 'free_disk_bytes': 12 << 30,
                      'output_bytes': 20 << 20, 'threads': 1},
           'physical_access': 'Only the six-game completed manifest and one pinned 29-column Parquet, fixed T config/three model states/recipe/encoder/calibrator, accepted profile manifest and all four archives loaded by Runtime. No raw CSV/network/old attempt/other game/model fit.',
           'output_scope': 'Fresh owned score_rows.json private <=16MiB plus aggregate receipt/SUPERVISOR/worker log; no individual export.',
           'reviewer_output_scope': 'After completion, independent verifier reads only new score_rows.json and receipt.json, recomputes accepted-key digest and pooled/per-game NLL/Brier; aggregate verdict only.',
           'score_rules': 'Observed-type seven-leaf, source quality and complete prefix, up to32 same-game/pitcher prior tokens, relevant unknown floor, exact same-PA predecessor, fixed FIT marginal, NLL clip1e-12, seven-class Brier, simplex 1e-10.',
           'invocation': ['/Users/song/Projects/pitcheezy/.venv-observer/bin/python', '-I',
                          str(root / 'experiments/ws_fullseason/supervise_audit.py'), '--worker',
                          'fullseason_postcutoff_offline.py', '--registration', '<registration>', '--grant', '<grant>']}
    return reg


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--registration', required=True, type=Path)
    a = p.parse_args()
    reg = build(a.manifest, a.output)
    a.registration.write_text(json.dumps(reg, sort_keys=True, indent=2) + '\n')
