"""Export an existing real replay request and frozen Observer response for C0.

No training or performance scoring. Writes only the explicit task run directory.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['PITCHEEZY_OBSERVER_RUNTIME'] = 'standalone'
    os.environ['PITCHEEZY_OBSERVER_RUN'] = str(args.run.resolve())
    from observer_app.dataset import DemoDataset
    from observer_app.recommender import Recommender
    from observer_app.settings import ARTIFACT_ROOT, BUNDLE, CONFIG
    dataset_path = ARTIFACT_ROOT / 'runs/observer-improvement-v2/dataset.json'
    dataset = DemoDataset(dataset_path)
    game = min(dataset.games.values(), key=lambda x: (x['date'], x['id']))
    pa = min(game['plate_appearances'], key=lambda x: x['id'])
    pitch = pa['pitches'][0]
    assert pitch['request']['batter_profile']['as_of'] < pitch['request']['date']
    recommender = Recommender()
    result = recommender.recommend(pitch, pa)
    assert result['status'] == 'ready', result
    restored = Recommender()
    replay = restored.recommend(pitch, pa)
    assert result == replay
    changed = copy.deepcopy(pitch)
    changed['actual'] = {'pitch_type': 'POISON', 'x': 999, 'event': 'home_run'}
    assert restored.recommend(changed, pa) == result
    for candidate in result['candidates']:
        assert 0 <= candidate['value'] <= 1
        assert abs(candidate['delta_pp'] - 100 * (candidate['value'] - result['baseline_value'])) < 1e-9
    output = {
        'schema_version': 1, 'contract_version': 'C0-v1',
        'value_spec_version': 'defense-we-pa-v1',
        'purpose': 'real replay integration example; not performance evaluation',
        'source_dataset_sha256': sha(dataset_path),
        'source_game_id': game['id'], 'pitch_id': pitch['id'],
        'input': {'pitch': {'id': pitch['id'], 'request': pitch['request']},
                  'pa': {k: pa[k] for k in ('zone_bounds', 'repertoire_counts')}},
        'output': result, 'model_identity': recommender.identity,
        'bundle_manifest_sha256': sha(BUNDLE / 'bundle_manifest.json'),
        'runtime_manifest_sha256': sha(ROOT / 'apps/observer/runtime_src/runtime_manifest.json'),
        'config': CONFIG, 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': sha(__file__),
        'verification': {'bundle_restored': True, 'cache_round_trip_equal': result == replay,
                         'actual_field_poison_invariant': True, 'probability_to_pp_equal': True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pitch_id': pitch['id'], 'status': result['status'],
                      'model_identity': recommender.identity, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
