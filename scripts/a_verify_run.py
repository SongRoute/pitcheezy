"""Read-only integrity verification of the committed A run and frozen bundle."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    config_path = ROOT / 'configs/EXP-A-S0S1-001.json'
    config = json.loads(config_path.read_text())
    run = Path(config['artifact_dir'])
    report = ROOT / 'results/EXP-A-S0S1-001'
    result = json.loads((report / 'results.json').read_text())
    checks = {
        'result_copy': sha(report / 'results.json') == sha(run / 'results.json'),
        'config': sha(config_path) == result['evaluation_config_sha256'],
        'evaluation_code': sha(ROOT / 'scripts/a_small_eval.py') == result['evaluation_code_sha256'],
        'selection': sha(run / 'selection_manifest.json') == result['source_manifest_sha256'],
        'selection_copy': sha(report / 'selection_manifest.json') == result['source_manifest_sha256'],
    }
    filenames = {'baseline': 'april_count_baseline.pkl', 'predictions': 'predictions.npz',
                 'internal_we': 'internal_we_diagnostics.json', 'observer_zone': 'observer_zone_diagnostics.json',
                 'event_cases': 'event_cases.json', 'selected_pitch_keys': 'selected_pitch_keys.json'}
    for key, filename in filenames.items():
        checks[filename] = sha(run / filename) == result['output_sha256'][key]
    selection = json.loads((run / 'selection_manifest.json').read_text())
    for split, path in selection['dataset_paths'].items():
        checks[split] = sha(path) == selection['sha256'][split + '_full_games.parquet']
    bundle = Path(config['bundle_dir'])
    checks['bundle_manifest'] = sha(bundle / 'bundle_manifest.json') == selection['sha256']['bundle_manifest']
    for name, expected in json.loads((bundle / 'bundle_manifest.json').read_text())['sha256'].items():
        checks['bundle/' + name] = sha(bundle / name) == expected
    print(json.dumps({'all_checks_pass': all(checks.values()), 'checks': checks}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
