"""Compare frozen original and vendored prediction engines in separate Python processes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
APP = REPO/'apps/observer'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_requests(dataset):
    """First chronological PA per supported pitcher/batter-hand; no outcome ranking."""
    cases, seen = [], set()
    for game in sorted(dataset['games'], key=lambda item: (item['date'], item['id'])):
        for pa in sorted(game['plate_appearances'], key=lambda item: item['id']):
            request = pa['pitches'][0]['request']
            key = (request['pitcher_id'], request['batter_stand'])
            if key in seen:
                continue
            seen.add(key)
            cases.append({'pitch_id': pa['pitches'][0]['id'], 'request': request})
    return cases


def worker(args):
    # Called by -I interpreters. Only the chosen mode adds its approved source tree.
    if args.worker == 'original':
        sys.path[:0] = [str(REPO/'experiments/pitchmdp/scripts'), str(REPO/'experiments/pitchmdp')]
        from minimal_pitch_service import Engine
        engine = Engine(args.bundle, device='cpu')
    else:
        sys.path.insert(0, str(APP/'backend'))
        from observer_app.standalone_engine import load_engine
        engine = load_engine(args.bundle, runtime_root=args.runtime, device='cpu')
    import numpy as np
    import pandas as pd
    import scipy
    import torch
    from pitchmdp.game import terminal_values
    from pitchmdp.planner import solve_pa
    arrays, summaries = {}, []
    for index, case in enumerate(json.loads(args.cases.read_text())):
        row = engine.predict_counts(case['request'])
        for kind, probabilities in row['probabilities'].items():
            arrays[f'case{index}_{kind}'] = probabilities
        counts = engine.metadata.get('repertoire_counts', {}).get(str(row['pitcher_id']))
        weights = np.array([counts.get(kind, 0) for kind in row['pitch_types']], dtype=float) if counts is not None else None
        if weights is not None:
            weights /= weights.sum()
        terminal = terminal_values(row['state'], engine.we, engine.advancement)
        plan = solve_pa(row['probabilities']['blend'], terminal, [0]*len(row['pitch_types']), baseline_policy=weights)
        for kind in ('values','q_values','baseline_values','myopic_q_values','policy'):
            arrays[f'case{index}_{kind}'] = getattr(plan, kind)
        arrays[f'case{index}_rankings'] = np.argsort(-plan.q_values, axis=-1, kind='stable')
        summaries.append({'pitch_id': case['pitch_id'], 'pitch_types': row['pitch_types'],
                          'delivery_tiers': row['delivery_tiers'], 'terminal_values': terminal,
                          'solver': plan.diagnostics})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'predictions.npz', **arrays)
    scientific = ('pitchmdp', 'minimal_pitch_service', 'representation_adapters', 'run_sequence_frequency_baselines',
                  'run_sequence_context_frequency', 'diagnose_sequence_legality')
    module_paths = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                    if getattr(module, '__file__', None) and
                    any(name == prefix or name.startswith(prefix+'.') for prefix in scientific)}
    package_paths = {module.__name__: str(Path(module.__file__).resolve()) for module in [np, pd, scipy, torch]}
    if args.worker == 'standalone':
        if any(not Path(path).is_relative_to(args.runtime.resolve()) for path in module_paths.values()):
            raise RuntimeError('Standalone scientific import escaped generated runtime')
        if any(not Path(path).is_relative_to(Path(sys.prefix).resolve()) for path in package_paths.values()):
            raise RuntimeError('Standalone dependency loaded outside its own virtual environment')
        if any('experiments/pitchmdp' in str(path) or '/.venv/' in str(path) for path in sys.path):
            raise RuntimeError('Standalone sys.path contains forbidden research/original environment path')
    report = {'cases': summaries, 'mode': args.worker, 'sys_prefix': sys.prefix,
              'python': sys.version, 'sys_path': sys.path, 'scientific_module_paths': module_paths,
              'scientific_package_paths': package_paths,
              'versions': {module.__name__: module.__version__ for module in [np, pd, scipy, torch]},
              'prediction_sha256': sha256(args.output/'predictions.npz')}
    (args.output/'report.json').write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['original','standalone'])
    parser.add_argument('--reference-python', type=Path)
    parser.add_argument('--standalone-python', type=Path)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--runtime', type=Path, default=APP/'runtime_src')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if not all([args.reference_python, args.standalone_python, args.dataset]):
        parser.error('Comparison requires both Python executables and an actual replay dataset')
    dataset = json.loads(args.dataset.read_text())
    manifest = json.loads(args.dataset.with_name('dataset_manifest.json').read_text())
    if sha256(args.dataset) != manifest['sha256']:
        raise ValueError('Dataset hash mismatch')
    cases = select_requests(dataset)
    if not cases:
        raise ValueError('At least one supported replay request required')
    args.output.mkdir(parents=True, exist_ok=False)
    case_path = args.output/'cases.json'
    case_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    sources = {'comparator': sha256(Path(__file__)), 'runtime_manifest': sha256(args.runtime/'runtime_manifest.json'),
               'bundle_manifest': sha256(args.bundle/'bundle_manifest.json'), 'cases': sha256(case_path)}
    (args.output/'config.json').write_text(json.dumps({'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_sha256': sha256(args.dataset), 'selection': 'first chronological PA per pitcher/batter-hand; all12count states',
        'tolerance': {'atol': 1e-12, 'rtol': 0.}, 'source_hashes': sources, 'original_frozen_bundle_modified': False}, indent=2))
    for mode, executable in [('original', args.reference_python), ('standalone', args.standalone_python)]:
        command = [str(executable), '-I', str(Path(__file__).resolve()), '--worker', mode,
                   '--bundle', str(args.bundle), '--runtime', str(args.runtime), '--cases', str(case_path),
                   '--output', str(args.output/mode)]
        with (args.output/(mode+'.log')).open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    import numpy as np
    comparison = {}
    with np.load(args.output/'original/predictions.npz') as old, np.load(args.output/'standalone/predictions.npz') as new:
        if old.files != new.files:
            raise ValueError('Prediction/plan archive keys differ')
        for key in old.files:
            np.testing.assert_allclose(new[key], old[key], rtol=0., atol=1e-12, err_msg=key)
            comparison[key] = float(np.max(np.abs(new[key]-old[key])))
    original = json.loads((args.output/'original/report.json').read_text())
    standalone = json.loads((args.output/'standalone/report.json').read_text())
    if len(original['cases']) != len(standalone['cases']):
        raise ValueError('Regression case count differs')
    for old_case, new_case in zip(original['cases'], standalone['cases']):
        for key in ('pitch_id', 'pitch_types', 'delivery_tiers'):
            if old_case[key] != new_case[key]:
                raise ValueError('Pitch identity/type order or delivery tiers changed: '+key)
        for group in ('terminal_values', 'solver'):
            if old_case[group].keys() != new_case[group].keys():
                raise ValueError('Result metadata keys changed: '+group)
            for key, value in old_case[group].items():
                if isinstance(value, (int, float)):
                    np.testing.assert_allclose(new_case[group][key], value, rtol=0., atol=1e-12)
                elif new_case[group][key] != value:
                    raise ValueError('Result metadata changed: '+group+'/'+key)
    end_sources = {'comparator': sha256(Path(__file__)), 'runtime_manifest': sha256(args.runtime/'runtime_manifest.json'),
                   'bundle_manifest': sha256(args.bundle/'bundle_manifest.json'), 'cases': sha256(case_path)}
    if sources != end_sources:
        raise RuntimeError('Equivalence inputs changed during verification')
    result = {'passed': True, 'cases': len(cases), 'count_states_per_case': 12,
              'max_absolute_difference': max(comparison.values()), 'arrays': comparison,
              'strict_tolerance': 1e-12, 'module_isolation_verified': True,
              'scope': 'Same-machine isolated environment regression; not a cross-platform portability claim'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ['passed','cases','max_absolute_difference','module_isolation_verified']}))


if __name__ == '__main__':
    main()
