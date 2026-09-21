"""One evaluator repair in a new run; reuse the immutable final policy comparison."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import pandas as pd
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.data import hash_file
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.model import eligible, outcome_labels
from pitchmdp.sequence_model import classification_metrics
from regularized_policy_evaluator import RegularizedPolicyEvaluator
import run_service_validation as frozen
from run_sequence_pilot import dump


def full_grid(data):
    frame = ORIGINAL_GRID(data)
    state = data['state']
    frame['inning'], frame['inning_topbot'] = state.inning, state.half
    frame['home_score'], frame['away_score'] = state.home_score, state.away_score
    return frame


ORIGINAL_GRID = frozen.grid_from_data


def prepare(root, source, output, config):
    frozen.verify_sources(source)
    frozen.verify_files(source, 'diagnostic_hashes.json')
    if output.exists():
        raise ValueError('Use a new repair output directory')
    frame = add_batter_style_history(prepare_frame(config))
    old_config = json.loads((source/'config.json').read_text())
    if frame.attrs['sequence_data_identity'] != old_config['data_identity']:
        raise ValueError('Original diagnostic data changed')
    parts = frozen.partition(frame)
    july = parts['judge'].loc[eligible(parts['judge'])]
    print('FIT_REGULARIZED_EVALUATOR', len(july), flush=True)
    repaired = RegularizedPolicyEvaluator().fit(july)
    with (source/'judge.pkl').open('rb') as stream:
        judges = pickle.load(stream)
    judges['outcomes'], judges['count'] = repaired, repaired.parent
    diagnostics = json.loads((source/'diagnostics.json').read_text())
    for name in ('diagnosis', 'guard'):
        part = parts[name].loc[eligible(parts[name])]
        y = outcome_labels(part)
        diagnostics[name]['judge_outcomes'] = classification_metrics(y, frozen.legal(repaired.predict(part), part))
        diagnostics[name]['judge_count_baseline'] = classification_metrics(y, frozen.legal(repaired.parent.predict(part), part))
    diagnostics['evaluator_repair'] = {'candidate_count': 1, 'fit_report': repaired.report,
        'reason': 'Original high-cardinality evaluator failed count-baseline predictive gate',
        'source_run': str(source), 'source_diagnostic_manifest_sha256': hash_file(source/'diagnostic_hashes.json')}
    # Copy before modifying any run file. Never hard-link mutable pickle/JSON files.
    shutil.copytree(source, output)
    dump(output/'diagnostics_before_repair.json', json.loads((source/'diagnostics.json').read_text()))
    dump(output/'diagnostics.json', diagnostics)
    with (output/'judge.pkl').open('wb') as stream:
        pickle.dump(judges, stream)
    dump(output/'evaluator_repair.json', diagnostics['evaluator_repair'])
    fit = json.loads((output/'judge_fit.json').read_text())
    fit['outcomes'] = repaired.report
    dump(output/'judge_fit.json', fit)
    sources = json.loads((output/'source_hashes.json').read_text())
    for rel in ('scripts/regularized_policy_evaluator.py', 'scripts/run_repaired_service_validation.py', 'docs/SERVICE_EVALUATOR_REPAIR.md'):
        sources[rel] = hash_file(PROJECT/rel)
        target = output/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT/rel, target)
    dump(output/'source_hashes.json', sources)
    old_config['evaluator_repair'] = diagnostics['evaluator_repair']
    old_config['sources'] = sources
    dump(output/'config.json', old_config)
    frozen.file_manifest(output, 'diagnostic_hashes.json')
    print('REPAIRED_DIAGNOSIS_COMPLETE', output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'finalize'])
    parser.add_argument('--repair', choices=['none', 'class_bias'], default='class_bias')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(config['artifact_root']).resolve()
    output = (args.output or root/'runs/service-validation-v2').resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to('/Volumes/T7 Shield'):
        raise ValueError('Approved mounted SSD required')
    if not output.is_relative_to(root/'runs') or output == root/'runs' or Path(sys.executable).absolute() != Path(config['python']).absolute():
        raise ValueError('Use approved output and configured Python')
    if args.stage == 'prepare':
        prepare(root, root/'runs/service-validation-v1', output, config)
    else:
        # The new evaluator requires the complete pre-pitch game context. Extend
        # the old finalizer's grid at this adapter boundary without editing it.
        frozen.grid_from_data = full_grid
        frozen.finalize(root, output, args.repair)


if __name__ == '__main__':
    main()
