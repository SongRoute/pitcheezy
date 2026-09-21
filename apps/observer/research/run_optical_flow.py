"""Freeze and run one sparse-flow alternative on the unchanged four annotated windows."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from optical_flow_tracker import track_flow_video, compare_endpoint, REPO

ROOT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/observer-improvement-v2/video-lab')
SPEC_PATH = Path(__file__).with_name('optical_flow_spec.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-manifest', type=Path, default=ROOT/'seed-set-v1/seed_manifest.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    seed_manifest = json.loads(args.seed_manifest.read_text())
    if seed_manifest['source_spec_sha256'] != spec['seed_reference_spec_sha256'] or sha(seed_manifest['source_spec']) != spec['seed_reference_spec_sha256']:
        raise ValueError('Expected unchanged frozen four-window reference specification')
    if len(seed_manifest['windows']) != 4:
        raise ValueError('Exactly the original four windows required')
    paths = [Path(__file__), Path(__file__).with_name('optical_flow_tracker.py'), SPEC_PATH,
             REPO/'apps/observer/backend/observer_app/video_lab.py', args.seed_manifest,
             Path(seed_manifest['source_spec'])]
    annotations = []
    for window in seed_manifest['windows']:
        path = Path(window['annotation'])
        if sha(path) != window['annotation_sha256']:
            raise ValueError('Frozen seed annotation changed')
        annotation = json.loads(path.read_text())
        if annotation['label_source'] != 'assistant_visual_estimate' or annotation['review_status'] != 'unreviewed':
            raise ValueError('Retain original assistant/unreviewed label provenance')
        if annotation.get('calibration_corners') is not None:
            raise ValueError('Frozen seed set authorizes pixel tracking only')
        annotations.append(annotation)
        paths.append(path)
    hashes = {str(path.resolve()): sha(path) for path in paths}
    output = (args.output or ROOT/datetime.now(timezone.utc).strftime('optical-flow-v1-%Y%m%dT%H%M%SZ')).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('New local approved video-lab output required')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'config.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'spec': spec,
        'reference_hashes': hashes, 'source_and_thresholds_frozen_before_inference': True,
        'new_holdout': False, 'service_default_changed': False})
    shutil.copyfile(SPEC_PATH, output/'optical_flow_spec.json')
    records = []
    for window, annotation in zip(seed_manifest['windows'], annotations):
        result = track_flow_video(annotation, spec)
        # Reference is passed only AFTER tracking finishes, never into tracker inputs.
        comparison = compare_endpoint(result, window['endpoint_reference'], window['endpoint_uncertainty_pixels'])
        result['endpoint_reference_comparison'] = comparison
        write(output/(window['id']+'.json'), result)
        records.append({'id': window['id'], 'status': result['status'], 'abstain_reason': result['abstain_reason'],
                        'processed_frames': result['processed_frames'], 'expected_frames': result['expected_frames'],
                        'seed_points': result['seed_points'], 'comparison': comparison})
    if hashes != {str(path.resolve()): sha(path) for path in paths}:
        raise RuntimeError('Source, thresholds, or frozen annotation changed during tracking')
    complete = sum(item['status'] == 'tracked' for item in records)
    summary = {'windows': records, 'total': len(records), 'completed': complete, 'abstained': len(records)-complete,
        'endpoint_comparisons': complete,
        'within_assistant_reference_uncertainty_count': sum(item['comparison']['within_reference_uncertainty'] is True for item in records),
        'accuracy_claim': None, 'label_source': 'assistant_visual_estimate', 'review_status': 'unreviewed',
        'independent_test': False, 'integrity_ok': True, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'One post-hoc alternative on the same tiny exploratory four-window set; abstentions remain abstentions.'}
    write(output/'summary.json', summary)
    print(json.dumps({'output': str(output), 'total': len(records), 'completed': complete, 'abstained': len(records)-complete,
                      'within_reference_uncertainty_count': summary['within_assistant_reference_uncertainty_count']}))


if __name__ == '__main__':
    main()
