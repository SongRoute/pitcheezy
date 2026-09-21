"""Read-only result checks plus paired final forecast intervals; append audit JSON."""
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from pitchmdp.data import hash_file
from policy_evaluation_models import OutcomeCalibration
from run_service_validation import verify_sources, verify_files
from run_sequence_frequency_blend import paired_fixed_predictors
from run_sequence_pilot import dump


def main():
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to('/Volumes/T7 Shield'):
        raise ValueError('Mounted approved SSD required')
    output = root/'runs/service-validation-v2'
    verify_sources(output)
    verify_files(output, 'completion_hashes.json')
    result = json.loads((output/'results.json').read_text())
    calibration = json.loads((output/'calibration.json').read_text())
    with np.load(output/'evaluation_predictions.npz') as data:
        original = data['p']
        candidate = OutcomeCalibration(calibration['bias']).apply(original)
        selected = candidate if calibration['accepted'] else original
        intervals = {'selected_vs_original': paired_fixed_predictors(data['y'], selected, original, data['game_pk']),
                     'candidate_vs_original_even_if_rejected': paired_fixed_predictors(data['y'], candidate, original, data['game_pk'])}
    scores = result['policy']['comparisons']['original']
    primary = {name: row['game_interval'][0] > 0 for name, row in scores.items() if name != 'we_full_minus_re_full'}
    audit = {'source_hashes_ok': True, 'completion_artifact_hashes_ok': True,
             'forecast_intervals': intervals, 'primary_positive_lower_bounds': primary,
             'minimal_evaluator_gate_passed': result['minimal_evaluator_gate_passed'],
             'scope': 'Fixed-model game bootstrap; no additional model/period selection. Source period previously explored.',
             'audit_source_sha256': hash_file(Path(__file__))}
    if (output/'audit.json').exists():
        raise ValueError('Audit already exists')
    dump(output/'audit.json', audit)
    dump(output/'audit_hashes.json', {'audit.json': hash_file(output/'audit.json')})
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
