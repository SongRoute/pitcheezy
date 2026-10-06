"""Generate the UNDATED P47 preparation DRAFT only; no launch/registration/real-data IO."""
import copy
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/pitchmdp'))
from pitchmdp import direct_pitcher_id_registry as pid

SOURCE = REPO / 'configs/DRAFT-CUDA-BATTER-ID-20261006.json'
SOURCE_SHA256 = 'bc94ae86a2482c920ebd3cb8a898774bcbbb4c5862d58639556621f4b7a442e2'
OUT = REPO / 'configs/DRAFT-CUDA-BATTER-DROPOUT-20261006.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P47-001-batter-dropout-2026-10-06.md'


def build():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('P43 DRAFT source differs from its pin')
    source = json.loads(SOURCE.read_text())
    cfg = {k: copy.deepcopy(source[k]) for k in
           ('protocol', 'export', 'training', 'calibration', 'data_rules', 'device', pid.BATTER_SIDECAR_KEY)}
    cfg.update(experiment_id='EXP-P47-001', status='DRAFT', **{pid.MARKER: True, pid.DESIGN_KEY: pid.P47_DESIGN},
               decision='None. Preparation only; no smoke, timing, main fit, registration or real-data scoring authorized.',
               purpose='Post-P40/P43 adaptive repeated-DEV exploratory cold-start repair. ' + PREREG,
               source_configs={'p43_draft': {'path': str(SOURCE.relative_to(REPO)), 'sha256': SOURCE_SHA256}},
               control_reuse='none; all four arms refit at matched seeds 80-83; no cross-family reuse or pooling',
               seeds=list(pid.P47_SEEDS), fits=pid.P47_FITS,
               seed_rule='80-83 reserved for this DRAFT; new RNG, not new data; 0-79 used/reserved.',
               arms=[copy.deepcopy(pid.BASELINES[n]) for n in pid.P47_ARM_ORDER],
               pitcher_id_data_rule=source['pitcher_id_data_rule'].replace(
                   'Row shuffle uses only the split\'s own batter column and a fixed seed; scorer subsets '
                   'use the TRUE batter\'s TRAIN vocabulary membership for every arm.',
                   'No row shuffle. Strata use TRUE pitcher and batter TRAIN membership; dropout is training-only. '
                   'The frozen-zero UNK gate applies in both training and evaluation.'),
               comparison={'reference_arm': pid.REFERENCE, 'scorer': source['comparison']['scorer'],
                           pid.CONTRAST_KEY: copy.deepcopy(pid.P47_CONTRASTS),
                           'scope': 'Five unique arm contrasts on every TRUE-ID 2x2 cell, both batter margins, and '
                                    'all rows; both dev and mlb_dev; each seed and ensemble. Two primary '
                                    'contrast/split/subset triples; all other comparisons secondary. NLL primary, '
                                    'Brier, accuracy, top-label ECE10; no multiplicity adjustment or verdict.',
                           'not_compared': 'other families, MPS scores, G15, G0 policy, OPE'},
               queue={**copy.deepcopy(source['queue']),
                      'batch': 'P47 preparation DRAFT: four arms x four seeds = 16 fits, cap 60',
                      'start_after': 'Distinct review and Root acceptance; new dated window; UUID-only GPU3 '
                                     'preflight; common exclusive lock; TRAIN-only smoke and full-TRAIN timing '
                                     'reviewed; formal registration before any real DEV scoring or main fit.',
                      'clock_rule': 'No clock selected in preparation. Fit projection 4894 s (81.57 min), then '
                                    '15 min stop grace + 20 min scoring/report reserve. Preflight time extra.',
                      'stop_rule': 'Plan only: inherited max_attempts and budgets unchanged; no partial-family scoring.'},
               exposure='Same export 49097109 and repeatedly exposed <=2025 DEV as P24-P46. Adaptive after P40/P43. '
                        'No new data, independent confirmation, model promotion, 2026 reads or policy evaluation.',
               execution={'enabled': False, 'executable': False,
                          'blocked_until': ['distinct reviewer fixed-source review', 'Root acceptance',
                                            'fresh dated window and TRAIN-only preflight protocol review',
                                            'GPU3 exclusive-lock smoke and timing PASS',
                                            'registration before real DEV scoring or main fit']})
    return cfg


if __name__ == '__main__':
    with OUT.open('x') as handle:
        handle.write(json.dumps(build(), indent=2, ensure_ascii=False) + '\n')
    print(OUT)
