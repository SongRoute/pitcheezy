"""Write configs/DRAFT-CUDA-EMBED-TYPE-20261005.json (tentative EXP-P37-001 DRAFT) from the registered EXP-P36-001 config.

Only family fields change (marker, arms, seeds, fits, comparison, purpose, notes, queue placeholders, status DRAFT,
execution disabled, registration block removed); export, training (cap 60), calibration, device and data rules are
copied byte-for-value from EXP-P36-001 (b6479afa...).  Deterministic: same input -> same bytes.
  python experiments/pitchmdp/docs/launch/make_p37_config.py
"""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/pitchmdp'))
from pitchmdp import direct_embed_type_registry as et  # noqa: E402

SOURCE = REPO / 'configs/EXP-P36-001.json'
SOURCE_SHA256 = 'b6479afa43a59df91774e0b714da2397692bdb06959fabb7716a129a5826b87e'
OUT = REPO / 'configs/DRAFT-CUDA-EMBED-TYPE-20261005.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P37-001-embed-type-2026-10-05.md'


def build():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('EXP-P36-001 config differs from its registered sha')
    p36 = json.loads(SOURCE.read_text())
    keep = ('protocol', 'export', 'training', 'calibration', 'device', 'data_rules')
    config = {k: p36[k] for k in keep}
    config.update({
        'experiment_id': 'EXP-P37-001',
        'status': 'DRAFT',
        et.MARKER: True,
        'decision': 'None. DRAFT only: CUDA smoke allowed after distinct review; real fits need registration '
                    '(status REGISTERED, execution.enabled true) after the same-model distinct review of this '
                    'pre-registration, config and wiring.',
        'purpose': 'Exploratory embedding-map control (pre-registration ' + PREREG + ', written 2026-10-05 after the '
                   'P36 report ce2861b and before any P37 fit): periodic_residual_counts vs linear_residual_counts '
                   'with the same residual backbone, embedded columns, embedding width 8 and backbone input width; '
                   'parameter-near-matched (+264), not compute-matched. Descriptive repeated-DEV only.',
        'source_configs': {**p36['source_configs'],
                           'robust_clip_registered': {'path': 'configs/EXP-P36-001.json', 'sha256': SOURCE_SHA256}},
        'arm_notes': {
            'control': p36['arm_notes']['control'],
            'periodic_residual_counts': p36['arm_notes']['periodic_residual_counts'],
            'linear_residual_counts': 'EXP-P32-001 arm unchanged (frozen LR embedding ReLU(w*x+b) per scalar, width 8, '
                                      '+ residual backbone, blocks Dropout .1, counts workload), old code path '
                                      '(direct_embed_residual_factory); keeps its embed_residual flag',
            'epoch_cap': 'Epoch cap 60, patience 5 for every arm (as EXP-P34/P35/P36). fit.json records cap, epochs_run, '
                         'best_epoch and stopped_by per fit',
            'parameters': 'Allocated parameters at the registered shapes: ' + json.dumps(et.PARAMETERS, sort_keys=True)
                          + ' (re-derived from the frozen networks by tests; equal to EXP-P32-001 fit.json)',
            'compute': 'NOT compute-matched: the periodic map computes 2 x 16 cos/sin and a shared Linear(32, 8) per '
                       'embedded scalar, the LR map one affine map per scalar; backbone identical. Seconds, updates, '
                       'seconds per update, epochs and peak device bytes are reported per fit, never used to adjust NLL',
        },
        'control_reuse': 'none: fresh control and both embedded arms on seeds 39-43; earlier P24-P36 tasks are never mixed in',
        'seeds': list(et.SEEDS),
        'seed_rule': 'only this registry family may use seeds 39-43: exactly [39, 40, 41, 42, 43], no repeat_tag; '
                     'disjoint from every earlier block 0-38 (direct_embed_type_registry.EARLIER_SEEDS); fixed in the '
                     'pre-registration before any P37 metric',
        'fits': et.FITS,
        'arms': [et.BASELINES[name] for name in et.ARM_ORDER],
        'comparison': {
            'control_arm': 'control',
            'scorer': p36['comparison']['scorer'],
            'scope': p36['comparison']['scope'],
            'not_compared': 'P24-P36 tasks (including the EXP-P32-001 run of the same pair), frozen MPS numbers or other '
                            'families (different control/code/seeds/cap); no saved-probability ensemble grid',
            et.CONTRAST_KEY: et.CONTRASTS,
            'embed_type_scope': '1 pre-registered pair (candidate minus reference, same seeds, seed-mean ensemble and '
                                'single-fit, whole-game bootstrap): embedding_map. Parameter-near-matched, not '
                                'compute-matched. Exploratory, post-selection, repeated DEV. No ranking, winner or '
                                'automatic selection; every arm also vs control.',
        },
        'queue': {**{k: p36['queue'][k] for k in ('timezone', 'task_timeout_seconds', 'family_budget_seconds',
                                                  'max_attempts', 'stop_rule')},
                  'date': '2026-10-05',
                  'batch': 'P37 embedding-map control 3 arms x fresh seeds 39-43, epoch cap 60 (DRAFT)',
                  'launch_by': '19:00', 'no_new_fit_after': '21:30', 'kill_own_after': '22:30',
                  'aggregate_by': '23:30',
                  'date_note': 'Planned 2026-10-05 KST bounds, fixed before registration: every fit ends by 22:30 '
                               '(<= 23:00 overall fit cutoff); if registration has not happened by 19:00 the family '
                               'is NOT launched today and needs a new dated plan',
                  'order': 'priority then seed then arm (all 3 arms priority 1; config order)',
                  'start_after': 'no other family queue running on GPU3/4; common exclusive GPU3/4 lock held; at most '
                                 '2 fits in parallel'},
        'exposure': 'DEV / MLB DEV are the same repeatedly exposed <=2025 rows used for P24-P36 and EXP-P30-002; this '
                    'exact pair ran once in EXP-P32-001 (seeds 16-18, cap 30; embedding_type dev -0.00126, mlb_dev '
                    '-0.00012). Descriptive only; no unseen validation, no 2026, no G0 or policy change.',
        'execution': {'enabled': False},
    })
    return config


def main():
    text = json.dumps(build(), indent=2, ensure_ascii=False) + '\n'
    OUT.write_text(text)
    print(OUT.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main()
