"""Write configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json (tentative EXP-P40-001 DRAFT) from the P38 DRAFT.

Only family fields change: design key, four arms (reference, P38 candidate, row-shuffled-id control, ID-dropout),
fresh seeds 60-64, fits 20, comparison contrasts, purpose, notes, queue clock, experiment id.  Export, training (cap
60), calibration, data rules, device (GPU3 only, max_parallel 1) and the pitcher-ID data rule are copied by value
from the P38 DRAFT (8518f754...), which itself copies EXP-P36-001.  Deterministic: same input -> same bytes.

Clock (one KST date, 2026-10-06 four-hour window that ends 04:28:47): launch_by 02:05, no_new_fit_after >= launch_by +
P40_FAMILY_PROJECTION_SECONDS (6,117.5 s, ceil 5 min), kill_own_after = +15 min, aggregate_by = +20 min, <= 04:28.
Re-date with --clock YYYY-MM-DD HH:MM (only queue clock fields change).
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/pitchmdp'))
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

SOURCE = REPO / 'configs/DRAFT-CUDA-PITCHER-ID-20261005.json'
SOURCE_SHA256 = '8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537'
P38_REGISTERED_SHA256 = 'e80ff2e4082078fb045c4980e03afa237dd205e96b38a0df3dcbae76f06e2755'
OUT = REPO / 'configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P40-001-pitcher-id-controls-2026-10-06.md'
DEFAULT_CLOCK = ('2026-10-06', '02:05')
KILL_AFTER_NO_NEW = timedelta(minutes=15)
AGGREGATE_AFTER_KILL = timedelta(minutes=20)
LAST_AGGREGATE = '04:28'


def clock(date, launch_by):
    start = datetime.strptime(date + ' ' + launch_by, '%Y-%m-%d %H:%M')
    no_new = start + timedelta(seconds=pid.P40_FAMILY_PROJECTION_SECONDS)
    if no_new.second or no_new.microsecond:
        no_new += timedelta(seconds=60 - no_new.second, microseconds=-no_new.microsecond)
    no_new += timedelta(minutes=-no_new.minute % 5)
    kill = no_new + KILL_AFTER_NO_NEW
    aggregate = kill + AGGREGATE_AFTER_KILL
    if aggregate.date() != start.date() or aggregate.strftime('%H:%M') > LAST_AGGREGATE:
        raise SystemExit('Window does not end by 04:28 on the launch date; choose an earlier launch_by')
    return {'date': date, 'launch_by': launch_by, 'no_new_fit_after': no_new.strftime('%H:%M'),
            'kill_own_after': kill.strftime('%H:%M'), 'aggregate_by': aggregate.strftime('%H:%M')}


def build(date=DEFAULT_CLOCK[0], launch_by=DEFAULT_CLOCK[1]):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('P38 DRAFT config differs from its pinned sha')
    p38 = json.loads(SOURCE.read_text())
    config = {k: p38[k] for k in ('protocol', 'export', 'training', 'calibration', 'data_rules', 'device')}
    config.update({
        'experiment_id': 'EXP-P40-001',
        'status': 'DRAFT',
        pid.MARKER: True,
        pid.DESIGN_KEY: pid.P40_DESIGN,
        'decision': 'None. DRAFT only: GPU3 TRAIN smoke allowed after source/protocol preflight; real fits need '
                    'registration (status REGISTERED, execution.enabled true) after a distinct review and Root '
                    'acceptance of this pre-registration, config and wiring.',
        'purpose': 'Exploratory control study of the P38 pitcher-ID signal (pre-registration ' + PREREG + ', written '
                   '2026-10-06 after the P38 score and before any P40 output): is the P38 candidate gain explained by '
                   'pitcher identity information or by the added id path? Primary contrast candidate vs a '
                   'row-shuffled-id control with identical network, parameters and compute; P38 replication on fresh '
                   'seeds; ID dropout 0.2 as a fixed cold-start regulariser. Descriptive repeated-DEV NLL / Brier / '
                   'ECE only; no policy, RL, OPE or service claim.',
        'source_configs': {**p38['source_configs'],
                           'p38_draft': {'path': 'configs/DRAFT-CUDA-PITCHER-ID-20261005.json',
                                         'sha256': SOURCE_SHA256},
                           'p38_registered': {'path': 'configs/EXP-P38-001.json', 'sha256': P38_REGISTERED_SHA256}},
        'arm_notes': {
            pid.REFERENCE: p38['arm_notes'][pid.REFERENCE],
            pid.CANDIDATE: p38['arm_notes'][pid.CANDIDATE] + ' (P38 entry verbatim)',
            pid.SHUFFLE: 'Candidate network, parameters and compute; before the vocabulary lookup the split\'s '
                         '`pitcher` column is permuted across rows by np.random.default_rng([40001, split position]) '
                         '(fixed, independent of labels, features and the model seed). Index frequencies and UNK '
                         'shares are unchanged; the row -> pitcher association is destroyed (information-free).',
            pid.DROPOUT: 'Candidate + ID dropout: in training mode each row\'s index is replaced by UNK with '
                         'probability 0.2 (private device generator seeded id_seed + 4040; global RNG untouched); '
                         'eval / calibration / prediction use the true index. Fixed, not tuned.',
            'epoch_cap': p38['arm_notes']['epoch_cap'],
            'parameters': f'Reference {pid.BASE_PARAMETERS}; the three id arms reference + (n_vocab + 1) x 8 + 128 x '
                          '8 (equal among them; n_vocab measured from TRAIN, recorded per fit, checked by the scorer)',
            'capacity': 'Candidate vs row-shuffle is parameter- and compute-matched with information removed. The '
                        'control matches the id path, not a generic wider backbone; a learned per-row noise index '
                        'is the closest information-free use of the same parameters.',
        },
        'control_reuse': 'none: fresh four arms on seeds 60-64; P38 tasks (seeds 44-48) are never mixed in',
        'seeds': list(pid.P40_SEEDS),
        'seed_rule': 'exactly [60, 61, 62, 63, 64], no repeat_tag; disjoint from 0-51 used before and from 52-59 '
                     'reserved for the MPS sibling (EXP-P41); fixed before any P40 output. New model randomness only, '
                     'not new data.',
        'fits': pid.P40_FITS,
        'arms': [pid.BASELINES[name] for name in pid.P40_ARM_ORDER],
        'pitcher_id_data_rule': p38['pitcher_id_data_rule'] + ' Row shuffle uses only the split\'s own pitcher '
                                'column and a fixed seed; subsets in the scorer use the TRUE pitcher\'s TRAIN '
                                'vocabulary membership for every arm.',
        'comparison': {
            'reference_arm': pid.REFERENCE,
            'scorer': p38['comparison']['scorer'],
            'scope': 'unblended calibrated direct-head probabilities on dev and mlb_dev; four fixed contrasts on all '
                     'rows and TRAIN-vocabulary in-vocab / UNK subsets, every seed and the seed-mean ensemble; '
                     'per-arm NLL / Brier / accuracy / top-label ECE; descriptive repeated-DEV only; no decision, '
                     'ranking, automatic selection or winner',
            'not_compared': 'P24-P39 tasks, frozen MPS numbers or other families',
            pid.CONTRAST_KEY: pid.P40_CONTRASTS,
        },
        'queue': {**{k: p38['queue'][k] for k in ('timezone', 'task_timeout_seconds', 'family_budget_seconds',
                                                  'max_attempts')},
                  **clock(date, launch_by),
                  'batch': 'P40 pitcher-ID controls 4 arms x fresh seeds 60-64, epoch cap 60 (DRAFT)',
                  'stop_rule': 'any task failing twice aborts the family (no partial comparison, failures reported); '
                               'GPU3 TRAIN smoke 20/20 valid before registration; no resource-ratio gate (the id '
                               'arms share the P38 candidate compute, measured 1.0729 in P38 v2); smoke per-epoch '
                               'seconds reported descriptively',
                  'clock_rule': 'no_new_fit_after >= launch_by + 6,117.5 s (20 x 1.25 x P38 max task wall 244.7 s, ceil '
                                '5 min), kill_own_after = +15 min, aggregate_by = +20 min, <= 04:28; re-dating '
                                'changes only these fields',
                  'order': 'priority then seed then arm (all arms priority 1; config order)',
                  'start_after': 'no other Pitcheezy queue, probe or scorer running; GPU3 has no compute process; '
                                 'common exclusive GPU3 lock held; exactly one fit at a time on GPU3 (UUID '
                                 + pid.RETAINED_GPU + ')',
                  'resource_rule': p38['queue']['resource_rule']},
        'exposure': 'DEV / MLB DEV are the same repeatedly exposed <=2025 rows used for P24-P38. Designed after '
                    'reading the P38 score (post-selection). Exploratory only, not an independent test, no '
                    'promotion. No 2026 data.',
        'execution': {'enabled': False, 'executable': False,
                      'blocked_until': ['distinct review of this packet and Root acceptance', 'GPU3 free',
                                        'GPU3 TRAIN smoke 20/20 valid',
                                        'registration with fresh config sha (make_p40_registered_config.py)']},
    })
    return config


def main(argv):
    if argv and (len(argv) != 3 or argv[0] != '--clock'):
        raise SystemExit('usage: make_p40_config.py [--clock YYYY-MM-DD HH:MM]')
    text = json.dumps(build(*argv[1:]) if argv else build(), indent=2, ensure_ascii=False) + '\n'
    OUT.write_text(text)
    print(OUT.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main(sys.argv[1:])
