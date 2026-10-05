"""Write configs/DRAFT-CUDA-PITCHER-ID-20261005.json (tentative EXP-P38-001 DRAFT) from the registered EXP-P36-001 config.

Only family fields change (marker, two arms, seeds 44-48, fits 10, comparison, purpose, notes, queue clock, status DRAFT,
execution disabled + executable false, registration block removed); export, training (cap 60), calibration and data
rules are copied byte-for-value from EXP-P36-001 (b6479afa...); the device block is the single-GPU rule of 2026-10-05
(GPU3 UUID only, max_parallel 1; GPU4 released).  The pitcher-ID data rule is appended to data_rules' sibling key
pitcher_id_data_rule (data_rules itself stays the P36 text).  Deterministic: same input -> same bytes.

The queue clock is the ONLY dated part.  Default window (2026-10-05 KST): launch_by 20:45, no_new_fit_after 22:15,
kill_own_after 22:30, aggregate_by 22:50 (< 23:00).  Rule (fixed here, before any P38 measurement): no_new_fit_after >=
launch_by + FAMILY_PROJECTION_SECONDS (5,292 s), kill_own_after >= no_new_fit_after + 15 min, aggregate_by >=
kill_own_after + 20 min and <= 23:00.  If P37 is not terminal / GPU3 not free by launch_by, re-date with
  python experiments/pitchmdp/docs/launch/make_p38_config.py --clock YYYY-MM-DD HH:MM
which recomputes only queue clock fields (tested: every other key byte-identical) and needs a fresh review of the
clock diff only.  Without arguments it writes the default DRAFT.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/pitchmdp'))
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

SOURCE = REPO / 'configs/EXP-P36-001.json'
SOURCE_SHA256 = 'b6479afa43a59df91774e0b714da2397692bdb06959fabb7716a129a5826b87e'
OUT = REPO / 'configs/DRAFT-CUDA-PITCHER-ID-20261005.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P38-001-pitcher-id-2026-10-05.md'
DEFAULT_CLOCK = ('2026-10-05', '20:45')
KILL_AFTER_NO_NEW = timedelta(minutes=15)
AGGREGATE_AFTER_KILL = timedelta(minutes=20)
LAST_AGGREGATE = '23:00'


def clock(date, launch_by):
    """Deterministic queue clock from (date, launch_by): ceil to 5 min; refuses windows that end after 23:00."""
    start = datetime.strptime(date + ' ' + launch_by, '%Y-%m-%d %H:%M')
    no_new = start + timedelta(seconds=pid.FAMILY_PROJECTION_SECONDS)
    if no_new.second:  # ceil to the minute, then to 5 minutes (never before the projected end)
        no_new += timedelta(seconds=60 - no_new.second)
    no_new += timedelta(minutes=-no_new.minute % 5)
    kill = no_new + KILL_AFTER_NO_NEW
    aggregate = kill + AGGREGATE_AFTER_KILL
    if aggregate.date() != start.date() or aggregate.strftime('%H:%M') > LAST_AGGREGATE:
        raise SystemExit('Window does not end by 23:00 on the launch date; choose an earlier launch_by')
    return {'date': date, 'launch_by': launch_by, 'no_new_fit_after': no_new.strftime('%H:%M'),
            'kill_own_after': kill.strftime('%H:%M'), 'aggregate_by': aggregate.strftime('%H:%M')}


def build(date=DEFAULT_CLOCK[0], launch_by=DEFAULT_CLOCK[1]):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('EXP-P36-001 config differs from its registered sha')
    p36 = json.loads(SOURCE.read_text())
    config = {k: p36[k] for k in ('protocol', 'export', 'training', 'calibration', 'data_rules')}
    config['device'] = json.loads(json.dumps(pid.DEVICE))  # Song 2026-10-05: GPU3 only, 1 fit; GPU4 released
    config.update({
        'experiment_id': 'EXP-P38-001',
        'status': 'DRAFT',
        pid.MARKER: True,
        'decision': 'None. DRAFT only: CUDA smoke allowed after distinct review; real fits need registration '
                    '(status REGISTERED, execution.enabled true) after the same-model distinct review of this '
                    'pre-registration, config and wiring, P37 terminal and GPU3 free.',
        'purpose': 'Exploratory information change (pre-registration ' + PREREG + ', written 2026-10-05 after the P36 '
                   'report and before any P38 fit): TRAIN-only pitcher-ID embedding (export column `pitcher`, never '
                   'read by a direct head before) added to the frozen periodic_residual_counts arm vs the same arm. '
                   'Not parameter- or compute-matched. Descriptive repeated-DEV NLL/Brier only; no policy, RL, OPE '
                   'or service claim.',
        'source_configs': {**p36['source_configs'],
                           'robust_clip_registered': {'path': 'configs/EXP-P36-001.json', 'sha256': SOURCE_SHA256},
                           'model_author_commit': '684f4b4 (codex/next-information-1005; cherry-picked with -x)'},
        'arm_notes': {
            pid.REFERENCE: p36['arm_notes'][pid.REFERENCE],
            pid.CANDIDATE: 'Reference entry + pitcher_id block (direct_pitcher_id_model): TRAIN-only vocabulary of '
                           'pitcher ids with >= 100 TRAIN rows (sorted), UNK 0 otherwise (rare TRAIN pitchers and ids '
                           'first seen after TRAIN), index appended as the last context column and stripped before '
                           'the frozen maps; z += P @ T[idx] (T zeros [n_vocab + 1, 8], P [128, 8] private generator '
                           'seed + 7919) so the untrained network equals the reference at the same seed',
            'epoch_cap': 'Epoch cap 60, patience 5 for both arms (as EXP-P34-P37). fit.json records cap, epochs_run, '
                         'best_epoch and stopped_by per fit',
            'parameters': f'Reference {pid.BASE_PARAMETERS}; candidate reference + (n_vocab + 1) x 8 + 128 x 8 '
                          '(n_vocab measured from TRAIN at run time, recorded per fit, checked by the scorer)',
            'capacity': 'NOT parameter-matched, NOT compute-matched. A capacity-matched information-free control is '
                        'not included in this bounded two-arm study; that limits causal attribution of a difference '
                        'to identity information rather than added capacity.',
        },
        'control_reuse': 'none: fresh reference and candidate on seeds 44-48; earlier P24-P37 tasks are never mixed in; '
                         'no flatten control in this two-arm family (the reference arm is the comparator)',
        'seeds': list(pid.SEEDS),
        'seed_rule': 'only this registry family may use seeds 44-48: exactly [44, 45, 46, 47, 48], no repeat_tag; '
                     'disjoint from every registered block 0-38 and the P37 DRAFT 39-43 '
                     '(direct_pitcher_id_registry.EARLIER_SEEDS); fixed before any P38 output. New model randomness '
                     'only, not new data.',
        'fits': pid.FITS,
        'arms': [pid.BASELINES[name] for name in pid.ARM_ORDER],
        'pitcher_id_data_rule': 'Vocabulary from the TRAIN split `pitcher` array only (labels never read): rows dated '
                                f'<= {pid.LAST_TRAIN_DATE} (manifest TRAIN max_date <= cutoff and every later split '
                                'min_date > cutoff, checked by runner and scorer), ids with >= '
                                f'{pid.MIN_TRAIN_PITCHES} TRAIN rows; DEV / mlb_dev never used for vocabulary, '
                                'min count, embedding width, calibration or selection. No 2026 rows.',
        'comparison': {
            'reference_arm': pid.REFERENCE,
            'scorer': p36['comparison']['scorer'],
            'scope': 'unblended calibrated direct-head probabilities, candidate minus reference on dev and mlb_dev, '
                     'all rows and TRAIN-vocabulary in-vocab / UNK subsets, every seed and the seed-mean ensemble; '
                     'descriptive repeated-DEV only; no decision, ranking, automatic selection or winner',
            'not_compared': 'P24-P37 tasks, frozen MPS numbers or other families',
            pid.CONTRAST_KEY: pid.CONTRASTS,
        },
        'queue': {**{k: p36['queue'][k] for k in ('timezone', 'task_timeout_seconds', 'family_budget_seconds',
                                                  'max_attempts')},
                  **clock(date, launch_by),
                  'batch': 'P38 pitcher-ID 2 arms x fresh seeds 44-48, epoch cap 60 (DRAFT)',
                  'stop_rule': 'any task failing twice aborts the family (no partial comparison, failures reported); '
                               'TRAIN-only smoke resource gate before registration (pre-registration section 4: '
                               'candidate / reference median train seconds per epoch <= 1.25 over 5 smoke seeds)',
                  'clock_rule': 'no_new_fit_after >= launch_by + 5,292 s (projection, ceil 5 min), kill_own_after = '
                                '+15 min, aggregate_by = +20 min, <= 23:00; re-dating changes only these fields',
                  'order': 'priority then seed then arm (both arms priority 1; config order)',
                  'start_after': 'EXP-P37-001 terminal (pipeline state complete / incomplete / blocked) and no other '
                                 'Pitcheezy queue, probe or scorer running; GPU3 has no compute process; common '
                                 'exclusive GPU3 lock held; exactly one fit at a time on GPU3 (UUID '
                                 + pid.RETAINED_GPU + ')',
                  'resource_rule': 'Song 2026-10-05: Pitcheezy uses ONLY physical GPU3 (UUID ' + pid.RETAINED_GPU +
                                   '), max 1 fit; GPU4 (UUID ' + pid.RELEASED_GPUS[0] + ') is released to another '
                                   'project: never allowed, probed or used as fallback (wait if GPU3 is busy).'},
        'exposure': 'DEV / MLB DEV are the same repeatedly exposed <=2025 rows used for P24-P37. Designed after '
                    'reading P24-P36 reports (post-selection). Exploratory only, not an independent test, no '
                    'promotion. No 2026 data.',
        'execution': {'enabled': False, 'executable': False,
                      'blocked_until': ['distinct review of this packet', 'P37 terminal and GPU3 free',
                                        'GPU3 TRAIN smoke 10/10 + resource gate pass',
                                        'registration with fresh config sha (make_p38_registered_config.py)']},
    })
    return config


def main(argv):
    if argv and (len(argv) != 3 or argv[0] != '--clock'):
        raise SystemExit('usage: make_p38_config.py [--clock YYYY-MM-DD HH:MM]')
    text = json.dumps(build(*argv[1:]) if argv else build(), indent=2, ensure_ascii=False) + '\n'
    OUT.write_text(text)
    print(OUT.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main(sys.argv[1:])
