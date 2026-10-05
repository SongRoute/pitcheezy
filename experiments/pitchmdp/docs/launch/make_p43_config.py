"""Write configs/DRAFT-CUDA-BATTER-ID-20261006.json (tentative EXP-P43-001 DRAFT) from the P40 DRAFT.

Only family fields change: design key p43_batter_id_v1, the batter-sidecar pin, four arms (reference, batter id,
row-shuffled batter id, pitcher + batter id), fresh seeds 70-74, fits 20, contrasts, purpose, notes, queue clock,
experiment id.  Export, training (cap 60), calibration, data rules and device (GPU3 only, max_parallel 1) are copied
by value from the P40 DRAFT (3dba5445...), which copies P38 / EXP-P36-001.  Deterministic: same input -> same bytes.

Clock: the DRAFT carries NO launch window (P40 / P42 hold GPU3 tonight).  The registering lead re-dates with
--clock YYYY-MM-DD HH:MM [LAST_AGGREGATE]; only the queue clock fields change, and that diff is reviewed.
no_new_fit_after >= launch_by + P43_FAMILY_PROJECTION_SECONDS (6,117.5 s, ceil 5 min), kill +15 min, aggregate +20 min.
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'experiments/pitchmdp'))
from pitchmdp import batter_sidecar as bs  # noqa: E402
from pitchmdp import direct_pitcher_id_registry as pid  # noqa: E402

SOURCE = REPO / 'configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json'
SOURCE_SHA256 = '3dba5445b3a76dbc5f19c73f1cd60f6a69fd18312c775a6f82e06db656ab6944'
P40_REGISTERED_SHA256 = '61d210eca2aa028808b5fce07252ebb6bd054c7ea31fdc096631460033a29fe9'
OUT = REPO / 'configs/DRAFT-CUDA-BATTER-ID-20261006.json'
PREREG = 'experiments/pitchmdp/docs/PREREG-DRAFT-EXP-P43-001-batter-id-2026-10-06.md'
# Built 2026-10-06 01:14 KST on the Mac mini from the local verified copy of export 49097109 and raw statcast
# 2023/2024/2025 (ids only); 0.74 s, 0 missing rows, 0 pitcher mismatches.  Rebuilding on newsrv must give this sha.
SIDECAR = {'dirname': 'batter-sidecar-49097109', 'format': bs.FORMAT,
           'sidecar_sha256': 'ad2eacc64d46985a57527167486d7762a435f02d51fc5ed86a9e2e2644c160a4'}
KILL_AFTER_NO_NEW = timedelta(minutes=15)
AGGREGATE_AFTER_KILL = timedelta(minutes=20)
NO_CLOCK = {'date': None, 'launch_by': None, 'no_new_fit_after': None, 'kill_own_after': None, 'aggregate_by': None}


def clock(date, launch_by, last_aggregate='23:59'):
    start = datetime.strptime(date + ' ' + launch_by, '%Y-%m-%d %H:%M')
    no_new = start + timedelta(seconds=pid.P43_FAMILY_PROJECTION_SECONDS)
    if no_new.second or no_new.microsecond:
        no_new += timedelta(seconds=60 - no_new.second, microseconds=-no_new.microsecond)
    no_new += timedelta(minutes=-no_new.minute % 5)
    kill = no_new + KILL_AFTER_NO_NEW
    aggregate = kill + AGGREGATE_AFTER_KILL
    if aggregate.date() != start.date() or aggregate.strftime('%H:%M') > last_aggregate:
        raise SystemExit(f'Window does not end by {last_aggregate} on the launch date; choose an earlier launch_by')
    return {'date': date, 'launch_by': launch_by, 'no_new_fit_after': no_new.strftime('%H:%M'),
            'kill_own_after': kill.strftime('%H:%M'), 'aggregate_by': aggregate.strftime('%H:%M')}


def build(date=None, launch_by=None, last_aggregate='23:59'):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('P40 DRAFT config differs from its pinned sha')
    p40 = json.loads(SOURCE.read_text())
    config = {k: p40[k] for k in ('protocol', 'export', 'training', 'calibration', 'data_rules', 'device')}
    config.update({
        'experiment_id': 'EXP-P43-001',
        'status': 'DRAFT',
        pid.MARKER: True,
        pid.DESIGN_KEY: pid.P43_DESIGN,
        pid.BATTER_SIDECAR_KEY: dict(SIDECAR),
        'decision': 'None. DRAFT only: GPU3 TRAIN smoke allowed after source/protocol preflight and after P40/P42 '
                    'release GPU3; real fits need registration (status REGISTERED, execution.enabled true, a dated '
                    'queue clock) after a distinct review and Root acceptance of this pre-registration, config and '
                    'wiring.',
        'purpose': 'Exploratory study (pre-registration ' + PREREG + ', written 2026-10-06 before any P43 output): '
                   'does a TRAIN-only BATTER identity embedding, never given to a direct head before, lower calibrated '
                   'NLL / Brier beyond the 17-d batter style summary the reference already reads (P25: removing that '
                   'summary costs +0.0075 mlb_dev NLL)? Primary contrast batter id vs a row-shuffled batter id '
                   '(identical network, parameters and compute, information removed); capacity control; pitcher + '
                   'batter additivity. Descriptive repeated-DEV NLL / Brier / ECE only; no policy, RL, OPE or service '
                   'claim.',
        'source_configs': {**p40['source_configs'],
                           'p40_draft': {'path': 'configs/DRAFT-CUDA-PITCHER-ID-CONTROLS-20261006.json',
                                         'sha256': SOURCE_SHA256},
                           'p40_registered': {'path': 'configs/EXP-P40-001.json', 'sha256': P40_REGISTERED_SHA256}},
        'arm_notes': {
            pid.REFERENCE: p40['arm_notes'][pid.REFERENCE],
            pid.BATTER: 'P38 id path on the batter column: TRAIN-only vocabulary (batters with >= 100 TRAIN rows, '
                        'sorted ids, UNK 0), zero-init table [n_vocab + 1, 8], projection 128 x 8 from the private '
                        'generator seed + 7919 + 1009, z += P T[idx]. Batter ids come from the pinned sidecar.',
            pid.BATTER_SHUFFLE: 'Batter-id network, parameters and compute; before the vocabulary lookup the split\'s '
                                '`batter` column is permuted across rows by np.random.default_rng([43001, split '
                                'position]) (fixed, independent of labels, features and the model seed). Index '
                                'frequencies / UNK share kept, row -> batter association destroyed (information-free).',
            pid.PITCHER_BATTER: 'Two id paths, additive: the P38 pitcher path verbatim (projection seed + 7919) and the '
                                'batter path above (seed + 7919 + 1009). Index columns (pitcher, batter).',
            'epoch_cap': p40['arm_notes']['epoch_cap'],
            'parameters': f'Reference {pid.BASE_PARAMETERS}; each id path adds (n_vocab + 1) x 8 + 128 x 8 (n_vocab '
                          'per column measured from TRAIN, recorded per fit, re-derived and checked by the scorer)',
            'capacity': 'Batter id vs row-shuffle is parameter- and compute-matched with information removed; the '
                        'control matches the id path, not a generic wider backbone. Pitcher + batter vs batter adds '
                        'the pitcher path (not matched; P40 holds the pitcher controls).',
        },
        'control_reuse': 'none: fresh four arms on seeds 70-74; P38 / P40 / P42 tasks are never mixed in',
        'seeds': list(pid.P43_SEEDS),
        'seed_rule': 'exactly [70, 71, 72, 73, 74], no repeat_tag; disjoint from 0-69 used or reserved before '
                     '(P40 60-64, P42 DRAFT 65-69, P41 MPS 49-53); fixed before any P43 output. New model randomness '
                     'only, not new data.',
        'fits': pid.P43_FITS,
        'arms': [pid.BASELINES[name] for name in pid.P43_ARM_ORDER],
        'pitcher_id_data_rule': 'Vocabularies per id column from the TRAIN split only (labels never read): rows dated '
                                '<= 20250430 (manifest TRAIN max_date <= cutoff and every later split min_date > '
                                'cutoff, checked by runner and scorer), ids with >= 100 TRAIN rows (the P38 rule, not '
                                'tuned); DEV / mlb_dev never used for vocabulary, min count, width, calibration or '
                                'selection. `batter` is joined by the export keys from raw statcast 2023-2025 (id '
                                'columns only, no 2026 / holdout file), every row matched once and raw pitcher == '
                                'export pitcher on every row; the sidecar is pinned by sha and verified per split. '
                                'Row shuffle uses only the split\'s own batter column and a fixed seed; scorer subsets '
                                'use the TRUE batter\'s TRAIN vocabulary membership for every arm. No 2026 rows.',
        'comparison': {
            'reference_arm': pid.REFERENCE,
            'scorer': p40['comparison']['scorer'],
            'scope': 'unblended calibrated direct-head probabilities on dev and mlb_dev; four fixed contrasts on all '
                     'rows and TRAIN batter-vocabulary in-vocab / UNK subsets, every seed and the seed-mean ensemble; '
                     'per-arm NLL / Brier / accuracy / top-label ECE; descriptive repeated-DEV only; no decision, '
                     'ranking, automatic selection or winner',
            'not_compared': 'P24-P42 tasks, frozen MPS numbers or other families',
            pid.CONTRAST_KEY: pid.P43_CONTRASTS,
        },
        'queue': {**{k: p40['queue'][k] for k in ('timezone', 'task_timeout_seconds', 'family_budget_seconds',
                                                  'max_attempts')},
                  **(clock(date, launch_by, last_aggregate) if date else NO_CLOCK),
                  'batch': 'P43 batter-ID 4 arms x fresh seeds 70-74, epoch cap 60 (DRAFT)',
                  'stop_rule': 'any task failing twice aborts the family (no partial comparison, failures reported); '
                               'GPU3 TRAIN smoke 20/20 valid before registration; smoke per-epoch seconds reported '
                               'descriptively (the batter path is the P38 id path; the two-path arm adds one gather)',
                  'clock_rule': 'DRAFT has no window; registration re-dates with make_p43_config.py --clock: '
                                'no_new_fit_after >= launch_by + 6,117.5 s (20 x 1.25 x P38 max task wall 244.7 s, '
                                'ceil 5 min), kill_own_after = +15 min, aggregate_by = +20 min; only these fields change',
                  'order': 'priority then seed then arm (all arms priority 1; config order)',
                  'start_after': 'P40 and P42 queues finished or stopped by their owner; no other Pitcheezy queue, '
                                 'probe or scorer running; GPU3 has no compute process; common exclusive GPU3 lock '
                                 'held; exactly one fit at a time on GPU3 (UUID ' + pid.RETAINED_GPU + ')',
                  'resource_rule': p40['queue']['resource_rule']},
        'exposure': 'DEV / MLB DEV are the same repeatedly exposed <=2025 rows used for P24-P42. Designed after reading '
                    'P25 and P38 (post-selection). Exploratory only, not an independent test, no promotion. No 2026 '
                    'data. An unseen <=2025 confirmation set is proposed separately (metadata only, not part of this '
                    'family).',
        'execution': {'enabled': False, 'executable': False,
                      'blocked_until': ['distinct review of this packet and Root acceptance',
                                        'P40 / P42 release GPU3', 'batter sidecar rebuilt on newsrv with the pinned sha',
                                        'GPU3 TRAIN smoke 20/20 valid', 'registration with a dated clock and fresh '
                                        'config sha']},
    })
    return config


def main(argv):
    if argv and (len(argv) not in (3, 4) or argv[0] != '--clock'):
        raise SystemExit('usage: make_p43_config.py [--clock YYYY-MM-DD HH:MM [LAST_AGGREGATE]]')
    text = json.dumps(build(*argv[1:]) if argv else build(), indent=2, ensure_ascii=False) + '\n'
    OUT.write_text(text)
    print(OUT.relative_to(REPO), hashlib.sha256(text.encode()).hexdigest())


if __name__ == '__main__':
    main(sys.argv[1:])
