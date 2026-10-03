"""Torch-free registry of the DRAFT periodic workload x head-dropout family (DRAFT-CUDA-WORKLOAD-DROPOUT-20261004).

Post-result follow-up chosen AFTER reading P27 (periodic_counts / periodic_block
lowered calibrated direct NLL vs control on dev and mlb_dev) and P29 (head dropout
on periodic alone: uncertain incremental effect).  Every family so far is exposed
repeated DEV; this one is descriptive only: no promotion, verdict, ranking, winner
or 2026.  Fresh MODEL RNG seeds are NOT new data and NOT an independent
confirmatory test (same export, same DEV rows).

Question: do extra workload and head dropout combine, and do the signs hold across
fresh model seeds?  Flatten control + periodic 3 x 3 (workload x head dropout),
10 arms x 5 fresh seeds [3..7] = 50 fits, one immutable export, fixed budget
(30 epochs, patience 5, lr 5e-4, batch 1024, AdamW weight decay .01, uniform loss),
external calibration:

  workload \\ dropout  0 (exact old arms)          .1                    .2
  none                periodic (P24/P29 spec)      periodic_d10          periodic_d20
  counts              periodic_counts (P27 combo)  periodic_counts_d10   periodic_counts_d20
  block               periodic_block  (P27 combo)  periodic_block_d10    periodic_block_d20

plus control = MatrixModel flatten_mlp 128, no workload.  The dropout-0 arms are the
exact earlier config entries (same flags, same code path).  The six new arms reuse
direct_dropout_model.PeriodicHeadDropoutModel unedited (frozen periodic network,
nn.Dropout(p) after BOTH head GELUs only; the PLR numeric embeddings and the raw
inputs are never dropped) with the P27 continuous map of their workload level:
none 40 indices (11..51 without 32), counts + 52, block + 52..55 (rest_missing 55
included, exactly as P27).  periodic_d10 has the same network as P29
periodic_dropout (different flag/name only).

Seeds: exactly [3, 4, 5, 6, 7] with no repeat_tag.  A second repetition is allowed
only as seeds [8..12] with the explicit repeat_tag below and the SAME frozen arms;
nothing else (in particular never seeds 0-2 and never a free seed list).

Fixed contrasts (candidate - reference, each changes exactly one factor):
  head_dropout  same workload, dropout d vs 0      (3 workloads x {.1, .2} = 6)
  workload      same dropout, counts/block vs none (3 dropouts x {counts, block} = 6)
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import BASE_PERIODIC, BLOCK, COUNTS, N_CONTEXT

FLAG = 'workload_dropout'
CONTRAST_KEY = 'workload_dropout_contrasts'
DROPOUTS = (0., .1, .2)
SUFFIX = {0.: '', .1: '_d10', .2: '_d20'}
WORKLOADS = {'none': [], 'counts': list(COUNTS), 'block': list(BLOCK)}
STEM = {'none': 'periodic', 'counts': 'periodic_counts', 'block': 'periodic_block'}
CONTINUOUS = {level: [*BASE_PERIODIC, *range(N_CONTEXT, N_CONTEXT + len(cols))] for level, cols in WORKLOADS.items()}
FIRST_SEEDS = [3, 4, 5, 6, 7]
SEED_BLOCKS = {None: FIRST_SEEDS, 'repeat2_seeds_8_12': [8, 9, 10, 11, 12]}


def arm_name(level, dropout):
    return STEM[level] + SUFFIX[dropout]


# Exact earlier config entries of the dropout-0 arms (old behavior, old flags).
BASELINES = {
    'control': {'name': 'control', 'priority': 1, 'model': 'flatten_mlp', 'width': 128, 'workload': []},
    'periodic': {'name': 'periodic', 'priority': 1, 'model': 'periodic_flatten_mlp', 'width': 128, 'workload': [],
                 'context_continuous': CONTINUOUS['none']},
    **{STEM[level]: {'name': STEM[level], 'priority': 1, 'combo': True, 'workload': WORKLOADS[level],
                     'family': 'combination', 'model': 'periodic_flatten_mlp', 'width': 128,
                     'context_continuous': CONTINUOUS[level]} for level in ('counts', 'block')},
}
# New arm name -> exact config entry without name / priority / flag / workload.
ARMS = {arm_name(level, d): {'family': FLAG, 'model': 'periodic_head_dropout_mlp',
                             'base_model': 'periodic_flatten_mlp', 'width': 128, 'head_dropout': d,
                             'workload_level': level, 'context_continuous': CONTINUOUS[level]}
        for level in WORKLOADS for d in DROPOUTS[1:]}
ARM_ORDER = ('control', *(arm_name(level, d) for level in WORKLOADS for d in DROPOUTS))
FITS = len(ARM_ORDER) * len(FIRST_SEEDS)

CONTRASTS = [
    *({'name': f'{arm_name(level, d)}_vs_{arm_name(level, 0.)}', 'candidate': arm_name(level, d),
       'reference': arm_name(level, 0.), 'isolates': 'head_dropout', 'held': {'workload': level}}
      for level in WORKLOADS for d in DROPOUTS[1:]),
    *({'name': f'{arm_name(level, d)}_vs_{arm_name("none", d)}', 'candidate': arm_name(level, d),
       'reference': arm_name('none', d), 'isolates': 'workload', 'held': {'head_dropout': d}}
      for d in DROPOUTS for level in ('counts', 'block')),
]


def check_arm(arm):
    """A new arm is exactly its registry entry (+ name, priority 1, flag, its level's workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', FLAG, 'workload')}
    if arm.get(FLAG) is not True or name not in ARMS or rest != ARMS[name]:
        raise ValueError('Workload-dropout arm differs from the fixed registry: ' + str(name))
    if arm.get('workload') != WORKLOADS[ARMS[name]['workload_level']] or arm.get('priority') != 1:
        raise ValueError(name + ': workload differs from its level or priority is not 1')
    return arm


def check_seeds(config):
    """Seeds [3..7] (no repeat_tag) or, only with the explicit repeat tag, [8..12]; nothing else."""
    tag = config.get('repeat_tag')
    if tag not in SEED_BLOCKS or config.get('seeds') != SEED_BLOCKS[tag]:
        raise ValueError('Workload-dropout seeds must be exactly [3..7], or [8..12] with the explicit repeat_tag')
    if config.get('fits') != FITS:
        raise ValueError(f'Workload-dropout family is exactly {FITS} fits')
    return config['seeds']


def check_family(config):
    """The whole family is the ten fixed arms in order; dropout-0 arms are the exact earlier entries."""
    if [a.get('name') for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Workload-dropout family must be exactly ' + ', '.join(ARM_ORDER))
    for arm in config['arms']:
        if arm['name'] in BASELINES:
            if arm != BASELINES[arm['name']]:
                raise ValueError(arm['name'] + ': dropout-0 arm differs from its exact earlier spec')
        else:
            check_arm(arm)
    check_seeds(config)
    return config


def factors(arm):
    """(workload level, head dropout, continuous map) of a periodic family arm."""
    if arm['name'] in ARMS:
        spec = ARMS[arm['name']]
        return spec['workload_level'], spec['head_dropout'], arm['context_continuous']
    if arm.get('model') != 'periodic_flatten_mlp':
        raise ValueError(arm['name'] + ': not a periodic family arm')
    level = [k for k, v in WORKLOADS.items() if v == arm['workload']]
    return level[0], 0., arm['context_continuous']


def check_contrasts(config):
    """comparison.workload_dropout_contrasts (or None): exactly the fixed list, each changing one factor."""
    contrasts = config.get('comparison', {}).get(CONTRAST_KEY)
    if contrasts is None:
        return None
    if contrasts != CONTRASTS:
        raise ValueError(CONTRAST_KEY + ' differ from the fixed workload / head-dropout pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        (cw, cd, cc), (rw, rd, rc) = factors(arms[c['candidate']]), factors(arms[c['reference']])
        if cc != CONTINUOUS[cw] or rc != CONTINUOUS[rw]:
            raise ValueError(c['name'] + ': continuous map differs from its workload level')
        changed = {'workload'} if cw != rw else set()
        changed |= {'head_dropout'} if cd != rd else set()
        held = {'workload': cw} if c['isolates'] == 'head_dropout' else {'head_dropout': cd}
        reference_is_zero = rd == 0. if c['isolates'] == 'head_dropout' else rw == 'none'
        if changed != {c['isolates']} or c['held'] != held or not reference_is_zero:
            raise ValueError(c['name'] + ': contrast must change exactly the declared factor vs its reference')
    return contrasts
