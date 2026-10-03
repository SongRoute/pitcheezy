"""Torch-free registry of the DRAFT architecture x dropout family (DRAFT-CUDA-DROPOUT-RUN-20261004).

Exploratory follow-up chosen AFTER reading P25 (dropout .1 at flatten 128 lowered
calibrated direct NLL on dev and mlb_dev vs its control, 3/3 seeds; weight decay
alone neutral) and P24 (periodic / residual).  Both are exposed repeated DEV; this
family is descriptive only: no promotion, verdict, ranking or 2026.

3 x 2 factorial, 6 arms x seeds 0-2 = 18 fits, one immutable export, fixed budget
(30 epochs, patience 5, lr 5e-4, batch 1024, AdamW weight decay .01, uniform loss),
external calibration, no workload:

  architecture  dropout 0 (fresh control)              dropout .1
  flatten       control  = MatrixModel flatten_mlp 128   dropout = frontier ARMS['dropout'] (exact)
  residual      residual = frozen DirectResidualModel    residual_dropout
  periodic      periodic = frozen DirectPeriodicModel    periodic_dropout

The dropout arms add nn.Dropout(.1) after each hidden activation stage of the HEAD
that the base network does not already follow with dropout, as in
FlattenRegularizedModel (after each hidden GELU):
  periodic_dropout  after both head GELUs; the PLR numeric embeddings (ReLU) and
                    the raw inputs are not dropped.
  residual_dropout  after the output-head GELU only.  The frozen residual blocks
                    ALREADY carry Dropout(.1) after their GELU (registered
                    architecture), so the residual contrast is "head dropout .1
                    added on top of block dropout .1", NOT "no dropout vs .1".
Dropout has no parameters: each dropout arm has the parameter count of its base.

Fixed regularization contrasts (candidate - reference, same architecture):
dropout_vs_control, residual_dropout_vs_residual, periodic_dropout_vs_periodic.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import BASE_PERIODIC

DROPOUT = .1
WEIGHT_DECAY = .01
PERIODIC_CONTINUOUS = list(BASE_PERIODIC)  # the 40 base periodic indices (11..51 without left_share 32)

# name -> exact config entry without name / priority / dropout_head / workload
ARMS = {
    'residual_dropout': {'family': 'architecture_dropout', 'model': 'residual_head_dropout_mlp',
                         'base_model': 'residual_mlp', 'width': 128, 'head_dropout': DROPOUT},
    'periodic_dropout': {'family': 'architecture_dropout', 'model': 'periodic_head_dropout_mlp',
                         'base_model': 'periodic_flatten_mlp', 'width': 128, 'head_dropout': DROPOUT,
                         'context_continuous': PERIODIC_CONTINUOUS},
}
# reference arm name -> exact config entry of the fresh same-architecture no-dropout arm
REFERENCES = {
    'control': {'name': 'control', 'priority': 1, 'model': 'flatten_mlp', 'width': 128, 'workload': []},
    'residual': {'name': 'residual', 'priority': 1, 'model': 'residual_mlp', 'width': 128, 'workload': []},
    'periodic': {'name': 'periodic', 'priority': 1, 'model': 'periodic_flatten_mlp', 'width': 128, 'workload': [],
                 'context_continuous': PERIODIC_CONTINUOUS},
}
CONTRASTS = [
    {'name': 'dropout_vs_control', 'candidate': 'dropout', 'reference': 'control',
     'architecture': 'flatten_mlp', 'isolates': 'dropout'},
    {'name': 'residual_dropout_vs_residual', 'candidate': 'residual_dropout', 'reference': 'residual',
     'architecture': 'residual_mlp', 'isolates': 'dropout'},
    {'name': 'periodic_dropout_vs_periodic', 'candidate': 'periodic_dropout', 'reference': 'periodic',
     'architecture': 'periodic_flatten_mlp', 'isolates': 'dropout'},
]
ARM_ORDER = ('control', 'dropout', 'residual', 'residual_dropout', 'periodic', 'periodic_dropout')


def check_arm(arm):
    """A dropout-head arm is exactly its registry entry (+ name, priority 1, flag, no workload)."""
    name = arm.get('name')
    rest = {k: v for k, v in arm.items() if k not in ('name', 'priority', 'dropout_head', 'workload')}
    if arm.get('dropout_head') is not True or name not in ARMS or rest != ARMS[name]:
        raise ValueError('Dropout arm differs from the fixed registry: ' + str(name))
    if arm.get('workload') != [] or arm.get('priority') != 1:
        raise ValueError('Dropout arms keep the control inputs (no workload) and priority 1')
    return arm


def check_family(config):
    """The whole family is the six fixed arms in order (references exact, frontier 'dropout' exact)."""
    from pitchmdp.direct_frontier_registry import ARMS as FRONTIER
    arms = {a['name']: a for a in config['arms']}
    if [a['name'] for a in config['arms']] != list(ARM_ORDER):
        raise ValueError('Dropout family must be exactly ' + ', '.join(ARM_ORDER))
    for name, ref in REFERENCES.items():
        if arms[name] != ref:
            raise ValueError(name + ': same-architecture no-dropout arm differs from the base spec')
    dropout = {k: v for k, v in arms['dropout'].items() if k not in ('name', 'priority', 'frontier', 'workload')}
    if arms['dropout'].get('frontier') is not True or dropout != FRONTIER['dropout'] \
            or FRONTIER['dropout']['dropout'] != DROPOUT or FRONTIER['dropout']['weight_decay'] != WEIGHT_DECAY:
        raise ValueError('dropout must be the exact frontier regularization arm (p .1, wd .01)')
    for name in ARMS:
        check_arm(arms[name])
    return config


def check_contrasts(config):
    """comparison.regularization_contrasts (or None): exactly the fixed list, arms checked before any read."""
    contrasts = config.get('comparison', {}).get('regularization_contrasts')
    if contrasts is None:
        return None
    if contrasts != CONTRASTS:
        raise ValueError('regularization_contrasts differ from the fixed same-architecture dropout pairs')
    check_family(config)
    arms = {a['name']: a for a in config['arms']}
    for c in contrasts:
        cand, ref = arms[c['candidate']], arms[c['reference']]
        base = cand.get('base_model', 'flatten_mlp' if cand.get('model') == 'regularized_flatten_mlp' else None)
        if ref['model'] != c['architecture'] or base != c['architecture'] or cand['workload'] or ref['workload'] \
                or cand.get('context_continuous') != ref.get('context_continuous'):
            raise ValueError(c['name'] + ': candidate and reference differ beyond dropout')
    return contrasts
