"""Model factory for the DRAFT data / pitcher-information / combination arms (see direct_combo_registry).

No new network or fit loop: data arms are the control MatrixModel flatten_mlp 128,
pitcher-information arms are direct_frontier_model.MaskedFlattenModel with the
registry's named columns, combination arms are the frozen DirectResidualModel /
DirectPeriodicModel whose input context is widened by the exported workload
scalars (52 + 1 or 52 + 4; the runner appends them after the 52 stripped context
columns).  load_combo refuses an archive of another kind, mask, continuous map or
context width.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_frontier_model import MaskedFlattenModel
from pitchmdp.direct_periodic_model import DirectPeriodicModel
from pitchmdp.direct_residual_model import DirectResidualModel
from pitchmdp.matrix_models import MatrixModel


def build_combo(arm, seed, device, context_names=None):
    """Unfitted model for a checked combination arm (config dict); masked arms need the verified names."""
    model = arm['model']
    if model == 'flatten_mlp':
        return MatrixModel('flatten_mlp', seed=seed, width=arm['width'], device=device)
    if model == 'masked_flatten_mlp':
        if not context_names:
            raise ValueError('Pitcher-information arms need the verified export context names')
        return MaskedFlattenModel(seed, arm['drop_context'], arm['drop_history'], context_names, device)
    if model == 'residual_mlp':
        return DirectResidualModel(seed, device=device)
    if model == 'periodic_flatten_mlp':
        return DirectPeriodicModel(seed, arm['context_continuous'], device=device)
    raise ValueError('Unknown combination model: ' + str(model))


def load_combo(arm, path, device):
    model_kind = arm['model']
    if model_kind == 'flatten_mlp':
        model = MatrixModel.load(path, device)
        if model.kind != 'flatten_mlp' or model.width != arm['width']:
            raise ValueError('Archive is not this data arm')
    elif model_kind == 'masked_flatten_mlp':
        model = MaskedFlattenModel.load(path, device)
        if model.drop_context != arm['drop_context'] or model.drop_history != arm['drop_history']:
            raise ValueError('Archive is not this pitcher-information arm')
    elif model_kind == 'residual_mlp':
        model = DirectResidualModel.load(path, device)
    elif model_kind == 'periodic_flatten_mlp':
        model = DirectPeriodicModel.load(path, device)
        if model.context_continuous != list(arm['context_continuous']):
            raise ValueError('Archive continuous map differs from the arm')
    else:
        raise ValueError('Unknown combination model: ' + str(model_kind))
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    return model
