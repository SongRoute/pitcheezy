"""Model factory for the DRAFT embedded-input residual arms (see direct_embed_residual_registry).

periodic_residual_counts -> DirectPeriodicResidualModel, linear_residual_counts ->
DirectLinearResidualModel (direct_periodic_residual_model), both with the counts
continuous map; the runner appends the workload scalar after the 52 stripped context
columns exactly as for the P27/P30/P31 counts arms.  load refuses an archive of the
other embedding, another architecture, a continuous map other than the registered one
(same length or not; same-model review F4 of 1b1ff30) or another context width.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_embed_residual_registry import ARMS
from pitchmdp.direct_periodic_residual_model import (ARCHITECTURES, DirectLinearResidualModel,
                                                     DirectPeriodicResidualModel)

MODELS = {'periodic': DirectPeriodicResidualModel, 'linear': DirectLinearResidualModel}


def build_embed_residual(arm, seed, device):
    """Unfitted model for a checked embedded-residual arm (config dict)."""
    spec = ARMS[arm['name']]
    return MODELS[spec['embedding']](seed, spec['context_continuous'], device=device)


def load_embed_residual(arm, path, device):
    spec = ARMS[arm['name']]
    cls = MODELS[spec['embedding']]
    model = cls.load(path, device)  # refuses the other embedding's format / kind
    if type(model) is not cls or model.kind != spec['model'] or model.architecture != ARCHITECTURES[spec['embedding']]:
        raise ValueError('Archive is not this embedded-residual arm')
    if model.context_continuous != list(spec['context_continuous']) \
            or model.net.config['context_continuous'] != list(spec['context_continuous']):
        raise ValueError('Archived continuous map differs from the registered arm')
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    return model
