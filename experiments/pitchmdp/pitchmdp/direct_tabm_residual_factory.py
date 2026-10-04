"""Model factory for the DRAFT TabM-mini x periodic-residual arm (see direct_tabm_residual_registry).

tabm_residual_counts -> direct_tabm_residual_model.DirectTabMResidualModel (frozen PLR-lite
embedding + frozen residual backbone + TabM-mini k=8) with the counts continuous map; the
runner appends the workload scalar after the 52 stripped context columns exactly as for the
P27/P30/P31/P32 counts arms.  load refuses an archive of another kind or format, another
architecture (backbone, embedding, k, scaling init), a continuous map other than the
registered one (same length or not) or another context width.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_tabm_residual_model import ARCHITECTURE, DirectTabMResidualModel
from pitchmdp.direct_tabm_residual_registry import ARMS


def build_tabm_residual(arm, seed, device):
    """Unfitted model for a checked TabM-residual arm (config dict)."""
    spec = ARMS[arm['name']]
    return DirectTabMResidualModel(seed, spec['context_continuous'], spec['k'], spec['scaling_init'], device)


def load_tabm_residual(arm, path, device):
    spec = ARMS[arm['name']]
    model = DirectTabMResidualModel.load(path, device)  # refuses another format / kind
    expected = {**ARCHITECTURE, 'k': spec['k'], 'scaling_init': spec['scaling_init']}
    if type(model) is not DirectTabMResidualModel or model.kind != spec['model'] or model.architecture != expected:
        raise ValueError('Archive is not this TabM-residual arm')
    if model.context_continuous != list(spec['context_continuous']) \
            or model.net.config['context_continuous'] != list(spec['context_continuous']):
        raise ValueError('Archived continuous map differs from the registered arm')
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    return model
