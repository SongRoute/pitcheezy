"""Model factory for the DRAFT PLE-embedding arms (see direct_ple_registry).

ple_residual_counts -> direct_ple_model.DirectPLEResidualModel and ple_tabm_residual_counts ->
DirectPLETabMResidualModel (k=8, scaling init normal), both with the counts continuous map; the runner appends the
workload scalar after the 52 stripped context columns exactly as for the P32/P34 counts arms.  Bins are computed in
fit from the TRAIN argument only.  load refuses an archive of another kind or format (class loaders), another
architecture (backbone, embedding incl. n_bins / d, k, scaling init), a continuous map other than
CONTINUOUS['counts'] (same length or not), another context width, or bins whose hash differs from the hash the
run recorded for this task (fit.json ``ple_bins_sha256``).
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_ple_model import (ARCHITECTURE, TABM_ARCHITECTURE, DirectPLEResidualModel,
                                       DirectPLETabMResidualModel, bins_sha256)
from pitchmdp.direct_ple_registry import ARMS, CONTINUOUS, FLAGS


def _tabm(name):
    return FLAGS[name] == 'ple_tabm_residual'


def build_ple(arm, seed, device):
    """Unfitted model for a checked PLE arm (config dict)."""
    spec = ARMS[arm['name']]
    if _tabm(arm['name']):
        return DirectPLETabMResidualModel(seed, spec['context_continuous'], spec['k'], spec['scaling_init'], device)
    return DirectPLEResidualModel(seed, spec['context_continuous'], device=device)


def load_ple(arm, path, device, recorded_bins_sha256):
    """Strict load; ``recorded_bins_sha256`` is the run's own record for this task (never the archive's)."""
    spec = ARMS[arm['name']]
    if _tabm(arm['name']):
        cls, expected = DirectPLETabMResidualModel, {**TABM_ARCHITECTURE, 'k': spec['k'],
                                                     'scaling_init': spec['scaling_init']}
    else:
        cls, expected = DirectPLEResidualModel, ARCHITECTURE
    model = cls.load(path, device)  # refuses another format / kind / malformed or tampered bins
    embedding = model.architecture['embedding']
    if type(model) is not cls or model.kind != spec['model'] or model.architecture != expected \
            or (embedding['n_bins'], embedding['d_embedding']) != (spec['n_bins'], spec['d_embedding']):
        raise ValueError('Archive is not this PLE arm')
    if spec['context_continuous'] != CONTINUOUS or model.context_continuous != CONTINUOUS \
            or model.net.config['context_continuous'] != CONTINUOUS:
        raise ValueError("Archived continuous map differs from CONTINUOUS['counts']")
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    if not recorded_bins_sha256 or bins_sha256(model.ple_bins) != recorded_bins_sha256 \
            or model.net.config['ple_bins_sha256'] != recorded_bins_sha256:
        raise ValueError('Archived PLE bins differ from the bins hash recorded by the run')
    return model
