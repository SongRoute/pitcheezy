"""Model factory for the DRAFT robust-clip arm (see direct_robust_clip_registry).

robust_clip_residual_counts -> direct_robust_clip_model.DirectRobustClipResidualModel with the counts continuous map
(the same 41-column list as periodic_residual_counts); the runner appends the workload scalar after the 52 stripped
context columns exactly as for the P32/P34/P35 counts arms.  Robust-scale statistics are computed in fit from the
TRAIN argument only and never recomputed at predict or load time.  Both build and load refuse a module file whose
sha256 differs from the registry pin.  load refuses an archive of another kind or format (class loader), another
architecture, a continuous map other than CONTINUOUS['counts'] (same length or not), another context width, or
statistics whose hash differs from the hash the run recorded for this task (fit.json ``rc_stats_sha256``).
The two reference arms (residual_counts, periodic_residual_counts) are NOT built here: they keep their old paths.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_robust_clip_registry import ARMS, CONTINUOUS, check_module


def build_robust_clip(arm, seed, device):
    """Unfitted model for a checked robust-clip arm (config dict)."""
    check_module()
    from pitchmdp.direct_robust_clip_model import DirectRobustClipResidualModel
    return DirectRobustClipResidualModel(seed, ARMS[arm['name']]['context_continuous'], device=device)


def load_robust_clip(arm, path, device, recorded_stats_sha256):
    """Strict load; ``recorded_stats_sha256`` is the run's own record for this task (never the archive's)."""
    check_module()
    from pitchmdp.direct_robust_clip_model import ARCHITECTURE, DirectRobustClipResidualModel, stats_sha256
    spec = ARMS[arm['name']]
    model = DirectRobustClipResidualModel.load(path, device)  # refuses another format / kind / tampered statistics
    if type(model) is not DirectRobustClipResidualModel or model.kind != spec['model'] \
            or model.architecture != ARCHITECTURE:
        raise ValueError('Archive is not this robust-clip arm')
    if spec['context_continuous'] != CONTINUOUS or model.context_continuous != CONTINUOUS \
            or model.net.config['context_continuous'] != CONTINUOUS:
        raise ValueError("Archived continuous map differs from CONTINUOUS['counts']")
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    if not recorded_stats_sha256 or stats_sha256(*model.rc_stats) != recorded_stats_sha256 \
            or model.net.config['rc_stats_sha256'] != recorded_stats_sha256:
        raise ValueError('Archived robust-scale statistics differ from the hash recorded by the run')
    return model
