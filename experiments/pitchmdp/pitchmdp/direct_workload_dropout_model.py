"""Model factory for the DRAFT periodic workload x head-dropout arms (see direct_workload_dropout_registry).

No new network or fit loop: every new arm is direct_dropout_model.PeriodicHeadDropoutModel
(unedited; frozen periodic network + Dropout(p) after both head GELUs) with its
level's continuous map; the runner appends the workload scalars after the 52
stripped context columns exactly as for the P27 combination arms.  load refuses an
archive of another kind, head dropout, continuous map or context width.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutModel
from pitchmdp.direct_workload_dropout_registry import ARMS


def build_workload_dropout(arm, seed, device):
    """Unfitted model for a checked workload-dropout arm (config dict)."""
    spec = ARMS[arm['name']]
    return PeriodicHeadDropoutModel(seed, spec['context_continuous'], spec['head_dropout'], device)


def load_workload_dropout(arm, path, device):
    spec = ARMS[arm['name']]
    model = PeriodicHeadDropoutModel.load(path, device)
    if model.architecture['head_dropout'] != spec['head_dropout'] \
            or model.context_continuous != list(spec['context_continuous']):
        raise ValueError('Archive is not this workload-dropout arm')
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    return model
