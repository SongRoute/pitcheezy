"""Model factory for the DRAFT TabM-mini arms (see direct_tabm_registry).

Every new arm is direct_tabm_model.DirectTabMModel (frozen periodic network + TabM-mini
k=8 head) with the counts continuous map; the runner appends the workload scalar after
the 52 stripped context columns exactly as for the P27/P30 counts arms.  load refuses
an archive of another kind, k, head dropout, scaling init, continuous map or context width.
"""
from __future__ import annotations

from pitchmdp.direct_combo_registry import N_CONTEXT
from pitchmdp.direct_tabm_model import DirectTabMModel
from pitchmdp.direct_tabm_registry import ARMS


def build_tabm(arm, seed, device):
    """Unfitted model for a checked TabM-mini arm (config dict)."""
    spec = ARMS[arm['name']]
    return DirectTabMModel(seed, spec['context_continuous'], spec['head_dropout'], spec['k'], spec['scaling_init'],
                           device)


def load_tabm(arm, path, device):
    spec = ARMS[arm['name']]
    model = DirectTabMModel.load(path, device)
    a = model.architecture
    if (a['head_dropout'], a['k'], a['scaling_init']) != (spec['head_dropout'], spec['k'], spec['scaling_init']) \
            or model.context_continuous != list(spec['context_continuous']):
        raise ValueError('Archive is not this TabM-mini arm')
    if model.net.config['n_context'] != N_CONTEXT + len(arm['workload']):
        raise ValueError('Archive context width differs from the arm (52 + workload)')
    return model
