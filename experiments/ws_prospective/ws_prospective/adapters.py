"""Forecaster adapters named in the registration.

Implemented here: ``FrozenFitMarginal`` (comparator; a code constant, no data
read).  The three model adapters are *declared* with their pinned entry points
but deliberately not implemented in the preparation step: wiring them to the
pinned seq_shadow source (GUMBO body -> captured_bridge -> StreamState.decide
-> Runtime.predict for every registered candidate pitch type) is an execution
step that needs its own review before Game 1 (REGISTRATION open item O-1).
"""
from __future__ import annotations

from typing import Dict, List

from . import LEAVES
from .context import PrePitchContext

# FIT seven-leaf counts of the T-FINAL fit window, as fixed in
# seq_shadow/postcutoff_offline.py FIT_COUNTS / FIT_TOTAL (no smoothing).
FIT_COUNTS = (258626, 132039, 2144, 78550, 124789, 51974, 72383)
FIT_TOTAL = 720505
assert sum(FIT_COUNTS) == FIT_TOTAL


class FrozenFitMarginal:
    model_id = "FIT_MARGINAL_7"
    leaf_order = LEAVES

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        return {"*": [c / FIT_TOTAL for c in FIT_COUNTS]}


class _Declared:
    model_id = ""
    leaf_order = LEAVES
    entry_point = ""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            f"{self.model_id}: adapter over {self.entry_point} is registered but not yet "
            "implemented/reviewed (open item O-1); RS-01 preparation runs no real model")


class TFinalAdapter(_Declared):
    model_id = "T_FINAL_3SEED_TEMP"
    entry_point = "pitcheezy.seq_shadow.fullseason_runtime:Runtime.predict"


class N1Cal2Adapter(_Declared):
    model_id = "SEQ001_N1_3SEED_CAL2"
    entry_point = "pitcheezy.seq_shadow.adapter:SeqAdapter.forecast (sel_ensemble)"


class Step5Adapter(_Declared):
    model_id = "STEP5_2024_EARLY_3HEAD"
    entry_point = "step5/pipeline.py:decide (17 actions x 7 leaves)"


REGISTRY = {c.model_id: c for c in (FrozenFitMarginal, TFinalAdapter, N1Cal2Adapter, Step5Adapter)}
