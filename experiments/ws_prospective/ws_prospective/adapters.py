"""Forecaster adapters named in the registration.

Contract for every adapter (see ``AdapterBase``):

* ``build(reg, game, out_dir)`` constructs it from the executable registration
  (``reg["model_sources"][model_id]`` pins its source files and artifacts) and
  one ``reg["execution"]["games"][i]`` entry: game_pk, official_date,
  store_path, profile_path, profile_sha256;
* ``predict(ctx)`` returns {candidate_pitch_type: 7 probabilities in LEAVES
  order} or raises ``runner.Abstain(reason)``.  It may read only the stored
  snapshot named by ``ctx.source_seq`` (verified against ``ctx.source_sha``),
  which by construction does not contain the target pitch; contexts without a
  source body (post-hoc) must abstain.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Dict, List

from . import LEAVES
from .context import PrePitchContext

# FIT seven-leaf counts of the T-FINAL fit window, as fixed in
# seq_shadow/postcutoff_offline.py FIT_COUNTS / FIT_TOTAL (no smoothing).
FIT_COUNTS = (258626, 132039, 2144, 78550, 124789, 51974, 72383)
FIT_TOTAL = 720505
assert sum(FIT_COUNTS) == FIT_TOTAL


class PinMismatch(ValueError):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_pins(root: Path, files: Dict[str, str]) -> None:
    """files: {relative path under root (or absolute): sha256}."""
    for rel, digest in files.items():
        p = Path(rel) if Path(rel).is_absolute() else Path(root) / rel
        if not p.is_file() or sha256_file(p) != digest:
            raise PinMismatch(f"pinned file changed or missing: {p}")


def pinned_sys_path(entries: List[str]) -> None:
    """Prepend pinned source roots once (verified by verify_pins beforehand)."""
    for e in reversed(entries):
        if e not in sys.path:
            sys.path.insert(0, e)
    importlib.invalidate_caches()


class AdapterBase:
    model_id = ""
    leaf_order = LEAVES

    @classmethod
    def build(cls, reg: dict, game: dict, out_dir: Path) -> "AdapterBase":
        raise NotImplementedError

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        raise NotImplementedError


class FrozenFitMarginal(AdapterBase):
    model_id = "FIT_MARGINAL_7"

    @classmethod
    def build(cls, reg, game, out_dir):
        return cls()

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        return {"*": [c / FIT_TOTAL for c in FIT_COUNTS]}


def registry() -> dict:
    """model_id -> adapter class. Real adapters are imported lazily so process B
    (lightgbm venv) never imports torch and process A never imports lightgbm."""
    reg = {FrozenFitMarginal.model_id: FrozenFitMarginal}
    for mod, name in (("adapter_t", "TFinalAdapter"), ("adapter_n1", "N1Cal2Adapter"),
                      ("adapter_step5", "Step5Adapter"), ("fit_type_freq", "FitTypeFreq7")):
        mid = {"TFinalAdapter": "T_FINAL_3SEED_TEMP", "N1Cal2Adapter": "SEQ001_N1_3SEED_CAL2",
               "Step5Adapter": "STEP5_2024_EARLY_3HEAD", "FitTypeFreq7": "FIT_TYPE_FREQ_7"}[name]

        def _lazy(mod=mod, name=name):
            return getattr(importlib.import_module(f"{__package__}.{mod}"), name)
        reg[mid] = _lazy
    return reg


def resolve(model_id: str):
    entry = registry()[model_id]
    return entry if isinstance(entry, type) else entry()
