"""STEP5_2024_EARLY_3HEAD adapter (process B: lightgbm 4.7.0 venv, no torch).

Replicates the scored cells of the accepted Step 5 shadow run (``shadow_runner`` -> ``live_adapter.live_decide``
-> ``pipeline.decide``) for one body: ``step5_bridge.prepare`` (pre-pitch state, support, 17 x 199 F1 matrix),
then exactly decide's head call ``seven_leaves(*(heads['predict'][h](x[eligible]) for h in HEADS), temperature)``.
The heads are loaded in the accepted real-run form ``shadow_runner.load_heads_smoke_form`` (reviewer gate
MODEL-GATE.json first, bounded read, all three sha256 before any Booster, ``Booster(model_str=...)``, header and
iteration checks, ``num_threads=1``). Only provisionally eligible actions are scored (as decide); the returned
dict holds those among the registered candidate codes. If none, the context abstains.

Registration (``reg["model_sources"]["STEP5_2024_EARLY_3HEAD"]``):
  python               the lightgbm venv interpreter (enforced)
  seq_shadow_src_root  src root holding pitcheezy/seq_shadow (only its stdlib loader is used here)
  step5_shadow         {dir, sha256: {shadow_sources.py, shadow_guard.py, shadow_runner.py, shadow_config.json}}
  model_gate           {path, sha256} of the reviewer compatibility PASS
  source_pins, pin_roots, profile_scope, candidate_pitch_types  (as for N1)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

from . import LEAVES
from . import step5_bridge as B
from .adapters import AdapterBase, PinMismatch, sha256_file
from .runner import Abstain


class Step5Adapter(AdapterBase):
    model_id = "STEP5_2024_EARLY_3HEAD"

    def __init__(self, *, game, env, profile, heads, candidates):
        self.game, self.env, self.profile, self.heads = game, env, profile, heads
        self.candidates = list(candidates)
        self.last: dict = {}

    @classmethod
    def build(cls, reg, game, out_dir):
        spec = reg["model_sources"][cls.model_id]
        if "python" in spec and Path(sys.executable) != Path(spec["python"]):
            raise PinMismatch(f"Step 5 adapter must run under {spec['python']}, not {sys.executable}")
        SS = B.import_seq_shadow(spec)
        env = B.load_step5(SS, spec["step5_shadow"])
        F, P, RUN, cfg = env.src.F, env.src.P, env.RUN, env.config
        if list(P.LEAVES) != list(LEAVES):
            raise PinMismatch("Step 5 leaf order differs from the registered LEAVES")
        registry = P.load_registry(cfg["models"]["path"], cfg["models"]["sha256"], env.encoder_sha256)
        gate = spec["model_gate"]
        if sha256_file(gate["path"]) != gate["sha256"]:
            raise PinMismatch("model gate receipt differs from its pin")
        heads = RUN.load_heads_smoke_form(P, registry, gate["path"], RUN.AUTHOR, cfg)
        # decide's registry_ok, checked once (heads never change afterwards)
        if not (all(heads.get(k) == registry[k] for k in ("sha256", "temperature", "selected_iteration"))
                and env.encoder_sha256 == registry["encoder_sha256"]
                and all(isinstance(t, float) and math.isfinite(t) and t > 0 for t in heads["temperature"].values())):
            raise PinMismatch("head hash, temperature, iteration or encoder differs from the registry")
        B.check_no_unpinned(spec, reg)
        profile = B.load_game_profile(env.src, game, spec.get("profile_scope", B.ACTUAL_SCOPE))
        bad = [c for c in spec["candidate_pitch_types"] if c not in F.ACTIONS]
        if bad:
            raise PinMismatch(f"candidate codes outside the 17 Step 5 actions: {bad}")
        return cls(game=game, env=env, profile=profile, heads=heads, candidates=spec["candidate_pitch_types"])

    def predict(self, ctx) -> Dict[str, List[float]]:
        import numpy as np
        F, P = self.env.src.F, self.env.src.P
        feed = B.read_feed(self.game, ctx, F)
        prep = B.prepare(self.env, feed, ctx, self.profile)
        try:  # only supported rows are ever passed to a head (decide)
            rows = np.ascontiguousarray(prep.x_f1[prep.eligible])
            leaves = P.seven_leaves(*(self.heads["predict"][h](rows) for h in P.HEADS), self.heads["temperature"])
        except Exception as exc:  # noqa: BLE001
            raise Abstain(f"predictor_error:predict:{type(exc).__name__}:{exc}"[:300]) from None
        out = {F.ACTIONS[i]: [float(v) for v in leaf] for i, leaf in zip(prep.eligible, leaves)
               if F.ACTIONS[i] in self.candidates}
        self.last = {"vectors_sha256": prep.vectors_sha256, "eligible": [F.ACTIONS[i] for i in prep.eligible]}
        if not out:
            raise Abstain("NO_REGISTERED_CANDIDATE_SCORED")
        return out
