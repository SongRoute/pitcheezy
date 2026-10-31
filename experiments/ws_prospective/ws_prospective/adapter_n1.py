"""SEQ001_N1_3SEED_CAL2 adapter (process A, .venv with torch, no lightgbm).

Replicates the forecast of the accepted ``scripts/seq_shadow_runner.py`` (``_seq_first_decision``) for one body:
Step 5 pre-pitch state and raw 91 columns (``step5_bridge.prepare``) -> ``features.apply_encoder(raw, scale-002)``
(asserted bitwise equal to ``seq_shadow.adapter.encode_for_seq``) -> context tokens ``seq_shadow.tokens.build`` from
earlier pitches in the same body -> ``SeqAdapter.forecast`` -> ``sel_ensemble`` (mean over the 3 seeds of each
seed's CAL2-selected arm), one 7-leaf row per Step 5 action. N1 is conditioned on the candidate pitch type through
the 17 candidate rows (type one-hot + the pitcher's strict-prior per-type profile columns); all 17 rows are
forecast whenever Step 5 built its matrix (as in the accepted runner), and the registered candidate codes are
returned (PO is not a registered candidate).

Registration (``reg["model_sources"]["SEQ001_N1_3SEED_CAL2"]``):
  seq_shadow_src_root  src root holding pitcheezy/seq_shadow (codex-ws-postseason-six-1017 @7a64583)
  config               {path, sha256} of configs/seq_shadow.json rev 2 (pins weights, temperatures, CAL2 result,
                       scale-002 encoder, architecture, train_seq.py and the Step 5 shadow dir + 4 sha256)
  source_pins          {absolute path: sha256} of every source module imported (verified before import)
  pin_roots            directories whose loaded modules must all be in source_pins (checked after import)
  profile_scope        'ACTUAL_S2_ONE_PASS' (tests: 'SYNTHETIC_OWNED_FIXTURE')
  candidate_pitch_types
"""
from __future__ import annotations

import json
from typing import Dict, List

from . import LEAVES
from . import step5_bridge as B
from .adapters import AdapterBase, PinMismatch
from .runner import Abstain


class N1Cal2Adapter(AdapterBase):
    model_id = "SEQ001_N1_3SEED_CAL2"

    def __init__(self, *, game, env, profile, config, seq_adapter, encoder_seq, token_kw, candidates, mods):
        self.game, self.env, self.profile, self.config = game, env, profile, config
        self.seq_adapter, self.encoder_seq, self.token_kw = seq_adapter, encoder_seq, token_kw
        self.candidates = list(candidates)
        self.AD, self.TOK = mods
        self.last: dict = {}

    @classmethod
    def build(cls, reg, game, out_dir):
        spec = reg["model_sources"][cls.model_id]
        SS = B.import_seq_shadow(spec)
        raw = SS.read_pinned(spec["config"]["path"], spec["config"]["sha256"])
        config = json.loads(raw)
        if not str(config.get("status", "")).startswith("provisional") or \
                [k for k in SS.REQUIRED_KEYS if k not in config]:
            raise PinMismatch("seq shadow configuration incomplete")
        env = B.load_step5(SS, config["step5_shadow"])
        from pitcheezy.seq_shadow import adapter as AD
        from pitcheezy.seq_shadow import calibration as CAL
        from pitcheezy.seq_shadow import model as MOD
        from pitcheezy.seq_shadow import tokens as TOK
        F, P = env.src.F, env.src.P
        # seq_shadow_runner.load_seq, unchanged order
        encoder_seq, _ = AD.load_seq_encoder(config)
        CAL.load_calibrators(config, [int(s) for s in config["model"]["seeds"]])
        ensemble = MOD.load_ensemble(config)
        calibrators = CAL.load_calibrators(config, ensemble.seeds, ensemble.temperatures)
        seq_adapter = AD.SeqAdapter(ensemble, calibrators, config["model"]["candidate_rows"])
        # run_seq_shadow refusals
        if list(P.LEAVES) != list(config["metrics"]["leaf_order"]) or list(P.LEAVES) != list(LEAVES) \
                or len(F.ACTIONS) != config["model"]["candidate_rows"]:
            raise PinMismatch("leaf order or action count differs from the registration")
        arch = ensemble.architecture["context"]
        if len(F.TYPES) != arch["pitch_type_vocab"]:
            raise PinMismatch("pitch type vocabulary differs from the 19 frozen types")
        token_kw = {"length": ensemble.length, "types": F.TYPES, "description": env.RUN.DESCRIPTION,
                    "pitch_number_range": config["tokens"]["pitch_number_range"],
                    "balls_vocab": arch["balls_vocab"], "strikes_vocab": arch["strikes_vocab"],
                    "missing_type": config["tokens"]["missing_type_policy"]}
        B.check_no_unpinned(spec, reg)
        profile = B.load_game_profile(env.src, game, spec.get("profile_scope", B.ACTUAL_SCOPE))
        bad = [c for c in spec["candidate_pitch_types"] if c not in F.ACTIONS]
        if bad:
            raise PinMismatch(f"candidate codes outside the 17 Step 5 actions: {bad}")
        return cls(game=game, env=env, profile=profile, config=config, seq_adapter=seq_adapter,
                   encoder_seq=encoder_seq, token_kw=token_kw, candidates=spec["candidate_pitch_types"],
                   mods=(AD, TOK))

    def predict(self, ctx) -> Dict[str, List[float]]:
        import numpy as np
        F = self.env.src.F
        feed = B.read_feed(self.game, ctx, F)
        prep = B.prepare(self.env, feed, ctx, self.profile)
        x_seq = F.apply_encoder(prep.raw, self.encoder_seq)
        if x_seq.dtype != np.float32 or not np.array_equal(x_seq, self.AD.encode_for_seq(prep.raw, self.encoder_seq)):
            raise RuntimeError("scale-002 encoding differs between features.apply_encoder and encode_for_seq")
        try:
            tok, counts = self.TOK.build(feed, prep.state, **self.token_kw)
            fc = self.seq_adapter.forecast(x_seq, tok)
        except Exception as exc:  # noqa: BLE001 - as the accepted runner: an abstention with the exact reason
            raise Abstain(f"seq_predictor_error:{type(exc).__name__}:{exc}"[:300]) from None
        sel = fc["sel_ensemble"]
        self.last = {"vectors_sha256": prep.vectors_sha256, "token_counts": counts,
                     "forward_ms": fc["forward_ms"], "calibration_ms": fc["calibration_ms"]}
        return {a: [float(v) for v in sel[i]] for i, a in enumerate(F.ACTIONS) if a in self.candidates}
