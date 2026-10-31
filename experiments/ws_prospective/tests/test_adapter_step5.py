"""STEP5_2024_EARLY_3HEAD adapter (process B): parity with the accepted Step 5 shadow run, leakage, restart, timing.

Runs only under the lightgbm venv (skipped elsewhere). The B interpreter has no pytest, so this file is plain
``unittest`` (pytest collects it too):

    cd experiments/ws_prospective && $B -B tests/test_adapter_step5.py

Synthetic only: the accepted seq-shadow test game, a fixture profile, a fake store. The REAL frozen 2024 heads are
loaded through the accepted ``shadow_runner.load_heads_smoke_form`` (reviewer gate first). The reference is the
pinned ``shadow_runner.run_shadow`` with those same real heads on the same bodies.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import resource
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

try:
    import lightgbm  # noqa: F401
except ImportError:
    lightgbm = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import fixtures_s5 as FX  # noqa: E402

MID = "STEP5_2024_EARLY_3HEAD"
TRANSIENT_TIMING = ("timing_unknown", "unavailable_at_decision", "disconnected", "stale", "conflicted",
                    "uncertain_ordering", "late")


def _outcome(adapter, ctx):
    from ws_prospective.runner import Abstain
    try:
        return adapter.predict(ctx)
    except Abstain as e:
        return ("ABSTAIN", str(e))


@unittest.skipIf(lightgbm is None, "Step 5 adapter tests run under the lightgbm venv (process B) only")
class Step5AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ws_prospective import step5_bridge as B
        from ws_prospective.adapters import PinMismatch
        cls.B, cls.PinMismatch = B, PinMismatch
        cls.tmp = Path(tempfile.mkdtemp(prefix="rs01-step5-"))
        tmp = cls.tmp
        cls.ms = FX.model_sources()
        spec = cls.ms[MID]
        SS = B.import_seq_shadow(spec)
        env = B.load_step5(SS, spec["step5_shadow"])
        import profile_reader
        cls.F, cls.P = env.src.F, env.src.P
        cls.profile_path, cls.profile_sha = FX.write_profile(tmp / "profile", cls.F, profile_reader)
        cls.bodies = FX.script()
        cls.store = tmp / "store.sqlite3"
        cls.refs = FX.write_store(cls.store, cls.bodies, cls.F)
        cls.game = {"game_pk": FX.GAME, "official_date": FX.DAY, "store_path": str(cls.store),
                    "profile_path": str(cls.profile_path), "profile_sha256": cls.profile_sha}
        cls.reg = {"model_sources": cls.ms}
        from ws_prospective.adapter_step5 import Step5Adapter
        cls.cls = Step5Adapter
        (tmp / "out").mkdir()
        t0 = time.perf_counter()
        cls.adapter = Step5Adapter.build(cls.reg, cls.game, tmp / "out")
        cls.build_ms = (time.perf_counter() - t0) * 1000
        cls.ctxs = FX.contexts(cls.bodies, cls.refs, cls.profile_sha)
        cls.results, cls.vectors, cls.ms_each = {}, {}, []
        for i, ctx in cls.ctxs:
            t0 = time.perf_counter()
            cls.results[i] = _outcome(cls.adapter, ctx)
            cls.ms_each.append((time.perf_counter() - t0) * 1000)
            if isinstance(cls.results[i], dict):
                cls.vectors[i] = cls.adapter.last["vectors_sha256"]
        cls.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        again = Step5Adapter.build(cls.reg, cls.game, tmp / "out")
        cls.restart = {i: _outcome(again, ctx) for i, ctx in cls.ctxs}
        cls.out_files = list((tmp / "out").iterdir())
        cls.refusals = cls.build_refusals(tmp / "refusals")
        cls.reference(SS, spec)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def build_refusals(cls, tmp):
        out = {}
        bad = copy.deepcopy(cls.ms)
        k = next(iter(bad[MID]["source_pins"]))
        bad[MID]["source_pins"][k] = "0" * 64
        gate = copy.deepcopy(cls.ms)
        gate[MID]["model_gate"]["sha256"] = "0" * 64
        py = copy.deepcopy(cls.ms)
        py[MID]["python"] = "/usr/bin/false-python"
        cases = {"source_pin": ({"model_sources": bad}, cls.game), "gate": ({"model_sources": gate}, cls.game),
                 "python": ({"model_sources": py}, cls.game),
                 "profile_sha": (cls.reg, dict(cls.game, profile_sha256="1" * 64)),
                 "not_strictly_prior": (cls.reg, dict(cls.game, official_date=FX.CUTOFF))}
        for name, (reg, game) in cases.items():
            try:
                cls.cls.build(reg, game, tmp)
                out[name] = None
            except cls.PinMismatch as e:
                out[name] = str(e)
        return out

    @classmethod
    def reference(cls, SS, spec):
        mods = SS.load_step5_shadow({"step5_shadow": spec["step5_shadow"]})
        S, G, RUN = mods.S, mods.G, mods.RUN
        step5 = S.load_config()
        src = S.load(step5)
        profile = src.A.load_committed(src.F, cls.profile_path.parent)
        registry = src.P.load_registry(step5["models"]["path"], step5["models"]["sha256"], step5["encoder"]["sha256"])
        heads = RUN.load_heads_smoke_form(src.P, registry, spec["model_gate"]["path"], RUN.AUTHOR, step5)
        replay = FX.write_replay(cls.tmp / "replay", cls.bodies, src.F)
        cfg = copy.deepcopy(step5)
        cfg["protected"] = {"processes": [], "command_words": []}
        cfg["heavy_lock_path"] = str(cls.tmp / "no-heavy-lock")
        cfg["caps"]["min_free_memory_bytes"] = 1 << 30
        g = G.Guard(cfg, run=FX.fake_run, arm_hard_stops=False)
        g.preflight()
        rc, cls.summary = RUN.run_shadow(replay, cls.tmp / "step5", config=cfg, src=src, profile=profile, heads=heads,
                                         guard=g, mode="synthetic")
        assert rc == 0
        db = cls.tmp / "step5" / "shadow-ledger.sqlite3"
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            cls.ref_rows = con.execute("SELECT replay_file, decision, outcome, reason, action_probabilities, "
                                       "vectors_sha256, model_kind FROM shadow_forecasts "
                                       "WHERE replay_file IS NOT NULL").fetchall()

    # ---------------------------------------------------------------------------------------------- 1. parity
    def test_parity_with_accepted_step5_shadow_run(self):
        import numpy as np
        cands = set(FX.MODEL_SOURCES[MID]["candidate_pitch_types"])
        scored = abstained = timing = cells = 0
        max_diff = 0.0
        self.assertTrue(self.ref_rows)
        for replay_file, decision, outcome, reason, probs_json, vsha, kind in self.ref_rows:
            i = int(replay_file.split("_")[1]) - 1
            got = self.results[i]
            if decision == "scored":
                self.assertEqual(kind, "frozen_2024_comparator")
                self.assertIsInstance(got, dict, (replay_file, got))
                self.assertEqual(self.vectors[i], vsha)
                ref = {c["action"]: c["seven_leaf"] for c in json.loads(probs_json) if c["status"] == "scored"}
                self.assertEqual(sorted(got), sorted(a for a in ref if a in cands))
                for a, p in got.items():
                    max_diff = max(max_diff, float(np.max(np.abs(np.array(p) - np.array(ref[a])))))
                    cells += 1
                scored += 1
            elif outcome in TRANSIENT_TIMING:
                timing += 1  # timing gates are the RS-01 runner's job (documented omission)
            else:
                self.assertEqual(got, ("ABSTAIN", f"{outcome}:{reason}"), replay_file)
                abstained += 1
        self.assertGreaterEqual(scored, 6)
        self.assertEqual(max_diff, 0.0)
        print(f"\nSTEP5 PARITY scored={scored} cells={cells} abstained={abstained} ref_timing_abstain={timing} "
              f"max_abs_diff={max_diff} adapter_contexts={len(self.ctxs)} "
              f"adapter_scored={sum(isinstance(r, dict) for r in self.results.values())}")

    # ---------------------------------------------------------------------------------------------- 2. leakage
    def test_no_source_body_and_sha_mismatch_abstain(self):
        _, ctx = self.ctxs[1]
        self.assertEqual(_outcome(self.adapter, dataclasses.replace(ctx, source_seq=None)), ("ABSTAIN", "NO_SOURCE_BODY"))
        self.assertEqual(_outcome(self.adapter, dataclasses.replace(ctx, source_sha="0" * 64)),
                         ("ABSTAIN", "SOURCE_SHA_MISMATCH"))

    def test_body_with_target_or_other_slot_is_refused(self):
        names = [n for n, _ in self.bodies]
        by = {names[i]: c for i, c in self.ctxs}
        seq = {names[i]: ref for i, ref in enumerate(self.refs)}
        ctx = by["pa8_p1"]
        later = dataclasses.replace(ctx, source_seq=seq["pa8_p2"][0], source_sha=seq["pa8_p2"][1])
        self.assertEqual(_outcome(self.adapter, later), ("ABSTAIN", "BODY_CONTAINS_TARGET"))
        earlier = dataclasses.replace(ctx, source_seq=seq["pa8_open"][0], source_sha=seq["pa8_open"][1])
        self.assertEqual(_outcome(self.adapter, earlier), ("ABSTAIN", "CONTEXT_STATE_MISMATCH"))

    def test_future_bodies_do_not_change_a_forecast(self):
        k = 6
        store = self.tmp / "prefix.sqlite3"
        refs = FX.write_store(store, self.bodies[:k + 1], self.F)
        a = self.cls(game=dict(self.game, store_path=str(store)), env=self.adapter.env, profile=self.adapter.profile,
                     heads=self.adapter.heads, candidates=self.adapter.candidates)
        for i, ctx in FX.contexts(self.bodies[:k + 1], refs, self.profile_sha):
            self.assertEqual(_outcome(a, ctx), self.results[i])

    # ---------------------------------------------------------------------------------------------- 3. restart
    def test_restart_rebuild_gives_identical_forecasts_and_writes_nothing(self):
        self.assertEqual(self.restart, self.results)
        self.assertEqual(self.out_files, [])

    def test_refusals(self):
        r = self.refusals
        self.assertIn("pinned file changed or missing", r["source_pin"])
        self.assertIn("model gate", r["gate"])
        self.assertIn("must run under", r["python"])
        self.assertIn("profile_sha256", r["profile_sha"])
        self.assertIn("strictly_prior_to_official_date", r["not_strictly_prior"])
        self.assertNotIn("torch", sys.modules)

    # ---------------------------------------------------------------------------------------------- 4. timing
    def test_timing_and_rss(self):
        ms = sorted(self.ms_each)
        rss_mb = self.rss / (1024 * 1024) if sys.platform == "darwin" else self.rss / 1024
        print(f"\nSTEP5 TIMING n={len(ms)} p50_ms={ms[len(ms) // 2]:.2f} max_ms={ms[-1]:.2f} "
              f"build_ms={self.build_ms:.0f} peak_rss_mb={rss_mb:.0f} "
              "(process peak after build+predict, before the reference run; synthetic profile)")
        self.assertLess(ms[-1], 2000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
