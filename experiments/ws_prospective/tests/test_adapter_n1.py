"""SEQ001_N1_3SEED_CAL2 adapter (process A): parity with the accepted seq-shadow runner, leakage, restart, timing.

The whole computation runs in a child interpreter (same python A) so this pytest process never imports the
seq_shadow / Step 5 modules: process A also hosts the T adapter, whose own closure check must not see modules a
test (or the accepted reference runner) loaded. Synthetic only: the accepted seq-shadow test game (19 bodies), a fixture profile and a fake store. The REAL pinned
N1 weights, temperatures, CAL2 result and scale-002 encoder are loaded (allowed: pinned model files); no 2026 row,
store, capture or profile is opened. The reference is the accepted ``scripts/seq_shadow_runner.run_seq_shadow``
(byte-identical copy in the codex-ws-postseason-six-1017 worktree) on the same bodies, with the Step 5 fixture
ledger from the pinned ``shadow_runner.run_shadow`` (synthetic heads, as in the accepted tests).
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import os
import resource
import sqlite3
import sys
import time
from pathlib import Path

import pytest
import subprocess

try:
    import torch  # noqa: F401  (only to decide skipping; the child does the work)
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fixtures_s5 as FX  # noqa: E402

MID = "SEQ001_N1_3SEED_CAL2"
RUNNER_PATH = Path("/Users/song/Projects/pitcheezy-worktrees/codex-ws-postseason-six-1017/scripts/seq_shadow_runner.py")
RUNNER_SHA = "3657906ca2bbd84bcf29670aeac3743ca88a7e1c6a5535983d6d7602a28965b8"  # == main scripts/seq_shadow_runner.py


def _outcome(adapter, ctx):
    from ws_prospective.runner import Abstain
    try:
        return adapter.predict(ctx)
    except Abstain as e:
        return ["ABSTAIN", str(e)]


class World:
    def __init__(self, tmp: Path):
        from ws_prospective import step5_bridge as B
        self.tmp = tmp
        self.ms = FX.model_sources()
        spec = self.ms[MID]
        SS = B.import_seq_shadow(spec)
        self.SS = SS
        self.config = json.loads(SS.read_pinned(spec["config"]["path"], spec["config"]["sha256"]))
        env = B.load_step5(SS, self.config["step5_shadow"])
        import profile_reader  # pinned, on sys.path after shadow_sources.load
        self.F, self.P = env.src.F, env.src.P
        self.profile_path, self.profile_sha = FX.write_profile(tmp / "profile", self.F, profile_reader)
        self.bodies = FX.script()
        self.store = tmp / "store.sqlite3"
        self.refs = FX.write_store(self.store, self.bodies, self.F)
        self.game = {"game_pk": FX.GAME, "official_date": FX.DAY, "store_path": str(self.store),
                     "profile_path": str(self.profile_path), "profile_sha256": self.profile_sha}
        self.reg = {"model_sources": self.ms}
        from ws_prospective.adapter_n1 import N1Cal2Adapter
        self.cls = N1Cal2Adapter
        (tmp / "out").mkdir()
        t0 = time.perf_counter()
        self.adapter = N1Cal2Adapter.build(self.reg, self.game, tmp / "out")
        self.build_ms = (time.perf_counter() - t0) * 1000
        self.ctxs = FX.contexts(self.bodies, self.refs, self.profile_sha)
        self.results, self.vectors, self.ms_each = {}, {}, []
        for i, ctx in self.ctxs:
            t0 = time.perf_counter()
            self.results[i] = _outcome(self.adapter, ctx)
            self.ms_each.append((time.perf_counter() - t0) * 1000)
            if isinstance(self.results[i], dict):
                self.vectors[i] = self.adapter.last["vectors_sha256"]
        self.rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # restart and build refusals run BEFORE the accepted reference imports its ledger/summary modules
        # (check_no_unpinned would rightly refuse a process that has loaded unpinned seq_shadow modules)
        again = N1Cal2Adapter.build(self.reg, self.game, tmp / "out")
        self.restart = {i: _outcome(again, ctx) for i, ctx in self.ctxs}
        self.out_files = list((tmp / "out").iterdir())
        self.refusals = self.build_refusals(tmp / "refusals")
        self.leakage = self.leakage_cases()
        self.reference()

    def leakage_cases(self):
        names = [n for n, _ in self.bodies]
        by = {names[i]: c for i, c in self.ctxs}
        seq = {names[i]: ref for i, ref in enumerate(self.refs)}
        ctx = by["pa8_p1"]  # slot: PA 8 pitch 2
        out = {"no_source": _outcome(self.adapter, dataclasses.replace(ctx, source_seq=None)),
               "sha_mismatch": _outcome(self.adapter, dataclasses.replace(ctx, source_sha="0" * 64)),
               "later_body": _outcome(self.adapter, dataclasses.replace(ctx, source_seq=seq["pa8_p2"][0],
                                                                         source_sha=seq["pa8_p2"][1])),
               "earlier_body": _outcome(self.adapter, dataclasses.replace(ctx, source_seq=seq["pa8_open"][0],
                                                                           source_sha=seq["pa8_open"][1])),
               "target_absent_from_named_body": all(
                   c.state.pitch_number == len([e for e in self.bodies[i][1]["liveData"]["plays"]["currentPlay"]
                                                ["playEvents"] if e.get("isPitch")]) + 1 for i, c in self.ctxs)}
        k = 6  # pa8_p2: a store that holds only bodies 0..k
        store = self.tmp / "prefix.sqlite3"
        refs = FX.write_store(store, self.bodies[:k + 1], self.F)
        a = self.cls(game=dict(self.game, store_path=str(store)), env=self.adapter.env, profile=self.adapter.profile,
                     config=self.adapter.config, seq_adapter=self.adapter.seq_adapter,
                     encoder_seq=self.adapter.encoder_seq, token_kw=self.adapter.token_kw,
                     candidates=self.adapter.candidates, mods=(self.adapter.AD, self.adapter.TOK))
        out["prefix_store"] = {str(i): _outcome(a, c) for i, c in FX.contexts(self.bodies[:k + 1], refs, self.profile_sha)}
        return out

    def build_refusals(self, tmp):
        from ws_prospective import step5_bridge as B
        from ws_prospective.adapters import PinMismatch
        out = {}
        bad = copy.deepcopy(self.ms)
        k = next(iter(bad[MID]["source_pins"]))
        bad[MID]["source_pins"][k] = "0" * 64
        cases = {"source_pin": ({"model_sources": bad}, self.game),
                 "profile_sha": (self.reg, dict(self.game, profile_sha256="1" * 64)),
                 "not_strictly_prior": (self.reg, dict(self.game, official_date=FX.CUTOFF))}
        actual = copy.deepcopy(self.ms)
        actual[MID]["profile_scope"] = B.ACTUAL_SCOPE
        cases["synthetic_profile_as_actual"] = ({"model_sources": actual}, self.game)
        for name, (reg, game) in cases.items():
            try:
                self.cls.build(reg, game, tmp)
                out[name] = None
            except PinMismatch as e:
                out[name] = str(e)
        return out

    def reference(self):
        from ws_prospective import step5_bridge as B
        assert B.sha256_file(RUNNER_PATH) == RUNNER_SHA
        spec = importlib.util.spec_from_file_location("seq_shadow_runner_ref", RUNNER_PATH)
        RUNNER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RUNNER)
        SS = self.SS
        assert RUNNER.SS is SS  # one seq_shadow package in this process
        mods = SS.load_step5_shadow(self.config)
        S, G, RUN = mods.S, mods.G, mods.RUN
        step5 = S.load_config()
        src = S.load(step5)
        profile = src.A.load_committed(src.F, self.profile_path.parent)
        registry = src.P.load_registry(step5["models"]["path"], step5["models"]["sha256"], step5["encoder"]["sha256"])
        heads = RUN.synthetic_heads(src.P, registry)  # Step 5 heads are placeholders in the accepted N1 run too
        replay = FX.write_replay(self.tmp / "replay", self.bodies, src.F)
        s5cfg = copy.deepcopy(step5)
        s5cfg["protected"] = {"processes": [], "command_words": []}
        s5cfg["heavy_lock_path"] = str(self.tmp / "no-heavy-lock")
        s5cfg["caps"]["min_free_memory_bytes"] = 1 << 30
        rc, _ = RUN.run_shadow(replay, self.tmp / "step5", config=s5cfg, src=src, profile=profile, heads=heads,
                               guard=G.Guard(s5cfg, run=FX.fake_run, arm_hard_stops=False), mode="synthetic")
        assert rc == 0
        cfg = copy.deepcopy(self.config)
        cfg["protected"] = {"processes": [], "command_words": []}
        cfg["heavy_lock_path"] = str(self.tmp / "no-heavy-lock")
        cfg["caps"]["min_free_memory_bytes"] = 1 << 30
        seq = RUNNER.load_seq(cfg)
        g = G.Guard(cfg, run=FX.fake_run, arm_hard_stops=False)
        g.preflight()
        ledger = self.tmp / "step5" / "shadow-ledger.sqlite3"
        rc, self.summary = RUNNER.run_seq_shadow(
            replay, self.tmp / "main", config=cfg, config_sha256=spec_sha(cfg), step5_config=step5,
            mods=mods, src=src, profile=profile, heads=heads, guard=g, mode="synthetic", seq=seq,
            step5_ledger={"path": ledger, "sha256": SS.file_sha256(ledger),
                          "summary": json.loads((ledger.parent / "SUMMARY.json").read_text()), "summary_sha256": None})
        assert rc == 0, self.summary.get("integrity_stop")
        db = self.tmp / "main" / RUNNER.LEDGER_NAME
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            self.ref_rows = con.execute("SELECT replay_file, decision, reason, sel_ensemble_json, vectors_sha256 "
                                        "FROM seq_forecasts WHERE replay_file IS NOT NULL").fetchall()


    def report(self):
        names = [n for n, _ in self.bodies]
        return {"names": names, "actions": list(self.F.ACTIONS),
                "results": {str(i): r for i, r in self.results.items()},
                "vectors": {str(i): v for i, v in self.vectors.items()},
                "ref_rows": [list(r) for r in self.ref_rows],
                "restart": {str(i): r for i, r in self.restart.items()},
                "out_files": [str(p) for p in self.out_files], "refusals": self.refusals, "leakage": self.leakage,
                "ms_each": self.ms_each, "build_ms": self.build_ms, "rss": self.rss,
                "lightgbm_loaded": "lightgbm" in sys.modules}


def spec_sha(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    if not HAVE_TORCH:
        pytest.skip("N1 adapter runs in process A (torch) only")
    tmp = tmp_path_factory.mktemp("n1")
    out = tmp / "report.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    p = subprocess.run([sys.executable, "-B", __file__, "--child", str(tmp / "work"), str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-4000:]
    return json.loads(out.read_text())


def _index(replay_file):
    return int(replay_file.split("_")[1]) - 1


# ------------------------------------------------------------------------------------------------- 1. parity
def test_parity_with_accepted_seq_shadow_runner(world):
    actions, results = world["actions"], world["results"]
    scored = abstained = 0
    max_diff = 0.0
    assert world["ref_rows"], "accepted runner produced no first decisions"
    for replay_file, decision, reason, sel_json, vsha in world["ref_rows"]:
        i = str(_index(replay_file))
        got = results[i]
        if decision == "scored":
            assert isinstance(got, dict), (replay_file, got)
            sel = json.loads(sel_json)
            assert world["vectors"][i] == vsha
            assert sorted(got) == sorted(FX.MODEL_SOURCES[MID]["candidate_pitch_types"])
            for code, probs in got.items():
                ref = sel[actions.index(code)]
                max_diff = max(max_diff, max(abs(a - b) for a, b in zip(probs, ref)))
            scored += 1
        else:
            assert got == ["ABSTAIN", reason], (replay_file, got, reason)
            abstained += 1
    assert scored >= 6 and max_diff == 0.0
    print(f"\nN1 PARITY scored={scored} abstained={abstained} max_abs_diff={max_diff} "
          f"adapter_contexts={len(results)} adapter_scored={sum(isinstance(r, dict) for r in results.values())}")


def test_extra_contexts_are_the_omitted_lifecycle_cases(world):
    """Bodies the accepted runner did not decide on (duplicates, revisions, quarantined substitution) give the
    adapter a pure-function answer; same slot + same pre-pitch inputs -> the same vector."""
    r = {world["names"][int(i)]: v for i, v in world["results"].items()}
    assert r["pa8_p1"] == r["pa8_p1_duplicate"] == r["pa8_p1_revision"]  # pitchData change is not read
    assert r["pa8_p2_untyped"] == ["ABSTAIN", "missing_input:previous_pitch_type_not_yet_classified"]
    assert r["pa9_open_B"] == ["ABSTAIN", "unsupported_player:no_prior_pitcher_history"]


# ------------------------------------------------------------------------------------------------- 2. leakage
def test_no_source_body_and_sha_mismatch_abstain(world):
    assert world["leakage"]["no_source"] == ["ABSTAIN", "NO_SOURCE_BODY"]
    assert world["leakage"]["sha_mismatch"] == ["ABSTAIN", "SOURCE_SHA_MISMATCH"]


def test_body_with_target_or_other_slot_is_refused(world):
    assert world["leakage"]["later_body"] == ["ABSTAIN", "BODY_CONTAINS_TARGET"]
    assert world["leakage"]["earlier_body"] == ["ABSTAIN", "CONTEXT_STATE_MISMATCH"]
    assert world["leakage"]["target_absent_from_named_body"] is True


def test_future_bodies_do_not_change_a_forecast(world):
    pre = world["leakage"]["prefix_store"]
    assert pre and all(world["results"][i] == v for i, v in pre.items())


# ------------------------------------------------------------------------------------------------- 3. restart
def test_restart_rebuild_gives_identical_forecasts_and_writes_nothing(world):
    assert world["restart"] == world["results"]
    assert world["out_files"] == []  # stateless: no ledger, no journal


def test_pins_profile_and_registration_refusals(world):
    r = world["refusals"]
    assert "pinned file changed or missing" in r["source_pin"]
    assert "profile_sha256" in r["profile_sha"]
    assert "strictly_prior_to_official_date" in r["not_strictly_prior"]
    assert "read_scope" in r["synthetic_profile_as_actual"]
    assert world["lightgbm_loaded"] is False


# ------------------------------------------------------------------------------------------------- 4. timing
def test_timing_and_rss(world):
    ms = sorted(world["ms_each"])
    rss = world["rss"]
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    print(f"\nN1 TIMING n={len(ms)} p50_ms={ms[len(ms) // 2]:.2f} max_ms={ms[-1]:.2f} build_ms={world['build_ms']:.0f} "
          f"peak_rss_mb={rss_mb:.0f} (child peak after build+predict, before the reference run; synthetic profile)")
    assert ms[-1] < 2000


def _child(work, out):
    sys.dont_write_bytecode = True
    work = Path(work)
    work.mkdir(parents=True)
    Path(out).write_text(json.dumps(World(work).report()))


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--child":
    _child(sys.argv[2], sys.argv[3])
