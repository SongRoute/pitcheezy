"""T_FINAL adapter contracts on synthetic GUMBO bodies only.

No real 2026 rows, no store from ws-ops/rehearsal runs, no network, no fitting.
The pinned model/encoder/calibrator/profile bytes are loaded (allowed); tests
that need them are skipped when they are absent.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import resource
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import pytest

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import STORE_DDL, FakeClock  # noqa: E402
from ws_prospective import LEAVES  # noqa: E402
from ws_prospective import adapter_t, store_source  # noqa: E402
from ws_prospective.context import build_context  # noqa: E402
from ws_prospective.ledger import Ledger, read_rows  # noqa: E402
from ws_prospective.runner import Abstain, GameRunner, Limits  # noqa: E402
from ws_prospective.store_source import LiveStoreSource, parse_feed  # noqa: E402

REG_V1 = Path("/Users/song/Projects/pitcheezy-pm-jobs/20261018-rs01-prospective-shadow/REGISTRATION.json")
SOURCE_ROOT = Path("/Users/song/Projects/pitcheezy-worktrees/codex-ws-postseason-six-1017")
ADAPTER_CONFIG = Path("/Users/song/Projects/pitcheezy-pm-jobs/20261016-fullseason-shadow-004/parity/ADAPTER-CONFIG.json")
ADAPTER_CONFIG_SHA = "3abaa2c39420d7f7d2dd7e865406070826487bcaec9018b1958eee46fea02a29"
PROFILE_DIR = Path("/Users/song/Projects/pitcheezy-pm-jobs/20261016-autonomous-profile-stream-005/profiles/attempt-001/bundle")
PROFILE_SHA = "3455dd1af73944c407f1fea88af807f001809dc28fbb4362d37c3b8c4d3d11cb"
TYPES16 = ["FF", "SI", "FC", "SL", "ST", "SV", "CU", "KC", "CS", "CH", "FS", "FO", "SC", "KN", "EP", "FA"]

GAME, DATE = 999101, "2026-10-24"
T0 = 1_800_000_000.0

HAVE_ARTIFACTS = (SOURCE_ROOT.is_dir() and ADAPTER_CONFIG.is_file()
                  and (PROFILE_DIR / "manifest.json").is_file())
heavy = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="pinned T artifacts/source absent")


# ------------------------------------------------------------ synthetic GUMBO
class SyntheticGame:
    """Builds live GUMBO bodies (full game prefix) pitch by pitch.

    Strikeouts and walks only (no balls in play), so every observed pitch is a
    source-native six-field token; same pitcher throughout (top 1st)."""

    PITCHER = 10

    def __init__(self):
        self.plays = []
        self.outs = 0
        self.offense = {}
        self.uid = 0

    def _new_play(self, batter):
        self.plays.append({
            "about": {"atBatIndex": len(self.plays), "isComplete": False, "inning": 1,
                      "isTopInning": True},
            "count": {"balls": 0, "strikes": 0, "outs": self.outs},
            "matchup": {"pitcher": {"id": self.PITCHER}, "batter": {"id": batter},
                        "batSide": {"code": "L" if batter % 2 else "R"},
                        "pitchHand": {"code": "R"}},
            "result": {"homeScore": 0, "awayScore": 0}, "playEvents": []})

    def start_pa(self, batter):
        self._new_play(batter)

    def pitch(self, code, call):
        play = self.plays[-1]
        c = play["count"]
        balls, strikes = c["balls"], c["strikes"]
        if call == "B":
            balls += 1
        elif call in ("C", "S"):
            strikes += 1
        elif call == "F":
            strikes = min(2, strikes + 1)
        is_out = strikes == 3
        if is_out:
            self.outs += 1
        self.uid += 1
        n = len(play["playEvents"]) + 1
        play["playEvents"].append({
            "isPitch": True, "pitchNumber": n, "playId": f"syn-{self.uid:04d}", "index": n - 1,
            "details": {"isInPlay": False, "isOut": is_out, "call": {"code": call},
                        "type": {"code": code}},
            "count": {"balls": balls, "strikes": strikes, "outs": self.outs}})
        play["count"] = {"balls": balls, "strikes": strikes, "outs": self.outs}
        if is_out or balls == 4:
            play["about"]["isComplete"] = True
            if balls == 4:  # forced advance
                b = play["matchup"]["batter"]["id"]
                if "first" in self.offense:
                    if "second" in self.offense:
                        self.offense["third"] = self.offense["second"]
                    self.offense["second"] = self.offense["first"]
                self.offense["first"] = {"id": b}

    def feed(self):
        cur = self.plays[-1]
        return copy.deepcopy({
            "gameData": {"game": {"pk": GAME}, "datetime": {"officialDate": DATE},
                         "status": {"abstractGameState": "Live"}},
            "liveData": {"plays": {"allPlays": self.plays, "currentPlay": cur},
                         "linescore": {"currentInning": 1, "isTopInning": True, "outs": self.outs,
                                       "teams": {"home": {"runs": 0}, "away": {"runs": 0}},
                                       "offense": dict(self.offense)}}})


SCRIPT = [  # (batter, [(type, call), ...]) - K, BB, K, partial PA
    (21, [("FF", "B"), ("SL", "C"), ("FF", "S"), ("CH", "F"), ("SL", "S")]),
    (22, [("SI", "B"), ("SI", "B"), ("FF", "C"), ("CH", "B"), ("SI", "B")]),
    (23, [("FC", "C"), ("CU", "B"), ("FF", "F"), ("CU", "S")]),
    (24, [("FF", "B"), ("SL", "C"), ("ST", "F")]),
]


def synthetic_bodies():
    """List of feeds in publication order: PA start, then one body per pitch."""
    g, out = SyntheticGame(), []
    for batter, pitches in SCRIPT:
        g.start_pa(batter)
        out.append(g.feed())
        for code, call in pitches:
            g.pitch(code, call)
            out.append(g.feed())
    return out


def canonical(feed) -> bytes:  # live_store._canonical
    return json.dumps(feed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


class ShaStore:
    """Fake observer store (fakes.STORE_DDL) with real canonical gzip bodies and sha."""

    def __init__(self, path: Path):
        self.path = Path(path)
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(STORE_DDL)
        con.close()
        self.seen = set()

    def publish(self, feed, fetched: float) -> int:
        raw = canonical(feed)
        sha = hashlib.sha256(raw).hexdigest()
        with sqlite3.connect(self.path, isolation_level=None) as con:
            cur = con.execute(
                "INSERT INTO snapshots (game_pk,event_wall,version,sha,first_fetched,fetched,source,"
                "body) VALUES (?,?,?,?,?,?,?,?)",
                (GAME, fetched, 1, sha, fetched, fetched, "live_poll", gzip.compress(raw, compresslevel=5)))
            seq = cur.lastrowid
            for play in feed["liveData"]["plays"]["allPlays"]:
                ab = play["about"]["atBatIndex"] + 1
                for ev in play["playEvents"]:
                    if ev.get("isPitch") and ev["playId"] not in self.seen:
                        self.seen.add(ev["playId"])
                        con.execute(
                            "INSERT INTO pitch_events (game_pk,provider_event_id,at_bat_number,"
                            "pitch_number,outcome,revision,event_wall,sha,first_received,received,"
                            "receipt_clock,store_seq,body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (GAME, ev["playId"], ab, ev["pitchNumber"], "new", 1, fetched, "h",
                             fetched, fetched, "observer_local", seq,
                             gzip.compress(json.dumps(ev, sort_keys=True).encode())))
        return seq


def model_source():
    return {"source_root": str(SOURCE_ROOT), "source_commit": "7a64583",
            "source_pins": dict(adapter_t.PINNED_SOURCE_PINS_7A64583),
            "normalizer_sha256": adapter_t.ACCEPTED_NORMALIZER_SHA256,
            "adapter_config_path": str(ADAPTER_CONFIG), "adapter_config_sha256": ADAPTER_CONFIG_SHA,
            "profile_dir": str(PROFILE_DIR), "profile_manifest_sha256": PROFILE_SHA}


def make_reg(types=TYPES16):
    return {"model_sources": {adapter_t.MODEL_ID: model_source()}, "candidate_pitch_types": list(types)}


def game_entry(store_path):
    return {"game_pk": GAME, "official_date": DATE, "store_path": str(store_path),
            "profile_path": "unused-by-T", "profile_sha256": "p" * 64}


class Recording:
    """Wraps the adapter as a runner forecaster and records ctx -> output."""

    def __init__(self, inner):
        self.inner, self.model_id, self.leaf_order = inner, inner.model_id, inner.leaf_order
        self.calls = []

    def predict(self, ctx):
        t0 = time.perf_counter()
        try:
            out = self.inner.predict(ctx)
        except Abstain as e:
            self.calls.append((ctx, None, str(e), time.perf_counter() - t0))
            raise
        self.calls.append((ctx, out, None, time.perf_counter() - t0))
        return out


def drive(tmp_path, bodies, adapter_wrapper, store, *, start_fetched=T0):
    clock = FakeClock()
    led = Ledger(tmp_path / f"game_{GAME}" / "ledger-A.sqlite", game_pk=GAME,
                 registration_sha256="r" * 64, code_commit="test", clock=clock)
    runner = GameRunner(GAME, LiveStoreSource(store.path, GAME), [adapter_wrapper], led, "p" * 64,
                        Limits(max_rss_mb=4096))
    for i, feed in enumerate(bodies):
        store.publish(feed, start_fetched + 7.0 * i)
        runner.step()
        clock.advance(7)
    led.close()
    return led


def accepted_chain_ff(state_path, forwarded):
    """Independent re-statement of fullseason_capture_native_replay.run (FF only)."""
    from pitcheezy.seq_shadow import captured_bridge, captured_native, captured_native_state
    from pitcheezy.seq_shadow.fullseason_runtime import Runtime
    config = json.loads(ADAPTER_CONFIG.read_text())
    runtime = Runtime(config, str(PROFILE_DIR), PROFILE_SHA)
    scorer = captured_native_state.NativeRuntime(runtime, allow_native_experimental=True)
    state = captured_native_state.NativeShadow(state_path, game_pk=GAME, game_date=DATE,
                                               normalizer_sha256=adapter_t.ACCEPTED_NORMALIZER_SHA256)
    out = []
    try:
        for seq, (body, fetched) in enumerate(forwarded, 1):
            sha = hashlib.sha256(body).hexdigest()
            feed = json.loads(body)
            received_ns = int(fetched * 1_000_000_000)
            step = captured_bridge.step(feed, game_pk=GAME, game_date=DATE, receipt_seq=seq,
                                        source_sha256=sha, candidate_pitch_type="FF")
            natives = {tuple(r["key"]): r for r in captured_native.inspect_tokens(
                feed, game_pk=GAME, game_date=DATE, receipt_seq=seq, source_sha256=sha)}
            for event in step["events"]:
                state.ingest(event, natives.get(tuple(event["key"])), received_ns=received_ns,
                             source_sha256=sha)
            state.observe_receipt(seq, sha, received_ns)
            cand = step["candidate"]
            if cand is None:
                out.append(("bridge_" + step["abstain_reason"], None))
                continue
            decision = state.decide(cand, received_ns=received_ns)
            result = scorer.predict(decision)
            out.append((result["status"], result["probabilities"]))
    finally:
        state.close()
    return out


# ---------------------------------------------------------------- light tests
def test_registration_keys_documented():
    assert set(adapter_t.REQUIRED_KEYS) <= set(adapter_t.REGISTRATION_KEYS)
    assert set(adapter_t.NATIVE_SOURCE_FILES) <= set(adapter_t.PINNED_SOURCE_PINS_7A64583)


def test_synthetic_bodies_are_valid_bridge_inputs():
    bodies = synthetic_bodies()
    n_state = 0
    for feed in bodies:
        pitches, state, final = parse_feed(feed, {})
        n_state += state is not None
        if state is not None:  # target never inside the body
            assert all((p.at_bat_index, p.pitch_number) != (state.at_bat_index, state.pitch_number)
                       for p in pitches)
    assert n_state == len(bodies) - 3  # 3 PA-ending bodies show no next state


@heavy
def test_normalizer_and_pins_match_pinned_root():
    from ws_prospective.adapters import verify_pins
    verify_pins(SOURCE_ROOT, adapter_t.PINNED_SOURCE_PINS_7A64583)
    cfg = json.loads(ADAPTER_CONFIG.read_text())
    for rel, d in cfg["source_pins"].items():
        assert adapter_t.PINNED_SOURCE_PINS_7A64583[rel] == d


# ---------------------------------------------------------------- heavy tests
@pytest.fixture(scope="module")
def run_once(tmp_path_factory):
    """One adapter run over the synthetic game through the real runner."""
    if not HAVE_ARTIFACTS:
        pytest.skip("pinned T artifacts/source absent")
    tmp = tmp_path_factory.mktemp("adapter_t")
    store = ShaStore(tmp / "store.sqlite3")
    reads = []
    orig = store_source.read_snapshot_raw

    def spy(path, seq):
        snap = orig(path, seq)
        reads.append(snap)
        return snap
    store_source.read_snapshot_raw = spy
    try:
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t_build = time.perf_counter()
        adapter = adapter_t.TFinalAdapter.build(make_reg(), game_entry(store.path), tmp / "out")
        t_build = time.perf_counter() - t_build
        rec = Recording(adapter)
        bodies = synthetic_bodies()
        drive(tmp, bodies, rec, store)
    finally:
        store_source.read_snapshot_raw = orig
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"tmp": tmp, "store": store, "adapter": adapter, "rec": rec, "reads": reads,
            "bodies": bodies, "build_s": t_build, "rss": (rss0, rss1)}


@heavy
def test_parity_with_accepted_chain_ff(run_once, tmp_path):
    rec, reads = run_once["rec"], run_once["reads"]
    assert len(reads) == len(rec.calls) > 0
    forwarded = [(r["body_bytes"], r["fetched"]) for r in reads]
    direct = accepted_chain_ff(tmp_path / "direct.sqlite3", forwarded)
    scored = abstained = 0
    max_diff = 0.0
    for (ctx, out, reason, _), (status, p) in zip(rec.calls, direct):
        if status == "experimental_conditional_score":
            assert out is not None and "FF" in out
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(out["FF"], p)))
            scored += 1
        else:
            abstained += 1
            assert out is None or "FF" not in out
    print(f"\nPARITY scored={scored} abstained={abstained} max_abs_diff={max_diff:.3e}")
    assert scored >= 1
    assert max_diff <= 1e-12
    run_once["parity"] = (scored, abstained, max_diff)


@heavy
def test_all_candidate_types_valid_distributions(run_once):
    adapter, rec = run_once["adapter"], run_once["rec"]
    assert adapter.types == TYPES16 and adapter.skipped_types == []
    full = [out for _, out, _, _ in rec.calls if out is not None]
    assert full, "no scored decision"
    distinct = set()
    for out in full:
        assert set(out) == set(TYPES16)
        for t, p in out.items():
            assert len(p) == len(LEAVES) and min(p) >= 0 and abs(sum(p) - 1) < 1e-9
            distinct.add(tuple(round(x, 12) for x in p))
    assert len(distinct) > len(full)  # candidate type changes the forecast
    reasons = [r for _, out, r, _ in rec.calls if out is None]
    print(f"\nTYPES scored_calls={len(full)} abstain_calls={len(reasons)} reasons={sorted(set(reasons))}")


@heavy
def test_no_leakage_target_absent_from_read_body(run_once):
    rec, reads = run_once["rec"], run_once["reads"]
    for (ctx, _, _, _), snap in zip(rec.calls, reads):
        assert snap["seq"] == ctx.source_seq and snap["sha256"] == ctx.source_sha
        feed = json.loads(snap["body_bytes"])
        keys = {(p["about"]["atBatIndex"] + 1, e["pitchNumber"])
                for p in feed["liveData"]["plays"]["allPlays"] for e in p["playEvents"] if e.get("isPitch")}
        assert (ctx.state.at_bat_index, ctx.state.pitch_number) not in keys
        assert len(keys) == ctx.slot_ordinal - 1


@heavy
def test_abstain_without_source_and_sha_mismatch(run_once):
    adapter, rec = run_once["adapter"], run_once["rec"]
    ctx = rec.calls[-1][0]
    with pytest.raises(Abstain, match="NO_SOURCE_BODY"):
        adapter.predict(build_context(GAME, [], 1, ctx.state, ctx.profile_sha256))
    bad = ctx.__class__(**{**ctx.__dict__, "source_sha": "0" * 64})
    with pytest.raises(ValueError, match="source sha"):
        adapter.predict(bad)


@heavy
def test_restart_same_probabilities_and_ordinal_continues(run_once):
    tmp, store, rec = run_once["tmp"], run_once["store"], run_once["rec"]
    adapter = run_once["adapter"]
    last_ctx, last_out = next((c, o) for c, o, _, _ in reversed(rec.calls) if o is not None)
    before = adapter_t.state_receipts(adapter.state_path)
    adapter.close()
    again = adapter_t.TFinalAdapter.build(make_reg(), game_entry(store.path), tmp / "out")
    try:
        same = again.predict(last_ctx)
        assert same == last_out
        mid = adapter_t.state_receipts(again.state_path)  # repeat body -> next ordinal, same sha
        assert mid[:-1] == before and mid[-1][0] == before[-1][0] + 1
        assert mid[-1][1:] == before[-1][1:]
        before = mid
        # one more pitch + forwarded body -> ordinal continues gaplessly
        g = SyntheticGame()
        for batter, pitches in SCRIPT:
            g.start_pa(batter)
            for code, call in pitches:
                g.pitch(code, call)
        g.pitch("FF", "B")
        feed = g.feed()
        seq = store.publish(feed, T0 + 7.0 * (len(run_once["bodies"]) + 1))
        pitches, state, _ = parse_feed(feed, {})
        con = sqlite3.connect(store.path)
        sha = con.execute("SELECT sha FROM snapshots WHERE seq=?", (seq,)).fetchone()[0]
        con.close()
        accepted = sorted(pitches, key=lambda p: p.ordinal)
        ctx = build_context(GAME, accepted, len(accepted) + 1, state, "p" * 64,
                            source_seq=seq, source_sha=sha)
        out = again.predict(ctx)
        assert set(out) == set(TYPES16)
        after = adapter_t.state_receipts(again.state_path)
        assert [r[0] for r in after] == list(range(1, len(before) + 2))
        assert after[-1][1] == sha
    finally:
        again.close()


@heavy
def test_unknown_prior_token_abstains_with_native_reason(tmp_path):
    """A prior pitch without a type code is not a native token -> later decisions abstain."""
    g, bodies = SyntheticGame(), []
    g.start_pa(21)
    g.pitch("FF", "B")
    del g.plays[-1]["playEvents"][0]["details"]["type"]
    bodies.append(g.feed())
    g.pitch("SL", "C")
    bodies.append(g.feed())
    store = ShaStore(tmp_path / "store.sqlite3")
    adapter = adapter_t.TFinalAdapter.build(make_reg(["FF", "SL", "PO", "XX"]), game_entry(store.path),
                                            tmp_path / "out")
    try:
        assert adapter.types == ["FF", "SL", "PO"] and adapter.skipped_types == ["XX"]
        rec = Recording(adapter)
        drive(tmp_path, bodies, rec, store)
        assert [r for _, _, r, _ in rec.calls] == ["native_prior_quality_unknown"] * 2
        rows = read_rows(tmp_path / f"game_{GAME}" / "ledger-A.sqlite", "forecasts")
        assert {r["status"] for r in rows} == {"ABSTAIN"}
    finally:
        adapter.close()


@heavy
def test_timing_and_rss(run_once):
    rec = run_once["rec"]
    scored_ms = sorted(dt * 1000 for _, o, _, dt in rec.calls if o is not None)
    all_ms = sorted(dt * 1000 for *_, dt in rec.calls)
    rss0, rss1 = run_once["rss"]
    div = 1 if sys.platform == "darwin" else 1024
    report = {"predict_calls": len(all_ms), "scored_calls": len(scored_ms),
              "scored_p50_ms": statistics.median(scored_ms), "scored_max_ms": max(scored_ms),
              "all_max_ms": max(all_ms), "build_s": run_once["build_s"],
              "peak_rss_mb_before_build": rss0 / div / 1048576,
              "peak_rss_mb_after_run": rss1 / div / 1048576}
    print("\nTIMING " + json.dumps(report))
    assert max(all_ms) < 2000
