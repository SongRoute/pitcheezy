"""Synthetic-only contracts for RS-01 (no real rows, no models, no network)."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fakes import (LEAVES, CrashForecaster, FakeClock, FakeStore, ScriptedSource,  # noqa: E402
                   SpyForecaster, leaf_of, make_pitch, pitch_event, type_of)
from ws_prospective import analyzer, metrics  # noqa: E402
from ws_prospective.context import PreState, build_context  # noqa: E402
from ws_prospective.ledger import Ledger, LedgerImmutableError, read_rows  # noqa: E402
from ws_prospective.runner import GameRunner, Limits  # noqa: E402
from ws_prospective.seal import SealedError, check_unsealed, file_sha256  # noqa: E402
from ws_prospective.store_source import LiveStoreSource, final_label_rows  # noqa: E402

GAME = 999001
S = 10**9


def _ledger(tmp, clock, reg_sha="r" * 64, **kw):
    return Ledger(tmp / "game_999001" / "ledger.sqlite", game_pk=GAME, registration_sha256=reg_sha,
                  code_commit="deadbeef", clock=clock, meta=kw.get("meta"))


def _script(clock, n=6):
    t0 = clock.wall_ns()
    pitches, ab, pn = [], 1, 0
    for i in range(1, n + 1):
        pn += 1
        if pn > 3:
            ab, pn = ab + 1, 1
        pitches.append(make_pitch(i, ab, pn, "FF" if i % 2 else "SL", t0 + i * 20 * S))
    return pitches


def _drive(runner, clock, until_s, step_s=5):
    for _ in range(int(until_s / step_s)):
        runner.step()
        clock.advance(step_s)


# ---------------------------------------------------------------- no leakage
def test_forecaster_never_sees_current_or_future_pitch(tmp_path):
    clock = FakeClock()
    pitches = _script(clock)
    spy = SpyForecaster()
    r = GameRunner(GAME, ScriptedSource(clock, pitches), [spy], _ledger(tmp_path, clock), "p" * 64)
    _drive(r, clock, 130)
    assert spy.seen, "runner produced no forecasts"
    for ctx in spy.seen:
        n = ctx.slot_ordinal
        assert len(ctx.history) == n - 1
        uids = {p.pitch_uid for p in pitches if p.ordinal >= n}
        assert not (uids & set(json.dumps(ctx.__dict__, default=str).split('"')))
        target = pitches[n - 1]
        assert (ctx.state.at_bat_index, ctx.state.pitch_number) == (target.at_bat_index,
                                                                   target.pitch_number)
        assert not hasattr(ctx.state, "pitch_type")


def test_build_context_truncates_even_if_store_already_has_the_pitch():
    clock = FakeClock()
    pitches = _script(clock)
    st = PreState(1, 3, 100, 201, 0, 0, 0, 1, True)
    a = build_context(GAME, pitches, 3, st, "p")
    changed = [p if p.ordinal != 3 else make_pitch(3, 1, 3, "CU", p.first_receipt_wall_ns)
               for p in pitches]
    b = build_context(GAME, changed, 3, st, "p")
    assert a.sha256() == b.sha256()  # mutating pitch n (or later) cannot change slot n's context


# ------------------------------------------------------- immutability/restart
def test_ledger_rejects_update_and_delete(tmp_path):
    clock = FakeClock()
    led = _ledger(tmp_path, clock)
    iid = led.write_intent(slot_key="game_start", slot_ordinal=1, target_at_bat_number=1,
                           target_pitch_number=1, model_id="m", context_asof_pitch_uid=None,
                           context_asof_receipt_wall_ns=None, context_sha256="c")
    led.write_forecast(iid, status="OK", leaf_order=LEAVES, probs={"*": [1 / 7] * 7},
                       abstain_reason=None, generated_wall_ns=1, generated_mono_ns=1,
                       compute_ns=1, provisional_timing="PRE_RECEIPT")
    for sql in ("UPDATE forecasts SET probs_json='x'", "DELETE FROM forecasts",
                "UPDATE intents SET model_id='z'", "DELETE FROM sessions",
                "UPDATE ledger_meta SET value='x'"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            led.con.execute(sql)
    assert led.con.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_restart_keeps_first_forecast_and_never_recomputes(tmp_path):
    clock = FakeClock()
    pitches = _script(clock)
    src = ScriptedSource(clock, pitches)
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, src, [SpyForecaster("m")], led, "p" * 64)
    _drive(r, clock, 70)
    path = led.path
    before = read_rows(path, "forecasts")
    led.close()
    # restart: same slots must not be scored again (a crash forecaster would show up as ERROR)
    led2 = _ledger(tmp_path, clock)
    r2 = GameRunner(GAME, src, [CrashForecaster("m")], led2, "p" * 64)
    r2.step()
    after = read_rows(path, "forecasts")
    assert after[:len(before)] == before
    assert len(read_rows(path, "sessions")) == 2
    new = after[len(before):]
    assert all(f["status"] == "ERROR" for f in new)  # only genuinely new slots were attempted
    assert len(new) <= 1


def test_restart_with_different_frozen_inputs_is_refused(tmp_path):
    clock = FakeClock()
    _ledger(tmp_path, clock, meta={"profile_sha256": "a"}).close()
    with pytest.raises(LedgerImmutableError):
        _ledger(tmp_path, clock, meta={"profile_sha256": "b"})
    with pytest.raises(LedgerImmutableError):
        _ledger(tmp_path, clock, reg_sha="x" * 64)


# --------------------------------------------------------- post-hoc exclusion
class _SlowSource(ScriptedSource):
    """Pitch n lands while the forecaster is computing (second poll sees it)."""

    def _before_poll(self):
        if self.polls % 2 == 0:  # the re-poll after compute
            self.clock.advance(30)


def test_pitch_arriving_during_compute_is_marked_post_hoc(tmp_path):
    clock = FakeClock()
    pitches = _script(clock, 3)
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, _SlowSource(clock, pitches), [SpyForecaster("m")], led, "p" * 64)
    r.step()
    f = read_rows(led.path, "forecasts")
    assert f and f[0]["provisional_timing"] == "POST_HOC_REPLAY"


def test_backfilled_slots_are_post_hoc_and_get_no_source_body(tmp_path):
    clock = FakeClock()
    pitches = _script(clock, 4)
    clock.advance(200)  # runner starts late: all 4 pitches already in the store
    spy = SpyForecaster("m")
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, ScriptedSource(clock, pitches), [spy], led, "p" * 64,
                   Limits(record_post_hoc_replays=True))
    r.step()
    rows = read_rows(led.path, "forecasts")
    assert len(rows) == 4 and {x["provisional_timing"] for x in rows} == {"POST_HOC_REPLAY"}
    assert all(c.source_seq is None for c in spy.seen)


# --------------------------------------------------------------- live store
def _store_game(tmp_path, t0):
    st = FakeStore(tmp_path / "store.sqlite3", GAME)
    return st


def test_live_store_source_end_to_end_and_grading(tmp_path):
    clock = FakeClock()
    t0 = clock.wall_ns() / S
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    src = LiveStoreSource(store.path, GAME)
    spy = SpyForecaster("m", conditional=True)
    base = SpyForecaster("base", conditional=False)
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, src, [spy, base], led, "p" * 64)
    store.new_plate_appearance(1, received=t0)
    plan = [(1, pitch_event(1, "FF", "B")), (1, pitch_event(2, "SL", "S")),
            (1, pitch_event(3, "FF", "X")), (2, pitch_event(1, "FF", "C"))]
    for i, (ab, ev) in enumerate(plan):
        r.step()  # forecast for the upcoming pitch, before it is stored
        clock.advance(20)
        if ab == 2 and i == 3:
            store.new_plate_appearance(2, received=clock.wall_ns() / S - 10)
            r.step()
            clock.advance(5)
        store.add_pitch(ab, ev, received=clock.wall_ns() / S, uid=f"u{i}",
                        plate_done=ev["details"]["call"]["code"] == "X")
    # a later correction of pitch u1's call must not change the stored forecast
    store.add_pitch(1, pitch_event(2, "SL", "F"), received=clock.wall_ns() / S + 100, uid="u1",
                    outcome="correction", revision=2, first=t0 + 40)
    fc = read_rows(led.path, "forecasts")
    assert {f["provisional_timing"] for f in fc} == {"PRE_RECEIPT"}
    labels = final_label_rows(store.path, GAME, revision="final")
    graded, excl = analyzer.grade_game(led.path, labels, leaf_of=leaf_of, type_of=type_of,
                                       leaf_order=LEAVES)
    assert excl[("m", "ELIGIBLE")] == 4 and excl[("base", "ELIGIBLE")] == 4
    first = final_label_rows(store.path, GAME, revision="first")
    assert leaf_of(first["u1"]["event"]) == "narrow_whiff"
    assert leaf_of(labels["u1"]["event"]) == "foul"


def test_reader_is_read_only(tmp_path):
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    from ws_prospective.store_source import open_ro
    con = open_ro(store.path)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO snapshots (game_pk,event_wall,version,sha,first_fetched,fetched,"
                    "source,body) VALUES (1,1,1,'s',1,1,'x',x'00')")
    con.close()


def test_generated_after_first_receipt_excluded_at_analysis(tmp_path):
    clock = FakeClock()
    led = _ledger(tmp_path, clock)
    iid = led.write_intent(slot_key="game_start", slot_ordinal=1, target_at_bat_number=1,
                           target_pitch_number=1, model_id="m", context_asof_pitch_uid=None,
                           context_asof_receipt_wall_ns=None, context_sha256="c")
    led.write_forecast(iid, status="OK", leaf_order=LEAVES, probs={"*": [1 / 7] * 7},
                       abstain_reason=None, generated_wall_ns=5 * S, generated_mono_ns=1,
                       compute_ns=1, provisional_timing="PRE_RECEIPT")
    labels = {"u0": {"at_bat_number": 1, "pitch_number": 1, "revision": 1,
                     "first_received_ns": 4 * S, "receipt_clock": "observer_local",
                     "event": pitch_event(1, "FF", "B")}}
    _, excl = analyzer.grade_game(led.path, labels, leaf_of=leaf_of, type_of=type_of,
                                  leaf_order=LEAVES)
    assert excl[("m", "GENERATED_AFTER_FIRST_RECEIPT")] == 1
    labels["u0"]["first_received_ns"] = 6 * S
    labels["u0"]["receipt_clock"] = "unverified"
    _, excl = analyzer.grade_game(led.path, labels, leaf_of=leaf_of, type_of=type_of,
                                  leaf_order=LEAVES)
    assert excl[("m", "RECEIPT_CLOCK_UNVERIFIED")] == 1


# --------------------------------------------------------------------- seal
def _registration(tmp_path, *, review="PASS", not_before="2026-11-05T00:00:00Z"):
    reg = {"review": {"status": review},
           "targets": {"leaf_order": LEAVES},
           "analysis": {"seal": {"not_before_utc": not_before, "analyst": "fable_analyst",
                                 "unseal_file": "UNSEAL.json", "look_marker": "LOOK.done"},
                        "label_revision_primary": "final", "label_revision_sensitivity": ["first"],
                        "bootstrap": {"seed": 20261024, "B": 200}, "min_paired_pitches": 1,
                        "comparisons": [{"candidate": "m", "comparator": "base"}]}}
    p = tmp_path / "REGISTRATION.json"
    p.write_text(json.dumps(reg))
    return p


def _unseal(tmp_path, reg_path, analyst="fable_analyst"):
    (tmp_path / "UNSEAL.json").write_text(json.dumps(
        {"registration_sha256": file_sha256(reg_path), "analyst": analyst}))


DURING = dt.datetime(2026, 10, 27, tzinfo=dt.timezone.utc)
AFTER = dt.datetime(2026, 11, 6, tzinfo=dt.timezone.utc)


def test_analyzer_refuses_during_series_even_with_unseal(tmp_path):
    reg = _registration(tmp_path)
    _unseal(tmp_path, reg)
    with pytest.raises(SealedError, match="sealed until"):
        check_unsealed(reg, tmp_path, now_utc=DURING)
    called = []
    with pytest.raises(SealedError):
        analyzer.sealed_look(reg, tmp_path, {}, leaf_of=leaf_of, type_of=type_of,
                             label_loader=lambda *a: called.append(a), now_utc=DURING)
    assert not called, "labels were read while sealed"


@pytest.mark.parametrize("case", ["no_review", "no_unseal", "wrong_analyst", "reg_changed"])
def test_analyzer_refuses_without_every_seal_condition(tmp_path, case):
    reg = _registration(tmp_path, review="PENDING" if case == "no_review" else "PASS")
    if case != "no_unseal":
        _unseal(tmp_path, reg, analyst="someone_else" if case == "wrong_analyst" else "fable_analyst")
    if case == "reg_changed":
        reg.write_text(reg.read_text().replace("200", "201"))
    with pytest.raises(SealedError):
        check_unsealed(reg, tmp_path, now_utc=AFTER)


def test_single_look_then_refuses_second(tmp_path):
    clock = FakeClock()
    t0 = clock.wall_ns() / S
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, LiveStoreSource(store.path, GAME),
                   [SpyForecaster("m"), SpyForecaster("base", conditional=False)], led, "p" * 64)
    store.new_plate_appearance(1, received=t0)
    for i, call in enumerate("BSFC"):
        r.step()
        clock.advance(20)
        store.add_pitch(1, pitch_event(i + 1, "FF", call), received=clock.wall_ns() / S, uid=f"u{i}")
    led.close()
    reg = _registration(tmp_path)
    _unseal(tmp_path, reg)
    games = {GAME: {"ledgers": [led.path], "store": store.path}}
    out = analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                               label_loader=final_label_rows, now_utc=AFTER)
    comp = out["revisions"]["final"]["comparisons"][0]
    assert comp["n_paired"] == 4 and set(comp["metrics"]) == {"nll", "brier", "ece", "top1"}
    with pytest.raises(SealedError, match="already taken"):
        analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                             label_loader=final_label_rows, now_utc=AFTER)


def test_ops_check_has_no_labels_or_metrics(tmp_path):
    clock = FakeClock()
    led = _ledger(tmp_path, clock)
    r = GameRunner(GAME, ScriptedSource(clock, _script(clock)), [SpyForecaster("m")], led, "p" * 64)
    _drive(r, clock, 60)
    rep = analyzer.ops_check(led.path)
    text = json.dumps(rep)
    assert rep["kind"].startswith("OPS_METADATA_ONLY")
    for banned in ("nll", "brier", "ece", "top1", "label", "pitch_type"):
        assert banned not in text.replace("NO_LABELS_NO_METRICS", "")


# ------------------------------------------------------------------ metrics
def test_metrics_known_values_and_bootstrap_determinism():
    P = np.array([[0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 0, 1])
    assert metrics.top1(P, y) == pytest.approx(1.0)  # argmax ties -> index 0
    assert metrics.nll(P, y) == pytest.approx(-(np.log(.5) + np.log(.9) + np.log(.8)) / 3)
    assert metrics.brier(P, y) == pytest.approx((0.5 + 0.02 + 0.08) / 3)
    U = np.full_like(P, 0.5)
    a = metrics.cluster_bootstrap_paired(P, U, y, [1, 1, 2], seed=7, B=50)
    b = metrics.cluster_bootstrap_paired(P, U, y, [1, 1, 2], seed=7, B=50)
    assert a == b and a["nll"]["n_games"] == 2


# ------------------------------------------------------------ labels / cli
def test_label_rule_matches_registered_mapping():
    from ws_prospective import labels as L
    ev = lambda code, **kw: dict({"details": {"call": {"code": code}}}, **kw)  # noqa: E731
    assert L.leaf_of(ev("B")) == "ball" and L.leaf_of(ev("S")) == "narrow_whiff"
    assert L.leaf_of(ev("C")) == "other_strike" and L.leaf_of(ev("F")) == "foul"
    assert L.leaf_of(ev("L", count={"strikes": 3})) == "other_strike"
    assert L.leaf_of(ev("L", count={"strikes": 2})) == "foul"
    assert L.leaf_of(ev("H")) == "hit_by_pitch"
    assert L.leaf_of(ev("X", hitData={"trajectory": "ground_ball"})) == "ground_ball_in_play"
    assert L.leaf_of(ev("X", hitData={"trajectory": "fly_ball"})) == "other_in_play"
    assert L.leaf_of(ev("X")) is None
    assert L.leaf_of(ev("X", hitData={"trajectory": "ground_ball"},
                        _play_event_type="field_error")) is None
    assert L.leaf_of(ev("VB")) is None and L.leaf_of(ev("??")) is None


def test_mismatched_state_intent_does_not_block_correct_target(tmp_path):
    clock = FakeClock()
    led = _ledger(tmp_path, clock)
    for tgt in ((1, 4), (2, 1)):  # stale state first, then the real next PA
        iid = led.write_intent(slot_key="after:u2", slot_ordinal=4, target_at_bat_number=tgt[0],
                               target_pitch_number=tgt[1], model_id="m",
                               context_asof_pitch_uid="u2", context_asof_receipt_wall_ns=1,
                               context_sha256="c")
        assert iid is not None
        led.write_forecast(iid, status="OK", leaf_order=LEAVES, probs={"*": [1 / 7] * 7},
                           abstain_reason=None, generated_wall_ns=1, generated_mono_ns=1,
                           compute_ns=1, provisional_timing="PRE_RECEIPT")
    labels = {u: {"at_bat_number": ab, "pitch_number": pn, "revision": 1,
                  "first_received_ns": 9 * S, "receipt_clock": "observer_local",
                  "event": pitch_event(pn, "FF", "B")}
              for u, ab, pn in (("u0", 1, 1), ("u1", 1, 2), ("u2", 1, 3), ("u3", 2, 1))}
    _, excl = analyzer.grade_game(led.path, labels, leaf_of=leaf_of, type_of=type_of,
                                  leaf_order=LEAVES)
    assert excl[("m", "ELIGIBLE")] == 1 and excl[("m", "NO_FORECAST")] == 3


def test_cli_run_refuses_non_executable_registration(tmp_path, capsys):
    from ws_prospective import cli
    reg = tmp_path / "R.json"
    reg.write_text(json.dumps({"execute": False, "review": {"status": "PENDING"}}))
    rc = cli.main(["run", "--registration", str(reg), "--game", "1", "--store",
                   str(tmp_path / "s"), "--out", str(tmp_path / "o"), "--proc", "A"])
    assert rc == 2 and "refused" in capsys.readouterr().out
    assert not (tmp_path / "o").exists()


def test_declared_model_adapters_cannot_run():
    from ws_prospective.adapters import REGISTRY
    for mid, cls in REGISTRY.items():
        if mid != "FIT_MARGINAL_7":
            with pytest.raises(NotImplementedError):
                cls()


def test_open_reader_transaction_does_not_block_writer(tmp_path):
    from ws_prospective.store_source import open_ro
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    store.new_plate_appearance(1, received=1.0)
    ro = open_ro(store.path)
    ro.execute("BEGIN")
    ro.execute("SELECT COUNT(*) FROM snapshots").fetchone()
    store.add_pitch(1, pitch_event(1, "FF", "B"), received=2.0, uid="w1")  # writer commits
    ro.execute("COMMIT")
    assert ro.execute("SELECT COUNT(*) FROM pitch_events").fetchone()[0] == 1
    ro.close()
