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


def _played_game(tmp_path, reg_path, store_name="store.sqlite3"):
    """Run a 4-pitch synthetic game whose ledger pins reg_path; back up the store."""
    clock = FakeClock()
    t0 = clock.wall_ns() / S
    store = FakeStore(tmp_path / store_name, GAME)
    led = Ledger(tmp_path / "game_999001" / "ledger-A.sqlite", game_pk=GAME,
                 registration_sha256=file_sha256(reg_path), code_commit="c", clock=clock,
                 meta={"profile_sha256": "p" * 64, "store_path": str(store.path)})
    r = GameRunner(GAME, LiveStoreSource(store.path, GAME),
                   [SpyForecaster("m"), SpyForecaster("base", conditional=False)], led, "p" * 64)
    store.new_plate_appearance(1, received=t0)
    for i, call in enumerate("BSFC"):
        r.step()
        clock.advance(20)
        store.add_pitch(1, pitch_event(i + 1, "FF", call), received=clock.wall_ns() / S, uid=f"u{i}")
    led.close()
    copy = tmp_path / "store-backup.sqlite3"
    from ws_prospective.store_source import backup_store
    digest = backup_store(store.path, copy)  # sqlite online backup (a plain file copy misses the WAL)
    with open(tmp_path / "STORE-BACKUPS.jsonl", "a") as f:
        f.write(json.dumps({"copy": str(copy), "sha256": digest, "games": [GAME]}) + "\n")
    return {GAME: {"ledgers": [led.path], "store": copy, "store_sha256": digest}}


def test_primary_selection_falls_back_only_on_pair_floor(tmp_path):
    reg = _exec_registration(tmp_path)
    body = json.loads(reg.read_text())
    body["analysis"]["min_paired_pitches"] = 5  # synthetic game has 4 pairs -> primary below floor
    body["analysis"]["comparisons"] = [{"candidate": "m", "comparator": "base"},
                                       {"candidate": "base", "comparator": "m"}]
    body["analysis"]["primary"] = {"candidate": "m", "comparator": "base", "metric": "nll"}
    body["analysis"]["fallback_primary"] = {"candidate": "base", "comparator": "m", "metric": "nll"}
    reg.write_text(json.dumps(body))
    games = _played_game(tmp_path, reg)
    _unseal(tmp_path, reg)
    out = analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                               label_loader=final_label_rows, now_utc=AFTER)
    sel = out["revisions"]["final"]["primary_selection"]
    assert sel["selected"] == "none" and sel["n_primary"] == 4 and sel["n_fallback"] == 4


def _exec_registration(tmp_path, **kw):
    reg = _registration(tmp_path, **kw)
    body = json.loads(reg.read_text())
    body["execution"] = {"games": [{"game_pk": GAME, "profile_sha256": "p" * 64,
                                    "store_path": str(tmp_path / "store.sqlite3")}]}
    reg.write_text(json.dumps(body))
    return reg


def test_single_look_then_refuses_second(tmp_path):
    reg = _exec_registration(tmp_path)
    games = _played_game(tmp_path, reg)
    _unseal(tmp_path, reg)
    out = analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                               label_loader=final_label_rows, now_utc=AFTER)
    comp = out["revisions"]["final"]["comparisons"][0]
    assert comp["n_paired"] == 4 and set(comp["metrics"]) == {"nll", "brier", "ece", "top1"}
    assert out["revisions"]["final"]["primary_selection"]["selected"] == "none"  # no primary declared
    with pytest.raises(SealedError, match="already taken"):
        analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                             label_loader=final_label_rows, now_utc=AFTER)


@pytest.mark.parametrize("case", ["registration_edited", "store_copy_changed", "profile_differs",
                                  "store_path_differs", "backup_not_recorded"])
def test_look_refuses_when_ledger_pins_disagree(tmp_path, case):
    reg = _exec_registration(tmp_path)
    games = _played_game(tmp_path, reg)
    body = json.loads(reg.read_text())
    if case == "registration_edited":  # e.g. analyst swaps the comparison after the Series
        body["analysis"]["min_paired_pitches"] = 2
    elif case == "profile_differs":
        body["execution"]["games"][0]["profile_sha256"] = "q" * 64
    elif case == "store_path_differs":
        body["execution"]["games"][0]["store_path"] = "/elsewhere/store.sqlite3"
    reg.write_text(json.dumps(body))
    if case == "store_copy_changed":
        games[GAME]["store_sha256"] = "0" * 64
    if case == "backup_not_recorded":
        (tmp_path / "STORE-BACKUPS.jsonl").write_text("")
    _unseal(tmp_path, reg)  # UNSEAL quotes the (possibly edited) file: must still refuse
    called = []
    with pytest.raises(SealedError):
        analyzer.sealed_look(reg, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                             label_loader=lambda *a, **k: called.append(a), now_utc=AFTER)
    assert not called and not (tmp_path / "LOOK.done").exists()


@pytest.mark.parametrize("later", [True, False])
def test_addendum_may_only_move_seal_later(tmp_path, later):
    reg = _registration(tmp_path)
    add = tmp_path / "ADDENDUM-001.json"
    add.write_text(json.dumps({"not_before_utc": "2026-11-10T00:00:00Z" if later
                               else "2026-11-01T00:00:00Z"}))
    (tmp_path / "UNSEAL.json").write_text(json.dumps(
        {"registration_sha256": file_sha256(reg), "analyst": "fable_analyst",
         "addenda": [{"file": add.name, "sha256": file_sha256(add)}]}))
    with pytest.raises(SealedError):
        check_unsealed(reg, tmp_path, now_utc=AFTER)  # later: still sealed on 11-06; earlier: refused
    if later:
        assert check_unsealed(reg, tmp_path, now_utc=dt.datetime(2026, 11, 11, tzinfo=dt.timezone.utc))


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
    gb = {"trajectory": "ground_ball"}
    assert L.leaf_of(ev("X", hitData=gb, _play_event_type="field_out")) == "ground_ball_in_play"
    assert L.leaf_of(ev("X", hitData={"trajectory": "fly_ball"},
                        _play_event_type="home_run")) == "other_in_play"
    assert L.leaf_of(ev("X", hitData=gb, _play_event_type="grounded_into_double_play")) == \
        "ground_ball_in_play"
    assert L.leaf_of(ev("X", _play_event_type="single")) is None  # no trajectory
    for held in ("field_error", "fielders_choice", "catcher_interf", "triple_play",
                 "batter_interference", "some_new_event", None):
        assert L.leaf_of(ev("X", hitData=gb, _play_event_type=held)) is None, held
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


def test_fit_marginal_adapter_contract():
    from ws_prospective.adapters import FrozenFitMarginal, resolve
    a = resolve("FIT_MARGINAL_7").build({}, {}, None)
    assert isinstance(a, FrozenFitMarginal)
    assert sum(a.predict(None)["*"]) == pytest.approx(1.0)


def test_cli_backup_store_records_sha(tmp_path):
    from ws_prospective import cli
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    store.new_plate_appearance(1, received=1.0)
    store.add_pitch(1, pitch_event(1, "FF", "B"), received=2.0, uid="b1")
    rec = tmp_path / "STORE-BACKUPS.jsonl"
    assert cli.main(["backup-store", "--store", str(store.path), "--dest",
                     str(tmp_path / "copy.sqlite3"), "--record", str(rec), "--games", str(GAME)]) == 0
    row = json.loads(rec.read_text().splitlines()[0])
    assert row["sha256"] == file_sha256(tmp_path / "copy.sqlite3")
    assert len(final_label_rows(tmp_path / "copy.sqlite3", GAME, revision="store_final")) == 1
    with pytest.raises(FileExistsError):
        cli.main(["backup-store", "--store", str(store.path), "--dest",
                  str(tmp_path / "copy.sqlite3"), "--record", str(rec)])


def _dry_plan(tmp_path, store_copy, *, approved=True):
    reg = tmp_path / "MODELREG.json"
    reg.write_text(json.dumps({"model_sources": {}, "candidate_pitch_types": ["FF"]}))
    plan = {"execute": approved, "review": {"status": "PASS" if approved else "PENDING"},
            "model_registration": {"path": str(reg), "sha256": file_sha256(reg)},
            "input": {"game_pk": GAME, "official_date": "2026-10-18",
                      "store_copy_sha256": file_sha256(store_copy)},
            "processes": {"A": {"python": sys.executable, "models": ["FIT_MARGINAL_7"]}},
            "bounds": {"nice": 0, "max_rss_mb": 4096, "max_compute_ms": 2000, "max_wall_s": 60,
                       "min_free_memory_mb": 0, "min_free_disk_gb": 0, "max_output_mb": 64}}
    p = tmp_path / "DRYRUN-REGISTRATION.json"
    p.write_text(json.dumps(plan))
    return p


def test_dryrun_replays_backup_copy_without_storing_probabilities(tmp_path, monkeypatch):
    from ws_prospective import cli
    from ws_prospective.store_source import backup_store
    monkeypatch.setattr(cli, "_in_protected_window", lambda now_utc=None: False)
    store = FakeStore(tmp_path / "store.sqlite3", GAME)
    store.new_plate_appearance(1, received=1.0)
    for i, call in enumerate("BSF"):
        store.add_pitch(1, pitch_event(i + 1, "FF", call), received=2.0 + i, uid=f"d{i}")
    copy = tmp_path / "copy.sqlite3"
    backup_store(store.path, copy)
    plan = _dry_plan(tmp_path, copy)
    out = tmp_path / "dry"
    assert cli.main(["dryrun", "--plan", str(plan), "--store", str(copy), "--out", str(out),
                     "--proc", "A"]) == 0
    summ = json.loads((out / "DRYRUN-SUMMARY-A.json").read_text())
    assert summ["kind"].startswith("DRYRUN_METADATA_ONLY") and summ["store_pitch_keys"] == 3
    fc = read_rows(out / f"game_{GAME}" / "ledger-A.sqlite", "forecasts")
    assert fc and all(r["probs_json"] is None for r in fc)  # vectors never persisted
    text = json.dumps(summ)
    assert "probs" not in text and "0.358" not in text
    # refusals: second run into an existing dir, unapproved plan, protected window
    assert cli.main(["dryrun", "--plan", str(plan), "--store", str(copy), "--out", str(out),
                     "--proc", "A"]) == 2
    bad = _dry_plan(tmp_path, copy, approved=False)
    assert cli.main(["dryrun", "--plan", str(bad), "--store", str(copy), "--out",
                     str(tmp_path / "dry2"), "--proc", "A"]) == 2
    monkeypatch.setattr(cli, "_in_protected_window", lambda now_utc=None: True)
    assert cli.main(["dryrun", "--plan", str(plan), "--store", str(copy), "--out",
                     str(tmp_path / "dry3"), "--proc", "A"]) == 2


def test_protected_window_bounds():
    from ws_prospective.cli import _in_protected_window
    utc = dt.timezone.utc
    assert _in_protected_window(dt.datetime(2026, 10, 18, 21, 0, tzinfo=utc))   # 06:00 KST
    assert not _in_protected_window(dt.datetime(2026, 10, 19, 1, 0, tzinfo=utc))  # 10:00 KST
    assert not _in_protected_window(dt.datetime(2026, 10, 18, 11, 0, tzinfo=utc))


# ------------------------------------------------- per-game EXEC files (REVIEW-EXEC-001 B-2 1a)
PKS7 = [849805, 849802, 849804, 849803, 849799, 849801, 849800]
PH = "<PLACEHOLDER: CL-04>"


def _row(pk, tmp_path, filled):
    if not filled:
        return {"game_pk": pk, "official_date": PH, "first_pitch_utc": PH, "store_path": PH,
                "profile_path": PH, "profile_sha256": PH}
    return {"game_pk": pk, "official_date": "2026-10-24", "first_pitch_utc": "2026-10-25T00:08:00Z",
            "store_path": str(tmp_path / f"store-{pk}.sqlite3"), "profile_path": f"/p/{pk}/PROFILE.json",
            "profile_sha256": f"{pk:064d}"}


def _exec_files(tmp_path, n=7, mutate=None):
    base = json.loads(_registration(tmp_path).read_text())
    base["parent"] = {"file": "REGISTRATION-EXEC-v3.draft.json", "sha256": "a" * 64}
    files = []
    for i in range(n):
        body = json.loads(json.dumps(base))
        body["execution"] = {"games": [_row(pk, tmp_path, j <= i) for j, pk in enumerate(PKS7)]}
        if mutate:
            mutate(i, body)
        p = tmp_path / f"REGISTRATION-EXEC-G{i + 1}.json"
        p.write_text(json.dumps(body, sort_keys=True))
        files.append(p)
    return files


def _play_pk(tmp_path, pk, exec_path):
    from ws_prospective.store_source import backup_store
    clock = FakeClock()
    store = FakeStore(tmp_path / f"store-{pk}.sqlite3", pk)
    led = Ledger(tmp_path / f"game_{pk}" / "ledger-A.sqlite", game_pk=pk,
                 registration_sha256=file_sha256(exec_path), code_commit="c", clock=clock,
                 meta={"profile_sha256": f"{pk:064d}", "store_path": str(store.path)})
    r = GameRunner(pk, LiveStoreSource(store.path, pk),
                   [SpyForecaster("m"), SpyForecaster("base", conditional=False)], led, "p" * 64)
    store.new_plate_appearance(1, received=clock.wall_ns() / S)
    for i, call in enumerate("BSFC"):
        r.step()
        clock.advance(20)
        store.add_pitch(1, pitch_event(i + 1, "FF", call), received=clock.wall_ns() / S,
                        uid=f"{pk}-{i}")
    led.close()
    copy = tmp_path / f"copy-{pk}.sqlite3"
    digest = backup_store(store.path, copy)
    with open(tmp_path / "STORE-BACKUPS.jsonl", "a") as f:
        f.write(json.dumps({"copy": str(copy), "sha256": digest, "games": [pk]}) + "\n")
    return {"ledgers": [led.path], "store": copy, "store_sha256": digest}


def _unseal_set(tmp_path, files, pks=PKS7):
    (tmp_path / "UNSEAL.json").write_text(json.dumps(
        {"analyst": "fable_analyst",
         "registrations": [{"file": f.name, "sha256": file_sha256(f), "game_pk": pk}
                           for f, pk in zip(files, pks)]}))


def test_per_game_exec_files_pool_at_the_single_look(tmp_path):
    files = _exec_files(tmp_path)
    games = {pk: _play_pk(tmp_path, pk, f) for pk, f in zip(PKS7, files)}
    _unseal_set(tmp_path, files)
    out = analyzer.sealed_look(files, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                               label_loader=final_label_rows, now_utc=AFTER)
    comp = out["revisions"]["final"]["comparisons"][0]
    assert comp["n_paired"] == 28 and comp["metrics"]["nll"]["n_games"] == 7
    assert out["exec_sha256_by_game"] == {str(pk): file_sha256(f) for pk, f in zip(PKS7, files)}
    assert out["registration_sha256"] == file_sha256(files[0])
    with pytest.raises(SealedError, match="already taken"):
        analyzer.sealed_look(files, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                             label_loader=final_label_rows, now_utc=AFTER)


@pytest.mark.parametrize("case", ["differs_outside_games", "other_parent", "ledger_pins_other_game_sha",
                                  "unseal_lists_other_shas", "unseal_swaps_games", "row_rewritten"])
def test_per_game_exec_refusals(tmp_path, case):
    def mutate(i, body):
        if case == "differs_outside_games" and i == 3:
            body["analysis"]["min_paired_pitches"] = 2
        if case == "other_parent" and i == 5:
            body["parent"]["sha256"] = "b" * 64
        if case == "row_rewritten" and i == 6:
            body["execution"]["games"][0]["profile_sha256"] = "c" * 64
    files = _exec_files(tmp_path, mutate=mutate)
    pins = list(files)
    if case == "ledger_pins_other_game_sha":
        pins[2], pins[3] = files[3], files[2]  # game 3's ledger launched with game 4's file
    games = {pk: _play_pk(tmp_path, pk, f) for pk, f in zip(PKS7, pins)}
    if case == "unseal_lists_other_shas":
        _unseal_set(tmp_path, files[:6], PKS7[:6])
    elif case == "unseal_swaps_games":
        _unseal_set(tmp_path, files, [PKS7[1], PKS7[0]] + PKS7[2:])
    else:
        _unseal_set(tmp_path, files)
    called = []
    with pytest.raises(SealedError):
        analyzer.sealed_look(files, tmp_path, games, leaf_of=leaf_of, type_of=type_of,
                             label_loader=lambda *a, **k: called.append(a), now_utc=AFTER)
    assert not called and not (tmp_path / "LOOK.done").exists()


def test_per_game_exec_still_sealed_before_date(tmp_path):
    files = _exec_files(tmp_path, n=2)
    _unseal_set(tmp_path, files, PKS7[:2])
    from ws_prospective.seal import check_unsealed_set
    with pytest.raises(SealedError, match="sealed until"):
        check_unsealed_set(files, tmp_path, now_utc=DURING)
