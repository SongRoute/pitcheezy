"""Synthetic stand-ins: a fake live store (same SQLite schema subset as the
observer store), a scripted in-memory source, fake forecasters and a fake
clock.  No real 2026 rows, no models, no network."""
from __future__ import annotations

import gzip
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ws_prospective.context import PreState, PrePitchContext, StorePitch
from ws_prospective.runner import Abstain, SourceSnapshot

LEAVES = ["ball", "foul", "hit_by_pitch", "narrow_whiff", "other_strike",
          "ground_ball_in_play", "other_in_play"]


class FakeClock:
    def __init__(self, start_ns: int = 1_800_000_000 * 10**9):
        self.t = start_ns

    def advance(self, s: float) -> None:
        self.t += int(s * 1e9)

    def wall_ns(self) -> int:
        return self.t

    def mono_ns(self) -> int:
        return self.t - 10**18 if self.t > 10**18 else self.t


def make_pitch(ordinal, ab, pn, ptype, first_ns, uid=None, clock="observer_local"):
    return StorePitch(pitch_uid=uid or f"p{ordinal}", ordinal=ordinal, at_bat_index=ab,
                      pitch_number=pn, pitcher_id=100, batter_id=200 + ab, pitch_type=ptype,
                      balls_before=0, strikes_before=0, outs_before=0, inning=1,
                      top_inning=True, first_receipt_wall_ns=first_ns,
                      accepted_wall_ns=first_ns, receipt_clock=clock)


def pre_state(ab, pn):
    return PreState(ab, pn, 100, 200 + ab, 0, 0, 0, 1, True)


@dataclass
class ScriptedSource:  # noqa: D101
    """Exposes pitches whose first receipt is <= the fake clock."""

    clock: FakeClock
    pitches: List[StorePitch]
    final_after_ns: Optional[int] = None
    polls: int = 0

    def poll(self) -> SourceSnapshot:
        self.polls += 1
        self._before_poll()
        now = self.clock.wall_ns()
        acc = [p for p in self.pitches if p.first_receipt_wall_ns <= now]
        final = self.final_after_ns is not None and now >= self.final_after_ns
        nxt = None
        remaining = [p for p in self.pitches if p.first_receipt_wall_ns > now]
        if remaining and not final:
            p = remaining[0]
            nxt = pre_state(p.at_bat_index, p.pitch_number)
        return SourceSnapshot(acc, nxt, final)

    def _before_poll(self) -> None:
        pass


class SpyForecaster:
    """Records every context it sees; fails loudly on forbidden content."""

    leaf_order = LEAVES

    def __init__(self, model_id="spy", conditional=True, abstain_on=()):
        self.model_id = model_id
        self.conditional = conditional
        self.abstain_on = set(abstain_on)
        self.seen: List[PrePitchContext] = []

    def predict(self, ctx: PrePitchContext) -> Dict[str, List[float]]:
        self.seen.append(ctx)
        if ctx.slot_ordinal in self.abstain_on:
            raise Abstain("UNKNOWN_PITCHER")
        k = len(ctx.history)
        base = [0.3, 0.2, 0.01, 0.1, 0.19, 0.1, 0.1]
        if self.conditional:
            return {"FF": base, "SL": base[::-1][:6] + [1 - sum(base[::-1][:6])]}
        return {"*": base}


class CrashForecaster(SpyForecaster):
    def predict(self, ctx):
        raise RuntimeError("restart recomputed score")


# ---- fake observer store (subset of live_store schema) --------------------
STORE_DDL = """
CREATE TABLE snapshots (seq INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL,
 event_wall REAL NOT NULL, version INTEGER NOT NULL, feed_ts TEXT, sha TEXT NOT NULL,
 first_fetched REAL NOT NULL, fetched REAL NOT NULL, source TEXT NOT NULL, body BLOB NOT NULL,
 requested REAL);
CREATE TABLE pitch_events (id INTEGER PRIMARY KEY AUTOINCREMENT, game_pk INTEGER NOT NULL,
 provider_event_id TEXT, at_bat_number INTEGER, pitch_number INTEGER, outcome TEXT NOT NULL,
 reason TEXT, revision INTEGER, provider_ts TEXT, provider_event_time TEXT,
 event_wall REAL NOT NULL, sha TEXT NOT NULL, first_received REAL, received REAL NOT NULL,
 receipt_clock TEXT NOT NULL, store_seq INTEGER NOT NULL, body BLOB NOT NULL);
"""


def _gz(obj) -> bytes:
    return gzip.compress(json.dumps(obj, sort_keys=True).encode())


def pitch_event(pn, code, call, desc="x"):
    return {"isPitch": True, "pitchNumber": pn, "playId": None,
            "details": {"type": {"code": code}, "call": {"code": call}, "description": desc},
            "count": {"balls": 0, "strikes": 0}}


class FakeStore:
    def __init__(self, path: Path, game_pk: int):
        self.path, self.game_pk = Path(path), game_pk
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(STORE_DDL)
        con.close()
        self.plays: List[dict] = []

    def _con(self):
        return sqlite3.connect(self.path, isolation_level=None)

    def add_pitch(self, ab: int, ev: dict, *, received: float, uid: str,
                  revision=1, outcome="new", first=None, clock="observer_local",
                  plate_done=False):
        while len(self.plays) < ab:
            self.plays.append({"about": {"atBatIndex": len(self.plays), "inning": 1,
                                         "isTopInning": True, "isComplete": False},
                               "matchup": {"pitcher": {"id": 100},
                                           "batter": {"id": 200 + len(self.plays) + 1}},
                               "playEvents": [], "count": {"balls": 0, "strikes": 0, "outs": 0}})
        ev = dict(ev, playId=uid)
        play = self.plays[ab - 1]
        if outcome == "new":
            play["playEvents"].append(ev)
        else:
            play["playEvents"] = [ev if e.get("playId") == uid else e for e in play["playEvents"]]
        play["about"]["isComplete"] = plate_done
        feed = {"gameData": {"status": {"abstractGameState": "Live"}},
                "liveData": {"plays": {"allPlays": self.plays, "currentPlay": self.plays[-1]},
                             "linescore": {"offense": {}}}}
        with self._con() as con:
            cur = con.execute("INSERT INTO snapshots (game_pk,event_wall,version,sha,first_fetched,"
                              "fetched,source,body) VALUES (?,?,?,?,?,?,?,?)",
                              (self.game_pk, received, 1, "s", received, received, "live_poll",
                               _gz(feed)))
            con.execute("INSERT INTO pitch_events (game_pk,provider_event_id,at_bat_number,"
                        "pitch_number,outcome,revision,event_wall,sha,first_received,received,"
                        "receipt_clock,store_seq,body) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (self.game_pk, uid, ab, ev["pitchNumber"], outcome, revision, received,
                         "h", first if first is not None else received, received, clock,
                         cur.lastrowid, _gz(ev)))

    def new_plate_appearance(self, ab: int, *, received: float):
        """Publish a snapshot whose currentPlay is a fresh PA with no pitches."""
        if self.plays:
            self.plays[-1]["about"]["isComplete"] = True
        self.plays.append({"about": {"atBatIndex": ab - 1, "inning": 1, "isTopInning": True,
                                     "isComplete": False},
                           "matchup": {"pitcher": {"id": 100}, "batter": {"id": 200 + ab}},
                           "playEvents": [], "count": {"balls": 0, "strikes": 0, "outs": 0}})
        feed = {"gameData": {"status": {"abstractGameState": "Live"}},
                "liveData": {"plays": {"allPlays": self.plays, "currentPlay": self.plays[-1]},
                             "linescore": {"offense": {}}}}
        with self._con() as con:
            con.execute("INSERT INTO snapshots (game_pk,event_wall,version,sha,first_fetched,"
                        "fetched,source,body) VALUES (?,?,?,?,?,?,?,?)",
                        (self.game_pk, received, 1, "s", received, received, "live_poll",
                         _gz(feed)))


def leaf_of(ev: dict):
    call = ev.get("details", {}).get("call", {}).get("code")
    return {"B": "ball", "F": "foul", "S": "narrow_whiff", "C": "other_strike",
            "X": "other_in_play", "H": "hit_by_pitch"}.get(call)


def type_of(ev: dict):
    return ev.get("details", {}).get("type", {}).get("code")
