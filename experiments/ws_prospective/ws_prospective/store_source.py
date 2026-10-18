"""Read-only source over the live service store (``store.sqlite3``).

Why the store and not a new capture: the publisher already is the single MLB
poller for the date (one poll stream per output folder; a second poller would
double MLB traffic and break the "one poller per game" rule owned by CL-01 /
CL-03).  The store is SQLite in WAL mode, so a reader never blocks the writer
as long as it

* opens ``file:<path>?mode=ro`` with ``PRAGMA query_only=ON`` (no writes, no
  checkpoint, never ``immutable=1`` because the file is live),
* keeps every read short (autocommit, one connection per poll, closed at once),
* reads with monotone cursors (``snapshots.seq``, ``pitch_events.id``) because
  rows are append-only.

Per poll it reads the newest stored snapshot of the game (the publisher's own
canonical GUMBO body) and the accepted ``pitch_events`` receipts.  From the
snapshot it derives (a) the accepted pitches so far and (b) the pre-pitch
state of the next slot when the feed already shows it (current plate
appearance not complete).  The current pitch is never visible here because a
slot is only scored while the store has no event for it; once it appears the
runner can only produce POST_HOC_REPLAY rows.

Note: 'before first receipt' is information-prospective (the model has not
seen pitch n), not physical pre-pitch: the feed lags the field (precedent
median ~19 s).  Physical timing is reported descriptively only.
"""
from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .context import PreState, StorePitch
from .runner import SourceSnapshot

ACCEPTED = ("new", "correction")


def _ns(ts) -> Optional[int]:
    """Store timestamps are epoch seconds (REAL) -> ns; None stays None."""
    if ts is None:
        return None
    return int(round(float(ts) * 1e9))


def open_ro(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, isolation_level=None,
                          timeout=5)
    con.execute("PRAGMA query_only=ON")
    return con


def _receipts(con, game_pk: int) -> Dict[Tuple[int, int], tuple]:
    out: Dict[Tuple[int, int], tuple] = {}
    for pid, ab, pn, first, recv, clock, etime in con.execute(
            "SELECT provider_event_id, at_bat_number, pitch_number, first_received, received, "
            "receipt_clock, provider_event_time "
            "FROM pitch_events WHERE game_pk=? AND outcome IN (?,?) ORDER BY id",
            (game_pk, *ACCEPTED)):
        key = (int(ab), int(pn))
        if key not in out:  # first accepted revision carries the first receipt
            out[key] = (str(pid), _ns(first), _ns(recv), clock, etime)
    return out


def _latest_body(con, game_pk: int, with_fetched: bool = False):
    row = con.execute("SELECT seq, sha, body, fetched FROM snapshots WHERE game_pk=? "
                      "ORDER BY seq DESC LIMIT 1", (game_pk,)).fetchone()
    if row is None:
        return (None, None, None, None) if with_fetched else (None, None, None)
    out = (int(row[0]), str(row[1]), json.loads(gzip.decompress(row[2])))
    return out + (_ns(row[3]),) if with_fetched else out


def read_snapshot_raw(store_path: Path, seq: int) -> dict:
    """Exact stored snapshot for real adapters: inflated canonical bytes + receipt fields.

    Verifies sha256(inflated bytes) == snapshots.sha before returning."""
    import hashlib
    con = open_ro(store_path)
    try:
        row = con.execute("SELECT seq, game_pk, sha, first_fetched, fetched, source, body "
                          "FROM snapshots WHERE seq=?", (int(seq),)).fetchone()
    finally:
        con.close()
    if row is None:
        raise KeyError(f"snapshot seq {seq} not in store")
    raw = gzip.decompress(row[6])
    if hashlib.sha256(raw).hexdigest() != row[2]:
        raise ValueError("stored snapshot body does not match its sha")
    return {"seq": int(row[0]), "game_pk": int(row[1]), "sha256": str(row[2]),
            "first_fetched": float(row[3]), "fetched": float(row[4]), "source": row[5],
            "body_bytes": raw}


def read_snapshot_body(store_path: Path, seq: int) -> dict:
    """For real adapters: the exact body a PRE_RECEIPT context was built from."""
    con = open_ro(store_path)
    try:
        row = con.execute("SELECT body FROM snapshots WHERE seq=?", (int(seq),)).fetchone()
    finally:
        con.close()
    return json.loads(gzip.decompress(row[0]))


def parse_feed(feed: dict, receipts) -> Tuple[List[StorePitch], Optional[PreState], bool]:
    plays = feed.get("liveData", {}).get("plays", {})
    status = feed.get("gameData", {}).get("status", {}).get("abstractGameState")
    pitches: List[StorePitch] = []
    ordinal = 0
    for play in plays.get("allPlays", []):
        about = play.get("about", {})
        ab = int(about.get("atBatIndex", 0)) + 1
        m = play.get("matchup", {})
        pitcher = int(m.get("pitcher", {}).get("id", 0))
        batter = int(m.get("batter", {}).get("id", 0))
        balls = strikes = 0
        outs_before = 0  # not derivable per pitch from GUMBO here; not used by any forecaster
        for ev in play.get("playEvents", []):
            if not ev.get("isPitch"):
                continue
            ordinal += 1
            pn = int(ev.get("pitchNumber", 0))
            pid, first, recv, clock, etime = receipts.get(
                (ab, pn), (ev.get("playId") or f"{ab}-{pn}", None, None, None, None))
            pitches.append(StorePitch(
                pitch_uid=str(pid), ordinal=ordinal, at_bat_index=ab, pitch_number=pn,
                pitcher_id=pitcher, batter_id=batter,
                pitch_type=(ev.get("details", {}).get("type") or {}).get("code"),
                balls_before=balls, strikes_before=strikes, outs_before=outs_before,
                inning=int(about.get("inning", 0)), top_inning=bool(about.get("isTopInning")),
                first_receipt_wall_ns=first, accepted_wall_ns=recv,
                receipt_clock=clock, provider_event_time=etime))
            c = ev.get("count", {})
            balls, strikes = int(c.get("balls", balls)), int(c.get("strikes", strikes))
    state = None
    cur = plays.get("currentPlay")
    if cur and status != "Final" and not cur.get("about", {}).get("isComplete", False):
        about, m = cur.get("about", {}), cur.get("matchup", {})
        n_p = sum(1 for e in cur.get("playEvents", []) if e.get("isPitch"))
        c = cur.get("count", {})
        off = feed.get("liveData", {}).get("linescore", {}).get("offense", {})
        state = PreState(
            at_bat_index=int(about.get("atBatIndex", 0)) + 1, pitch_number=n_p + 1,
            pitcher_id=int(m.get("pitcher", {}).get("id", 0)),
            batter_id=int(m.get("batter", {}).get("id", 0)),
            balls=int(c.get("balls", 0)), strikes=int(c.get("strikes", 0)),
            outs=int(c.get("outs", 0)), inning=int(about.get("inning", 0)),
            top_inning=bool(about.get("isTopInning")),
            on_1b="first" in off, on_2b="second" in off, on_3b="third" in off)
    return pitches, state, status == "Final"


class LiveStoreSource:
    def __init__(self, store_path: Path, game_pk: int):
        self.store_path = Path(store_path)
        self.game_pk = int(game_pk)
        self._cache: tuple = (None, None)  # (snapshot seq, max pitch_events id) -> snapshot
        self._cached: Optional[SourceSnapshot] = None

    def poll(self) -> SourceSnapshot:
        if not self.store_path.exists():
            return SourceSnapshot([], None, False)
        con = open_ro(self.store_path)
        try:
            con.execute("BEGIN")  # one short read snapshot for both queries (WAL: never blocks the writer)
            key = con.execute("SELECT (SELECT MAX(seq) FROM snapshots WHERE game_pk=?), "
                              "(SELECT MAX(id) FROM pitch_events WHERE game_pk=?)",
                              (self.game_pk, self.game_pk)).fetchone()
            if self._cached is not None and tuple(key) == self._cache:
                con.execute("COMMIT")
                return self._cached
            receipts = _receipts(con, self.game_pk)
            seq, sha, feed, fetched = _latest_body(con, self.game_pk, with_fetched=True)
            con.execute("COMMIT")
        finally:
            con.close()
        if feed is None:
            return SourceSnapshot([], None, False)
        pitches, state, final = parse_feed(feed, receipts)
        self._cache = tuple(key)
        self._cached = SourceSnapshot(pitches, state, final, source_seq=seq, source_sha=sha,
                                      source_fetched_ns=fetched)
        return self._cached


def final_label_rows(store_path: Path, game_pk: int, *, revision: str) -> Dict[str, dict]:
    """Sealed-analysis only: playEvent bodies by provider id.

    revision='store_final' -> last accepted revision in the store (registered primary;
                              the launcher stops polling settle_s=900 s after Final)
    revision='store_first' -> first accepted revision (registered sensitivity)
    first_received_ns is the minimum over all accepted revisions of the pitch.
    """
    aliases = {"final": "store_final", "first": "store_first"}
    revision = aliases.get(revision, revision)
    if revision not in ("store_final", "store_first"):
        raise ValueError(revision)
    con = open_ro(store_path)
    try:
        rows = con.execute(
            "SELECT provider_event_id, at_bat_number, pitch_number, revision, first_received, "
            "receipt_clock, provider_event_time, body "
            "FROM pitch_events WHERE game_pk=? AND outcome IN (?,?) ORDER BY id",
            (int(game_pk), *ACCEPTED)).fetchall()
        _, _, feed = _latest_body(con, int(game_pk))
    finally:
        con.close()
    play_type = {}
    for play in ((feed or {}).get("liveData", {}).get("plays", {}).get("allPlays", [])):
        play_type[int(play.get("about", {}).get("atBatIndex", -1)) + 1] = \
            (play.get("result") or {}).get("eventType")
    min_first: Dict[str, float] = {}
    for r in rows:
        if r[4] is not None:
            min_first[str(r[0])] = min(min_first.get(str(r[0]), r[4]), r[4])
    out: Dict[str, dict] = {}
    for pid, ab, pn, rev, first, clock, etime, body in rows:
        if revision == "store_first" and str(pid) in out:
            continue
        out[str(pid)] = {"at_bat_number": ab, "pitch_number": pn, "revision": rev,
                         "first_received_ns": _ns(min_first.get(str(pid), first)),
                         "receipt_clock": clock,
                         "provider_event_time": etime,
                         "event": dict(json.loads(gzip.decompress(body)),
                                       _play_event_type=play_type.get(int(ab)))}
    return out


def backup_store(store_path: Path, dest: Path) -> str:
    """Label-source copy for the sealed look (B4): SQLite online backup from a
    read-only connection (includes WAL content, never blocks the writer for
    long), then journal_mode=DELETE on the copy.  Returns sha256 of the copy.
    Refuses to overwrite an existing destination."""
    import hashlib
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    src = open_ro(store_path)
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
