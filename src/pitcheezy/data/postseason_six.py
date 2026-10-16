"""Bounded, exact-game Statcast acquisition for the fixed Oct 12–15 L cohort.

Only source/generated use is authorized by this module. Actual requests need an
external reviewed registration and grant. The regular-season fetcher is unchanged.
"""
from __future__ import annotations

import hashlib
import http.client
import importlib.metadata
import inspect
import json
import signal
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pitcheezy.data import statcast_fetch as sf
from experiments.ws_campaign.source.abc_corpus import PROJECT

GAMES = ((849812, "2026-10-12"), (849820, "2026-10-12"),
         (849813, "2026-10-13"), (849811, "2026-10-14"),
         (849810, "2026-10-15"), (849818, "2026-10-15"))
HOST = "baseballsavant.mlb.com"
PATH = "/statcast_search/csv?all=true&type=details&game_pk={}"
MAX_BODY = 8 * 1024 * 1024
MAX_ROWS = 2000
MAX_COLUMNS = 150
MAX_OUTPUT = 100 * 1024 * 1024
READ_SIZE = 64 * 1024
TIMEOUT = 15
MAX_WALL_SECONDS = 180
EXPECTED_COLUMNS = 29
SCHEDULE_SHA256 = "8939bbe1618fb0e6432f630588f1b9f576940944736565dcd1a8130415636314"


class CollectionStop(ValueError):
    pass


def fetch_game(game_pk: int) -> bytes:
    """One direct TLS GET; no proxy, redirect, retry, disk cache, or wider dates."""
    if game_pk not in dict(GAMES):
        raise CollectionStop("unregistered game")
    connection = http.client.HTTPSConnection(HOST, timeout=TIMEOUT)
    try:
        connection.request("GET", PATH.format(game_pk), headers={"Accept": "text/csv", "Accept-Encoding": "identity", "User-Agent": "pitcheezy-postseason-six/1"})
        response = connection.getresponse()
        if response.status != 200:
            raise CollectionStop("HTTP status is not 200")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdecimal() or int(length) > MAX_BODY):
            raise CollectionStop("invalid or oversized Content-Length")
        parts, size = [], 0
        while True:
            piece = response.read(min(READ_SIZE, MAX_BODY + 1 - size))
            if not piece:
                break
            size += len(piece)
            if size > MAX_BODY:
                raise CollectionStop("response body exceeds cap")
            parts.append(piece)
        if length is not None and size != int(length):
            raise CollectionStop("truncated response body")
        return b"".join(parts)
    finally:
        connection.close()


def parse_csv(body: bytes) -> pd.DataFrame:
    """Use pybaseball 2.2.7's pure CSV parser, not its unbounded HTTP/cache path."""
    from pybaseball.datasources.statcast import get_statcast_data_from_csv

    if not body or len(body) > MAX_BODY:
        raise CollectionStop("empty or oversized CSV body")
    try:
        frame = get_statcast_data_from_csv(body.decode("utf-8-sig"))
    except (UnicodeError, ValueError, pd.errors.ParserError) as exc:
        raise CollectionStop("CSV parse failed") from exc
    if frame is None or not 0 < len(frame) <= MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise CollectionStop("CSV shape outside registered bounds")
    if frame.columns.duplicated().any() or set(PROJECT).difference(frame.columns):
        raise CollectionStop("CSV schema missing or duplicate columns")
    return frame


def census(frame: pd.DataFrame, game_pk: int, official_date: str) -> dict:
    """No rows are silently dropped; reject ambiguous source identity."""
    keys = sf.SORT_KEYS
    numeric = frame[list(keys)].apply(pd.to_numeric, errors="coerce")
    valid = numeric.notna().all(axis=1)
    for column in keys:
        valid &= numeric[column].gt(0) & numeric[column].lt(2**53) & numeric[column].eq(numeric[column].floordiv(1))
    typed = frame["game_type"].astype("string")
    dated = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pitcher = pd.to_numeric(frame["pitcher"], errors="coerce")
    valid_pitcher = pitcher.notna() & pitcher.gt(0) & pitcher.lt(2**53) & pitcher.eq(pitcher.floordiv(1))
    duplicates = int(numeric.loc[valid].duplicated(list(keys), keep=False).sum())
    report = {"game_pk": game_pk, "officialDate": official_date, "gameType": "L",
              "rows": int(len(frame)), "source_columns": int(len(frame.columns)),
              "null_or_invalid_physical_key_rows": int((~valid).sum()),
              "null_or_invalid_pitcher_rows": int((~valid_pitcher).sum()),
              "duplicate_physical_key_rows": duplicates,
              "wrong_game_pk_rows": int((numeric["game_pk"] != game_pk).fillna(True).sum()),
              "wrong_game_type_rows": int((typed != "L").fillna(True).sum()),
              "wrong_or_missing_game_date_rows": int((dated != official_date).fillna(True).sum())}
    if any(report[k] for k in ("null_or_invalid_physical_key_rows", "duplicate_physical_key_rows",
                              "wrong_game_pk_rows", "wrong_game_type_rows", "wrong_or_missing_game_date_rows")):
        raise CollectionStop("source identity/quality census STOP: " + json.dumps(report, sort_keys=True))
    return report


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(out: Path, *, fetch=fetch_game) -> dict:
    """Fresh output only; six exact requests or fail closed with partial evidence."""
    out = Path(out)
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise CollectionStop("existing wall timer")
    if out.exists() or not out.parent.is_dir() or out.parent.is_symlink():
        raise CollectionStop("fresh output directory required")
    out.mkdir(mode=0o700)
    raw_dir = out / "raw"
    raw_dir.mkdir(mode=0o700)
    records, raw_receipts, frames = [], [], []
    raw_digest = hashlib.sha256()
    started = datetime.now(timezone.utc).isoformat()
    attempted = 0
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timeout(_signum, _frame):
        raise CollectionStop("total wall deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(MAX_WALL_SECONDS)
    try:
        for game_pk, official_date in GAMES:
            attempted += 1
            body = fetch(game_pk)
            if not isinstance(body, bytes) or not 0 < len(body) <= MAX_BODY:
                raise CollectionStop("fetch returned invalid body")
            raw_path = raw_dir / f"game-{game_pk}.csv"
            with raw_path.open("xb") as stream:
                stream.write(body)
            retrieved = datetime.now(timezone.utc).isoformat()
            raw_receipts.append({"game_pk": game_pk, "file": str(raw_path.relative_to(out)),
                                 "bytes": len(body), "sha256": _sha(body), "retrieved_at_utc": retrieved})
            raw_digest.update(game_pk.to_bytes(8, "big"))
            raw_digest.update(len(body).to_bytes(8, "big"))
            raw_digest.update(body)
            frame = parse_csv(body)
            quality = census(frame, game_pk, official_date)
            records.append({**quality, "raw_csv": str(raw_path.relative_to(out)),
                            "raw_bytes": len(body), "raw_sha256": _sha(body),
                            "retrieved_at_utc": retrieved})
            frames.append(frame[PROJECT].copy())
        projected = pd.concat(frames, ignore_index=True)
        if len(projected) > len(GAMES) * MAX_ROWS:
            raise CollectionStop("total row cap")
        projected["game_date"] = pd.to_datetime(projected["game_date"], errors="raise").dt.normalize()
        table = sf.cast_policy(pa.Table.from_pandas(projected, preserve_index=False), final=True)
        table = sf.sort_table(table)
        if table.num_columns != EXPECTED_COLUMNS or table.column_names != PROJECT:
            raise CollectionStop("projected schema changed")
        dest = out / "postseason_six.parquet"
        pq.write_table(table, dest, compression=sf.PARQUET_COMPRESSION, version=sf.PARQUET_VERSION)
        if dest.stat().st_size > MAX_OUTPUT:
            raise CollectionStop("parquet output exceeds cap")
        manifest = {"schema": "postseason-six-exact-game-v1", "status": "COMPLETE",
                    "data_role": "EXPOSED DEVELOPMENT; prior exposure UNKNOWN",
                    "arrival": "UNKNOWN; corrected retrospective source", "schedule_projection_sha256": SCHEDULE_SHA256,
                    "source_route": f"https://{HOST}" + PATH.format("{game_pk}"),
                    "requests": len(GAMES), "retry_count": 0, "redirects_allowed": False,
                    "cache_used": False, "started_at_utc": started,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "raw_order": "each original CSV retained; projected Parquet sorted by physical key",
                    "versions": {"pybaseball": importlib.metadata.version("pybaseball"),
                                 "pandas": pd.__version__, "pyarrow": pa.__version__},
                    "games": records, "raw_total_bytes": sum(r["raw_bytes"] for r in records),
                    "raw_aggregate_sha256": raw_digest.hexdigest(),
                    "raw_aggregate_sha256_semantics": "SHA256 of game_pk uint64 big-endian, body length uint64 big-endian, raw bytes for each fixed game in GAMES order",
                    "projected": {"file": dest.name, "bytes": dest.stat().st_size,
                                  "sha256": _file_sha(dest), "rows": table.num_rows,
                                  "columns": table.column_names, "row_groups": pq.read_metadata(dest).num_row_groups}}
        (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        return manifest
    except BaseException as exc:
        (out / "STOP.json").write_text(json.dumps({"status": "STOP", "reason": type(exc).__name__,
            "attempted_requests": attempted, "raw_receipts": raw_receipts,
            "completed_games": records, "source_route": f"https://{HOST}" + PATH.format("{game_pk}"),
            "schedule_projection_sha256": SCHEDULE_SHA256}, sort_keys=True, indent=2) + "\n")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def gated(registration: Path, grant: Path) -> dict:
    """Managed CLI consistency gate; grant/review files are local provenance."""
    reg_bytes = Path(registration).read_bytes()
    reg = json.loads(reg_bytes)
    repo = Path(__file__).resolve().parents[3]
    pins = reg.get("source_pins")
    required = ("scripts/fetch_data.py", "src/pitcheezy/data/postseason_six.py",
                "src/pitcheezy/data/statcast_fetch.py", "experiments/ws_campaign/source/abc_corpus.py")
    if (reg.get("schema") != "postseason-six-acquisition-v1" or reg.get("execute") is not False
            or reg.get("worktree") != str(repo) or not isinstance(pins, dict)
            or set(pins) != set(required) or reg.get("games") != [list(x) for x in GAMES]
            or reg.get("source_route") != f"https://{HOST}" + PATH.format("{game_pk}")
            or reg.get("max_requests") != len(GAMES) or reg.get("max_body_bytes_each") != MAX_BODY
            or reg.get("max_rows_each") != MAX_ROWS or reg.get("max_columns_each") != MAX_COLUMNS
            or reg.get("max_output_bytes") != MAX_OUTPUT or reg.get("wall_seconds") != MAX_WALL_SECONDS
            or reg.get("socket_timeout_seconds") != TIMEOUT
            or reg.get("schedule_projection_sha256") != SCHEDULE_SHA256):
        raise CollectionStop("registration contract changed")
    for name, digest in pins.items():
        if _file_sha(repo / name) != digest:
            raise CollectionStop("source pin changed")
    from pybaseball.datasources import statcast as parser
    parser_files = {"csv_parser": Path(inspect.getsourcefile(parser)),
                    "postprocessing": Path(inspect.getsourcefile(parser.postprocessing))}
    if (reg.get("parser_source_pins") != {name: {"path": str(path), "sha256": _file_sha(path)}
                                          for name, path in parser_files.items()}
            or reg.get("versions") != {"pybaseball": importlib.metadata.version("pybaseball"),
                                       "pandas": pd.__version__, "pyarrow": pa.__version__}):
        raise CollectionStop("CSV parser/dependency pin changed")
    approval = json.loads(Path(grant).read_bytes())
    if (approval.get("approved_execute") is not True or approval.get("authorized_by") != "root"
            or approval.get("verdict") != "PASS" or approval.get("attempt_id") != reg.get("attempt_id")
            or approval.get("registration_sha256") != _sha(reg_bytes)
            or approval.get("reviewer") in (None, "", reg.get("author"))
            or not isinstance(approval.get("review_path"), str)
            or not isinstance(approval.get("review_sha256"), str)):
        raise CollectionStop("exact Root grant required")
    review_bytes = Path(approval["review_path"]).read_bytes()
    review = json.loads(review_bytes)
    if (_sha(review_bytes) != approval["review_sha256"]
            or review.get("registration_sha256") != _sha(reg_bytes)
            or review.get("reviewer") != approval["reviewer"]
            or not str(review.get("verdict", "")).startswith("PASS")):
        raise CollectionStop("independent source review required")
    return run(Path(reg["output"]))
