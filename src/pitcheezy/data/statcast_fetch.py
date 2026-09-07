"""Statcast 수집 로직 (규약 §5). CLI는 scripts/fetch_data.py.

흐름
    시즌별로 3/15~11/15를 14일 창으로 나눠 pybaseball.statcast()로 받는다.
    창마다 {out_dir}/{version}/_parts/{season}/ 에 parquet + .done 마커를 남겨 재실행 시 이어받는다.
    모든 창이 끝나면 pyarrow로 합쳐 statcast_{season}.parquet 을 만들고,
    manifest.json(행 수·sha256·쿼리 파라미터)을 쓰고 data/versions.md 에 한 줄 append 한다.

확인된 pybaseball 2.2.7 동작 (site-packages/pybaseball/statcast.py, utils.py)
    - statcast()는 하루 단위 요청을 스레드로 병렬 실행하고, 검색 URL이 R|PO|S 로 고정되어
      정규·포스트·스프링이 함께 온다 → 여기서 game_type == "R" 만 남긴다 (규약 §5, ADR-2).
    - 2021년 이후 날짜는 3/15~11/15 밖이면 요청 없이 건너뛴다. SEASON_LO/HI 는 이에 맞춘다.
    - 반환 프레임은 (game_date, game_pk, at_bat_number, pitch_number) 내림차순, nullable dtype.
      pandas 3 에서는 game_date 가 문자열로 남으므로 여기서 datetime 으로 정규화한다.
    - 전부 결측인 컬럼은 창마다 Int64/Float64 로 갈릴 수 있어 조립은 pyarrow permissive 승격으로 한다.
    - 요청에 timeout·재시도가 없다 → 창 단위로 재시도한다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# 고정 파라미터 (규약 §5 "파라미터 고정"). 바꾸면 데이터 버전이 달라져야 한다.
WINDOW_DAYS = 14
SEASON_LO = (3, 15)  # pybaseball statcast_date_range 의 2021+ 클램프와 동일
SEASON_HI = (11, 15)
GAME_TYPE = "R"
SORT_KEYS = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
SAMPLE_PITCHERS = 2
SAMPLE_PITCHES = 100

VERSION_RE = re.compile(r"^d\d{8}-[A-Za-z0-9]+$")
MANIFEST_NAME = "manifest.json"
PARTS_DIR = "_parts"
LEDGER_HEADER = (
    "# 데이터 버전\n\n"
    "| 버전 ID | 시즌 | pybaseball 쿼리 파라미터 | 행 수 | parquet sha256 | 수집 스크립트 커밋 | frozen |\n"
    "|---|---|---|---|---|---|---|\n"
)

Fetcher = Callable[[str, str], Optional[pd.DataFrame]]


def log(msg: str) -> None:
    print(msg, flush=True)


def check_version(version: str) -> None:
    if not VERSION_RE.match(version):
        raise ValueError(f"버전 ID 형식 오류: {version!r} (d{{YYYYMMDD}}-{{tag}}, 예 d20260908-s2326)")


def season_path(version_dir: Path, season: int) -> Path:
    return version_dir / f"statcast_{season}.parquet"


# ---------------------------------------------------------------- 날짜 창


def season_windows(
    season: int, floor: Optional[date] = None, ceiling: Optional[date] = None
) -> list[tuple[date, date]]:
    """[max(3/15, floor), min(11/15, ceiling)] 를 WINDOW_DAYS 일씩 자른다 (양끝 포함). 범위가 비면 []."""
    lo = date(season, *SEASON_LO)
    hi = date(season, *SEASON_HI)
    if floor is not None:
        lo = max(lo, floor)
    if ceiling is not None:
        hi = min(hi, ceiling)
    windows: list[tuple[date, date]] = []
    cur = lo
    while cur <= hi:
        end = min(cur + timedelta(days=WINDOW_DAYS - 1), hi)
        windows.append((cur, end))
        cur = end + timedelta(days=1)
    return windows


# ---------------------------------------------------------------- 창 수집


def default_fetcher(start: str, end: str) -> Optional[pd.DataFrame]:
    from pybaseball import statcast  # 지연 import: 테스트·--sample 은 pybaseball 없이도 돈다

    return statcast(start_dt=start, end_dt=end, verbose=False)


def _retryable() -> tuple[type[BaseException], ...]:
    excs: list[type[BaseException]] = [ConnectionError, TimeoutError]
    try:
        import requests

        excs.append(requests.RequestException)
    except ImportError:
        pass
    try:
        from pybaseball.statcast import StatcastException

        excs.append(StatcastException)
    except ImportError:
        pass
    return tuple(excs)


def normalize(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """game_type == R 필터, game_date datetime 정규화, SORT_KEYS 오름차순 정렬."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    missing = [c for c in ["game_type", *SORT_KEYS] if c not in raw.columns]
    if missing:
        raise ValueError(f"예상 컬럼 없음: {missing}. Savant 스키마가 바뀌었는지 확인")
    mask = raw["game_type"].eq(GAME_TYPE).fillna(False).astype(bool)
    df = raw.loc[mask].copy()
    if len(df) == 0:
        return df.reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    return df.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)


def fetch_window(
    start: date,
    end: date,
    fetcher: Optional[Fetcher] = None,
    retries: int = 3,
    backoff: float = 5.0,
) -> tuple[pd.DataFrame, int]:
    """한 창을 받아 (정규시즌 프레임, 받은 원본 행 수) 를 돌려준다. 네트워크 오류는 지수 백오프로 재시도."""
    fetcher = fetcher or default_fetcher
    retryable = _retryable()
    raw: Optional[pd.DataFrame] = None
    for attempt in range(1, retries + 1):
        try:
            raw = fetcher(start.isoformat(), end.isoformat())
            break
        except retryable as exc:
            if attempt >= retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            log(f"    재시도 {attempt}/{retries - 1}: {type(exc).__name__}: {exc} ({wait:g}s 대기)")
            time.sleep(wait)
    rows_raw = 0 if raw is None else int(len(raw))
    return normalize(raw), rows_raw


# ---------------------------------------------------------------- 시즌 수집


@dataclass
class SeasonFetch:
    season: int
    path: Path
    windows_total: int
    windows_fetched: int
    skipped: bool  # 최종본이 이미 있어 아무것도 하지 않음


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def assemble_parts(parts_dir: Path, stems: list[str]) -> pa.Table:
    """현재 창 목록(stems)에 해당하는 part 만 합친다. 다른 창 배치로 남은 part 는 경고만."""
    wanted = [parts_dir / f"{s}.parquet" for s in stems]
    parts = [p for p in wanted if p.exists()]
    extra = sorted(set(parts_dir.glob("*.parquet")) - set(wanted))
    if extra:
        log(f"  경고: 현재 창 배치에 없는 part {len(extra)}개는 조립에서 제외: {[p.name for p in extra]}")
    if not parts:
        raise RuntimeError(f"조립할 창이 없음 (모든 창이 0행): {parts_dir}")
    tables = [pq.read_table(p).replace_schema_metadata(None) for p in parts]
    try:
        table = pa.concat_tables(tables, promote_options="permissive")
    except TypeError as exc:  # pyarrow < 14
        raise RuntimeError("pyarrow >= 14 필요 (concat_tables promote_options)") from exc
    return table.sort_by([(k, "ascending") for k in SORT_KEYS])


def fetch_season(
    season: int,
    version_dir: Path,
    floor: Optional[date] = None,
    ceiling: Optional[date] = None,
    fetcher: Optional[Fetcher] = None,
) -> SeasonFetch:
    """창 단위로 받아 part 저장 → 전부 끝나면 statcast_{season}.parquet 조립. 재실행 시 완료 창은 건너뜀."""
    final = season_path(version_dir, season)
    windows = season_windows(season, floor, ceiling)
    if final.exists():
        log(f"[{season}] 최종본 있음 → 건너뜀 (재수집하려면 삭제): {final}")
        return SeasonFetch(season, final, len(windows), 0, True)
    if not windows:
        raise ValueError(f"[{season}] 수집 범위가 비었음 (floor={floor}, ceiling={ceiling})")

    parts_dir = version_dir / PARTS_DIR / str(season)
    parts_dir.mkdir(parents=True, exist_ok=True)
    stems: list[str] = []
    fetched = 0
    for start, end in windows:
        stem = f"{start.isoformat()}_{end.isoformat()}"
        stems.append(stem)
        part = parts_dir / f"{stem}.parquet"
        done = parts_dir / f"{stem}.done"
        if done.exists():
            info = json.loads(done.read_text(encoding="utf-8"))
            if info.get("rows", 0) == 0 or part.exists():
                log(f"[{season}] {start}..{end}  완료 창 건너뜀 (R {info.get('rows', 0):,}행)")
                continue
            log(f"[{season}] {start}..{end}  마커만 있고 parquet 없음 → 재수집")
        df, rows_raw = fetch_window(start, end, fetcher=fetcher)
        if len(df):
            _write_parquet_atomic(df, part)
        done.write_text(
            json.dumps(
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "rows_raw": rows_raw,
                    "rows": int(len(df)),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            ),
            encoding="utf-8",
        )
        fetched += 1
        log(
            f"[{season}] {start}..{end}  받음 {rows_raw:,}행 → R {len(df):,}행"
            + ("  저장" if len(df) else "  (빈 창)")
        )

    table = assemble_parts(parts_dir, stems)
    _write_table_atomic(table, final)
    log(f"[{season}] 조립 완료 {table.num_rows:,}행 → {final}")
    return SeasonFetch(season, final, len(windows), fetched, False)


# ---------------------------------------------------------------- 요약·manifest·장부


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def season_summary(path: Path) -> dict:
    rows = pq.read_metadata(path).num_rows
    first = last = None
    if rows:
        mm = pc.min_max(pq.read_table(path, columns=["game_date"])["game_date"]).as_py()
        first, last = str(mm["min"])[:10], str(mm["max"])[:10]
    return {
        "file": path.name,
        "rows": int(rows),
        "sha256": sha256_file(path),
        "first_game_date": first,
        "last_game_date": last,
    }


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def git_commit(repo_root: Path) -> str:
    """짧은 HEAD 해시. 워킹트리에 변경(추적·미추적)이 있으면 -dirty 접미. git 없으면 unknown."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        head = run("rev-parse", "--short", "HEAD")
        return head + ("-dirty" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_manifest(version_dir: Path) -> dict:
    path = version_dir / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"manifest 없음: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    version_dir: Path, version: str, summaries: dict[int, dict], query: dict, fetch_commit: str
) -> Path:
    """기존 manifest 가 있으면 시즌 항목을 병합한다 (시즌을 나눠 실행해도 누적)."""
    path = version_dir / MANIFEST_NAME
    manifest: dict = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fetch_commit": fetch_commit,
            "versions": {
                "pybaseball": _pkg_version("pybaseball"),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "query": query,
        }
    )
    seasons = manifest.setdefault("seasons", {})
    for season, summary in summaries.items():
        seasons[str(season)] = summary
    manifest["seasons"] = dict(sorted(seasons.items()))
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def ledger_row(manifest: dict) -> str:
    """data/versions.md 표의 한 행 (7열). sha256 는 규약대로 시즌별 전체 64자."""
    q = manifest["query"]
    seasons = manifest["seasons"]
    params = (
        f"statcast(start,end) {q['window_days']}일 창 {q['season_lo']}~{q['season_hi']}"
        + (f", floor {q['floor']}" if q.get("floor") else "")
        + f", ceiling {q['ceiling']}, game_type={q['game_type']}"
        + f", pybaseball {manifest['versions']['pybaseball']}"
    )
    total = sum(s["rows"] for s in seasons.values())
    rows = "<br>".join(f"{k}: {v['rows']:,}" for k, v in seasons.items()) + f"<br>계 {total:,}"
    hashes = "<br>".join(f"{k}: {v['sha256']}" for k, v in seasons.items())
    cells = [manifest["version"], ", ".join(seasons), params, rows, hashes, manifest["fetch_commit"], ""]
    return "| " + " | ".join(cells) + " |"


def has_ledger_row(ledger: Path, version: str) -> bool:
    if not ledger.exists():
        return False
    prefix = f"| {version} |"
    return any(line.startswith(prefix) for line in ledger.read_text(encoding="utf-8").splitlines())


def append_ledger(ledger: Path, version: str, row: str) -> bool:
    """버전 행이 없을 때만 append (멱등). 장부 파일이 없으면 헤더부터 만든다."""
    if has_ledger_row(ledger, version):
        return False
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(LEDGER_HEADER, encoding="utf-8")
    text = ledger.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    ledger.write_text(text + row + "\n", encoding="utf-8")
    return True


def verify(version_dir: Path, ledger: Path) -> list[str]:
    """manifest 의 시즌마다 파일 존재·행 수·sha256 을 대조하고 장부 행을 확인. 문제 목록을 돌려준다."""
    try:
        manifest = load_manifest(version_dir)
    except FileNotFoundError as exc:
        return [str(exc)]
    problems: list[str] = []
    for season, info in manifest["seasons"].items():
        path = version_dir / info["file"]
        if not path.exists():
            problems.append(f"{season}: 파일 없음 {path}")
            continue
        try:
            rows = pq.read_metadata(path).num_rows
        except Exception as exc:  # 깨진 parquet
            problems.append(f"{season}: parquet 읽기 실패 {type(exc).__name__}: {exc}")
            continue
        sha = sha256_file(path)
        ok = True
        if rows != info["rows"]:
            problems.append(f"{season}: 행 수 불일치 manifest {info['rows']:,} vs 파일 {rows:,}")
            ok = False
        if sha != info["sha256"]:
            problems.append(f"{season}: sha256 불일치 manifest {info['sha256'][:16]}… vs 파일 {sha[:16]}…")
            ok = False
        log(f"[{season}] {'OK' if ok else 'FAIL'} rows={rows:,} sha256={sha}")
    if not has_ledger_row(ledger, manifest["version"]):
        problems.append(f"장부에 {manifest['version']} 행 없음: {ledger}")
    return problems


# ---------------------------------------------------------------- 픽스처 샘플


def make_sample(
    season_parquet: Path, n_pitchers: int = SAMPLE_PITCHERS, n_pitches: int = SAMPLE_PITCHES
) -> pd.DataFrame:
    """투구 수 상위 n_pitchers 명(동률은 id 작은 쪽)을 골라, 투수별로 시간순 타석을 통째로 누적해
    n_pitches 구 이상이 되는 시점까지 담는다 (에피소드 = 타석이라 타석을 자르지 않음). 전 컬럼 유지."""
    df = pd.read_parquet(season_parquet)
    df = df.loc[df["game_type"].eq(GAME_TYPE).fillna(False).astype(bool)]
    counts = df.groupby("pitcher").size().rename("n").reset_index()
    counts = counts.sort_values(["n", "pitcher"], ascending=[False, True], kind="mergesort")
    pieces = []
    for pitcher in counts["pitcher"].head(n_pitchers).tolist():
        sub = df.loc[df["pitcher"] == pitcher].sort_values(SORT_KEYS, kind="mergesort")
        sizes = sub.groupby(["game_pk", "at_bat_number"], sort=False).size()  # 시간순 타석
        need = int((sizes.cumsum() < n_pitches).sum()) + 1
        keep = sizes.index[:need]
        pa_key = pd.MultiIndex.from_frame(sub[["game_pk", "at_bat_number"]])
        pieces.append(sub.loc[pa_key.isin(keep)])
    return pd.concat(pieces).sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)


def update_fixture_readme(
    readme: Path, version: str, season: int, pitchers: list[tuple[int, str]], rows: int
) -> str:
    """tests/fixtures/README.md 표의 sample.parquet 행을 교체하거나 append."""
    row = (
        f"| sample.parquet | {version} (시즌 {season}) | "
        + ", ".join(f"{pid} {name}".strip() for pid, name in pitchers)
        + f" | {rows:,} |"
    )
    if readme.exists():
        lines = readme.read_text(encoding="utf-8").splitlines()
    else:
        lines = ["# tests/fixtures", "", "| 파일 | 출처 데이터 버전 | 투수 | 행 수 |", "|---|---|---|---|"]
    for i, line in enumerate(lines):
        if line.startswith("| sample.parquet |"):
            lines[i] = row
            break
    else:
        lines.append(row)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return row


# ---------------------------------------------------------------- 모드별 진입점


def run_fetch(
    version: str,
    seasons: list[int],
    out_dir: Path,
    ledger: Path,
    repo_root: Path,
    floor: Optional[date] = None,
    ceiling: Optional[date] = None,
    fetcher: Optional[Fetcher] = None,
) -> Path:
    """시즌 수집 → manifest → 장부. 돌려주는 값은 manifest 경로."""
    check_version(version)
    if ceiling is None:
        ceiling = date.today() - timedelta(days=1)  # 당일 데이터는 미완
    version_dir = out_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    lo = f"{SEASON_LO[0]:02d}-{SEASON_LO[1]:02d}"
    hi = f"{SEASON_HI[0]:02d}-{SEASON_HI[1]:02d}"
    log(f"버전 {version} → {version_dir}")
    log(
        f"시즌 {', '.join(map(str, seasons))} | 창 {WINDOW_DAYS}일 {lo}~{hi}"
        f" | floor {floor} | ceiling {ceiling} | game_type={GAME_TYPE}"
    )
    for season in seasons:
        fetch_season(season, version_dir, floor, ceiling, fetcher)

    query = {
        "window_days": WINDOW_DAYS,
        "season_lo": lo,
        "season_hi": hi,
        "floor": floor.isoformat() if floor else None,
        "ceiling": ceiling.isoformat(),
        "game_type": GAME_TYPE,
        "sort_keys": SORT_KEYS,
        "note": "pybaseball statcast() 검색 URL은 R|PO|S 고정 → 후처리로 game_type=R 만 남김",
    }
    summaries = {s: season_summary(season_path(version_dir, s)) for s in seasons}
    path = write_manifest(version_dir, version, summaries, query, git_commit(repo_root))
    row = ledger_row(load_manifest(version_dir))
    added = append_ledger(ledger, version, row)
    log(f"manifest → {path}")
    log(("장부 행 추가: " if added else "장부에 이미 있음 (갱신하려면 수동 교체): ") + str(ledger))
    log(row)
    return path


def run_sample(
    version_dir: Path, out_path: Path, fixtures_dir: Path, season: Optional[int] = None
) -> pd.DataFrame:
    """manifest 의 시즌 parquet 에서 픽스처 샘플을 만든다. out_path 가 tests/fixtures 안이면 README 도 갱신."""
    manifest = load_manifest(version_dir)
    seasons = sorted(int(s) for s in manifest["seasons"])
    if not seasons:
        raise ValueError("manifest 에 시즌이 없음")
    season = season or seasons[-1]
    src = season_path(version_dir, season)
    if not src.exists():
        raise FileNotFoundError(f"시즌 parquet 없음: {src}")
    df = make_sample(src)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    pitchers: list[tuple[int, str]] = []
    for pid in sorted(df["pitcher"].unique()):
        name = df.loc[df["pitcher"] == pid, "player_name"].iloc[0] if "player_name" in df.columns else ""
        pitchers.append((int(pid), str(name)))
    log(f"샘플 {len(df):,}행 × {df.shape[1]}컬럼 (시즌 {season}, {src.name}) → {out_path}")
    for pid, name in pitchers:
        log(f"  투수 {pid} {name}: {int((df['pitcher'] == pid).sum())}구")
    if out_path.resolve().parent == fixtures_dir.resolve():
        row = update_fixture_readme(fixtures_dir / "README.md", manifest["version"], season, pitchers, len(df))
        log(f"README 갱신: {row}")
    else:
        log("tests/fixtures 밖 경로 → README 갱신 생략")
    return df
