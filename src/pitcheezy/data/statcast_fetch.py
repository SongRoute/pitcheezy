"""Statcast 수집 로직 (규약 §5). CLI는 scripts/fetch_data.py.

흐름
    시즌별 정규시즌 날짜 구간(SEASON_DATES, 코드 상수)을 월 단위 청크로 나눠 pybaseball.statcast()로 받는다.
    청크마다 {out}/{version}/_chunks/{season}/{start}_{end}.parquet + .done 마커를 남겨 재실행 시 이어받는다.
    모든 청크가 끝나면 합쳐 dtype 을 고정하고 (game_pk, at_bat_number, pitch_number) 로 정렬해
    statcast_{season}.parquet 을 쓴다. 홀드아웃 시즌은 holdout_{season}/ 아래 (학습 시즌과 같은 디렉터리에 두지 않음).
    끝나면 manifest.json 을 쓰고 data/versions.md 에 시즌(파일)마다 한 행을 append 한다.
    HOLDOUT_SEASONS(2026) 는 --seasons 로 항상 거부한다 (홀드아웃 전용). --freeze 는 수집 상한 ≥ 시즌 종료일일 때만.

확인된 pybaseball 2.2.7 동작 (site-packages/pybaseball/statcast.py, utils.py, datasources/statcast.py)
    - statcast()는 하루 단위 요청을 스레드로 병렬 실행한다. 검색 URL 이 R|PO|S 고정이라 정규·포스트·스프링이
      함께 온다 → 여기서 game_type == "R" 만 남긴다 (규약 §5, ADR-2).
    - 2021+ 시즌은 3/15~11/15 밖 날짜를 요청 없이 건너뛴다. SEASON_DATES 는 모두 그 안에 있다.
    - requests.get(url, timeout=None) 이고 재시도 없음. timeout=None 을 명시해 넘기므로 socket.setdefaulttimeout 은
      무시된다 (requests/urllib3 가 sock.settimeout(None) 을 다시 건다) → pybaseball 모듈이 보는 requests.get 을
      타임아웃 붙인 래퍼로 바꾼다 (_install_request_timeout) + 청크 단위 재시도.
    - 반환 dtype 이 pandas 버전에 따라 다르다 (pandas 3: game_date 가 str, 전부 결측인 컬럼이 Int64 /
      pandas 2: game_date 가 datetime, 문자열이 object). 그래서 dtype 은 여기서 명시적으로 고정한다 (target_type).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import types
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------- 고정 파라미터 (규약 §5 "파라미터 고정")
# 바꾸면 데이터 버전이 달라져야 한다.

# 정규시즌 첫 경기일 ~ 마지막 경기일. 출처: MLB Stats API schedule?gameType=R (조회 2026-09-11, PR 본문에 기록).
#   2023: 3/30 개막 ~ 10/1 최종일. 10/2 는 9/28 MIA@NYM 서스펜디드 경기의 재개 예정일 엔트리(재개 없이 종료) → 포함
#   2024: 3/20 서울 시리즈 ~ 9/30 NYM@ATL 순연 더블헤더 (예정 최종일 9/29)
#   2025: 3/18 도쿄 시리즈 ~ 9/28 최종일
#   2026: 3/25 NYY@SFG 개막 ~ 9/27 예정 최종일 (진행 중 — 실제 수집 상한은 실행일 기준 data_ceiling 으로 잘린다)
# 범위가 넓어도 game_type == R 필터가 있어 결과는 같다. 좁으면 경기를 놓치므로 순연·재개일까지 포함한다.
SEASON_DATES: dict[int, tuple[date, date]] = {
    2023: (date(2023, 3, 30), date(2023, 10, 2)),
    2024: (date(2024, 3, 20), date(2024, 9, 30)),
    2025: (date(2025, 3, 18), date(2025, 9, 28)),
    2026: (date(2026, 3, 25), date(2026, 9, 27)),
}
# 홀드아웃 전용 시즌 (CLAUDE.md: 2026 은 OPE·분해 전용, 전이 모델 학습·튜닝·모델 선택에 쓰지 않는다).
# 시즌이 끝났든 아니든 --seasons 로는 항상 거부하고 --holdout 으로만 받는다.
HOLDOUT_SEASONS: frozenset[int] = frozenset({2026})

# 수집 상한 = 실행일 − CEILING_LAG 일. 미국 서부 야간 경기는 UTC 09:00 (KST 18:00) 께 끝나므로 KST·UTC 어느 쪽
# 달력으로도 "이틀 전"은 항상 완료된 날이다 (전날은 실행 시각에 따라 진행 중일 수 있음).
# --freeze 도 같은 상한을 쓴다: 상한 ≥ 시즌 종료일이어야 마지막 날 경기까지 완료된 상태로 고정된다.
CEILING_LAG_DAYS = 2

GAME_TYPE = "R"
GAME_TYPE_COLUMN = "game_type"
DATE_COLUMN = "game_date"
SORT_KEYS = ("game_pk", "at_bat_number", "pitch_number")

# dtype 정책 (target_type). 이름으로 정해지는 컬럼이 우선, 나머지는 관측 타입으로: 숫자 → float64, 문자열 → string.
#   game_date → timestamp[ns] / INT64_COLUMNS → int64 / STRING_COLUMNS → string
#   어느 집합에도 없고 청크 안에서 전부 결측이면 타입을 미루고(null), 시즌 조립 때 다른 청크의 타입을 따른다.
#   끝까지 결측이면 float64. 이름 집합은 Savant CSV 문서(baseballsavant.mlb.com/csv-docs, 113필드) 기준.
# 여기 없는 정수 컬럼은 float64 로 저장될 뿐 값은 보존된다. 반대로 INT64 에 있는데 소수값이 오면 캐스팅이 실패한다 (의도).
INT64_COLUMNS = frozenset(
    {
        "game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "game_year",
        "balls", "strikes", "outs_when_up", "inning", "zone", "hit_location",
        "on_1b", "on_2b", "on_3b",
        "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        "home_score", "away_score", "bat_score", "fld_score",
        "post_home_score", "post_away_score", "post_bat_score", "post_fld_score",
        "home_score_diff", "bat_score_diff",
        "launch_speed_angle", "woba_denom", "babip_value", "iso_value", "hit_distance_sc", "spin_axis",
        "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
        "pitcher_days_since_prev_game", "batter_days_since_prev_game",
        "pitcher_days_until_next_game", "batter_days_until_next_game",
        "age_pit", "age_bat", "age_pit_legacy", "age_bat_legacy",
        # 구버전 CSV 의 중복 컬럼 (pandas 가 .1 접미로 읽음). 없으면 무시된다.
        "pitcher.1", "fielder_2.1",
    }
)
# 문자열로 고정하는 컬럼. 한 청크(월) 전체가 결측이어도 string 이어야 다른 청크와 합쳐진다 (sv_id 가 실제 사례).
STRING_COLUMNS = frozenset(
    {
        "pitch_type", "player_name", "events", "description", "des", "game_type", "stand", "p_throws",
        "home_team", "away_team", "type", "bb_type", "inning_topbot", "pitch_name",
        "if_fielding_alignment", "of_fielding_alignment", "sv_id",
    }
)

PARQUET_COMPRESSION = "snappy"
PARQUET_VERSION = "2.6"
ROW_GROUP_SIZE = 1 << 20

RETRIES = 3  # 첫 시도 이후 재시도 횟수
BACKOFF_SECONDS = 5.0  # 5, 10, 20
REQUEST_TIMEOUT_SECONDS = (30.0, 120.0)  # (connect, read). 하루치 CSV(~3MB)는 실측 10초 안팎

CHUNKS_DIR = "_chunks"
DRYRUN_DIR = "_dryrun"
MANIFEST_NAME = "manifest.json"

# ---------------------------------------------------------------- 장부 (data/versions.md)

LEDGER_COLUMNS = (
    "버전 ID", "시즌", "파일", "날짜 범위", "행 수", "parquet sha256", "pybaseball 버전", "수집 커밋", "frozen",
    "pyarrow 버전", "수집 상한일",
)
# 2026-09-12 이전 행은 앞 9열만 있다 (pyarrow 버전·수집 상한일 없음). 파서는 두 형태를 모두 읽는다.
LEDGER_LEGACY_COLUMN_COUNT = 9
LEDGER_HEADER_LINE = "| " + " | ".join(LEDGER_COLUMNS) + " |"
LEDGER_SEPARATOR_LINE = "|" + "---|" * len(LEDGER_COLUMNS)

Fetcher = Callable[[str, str], Optional[pd.DataFrame]]


class FetchError(RuntimeError):
    """수집 흐름의 사용자 오류·데이터 오류 (CLI 가 메시지만 출력하고 종료)."""


class SchemaError(FetchError):
    """Savant 응답에 예상 컬럼이 없음. 일시적 오류(HTML 에러 페이지 등)일 수 있어 재시도 대상."""


class DtypeError(FetchError):
    """dtype 정책과 데이터가 어긋남 (정수 컬럼에 소수값, 청크 간 타입 충돌). 재시도해도 같으므로 즉시 실패."""


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- 버전·경로


def version_id(tag: str, today: date) -> str:
    """d{YYYYMMDD}-{tag}"""
    if not tag or not tag.replace("-", "").replace("_", "").isalnum():
        raise FetchError(f"태그 형식 오류: {tag!r} (영숫자, 예 s2325)")
    return f"d{today:%Y%m%d}-{tag}"


def check_version_id(version: str) -> None:
    ok = (
        len(version) > 10
        and version[0] == "d"
        and version[1:9].isdigit()
        and version[9] == "-"
        and version[10:].replace("-", "").replace("_", "").isalnum()
    )
    if not ok:
        raise FetchError(f"버전 ID 형식 오류: {version!r} (d{{YYYYMMDD}}-{{tag}}, 예 d20260910-s2325)")


def season_relpath(season: int, holdout: bool) -> Path:
    """버전 디렉터리 기준 상대 경로. 홀드아웃은 holdout_{season}/ 아래."""
    name = f"statcast_{season}.parquet"
    return Path(f"holdout_{season}") / name if holdout else Path(name)


# ---------------------------------------------------------------- 날짜 청크


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """[start, end] 를 달력 월 단위로 자른다 (양끝 포함, 겹침 없음). start > end 면 []."""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def data_ceiling(today: date) -> date:
    """실행일 기준 완료된 마지막 날."""
    return today - timedelta(days=CEILING_LAG_DAYS)


def season_range(season: int, ceiling: date) -> tuple[date, date]:
    """수집 구간 = [시즌 시작, min(시즌 종료, ceiling)]. 구간이 비면 오류."""
    if season not in SEASON_DATES:
        raise FetchError(f"{season}: SEASON_DATES 에 없는 시즌 (있는 시즌: {sorted(SEASON_DATES)})")
    start, end = SEASON_DATES[season]
    end = min(end, ceiling)
    if end < start:
        raise FetchError(f"{season}: 수집 구간이 비었음 (시즌 시작 {start} > 상한 {ceiling})")
    return start, end


# ---------------------------------------------------------------- dtype 고정


def target_type(name: str, arrow_type: pa.DataType, all_null: bool = False, final: bool = True) -> pa.DataType:
    """컬럼 이름·원 타입 → 저장 타입. pandas 버전과 무관하게 같은 스키마가 나오게 한다.
    final=False(청크 단계)에서 이름 정책이 없는 전부-결측 컬럼은 null 로 미룬다."""
    if name == DATE_COLUMN:
        return pa.timestamp("ns")
    if name in INT64_COLUMNS:
        return pa.int64()
    if name in STRING_COLUMNS:
        return pa.string()
    t = arrow_type
    if (all_null or pa.types.is_null(t)) and not final:
        return pa.null()
    if pa.types.is_null(t) or pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t):
        return pa.float64()
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pa.string()
    raise DtypeError(f"컬럼 {name!r}: 예상 밖 타입 {t} (target_type 정책에 추가 필요)")


def cast_policy(table: pa.Table, *, final: bool = True) -> pa.Table:
    """target_type 정책으로 캐스팅. 정수 컬럼에 소수값이 있으면 실패한다 (safe cast)."""
    fields, columns = [], []
    for f in table.schema:
        col = table[f.name]
        t = target_type(f.name, f.type, col.null_count == table.num_rows, final)
        if pa.types.is_null(t):
            new = pa.nulls(table.num_rows, pa.null())  # Arrow 는 int64 → null 캐스팅을 지원하지 않아 직접 만든다
        else:
            try:
                new = col.cast(t, safe=True)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
                raise DtypeError(f"dtype 고정 실패 (컬럼 {f.name!r} → {t}): {exc}") from exc
        fields.append(pa.field(f.name, t))
        columns.append(new)
    return pa.Table.from_arrays(columns, schema=pa.schema(fields))


def normalize(raw: Optional[pd.DataFrame]) -> Optional[pa.Table]:
    """game_type == R 필터 → game_date 정규화 → 정책 스키마로 캐스팅 → SORT_KEYS 정렬. 정규시즌 행이 없으면 None."""
    if raw is None or len(raw) == 0:
        return None
    missing = [c for c in (GAME_TYPE_COLUMN, DATE_COLUMN, *SORT_KEYS) if c not in raw.columns]
    if missing:
        raise SchemaError(f"예상 컬럼 없음: {missing}. Savant 응답이 깨졌거나 스키마가 바뀜")
    mask = raw[GAME_TYPE_COLUMN].eq(GAME_TYPE).fillna(False).astype(bool).to_numpy()
    df = raw.loc[mask].copy().reset_index(drop=True)
    if len(df) == 0:
        return None
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN]).dt.normalize()
    table = pa.Table.from_pandas(df, preserve_index=False)
    return sort_table(cast_policy(table, final=False))


def sort_table(table: pa.Table) -> pa.Table:
    return table.sort_by([(k, "ascending") for k in SORT_KEYS]).combine_chunks()


def count_duplicate_keys(table: pa.Table) -> int:
    """SORT_KEYS 가 같은 행 수 (원본 그대로 두고 보고만 한다 — 중복 제거는 전처리 단계 일)."""
    keys = table.select(list(SORT_KEYS)).to_pandas()
    return int(keys.duplicated().sum())


def assemble(tables: list[pa.Table]) -> pa.Table:
    """청크 테이블을 합쳐 스키마를 최종 고정하고 정렬한다. 청크에 없던 컬럼·미룬(null) 컬럼은 다른 청크 타입을 따른다."""
    try:
        combined = pa.concat_tables(tables, promote_options="permissive")
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise DtypeError(
            "청크 스키마 병합 실패 (STRING_COLUMNS/INT64_COLUMNS 정책 추가 필요. "
            f"_chunks/ 가 다른 코드 버전으로 만들어졌다면 지우고 재실행): {exc}"
        ) from exc
    return sort_table(cast_policy(combined, final=True))


# ---------------------------------------------------------------- 청크 수집


def _install_request_timeout(timeout: tuple[float, float] = REQUEST_TIMEOUT_SECONDS) -> None:
    """pybaseball.datasources.statcast 가 쓰는 requests.get 에 타임아웃을 강제한다 (그 모듈 네임스페이스만 교체).
    pybaseball 은 timeout=None 을 명시해서 넘기므로 setdefault 가 아니라 덮어써야 한다. 멱등."""
    import requests

    import pybaseball.datasources.statcast as ds

    if getattr(ds.requests, "_pitcheezy_timeout", None) == timeout:
        return

    def get(url: str, **kwargs):  # noqa: ANN202 - requests.Response
        kwargs["timeout"] = timeout
        return requests.get(url, **kwargs)

    ds.requests = types.SimpleNamespace(get=get, _pitcheezy_timeout=timeout)


def default_fetcher(start: str, end: str) -> Optional[pd.DataFrame]:
    from pybaseball import cache, statcast  # 지연 import: 테스트는 pybaseball 없이 돈다

    _install_request_timeout()
    # pybaseball 의 디스크 캐시(~/.pybaseball, 최대 365일)가 켜져 있으면 옛 응답이 그대로 나온다.
    # 이 프로세스에서만 끈다 (cache.disable() 은 사용자 설정 파일을 덮어쓰므로 쓰지 않음). 재개는 _chunks/ 가 맡는다.
    if cache.config.enabled:
        cache.config.enabled = False
        log("    pybaseball 디스크 캐시가 켜져 있어 이 실행에서만 끔 (항상 Savant 에서 새로 받음)")
    return statcast(start_dt=start, end_dt=end, verbose=False)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, DtypeError):
        return False
    if isinstance(exc, (OSError, SchemaError, pd.errors.ParserError, pd.errors.EmptyDataError)):
        return True
    try:
        import requests

        if isinstance(exc, requests.RequestException):
            return True
    except ImportError:
        pass
    return type(exc).__name__ == "StatcastException"  # pybaseball.statcast.StatcastException (import 비용 회피)


def fetch_chunk(
    start: date,
    end: date,
    fetcher: Optional[Fetcher] = None,
    retries: int = RETRIES,
    backoff: float = BACKOFF_SECONDS,
) -> tuple[Optional[pa.Table], int]:
    """한 청크를 받아 (정규시즌 테이블 또는 None, 받은 원본 행 수). 네트워크·파싱 오류는 지수 백오프로 재시도."""
    fetcher = fetcher or default_fetcher
    for attempt in range(retries + 1):
        try:
            raw = fetcher(start.isoformat(), end.isoformat())
            rows_raw = 0 if raw is None else int(len(raw))
            return normalize(raw), rows_raw
        except Exception as exc:  # noqa: BLE001 - 재시도 여부는 _is_retryable 이 결정
            if not _is_retryable(exc) or attempt >= retries:
                raise
            wait = backoff * (2**attempt)
            log(f"    재시도 {attempt + 1}/{retries}: {type(exc).__name__}: {exc} ({wait:g}s 대기)")
            time.sleep(wait)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- 시즌 수집


@dataclass
class SeasonResult:
    season: int
    holdout: bool
    file: str  # 버전 디렉터리 기준 상대 경로
    start: str
    end: str
    rows: int
    columns: int
    rows_raw: int  # 필터 전 원본 행 수 (재사용한 최종본이면 -1)
    duplicate_keys: int
    sha256: str
    chunks_total: int
    chunks_fetched: int
    fetch_seconds: float  # 이번 실행에서 실제 요청에 쓴 시간 합
    reused: bool  # 최종본이 이미 있어 재수집·재조립하지 않음
    frozen: bool


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(
        table.replace_schema_metadata(None),
        tmp,
        compression=PARQUET_COMPRESSION,
        version=PARQUET_VERSION,
        row_group_size=ROW_GROUP_SIZE,
    )
    tmp.replace(path)


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_season(
    season: int,
    version_dir: Path,
    start: date,
    end: date,
    *,
    holdout: bool,
    frozen: bool,
    fetcher: Optional[Fetcher] = None,
    dry_run: bool = False,
    use_cache: bool = True,
) -> SeasonResult:
    """월 청크로 받아 _chunks/ 에 저장 → 전부 끝나면 시즌 parquet 조립. 재실행 시 완료 청크는 건너뜀.
    dry_run 이면 시즌 첫 하루만 받는다."""
    rel = season_relpath(season, holdout)
    final = version_dir / rel
    tag = f"[{season}{' holdout' if holdout else ''}]"
    chunks = [(start, start)] if dry_run else month_chunks(start, end)
    end = start if dry_run else end

    if use_cache and final.exists():
        meta = pq.read_metadata(final)
        log(f"{tag} 최종본 있음 → 재사용 {meta.num_rows:,}행 (재수집하려면 삭제): {final}")
        return SeasonResult(
            season, holdout, rel.as_posix(), start.isoformat(), end.isoformat(), meta.num_rows, meta.num_columns,
            -1, count_duplicate_keys(pq.read_table(final, columns=list(SORT_KEYS))), sha256_file(final),
            len(chunks), 0, 0.0, True, frozen,
        )

    chunks_dir = version_dir / CHUNKS_DIR / str(season)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pa.Table] = []
    fetched = 0
    rows_raw_total = 0
    seconds = 0.0
    for chunk_start, chunk_end in chunks:
        stem = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}"
        part = chunks_dir / f"{stem}.parquet"
        done = chunks_dir / f"{stem}.done"
        if use_cache and done.exists():
            info = json.loads(done.read_text(encoding="utf-8"))
            if info.get("rows", 0) == 0:
                log(f"{tag} {chunk_start}..{chunk_end}  완료 청크 건너뜀 (0행)")
                rows_raw_total += int(info.get("rows_raw", 0))
                continue
            if part.exists():
                log(f"{tag} {chunk_start}..{chunk_end}  완료 청크 건너뜀 ({info['rows']:,}행)")
                tables.append(pq.read_table(part))
                rows_raw_total += int(info.get("rows_raw", 0))
                continue
            log(f"{tag} {chunk_start}..{chunk_end}  마커만 있고 parquet 없음 → 재수집")
        t0 = time.monotonic()
        table, rows_raw = fetch_chunk(chunk_start, chunk_end, fetcher=fetcher)
        elapsed = time.monotonic() - t0
        rows = 0 if table is None else table.num_rows
        if table is not None:
            _write_parquet(table, part)
            tables.append(table)
        done.write_text(
            json.dumps(
                {
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                    "rows_raw": rows_raw,
                    "rows": rows,
                    "seconds": round(elapsed, 1),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
            ),
            encoding="utf-8",
        )
        fetched += 1
        rows_raw_total += rows_raw
        seconds += elapsed
        log(
            f"{tag} {chunk_start}..{chunk_end}  받음 {rows_raw:,}행 → R {rows:,}행"
            f"  ({elapsed:.1f}s){'' if rows else '  (빈 청크)'}"
        )

    if not tables:
        raise FetchError(f"{tag} 정규시즌 행이 0 — {start}..{end} 에 경기가 없거나 SEASON_DATES 가 틀림")
    combined = assemble(tables)
    dup = count_duplicate_keys(combined)
    if dup:
        log(f"{tag} 경고: SORT_KEYS 중복 {dup:,}행 (원본 그대로 저장, 전처리에서 처리)")
    _write_parquet(combined, final)
    log(f"{tag} 조립 완료 {combined.num_rows:,}행 × {combined.num_columns}컬럼 → {final}")
    return SeasonResult(
        season, holdout, rel.as_posix(), start.isoformat(), end.isoformat(), combined.num_rows,
        combined.num_columns, rows_raw_total, dup, sha256_file(final), len(chunks), fetched,
        round(seconds, 1), False, frozen,
    )


# ---------------------------------------------------------------- 장부·manifest


@dataclass
class LedgerRow:
    version: str
    season: int
    file: str
    start: str
    end: str
    rows: int
    sha256: str
    pybaseball_version: str
    commit: str
    frozen: bool
    pyarrow_version: str = ""  # 구형(9열) 행은 빈 값
    ceiling: str = ""  # 수집 상한일 YYYY-MM-DD (data_ceiling). 구형 행은 빈 값

    def render(self) -> str:
        cells = [
            self.version, str(self.season), self.file, f"{self.start}~{self.end}", str(self.rows),
            self.sha256, self.pybaseball_version, self.commit, "true" if self.frozen else "false",
            self.pyarrow_version, self.ceiling,
        ]
        return "| " + " | ".join(cells) + " |"

    @classmethod
    def parse(cls, line: str) -> Optional["LedgerRow"]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) not in (LEDGER_LEGACY_COLUMN_COUNT, len(LEDGER_COLUMNS)):
            return None
        if not cells[0].startswith("d") or not cells[1].isdigit():
            return None
        start, _, end = cells[3].partition("~")
        extra = cells[LEDGER_LEGACY_COLUMN_COUNT:] + [""] * (len(LEDGER_COLUMNS) - len(cells))
        return cls(
            cells[0], int(cells[1]), cells[2], start, end, int(cells[4].replace(",", "")),
            cells[5], cells[6], cells[7], cells[8].lower() == "true", extra[0], extra[1],
        )


def ledger_rows(ledger: Path) -> list[LedgerRow]:
    if not ledger.exists():
        return []
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        row = LedgerRow.parse(line)
        if row is not None:
            rows.append(row)
    return rows


def ledger_has_version(ledger: Path, version: str) -> bool:
    return any(r.version == version for r in ledger_rows(ledger))


def append_ledger(ledger: Path, rows: list[LedgerRow], *, allow_existing: bool = False) -> None:
    """행을 append 한다 (기존 행은 건드리지 않음). 같은 버전 ID 가 이미 있으면 오류.
    파일이 없으면 헤더부터 만든다. 헤더가 예상과 다르면 오류 (열이 어긋난 행을 쓰지 않기 위해)."""
    if not rows:
        return
    version = rows[0].version
    if not allow_existing and ledger_has_version(ledger, version):
        raise FetchError(f"장부에 {version} 이 이미 있음: {ledger} (기존 행은 덮어쓰지 않는다. 다른 태그를 쓸 것)")
    if ledger.exists():
        text = ledger.read_text(encoding="utf-8")
        if LEDGER_HEADER_LINE not in text.splitlines():
            raise FetchError(f"장부 헤더가 예상과 다름: {ledger}\n기대: {LEDGER_HEADER_LINE}")
        if text and not text.endswith("\n"):
            text += "\n"
    else:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        text = f"# 데이터 버전\n\n{LEDGER_HEADER_LINE}\n{LEDGER_SEPARATOR_LINE}\n"
    ledger.write_text(text + "".join(r.render() + "\n" for r in rows), encoding="utf-8")


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


DIRTY_IGNORED_PATHS = frozenset({"data/versions.md"})  # 장부는 수집이 직접 쓰는 파일이라 코드 변경으로 보지 않음


def git_commit(repo_root: Path) -> str:
    """짧은 HEAD 해시. 워킹트리에 변경(추적·미추적)이 있으면 -dirty 접미 (DIRTY_IGNORED_PATHS 제외). git 없으면 unknown."""

    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout

    try:
        head = run("rev-parse", "--short", "HEAD").strip()
        # porcelain v1: "XY path" — 첫 두 글자가 상태(공백 포함)라 strip 하면 안 된다. 경로는 레포 루트 기준(cwd=repo_root),
        # 이름 변경은 "old -> new".
        changed = [line[3:].split(" -> ")[-1] for line in run("status", "--porcelain").splitlines() if len(line) > 3]
        dirty = any(p not in DIRTY_IGNORED_PATHS for p in changed)
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "pybaseball": _pkg_version("pybaseball"),
    }


def write_manifest(
    version_dir: Path,
    version: str,
    results: list[SeasonResult],
    fetch_commit: str,
    *,
    today: date,
    ceiling: date,
    dry_run: bool,
) -> Path:
    manifest = {
        "version": version,
        "dry_run": dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "today": today.isoformat(),  # 버전 ID 의 날짜 (--date 또는 실행일)
        "data_ceiling": ceiling.isoformat(),  # 수집 상한일 = today − CEILING_LAG_DAYS. 장부의 "수집 상한일" 과 같음
        "fetch_commit": fetch_commit,
        "environment": environment(),  # pyarrow 항목이 장부의 "pyarrow 버전" 과 같음
        "parameters": {
            "season_dates": {str(s): [a.isoformat(), b.isoformat()] for s, (a, b) in SEASON_DATES.items()},
            "game_type": GAME_TYPE,
            "chunking": "calendar-month",
            "sort_keys": list(SORT_KEYS),
            "int64_columns": sorted(INT64_COLUMNS),
            "string_columns": sorted(STRING_COLUMNS),
            "parquet": {
                "compression": PARQUET_COMPRESSION,
                "version": PARQUET_VERSION,
                "row_group_size": ROW_GROUP_SIZE,
            },
            "note": "pybaseball statcast() 검색 URL 은 R|PO|S 고정 → 후처리로 game_type=R 만 남김. "
            "sha256 은 같은 pyarrow 버전에서 재현된다 (parquet footer 에 writer 버전이 들어감).",
        },
        "seasons": {str(r.season): asdict(r) for r in results},
    }
    path = version_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- 검증


def verify(version: str, out_dir: Path, ledger: Path) -> list[str]:
    """장부의 버전 행마다 파일 존재·행 수·sha256 을 대조. 문제 목록을 돌려준다 (비면 OK)."""
    rows = [r for r in ledger_rows(ledger) if r.version == version]
    if not rows:
        return [f"장부에 {version} 행 없음: {ledger}"]
    problems: list[str] = []
    for r in rows:
        path = out_dir / version / r.file
        if not path.exists():
            problems.append(f"{r.season}: 파일 없음 {path}")
            continue
        try:
            n = pq.read_metadata(path).num_rows
        except Exception as exc:  # noqa: BLE001 - 깨진 parquet
            problems.append(f"{r.season}: parquet 읽기 실패 {type(exc).__name__}: {exc}")
            continue
        sha = sha256_file(path)
        ok = True
        if n != r.rows:
            problems.append(f"{r.season}: 행 수 불일치 장부 {r.rows:,} vs 파일 {n:,}")
            ok = False
        if sha != r.sha256:
            problems.append(f"{r.season}: sha256 불일치 장부 {r.sha256[:16]}… vs 파일 {sha[:16]}…")
            ok = False
        log(f"[{r.season}] {'OK' if ok else 'FAIL'} {r.file} rows={n:,} sha256={sha}")
    return problems


# ---------------------------------------------------------------- 수집 진입점


@dataclass
class RunResult:
    version: str
    version_dir: Path
    ledger: Path
    manifest: Path
    results: list[SeasonResult]
    rows: list[LedgerRow]


def run_fetch(
    *,
    version: str,
    seasons: list[int],
    holdouts: list[int],
    out_dir: Path,
    ledger: Path,
    repo_root: Path,
    today: date,
    dry_run: bool = False,
    freeze: bool = False,
    fetcher: Optional[Fetcher] = None,
) -> RunResult:
    """학습 시즌(seasons) + 홀드아웃 시즌(holdouts) 수집 → manifest → 장부.
    dry_run 은 {out_dir}/_dryrun/{version}/ 에 쓰고 장부도 그 안의 versions.md 에 쓴다 (실제 장부 불변)."""
    check_version_id(version)
    if not seasons and not holdouts:
        raise FetchError("수집할 시즌이 없음 (--seasons 또는 --holdout)")
    overlap = sorted(set(seasons) & set(holdouts))
    if overlap:
        raise FetchError(f"학습 시즌과 홀드아웃이 겹침: {overlap}")
    if len(set(seasons)) != len(seasons) or len(set(holdouts)) != len(holdouts):
        raise FetchError("시즌이 중복 지정됨")
    ceiling = data_ceiling(today)
    for season in seasons:
        if season not in SEASON_DATES:
            raise FetchError(f"{season}: SEASON_DATES 에 없는 시즌")
        if season in HOLDOUT_SEASONS:
            raise FetchError(
                f"{season}: 홀드아웃 전용 시즌 (OPE·분해 전용, 학습 시즌으로 받지 않는다) → --holdout {season}"
            )
    for season in holdouts:
        if season not in SEASON_DATES:
            raise FetchError(f"{season}: SEASON_DATES 에 없는 시즌")
        season_end = SEASON_DATES[season][1]
        if freeze and ceiling < season_end:
            raise FetchError(
                f"{season}: --freeze 는 수집 상한(실행일 − {CEILING_LAG_DAYS}일 = {ceiling})이 "
                f"정규시즌 종료일({season_end}) 이상일 때만 가능"
            )

    if dry_run:
        version_dir = out_dir / DRYRUN_DIR / version
        ledger = version_dir / "versions.md"
    else:
        version_dir = out_dir / version
        if ledger_has_version(ledger, version):
            raise FetchError(f"장부에 {version} 이 이미 있음: {ledger} (기존 행은 덮어쓰지 않는다. 다른 태그를 쓸 것)")
    version_dir.mkdir(parents=True, exist_ok=True)

    plan = [(s, False, True) for s in seasons] + [(s, True, freeze) for s in holdouts]
    log(f"버전 {version} → {version_dir}{'  [dry-run: 시즌 첫 하루만]' if dry_run else ''}")
    log(f"기준일 {today} (상한 {ceiling}) | game_type={GAME_TYPE} | 정렬 {SORT_KEYS} | 청크 월 단위")
    log("환경 " + ", ".join(f"{k} {v}" for k, v in environment().items()))
    results: list[SeasonResult] = []
    for season, holdout, frozen in plan:
        start, end = season_range(season, ceiling)
        log(f"[{season}{' holdout' if holdout else ''}] 구간 {start}..{end}")
        results.append(
            fetch_season(
                season, version_dir, start, end, holdout=holdout, frozen=frozen,
                fetcher=fetcher, dry_run=dry_run, use_cache=not dry_run,
            )
        )

    fetch_commit = git_commit(repo_root)
    manifest = write_manifest(
        version_dir, version, results, fetch_commit, today=today, ceiling=ceiling, dry_run=dry_run
    )
    env = environment()
    rows = [
        LedgerRow(
            version, r.season, r.file, r.start, r.end, r.rows, r.sha256, env["pybaseball"], fetch_commit, r.frozen,
            env["pyarrow"], ceiling.isoformat(),
        )
        for r in results
    ]
    append_ledger(ledger, rows, allow_existing=dry_run)
    log(f"manifest → {manifest}")
    log(f"장부 {len(rows)}행 추가 → {ledger}")
    for r in rows:
        log(r.render())
    return RunResult(version, version_dir, ledger, manifest, results, rows)
