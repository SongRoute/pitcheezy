"""statcast_fetch 단위 테스트. 네트워크 없음 — pybaseball.statcast() 호출은 가짜 fetcher 로 대체.

가짜 프레임은 pybaseball 2.2.7 이 돌려주는 두 가지 모양을 흉내낸다.
    pandas 3: 문자열 str, game_date str, 숫자는 convert_dtypes 로 Int64/Float64 (전부 결측이면 Int64)
    pandas 2: 문자열 object, game_date datetime64, 숫자는 Int64/Float64
어느 쪽이든 같은 스키마·같은 sha256 이 나와야 한다 (dtype 고정).
"""

from __future__ import annotations

import random
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pitcheezy.data import statcast_fetch as sf

TODAY = date(2026, 9, 10)  # 2024·2025 는 끝난 시즌, 2026 은 진행 중
STRING_COLUMNS = ("pitch_type", "player_name", "events", "des", "game_type", "sv_id")


def pitch_rows(day, pitcher, game_pk, n_pa, per_pa, game_type="R", bat_speed=71.5, zone=5):
    rows = []
    for ab in range(1, n_pa + 1):
        for pn in range(1, per_pa + 1):
            rows.append(
                {
                    "pitch_type": "FF",
                    "game_date": day.isoformat(),
                    "release_speed": 95.0 + pn / 10,
                    "player_name": f"P{pitcher}",
                    "batter": 600000 + ab,
                    "pitcher": pitcher,
                    "events": "strikeout" if pn == per_pa else None,
                    "des": "Ball" if pn % 2 else "Called Strike",
                    "game_type": game_type,
                    "zone": zone,
                    "hit_location": None,  # 전부 결측 → pandas 3 에서 Int64, 우리 정책은 int64 (INT64_COLUMNS)
                    "umpire": None,  # 전부 결측, 정책 밖 → float64
                    "on_1b": 500000 if ab % 2 else None,
                    "game_pk": game_pk,
                    "at_bat_number": ab,
                    "pitch_number": pn,
                    "sv_id": f"{day:%y%m%d}_{ab:03d}{pn:02d}",
                    "bat_speed": bat_speed,
                    "delta_run_exp": -0.03 * pn,
                }
            )
    return rows


def frame(rows, style="pandas3", shuffle=False):
    """pybaseball.statcast() 반환 모양. 기본은 내림차순 정렬(pybaseball 이 그렇게 돌려줌)."""
    if shuffle:
        rows = list(rows)
        random.Random(0).shuffle(rows)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].isna().all():
            df[c] = df[c].astype("float64")  # read_csv 는 빈 컬럼을 float64 NaN 으로 읽는다 → convert_dtypes 가 Int64 로
    if style == "pandas2":
        for c in STRING_COLUMNS:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
        df["game_date"] = pd.to_datetime(df["game_date"])
    if not shuffle:
        df = df.sort_values(list(sf.SORT_KEYS), ascending=False)
    return df.convert_dtypes(convert_string=False)


def one_chunk_fetcher(rows_by_start):
    """{start iso: rows} → fetcher. 요청받은 (start, end) 를 기록한다."""
    calls = []

    def fetcher(start, end):
        calls.append((start, end))
        rows = rows_by_start.get(start)
        return None if rows is None else frame(rows)

    fetcher.calls = calls
    return fetcher


# ---------------------------------------------------------------- 버전·날짜


def test_version_id_and_format():
    assert sf.version_id("s2325", date(2026, 9, 10)) == "d20260910-s2325"
    sf.check_version_id("d20260910-s2325")
    for bad in ("s2325", "d2026091-s2325", "d20260910s2325", "d20260910-", "d20260910-a b"):
        with pytest.raises(sf.FetchError):
            sf.check_version_id(bad)
    with pytest.raises(sf.FetchError):
        sf.version_id("", TODAY)


def test_month_chunks_cover_range_without_overlap():
    chunks = sf.month_chunks(date(2024, 3, 20), date(2024, 9, 30))
    assert chunks[0] == (date(2024, 3, 20), date(2024, 3, 31))
    assert chunks[1] == (date(2024, 4, 1), date(2024, 4, 30))
    assert chunks[-1] == (date(2024, 9, 1), date(2024, 9, 30))
    assert len(chunks) == 7
    for (s1, e1), (s2, _) in zip(chunks, chunks[1:]):
        assert s1 <= e1 and (s2 - e1).days == 1  # 양끝 포함, 겹침·틈 없음
    assert sf.month_chunks(date(2025, 4, 1), date(2025, 4, 1)) == [(date(2025, 4, 1), date(2025, 4, 1))]
    assert sf.month_chunks(date(2025, 5, 1), date(2025, 4, 1)) == []
    # 12월 넘어가는 계산
    assert sf.month_chunks(date(2025, 12, 20), date(2026, 1, 3)) == [
        (date(2025, 12, 20), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 1, 3)),
    ]


def test_season_range_clips_to_ceiling():
    assert sf.season_range(2024, ceiling=date(2026, 9, 8)) == sf.SEASON_DATES[2024]
    assert sf.season_range(2026, ceiling=date(2026, 9, 8)) == (date(2026, 3, 25), date(2026, 9, 8))
    with pytest.raises(sf.FetchError, match="비었음"):
        sf.season_range(2026, ceiling=date(2026, 3, 1))
    with pytest.raises(sf.FetchError, match="없는 시즌"):
        sf.season_range(2022, ceiling=TODAY)
    assert sf.data_ceiling(date(2026, 9, 10)) == date(2026, 9, 8)


def test_season_dates_inside_pybaseball_window():
    # pybaseball 2.2.7 은 2021+ 시즌에서 3/15~11/15 밖 날짜를 요청 없이 건너뛴다
    for season, (start, end) in sf.SEASON_DATES.items():
        assert date(season, 3, 15) <= start <= end <= date(season, 11, 15)


# ---------------------------------------------------------------- dtype 고정·정렬


def test_normalize_filters_game_type_and_fixes_schema():
    day = date(2025, 4, 1)
    rows = pitch_rows(day, 1, 100, 2, 3) + pitch_rows(day, 1, 101, 1, 2, game_type="S") + pitch_rows(
        day, 1, 102, 1, 2, game_type="F"
    )
    table = sf.normalize(frame(rows))
    assert table.num_rows == 6
    assert set(table["game_type"].to_pylist()) == {"R"}
    schema = table.schema
    assert schema.field("game_date").type == pa.timestamp("ns")
    for c in ("game_pk", "at_bat_number", "pitch_number", "batter", "pitcher", "zone", "hit_location", "on_1b"):
        assert schema.field(c).type == pa.int64(), c
    for c in ("release_speed", "bat_speed", "delta_run_exp"):
        assert schema.field(c).type == pa.float64(), c
    assert schema.field("umpire").type == pa.null()  # 정책 없음 + 전부 결측 → 청크 단계에서는 미룸
    assert sf.assemble([table]).schema.field("umpire").type == pa.float64()  # 최종은 float64
    for c in STRING_COLUMNS:
        assert schema.field(c).type == pa.string(), c
    assert schema.metadata is None
    keys = list(zip(*(table[k].to_pylist() for k in sf.SORT_KEYS)))
    assert keys == sorted(keys)
    assert table["hit_location"].null_count == 6 and table["on_1b"].null_count == 3


def test_normalize_edge_cases():
    day = date(2025, 4, 1)
    assert sf.normalize(None) is None
    assert sf.normalize(pd.DataFrame()) is None
    assert sf.normalize(frame(pitch_rows(day, 1, 100, 1, 2, game_type="S"))) is None
    with pytest.raises(sf.SchemaError, match="예상 컬럼 없음"):
        sf.normalize(pd.DataFrame({"x": [1]}))
    with pytest.raises(sf.DtypeError, match="dtype 고정 실패"):
        sf.normalize(frame(pitch_rows(day, 1, 100, 1, 2, zone=1.5)))  # 정수 컬럼에 소수
    assert not sf._is_retryable(sf.DtypeError("x")) and sf._is_retryable(sf.SchemaError("x"))


def test_schema_and_sha256_identical_across_pandas_styles_and_row_order(tmp_path):
    day = date(2025, 4, 1)
    rows = pitch_rows(day, 7, 100, 4, 5) + pitch_rows(day, 8, 101, 3, 4)
    variants = {
        "pandas3": frame(rows, "pandas3"),
        "pandas2": frame(rows, "pandas2"),
        "shuffled": frame(rows, "pandas3", shuffle=True),
        "pandas2-shuffled": frame(rows, "pandas2", shuffle=True),
    }
    hashes = {}
    schemas = {}
    for name, raw in variants.items():
        table = sf.normalize(raw)
        path = tmp_path / f"{name}.parquet"
        sf._write_parquet(table, path)
        hashes[name] = sf.sha256_file(path)
        schemas[name] = pq.read_schema(path)
    assert len(set(hashes.values())) == 1, hashes
    assert all(s.equals(schemas["pandas3"]) for s in schemas.values())
    # 다시 써도 같다
    sf._write_parquet(sf.normalize(variants["pandas3"]), tmp_path / "again.parquet")
    assert sf.sha256_file(tmp_path / "again.parquet") == hashes["pandas3"]


def test_assemble_fills_missing_column_and_resorts():
    day1, day2 = date(2025, 4, 2), date(2025, 4, 15)
    t_late = sf.normalize(frame(pitch_rows(day2, 1, 101, 1, 3)))
    t_early = sf.normalize(frame(pitch_rows(day1, 1, 100, 2, 2))).drop_columns(["bat_speed"])
    combined = sf.assemble([t_late, t_early])
    assert combined.num_rows == 7
    assert combined.schema.field("bat_speed").type == pa.float64()
    assert combined["bat_speed"].null_count == 4
    assert combined["game_pk"].to_pylist() == [100, 100, 100, 100, 101, 101, 101]


def test_all_null_columns_do_not_collide_across_chunks():
    """한 청크에서 전부 결측인 컬럼: 이름 정책(STRING/INT64)이 있으면 그 타입, 없으면 미뤄서(null) 조립 때 맞춘다."""
    day1, day2 = date(2025, 4, 2), date(2025, 5, 2)
    rows_a = pitch_rows(day1, 1, 100, 1, 2)
    rows_b = pitch_rows(day2, 1, 101, 1, 2)
    for r in rows_a:
        r["sv_id"] = None  # 이름 정책 있음 (STRING_COLUMNS)
        r["new_text_col"] = None  # 이름 정책 없음 → 청크에서는 null 로 미룸
        r["new_num_col"] = None
    for r in rows_b:
        r["new_text_col"] = "x"
        r["new_num_col"] = 1.25
    raw_a = frame(rows_a)
    assert str(raw_a["new_num_col"].dtype) == "Int64"  # pandas 3 에서 전부 결측 → Int64 (실제 pybaseball 출력과 동일)
    t_a, t_b = sf.normalize(raw_a), sf.normalize(frame(rows_b))
    assert t_a.schema.field("sv_id").type == pa.string() and t_a["sv_id"].null_count == 2
    assert t_a.schema.field("new_text_col").type == pa.null() and t_a.schema.field("new_num_col").type == pa.null()
    assert t_a.schema.field("umpire").type == pa.null()  # 문서상 deprecated, 정책 없음, 전부 결측
    combined = sf.assemble([t_a, t_b])
    assert combined.schema.field("sv_id").type == pa.string()
    assert combined.schema.field("new_text_col").type == pa.string()
    assert combined.schema.field("new_num_col").type == pa.float64()
    assert combined.schema.field("umpire").type == pa.float64()  # 끝까지 결측 → float64
    assert combined["new_text_col"].to_pylist() == [None, None, "x", "x"]
    # 이름 정책이 없는 컬럼이 청크마다 다른 타입이면 명확히 실패
    rows_c = [dict(r, new_text_col=2.0) for r in rows_b]
    with pytest.raises(sf.DtypeError, match="병합 실패"):
        sf.assemble([t_b, sf.normalize(frame(rows_c))])


def test_count_duplicate_keys():
    day = date(2025, 4, 1)
    rows = pitch_rows(day, 1, 100, 1, 2)
    table = sf.normalize(frame(rows + rows[:1]))
    assert sf.count_duplicate_keys(table) == 1


# ---------------------------------------------------------------- 청크 수집·재시도


def test_fetch_chunk_retries_transient_errors_only():
    day = date(2025, 4, 1)
    calls = []

    def flaky(start, end):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return frame(pitch_rows(day, 1, 100, 1, 1))

    table, rows_raw = sf.fetch_chunk(day, day, fetcher=flaky, retries=3, backoff=0)
    assert len(calls) == 3 and table.num_rows == 1 and rows_raw == 1

    def down(start, end):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        sf.fetch_chunk(day, day, fetcher=down, retries=2, backoff=0)

    def bug(start, end):
        raise ValueError("programming error")

    with pytest.raises(ValueError):  # 재시도 대상 아님 → 즉시 전파
        sf.fetch_chunk(day, day, fetcher=bug, retries=3, backoff=0)

    garbage = [0]

    def html_then_ok(start, end):  # 에러 페이지 → 예상 컬럼 없음 → 재시도
        garbage[0] += 1
        return pd.DataFrame({"<html>": ["error"]}) if garbage[0] == 1 else frame(pitch_rows(day, 1, 100, 1, 1))

    table, _ = sf.fetch_chunk(day, day, fetcher=html_then_ok, retries=1, backoff=0)
    assert table.num_rows == 1


def test_request_timeout_is_forced_on_pybaseball(monkeypatch):
    """pybaseball 은 requests.get(url, timeout=None) 을 쓴다 → 우리가 끼운 래퍼가 timeout 을 덮어써야 한다."""
    pytest.importorskip("pybaseball")
    import requests

    import pybaseball.datasources.statcast as ds

    seen = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        raise requests.ConnectionError("stop here")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(ds, "requests", requests)  # 깨끗한 상태에서 시작
    sf._install_request_timeout((1.0, 2.0))
    sf._install_request_timeout((1.0, 2.0))  # 멱등
    with pytest.raises(requests.ConnectionError):
        ds.requests.get("https://example.invalid/x", timeout=None)
    assert seen["timeout"] == (1.0, 2.0)
    assert sf._is_retryable(requests.exceptions.ReadTimeout("t"))


def test_fetch_season_resumes_from_chunk_cache(tmp_path):
    start, end = date(2025, 4, 20), date(2025, 5, 3)  # 두 청크
    fetcher = one_chunk_fetcher(
        {"2025-04-20": pitch_rows(date(2025, 4, 20), 7, 100, 3, 4), "2025-05-01": pitch_rows(date(2025, 5, 1), 7, 101, 2, 2)}
    )
    r1 = sf.fetch_season(2025, tmp_path, start, end, holdout=False, frozen=True, fetcher=fetcher)
    assert fetcher.calls == [("2025-04-20", "2025-04-30"), ("2025-05-01", "2025-05-03")]
    assert (r1.chunks_total, r1.chunks_fetched, r1.rows, r1.rows_raw) == (2, 2, 16, 16)
    chunks_dir = tmp_path / sf.CHUNKS_DIR / "2025"
    assert (chunks_dir / "2025-04-20_2025-04-30.parquet").exists()
    assert (chunks_dir / "2025-04-20_2025-04-30.done").exists()
    assert r1.file == "statcast_2025.parquet" and (tmp_path / r1.file).exists()
    assert pq.read_metadata(tmp_path / r1.file).num_rows == 16

    def boom(start, end):
        raise AssertionError("완료 청크는 다시 요청하면 안 됨")

    # 최종본이 있으면 재사용
    r2 = sf.fetch_season(2025, tmp_path, start, end, holdout=False, frozen=True, fetcher=boom)
    assert r2.reused and r2.sha256 == r1.sha256 and r2.rows == 16 and r2.columns == r1.columns
    # 최종본만 지우면 완료 청크는 건너뛰고 재조립 → 같은 해시
    (tmp_path / r1.file).unlink()
    r3 = sf.fetch_season(2025, tmp_path, start, end, holdout=False, frozen=True, fetcher=boom)
    assert r3.chunks_fetched == 0 and r3.sha256 == r1.sha256
    # 마커만 있고 parquet 이 없으면 그 청크만 재수집
    (tmp_path / r1.file).unlink()
    (chunks_dir / "2025-05-01_2025-05-03.parquet").unlink()
    r4 = sf.fetch_season(2025, tmp_path, start, end, holdout=False, frozen=True, fetcher=fetcher)
    assert r4.chunks_fetched == 1 and fetcher.calls[-1] == ("2025-05-01", "2025-05-03") and r4.sha256 == r1.sha256
    # use_cache=False 면 전부 다시 받는다 (dry-run)
    (tmp_path / r1.file).unlink()
    n = len(fetcher.calls)
    sf.fetch_season(2025, tmp_path, start, end, holdout=False, frozen=True, fetcher=fetcher, use_cache=False)
    assert len(fetcher.calls) == n + 2


def test_fetch_season_empty_chunk_is_marked_and_zero_total_fails(tmp_path):
    fetcher = one_chunk_fetcher({"2025-04-01": pitch_rows(date(2025, 4, 1), 7, 100, 1, 2)})  # 5월 청크는 None
    r = sf.fetch_season(2025, tmp_path, date(2025, 4, 1), date(2025, 5, 2), holdout=False, frozen=True, fetcher=fetcher)
    assert r.rows == 2 and r.chunks_fetched == 2
    marker = tmp_path / sf.CHUNKS_DIR / "2025" / "2025-05-01_2025-05-02.done"
    assert marker.exists() and not marker.with_suffix(".parquet").exists()
    with pytest.raises(sf.FetchError, match="정규시즌 행이 0"):
        sf.fetch_season(2026, tmp_path, date(2026, 5, 1), date(2026, 5, 2), holdout=True, frozen=False, fetcher=fetcher)


def test_fetch_season_holdout_path_and_dry_run(tmp_path):
    fetcher = one_chunk_fetcher({"2026-03-25": pitch_rows(date(2026, 3, 25), 7, 100, 1, 2)})
    r = sf.fetch_season(
        2026, tmp_path, date(2026, 3, 25), date(2026, 9, 8), holdout=True, frozen=False, fetcher=fetcher, dry_run=True
    )
    assert fetcher.calls == [("2026-03-25", "2026-03-25")]  # 첫 하루만
    assert r.file == "holdout_2026/statcast_2026.parquet" and (tmp_path / r.file).exists()
    assert (r.start, r.end, r.frozen, r.holdout) == ("2026-03-25", "2026-03-25", False, True)


# ---------------------------------------------------------------- 장부


def test_ledger_row_roundtrip_and_append(tmp_path):
    ledger = tmp_path / "versions.md"
    row = sf.LedgerRow("d20260910-s2325", 2024, "statcast_2024.parquet", "2024-03-20", "2024-09-30", 123456, "a" * 64, "2.2.7", "abc1234", True, "25.0.1", "2026-09-08")
    rendered = row.render()
    assert rendered.count("|") == len(sf.LEDGER_COLUMNS) + 1 and rendered.endswith("| 25.0.1 | 2026-09-08 |")
    parsed = sf.LedgerRow.parse(rendered)
    assert parsed == row
    assert sf.LedgerRow.parse(sf.LEDGER_HEADER_LINE) is None and sf.LedgerRow.parse(sf.LEDGER_SEPARATOR_LINE) is None

    sf.append_ledger(ledger, [row])
    text = ledger.read_text(encoding="utf-8")
    assert sf.LEDGER_HEADER_LINE in text and text.count("d20260910-s2325") == 1
    assert sf.ledger_has_version(ledger, "d20260910-s2325")
    with pytest.raises(sf.FetchError, match="이미 있음"):
        sf.append_ledger(ledger, [row])
    row2 = sf.LedgerRow("d20260911-s2325", 2025, "statcast_2025.parquet", "2025-03-18", "2025-09-28", 1, "b" * 64, "2.2.7", "def5678", True)
    sf.append_ledger(ledger, [row2])
    assert [r.version for r in sf.ledger_rows(ledger)] == ["d20260910-s2325", "d20260911-s2325"]  # 기존 행 보존
    assert text.splitlines() == ledger.read_text(encoding="utf-8").splitlines()[: len(text.splitlines())]


def test_append_ledger_rejects_unexpected_header(tmp_path):
    ledger = tmp_path / "versions.md"
    ledger.write_text("# 데이터 버전\n\n| 버전 ID | 시즌 | 행 수 |\n|---|---|---|\n", encoding="utf-8")
    row = sf.LedgerRow("d20260910-s2325", 2024, "statcast_2024.parquet", "2024-03-20", "2024-09-30", 1, "a" * 64, "2.2.7", "abc", True)
    with pytest.raises(sf.FetchError, match="헤더"):
        sf.append_ledger(ledger, [row])


def test_ledger_parse_accepts_legacy_nine_column_rows():
    """2026-09-12 이전 행(pyarrow 버전·수집 상한일 없음)도 읽힌다. 새 열은 빈 값."""
    legacy = "| d20260911-s2325 | 2026 | holdout_2026/statcast_2026.parquet | 2026-03-25~2026-09-09 | 647896 | " + "c" * 64 + " | 2.2.7 | 9ebc6bb | false |"
    row = sf.LedgerRow.parse(legacy)
    assert row is not None
    assert (row.version, row.season, row.end, row.rows, row.commit, row.frozen) == ("d20260911-s2325", 2026, "2026-09-09", 647896, "9ebc6bb", False)
    assert row.pyarrow_version == "" and row.ceiling == ""
    assert sf.LedgerRow.parse(legacy + " x |") is None  # 10열은 어느 형식도 아님
    assert sf.LedgerRow.parse(row.render()).pyarrow_version == ""  # 다시 렌더하면 11열 (빈 칸)


def test_repo_ledger_has_expected_header_and_parses():
    repo_ledger = sf.Path(__file__).resolve().parents[2] / "data" / "versions.md"
    assert sf.LEDGER_HEADER_LINE in repo_ledger.read_text(encoding="utf-8").splitlines()
    rows = sf.ledger_rows(repo_ledger)
    assert rows, "레포 장부의 행이 하나도 파싱되지 않음"
    assert all(r.version.startswith("d") and r.sha256 and len(r.sha256) == 64 for r in rows)


def test_git_commit_ignores_ledger_changes(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("git"):
        pytest.skip("git 없음")

    def run(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "versions.md").write_text("# x\n", encoding="utf-8")
    (tmp_path / "code.py").write_text("a = 1\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    head = sf.git_commit(tmp_path)
    assert head != "unknown" and not head.endswith("-dirty")
    (tmp_path / "data" / "versions.md").write_text("# x\n| row |\n", encoding="utf-8")
    assert sf.git_commit(tmp_path) == head  # 장부 변경만으로는 dirty 아님
    (tmp_path / "code.py").write_text("a = 2\n", encoding="utf-8")
    assert sf.git_commit(tmp_path) == head + "-dirty"
    (tmp_path / "code.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("u\n", encoding="utf-8")  # 미추적 파일도 dirty
    assert sf.git_commit(tmp_path) == head + "-dirty"


# ---------------------------------------------------------------- run_fetch·verify


def full_fetcher():
    """어느 청크를 요청해도 청크 시작일 하루치 프레임을 돌려준다 (game_pk 는 날짜에서 유도해 청크 간 유일)."""
    calls = []

    def fetcher(start, end):
        calls.append((start, end))
        day = date.fromisoformat(start)
        return frame(pitch_rows(day, 7, int(start.replace("-", "")), 2, 3))

    fetcher.calls = calls
    return fetcher


def test_run_fetch_end_to_end_then_verify(tmp_path):
    out, ledger = tmp_path / "raw", tmp_path / "versions.md"
    fetcher = full_fetcher()
    run = sf.run_fetch(
        version="d20260910-s2325", seasons=[2024], holdouts=[2026], out_dir=out, ledger=ledger,
        repo_root=tmp_path, today=TODAY, fetcher=fetcher,
    )
    vdir = out / "d20260910-s2325"
    assert run.version_dir == vdir
    assert (vdir / "statcast_2024.parquet").exists()
    assert (vdir / "holdout_2026" / "statcast_2026.parquet").exists()
    assert not (vdir / "statcast_2026.parquet").exists()
    assert (vdir / sf.MANIFEST_NAME).exists()
    assert fetcher.calls[0] == ("2024-03-20", "2024-03-31") and fetcher.calls[6] == ("2024-09-01", "2024-09-30")
    assert fetcher.calls[7] == ("2026-03-25", "2026-03-31") and fetcher.calls[-1] == ("2026-09-01", "2026-09-08")

    rows = sf.ledger_rows(ledger)
    assert [(r.version, r.season, r.file, r.start, r.end, r.frozen) for r in rows] == [
        ("d20260910-s2325", 2024, "statcast_2024.parquet", "2024-03-20", "2024-09-30", True),
        ("d20260910-s2325", 2026, "holdout_2026/statcast_2026.parquet", "2026-03-25", "2026-09-08", False),
    ]
    assert rows[0].rows == 7 * 6 and rows[1].rows == 7 * 6
    assert rows[0].sha256 == sf.sha256_file(vdir / "statcast_2024.parquet")
    assert rows[0].pybaseball_version and rows[0].commit
    assert rows[0].pyarrow_version == pa.__version__ and rows[0].ceiling == "2026-09-08"
    assert rows[1].ceiling == "2026-09-08"  # 완료 시즌·홀드아웃 모두 같은 상한일 기록
    import json

    manifest = json.loads((vdir / sf.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["data_ceiling"] == "2026-09-08" and manifest["today"] == "2026-09-10"
    assert manifest["environment"]["pyarrow"] == pa.__version__

    assert sf.verify("d20260910-s2325", out, ledger) == []
    # 파일이 바뀌면 FAIL
    path = vdir / "statcast_2024.parquet"
    table = pq.read_table(path)
    sf._write_parquet(table.slice(1), path)
    problems = sf.verify("d20260910-s2325", out, ledger)
    assert len(problems) == 2 and "행 수" in problems[0] and "sha256" in problems[1]
    (vdir / "holdout_2026" / "statcast_2026.parquet").unlink()
    assert any("파일 없음" in p for p in sf.verify("d20260910-s2325", out, ledger))
    assert sf.verify("d20260101-none", out, ledger) == ["장부에 d20260101-none 행 없음: " + str(ledger)]


def test_run_fetch_rejects_existing_version_before_fetching(tmp_path):
    out, ledger = tmp_path / "raw", tmp_path / "versions.md"
    fetcher = full_fetcher()
    sf.run_fetch(version="d20260910-s2325", seasons=[2024], holdouts=[], out_dir=out, ledger=ledger, repo_root=tmp_path, today=TODAY, fetcher=fetcher)
    n = len(fetcher.calls)
    with pytest.raises(sf.FetchError, match="이미 있음"):
        sf.run_fetch(version="d20260910-s2325", seasons=[2025], holdouts=[], out_dir=out, ledger=ledger, repo_root=tmp_path, today=TODAY, fetcher=fetcher)
    assert len(fetcher.calls) == n


def test_run_fetch_guards(tmp_path):
    out, ledger = tmp_path / "raw", tmp_path / "versions.md"
    kw = dict(out_dir=out, ledger=ledger, repo_root=tmp_path, today=TODAY, fetcher=full_fetcher())
    with pytest.raises(sf.FetchError, match="홀드아웃 전용.*--holdout 2026"):  # 홀드아웃 전용 시즌은 학습 시즌으로 못 받음
        sf.run_fetch(version="d20260910-x", seasons=[2026], holdouts=[], **kw)
    with pytest.raises(sf.FetchError, match="겹침"):
        sf.run_fetch(version="d20260910-x", seasons=[2024], holdouts=[2024], **kw)
    with pytest.raises(sf.FetchError, match="중복"):
        sf.run_fetch(version="d20260910-x", seasons=[2024, 2024], holdouts=[], **kw)
    with pytest.raises(sf.FetchError, match="없음"):
        sf.run_fetch(version="d20260910-x", seasons=[], holdouts=[], **kw)
    with pytest.raises(sf.FetchError, match="없는 시즌"):
        sf.run_fetch(version="d20260910-x", seasons=[2022], holdouts=[], **kw)
    with pytest.raises(sf.FetchError, match="--freeze"):
        sf.run_fetch(version="d20260910-x", seasons=[], holdouts=[2026], freeze=True, **kw)
    with pytest.raises(sf.FetchError, match="형식 오류"):
        sf.run_fetch(version="bad", seasons=[2024], holdouts=[], **kw)
    # 시즌 종료 후에는 --freeze 가능 → frozen=true, 전체 구간
    run = sf.run_fetch(version="d20261005-x", seasons=[], holdouts=[2026], freeze=True, out_dir=out, ledger=ledger, repo_root=tmp_path, today=date(2026, 10, 5), fetcher=full_fetcher())
    assert run.rows[0].frozen is True and run.rows[0].end == "2026-09-27" and run.rows[0].ceiling == "2026-10-03"


def test_holdout_season_rejected_as_training_even_after_season_end(tmp_path):
    """2026 은 시즌 종료 여부와 무관하게 홀드아웃 전용 (CLAUDE.md: 학습·튜닝·모델 선택 금지)."""
    assert 2026 in sf.HOLDOUT_SEASONS
    kw = dict(out_dir=tmp_path / "raw", ledger=tmp_path / "versions.md", repo_root=tmp_path, fetcher=full_fetcher())
    with pytest.raises(sf.FetchError, match="홀드아웃 전용"):
        sf.run_fetch(version="d20261005-x", seasons=[2026], holdouts=[], today=date(2026, 10, 5), **kw)
    with pytest.raises(sf.FetchError, match="홀드아웃 전용"):  # 완료 시즌과 섞여 있어도 거부
        sf.run_fetch(version="d20261005-x", seasons=[2024, 2026], holdouts=[], today=date(2026, 10, 5), **kw)
    assert (tmp_path / "raw").exists() is False  # 수집 시작 전에 거부


def test_freeze_gate_uses_ceiling_not_today(tmp_path):
    """--freeze 는 today − CEILING_LAG_DAYS ≥ 시즌 종료일(2026-09-27) 일 때만 → 9/28 거부, 9/29 통과."""
    assert sf.SEASON_DATES[2026][1] == date(2026, 9, 27) and sf.CEILING_LAG_DAYS == 2
    kw = dict(out_dir=tmp_path / "raw", ledger=tmp_path / "versions.md", repo_root=tmp_path, fetcher=full_fetcher())
    with pytest.raises(sf.FetchError, match="--freeze.*2026-09-26.*2026-09-27"):
        sf.run_fetch(version="d20260928-x", seasons=[], holdouts=[2026], freeze=True, today=date(2026, 9, 28), **kw)
    assert not (tmp_path / "raw").exists()
    # --freeze 없이는 같은 날에도 수집 가능 (frozen=false, 상한 9/26 까지)
    run = sf.run_fetch(version="d20260928-y", seasons=[], holdouts=[2026], today=date(2026, 9, 28), **kw)
    assert run.rows[0].frozen is False and run.rows[0].end == "2026-09-26" and run.rows[0].ceiling == "2026-09-26"
    run = sf.run_fetch(version="d20260929-x", seasons=[], holdouts=[2026], freeze=True, today=date(2026, 9, 29), **kw)
    assert run.rows[0].frozen is True and run.rows[0].end == "2026-09-27" and run.rows[0].ceiling == "2026-09-27"


def test_run_fetch_dry_run_isolated_from_real_ledger(tmp_path):
    out, ledger = tmp_path / "raw", tmp_path / "versions.md"
    fetcher = full_fetcher()
    run = sf.run_fetch(
        version="d20260910-s2325", seasons=[2024, 2025], holdouts=[2026], out_dir=out, ledger=ledger,
        repo_root=tmp_path, today=TODAY, dry_run=True, fetcher=fetcher,
    )
    assert fetcher.calls == [("2024-03-20", "2024-03-20"), ("2025-03-18", "2025-03-18"), ("2026-03-25", "2026-03-25")]
    assert run.version_dir == out / sf.DRYRUN_DIR / "d20260910-s2325"
    assert not ledger.exists()  # 실제 장부 불변
    assert run.ledger == run.version_dir / "versions.md" and len(sf.ledger_rows(run.ledger)) == 3
    assert not (out / "d20260910-s2325").exists()
    # 같은 날 다시 돌려도 오류 없이 다시 받는다
    sf.run_fetch(
        version="d20260910-s2325", seasons=[2024], holdouts=[], out_dir=out, ledger=ledger,
        repo_root=tmp_path, today=TODAY, dry_run=True, fetcher=fetcher,
    )
    assert fetcher.calls[-1] == ("2024-03-20", "2024-03-20")


# ---------------------------------------------------------------- CLI


def test_cli_fetch_verify_and_errors(tmp_path, monkeypatch, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fetch_data_cli", sf.Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(sf, "default_fetcher", full_fetcher())

    out, ledger = tmp_path / "raw", tmp_path / "versions.md"
    common = ["--out", str(out), "--versions-file", str(ledger), "--date", "20260910"]
    assert cli.main(["--seasons", "2024", "--holdout", "2026", "--tag", "s2325", *common]) == 0
    printed = capsys.readouterr().out
    assert "완료 d20260910-s2325" in printed and "2026 holdout" in printed and "frozen=false" in printed
    # 홀드아웃 전용 시즌을 --seasons 로 → 2, 프리즈 게이트 미충족 → 2
    assert cli.main(["--seasons", "2026", "--tag", "x", *common]) == 2
    assert "홀드아웃 전용" in capsys.readouterr().err
    assert cli.main(["--holdout", "2026", "--tag", "x", "--freeze", "--out", str(out), "--versions-file", str(ledger), "--date", "20260928"]) == 2
    assert "--freeze" in capsys.readouterr().err
    assert cli.main(["--verify", "d20260910-s2325", "--out", str(out), "--versions-file", str(ledger)]) == 0
    assert cli.main(["--verify", "d20260910-none", "--out", str(out), "--versions-file", str(ledger)]) == 1
    # 같은 버전 재수집 거부 → 2
    assert cli.main(["--seasons", "2025", "--tag", "s2325", *common]) == 2
    assert "이미 있음" in capsys.readouterr().err
    # dry-run 은 별도 디렉터리
    assert cli.main(["--seasons", "2024", "--tag", "s2325", "--dry-run", *common]) == 0
    assert (out / sf.DRYRUN_DIR / "d20260910-s2325" / "statcast_2024.parquet").exists()
    # 인자 오류
    for argv in (["--tag", "x"], ["--seasons", "2024"], ["--seasons", "2024", "--tag", "x", "--date", "2026-09-10"], ["--verify", "v", "--tag", "x"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
