"""statcast_fetch 단위 테스트. 네트워크 없음 — pybaseball.statcast() 호출은 가짜 fetcher 로 대체."""

import json
from datetime import date, timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest

from pitcheezy.data import statcast_fetch as sf


def pitch_rows(day, pitcher, game_pk, n_pa, per_pa, game_type="R", bat_speed=71.5, ab_start=1):
    rows = []
    for ab in range(ab_start, ab_start + n_pa):
        for pn in range(1, per_pa + 1):
            rows.append(
                {
                    "game_date": day.isoformat(),
                    "game_pk": game_pk,
                    "at_bat_number": ab,
                    "pitch_number": pn,
                    "pitcher": pitcher,
                    "player_name": f"P{pitcher}",
                    "game_type": game_type,
                    "pitch_type": "FF",
                    "release_speed": 95.0,
                    "bat_speed": bat_speed,
                }
            )
    return rows


def frame(rows):
    """pybaseball.statcast() 반환 형태 흉내: 내림차순 정렬 + convert_dtypes(convert_string=False)."""
    df = pd.DataFrame(rows)
    return df.sort_values(sf.SORT_KEYS, ascending=False).convert_dtypes(convert_string=False)


def test_season_windows_bounds():
    w = sf.season_windows(2025)
    assert w[0] == (date(2025, 3, 15), date(2025, 3, 28))
    assert w[-1][1] == date(2025, 11, 15)
    assert all((e - s).days <= sf.WINDOW_DAYS - 1 for s, e in w)
    assert all(w[i][1] + timedelta(days=1) == w[i + 1][0] for i in range(len(w) - 1))
    d = date(2025, 4, 1)
    assert sf.season_windows(2025, floor=d, ceiling=d) == [(d, d)]
    assert sf.season_windows(2025, floor=date(2025, 5, 1), ceiling=date(2025, 4, 1)) == []
    # 3/15 이전 floor·11/15 이후 ceiling 은 클램프 (pybaseball 이 어차피 건너뛰는 구간)
    assert sf.season_windows(2025, floor=date(2025, 1, 1), ceiling=date(2025, 12, 31)) == w


def test_fetch_window_filters_sorts_normalizes():
    day = date(2025, 4, 1)
    rows = (
        pitch_rows(day, 1, 100, 2, 3)
        + pitch_rows(day, 1, 101, 1, 2, game_type="S")
        + pitch_rows(day, 1, 102, 1, 2, game_type="F")
    )

    def fetcher(start, end):
        assert (start, end) == ("2025-04-01", "2025-04-01")
        return frame(rows)

    df, rows_raw = sf.fetch_window(day, day, fetcher=fetcher)
    assert rows_raw == 10
    assert len(df) == 6 and set(df["game_type"]) == {"R"}
    assert pd.api.types.is_datetime64_any_dtype(df["game_date"])
    keys = list(zip(df["game_pk"], df["at_bat_number"], df["pitch_number"]))
    assert keys == sorted(keys)


def test_fetch_window_empty_and_missing_columns():
    df, rows_raw = sf.fetch_window(date(2025, 4, 1), date(2025, 4, 1), fetcher=lambda s, e: pd.DataFrame())
    assert rows_raw == 0 and df.empty
    with pytest.raises(ValueError, match="예상 컬럼 없음"):
        sf.fetch_window(date(2025, 4, 1), date(2025, 4, 1), fetcher=lambda s, e: pd.DataFrame({"x": [1]}))


def test_fetch_window_retries():
    calls = []

    def flaky(start, end):
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return frame(pitch_rows(date(2025, 4, 1), 1, 100, 1, 1))

    df, _ = sf.fetch_window(date(2025, 4, 1), date(2025, 4, 1), fetcher=flaky, retries=3, backoff=0)
    assert len(calls) == 3 and len(df) == 1

    def down(start, end):
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        sf.fetch_window(date(2025, 4, 1), date(2025, 4, 1), fetcher=down, retries=2, backoff=0)


def test_fetch_season_resume(tmp_path):
    d = date(2025, 4, 1)
    calls = []

    def fetcher(start, end):
        calls.append((start, end))
        return frame(pitch_rows(d, 7, 100, 3, 4))

    r1 = sf.fetch_season(2025, tmp_path, floor=d, ceiling=d, fetcher=fetcher)
    assert not r1.skipped and r1.windows_fetched == 1 and calls == [("2025-04-01", "2025-04-01")]
    parts = tmp_path / sf.PARTS_DIR / "2025"
    assert (parts / "2025-04-01_2025-04-01.parquet").exists()
    assert (parts / "2025-04-01_2025-04-01.done").exists()
    assert pq.read_metadata(r1.path).num_rows == 12

    def boom(start, end):
        raise AssertionError("완료 창은 다시 요청하면 안 됨")

    # 최종본이 있으면 아무것도 하지 않음
    assert sf.fetch_season(2025, tmp_path, floor=d, ceiling=d, fetcher=boom).skipped
    # 최종본만 지우면 완료 창은 건너뛰고 재조립
    r1.path.unlink()
    r3 = sf.fetch_season(2025, tmp_path, floor=d, ceiling=d, fetcher=boom)
    assert r3.windows_fetched == 0 and r3.path.exists()
    # 마커만 있고 parquet 이 없으면 재수집
    r3.path.unlink()
    (parts / "2025-04-01_2025-04-01.parquet").unlink()
    r4 = sf.fetch_season(2025, tmp_path, floor=d, ceiling=d, fetcher=fetcher)
    assert r4.windows_fetched == 1 and len(calls) == 2


def test_assemble_promotes_dtypes(tmp_path):
    # 창 1: bat_speed 전부 결측 → convert_dtypes 가 Int64 로 잡음 / 창 2: Float64 → 조립본은 double
    def fetcher(start, end):
        if start == "2025-04-01":
            return frame(pitch_rows(date(2025, 4, 2), 1, 100, 2, 2, bat_speed=float("nan")))
        return frame(pitch_rows(date(2025, 4, 15), 1, 101, 1, 3, bat_speed=70.25))

    r = sf.fetch_season(2025, tmp_path, floor=date(2025, 4, 1), ceiling=date(2025, 4, 15), fetcher=fetcher)
    assert r.windows_total == 2
    assert str(pq.read_schema(r.path).field("bat_speed").type) == "double"
    out = pd.read_parquet(r.path)
    assert len(out) == 7 and int(out["bat_speed"].notna().sum()) == 3
    assert list(out["game_date"].dt.day) == [2, 2, 2, 2, 15, 15, 15]


def test_make_sample_whole_plate_appearances(tmp_path):
    rows = []
    rows += pitch_rows(date(2025, 4, 1), 11, 100, 12, 5)  # 60구
    rows += pitch_rows(date(2025, 4, 6), 11, 105, 12, 5)  # +60 = 120구
    rows += pitch_rows(date(2025, 4, 2), 22, 101, 40, 3)  # 120구 (동률)
    rows += pitch_rows(date(2025, 4, 3), 33, 102, 5, 4)  # 20구 → 탈락
    df = sf.normalize(frame(rows))
    src = tmp_path / "statcast_2025.parquet"
    df.to_parquet(src, index=False)

    out = sf.make_sample(src)
    assert set(out["pitcher"]) == {11, 22}
    assert int((out["pitcher"] == 11).sum()) == 100  # 5구 타석 20개
    assert int((out["pitcher"] == 22).sum()) == 102  # 3구 타석 34개 (100 을 넘는 첫 타석까지)
    for pid in (11, 22):
        sub = out[out["pitcher"] == pid]
        orig = df[df["pitcher"] == pid].groupby(["game_pk", "at_bat_number"]).size()
        got = sub.groupby(["game_pk", "at_bat_number"]).size()
        assert all(orig[k] == v for k, v in got.items())  # 타석을 자르지 않음
    keys = list(zip(out["game_date"], out["game_pk"], out["at_bat_number"], out["pitch_number"]))
    assert keys == sorted(keys)
    assert list(out.columns) == list(df.columns)


def test_ledger_row_and_append(tmp_path):
    manifest = {
        "version": "d20260907-test",
        "fetch_commit": "abc1234",
        "versions": {"pybaseball": "2.2.7"},
        "query": {
            "window_days": 14, "season_lo": "03-15", "season_hi": "11-15",
            "floor": None, "ceiling": "2026-09-06", "game_type": "R",
        },
        "seasons": {"2025": {"rows": 1234, "sha256": "a" * 64}, "2026": {"rows": 10, "sha256": "b" * 64}},
    }
    row = sf.ledger_row(manifest)
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert len(cells) == 7
    assert cells[0] == "d20260907-test" and cells[1] == "2025, 2026" and cells[5] == "abc1234" and cells[6] == ""
    assert "floor" not in cells[2] and "ceiling 2026-09-06" in cells[2]
    assert "2025: 1,234" in cells[3] and "계 1,244" in cells[3]
    assert "2025: " + "a" * 64 in cells[4]

    ledger = tmp_path / "versions.md"
    assert sf.append_ledger(ledger, "d20260907-test", row) is True
    assert sf.append_ledger(ledger, "d20260907-test", row) is False
    text = ledger.read_text(encoding="utf-8")
    assert text.startswith("# 데이터 버전") and text.count("d20260907-test") == 1
    assert sf.has_ledger_row(ledger, "d20260907-test")


def test_run_fetch_manifest_and_verify(tmp_path):
    d = date(2025, 4, 1)

    def fetcher(start, end):
        return frame(pitch_rows(d, 7, 100, 3, 4))

    out_dir, ledger = tmp_path / "data", tmp_path / "versions.md"
    sf.run_fetch("d20260907-test", [2025], out_dir, ledger, repo_root=tmp_path, floor=d, ceiling=d, fetcher=fetcher)
    vdir = out_dir / "d20260907-test"
    m = sf.load_manifest(vdir)
    assert m["seasons"]["2025"]["rows"] == 12
    assert m["seasons"]["2025"]["first_game_date"] == "2025-04-01" == m["seasons"]["2025"]["last_game_date"]
    assert m["query"]["ceiling"] == "2025-04-01" and m["query"]["floor"] == "2025-04-01"
    assert sf.has_ledger_row(ledger, "d20260907-test")
    assert sf.verify(vdir, ledger) == []

    # manifest 의 해시가 어긋나면 실패로 보고
    m["seasons"]["2025"]["sha256"] = "0" * 64
    (vdir / sf.MANIFEST_NAME).write_text(json.dumps(m), encoding="utf-8")
    problems = sf.verify(vdir, ledger)
    assert len(problems) == 1 and "sha256" in problems[0]

    with pytest.raises(ValueError, match="형식 오류"):
        sf.run_fetch("bad-id", [2025], out_dir, ledger, repo_root=tmp_path, fetcher=fetcher)
