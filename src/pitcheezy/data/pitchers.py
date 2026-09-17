"""레귤러 선발 필터 (design.md §1: 시즌당 일정 이닝 이상). CLI 는 scripts/make_pitcher_pool.py.

이닝 = 투수에게 귀속된 아웃 / 3. 아웃 귀속: 타석 i 동안 난 아웃 = 다음 타석이 같은 하프이닝이면 outs_when_up 차이,
아니면 3 − outs_when_up[i] (경기 마지막 하프이닝은 3아웃 전에 끝날 수 있어 그만큼 과소 — 공식 IP 와 약간 다를 수 있음).
선발 = 하프이닝 1회의 첫 타석을 던진 투수.
풀 규칙 any_season: 창 안 어느 시즌이든 min_ip 이상이면 풀. 그 투수의 창 안 모든 투구를 학습에 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ("game_pk", "game_year", "inning", "inning_topbot", "at_bat_number", "pitch_number", "outs_when_up", "pitcher", "player_name", "p_throws", "pitch_type")
SORT = ["game_pk", "at_bat_number", "pitch_number"]
HALF = ["game_pk", "inning", "inning_topbot"]


def load_pitches(data_dir: Path, version: str, seasons: list[int]) -> pd.DataFrame:
    parts = [pq.read_table(Path(data_dir) / "raw" / version / f"statcast_{s}.parquet", columns=list(COLUMNS)).to_pandas() for s in seasons]
    return pd.concat(parts, ignore_index=True).sort_values(SORT, kind="stable").reset_index(drop=True)


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    """투수 × 시즌: 투구 수, 행동 가능 투구 수(구종 있음), 아웃, 이닝, 경기 수, 선발 수."""
    df = df.sort_values(SORT, kind="stable").reset_index(drop=True)
    pa_key = ["game_pk", "at_bat_number"]
    first = df.groupby(pa_key, sort=False).head(1).reset_index(drop=True)
    same_half = (first[HALF].shift(-1) == first[HALF]).all(axis=1).to_numpy()
    next_outs = first["outs_when_up"].shift(-1).to_numpy()
    outs = np.where(same_half, next_outs - first["outs_when_up"].to_numpy(), 3 - first["outs_when_up"].to_numpy())
    first = first.assign(outs=outs.astype(int))
    first["is_start"] = (first["inning"] == 1) & ~first.duplicated(HALF)
    by = ["pitcher", "game_year"]
    agg = first.groupby(by).agg(outs=("outs", "sum"), games=("game_pk", "nunique"), starts=("is_start", "sum")).reset_index()
    pitches = df.groupby(by).agg(pitches=("pitch_number", "size"), pitches_action=("pitch_type", lambda s: s.notna().sum()),
                                 name=("player_name", "first"), throws=("p_throws", "first")).reset_index()
    t = pitches.merge(agg, on=by, how="left").fillna({"outs": 0, "games": 0, "starts": 0})
    t["ip"] = t["outs"] / 3.0
    t = t.rename(columns={"pitcher": "mlbam_id", "game_year": "season"})
    return t[["mlbam_id", "name", "throws", "season", "pitches", "pitches_action", "outs", "ip", "games", "starts"]].sort_values(["season", "ip"], ascending=[True, False]).reset_index(drop=True)


def build_pool(seasons_tbl: pd.DataFrame, min_ip: float, rule: str = "any_season") -> pd.DataFrame:
    """풀 = 투수 목록 (mlbam_id, name, throws, 시즌별 자격 여부, 창 안 총 투구 수)."""
    if rule != "any_season":
        raise ValueError(f"알 수 없는 rule: {rule}")
    q = seasons_tbl.assign(qualified=seasons_tbl["ip"] >= min_ip)
    ids = q.loc[q["qualified"], "mlbam_id"].unique()
    sub = q[q["mlbam_id"].isin(ids)]
    pool = sub.groupby("mlbam_id").agg(
        name=("name", "last"), throws=("throws", "first"),
        seasons_qualified=("qualified", lambda s: int(s.sum())), seasons_present=("season", "nunique"),
        pitches_total=("pitches", "sum"), pitches_action_total=("pitches_action", "sum"), ip_total=("ip", "sum"), starts_total=("starts", "sum"),
    ).reset_index()
    wide = sub.pivot(index="mlbam_id", columns="season", values="qualified").fillna(False)
    wide.columns = [f"q{c}" for c in wide.columns]
    pool = pool.merge(wide, left_on="mlbam_id", right_index=True)
    return pool.sort_values("pitches_total", ascending=False).reset_index(drop=True)


def markdown_summary(seasons_tbl: pd.DataFrame, pool: pd.DataFrame, min_ip: float) -> str:
    lines = [f"## 시즌별 (이닝 ≥ {min_ip:g})", "", "| 시즌 | 투수 수 | 투구 수 합 | 투수당 투구 중앙값 | 행동 제외 비율 |", "|---|---|---|---|---|"]
    for s, g in seasons_tbl[seasons_tbl["ip"] >= min_ip].groupby("season"):
        excl = 1 - g["pitches_action"].sum() / g["pitches"].sum()
        lines.append(f"| {s} | {len(g)} | {g['pitches'].sum():,} | {int(g['pitches'].median()):,} | {excl:.2%} |")
    lines += ["", f"## 풀 (any_season): {len(pool)}명, 투구 {pool['pitches_total'].sum():,}", "",
              "| # | MLBAM | 이름 | 투 | 자격 시즌 | 창 안 시즌 | 투구 | 이닝 | 선발 |", "|---|---|---|---|---|---|---|---|---|"]
    for i, r in pool.iterrows():
        qs = ",".join(c[1:] for c in pool.columns if c.startswith("q") and c[1:].isdigit() and r[c])
        lines.append(f"| {i + 1} | {r.mlbam_id} | {r['name']} | {r.throws} | {qs} | {r.seasons_present} | {r.pitches_total:,} | {r.ip_total:.1f} | {r.starts_total} |")
    return "\n".join(lines) + "\n"
