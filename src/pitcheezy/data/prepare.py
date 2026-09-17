"""투구 표 준비: 원본 Statcast → 인터페이스 id (pitcher_idx, state_id, action_id, outcome_id) + 타석·득점 정보.

풀(레귤러 선발) 투수의 투구만. 시즌마다 하나의 DataFrame. 캐시 {runs}/_cache/pitches_{data_version}_{pool_version}_{season}.parquet
행동 제외(구종 PO·IN·UN·null, 위치 결측)는 action_id = −1 로 남긴다 (카운트는 진행하므로 타석 궤적에는 필요).
결과 미지(description 미지)는 outcome_id = −1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pitcheezy.interfaces import grid as G
from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import pitch_types as PT
from pitcheezy.interfaces import states as S

RAW_COLUMNS = (
    "game_pk", "game_year", "game_date", "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "pitcher", "batter", "stand", "p_throws", "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
    "pitch_type", "plate_x", "plate_z", "sz_top", "sz_bot", "description", "events", "bat_score", "post_bat_score",
)
SORT = ["game_pk", "at_bat_number", "pitch_number"]
HALF = ["game_pk", "inning", "inning_topbot"]

OUT_COLUMNS = (
    "season", "game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot",
    "pitcher", "pitcher_idx", "batter", "stand",
    "count_id", "base_out_id", "cluster_id", "pitch_id", "loc_id", "action_id", "outcome_id", "terminal",
    "bat_score", "post_bat_score",
)


def cache_path(runs_dir: Path, data_version: str, pool_version: str, season: int) -> Path:
    return Path(runs_dir) / "_cache" / f"pitches_{data_version}_{pool_version}_{season}.parquet"


def prepare_season(
    data_dir: Path, data_version: str, season: int, pitcher_index: dict[int, int], *, holdout: bool = False,
    cluster_of_batter: dict[int, int] | None = None,
) -> pd.DataFrame:
    """한 시즌. pitcher_index: MLBAM id → pitcher_idx (풀 순서). cluster_of_batter 없으면 K=1 (cluster 0)."""
    rel = f"holdout_{season}/statcast_{season}.parquet" if holdout else f"statcast_{season}.parquet"
    df = pq.read_table(Path(data_dir) / "raw" / data_version / rel, columns=list(RAW_COLUMNS)).to_pandas()
    df = df[df["pitcher"].isin(pitcher_index)].sort_values(SORT, kind="stable").reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame(
        {
            "season": np.full(n, season, dtype=np.int16),
            "game_pk": df["game_pk"].to_numpy(dtype=np.int64),
            "at_bat_number": df["at_bat_number"].to_numpy(dtype=np.int32),
            "pitch_number": df["pitch_number"].to_numpy(dtype=np.int16),
            "inning": df["inning"].to_numpy(dtype=np.int16),
            "inning_topbot": df["inning_topbot"].astype("string").to_numpy(dtype=object),
            "pitcher": df["pitcher"].to_numpy(dtype=np.int64),
            "pitcher_idx": df["pitcher"].map(pitcher_index).to_numpy(dtype=np.int32),
            "batter": df["batter"].to_numpy(dtype=np.int64),
            "stand": df["stand"].astype("string").to_numpy(dtype=object),
            "count_id": S.count_id(df["balls"].to_numpy(), df["strikes"].to_numpy()).astype(np.int16),
            "base_out_id": S.base_out_id(df["outs_when_up"].to_numpy(), df["on_1b"].to_numpy(), df["on_2b"].to_numpy(), df["on_3b"].to_numpy()).astype(np.int16),
        }
    )
    if cluster_of_batter is None:
        out["cluster_id"] = np.zeros(n, dtype=np.int16)
    else:
        out["cluster_id"] = df["batter"].map(cluster_of_batter).fillna(-1).to_numpy(dtype=np.int16)
    pid = PT.pitch_ids(df["pitch_type"].to_numpy(dtype=object))
    zn = G.z_norm(df["plate_z"].to_numpy(dtype=float), df["sz_bot"].to_numpy(dtype=float), df["sz_top"].to_numpy(dtype=float))
    px = df["plate_x"].to_numpy(dtype=float)
    has_loc = ~np.isnan(zn) & ~np.isnan(px)
    loc = np.full(n, -1, dtype=np.int64)
    loc[has_loc] = G.loc_id(px[has_loc], zn[has_loc])
    ok = (pid >= 0) & has_loc
    aid = np.full(n, -1, dtype=np.int64)
    aid[ok] = G.action_id(pid[ok], loc[ok])
    out["pitch_id"] = pid.astype(np.int16)
    out["loc_id"] = loc.astype(np.int16)
    out["action_id"] = aid.astype(np.int16)
    oid = O.outcome_ids(df["description"].to_numpy(dtype=object), df["events"].to_numpy(dtype=object))
    out["outcome_id"] = oid.astype(np.int16)
    out["terminal"] = (oid >= O.K)
    out["bat_score"] = df["bat_score"].to_numpy(dtype=np.int16)
    out["post_bat_score"] = df["post_bat_score"].to_numpy(dtype=np.int16)
    return out[list(OUT_COLUMNS)]


def state_ids(df: pd.DataFrame, K: int, *, collapse_base_out: bool = False) -> np.ndarray:
    """state_id 열 계산. collapse_base_out (B1): base_out 을 0 으로 접는다."""
    b = np.zeros(len(df), dtype=np.int64) if collapse_base_out else df["base_out_id"].to_numpy(dtype=np.int64)
    k = df["cluster_id"].to_numpy(dtype=np.int64)
    if K == 1:
        k = np.zeros_like(k)
    return S.state_id(df["count_id"].to_numpy(dtype=np.int64), b, k, K)


def load_or_prepare(
    runs_dir: Path, data_dir: Path, data_version: str, pool_version: str, season: int,
    pitcher_index: dict[int, int], *, holdout: bool = False,
) -> pd.DataFrame:
    p = cache_path(runs_dir, data_version, pool_version, season)
    if p.exists():
        return pd.read_parquet(p)
    df = prepare_season(data_dir, data_version, season, pitcher_index, holdout=holdout)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def pitcher_index_from_pool(pool: pd.DataFrame) -> dict[int, int]:
    """pool.parquet 순서 = pitcher_idx."""
    return {int(m): i for i, m in enumerate(pool["mlbam_id"].to_numpy())}


def pa_rewards(df: pd.DataFrame, RE24: np.ndarray) -> pd.DataFrame:
    """타석별 실제 ΔRE24 (타자 팀 관점) 와 메타. 타석 = (game_pk, at_bat_number).

    ΔRE = RE24[다음 타석 시작 상태](같은 하프이닝이면) + 타석 중 득점 − RE24[타석 첫 투구 상태].
    하프이닝이 끝나면 다음 RE = 0. 득점 = post_bat_score(마지막 투구) − bat_score(첫 투구).
    """
    d = df.sort_values(SORT, kind="stable")
    key = ["game_pk", "at_bat_number"]
    first = d.groupby(key, sort=False).head(1).set_index(key)
    last = d.groupby(key, sort=False).tail(1).set_index(key)
    pa = pd.DataFrame(
        {
            "inning": first["inning"], "inning_topbot": first["inning_topbot"], "pitcher_idx": first["pitcher_idx"],
            "b_start": first["base_out_id"].astype(int), "score_start": first["bat_score"].astype(int),
            "score_end": last["post_bat_score"].astype(int), "outcome_id": last["outcome_id"].astype(int),
            "n_pitches": d.groupby(key, sort=False).size(),
            "n_pitchers": d.groupby(key, sort=False)["pitcher_idx"].nunique(),
        }
    ).reset_index().sort_values(key, kind="stable").reset_index(drop=True)
    same_half = (pa[HALF].shift(-1) == pa[HALF]).all(axis=1).to_numpy()
    next_b = pa["b_start"].shift(-1).fillna(0).astype(int).to_numpy()
    re_next = np.where(same_half, RE24[next_b], 0.0)
    runs = (pa["score_end"] - pa["score_start"]).to_numpy(dtype=float)
    pa["delta_re24"] = re_next + runs - RE24[pa["b_start"].to_numpy()]
    pa["runs"] = runs
    return pa
