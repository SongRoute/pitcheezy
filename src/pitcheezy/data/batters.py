"""타자 군집 v0 — 성향 특징 k-means, 좌우 층화 (design.md §1-d, ① 타자 표현). CLI scripts/make_batter_clusters.py.

단위 = (타자 MLBAM id, stand). 스위치히터는 좌·우 따로. 특징은 학습 창(2023–25) 투구에서만 계산.
cluster_id = side × (K/2) + k  (side: L=0, R=1). 기본 군집(특징 없는 타자) = 그 side 에서 중심이 리그 평균에 가장 가까운 군집.
k-means 는 numpy 구현(시드 고정, n_init 재시작, 표준화 특징).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLUMNS = ("game_pk", "at_bat_number", "batter", "stand", "description", "events", "zone", "bb_type", "launch_speed", "launch_angle", "estimated_woba_using_speedangle")
SWING = frozenset({"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt", "missed_bunt", "bunt_foul_tip", "foul_pitchout"})
CONTACT = frozenset({"foul", "foul_tip", "hit_into_play", "foul_bunt", "bunt_foul_tip"})
SIDES = ("L", "R")


def load_pitches(data_dir: Path, version: str, seasons: list[int]) -> pd.DataFrame:
    return pd.concat([pq.read_table(Path(data_dir) / "raw" / version / f"statcast_{s}.parquet", columns=list(COLUMNS)).to_pandas() for s in seasons], ignore_index=True)


def batter_features(df: pd.DataFrame) -> pd.DataFrame:
    """(batter, stand) 별 성향 특징 + 타석 수."""
    d = df.copy()
    d["swing"] = d["description"].isin(SWING)
    d["contact"] = d["description"].isin(CONTACT)
    d["out_zone"] = d["zone"] >= 11
    d["chase"] = d["swing"] & d["out_zone"]
    d["inplay"] = d["description"] == "hit_into_play"
    d["gb"] = d["bb_type"] == "ground_ball"
    d["fb"] = d["bb_type"].isin(["fly_ball", "popup"])
    d["hard"] = d["inplay"] & (d["launch_speed"] >= 95)
    d["pa_end"] = d["events"].notna()
    d["k"] = d["events"].isin(["strikeout", "strikeout_double_play"])
    d["bb"] = d["events"].isin(["walk", "intent_walk"])
    g = d.groupby(["batter", "stand"])
    f = pd.DataFrame({
        "n_pitches": g.size(),
        "n_pa": g["pa_end"].sum(),
        "k_rate": g["k"].sum() / g["pa_end"].sum().clip(lower=1),
        "bb_rate": g["bb"].sum() / g["pa_end"].sum().clip(lower=1),
        "swing_rate": g["swing"].mean(),
        "chase_rate": g["chase"].sum() / g["out_zone"].sum().clip(lower=1),
        "contact_rate": g["contact"].sum() / g["swing"].sum().clip(lower=1),
        "gb_rate": g["gb"].sum() / g["inplay"].sum().clip(lower=1),
        "fb_rate": g["fb"].sum() / g["inplay"].sum().clip(lower=1),
        "hard_hit_rate": g["hard"].sum() / g["inplay"].sum().clip(lower=1),
        "xwoba_contact": g["estimated_woba_using_speedangle"].mean(),
        "avg_launch_angle": g["launch_angle"].mean(),
    }).reset_index()
    return f


def kmeans(X: np.ndarray, k: int, *, n_init: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        # k-means++ 초기화
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(c)
        for _ in range(max_iter):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (lab, C, inertia)
    return best


def cluster(features: pd.DataFrame, feature_names: list[str], K: int, *, min_pa: int, n_init: int, max_iter: int, seed: int):
    """→ (assign DataFrame[batter, stand, cluster_id, is_default, n_pa], centroids DataFrame, meta)"""
    if K % 2:
        raise ValueError("K 는 짝수 (좌우 층화)")
    per_side = K // 2
    rows, cents, meta = [], [], {}
    for si, side in enumerate(SIDES):
        f = features[(features["stand"] == side)]
        elig = f[f["n_pa"] >= min_pa].dropna(subset=feature_names)
        X = elig[feature_names].to_numpy(dtype=float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        lab, C, inertia = kmeans(Z, per_side, n_init=n_init, max_iter=max_iter, seed=seed + si)
        # 군집 번호를 k_rate 중심 오름차순으로 고정 (재현성·읽기 쉬움)
        order = np.argsort(C[:, feature_names.index("k_rate")])
        remap = {int(o): j for j, o in enumerate(order)}
        lab = np.array([remap[int(l)] for l in lab])
        C = C[order]
        default = int(np.argmin((C ** 2).sum(1)))  # 리그 평균(0 벡터)에 가장 가까운 중심
        rows.append(pd.DataFrame({"batter": elig["batter"].to_numpy(), "stand": side, "cluster_id": si * per_side + lab, "is_default": False, "n_pa": elig["n_pa"].to_numpy()}))
        rest = f[~f.index.isin(elig.index)]
        rows.append(pd.DataFrame({"batter": rest["batter"].to_numpy(), "stand": side, "cluster_id": si * per_side + default, "is_default": True, "n_pa": rest["n_pa"].to_numpy()}))
        for j in range(per_side):
            cents.append({"cluster_id": si * per_side + j, "stand": side, "n_batters": int((lab == j).sum()), **{n: float(C[j, i] * sd[i] + mu[i]) for i, n in enumerate(feature_names)}})
        meta[side] = {"n_eligible": int(len(elig)), "n_default": int(len(rest)), "default_cluster_id": si * per_side + default, "inertia": inertia, "feature_mean": dict(zip(feature_names, mu.tolist())), "feature_std": dict(zip(feature_names, sd.tolist()))}
    assign = pd.concat(rows, ignore_index=True).sort_values(["stand", "batter"]).reset_index(drop=True)
    assign["cluster_id"] = assign["cluster_id"].astype(np.int16)
    return assign, pd.DataFrame(cents), meta


def load_cluster_map(d: Path) -> tuple[dict[tuple[int, str], int], dict[str, int]]:
    """batter_clusters.parquet → ({(batter, stand): cluster_id}, {stand: 기본 cluster_id})"""
    import json
    a = pd.read_parquet(Path(d) / "batter_clusters.parquet")
    meta = json.loads((Path(d) / "meta.json").read_text())
    return {(int(b), s): int(c) for b, s, c in zip(a["batter"], a["stand"], a["cluster_id"])}, {s: meta["sides"][s]["default_cluster_id"] for s in SIDES}
