"""B2 재구현 (모듈 층) — Takamido & Nakamoto 류 Transformer 를 우리 결과 공간 11-way 의 투구 단위 분류기로.

docs/baselines.md B2, design.md §3-b. 정책 층(2스트라이크 반사실 argmin)·텐서·VI·OPE 는 여기 없다.
학습 2023–24 → 평가 2025. 지표는 transition.count.holdout_metrics 와 같은 정의(NLL·ECE 15빈·ECE_HR·top-1)라
EXP-P0-006·EXP-P1-001 과 한 표에 놓을 수 있다. 2026 은 어디에도 쓰지 않는다.

입력 (논문 구성을 우리 축에 맞춘 것)
  ⓐ 시퀀스: 같은 타석의 직전 6구 × 물리 피처 7 (plate_x, plate_z 정규화, 유효 구속, 회전수, 회전축, pfx_x, pfx_z).
     6구 미만은 앞쪽 0 패딩 + key padding mask. 좌타자는 x 축(plate_x, pfx_x) 부호 반전.
  ⓑ 현재 투구: 같은 물리 피처 7 + pitch_id 임베딩(9) + loc_id 임베딩(25).
  ⓒ 맥락 16: 볼, 스트라이크, 아웃, 이닝, 초말(0/1), 주자 1·2·3루(0/1), 점수차(bat−fld), 타순 회차,
     타자 전 시즌 K%·HR%·ISO·AVG·OPS, 전 시즌 결측 플래그.
출력: outcome 11 로짓. 카운트별 규칙 마스크(outcomes.rule_mask_table)를 −inf 로 막는다 (neural.SharedNet 과 동일).

논문과 다른 점 (재현이 아니라 이식이므로 전부 기록)
  - 출력: 인플레이(1)/헛스윙아웃(0) 이진 → outcome 11-way. 우리 전이 모델과 같은 저울에 놓기 위함.
  - 대상 투구: 2스트라이크 결정구만 → 모든 투구. 우리 전이 모델이 모든 투구에서 채점되므로.
  - 현재 투구를 윈도 안에 넣지 않고 따로 뺐다. 논문은 결정구를 포함한 창을 넣지만,
    11-way 예측에서는 "무엇을 던졌는가"가 명시 입력이어야 한다 (구종·위치가 곧 우리 행동 축).
  - 시즌 창 2018–25(2020 제외) → 2023–25. 투수 풀·타석 수가 다르므로 논문 수치와 나란히 두지 않는다.
  - 조기 종료 기준 val AUC → 2025 홀드아웃 NLL (다중분류라 AUC 가 그대로 쓰이지 않는다).
  - 좌타자 반전은 plate_x·pfx_x 만. spin_axis 는 원문 기술이 없어 그대로 둔다 (미해결 질문).
  - 인코더 활성은 torch 기본(relu). 원문 미기재.

타자 전 시즌 성적 (우리 원본 parquet 에서 직접 집계 — 외부 수치를 가져오지 않는다)
  타석(PA) = events 가 있는 투구 1행. events = truncated_pa (경기·이닝 중단) 는 PA 에서 뺀다.
  타수(AB) = PA − (walk, intent_walk, hit_by_pitch, sac_fly, sac_bunt, sac_fly_double_play, catcher_interf).
  H = single+double+triple+home_run, TB = 1·2·3·4 가중합.
  AVG = H/AB, SLG = TB/AB, ISO = SLG − AVG, OBP = (H+BB+HBP)/(AB+BB+HBP+SF), OPS = OBP+SLG,
  K% = (strikeout + strikeout_double_play)/PA, HR% = home_run/PA.
  시즌 Y 의 행에는 시즌 Y−1 집계를 붙인다. Y−1 이 우리 데이터(2023–25) 밖이거나(2023 행) Y−1 PA < 50 이면
  그 시즌 리그 평균(PA ≥ 50 타자들의 비가중 평균)으로 채우고 prev_season_missing = 1 을 세운다.
  2023 행에는 2022 가 없으므로 2023 시즌 자체의 리그 평균을 대치값으로 쓴다 (그 해 전체를 같은 상수로 채우므로 정보 누출 없음).

결측 물리 피처: 학습 평균/표준편차로 표준화한 뒤 NaN → 0 (= 학습 평균으로 대치).

절제 플래그 (기본 둘 다 True = EXP-P1-006 그대로)
  current_phys=False → 현재 투구 블록에서 연속 물리 피처 7 을 뺀다 (구종 임베딩 + 위치 셀 임베딩만).
     우리 전이 모델이 행동에 대해 가진 정보량(구종 9 × 위치 25)과 같아진다. EXP-P1-007.
  use_window=False → 6구 시퀀스 인코더를 통째로 뺀다 (풀링 벡터를 결합하지 않는다). EXP-P1-008.
  둘 다 Linear 입력 차원에서 빼므로 한 config 안에서 파라미터 수가 일정하다.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from pitcheezy.interfaces import grid as G
from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S
from pitcheezy.interfaces.grid import N_LOC
from pitcheezy.interfaces.pitch_types import N_PITCH
from pitcheezy.transition.count import _ece

log = logging.getLogger("pitcheezy")

N_PHYS = 7  # plate_x, plate_z_norm, speed, spin_rate, spin_axis, pfx_x, pfx_z
CTX_NAMES: tuple[str, ...] = (
    "balls", "strikes", "outs", "inning", "topbot", "on_1b", "on_2b", "on_3b", "score_diff", "thruorder",
    "prev_k_pct", "prev_hr_pct", "prev_iso", "prev_avg", "prev_ops", "prev_season_missing",
)
N_CTX = len(CTX_NAMES)  # 16
BATTER_STAT_COLS: tuple[str, ...] = ("k_pct", "hr_pct", "iso", "avg", "ops")
MIN_PREV_PA = 50

HP_DEFAULTS = {
    "d_model": 256, "n_heads": 4, "ff": 512, "n_layers": 2, "dropout": 0.1, "window": 6,
    "dense_pitch": 64, "dense_ctx": 32, "dense_head": 64,
    "lr": 1e-4, "batch_size": 512, "max_epochs": 200, "patience": 10, "device": "auto",
    "current_phys": True,  # False = 현재 투구의 연속 물리 피처 7 제거 (구종·위치 임베딩만)
    "use_window": True,    # False = 6구 시퀀스 인코더 제거
    "eval_subset": 0,  # > 0 이면 조기 종료용 홀드아웃 NLL 을 이 행 수의 고정 부분집합에서만 (최종 지표는 항상 전체)
}

# 원본에서 읽는 물리·맥락 열 (id 표에 없는 것만). 나머지는 prepare.OUT_COLUMNS 에서 온다.
RAW_JOIN_KEYS: tuple[str, ...] = ("game_pk", "at_bat_number", "pitch_number")
RAW_FEATURE_COLUMNS: tuple[str, ...] = RAW_JOIN_KEYS + (
    "plate_x", "plate_z", "sz_top", "sz_bot", "effective_speed", "release_speed",
    "release_spin_rate", "spin_axis", "pfx_x", "pfx_z", "fld_score", "n_thruorder_pitcher",
)
RAW_BATTER_COLUMNS: tuple[str, ...] = ("game_pk", "at_bat_number", "batter", "events")

K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
BB_EVENTS = frozenset({"walk", "intent_walk"})
HBP_EVENTS = frozenset({"hit_by_pitch"})
SF_EVENTS = frozenset({"sac_fly", "sac_fly_double_play"})
NON_AB_EVENTS = BB_EVENTS | HBP_EVENTS | SF_EVENTS | {"sac_bunt", "sac_bunt_double_play", "catcher_interf"}
NOT_A_PA_EVENTS = frozenset({"truncated_pa"})
TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}


FLAG_KEYS: tuple[str, ...] = ("current_phys", "use_window")


def hparams(cfg: dict | None) -> dict:
    hp = {**HP_DEFAULTS, **(cfg or {})}
    unknown = set(hp) - set(HP_DEFAULTS)
    if unknown:
        raise ValueError(f"b2 하이퍼파라미터 모름: {sorted(unknown)}")
    for k in FLAG_KEYS:  # yaml 의 "false" 같은 문자열이 조용히 참이 되지 않게
        if not isinstance(hp[k], bool):
            raise ValueError(f"b2.{k} 는 bool 이어야 함: {hp[k]!r}")
    return hp


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


# ---------------------------------------------------------------- 타자 전 시즌 성적
def batter_season_stats(pa: pd.DataFrame) -> pd.DataFrame:
    """타석 표(batter, events) → 타자별 PA·K%·HR%·ISO·AVG·OPS. events 는 PA 종결 사건 하나씩."""
    ev = pa["events"].astype("string").fillna("")
    keep = ~ev.isin(list(NOT_A_PA_EVENTS)) & (ev != "")
    d = pd.DataFrame({"batter": pa["batter"].to_numpy(dtype=np.int64)[keep.to_numpy()], "ev": ev[keep].to_numpy(dtype=object)})
    d["is_k"] = np.isin(d["ev"], list(K_EVENTS))
    d["is_bb"] = np.isin(d["ev"], list(BB_EVENTS))
    d["is_hbp"] = np.isin(d["ev"], list(HBP_EVENTS))
    d["is_sf"] = np.isin(d["ev"], list(SF_EVENTS))
    d["is_ab"] = ~np.isin(d["ev"], list(NON_AB_EVENTS))
    d["tb"] = np.array([TOTAL_BASES.get(e, 0) for e in d["ev"]], dtype=np.float64)
    d["is_h"] = d["tb"] > 0
    d["is_hr"] = d["ev"] == "home_run"
    g = d.groupby("batter", sort=True)
    out = pd.DataFrame(
        {
            "pa": g.size().astype(np.float64), "k": g["is_k"].sum().astype(np.float64), "bb": g["is_bb"].sum().astype(np.float64),
            "hbp": g["is_hbp"].sum().astype(np.float64), "sf": g["is_sf"].sum().astype(np.float64), "ab": g["is_ab"].sum().astype(np.float64),
            "h": g["is_h"].sum().astype(np.float64), "hr": g["is_hr"].sum().astype(np.float64), "tb": g["tb"].sum().astype(np.float64),
        }
    )
    ab = out["ab"].to_numpy()
    obp_den = (out["ab"] + out["bb"] + out["hbp"] + out["sf"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.where(ab > 0, out["h"].to_numpy() / ab, 0.0)
        slg = np.where(ab > 0, out["tb"].to_numpy() / ab, 0.0)
        obp = np.where(obp_den > 0, (out["h"] + out["bb"] + out["hbp"]).to_numpy() / obp_den, 0.0)
    res = pd.DataFrame(
        {
            "pa": out["pa"].to_numpy(), "k_pct": out["k"].to_numpy() / out["pa"].to_numpy(), "hr_pct": out["hr"].to_numpy() / out["pa"].to_numpy(),
            "iso": slg - avg, "avg": avg, "ops": obp + slg,
        },
        index=out.index,
    )
    res.index.name = "batter"
    return res


def load_batter_stats(data_dir: Path, data_version: str, season: int) -> pd.DataFrame:
    """원본 시즌 parquet → batter_season_stats. 전체 타석(풀 밖 투수 포함)."""
    t = pq.read_table(Path(data_dir) / "raw" / data_version / f"statcast_{season}.parquet", columns=list(RAW_BATTER_COLUMNS)).to_pandas()
    t = t[t["events"].notna()].drop_duplicates(subset=["game_pk", "at_bat_number"], keep="last")
    return batter_season_stats(t)


def league_means(stats: pd.DataFrame, min_pa: int = MIN_PREV_PA) -> np.ndarray:
    """PA ≥ min_pa 타자들의 비가중 평균 [5] (BATTER_STAT_COLS 순서)."""
    q = stats[stats["pa"] >= min_pa]
    if len(q) == 0:
        q = stats
    return np.array([float(q[c].mean()) if len(q) else 0.0 for c in BATTER_STAT_COLS], dtype=np.float64)


def prev_season_block(batter: np.ndarray, prev: pd.DataFrame | None, fallback: np.ndarray, min_pa: int = MIN_PREV_PA):
    """[N, 5] 전 시즌 성적 + [N] 결측 플래그. prev 가 없거나 PA < min_pa 면 fallback(리그 평균)."""
    n = len(batter)
    out = np.tile(fallback.astype(np.float64), (n, 1))
    miss = np.ones(n, dtype=np.float64)
    if prev is not None and len(prev):
        q = prev[prev["pa"] >= min_pa]
        idx = q.index.to_numpy(dtype=np.int64)
        if len(idx):
            pos = np.clip(np.searchsorted(idx, batter), 0, len(idx) - 1)
            hit = idx[pos] == batter
            vals = q[list(BATTER_STAT_COLS)].to_numpy(dtype=np.float64)
            out[hit] = vals[pos[hit]]
            miss[hit] = 0.0
    return out, miss


# ---------------------------------------------------------------- 피처
def load_raw_features(data_dir: Path, data_version: str, season: int) -> pd.DataFrame:
    p = Path(data_dir) / "raw" / data_version / f"statcast_{season}.parquet"
    t = pq.read_table(p, columns=list(RAW_FEATURE_COLUMNS)).to_pandas()
    return t.drop_duplicates(subset=list(RAW_JOIN_KEYS), keep="last")


def physical_features(d: pd.DataFrame) -> np.ndarray:
    """[N, 7] float64 (NaN 유지). 좌타자는 plate_x·pfx_x 부호 반전."""
    lh = (d["stand"].astype("string").to_numpy(dtype=object) == "L")
    sgn = np.where(lh, -1.0, 1.0)
    px = d["plate_x"].to_numpy(dtype=np.float64) * sgn
    zn = G.z_norm(d["plate_z"].to_numpy(dtype=np.float64), d["sz_bot"].to_numpy(dtype=np.float64), d["sz_top"].to_numpy(dtype=np.float64))
    spd = d["effective_speed"].to_numpy(dtype=np.float64)
    spd = np.where(np.isnan(spd), d["release_speed"].to_numpy(dtype=np.float64), spd)
    return np.column_stack(
        [px, zn, spd, d["release_spin_rate"].to_numpy(dtype=np.float64), d["spin_axis"].to_numpy(dtype=np.float64),
         d["pfx_x"].to_numpy(dtype=np.float64) * sgn, d["pfx_z"].to_numpy(dtype=np.float64)]
    )


def context_features(d: pd.DataFrame, prev_stats: np.ndarray, prev_missing: np.ndarray) -> np.ndarray:
    """[N, 16] float64. CTX_NAMES 순서."""
    balls, strikes = S.decode_count(d["count_id"].to_numpy(dtype=np.int64))
    outs, r1, r2, r3 = S.decode_base_out(d["base_out_id"].to_numpy(dtype=np.int64))
    topbot = (d["inning_topbot"].astype("string").to_numpy(dtype=object) == "Bot").astype(np.float64)
    thru = d["n_thruorder_pitcher"].to_numpy(dtype=np.float64)
    thru = np.where(np.isnan(thru), 1.0, thru)
    diff = d["bat_score"].to_numpy(dtype=np.float64) - d["fld_score"].to_numpy(dtype=np.float64)
    return np.column_stack(
        [balls, strikes, outs, d["inning"].to_numpy(dtype=np.float64), topbot, r1, r2, r3, diff, thru,
         prev_stats, prev_missing]
    ).astype(np.float64)


def pa_windows(d: pd.DataFrame, phys: np.ndarray, window: int):
    """[N, window, 7] 직전 투구 물리 피처 + [N, window] 패딩 마스크(True = 패딩). d 는 (경기, 타석, 투구) 정렬."""
    n = len(d)
    g = d["game_pk"].to_numpy(dtype=np.int64)
    ab = d["at_bat_number"].to_numpy(dtype=np.int64)
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = (g[1:] != g[:-1]) | (ab[1:] != ab[:-1])
    idx = np.arange(n)
    start = np.maximum.accumulate(np.where(new, idx, -1))
    win = np.zeros((n, window, N_PHYS), dtype=np.float64)
    pad = np.ones((n, window), dtype=bool)
    for j in range(1, window + 1):  # j=1 이 직전 구 → 창의 마지막 칸
        src = idx - j
        ok = src >= start
        col = window - j
        win[ok, col] = phys[src[ok]]
        pad[ok, col] = False
    return win, pad


def eligible_mask(d: pd.DataFrame) -> np.ndarray:
    """행동·결과가 있고 규칙 마스크를 어기지 않는 투구 (transition.neural._rows 와 같은 규칙)."""
    ok = (d["action_id"].to_numpy() >= 0) & (d["outcome_id"].to_numpy() >= 0)
    cnt = d["count_id"].to_numpy(dtype=np.int64)
    ok &= O.rule_mask_table()[cnt, np.maximum(d["outcome_id"].to_numpy(dtype=np.int64), 0)]
    return ok


class Dataset:
    """모델에 넣는 배열 묶음. 표준화 전(raw) 값으로 만들고 Scaler 로 나중에 표준화한다."""

    __slots__ = ("win", "pad", "cur", "pid", "lid", "ctx", "y", "count", "pitcher_idx", "state_id", "action_id")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    def __len__(self) -> int:
        return len(self.y)

    def take(self, sel: np.ndarray) -> "Dataset":
        return Dataset(**{k: getattr(self, k)[sel] for k in self.__slots__})


def build_dataset(ids: pd.DataFrame, raw: pd.DataFrame, prev: pd.DataFrame | None, fallback: np.ndarray, window: int) -> Dataset:
    """id 표(prepare.OUT_COLUMNS) + 원본 물리 열 → Dataset (적격 투구만). 창은 적격 여부와 무관하게 타석의 모든 투구에서 만든다."""
    d = ids.sort_values(list(RAW_JOIN_KEYS), kind="stable").reset_index(drop=True)
    d = d.merge(raw, on=list(RAW_JOIN_KEYS), how="left", validate="m:1")
    phys = physical_features(d)
    win, pad = pa_windows(d, phys, window)
    pstat, pmiss = prev_season_block(d["batter"].to_numpy(dtype=np.int64), prev, fallback)
    ctx = context_features(d, pstat, pmiss)
    sel = eligible_mask(d)
    sid = S.state_id(d["count_id"].to_numpy(dtype=np.int64), d["base_out_id"].to_numpy(dtype=np.int64), np.zeros(len(d), dtype=np.int64), 1)
    return Dataset(
        win=win[sel], pad=pad[sel], cur=phys[sel], pid=d["pitch_id"].to_numpy(dtype=np.int64)[sel], lid=d["loc_id"].to_numpy(dtype=np.int64)[sel],
        ctx=ctx[sel], y=d["outcome_id"].to_numpy(dtype=np.int64)[sel], count=d["count_id"].to_numpy(dtype=np.int64)[sel],
        pitcher_idx=d["pitcher_idx"].to_numpy(dtype=np.int64)[sel], state_id=sid[sel], action_id=d["action_id"].to_numpy(dtype=np.int64)[sel],
    )


class Scaler:
    """학습 행의 nan 평균·표준편차. 표준화 뒤 NaN → 0 (= 평균 대치)."""

    def __init__(self, phys_mean, phys_std, ctx_mean, ctx_std):
        self.phys_mean, self.phys_std = np.asarray(phys_mean, np.float64), np.asarray(phys_std, np.float64)
        self.ctx_mean, self.ctx_std = np.asarray(ctx_mean, np.float64), np.asarray(ctx_std, np.float64)

    @classmethod
    def fit(cls, ds: Dataset) -> "Scaler":
        pm, ps = np.nanmean(ds.cur, axis=0), np.nanstd(ds.cur, axis=0)
        cm, cs = np.nanmean(ds.ctx, axis=0), np.nanstd(ds.ctx, axis=0)
        return cls(pm, np.where(ps > 1e-8, ps, 1.0), cm, np.where(cs > 1e-8, cs, 1.0))

    def apply(self, ds: Dataset) -> Dataset:
        cur = np.nan_to_num((ds.cur - self.phys_mean) / self.phys_std, nan=0.0, posinf=0.0, neginf=0.0)
        win = np.nan_to_num((ds.win - self.phys_mean) / self.phys_std, nan=0.0, posinf=0.0, neginf=0.0)
        win[ds.pad] = 0.0
        ctx = np.nan_to_num((ds.ctx - self.ctx_mean) / self.ctx_std, nan=0.0, posinf=0.0, neginf=0.0)
        return Dataset(win=win.astype(np.float32), pad=ds.pad, cur=cur.astype(np.float32), pid=ds.pid, lid=ds.lid,
                       ctx=ctx.astype(np.float32), y=ds.y, count=ds.count, pitcher_idx=ds.pitcher_idx, state_id=ds.state_id, action_id=ds.action_id)

    def to_meta(self) -> dict:
        return {"phys_mean": self.phys_mean.tolist(), "phys_std": self.phys_std.tolist(),
                "ctx_mean": self.ctx_mean.tolist(), "ctx_std": self.ctx_std.tolist(), "ctx_names": list(CTX_NAMES)}


# ---------------------------------------------------------------- 모델
class B2Net(nn.Module):
    def __init__(self, hp: dict):
        super().__init__()
        d, w = int(hp["d_model"]), int(hp["window"])
        self.window = w
        self.current_phys = bool(hp.get("current_phys", True))
        self.use_window = bool(hp.get("use_window", True))
        if self.use_window:  # 절제 시에는 인코더를 만들지도 않는다 (파라미터 수가 config 를 그대로 반영하도록)
            self.inp = nn.Linear(N_PHYS, d)
            self.pos = nn.Embedding(w, d)
            layer = nn.TransformerEncoderLayer(d, int(hp["n_heads"]), int(hp["ff"]), float(hp["dropout"]), batch_first=True)
            self.enc = nn.TransformerEncoder(layer, int(hp["n_layers"]), enable_nested_tensor=False)  # 패딩 마스크 + MPS 에서 경로가 갈리지 않게
            self.register_buffer("posidx", torch.arange(w))
        dp, dc, dh = int(hp["dense_pitch"]), int(hp["dense_ctx"]), int(hp["dense_head"])
        self.e_pitch = nn.Embedding(N_PITCH, dp)
        self.e_loc = nn.Embedding(N_LOC, dp)
        self.cur = nn.Sequential(nn.Linear((N_PHYS if self.current_phys else 0) + 2 * dp, dp), nn.GELU())
        self.ctx = nn.Sequential(nn.Linear(N_CTX, dc), nn.GELU())
        self.head = nn.Sequential(nn.Linear((d if self.use_window else 0) + dp + dc, dh), nn.GELU(), nn.Linear(dh, O.N_OUTCOMES))
        self.register_buffer("rule", torch.from_numpy(O.rule_mask_table().astype(bool)))

    def forward(self, win, pad, cur, pid, lid, ctx, count):
        parts = []
        if self.use_window:
            all_pad = pad.all(dim=1)
            m = pad.clone()
            m[all_pad] = False  # 전부 패딩이면 softmax 가 NaN → 일단 풀고 아래에서 0 으로 덮는다
            h = self.enc(self.inp(win) + self.pos(self.posidx)[None], src_key_padding_mask=m)
            keep = (~m).to(h.dtype)[..., None]
            z = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            parts.append(z.masked_fill(all_pad[:, None], 0.0))
        blocks = [self.e_pitch(pid), self.e_loc(lid)]
        if self.current_phys:
            blocks.insert(0, cur)
        parts.append(self.cur(torch.cat(blocks, dim=-1)))
        parts.append(self.ctx(ctx))
        logits = self.head(torch.cat(parts, dim=-1))
        return logits.masked_fill(~self.rule[count], float("-inf"))


def _tensors(ds: Dataset, dev) -> list:
    return [
        torch.from_numpy(np.ascontiguousarray(ds.win)).to(dev), torch.from_numpy(np.ascontiguousarray(ds.pad)).to(dev),
        torch.from_numpy(np.ascontiguousarray(ds.cur)).to(dev), torch.from_numpy(np.ascontiguousarray(ds.pid)).to(dev),
        torch.from_numpy(np.ascontiguousarray(ds.lid)).to(dev), torch.from_numpy(np.ascontiguousarray(ds.ctx)).to(dev),
        torch.from_numpy(np.ascontiguousarray(ds.count)).to(dev), torch.from_numpy(np.ascontiguousarray(ds.y)).to(dev),
    ]


def _nll(net: B2Net, data: list, bs: int = 8192) -> float:
    net.eval()
    n, tot = len(data[0]), 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            t = [x[i:i + bs] for x in data]
            tot += float(nn.functional.cross_entropy(net(*t[:7]), t[7], reduction="sum"))
    return tot / max(n, 1)


def train(train_ds: Dataset, eval_ds: Dataset | None, hp: dict, seed: int, *, max_epochs: int | None = None):
    """조기 종료는 eval_ds NLL (patience). 반환 (net, 이력). 최적 에폭의 가중치를 복원한다."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = pick_device(hp["device"])
    net = B2Net(hp).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=float(hp["lr"]))
    data = _tensors(train_ds, dev)
    ev = None
    if eval_ds is not None:
        sub = int(hp.get("eval_subset") or 0)
        e = eval_ds
        if 0 < sub < len(eval_ds):
            e = eval_ds.take(np.sort(np.random.default_rng(12345).choice(len(eval_ds), sub, replace=False)))
        ev = _tensors(e, dev)
    n, bs = len(train_ds), int(hp["batch_size"])
    me = int(max_epochs) if max_epochs else int(hp["max_epochs"])
    hist, best, best_state, bad = [], (None, float("inf")), None, 0
    for ep in range(1, me + 1):
        t0 = time.time()
        net.train()
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            t = [x[idx] for x in data]
            loss = nn.functional.cross_entropy(net(*t[:7]), t[7])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        rec = {"epoch": ep, "train_nll": tot / n, "seconds": time.time() - t0}
        if ev is not None:
            rec["eval_nll"] = _nll(net, ev)
            if rec["eval_nll"] < best[1] - 1e-5:
                best, bad = (ep, rec["eval_nll"]), 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log.info("b2 ep %d train NLL %.4f%s (%.0fs)", ep, rec["train_nll"], f" eval NLL {rec['eval_nll']:.4f}" if "eval_nll" in rec else "", rec["seconds"])
        if ev is not None and bad >= int(hp["patience"]):
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"epochs": hist, "best_epoch": best[0] if best[0] else me, "device": str(dev),
                 "eval_subset_n": (len(ev[0]) if ev is not None else 0)}


def predict_probs(net: B2Net, ds: Dataset, dev, bs: int = 8192) -> np.ndarray:
    """[N, 11] float64. 행 합 1."""
    data = _tensors(ds, dev)
    out = np.empty((len(ds), O.N_OUTCOMES), dtype=np.float64)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            t = [x[i:i + bs] for x in data]
            p = torch.softmax(net(*t[:7]).float(), dim=-1).cpu().numpy().astype(np.float64)
            out[i:i + bs] = p / p.sum(-1, keepdims=True)
    return out


def metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """transition.count.holdout_metrics 와 같은 정의."""
    p_obs = probs[np.arange(len(y)), y]
    ece = [_ece(probs[:, k], (y == k).astype(float)) for k in range(O.N_OUTCOMES)]
    return {
        "holdout_nll": float(-np.log(np.maximum(p_obs, 1e-12)).mean()),
        "holdout_ece": float(np.mean(ece)), "holdout_ece_hr": float(ece[O.HR]),
        "holdout_ece_by_outcome": {O.OUTCOME_NAMES[k]: float(e) for k, e in enumerate(ece)},
        "holdout_n_pitches": int(len(y)), "holdout_accuracy_top1": float((probs.argmax(1) == y).mean()),
    }
