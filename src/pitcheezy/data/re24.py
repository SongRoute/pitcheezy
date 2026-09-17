"""RE24·dRE24 계산 (docs/interface-spec.md 공통 자산 0, design.md §0). CLI 는 scripts/make_re24.py.

정의
    RE24[b]            타석 시작 시점 상태 b(아웃×8+주자)에서 그 하프이닝이 끝날 때까지 난 득점의 평균
    dRE24[o, b]        종결 결과 o 가 상태 b(종결 투구 시점)에서 일어났을 때 RE24[다음 상태] + 그 플레이 득점 − RE24[b] 의 평균.
                       다음 상태 = 같은 하프이닝 다음 타석의 첫 투구 상태, 하프이닝이 끝났으면 RE 0.
                       인플레이 아웃 안의 병살·희생플라이는 구분하지 않고 평균 (ASM-7)
제외  경기의 마지막 하프이닝 (끝내기·홈팀 말 공격 생략 등으로 3아웃 전에 끝날 수 있음). 그 밖의 하프이닝은 모두 3아웃으로 끝난다.
비고  타석 중 주자 사건(도루·폭투 득점)은 RE24 쪽에는 "이닝 끝까지 득점"에 자연히 포함되고, dRE24 쪽에는 종결 투구 시점 상태를 쓰므로
      그 시점 이전 변화는 반영돼 있다. 타석 사이의 견제사 등은 다음 타석 첫 투구 상태에 섞여 들어간다 (작은 잡음, 미보정).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pitcheezy.interfaces.outcomes import N_TERMINAL, OUTCOME_NAMES, TERMINAL, is_terminal, outcome_ids
from pitcheezy.interfaces.states import N_BASE_OUT, base_out_id, decode_base_out

COLUMNS = (
    "game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number",
    "outs_when_up", "on_1b", "on_2b", "on_3b", "bat_score", "post_bat_score", "description", "events",
)
SORT = ["game_pk", "at_bat_number", "pitch_number"]
HALF = ["game_pk", "inning", "inning_topbot"]


def load_pitches(data_dir: Path, version: str, seasons: list[int]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        p = Path(data_dir) / "raw" / version / f"statcast_{s}.parquet"
        parts.append(pq.read_table(p, columns=list(COLUMNS)).to_pandas())
    return pd.concat(parts, ignore_index=True).sort_values(SORT, kind="stable").reset_index(drop=True)


@dataclass
class RE24Result:
    RE24: np.ndarray  # [24]
    RE24_n: np.ndarray  # [24] int
    dRE24: np.ndarray  # [8, 24]
    dRE24_n: np.ndarray  # [8, 24] int
    n_half_innings: int
    n_half_innings_excluded: int
    n_plate_appearances: int
    n_terminal_pa: int
    mean_delta_all: float  # 종결 타석 전체 Δ 평균 (진단, ≈0 기대)


def compute(df: pd.DataFrame, *, exclude_last_half_inning: bool = True) -> RE24Result:
    df = df.sort_values(SORT, kind="stable").reset_index(drop=True)
    df["base_out"] = base_out_id(df["outs_when_up"].to_numpy(), df["on_1b"].to_numpy(), df["on_2b"].to_numpy(), df["on_3b"].to_numpy())
    df["oid"] = outcome_ids(df["description"].to_numpy(dtype=object), df["events"].to_numpy(dtype=object))

    # 타석 첫·마지막 투구
    pa_key = ["game_pk", "at_bat_number"]
    first = df.groupby(pa_key, sort=False).head(1).set_index(pa_key)
    last = df.groupby(pa_key, sort=False).tail(1).set_index(pa_key)
    pa = pd.DataFrame(
        {
            "inning": first["inning"], "inning_topbot": first["inning_topbot"],
            "b_start": first["base_out"], "score_start": first["bat_score"],
            "b_end": last["base_out"], "oid": last["oid"], "score_before": last["bat_score"], "score_after": last["post_bat_score"],
        }
    ).reset_index()
    pa = pa.sort_values(pa_key, kind="stable").reset_index(drop=True)

    # 하프이닝: 끝 점수, 경기 마지막 하프이닝 여부
    half = pa.groupby(HALF, sort=False)
    pa["inning_end_score"] = half["score_after"].transform("max")
    order = pa.groupby("game_pk", sort=False)["inning"].transform("max")
    last_half = (pa["inning"] == order) & (
        pa["inning_topbot"] == pa.groupby("game_pk", sort=False)["inning_topbot"].transform(lambda s: s.iloc[-1])
    )
    n_half = int(half.ngroups)
    keep = ~last_half if exclude_last_half_inning else pd.Series(True, index=pa.index)
    n_excl = int(pa.loc[~keep].groupby(HALF).ngroups)
    pa = pa.loc[keep].reset_index(drop=True)

    # RE24
    runs_to_end = (pa["inning_end_score"] - pa["score_start"]).to_numpy(dtype=np.float64)
    b0 = pa["b_start"].to_numpy()
    RE24_n = np.bincount(b0, minlength=N_BASE_OUT).astype(np.int64)
    RE24 = np.bincount(b0, weights=runs_to_end, minlength=N_BASE_OUT) / np.maximum(RE24_n, 1)

    # dRE24: 다음 타석이 같은 하프이닝이면 그 시작 상태의 RE, 아니면 0
    same_half = (pa[HALF].shift(-1) == pa[HALF]).all(axis=1).to_numpy()
    next_b = pa["b_start"].shift(-1).fillna(0).astype(int).to_numpy()
    re_next = np.where(same_half, RE24[next_b], 0.0)
    runs_play = (pa["score_after"] - pa["score_before"]).to_numpy(dtype=np.float64)
    b_end = pa["b_end"].to_numpy()
    delta = re_next + runs_play - RE24[b_end]
    oid = pa["oid"].to_numpy()
    term = (oid >= 0) & is_terminal(np.maximum(oid, 0))
    t_idx = {o: i for i, o in enumerate(TERMINAL)}
    rows = np.array([t_idx.get(o, -1) for o in oid[term]])
    cols = b_end[term]
    flat = rows * N_BASE_OUT + cols
    dn = np.bincount(flat, minlength=N_TERMINAL * N_BASE_OUT).reshape(N_TERMINAL, N_BASE_OUT)
    dsum = np.bincount(flat, weights=delta[term], minlength=N_TERMINAL * N_BASE_OUT).reshape(N_TERMINAL, N_BASE_OUT)
    with np.errstate(invalid="ignore"):
        dRE24 = np.where(dn > 0, dsum / np.maximum(dn, 1), np.nan)

    return RE24Result(
        RE24=RE24, RE24_n=RE24_n, dRE24=dRE24, dRE24_n=dn.astype(np.int64),
        n_half_innings=n_half, n_half_innings_excluded=n_excl,
        n_plate_appearances=int(len(pa)), n_terminal_pa=int(term.sum()),
        mean_delta_all=float(delta[term].mean()),
    )


def state_label(b: int) -> str:
    o, r1, r2, r3 = decode_base_out(b)
    runners = "".join(ch if r else "-" for ch, r in (("1", r1), ("2", r2), ("3", r3)))
    return f"{o}out {runners}"


def markdown_tables(r: RE24Result) -> str:
    lines = ["## RE24 (행 = 주자, 열 = 아웃)", "", "| 주자 | 0아웃 | 1아웃 | 2아웃 |", "|---|---|---|---|"]
    for m in range(8):
        runners = "".join(ch if (m >> i) & 1 else "-" for i, ch in enumerate("123"))
        lines.append(f"| {runners} | " + " | ".join(f"{r.RE24[o * 8 + m]:.3f} (n={r.RE24_n[o * 8 + m]:,})" for o in range(3)) + " |")
    lines += ["", "## dRE24 (행 = 종결 결과, 열 = base_out_id 0..23)", "", "| 결과 | " + " | ".join(state_label(b) for b in range(N_BASE_OUT)) + " |", "|---|" + "---|" * N_BASE_OUT]
    for i, o in enumerate(TERMINAL):
        lines.append(f"| {OUTCOME_NAMES[o]} | " + " | ".join(f"{r.dRE24[i, b]:+.3f}" for b in range(N_BASE_OUT)) + " |")
    lines += ["", "## dRE24 셀 표본 수", "", "| 결과 | 최소 n | 최소 셀 | 합 |", "|---|---|---|---|"]
    for i, o in enumerate(TERMINAL):
        j = int(r.dRE24_n[i].argmin())
        lines.append(f"| {OUTCOME_NAMES[o]} | {r.dRE24_n[i, j]:,} | {state_label(j)} | {r.dRE24_n[i].sum():,} |")
    return "\n".join(lines) + "\n"
