"""결과 축 — docs/interface-spec.md 자산 1 "결과".

outcome_id 0..10: 비종결 {볼, 스트라이크, 파울} + 종결 {K, BB, HBP, 1B, 2B, 3B, HR, 인플레이 아웃}
다음 카운트는 count_rule 로 결정: 볼→b+1, 스트라이크→s+1, 파울→s<2면 s+1 / s=2면 유지

Statcast 매핑 (스펙 + 데이터에서 실제로 보인 값)
  종결은 events 로, 비종결은 description 으로 판정한다. 주자 사건(도루사 등)은 events 가 있어도 투구 결과는 description.
  볼        ← ball, blocked_ball, automatic_ball(피치클록), intent_ball, pitchout
  스트라이크 ← called_strike, swinging_strike, swinging_strike_blocked, foul_tip, bunt_foul_tip, missed_bunt, automatic_strike
              (foul_tip 은 s=2 면 events=strikeout 이라 K 로 먼저 잡힌다)
  파울      ← foul, foul_bunt, foul_pitchout (foul_bunt 는 s=2 면 events=strikeout → K)
  K   ← events strikeout, strikeout_double_play
  BB  ← events walk, intent_walk
  HBP ← description hit_by_pitch
  1B/2B/3B/HR ← description hit_into_play + events single/double/triple/home_run
  인플레이 아웃 ← description hit_into_play + 그 밖의 모든 events (field_out, force_out, double_play, sac_fly, sac_bunt,
              fielders_choice, triple_play, field_error, catcher_interf 등. ASM-7 "리그 평균으로 뭉갬")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .states import N_COUNTS, N_STRIKES, count_id, decode_count

OUTCOME_NAMES: tuple[str, ...] = ("ball", "strike", "foul", "K", "BB", "HBP", "1B", "2B", "3B", "HR", "in_play_out")
N_OUTCOMES = len(OUTCOME_NAMES)  # 11
BALL, STRIKE, FOUL, K, BB, HBP, SINGLE, DOUBLE, TRIPLE, HR, IN_PLAY_OUT = range(N_OUTCOMES)

NONTERMINAL: tuple[int, ...] = (BALL, STRIKE, FOUL)
TERMINAL: tuple[int, ...] = (K, BB, HBP, SINGLE, DOUBLE, TRIPLE, HR, IN_PLAY_OUT)
N_TERMINAL = len(TERMINAL)  # 8. dRE24 의 행 축 = TERMINAL 순서

COUNT_RULE: dict[int, Optional[str]] = {BALL: "ball", STRIKE: "strike", FOUL: "foul"}
OUTCOMES_COLUMNS = ("outcome_id", "name", "terminal", "count_rule", "statcast")

BALL_DESCRIPTIONS = frozenset({"ball", "blocked_ball", "automatic_ball", "intent_ball", "pitchout"})
STRIKE_DESCRIPTIONS = frozenset(
    {"called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip", "missed_bunt", "automatic_strike"}
)
FOUL_DESCRIPTIONS = frozenset({"foul", "foul_bunt", "foul_pitchout"})
IN_PLAY_DESCRIPTION = "hit_into_play"
HBP_DESCRIPTION = "hit_by_pitch"
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
BB_EVENTS = frozenset({"walk", "intent_walk"})
HIT_EVENTS: dict[str, int] = {"single": SINGLE, "double": DOUBLE, "triple": TRIPLE, "home_run": HR}

_STATCAST_DOC = {
    BALL: "description: " + ",".join(sorted(BALL_DESCRIPTIONS)),
    STRIKE: "description: " + ",".join(sorted(STRIKE_DESCRIPTIONS)),
    FOUL: "description: " + ",".join(sorted(FOUL_DESCRIPTIONS)),
    K: "events: " + ",".join(sorted(K_EVENTS)),
    BB: "events: " + ",".join(sorted(BB_EVENTS)),
    HBP: f"description: {HBP_DESCRIPTION}",
    SINGLE: "hit_into_play + events: single",
    DOUBLE: "hit_into_play + events: double",
    TRIPLE: "hit_into_play + events: triple",
    HR: "hit_into_play + events: home_run",
    IN_PLAY_OUT: "hit_into_play + events: 그 밖의 전부",
}


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def outcome_id(description: Optional[str], events: Optional[str]) -> Optional[int]:
    """Statcast (description, events) → outcome_id. 알 수 없는 description 은 None (호출 쪽에서 세고 걸러야 함)."""
    ev = None if _is_missing(events) else str(events)
    de = None if _is_missing(description) else str(description)
    if ev in K_EVENTS:
        return K
    if ev in BB_EVENTS:
        return BB
    if de == HBP_DESCRIPTION:
        return HBP
    if de == IN_PLAY_DESCRIPTION:
        return HIT_EVENTS.get(ev, IN_PLAY_OUT)
    if de in BALL_DESCRIPTIONS:
        return BALL
    if de in STRIKE_DESCRIPTIONS:
        return STRIKE
    if de in FOUL_DESCRIPTIONS:
        return FOUL
    return None


def outcome_ids(descriptions, events) -> np.ndarray:
    """배열 버전. 알 수 없으면 −1."""
    d = np.asarray(descriptions, dtype=object).ravel()
    e = np.asarray(events, dtype=object).ravel()
    out = np.fromiter((o if (o := outcome_id(a, b)) is not None else -1 for a, b in zip(d, e)), dtype=np.int64, count=len(d))
    return out.reshape(np.shape(descriptions))


def is_terminal(oid) -> np.ndarray | bool:
    o = np.asarray(oid, dtype=np.int64)
    out = o >= K
    return bool(out) if out.ndim == 0 else out


def next_count(cid: int, oid: int) -> Optional[int]:
    """비종결 결과 뒤 count_id. 종결이면 None. 규칙상 불가능한 조합(3볼에 볼 등)은 ValueError."""
    b, s = decode_count(cid)
    if oid == BALL:
        if b == 3:
            raise ValueError("3볼에 '볼' 은 BB 여야 함")
        return count_id(b + 1, s)
    if oid == STRIKE:
        if s == 2:
            raise ValueError("2스트라이크에 '스트라이크' 는 K 여야 함")
        return count_id(b, s + 1)
    if oid == FOUL:
        return count_id(b, min(s + 1, N_STRIKES - 1))
    if is_terminal(oid):
        return None
    raise ValueError(f"outcome_id 범위 밖: {oid}")


def rule_mask(cid) -> np.ndarray:
    """카운트 규칙 마스크 bool[O] (True = 허용). 스칼라 cid → [O], 배열 → [..., O].

    3볼에 '볼' 불가(→BB), 2스트에 '스트라이크' 불가(→K). 같은 규칙의 뒷면으로 BB 는 3볼, K 는 2스트에서만.
    """
    c = np.asarray(cid, dtype=np.int64)
    b, s = decode_count(c)
    b, s = np.asarray(b), np.asarray(s)
    m = np.ones(c.shape + (N_OUTCOMES,), dtype=bool)
    m[..., BALL] = b < 3
    m[..., BB] = b == 3
    m[..., STRIKE] = s < 2
    m[..., K] = s == 2
    return m


def rule_mask_table() -> np.ndarray:
    """[N_COUNTS, O] 전체 표."""
    return rule_mask(np.arange(N_COUNTS))


def outcomes_table() -> pd.DataFrame:
    """outcomes.parquet 내용."""
    return pd.DataFrame(
        {
            "outcome_id": np.arange(N_OUTCOMES, dtype=np.int32),
            "name": list(OUTCOME_NAMES),
            "terminal": [i in TERMINAL for i in range(N_OUTCOMES)],
            "count_rule": [COUNT_RULE.get(i) for i in range(N_OUTCOMES)],
            "statcast": [_STATCAST_DOC[i] for i in range(N_OUTCOMES)],
        }
    )
