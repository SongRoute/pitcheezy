"""계약 테스트용 소형 유효 텐서 빌더 (P=2, K=2). 실제 학습 산출물과 같은 규칙으로 만든다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import states as S
from pitcheezy.interfaces import tensor as T
from pitcheezy.interfaces import value as V
from pitcheezy.interfaces.grid import N_ACTIONS


def make_meta(K: int = 2, **over) -> dict:
    m = {
        "spec_version": T.SPEC_VERSION,
        "data_version": "test-fixture",
        "season_window": "2023-2024",
        "holdout_split": "season:2025",
        "model_arch": "count",
        "seed": 0,
        "train_commit": "0000000",
        "K": K,
        "cluster_file_version": "test",
        "pitch_type_map_version": "v1",
        "repertoire_min_pitches": 100,
        "row_sum_tol": 1e-5,
        "holdout_nll": None,
        "holdout_ece": None,
        "holdout_ece_hr": None,
        "excluded_pitchers": [],
    }
    m.update(over)
    return m


def make_tensor(n_pitchers: int = 2, K: int = 2, seed: int = 0, invalid_frac: float = 0.3) -> T.TransitionTensor:
    rng = np.random.default_rng(seed)
    S_ = S.n_states(K)
    shape = (n_pitchers, S_, N_ACTIONS, O.N_OUTCOMES)
    P = rng.random(shape, dtype=np.float32)
    # 규칙 마스크 적용 후 행 정규화
    cid = S.decode_state(np.arange(S_), K)[0]
    allowed = O.rule_mask_table()[cid]  # [S, O]
    P *= allowed[None, :, None, :]
    valid = rng.random(shape[:3]) >= invalid_frac
    P[~valid] = 0
    row = P.sum(-1, keepdims=True, dtype=np.float64)
    P = np.where(valid[..., None], P / np.where(row > 0, row, 1), 0).astype(np.float32)
    n_obs = np.where(valid, rng.integers(1, 500, shape[:3]), 0).astype(np.int32)
    pitchers = pd.DataFrame(
        {
            "pitcher_idx": np.arange(n_pitchers, dtype=np.int32),
            "mlbam_id": np.arange(100000, 100000 + n_pitchers, dtype=np.int64),
            "name": [f"Pitcher {i}" for i in range(n_pitchers)],
            "n_pitches_train": np.full(n_pitchers, 2500, dtype=np.int32),
        }
    )
    return T.TransitionTensor(P=P, valid=valid, n_obs=n_obs, pitchers=pitchers, meta=make_meta(K))


def make_value(t: T.TransitionTensor, seed: int = 0) -> V.ValueBundle:
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=t.valid.shape).astype(np.float32)
    Vv = np.where(t.valid, Q, -np.inf).max(-1)
    Vv = np.where(np.isfinite(Vv), Vv, 0).astype(np.float32)
    logits = np.where(t.valid, Q.astype(np.float64), -np.inf)
    pol = np.exp(logits - np.nanmax(np.where(np.isfinite(logits), logits, np.nan), axis=-1, initial=-np.inf, keepdims=True))
    pol = np.nan_to_num(pol)
    s = pol.sum(-1, keepdims=True)
    pol = np.where(s > 0, pol / np.where(s > 0, s, 1), 0).astype(np.float32)
    meta = {
        "transition_dir": "runs/test/s0/transition",
        "transition_sha256": {},
        "re24_version": "test",
        "dre24_version": "test",
        "terminal_reward": "dRE24[outcome, base_out]",
        "in_play_reward": "league_mean",
        "gamma": 1,
        "relax": {"method": "softmax", "temperature": 1.0},
        "lookup_mode": "snap",
        "seed": 0,
        "train_commit": "0000000",
    }
    return V.ValueBundle(Q=Q, V=Vv, policy=pol, meta=meta)


@pytest.fixture
def small_tensor() -> T.TransitionTensor:
    return make_tensor()


@pytest.fixture
def small_value(small_tensor) -> V.ValueBundle:
    return make_value(small_tensor)
