"""정책·OPE 계약: VI 항등식, 완화 정책, IPS 추정량, π_b 인수분해, tilt."""

import numpy as np
import pandas as pd
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.grid import N_ACTIONS
from pitcheezy.interfaces.states import count_id, n_states, state_id
from pitcheezy.ope import behavior as BH
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI
from pitcheezy.transition.smoothing import smooth_hierarchical


def test_next_state_table_and_reward_sign():
    nxt = VI.next_state_table(1)
    s = state_id(count_id(0, 0), 5, 0, 1)
    assert nxt[s, O.BALL] == state_id(count_id(1, 0), 5, 0, 1)
    assert nxt[s, O.STRIKE] == state_id(count_id(0, 1), 5, 0, 1)
    assert nxt[s, O.K] == -1
    s32 = state_id(count_id(3, 2), 5, 0, 1)
    assert nxt[s32, O.FOUL] == s32 and nxt[s32, O.BALL] == -1
    d = np.zeros((8, 24)); d[O.TERMINAL.index(O.HR), :] = 1.0
    R = VI.reward_table(d, 1)
    assert R[s, O.HR] == -1.0 and R[s, O.BALL] == 0.0


def test_value_iteration_deterministic_chain():
    """단일 투수, 행동 1개만 valid: 0-0 에서 스트라이크 확률 1 → 0-1 → 0-2 → K. K 보상 +0.3 → V(0-0) = 0.3."""
    K = 1; S_ = n_states(K)
    P = np.zeros((1, S_, N_ACTIONS, O.N_OUTCOMES), dtype=np.float32)
    valid = np.zeros((1, S_, N_ACTIONS), dtype=bool)
    for c in range(12):
        s = state_id(c, 0, 0, K)
        b, st = divmod(c, 3)
        P[0, s, 0, O.K if st == 2 else O.STRIKE] = 1.0
        valid[0, s, 0] = True
    d = np.zeros((8, 24)); d[O.TERMINAL.index(O.K), 0] = -0.3
    R = VI.reward_table(d, K)
    Q, V, it, delta = VI.value_iteration(P, valid, R, VI.next_state_table(K))
    assert V[0, state_id(count_id(0, 0), 0, 0, K)] == pytest.approx(0.3)
    assert V[0, state_id(count_id(0, 2), 0, 0, K)] == pytest.approx(0.3)
    assert V[0, state_id(count_id(0, 0), 1, 0, K)] == 0.0  # 다른 주자 상태는 valid 없음 → 0
    Vpi = VI.policy_evaluation(P, VI.relax(Q, valid, method="greedy"), R, VI.next_state_table(K))
    assert Vpi[0, state_id(count_id(0, 0), 0, 0, K)] == pytest.approx(0.3)


def test_relax_variants():
    Q = np.array([[[1.0, 0.5, 0.0, -1.0]]])
    valid = np.array([[[True, True, True, False]]])
    g = VI.relax(Q, valid, method="greedy"); assert g[0, 0].tolist() == [1, 0, 0, 0]
    u = VI.relax(Q, valid, method="uniform"); assert u[0, 0].tolist() == pytest.approx([1 / 3] * 3 + [0])
    k = VI.relax(Q, valid, method="topk", top_k=2); assert k[0, 0].tolist() == pytest.approx([0.5, 0.5, 0, 0])
    sm = VI.relax(Q, valid, method="softmax", temperature=0.5)
    assert sm[0, 0, 3] == 0 and sm[0, 0].sum() == pytest.approx(1) and sm[0, 0, 0] > sm[0, 0, 1] > sm[0, 0, 2]


def test_ips_estimators_and_bootstrap():
    df = pd.DataFrame({"game_pk": [1, 1, 1, 2, 2], "at_bat_number": [1, 1, 2, 1, 1], "pitch_number": [1, 2, 1, 1, 2],
                       "pitcher_idx": 0, "action_id": [3, 5, 4, -1, 7]})
    pe = np.array([0.2, 0.1, 0.5, np.nan, 0.0]); pb = np.array([0.1, 0.1, 0.25, np.nan, 0.2])
    w = IPS.pa_weights(df, pe, pb, clip=None).sort_values(["game_pk", "at_bat_number"])
    assert w["w_traj"].tolist() == pytest.approx([2.0, 2.0, 0.0])
    assert w["w_onestep"].tolist() == pytest.approx([3.0, 2.0, 0.0])
    assert w["n_decisions"].tolist() == [2, 1, 1] and w["n_zero"].tolist() == [0, 0, 1]
    wc = IPS.pa_weights(df, pe, pb, clip=1.5)
    assert wc.sort_values(["game_pk", "at_bat_number"])["w_traj"].tolist() == pytest.approx([1.5, 1.5, 0.0])
    r = np.array([1.0, -1.0, 5.0])
    est = IPS.estimate(w["w_traj"].to_numpy(), r)
    assert est["snips"] == pytest.approx(0.0) and est["ips"] == pytest.approx(0.0) and est["frac_zero_w"] == pytest.approx(1 / 3)
    b = IPS.bootstrap(np.ones(3), r, np.array([1, 1, 2]), n_boot=200, seed=0)
    assert b["ci_low"] <= r.mean() <= b["ci_high"]


def _toy_pitches(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"game_pk": rng.integers(1, 20, n), "at_bat_number": rng.integers(1, 60, n), "pitch_number": 1,
                         "pitcher_idx": rng.integers(0, 2, n), "pitch_id": rng.choice([0, 3, 6], n), "loc_id": rng.integers(0, 25, n)}).assign(
        action_id=lambda d: d.pitch_id * 25 + d.loc_id)


def test_behavior_factored_and_tilt():
    df = _toy_pitches()
    sid = np.zeros(len(df), dtype=int)
    pb = BH.fit_behavior(df, sid, 2, 1, alpha=5.0)
    assert pb.shape == (2, 288, 225) and np.allclose(pb.sum(-1), 1.0, atol=1e-5)
    assert (pb > 0).all()
    # 던진 구종(0,3,6)의 질량이 나머지보다 훨씬 큼
    thrown = pb[0, 0].reshape(9, 25)[[0, 3, 6]].sum(); assert thrown > 0.9
    pbl, pe, pbc, pec = BH.crossfit_logged(df, sid, 2, 1, alpha=5.0, n_folds=3, tilts={"t": (np.zeros((2, 288, 225)), np.ones((2, 288, 225), bool), 1.0)}, groups=IPS.coarse_groups())
    assert (pbc[~np.isnan(pbc)] >= pbl[~np.isnan(pbl)] - 1e-9).all()  # 그룹 질량 ≥ 행동 질량
    assert np.isnan(pbl).sum() == 0 and np.allclose(pe["t"], pbl)  # Q=0 이면 tilt = π_b
    Q = np.zeros((2, 288, 225)); Q[:, :, 0] = 5.0
    tl = BH.tilt(pb, Q, np.ones_like(pb, dtype=bool), 0.1)
    assert tl[0, 0, 0] > 0.99


def test_smoothing_rows_sum_and_mask():
    rng = np.random.default_rng(0)
    n = rng.integers(0, 3, (2, 12, 4, 3))
    cs = np.arange(12); ga = np.array([0, 0, 1, 1])
    mask = np.ones((12, 3), bool); mask[:, 2] = False
    P = smooth_hierarchical(n, count_of_state=cs, group_of_action=ga, n_counts=12, n_groups=2, alpha=1.0, rule_mask=mask)
    assert np.allclose(P.sum(-1), 1.0, atol=1e-6) and (P[..., 2] == 0).all()


def test_coarse_groups_and_coarsen():
    g = IPS.coarse_groups()
    assert g.shape == (225,) and g.max() == 80 and len(np.unique(g)) == 81
    pol = np.full((1, 1, 225), 1 / 225)
    c = IPS.coarsen(pol, g)
    assert np.allclose(c[0, 0], np.bincount(g)[g] / 225)
