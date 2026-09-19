"""정책·OPE 계약: VI 항등식, 완화 정책, IPS 추정량, π_b 인수분해, tilt, 시퀀스 맥락(C>1)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitcheezy.data import prepare as PR
from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.grid import N_ACTIONS, decode_action
from pitcheezy.interfaces.states import context_family_of_pitch, count_id, decode_state_full, n_states, state_id
from pitcheezy.interfaces.validate import validate_transition
from pitcheezy.ope import behavior as BH
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI
from pitcheezy.transition import count as TC
from pitcheezy.transition.smoothing import smooth_hierarchical

sys.path.insert(0, str(Path(__file__).parent / "interfaces"))
from conftest import make_tensor  # noqa: E402  계약 테스트용 소형 텐서 빌더


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


# ---------------------------------------------------------------- 시퀀스 맥락 (C>1)
CTX_META = {"data_version": "test", "season_window": "2024-2024", "holdout_split": "none", "seed": 0, "train_commit": "0000000",
            "cluster_file_version": "K1-none", "pitch_type_map_version": "v1"}


def _fixture_prepared(tmp_path):
    """tests/fixtures 의 실제 Statcast 행 → prepare_season 산출 표 (투수 2명, 196구)."""
    raw = pd.read_parquet(Path(__file__).parent / "fixtures" / "statcast_2024_d20260911-s2325_p2.parquet")
    d = tmp_path / "raw" / "vtest"
    d.mkdir(parents=True)
    raw.to_parquet(d / "statcast_2024.parquet", index=False)
    index = {int(m): i for i, m in enumerate(sorted(raw["pitcher"].unique()))}
    return PR.prepare_season(tmp_path, "vtest", 2024, index), index


def test_next_state_table_action_dependent_context():
    K, C = 1, 4
    nxt = VI.next_state_table(K, C)
    S_ = n_states(K, C)
    assert nxt.shape == (S_, N_ACTIONS, O.N_OUTCOMES) and nxt.dtype == np.int64
    c, b, k, _ = decode_state_full(np.arange(S_), K, C)
    fam = context_family_of_pitch(decode_action(np.arange(N_ACTIONS))[0])  # [A] 늘 1..3
    assert set(np.unique(fam).tolist()) == {1, 2, 3}
    nc = np.full((12, O.N_OUTCOMES), -1)  # 카운트 규칙표
    for ci in range(12):
        for o in O.NONTERMINAL:
            try:
                nc[ci, o] = O.next_count(ci, o)
            except ValueError:
                pass
    ok = nxt >= 0
    assert (ok == (nc[c][:, None, :] >= 0)).all()  # 종결·규칙 불가 = −1, 나머지는 전부 다음 상태
    assert (nxt[:, :, list(O.TERMINAL)] == -1).all()
    c2, b2, k2, x2 = decode_state_full(np.where(ok, nxt, 0), K, C)
    big = lambda a: np.broadcast_to(a, nxt.shape)  # noqa: E731
    assert (x2[ok] == big(fam[None, :, None])[ok]).all()  # 다음 맥락 = 행동의 구종 계열
    assert (c2[ok] == big(nc[c][:, None, :])[ok]).all()
    assert (b2[ok] == big(b[:, None, None])[ok]).all() and (k2[ok] == big(k[:, None, None])[ok]).all()
    # 2스트라이크 파울: 카운트·주자아웃은 그대로, 맥락만 행동을 따라간다
    s = state_id(count_id(3, 2), 7, 0, K, 2, C)
    assert decode_state_full(int(nxt[s, 0, O.FOUL]), K, C) == (count_id(3, 2), 7, 0, 1)
    assert nxt[s, 0, O.BALL] == -1


def test_value_iteration_and_policy_eval_accept_3d_next_state():
    """3차원 next 를 2차원의 브로드캐스트로 만들면 Q·V 가 완전히 같아야 한다 (C=1 경로 불변 확인)."""
    t = make_tensor(n_pitchers=2, K=2)
    d = np.zeros((8, 24)); d[O.TERMINAL.index(O.HR), :] = 1.0; d[O.TERMINAL.index(O.K), :] = -0.3
    R = VI.reward_table(d, 2)
    nxt2 = VI.next_state_table(2)
    nxt3 = np.broadcast_to(nxt2[:, None, :], (nxt2.shape[0], N_ACTIONS, O.N_OUTCOMES)).copy()
    Q2, V2, _, _ = VI.value_iteration(t.P, t.valid, R, nxt2)
    Q3, V3, _, _ = VI.value_iteration(t.P, t.valid, R, nxt3)
    np.testing.assert_allclose(Q3, Q2, atol=1e-12)
    np.testing.assert_allclose(V3, V2, atol=1e-12)
    pol = VI.relax(Q2, t.valid, method="greedy")
    np.testing.assert_allclose(VI.policy_evaluation(t.P, pol, R, nxt3), VI.policy_evaluation(t.P, pol, R, nxt2), atol=1e-12)


def test_count_fit_with_context_and_c1_unchanged(tmp_path):
    df, index = _fixture_prepared(tmp_path)
    pit = pd.DataFrame({"pitcher_idx": np.arange(2, dtype=np.int32), "mlbam_id": np.array(sorted(index), dtype=np.int64),
                        "name": ["a", "b"], "n_pitches_train": np.full(2, len(df) // 2, dtype=np.int32)})
    kw = dict(alpha=5.0, repertoire_min=1, meta=CTX_META)
    t4 = TC.fit(df, PR.state_ids(df, 1, C=4, context_kind="prev_pitch_family"), pit, 1, C=4, context_kind="prev_pitch_family", **kw)
    assert validate_transition(t4) == []
    assert t4.C == 4 and t4.P.shape == (2, n_states(1, 4), N_ACTIONS, O.N_OUTCOMES)
    assert t4.meta["context_kind"] == "prev_pitch_family" and len(t4.states) == n_states(1, 4)
    assert int(t4.n_obs.sum()) == int(((df["action_id"] >= 0) & (df["outcome_id"] >= 0)).sum())
    sid1 = PR.state_ids(df, 1)
    t1 = TC.fit(df, sid1, pit, 1, **kw)  # C 인자 없이 = 변경 전 호출
    t1c = TC.fit(df, sid1, pit, 1, C=1, context_kind=None, **kw)
    assert validate_transition(t1) == [] and t1.C == 1
    assert np.array_equal(t1.P, t1c.P) and np.array_equal(t1.valid, t1c.valid) and np.array_equal(t1.n_obs, t1c.n_obs)


def test_fit_behavior_with_context():
    df = _toy_pitches()
    rng = np.random.default_rng(0)
    sid = state_id(rng.integers(0, 12, len(df)), 0, 0, 1, rng.integers(0, 4, len(df)), 4)
    pb = BH.fit_behavior(df, sid, 2, 1, alpha=5.0, C=4)
    assert pb.shape == (2, n_states(1, 4), N_ACTIONS) and np.allclose(pb.sum(-1), 1.0, atol=1e-5)
    assert (pb > 0).all()
    pbl = BH.behavior_logged_crossfit(df, sid, 2, 1, alpha=5.0, n_folds=3, C=4)
    assert np.isnan(pbl).sum() == 0 and (pbl > 0).all()
