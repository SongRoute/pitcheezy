"""DR OPE 계약: q̂=0 이면 IPS 와 항등, 참 모델이면 정확·저분산, 결과 키 집합.

합성 MDP (K=1, 투수 2, 상태별 valid 행동 4개) 에서 π_b 로 타석 4000 개를 굴린 뒤,
같은 데이터로 π_e 를 평가한다. 참값은 VI.policy_evaluation(P_true, π_e) 의 시작 상태 평균.
"""

import numpy as np
import pandas as pd
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.grid import N_ACTIONS
from pitcheezy.interfaces.states import count_id, decode_state, n_states, state_id
from pitcheezy.ope import dr as DR
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI

K = 1
N_P = 2
ACTS = np.array([0, 40, 100, 180])  # 상태마다 이 4개만 valid (작게 → 빠르게)
BASE_OUT = np.array([0, 3, 9, 17])  # 시작 주자·아웃 상태 몇 개
CLIP = 20.0


@pytest.fixture(scope="module")
def world():
    """(P_true, valid, R, nxt, pb, pe, df, sid, pe_logged, pb_logged) — 합성 MDP + π_b 로 굴린 로그."""
    rng = np.random.default_rng(0)
    S_ = n_states(K)
    P = np.zeros((N_P, S_, N_ACTIONS, O.N_OUTCOMES), dtype=np.float32)
    valid = np.zeros((N_P, S_, N_ACTIONS), dtype=bool)
    allowed = O.rule_mask_table()[decode_state(np.arange(S_), K)[0]]  # [S, O]
    for p in range(N_P):
        for s in range(S_):
            row = rng.dirichlet(np.full(O.N_OUTCOMES, 0.3), len(ACTS)) * allowed[s][None]  # 뾰족한 행 → q̂ 가 실제로 정보를 준다
            row[:, list(O.TERMINAL)] *= 0.25  # 타석이 2~4구는 가게 (궤적 DR 이 의미를 갖도록)
            row[:, O.FOUL] *= 0.4             # 파울 자기전이로 너무 길어지지 않게
            P[p, s, ACTS] = (row / row.sum(-1, keepdims=True)).astype(np.float32)
            valid[p, s, ACTS] = True
    dre = rng.normal(0, 0.4, (8, 24))
    R = VI.reward_table(dre, K)
    nxt = VI.next_state_table(K)
    Q, _, _, _ = VI.value_iteration(P, valid, R, nxt)
    pb = VI.relax(Q, valid, method="softmax", temperature=1.0).astype(np.float64)   # 거의 균등
    pe = VI.relax(Q, valid, method="softmax", temperature=0.05).astype(np.float64)  # 훨씬 뾰족 → IPS 분산 큼
    # --- π_b 로 타석 시뮬레이션
    n_pa, rows = 4000, []
    cum_b = pb.cumsum(-1)
    for j in range(n_pa):
        p = int(rng.integers(N_P))
        s = state_id(count_id(0, 0), int(rng.choice(BASE_OUT)), 0, K)
        for t in range(30):
            a = int(np.searchsorted(cum_b[p, s], rng.random() * cum_b[p, s, -1]))
            o = int(np.searchsorted(P[p, s, a].cumsum(), rng.random() * P[p, s, a].sum()))
            rows.append((j // 20, j, t + 1, p, a, s, R[s, o] if nxt[s, o] < 0 else 0.0, nxt[s, o] < 0))
            if nxt[s, o] < 0 or t == 29:
                break
            s = int(nxt[s, o])
    df = pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "pitcher_idx", "action_id", "state_id", "rew", "term"])
    sid = df["state_id"].to_numpy()
    pe_logged = IPS.logged_probs(df, sid, pe)
    pb_logged = IPS.logged_probs(df, sid, pb)
    pa = df.groupby(["game_pk", "at_bat_number"], sort=True).agg(r=("rew", "sum")).reset_index()
    first = df.groupby(["game_pk", "at_bat_number"], sort=True).head(1)
    return dict(P=P, valid=valid, R=R, nxt=nxt, pb=pb, pe=pe, df=df, sid=sid, pe_logged=pe_logged, pb_logged=pb_logged,
                r=pa["r"].to_numpy(), games=pa["game_pk"].to_numpy(),
                f_p=first["pitcher_idx"].to_numpy(), f_s=first["state_id"].to_numpy())


def test_dr_reduces_to_ips_when_q_is_zero(world):
    w = world
    rho, has = IPS.pitch_ratios(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    S_, nP = n_states(K), N_P
    z_q, z_v = np.zeros((nP, S_, N_ACTIONS)), np.zeros((nP, S_))
    ot = DR.onestep_dr_terms(w["df"], w["sid"], rho, has, z_q, z_v)
    pw = IPS.pa_weights(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    assert (ot[["game_pk", "at_bat_number"]].to_numpy() == pw[["game_pk", "at_bat_number"]].to_numpy()).all()
    assert ot["sum_rho"].to_numpy() == pytest.approx(pw["w_onestep"].to_numpy())
    assert DR.estimate_dr1(ot, w["r"])["snips"] == pytest.approx(IPS.estimate(pw["w_onestep"].to_numpy(), w["r"])["snips"], abs=1e-12)
    tt = DR.traj_dr_terms(w["df"], w["sid"], rho, has, z_q, z_v, r_pa=w["r"])
    assert tt["w_last"].to_numpy() == pytest.approx(pw["w_traj"].to_numpy())
    assert float(tt["dr_plain"].mean()) == pytest.approx(IPS.estimate(pw["w_traj"].to_numpy(), w["r"])["ips"], abs=1e-12)


def test_traj_dr_accurate_and_lower_variance(world):
    w = world
    rho, has = IPS.pitch_ratios(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    V_e = VI.policy_evaluation(w["P"], w["pe"], w["R"], w["nxt"])
    q_e = DR.q_from_v(w["P"], w["R"], w["nxt"], V_e)
    truth = float(V_e[w["f_p"], w["f_s"]].mean())
    tt = DR.traj_dr_terms(w["df"], w["sid"], rho, has, q_e, V_e, r_pa=w["r"])
    dr_plain = tt["dr_plain"].to_numpy()
    b_dr = DR.bootstrap_traj_dr(dr_plain, w["games"], n_boot=300, seed=0)
    assert abs(float(dr_plain.mean()) - truth) < 3 * b_dr["boot_std"]
    pw = IPS.pa_weights(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    b_ips = IPS.bootstrap(pw["w_traj"].to_numpy(), w["r"], w["games"], n_boot=300, seed=0)
    assert b_dr["boot_std"] < b_ips["boot_std"]


def test_onestep_dr_lower_variance_and_keys(world):
    w = world
    rho, has = IPS.pitch_ratios(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    support = w["valid"]
    q_b = DR.behavior_q(w["P"], w["pb"], support, w["R"], w["nxt"])
    v_e_b, V_e, q_e = DR.dr_inputs(w["P"], w["pe"], q_b, w["R"], w["nxt"])
    ot = DR.onestep_dr_terms(w["df"], w["sid"], rho, has, q_b, v_e_b)
    est = DR.estimate_dr1(ot, w["r"])
    est.update(DR.bootstrap_dr1(ot, w["r"], w["games"], n_boot=300, seed=0))
    pw = IPS.pa_weights(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    ref = IPS.bootstrap(pw["w_onestep"].to_numpy(), w["r"], w["games"], n_boot=300, seed=0)
    assert np.isfinite(est["snips"]) and est["boot_std"] <= ref["boot_std"]
    # 키 집합: 결과 dict 가 aggregate()·log_wandb() 가 읽는 키를 모두 갖는다
    base = {"ips", "snips", "ess", "ess_frac", "w_max", "w_mean", "n_pa", "frac_zero_w", "ci_low", "ci_high", "boot_std", "n_boot", "n_games", "seed"}
    assert set(est) == base | {"dm", "corr"}
    tt = DR.traj_dr_terms(w["df"], w["sid"], rho, has, q_e, V_e)
    dr_plain, dr_wdr = DR.traj_dr_values(tt, w["r"])
    est_t = DR.estimate_traj_dr(dr_wdr, dr_plain, tt["w_last"].to_numpy())
    est_t.update(DR.bootstrap_traj_dr(dr_wdr, w["games"], n_boot=100, seed=0))
    assert set(est_t) == base
    for d in (est, est_t):
        assert all(np.isfinite(v) for k, v in d.items() if isinstance(v, float))


def test_shapes_and_no_nan(world):
    w = world
    rho, has = IPS.pitch_ratios(w["df"], w["pe_logged"], w["pb_logged"], clip=CLIP)
    q_b = DR.behavior_q(w["P"], w["pb"], w["valid"], w["R"], w["nxt"])
    assert q_b.shape == (N_P, n_states(K), N_ACTIONS) and np.isfinite(q_b).all()
    v_e_b, V_e, q_e = DR.dr_inputs(w["P"], w["pe"], q_b, w["R"], w["nxt"])
    assert v_e_b.shape == V_e.shape == (N_P, n_states(K)) and q_e.shape == q_b.shape
    n_pa = len(w["r"])
    ot = DR.onestep_dr_terms(w["df"], w["sid"], rho, has, q_b, v_e_b)
    tt = DR.traj_dr_terms(w["df"], w["sid"], rho, has, q_e, V_e, r_pa=w["r"])
    assert len(ot) == len(tt) == n_pa
    assert not ot.isna().to_numpy().any() and not tt.isna().to_numpy().any()
    assert (ot["n_dec"].to_numpy() > 0).all() and (tt["w_last"].to_numpy() >= 0).all()
    with pytest.raises(ValueError):
        DR.traj_dr_terms(w["df"], w["sid"], rho, has, q_e, V_e, r_pa=w["r"][:-1])


def test_q_from_v_accepts_action_dependent_nxt():
    """맥락 C>1 이면 next_state_table 이 [S, A, O] 다 → q_from_v 가 2-D 를 행동 축으로 브로드캐스트한 3-D 와 같은 Q 를 내야 한다."""
    rng = np.random.default_rng(7)
    n_p, S_, A, O_, chunk = 3, 9, 4, 6, 2  # 청크 경계도 지나가게 (3 투수 / 청크 2)
    P = rng.dirichlet(np.ones(O_), (n_p, S_, A)).astype(np.float32)
    R = rng.normal(0, 1.0, (S_, O_))
    V = rng.normal(0, 1.0, (n_p, S_))
    nxt = rng.integers(-1, S_, (S_, O_))  # −1 = 종결·불허
    q2 = DR.q_from_v(P, R, nxt, V, chunk=chunk)
    q3 = DR.q_from_v(P, R, np.broadcast_to(nxt[:, None, :], (S_, A, O_)), V, chunk=chunk)
    assert q2.shape == q3.shape == (n_p, S_, A)
    assert np.allclose(q2, q3)
    # 행동마다 다음 상태가 다르면 Q 도 달라진다 (3-D 경로가 정말 행동 축을 쓴다)
    nxt3 = rng.integers(0, S_, (S_, A, O_))
    assert not np.allclose(DR.q_from_v(P, R, nxt3, V, chunk=chunk), q3)
