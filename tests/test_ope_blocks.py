"""D37: 투수 블록 단위 OPE 재료(blocks.compute_rows) = 통짜 [P,S,A] 경로와 같은 행 단위 값."""

import numpy as np
import pandas as pd
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces.grid import N_ACTIONS
from pitcheezy.interfaces.states import decode_state_full, n_states
from pitcheezy.ope import behavior as BH
from pitcheezy.ope import blocks as BK
from pitcheezy.ope import dr as DR
from pitcheezy.ope import ips as IPS
from pitcheezy.policy import vi as VI

N_P = 3
SPECS = {"tilt_t0.02": ("tilt", 0.02), "softmax_t0.05": ("relax", "softmax", {"temperature": 0.05}), "topk3": ("relax", "topk", {"top_k": 3})}


def _world(C: int, kind: str | None):
    rng = np.random.default_rng(1)
    S_ = n_states(1, C)
    n = 4000
    pitch = rng.integers(0, 4, n)
    loc = rng.integers(0, 25, n)
    df = pd.DataFrame({"game_pk": rng.integers(0, 40, n), "at_bat_number": rng.integers(0, 30, n), "pitcher_idx": rng.integers(0, N_P, n), "pitch_id": pitch, "loc_id": loc})
    df["pitch_number"] = df.groupby(["game_pk", "at_bat_number"]).cumcount() + 1
    df["action_id"] = np.where(rng.random(n) < 0.05, -1, pitch * 25 + loc)
    sid = rng.integers(0, S_, n)
    valid = np.zeros((N_P, S_, N_ACTIONS), dtype=bool)
    valid[0, :, :75] = True; valid[1, :, :100] = True; valid[2, :, 25:100] = True
    allowed = O.rule_mask_table()[decode_state_full(np.arange(S_), 1, C)[0]]
    P = (rng.dirichlet(np.full(O.N_OUTCOMES, 0.5), (N_P, S_, N_ACTIONS)) * allowed[None, :, None, :]).astype(np.float64)
    P = (P / P.sum(-1, keepdims=True) * valid[..., None]).astype(np.float32)
    R = VI.reward_table(rng.normal(0, 0.4, (8, 24)), 1, C=C)
    nxt = VI.next_state_table(1, C, kind)
    Q = VI.value_iteration(P, valid, R, nxt)[0].astype(np.float32)
    rep_ok = np.ones((N_P, N_ACTIONS), dtype=bool); rep_ok[1, 75:100] = False
    return df, sid, valid, P, R, nxt, Q, rep_ok


@pytest.mark.parametrize("C,kind", [(1, None), (4, "prev_pitch_family")])
def test_compute_rows_matches_whole_array_path(C, kind):
    df, sid, valid, P, R, nxt, Q, rep_ok = _world(C, kind)
    groups = IPS.coarse_groups()
    rows = BK.compute_rows(df, sid, N_P, 1, C, specs=SPECS, valid=valid, rep_ok=rep_ok, Q=Q, alpha=5.0, n_folds=3, groups=groups, P=P, R=R, nxt=nxt, with_value=True, with_dr=True, block=2)
    # --- 통짜 경로 (변경 전 stage_ope 의 순서 그대로)
    support = valid & rep_ok[:, None, :]
    tilts = {k: (Q, support, v[1]) for k, v in SPECS.items() if v[0] == "tilt"}
    pb_l, pe_t, pbc_l, pec_t = BH.crossfit_logged(df, sid, N_P, 1, alpha=5.0, n_folds=3, tilts=tilts, groups=groups, C=C)
    np.testing.assert_array_equal(rows.pb, pb_l)
    np.testing.assert_array_equal(rows.pb_coarse, pbc_l)
    assert rows.n_support == int(support.sum()) and rows.n_valid == int(valid.sum())
    pb_full = BH.fit_behavior(df, sid, N_P, 1, alpha=5.0, C=C)
    q_b = DR.behavior_q(P, pb_full, support, R, nxt)
    kw = dict(atol=1e-8, rtol=0, equal_nan=True)
    np.testing.assert_allclose(rows.qb_row, DR.gather_rows(df, sid, q_b, np.zeros(q_b.shape[:2]))[0], **kw)
    for name, spec in SPECS.items():
        pr = rows.policies[name]
        if spec[0] == "tilt":
            pol = BH.tilt(pb_full, Q, support, spec[1])
            pe, pec = pe_t[name], pec_t[name]
        else:
            pol = IPS.restrict_support(VI.relax(Q.astype(np.float64), valid, method=spec[1], **spec[2]), support)
            pe, pec = IPS.logged_probs(df, sid, pol), IPS.logged_probs(df, sid, IPS.coarsen(pol, groups))
        np.testing.assert_array_equal(pr.pe, pe)
        np.testing.assert_array_equal(pr.pe_coarse, pec)
        V_e = VI.policy_evaluation(P, pol, R, nxt)
        v_e_b, _, q_e = DR.dr_inputs(P, pol.astype(np.float32), q_b, R, nxt, V_e=V_e)
        qe_row, V_row = DR.gather_rows(df, sid, q_e, V_e)
        np.testing.assert_allclose(pr.V_row, V_row, **kw)
        np.testing.assert_allclose(pr.qe_row, qe_row, **kw)
        np.testing.assert_allclose(pr.veb_row, DR.gather_rows(df, sid, q_e, v_e_b)[1], **kw)


def test_compute_rows_without_tensor():
    """P 없이(시드 ≠ 0 에서 P.npy 가 지워진 경우) π_e·π_b 행 값만."""
    df, sid, valid, P, R, nxt, Q, rep_ok = _world(1, None)
    rows = BK.compute_rows(df, sid, N_P, 1, 1, specs={"tilt_t0.02": ("tilt", 0.02)}, valid=valid, rep_ok=rep_ok, Q=Q, alpha=5.0, n_folds=3)
    full = BK.compute_rows(df, sid, N_P, 1, 1, specs={"tilt_t0.02": ("tilt", 0.02)}, valid=valid, rep_ok=rep_ok, Q=Q, alpha=5.0, n_folds=3, P=P, R=R, nxt=nxt, with_value=True)
    np.testing.assert_array_equal(rows.policies["tilt_t0.02"].pe, full.policies["tilt_t0.02"].pe)
    assert rows.policies["tilt_t0.02"].V_row is None and rows.qb_row is None
    with pytest.raises(ValueError):
        BK.compute_rows(df, sid, N_P, 1, 1, specs={}, valid=valid, rep_ok=rep_ok, Q=Q, alpha=5.0, n_folds=3, with_dr=True)


def test_rows_cache_roundtrip(tmp_path):
    df, sid, valid, P, R, nxt, Q, rep_ok = _world(1, None)
    rows = BK.compute_rows(df, sid, N_P, 1, 1, specs=SPECS, valid=valid, rep_ok=rep_ok, Q=Q, alpha=5.0, n_folds=3, groups=IPS.coarse_groups(), P=P, R=R, nxt=nxt, with_value=True, with_dr=True)
    BK.save_rows(tmp_path / "r.npz", rows, "k1")
    assert BK.load_rows(tmp_path / "r.npz", "other") is None and BK.load_rows(tmp_path / "none.npz", "k1") is None
    back = BK.load_rows(tmp_path / "r.npz", "k1")
    assert list(back.policies) == list(rows.policies) and (back.n_support, back.n_valid) == (rows.n_support, rows.n_valid)
    np.testing.assert_array_equal(back.pb, rows.pb); np.testing.assert_array_equal(back.qb_row, rows.qb_row)
    for name in rows.policies:
        for k in ("pe", "pe_coarse", "V_row", "qe_row", "veb_row"):
            np.testing.assert_array_equal(getattr(back.policies[name], k), getattr(rows.policies[name], k))


def test_block_size_keeps_16_for_existing_experiment_sizes():
    assert BK.block_size(288, 225, 11) == 16 and BK.block_size(1728, 225, 11) == 16 and BK.block_size(2016, 225, 11) == 16
    assert BK.block_size(12096, 225, 11) == 4
