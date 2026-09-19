"""두 실험의 주 정책을 같은 2026 타석에 짝지어 비교: Δ = SNIPS(A) − SNIPS(B) 의 경기 클러스터 부트스트랩 CI.

    .venv/bin/python scripts/ope_compare.py EXP-P0-009 EXP-P0-001 [--variant onestep_clip] [--n-boot 2000]
각 실험은 runs/{ID}/s0 의 텐서·Q 를 쓰고, π_b 교차 적합은 각자 설정으로 다시 계산한다 (K 가 달라도 타석 키로 맞춤).
정책 "behavior" 는 π_b 자체.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pitcheezy import experiment as E  # noqa: E402
from pitcheezy.data import prepare as PR  # noqa: E402
from pitcheezy.interfaces.tensor import TransitionTensor  # noqa: E402
from pitcheezy.interfaces.value import ValueBundle  # noqa: E402
from pitcheezy.ope import behavior as BH  # noqa: E402
from pitcheezy.ope import dr as DR  # noqa: E402
from pitcheezy.ope import ips as IPS  # noqa: E402
from pitcheezy.policy import vi as VI  # noqa: E402

COL = {"onestep_clip": "w_onestep", "traj_clip": "w_traj", "onestep2s_clip": "w_onestep_slice"}
DR_VARIANTS = ("onestep_dr", "traj_dr")
VARIANTS = list(COL) + list(DR_VARIANTS)


def weights_for(exp_id: str, data_dir: Path, runs_dir: Path, variant: str, model_seed: int = 0, policy: str | None = None):
    """(key, (num1, den1, num2, den2), r, 정책 이름). 추정량 = Σnum1/Σden1 + Σnum2/Σden2 (den2 합이 0 이면 둘째 항 없음).
    SNIPS 계열은 num1 = w·r, den1 = w 뿐이고, DR 은 둘째 항(직접법)이 붙는다."""
    cfg = E.load_config(REPO / "configs" / f"{exp_id}.yaml")
    ex = E.Experiment(cfg, model_seed, data_dir, runs_dir)  # arch=neural 이면 그 시드의 Q, count 면 항상 s0
    op = ex.p["ope"]
    ev = ex.pitches(op["eval_season"], holdout=True).reset_index(drop=True)
    sid = ex.sid(ev)
    pa = PR.pa_rewards(ev, ex.re24.RE24)
    key = pa[["game_pk", "at_bat_number", "n_pitchers"]]
    r = -pa["delta_re24"].to_numpy()
    z = np.zeros(len(pa))
    two_strike = ev["count_id"].to_numpy() % 3 == 2
    name = policy or ex.primary_name()  # --policy 로 주 정책 대신 메뉴의 다른 정책(예: tilt_t0.005)을 같은 텐서·Q 로 만든다
    if name == "behavior":
        if variant in DR_VARIANTS:
            raise ValueError(f"behavior 정책에는 DR 변형({variant})이 없다 — 제어변량과 정책이 같다. --variant 를 {list(COL)} 중에서 고르라")
        has = ev["action_id"] >= 0
        m = has if variant != "onestep2s_clip" else (has & two_strike)
        n = key.merge(ev[m].groupby(["game_pk", "at_bat_number"]).size().rename("n").reset_index(), on=["game_pk", "at_bat_number"], how="left")["n"].fillna(0).to_numpy(float)
        w = np.ones(len(pa)) if variant == "traj_clip" else n
        return key, (w * r, w, z, z), r, "behavior"
    valid = np.load(ex.s0_dir / "transition" / "valid.npy")  # P.npy 는 시드 ≠ 0 에서 지워져 있을 수 있다
    vb = ValueBundle.load(ex.s0_dir / "value", check_hash=False)
    support = ex.eval_support(ev, valid)
    if name.startswith("tilt_t"):
        tau = float(name[6:])
        pb_logged, pe, _, _ = BH.crossfit_logged(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]), tilts={"p": (vb.Q, support, tau)}, C=ex.C)
        pe_logged = pe["p"]
    else:
        pb_logged, _, _, _ = BH.crossfit_logged(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]), C=ex.C)
        pe_logged = IPS.logged_probs(ev, sid, IPS.restrict_support(vb.policy, support))
    if variant not in DR_VARIANTS:
        pw = key.merge(IPS.pa_weights(ev, pe_logged, pb_logged, clip=op.get("clip"), slice_mask=two_strike), on=["game_pk", "at_bat_number"], how="left")
        w = pw[COL[variant]].fillna(1.0 if variant == "traj_clip" else 0.0).to_numpy()
        return key, (w * r, w, z, z), r, name
    # --- DR: stage_ope 와 같은 재료 (pitcheezy.ope.dr 가 단일 출처)
    if not (ex.s0_dir / "transition" / "P.npy").exists():
        raise FileNotFoundError(f"{ex.s0_dir/'transition'/'P.npy'} 없음 — DR 은 전이 텐서가 필요하다 (prune 된 시드면 같은 config·시드로 재실행)")
    t = TransitionTensor.load(ex.s0_dir / "transition", check_hash=False, mmap=True)
    R = VI.reward_table(ex.re24.dRE24, ex.K, collapse_base_out=bool(ex.p["policy"].get("reward_collapse_base_out", False)))
    nxt = VI.next_state_table(ex.K)
    pb_full = BH.fit_behavior(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]))
    pol_arr = BH.tilt(pb_full, vb.Q, support, float(name[6:])) if name.startswith("tilt_t") else IPS.restrict_support(vb.policy, support)
    q_b = DR.behavior_q(t.P, pb_full, support, R, nxt)
    v_e_b, V_e, q_e = DR.dr_inputs(t.P, pol_arr, q_b, R, nxt)
    rho, has = IPS.pitch_ratios(ev, pe_logged, pb_logged, clip=op.get("clip"))
    if variant == "onestep_dr":
        ot = key.merge(DR.onestep_dr_terms(ev, sid, rho, has, q_b, v_e_b), on=["game_pk", "at_bat_number"], how="left").fillna(0.0)
        return key, (ot["sum_rho"].to_numpy() * r - ot["sum_rho_q"].to_numpy(), ot["sum_rho"].to_numpy(), ot["sum_v"].to_numpy(), ot["n_dec"].to_numpy()), r, name
    tt = key.merge(DR.traj_dr_terms(ev, sid, rho, has, q_e, V_e), on=["game_pk", "at_bat_number"], how="left").fillna(0.0)
    return key, (DR.traj_dr_values(tt, r)[1], np.ones(len(pa)), z, z), r, name  # WDR 평균


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--variant", default="onestep_clip", choices=VARIANTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=0, help="arch=neural 실험에서 쓸 학습 시드 (count 실험은 무시)")
    ap.add_argument("--policy", default=None, help="두 실험 모두 주 정책 대신 이 정책으로 (tilt_t{τ} | behavior). 고감도 τ 읽기용; 판정 규칙(D22)은 주 정책")
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    ka, pta, ra, na = weights_for(a.a, a.data_dir, a.runs_dir, a.variant, a.model_seed, a.policy)
    kb, ptb, rb, nb = weights_for(a.b, a.data_dir, a.runs_dir, a.variant, a.model_seed, a.policy)
    assert (ka[["game_pk", "at_bat_number"]].to_numpy() == kb[["game_pk", "at_bat_number"]].to_numpy()).all(), "타석 키 불일치"
    keep = (ka["n_pitchers"] == 1).to_numpy()
    games = ka["game_pk"].to_numpy()[keep]
    ug, inv = np.unique(games, return_inverse=True); n_g = len(ug)
    ga = [np.bincount(inv, weights=x[keep], minlength=n_g) for x in pta]  # 경기별 (num1, den1, num2, den2)
    gb = [np.bincount(inv, weights=x[keep], minlength=n_g) for x in ptb]
    two_a, two_b = ga[3].sum() > 0, gb[3].sum() > 0  # 둘째 항(직접법)이 있는가

    def _est(g, m, two):
        return (m * g[0]).sum() / (m * g[1]).sum() + ((m * g[2]).sum() / (m * g[3]).sum() if two else 0.0)

    one = np.ones(n_g)
    rng = np.random.default_rng(a.seed)
    d = np.empty(a.n_boot)
    for i in range(a.n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        d[i] = _est(ga, m, two_a) - _est(gb, m, two_b)
    pa_, pb_ = _est(ga, one, two_a), _est(gb, one, two_b)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_le0 = float((d <= 0).mean())
    verdict = "A > B (CI 가 0 을 제외)" if lo > 0 else ("A < B (CI 가 0 을 제외)" if hi < 0 else "구분 안 됨")
    out = {"a": a.a, "b": a.b, "policy_a": na, "policy_b": nb, "variant": a.variant, "snips_a": pa_, "snips_b": pb_, "delta": pa_ - pb_, "ci": [float(lo), float(hi)], "p_delta_le_0": p_le0, "n_pa": int(keep.sum()), "n_games": int(n_g), "n_boot": a.n_boot, "model_seed": a.model_seed, "policy_override": a.policy, "verdict": verdict}
    print(f"[model seed {a.model_seed}] {a.a}({na}) − {a.b}({nb}) [{a.variant}]: Δ = {pa_ - pb_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ≤0)={p_le0:.3f}  → {verdict}")
    d_dir = a.runs_dir / "_diag" / "compare"; d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / (f"{a.a}_vs_{a.b}_{a.variant}" + (f"_ms{a.model_seed}" if a.model_seed else "") + (f"_{a.policy}" if a.policy else "") + ".json")).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
