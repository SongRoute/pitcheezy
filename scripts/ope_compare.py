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
from pitcheezy.ope import ips as IPS  # noqa: E402

COL = {"onestep_clip": "w_onestep", "traj_clip": "w_traj", "onestep2s_clip": "w_onestep_slice"}


def weights_for(exp_id: str, data_dir: Path, runs_dir: Path, variant: str):
    cfg = E.load_config(REPO / "configs" / f"{exp_id}.yaml")
    ex = E.Experiment(cfg, 0, data_dir, runs_dir)
    op = ex.p["ope"]
    ev = ex.pitches(op["eval_season"], holdout=True).reset_index(drop=True)
    sid = ex.sid(ev)
    pa = PR.pa_rewards(ev, ex.re24.RE24)
    key = pa[["game_pk", "at_bat_number", "n_pitchers"]]
    r = -pa["delta_re24"].to_numpy()
    two_strike = ev["count_id"].to_numpy() % 3 == 2
    if ex.primary_name() == "behavior":
        has = ev["action_id"] >= 0
        m = has if variant != "onestep2s_clip" else (has & two_strike)
        n = key.merge(ev[m].groupby(["game_pk", "at_bat_number"]).size().rename("n").reset_index(), on=["game_pk", "at_bat_number"], how="left")["n"].fillna(0).to_numpy(float)
        w = np.ones(len(pa)) if variant == "traj_clip" else n
        return key, w, r, "behavior"
    t = TransitionTensor.load(ex.s0_dir / "transition", check_hash=False, mmap=True)
    vb = ValueBundle.load(ex.s0_dir / "value", check_hash=False)
    support = ex.eval_support(ev, t.valid)
    name = ex.primary_name()
    if name.startswith("tilt_t"):
        tau = float(name[6:])
        pb_logged, pe, _, _ = BH.crossfit_logged(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]), tilts={"p": (vb.Q, support, tau)})
        pe_logged = pe["p"]
    else:
        pb_logged, _, _, _ = BH.crossfit_logged(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]))
        pe_logged = IPS.logged_probs(ev, sid, IPS.restrict_support(vb.policy, support))
    pw = key.merge(IPS.pa_weights(ev, pe_logged, pb_logged, clip=op.get("clip"), slice_mask=two_strike), on=["game_pk", "at_bat_number"], how="left")
    w = pw[COL[variant]].fillna(1.0 if variant == "traj_clip" else 0.0).to_numpy()
    return key, w, r, name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--variant", default="onestep_clip", choices=list(COL))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    ka, wa, ra, na = weights_for(a.a, a.data_dir, a.runs_dir, a.variant)
    kb, wb, rb, nb = weights_for(a.b, a.data_dir, a.runs_dir, a.variant)
    assert (ka[["game_pk", "at_bat_number"]].to_numpy() == kb[["game_pk", "at_bat_number"]].to_numpy()).all(), "타석 키 불일치"
    keep = (ka["n_pitchers"] == 1).to_numpy()
    wa, wb, r, games = wa[keep], wb[keep], ra[keep], ka["game_pk"].to_numpy()[keep]
    ug, inv = np.unique(games, return_inverse=True); n_g = len(ug)
    wra, wga = np.bincount(inv, weights=wa * r, minlength=n_g), np.bincount(inv, weights=wa, minlength=n_g)
    wrb, wgb = np.bincount(inv, weights=wb * r, minlength=n_g), np.bincount(inv, weights=wb, minlength=n_g)
    rng = np.random.default_rng(a.seed)
    d = np.empty(a.n_boot)
    for i in range(a.n_boot):
        m = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g)).astype(float)
        d[i] = (m * wra).sum() / (m * wga).sum() - (m * wrb).sum() / (m * wgb).sum()
    pa_, pb_ = (wa * r).sum() / wa.sum(), (wb * r).sum() / wb.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    p_le0 = float((d <= 0).mean())
    verdict = "A > B (CI 가 0 을 제외)" if lo > 0 else ("A < B (CI 가 0 을 제외)" if hi < 0 else "구분 안 됨")
    out = {"a": a.a, "b": a.b, "policy_a": na, "policy_b": nb, "variant": a.variant, "snips_a": pa_, "snips_b": pb_, "delta": pa_ - pb_, "ci": [float(lo), float(hi)], "p_delta_le_0": p_le0, "n_pa": int(keep.sum()), "n_games": int(n_g), "n_boot": a.n_boot, "verdict": verdict}
    print(f"{a.a}({na}) − {a.b}({nb}) [{a.variant}]: Δ = {pa_ - pb_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ≤0)={p_le0:.3f}  → {verdict}")
    d_dir = a.runs_dir / "_diag" / "compare"; d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / f"{a.a}_vs_{a.b}_{a.variant}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
