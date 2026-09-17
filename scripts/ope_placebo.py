"""위약 검사 (진단): tilt 1스텝 SNIPS 가 Q 의 정보 때문인지, 추정량의 편향인지.

    .venv/bin/python scripts/ope_placebo.py --config configs/EXP-P0-001.yaml
Q 변형: real / neg(−Q) / perm_action(상태 안에서 행동 뒤섞기) / perm_pitcher(투수 축 뒤섞기) / zero(=π_b).
기대: real > zero ≈ perm ≈ π_b, neg < zero. 아니면 추정량을 의심한다.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = E.load_config(a.config)
    ex = E.Experiment(cfg, 0, a.data_dir, a.runs_dir)
    op = ex.p["ope"]
    ev = ex.pitches(op["eval_season"], holdout=True).reset_index(drop=True)
    sid = ex.sid(ev)
    t = TransitionTensor.load(ex.s0_dir / "transition", check_hash=False, mmap=True)
    vb = ValueBundle.load(ex.s0_dir / "value", check_hash=False)
    support = ex.eval_support(ev, t.valid)
    rng = np.random.default_rng(0)
    Q = vb.Q.astype(np.float64)
    variants = {"real": Q, "neg": -Q, "zero": np.zeros_like(Q)}
    Qp = Q.copy()
    for p in range(Q.shape[0]):
        for s in range(Q.shape[1]):
            Qp[p, s] = Q[p, s, rng.permutation(Q.shape[2])]
    variants["perm_action"] = Qp
    variants["perm_pitcher"] = Q[rng.permutation(Q.shape[0])]
    tilts = {k: (v, support, a.tau) for k, v in variants.items()}
    pb_logged, pe, _, _ = BH.crossfit_logged(ev, sid, ex.n_p, ex.K, alpha=float(op["behavior_alpha"]), n_folds=int(op["n_folds"]), tilts=tilts)
    pa = PR.pa_rewards(ev, ex.re24.RE24)
    pa = pa[pa["n_pitchers"] == 1].reset_index(drop=True)
    r = -pa["delta_re24"].to_numpy(); games = pa["game_pk"].to_numpy(); key = pa[["game_pk", "at_bat_number"]]
    out = {}
    for k in variants:
        pw = key.merge(IPS.pa_weights(ev, pe[k], pb_logged, clip=op.get("clip")), on=["game_pk", "at_bat_number"], how="left")
        w = pw["w_onestep"].fillna(0).to_numpy()
        est = IPS.estimate(w, r); est.update(IPS.bootstrap(w, r, games, n_boot=500, seed=0))
        out[k] = est
        print(f"{k:14s} tilt τ={a.tau}: 1스텝 SNIPS {est['snips']:+.4f} [{est['ci_low']:+.4f}, {est['ci_high']:+.4f}] ESS {100*est['ess_frac']:.0f}%", flush=True)
    (a.runs_dir / "_diag" / cfg["id"]).mkdir(parents=True, exist_ok=True)
    (a.runs_dir / "_diag" / cfg["id"] / f"placebo_tau{a.tau}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
