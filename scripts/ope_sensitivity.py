"""OPE 추정량 민감도 (진단, 모델 선택 아님): π_b 평활 α·폴드 수·클립을 바꿔 같은 정책의 1스텝·궤적 SNIPS 가 얼마나 흔들리는지.

    .venv/bin/python scripts/ope_sensitivity.py --config configs/EXP-P0-001.yaml
산출 runs/_diag/{ID}/ope_sensitivity.json + 표 stdout. 텐서·Q 는 runs/{ID}/s0 재사용.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from pitcheezy import experiment as E  # noqa: E402

GRID = [
    {"behavior_alpha": 5, "n_folds": 5, "clip": 20}, {"behavior_alpha": 10, "n_folds": 5, "clip": 20}, {"behavior_alpha": 20, "n_folds": 5, "clip": 20}, {"behavior_alpha": 40, "n_folds": 5, "clip": 20},
    {"behavior_alpha": 10, "n_folds": 10, "clip": 20}, {"behavior_alpha": 10, "n_folds": 2, "clip": 20}, {"behavior_alpha": 10, "n_folds": 1, "clip": 20},
    {"behavior_alpha": 10, "n_folds": 5, "clip": 5}, {"behavior_alpha": 10, "n_folds": 5, "clip": 100},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cfg = E.load_config(a.config)
    rows = []
    for g in GRID:
        c = copy.deepcopy(cfg)
        c["params"]["ope"].update(g)
        c["params"]["ope"]["menu"] = ["primary", "behavior", "uniform"]
        c["params"]["ope"]["n_boot"] = 300
        ex = E.Experiment(c, 0, a.data_dir, a.runs_dir)
        ex.seed_dir = a.runs_dir / "_diag" / cfg["id"] / f"a{g['behavior_alpha']}_f{g['n_folds']}_c{g['clip']}"
        ex.seed_dir.mkdir(parents=True, exist_ok=True)
        s = ex.stage_ope()
        r = s["results"]
        prim = ex.primary_name()
        row = {**g, "primary": prim,
               "onestep": r[f"{prim}/onestep_clip"]["snips"], "onestep_ci": [r[f"{prim}/onestep_clip"]["ci_low"], r[f"{prim}/onestep_clip"]["ci_high"]], "onestep_ess": r[f"{prim}/onestep_clip"]["ess_frac"],
               "traj": r[f"{prim}/traj_clip"]["snips"], "traj_ess": r[f"{prim}/traj_clip"]["ess_frac"], "traj_wmean": r[f"{prim}/traj_clip"]["w_mean"],
               "behavior_onestep": r["behavior/onestep_clip"]["snips"], "uniform_onestep": r["uniform/onestep_clip"]["snips"]}
        rows.append(row)
        print(f"α={g['behavior_alpha']:>3} folds={g['n_folds']:>2} clip={g['clip']:>3} | 1스텝 {row['onestep']:+.4f} [{row['onestep_ci'][0]:+.4f},{row['onestep_ci'][1]:+.4f}] ESS {100*row['onestep_ess']:.0f}% | 궤적 {row['traj']:+.4f} ESS {100*row['traj_ess']:.1f}% wmean {row['traj_wmean']:.2f} | π_b {row['behavior_onestep']:+.4f} 균등 {row['uniform_onestep']:+.4f}", flush=True)
    out = a.runs_dir / "_diag" / cfg["id"] / "ope_sensitivity.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
