"""실험 실행 진입점 (CLAUDE.md 실험 절).

    .venv/bin/python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --seed 0
    .venv/bin/python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --aggregate   # results/{ID}.json 갱신

- 설정 파일명 = 실험 ID. 실험은 configs/ 파일로만.
- 산출 $PITCHEEZY_RUNS_DIR/{ID}/s{seed}/ (transition·value 는 결정론적 → s0 에만). 요약 results/{ID}.json (커밋). W&B run {ID}/s{seed}.
- 단계별 산출물이 있으면 건너뛴다 (체크포인트).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pitcheezy import experiment as E  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true", help="시드 실행 없이 results/{ID}.json 만 갱신")
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    ap.add_argument("--runs-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_RUNS_DIR", REPO / "runs")))
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = E.load_config(a.config)
    if a.no_wandb:
        cfg.setdefault("wandb", {})["mode"] = "disabled"
    if not a.aggregate:
        if a.seed is None:
            ap.error("--seed 필요 (또는 --aggregate)")
        rec = E.Experiment(cfg, a.seed, a.data_dir, a.runs_dir).run()
        E.log_wandb(cfg, rec)
    out = E.aggregate(cfg, a.runs_dir, REPO / "results")
    main_pol = out["ope"].get("primary") or next(iter(out["ope"]["policies"]))
    p = out["ope"]["policies"][main_pol]
    print(f"OK results/{cfg['id']}.json  seeds={out['seeds']}  {main_pol}: SNIPS {p['point_snips']:+.4f} CI[{p['ci_low_min']:+.4f}, {p['ci_high_max']:+.4f}]  "
          f"holdout NLL {out['transition']['holdout_nll']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
