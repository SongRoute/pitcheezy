"""RE24·dRE24 표 산출 (공통 자산 0). 설정은 configs/re24_*.yaml 로만.

    .venv/bin/python scripts/make_re24.py --config configs/re24_2023-2025.yaml

산출 data/re24/{re24_version}/ : RE24.npy, dRE24.npy, meta.json, sha256.txt, table.md (사람용). 커밋 대상.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from pitcheezy.data import re24 as R
from pitcheezy.interfaces.re24 import RE24Table, validate

REPO = Path(__file__).resolve().parents[1]


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return out + ("-dirty" if dirty else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path(os.environ.get("PITCHEEZY_DATA_DIR", REPO / "data")))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    seasons = [int(s) for s in cfg["seasons"]]
    if any(s >= 2026 for s in seasons):
        print("FAIL 2026 은 학습 창에 넣을 수 없음 (OPE 전용)", file=sys.stderr)
        return 1
    commit = git_commit()
    version = f"re24-{seasons[0]}_{seasons[-1]}-{cfg['data_version']}-{commit}"
    df = R.load_pitches(a.data_dir, cfg["data_version"], seasons)
    r = R.compute(df, exclude_last_half_inning=bool(cfg.get("exclude_last_half_inning", True)))
    min_n = int(r.dRE24_n.min())
    if min_n < int(cfg.get("min_cell_n", 1)):
        print(f"FAIL dRE24 셀 표본 부족: 최소 {min_n}", file=sys.stderr)
        return 1
    meta = {
        "re24_version": version,
        "data_version": cfg["data_version"],
        "season_window": f"{seasons[0]}-{seasons[-1]}",
        "seasons": seasons,
        "compute_commit": commit,
        "config": str(a.config.relative_to(REPO)) if a.config.is_absolute() else str(a.config),
        "exclude_last_half_inning": bool(cfg.get("exclude_last_half_inning", True)),
        "n_pitches": int(len(df)),
        "n_plate_appearances": r.n_plate_appearances,
        "n_terminal_plate_appearances": r.n_terminal_pa,
        "n_half_innings": r.n_half_innings,
        "n_half_innings_excluded": r.n_half_innings_excluded,
        "RE24_n": r.RE24_n.tolist(),
        "dRE24_n": r.dRE24_n.tolist(),
        "dRE24_min_cell_n": min_n,
        "mean_delta_terminal": r.mean_delta_all,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    table = RE24Table(RE24=r.RE24.astype(np.float64), dRE24=r.dRE24.astype(np.float64), meta=meta)
    problems = validate(table)
    if problems:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        return 1
    out = REPO / "data" / "re24" / version
    table.save(out)
    (out / "table.md").write_text(f"# {version}\n\n" + R.markdown_tables(r), encoding="utf-8")
    print(f"OK {out.relative_to(REPO)}  PA={r.n_plate_appearances:,} terminal={r.n_terminal_pa:,} "
          f"half_innings={r.n_half_innings:,} (excluded {r.n_half_innings_excluded:,}) dRE24 min n={min_n} mean Δ={r.mean_delta_all:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
