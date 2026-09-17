"""타자 군집 v0 산출. 설정은 configs/batters_*.yaml.

    .venv/bin/python scripts/make_batter_clusters.py --config configs/batters_k6.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from pitcheezy.data import batters as B
from pitcheezy.interfaces._io import write_sha256

REPO = Path(__file__).resolve().parents[1]
FILES = ("batter_clusters.parquet", "centroids.parquet", "meta.json")


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
        print("FAIL 2026 은 학습 창에 넣을 수 없음", file=sys.stderr); return 1
    commit = git_commit()
    version = f"clusters-K{cfg['K']}-{seasons[0]}_{seasons[-1]}-{cfg['data_version']}-{commit}"
    df = B.load_pitches(a.data_dir, cfg["data_version"], seasons)
    feats = B.batter_features(df)
    assign, cents, meta_sides = B.cluster(feats, list(cfg["features"]), int(cfg["K"]), min_pa=int(cfg["min_pa"]), **{k: int(v) for k, v in cfg["kmeans"].items()})
    out = REPO / "data" / "batters" / version
    out.mkdir(parents=True, exist_ok=True)
    assign.to_parquet(out / "batter_clusters.parquet", index=False)
    cents.to_parquet(out / "centroids.parquet", index=False)
    meta = {"cluster_version": version, "data_version": cfg["data_version"], "seasons": seasons, "K": int(cfg["K"]), "min_pa": int(cfg["min_pa"]), "features": list(cfg["features"]),
            "kmeans": cfg["kmeans"], "compute_commit": commit, "n_batter_stand": int(len(assign)), "sides": meta_sides, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    write_sha256(out, FILES)
    lines = [f"# {version}", "", f"타자×stand {len(assign)} (특징 있음 {int((~assign.is_default).sum())}, 기본 군집 {int(assign.is_default.sum())}). 군집 번호는 side 안에서 k_rate 중심 오름차순.", "",
             "| cluster_id | stand | 타자 수 | " + " | ".join(cfg["features"]) + " |", "|---|---|---|" + "---|" * len(cfg["features"])]
    for _, r in cents.iterrows():
        lines.append(f"| {r.cluster_id} | {r.stand} | {r.n_batters} | " + " | ".join(f"{r[f]:.3f}" for f in cfg["features"]) + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print(f"OK {out.relative_to(REPO)}  n={len(assign)} " + " ".join(f"{s}:{m['n_eligible']}+{m['n_default']}(default {m['default_cluster_id']})" for s, m in meta_sides.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
