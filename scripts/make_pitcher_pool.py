"""레귤러 선발 풀 산출. 설정은 configs/pitchers_*.yaml 로만.

    .venv/bin/python scripts/make_pitcher_pool.py --config configs/pitchers_ip100.yaml

산출 data/pitchers/{pool_version}/ : pitchers.parquet(풀), seasons.parquet(투수×시즌 전체), meta.json, sha256.txt, table.md
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

from pitcheezy.data import pitchers as P
from pitcheezy.interfaces._io import write_sha256

REPO = Path(__file__).resolve().parents[1]
FILES = ("pitchers.parquet", "seasons.parquet", "meta.json")


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
        print("FAIL 2026 은 풀 산정에 쓰지 않는다 (OPE 전용)", file=sys.stderr)
        return 1
    min_ip = float(cfg["min_ip"])
    commit = git_commit()
    version = f"pool-ip{min_ip:g}-{seasons[0]}_{seasons[-1]}-{cfg['data_version']}-{commit}"
    df = P.load_pitches(a.data_dir, cfg["data_version"], seasons)
    st = P.season_table(df)
    pool = P.build_pool(st, min_ip, cfg.get("rule", "any_season"))
    out = REPO / "data" / "pitchers" / version
    out.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out / "pitchers.parquet", index=False)
    st.to_parquet(out / "seasons.parquet", index=False)
    meta = {
        "pool_version": version, "data_version": cfg["data_version"], "seasons": seasons, "min_ip": min_ip, "rule": cfg.get("rule", "any_season"),
        "compute_commit": commit, "config": str(a.config), "n_pitchers": int(len(pool)), "n_pitches_total": int(pool["pitches_total"].sum()),
        "n_pitchers_per_season": {int(s): int((g["ip"] >= min_ip).sum()) for s, g in st.groupby("season")},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_sha256(out, FILES)
    (out / "table.md").write_text(f"# {version}\n\n" + P.markdown_summary(st, pool, min_ip), encoding="utf-8")
    print(f"OK {out.relative_to(REPO)}  pitchers={len(pool)} pitches={meta['n_pitches_total']:,} per_season={meta['n_pitchers_per_season']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
