"""E4 workload column-quality audit on TRAIN rows only (not yet run; PM approval required).

Reads only key/date/pitcher and the two Savant workload columns from the
2023-2025 raw files, filtered at read time to game_date <= 2025-04-30. No
outcome, description, score or physics column is opened; no 2026 file.
Rows before 2023-05-15 are loaded only as history for days-since rebuilds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.workload_features import INPUT_COLUMNS, TRAIN_END, audit

FILES = [f"statcast_{year}.parquet" for year in (2023, 2024, 2025)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "holdout" in str(args.raw_root) or args.out.exists():
        raise SystemExit("Refusing holdout root or existing output")
    end = pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)
    frames = [pq.read_table(args.raw_root / name, columns=list(INPUT_COLUMNS),
                            filters=[("game_date", "<", end)]).to_pandas() for name in FILES]
    report = audit(pd.concat(frames, ignore_index=True))
    report["source_files"] = FILES
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
