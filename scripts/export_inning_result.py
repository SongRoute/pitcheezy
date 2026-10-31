#!/usr/bin/env python3
"""Export the frozen inning result and a development-only unavailable example."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/observer/backend"))

from observer_app.inning_result import convert_inning_result, development_unavailable  # noqa: E402


def _encoded(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def export(source: Path, anchor: Path, output_dir: Path) -> None:
    bounded = convert_inning_result(source, anchor)
    unavailable = development_unavailable(bounded)
    files = {"bounded.json": _encoded(bounded), "development_unavailable.json": _encoded(unavailable)}
    manifest = {
        "schema_version": "inning-result-fixtures-v1",
        "fixtures": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "synthetic": name == "development_unavailable.json", "description": "Synthetic missing_evaluation_result example; not a new evaluation" if name == "development_unavailable.json" else "Conversion of frozen C evaluation and decision anchor"}
            for name, data in files.items()
        },
    }
    files["manifest.json"] = _encoded(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    for name, data in files.items():
        path = output_dir / name
        with path.open("xb") as handle:
            handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/EXP-C-INNING-001/conditional_keep_777063.json")
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.anchor, args.output_dir)


if __name__ == "__main__":
    main()
