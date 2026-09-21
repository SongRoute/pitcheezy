"""Check the local pilot setup; no training, installs, or data-row reads.

Run with the existing repository .venv. --prepare creates output directories
and writes a timestamped report on the mounted SSD. Configuration fields for
training/logging are a contract for future implementation, not active hooks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = json.loads((project / "configs/local.json").read_text())
    volume = Path("/Volumes/T7 Shield")
    root = Path(config["artifact_root"])
    # Do not create an internal /Volumes fallback if the SSD is disconnected.
    if not volume.is_mount():
        raise SystemExit("T7 Shield is not mounted; no output directories created.")
    if not root.resolve().is_relative_to(volume.resolve()):
        raise SystemExit("Artifact root must resolve inside the mounted T7 Shield.")
    if Path(sys.executable).absolute() != Path(config["python"]).absolute():
        raise SystemExit(f"Use the configured interpreter: {config['python']}")

    import pyarrow.parquet as pq
    import torch

    allowed = [f"statcast_{year}.parquet" for year in (2023, 2024, 2025)]
    if config["raw_allowlist"] != allowed:
        raise SystemExit("Raw allowlist must contain exactly the three approved files.")
    raw_files = []
    for name in allowed:
        path = Path(config["raw_root"]) / name
        metadata = pq.read_metadata(path)
        raw_files.append({"name": name, "bytes": path.stat().st_size,
                          "rows": metadata.num_rows,
                          "columns": metadata.num_columns})

    mps = torch.backends.mps.is_available()
    device = "mps" if mps else "cpu"
    value = (torch.ones((2, 2), device=device) @
             torch.ones((2, 2), device=device)).sum().item()
    if value != 8:
        raise SystemExit("Tensor smoke check failed.")
    packages = sorted(
        [{"name": d.metadata["Name"], "version": d.version}
         for d in importlib.metadata.distributions() if d.metadata["Name"]],
        key=lambda d: d["name"].lower(),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "device_smoke_test": device,
        "internal_free_bytes": shutil.disk_usage(project).free,
        "ssd_free_bytes": shutil.disk_usage(volume).free,
        "raw_metadata": raw_files, "packages": packages,
        "tmux": shutil.which("tmux"), "uv": shutil.which("uv"),
        "cohort_period_status": config["cohort"]["period_status"],
        "environment_note": "Installed-version snapshot, not a resolved lockfile.",
        "data_note": "Only approved parquet footers read; no row profiling or hashes.",
    }
    if args.prepare:
        for directory in ("raw", "cache", "processed", "runs", "checkpoints", "logs", "reports", "wandb"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        # Verify real write/read access, not merely permission bits on ExFAT.
        with tempfile.TemporaryFile(dir=root / "reports") as probe:
            probe.write(b"pitchmdp")
            probe.flush()
            probe.seek(0)
            if probe.read() != b"pitchmdp":
                raise SystemExit("SSD write/read probe failed.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = root / "reports" / f"setup-{stamp}.json"
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(f"Setup report: {destination}")
    print(json.dumps({k: v for k, v in report.items() if k != "packages"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
