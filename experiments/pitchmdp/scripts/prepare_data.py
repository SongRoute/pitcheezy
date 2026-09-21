"""Prepare the three approved Statcast seasons and freeze the pitcher cohort."""
from pathlib import Path
import argparse
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp.data import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/local.json")
    args = parser.parse_args()
    prepare(args.config)
