#!/usr/bin/env python3
"""Import one frozen C inning decision into an explicitly selected SQLite database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/observer/backend"))

from observer_app.inning_decision_store import InningDecisionRepository  # noqa: E402
from observer_app.store import Store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    args = parser.parse_args()
    print(InningDecisionRepository(Store(args.database)).import_result(args.source, args.anchor))


if __name__ == "__main__":
    main()
