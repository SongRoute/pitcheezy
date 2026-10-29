#!/usr/bin/env python3
"""Run the built-in synthetic evaluation fixture, never external pitch payloads."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.prepitch_evaluation import evaluate, synthetic_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    roster, predictions = synthetic_fixture(config)
    result = evaluate(config, roster, predictions)
    # Exclusive creation preserves previous evidence even on accidental reruns.
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print('SYNTHETIC_ONLY: contract check saved; real data gate CLOSED; no scientific winner.')


if __name__ == '__main__':
    main()
