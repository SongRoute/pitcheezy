#!/usr/bin/env python3
"""Evaluate the built-in synthetic pitch-type fixture; no external payload input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.pitch_type_evaluation import evaluate, synthetic_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = evaluate(config, *synthetic_fixture(config))
    result['source_sha256']['evaluate_pitch_type.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with args.output.open('x') as handle:
        handle.write(payload)
    print('SYNTHETIC_ONLY: frozen frequency baselines; hypothetical candidate, no fitted tree or scientific winner.')


if __name__ == '__main__':
    main()
