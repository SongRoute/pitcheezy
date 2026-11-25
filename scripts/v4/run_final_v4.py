"""Run EXP-EXPL-1121-FINAL stages in order (original use: `run_final_v4.py select`).

    python scripts/v4/run_final_v4.py select [predict score ...]
Each stage = `python -m experiments.lit_base.expl <stage> --part FINAL` from the repo root.
"""
import os
import subprocess
import sys

from _common import PY, REPO, heavy_lock, stamp

for k in ('PITCHEEZY_OUT_ROOT', 'PITCHEEZY_LANE'):
    assert k not in os.environ, k
with heavy_lock():
    for stage in sys.argv[1:]:
        stamp('START', stage)
        r = subprocess.run([PY, '-B', '-m', 'experiments.lit_base.expl', stage, '--part', 'FINAL'], cwd=REPO)
        stamp('END', stage, 'rc', r.returncode)
        if r.returncode != 0:
            sys.exit(r.returncode)
