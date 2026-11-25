"""S4 of the v4 run: member predictions (split processes) -> FINAL predict -> FINAL score.

    python scripts/v4/run_v4_rest.py
Note: `score` refuses when score/result.json already exists (E_PS is scored once per output root).
To re-check the published numbers without retraining or re-predicting, use scripts/rescore_table.py instead.
"""
import pathlib
import sys
import subprocess

from _common import PY, REPO, heavy_lock, stamp

HERE = pathlib.Path(__file__).resolve().parent
steps = [[PY, '-B', str(HERE / 'v4_member.py'), 'X-GBT'],
         [PY, '-B', str(HERE / 'v4_member.py'), 'X-GRU-A3-plus', 'X-GRU-A5-b24'],
         [PY, '-B', '-m', 'experiments.lit_base.expl', 'predict', '--part', 'FINAL'],
         [PY, '-B', '-m', 'experiments.lit_base.expl', 'score', '--part', 'FINAL']]
with heavy_lock():
    for st in steps:
        stamp('START', st[2:])
        r = subprocess.run(st, cwd=REPO)
        stamp('END rc', r.returncode)
        if r.returncode:
            sys.exit(r.returncode)
