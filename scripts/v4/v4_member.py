"""Predict E_PS for the named v4 members in this process only (so lightgbm and torch never load together).

    python scripts/v4/v4_member.py X-GBT
    python scripts/v4/v4_member.py X-GRU-A3-plus X-GRU-A5-b24
"""
import os
import sys

from _common import REPO

assert 'KMP_DUPLICATE_LIB_OK' not in os.environ and 'PITCHEEZY_OUT_ROOT' not in os.environ and 'PITCHEEZY_LANE' not in os.environ
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from experiments.lit_base import expl as X, final2 as F2, run as R  # noqa: E402

cfg = R.load_cfg(X.CONFIGS['FINAL']); R.require_registered(cfg, False); sels = X.pool_selections(cfg, False)
log = R.logger(R.out_root(cfg, False) / 'logs' / 'predict-members-split.log')
for f in sys.argv[1:]:
    F2.eps_member(cfg, False, f, sels, log)
print('loaded', [m for m in ('torch', 'lightgbm') if m in sys.modules], flush=True)
