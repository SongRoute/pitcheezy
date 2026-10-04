"""Distinct same-model review tests of the TabM-residual wiring 6238d6c (same_model_review; Codex_review_pending).

epoch_record vs the frozen early-stop loop (matrix_models / direct_residual_model / direct_tabm_model):
``stale`` counts consecutive non-improving epochs since best_epoch and the loop breaks as soon as
``stale >= patience``, so at the end ``epochs_run - best_epoch == stale``.  Hence a patience stop
has run - best == patience exactly, and a fit that ran to the cap has run - best <= patience
(== patience when the break fires at the cap epoch itself).  A report with run - best > patience
cannot come from these loops.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export helpers
import test_cuda_tabm_residual as base  # noqa: E402

NEW = base.NEW


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


@pytest.mark.parametrize('report', [
    {'epochs_run': 30, 'best_epoch': 10},   # run < cap, 20 stale epochs: the loop would have stopped at 15
    {'epochs_run': 60, 'best_epoch': 50},   # at the cap, 10 stale epochs: the loop would have stopped at 55
    {'epochs_run': 17, 'best_epoch': 11},   # run < cap, 6 stale > patience 5
])
def test_epoch_record_refuses_reports_the_frozen_loop_cannot_produce(report):
    with pytest.raises(ValueError, match='inconsistent'):
        rc.epoch_record(report, 60, 5)


@pytest.mark.parametrize('arm', ['control', 'periodic_residual_counts', 'tabm_counts', NEW, 'tabm_counts_d10'])
def test_epoch_record_agrees_with_the_real_loop_of_every_arm_class(arm):
    """lr 0 (AdamW decay is lr-scaled) keeps the early-stop loss flat: best epoch 1, then a patience stop."""
    spec = rc.arm_spec(base.load(), arm)
    train, y = base.inputs(spec, 200, 31)
    early, ey = base.inputs(spec, 60, 32)
    cases = {(10, 2): (3, 1, 'patience'),    # patience stop before the cap
             (3, 2): (3, 1, 'patience'),     # patience stop exactly at the cap epoch
             (2, 2): (2, 1, 'cap'),          # cap reached with 1 stale epoch
             (1, 2): (1, 1, 'cap')}          # cap reached at the best epoch
    for (epochs, patience), (run, best, stopped_by) in cases.items():
        model = rc.build_model(spec, 24, 'cpu').fit(train, y, early, ey, epochs=epochs, patience=patience,
                                                    batch_size=64, learning_rate=0.)
        record = rc.epoch_record(model.report, epochs, patience)
        assert (record['epochs_run'], record['best_epoch'], record['stopped_by']) == (run, best, stopped_by), arm
        assert len(model.report['history']) == record['epochs_run']


def test_timing_projection_fields_present_for_every_arm_class(tmp_path):
    """Amendment 1 projection needs report.seconds, report.optimizer_updates and seconds_total in every fit.json."""
    path, digest, _ = tf.make_export(tmp_path / 'export', tf.lineage(base.load()))
    cfg = base.load()
    for arm in base.reg.ARM_ORDER:
        fit = base.read(rc.run_task(cfg, path, digest, tmp_path / 'smoke', arm, 26, 'cpu', {}, smoke=True) / 'fit.json')
        report = fit['report']
        assert report['seconds'] > 0 and fit['seconds_total'] >= report['seconds'], arm
        assert report['optimizer_updates'] == rc.SMOKE['epochs'] * int(np.ceil(report['training_rows'] / 1024)), arm
        assert fit['epoch_cap']['cap'] == rc.SMOKE['epochs'], arm


def test_registered_updates_per_epoch_constant():
    """1,224 = ceil(1,252,824 D100 TRAIN rows / batch 1024) (docs/contracts/ML-BRIDGE-RUNNER-v1.md)."""
    assert int(np.ceil(1_252_824 / base.reg.TRAINING['batch_size'])) == 1224
