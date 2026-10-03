"""Disposition tests for the TabM-mini integration review (805aadc, findings 3 and 5)."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cuda_tabm_mini as tt  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402

export = tt.export


def test_queue_declares_start_after_p30_and_common_lock_dir():
    """Finding 3: the DRAFT states it never runs concurrently with P30 and names the common per-GPU lock dir."""
    q = tt.load(unpin=False)['queue']
    assert 'EXP-P30-001 queue terminal' in q['start_after']
    assert 'never concurrent' in q['start_after']
    assert '--lock-dir /home/sds733/pitcheezy-autonomy/gpu-locks' in q['start_after']
    assert tt.rc.config_check(tt.load(unpin=False))['queue']['start_after'] == q['start_after']


@pytest.mark.parametrize('name', ['check_contrasts', 'regularization'])
def test_tabm_scorer_refuses_other_contrast_lists_explicitly(export, tmp_path, monkeypatch, name):
    """Finding 5: a TabM config is scored only with its own contrasts, refused before any read."""
    cfg = tt.load(registered=True)
    if name == 'check_contrasts':
        monkeypatch.setattr(tt.sc, 'check_contrasts', lambda config: [{'name': 'factorial'}])
    else:
        monkeypatch.setattr(tt.sc.dropout_registry, 'check_contrasts', lambda config: [{'name': 'regularization'}])
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, copy.deepcopy(cfg), export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()
