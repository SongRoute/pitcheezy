"""Reviewer tests (same-model review of b79888f): EMA config pins and the F6 raw-trajectory check. CPU only."""
import json
from pathlib import Path
import sys

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ema_trajectory_check as et  # noqa: E402
import run_cuda_direct as rc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers

torch.set_num_threads(1)
RUN = REPO / 'configs' / 'DRAFT-CUDA-EMA-20261004.json'
SIBLING = REPO / 'configs' / 'DRAFT-CUDA-EMBED-RESIDUAL-20261004.json'
QUEUE_LIMITS = ('task_timeout_seconds', 'family_budget_seconds', 'max_attempts')


def load(registered=False):
    return tf.load(RUN, registered, True)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def test_config_blocks_equal_the_earlier_family_and_every_pin_is_current():
    """Mutant missed by test_cuda_ema.py: a third GPU UUID, another calibration chain or a stale design-note pin."""
    cfg, old = json.loads(RUN.read_text()), json.loads(SIBLING.read_text())
    assert cfg['training'] == rc.TRAINING == old['training'] and cfg['calibration'] == old['calibration']
    assert cfg['export'] == old['export'] and cfg['device'] == old['device']
    assert cfg['device']['cpu_fallback'] is False and len(cfg['device']['allowed_gpu_uuids']) == 2
    assert {k: cfg['queue'][k] for k in QUEUE_LIMITS} == {k: old['queue'][k] for k in QUEUE_LIMITS}
    assert cfg['execution'] == {'enabled': False} and cfg['status'] == 'DRAFT'
    for source in cfg['source_configs'].values():
        assert source['sha256'] == rc.sha256_file(REPO / source['path']), source['path']
    assert cfg['source_configs']['design_note']['path'] == 'experiments/pitchmdp/docs/DESIGN-EMA-WEIGHTS-20261004.md'


def smoke_family(tmp_path, export, seed=19):
    path, digest, _ = export
    cfg = load()
    out = tmp_path / 'fam-smoke'
    for arm in ('control', 'ema', 'dropout', 'dropout_ema'):
        rc.run_task(cfg, path, digest, out, arm, seed, 'cpu', {'device_uuid': 'cpu'}, smoke=True)
    return cfg, out, digest


def test_trajectory_check_is_exact_on_cpu_smoke(export, tmp_path):
    cfg, out, digest = smoke_family(tmp_path, export)
    report = et.check(cfg, out, digest, smoke=True, device='cpu', seeds=[19])
    assert report['all_exact'] and set(report['pairs']) == {'ema_vs_control/19', 'dropout_ema_vs_dropout/19'}
    for pair in report['pairs'].values():
        assert pair['epochs_compared'] == rc.SMOKE['epochs'] and pair['max_abs_diff'] == {
            'train_weighted_nll': 0., 'raw_earlystop_conditional_nll': 0.}
    with pytest.raises(FileNotFoundError):  # all 20 tasks by default
        et.check(cfg, out, digest, smoke=True, device='cpu')
    with pytest.raises(ValueError, match='Unregistered seed'):
        et.check(cfg, out, digest, smoke=True, device='cpu', seeds=[18])


def fake(tmp_path, export, histories, weights=None, sources=None):
    """Fake EMA tasks for seed 19 with given histories (no fit)."""
    path, digest, splits = export
    cfg = load()
    root = tmp_path / 'fake'
    for arm, history in histories.items():
        dest = tf.fake_task(root, cfg, digest, arm, 19, splits, (sources or {}).get(arm) or rc.own_hashes())
        fit = json.loads((dest / 'fit.json').read_text())
        w = (weights or {}).get(arm, 'ema' if arm.endswith('ema') else None)
        fit['report'] = {'history': history, **({'weights': w} if w else {})}
        rc.write_json_atomic(dest / 'fit.json', fit)
        tf.reseal(dest)
    return cfg, root, digest


def hist(values, raw=None):
    return [{'epoch': i + 1, 'train_weighted_nll': v, 'earlystop_conditional_nll': v + 1.,
             **({'raw_earlystop_conditional_nll': raw[i]} if raw else {})} for i, v in enumerate(values)]


def test_trajectory_check_reports_a_difference_and_compares_only_the_raw_scores(export, tmp_path):
    base = hist([2., 1.5, 1.2])
    ema = hist([2., 1.5, 1.2, 1.1], raw=[3., 2.5, 2.2, 2.1])  # EMA score differs (not compared), raw equal
    drop = hist([2., 1.5])
    drop_ema = hist([2., 1.5 + 1e-7], raw=[3., 2.5 + 1e-7])
    cfg, root, digest = fake(tmp_path, export, {'control': base, 'ema': ema, 'dropout': drop, 'dropout_ema': drop_ema})
    report = et.check(cfg, root, digest, smoke=False, seeds=[19])
    one, two = report['pairs']['ema_vs_control/19'], report['pairs']['dropout_ema_vs_dropout/19']
    assert one['exact'] and one['epochs_compared'] == 3 and one['epochs_run'] == [4, 3]
    assert not two['exact'] and two['max_abs_diff']['train_weighted_nll'] == pytest.approx(1e-7)
    assert not report['all_exact']


@pytest.mark.parametrize('case', ['base_flagged_ema', 'candidate_not_ema', 'tampered', 'sources', 'epochs'])
def test_trajectory_check_fails_closed(export, tmp_path, case):
    h = {'control': hist([2.]), 'ema': hist([2.], raw=[3.]), 'dropout': hist([2.]), 'dropout_ema': hist([2.], raw=[3.])}
    weights, sources = {}, {}
    if case == 'base_flagged_ema':
        weights['control'] = 'ema'
    if case == 'candidate_not_ema':
        weights['ema'] = 'raw'
    if case == 'sources':
        sources['dropout'] = {**rc.own_hashes(), rc.SELF: '0' * 64}
    if case == 'epochs':
        h['ema'][0]['epoch'] = 2
    cfg, root, digest = fake(tmp_path, export, h, weights, sources)
    if case == 'tampered':  # fit.json edited after sealing
        (rc.task_dir(root, 'ema', 19, False) / 'fit.json').write_text('{}')
    with pytest.raises(ValueError):
        et.check(cfg, root, digest, smoke=False, seeds=[19])


def test_trajectory_check_refuses_a_non_ema_config(tmp_path):
    cfg = tf.load(SIBLING, False, True)
    with pytest.raises(ValueError, match='Not the EMA family'):
        et.check(cfg, tmp_path, 'x', smoke=False)
