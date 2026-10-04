"""Synthetic CPU tests: EMA weight-averaging runner/scorer integration (DRAFT-CUDA-EMA-20261004)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_ema_registry as emr  # noqa: E402
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_tabm_registry as tr  # noqa: E402
from pitchmdp.direct_ema_model import FlattenEMAModel  # noqa: E402
from pitchmdp.direct_frontier_model import FlattenRegularizedModel  # noqa: E402
from pitchmdp.matrix_models import MatrixModel  # noqa: E402

torch.set_num_threads(1)
RUN = REPO / 'configs' / 'DRAFT-CUDA-EMA-20261004.json'
FRONTIER = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-RUN-20261004.json'
OTHERS = [REPO / 'configs' / n for n in ('DRAFT-CUDA-EMBED-RESIDUAL-20261004.json', 'DRAFT-CUDA-TABM-MINI-20261004.json',
                                          'EXP-P30-001.json', 'DRAFT-CUDA-DROPOUT-RUN-20261004.json',
                                          'DRAFT-CUDA-FRONTIER-RUN-20261004.json')]


def load(registered=False, unpin=True):
    return tf.load(RUN, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def test_config_is_draft_exact_4_arm_family_on_seeds_19_23():
    cfg = load(unpin=False)
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert 'registration' not in cfg and cfg['experiment_id'] == 'DRAFT-CUDA-EMA-20261004'
    assert [a['name'] for a in cfg['arms']] == list(emr.ARM_ORDER)
    assert cfg['seeds'] == [19, 20, 21, 22, 23] and cfg['fits'] == 20 == 4 * len(cfg['seeds'])
    old = {a['name']: a for a in json.loads(FRONTIER.read_text())['arms']}
    arms = {a['name']: a for a in cfg['arms']}
    assert arms['control'] == old['control'] and arms['dropout'] == old['dropout']
    assert cfg['comparison'][emr.CONTRAST_KEY] == emr.CONTRASTS
    for path in OTHERS:  # seed blocks of every earlier family are disjoint from 19-23
        assert not set(json.loads(path.read_text())['seeds']) & set(cfg['seeds'])


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][1].update(ema_decay=.99),
    lambda c: c['arms'][3].update(dropout=.2),
    lambda c: c['arms'][1].update(workload=['prior_records_per100']),
    lambda c: c['arms'][2].update(weight_decay=1e-4),
    lambda c: c['arms'][1].update(dropout=0),          # int instead of float: strict
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].reverse(),
    lambda c: c.update(seeds=[19, 20, 21]),
    lambda c: c.update(fits=12),
    lambda c: c.update(repeat_tag='x'),
])
def test_config_check_refuses_any_family_or_seed_drift(edit):
    cfg = load()
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', OTHERS, ids=lambda p: p.name)
def test_other_families_unchanged_and_refuse_an_ema_arm(path):
    cfg = json.loads(path.read_text())
    cfg['export'].pop('manifest_sha256', None)
    rc.config_check(copy.deepcopy(cfg))
    assert sc.ema_registry.check_contrasts(cfg) is None
    cfg['arms'].append(dict(emr.ARMS['ema'], name='ema', priority=1, ema=True, workload=[]))
    with pytest.raises(ValueError):
        rc.config_check(cfg)


def test_contrast_list_is_fixed_and_isolates_one_factor():
    arms = {a['name']: a for a in load()['arms']}
    for c in emr.CONTRASTS:
        cand, ref = arms[c['candidate']], arms[c['reference']]
        diff = {k for k in ('dropout', 'ema_decay') if cand.get(k, 0.) != ref.get(k, 0.)}
        assert diff == ({'ema_decay'} if c['isolates'] == 'ema' else {'dropout'})
    cfg = load()
    cfg['comparison'][emr.CONTRAST_KEY][0]['held'] = {}
    with pytest.raises(ValueError):
        emr.check_contrasts(cfg)


def test_modules_pinned_and_dispatch_keeps_old_paths():
    assert {'pitchmdp/direct_ema_model.py', 'pitchmdp/direct_ema_registry.py'} <= set(rc.MODEL_SOURCES)
    assert 'pitchmdp/direct_ema_registry.py' in sc.SCORER_SOURCES
    arms = {a['name']: a for a in load()['arms']}
    assert type(rc.build_model(arms['control'], 19, 'cpu')) is MatrixModel
    assert type(rc.build_model(arms['dropout'], 19, 'cpu')) is FlattenRegularizedModel
    for name, p in (('ema', 0.), ('dropout_ema', .1)):
        model = rc.build_model(arms[name], 19, 'cpu')
        assert type(model) is FlattenEMAModel and model.architecture['dropout'] == p
        assert model.architecture['ema_decay'] == .999
        with pytest.raises(ValueError):
            rc.build_model(arms[name], 19, 'cpu', context_names=['x'])


def test_runner_smoke_task_cpu_loader_and_seed_refusal(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'dropout_ema', 19, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    identity = json.loads((dest / 'identity.json').read_text())
    assert identity['seed'] == 19 and identity['arm']['ema'] is True and identity['arm']['dropout'] == .1
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    assert fit['parameter_count'] == 141834
    with np.load(dest / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(axis=1), 1.) and p['dev'].shape[1] == 10
    archive = next(dest.glob('*.pt'))
    arms = {a['name']: a for a in cfg['arms']}
    assert rc.load_model(arms['dropout_ema'], archive, 'cpu').report['weights'] == 'ema'
    with pytest.raises(ValueError):
        rc.load_model(arms['ema'], archive, 'cpu')  # other decay/dropout arm
    with pytest.raises(ValueError):
        rc.load_model(arms['dropout'], archive, 'cpu')  # EMA archive is not the frontier arm
    for seed in (0, 3, 8, 13, 16, 24):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'ema', seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', 'ema', 19, 'cpu', {})


@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the scorer default stays 10,000


def test_scorer_reports_all_ema_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    assert result['decision'] is None and result['ranking'] is None
    assert not {'factorial_contrasts', 'regularization_contrasts', 'workload_dropout_contrasts', 'tabm_contrasts',
                'embed_residual_contrasts'} & set(result)
    assert result['seed_block']['seeds'] == [19, 20, 21, 22, 23] and result['seed_block']['repeat_tag'] is None
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(emr.ARM_ORDER)
        block = result['ema_contrasts'][split]
        assert list(block) == [c['name'] for c in emr.CONTRASTS]
        vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in emr.CONTRASTS:
            one = block[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {str(s) for s in cfg['seeds']}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert '가중치 평균(EMA) 대비' in table and 'dropout_ema_vs_dropout' in table


def test_scorer_refuses_incomplete_missing_or_mixed_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=(('dropout_ema', 23),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][emr.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    for key, value in ((er.CONTRAST_KEY, er.CONTRASTS), (tr.CONTRAST_KEY, tr.CONTRASTS)):
        both = copy.deepcopy(cfg)
        both['comparison'][key] = value
        with pytest.raises(ValueError):
            tf.run_score(tmp_path, both, export, tmp_path / 'no-family-needed')
