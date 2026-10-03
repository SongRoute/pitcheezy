"""Synthetic CPU tests: architecture x dropout family (DRAFT-CUDA-DROPOUT-RUN-20261004) in the CUDA runner and scorer."""
import copy
import json
from pathlib import Path
import subprocess
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
from pitchmdp import direct_dropout_registry as dr  # noqa: E402
from pitchmdp import direct_dropout_model as dm  # noqa: E402
from pitchmdp.direct_periodic_model import DirectPeriodicModel, PeriodicDirectNetwork  # noqa: E402
from pitchmdp.direct_residual_model import DirectResidualModel, ResidualMLPNetwork  # noqa: E402

DROPOUT_RUN = REPO / 'configs' / 'DRAFT-CUDA-DROPOUT-RUN-20261004.json'
COMBO = REPO / 'configs' / 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json'
FRONTIER_RUN = REPO / 'configs' / 'DRAFT-CUDA-FRONTIER-RUN-20261004.json'
CONT = dr.PERIODIC_CONTINUOUS


def load(registered=False, unpin=True):
    return tf.load(DROPOUT_RUN, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


def arrays(n, seed):
    a = tf.split_arrays(n, seed, 0)
    return (a['tokens'], a['valid'], a['context']), a['y']


# ------------------------------------------------------------------ config

def test_config_is_draft_exact_3x2_factorial():
    cfg = load(unpin=False)
    combo = {a['name']: a for a in json.loads(COMBO.read_text())['arms']}
    frontier = {a['name']: a for a in json.loads(FRONTIER_RUN.read_text())['arms']}
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False}
    assert [a['name'] for a in cfg['arms']] == list(dr.ARM_ORDER) and cfg['fits'] == 18 == 6 * len(cfg['seeds'])
    arms = {a['name']: a for a in cfg['arms']}
    for name in ('control', 'residual', 'periodic'):
        assert arms[name] == combo[name] == dr.REFERENCES[name]  # fresh same-architecture no-dropout arms
    assert arms['dropout'] == frontier['dropout']
    assert len(CONT) == 40 and arms['periodic_dropout']['context_continuous'] == arms['periodic']['context_continuous'] == CONT
    assert all(a['workload'] == [] for a in cfg['arms'])
    assert cfg['training'] == rc.TRAINING and cfg['comparison']['control_arm'] == 'control'
    assert cfg['comparison']['regularization_contrasts'] == dr.CONTRASTS and 'factorial_contrasts' not in cfg['comparison']
    assert cfg['export'] == json.loads(COMBO.read_text())['export']
    assert sc.check_contrasts(cfg) is None and dr.check_contrasts(cfg) == dr.CONTRASTS


@pytest.mark.parametrize('edit', [
    lambda arms: arms[3].update(head_dropout=.2),
    lambda arms: arms[3].update(workload=['prior_records_per100']),
    lambda arms: arms[5].update(context_continuous=CONT[:-1]),
    lambda arms: arms[5].update(model='residual_head_dropout_mlp'),
    lambda arms: arms[1].update(weight_decay=1e-4),
    lambda arms: arms[2].update(workload=['prior_records_per100']),
    lambda arms: arms[4].update(context_continuous=[*CONT, 52]),
    lambda arms: arms.pop(4),
    lambda arms: arms.insert(0, arms.pop(3)),
    lambda arms: arms[3].pop('dropout_head'),
])
def test_config_check_refuses_any_family_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg['arms'])
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[1].update(reference='control'),             # cross-architecture
    lambda cs: cs[2].update(candidate='periodic', reference='periodic_dropout'),  # reversed sign
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
    lambda cs: cs.clear(),
])
def test_regularization_contrasts_are_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison']['regularization_contrasts'])
    with pytest.raises(ValueError):
        dr.check_contrasts(cfg)


def test_modules_are_pinned_sources_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_dropout_model.py', 'pitchmdp/direct_dropout_registry.py'} <= set(rc.MODEL_SOURCES)
    assert set(rc.own_hashes()) == {rc.SELF, *rc.MODEL_SOURCES}
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(DROPOUT_RUN)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ networks

def pair(kind, p):
    torch.manual_seed(7)
    if kind == 'residual':
        base = ResidualMLPNetwork(52, 38, 6)
        torch.manual_seed(7)
        return base, dm.ResidualHeadDropoutNetwork(52, 38, 6, head_dropout=p)
    base = PeriodicDirectNetwork(52, 38, 6, CONT)
    torch.manual_seed(7)
    return base, dm.PeriodicHeadDropoutNetwork(52, 38, 6, CONT, head_dropout=p)


@pytest.mark.parametrize('kind', ['residual', 'periodic'])
def test_dropout_networks_same_parameters_and_placement(kind):
    base, net = pair(kind, .1)
    assert [p.shape for p in base.parameters()] == [p.shape for p in net.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(base.parameters(), net.parameters()))  # identical init order
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in net.parameters())
    drops = [m for m in net.modules() if isinstance(m, torch.nn.Dropout)]
    if kind == 'residual':  # 2 frozen block dropouts + 1 head dropout after the output GELU
        assert [d.p for d in drops] == [.1, .1, .1] and isinstance(net.output[2], torch.nn.Dropout)
        assert isinstance(net.output[1], torch.nn.GELU)
    else:  # after both head GELUs; never in the embedding
        assert [d.p for d in drops] == [.1, .1]
        assert [type(m).__name__ for m in net.head] == ['Linear', 'GELU', 'Dropout', 'Linear', 'GELU', 'Dropout', 'Linear']
    x, _ = arrays(64, 3)
    t = [torch.as_tensor(a) for a in x]
    base.eval(), net.eval()
    assert torch.equal(base(*t), net(*t))  # eval: dropout off, same weights -> identical logits
    net.train()
    torch.manual_seed(1)
    a = net(*t)
    torch.manual_seed(2)
    assert not torch.allclose(a, net(*t))  # train: dropout active and stochastic


@pytest.mark.parametrize('kind', ['residual', 'periodic'])
def test_zero_dropout_matches_base_bitwise_in_train_mode(kind):
    base, net = pair(kind, 0.)
    x, _ = arrays(64, 4)
    t = [torch.as_tensor(a) for a in x]
    base.train(), net.train()
    torch.manual_seed(3)
    a = base(*t)
    torch.manual_seed(3)
    assert torch.equal(a, net(*t))


def small_fit(model, seed=0):
    train, y = arrays(300, 11)
    early, ey = arrays(90, 12)
    return model.fit(train, y, early, ey, epochs=3, patience=2, batch_size=64)


@pytest.mark.parametrize('kind', ['residual', 'periodic'])
def test_zero_dropout_training_reproduces_frozen_base_bitwise(kind):
    if kind == 'residual':
        base, ours = DirectResidualModel(5, device='cpu'), dm.ResidualHeadDropoutModel(5, 0., 'cpu')
    else:
        base, ours = DirectPeriodicModel(5, CONT, device='cpu'), dm.PeriodicHeadDropoutModel(5, CONT, 0., 'cpu')
    small_fit(base), small_fit(ours)
    assert [h['earlystop_conditional_nll'] for h in base.report['history']] == \
        [h['earlystop_conditional_nll'] for h in ours.report['history']]
    assert all(torch.equal(a, b) for a, b in zip(base.net.parameters(), ours.net.parameters()))
    x, _ = arrays(50, 13)
    assert np.array_equal(base.logits(x), ours.logits(x))
    assert base.report['parameter_count'] == ours.report['parameter_count']
    drop = small_fit(dm.ResidualHeadDropoutModel(5, .1, 'cpu') if kind == 'residual'
                     else dm.PeriodicHeadDropoutModel(5, CONT, .1, 'cpu'))
    assert drop.report['parameter_count'] == base.report['parameter_count']
    assert drop.report['history'][0]['train_weighted_nll'] != base.report['history'][0]['train_weighted_nll']


def test_periodic_dropout_keeps_zero_current_physics_guard():
    model = small_fit(dm.PeriodicHeadDropoutModel(0, CONT, .1, 'cpu'))
    x, _ = arrays(20, 14)
    bad = (x[0].copy(), x[1], x[2])
    bad[0][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='physics'):
        model.logits(bad)


@pytest.mark.parametrize('name', ['residual_dropout', 'periodic_dropout'])
def test_checkpoint_roundtrip_and_architecture_guard(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    model = small_fit(rc.build_model(spec, 1, 'cpu'))
    model.save(tmp_path / 'm.pt')
    again = rc.load_model(spec, tmp_path / 'm.pt', 'cpu')
    assert type(again) is type(model) and again.architecture == model.architecture
    assert again.architecture['head_dropout'] == .1 and again.net.config['head_dropout'] == .1
    assert [m.p for m in again.net.modules() if isinstance(m, torch.nn.Dropout)][-1] == .1
    x, _ = arrays(30, 15)
    assert np.array_equal(model.logits(x), again.logits(x))
    other = 'periodic_dropout' if name == 'residual_dropout' else 'residual_dropout'
    with pytest.raises((ValueError, KeyError)):
        rc.load_model(rc.arm_spec(cfg, other), tmp_path / 'm.pt', 'cpu')
    base = rc.arm_spec(cfg, name.replace('_dropout', ''))
    with pytest.raises(ValueError):
        rc.load_model(base, tmp_path / 'm.pt', 'cpu')  # frozen base loader refuses the dropout archive
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['network_config']['head_dropout'] = 0.
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError):
        rc.load_model(spec, tmp_path / 'bad.pt', 'cpu')
    zero = small_fit(rc.build_model(spec, 1, 'cpu'))
    zero.architecture['head_dropout'] = 0.
    zero.save(tmp_path / 'zero.pt')
    with pytest.raises(ValueError):
        rc.load_model(spec, tmp_path / 'zero.pt', 'cpu')


@pytest.mark.parametrize('name', ['residual_dropout', 'periodic_dropout'])
def test_runner_smoke_task_cpu(export, tmp_path, name):
    path, digest, splits = export
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', name, 0, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 52
    model = rc.load_model(spec, dest / 'model.pt', 'cpu')
    assert model.kind == spec['model'] and model.architecture['head_dropout'] == .1
    state = json.loads((dest / 'state.json').read_text())
    assert set(state['identity']['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    with pytest.raises(ValueError):
        rc.build_model(spec, 0, 'cpu', ['x'] * 52)


# ------------------------------------------------------------------ scorer

def test_scorer_reports_regularization_contrasts(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    result = tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    assert result['decision'] is None and result['ranking'] is None and 'factorial_contrasts' not in result
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(dr.ARM_ORDER)
        assert all('paired_vs_control' in arms[a] for a in dr.ARM_ORDER[1:])  # baseline vs control retained
        block = result['regularization_contrasts'][split]
        assert list(block) == [c['name'] for c in dr.CONTRASTS]
        for c in dr.CONTRASTS:
            one = block[c['name']]
            assert one['isolates'] == 'dropout' and set(one['seeds']) == {'0', '1', '2'}
            vs = lambda a: 0. if a == 'control' else arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
            # candidate MINUS reference (sign), same helper as vs-control
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    table = (tmp_path / 'score' / 'table_ko.md').read_text()
    assert 'periodic_dropout_vs_periodic' in table


def test_scorer_refuses_bad_contrast_or_incomplete_family_before_reading(export, tmp_path):
    path, digest, splits = export
    cfg = load(registered=True)
    bad = copy.deepcopy(cfg)
    bad['comparison']['regularization_contrasts'][1]['reference'] = 'control'
    with pytest.raises(ValueError, match='regularization_contrasts'):
        tf.run_score(tmp_path, bad, export, tmp_path / 'no-family-needed')
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=(('periodic_dropout', 2),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


def test_older_families_have_no_regularization_block():
    for path in (COMBO, FRONTIER_RUN):
        assert dr.check_contrasts(tf.load(path, True, True)) is None
