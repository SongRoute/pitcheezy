"""Synthetic checks for the E1 residual direct head (CPU, tiny; no real data, no real fit)."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(PROJECT / 'tests')]
import run_ml_direct_residual as runner  # noqa: E402
import run_ml_direct_head as direct  # noqa: E402
from pitchmdp.data import hash_file  # noqa: E402
from pitchmdp.direct_residual_model import (ARCHITECTURE, FORMAT, DirectResidualModel,  # noqa: E402
                                            ResidualMLPNetwork, parameter_count)
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from pitchmdp.matrix_models import MatrixModel, MatrixNetwork  # noqa: E402
from test_run_ml_direct_head import Context, archives, arm, frame  # noqa: E402

REPO = PROJECT.parents[1]
CONFIG = json.loads((REPO / 'configs' / 'DRAFT-DIRECT-RESIDUAL-20261003.json').read_text())
P15 = json.loads((REPO / 'configs' / 'EXP-P15-001.yaml').read_text())


def arrays(n, rng, length=3, channels=24, n_context=5):
    tokens = rng.normal(size=(n, length, channels)).astype(np.float32)
    tokens[:, -1, -11:] = 0
    valid = np.ones((n, length), bool)
    valid[: n // 3, 0] = False
    return (tokens, valid, rng.normal(size=(n, n_context)).astype(np.float32)), rng.integers(0, 10, n)


class FlattenThroughMirror(DirectResidualModel):
    """The mirrored loop with the control network: must equal MatrixModel.fit exactly."""
    def network(self, n_context, n_token, length):
        return MatrixNetwork('flatten_mlp', n_context, n_token, length, 128, 10)


def tiny_fit(cls, seed=0, **kw):
    rng = np.random.default_rng(7)
    (train, y), (early, ey) = arrays(300, rng), arrays(100, rng)
    return cls(seed=seed, device='cpu', **kw).fit(train, y, early, ey, epochs=4, patience=2, batch_size=64,
                                                   learning_rate=.0005), early


def test_mirrored_fit_reproduces_baseline_matrix_model_fit():
    mirror, early = tiny_fit(FlattenThroughMirror)
    base, _ = tiny_fit(lambda seed, device: MatrixModel('flatten_mlp', seed=seed, width=128, device=device))
    for key, value in base.net.state_dict().items():
        assert torch.equal(value, mirror.net.state_dict()[key])
    assert mirror.report['history'] == base.report['history']
    for key in ('best_epoch', 'epochs_run', 'optimizer_updates', 'parameter_count', 'network', 'weight_ess', 'selection'):
        assert mirror.report[key] == base.report[key]
    assert np.array_equal(mirror.logits(early), base.logits(early))


def test_fixed_architecture_and_parameter_counts():
    shape = CONFIG['candidate_architecture']['input_contract']
    assert parameter_count(**shape) == CONFIG['candidate_architecture']['parameter_count'] == 170634
    control = MatrixNetwork('flatten_mlp', shape['n_context'], shape['n_token'], shape['length'], 128, 10)
    assert sum(p.numel() for p in control.parameters()) == CONFIG['control_parameter_count'] == 141834
    assert shape['length'] * (shape['n_token'] + 1) + shape['n_context'] == CONFIG['candidate_architecture']['n_flat']
    net = ResidualMLPNetwork(**shape)
    assert len(net.blocks) == 2 and net.input.out_features == 128
    for block in net.blocks:
        kinds = [type(m) for m in block]
        assert kinds == [torch.nn.LayerNorm, torch.nn.Linear, torch.nn.GELU, torch.nn.Dropout, torch.nn.Linear]
        assert block[1].out_features == 256 and block[3].p == .1
    assert [type(m) for m in net.output] == [torch.nn.LayerNorm, torch.nn.GELU, torch.nn.Linear]
    with pytest.raises(ValueError, match='fixed registered'):
        DirectResidualModel(architecture={**ARCHITECTURE, 'dropout': .2})


def test_residual_path_and_masking():
    net = ResidualMLPNetwork(5, 24, 3).eval()
    (tokens, valid, context), _ = arrays(8, np.random.default_rng(1))
    t, v, c = (torch.as_tensor(a) for a in (tokens, valid, context))
    z = net.input(torch.cat((t.masked_fill(~v[..., None], 0.).flatten(1), v.float(), c), 1))
    for block in net.blocks:
        z = z + block(z)
    assert torch.allclose(net(t, v, c), net.output(z))
    masked = t.clone()
    masked[~v] = 99.                                   # masked past tokens never reach the network
    assert torch.equal(net(masked, v, c), net(t, v, c))
    net.train()
    torch.manual_seed(0)
    assert not torch.equal(net(t, v, c), net(t, v, c))  # dropout active only in training


def test_checkpoint_roundtrip_schema_and_cross_loader_rejection(tmp_path):
    model, early = tiny_fit(DirectResidualModel, seed=3)
    assert model.report['kind'] == 'residual_mlp' and model.report['device'] == 'cpu'
    assert model.report['architecture'] == ARCHITECTURE and model.report['parameter_count'] == parameter_count(5, 24, 3)
    a = model.logits(early)
    assert np.array_equal(a, model.logits(early))      # eval logits are deterministic (dropout off)
    model.temperature, model.delivery_temperature = 1.3, .9
    model.save(tmp_path / 'model.pt')
    payload = torch.load(tmp_path / 'model.pt', weights_only=True)
    assert set(payload) == {'format', 'kind', 'seed', 'architecture', 'temperature', 'delivery_temperature',
                            'report', 'network_config', 'state_dict'} and payload['format'] == FORMAT
    loaded = DirectResidualModel.load(tmp_path / 'model.pt', 'cpu')
    assert np.array_equal(loaded.logits(early), a) and loaded.seed == 3
    assert (loaded.temperature, loaded.delivery_temperature) == (1.3, .9) and loaded.report == model.report
    assert np.allclose(loaded.predict(early).sum(1), 1.)
    with pytest.raises(ValueError, match='Unrecognized matrix model'):
        MatrixModel.load(tmp_path / 'model.pt')          # the sealed loader never mistakes it for flatten_mlp
    base, _ = tiny_fit(lambda seed, device: MatrixModel('flatten_mlp', seed=seed, device=device))
    base.save(tmp_path / 'base.pt')
    with pytest.raises(ValueError, match='Unrecognized direct residual'):
        DirectResidualModel.load(tmp_path / 'base.pt')
    payload['network_config'] = {**payload['network_config'], 'blocks': 3}
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='differs'):
        DirectResidualModel.load(tmp_path / 'tampered.pt')


def test_draft_config_matches_runner_and_refuses_real_runs():
    runner.config_check(CONFIG, smoke=True)
    with pytest.raises(ValueError, match='registered'):
        runner.config_check(CONFIG)
    registered = copy.deepcopy(CONFIG)
    registered['execution']['enabled'] = True
    with pytest.raises(ValueError, match='registered'):
        runner.config_check(registered)
    registered['status'] = 'REGISTERED'
    runner.config_check(registered)
    for path, value, message in ((('candidate_architecture', 'dropout'), .2, 'fixed registered'),
                                 (('candidate_architecture', 'blocks'), 3, 'fixed registered'),
                                 (('candidate_architecture', 'training', 'learning_rate'), .001, 'fixed registered'),
                                 (('candidate_architecture', 'hpo_trials'), 4, 'fixed registered'),
                                 (('candidate_architecture', 'parameter_count'), 141834, 'parameter count'),
                                 (('control_training', 'width'), 256, 'Control'),
                                 (('mix', 'direct_weight'), .7, 'Controls'),
                                 (('control', 'preparation_sha256'), '', 'pinned')):
        bad = copy.deepcopy(CONFIG)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            runner.config_check(bad, smoke=True)
    training = P15['unchanged_from_g0']['training']
    assert CONFIG['control_training'] == training
    assert CONFIG['candidate_architecture']['training'] == {k: training[k] for k in runner.BUDGET}
    assert CONFIG['control']['preparation_sha256'] == json.loads(
        (REPO / 'configs' / 'DRAFT-DIRECT-RECENCY-20261003.json').read_text())['control']['preparation_sha256']
    assert CONFIG['parent_preparation_sha256'] == P15['parent_preparation_sha256']


def local_file(tmp_path):
    local = tmp_path / 'local.json'
    local.write_text('{}')
    return local


def test_identity_tracks_sources_device_and_rejects_changed_source(tmp_path, monkeypatch):
    local = local_file(tmp_path)
    ident = runner.identity(CONFIG, local)
    assert ident['source_hashes'] == direct.source_hashes()
    assert ident['residual_source_hashes'] == {n: hash_file(PROJECT / n) for n in runner.OWN_SOURCES}
    assert 'pitchmdp/direct_residual_model.py' in runner.OWN_SOURCES and ident['architecture'] == ARCHITECTURE
    assert ident['device_option'] == 'auto' and ident != runner.identity(CONFIG, local, 'cpu')
    with pytest.raises(ValueError, match='device'):
        runner.identity(CONFIG, local, 'tpu')
    stale = copy.deepcopy(ident)
    stale['residual_source_hashes']['pitchmdp/direct_residual_model.py'] = '0' * 64
    with pytest.raises(ValueError, match='Sources changed'):
        runner.sources_unchanged({'identity': stale})
    runner.sources_unchanged({'identity': ident})


def synthetic(tmp_path, device='cpu'):
    f = pd.concat([frame()] * 4, ignore_index=True)
    f['game_pk'] = f.game_pk + 10 * np.repeat(np.arange(4), 8)
    f['game_date'] = f.game_date + pd.to_timedelta(np.repeat(np.arange(4), 8) * 7, unit='D')
    f['description'] = np.tile(['ball', 'called_strike', 'foul', 'swinging_strike'], 8)
    store = MatrixHistoryStore.from_frame(f)
    parts = {'train': f.iloc[:20], 'earlystop': f.iloc[20:26], 'temperature': f.iloc[20:26],
             'blend': f.iloc[26:29], 'dev': f.iloc[29:]}
    data = (None, {'samples': {}, 'clusters': {'cluster_counts': {}}}, store, Context(), parts, None)
    tokens, _, context = runner.training_arrays(direct.direct_arrays(store, Context(), [0]))
    shape = {'n_context': int(context.shape[1]), 'n_token': int(tokens.shape[2]), 'length': int(tokens.shape[1])}
    config = copy.deepcopy(CONFIG)
    config['candidate_architecture'].update(input_contract=shape, parameter_count=parameter_count(**shape))
    prep = {'identity': runner.identity(config, local_file(tmp_path), device)}
    return config, data, prep


def test_fit_and_predict_wiring_resume_and_guards(tmp_path):
    config, data, prep = synthetic(tmp_path)
    out = tmp_path / 'family'
    runner.fit(config, data, out, prep, 0, smoke=False)
    fitted = json.loads((out / '0' / 'fit.json').read_text())
    assert fitted['report']['kind'] == 'residual_mlp' and fitted['report']['device'] == 'cpu'
    assert fitted['report']['parameter_count'] == config['candidate_architecture']['parameter_count']
    stamp = (out / '0' / 'model.pt').stat().st_mtime_ns
    runner.fit(config, data, out, prep, 0, smoke=False)                 # verified reuse, no refit
    assert (out / '0' / 'model.pt').stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match='identity changed'):
        runner.fit(config, data, out, {'identity': runner.identity(config, local_file(tmp_path), 'auto')}, 0, smoke=False)
    runner.predict(data, out, prep, 0, smoke=False)
    with np.load(out / '0' / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(1), 1.) and np.all(p['dev_delivery_level'] == direct.NO_PHYSICS_LEVEL)
        assert len(p['blend']) == 3 and len(p['dev']) == 3
    runner.predict(data, out, prep, 0, smoke=False)                     # verified reuse
    (out / '0' / 'prediction_state.json').unlink()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    with pytest.raises(ValueError, match='Incomplete prediction'):
        runner.predict(data, out, prep, 0, smoke=False)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    wrong = copy.deepcopy(config)
    wrong['candidate_architecture']['input_contract'] = CONFIG['candidate_architecture']['input_contract']
    with pytest.raises(ValueError, match='Input contract'):
        runner.fit(wrong, data, tmp_path / 'other', prep, 1, smoke=False)
    (tmp_path / 'partial' / '1').mkdir(parents=True)
    (tmp_path / 'partial' / '1' / 'model.pt').write_bytes(b'half')
    with pytest.raises(ValueError, match='Incomplete fit'):
        runner.fit(config, data, tmp_path / 'partial', prep, 1, smoke=False)


def test_fit_inputs_never_contain_current_physics(tmp_path, monkeypatch):
    config, data, prep = synthetic(tmp_path)
    seen = []

    class Spy(DirectResidualModel):
        def fit(self, train, labels, early, early_labels, **kwargs):
            seen.append((train[0][:, -1, :8].copy(), early[0][:, -1, :8].copy(), kwargs))
            return super().fit(train, labels, early, early_labels, **kwargs)

    monkeypatch.setattr(runner, 'DirectResidualModel', Spy)   # test-only spy on this runner, not the sealed modules
    runner.fit(config, data, tmp_path / 'spy', prep, 0, smoke=False)
    train_physics, early_physics, kwargs = seen[0]
    assert not train_physics.any() and not early_physics.any()
    assert kwargs == {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}


def test_prepare_refuses_unsafe_or_changed_output(tmp_path):
    dest = tmp_path / 'out'
    dest.mkdir()
    (dest / 'stray.txt').write_text('x')
    config, data, prep = synthetic(tmp_path)
    with pytest.raises(ValueError, match='Incomplete preparation'):
        runner.prepare(config, data, dest, prep['identity'])


def test_descriptive_comparison_against_fixed_controls():
    baseline = archives(np.random.default_rng(0))
    g0, uniform = arm(baseline, (1, 2, 3)), arm(baseline, (4, 5, 6))
    same = runner.recency.describe(uniform, uniform, g0, baseline, CONFIG['mix']['direct_weight'])
    assert all(same[a]['after_june_blend']['paired']['nll']['delta'] == 0 for a in ('direct', 'mix'))
    assert 'N' not in same['direct']['after_june_blend']


def test_auto_cpu_preparation_is_not_comparable_to_mps_control(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
    config, data, prep = synthetic(tmp_path, device='auto')
    assert prep['identity']['resolved_device'] == 'cpu'
    made = runner.prepare(config, data, tmp_path / 'auto_cpu', prep['identity'])
    assert made['comparable_device_to_control'] is False
    explicit = runner.identity(config, local_file(tmp_path), 'mps')
    assert explicit['resolved_device'] == 'mps'
    made_mps = runner.prepare(config, data, tmp_path / 'explicit_mps', explicit)
    assert made_mps['comparable_device_to_control'] is True
