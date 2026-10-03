"""Synthetic checks for the periodic-embedding direct head (CPU, tiny; no real data, no real fit)."""
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
import run_ml_direct_periodic as runner  # noqa: E402
import run_ml_direct_head as direct  # noqa: E402
from pitchmdp.data import hash_file  # noqa: E402
from pitchmdp.direct_periodic_model import (ARCHITECTURE, FORMAT, DirectPeriodicModel,  # noqa: E402
                                            PeriodicDirectNetwork, context_map, parameter_count)
from pitchmdp.direct_residual_model import DirectResidualModel  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from pitchmdp.matrix_models import MatrixModel, MatrixNetwork  # noqa: E402
from test_run_ml_direct_head import archives, arm, frame  # noqa: E402

REPO = PROJECT.parents[1]
CONFIG = json.loads((REPO / 'configs' / 'DRAFT-DIRECT-PERIODIC-20261003.json').read_text())
RESIDUAL = json.loads((REPO / 'configs' / 'DRAFT-DIRECT-RESIDUAL-20261003.json').read_text())
PROFILE = CONFIG['candidate_architecture']['input_map']['pitcher_profile_columns']
MAP = context_map(PROFILE)


def arrays(n, rng, length=3, channels=24, n_context=MAP['n_context']):
    tokens = rng.normal(size=(n, length, channels)).astype(np.float32)
    tokens[:, -1, :8] = 0                                   # direct head: no current physics
    tokens[:, -1, -11:] = 0                                 # no current outcome
    valid = np.ones((n, length), bool)
    valid[: n // 3, 0] = False
    return (tokens, valid, rng.normal(size=(n, n_context)).astype(np.float32)), rng.integers(0, 10, n)


class FlattenThroughMirror(DirectPeriodicModel):
    """The inherited loop with the control network: must equal MatrixModel.fit exactly."""
    def network(self, n_context, n_token, length):
        return MatrixNetwork('flatten_mlp', n_context, n_token, length, 128, 10)


def tiny_fit(cls, seed=0, **kw):
    rng = np.random.default_rng(7)
    (train, y), (early, ey) = arrays(300, rng), arrays(100, rng)
    return cls(seed=seed, device='cpu', **kw).fit(train, y, early, ey, epochs=4, patience=2, batch_size=64,
                                                   learning_rate=.0005), early


def test_input_column_mapping_on_registered_contract():
    arch = CONFIG['candidate_architecture']
    cmap = arch['input_map']
    assert MAP['n_context'] == arch['input_contract']['n_context'] == 52
    assert cmap['context_raw'] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 32]      # ordinals, flags, left_share
    assert cmap['context_continuous'] == [*range(11, 32), *range(33, 52)] == MAP['continuous']
    assert PROFILE[4] == 'left_share' and 28 + 4 not in cmap['context_continuous']
    net = PeriodicDirectNetwork(**arch['input_contract'], context_continuous=cmap['context_continuous'])
    assert net.n_embedded == arch['n_embedded'] == 5 * 8 + 40
    assert net.head[0].in_features == arch['n_head_input'] == arch['n_raw'] + 80 * 8 == 846
    assert arch['n_raw'] + 80 == RESIDUAL['candidate_architecture']['n_flat']   # every control column used once
    for bad in (['release_spin_rate', 'effective_speed', 'pfx_x', 'pfx_z', 'left_share'],
                [*PROFILE[:4], 'right_share', *PROFILE[5:]], [*PROFILE, 'pitcher_id']):
        with pytest.raises(ValueError, match='layout'):
            context_map(bad)


def test_fixed_embedding_and_parameter_counts():
    arch = CONFIG['candidate_architecture']
    assert {k: arch[k] for k in ARCHITECTURE} == ARCHITECTURE
    assert (ARCHITECTURE['sigma'], ARCHITECTURE['n_frequencies'], ARCHITECTURE['d_embedding']) == (.01, 16, 8)
    count = parameter_count(**arch['input_contract'], context_continuous=MAP['continuous'])
    assert count == arch['parameter_count'] == 286738 == sum(arch['parameter_breakdown'].values())
    assert CONFIG['control_parameter_count'] == 141834
    net = PeriodicDirectNetwork(**arch['input_contract'], context_continuous=MAP['continuous'])
    assert [type(m) for m in net.head] == [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]
    assert net.head[0].out_features == 256 and net.head[2].out_features == 256
    assert net.frequencies.abs().max() <= .03 + 1e-7                      # truncated at 3 sigma
    with pytest.raises(ValueError, match='fixed registered'):
        DirectPeriodicModel(architecture={**ARCHITECTURE, 'sigma': .1})


def test_masks_and_zero_current_physics():
    torch.manual_seed(0)
    net = PeriodicDirectNetwork(MAP['n_context'], 24, 3, MAP['continuous']).eval()
    (tokens, valid, context), _ = arrays(9, np.random.default_rng(1))
    t, v, c = (torch.as_tensor(a) for a in (tokens, valid, context))
    e = net.embed(t.masked_fill(~v[..., None], 0.), v, c)
    assert torch.all(e[:3, :8] == 0) and torch.any(e[3:, :8] != 0)      # padded past physics embed to exactly 0
    masked = t.clone()
    masked[~v] = 99.                                                      # padding payload is never evidence
    assert torch.equal(net(masked, v, c), net(t, v, c))
    raw_only = c.clone()
    raw_only[:, MAP['continuous']] += 1.
    assert not torch.equal(net(t, v, raw_only), net(t, v, c))            # continuous context reaches the head
    moved = t.clone()
    moved[:, -1, :8] = 1.
    model = DirectPeriodicModel(context_continuous=MAP['continuous'], device='cpu')
    model.net = net
    assert np.isfinite(model.logits((tokens, valid, context))).all()
    with pytest.raises(ValueError, match='Current-pitch physics'):
        model.logits((moved.numpy(), valid, context))
    rng = np.random.default_rng(2)
    (train, y), (early, ey) = arrays(40, rng), arrays(20, rng)
    train[0][0, -1, 0] = .5
    with pytest.raises(ValueError, match='Current-pitch physics'):
        DirectPeriodicModel(context_continuous=MAP['continuous'], device='cpu').fit(train, y, early, ey, epochs=1)


def test_inherited_fit_reproduces_baseline_matrix_model_fit():
    mirror, early = tiny_fit(FlattenThroughMirror, context_continuous=MAP['continuous'])
    base, _ = tiny_fit(lambda seed, device: MatrixModel('flatten_mlp', seed=seed, width=128, device=device))
    for key, value in base.net.state_dict().items():
        assert torch.equal(value, mirror.net.state_dict()[key])
    assert mirror.report['history'] == base.report['history']
    assert np.array_equal(mirror.logits(early), base.logits(early))


def test_train_predict_save_load_equivalence_and_cross_loader_rejection(tmp_path):
    model, early = tiny_fit(DirectPeriodicModel, seed=3, context_continuous=MAP['continuous'])
    assert model.report['kind'] == 'periodic_flatten_mlp' and model.report['device'] == 'cpu'
    assert model.report['architecture'] == ARCHITECTURE
    assert model.report['parameter_count'] == parameter_count(MAP['n_context'], 24, 3, MAP['continuous'])
    a = model.logits(early)
    model.temperature, model.delivery_temperature = 1.3, .9
    model.save(tmp_path / 'model.pt')
    payload = torch.load(tmp_path / 'model.pt', weights_only=True)
    assert payload['format'] == FORMAT and payload['network_config']['context_continuous'] == MAP['continuous']
    loaded = DirectPeriodicModel.load(tmp_path / 'model.pt', 'cpu')
    assert np.array_equal(loaded.logits(early), a) and loaded.seed == 3
    assert np.array_equal(loaded.predict(early), model.predict(early)) and loaded.report == model.report
    with pytest.raises(ValueError, match='Unrecognized matrix model'):
        MatrixModel.load(tmp_path / 'model.pt')
    with pytest.raises(ValueError, match='Unrecognized direct residual'):
        DirectResidualModel.load(tmp_path / 'model.pt')
    base, _ = tiny_fit(lambda seed, device: MatrixModel('flatten_mlp', seed=seed, device=device))
    base.save(tmp_path / 'base.pt')
    with pytest.raises(ValueError, match='Unrecognized direct periodic'):
        DirectPeriodicModel.load(tmp_path / 'base.pt')
    payload['network_config'] = {**payload['network_config'], 'n_frequencies': 48}
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='differs'):
        DirectPeriodicModel.load(tmp_path / 'tampered.pt')


def test_draft_config_matches_runner_and_refuses_real_runs():
    runner.config_check(CONFIG, smoke=True)
    with pytest.raises(ValueError, match='registered'):
        runner.config_check(CONFIG)
    registered = copy.deepcopy(CONFIG)
    registered['execution']['enabled'], registered['status'] = True, 'REGISTERED'
    runner.config_check(registered)
    for path, value, message in ((('candidate_architecture', 'sigma'), .1, 'fixed registered'),
                                 (('candidate_architecture', 'n_frequencies'), 48, 'fixed registered'),
                                 (('candidate_architecture', 'hpo_trials'), 4, 'fixed registered'),
                                 (('candidate_architecture', 'parameter_count'), 141834, 'parameter count'),
                                 (('candidate_architecture', 'input_map', 'context_continuous'), list(range(11, 52)), 'context map'),
                                 (('control_training', 'width'), 256, 'Control'),
                                 (('mix', 'direct_weight'), .7, 'Controls')):
        bad = copy.deepcopy(CONFIG)
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            runner.config_check(bad, smoke=True)
    for key in ('control', 'control_training', 'control_parameter_count', 'parent_preparation_sha256', 'mix', 'seeds'):
        assert CONFIG[key] == RESIDUAL[key]
    assert CONFIG['candidate_architecture']['training'] == runner.BUDGET


def local_file(tmp_path):
    local = tmp_path / 'local.json'
    local.write_text('{}')
    return local


class Context:
    """Sharing-style context: 52 direct columns plus seven routing columns stripped by training_arrays."""
    def transform(self, part):
        rng = np.random.default_rng(len(part))
        return np.column_stack((rng.normal(size=(len(part), MAP['n_context'])), np.zeros((len(part), 7)))).astype(np.float32)


def synthetic(tmp_path, device='cpu'):
    f = pd.concat([frame()] * 4, ignore_index=True)
    f['game_pk'] = f.game_pk + 10 * np.repeat(np.arange(4), 8)
    f['game_date'] = f.game_date + pd.to_timedelta(np.repeat(np.arange(4), 8) * 7, unit='D')
    f['description'] = np.tile(['ball', 'called_strike', 'foul', 'swinging_strike'], 8)
    store = MatrixHistoryStore.from_frame(f)
    parts = {'train': f.iloc[:20], 'earlystop': f.iloc[20:26], 'temperature': f.iloc[20:26],
             'blend': f.iloc[26:29], 'dev': f.iloc[29:]}
    data = (None, {'samples': {}, 'clusters': {'cluster_counts': {}, 'columns': list(PROFILE)}}, store, Context(), parts, None)
    tokens, _, context = runner.training_arrays(direct.direct_arrays(store, Context(), [0]))
    shape = {'n_context': int(context.shape[1]), 'n_token': int(tokens.shape[2]), 'length': int(tokens.shape[1])}
    config = copy.deepcopy(CONFIG)
    config['candidate_architecture'].update(input_contract=shape,
                                            parameter_count=parameter_count(**shape, context_continuous=MAP['continuous']))
    prep = {'identity': runner.identity(config, local_file(tmp_path), device)}
    return config, data, prep


def test_identity_tracks_sources_map_and_device(tmp_path):
    local = local_file(tmp_path)
    ident = runner.identity(CONFIG, local)
    assert ident['source_hashes'] == direct.source_hashes()
    assert ident['periodic_source_hashes'] == {n: hash_file(PROJECT / n) for n in runner.OWN_SOURCES}
    assert {'pitchmdp/direct_periodic_model.py', 'pitchmdp/direct_residual_model.py'} <= set(runner.OWN_SOURCES)
    assert ident['architecture'] == ARCHITECTURE and ident['input_map'] == CONFIG['candidate_architecture']['input_map']
    assert ident['device_option'] == 'auto' and ident != runner.identity(CONFIG, local, 'cpu')
    with pytest.raises(ValueError, match='device'):
        runner.identity(CONFIG, local, 'tpu')
    stale = copy.deepcopy(ident)
    stale['periodic_source_hashes']['pitchmdp/direct_periodic_model.py'] = '0' * 64
    with pytest.raises(ValueError, match='Sources changed'):
        runner.sources_unchanged({'identity': stale})
    runner.sources_unchanged({'identity': ident})


def test_fit_predict_wiring_verified_reuse_and_no_partial_resume(tmp_path):
    config, data, prep = synthetic(tmp_path)
    out = tmp_path / 'family'
    runner.fit(config, data, out, prep, 0, smoke=False)
    fitted = json.loads((out / '0' / 'fit.json').read_text())
    assert fitted['report']['kind'] == 'periodic_flatten_mlp' and fitted['report']['device'] == 'cpu'
    assert fitted['report']['network']['context_continuous'] == MAP['continuous']
    stamp = (out / '0' / 'model.pt').stat().st_mtime_ns
    runner.fit(config, data, out, prep, 0, smoke=False)                 # verified reuse, no refit
    assert (out / '0' / 'model.pt').stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match='identity changed'):
        runner.fit(config, data, out, {'identity': runner.identity(config, local_file(tmp_path), 'auto')}, 0, smoke=False)
    runner.predict(data, out, prep, 0, smoke=False)
    with np.load(out / '0' / 'predictions.npz') as p:
        assert np.allclose(p['dev'].sum(1), 1.) and np.all(p['dev_delivery_level'] == direct.NO_PHYSICS_LEVEL)
    model = DirectPeriodicModel.load(out / '0' / 'model.pt', 'cpu')
    ta = runner.training_arrays(direct.direct_arrays(data[2], Context(), data[4]['dev'].index.to_numpy()))
    assert np.array_equal(model.logits(ta), DirectPeriodicModel.load(out / '0' / 'model.pt', 'cpu').logits(ta))
    (out / '0' / 'prediction_state.json').unlink()                       # interrupted after the archive was written
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    with pytest.raises(ValueError, match='Incomplete prediction'):
        runner.predict(data, out, prep, 0, smoke=False)
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (out / '0').iterdir()}
    (tmp_path / 'partial' / '1').mkdir(parents=True)
    (tmp_path / 'partial' / '1' / 'model.pt').write_bytes(b'half')
    with pytest.raises(ValueError, match='Incomplete fit'):
        runner.fit(config, data, tmp_path / 'partial', prep, 1, smoke=False)
    moved = (data[0], {**data[1], 'clusters': {'columns': [*PROFILE[:4], 'type_X', *PROFILE[5:]]}}, *data[2:])
    with pytest.raises(ValueError, match='profile columns'):
        runner.fit(config, moved, tmp_path / 'moved', prep, 2, smoke=False)
    dest = tmp_path / 'stray'
    dest.mkdir()
    (dest / 'x.txt').write_text('x')
    with pytest.raises(ValueError, match='Incomplete preparation'):
        runner.prepare(config, data, dest, prep['identity'])


def test_fit_inputs_never_contain_current_physics(tmp_path, monkeypatch):
    config, data, prep = synthetic(tmp_path)
    seen = []

    class Spy(DirectPeriodicModel):
        def fit(self, train, labels, early, early_labels, **kwargs):
            seen.append((train[0][:, -1, :8].copy(), early[0][:, -1, :8].copy(), self.context_continuous, kwargs))
            return super().fit(train, labels, early, early_labels, **kwargs)

    monkeypatch.setattr(runner, 'DirectPeriodicModel', Spy)
    runner.fit(config, data, tmp_path / 'spy', prep, 0, smoke=False)
    train_physics, early_physics, continuous, kwargs = seen[0]
    assert not train_physics.any() and not early_physics.any() and continuous == MAP['continuous']
    assert kwargs == {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005}


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
    assert explicit['resolved_device'] == 'mps' and explicit != prep['identity']
    assert runner.prepare(config, data, tmp_path / 'explicit_mps', explicit)['comparable_device_to_control'] is True
