"""Synthetic CPU checks for the embedded-input residual networks (DRAFT; no real data, no registered fit)."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp.direct_combo_registry import BASE_PERIODIC  # noqa: E402
from pitchmdp.direct_periodic_model import DirectPeriodicModel, PeriodicDirectNetwork  # noqa: E402
from pitchmdp.direct_periodic_model import parameter_count as periodic_count  # noqa: E402
from pitchmdp.direct_residual_model import DirectResidualModel, ResidualMLPNetwork  # noqa: E402
from pitchmdp.direct_residual_model import parameter_count as residual_count  # noqa: E402

torch.set_num_threads(1)
CONTINUOUS = [*BASE_PERIODIC, 52]                     # periodic_counts map (context 52 + prior_records_per100)
REAL = dict(n_context=53, n_token=38, length=6)       # direct contract; frozen counts reproduce 286,738 / 170,634 at 52
SMALL_CONTINUOUS = [2, 3, 5]
MODELS = {'periodic': pr.DirectPeriodicResidualModel, 'linear': pr.DirectLinearResidualModel}


def arrays(n, rng, length=3, channels=24, n_context=7):
    tokens = rng.normal(size=(n, length, channels)).astype(np.float32)
    tokens[:, -1, :8] = 0
    tokens[:, -1, -11:] = 0
    valid = np.ones((n, length), bool)
    valid[: n // 3, 0] = False
    return (tokens, valid, rng.normal(size=(n, n_context)).astype(np.float32)), rng.integers(0, 10, n)


def tiny_fit(embedding, seed=0):
    rng = np.random.default_rng(7)
    (train, y), (early, ey) = arrays(300, rng), arrays(100, rng)
    model = MODELS[embedding](seed, SMALL_CONTINUOUS, device='cpu')
    return model.fit(train, y, early, ey, epochs=3, patience=2, batch_size=64, learning_rate=.0005), early


def small_net(embedding, seed=0):
    torch.manual_seed(seed)
    return pr.NETWORKS[embedding](7, 24, 3, SMALL_CONTINUOUS, **pr.network_kwargs(embedding, pr.ARCHITECTURES[embedding]))


def test_frozen_shapes_and_parameter_counts_on_the_real_contract():
    assert periodic_count(52, 38, 6, BASE_PERIODIC) == 286738 and residual_count(52, 38, 6) == 170634
    assert len(CONTINUOUS) == 41
    n_embedded = 5 * 8 + 41
    n_raw = 5 * 30 + 38 + 6 + 53 - 41
    d_in = n_raw + 8 * n_embedded
    backbone = d_in * 128 + 128 + 2 * (2 * 128 + 128 * 256 + 256 + 256 * 128 + 128) + 2 * 128 + 128 * 10 + 10
    assert pr.parameter_count('periodic', **REAL, context_continuous=CONTINUOUS) == \
        n_embedded * 16 + 32 * 8 + 8 + backbone == 244898
    assert pr.parameter_count('linear', **REAL, context_continuous=CONTINUOUS) == 2 * n_embedded * 8 + backbone == 244634
    assert periodic_count(53, 38, 6, CONTINUOUS) == 288802 and residual_count(53, 38, 6) == 170762


def test_architectures_are_the_frozen_backbone_and_embedding():
    for embedding in pr.ARCHITECTURES:
        assert pr.ARCHITECTURES[embedding]['backbone'] == {'kind': 'residual_mlp', **pr.residual.ARCHITECTURE}
    assert pr.residual.ARCHITECTURE['activation'] == 'gelu'
    assert pr.ARCHITECTURES['periodic']['embedding'] == pr.periodic.EMBEDDING
    assert pr.ARCHITECTURES['linear']['embedding'] == {'type': 'LR', 'd_embedding': 8, 'activation': 'relu'}
    for embedding, net in (('periodic', small_net('periodic')), ('linear', small_net('linear'))):
        assert not hasattr(net, 'head') and net.config['kind'] == pr.KINDS[embedding]
        assert (net.input.out_features, len(net.blocks), net.blocks[0][3].p) == (128, 2, .1)
    assert not hasattr(small_net('linear'), 'frequencies')


def test_periodic_embedding_initialisation_and_values_equal_the_frozen_network():
    net = small_net('periodic', seed=3)
    torch.manual_seed(3)
    frozen = PeriodicDirectNetwork(7, 24, 3, SMALL_CONTINUOUS)
    assert torch.equal(net.frequencies, frozen.frequencies)
    assert all(torch.equal(a, b) for a, b in zip(net.embedding.parameters(), frozen.embedding.parameters()))
    (tokens, valid, context), _ = arrays(20, np.random.default_rng(1))
    args = [torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(context)]
    args[0] = args[0].masked_fill(~args[1][..., None], 0.)
    assert torch.equal(net.embed(*args), frozen.embed(*args))


def test_backbone_mirrors_residual_mlp_network_blocks_and_output():
    net = small_net('periodic').eval()
    ref = ResidualMLPNetwork(7, 24, 3).eval()
    assert [(k, v.shape) for k, v in ref.blocks.state_dict().items()] == \
        [(k, v.shape) for k, v in net.blocks.state_dict().items()]
    ref.blocks.load_state_dict(net.blocks.state_dict())
    ref.output.load_state_dict(net.output.state_dict())
    x = torch.randn(11, net.input.in_features)
    z = net.input(x)
    for block in ref.blocks:
        z = z + block(z)
    assert torch.allclose(net.backbone(x), ref.output(z), atol=0, rtol=0)


@pytest.mark.parametrize('embedding', ['periodic', 'linear'])
def test_padded_past_physics_embed_to_zero_and_gradients_reach_the_embedding(embedding):
    net = small_net(embedding)
    (tokens, valid, context), y = arrays(30, np.random.default_rng(2))
    t, v, c = torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(context)
    e = net.embed(t.masked_fill(~v[..., None], 0.), v, c)
    assert e.shape == (30, net.n_embedded, 8)
    assert torch.count_nonzero(e[:10, :8]) == 0 and torch.count_nonzero(e[10:, :8]) > 0
    torch.nn.functional.cross_entropy(net(t, v, c), torch.as_tensor(y)).backward()
    first = net.frequencies if embedding == 'periodic' else net.weight
    assert first.grad is not None and torch.count_nonzero(first.grad) > 0


@pytest.mark.parametrize('embedding', ['periodic', 'linear'])
def test_fit_report_save_load_round_trip_and_refusals(embedding, tmp_path):
    model, early = tiny_fit(embedding)
    report = model.report
    assert report['architecture'] == pr.ARCHITECTURES[embedding] and report['network']['kind'] == pr.KINDS[embedding]
    assert report['parameter_count'] == sum(p.numel() for p in model.net.parameters()) and report['epochs_run'] >= 1
    path = tmp_path / 'model.pt'
    model.save(path)
    loaded = MODELS[embedding].load(path, device='cpu')
    assert np.array_equal(loaded.logits(early), model.logits(early))
    assert loaded.context_continuous == SMALL_CONTINUOUS
    other = MODELS['linear' if embedding == 'periodic' else 'periodic']
    with pytest.raises(ValueError, match='Unrecognized'):
        other.load(path, device='cpu')
    with pytest.raises(ValueError, match='Unrecognized'):
        DirectPeriodicModel.load(path, device='cpu')
    with pytest.raises(ValueError, match='differs'):
        MODELS[embedding](0, SMALL_CONTINUOUS, {**pr.ARCHITECTURES[embedding], 'backbone': {**pr.ARCHITECTURES[embedding]['backbone'], 'blocks': 3}}, device='cpu')
    payload = torch.load(path, weights_only=True)
    payload['network_config']['hidden'] = 512
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='Archived network differs'):
        MODELS[embedding].load(tmp_path / 'tampered.pt', device='cpu')


def test_seeded_fits_repeat_exactly_and_current_physics_is_refused():
    a, early = tiny_fit('periodic', seed=4)
    b, _ = tiny_fit('periodic', seed=4)
    assert np.array_equal(a.logits(early), b.logits(early))
    bad = list(early)
    bad[0] = bad[0].copy()
    bad[0][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='Current-pitch physics'):
        a.logits(bad)


# --- added by same-model review (docs/reviews/OPUS-review-periodic-residual-1b1ff30-2026-10-04.md) ---

def test_review_architecture_is_a_deep_copy_not_an_alias_of_the_fixed_constant():
    for embedding, model_cls in MODELS.items():
        model = model_cls(0, SMALL_CONTINUOUS, device='cpu')
        model.architecture['backbone']['blocks'] = 3
        model.architecture['embedding']['d_embedding'] = 4
        assert pr.ARCHITECTURES[embedding]['backbone']['blocks'] == 2
        assert pr.ARCHITECTURES[embedding]['embedding']['d_embedding'] == 8
        with pytest.raises(ValueError, match='differs'):
            model_cls(0, SMALL_CONTINUOUS, model.architecture, device='cpu')


@pytest.mark.parametrize('embedding', ['periodic', 'linear'])
def test_review_exact_parameter_names_leave_no_discarded_module(embedding):
    backbone = ['input.weight', 'input.bias'] + [f'blocks.{b}.{i}.{p}' for b in range(2) for i in (0, 1, 4)
                                                for p in ('weight', 'bias')] + \
               [f'output.{i}.{p}' for i in (0, 2) for p in ('weight', 'bias')]
    first = ['frequencies', 'embedding.weight', 'embedding.bias'] if embedding == 'periodic' else ['weight', 'bias']
    net = small_net(embedding)
    assert sorted(net.state_dict()) == sorted(first + backbone)
    assert sorted(n for n, _ in net.named_parameters()) == sorted(first + backbone)
    assert not hasattr(net, 'head') and (embedding == 'periodic' or not hasattr(net, 'embedding'))


def test_review_periodic_backbone_input_is_the_frozen_flatten_head_input():
    torch.manual_seed(5)
    frozen = PeriodicDirectNetwork(7, 24, 3, SMALL_CONTINUOUS)
    frozen.head = torch.nn.Identity()
    net = small_net('periodic', seed=5)
    net.backbone = lambda x: x
    (tokens, valid, context), _ = arrays(20, np.random.default_rng(3))
    args = [torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(context)]
    assert torch.equal(net(*args), frozen(*args))


def test_review_linear_and_periodic_share_backbone_initialisation_for_a_seed():
    periodic_params = dict(small_net('periodic', seed=6).named_parameters())
    linear_params = dict(small_net('linear', seed=6).named_parameters())
    shared = [k for k in linear_params if k.startswith(('input.', 'blocks.', 'output.'))]
    assert len(shared) == 18 and all(torch.equal(periodic_params[k], linear_params[k]) for k in shared)


def test_review_frozen_and_new_archives_do_not_cross_load(tmp_path):
    rng = np.random.default_rng(7)
    (train, y), (early, ey) = arrays(120, rng), arrays(40, rng)
    budget = dict(epochs=1, patience=1, batch_size=64)
    DirectResidualModel(0, device='cpu').fit(train, y, early, ey, **budget).save(tmp_path / 'residual.pt')
    DirectPeriodicModel(0, SMALL_CONTINUOUS, device='cpu').fit(train, y, early, ey, **budget).save(tmp_path / 'periodic.pt')
    for model_cls in MODELS.values():
        for name in ('residual.pt', 'periodic.pt'):
            with pytest.raises(ValueError, match='Unrecognized'):
                model_cls.load(tmp_path / name, device='cpu')
    MODELS['periodic'](0, SMALL_CONTINUOUS, device='cpu').fit(train, y, early, ey, **budget).save(tmp_path / 'new.pt')
    with pytest.raises(ValueError, match='Unrecognized'):
        DirectResidualModel.load(tmp_path / 'new.pt', device='cpu')


@pytest.mark.parametrize('embedding', ['periodic', 'linear'])
def test_review_fit_refuses_current_physics_and_tampered_embedding_archives(embedding, tmp_path):
    rng = np.random.default_rng(7)
    (train, y), (early, ey) = arrays(120, rng), arrays(40, rng)
    for which in (0, 1):
        bad = [list(train), list(early)]
        bad[which][0] = bad[which][0].copy()
        bad[which][0][:, -1, 3] = .5
        with pytest.raises(ValueError, match='Current-pitch physics'):
            MODELS[embedding](0, SMALL_CONTINUOUS, device='cpu').fit(bad[0], y, bad[1], ey, epochs=1, patience=1)
    MODELS[embedding](0, SMALL_CONTINUOUS, device='cpu').fit(train, y, early, ey, epochs=1, patience=1).save(tmp_path / 'm.pt')
    keys = {'d_embedding': 4, 'width': 64, 'dropout': 0.}
    if embedding == 'periodic':
        keys.update(n_frequencies=8, sigma=.1)
    for key, value in keys.items():
        payload = torch.load(tmp_path / 'm.pt', weights_only=True)
        payload['network_config'][key] = value
        torch.save(payload, tmp_path / 't.pt')
        with pytest.raises(ValueError, match='Archived network differs'):
            MODELS[embedding].load(tmp_path / 't.pt', device='cpu')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['architecture']['embedding']['d_embedding'] = 4
    torch.save(payload, tmp_path / 't.pt')
    with pytest.raises(ValueError, match='differs from the fixed'):
        MODELS[embedding].load(tmp_path / 't.pt', device='cpu')
