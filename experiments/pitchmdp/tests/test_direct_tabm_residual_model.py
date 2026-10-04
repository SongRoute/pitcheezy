"""Synthetic CPU tests: TabM-mini adapter on the frozen periodic residual backbone (DRAFT)."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic arrays
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp import direct_tabm_model as tm  # noqa: E402
from pitchmdp import direct_tabm_residual_model as trm  # noqa: E402
from pitchmdp.direct_dropout_registry import PERIODIC_CONTINUOUS as CONT  # noqa: E402

KW = pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic'])


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    """Mac shares the machine with live recording: keep focused tests on one native thread."""
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def arrays(n, seed):
    a = tf.split_arrays(n, seed, 0)
    return (a['tokens'], a['valid'], a['context']), a['y']


def tensors(n, seed):
    x, y = arrays(n, seed)
    return [torch.as_tensor(a) for a in x], torch.as_tensor(y)


def small_fit(model):
    train, y = arrays(300, 11)
    early, ey = arrays(90, 12)
    return model.fit(train, y, early, ey, epochs=3, patience=2, batch_size=64)


def nets(k=8, seed=7):
    torch.manual_seed(seed)
    base = pr.PeriodicResidualNetwork(52, 38, 6, CONT, **KW)
    torch.manual_seed(seed)
    ours = trm.PeriodicResidualTabMNetwork(52, 38, 6, CONT, **KW, k=k)
    return base, ours


def test_shared_parameters_equal_the_frozen_residual_initialisation_and_only_member_parameters_are_added():
    base, net = nets()
    shared, ours = dict(base.named_parameters()), dict(net.named_parameters())
    removed = {'output.2.weight', 'output.2.bias'}
    for name in set(shared) - removed:
        assert torch.equal(shared[name], ours[name]), name  # same RNG order up to the discarded final Linear
    assert set(ours) - set(shared) == {'scaling', 'output_weight', 'output_bias'}
    assert set(shared) - set(ours) == removed
    d_in = net.input.in_features
    assert ours['scaling'].shape == (8, d_in) and ours['output_weight'].shape == (8, 128, 10)
    assert ours['output_bias'].shape == (8, 10)
    assert [type(m).__name__ for m in net.output] == ['LayerNorm', 'GELU']
    extra, dropped = 8 * d_in + 8 * 128 * 10 + 8 * 10, 128 * 10 + 10
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in base.parameters()) + extra - dropped
    assert trm.parameter_count(52, 38, 6, CONT) == sum(p.numel() for p in net.parameters())
    assert net.n_raw + net.n_embedded * 8 == d_in and net.n_embedded == 5 * 8 + len(CONT)


def test_scaling_is_chunked_like_tabm_and_output_init_bound_uses_width():
    _, net = nets(k=4, seed=0)
    r = net.scaling.detach()
    start = 0
    for size in tm.feature_chunks(net.n_raw, net.n_embedded, 8):
        block = r[:, start:start + size]
        assert torch.equal(block, block[:, :1].expand_as(block))
        start += size
    assert start == r.shape[1] and not torch.allclose(r[0], r[1])
    bound = 1 / np.sqrt(128)
    assert net.output_weight.abs().max() <= bound and net.output_bias.abs().max() <= bound


def test_k1_with_unit_scaling_and_copied_output_equals_the_frozen_residual_forward():
    base, net = nets(k=1)
    with torch.no_grad():
        net.scaling.fill_(1.)
        net.output_weight.copy_(base.output[2].weight.T[None])
        net.output_bias.copy_(base.output[2].bias[None])
    base.eval(), net.eval()
    x, _ = tensors(40, 3)
    member = net(*x)
    assert member.shape == (40, 1, 10)
    assert torch.allclose(member[:, 0], base(*x), atol=1e-5, rtol=0)


def test_each_member_equals_the_frozen_network_with_input_columns_scaled_by_its_r_and_its_own_output():
    base, net = nets(k=3, seed=5)
    net.eval()
    x, _ = tensors(35, 6)
    member = net(*x)
    for m in range(3):
        ref = copy.deepcopy(base)
        ref.load_state_dict({k: v for k, v in net.state_dict().items() if k in base.state_dict()}, strict=False)
        with torch.no_grad():
            ref.input.weight.mul_(net.scaling[m][None, :])  # Linear(x * r) == Linear with weight * r (columns)
            ref.output[2].weight.copy_(net.output_weight[m].T)
            ref.output[2].bias.copy_(net.output_bias[m])
        ref.eval()
        assert torch.allclose(member[:, m], ref(*x), atol=1e-5, rtol=0)


def test_members_differ_and_padded_past_physics_does_not_change_the_output():
    _, net = nets(k=3)
    net.eval()
    x, _ = tensors(30, 4)
    member = net(*x)
    assert member.shape == (30, 3, 10) and not torch.allclose(member[:, 0], member[:, 1])
    tokens, valid, context = x
    assert (~valid).any()
    noisy = tokens.clone()
    noisy[~valid] = torch.randn_like(noisy[~valid])
    assert torch.equal(net(noisy, valid, context), member)


class _K1Wrapper(nn.Module):
    """k=1 adapter around the frozen periodic residual network (no parameters added)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config

    def forward(self, *x):
        return self.base(*x)[:, None, :]


class _K1Model(trm.DirectTabMResidualModel):
    def network(self, n_context, n_token, length):
        return _K1Wrapper(pr.PeriodicResidualNetwork(n_context, n_token, length, self.context_continuous, **KW,
                                                     n_classes=self.n_classes))


def test_fit_loop_mirrors_frozen_periodic_residual_fit_with_k1_wrapper():
    base = small_fit(pr.DirectPeriodicResidualModel(5, CONT, device='cpu'))
    ours = small_fit(_K1Model(5, CONT, k=1, device='cpu'))
    assert [h['train_weighted_nll'] for h in base.report['history']] == \
        [h['train_weighted_nll'] for h in ours.report['history']]
    assert np.allclose([h['earlystop_conditional_nll'] for h in base.report['history']],
                       [h['earlystop_conditional_nll'] for h in ours.report['history']], rtol=0, atol=1e-6)
    assert base.report['best_epoch'] == ours.report['best_epoch']
    assert all(torch.equal(a, b) for a, b in zip(base.net.parameters(), ours.net.parameters()))
    x, _ = arrays(50, 13)
    assert np.allclose(torch.softmax(torch.as_tensor(base.logits(x)), -1),
                       torch.softmax(torch.as_tensor(ours.logits(x)), -1), atol=1e-6)


def test_fit_predict_report_architecture_and_zero_current_physics_guard():
    model = small_fit(trm.DirectTabMResidualModel(0, CONT, k=4, device='cpu'))
    x, _ = arrays(30, 14)
    member = model.member_logits(x)
    assert member.shape == (30, 4, 10)
    assert np.allclose(model.predict(x), torch.softmax(torch.as_tensor(member), -1).mean(1).numpy(), atol=1e-6)
    a = model.report['architecture']
    assert a['backbone'] == pr.ARCHITECTURES['periodic']['backbone']
    assert a['embedding'] == pr.ARCHITECTURES['periodic']['embedding']
    assert a['ensemble'] == 'tabm-mini' and a['k'] == 4 and a['train_loss'] == 'mean_member_ce'
    assert 'head_dropout' not in a and model.report['network']['kind'] == trm.KIND == model.kind
    assert model.report['network']['dropout'] == .1 and model.report['network']['blocks'] == 2
    bad = (x[0].copy(), x[1], x[2])
    bad[0][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='physics'):
        model.logits(bad)


def test_architecture_constant_is_not_aliased_by_instances():
    model = trm.DirectTabMResidualModel(0, CONT, device='cpu')
    model.architecture['backbone']['width'] = 999
    assert trm.ARCHITECTURE['backbone']['width'] == 128 and pr.ARCHITECTURES['periodic']['backbone']['width'] == 128
    assert trm.DirectTabMResidualModel(0, CONT, device='cpu').architecture == trm.ARCHITECTURE


def test_same_seed_is_deterministic_and_seeds_differ():
    a = small_fit(trm.DirectTabMResidualModel(3, CONT, k=2, device='cpu'))
    b = small_fit(trm.DirectTabMResidualModel(3, CONT, k=2, device='cpu'))
    c = small_fit(trm.DirectTabMResidualModel(4, CONT, k=2, device='cpu'))
    x, _ = arrays(20, 15)
    assert np.array_equal(a.logits(x), b.logits(x))
    assert not np.allclose(a.logits(x), c.logits(x))


def test_checkpoint_roundtrip_and_cross_format_refusals(tmp_path):
    model = small_fit(trm.DirectTabMResidualModel(1, CONT, k=3, device='cpu'))
    model.temperature = 1.3
    model.save(tmp_path / 'm.pt')
    again = trm.DirectTabMResidualModel.load(tmp_path / 'm.pt', 'cpu')
    x, _ = arrays(25, 16)
    assert np.array_equal(model.logits(x), again.logits(x)) and again.temperature == 1.3
    assert again.architecture == model.architecture and again.context_continuous == list(CONT)
    for key, value in (('k', 4), ('dropout', .2), ('scaling_init', 'random-signs')):
        payload = torch.load(tmp_path / 'm.pt', weights_only=True)
        payload['network_config'][key] = value
        torch.save(payload, tmp_path / 'bad.pt')
        with pytest.raises(ValueError, match='differs'):
            trm.DirectTabMResidualModel.load(tmp_path / 'bad.pt', 'cpu')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['architecture']['backbone']['hidden'] = 512
    torch.save(payload, tmp_path / 'bad_arch.pt')
    with pytest.raises(ValueError, match='differs'):
        trm.DirectTabMResidualModel.load(tmp_path / 'bad_arch.pt', 'cpu')
    small_fit(tm.DirectTabMModel(1, CONT, 0., k=3, device='cpu')).save(tmp_path / 'tabm.pt')
    small_fit(pr.DirectPeriodicResidualModel(1, CONT, device='cpu')).save(tmp_path / 'resid.pt')
    for other in ('tabm.pt', 'resid.pt'):
        with pytest.raises(ValueError, match='Unrecognized'):
            trm.DirectTabMResidualModel.load(tmp_path / other, 'cpu')
    with pytest.raises(ValueError, match='Unrecognized'):
        tm.DirectTabMModel.load(tmp_path / 'm.pt', 'cpu')
    with pytest.raises(ValueError, match='Unrecognized'):
        pr.DirectPeriodicResidualModel.load(tmp_path / 'm.pt', 'cpu')


@pytest.mark.parametrize('kwargs', [{'k': 0}, {'k': 2.5}, {'k': True}, {'scaling_init': 'ones'}])
def test_invalid_settings_refused(kwargs):
    with pytest.raises(ValueError):
        trm.DirectTabMResidualModel(0, CONT, device='cpu', **kwargs)
    with pytest.raises(ValueError):
        trm.PeriodicResidualTabMNetwork(52, 38, 6, CONT, **KW, **kwargs)


# --- added by same-model review (OPUS-review-tabm-residual-45cb770-2026-10-04) ---

def test_shared_dropout_draws_independent_masks_per_member_in_training_only():
    """[N, k, width] Dropout is elementwise (official TabM semantics); eval mode is deterministic."""
    _, net = nets(k=4, seed=2)
    with torch.no_grad():  # identical members: any train-mode difference comes from dropout alone
        net.scaling.fill_(1.)
        net.output_weight.copy_(net.output_weight[:1].expand_as(net.output_weight))
        net.output_bias.copy_(net.output_bias[:1].expand_as(net.output_bias))
    x, _ = tensors(16, 8)
    net.eval()
    member = net(*x)
    assert all(torch.equal(member[:, 0], member[:, m]) for m in range(1, 4))
    net.train()
    torch.manual_seed(0)
    member = net(*x)
    assert not torch.allclose(member[:, 0], member[:, 1])


def test_empty_batch_shapes_and_report_width():
    model = small_fit(trm.DirectTabMResidualModel(2, CONT, k=3, device='cpu'))
    x, _ = arrays(5, 17)
    empty = tuple(a[:0] for a in x)
    assert model.logits(empty).shape == (0, 10) and model.member_logits(empty).shape == (0, 3, 10)
    assert model.predict(empty).shape == (0, 10)
    assert model.width == 128 == model.report['network']['width']


@pytest.mark.parametrize('part, key', [('architecture', 'k'), ('architecture', 'scaling_init'),
                                       ('architecture', 'backbone'), ('network_config', 'kind'),
                                       ('network_config', 'context_continuous')])
def test_load_refuses_archives_with_missing_keys_with_value_error(tmp_path, part, key):
    model = small_fit(trm.DirectTabMResidualModel(1, CONT, k=2, device='cpu'))
    model.save(tmp_path / 'm.pt')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload[part].pop(key)
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='differs'):
        trm.DirectTabMResidualModel.load(tmp_path / 'bad.pt', 'cpu')
