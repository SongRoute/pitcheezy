"""Synthetic CPU tests: TabM-mini shared-weight ensemble head on the frozen periodic direct network (DRAFT)."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic arrays
from pitchmdp import direct_tabm_model as tm  # noqa: E402
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutModel, PeriodicHeadDropoutNetwork  # noqa: E402
from pitchmdp.direct_dropout_registry import PERIODIC_CONTINUOUS as CONT  # noqa: E402


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


def test_network_reuses_frozen_modules_and_adds_only_member_parameters():
    torch.manual_seed(7)
    base = PeriodicHeadDropoutNetwork(52, 38, 6, CONT, head_dropout=.1)
    torch.manual_seed(7)
    net = tm.PeriodicTabMNetwork(52, 38, 6, CONT, head_dropout=.1, k=8)
    shared = dict(base.named_parameters())
    ours = dict(net.named_parameters())
    for name in ('frequencies', 'embedding.weight', 'embedding.bias', 'head.0.weight', 'head.0.bias',
                 'head.3.weight', 'head.3.bias'):
        assert torch.equal(shared[name], ours[name])  # frozen modules built in the original order
    assert [type(m).__name__ for m in net.head] == ['Linear', 'GELU', 'Dropout', 'Linear', 'GELU', 'Dropout']
    d_in = net.head[0].in_features
    assert set(ours) - set(shared) == {'scaling', 'output_weight', 'output_bias'}
    assert ours['scaling'].shape == (8, d_in) and ours['output_weight'].shape == (8, 256, 10)
    assert ours['output_bias'].shape == (8, 10)
    extra = 8 * d_in + 8 * 256 * 10 + 8 * 10
    removed = 256 * 10 + 10
    assert sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in base.parameters()) + extra - removed
    assert tm.parameter_count(52, 38, 6, CONT) == sum(p.numel() for p in net.parameters())


def test_scaling_init_is_one_normal_value_per_member_and_feature_chunk():
    torch.manual_seed(0)
    net = tm.PeriodicTabMNetwork(52, 38, 6, CONT, k=4)
    r = net.scaling.detach()
    chunks = tm.feature_chunks(net.n_raw, net.n_embedded, 8)
    assert sum(chunks) == r.shape[1] and net.n_embedded == 5 * 8 + len(CONT)
    start = 0
    firsts = []
    for size in chunks:
        block = r[:, start:start + size]
        assert torch.equal(block, block[:, :1].expand_as(block))  # constant within a chunk
        firsts.append(block[:, 0])
        start += size
    firsts = torch.stack(firsts, 1)
    assert not torch.allclose(firsts[0], firsts[1])  # members differ
    assert (firsts.abs() != 1).any()  # 'normal', not random signs
    torch.manual_seed(0)
    signs = tm.PeriodicTabMNetwork(52, 38, 6, CONT, k=4, scaling_init='random-signs').scaling.detach()
    assert set(signs.unique().tolist()) == {-1., 1.}
    with pytest.raises(ValueError):
        tm.PeriodicTabMNetwork(52, 38, 6, CONT, k=4, scaling_init='ones')
    with pytest.raises(ValueError):
        tm.init_scaling(torch.empty(2, 5), [2, 2], 'normal')


def test_forward_member_shape_shared_head_and_padding_invariance():
    torch.manual_seed(1)
    net = tm.PeriodicTabMNetwork(52, 38, 6, CONT, k=3).eval()
    t, _ = tensors(40, 3)
    out = net(*t)
    assert out.shape == (40, 3, 10)
    assert not torch.allclose(out[:, 0], out[:, 1])  # member scaling makes predictions differ
    with torch.no_grad():  # identical scalings and output layers -> identical members (head is shared)
        net.scaling.copy_(net.scaling[:1].expand_as(net.scaling))
        net.output_weight.copy_(net.output_weight[:1].expand_as(net.output_weight))
        net.output_bias.copy_(net.output_bias[:1].expand_as(net.output_bias))
    out = net(*t)
    assert torch.allclose(out[:, 0], out[:, 2])
    tokens = t[0].clone()
    pad = ~t[1]
    tokens[pad] = 123.  # values at padded positions must not matter
    assert torch.equal(net(tokens, t[1], t[2]), out)


def test_member_loss_is_mean_of_member_losses_not_loss_of_mean():
    torch.manual_seed(2)
    member = torch.randn(5, 4, 10)
    y = torch.randint(0, 10, (5,))
    expected = torch.stack([nn.functional.cross_entropy(member[:, j], y, reduction='none') for j in range(4)], 1).mean(1)
    assert torch.allclose(tm.member_loss(member, y), expected)
    mean_prediction = nn.functional.nll_loss(tm.ensemble_logits(member), y, reduction='none')
    assert (mean_prediction <= tm.member_loss(member, y) + 1e-6).all()  # Jensen: they differ in general
    assert not torch.allclose(mean_prediction, tm.member_loss(member, y))


def test_ensemble_logits_average_member_probabilities():
    torch.manual_seed(3)
    member = torch.randn(6, 5, 10)
    probs = torch.softmax(tm.ensemble_logits(member), -1)
    assert torch.allclose(probs, torch.softmax(member, -1).mean(1), atol=1e-6)
    assert torch.allclose(torch.exp(tm.ensemble_logits(member)).sum(-1), torch.ones(6), atol=1e-6)


class _K1Wrapper(nn.Module):
    """k=1 adapter around the frozen periodic dropout network (no parameters added)."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.config = base.config

    def forward(self, *x):
        return self.base(*x)[:, None, :]


class _K1Model(tm.DirectTabMModel):
    def network(self, n_context, n_token, length):
        a = self.architecture
        return _K1Wrapper(PeriodicHeadDropoutNetwork(n_context, n_token, length, self.context_continuous, a['width'],
                                                     a['n_frequencies'], a['d_embedding'], a['sigma'],
                                                     a['head_dropout'], self.n_classes))


@pytest.mark.parametrize('p', [0., .1])
def test_fit_loop_mirrors_frozen_periodic_fit_with_k1_wrapper(p):
    base = small_fit(PeriodicHeadDropoutModel(5, CONT, p, 'cpu'))
    ours = small_fit(_K1Model(5, CONT, p, k=1, device='cpu'))
    assert [h['train_weighted_nll'] for h in base.report['history']] == \
        [h['train_weighted_nll'] for h in ours.report['history']]
    assert np.allclose([h['earlystop_conditional_nll'] for h in base.report['history']],
                       [h['earlystop_conditional_nll'] for h in ours.report['history']], rtol=0, atol=1e-6)
    assert base.report['best_epoch'] == ours.report['best_epoch']
    assert all(torch.equal(a, b) for a, b in zip(base.net.parameters(), ours.net.parameters()))
    x, _ = arrays(50, 13)
    assert np.allclose(torch.softmax(torch.as_tensor(base.logits(x)), -1),
                       torch.softmax(torch.as_tensor(ours.logits(x)), -1), atol=1e-6)


def test_fit_predict_report_and_zero_current_physics_guard():
    model = small_fit(tm.DirectTabMModel(0, CONT, 0., k=4, device='cpu'))
    x, _ = arrays(30, 14)
    member = model.member_logits(x)
    assert member.shape == (30, 4, 10)
    probs = model.predict(x)
    assert np.allclose(probs, torch.softmax(torch.as_tensor(member), -1).mean(1).numpy(), atol=1e-6)
    assert model.report['architecture']['ensemble'] == 'tabm-mini' and model.report['architecture']['k'] == 4
    assert model.report['network']['kind'] == tm.KIND and model.kind == tm.KIND
    bad = (x[0].copy(), x[1], x[2])
    bad[0][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='physics'):
        model.logits(bad)
    with pytest.raises(ValueError, match='physics'):
        small_fit_bad = tm.DirectTabMModel(0, CONT, 0., k=2, device='cpu')
        train, y = arrays(50, 1)
        train = (train[0].copy(), train[1], train[2])
        train[0][:, -1, 0] = 1.
        small_fit_bad.fit(train, y, *arrays(20, 2), epochs=1)


def test_same_seed_is_deterministic_and_seeds_differ():
    a = small_fit(tm.DirectTabMModel(3, CONT, .1, k=2, device='cpu'))
    b = small_fit(tm.DirectTabMModel(3, CONT, .1, k=2, device='cpu'))
    c = small_fit(tm.DirectTabMModel(4, CONT, .1, k=2, device='cpu'))
    x, _ = arrays(20, 15)
    assert np.array_equal(a.logits(x), b.logits(x))
    assert not np.allclose(a.logits(x), c.logits(x))


def test_checkpoint_roundtrip_and_guards(tmp_path):
    model = small_fit(tm.DirectTabMModel(1, CONT, .1, k=3, device='cpu'))
    model.temperature = 1.3
    model.save(tmp_path / 'm.pt')
    again = tm.DirectTabMModel.load(tmp_path / 'm.pt', 'cpu')
    x, _ = arrays(25, 16)
    assert np.array_equal(model.logits(x), again.logits(x)) and again.temperature == 1.3
    assert again.architecture == model.architecture and again.context_continuous == list(CONT)
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['network_config']['k'] = 4
    torch.save(payload, tmp_path / 'bad_k.pt')
    with pytest.raises(ValueError, match='differs'):
        tm.DirectTabMModel.load(tmp_path / 'bad_k.pt', 'cpu')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['format'] = 'direct_dropout_periodic_v1'
    torch.save(payload, tmp_path / 'bad_format.pt')
    with pytest.raises(ValueError, match='Unrecognized'):
        tm.DirectTabMModel.load(tmp_path / 'bad_format.pt', 'cpu')
    other = small_fit(PeriodicHeadDropoutModel(1, CONT, .1, 'cpu'))
    other.save(tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='Unrecognized'):
        tm.DirectTabMModel.load(tmp_path / 'other.pt', 'cpu')


@pytest.mark.parametrize('kwargs', [{'head_dropout': 1.}, {'k': 0}, {'k': 2.5}, {'scaling_init': 'ones'}])
def test_invalid_settings_refused(kwargs):
    with pytest.raises(ValueError):
        tm.DirectTabMModel(0, CONT, device='cpu', **kwargs)
