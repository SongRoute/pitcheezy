"""Reviewer-added CPU tests for the DRAFT TabM-mini head (OPUS review of f03bd04): gaps the k=1 wrapper cannot see."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_direct_tabm_model as base_tests  # noqa: E402  shared fixtures / synthetic arrays
from pitchmdp import direct_tabm_model as tm  # noqa: E402
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutNetwork  # noqa: E402
from pitchmdp.direct_dropout_registry import PERIODIC_CONTINUOUS as CONT  # noqa: E402

one_thread = base_tests.one_thread


def test_unit_scaling_k1_forward_equals_frozen_network():
    """The TabM forward is a copy of the frozen forward; with r=1 and the frozen output layer it must match it."""
    torch.manual_seed(21)
    frozen = PeriodicHeadDropoutNetwork(52, 38, 6, CONT, head_dropout=.1).eval()
    torch.manual_seed(21)
    net = tm.PeriodicTabMNetwork(52, 38, 6, CONT, head_dropout=.1, k=1).eval()
    with torch.no_grad():
        net.scaling.fill_(1.)
        net.output_weight.copy_(frozen.head[-1].weight.T[None])
        net.output_bias.copy_(frozen.head[-1].bias[None])
    t, _ = base_tests.tensors(40, 22)
    assert torch.allclose(net(*t)[:, 0], frozen(*t), atol=1e-6)


def test_scaling_chunks_align_with_the_features_actually_concatenated():
    """Perturbing one embedded scalar moves exactly one d_embedding chunk; one raw scalar moves one raw column."""
    torch.manual_seed(23)
    net = tm.PeriodicTabMNetwork(52, 38, 6, CONT, k=2).eval()
    seen = []
    net.head[0].register_forward_hook(lambda m, inputs, out: seen.append(inputs[0][:, 0]))
    tokens, valid, context = base_tests.tensors(8, 24)[0]
    chunks = np.cumsum([0, *tm.feature_chunks(net.n_raw, net.n_embedded, 8)])
    with torch.no_grad():
        net.scaling.fill_(1.)  # backbone input seen by the hook is then x itself
        changed = context.clone()
        changed[:, CONT[3]] += 5.
        seen.clear()
        net(tokens, valid, context)
        net(tokens, valid, changed)
        idx = (seen[0] - seen[1]).abs().amax(0).nonzero().flatten()
        feature = 5 * 8 + 3  # embedded order: past physics (5 tokens x 8) then continuous context
        lo, hi = chunks[net.n_raw + feature], chunks[net.n_raw + feature + 1]
        assert len(idx) and idx.min() >= lo and idx.max() < hi and hi - lo == 8
        raw_col = next(i for i in range(52) if i not in CONT)
        changed = context.clone()
        changed[:, raw_col] += 5.
        seen.clear()
        net(tokens, valid, context)
        net(tokens, valid, changed)
        idx = (seen[0] - seen[1]).abs().amax(0).nonzero().flatten()
        assert len(idx) == 1 and idx.item() < net.n_raw  # one raw scalar = one size-1 chunk


def test_early_stop_monitors_ensemble_nll_not_mean_member_ce():
    model = base_tests.small_fit(tm.DirectTabMModel(6, CONT, 0., k=4, device='cpu'))
    early, ey = base_tests.arrays(90, 12)
    logits = model.logits(early)
    ensemble = float((logsumexp(logits, axis=1) - logits[np.arange(len(ey)), ey]).mean())
    best = model.report['history'][model.report['best_epoch'] - 1]['earlystop_conditional_nll']
    assert abs(ensemble - best) < 1e-5
    member = float(tm.member_loss(torch.as_tensor(model.member_logits(early)), torch.as_tensor(ey)).mean())
    assert member > ensemble + 1e-4  # k>1: the monitored score is the ensemble NLL (Jensen gap is visible)


def test_temperature_on_log_mean_probability_is_a_valid_power_calibration():
    model = base_tests.small_fit(tm.DirectTabMModel(7, CONT, 0., k=3, device='cpu'))
    x, _ = base_tests.arrays(20, 25)
    model.temperature = 1.7
    p_bar = torch.softmax(torch.as_tensor(model.member_logits(x)), -1).mean(1).double().numpy()
    expected = p_bar ** (1 / 1.7)
    expected /= expected.sum(1, keepdims=True)
    assert np.allclose(model.predict(x), expected, atol=1e-5)


@pytest.mark.parametrize('where,key,value', [('network_config', 'head_dropout', .2),
                                             ('network_config', 'scaling_init', 'random-signs'),
                                             ('architecture', 'k', 4), ('architecture', 'ensemble', 'tabm'),
                                             ('kind', None, 'periodic_head_dropout_mlp')])
def test_load_refuses_each_tampered_field(tmp_path, where, key, value):
    model = base_tests.small_fit(tm.DirectTabMModel(1, CONT, .1, k=3, device='cpu'))
    model.save(tmp_path / 'm.pt')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    if key is None:
        payload[where] = value
    else:
        payload[where][key] = value
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError):
        tm.DirectTabMModel.load(tmp_path / 'bad.pt', 'cpu')
