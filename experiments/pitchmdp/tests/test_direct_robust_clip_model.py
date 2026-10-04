"""Synthetic CPU checks for the robust-clip residual network (T100 preparation; no real data, no registered fit)."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import direct_robust_clip_model as rc  # noqa: E402
from pitchmdp.direct_residual_model import DirectResidualModel  # noqa: E402
from pitchmdp.direct_workload_dropout_registry import CONTINUOUS  # noqa: E402

CONT = CONTINUOUS['counts']  # the registered 41-column `counts` map (P32/P34/P35 arms)
SHAPE = (53, 38, 6)  # n_context, n_token, length of the counts export: 81 transformed scalars
N_SCALARS = 5 * 8 + len(CONT)
FROZEN = {  # sha256 at 387acd7; this module must not change them
    'direct_periodic_model.py': '18f0b96a9b8a0d9eb39af2de4b0cc1b7a10a0dd42ca92977f726042e1aec80d3',
    'direct_residual_model.py': '7feb9cfdcad85f4cb02c50a75e2ba2b528dd5546e600526b0334dcf0cf8e2d59',
    'direct_ple_model.py': 'd3575019f971b06ba26060a1fa31e3894a070fd3b4b88191393743cb16ea175e',
    'matrix_models.py': 'b6cc77fe3e337cf96f8c7cb31f02e8fd8a4bf71bb20713d01c462cdbd33d2fcd',
}


@pytest.fixture(autouse=True)
def one_thread(monkeypatch):
    """Mac is in protected light-work mode: one native thread for focused tests."""
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)


def arrays(n, seed, constant=None):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, SHAPE[2], SHAPE[1])).astype(np.float32)
    tokens[:, -1, :8] = 0.
    tokens[:, -1, -11:] = 0.
    valid = np.ones((n, SHAPE[2]), dtype=bool)
    valid[:, 0] = rng.random(n) > .5
    context = rng.normal(size=(n, SHAPE[0])).astype(np.float32)
    context[:, CONT[0]] = rng.standard_t(1, size=n)  # heavy tail: the clip must act
    if constant is not None:
        context[:, constant] = 1.25
    return (tokens, valid, context), rng.integers(0, 10, n).astype(np.int64)


def small_fit(model, train_seed=11, early_seed=12, n=300, epochs=2, checkpoint=None):
    train, y = arrays(n, train_seed)
    early, ey = arrays(90, early_seed)
    return model.fit(train, y, early, ey, epochs=epochs, patience=2, batch_size=64, checkpoint=checkpoint)


def stats(seed=0, n=400):
    return rc.compute_stats(arrays(n, seed)[0], CONT)


def test_frozen_modules_unchanged():
    for name, sha in FROZEN.items():
        assert hashlib.sha256((PROJECT / 'pitchmdp' / name).read_bytes()).hexdigest() == sha, name


# ------------------------------------------------------------------ statistics: TRAIN only, documented rule

def test_statistics_come_from_the_train_argument_only():
    a = small_fit(rc.DirectRobustClipResidualModel(0, CONT, device='cpu'), early_seed=12)
    b = small_fit(rc.DirectRobustClipResidualModel(0, CONT, device='cpu'), early_seed=99)
    c = small_fit(rc.DirectRobustClipResidualModel(0, CONT, device='cpu'), train_seed=21)
    sha = a.net.config['rc_stats_sha256']
    assert sha == b.net.config['rc_stats_sha256'] == rc.stats_sha256(*rc.compute_stats(arrays(300, 11)[0], CONT)[:2])
    assert c.net.config['rc_stats_sha256'] != sha
    pooled = [np.concatenate(p) for p in zip(arrays(300, 11)[0], arrays(90, 12)[0])]
    assert rc.stats_sha256(*rc.compute_stats(pooled, CONT)[:2]) != sha
    assert a.report['rc_stats_sha256'] == sha and sum(a.report['rc_rule_counts'].values()) == N_SCALARS


def test_rule_matches_the_paper_formula_per_column():
    (tokens, valid, context), _ = arrays(500, 4)
    center, scale, rules = rc.compute_stats((tokens, valid, context), CONT)
    assert center.shape == scale.shape == (N_SCALARS,) and center.dtype == scale.dtype == np.float32
    col = context[:, CONT[3]].astype(np.float64)
    q25, q50, q75 = np.quantile(col, [.25, .5, .75])
    assert center[40 + 3] == np.float32(q50) and scale[40 + 3] == np.float32(1. / (q75 - q25)) and rules[43] == 'iqr'
    phys = tokens[valid[:, 0], 0, 2].astype(np.float64)  # past position 0, channel 2: valid rows only
    assert center[2] == np.float32(np.median(phys))


def test_degenerate_columns_use_range_then_zero():
    (tokens, valid, context), _ = arrays(400, 5)
    context[:, CONT[1]] = 0.
    context[:300, CONT[1]] = 0.
    context[-1, CONT[1]] = 4.  # IQR 0, range 4 -> s = 2/4
    context[:, CONT[2]] = 1.25  # constant -> s = 0, transformed value 0
    center, scale, rules = rc.compute_stats((tokens, valid, context), CONT)
    assert rules[41] == 'range' and scale[41] == np.float32(.5) and center[41] == 0.
    assert rules[42] == 'constant' and scale[42] == 0.
    net = rc.RobustClipResidualNetwork(*SHAPE, CONT, center, scale)
    _, ctx = net.transform(torch.as_tensor(tokens), torch.as_tensor(context))
    assert torch.all(ctx[:, CONT[2]] == 0.)


def test_padded_past_positions_never_enter_statistics():
    (tokens, valid, context), _ = arrays(300, 3)
    assert (~valid[:, 0]).any()
    noisy = tokens.copy()
    noisy[~valid] = 1e3 * np.random.default_rng(0).normal(size=noisy[~valid].shape)
    assert rc.stats_sha256(*rc.compute_stats((noisy, valid, context), CONT)[:2]) == \
        rc.stats_sha256(*rc.compute_stats((tokens, valid, context), CONT)[:2])
    none = valid.copy()
    none[:, 0] = False
    with pytest.raises(ValueError, match='no rows'):
        rc.compute_stats((tokens, none, context), CONT)


def test_fail_closed_on_bad_inputs_and_statistics():
    (tokens, valid, context), _ = arrays(100, 6)
    bad = context.copy()
    bad[0, CONT[0]] = np.nan
    with pytest.raises(ValueError, match='finite'):
        rc.compute_stats((tokens, valid, bad), CONT)
    center, scale, _ = stats()
    with pytest.raises(ValueError):
        rc.check_stats(center.astype(np.float64), scale)
    with pytest.raises(ValueError):
        rc.check_stats(center, -scale - 1)
    with pytest.raises(ValueError, match='cover'):
        rc.RobustClipResidualNetwork(*SHAPE, CONT, center[:-1], scale[:-1])
    model = rc.DirectRobustClipResidualModel(0, CONT, device='cpu')
    with pytest.raises(RuntimeError, match='TRAIN only'):
        model.network(*SHAPE)
    with pytest.raises(ValueError, match='differs'):
        rc.DirectRobustClipResidualModel(0, CONT, architecture={**rc.ARCHITECTURE, 'preprocessing': {}})


# ------------------------------------------------------------------ transform: exact formula, untouched inputs

def test_transform_matches_float64_reference_and_leaves_other_inputs_unchanged():
    (tokens, valid, context), _ = arrays(200, 7)
    center, scale, _ = stats(1)
    net = rc.RobustClipResidualNetwork(*SHAPE, CONT, center, scale)
    t, c = net.transform(torch.as_tensor(tokens), torch.as_tensor(context))
    x = np.concatenate((tokens[:, :-1, :8].reshape(len(tokens), -1), context[:, CONT]), axis=1).astype(np.float64)
    z = scale.astype(np.float64) * (x - center.astype(np.float64))
    expect = z / np.sqrt(1. + (z / 3.) ** 2)
    got = np.concatenate((t[:, :-1, :8].reshape(len(tokens), -1).numpy(), c[:, CONT].numpy()), axis=1)
    assert np.allclose(got, expect, rtol=1e-5, atol=1e-6)
    assert np.abs(got).max() < 3.
    assert torch.equal(t[:, :, 8:], torch.as_tensor(tokens)[:, :, 8:])  # one-hots/outcome channels
    assert torch.equal(t[:, -1], torch.as_tensor(tokens)[:, -1])  # current token untouched
    raw = [i for i in range(SHAPE[0]) if i not in CONT]
    assert torch.equal(c[:, raw], torch.as_tensor(context)[:, raw])
    assert (np.abs(context[:, CONT[0]]) > 50).any()  # heavy tail present in the synthetic data ...
    assert np.abs(got[:, 40]).max() < 3.  # ... and bounded after the clip


def test_padded_positions_do_not_change_logits():
    (tokens, valid, context), _ = arrays(64, 8)
    net = rc.RobustClipResidualNetwork(*SHAPE, CONT, *stats(2)[:2]).eval()
    noisy = tokens.copy()
    noisy[~valid] = 77.
    with torch.no_grad():
        a = net(torch.as_tensor(tokens), torch.as_tensor(valid), torch.as_tensor(context))
        b = net(torch.as_tensor(noisy), torch.as_tensor(valid), torch.as_tensor(context))
    assert torch.equal(a, b)


def test_initial_parameters_equal_the_residual_counts_network():
    torch.manual_seed(5)
    ours = rc.RobustClipResidualNetwork(*SHAPE, CONT, *stats(3)[:2])
    torch.manual_seed(5)
    base = DirectResidualModel(5, device='cpu').network(*SHAPE)
    params = dict(ours.named_parameters())
    assert params.keys() == dict(base.named_parameters()).keys()
    assert all(torch.equal(params[k], v) for k, v in base.named_parameters())
    assert set(ours.state_dict()) - set(base.state_dict()) == {'rc_center', 'rc_scale'}


def test_identity_statistics_reduce_to_the_frozen_network_on_small_inputs():
    """With c = 0, s = 1 the clip is near identity for |x| << 3; logits approach the frozen forward."""
    (tokens, valid, context), _ = arrays(32, 9)
    tokens *= 1e-3
    context *= 1e-3
    torch.manual_seed(1)
    ours = rc.RobustClipResidualNetwork(*SHAPE, CONT, np.zeros(N_SCALARS, np.float32), np.ones(N_SCALARS, np.float32))
    torch.manual_seed(1)
    base = DirectResidualModel(1, device='cpu').network(*SHAPE)
    ours.eval(), base.eval()
    args = [torch.as_tensor(a) for a in (tokens, valid, context)]
    with torch.no_grad():
        assert torch.allclose(ours(*args), base(*args), atol=1e-6)


# ------------------------------------------------------------------ guard, archive

def test_current_physics_guard():
    (tokens, valid, context), y = arrays(50, 10)
    tokens[0, -1, 0] = 1.
    model = rc.DirectRobustClipResidualModel(0, CONT, device='cpu')
    with pytest.raises(ValueError, match='Current-pitch physics'):
        model.fit((tokens, valid, context), y, *arrays(20, 11))


def test_archive_round_trip_and_tamper_refusal(tmp_path):
    path = tmp_path / 'm.pt'
    model = small_fit(rc.DirectRobustClipResidualModel(3, CONT, device='cpu'), checkpoint=path)
    loaded = rc.DirectRobustClipResidualModel.load(path, device='cpu')
    x, _ = arrays(40, 30)
    assert np.array_equal(model.logits(x), loaded.logits(x))
    assert loaded.report['rc_stats_sha256'] == model.net.config['rc_stats_sha256']
    payload = torch.load(path, weights_only=True)
    payload['state_dict']['rc_scale'] = payload['state_dict']['rc_scale'] * 2
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='recorded hash'):
        rc.DirectRobustClipResidualModel.load(tmp_path / 'bad.pt', device='cpu')
    payload = torch.load(path, weights_only=True)
    payload['format'] = 'direct_residual_model_v1'
    torch.save(payload, tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='Unrecognized'):
        rc.DirectRobustClipResidualModel.load(tmp_path / 'other.pt', device='cpu')
    with pytest.raises(ValueError):
        DirectResidualModel.load(path, device='cpu')
