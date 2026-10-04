"""Synthetic CPU tests for the DRAFT weight-averaging (EMA) arms (no real data)."""
import copy

import numpy as np
import pytest
import torch

from pitchmdp import direct_ema_model as em
from pitchmdp import direct_frontier_model as fm
from pitchmdp.matrix_models import MatrixModel

FIT = dict(epochs=3, patience=5, batch_size=64, learning_rate=.0005)


def arrays(n, seed):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 38)).astype(np.float32)
    tokens[:, -1, :8] = 0.
    tokens[:, -1, -11:] = 0.
    valid = np.ones((n, 6), dtype=bool)
    valid[:, 0] = rng.random(n) > .5
    return (tokens, valid, rng.normal(size=(n, 52)).astype(np.float32)), rng.integers(0, 10, n)


def fitted(model, seed=0, **kwargs):
    (x, y), (e, ey) = arrays(256, seed), arrays(96, seed + 1)
    return model.fit(x, y, e, ey, **{**FIT, **kwargs})


def test_decay_warmup_and_update_math():
    assert em.ema_decay_at(.999, 0) == pytest.approx(.1)
    assert em.ema_decay_at(.999, 90) == pytest.approx(91 / 100)
    assert em.ema_decay_at(.999, 10 ** 6) == .999
    for bad in ((0., 1), (1., 1), (.5, -1)):
        with pytest.raises(ValueError):
            em.ema_decay_at(*bad)
    ema, w = [torch.zeros(3)], [torch.ones(3)]
    expected = 0.
    for step in range(5):
        d = em.update_ema(ema, w, step, .999)
        expected = d * expected + (1 - d) * 1.
    np.testing.assert_allclose(ema[0].numpy(), expected, rtol=1e-6)
    with pytest.raises(ValueError):
        em.update_ema([torch.zeros(1)], [torch.ones(1), torch.ones(1)], 0, .9)


@pytest.mark.parametrize('dropout', [0., .1])
def test_raw_trajectory_equals_base_arm(dropout):
    """EMA consumes no randomness: same seed -> same training losses and raw early-stop NLL as the base arm."""
    base = fitted(fm.FlattenRegularizedModel(2, dropout, .01, 'cpu'), 3)
    ema = fitted(em.FlattenEMAModel(2, dropout, .999, 'cpu'), 3)
    assert len(base.report['history']) == len(ema.report['history']) == 3
    for b, e in zip(base.report['history'], ema.report['history']):
        assert b['train_weighted_nll'] == e['train_weighted_nll']
        assert b['earlystop_conditional_nll'] == e['raw_earlystop_conditional_nll']
    assert ema.report['optimizer_updates'] == base.report['optimizer_updates'] == ema.report['ema_updates']


def test_neutral_base_still_reproduces_control():
    probe, _ = arrays(40, 9)
    control = fitted(MatrixModel('flatten_mlp', seed=2, width=128, device='cpu'), 3).logits(probe)
    neutral = fitted(fm.FlattenRegularizedModel(2, 0., .01, 'cpu'), 3).logits(probe)
    np.testing.assert_array_equal(control, neutral)
    assert not np.array_equal(control, fitted(em.FlattenEMAModel(2, 0., .999, 'cpu'), 3).logits(probe))


def test_saved_weights_are_the_shadow_average():
    """Replay the same fit by hand: final weights equal the EMA of the raw steps, not the raw weights."""
    (x, y), (e, ey) = arrays(128, 5), arrays(48, 6)
    model = em.FlattenEMAModel(4, 0., .99, 'cpu').fit(x, y, e, ey, epochs=1, patience=5, batch_size=32)
    base = fm.FlattenRegularizedModel(4, 0., .01, 'cpu')
    torch.manual_seed(4)
    net = base.network(52, 38, 6)
    shadow = copy.deepcopy(net)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.01)
    tensors = [torch.as_tensor(a) for a in (x[0], x[1], x[2], y.astype(np.int64))]
    rng = np.random.default_rng(4)
    net.train()
    for step, ids in enumerate(np.array_split(rng.permutation(len(y)), np.arange(32, len(y), 32))):
        ix = torch.as_tensor(ids)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(*(a[ix] for a in tensors[:3])), tensors[3][ix])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()
        em.update_ema(list(shadow.parameters()), list(net.parameters()), step, .99)
    for (k, a), b in zip(model.net.state_dict().items(), shadow.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=k)
    assert any(not torch.equal(a, b) for a, b in zip(model.net.state_dict().values(), net.state_dict().values()))
    assert model.report['weights'] == 'ema' and model.report['best_epoch'] == 1


@pytest.mark.parametrize('arm', list(em.ARMS))
def test_every_arm_trains_saves_loads(arm, tmp_path):
    model = fitted(em.build_ema(arm, 1, 'cpu'))
    probe, _ = arrays(40, 9)
    model.save(tmp_path / 'm.pt')
    loaded = em.load_ema(arm, tmp_path / 'm.pt', 'cpu')
    np.testing.assert_array_equal(model.logits(probe), loaded.logits(probe))
    assert loaded.report == model.report and loaded.kind == 'ema_flatten_mlp'
    assert model.report['parameter_count'] == fm.flatten_parameter_count(128)  # EMA adds no parameters
    other = next(a for a in em.ARMS if a != arm)
    with pytest.raises(ValueError):
        em.load_ema(other, tmp_path / 'm.pt', 'cpu')
    with pytest.raises(ValueError):
        fm.FlattenRegularizedModel.load(tmp_path / 'm.pt', 'cpu')
    history = model.report['history']
    assert all({'raw_earlystop_conditional_nll', 'ema_decay_last'} <= set(h) for h in history)


def test_registry_guards():
    arms = [dict(em.ARMS[n] if n in em.ARMS else {}, name=n, priority=1, workload=[]) for n in em.ARM_ORDER]
    arms[0] = dict(em.check_family.__globals__['CONTROL'])
    from pitchmdp.direct_frontier_registry import ARMS as FRONTIER
    arms[2] = dict(FRONTIER['dropout'], name='dropout', priority=1, frontier=True, workload=[])
    for a in (arms[1], arms[3]):
        a['ema'] = True
    config = {'arms': arms, 'seeds': [19, 20, 21, 22, 23], 'fits': 20,
              'comparison': {'ema_contrasts': copy.deepcopy(em.CONTRASTS)}}
    assert em.check_contrasts(config) == em.CONTRASTS
    assert em.check_contrasts({'arms': arms}) is None
    for mutate in (lambda c: c['arms'][1].update(ema_decay=.99),
                   lambda c: c['arms'][3].update(dropout=.2),
                   lambda c: c['arms'][1].update(workload=['counts']),
                   lambda c: c['arms'][2].update(weight_decay=1e-4),
                   lambda c: c['arms'].reverse(),
                   lambda c: c['comparison']['ema_contrasts'].pop(),
                   lambda c: c.update(seeds=[16, 17, 18]),
                   lambda c: c.update(fits=12)):
        bad = copy.deepcopy(config)
        mutate(bad)
        with pytest.raises(ValueError):
            em.check_contrasts(bad)
    with pytest.raises(ValueError):
        em.build_ema('control', 0, 'cpu')
    with pytest.raises(ValueError):
        em.FlattenEMAModel(0, 0., 1.)


# ------------------------------------------------------------------ reviewer additions (same_model_review 2026-10-04)

def stop_rule(scores, patience):
    """(best_epoch, epochs_run) of the shared early-stop rule (min delta 1e-5, patience) over a score list."""
    best, best_epoch, stale = np.inf, 0, 0
    for i, s in enumerate(scores):
        if s < best - 1e-5:
            best, best_epoch, stale = s, i + 1, 0
        else:
            stale += 1
        if stale >= patience:
            return best_epoch, i + 1
    return best_epoch, len(scores)


def signal(n, seed):
    """Synthetic arrays whose labels depend on the context, so early stopping fires after a few epochs."""
    (tokens, valid, context), _ = arrays(n, seed)
    noise = np.random.default_rng(seed + 100).normal(size=(n, 10))
    return (tokens, valid, context), np.argmax(context[:, :10] + 1.5 * noise, axis=1)


def test_vanishing_decay_is_the_base_arm_including_stopping():
    """decay -> 0 makes the shadow equal the raw weights: stop, best state and logits must equal the base arm."""
    kw = dict(epochs=12, patience=2, batch_size=32, learning_rate=.003)
    (x, y), (e, ey) = arrays(256, 0), arrays(96, 1)
    base = fm.FlattenRegularizedModel(2, 0., .01, 'cpu').fit(x, y, e, ey, **kw)
    ema = em.FlattenEMAModel(2, 0., 1e-9, 'cpu').fit(x, y, e, ey, **kw)
    assert base.report['epochs_run'] < 12  # the patience branch was exercised
    assert [h['earlystop_conditional_nll'] for h in base.report['history']] == \
        [h['earlystop_conditional_nll'] for h in ema.report['history']]
    assert (base.report['best_epoch'], base.report['epochs_run']) == (ema.report['best_epoch'], ema.report['epochs_run'])
    np.testing.assert_array_equal(base.logits(e), ema.logits(e))
    # review F2 disposition: replaying the shared rule on the raw score gives the base arm's epochs
    assert ema.report['raw_rule_best_epoch'] == base.report['best_epoch']
    assert ema.report['raw_rule_stop_epoch'] == base.report['epochs_run']


def test_stopping_and_best_epoch_follow_the_ema_score():
    """On data where the raw and EMA rules stop at different epochs, the fit must follow the EMA score."""
    (x, y), (e, ey) = signal(1024, 0), signal(256, 1)
    model = em.FlattenEMAModel(0, 0., .99, 'cpu').fit(x, y, e, ey, epochs=6, patience=2, batch_size=64,
                                                      learning_rate=.001)
    history = model.report['history']
    ema_rule = stop_rule([h['earlystop_conditional_nll'] for h in history], 2)
    raw_rule = stop_rule([h['raw_earlystop_conditional_nll'] for h in history], 2)
    assert ema_rule != raw_rule  # precondition: the two rules disagree on this data
    assert (model.report['best_epoch'], model.report['epochs_run']) == ema_rule
    rbest, rstop = model.report['raw_rule_best_epoch'], model.report['raw_rule_stop_epoch']
    assert rbest == raw_rule[0] and rstop in (None, raw_rule[1])  # None when the EMA run stopped first


def test_archive_negatives(tmp_path):
    (x, y), (e, ey) = arrays(64, 0), arrays(32, 1)
    small = dict(epochs=1, patience=1, batch_size=32)
    fm.FlattenRegularizedModel(0, .1, .01, 'cpu').fit(x, y, e, ey, **small).save(tmp_path / 'reg.pt')
    MatrixModel('flatten_mlp', seed=0, width=128, device='cpu').fit(x, y, e, ey, **small).save(tmp_path / 'mat.pt')
    for path in ('reg.pt', 'mat.pt'):
        with pytest.raises(ValueError):
            em.load_ema('dropout_ema', tmp_path / path, 'cpu')
    em.build_ema('ema', 0, 'cpu').fit(x, y, e, ey, **small).save(tmp_path / 'ema.pt')
    with pytest.raises(ValueError):
        MatrixModel.load(tmp_path / 'ema.pt', 'cpu')
    for tamper in (lambda p: p['architecture'].update(ema_decay=.99),
                   lambda p: p['network_config'].update(dropout=.1),
                   lambda p: p.update(kind='regularized_flatten_mlp')):
        payload = torch.load(tmp_path / 'ema.pt', map_location='cpu', weights_only=True)
        tamper(payload)
        torch.save(payload, tmp_path / 'bad.pt')
        with pytest.raises(ValueError):
            em.load_ema('ema', tmp_path / 'bad.pt', 'cpu')


def test_registry_rejects_extra_keys_and_loose_flags():
    from pitchmdp.direct_frontier_registry import ARMS as FRONTIER
    arms = [dict(em.ARMS[n] if n in em.ARMS else {}, name=n, priority=1, workload=[]) for n in em.ARM_ORDER]
    arms[0] = dict(em.check_family.__globals__['CONTROL'])
    arms[2] = dict(FRONTIER['dropout'], name='dropout', priority=1, frontier=True, workload=[])
    for a in (arms[1], arms[3]):
        a['ema'] = True
    config = {'arms': arms, 'comparison': {'ema_contrasts': copy.deepcopy(em.CONTRASTS)},
              'seeds': [19, 20, 21, 22, 23], 'fits': 20}
    assert em.check_family(copy.deepcopy(config))
    for mutate in (lambda c: c['arms'][1].update(ema_start=0),
                   lambda c: c['arms'][1].update(ema=1),
                   lambda c: c['arms'][3].update(priority=2),
                   lambda c: c['arms'][2].pop('frontier'),
                   lambda c: c['arms'][0].update(ema=True),
                   lambda c: c['arms'].append(dict(c['arms'][1], name='ema2')),
                   lambda c: c['comparison']['ema_contrasts'].reverse(),
                   lambda c: c['arms'][1].update(priority=True),
                   lambda c: c.update(seeds=[16, 17, 18, 19, 20]),
                   lambda c: c.update(seeds=[19., 20., 21., 22., 23.]),
                   lambda c: c.update(fits=12),
                   lambda c: c.update(repeat_tag='r1')):
        bad = copy.deepcopy(config)
        mutate(bad)
        with pytest.raises(ValueError):
            em.check_contrasts(bad)


def test_shadow_swap_is_restored_when_scoring_fails(monkeypatch):
    model = em.FlattenEMAModel(0, 0., .999, 'cpu')
    calls, logits = [], em.FlattenEMAModel.logits

    def failing(self, arrays, batch_size=4096):
        calls.append(1)
        if len(calls) == 2:  # the shadow-weight score of epoch 1
            raise RuntimeError('scoring failed')
        return logits(self, arrays, batch_size)
    monkeypatch.setattr(em.FlattenEMAModel, 'logits', failing)
    with pytest.raises(RuntimeError):
        fitted(model)
    assert all(p.requires_grad for p in model.net.parameters())  # raw net back in place, not the shadow


def test_raw_rule_replay_stops_where_the_raw_rule_stops():
    """Review F2 disposition: raw_rule_* equal the shared rule on the raw score, truncated at the raw stop."""
    hit = 0
    for seed in range(3):
        (x, y), (e, ey) = signal(1024, seed), signal(256, seed + 10)
        model = em.FlattenEMAModel(0, 0., .99, 'cpu').fit(x, y, e, ey, epochs=10, patience=2, batch_size=64,
                                                          learning_rate=.001)
        history = model.report['history']
        raw_rule = stop_rule([h['raw_earlystop_conditional_nll'] for h in history], 2)
        if raw_rule[1] < len(history):  # the raw rule stopped before the EMA run ended
            hit += 1
            assert (model.report['raw_rule_best_epoch'], model.report['raw_rule_stop_epoch']) == raw_rule
    assert hit  # precondition: exercised at least once


def test_replay_stop_rule_is_truncated_at_the_stop():
    assert em.replay_stop_rule([3., 2., 2.5, 2.6, 1.], 2) == (2, 4)  # the later 1.0 must not count
    assert em.replay_stop_rule([3., 2., 1.], 2) == (3, None)
    assert em.replay_stop_rule([3., 3. - 5e-6, 3.], 2) == (1, 3)  # min delta 1e-5
