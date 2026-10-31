"""Optional F4 inference reuse: untrained CPU networks and synthetic inputs only."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import softmax

from pitchmdp.matrix_lazy_model import DualStreamNetwork, LazyJointDelivery, LazyMatrixModel
from pitchmdp.matrix_long_history import LazyPitchBatch, MatrixLongHistoryStore
from test_matrix_long_history import observations


class ActionContext:
    def transform(self, frame):
        return np.column_stack((frame.balls, frame.strikes, frame.pitch_type.eq('SL'),
                                frame.pitch_number)).astype(np.float32)


class FrozenSyntheticPool:
    draws = 400

    def sample(self, frame):
        pools = []
        for row in frame.itertuples():
            seed = int(row.at_bat_number)*7+int(row.pitch_number)*2+int(row.pitch_type == 'SL')
            pools.append(np.random.default_rng(seed).normal(size=(self.draws, 8)).astype(np.float32))
        return np.stack(pools), frame.at_bat_number.to_numpy() % 3


def batch(length):
    template = observations().iloc[0].to_dict()
    frame = pd.DataFrame([{**template, 'game_pk': 100, 'at_bat_number': i//2+1,
        'pitch_number': i % 2+1, 'pitch_type': 'FF' if i % 2 == 0 else 'SL',
        'balls': i % 2, 'effective_speed': 85.+i/20} for i in range(164)])
    store = MatrixLongHistoryStore.from_frame(frame, long_length=length)
    # Duplicate row with distinct candidate actions and nonmonotonic query order.
    return LazyPitchBatch(store, ActionContext(), [163, 0, 64, 130, 130],
                          candidate_pitch_types=['FF', 'SL', 'FF', 'FF', 'SL'])


def untrained(query, width=8):
    torch.manual_seed(317)
    torch.set_num_threads(1)
    model = LazyMatrixModel(width=width, device='cpu')
    arrays = query.gather()
    model.net = DualStreamNetwork(arrays[0].shape[-1], arrays[-1].shape[-1], width)
    model.feature_report = query.store.report()
    model.delivery_temperature = 1.37
    return model


def original_forward(net, h5, h5_valid, long, long_valid, context):
    """Literal original forward computation, independent of extracted methods."""
    h5 = h5.masked_fill(~h5_valid[..., None], 0.)
    long = long.masked_fill(~long_valid[..., None], 0.)
    short = net.h5_path(torch.cat((h5.flatten(1), h5_valid.float()), dim=1))
    codes = net.position_codes[None].expand(len(long), -1, -1)
    encoded = net.long_path(torch.cat((long, codes), dim=-1))
    encoded = encoded.masked_fill(~long_valid[..., None], 0.)
    pooled = encoded.sum(1)/long_valid.sum(1, keepdim=True).clamp(min=1)
    return net.head(torch.cat((short, pooled, context), dim=1))


def test_forward_and_gradients_preserve_original_computation_and_parameter_names():
    query = batch(32); model = untrained(query)
    values = [torch.as_tensor(a) for a in query.gather()]
    values[0][~values[1]] = 999.
    values[2][~values[3]] = 999.
    original = deepcopy(model.net)
    before = original_forward(original, *values)
    after = model.net(*values)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    before.square().sum().backward(); after.square().sum().backward()
    assert tuple(original.state_dict()) == tuple(model.net.state_dict())
    for p, q in zip(original.parameters(), model.net.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    with torch.no_grad():
        pooled = model.net.encode_long(values[2], values[3])
        actual = model.net.forward_with_pooled(values[0], values[1], pooled, values[4])
    torch.testing.assert_close(actual, before.detach(), rtol=0, atol=0)


@pytest.mark.parametrize('length', [0, 32, 128])
@pytest.mark.parametrize('pitch_chunk,model_batch_size', [(1, 37), (3, 256), (8, 513)])
def test_exact400_logits_probabilities_and_draw_owner_order(length, pitch_chunk, model_batch_size):
    query = batch(length); model = untrained(query)
    frozen = FrozenSyntheticPool()
    optimized = LazyJointDelivery(frozen, pitch_chunk, model_batch_size)
    reference = LazyJointDelivery(frozen, pitch_chunk, model_batch_size, reuse_long=False)
    plain_scores, plain_tiers = reference.logits(model, query)
    rows = []
    hook = model.net.long_path.register_forward_pre_hook(lambda module, args: rows.append(len(args[0])))
    scores, tiers = optimized.logits(model, query)
    hook.remove()
    assert scores.shape == (5, 400, 10)
    assert sum(rows) == len(query)  # Each query once, not 400 repeated long histories.
    assert max(rows) <= pitch_chunk
    np.testing.assert_allclose(scores, plain_scores, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(tiers, plain_tiers)
    for actual, expected in zip(optimized.predict(model, query), reference.predict(model, query)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    # The temperature objective must remain float32 and use all 400 logits.
    for temperature in (.5, 1., 2.5):
        p = softmax(scores/temperature, axis=-1).mean(1)
        original = softmax(plain_scores/temperature, axis=-1).mean(1)
        assert p.dtype == np.float32
        np.testing.assert_allclose(p, original, rtol=0, atol=1e-6)
    assert not np.array_equal(scores[3], scores[4])
    # Direct one-query comparison also catches chunk/query order corruption.
    single, _ = reference.logits(model, query.subset(np.array([4])))
    np.testing.assert_allclose(scores[4], single[0], rtol=0, atol=1e-6)


def test_full_width128_empty_and_feature_contract_checks():
    query = batch(128); model = untrained(query, width=128)
    optimized = LazyJointDelivery(FrozenSyntheticPool(), pitch_chunk=3, model_batch_size=257)
    reference = LazyJointDelivery(FrozenSyntheticPool(), pitch_chunk=3, model_batch_size=257, reuse_long=False)
    np.testing.assert_allclose(optimized.logits(model, query)[0], reference.logits(model, query)[0], rtol=0, atol=1e-6)
    empty = query.subset(np.array([], dtype=int))
    assert optimized.logits(model, empty)[0].shape == (0, 400, 10)
    assert optimized.predict(model, empty)[0].shape == (0, 10)
    model.feature_report = {**model.feature_report, 'long_length': 32}
    with pytest.raises(ValueError, match='feature contract'):
        optimized.logits(model, query)
