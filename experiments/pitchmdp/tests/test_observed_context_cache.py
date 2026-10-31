"""Optional observed-context memoization: synthetic CPU-only equivalence checks."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import torch

from pitchmdp.archetypes import STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.sequence_model import SequenceContext
from pitchmdp.matrix_sharing import ContinuousPitcherContext
from pitchmdp.matrix_observed_context_cache import FrozenObservedContext, GUARDED_COLUMNS
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_lazy_model import LazyMatrixModel, DualStreamNetwork, LazyJointDelivery
from test_matrix_long_history import observations
from test_lazy_delivery_reuse import FrozenSyntheticPool


def fixture():
    frame = observations()
    frame['split'] = 'train'
    frame['outs_when_up'] = np.arange(len(frame)) % 3
    frame['inning'] = 4; frame['home_score'] = 2; frame['away_score'] = 1
    frame['inning_topbot'] = 'Top'; frame['bases'] = np.arange(len(frame)) % 8
    for i, col in enumerate(STYLE_COLUMNS): frame[col] = .2+i*.05+np.arange(len(frame))*.003
    for i, col in enumerate(RELIABILITY_COLUMNS): frame[col] = .4+i*.02
    base = SequenceContext().fit(frame)
    # Twenty-three synthetic static pitcher profile coordinates + count = 24;
    # the real frozen context has this same dimension and transformation.
    clusters = {'columns': [f'profile{i}' for i in range(23)],
        'pitcher_cluster': {'11': 0}, 'pitcher_profiles': {'11': np.arange(23).tolist()},
        'pitcher_counts': {'11': len(frame)}}
    return frame, ContinuousPitcherContext(base, clusters)


def test_chunks_repeats_order_empty_lookup_and_no_rng_mutation():
    frame, base = fixture()
    numpy_state = np.random.get_state(); torch_state = torch.get_rng_state().clone()
    cache = FrozenObservedContext(base, frame, [12, 8, 2, 8, 5], chunk_size=2)
    for rows in ([12, 5, 8, 8, 2], [2], []):
        query = frame.iloc[rows]
        np.testing.assert_array_equal(cache.transform(query), base.transform(query))
        assert cache.transform(query).tobytes() == base.transform(query).tobytes()
    assert cache.report() == base.report()
    assert cache.cache_report()['value_bytes'] == 4*52*4
    assert cache.cache_report()['exact_input_snapshot_bytes'] > 0
    assert cache.cache_report()['adoption'] is None
    np.testing.assert_array_equal(torch.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert numpy_state[0] == after[0] and numpy_state[2:] == after[2:]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    actual = cache.transform(frame.iloc[[8]])
    actual[:] = 99
    np.testing.assert_array_equal(cache.transform(frame.iloc[[8]]), base.transform(frame.iloc[[8]]))
    with pytest.raises(ValueError, match='not registered'): cache.transform(frame.iloc[[3]])


@pytest.mark.parametrize('column', GUARDED_COLUMNS)
def test_changed_observed_inputs_are_rejected_including_source_mutation(column):
    frame, base = fixture(); cache = FrozenObservedContext(base, frame, np.arange(len(frame)))
    if column == 'game_date': frame.loc[8, column] += pd.Timedelta(days=1)
    elif isinstance(frame.loc[8, column], str): frame.loc[8, column] = 'changed'
    else: frame.loc[8, column] += 1
    with pytest.raises(ValueError, match='dependency changed'): cache.transform(frame.iloc[[8]])


def test_candidate_type_current_physics_outcome_support_do_not_enter_context():
    frame, base = fixture(); cache = FrozenObservedContext(base, frame, np.arange(len(frame)))
    query = frame.iloc[[12, 8, 8]].copy()
    query['pitch_type'] = ['SL', 'FF', 'UNSEEN_TYPE']
    for column in ['plate_x', 'plate_z', 'effective_speed', 'release_spin_rate']:
        query[column] = -999.
    query['description'] = 'hit_by_pitch'; query['events'] = 'home_run'; query['supported_pa'] = False
    np.testing.assert_array_equal(cache.transform(query), base.transform(query))
    np.testing.assert_array_equal(cache.transform(query), base.transform(frame.iloc[[12, 8, 8]]))
    changed_index = query.copy(); changed_index.index = [0, 1, 2]
    with pytest.raises(ValueError, match='dependency changed'): cache.transform(changed_index)


@pytest.mark.parametrize('length', [0, 32, 128])
def test_lazy_inputs_logits_gradients_fit_and400draw_delivery_are_exact(length):
    frame, base = fixture(); cache = FrozenObservedContext(base, frame, np.arange(len(frame)), chunk_size=3)
    store = MatrixLongHistoryStore.from_frame(frame, long_length=length)
    left = LazyPitchBatch(store, base, np.array([12, 8, 8, 2]),
        current=np.full((4, 8), .25, np.float32), candidate_pitch_types=['SL', 'FF', 'SL', 'FF'])
    right = LazyPitchBatch(store, cache, left.rows, current=left.current, candidate_pitch_types=left.candidate_pitch_types)
    for a, b in zip(left.gather(), right.gather()): np.testing.assert_array_equal(a, b)
    torch.set_num_threads(1); torch.manual_seed(28)
    net = DualStreamNetwork(left.gather()[0].shape[-1], 52, width=8); other = deepcopy(net)
    a = net(*[torch.as_tensor(v) for v in left.gather()])
    b = other(*[torch.as_tensor(v) for v in right.gather()])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(); b.square().sum().backward()
    for p, q in zip(net.parameters(), other.parameters()): torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    train = LazyPitchBatch(store, base, np.arange(len(frame)))
    cached = LazyPitchBatch(store, cache, train.rows)
    y = np.arange(len(frame)) % 10
    reference = LazyMatrixModel(seed=3, width=8, device='cpu').fit(train,y,train,y,epochs=2,patience=2,batch_size=3)
    accelerated = LazyMatrixModel(seed=3, width=8, device='cpu').fit(cached,y,cached,y,epochs=2,patience=2,batch_size=3)
    assert reference.report['history'] == accelerated.report['history']
    assert reference.report['optimizer_updates'] == accelerated.report['optimizer_updates']
    for key, value in reference.net.state_dict().items(): torch.testing.assert_close(value, accelerated.net.state_dict()[key], rtol=0, atol=0)
    delivery = LazyJointDelivery(FrozenSyntheticPool(), pitch_chunk=3, model_batch_size=257)
    for x,z in zip(delivery.logits(reference,left), delivery.logits(reference,right)): np.testing.assert_array_equal(x,z)


def test_generic_action_dependent_encoders_cannot_be_cached():
    frame, base = fixture()
    class Unknown:
        def transform(self, rows): return rows.pitch_type.eq('SL').to_numpy()
    with pytest.raises(ValueError, match='Only the frozen'): FrozenObservedContext(Unknown(), frame, [1])
    with pytest.raises(ValueError, match='canonical'): FrozenObservedContext(base, frame.set_index('pitch_number'), [1])
