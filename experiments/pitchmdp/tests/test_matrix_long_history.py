"""Synthetic leakage, bounded expansion and lazy delivery regression tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_lazy_model import LazyMatrixModel, LazyJointDelivery, DualStreamNetwork
from pitchmdp.sequence_delivery import JointDelivery


def observations():
    rows = []
    plan = [('2024-04-01', 10, [(1, 7, 2), (2, 8, 1), (3, 7, 2)]),
            ('2024-04-02', 20, [(1, 7, 2), (2, 8, 1), (3, 7, 2)]),
            ('2024-04-02', 21, [(1, 7, 2)]),
            ('2024-04-03', 30, [(1, 7, 2)])]
    for date, game, pas in plan:
        for pa, batter, pitches in pas:
            for pitch in range(1, pitches + 1):
                i = len(rows)
                rows.append(dict(game_date=pd.Timestamp(date), game_pk=game, at_bat_number=pa,
                    pitch_number=pitch, batter=batter, pitcher=11, p_throws='R', stand='L',
                    split='train' if game == 10 else 'dev', pitch_type='FF' if pitch == 1 else 'SL',
                    description='ball' if pitch == 1 else 'called_strike', events=None,
                    supported_pa=True, balls=pitch - 1, strikes=0, effective_speed=90. + i,
                    release_spin_rate=2000. + i, spin_axis=20. + i, pfx_x=.1 + i / 100,
                    pfx_z=.3, plate_x=i / 10, plate_z=2.5))
    return pd.DataFrame(rows)


class Context:
    def report(self):
        return {'kind': 'synthetic_count_context', 'columns': 4}

    def transform(self, frame):
        return np.column_stack((frame.balls, frame.strikes, np.zeros((len(frame), 2)))).astype(np.float32)


def test_second_stream_excludes_current_pa_and_other_same_date_games():
    store = MatrixLongHistoryStore.from_frame(observations(), long_length=32)
    indices = store.long_indices([8, 9, 10, 12])
    observed = [row[row >= 0].tolist() for row in indices]
    assert observed[0] == [0, 1, 3, 4, 5, 6]
    assert observed[1] == observed[0]  # Current PA's first pitch is never in second stream.
    assert observed[2] == [0, 1, 3, 4]  # Game 20 is unavailable to same-date game 21.
    assert observed[3] == [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]
    assert store.report()['link_storage_bytes'] == len(store.frame) * 16
    h5, _, _, _ = store.gather([9])
    assert store.base.indices[9, -1] == 8
    assert 8 not in observed[1]
    assert h5.shape[1] == 6


def test_current_future_labels_and_retrospective_support_do_not_leak():
    frame = observations()
    store = MatrixLongHistoryStore.from_frame(frame, long_length=128)
    changed = frame.copy()
    changed.loc[8:, 'description'] = 'hit_by_pitch'
    changed['supported_pa'] = False
    other = MatrixLongHistoryStore.from_frame(changed, normalizer=store.normalizer,
                    type_vocabulary=store.base.type_vocabulary, long_length=128)
    for left, right in zip(store.gather([8]), other.gather([8])):
        np.testing.assert_array_equal(left, right)
    assert (other.gather([8])[0][:, -1, -11:] == 0).all()


def test_zero_history_same_capacity_and_links_shared():
    store = MatrixLongHistoryStore.from_frame(observations(), long_length=128)
    zero = store.with_length(0)
    assert zero.previous_global is store.previous_global
    h5, hmask, long, mask = zero.gather([12])
    assert long.shape == (1, 128, h5.shape[2])
    assert not mask.any() and not long.any()
    assert hmask.any()


def test_candidate_type_clone_and_current_physical_override():
    store = MatrixLongHistoryStore.from_frame(observations(), long_length=32)
    batch = LazyPitchBatch(store, Context(), [8], current=np.full((1, 8), 2.), candidate_pitch_types=['SL'])
    assert batch.frame().pitch_type.tolist() == ['SL']
    assert store.frame.iloc[8].pitch_type == 'FF'
    tokens = batch.gather()[0]
    np.testing.assert_array_equal(tokens[:, -1, :8], np.full((1, 8), 2.))
    assert tokens[0, -1, 8 + store.base.type_map['SL']] == 1


def test_lazy_fit_all_arms_capacity_save_load_and_bounded_gathers(tmp_path):
    base = MatrixLongHistoryStore.from_frame(observations(), long_length=128)
    counts = []
    for length in (0, 32, 128):
        store = base.with_length(length)
        batch = LazyPitchBatch(store, Context(), np.arange(10))
        labels = np.arange(10) % 10
        model = LazyMatrixModel(seed=0, width=8, device='cpu').fit(batch, labels, batch, labels,
                                                                epochs=2, batch_size=3)
        assert model.report['max_expanded_training_rows'] == 3
        counts.append(model.report['parameter_count'])
        before = model.logits(batch, batch_size=2)
        path = tmp_path / f'model{length}.pt'
        model.save(path)
        restored = LazyMatrixModel.load(path, device='cpu')
        np.testing.assert_array_equal(before, restored.logits(batch, batch_size=2))
        np.testing.assert_allclose(restored.predict(batch).sum(1), 1., atol=1e-6)
    assert len(set(counts)) == 1


def test_masked_padding_is_neutral_and_position_information_can_matter():
    store = MatrixLongHistoryStore.from_frame(observations(), long_length=32)
    batch = LazyPitchBatch(store, Context(), [8, 12])
    values = [torch.as_tensor(a) for a in batch.gather()]
    network = DualStreamNetwork(values[0].shape[-1], 4, width=8)
    before = network(*values).detach().numpy()
    values[0][~values[1]] = 999.
    values[2][~values[3]] = 999.
    np.testing.assert_array_equal(before, network(*values).detach().numpy())
    assert not torch.equal(network.position_codes[0], network.position_codes[-1])


def test_lazy_delivery_replaces_current_physics_and_propagates_candidate_type():
    store = MatrixLongHistoryStore.from_frame(observations(), long_length=32)
    batch = LazyPitchBatch(store, Context(), [8, 9])
    model = LazyMatrixModel(seed=0, width=8, device='cpu').fit(batch, np.array([0, 1]), batch,
                                                            np.array([0, 1]), epochs=1, batch_size=1)
    delivery = JointDelivery().fit(store.frame.loc[store.frame.split.eq('train')], store.normalizer, draws=4)
    lazy = LazyJointDelivery(delivery, pitch_chunk=1, model_batch_size=2)
    query = LazyPitchBatch(store, Context(), [8], candidate_pitch_types=['SL'])
    before = lazy.predict(model, query)[0]
    store.physical[8] = 99999.
    np.testing.assert_array_equal(before, lazy.predict(model, query)[0])
    lazy.calibrate(model, query, np.array([0]))
    assert .5 <= model.delivery_temperature <= 2.5
    assert lazy.predict(model, query)[0].dtype == np.float64


def test_ambiguous_suspended_game_and_invalid_lengths_rejected():
    frame = observations()
    frame.loc[5, 'game_pk'] = 10
    with pytest.raises(ValueError):
        MatrixLongHistoryStore.from_frame(frame)
    with pytest.raises(ValueError):
        MatrixLongHistoryStore.from_frame(observations(), long_length=64)


def test_artifact_adapter_freezes_keys_auxiliary_bytes_and_completed_members(tmp_path):
    from pitchmdp.matrix_long_experiment import fit_member, predict_member
    frame = observations()
    extra = frame.loc[frame.game_pk.eq(30)].copy()
    extra['game_pk'] = 40
    extra['game_date'] = pd.Timestamp('2024-04-04')
    frame = pd.concat([frame, extra], ignore_index=True)
    store = MatrixLongHistoryStore.from_frame(frame, long_length=0)
    batches = {name: LazyPitchBatch(store, Context(), np.flatnonzero(frame.game_pk.eq(game)))
               for name, game in zip(('train', 'earlystop', 'temperature', 'blend', 'dev'), (10, 20, 21, 30, 40))}
    delivery = JointDelivery().fit(frame.loc[frame.split.eq('train')], store.normalizer, draws=400)
    kwargs = dict(parent_preparation_sha256='a' * 64, batches=batches, frozen_delivery=delivery,
                  cell='F4-H0', seed=0, budget={'epochs': 1, 'patience': 1, 'batch_size': 2, 'learning_rate': .0005},
                  width=8, device='cpu')
    state = fit_member(tmp_path / 'member', **kwargs)
    assert fit_member(tmp_path / 'member', **kwargs) == state
    predicted = predict_member(tmp_path / 'member', batches=batches, frozen_delivery=delivery, device='cpu')
    assert predict_member(tmp_path / 'member', batches=batches, frozen_delivery=delivery, device='cpu') == predicted
    with np.load(tmp_path / 'member/predictions.npz') as archive:
        assert archive['dev'].dtype == np.float64
        np.testing.assert_array_equal(archive['dev_keys'][:, 0], archive['dev_game_pk'])
        assert archive['dev'].shape == (2, 10)
    delivery.fallback[0, 0] += .25
    with pytest.raises(ValueError, match='identity changed'):
        fit_member(tmp_path / 'member', **kwargs)
