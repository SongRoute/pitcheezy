"""Synthetic CPU checks for the EXP-P38-001 DRAFT pitcher-ID arm (no real data, no registered fit)."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp import direct_pitcher_id_model as pid  # noqa: E402

torch.set_num_threads(1)
CONTINUOUS = [2, 3, 5]
CONFIG = PROJECT.parents[1] / 'configs' / 'EXP-P38-001.json'


def arrays(n, rng, n_context=7):
    tokens = rng.normal(size=(n, 3, 24)).astype(np.float32)
    tokens[:, -1, :8] = 0
    tokens[:, -1, -11:] = 0
    valid = np.ones((n, 3), bool)
    valid[: n // 3, 0] = False
    return (tokens, valid, rng.normal(size=(n, n_context)).astype(np.float32)), rng.integers(0, 10, n)


def test_vocab_is_train_only_sorted_with_min_count_and_unk_for_unseen_ids():
    train = np.array([30] * 5 + [10] * 3 + [20] * 1, dtype=np.int64)
    vocab = pid.build_vocab(train, min_count=3)
    assert vocab['ids'].tolist() == [10, 30] and vocab['train_unk_rows'] == 1 and vocab['train_pitchers'] == 3
    # 20 is rare in TRAIN, 99 never appears in TRAIN (DEV-only cold start): both share UNK
    assert pid.map_ids([10, 30, 20, 99], vocab['ids']).tolist() == [1, 2, pid.UNK, pid.UNK]
    assert pid.map_ids([5, 99], np.array([], np.int64)).tolist() == [0, 0]


def test_vocab_refuses_rows_after_the_train_cutoff():
    ids = np.array([1, 1, 2])
    pid.build_vocab(ids, np.array([20250429, 20250430, 20250101]), 20250430, min_count=1)
    with pytest.raises(ValueError, match='after the TRAIN cutoff'):
        pid.build_vocab(ids, np.array([20250429, 20250501, 20250101]), 20250430, min_count=1)


def test_vocab_ignores_later_splits():
    rng = np.random.default_rng(0)
    train = rng.integers(0, 50, 2000)
    later = np.concatenate([train, rng.integers(50, 80, 500)])  # new pitchers only after TRAIN
    v = pid.build_vocab(train, min_count=10)
    assert v['ids_sha256'] == pid.build_vocab(train.copy(), min_count=10)['ids_sha256']
    assert set(pid.map_ids(later[2000:], v['ids']).tolist()) == {pid.UNK}


def test_untrained_network_equals_the_baseline_and_keeps_the_global_rng_stream():
    rng = np.random.default_rng(1)
    (tokens, valid, context), _ = arrays(40, rng)
    torch.manual_seed(3)
    baseline = pr.PeriodicResidualNetwork(7, 24, 3, CONTINUOUS, **pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic']))
    after_baseline = torch.rand(4)
    torch.manual_seed(3)
    net = pid.PitcherIdPeriodicResidualNetwork(8, 24, 3, CONTINUOUS, n_ids=5, id_seed=11,
                                               **pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic']))
    assert torch.equal(torch.rand(4), after_baseline)
    (_, _, with_id), _ = pid.id_inputs((tokens, valid, context), rng.integers(0, 9, 40), np.arange(1, 6))
    baseline.eval(), net.eval()
    t, v = torch.as_tensor(tokens), torch.as_tensor(valid)
    assert torch.equal(net(t, v, torch.as_tensor(with_id)), baseline(t, v, torch.as_tensor(context)))
    extra = sum(p.numel() for p in net.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert extra == (5 + 1) * pid.D_ID + 128 * pid.D_ID


def test_id_changes_logits_after_training_and_unk_rows_share_one_vector():
    rng = np.random.default_rng(2)
    (train, y), (early, ey) = arrays(400, rng), arrays(120, rng)
    pitcher, epitcher = rng.integers(0, 6, 400), rng.integers(0, 9, 120)  # 6-8 unseen in TRAIN
    y = np.where(pitcher == 1, 4, y)  # pitcher 1 carries real signal
    vocab = pid.build_vocab(pitcher, min_count=1)
    train_in, _ = pid.id_inputs(train, pitcher, vocab['ids'])
    early_in, unk_share = pid.id_inputs(early, epitcher, vocab['ids'])
    assert 0 < unk_share < 1
    model = pid.DirectPitcherIdModel(0, CONTINUOUS, device='cpu', vocab_ids=vocab['ids'])
    model.fit(train_in, y, early_in, ey, epochs=3, patience=2, batch_size=64, learning_rate=.005)
    table = model.net.id_table.detach()
    assert table[1:].abs().sum() > 0
    probe = tuple(a[:1].repeat(2, 0) if a.ndim > 1 else a for a in (early_in[0], early_in[1], early_in[2]))
    probe[2][0, -1], probe[2][1, -1] = 2, 0  # pitcher 1 vs UNK, everything else identical
    out = model.logits(probe)
    assert not np.allclose(out[0], out[1])


def test_round_trip_and_bad_index_refused(tmp_path):
    rng = np.random.default_rng(4)
    (train, y), (early, ey) = arrays(200, rng), arrays(60, rng)
    ids = np.array([3, 7, 9])
    tr, _ = pid.id_inputs(train, rng.choice([3, 7, 9, 11], 200), ids)
    ea, _ = pid.id_inputs(early, rng.choice([3, 7, 9, 11], 60), ids)
    model = pid.DirectPitcherIdModel(1, CONTINUOUS, device='cpu', vocab_ids=ids)
    model.fit(tr, y, ea, ey, epochs=2, patience=2, batch_size=64)
    model.save(tmp_path / 'm.pt')
    loaded = pid.DirectPitcherIdModel.load(tmp_path / 'm.pt', 'cpu')
    assert np.array_equal(loaded.vocab_ids, ids)
    np.testing.assert_array_equal(loaded.logits(ea), model.logits(ea))
    bad = (ea[0], ea[1], ea[2].copy())
    bad[2][0, -1] = 4  # beyond n_ids = 3
    with pytest.raises(ValueError, match='pitcher-id index'):
        model.logits(bad)
    with pytest.raises(ValueError, match='TRAIN vocabulary'):
        pid.DirectPitcherIdModel(1, CONTINUOUS, device='cpu').fit(tr, y, ea, ey, epochs=1)


def test_draft_config_is_two_arms_ten_fits_gpu3_only_unused_seeds():
    config = json.loads(CONFIG.read_text())
    assert config['status'] == 'DRAFT' and config['execution']['executable'] is False
    assert [a['name'] for a in config['arms']] == ['periodic_residual_counts', 'pitcher_id_periodic_residual_counts']
    base, arm = config['arms']
    assert {k: v for k, v in arm.items() if k not in ('name', 'pitcher_id', 'family')} == \
        {k: v for k, v in base.items() if k not in ('name', 'family')}
    assert arm['pitcher_id'] == {'min_train_pitches': pid.MIN_TRAIN_PITCHES, 'd_id': pid.D_ID, 'unk_index': pid.UNK,
                                 'vocab_source': 'train', 'last_train_date': 20250430}
    assert config['seeds'] == [44, 45, 46, 47, 48] and config['fits'] == 10
    assert config['device']['allowed_gpu_uuids'] == ['GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9']
    assert config['device']['max_parallel'] == 1
    assert config['training'] == {'epochs': 60, 'patience': 5, 'batch_size': 1024, 'learning_rate': 0.0005,
                                  'loss_weighting': 'uniform', 'hpo_trials': 0}
