"""전이 모델 ⓑ (공유 신경망 + 투수 임베딩) 계약: 텐서 스키마, 규칙 마스크, 투수 임베딩의 역할, 시드 재현."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pitcheezy.interfaces import outcomes as O  # noqa: E402
from pitcheezy.interfaces import states as S  # noqa: E402
from pitcheezy.interfaces.grid import N_ACTIONS  # noqa: E402
from pitcheezy.interfaces.validate import validate_transition  # noqa: E402
from pitcheezy.transition import neural as TN  # noqa: E402

HP = {"hidden": 32, "cat_dim": 4, "pitcher_dim": 4, "batch_size": 256, "max_epochs": 3, "patience": 2, "device": "cpu", "lr": 0.01}
META = {"data_version": "test", "season_window": "t", "holdout_split": "none", "seed": 0, "train_commit": "t", "cluster_file_version": "K1-none", "pitch_type_map_version": "v1"}


def synth(n=4000, seed=0):
    """투수 2명. 투수 0 은 구종 0 에서 헛스윙(STRIKE)이 많고, 투수 1 은 볼이 많다. 규칙 마스크를 지키는 결과만."""
    rng = np.random.default_rng(seed)
    p = rng.integers(0, 2, n)
    c = rng.integers(0, S.N_COUNTS, n)
    b = rng.integers(0, S.N_BASE_OUT, n)
    a = rng.integers(0, 25, n)  # 구종 0 의 위치 25개
    mask = O.rule_mask_table()[c]
    pref = np.where(p == 0, O.STRIKE, O.BALL)
    o = np.empty(n, dtype=np.int64)
    for i in range(n):
        w = mask[i].astype(float)
        if mask[i, pref[i]]:
            w[pref[i]] += 8.0
        o[i] = rng.choice(O.N_OUTCOMES, p=w / w.sum())
    df = pd.DataFrame({"pitcher_idx": p, "action_id": a, "outcome_id": o, "pitch_id": np.zeros(n, dtype=int)})
    return df, S.state_id(c, b, np.zeros(n, dtype=np.int64), 1)


def pitchers(n=2):
    return pd.DataFrame({"pitcher_idx": np.arange(n, dtype=np.int32), "mlbam_id": np.arange(n, dtype=np.int64) + 1, "name": ["a", "b"][:n], "n_pitches_train": np.zeros(n, dtype=np.int32)})


def test_tensor_contract_and_rule_mask():
    df, sid = synth()
    t = TN.fit(df, sid, pitchers(), 1, hp=TN.hparams(HP), seed=0, n_epochs=2, repertoire_min=100, meta=META)
    assert validate_transition(t) == []
    assert t.P.shape == (2, S.n_states(1), N_ACTIONS, O.N_OUTCOMES) and t.P.dtype == np.float32
    assert t.valid[:, :, :25].all() and not t.valid[:, :, 25:].any()  # 던진 구종만 valid
    cid = S.decode_state(np.arange(S.n_states(1)), 1)[0]
    banned = ~O.rule_mask_table()[cid]  # [S, O]
    assert (t.P[:, :, :25, :][:, banned[:, None, :].repeat(25, 1)] == 0).all()
    assert int(t.n_obs.sum()) == len(df)


def test_pitcher_embedding_separates_pitchers_and_control_does_not():
    df, sid = synth()
    s00 = S.state_id(0, 0, 0, 1)
    t = TN.fit(df, sid, pitchers(), 1, hp=TN.hparams({**HP, "max_epochs": 8}), seed=0, n_epochs=8, repertoire_min=100, meta=META)
    assert t.P[0, s00, 0, O.STRIKE] > t.P[1, s00, 0, O.STRIKE] + 0.1
    assert t.P[1, s00, 0, O.BALL] > t.P[0, s00, 0, O.BALL] + 0.1
    t0 = TN.fit(df, sid, pitchers(), 1, hp=TN.hparams({**HP, "pitcher_dim": 0}), seed=0, n_epochs=2, repertoire_min=100, meta=META)
    np.testing.assert_allclose(t0.P[0], t0.P[1], atol=1e-6)  # 임베딩 없으면 모든 투수가 같은 분포


def test_early_stopping_picks_epoch_and_seed_reproducible():
    df, sid = synth()
    ev, ev_sid = synth(1000, seed=1)
    net, info = TN.train(TN._rows(df, sid, 1), 2, 1, TN.hparams(HP), 0, eval_rows=TN._rows(ev, ev_sid, 1))
    assert 1 <= info["best_epoch"] <= 3 and "eval_nll" in info["epochs"][0]
    a = TN.fit(df, sid, pitchers(), 1, hp=TN.hparams(HP), seed=3, n_epochs=1, repertoire_min=100, meta=META)
    b = TN.fit(df, sid, pitchers(), 1, hp=TN.hparams(HP), seed=3, n_epochs=1, repertoire_min=100, meta=META)
    np.testing.assert_allclose(a.P, b.P, atol=1e-6)


def test_unknown_hparam_rejected():
    with pytest.raises(ValueError):
        TN.hparams({"hiden": 3})
