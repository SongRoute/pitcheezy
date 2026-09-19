"""B2 재구현(모듈 층) 계약: 피처 빌더(창·패딩·좌타 반전), 타자 전 시즌 규칙, 마스크된 로짓, 학습 루프. 원본 데이터 없이 합성 표로만."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pitcheezy.baselines import b2 as B2  # noqa: E402
from pitcheezy.interfaces import outcomes as O  # noqa: E402
from pitcheezy.interfaces import states as S  # noqa: E402

HP = {"d_model": 32, "n_heads": 2, "ff": 32, "n_layers": 1, "dropout": 0.0, "dense_pitch": 8, "dense_ctx": 8,
      "dense_head": 8, "batch_size": 64, "max_epochs": 2, "patience": 2, "device": "cpu", "lr": 1e-3}


def synth(n_pa=80, seed=0):
    """타석당 1~7구. id 표(prepare.OUT_COLUMNS 부분집합) + 원본 물리 표를 따로 만든다."""
    rng = np.random.default_rng(seed)
    ids, raw = [], []
    for i in range(n_pa):
        npitch = int(rng.integers(1, 8))
        stand = "L" if i % 3 == 0 else "R"
        for j in range(1, npitch + 1):
            cid = int(rng.integers(0, S.N_COUNTS))
            mask = O.rule_mask_table()[cid]
            o = int(rng.choice(np.flatnonzero(mask)))
            ids.append(
                {"game_pk": 1 + i // 20, "at_bat_number": i, "pitch_number": j, "inning": 1 + i % 9,
                 "inning_topbot": "Top" if i % 2 else "Bot", "pitcher": 100 + i % 3, "pitcher_idx": i % 3,
                 "batter": 500 + i % 7, "stand": stand, "count_id": cid, "base_out_id": int(rng.integers(0, S.N_BASE_OUT)),
                 "cluster_id": 0, "pitch_id": int(rng.integers(0, 9)), "loc_id": int(rng.integers(0, 25)),
                 "action_id": 0, "outcome_id": o, "bat_score": int(rng.integers(0, 6))}
            )
            ids[-1]["action_id"] = ids[-1]["pitch_id"] * 25 + ids[-1]["loc_id"]
            raw.append(
                {"game_pk": ids[-1]["game_pk"], "at_bat_number": i, "pitch_number": j,
                 "plate_x": float(rng.normal()), "plate_z": 2.0 + float(rng.normal(0, 0.5)), "sz_top": 3.4, "sz_bot": 1.6,
                 "effective_speed": 90.0 + float(rng.normal()), "release_speed": 89.0, "release_spin_rate": 2200.0 + float(rng.normal(0, 100)),
                 "spin_axis": float(rng.integers(0, 360)), "pfx_x": float(rng.normal()), "pfx_z": float(rng.normal()),
                 "fld_score": int(rng.integers(0, 6)), "n_thruorder_pitcher": 1 + i % 3}
            )
    return pd.DataFrame(ids), pd.DataFrame(raw)


def make_ds(seed=0, window=6):
    ids, raw = synth(seed=seed)
    fb = np.array([0.22, 0.03, 0.15, 0.25, 0.72])
    return B2.build_dataset(ids, raw, None, fb, window), ids, raw


# ---------------------------------------------------------------- 피처
def test_window_shapes_and_padding_mask():
    ds, ids, _ = make_ds()
    assert ds.win.shape == (len(ds), 6, B2.N_PHYS) and ds.pad.shape == (len(ds), 6)
    assert ds.ctx.shape == (len(ds), B2.N_CTX) == (len(ds), 16)
    d = ids.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    pn = d["pitch_number"].to_numpy()  # 모든 행이 적격이므로 ds 와 행 순서가 같다
    assert len(ds) == len(d)
    n_prev = (~ds.pad).sum(1)
    np.testing.assert_array_equal(n_prev, np.minimum(pn - 1, 6))
    assert ds.pad[pn == 1].all()  # 타석 첫 구는 전부 패딩
    assert (ds.win[pn == 1] == 0).all()
    assert not ds.pad[pn > 1][:, -1].any()  # 창의 마지막 칸 = 직전 구


def test_window_holds_previous_pitch_not_current():
    ds, ids, raw = make_ds()
    d = ids.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    d = d.merge(raw, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    i = int(np.flatnonzero(d["pitch_number"].to_numpy() >= 3)[0])
    np.testing.assert_allclose(ds.win[i, -1], ds.cur[i - 1])  # 직전 구
    np.testing.assert_allclose(ds.win[i, -2], ds.cur[i - 2])
    assert not np.allclose(ds.win[i, -1], ds.cur[i])


def test_lhb_x_axis_flip():
    ids, raw = synth()
    ids2 = ids.copy()
    ids2["stand"] = np.where(ids["stand"].to_numpy() == "L", "R", "L")  # 좌우 바꾸면 x 부호만 뒤집힌다
    fb = np.zeros(5)
    a = B2.build_dataset(ids, raw, None, fb, 6)
    b = B2.build_dataset(ids2, raw, None, fb, 6)
    np.testing.assert_allclose(a.cur[:, 0], -b.cur[:, 0])  # plate_x
    np.testing.assert_allclose(a.cur[:, 5], -b.cur[:, 5])  # pfx_x
    np.testing.assert_allclose(a.cur[:, 1:5], b.cur[:, 1:5])  # 나머지는 그대로 (spin_axis 포함)


def test_context_columns():
    ds, ids, raw = make_ds()
    d = ids.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    d = d.merge(raw, on=["game_pk", "at_bat_number", "pitch_number"], how="left")
    balls, strikes = S.decode_count(d["count_id"].to_numpy())
    np.testing.assert_array_equal(ds.ctx[:, 0], balls)
    np.testing.assert_array_equal(ds.ctx[:, 1], strikes)
    np.testing.assert_array_equal(ds.ctx[:, 8], d["bat_score"].to_numpy() - d["fld_score"].to_numpy())
    assert (ds.ctx[:, B2.CTX_NAMES.index("prev_season_missing")] == 1).all()  # prev 없음 → 전부 결측


def test_scaler_zero_fills_nan_and_masks_padding():
    ids, raw = synth()
    raw = raw.copy()
    raw.loc[raw.index[:10], "release_spin_rate"] = np.nan
    ds = B2.build_dataset(ids, raw, None, np.zeros(5), 6)
    sc = B2.Scaler.fit(ds)
    z = sc.apply(ds)
    assert np.isfinite(z.cur).all() and np.isfinite(z.win).all() and np.isfinite(z.ctx).all()
    assert (z.win[z.pad] == 0).all()


# ---------------------------------------------------------------- 타자 전 시즌 성적
def test_batter_season_stats_rule():
    """손으로 만든 타석 표. 타자 1: 10 PA = 2 HR, 1 단타, 1 2루타, 2 삼진, 1 볼넷, 1 사구, 1 희생플라이, 1 범타."""
    ev = ["home_run", "home_run", "single", "double", "strikeout", "strikeout", "walk", "hit_by_pitch", "sac_fly", "field_out"]
    pa = pd.DataFrame({"batter": [1] * len(ev) + [2] * 3, "events": ev + ["single", "field_out", "truncated_pa"]})
    st = B2.batter_season_stats(pa)
    r = st.loc[1]
    assert r["pa"] == 10
    ab = 10 - 1 - 1 - 1  # 볼넷·사구·희생플라이 제외 = 7
    h, tb = 4, 4 + 4 + 1 + 2
    assert r["avg"] == pytest.approx(h / ab)
    assert r["iso"] == pytest.approx(tb / ab - h / ab)
    obp = (h + 1 + 1) / (ab + 1 + 1 + 1)
    assert r["ops"] == pytest.approx(obp + tb / ab)
    assert r["k_pct"] == pytest.approx(0.2) and r["hr_pct"] == pytest.approx(0.2)
    assert st.loc[2]["pa"] == 2  # truncated_pa 는 PA 가 아니다


def test_prev_season_block_uses_league_mean_below_min_pa():
    prev = pd.DataFrame({"pa": [200.0, 10.0], "k_pct": [0.1, 0.9], "hr_pct": [0.05, 0.0], "iso": [0.2, 0.0], "avg": [0.3, 0.0], "ops": [0.9, 0.0]},
                        index=pd.Index([7, 8], name="batter"))
    fb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vals, miss = B2.prev_season_block(np.array([7, 8, 9]), prev, fb)
    np.testing.assert_allclose(vals[0], [0.1, 0.05, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(vals[1], fb)  # PA < 50 → 리그 평균
    np.testing.assert_allclose(vals[2], fb)  # 전 시즌에 없는 타자
    np.testing.assert_array_equal(miss, [0.0, 1.0, 1.0])


def test_league_means_only_qualified():
    st = pd.DataFrame({"pa": [100.0, 100.0, 5.0], "k_pct": [0.2, 0.3, 9.0], "hr_pct": [0.0] * 3, "iso": [0.0] * 3, "avg": [0.0] * 3, "ops": [0.0] * 3})
    assert B2.league_means(st)[0] == pytest.approx(0.25)


# ---------------------------------------------------------------- 모델·학습
def test_forward_masked_logits_normalise_to_one():
    ds, _, _ = make_ds()
    ds = B2.Scaler.fit(ds).apply(ds)
    hp = B2.hparams(HP)
    net = B2.B2Net(hp)
    p = B2.predict_probs(net, ds, torch.device("cpu"))
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-6)
    banned = ~O.rule_mask_table()[ds.count]
    assert (p[banned] == 0).all()
    assert np.isfinite(p).all()  # 타석 첫 구(전부 패딩)도 NaN 이 아니다


def test_train_two_epochs_returns_finite_nll():
    tr, _, _ = make_ds(seed=0)
    ev, _, _ = make_ds(seed=1)
    sc = B2.Scaler.fit(tr)
    tr, ev = sc.apply(tr), sc.apply(ev)
    net, info = B2.train(tr, ev, B2.hparams(HP), 0, max_epochs=2)
    assert len(info["epochs"]) == 2 and 1 <= info["best_epoch"] <= 2
    assert all(np.isfinite(e["train_nll"]) and np.isfinite(e["eval_nll"]) for e in info["epochs"])
    m = B2.metrics(B2.predict_probs(net, ev, torch.device("cpu")), ev.y)
    assert np.isfinite(m["holdout_nll"]) and 0.0 <= m["holdout_ece"] <= 1.0 and m["holdout_n_pitches"] == len(ev)
    assert set(m["holdout_ece_by_outcome"]) == set(O.OUTCOME_NAMES)


def test_unknown_hparam_rejected():
    with pytest.raises(ValueError):
        B2.hparams({"d_modl": 8})
