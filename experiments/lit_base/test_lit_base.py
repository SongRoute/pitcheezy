"""EXP-LIT-BASE-001 작은 합성 검사 (실자료 없음). `.venv/bin/python -m pytest experiments/lit_base/test_lit_base.py -q`"""
import numpy as np
import pytest

from experiments.lit_base import data as D
from experiments.lit_base import models as M
from experiments.lit_base import run as R


def test_type9_matches_prepitch_targets():
    targets = pytest.importorskip('experiments.gpu14h_prepitch.train', reason='옛 실험 모듈 experiments.gpu14h_prepitch는 이 공개 레포에 없음(원래 레포에만 있음)').targets
    assert np.array_equal(D.TYPE9, targets(np.arange(len(D.TYPES))))


def test_oof_encoding_never_uses_own_label():
    # one pitcher, one game per fold: the row's own label must not enter its encoding
    n = 10; game = np.arange(n); p = np.full(n, 7); b = np.arange(n); z = np.zeros(n, np.int64)
    y = np.zeros(n, np.int64); y[3] = 5
    enc = D.oof_encodings(4.0, 10, game, p, b, z, z, z, y)
    # row 3 (label 5) sees only the other nine rows (all label 0): its class-5 mass is the prior share only
    te = D.TargetEncoder(4.0).fit(p[game != 3], b[game != 3], z[game != 3], z[game != 3], z[game != 3], y[game != 3])
    assert np.allclose(enc['p'][3], te.transform(p[3:4], b[3:4], z[3:4], z[3:4], z[3:4])['p'][0])
    assert enc['p'][3, 5] < enc['p'][4, 5]


def test_token_features_pa_prefix():
    tok = {k: np.zeros((1, D.CONTEXT), np.int64) for k in ('type', 'out', 'balls', 'strikes', 'same')}
    tok['valid'] = np.zeros((1, D.CONTEXT), bool); tok['valid'][0, :4] = True
    tok['same'][0, :2] = 1; tok['type'][0, :4] = [7, 15, 7, 1]          # FF, SL (same PA), FF, CH (earlier PA)
    f = D.token_features(tok)
    assert f['pa_n'][0, 0] == 0.2 and f['pa_mix'][0, 0] == 0.5 and f['pa_mix'][0, 3] == 0.5
    assert f['last1_type'][0, 2 + 0] == 1 and f['last2_type'][0, 2 + 3] == 1
    assert np.isclose(f['game_mix'][0].sum(), 1.0)


def test_context_strictly_earlier():
    build_context_index = pytest.importorskip('experiments.ws_seq.train_seq', reason='옛 실험 모듈 experiments.ws_seq는 이 공개 레포에 없음(원래 레포에만 있음)').build_context_index
    pit = np.array([1, 1, 1, 2]); g = np.array([5, 5, 5, 5]); ab = np.array([1, 1, 2, 2]); pn = np.array([1, 2, 1, 1])
    ctx = build_context_index(pit, g, ab, pn, 4)
    assert (ctx[0] == -1).all() and ctx[1, 0] == 0 and list(ctx[2, :2]) == [1, 0] and (ctx[3] == -1).all()


def test_temperature_recovers():
    rng = np.random.default_rng(0); z = rng.standard_normal((5000, 9)) * 2
    y = np.array([rng.choice(9, p=p) for p in M.softmax_np(z)])
    q_hot = M.softmax_np(z * 1.5)                      # over-confident by 1.5
    assert abs(M.fit_temperature(q_hot, y) - 1.5) < 0.1


def test_score_refuses_without_all_finals(tmp_path):
    cfg = R.load_cfg(); cfg['resources'] = {**cfg['resources'], 'output_root': str(tmp_path)}
    with pytest.raises(SystemExit):
        R.score(cfg, False, lambda m: None)


def test_real_run_refused_while_draft():
    cfg = R.load_cfg()
    if cfg['status'] != 'REGISTERED':
        with pytest.raises(SystemExit):
            R.require_registered(cfg, False)


def test_window_guard():
    import datetime as dt
    cfg = R.load_cfg()
    a, _ = cfg['resources']['forbidden_windows_kst'][0]
    inside = dt.datetime.fromisoformat(a).replace(tzinfo=R.KST) + dt.timedelta(minutes=5)
    assert R.in_window(cfg, inside) and not R.in_window(cfg, inside - dt.timedelta(hours=2))


def _preq_module():
    import importlib.util
    import os
    path = os.path.expandvars('${PITCHEEZY_REPO}/experiments/preq/preq.py')
    if not os.path.isfile(path):
        pytest.skip('곁가지 실험 코드 experiments/preq/preq.py가 이 공개 레포에 없음(PITCHEEZY_REPO에 원래 레포를 두면 실행됨)')
    spec = importlib.util.spec_from_file_location('preq_ref', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_tilt_matches_preq001():
    from experiments.lit_base import expl as X
    rng = np.random.default_rng(1); q = M.softmax_np(rng.standard_normal((50, 9))); m = M.softmax_np(rng.standard_normal((50, 9)))
    c = rng.integers(0, 20, (50, 9)).astype(float); ref = _preq_module()
    for k in (10.0, 500.0, float('inf')):
        assert np.allclose(X.tilt(q, m, c, k, [0.05, 20.0]), ref.tilt(q, m, c, k, [0.05, 20.0]), atol=0, rtol=0)


def test_asof_counts_strictly_earlier_days():
    from experiments.lit_base import expl as X
    pit = np.array([1, 1, 2, 1, 1]); day = np.array([0, 0, 0, 1, 2]); y = np.array([3, 4, 5, 3, 0])
    c = X.asof_counts(pit, day, y)
    assert c[:3].sum() == 0                         # first day: nothing earlier (same-day rows never count)
    assert c[3, 3] == 1 and c[3, 4] == 1 and c[3].sum() == 2
    assert c[4, 3] == 2 and c[4].sum() == 3


def test_gru_attn_masks_invalid_slots():
    import torch
    hp = dict(encoder='gru_attn', window=32, type_dim=8, hidden=16, pitcher_dim=4, batter_dim=4, context_dim=8, head_hidden=8, dropout=0.0, layers=1)
    net = M.build_seq(hp, 5, 5, 0).eval(); B = 3
    tok = {k: torch.randint(0, 3, (B, 32)) for k in ('type', 'out', 'balls', 'strikes', 'same')}
    tok['valid'] = torch.zeros(B, 32, dtype=torch.bool); tok['valid'][1, :4] = True; tok['valid'][2, :] = True
    args = (torch.randn(B, 11), torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    z1 = net(*args, tok)
    tok2 = {k: v.clone() for k, v in tok.items()}; tok2['type'][:2, 4:] = 7     # change only invalid slots of rows 0-1
    z2 = net(*args, tok2)
    assert torch.isfinite(z1).all() and torch.allclose(z1[:2], z2[:2])
