"""Synthetic CPU checks for the PLE (version B) residual networks of EXP-P35-001 (DRAFT; no real data, no registered fit)."""
import copy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402
from pitchmdp import direct_ple_model as ple  # noqa: E402
from pitchmdp import direct_tabm_residual_model as trm  # noqa: E402
from pitchmdp.direct_workload_dropout_registry import CONTINUOUS  # noqa: E402

CONT = CONTINUOUS['counts']  # the registered 41-column `counts` map (P32/P34 arms)
SHAPE = (53, 38, 6)  # n_context, n_token, length of the counts export: n_raw 206, 81 embedded scalars
N_EMB = 5 * 8 + len(CONT)
KW = ple.network_kwargs(ple.ARCHITECTURE)
TKW = ple.network_kwargs(ple.TABM_ARCHITECTURE, tabm=True)
PR_KW = pr.network_kwargs('periodic', pr.ARCHITECTURES['periodic'])
FROZEN = {  # sha256 at a5fe718 (P34 register); this module must not change them
    'direct_periodic_model.py': '18f0b96a9b8a0d9eb39af2de4b0cc1b7a10a0dd42ca92977f726042e1aec80d3',
    'direct_periodic_residual_model.py': 'ee1b3d3f84b19f3564c853acf2bd12c640b1e5e597b8c62d485f839aeb8c992b',
    'direct_residual_model.py': '7feb9cfdcad85f4cb02c50a75e2ba2b528dd5546e600526b0334dcf0cf8e2d59',
    'direct_tabm_model.py': 'c968aec94fd7278d30dbbc53cb81fe4fcfc23c0b33b912a48ee09122e216b319',
    'direct_tabm_residual_model.py': '6a9daff6c3de13a753357d8615bd9f0d5a7bfe796b350f7f7bd0a5e60c9287cf',
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
    if constant is not None:
        context[:, constant] = 1.25
    return (tokens, valid, context), rng.integers(0, 10, n).astype(np.int64)


def small_fit(model, train_seed=11, early_seed=12, n=300, constant=None, epochs=2):
    train, y = arrays(n, train_seed, constant)
    early, ey = arrays(90, early_seed)
    return model.fit(train, y, early, ey, epochs=epochs, patience=2, batch_size=64)


def synthetic_bins(seed=0):
    return ple.compute_bins(arrays(400, seed)[0], CONT)


def tensors(n, seed):
    x, _ = arrays(n, seed)
    return [torch.as_tensor(a) for a in x]


def reference_ple(x, edges):
    """Equation 1 of Gorishniy et al. 2022, unpadded, float64, one feature (T = len(edges) - 1)."""
    t = len(edges) - 1
    out = np.zeros((len(x), t))
    for i in range(1, t + 1):
        z = (x - edges[i - 1]) / (edges[i] - edges[i - 1])
        if i > 1:
            z = np.where(x < edges[i - 1], 0., z)
        if i < t:
            z = np.where(x >= edges[i], 1., z)
        out[:, i - 1] = z
    return out


# ------------------------------------------------------------------ bins: TRAIN only, B1 rule, fail closed

def test_bins_come_from_the_train_argument_only():
    a = small_fit(ple.DirectPLEResidualModel(0, CONT, device='cpu'), early_seed=12)
    b = small_fit(ple.DirectPLEResidualModel(0, CONT, device='cpu'), early_seed=99)  # other early-stop rows
    c = small_fit(ple.DirectPLEResidualModel(0, CONT, device='cpu'), train_seed=21)  # other TRAIN rows
    sha = a.net.config['ple_bins_sha256']
    assert sha == b.net.config['ple_bins_sha256'] == ple.bins_sha256(ple.compute_bins(arrays(300, 11)[0], CONT))
    assert torch.equal(a.net.ple_edges, b.net.ple_edges) and torch.equal(a.net.ple_counts, b.net.ple_counts)
    assert c.net.config['ple_bins_sha256'] != sha
    pooled = [np.concatenate(p) for p in zip(arrays(300, 11)[0], arrays(90, 12)[0])]
    assert ple.bins_sha256(ple.compute_bins(pooled, CONT)) != sha  # TRAIN + early-stop would be detectable


def test_padded_past_positions_never_enter_binning():
    (tokens, valid, context), _ = arrays(300, 3)
    assert (~valid[:, 0]).any()
    noisy = tokens.copy()
    noisy[~valid] = 1e3 * np.random.default_rng(0).normal(size=noisy[~valid].shape)
    assert ple.bins_sha256(ple.compute_bins((noisy, valid, context), CONT)) == \
        ple.bins_sha256(ple.compute_bins((tokens, valid, context), CONT))
    bins = ple.compute_bins((tokens, valid, context), CONT)
    assert bins[0][0] == tokens[valid[:, 0], 0, 0].min() and bins[0][-1] == tokens[valid[:, 0], 0, 0].max()
    none = valid.copy()
    none[:, 0] = False
    with pytest.raises(ValueError, match='no rows'):
        ple.compute_bins((tokens, none, context), CONT)


def test_quantile_edges_match_the_documented_rule_and_hash_definition():
    (tokens, valid, context), _ = arrays(500, 4)
    bins = ple.compute_bins((tokens, valid, context), CONT)
    assert len(bins) == N_EMB and all(b.dtype == np.float32 for b in bins)
    col = context[:, CONT[3]].astype(np.float64)
    expect = np.unique(np.quantile(col, np.linspace(0, 1, 49)).astype(np.float32))
    assert np.array_equal(bins[40 + 3], expect) and len(expect) == 49  # continuous data: T_j = 48
    digest = hashlib.sha256()
    for e in bins:
        digest.update(np.int64(len(e) - 1).astype('<i8').tobytes() + e.astype('<f4').tobytes())
    assert ple.bins_sha256(bins) == digest.hexdigest()


def test_constant_column_gets_t0_records_its_name_and_embeds_linearly_after_a_step():
    pos = CONT.index(40)  # one pitch-type-proportion column, constant in this TRAIN
    model = small_fit(ple.DirectPLEResidualModel(1, CONT, device='cpu'), constant=40, epochs=1)
    cfg, j = model.net.config, 40 + pos
    assert cfg['ple_n_bins'][j] == 0 and cfg['ple_degenerate_columns'] == ['context40']
    assert model.net.ple_counts[j] == 0 and np.array_equal(model.ple_bins[j], np.array([1.25], np.float32))
    assert torch.equal(model.net.ple_weight[j], torch.zeros(48, 8))  # masked: no gradient, stays 0 under AdamW
    x = torch.randn(7, N_EMB)
    assert torch.equal(model.net.ple_encode(x)[:, j], torch.zeros(7, 48))
    assert all(t == 48 for i, t in enumerate(cfg['ple_n_bins']) if i != j)


def test_fail_closed_non_finite_too_many_bins_and_non_increasing_edges():
    (tokens, valid, context), _ = arrays(100, 5)
    bad = context.copy()
    bad[3, CONT[0]] = np.nan
    with pytest.raises(ValueError, match='finite'):
        ple.compute_bins((tokens, valid, bad), CONT)
    bins = list(synthetic_bins())
    for edges, match in ((np.array([0., np.inf], np.float32), 'finite'),
                         (np.arange(50, dtype=np.float32), 'more bins'),
                         (np.array([0., 1., 1.], np.float32), 'strictly increasing'),
                         (np.array([0., 2., 1.], np.float32), 'strictly increasing'),
                         (np.array([0., 1.]), 'float32'),
                         (np.zeros(0, np.float32), 'non-empty')):
        with pytest.raises(ValueError, match=match):
            ple.check_bins(bins[:5] + [edges] + bins[6:])
        with pytest.raises(ValueError):
            ple.PLEResidualNetwork(*SHAPE, CONT, bins[:5] + [edges] + bins[6:], **KW)
    with pytest.raises(ValueError, match='cover'):
        ple.PLEResidualNetwork(*SHAPE, CONT, bins[:-1], **KW)
    tiny = np.array([0., np.float32(1e-39)], np.float32)  # subnormal width: 1 / width overflows float32
    with pytest.raises(ValueError, match='scale'):
        ple.PLEResidualNetwork(*SHAPE, CONT, bins[:5] + [tiny] + bins[6:], **KW)


def test_float32_edge_collapse_is_handled_by_unique_after_cast():
    lo = np.float32(1.)
    hi = np.nextafter(lo, np.float32(2.))
    (tokens, valid, context), _ = arrays(400, 6)
    context[:, CONT[0]] = np.where(np.arange(400) % 2 == 0, lo, hi)  # level 24/48 sits at sorted position 199.5
    q = np.quantile(context[:, CONT[0]].astype(np.float64), np.linspace(0, 1, 49))
    assert len(np.unique(q)) > 2  # float64 edges between the two float32 values ...
    assert np.any(np.diff(np.unique(q).astype(np.float32)) <= 0)  # ... collapse if unique ran before the cast
    bins = ple.compute_bins((tokens, valid, context), CONT)
    assert np.array_equal(bins[40], np.array([lo, hi], np.float32))  # T_j = 1
    net = ple.PLEResidualNetwork(*SHAPE, CONT, bins, **KW)
    assert torch.isfinite(net.ple_scale).all() and torch.isfinite(net.ple_shift).all()


# ------------------------------------------------------------------ encoding: package layout, Eq. 1, mixed T_j

def test_mixed_t_padded_batch_matches_equation_1_with_package_layout_and_clamps():
    bins = list(synthetic_bins())
    picks = {0: np.array([.5], np.float32),                                   # T = 0
             1: np.array([-1., 2.], np.float32),                              # T = 1: unclamped both sides
             2: np.array([-1., 0., 3.], np.float32),                          # T = 2
             3: np.linspace(-2., 2., 49).astype(np.float32)}                  # T = 48
    for j, e in picks.items():
        bins[j] = e
    net = ple.PLEResidualNetwork(*SHAPE, CONT, bins, **KW)
    x = torch.zeros(9, N_EMB)
    probe = torch.tensor([-5., -1.5, -1., -.3, 0., .7, 2., 3., 9.])  # below, at and above the edges
    x[:, :4] = probe[:, None]
    enc = net.ple_encode(x).double().numpy()
    assert np.array_equal(enc[:, 0], np.zeros((9, 48)))
    for j in (1, 2, 3):
        t = len(bins[j]) - 1
        ref = reference_ple(probe.double().numpy(), bins[j].astype(np.float64))
        assert np.allclose(enc[:, j, :t - 1], ref[:, :t - 1], atol=1e-6)  # leading pieces at 0..T-2
        assert np.allclose(enc[:, j, -1], ref[:, -1], atol=1e-6)          # last piece at index 47
        assert np.array_equal(enc[:, j, t - 1:-1], np.zeros((9, 48 - t)))  # padding in between
    assert enc[0, 1, -1] < 0 and enc[-1, 1, -1] > 1                        # T = 1: no clamp
    assert enc[0, 2, 0] < 0 and enc[-1, 2, -1] > 1                         # first/last pieces extrapolate (N14)
    assert enc[0, 3, 0] < 0 and enc[-1, 3, -1] > 1 and enc[:, 3, 1:-1].min() >= 0 and enc[:, 3, 1:-1].max() <= 1
    assert net.config['ple_n_bins'][:4] == [0, 1, 2, 48] and net.config['ple_degenerate_columns'] == ['past0.phys0']
    mask = net.ple_mask.bool()
    assert mask[0].sum() == 0 and mask[1].tolist() == [False] * 47 + [True]
    assert mask[2].tolist() == [True] + [False] * 46 + [True] and bool(mask[3].all())


def test_zero_init_ple_term_means_the_network_equals_the_linear_embedding_path_at_init():
    torch.manual_seed(3)
    net = ple.PLEResidualNetwork(*SHAPE, CONT, synthetic_bins(), **KW)
    net.eval()
    tokens, valid, context = tensors(40, 7)
    tokens = tokens.masked_fill(~valid[..., None], 0.)
    x = torch.cat((tokens[:, :-1, :8].flatten(1), context[:, net.continuous]), dim=1)
    keep = torch.cat((valid[:, :-1, None].expand(-1, -1, 8).flatten(1), torch.ones(40, len(CONT), dtype=torch.bool)), 1)
    linear = (x[..., None] * net.linear0_weight + net.linear0_bias) * keep[..., None]
    assert torch.equal(net.ple_weight, torch.zeros(N_EMB, 48, 8))
    assert torch.allclose(net.embed(tokens, valid, context), linear, atol=1e-6, rtol=0)
    assert (net.embed(tokens, valid, context)[~keep] == 0).all()  # padded past positions embed to exactly 0
    noisy = tensors(40, 7)[0].clone()
    noisy[~valid] = torch.randn_like(noisy[~valid])
    assert torch.equal(net(noisy, valid, context), net(*tensors(40, 7)))


def test_linear0_init_is_linear_embeddings_uniform_drawn_last_like_the_p32_linear_network():
    torch.manual_seed(5)
    ours = ple.PLEResidualNetwork(*SHAPE, CONT, synthetic_bins(), **KW)
    torch.manual_seed(5)
    lr = pr.LinearResidualNetwork(*SHAPE, CONT, **pr.network_kwargs('linear', pr.ARCHITECTURES['linear']))
    assert torch.equal(ours.linear0_weight, lr.weight) and torch.equal(ours.linear0_bias, lr.bias)
    bound = 8 ** -.5
    assert ours.linear0_weight.abs().max() <= bound and ours.linear0_bias.abs().max() <= bound


# ------------------------------------------------------------------ backbone identity and parameter counts

@pytest.mark.parametrize('tabm', [False, True])
def test_every_backbone_parameter_equals_the_frozen_periodic_counterpart_for_one_seed(tabm):
    bins = synthetic_bins()
    torch.manual_seed(7)
    base = trm.PeriodicResidualTabMNetwork(*SHAPE, CONT, **PR_KW, k=8) if tabm else \
        pr.PeriodicResidualNetwork(*SHAPE, CONT, **PR_KW)
    torch.manual_seed(7)
    net = ple.PLEResidualTabMNetwork(*SHAPE, CONT, bins, **TKW) if tabm else ple.PLEResidualNetwork(*SHAPE, CONT, bins, **KW)
    shared, ours = dict(base.named_parameters()), dict(net.named_parameters())
    assert set(shared) - set(ours) == {'frequencies', 'embedding.weight', 'embedding.bias'}
    assert set(ours) - set(shared) == {'linear0_weight', 'linear0_bias', 'ple_weight'}
    for name in set(shared) & set(ours):
        assert torch.equal(shared[name], ours[name]), name
    assert net.input.in_features == base.input.in_features == 206 + N_EMB * 8
    assert net.config['kind'] == ple.KINDS['tabm' if tabm else 'single']
    assert 'n_frequencies' not in net.config and 'sigma' not in net.config


def test_parameter_counts_against_the_review_n3_bounds():
    full = synthetic_bins()  # continuous synthetic columns: every T_j = 48
    assert pr.parameter_count('periodic', *SHAPE, CONT) == 244_898  # = P34 probe table (frozen arm)
    assert trm.parameter_count(*SHAPE, CONT) == 260_760
    single, tabm = ple.parameter_count(*SHAPE, CONT, full), ple.parameter_count(*SHAPE, CONT, full, tabm=True)
    assert single == (275_738, 275_738) and tabm == (291_600, 291_600)  # N3 bounds minus 648 (no Linear_j bias)
    assert 276_386 - 648 == single[0] and 292_248 - 648 == tabm[0]
    mixed = list(full)
    mixed[0], mixed[1] = np.array([1.], np.float32), np.array([0., 1., 2.], np.float32)  # T = 0 and T = 2
    single = ple.parameter_count(*SHAPE, CONT, mixed)
    assert single[0] == 275_738 and single[1] == 244_634 + 8 * (sum(48 for _ in range(N_EMB - 2)) + 0 + 2)


# ------------------------------------------------------------------ training: masks, seeds, predict

def test_padded_weight_entries_stay_exactly_zero_after_optimizer_steps_and_pieces_learn():
    bins = list(ple.compute_bins(arrays(300, 11)[0], CONT))  # the TRAIN of small_fit: every real piece sees data
    bins[1], bins[2] = np.array([-1., 2.], np.float32), np.array([-1., 0., 3.], np.float32)
    model = ple.DirectPLEResidualModel(2, CONT, device='cpu')
    model.network = lambda n_context, n_token, length: ple.PLEResidualNetwork(
        n_context, n_token, length, CONT, bins, **KW)
    small_fit(model, epochs=1)
    w, mask = model.net.ple_weight.detach(), model.net.ple_mask.bool()
    assert (~mask).sum() == 47 + 46 and torch.equal(w[~mask], torch.zeros(int((~mask).sum()), 8))
    assert (w[mask] != 0).all()  # every real piece received gradient
    assert model.report['optimizer_updates'] == 5 and model.report['parameter_count'] == 275_738
    assert model.report['network']['ple_effective_parameter_count'] == 275_738 - 8 * 93


def test_bins_and_hash_identical_across_seeds_and_heads_parameters_differ():
    fits = [small_fit(ple.DirectPLEResidualModel(s, CONT, device='cpu')) for s in (29, 30)]
    fits.append(small_fit(ple.DirectPLETabMResidualModel(31, CONT, k=2, device='cpu')))
    shas = {m.net.config['ple_bins_sha256'] for m in fits}
    assert len(shas) == 1 and all(m.report['network']['ple_bins_sha256'] in shas for m in fits)
    assert not torch.equal(fits[0].net.linear0_weight, fits[1].net.linear0_weight)


def test_same_seed_is_deterministic():
    a = small_fit(ple.DirectPLEResidualModel(3, CONT, device='cpu'))
    b = small_fit(ple.DirectPLEResidualModel(3, CONT, device='cpu'))
    x, _ = arrays(20, 15)
    assert np.array_equal(a.logits(x), b.logits(x))


def test_predict_and_load_never_recompute_bins(tmp_path, monkeypatch):
    model = small_fit(ple.DirectPLEResidualModel(4, CONT, device='cpu'))
    model.save(tmp_path / 'm.pt')
    before = model.net.ple_edges.clone()
    monkeypatch.setattr(ple, 'compute_bins', lambda *a, **k: pytest.fail('bins recomputed outside fit'))
    x, _ = arrays(30, 16)
    p = model.predict(x)
    again = ple.DirectPLEResidualModel.load(tmp_path / 'm.pt', 'cpu')
    assert np.array_equal(again.predict(x), p) and torch.equal(model.net.ple_edges, before)
    assert ple.bins_sha256(again.ple_bins) == model.net.config['ple_bins_sha256']


def test_network_requires_fit_bins():
    with pytest.raises(RuntimeError, match='TRAIN'):
        ple.DirectPLEResidualModel(0, CONT, device='cpu').network(*SHAPE)
    with pytest.raises(RuntimeError, match='TRAIN'):
        ple.DirectPLETabMResidualModel(0, CONT, device='cpu').network(*SHAPE)


def test_tabm_fit_member_logits_report_and_physics_guard():
    model = small_fit(ple.DirectPLETabMResidualModel(0, CONT, k=4, device='cpu'))
    x, _ = arrays(30, 14)
    member = model.member_logits(x)
    assert member.shape == (30, 4, 10)
    assert np.allclose(model.predict(x), torch.softmax(torch.as_tensor(member), -1).mean(1).numpy(), atol=1e-6)
    a = model.report['architecture']
    assert a['embedding'] == ple.EMBEDDING and a['k'] == 4 and a['train_loss'] == 'mean_member_ce'
    assert model.report['network']['kind'] == ple.KINDS['tabm'] == model.kind
    bad = (x[0].copy(), x[1], x[2])
    bad[0][:, -1, 0] = 1.
    with pytest.raises(ValueError, match='physics'):
        model.logits(bad)
    with pytest.raises(NotImplementedError):
        model.net.backbone(torch.zeros(1, model.net.input.in_features))


# ------------------------------------------------------------------ archives

@pytest.mark.parametrize('tabm', [False, True])
def test_checkpoint_roundtrip_and_refusals(tmp_path, tabm):
    cls = ple.DirectPLETabMResidualModel if tabm else ple.DirectPLEResidualModel
    model = small_fit(cls(1, CONT, k=3, device='cpu') if tabm else cls(1, CONT, device='cpu'))
    model.temperature = 1.3
    model.save(tmp_path / 'm.pt')
    again = cls.load(tmp_path / 'm.pt', 'cpu')
    x, _ = arrays(25, 16)
    assert np.array_equal(model.logits(x), again.logits(x)) and again.temperature == 1.3
    assert again.architecture == model.architecture and again.context_continuous == list(CONT)
    edits = [('network_config', 'n_bins', 32), ('network_config', 'd_embedding', 12), ('network_config', 'dropout', .2),
             ('network_config', 'ple_bins_sha256', '0' * 64), ('ple_bins_sha256', None, '0' * 64),
             ('network_config', 'kind', pr.KINDS['periodic'])] + ([('network_config', 'k', 4)] if tabm else [])
    for part, key, value in edits:
        payload = torch.load(tmp_path / 'm.pt', weights_only=True)
        if key is None:
            payload[part] = value
        else:
            payload[part][key] = value
        torch.save(payload, tmp_path / 'bad.pt')
        with pytest.raises(ValueError, match='differ'):
            cls.load(tmp_path / 'bad.pt', 'cpu')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['state_dict']['ple_edges'][0, 0] -= 1.  # bins buffer no longer matches the recorded hash
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='hash'):
        cls.load(tmp_path / 'bad.pt', 'cpu')
    payload = torch.load(tmp_path / 'm.pt', weights_only=True)
    payload['architecture']['embedding']['n_bins'] = 32
    torch.save(payload, tmp_path / 'bad.pt')
    with pytest.raises(ValueError, match='differs'):
        cls.load(tmp_path / 'bad.pt', 'cpu')
    other = ple.DirectPLEResidualModel if tabm else ple.DirectPLETabMResidualModel
    with pytest.raises(ValueError, match='Unrecognized'):
        other.load(tmp_path / 'm.pt', 'cpu')
    frozen = trm.DirectTabMResidualModel if tabm else pr.DirectPeriodicResidualModel
    with pytest.raises(ValueError, match='Unrecognized'):
        frozen.load(tmp_path / 'm.pt', 'cpu')


def test_architecture_constants_are_not_aliased():
    model = ple.DirectPLEResidualModel(0, CONT, device='cpu')
    model.architecture['embedding']['n_bins'] = 1
    model.architecture['backbone']['width'] = 999
    assert ple.ARCHITECTURE['embedding']['n_bins'] == 48 and pr.ARCHITECTURES['periodic']['backbone']['width'] == 128
    tabm = ple.DirectPLETabMResidualModel(0, CONT, device='cpu')
    tabm.architecture['embedding']['d_embedding'] = 1
    assert ple.TABM_ARCHITECTURE['embedding']['d_embedding'] == 8 == ple.ARCHITECTURE['embedding']['d_embedding']
    with pytest.raises(ValueError, match='differs'):
        ple.DirectPLEResidualModel(0, CONT, copy.deepcopy(model.architecture), device='cpu')


def test_frozen_modules_are_byte_identical():
    for name, digest in FROZEN.items():
        assert hashlib.sha256((PROJECT / 'pitchmdp' / name).read_bytes()).hexdigest() == digest, name
