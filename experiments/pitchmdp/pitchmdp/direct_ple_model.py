"""Piecewise-linear (PLE, version B) embeddings in front of the residual backbone (EXP-P35-001, DRAFT).

Pre-registration: docs/PREREG-DRAFT-EXP-P35-001-ple-embedding-2026-10-04.md (§2 primary-source re-read, §4 network),
review OPUS-review-p35-prereg-4ccd13f and its disposition (B1 T_j = 0 rule, N4 padded weight, N14 disclosure).

Source: Gorishniy, Rubachev, Babenko 2022, "On Embeddings for Numerical Features in Tabular Deep Learning"
(arXiv:2203.05556v4, §3.2 Eq. 1, quantile bins §3.2.1) and the official package rtdl_num_embeddings
(https://github.com/yandex-research/rtdl-num-embeddings, MIT, read at commit a8fc250 = package 0.0.12):
``PiecewiseLinearEmbeddings(bins, d, activation=False, version="B")`` = ``LinearEmbeddings`` (weight and bias
U(-d^-1/2, d^-1/2)) + a per-feature PLE linear layer with NO bias and zero-initialised weight, so the embedding
starts as a linear embedding.  The README names version B "the version used in the different paper about the TabM
model" (TabM, arXiv:2410.24210v3, §D.8).  Written here from that source; no third-party code is copied and the
package is not installed.  NOT a faithful reproduction: d_embedding 8 (README start 12) to hold the backbone input
width of the periodic arms; n_bins 48 (package default); numpy float64 quantiles cast to float32 before unique
(package: float32 torch.quantile); a column with one distinct TRAIN value gets T_j = 0 (package raises); physics
columns are binned on rows whose past position is valid only; our residual backbone, ten-class head, AdamW loop
(weight decay .01 also on the PLE parameters; the global grad-norm clip 5 now includes them), no tuning.

Exact embedding (fixed):

  x        = cat(t[:, :-1, :8].flatten(1), context[:, continuous])            # 81 scalars, as the periodic arms
  edges_j  = unique(float32(quantile_float64(TRAIN_j, linspace(0, 1, 49))))   # b_0 < ... < b_{T_j}, T_j <= 48
  z_j,t    = (x_j - b_{t-1}) / (b_t - b_{t-1})                                # per bin, scale/shift as in the package
  PLE_j    = [clamp_max(z_1, 1), clamp(z_2..z_{T-1}, 0, 1), clamp_min(z_T, 0)] # 2022 Eq. 1; first/last pieces extrapolate
             T_j = 1: one UNCLAMPED piece;  T_j = 0 (one distinct TRAIN value): no PLE term
             padded to 48 in the package layout: pieces 1..T_j-1 at 0..T_j-2, piece T_j at index 47, zeros between
  e_j      = w0_j * x_j + c0_j + einsum(PLE_j, W_j * mask_j)                  # w0, c0 [81, 8]; W [81, 48, 8] zero init
  e        = e * keep                                                         # padded past positions -> exactly 0

Bins are computed in ``fit`` from the ``train`` argument only (never the early-stop, temperature, DEV or mlb_dev
rows), stored as persistent buffers (``ple_edges`` [81, 49], ``ple_counts`` [81]), hashed (``ple_bins_sha256``:
sha256 over, per feature in embedding order, T_j as little-endian int64 then its T_j + 1 float32 edges, little-endian;
one value for T_j = 0) and never recomputed at predict or load time.  ``mask`` keeps padded weight entries out of
the forward pass and their gradient exactly zero, so AdamW leaves them at exactly 0.

Networks: built through the frozen PeriodicResidualNetwork / PeriodicResidualTabMNetwork constructors (identical RNG
order), then the periodic ``frequencies`` and ``embedding`` are deleted and w0, c0 are drawn LAST (P32
LinearResidualNetwork pattern; the zero W draws nothing), so for one seed every backbone parameter (and r, W, b for
TabM) equals the periodic counterpart's initial value.  Training is the frozen DirectResidualModel.fit (single head)
or DirectTabMModel.fit (TabM, as P34), unchanged; only bins are computed before it.  Own archive formats; load
refuses another kind, architecture, k, n_bins, d or a bins buffer that does not match the recorded hash.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_periodic_residual_model as pr
from pitchmdp import direct_tabm_model as tm
from pitchmdp import direct_tabm_residual_model as trm
from pitchmdp.direct_residual_model import DirectResidualModel

N_BINS = 48
D_EMBEDDING = periodic.EMBEDDING['d_embedding']
EMBEDDING = {'type': 'PLE', 'version': 'B', 'bins': 'quantile_train', 'n_bins': N_BINS, 'd_embedding': D_EMBEDDING,
             'activation': False, 'ple_linear_bias': False, 'ple_linear_init': 'zeros',
             'degenerate_columns': 'T_j=0'}
ARCHITECTURE = {'backbone': {'kind': 'residual_mlp', **pr.BACKBONE}, 'embedding': EMBEDDING}
TABM_ARCHITECTURE = {**copy.deepcopy(ARCHITECTURE), **trm.ENSEMBLE}
KINDS = {'single': 'ple_residual_mlp', 'tabm': 'ple_residual_tabm_mini_mlp'}
FORMATS = {'single': 'direct_ple_residual_model_v1', 'tabm': 'direct_tabm_mini_ple_residual_v1'}
DERIVED = ('ple_bins_sha256', 'ple_n_bins', 'ple_degenerate_columns', 'ple_effective_parameter_count')


def feature_labels(length, context_continuous):
    """Embedded-scalar labels in embedding order: past position p, physics channel c, then context indices."""
    return [f'past{p}.phys{c}' for p in range(length - 1) for c in range(periodic.N_PHYSICAL)] + \
        [f'context{int(i)}' for i in context_continuous]


def compute_bins(arrays, context_continuous, n_bins=N_BINS):
    """Per-feature quantile edges from ONE split (call with TRAIN only); float32 1-D arrays, embedding order.

    Physics scalars use the rows whose past position is valid; context scalars use every row.  One distinct value
    -> a single edge (T_j = 0, B1).  Fails closed on empty or non-finite columns and on invalid edges.
    """
    tokens, valid, context = (np.asarray(a) for a in arrays)
    levels = np.linspace(0., 1., n_bins + 1)
    columns = [tokens[valid[:, p], p, c] for p in range(tokens.shape[1] - 1) for c in range(periodic.N_PHYSICAL)]
    columns += [context[:, int(i)] for i in context_continuous]
    bins = []
    for values in columns:
        values = np.asarray(values, dtype=np.float32)
        if not len(values):
            raise ValueError('PLE binning column has no rows')
        if not np.isfinite(values).all():
            raise ValueError('PLE binning values must be finite')
        if values.min() == values.max():
            bins.append(values[:1].copy())
            continue
        quantiles = np.quantile(values.astype(np.float64), levels, method='linear')
        bins.append(np.unique(quantiles.astype(np.float32)))  # unique AFTER the float32 cast (review N5)
    check_bins(bins, n_bins)
    return tuple(bins)


def check_bins(bins, n_bins=N_BINS):
    """Validate edges; return the per-feature bin counts T_j."""
    counts = []
    for edges in bins:
        edges = np.asarray(edges)
        if edges.dtype != np.float32 or edges.ndim != 1 or not len(edges):
            raise ValueError('PLE edges must be non-empty 1-D float32 arrays')
        if not np.isfinite(edges).all():
            raise ValueError('PLE edges must be finite')
        if len(edges) - 1 > n_bins:
            raise ValueError('PLE feature has more bins than n_bins')
        if np.any(edges[1:] <= edges[:-1]):
            raise ValueError('PLE float32 edges must be strictly increasing')
        counts.append(len(edges) - 1)
    return counts


def bins_sha256(bins):
    digest = hashlib.sha256()
    for edges in bins:
        edges = np.asarray(edges)
        digest.update(np.int64(len(edges) - 1).astype('<i8').tobytes())
        digest.update(edges.astype('<f4').tobytes())
    return digest.hexdigest()


def encoding_tables(bins, n_bins=N_BINS):
    """Package-layout scale/shift [F, n_bins], piece mask [F, n_bins] and single-bin flags [F] (float32 arithmetic)."""
    counts = check_bins(bins, n_bins)
    f = len(bins)
    scale, shift = torch.zeros(f, n_bins), torch.zeros(f, n_bins)
    piece = torch.zeros(f, n_bins, dtype=torch.bool)
    for j, edges in enumerate(bins):
        t = counts[j]
        if not t:
            continue
        edges = torch.as_tensor(np.asarray(edges))
        width = edges.diff()
        w, b = 1. / width, -edges[:-1] / width
        if not (torch.isfinite(w).all() and torch.isfinite(b).all()):
            raise ValueError('PLE bin scale is not finite in float32')
        scale[j, -1], shift[j, -1], piece[j, -1] = w[-1], b[-1], True
        scale[j, :t - 1], shift[j, :t - 1], piece[j, :t - 1] = w[:-1], b[:-1], True
    return scale, shift, piece, torch.tensor(counts) == 1


class _PLEEmbedding:
    """Replaces the inherited periodic embed(); installed after the periodic counterpart's constructor."""

    def _install_ple(self, bins, n_bins, d_embedding, length):
        if len(bins) != self.n_embedded:
            raise ValueError('PLE bins do not cover the embedded scalars')
        counts = check_bins(bins, n_bins)
        scale, shift, piece, single = encoding_tables(bins, n_bins)
        for name in ('n_frequencies', 'sigma'):
            self.config.pop(name)
        del self.frequencies, self.embedding
        edges = np.stack([np.concatenate((e, np.repeat(e[-1:], n_bins + 1 - len(e)))) for e in bins])
        self.register_buffer('ple_edges', torch.as_tensor(edges))
        self.register_buffer('ple_counts', torch.tensor(counts, dtype=torch.long))
        self.register_buffer('ple_scale', scale, persistent=False)
        self.register_buffer('ple_shift', shift, persistent=False)
        self.register_buffer('ple_mask', piece.float(), persistent=False)
        self.register_buffer('ple_single', single, persistent=False)
        bound = d_embedding ** -.5  # LinearEmbeddings.reset_parameters: weight, then bias
        self.linear0_weight = nn.Parameter(torch.empty(self.n_embedded, d_embedding).uniform_(-bound, bound))
        self.linear0_bias = nn.Parameter(torch.empty(self.n_embedded, d_embedding).uniform_(-bound, bound))
        self.ple_weight = nn.Parameter(torch.zeros(self.n_embedded, n_bins, d_embedding))  # version B: zeros, no bias
        labels = feature_labels(length, self.config['context_continuous'])
        padded = int((~piece).sum()) * d_embedding
        self.config.update(n_bins=n_bins, ple_bins_sha256=bins_sha256(bins), ple_n_bins=counts,
                           ple_degenerate_columns=[labels[j] for j, t in enumerate(counts) if not t],
                           ple_effective_parameter_count=sum(p.numel() for p in self.parameters()) - padded)

    def ple_encode(self, x):
        z = torch.addcmul(self.ple_shift, self.ple_scale, x[..., None])
        last = z[..., -1:]
        return torch.cat((z[..., :1].clamp_max(1.), z[..., 1:-1].clamp(0., 1.),
                          torch.where(self.ple_single[:, None], last, last.clamp_min(0.))), dim=-1)

    def embed(self, tokens, valid, context):
        x = torch.cat((tokens[:, :-1, :periodic.N_PHYSICAL].flatten(1), context[:, self.continuous]), dim=1)
        e = torch.addcmul(self.linear0_bias, self.linear0_weight, x[..., None]) + \
            torch.einsum('nfb,fbd->nfd', self.ple_encode(x), self.ple_weight * self.ple_mask[..., None])
        keep = torch.cat((valid[:, :-1, None].expand(-1, -1, periodic.N_PHYSICAL).flatten(1),
                          torch.ones_like(context[:, self.continuous], dtype=torch.bool)), dim=1)
        return e * keep[..., None]

    def archived_bins(self):
        return tuple(e[:t + 1].copy() for e, t in zip(self.ple_edges.cpu().numpy(), self.ple_counts.tolist()))


class PLEResidualNetwork(_PLEEmbedding, pr.PeriodicResidualNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, bins, width=128, hidden=256, blocks=2,
                 dropout=.1, n_bins=N_BINS, d_embedding=D_EMBEDDING, n_classes=10):
        check_bins(bins, n_bins)
        pr.PeriodicResidualNetwork.__init__(self, n_context, n_token, length, context_continuous, width, hidden, blocks,
                                            dropout, d_embedding=d_embedding, n_classes=n_classes)
        self.config.update(kind=KINDS['single'])
        self._install_ple(bins, n_bins, d_embedding, length)


class PLEResidualTabMNetwork(_PLEEmbedding, trm.PeriodicResidualTabMNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, bins, width=128, hidden=256, blocks=2,
                 dropout=.1, n_bins=N_BINS, d_embedding=D_EMBEDDING, k=8, scaling_init='normal', n_classes=10):
        check_bins(bins, n_bins)
        trm.PeriodicResidualTabMNetwork.__init__(self, n_context, n_token, length, context_continuous, width, hidden,
                                                 blocks, dropout, d_embedding=d_embedding, k=k,
                                                 scaling_init=scaling_init, n_classes=n_classes)
        self.config.update(kind=KINDS['tabm'])
        self._install_ple(bins, n_bins, d_embedding, length)


def network_kwargs(architecture, tabm=False):
    b, e = architecture['backbone'], architecture['embedding']
    kwargs = {**{k: b[k] for k in ('width', 'hidden', 'blocks', 'dropout')}, 'n_bins': e['n_bins'],
              'd_embedding': e['d_embedding']}
    if tabm:
        kwargs.update(k=architecture['k'], scaling_init=architecture['scaling_init'])
    return kwargs


def _network_from_archive(cls, payload, kind, expected):
    config, state = dict(payload['network_config']), payload['state_dict']
    if config.pop('kind', None) != kind or 'ple_edges' not in state or 'ple_counts' not in state \
            or {k: config.get(k) for k in expected} != expected:
        raise ValueError('Archived network differs from the archived architecture')
    derived = {k: config.pop(k, None) for k in DERIVED}
    edges, counts = state['ple_edges'].numpy(), state['ple_counts'].tolist()
    if edges.ndim != 2 or len(edges) != len(counts) or any(not 0 <= t < edges.shape[1] for t in counts):
        raise ValueError('Archived PLE bins are malformed')
    bins = tuple(e[:t + 1].copy() for e, t in zip(edges, counts))
    if bins_sha256(bins) != derived['ple_bins_sha256'] or derived['ple_bins_sha256'] != payload.get('ple_bins_sha256'):
        raise ValueError('Archived PLE bins differ from the recorded bins hash')
    net = cls(**config, bins=bins)
    net.load_state_dict(state)
    if {k: net.config[k] for k in DERIVED} != derived:
        raise ValueError('Archived network differs from the archived architecture')
    return net


class DirectPLEResidualModel(DirectResidualModel):
    """Frozen DirectResidualModel.fit with the PLE residual network; bins from the TRAIN argument only."""

    KIND, FORMAT = KINDS['single'], FORMATS['single']

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None):
        self.architecture = copy.deepcopy(ARCHITECTURE if architecture is None else architecture)
        if self.architecture != ARCHITECTURE:
            raise ValueError('PLE-residual architecture differs from the fixed registered network')
        self.context_continuous = [int(i) for i in context_continuous]
        self.kind, self.seed, self.width, self.n_classes = KINDS['single'], seed, self.architecture['backbone']['width'], 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}
        self.ple_bins = None

    def network(self, n_context, n_token, length):
        if self.ple_bins is None:
            raise RuntimeError('PLE bins are computed in fit from TRAIN only')
        return PLEResidualNetwork(n_context, n_token, length, self.context_continuous, self.ple_bins,
                                  **network_kwargs(self.architecture), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        train, calibration = periodic._no_current_physics(train), periodic._no_current_physics(calibration)
        self.ple_bins = compute_bins(train, self.context_continuous, self.architecture['embedding']['n_bins'])
        return super().fit(train, labels, calibration, calibration_labels, **kwargs)

    def logits(self, arrays, batch_size=4096):
        return super().logits(periodic._no_current_physics(arrays), batch_size)

    def save(self, path):
        torch.save({'format': FORMATS['single'], 'kind': self.kind, 'seed': self.seed,
                    'architecture': self.architecture, 'temperature': self.temperature,
                    'delivery_temperature': self.delivery_temperature, 'report': self.report,
                    'network_config': self.net.config, 'ple_bins_sha256': self.net.config['ple_bins_sha256'],
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMATS['single'] or payload.get('kind') != KINDS['single']:
            raise ValueError('Unrecognized PLE-residual model archive')
        config = payload['network_config']
        if not isinstance(config, dict) or 'context_continuous' not in config:
            raise ValueError('Archived network differs from the archived architecture')
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device)
        expected = {**network_kwargs(model.architecture), 'n_classes': model.n_classes}
        model.net = _network_from_archive(PLEResidualNetwork, payload, model.kind, expected).to(model.device)
        model.ple_bins = model.net.archived_bins()
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


class DirectPLETabMResidualModel(tm.DirectTabMModel):
    """Frozen DirectTabMModel.fit (P34 path) with the PLE TabM-residual network; bins from the TRAIN argument only."""

    KIND, FORMAT = KINDS['tabm'], FORMATS['tabm']  # the inherited class attributes name the P31 periodic TabM

    def __init__(self, seed=0, context_continuous=(), k=trm.ENSEMBLE['k'], scaling_init=trm.ENSEMBLE['scaling_init'],
                 device=None):
        tm.DirectTabMModel.__init__(self, seed, context_continuous, 0., k, scaling_init, device)
        self.kind = KINDS['tabm']
        self.width = pr.BACKBONE['width']
        self.architecture = {**copy.deepcopy(ARCHITECTURE), **trm.ENSEMBLE, 'k': int(k), 'scaling_init': scaling_init}
        self.ple_bins = None

    def network(self, n_context, n_token, length):
        if self.ple_bins is None:
            raise RuntimeError('PLE bins are computed in fit from TRAIN only')
        return PLEResidualTabMNetwork(n_context, n_token, length, self.context_continuous, self.ple_bins,
                                      **network_kwargs(self.architecture, tabm=True), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        # The frozen DirectTabMModel.fit guards and copies both splits itself; keeping guarded copies here would hold
        # one extra host copy of TRAIN + early-stop for the whole fit (same-model review c953bfd).
        periodic._no_current_physics(calibration)
        self.ple_bins = compute_bins(periodic._no_current_physics(train), self.context_continuous,
                                     self.architecture['embedding']['n_bins'])
        return super().fit(train, labels, calibration, calibration_labels, **kwargs)

    def save(self, path):
        torch.save({'format': FORMATS['tabm'], 'kind': self.kind, 'seed': self.seed,
                    'architecture': self.architecture, 'temperature': self.temperature,
                    'delivery_temperature': self.delivery_temperature, 'report': self.report,
                    'network_config': self.net.config, 'ple_bins_sha256': self.net.config['ple_bins_sha256'],
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMATS['tabm'] or payload.get('kind') != KINDS['tabm']:
            raise ValueError('Unrecognized PLE TabM-residual archive')
        config, a = payload['network_config'], payload['architecture']
        if not isinstance(a, dict) or set(a) != set(TABM_ARCHITECTURE) or not isinstance(config, dict) \
                or 'context_continuous' not in config:
            raise ValueError('Archived network differs from the archived architecture')
        model = cls(payload['seed'], config['context_continuous'], a.get('k'), a.get('scaling_init'), device)
        if model.architecture != a:
            raise ValueError('Archived network differs from the archived architecture')
        expected = {**network_kwargs(model.architecture, tabm=True), 'n_classes': model.n_classes}
        model.net = _network_from_archive(PLEResidualTabMNetwork, payload, model.kind, expected).to(model.device)
        model.ple_bins = model.net.archived_bins()
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


def parameter_count(n_context, n_token, length, context_continuous, bins, tabm=False):
    """(allocated, effective) parameter counts; padded PLE weight entries are allocated but fixed at 0."""
    if tabm:
        net = PLEResidualTabMNetwork(n_context, n_token, length, context_continuous, bins,
                                     **network_kwargs(TABM_ARCHITECTURE, tabm=True))
    else:
        net = PLEResidualNetwork(n_context, n_token, length, context_continuous, bins, **network_kwargs(ARCHITECTURE))
    return sum(p.numel() for p in net.parameters()), net.config['ple_effective_parameter_count']
