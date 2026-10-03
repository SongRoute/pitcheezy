"""Periodic numerical embeddings for the direct pre-pitch head (research frontier, DRAFT).

Adaptation of the "PLR" periodic embeddings of Gorishniy, Rubachev, Babenko 2022,
"On Embeddings for Numerical Features in Tabular Deep Learning"
(https://arxiv.org/abs/2203.05556; official package rtdl_num_embeddings,
https://github.com/yandex-research/rtdl-num-embeddings, MIT license, read at
commit a8fc250).  Written here from the package's documented formula
ReLU(Linear(cat(cos v, sin v))), v = 2*pi*c*x, c ~ TruncNormal(0, sigma, +-3 sigma),
with the README's ``lite`` option (outer Linear shared across features).
No third-party code is copied.  NOT a faithful reproduction: fixed sigma .01
(package default, documented as safe), 16 frequencies and d_embedding 8
(package defaults 48 and 24; reduced to bound head growth), no tuning, our
ten-class head and flattened token/mask/context contract.

Exact network (fixed, registered in config):

  t      = tokens with masked positions zeroed                      # as flatten_mlp
  x_num  = cat(t[:, :-1, :8].flatten(1), context[:, continuous])    # 80 scalars
  e      = ReLU(Linear(32, 8)(cat(cos, sin)(2*pi*c*x_num)))         # [N, 80, 8]
  e      = e * mask                    # padded past physics embed to exactly 0
  raw    = cat(t[:, :-1, 8:].flatten(1), t[:, -1], valid, context[:, raw])
  logits = flatten_mlp head(cat(raw, e.flatten(1)))                 # Linear(.,256) GELU Linear(256,256) GELU Linear(256,10)

Embedded: past-token physical channels (standardized measurements) and the
TRAIN-fitted continuous context columns.  Raw: past/current pitch-type and
outcome one-hots, the padding mask, the current token (its physics is zero by
the direct-head contract and is never embedded, because embed(0) != 0),
game-state ordinals, binary flags and the near-binary pitcher left-hand share.

Training is DirectResidualModel.fit (itself a tested mirror of MatrixModel.fit);
only the network factory, the zero-current-physics guard and the archive differ.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pitchmdp.direct_residual_model import DirectResidualModel
from pitchmdp.matrix_models import _validate
from pitchmdp.matrix_sharing import CLUSTER_FEATURES

KIND = 'periodic_flatten_mlp'
FORMAT = 'direct_periodic_model_v1'
N_PHYSICAL = 8
EMBEDDING = {'type': 'PLR-lite', 'n_frequencies': 16, 'd_embedding': 8, 'sigma': .01,
             'activation': 'relu', 'frequencies_trainable': True}
ARCHITECTURE = {'backbone': 'flatten_mlp', 'width': 128, **EMBEDDING}
BASE_CONTEXT = 28  # SequenceContext: 11 game-state/flag columns + 6 style priors + 6 reliabilities + 5 memberships


def context_map(cluster_columns):
    """Continuous/raw split of the 52-wide direct context from the frozen pitcher-profile columns.

    Layout: SequenceContext (0-10 game state and flags, 11-27 batter archetype
    features), pitcher profile (CLUSTER_FEATURES means, left_share, type
    proportions), log1p(TRAIN count)/10.  Routing columns are already stripped.
    """
    columns = list(cluster_columns)
    if columns[:len(CLUSTER_FEATURES)] != list(CLUSTER_FEATURES) or columns[len(CLUSTER_FEATURES)] != 'left_share' \
            or not all(c.startswith('type_') for c in columns[len(CLUSTER_FEATURES) + 1:]):
        raise ValueError('Pitcher-profile columns differ from the direct context layout')
    left = BASE_CONTEXT + len(CLUSTER_FEATURES)
    n_context = BASE_CONTEXT + len(columns) + 1
    continuous = [i for i in range(11, n_context) if i != left]
    return {'n_context': n_context, 'continuous': continuous,
            'raw': [i for i in range(n_context) if i not in continuous]}


class PeriodicDirectNetwork(nn.Module):
    def __init__(self, n_context, n_token, length, context_continuous, width=128, n_frequencies=16,
                 d_embedding=8, sigma=.01, n_classes=10):
        super().__init__()
        continuous = [int(i) for i in context_continuous]
        if len(set(continuous)) != len(continuous) or not all(0 <= i < n_context for i in continuous) or n_token <= N_PHYSICAL:
            raise ValueError('Invalid continuous context map')
        self.config = dict(kind=KIND, n_context=n_context, n_token=n_token, length=length,
                           context_continuous=continuous, width=width, n_frequencies=n_frequencies,
                           d_embedding=d_embedding, sigma=sigma, n_classes=n_classes)
        self.register_buffer('continuous', torch.tensor(continuous, dtype=torch.long), persistent=False)
        self.register_buffer('raw', torch.tensor([i for i in range(n_context) if i not in continuous], dtype=torch.long),
                             persistent=False)
        self.n_embedded = (length - 1) * N_PHYSICAL + len(continuous)
        self.frequencies = nn.Parameter(torch.empty(self.n_embedded, n_frequencies))
        nn.init.trunc_normal_(self.frequencies, 0., sigma, a=-3 * sigma, b=3 * sigma)
        self.embedding = nn.Linear(2 * n_frequencies, d_embedding)
        n_raw = (length - 1) * (n_token - N_PHYSICAL) + n_token + length + n_context - len(continuous)
        self.head = nn.Sequential(nn.Linear(n_raw + self.n_embedded * d_embedding, width * 2), nn.GELU(),
                                  nn.Linear(width * 2, width * 2), nn.GELU(), nn.Linear(width * 2, n_classes))

    def embed(self, tokens, valid, context):
        past = tokens[:, :-1, :N_PHYSICAL]
        x = torch.cat((past.flatten(1), context[:, self.continuous]), dim=1)
        v = 2 * math.pi * self.frequencies * x[..., None]
        e = torch.relu(self.embedding(torch.cat((torch.cos(v), torch.sin(v)), -1)))
        keep = torch.cat((valid[:, :-1, None].expand(-1, -1, N_PHYSICAL).flatten(1),
                          torch.ones_like(context[:, self.continuous], dtype=torch.bool)), dim=1)
        return e * keep[..., None]

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(), context[:, self.raw]), dim=1)
        return self.head(torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1))


def parameter_count(n_context, n_token, length, context_continuous):
    net = PeriodicDirectNetwork(n_context, n_token, length, context_continuous)
    return sum(p.numel() for p in net.parameters())


def _no_current_physics(arrays):
    arrays = _validate(arrays)
    if np.any(arrays[0][:, -1, :N_PHYSICAL] != 0):
        raise ValueError('Current-pitch physics must be zero for the direct head')
    return arrays


class DirectPeriodicModel(DirectResidualModel):
    """DirectResidualModel training loop with the periodic network, a zero-current-physics guard and its own archive."""

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None):
        self.architecture = dict(ARCHITECTURE if architecture is None else architecture)
        if self.architecture != ARCHITECTURE:
            raise ValueError('Periodic architecture differs from the fixed registered network')
        self.context_continuous = [int(i) for i in context_continuous]
        self.kind, self.seed, self.width, self.n_classes = KIND, seed, self.architecture['width'], 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}

    def network(self, n_context, n_token, length):
        a = self.architecture
        return PeriodicDirectNetwork(n_context, n_token, length, self.context_continuous, a['width'],
                                     a['n_frequencies'], a['d_embedding'], a['sigma'], self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        return super().fit(_no_current_physics(train), labels, _no_current_physics(calibration), calibration_labels, **kwargs)

    def logits(self, arrays, batch_size=4096):
        return super().logits(_no_current_physics(arrays), batch_size)

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized direct periodic model archive')
        config = dict(payload['network_config'])
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device)
        if config.pop('kind') != KIND or {k: config[k] for k in ('width', 'n_frequencies', 'd_embedding', 'sigma')} != \
                {k: model.architecture[k] for k in ('width', 'n_frequencies', 'd_embedding', 'sigma')}:
            raise ValueError('Archived network differs from the archived architecture')
        model.net = PeriodicDirectNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
