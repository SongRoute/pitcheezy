"""Numerical embeddings in front of the residual backbone for the direct pre-pitch head (research frontier, DRAFT).

Combination deferred by direct_frontier_model ("periodic+residual combination"):
Gorishniy, Rubachev, Babenko 2022, "On Embeddings for Numerical Features in
Tabular Deep Learning" (https://arxiv.org/abs/2203.05556) pair numerical
embeddings with MLP and ResNet backbones.  Here the frozen PLR-lite embedding of
direct_periodic_model feeds the frozen-architecture ResNet backbone of
direct_residual_model instead of the flatten_mlp head.  NOT a faithful
reproduction: same deviations as the two frozen modules (fixed sigma .01, 16
frequencies, d_embedding 8, width 128 / hidden 256 / 2 blocks / dropout .1, GELU,
no tuning, our ten-class head and flattened contract).

Exact networks (fixed, registered in config):

  raw, e = as DirectPeriodicModel                      # same columns, same mask, e [N, n_embedded, 8]
  z      = Linear(n_raw + 8 * n_embedded, 128)(cat(raw, e.flatten(1)))
  2 x  z = z + Linear(256, 128)(Dropout(.1)(GELU(Linear(128, 256)(LayerNorm(z)))))
  logits = Linear(128, 10)(GELU(LayerNorm(z)))         # residual blocks/output as ResidualMLPNetwork

periodic_residual_mlp: e = ReLU(Linear(32, 8)(cat(cos, sin)(2*pi*c*x)))  (PLR-lite, frozen embed())
linear_residual_mlp:   e = ReLU(w * x + b), w, b [n_embedded, 8]         (per-feature "LR" embedding of
                       the same paper; control that keeps the backbone, the embedded columns and the
                       backbone input width but drops the periodic map.  Init U(-1/sqrt(8), 1/sqrt(8))
                       for w and b, chosen here; not checked against the package.)

The periodic network is built through PeriodicDirectNetwork.__init__ (same
frequency/embedding initialisation order); its flatten head is then discarded and
the residual modules are created, so parameters and RNG use differ from
DirectPeriodicModel only after the embedding.  The residual part mirrors
ResidualMLPNetwork module for module (tested by copying weights).  Training is
DirectResidualModel.fit (tested mirror of MatrixModel.fit); calibration stays
external.  Frozen modules are not edited.  Own archive formats; load refuses
another kind, architecture or continuous map.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_residual_model as residual
from pitchmdp.direct_residual_model import DirectResidualModel

KINDS = {'periodic': 'periodic_residual_mlp', 'linear': 'linear_residual_mlp'}
FORMATS = {'periodic': 'direct_periodic_residual_model_v1', 'linear': 'direct_linear_residual_model_v1'}
BACKBONE = {k: residual.ARCHITECTURE[k] for k in ('width', 'hidden', 'blocks', 'dropout', 'normalization', 'activation')}
ARCHITECTURES = {  # nested: both frozen dicts use the key 'activation' (backbone gelu, embedding relu)
    'periodic': {'backbone': {'kind': 'residual_mlp', **BACKBONE}, 'embedding': dict(periodic.EMBEDDING)},
    'linear': {'backbone': {'kind': 'residual_mlp', **BACKBONE},
               'embedding': {'type': 'LR', 'd_embedding': periodic.EMBEDDING['d_embedding'], 'activation': 'relu'}},
}


class PeriodicResidualNetwork(periodic.PeriodicDirectNetwork):
    embedding_kind = 'periodic'

    def __init__(self, n_context, n_token, length, context_continuous, width=128, hidden=256, blocks=2, dropout=.1,
                 n_frequencies=16, d_embedding=8, sigma=.01, n_classes=10):
        super().__init__(n_context, n_token, length, context_continuous, width, n_frequencies, d_embedding, sigma,
                         n_classes)
        d_in = self.head[0].in_features
        del self.head
        self.d_embedding = d_embedding
        self.config.update(kind=KINDS[self.embedding_kind], hidden=hidden, blocks=blocks, dropout=dropout)
        self.input = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                                  nn.Dropout(dropout), nn.Linear(hidden, width))
                                    for _ in range(blocks))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, n_classes))

    def backbone(self, x):
        z = self.input(x)
        for block in self.blocks:
            z = z + block(z)
        return self.output(z)

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, periodic.N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(),
                         context[:, self.raw]), dim=1)
        return self.backbone(torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1))


class LinearResidualNetwork(PeriodicResidualNetwork):
    embedding_kind = 'linear'

    def __init__(self, n_context, n_token, length, context_continuous, width=128, hidden=256, blocks=2, dropout=.1,
                 d_embedding=8, n_classes=10):
        super().__init__(n_context, n_token, length, context_continuous, width, hidden, blocks, dropout,
                         d_embedding=d_embedding, n_classes=n_classes)
        for name in ('n_frequencies', 'sigma'):
            self.config.pop(name)
        del self.frequencies, self.embedding
        bound = 1 / math.sqrt(d_embedding)
        self.weight = nn.Parameter(torch.empty(self.n_embedded, d_embedding).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(self.n_embedded, d_embedding).uniform_(-bound, bound))

    def embed(self, tokens, valid, context):
        x = torch.cat((tokens[:, :-1, :periodic.N_PHYSICAL].flatten(1), context[:, self.continuous]), dim=1)
        e = torch.relu(x[..., None] * self.weight + self.bias)
        keep = torch.cat((valid[:, :-1, None].expand(-1, -1, periodic.N_PHYSICAL).flatten(1),
                          torch.ones_like(context[:, self.continuous], dtype=torch.bool)), dim=1)
        return e * keep[..., None]


NETWORKS = {'periodic': PeriodicResidualNetwork, 'linear': LinearResidualNetwork}


def network_kwargs(embedding, architecture):
    b, e = architecture['backbone'], architecture['embedding']
    kwargs = {k: b[k] for k in ('width', 'hidden', 'blocks', 'dropout')}
    kwargs['d_embedding'] = e['d_embedding']
    if embedding == 'periodic':
        kwargs.update(n_frequencies=e['n_frequencies'], sigma=e['sigma'])
    return kwargs


def parameter_count(embedding, n_context, n_token, length, context_continuous):
    net = NETWORKS[embedding](n_context, n_token, length, context_continuous,
                              **network_kwargs(embedding, ARCHITECTURES[embedding]))
    return sum(p.numel() for p in net.parameters())


class DirectEmbeddingResidualModel(DirectResidualModel):
    """DirectResidualModel training loop with an embedded-input residual network and a zero-current-physics guard."""

    embedding = None

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None):
        fixed = ARCHITECTURES[self.embedding]
        self.architecture = copy.deepcopy(fixed if architecture is None else architecture)  # nested: no aliasing
        if self.architecture != fixed:
            raise ValueError('Embedding-residual architecture differs from the fixed registered network')
        self.context_continuous = [int(i) for i in context_continuous]
        self.kind, self.seed, self.width, self.n_classes = KINDS[self.embedding], seed, self.architecture['backbone']['width'], 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}

    def network(self, n_context, n_token, length):
        return NETWORKS[self.embedding](n_context, n_token, length, self.context_continuous,
                                        **network_kwargs(self.embedding, self.architecture), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        return super().fit(periodic._no_current_physics(train), labels, periodic._no_current_physics(calibration),
                           calibration_labels, **kwargs)

    def logits(self, arrays, batch_size=4096):
        return super().logits(periodic._no_current_physics(arrays), batch_size)

    def save(self, path):
        torch.save({'format': FORMATS[self.embedding], 'kind': self.kind, 'seed': self.seed,
                    'architecture': self.architecture, 'temperature': self.temperature,
                    'delivery_temperature': self.delivery_temperature, 'report': self.report,
                    'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMATS[cls.embedding] or payload.get('kind') != KINDS[cls.embedding]:
            raise ValueError('Unrecognized embedding-residual model archive')
        config = dict(payload['network_config'])
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device)
        expected = network_kwargs(cls.embedding, model.architecture)
        if config.pop('kind') != model.kind or {k: config[k] for k in expected} != expected:
            raise ValueError('Archived network differs from the archived architecture')
        model.net = NETWORKS[cls.embedding](**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


class DirectPeriodicResidualModel(DirectEmbeddingResidualModel):
    embedding = 'periodic'


class DirectLinearResidualModel(DirectEmbeddingResidualModel):
    embedding = 'linear'
