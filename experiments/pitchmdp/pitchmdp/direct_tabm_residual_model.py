"""TabM-mini adapter on the periodic-embedding residual backbone (research frontier, DRAFT).

Pre-registration: docs/DRAFT-TABM-RESIDUAL-DESIGN-20261004.md (committed before this code).
Combines two frozen DRAFT modules without editing them:

  * direct_periodic_residual_model.PeriodicResidualNetwork: frozen PLR-lite embedding in
    front of the frozen ResNet-style residual_mlp backbone;
  * direct_tabm_model: TabM-mini (Gorishniy, Kotelnikov, Babenko, ICLR 2025,
    https://arxiv.org/abs/2410.24210; package https://github.com/yandex-research/tabm
    commit 28e47ae) -- one ElementwiseAffine k x d_in (no bias) on the backbone input,
    fully shared backbone, per-member output layer, mean member loss, mean member
    probability.

Our adaptation, not in the package: the package's TabM backbone is an MLP; here the
shared backbone is the residual_mlp.  The scaling acts on the backbone input and the
per-member output replaces only the final Linear(128, 10).  Same deviations as the two
frozen modules (k=8, fixed sigma .01, 16 frequencies, d_embedding 8, width 128 / hidden
256 / 2 blocks / dropout .1, GELU, our AdamW loop, no tuning).  Not a faithful reproduction.

Exact network (fixed):

  x      = cat(raw, e.flatten(1))                     # as PeriodicResidualNetwork
  z      = Linear(D, 128)(x[:, None, :] * r)          # r [k, D]; input layer shared
  2 x  z = z + Linear(256, 128)(Dropout(.1)(GELU(Linear(128, 256)(LayerNorm(z)))))   # shared
  h      = GELU(LayerNorm(z))                         # shared
  member = einsum('nkd,kdc->nkc', h, W) + b           # W [k, 128, 10], b [k, 10]

The network is built through PeriodicResidualNetwork.__init__ (identical RNG order); its
final Linear is then discarded and r, W, b are drawn, so for one seed every shared
parameter equals the periodic_residual initialisation.  Training, ensemble logits and
batching are DirectTabMModel's (mean member CE; log mean member softmax; external
temperature on log p-bar).  Own archive format; load refuses another kind, k, scaling
init or backbone/embedding architecture.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_periodic_residual_model as pr
from pitchmdp import direct_tabm_model as tm

KIND = 'periodic_residual_tabm_mini_mlp'
FORMAT = 'direct_tabm_mini_periodic_residual_v1'
ENSEMBLE = dict(tm.ENSEMBLE)
ARCHITECTURE = {**copy.deepcopy(pr.ARCHITECTURES['periodic']), **ENSEMBLE}


class PeriodicResidualTabMNetwork(pr.PeriodicResidualNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, width=128, hidden=256, blocks=2, dropout=.1,
                 n_frequencies=16, d_embedding=8, sigma=.01, k=8, scaling_init='normal', n_classes=10):
        if isinstance(k, bool) or int(k) != k or k < 1 or scaling_init not in tm.SCALING_INITS:
            raise ValueError('Invalid TabM-mini ensemble settings')
        super().__init__(n_context, n_token, length, context_continuous, width, hidden, blocks, dropout,
                         n_frequencies, d_embedding, sigma, n_classes)
        norm, act, _ = self.output  # frozen final Linear created (RNG order kept) and discarded
        self.output = nn.Sequential(norm, act)
        d_in = self.input.in_features
        self.n_raw = d_in - self.n_embedded * d_embedding
        self.scaling = nn.Parameter(tm.init_scaling(torch.empty(k, d_in),
                                                    tm.feature_chunks(self.n_raw, self.n_embedded, d_embedding),
                                                    scaling_init))
        bound = 1 / math.sqrt(width)
        self.output_weight = nn.Parameter(torch.empty(k, width, n_classes).uniform_(-bound, bound))
        self.output_bias = nn.Parameter(torch.empty(k, n_classes).uniform_(-bound, bound))
        self.config.update(kind=KIND, k=int(k), scaling_init=scaling_init)

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, periodic.N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(),
                         context[:, self.raw]), dim=1)
        x = torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1)
        z = self.input(x[:, None, :] * self.scaling)
        for block in self.blocks:
            z = z + block(z)
        return torch.einsum('nkd,kdc->nkc', self.output(z), self.output_weight) + self.output_bias


def network_kwargs(architecture):
    return {**pr.network_kwargs('periodic', architecture), 'k': architecture['k'],
            'scaling_init': architecture['scaling_init']}


def parameter_count(n_context, n_token, length, context_continuous, k=ENSEMBLE['k']):
    net = PeriodicResidualTabMNetwork(n_context, n_token, length, context_continuous,
                                      **{**network_kwargs(ARCHITECTURE), 'k': k})
    return sum(p.numel() for p in net.parameters())


class DirectTabMResidualModel(tm.DirectTabMModel):
    """DirectTabMModel (mean member loss, ensemble logits) with the periodic residual backbone and its own archive."""

    KIND, FORMAT = KIND, FORMAT

    def __init__(self, seed=0, context_continuous=(), k=ENSEMBLE['k'], scaling_init=ENSEMBLE['scaling_init'],
                 device=None):
        tm.DirectTabMModel.__init__(self, seed, context_continuous, 0., k, scaling_init, device)
        self.kind = KIND
        self.width = pr.BACKBONE['width']
        self.architecture = {**copy.deepcopy(pr.ARCHITECTURES['periodic']), **ENSEMBLE, 'k': int(k),
                             'scaling_init': scaling_init}

    def network(self, n_context, n_token, length):
        return PeriodicResidualTabMNetwork(n_context, n_token, length, self.context_continuous,
                                           **network_kwargs(self.architecture), n_classes=self.n_classes)

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized direct TabM-mini residual archive')
        config, a = dict(payload['network_config']), payload['architecture']
        model = cls(payload['seed'], config['context_continuous'], a.get('k'), a.get('scaling_init'), device)
        expected = network_kwargs(model.architecture)
        if model.architecture != a or config.pop('kind') != KIND or {k: config.get(k) for k in expected} != expected:
            raise ValueError('Archived network differs from the archived architecture')
        model.net = PeriodicResidualTabMNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
