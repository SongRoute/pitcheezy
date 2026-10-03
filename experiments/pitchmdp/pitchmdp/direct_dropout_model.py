"""Head-dropout variants of the frozen residual / periodic direct heads (DRAFT, see direct_dropout_registry).

Additive only: direct_residual_model.py and direct_periodic_model.py are NOT edited.
Each network subclasses the frozen one, lets its __init__ build every module in the
original order (identical initialization and parameter count), then re-wraps the
SAME Linear / norm / activation modules with nn.Dropout(head_dropout) after each
hidden head activation (see the registry for placement).  Training is the inherited
DirectResidualModel.fit (tested mirror of MatrixModel.fit, AdamW weight decay .01);
the periodic variant keeps DirectPeriodicModel's zero-current-physics guard.
head_dropout 0 reproduces the base network's logits and training exactly (tested).
Own archive formats; load refuses another kind, frozen architecture or dropout.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_residual_model as residual
from pitchmdp.direct_dropout_registry import ARMS


class ResidualHeadDropoutNetwork(residual.ResidualMLPNetwork):
    def __init__(self, n_context, n_token, length, width=128, hidden=256, blocks=2, dropout=.1, head_dropout=0.,
                 n_classes=10):
        super().__init__(n_context, n_token, length, width, hidden, blocks, dropout, n_classes)
        norm, act, out = self.output
        self.output = nn.Sequential(norm, act, nn.Dropout(head_dropout), out)
        self.config.update(kind=ResidualHeadDropoutModel.KIND, head_dropout=float(head_dropout))


class PeriodicHeadDropoutNetwork(periodic.PeriodicDirectNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, width=128, n_frequencies=16, d_embedding=8,
                 sigma=.01, head_dropout=0., n_classes=10):
        super().__init__(n_context, n_token, length, context_continuous, width, n_frequencies, d_embedding, sigma,
                         n_classes)
        first, act1, second, act2, out = self.head
        self.head = nn.Sequential(first, act1, nn.Dropout(head_dropout), second, act2, nn.Dropout(head_dropout), out)
        self.config.update(kind=PeriodicHeadDropoutModel.KIND, head_dropout=float(head_dropout))


class _HeadDropoutArchive:
    """save / load shared by both variants: own format, frozen base architecture + head dropout."""
    KIND = FORMAT = BASE = NETWORK = None

    def _check(self, head_dropout):
        if not 0 <= head_dropout < 1:
            raise ValueError('Invalid head dropout')
        self.kind = self.KIND
        self.architecture = {**self.BASE.ARCHITECTURE, 'head_dropout': float(head_dropout)}

    def save(self, path):
        torch.save({'format': self.FORMAT, 'kind': self.KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != cls.FORMAT or payload.get('kind') != cls.KIND:
            raise ValueError('Unrecognized head-dropout archive')
        config = dict(payload['network_config'])
        model = cls._from_archive(payload, config, device)
        if model.architecture != payload['architecture'] or config.pop('kind') != cls.KIND \
                or config['head_dropout'] != model.architecture['head_dropout'] \
                or any(config[k] != v for k, v in cls.BASE.ARCHITECTURE.items() if k in config):
            raise ValueError('Archived network differs from the archived architecture')
        model.net = cls.NETWORK(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


class ResidualHeadDropoutModel(_HeadDropoutArchive, residual.DirectResidualModel):
    KIND, FORMAT, BASE, NETWORK = 'residual_head_dropout_mlp', 'direct_dropout_residual_v1', residual, \
        ResidualHeadDropoutNetwork

    def __init__(self, seed=0, head_dropout=0., device=None):
        residual.DirectResidualModel.__init__(self, seed, None, device)
        self._check(head_dropout)

    def network(self, n_context, n_token, length):
        a = self.architecture
        return ResidualHeadDropoutNetwork(n_context, n_token, length, a['width'], a['hidden'], a['blocks'],
                                          a['dropout'], a['head_dropout'], self.n_classes)

    @classmethod
    def _from_archive(cls, payload, config, device):
        return cls(payload['seed'], payload['architecture']['head_dropout'], device)


class PeriodicHeadDropoutModel(_HeadDropoutArchive, periodic.DirectPeriodicModel):
    KIND, FORMAT, BASE, NETWORK = 'periodic_head_dropout_mlp', 'direct_dropout_periodic_v1', periodic, \
        PeriodicHeadDropoutNetwork

    def __init__(self, seed=0, context_continuous=(), head_dropout=0., device=None):
        periodic.DirectPeriodicModel.__init__(self, seed, context_continuous, None, device)
        self._check(head_dropout)

    def network(self, n_context, n_token, length):
        a = self.architecture
        return PeriodicHeadDropoutNetwork(n_context, n_token, length, self.context_continuous, a['width'],
                                          a['n_frequencies'], a['d_embedding'], a['sigma'], a['head_dropout'],
                                          self.n_classes)

    @classmethod
    def _from_archive(cls, payload, config, device):
        return cls(payload['seed'], config['context_continuous'], payload['architecture']['head_dropout'], device)


MODELS = {'residual_head_dropout_mlp': ResidualHeadDropoutModel, 'periodic_head_dropout_mlp': PeriodicHeadDropoutModel}


def build_dropout(arm, seed, device):
    """Unfitted model for a checked dropout-head arm (config dict)."""
    spec = ARMS[arm['name']]
    if spec['model'] == 'residual_head_dropout_mlp':
        return ResidualHeadDropoutModel(seed, spec['head_dropout'], device)
    return PeriodicHeadDropoutModel(seed, spec['context_continuous'], spec['head_dropout'], device)


def load_dropout(arm, path, device):
    """Fitted model of a registered dropout-head arm; refuses another kind, dropout or continuous map."""
    spec = ARMS[arm['name']]
    model = MODELS[spec['model']].load(path, device)
    if model.architecture['head_dropout'] != spec['head_dropout'] or model.net.config['n_context'] != 52:
        raise ValueError('Archive is not this dropout arm')
    if 'context_continuous' in spec and model.context_continuous != spec['context_continuous']:
        raise ValueError('Archive continuous map differs from the arm')
    return model
