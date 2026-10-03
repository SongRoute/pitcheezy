"""ResNet-style backbone for the direct pre-pitch head (research frontier E1).

Adaptation of the tabular ResNet of Gorishniy et al. 2021, "Revisiting Deep
Learning Models for Tabular Data" (https://arxiv.org/abs/2106.11959; official
code https://github.com/yandex-research/rtdl-revisiting-models).  It is NOT a
faithful replication: no per-feature preprocessing, no HPO, our ten-class head
and our flattened input contract.  Exact network (fixed, registered in config):

  x = concat(flatten(masked tokens), mask, context)          # same as flatten_mlp
  z = Linear(n_flat, 128)
  2 x  z = z + Linear(256, 128)(Dropout(.1)(GELU(Linear(128, 256)(LayerNorm(z)))))
  logits = Linear(128, 10)(GELU(LayerNorm(z)))

Differences from the paper block: GELU instead of ReLU (kept equal to the
flatten_mlp control so activation is not a second factor), one dropout of .1
after the hidden activation and no residual-branch dropout.

The training procedure is a line-for-line mirror of MatrixModel.fit (torch
branch): same seeding, AdamW(lr, weight_decay .01), shuffled mini-batches,
per-row weighted cross entropy, grad-norm clip 5, unweighted early-stop NLL
with min delta 1e-5 and patience, best-state restore.  MatrixModel.fit builds
MatrixNetwork inline, so the only difference here is the network factory;
tests check that this loop with the flatten_mlp network reproduces
MatrixModel.fit exactly.  Calibration stays external (JointDelivery).
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from pitchmdp.matrix_models import MatrixModel, _validate

KIND = 'residual_mlp'
FORMAT = 'direct_residual_model_v1'
ARCHITECTURE = {'width': 128, 'hidden': 256, 'blocks': 2, 'dropout': .1,
                'normalization': 'layernorm', 'activation': 'gelu'}


class ResidualMLPNetwork(nn.Module):
    def __init__(self, n_context, n_token, length, width=128, hidden=256, blocks=2, dropout=.1, n_classes=10):
        super().__init__()
        self.config = dict(kind=KIND, n_context=n_context, n_token=n_token, length=length, width=width,
                           hidden=hidden, blocks=blocks, dropout=dropout, n_classes=n_classes)
        self.input = nn.Linear(length * (n_token + 1) + n_context, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                                  nn.Dropout(dropout), nn.Linear(hidden, width))
                                    for _ in range(blocks))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, n_classes))

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        z = self.input(torch.cat((tokens.flatten(1), valid.float(), context), dim=1))
        for block in self.blocks:
            z = z + block(z)
        return self.output(z)


def parameter_count(n_context, n_token, length, architecture=ARCHITECTURE):
    net = ResidualMLPNetwork(n_context, n_token, length, *(architecture[k] for k in ('width', 'hidden', 'blocks', 'dropout')))
    return sum(p.numel() for p in net.parameters())


class DirectResidualModel(MatrixModel):
    """Same predict/logits/report as MatrixModel; own network, fit mirror and archive format."""

    def __init__(self, seed=0, architecture=None, device=None):
        self.architecture = dict(ARCHITECTURE if architecture is None else architecture)
        if self.architecture != ARCHITECTURE:
            raise ValueError('Residual architecture differs from the fixed registered network')
        self.kind, self.seed, self.width, self.n_classes = KIND, seed, self.architecture['width'], 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}

    def network(self, n_context, n_token, length):
        a = self.architecture
        return ResidualMLPNetwork(n_context, n_token, length, a['width'], a['hidden'], a['blocks'], a['dropout'], self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None, sample_weight=None):
        # Mirror of MatrixModel.fit (torch branch); keep in sync and covered by the equivalence test.
        train, calibration = _validate(train), _validate(calibration)
        y, ey = np.asarray(labels, dtype=np.int64), np.asarray(calibration_labels, dtype=np.int64)
        if len(y) != len(train[0]) or len(ey) != len(calibration[0]) or not len(y) or not len(ey):
            raise ValueError('Nonempty aligned training and early-stop labels required')
        if not np.isin(y, range(10)).all() or not np.isin(ey, range(10)).all():
            raise ValueError('Labels must follow the ten-class contract')
        if train[0].shape[1:] != calibration[0].shape[1:] or train[2].shape[1:] != calibration[2].shape[1:]:
            raise ValueError('Training and early-stop input contracts differ')
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError('Positive training budgets required')
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weight.shape != y.shape or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError('Invalid training sample weights')
        weight = weight / weight.mean()
        self.temperature = self.delivery_temperature = 1.
        start = time.perf_counter()
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.net = self.network(train[2].shape[1], train[0].shape[2], train[0].shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=.01)
        tensors = [torch.as_tensor(a, device=self.device) for a in (*train, y, weight.astype(np.float32))]
        rng = np.random.default_rng(self.seed)
        best, best_state, best_epoch, stale, updates = np.inf, None, 0, 0, 0
        history = []
        for epoch in range(epochs):
            self.net.train()
            total = 0.
            for ids in np.array_split(rng.permutation(len(y)), np.arange(batch_size, len(y), batch_size)):
                ix = torch.as_tensor(ids, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                losses = nn.functional.cross_entropy(self.net(*(a[ix] for a in tensors[:3])), tensors[3][ix], reduction='none')
                loss = (losses * tensors[4][ix]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total += float(loss.item()) * len(ids)
                updates += 1
            logits = self.logits(calibration)  # eval mode: dropout off
            score = float((logsumexp(logits, axis=1) - logits[np.arange(len(ey)), ey]).mean())
            if not np.isfinite(score):
                raise ValueError('Nonfinite early-stop loss')
            history.append({'epoch': epoch + 1, 'train_weighted_nll': total / len(y), 'earlystop_conditional_nll': score})
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        self.net.load_state_dict(best_state)
        self.report = self._base_report(len(y), len(ey), weight)
        self.report.update(network=self.net.config, parameter_count=sum(p.numel() for p in self.net.parameters()),
                           best_epoch=best_epoch, epochs_run=len(history), optimizer_updates=updates,
                           history=history, seconds=time.perf_counter() - start, architecture=self.architecture)
        if checkpoint:
            self.save(checkpoint)
        return self

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized direct residual model archive')
        model = cls(payload['seed'], payload['architecture'], device)
        config = dict(payload['network_config'])
        if config.pop('kind') != KIND or {k: config[k] for k in ('width', 'hidden', 'blocks', 'dropout')} != \
                {k: model.architecture[k] for k in ('width', 'hidden', 'blocks', 'dropout')}:
            raise ValueError('Archived network differs from the archived architecture')
        model.net = ResidualMLPNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
