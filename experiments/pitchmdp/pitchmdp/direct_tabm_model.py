"""TabM-mini shared-weight ensemble head on the frozen periodic direct network (research frontier A4, DRAFT).

Adaptation of TabM-mini from Gorishniy, Kotelnikov, Babenko, "TabM: Advancing
Tabular Deep Learning with Parameter-Efficient Ensembling" (ICLR 2025,
https://arxiv.org/abs/2410.24210; official package https://github.com/yandex-research/tabm,
Apache-2.0, ``tabm.py`` read at commit 28e47ae on 2026-10-04).  Written here from
the package's documented structure; no third-party code is copied:

  * MLPBackboneMiniEnsemble: one ElementwiseAffine (k x d_in scaling, no bias)
    in front of a fully shared MLP; the k members differ only in that scaling;
  * scaling init 'normal' when numerical embeddings precede the backbone, with one
    random scalar per feature chunk (``start_scaling_init_chunks = d_features``);
  * LinearEnsemble output: k separate output layers, weights and biases
    U(-1/sqrt(d_in), 1/sqrt(d_in));
  * training optimizes the MEAN of the k member losses (README: "the mean loss
    must be optimized, not the loss of the mean prediction");
  * inference averages member PROBABILITIES.

NOT a faithful reproduction: k=8 (package default 32), our frozen periodic PLR-lite
embedding, flatten_mlp head width and GELU (package default ReLU, d_block 512),
our AdamW/early-stop loop and ten-class head, no tuning, same mini-batch for all
members (the package's default "conventional MLP-like" input mode).

Exact network (fixed; the periodic embedding and shared head Linear layers are the
frozen PeriodicHeadDropoutNetwork modules, built in the original order):

  x      = cat(raw, periodic_embed(...).flatten(1))                 # [N, D] as periodic
  x_k    = x[:, None, :] * r                                        # r [k, D], no bias
  h      = Dropout(p)(GELU(Linear(D, 256)(x_k)))                    # shared across members
  h      = Dropout(p)(GELU(Linear(256, 256)(h)))                    # shared across members
  member = einsum('nkd,kdc->nkc', h, W) + b                         # W [k, 256, 10], b [k, 10]
  forward returns member logits [N, k, 10]

DirectTabMModel.logits returns log(mean_k softmax(member)), a valid logit vector, so
early stopping (ensemble conditional NLL, as TabM monitors the ensemble) and the
unchanged external temperature/delivery calibration see one ten-class prediction.
fit is a line-for-line mirror of DirectResidualModel.fit except that the per-row loss
is the mean over members of the member cross entropy (tested: with a k=1 wrapper
around the frozen periodic network it reproduces PeriodicHeadDropoutModel.fit
bitwise).  Frozen modules are not edited.  Own archive format; load refuses another
kind, k, dropout, scaling init or base architecture.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutNetwork
from pitchmdp.matrix_models import _validate

KIND = 'periodic_tabm_mini_mlp'
FORMAT = 'direct_tabm_mini_periodic_v1'
ENSEMBLE = {'ensemble': 'tabm-mini', 'k': 8, 'scaling_init': 'normal', 'scaling_chunks': 'per_feature',
            'scaling_bias': False, 'output': 'per_member_linear', 'train_loss': 'mean_member_ce',
            'prediction': 'mean_member_probability'}
SCALING_INITS = ('normal', 'random-signs')


def feature_chunks(n_raw, n_embedded, d_embedding):
    """Backbone-input chunk sizes: one per raw scalar, d_embedding per embedded scalar (input order)."""
    return [1] * n_raw + [d_embedding] * n_embedded


def init_scaling(r, chunks, distribution):
    """One random value per (member, chunk), repeated across the chunk (official init_scaling_ semantics)."""
    if sum(chunks) != r.shape[-1]:
        raise ValueError('Scaling chunks do not cover the backbone input')
    with torch.no_grad():
        start = 0
        for size in chunks:
            value = torch.empty(r.shape[0], 1)
            if distribution == 'normal':
                nn.init.normal_(value)
            elif distribution == 'random-signs':
                value.bernoulli_(.5).mul_(2).add_(-1)
            else:
                raise ValueError('Unknown scaling initialization')
            r[:, start:start + size] = value
            start += size
    return r


class PeriodicTabMNetwork(PeriodicHeadDropoutNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, width=128, n_frequencies=16, d_embedding=8,
                 sigma=.01, head_dropout=0., k=8, scaling_init='normal', n_classes=10):
        super().__init__(n_context, n_token, length, context_continuous, width, n_frequencies, d_embedding, sigma,
                         head_dropout, n_classes)
        if k < 1 or scaling_init not in SCALING_INITS:
            raise ValueError('Invalid TabM-mini ensemble settings')
        first, act1, drop1, second, act2, drop2, out = self.head
        self.head = nn.Sequential(first, act1, drop1, second, act2, drop2)  # shared; frozen modules reused
        d_in, d_hidden = first.in_features, out.in_features
        self.n_raw = d_in - self.n_embedded * d_embedding
        self.scaling = nn.Parameter(init_scaling(torch.empty(k, d_in), feature_chunks(self.n_raw, self.n_embedded,
                                                                                       d_embedding), scaling_init))
        bound = 1 / math.sqrt(d_hidden)
        self.output_weight = nn.Parameter(torch.empty(k, d_hidden, n_classes).uniform_(-bound, bound))
        self.output_bias = nn.Parameter(torch.empty(k, n_classes).uniform_(-bound, bound))
        self.config.update(kind=KIND, k=int(k), scaling_init=scaling_init)

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, periodic.N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(),
                         context[:, self.raw]), dim=1)
        x = torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1)
        h = self.head(x[:, None, :] * self.scaling)
        return torch.einsum('nkd,kdc->nkc', h, self.output_weight) + self.output_bias


def member_loss(member_logits, y):
    """Per-row mean over members of the member cross entropy ([N, k, C], [N] -> [N])."""
    n, k, c = member_logits.shape
    losses = nn.functional.cross_entropy(member_logits.reshape(n * k, c), y.repeat_interleave(k), reduction='none')
    return losses.view(n, k).mean(1)


def ensemble_logits(member_logits):
    """log of the mean member probability ([N, k, C] -> [N, C]); softmax of it is the ensemble prediction."""
    return torch.logsumexp(torch.log_softmax(member_logits, -1), dim=1) - math.log(member_logits.shape[1])


def parameter_count(n_context, n_token, length, context_continuous, k=ENSEMBLE['k']):
    net = PeriodicTabMNetwork(n_context, n_token, length, context_continuous, k=k)
    return sum(p.numel() for p in net.parameters())


class DirectTabMModel(periodic.DirectPeriodicModel):
    """Periodic direct model with a TabM-mini head: own fit mirror (mean member loss), ensemble logits, archive."""

    KIND, FORMAT = KIND, FORMAT

    def __init__(self, seed=0, context_continuous=(), head_dropout=0., k=ENSEMBLE['k'],
                 scaling_init=ENSEMBLE['scaling_init'], device=None):
        periodic.DirectPeriodicModel.__init__(self, seed, context_continuous, None, device)
        if not 0 <= head_dropout < 1 or int(k) != k or k < 1 or scaling_init not in SCALING_INITS:
            raise ValueError('Invalid TabM-mini settings')
        self.kind = KIND
        self.architecture = {**periodic.ARCHITECTURE, 'head_dropout': float(head_dropout), **ENSEMBLE,
                             'k': int(k), 'scaling_init': scaling_init}

    def network(self, n_context, n_token, length):
        a = self.architecture
        return PeriodicTabMNetwork(n_context, n_token, length, self.context_continuous, a['width'], a['n_frequencies'],
                                   a['d_embedding'], a['sigma'], a['head_dropout'], a['k'], a['scaling_init'],
                                   self.n_classes)

    def member_logits(self, arrays, batch_size=4096):
        arrays = periodic._no_current_physics(arrays)
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(arrays[0]), batch_size):
                inputs = [torch.as_tensor(a[start:start + batch_size], device=self.device) for a in arrays]
                outputs.append(self.net(*inputs).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.architecture['k'], 10), dtype=np.float32)

    def logits(self, arrays, batch_size=4096):
        member = self.member_logits(arrays, batch_size)
        if not len(member):
            return np.empty((0, 10), dtype=np.float32)
        return ensemble_logits(torch.as_tensor(member)).numpy()

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None, sample_weight=None):
        # Mirror of DirectResidualModel.fit; only the loss line differs (mean member CE).  Covered by the k=1 test.
        train, calibration = periodic._no_current_physics(train), periodic._no_current_physics(calibration)
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
                losses = member_loss(self.net(*(a[ix] for a in tensors[:3])), tensors[3][ix])
                loss = (losses * tensors[4][ix]).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total += float(loss.item()) * len(ids)
                updates += 1
            logits = self.logits(calibration)  # eval mode: dropout off; ensemble (mean-probability) logits
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
            raise ValueError('Unrecognized direct TabM-mini archive')
        config, a = dict(payload['network_config']), payload['architecture']
        model = cls(payload['seed'], config['context_continuous'], a['head_dropout'], a['k'], a['scaling_init'], device)
        if model.architecture != a or config.pop('kind') != KIND \
                or any(config[key] != model.architecture[key] for key in
                       ('width', 'n_frequencies', 'd_embedding', 'sigma', 'head_dropout', 'k', 'scaling_init')):
            raise ValueError('Archived network differs from the archived architecture')
        model.net = PeriodicTabMNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
