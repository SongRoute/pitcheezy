"""Bounded NEXT-batch frontier arms for the direct pre-pitch head (DRAFT-CUDA-FRONTIER-20261004).

Purpose: explain the flatten-MLP direct head's performance by capacity, input
information and regularization, not by tuning.  Every arm is ONE disclosed
change from the CUDA control (MatrixModel flatten_mlp width 128, 141,834
parameters at n_context 52, n_token 38, length 6) and keeps its inputs, budget
(30 epochs, patience 5, lr 5e-4, batch 1024, uniform loss), early stop and
external calibration.  The fixed registry ARMS is the whole family; widths,
masks, dropout and weight decay are not free parameters.

  capacity   width145 / width203: plain MatrixModel('flatten_mlp', width=w), no new
             class.  flatten params = h^2 + (n_flat+12) h + 10 with h = 2w and
             n_flat = 286, so 170,530 (DirectResidual 170,634, -0.06 %) and
             285,834 (DirectPeriodic 286,738, -0.32 %).
  information MaskedFlattenModel = control network with named inputs zeroed in
             fit AND logits (same transform, saved with the archive):
               no_batter          context 11-27, the 17 SequenceContext batter
                                  archetype columns (6 standardized style rates,
                                  6 reliabilities, 5 soft memberships)
               no_batter_cluster  context 23-27, the 5 soft memberships only
                                  (Archetypes.numeric_features puts them LAST)
               no_history         every past token position marked invalid, so
                                  MatrixNetwork zeroes its channels and mask bit;
                                  the current token (action/type, zero physics)
                                  and all context are untouched
             The network and parameter count stay the control's; zeroed inputs
             only leave their first-layer weights without data gradient (recorded).
  regularization FlattenRegularizedModel = control network with nn.Dropout(p) after
             each hidden GELU and AdamW(weight_decay=wd): dropout (p .1, wd .01),
             weight_decay (p 0, wd 1e-4), dropout_weight_decay (p .1, wd 1e-4).
             Its fit is DirectResidualModel.fit (a tested mirror of
             MatrixModel.fit) with only the weight decay read from the arm;
             p 0 + wd .01 reproduces MatrixModel.fit exactly (tested).

Deferred (separate task): periodic+residual combination and TabM.

API: build_frontier(arm, seed, device, context_names=None) -> unfitted model with
the MatrixModel fit/logits/predict/save/report surface; load_frontier(arm, path,
device) -> fitted model, refusing an archive of another arm.  context_names (the
exporter's 52 ordered names, see direct_context_names) is required by the
information arms and ignored otherwise.  No pandas import: runs in the CUDA venv.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from pitchmdp.direct_residual_model import DirectResidualModel
from pitchmdp.matrix_models import MatrixModel, MatrixNetwork, _validate

from pitchmdp.direct_frontier_registry import (  # noqa: F401  torch-free registry, re-exported
    ARMS, BASE_CONTEXT_NAMES, BATTER_CLUSTER, BATTER_RELIABILITY, BATTER_STYLE, CONTROL_WEIGHT_DECAY, CONTROL_WIDTH,
    GAME_STATE, MAX_WIDTH, N_PHYSICAL, PARAMETER_TOLERANCE, PITCHER_CLUSTER_FEATURES, STYLE_NAMES, SUPPORT_NAME,
    arm_spec, check_config, direct_context_names, flatten_parameter_count)


# ------------------------------------------------------------------ information arms

class MaskedFlattenModel(MatrixModel):
    """Control flatten_mlp with a fixed, self-describing input mask applied in fit and logits.

    The mask is idempotent (MatrixModel.fit early-stops through self.logits on
    already-masked rows) and never touches the current token or its mask bit.
    """
    FORMAT = 'direct_frontier_masked_v1'

    def __init__(self, seed=0, drop_context=(), drop_history=False, context_names=None, device=None,
                 width=CONTROL_WIDTH):
        super().__init__('flatten_mlp', seed, width, device=device)
        names = list(context_names or [])
        drop = list(drop_context)
        if drop and not names:
            raise ValueError('Dropping context columns requires the exporter context names')
        if names and (names[:len(BASE_CONTEXT_NAMES)] != list(BASE_CONTEXT_NAMES) or len(set(names)) != len(names)):
            raise ValueError('Context names differ from the frozen SequenceContext layout')
        if len(set(drop)) != len(drop) or any(names.count(c) != 1 for c in drop):
            raise ValueError('Every dropped column must name exactly one context column')
        self.drop_context, self.drop_history, self.context_names = drop, bool(drop_history), names
        self.drop_index = [names.index(c) for c in drop]

    def transform(self, arrays):
        tokens, valid, context = _validate(arrays)
        if self.context_names and context.shape[1] != len(self.context_names):
            raise ValueError('Context width differs from the archived context names')
        if self.drop_index:
            context = context.copy()
            context[:, self.drop_index] = 0.
        if self.drop_history:
            current = tokens[:, -1].copy()
            valid = valid.copy()
            valid[:, :-1] = False
            tokens = np.where(valid[..., None], tokens, 0.).astype(np.float32)
            if not np.array_equal(tokens[:, -1], current) or not valid[:, -1].all():
                raise RuntimeError('History mask altered the current token')
        return tokens, valid, context

    def masked_inputs(self, n_context, n_token, length):
        """Flattened input positions zeroed by the mask (tokens, mask bits, context)."""
        history = (length - 1) * (n_token + 1) if self.drop_history else 0
        return history + len(self.drop_index)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        super().fit(self.transform(train), labels, self.transform(calibration), calibration_labels, **kwargs)
        config = self.net.config
        zeroed = self.masked_inputs(config['n_context'], config['n_token'], config['length'])
        self.report.update(mask=self.mask_spec(), zeroed_flat_inputs=zeroed,
                           first_layer_weights_without_data_gradient=zeroed * 2 * self.width)
        return self

    def logits(self, arrays, batch_size=4096):
        return super().logits(self.transform(arrays), batch_size)

    def mask_spec(self):
        return {'drop_context': self.drop_context, 'drop_context_index': self.drop_index,
                'drop_history': self.drop_history, 'context_names': self.context_names}

    def save(self, path):
        super().save(path)
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        payload.update(format=self.FORMAT, mask=self.mask_spec())
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        if payload.get('format') != cls.FORMAT or payload.get('kind') != 'flatten_mlp':
            raise ValueError('Unrecognized masked frontier archive')
        mask = payload['mask']
        model = cls(payload['seed'], mask['drop_context'], mask['drop_history'], mask['context_names'], device,
                    payload['width'])
        if model.drop_index != mask['drop_context_index']:
            raise ValueError('Archived mask indices differ from the archived names')
        model.net = MatrixNetwork(**payload['network_config']).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


# ------------------------------------------------------------------ regularization arms

class DropoutFlattenNetwork(nn.Module):
    """MatrixNetwork flatten_mlp with Dropout(p) after each hidden GELU; same Linear init order and count."""

    def __init__(self, n_context, n_token, length, width=CONTROL_WIDTH, dropout=0., n_classes=10):
        super().__init__()
        self.config = dict(kind='regularized_flatten_mlp', n_context=n_context, n_token=n_token, length=length,
                           width=width, dropout=dropout, n_classes=n_classes)
        n_flat = length * (n_token + 1) + n_context
        self.head = nn.Sequential(nn.Linear(n_flat, width * 2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width * 2, width * 2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width * 2, n_classes))

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        return self.head(torch.cat((tokens.flatten(1), valid.float(), context), dim=1))


class FlattenRegularizedModel(DirectResidualModel):
    """DirectResidualModel surface with the dropout flatten network and a fixed AdamW weight decay."""
    KIND, FORMAT = 'regularized_flatten_mlp', 'direct_frontier_regularized_v1'

    def __init__(self, seed=0, dropout=0., weight_decay=CONTROL_WEIGHT_DECAY, device=None, width=CONTROL_WIDTH):
        if not 0 <= dropout < 1 or not weight_decay >= 0 or not 0 < width <= MAX_WIDTH:
            raise ValueError('Invalid regularization arm')
        self.architecture = {'backbone': 'flatten_mlp', 'width': width, 'dropout': float(dropout),
                             'weight_decay': float(weight_decay)}
        self.kind, self.seed, self.width, self.n_classes = self.KIND, seed, width, 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}

    def network(self, n_context, n_token, length):
        return DropoutFlattenNetwork(n_context, n_token, length, self.width, self.architecture['dropout'], self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None, sample_weight=None):
        # Copy of DirectResidualModel.fit (mirror of MatrixModel.fit); ONLY the AdamW weight decay is the arm's.
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
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate,
                                      weight_decay=self.architecture['weight_decay'])
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
        torch.save({'format': self.FORMAT, 'kind': self.KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != cls.FORMAT or payload.get('kind') != cls.KIND:
            raise ValueError('Unrecognized regularized frontier archive')
        a = payload['architecture']
        model = cls(payload['seed'], a['dropout'], a['weight_decay'], device, a['width'])
        config = dict(payload['network_config'])
        if config.pop('kind') != cls.KIND or (config['width'], config['dropout']) != (a['width'], a['dropout']):
            raise ValueError('Archived network differs from the archived architecture')
        model.net = DropoutFlattenNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


# ------------------------------------------------------------------ factory

def build_frontier(arm, seed, device, context_names=None):
    """Unfitted model for a registered arm; context_names is required for information arms."""
    spec = arm_spec(arm)
    if spec['model'] == 'flatten_mlp':
        return MatrixModel('flatten_mlp', seed=seed, width=spec['width'], device=device)
    if spec['model'] == 'masked_flatten_mlp':
        if not context_names:
            raise ValueError('Information arms need the exporter context names')
        return MaskedFlattenModel(seed, spec['drop_context'], spec['drop_history'], context_names, device)
    return FlattenRegularizedModel(seed, spec['dropout'], spec['weight_decay'], device)


def load_frontier(arm, path, device):
    """Fitted model of a registered arm; refuses an archive whose kind/width/mask/regularization differs."""
    spec = arm_spec(arm)
    if spec['model'] == 'flatten_mlp':
        model = MatrixModel.load(path, device)
        if model.kind != 'flatten_mlp' or model.width != spec['width']:
            raise ValueError('Archive is not this capacity arm')
    elif spec['model'] == 'masked_flatten_mlp':
        model = MaskedFlattenModel.load(path, device)
        if model.drop_context != spec['drop_context'] or model.drop_history != spec['drop_history']:
            raise ValueError('Archive is not this information arm')
    else:
        model = FlattenRegularizedModel.load(path, device)
        if (model.architecture['dropout'], model.architecture['weight_decay']) != (spec['dropout'], spec['weight_decay']):
            raise ValueError('Archive is not this regularization arm')
    return model
