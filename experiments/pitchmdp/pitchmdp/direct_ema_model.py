"""Weight-averaging (EMA of weights) arms for the direct pre-pitch head (DRAFT-CUDA-EMA-20261004).

Additive module; frozen models are not edited.  FlattenEMAModel is the frontier
FlattenRegularizedModel (flatten_mlp network with Dropout(p) after each hidden GELU,
AdamW with the arm's weight decay) plus ONE change: a shadow copy of the weights is
updated after every optimizer step,

  ema <- d_t * ema + (1 - d_t) * w,   d_t = min(decay, (1 + t) / (10 + t)),

and the shadow weights (not the raw ones) give the per-epoch early-stop NLL, the
best-epoch state and the saved / predicted model.  The raw early-stop NLL is still
recorded per epoch for diagnosis.  The update consumes no randomness, so the raw
trajectory equals the base arm's for the same seed (tested); stopping can differ
because it follows the EMA score.  The fit loop is a copy of
FlattenRegularizedModel.fit (itself a mirror of MatrixModel.fit); the EMA lines are
the only additions.  Calibration stays external (JointDelivery).

API: build_ema(arm, seed, device) -> unfitted model; load_ema(arm, path, device) ->
fitted model, refusing an archive of another arm.  No pandas import.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn

from pitchmdp.direct_ema_registry import (  # noqa: F401  torch-free registry, re-exported
    ARM_ORDER, ARMS, CONTRASTS, EMA_DECAY, arm_spec, check_arm, check_contrasts, check_family, ema_decay_at)
from pitchmdp.direct_frontier_model import DropoutFlattenNetwork, FlattenRegularizedModel
from pitchmdp.direct_frontier_registry import CONTROL_WEIGHT_DECAY, CONTROL_WIDTH
from pitchmdp.matrix_models import _validate


@torch.no_grad()
def update_ema(ema_params, params, step, decay):
    """In-place ema <- d * ema + (1 - d) * w with the warm-up decay of `step`; returns d."""
    d = ema_decay_at(decay, step)
    for e, w in zip(ema_params, params, strict=True):
        e.lerp_(w.detach(), 1. - d)
    return d


def replay_stop_rule(scores, patience):
    """(best_epoch, stop_epoch or None) of the shared early-stop rule (min delta 1e-5) replayed on a score list.

    Review F2: applied to the raw-weight scores of an EMA run, this is the epoch the control rule would pick,
    truncated at the raw stop; stop_epoch is None when the EMA run ended before the raw rule stopped.
    """
    best, best_epoch, stale = np.inf, 0, 0
    for epoch, score in enumerate(scores, 1):
        if score < best - 1e-5:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            return best_epoch, epoch
    return best_epoch, None


class FlattenEMAModel(FlattenRegularizedModel):
    """FlattenRegularizedModel with EMA weights for early stopping, the best state and the saved model."""
    KIND, FORMAT = 'ema_flatten_mlp', 'direct_ema_flatten_v1'

    def __init__(self, seed=0, dropout=0., ema_decay=EMA_DECAY, device=None, width=CONTROL_WIDTH,
                 weight_decay=CONTROL_WEIGHT_DECAY):
        super().__init__(seed, dropout, weight_decay, device, width)
        if not 0 < ema_decay < 1:
            raise ValueError('EMA decay must lie in (0, 1)')
        self.kind = self.KIND
        self.architecture['ema_decay'] = float(ema_decay)

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None, sample_weight=None):
        # Copy of FlattenRegularizedModel.fit; additions are marked EMA.
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
        # EMA: shadow network; the flatten head has no buffers, so parameters are the whole state.
        if any(True for _ in self.net.buffers()):
            raise RuntimeError('EMA arm expects a network without buffers')
        ema_net = copy.deepcopy(self.net).requires_grad_(False)
        ema_params, raw_params = list(ema_net.parameters()), list(self.net.parameters())
        decay, last_decay = self.architecture['ema_decay'], None
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
                last_decay = update_ema(ema_params, raw_params, updates, decay)  # EMA
                total += float(loss.item()) * len(ids)
                updates += 1
            raw_logits = self.logits(calibration)  # EMA: diagnostic raw-weight score (eval mode)
            raw_score = float((logsumexp(raw_logits, axis=1) - raw_logits[np.arange(len(ey)), ey]).mean())
            self.net, ema_net = ema_net, self.net  # EMA: score the shadow weights
            try:
                logits = self.logits(calibration)
            finally:
                self.net, ema_net = ema_net, self.net
            score = float((logsumexp(logits, axis=1) - logits[np.arange(len(ey)), ey]).mean())
            if not np.isfinite(score) or not np.isfinite(raw_score):
                raise ValueError('Nonfinite early-stop loss')
            history.append({'epoch': epoch + 1, 'train_weighted_nll': total / len(y), 'earlystop_conditional_nll': score,
                            'raw_earlystop_conditional_nll': raw_score, 'ema_decay_last': last_decay})
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in ema_net.state_dict().items()}  # EMA
            else:
                stale += 1
            if stale >= patience:
                break
        self.net.load_state_dict(best_state)  # EMA: the model is the best-epoch shadow weights
        raw_best = min(history, key=lambda h: h['raw_earlystop_conditional_nll'])
        rbest_epoch, rstop = replay_stop_rule([h['raw_earlystop_conditional_nll'] for h in history], patience)  # F2
        self.report = self._base_report(len(y), len(ey), weight)
        self.report.update(network=self.net.config, parameter_count=sum(p.numel() for p in self.net.parameters()),
                           best_epoch=best_epoch, epochs_run=len(history), optimizer_updates=updates,
                           history=history, seconds=time.perf_counter() - start, architecture=self.architecture,
                           weights='ema', ema_updates=updates,
                           raw_best_epoch_within_run=raw_best['epoch'],
                           raw_best_earlystop_conditional_nll_within_run=raw_best['raw_earlystop_conditional_nll'],
                           raw_rule_best_epoch=rbest_epoch, raw_rule_stop_epoch=rstop)  # None: raw rule not reached
        if checkpoint:
            self.save(checkpoint)
        return self

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != cls.FORMAT or payload.get('kind') != cls.KIND:
            raise ValueError('Unrecognized EMA flatten archive')
        a = payload['architecture']
        model = cls(payload['seed'], a['dropout'], a['ema_decay'], device, a['width'], a['weight_decay'])
        config = dict(payload['network_config'])
        if config.pop('kind') != 'regularized_flatten_mlp' or (config['width'], config['dropout']) != (a['width'], a['dropout']):
            raise ValueError('Archived network differs from the archived architecture')
        model.net = DropoutFlattenNetwork(**config).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


def build_ema(arm, seed, device):
    """Unfitted model of a registered EMA arm (control / dropout come from their own frozen factories)."""
    spec = arm_spec(arm)
    return FlattenEMAModel(seed, spec['dropout'], spec['ema_decay'], device, spec['width'], spec['weight_decay'])


def load_ema(arm, path, device):
    """Fitted model of a registered EMA arm; refuses an archive whose regularization or decay differs."""
    spec = arm_spec(arm)
    model = FlattenEMAModel.load(path, device)
    a = model.architecture
    if (a['width'], a['dropout'], a['weight_decay'], a['ema_decay']) != \
            (spec['width'], spec['dropout'], spec['weight_decay'], spec['ema_decay']):
        raise ValueError('Archive is not this EMA arm')
    return model
