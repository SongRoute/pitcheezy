"""Lazy, capacity-matched H5-only and long-batter-history MLP adaptations.

The H5 trunk and max128 second-stream network are identical for all arms. The
0/32/128 distinction changes only observed history masks. No full training
sequence tensor or full-pool 400-draw tensor is allocated on an accelerator.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
import torch
from torch import nn

from .matrix_long_history import LazyPitchBatch


class DualStreamNetwork(nn.Module):
    def __init__(self, n_token, n_context, width=128, h5_length=6, max_history=128):
        super().__init__()
        self.config = dict(n_token=n_token, n_context=n_context, width=width,
                           h5_length=h5_length, max_history=max_history)
        self.h5_path = nn.Sequential(nn.Linear((n_token + 1) * h5_length, width * 2), nn.GELU(),
                                     nn.Linear(width * 2, width * 2), nn.GELU())
        positions = torch.arange(max_history, dtype=torch.float32)[:, None]
        divisors = torch.exp(torch.arange(0, 8, 2, dtype=torch.float32) * (-np.log(10000.) / 8))
        codes = torch.zeros(max_history, 8)
        codes[:, 0::2], codes[:, 1::2] = torch.sin(positions * divisors), torch.cos(positions * divisors)
        self.register_buffer('position_codes', codes)
        self.long_path = nn.Sequential(nn.Linear(n_token + 8, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(width * 3 + n_context, 64), nn.GELU(), nn.Linear(64, 10))

    def forward(self, h5, h5_valid, long, long_valid, context):
        h5 = h5.masked_fill(~h5_valid[..., None], 0.)
        long = long.masked_fill(~long_valid[..., None], 0.)
        short = self.h5_path(torch.cat((h5.flatten(1), h5_valid.float()), dim=1))
        codes = self.position_codes[None].expand(len(long), -1, -1)
        encoded = self.long_path(torch.cat((long, codes), dim=-1))
        encoded = encoded.masked_fill(~long_valid[..., None], 0.)
        pooled = encoded.sum(1) / long_valid.sum(1, keepdim=True).clamp(min=1)
        return self.head(torch.cat((short, pooled, context), dim=1))


def _tensors(batch, device):
    arrays = batch.gather()
    h5, h5_valid, long, long_valid, context = arrays
    if (h5.ndim != 3 or h5.shape[1] != 6 or h5_valid.shape != h5.shape[:2] or
            long.shape != (len(h5), 128, h5.shape[2]) or long_valid.shape != long.shape[:2] or
            h5_valid.dtype != np.bool_ or long_valid.dtype != np.bool_ or not h5_valid[:, -1].all()):
        raise ValueError('Invalid fixed-capacity dual-stream input shapes/masks')
    if any(not np.isfinite(a).all() for a in (h5, long, context)) or np.any(h5[:, -1, -11:] != 0):
        raise ValueError('Invalid dual-stream values or exposed current outcome')
    return [torch.as_tensor(value, device=device) for value in arrays]


def _feature_contract(report):
    # Storage size is diagnostic; a new source frame may have more observed rows.
    return {key: value for key, value in report.items() if key != 'link_storage_bytes'}


class LazyMatrixModel:
    def __init__(self, seed=0, width=128, device=None):
        self.kind = 'dual_stream_mlp'
        self.seed, self.width = seed, width
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.

    def fit(self, train, labels, earlystop, earlystop_labels, *, epochs=30, patience=5,
            batch_size=256, learning_rate=.0005, sample_weight=None, checkpoint=None):
        if not isinstance(train, LazyPitchBatch) or not isinstance(earlystop, LazyPitchBatch):
            raise ValueError('LazyPitchBatch inputs are required; full sequence arrays are not accepted')
        y, ey = np.asarray(labels), np.asarray(earlystop_labels)
        for batch, values in ((train, y), (earlystop, ey)):
            if (not len(batch) or values.shape != (len(batch),) or not np.issubdtype(values.dtype, np.integer)
                    or not np.isin(values, range(10)).all()):
                raise ValueError('Aligned nonempty ten-class integer labels are required')
        if train.store.report()['h5'] != earlystop.store.report()['h5'] or train.store.long_length != earlystop.store.long_length:
            raise ValueError('Training and early-stop feature contracts must match')
        if epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError('Positive training budgets required')
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weight.shape != y.shape or not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError('Invalid sample weights')
        weight = (weight / weight.mean()).astype(np.float32)
        self.temperature = self.delivery_temperature = 1.
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        one = train.subset(np.array([0])).gather()
        self.feature_report = train.store.report()
        self.net = DualStreamNetwork(one[0].shape[-1], one[-1].shape[-1], self.width).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=.01)
        rng = np.random.default_rng(self.seed)
        best, best_state, best_epoch, stale, updates = np.inf, None, 0, 0, 0
        history, start = [], time.perf_counter()
        max_expanded = 0
        for epoch in range(epochs):
            self.net.train()
            total = 0.
            order = rng.permutation(len(y))
            for begin in range(0, len(y), batch_size):
                indices = order[begin:begin + batch_size]
                tensors = _tensors(train.subset(indices), self.device)
                max_expanded = max(max_expanded, len(indices))
                target = torch.as_tensor(y[indices].astype(np.int64), device=self.device)
                batch_weight = torch.as_tensor(weight[indices], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                losses = nn.functional.cross_entropy(self.net(*tensors), target, reduction='none')
                loss = (losses * batch_weight).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total += float(loss.item()) * len(indices)
                updates += 1
            logits = self.logits(earlystop, batch_size=batch_size)
            score = float((logsumexp(logits, axis=1) - logits[np.arange(len(ey)), ey]).mean())
            if not np.isfinite(score):
                raise ValueError('Nonfinite early-stop NLL')
            history.append({'epoch': epoch + 1, 'train_weighted_nll': total / len(y), 'earlystop_conditional_nll': score})
            if score < best - 1e-5:
                best, best_epoch, stale = score, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        self.net.load_state_dict(best_state)
        self.report = {'kind': self.kind, 'seed': self.seed, 'device': self.device,
                       'network': self.net.config, 'features': self.feature_report,
                       'parameter_count': sum(p.numel() for p in self.net.parameters()),
                       'training_rows': len(y), 'earlystop_rows': len(ey), 'epochs_run': len(history),
                       'best_epoch': best_epoch, 'optimizer_updates': updates, 'history': history,
                       'batch_size': batch_size, 'max_expanded_training_rows': max_expanded,
                       'weight_ess': float(weight.sum(dtype=np.float64) ** 2 / np.square(weight.astype(np.float64)).sum()),
                       'seconds': time.perf_counter() - start,
                       'adaptation': 'capacity-matched dual-stream MLP: H5 flatten plus nonlinear token encoder, fixed positional codes, masked mean; not a paper reproduction',
                       'calibration': 'external temperature CAL then ensemble/blend CAL; no DEV fitting'}
        if checkpoint:
            self.save(checkpoint)
        return self

    def logits(self, batch, batch_size=256):
        if not isinstance(batch, LazyPitchBatch) or batch_size < 1:
            raise ValueError('Lazy batch and positive inference batch size required')
        if hasattr(self, 'feature_report') and _feature_contract(batch.store.report()) != _feature_contract(self.feature_report):
            raise ValueError('Prediction stream differs from fitted feature contract')
        self.net.eval()
        result = []
        with torch.no_grad():
            for begin in range(0, len(batch), batch_size):
                tensors = _tensors(batch.subset(slice(begin, begin + batch_size)), self.device)
                result.append(self.net(*tensors).cpu().numpy())
        return np.concatenate(result) if result else np.empty((0, 10), dtype=np.float32)

    def predict(self, batch, batch_size=256):
        return softmax(self.logits(batch, batch_size) / self.temperature, axis=1)

    def save(self, path):
        torch.save({'format': 'lazy_matrix_model_v1', 'seed': self.seed, 'width': self.width,
                    'network': self.net.config, 'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()},
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'feature_report': self.feature_report, 'report': self.report}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        if payload.get('format') != 'lazy_matrix_model_v1':
            raise ValueError('Unknown lazy model archive')
        model = cls(payload['seed'], payload['width'], device)
        model.net = DualStreamNetwork(**payload['network']).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.feature_report, model.report = payload['feature_report'], payload['report']
        return model


class LazyJointDelivery:
    """Apply a frozen JointDelivery pool to lazy dual-stream models.

    Current physical replacement and candidate type are kept together. Logits are
    expanded for a bounded pitch block and model evaluation gathers <=batch_size
    histories at a time. Calibration caches only float32 logits, not histories.
    """
    def __init__(self, delivery, pitch_chunk=16, model_batch_size=256):
        if pitch_chunk < 1 or model_batch_size < 1:
            raise ValueError('Positive inference chunk sizes required')
        self.delivery, self.draws = delivery, delivery.draws
        self.pitch_chunk, self.model_batch_size = pitch_chunk, model_batch_size

    def logits(self, model, batch):
        pieces, levels = [], []
        for begin in range(0, len(batch), self.pitch_chunk):
            selected = batch.subset(slice(begin, begin + self.pitch_chunk))
            samples, tier = self.delivery.sample(selected.frame())
            repeated = LazyPitchBatch(batch.store, batch.context, np.repeat(selected.rows, self.draws),
                           current=samples.reshape(-1, 8),
                           candidate_pitch_types=None if selected.candidate_pitch_types is None else
                           np.repeat(selected.candidate_pitch_types, self.draws))
            scores = model.logits(repeated, batch_size=self.model_batch_size).reshape(len(selected), self.draws, 10)
            pieces.append(scores)
            levels.append(tier)
        return (np.concatenate(pieces), np.concatenate(levels)) if pieces else (
                np.empty((0, self.draws, 10), dtype=np.float32), np.empty(0, dtype=np.int64))

    def predict(self, model, batch):
        calibrated, raw, levels = [], [], []
        for begin in range(0, len(batch), self.pitch_chunk):
            logits, tier = self.logits(model, batch.subset(slice(begin, begin + self.pitch_chunk)))
            z = logits.astype(np.float64)
            calibrated.append(softmax(z / model.delivery_temperature, axis=-1).mean(1))
            raw.append(softmax(z, axis=-1).mean(1))
            levels.append(tier)
        if not calibrated:
            return np.empty((0, 10)), np.empty((0, 10)), np.empty(0, dtype=np.int64)
        p, r = np.concatenate(calibrated), np.concatenate(raw)
        for values in (p, r):
            if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1., rtol=0, atol=1e-6):
                raise ValueError('Invalid integrated probability mass')
        return p, r, np.concatenate(levels)

    def calibrate(self, model, batch, labels):
        y = np.asarray(labels)
        if y.shape != (len(batch),) or not len(y) or not np.issubdtype(y.dtype, np.integer) or not np.isin(y, range(10)).all():
            raise ValueError('Aligned temperature-CAL integer labels required')
        logits, _ = self.logits(model, batch)
        # Preserve existing conditional-logit float32 temperature objective;
        # final saved integrated probabilities use float64 as in ML1/ML2.
        def loss(temperature):
            p = softmax(logits / temperature, axis=-1).mean(1)
            return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        fit = minimize_scalar(loss, bounds=(.5, 2.5), method='bounded')
        model.delivery_temperature = float(fit.x)
        model.report.update(delivery_temperature=float(fit.x), delivery_calibration_rows=len(y),
                            calibration_integrated_log_loss=float(fit.fun),
                            calibration_numeric_contract='float32 temperature objective; float64 final integration')
        return model
