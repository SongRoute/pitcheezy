"""Takamido-inspired physical sequence + game/batter context predictors.

Conditional tokens include the hypothetical current delivery. Pre-pitch
probabilities must integrate that delivery, never insert its logged realization.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.optimize import minimize_scalar
import torch
from torch import nn

from .archetypes import Archetypes


class SequenceContext:
    """No player identity, physical realization, or outcome column is an input."""
    def fit(self, train):
        if not train.split.eq('train').all():
            raise ValueError('Context fitting requires training rows only')
        self.archetypes = Archetypes(n_clusters=5, seed=42).fit(train)
        return self

    def transform(self, frame):
        def col(name):
            return pd.to_numeric(frame[name], errors='coerce').fillna(0).to_numpy(np.float32)
        bases = col('bases').astype(int)
        field_sign = np.where(frame.inning_topbot.eq('Top'), 1., -1.)
        numeric = [col('balls')/3, col('strikes')/2, col('outs_when_up')/2,
                   col('inning')/9, field_sign*(col('home_score')-col('away_score'))/5,
                   frame.inning_topbot.eq('Top').to_numpy(float),
                   (bases & 1).astype(float), ((bases >> 1) & 1).astype(float), ((bases >> 2) & 1).astype(float),
                   frame.stand.eq('L').to_numpy(float), frame.p_throws.eq('L').to_numpy(float)]
        numeric += list(self.archetypes.numeric_features(frame).T)
        return np.column_stack(numeric).astype(np.float32)

    def report(self):
        return {'features': ['balls', 'strikes', 'outs', 'inning', 'defensive_score_lead', 'defender_is_home',
                             'first_base', 'second_base', 'third_base', 'batter_left', 'pitcher_left',
                             'six_batter_style_rates', 'six_style_reliabilities', 'five_soft_memberships'],
                'batter_id_input': False, 'pitcher_id_input': False, 'archetypes': self.archetypes.report()}


class SequenceNetwork(nn.Module):
    def __init__(self, kind, n_context, n_physical=8, n_classes=10, width=128, length=6):
        super().__init__()
        self.kind, self.length = kind, length
        self.config = dict(kind=kind, n_context=n_context, n_physical=n_physical, n_classes=n_classes,
                           width=width, length=length)
        self.context_path = nn.Sequential(nn.Linear(n_context, 32), nn.GELU())
        if kind == 'transformer':
            self.token_projection = nn.Linear(n_physical, width)
            self.position = nn.Parameter(torch.empty(length, width))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(width, 4, width*2, dropout=.1,
                                               activation='gelu', batch_first=True)
            self.sequence_path = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            n_sequence = width
        elif kind in ('flatten_mlp', 'current_only'):
            n_input = (n_physical+1)*length if kind == 'flatten_mlp' else n_physical
            self.sequence_path = nn.Sequential(nn.Linear(n_input, width*2), nn.GELU(),
                                               nn.Linear(width*2, width*2), nn.GELU())
            n_sequence = width*2
        else:
            raise ValueError(f'Unknown architecture {kind}')
        self.output = nn.Sequential(nn.Linear(n_sequence+32, 64), nn.GELU(), nn.Linear(64, n_classes))

    def forward(self, tokens, valid, context):
        # Mask before processing, so arbitrary padding payload is never evidence.
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        if self.kind == 'transformer':
            z = self.token_projection(tokens)+self.position
            z = self.sequence_path(z, src_key_padding_mask=~valid)
            z = (z*valid[..., None]).sum(1)/valid.sum(1, keepdim=True).clamp(min=1)
        elif self.kind == 'flatten_mlp':
            z = self.sequence_path(torch.cat([tokens.flatten(1), valid.float()], dim=1))
        else:
            z = self.sequence_path(tokens[:, -1])
        return self.output(torch.cat([z, self.context_path(context)], dim=1))


def classification_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not len(y):
        return {'n': 0}
    if p.shape[0] != len(y) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid predicted probability mass')
    result = {'n': len(y), 'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
              'brier_multiclass': float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean()),
              'accuracy': float((p.argmax(1) == y).mean())}
    confidence, correct = p.max(1), p.argmax(1) == y
    bins = np.minimum((confidence*10).astype(int), 9)
    result['top_label_ece10'] = float(sum(abs(correct[m].mean()-confidence[m].mean())*m.mean()
                                  for i in range(10) if (m := (bins == i)).any()))
    result['class_rates'] = [{'class': i, 'observed': float((y == i).mean()), 'predicted': float(p[:, i].mean())}
                             for i in range(p.shape[1])]
    if p.shape[1] == 2 and len(np.unique(y)) == 2:
        from scipy.stats import rankdata
        ranks = rankdata(p[:, 1])
        n1, n0 = int(y.sum()), int((y == 0).sum())
        result['auc'] = float((ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))
    return result


class SequenceModel:
    def __init__(self, kind='transformer', seed=42, width=128, n_classes=10):
        self.kind, self.seed, self.width, self.n_classes = kind, seed, width, n_classes
        self.temperature = 1.
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None):
        """train/calibration are (tokens, valid_mask, context), fixed across models."""
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.net = SequenceNetwork(self.kind, train[2].shape[1], train[0].shape[2], self.n_classes, self.width).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=.01)
        arrays = [torch.as_tensor(a, device=self.device) for a in [*train, np.asarray(labels, np.int64)]]
        rng = np.random.default_rng(self.seed)
        best, best_epoch, stale, history, best_state = np.inf, 0, 0, [], None
        start = time.perf_counter()
        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(len(labels))
            total = 0.
            for begin in range(0, len(labels), batch_size):
                index = torch.as_tensor(order[begin:begin+batch_size], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(self.net(*(a[index] for a in arrays[:3])), arrays[3][index])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total += float(loss.item())*len(index)
            logits = self.logits(calibration)
            score = float((logsumexp(logits, axis=1)-logits[np.arange(len(calibration_labels)), calibration_labels]).mean())
            record = {'epoch': epoch+1, 'train_log_loss': total/len(labels), 'calibration_log_loss': score,
                      'seconds': time.perf_counter()-start}
            history.append(record)
            print(self.kind, record, flush=True)
            if score < best-1e-5:
                best, best_epoch, stale = score, epoch+1, 0
                best_state = {k: v.detach().cpu().clone() for k,v in self.net.state_dict().items()}
                if checkpoint:
                    torch.save({'model': best_state, 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                                'network_config': self.net.config, 'rng_numpy': rng.bit_generator.state,
                                'torch_rng': torch.get_rng_state(), 'history': history}, checkpoint)
            else:
                stale += 1
            if stale >= patience:
                break
        self.net.load_state_dict(best_state)
        logits = self.logits(calibration)
        def objective(t):
            z = logits/t
            return float((logsumexp(z, axis=1)-z[np.arange(len(calibration_labels)), calibration_labels]).mean())
        self.temperature = float(minimize_scalar(objective, bounds=(.5, 2.5), method='bounded').x)
        self.report = {'kind': self.kind, 'seed': self.seed, 'network': self.net.config,
                       'parameter_count': sum(p.numel() for p in self.net.parameters()),
                       'training_rows': len(labels), 'calibration_rows': len(calibration_labels),
                       'epochs_run': len(history), 'best_epoch': best_epoch, 'history': history,
                       'temperature_conditional': self.temperature, 'seconds': time.perf_counter()-start,
                       'selection': 'Calibration conditional log loss only; DEV not used.', 'device': self.device}
        return self

    def logits(self, arrays, batch_size=4096):
        self.net.eval()
        out = []
        with torch.no_grad():
            for begin in range(0, len(arrays[0]), batch_size):
                tensors = [torch.as_tensor(a[begin:begin+batch_size], device=self.device) for a in arrays]
                out.append(self.net(*tensors).cpu().numpy())
        return np.concatenate(out)

    def predict(self, arrays):
        return softmax(self.logits(arrays)/self.temperature, axis=1)

    def save(self, path: Path):
        torch.save({'network_config': self.net.config, 'state_dict': {k:v.cpu() for k,v in self.net.state_dict().items()},
                    'seed': self.seed, 'temperature': self.temperature, 'report': self.report,
                    'delivery_temperature': getattr(self, 'delivery_temperature', self.temperature)}, path)

    @classmethod
    def load(cls, path: Path):
        item = torch.load(path, map_location='cpu', weights_only=False)
        cfg = item['network_config']
        model = cls(cfg['kind'], item['seed'], cfg['width'], cfg['n_classes'])
        model.net = SequenceNetwork(**cfg).to(model.device)
        model.net.load_state_dict(item['state_dict'])
        model.temperature, model.report = item['temperature'], item['report']
        model.delivery_temperature = item.get('delivery_temperature', model.temperature)
        return model
