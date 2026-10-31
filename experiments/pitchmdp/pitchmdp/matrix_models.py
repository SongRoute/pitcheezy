"""Common-input ML2 models; no legacy model or feature behavior is changed.

The Melville-inspired cell is an explicit common-context adaptation, not an
exact paper reproduction: no batter identity embedding and a shared 10-class
output replace the paper-specific representation and outcome task.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp, softmax
import torch
from torch import nn

KINDS = ('linear', 'flatten_mlp', 'transformer', 'lstm', 'gru', 'melville', 'lightgbm')


def _validate(arrays):
    tokens, valid, context = [np.asarray(a) for a in arrays]
    if tokens.ndim != 3 or valid.shape != tokens.shape[:2] or context.ndim != 2 or len(context) != len(tokens):
        raise ValueError('Expected aligned tokens [N,L,C], mask [N,L], context [N,D]')
    if valid.dtype != np.bool_ or not valid[:, -1].all():
        raise ValueError('Boolean mask must mark the current last token valid')
    if not np.isfinite(tokens).all() or not np.isfinite(context).all():
        raise ValueError('All supplied inputs must be finite')
    if tokens.shape[2] < 20 or np.any(tokens[:, -1, -11:] != 0):
        raise ValueError('Enriched tokens require eleven zero current-outcome channels')
    return tokens.astype(np.float32), valid, context.astype(np.float32)


def _flat(arrays):
    tokens, valid, context = _validate(arrays)
    tokens = np.where(valid[..., None], tokens, 0.)
    return np.column_stack((tokens.reshape(len(tokens), -1), valid.astype(np.float32), context))


class MatrixNetwork(nn.Module):
    def __init__(self, kind, n_context, n_token, length, width=128, n_classes=10):
        super().__init__()
        self.config = dict(kind=kind, n_context=n_context, n_token=n_token,
                           length=length, width=width, n_classes=n_classes)
        self.kind, self.length = kind, length
        n_flat = length * (n_token + 1) + n_context
        if kind == 'linear':
            self.head = nn.Linear(n_flat, n_classes)
        elif kind == 'flatten_mlp':
            self.head = nn.Sequential(nn.Linear(n_flat, width * 2), nn.GELU(),
                                      nn.Linear(width * 2, width * 2), nn.GELU(), nn.Linear(width * 2, n_classes))
        elif kind == 'transformer':
            if width % 4:
                raise ValueError('Transformer width must be divisible by four')
            self.projection = nn.Linear(n_token, width)
            self.position = nn.Parameter(torch.empty(length, width))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                                               activation='gelu', batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(width + n_context, n_classes)
        elif kind in ('lstm', 'gru'):
            self.cell = (nn.LSTMCell if kind == 'lstm' else nn.GRUCell)(n_token, width)
            self.head = nn.Linear(width + n_context, n_classes)
        elif kind == 'melville':
            self.outcome_embedding = nn.Linear(11, width, bias=False)
            # Additive past update h <- h + MLP(x_past, e(y_past)).
            self.update = nn.Sequential(nn.Linear(width + n_token - 11, width),
                                        nn.Tanh(), nn.Linear(width, width), nn.Tanh())
            self.head = nn.Sequential(nn.Linear(width + n_token - 11 + n_context, width),
                                      nn.GELU(), nn.Linear(width, n_classes))
        else:
            raise ValueError('Unsupported torch model: ' + kind)

    def forward(self, tokens, valid, context):
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        if self.kind in ('linear', 'flatten_mlp'):
            return self.head(torch.cat((tokens.flatten(1), valid.float(), context), dim=1))
        if self.kind == 'transformer':
            z = self.encoder(self.projection(tokens) + self.position, src_key_padding_mask=~valid)
            z = (z * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        else:
            width = self.config['width']
            z = tokens.new_zeros((len(tokens), width))
            c = z.clone()
            stop = self.length - 1 if self.kind == 'melville' else self.length
            for j in range(stop):
                mask = valid[:, j, None]
                if self.kind == 'lstm':
                    next_z, next_c = self.cell(tokens[:, j], (z, c))
                    z, c = torch.where(mask, next_z, z), torch.where(mask, next_c, c)
                elif self.kind == 'gru':
                    z = torch.where(mask, self.cell(tokens[:, j], z), z)
                else:
                    update = self.update(torch.cat((tokens[:, j, :-11],
                                                   self.outcome_embedding(tokens[:, j, -11:])), dim=1))
                    z = torch.where(mask, z + update, z)
            if self.kind == 'melville':
                return self.head(torch.cat((z, tokens[:, -1, :-11], context), dim=1))
        return self.head(torch.cat((z, context), dim=1))


class MatrixModel:
    def __init__(self, kind='flatten_mlp', seed=0, width=128, n_classes=10, device=None,
                 lightgbm_params=None):
        if kind not in KINDS or n_classes != 10:
            raise ValueError('ML2 requires a supported model kind and exactly ten classes')
        self.kind, self.seed, self.width, self.n_classes = kind, seed, width, n_classes
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = dict(lightgbm_params or {})

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None, sample_weight=None):
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
        if self.kind == 'lightgbm':
            self._fit_lightgbm(train, y, calibration, ey, weight, epochs, patience, learning_rate)
            self.report['seconds'] = time.perf_counter() - start
            if checkpoint:
                self.save(checkpoint)
            return self
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.net = MatrixNetwork(self.kind, train[2].shape[1], train[0].shape[2],
                                 train[0].shape[1], self.width, self.n_classes).to(self.device)
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
            logits = self.logits(calibration)
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
                           history=history, seconds=time.perf_counter() - start)
        if checkpoint:
            self.save(checkpoint)
        return self

    def _base_report(self, n, early_n, weight):
        return {'kind': self.kind, 'seed': self.seed, 'device': self.device,
                'training_rows': n, 'calibration_rows': early_n, 'earlystop_rows': early_n,
                'weight_ess': float(weight.sum() ** 2 / np.square(weight).sum()),
                'selection': 'unweighted conditional NLL on early-stop rows; no CAL/DEV model selection',
                'calibration': 'none in fit; external JointDelivery temperature CAL then ensemble/blend CAL',
                'information': 'identical ordered physical/type/prior-outcome tokens, mask, context',
                'melville_adaptation': ('additive recurrent update of past tokens and learned prior-outcome embedding; '
                                        'common context instead of paper batter embedding; ten-class adaptation') if self.kind == 'melville' else None}

    def _fit_lightgbm(self, train, y, early, ey, weight, epochs, patience, learning_rate):
        import lightgbm as lgb  # Optional dependency only when this family is used.
        params = {'objective': 'multiclass', 'num_class': 10, 'learning_rate': learning_rate,
                  'num_leaves': 31, 'verbosity': -1, 'num_threads': 4, 'seed': self.seed,
                  'deterministic': True, 'force_col_wise': True, **self.lightgbm_params}
        if params['objective'] != 'multiclass' or params['num_class'] != 10:
            raise ValueError('LightGBM must preserve ten-class multiclass objective')
        self.booster = lgb.train(params, lgb.Dataset(_flat(train), label=y, weight=weight),
                                 num_boost_round=epochs, valid_sets=[lgb.Dataset(_flat(early), label=ey)],
                                 callbacks=[lgb.early_stopping(patience, verbose=False)])
        self.report = self._base_report(len(y), len(ey), weight)
        self.report.update(device='cpu', params=params, best_epoch=self.booster.best_iteration,
                           epochs_run=self.booster.current_iteration(),
                           parameter_count=None, input_shape=list(train[0].shape[1:]), n_context=train[2].shape[1])

    def logits(self, arrays, batch_size=4096):
        arrays = _validate(arrays)
        if not len(arrays[0]):
            return np.empty((0, 10), dtype=np.float32)
        if self.kind == 'lightgbm':
            return np.asarray(self.booster.predict(_flat(arrays), raw_score=True), dtype=np.float64)
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(arrays[0]), batch_size):
                inputs = [torch.as_tensor(a[start:start + batch_size], device=self.device) for a in arrays]
                outputs.append(self.net(*inputs).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, arrays):
        return softmax(self.logits(arrays) / self.temperature, axis=1)

    def save(self, path):
        payload = {'format': 'matrix_model_v1', 'kind': self.kind, 'seed': self.seed,
                   'width': self.width, 'n_classes': self.n_classes, 'temperature': self.temperature,
                   'delivery_temperature': self.delivery_temperature, 'report': self.report,
                   'lightgbm_params': self.lightgbm_params}
        if self.kind == 'lightgbm':
            payload['booster'] = self.booster.model_to_string()
        else:
            payload.update(network_config=self.net.config,
                           state_dict={k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()})
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=False)
        if payload.get('format') != 'matrix_model_v1':
            raise ValueError('Unrecognized matrix model archive')
        model = cls(payload['kind'], payload['seed'], payload['width'], payload['n_classes'],
                    device, payload['lightgbm_params'])
        if model.kind == 'lightgbm':
            import lightgbm as lgb
            model.booster = lgb.Booster(model_str=payload['booster'])
        else:
            model.net = MatrixNetwork(**payload['network_config']).to(model.device)
            model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
