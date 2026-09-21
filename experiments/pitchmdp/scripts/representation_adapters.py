"""New batter/history adapters; frozen sequence implementation remains unchanged."""
from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
from torch import nn

from pitchmdp.archetypes import Archetypes
from pitchmdp.sequence_model import SequenceModel, SequenceNetwork


def _require_train(frame):
    if not len(frame) or 'split' not in frame or not frame.split.eq('train').all():
        raise ValueError('Adapter fitting requires nonempty TRAIN rows only')
    if 'game_date' in frame and not pd.to_datetime(frame.game_date).between('2023-05-15', '2025-04-30').all():
        raise ValueError('Adapter fit dates outside approved TRAIN period')


def _base_features(frame):
    """Exactly the frozen eleven game/count/hand channels; no batting outcomes."""
    def col(name):
        return pd.to_numeric(frame[name], errors='coerce').fillna(0).to_numpy(np.float32)
    bases = col('bases').astype(int)
    top = frame.inning_topbot.eq('Top').to_numpy()
    return np.column_stack([col('balls')/3, col('strikes')/2, col('outs_when_up')/2,
        col('inning')/9, np.where(top, 1., -1.)*(col('home_score')-col('away_score'))/5,
        top.astype(float), (bases & 1).astype(float), ((bases >> 1) & 1).astype(float),
        ((bases >> 2) & 1).astype(float), frame.stand.eq('L').to_numpy(float),
        frame.p_throws.eq('L').to_numpy(float)]).astype(np.float32)


def _batter_keys(frame):
    keys = pd.to_numeric(frame.batter, errors='coerce')
    if (keys.notna() & ((keys % 1) != 0)).any():
        raise ValueError('Batter join keys must be integer IDs')
    return keys.astype('Int64')


class BatterContext:
    def __init__(self, base_context, mode, clusters=None):
        if mode not in ('hand', 'id', 'continuous', 'clusters', 'reference'):
            raise ValueError('Unknown batter representation')
        if mode == 'clusters' and clusters not in (3, 5, 10, 20):
            raise ValueError('Cluster-only representation requires K in 3, 5, 10, 20')
        if mode != 'clusters' and clusters is not None:
            raise ValueError('Cluster count is only applicable to cluster-only mode')
        self.base_context, self.mode, self.clusters = base_context, mode, clusters

    def fit(self, full_train, id_train=None):
        _require_train(full_train)
        self.fit_rows = len(full_train)
        self.fit_date_max = str(pd.to_datetime(full_train.game_date).max().date()) if 'game_date' in full_train else None
        if self.mode in ('continuous', 'reference') and self.fit_date_max is not None:
            base_max = self.base_context.archetypes.report().get('fit_date_max')
            if base_max is None or pd.Timestamp(base_max) > pd.Timestamp(self.fit_date_max):
                raise ValueError('Reference batting geometry extends beyond this TRAIN pool')
        if self.mode == 'clusters':
            self.archetypes = Archetypes(n_clusters=self.clusters, seed=42).fit(full_train)
        if self.mode == 'id':
            if id_train is None:
                raise ValueError('ID vocabulary requires explicit shared neural TRAIN sample')
            _require_train(id_train)
            if self.fit_date_max is not None and 'game_date' in id_train and pd.to_datetime(id_train.game_date).max() > pd.Timestamp(self.fit_date_max):
                raise ValueError('ID vocabulary dates extend beyond full TRAIN pool')
            key_columns = ['game_pk', 'at_bat_number', 'pitch_number']
            if set(key_columns).issubset(full_train.columns) and set(key_columns).issubset(id_train.columns):
                full_keys = pd.MultiIndex.from_frame(full_train[key_columns])
                if not pd.MultiIndex.from_frame(id_train[key_columns]).isin(full_keys).all():
                    raise ValueError('ID vocabulary rows must belong to full TRAIN pool')
            ids = sorted(int(value) for value in _batter_keys(id_train).dropna().unique())
            self.id_map = {value: index+1 for index, value in enumerate(ids)}
            self.vocab_size = len(self.id_map)+1
            self.vocabulary_rows = len(id_train)
        self.fitted = True
        return self

    def transform(self, frame):
        if not getattr(self, 'fitted', False):
            raise ValueError('BatterContext must be fitted before transformation')
        if self.mode == 'reference':
            return self.base_context.transform(frame)
        base = _base_features(frame)
        if self.mode == 'hand':
            return base
        if self.mode == 'id':
            indices = _batter_keys(frame).map(self.id_map).fillna(0).to_numpy(np.float32)
            return np.column_stack([base, indices]).astype(np.float32)
        if self.mode == 'continuous':
            batting = self.base_context.archetypes.numeric_features(frame)[:, :12]
        else:
            batting = self.archetypes.transform(frame)
        return np.column_stack([base, batting]).astype(np.float32)

    def report(self):
        result = {'mode': self.mode, 'base_channels': 11, 'fit_rows': self.fit_rows,
                  'fit_date_max': self.fit_date_max, 'game_count_handedness_preserved': True,
                  'continuous_normalization': 'Frozen reference TRAIN geometry',
                  'batter_id_input': self.mode == 'id'}
        if self.mode == 'id':
            result.update(vocab_size=self.vocab_size, observed_train_ids=len(self.id_map),
                          vocabulary_rows=self.vocabulary_rows, unknown_index=0,
                          unknown_embedding='Fixed zero; padding_idx=0', embedding_width=16)
        if self.mode == 'clusters':
            result['archetypes'] = self.archetypes.report()
        result['n_context'] = {'hand': 11, 'id': 12, 'continuous': 23, 'reference': 28,
                               'clusters': 11+(self.clusters or 0)}[self.mode]
        return result


class WindowStore:
    """Mask older past tokens while preserving six slots and current realization."""
    def __init__(self, store, history_length):
        if type(history_length) is not int or history_length not in range(1, 6):
            raise ValueError('History length must be an integer from 1 through 5')
        self.store, self.history_length = store, history_length
        self.frame, self.normalizer = store.frame, store.normalizer

    def gather(self, rows, current=None):
        tokens, valid = self.store.gather(rows, current=current)
        if tokens.shape[1] != 6 or valid.shape[1] != 6:
            raise ValueError('Window adapter requires original five-past plus current slots')
        tokens, valid = tokens.copy(), valid.copy()
        stop = 5-self.history_length
        tokens[:, :stop], valid[:, :stop] = 0., False
        return tokens, valid


class IDNetwork(nn.Module):
    """Categorical batter lookup before the original MLP context projection."""
    def __init__(self, vocab_size, width=128, n_classes=10, embedding_dim=16, n_physical=8, length=6):
        super().__init__()
        if vocab_size < 1 or embedding_dim != 16:
            raise ValueError('Expected a nonempty vocabulary including UNK and fixed embedding width 16')
        self.config = dict(kind='id_mlp', vocab_size=vocab_size, width=width, n_classes=n_classes,
                           embedding_dim=embedding_dim, n_physical=n_physical, length=length)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.backbone = SequenceNetwork('flatten_mlp', 11+embedding_dim, n_physical=n_physical,
                                        n_classes=n_classes, width=width, length=length)

    def forward(self, tokens, valid, context):
        if context.ndim != 2 or context.shape[1] != 12:
            raise ValueError('ID context must contain eleven numeric features and one categorical index')
        index = context[:, 11].long()
        if torch.any(context[:, 11] != index) or torch.any(index < 0) or torch.any(index >= self.embedding.num_embeddings):
            raise ValueError('ID context contains an invalid categorical index')
        features = torch.cat([context[:, :11], self.embedding(index)], dim=1)
        return self.backbone(tokens, valid, features)


class IDModel(SequenceModel):
    def __init__(self, vocab_size, seed=42, width=128, n_classes=10, embedding_dim=16):
        super().__init__('id_mlp', seed=seed, width=width, n_classes=n_classes)
        if vocab_size < 1 or embedding_dim != 16:
            raise ValueError('IDModel requires UNK-inclusive vocabulary and embedding width 16')
        self.vocab_size, self.embedding_dim = vocab_size, embedding_dim

    def fit(self, train, labels, calibration, calibration_labels, *, epochs=30, patience=5,
            batch_size=1024, learning_rate=.0005, checkpoint=None):
        """Original optimizer/early-stop/temperature schedule, with categorical embedding."""
        if not len(labels) or not len(calibration_labels) or epochs < 1:
            raise ValueError('Nonempty TRAIN/CAL labels and positive epochs required')
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.net = IDNetwork(self.vocab_size, self.width, self.n_classes,
                             self.embedding_dim, train[0].shape[2], train[0].shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=.01)
        tensors = [torch.as_tensor(value, device=self.device) for value in [*train, np.asarray(labels, np.int64)]]
        rng = np.random.default_rng(self.seed)
        best, best_epoch, stale, history, best_state = np.inf, 0, 0, [], None
        start = time.perf_counter()
        for epoch in range(epochs):
            self.net.train()
            order, total = rng.permutation(len(labels)), 0.
            for begin in range(0, len(labels), batch_size):
                index = torch.as_tensor(order[begin:begin+batch_size], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(self.net(*(value[index] for value in tensors[:3])), tensors[3][index])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.)
                optimizer.step()
                total += float(loss.item())*len(index)
            logits = self.logits(calibration)
            score = float((logsumexp(logits, axis=1)-logits[np.arange(len(calibration_labels)), calibration_labels]).mean())
            record = {'epoch': epoch+1, 'train_log_loss': total/len(labels),
                      'calibration_log_loss': score, 'seconds': time.perf_counter()-start}
            history.append(record)
            print(self.kind, record, flush=True)
            if score < best-1e-5:
                best, best_epoch, stale = score, epoch+1, 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
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
        def objective(temperature):
            scaled = logits/temperature
            return float((logsumexp(scaled, axis=1)-scaled[np.arange(len(calibration_labels)), calibration_labels]).mean())
        self.temperature = float(minimize_scalar(objective, bounds=(.5, 2.5), method='bounded').x)
        self.report = {'kind': self.kind, 'seed': self.seed, 'network': self.net.config,
            'parameter_count': sum(value.numel() for value in self.net.parameters()),
            'embedding_parameter_count': self.net.embedding.weight.numel(),
            'training_rows': len(labels), 'calibration_rows': len(calibration_labels),
            'epochs_run': len(history), 'best_epoch': best_epoch, 'history': history,
            'temperature_conditional': self.temperature, 'seconds': time.perf_counter()-start,
            'selection': 'Calibration conditional log loss only; DEV not used.', 'device': self.device}
        return self

    @classmethod
    def load(cls, path: Path, device=None):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        config = saved['network_config'].copy()
        if config.pop('kind') != 'id_mlp':
            raise ValueError('Checkpoint does not contain an IDModel')
        model = cls(config['vocab_size'], seed=saved['seed'], width=config['width'],
                    n_classes=config['n_classes'], embedding_dim=config['embedding_dim'])
        if device is not None:
            model.device = device
        model.net = IDNetwork(**config).to(model.device)
        model.net.load_state_dict(saved['state_dict'])
        model.temperature, model.report = saved['temperature'], saved['report']
        model.delivery_temperature = saved.get('delivery_temperature', model.temperature)
        return model
