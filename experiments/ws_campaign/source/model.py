"""Small conditional pitch model; delivery integration is explicit at query time.

Current-pitch coordinates are permitted ONLY in the conditional response model.
The pre-pitch evaluator marginalizes them using training-only delivery samples;
the target policy uses a declared Gaussian execution assumption instead.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

OUTCOMES = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple',
            'home_run', 'hbp', 'double_play')
CATEGORICAL = ('pitch_type', 'pitcher', 'batter', 'stand', 'p_throws', 'prev_pitch_type')
NUMERIC_INPUTS = ('balls', 'strikes', 'outs_when_up', 'inning', 'bases',
                  'home_score', 'away_score', 'inning_topbot', 'plate_x', 'plate_z',
                  'batter_pa_prior', 'batter_obp_prior', 'batter_k_prior')


def outcome_labels(frame: pd.DataFrame) -> np.ndarray:
    d, e = frame.description.fillna(''), frame.events.fillna('')
    labels = np.full(len(frame), -1, dtype=np.int64)
    labels[d.isin(['ball', 'blocked_ball', 'pitchout', 'intent_ball'])] = 0
    labels[d.isin(['called_strike', 'swinging_strike', 'swinging_strike_blocked',
                  'missed_bunt', 'foul_tip', 'bunt_foul_tip'])] = 1
    labels[d.isin(['foul', 'foul_bunt'])] = 2
    labels[(d == 'foul_bunt') & (frame.strikes == 2)] = 1
    labels[d == 'hit_by_pitch'] = 8
    for name in ['single', 'double', 'triple', 'home_run']:
        labels[(d == 'hit_into_play') & (e == name)] = OUTCOMES.index(name)
    labels[(d == 'hit_into_play') & e.isin(
        ['field_out', 'force_out', 'fielders_choice_out', 'sac_fly', 'sac_bunt'])] = 3
    labels[(d == 'hit_into_play') & e.isin(
        ['double_play', 'grounded_into_double_play', 'sac_fly_double_play', 'sac_bunt_double_play'])] = 9
    return labels


def eligible(frame: pd.DataFrame) -> np.ndarray:
    return ((outcome_labels(frame) >= 0) & frame.supported_pa.fillna(False).to_numpy()
            & frame.pitch_type.notna().to_numpy()
            & frame.plate_x.notna().to_numpy() & frame.plate_z.notna().to_numpy()
            & frame.balls.between(0, 3).to_numpy() & frame.strikes.between(0, 2).to_numpy())


def metrics(labels, probabilities) -> dict:
    y = np.asarray(labels, int)
    p = np.asarray(probabilities, float)
    if not len(y):
        return {'n': 0}
    assert p.shape == (len(y), len(OUTCOMES))
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.allclose(p.sum(1), 1, atol=1e-5)
    onehot = np.eye(len(OUTCOMES))[y]
    confidence = p.max(1)
    correct = p.argmax(1) == y
    bins = []
    ece = 0.
    for low in np.arange(0, 1, .1):
        mask = (confidence >= low) & (confidence < low + .1 + (1e-10 if low > .89 else 0))
        if mask.any():
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += mask.mean() * abs(acc - conf)
            bins.append({'low': float(low), 'n': int(mask.sum()), 'accuracy': acc, 'confidence': conf})
    return {'n': len(y), 'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'brier_multiclass': float(((p-onehot)**2).sum(1).mean()),
            'accuracy': float(correct.mean()), 'top_label_ece_10': float(ece),
            'calibration_bins': bins,
            'class_calibration': {name: {'n': int((y == i).sum()), 'observed_rate': float((y == i).mean()),
                                          'predicted_rate': float(p[:, i].mean())}
                                  for i, name in enumerate(OUTCOMES)}}


class Encoder:
    def __init__(self, variant='full'):
        self.variant = variant
        self.categories = [c for c in CATEGORICAL if not ((variant == 'minus_b' or variant.startswith('archetype')) and c == 'batter')
                           and not (variant in ('minus_c', 'archetype_no_history') and c == 'prev_pitch_type')]

    def fit(self, frame):
        self.maps = {c: {v: i+1 for i, v in enumerate(sorted(frame[c].fillna('UNKNOWN').astype(str).unique()))}
                     for c in self.categories}
        if self.variant.startswith('archetype'):
            from .archetypes import Archetypes
            self.archetypes = Archetypes(n_clusters=5, seed=42).fit(frame)
        return self

    def transform(self, frame):
        cats = np.column_stack([frame[c].fillna('UNKNOWN').astype(str).map(self.maps[c]).fillna(0).to_numpy(np.int64)
                                for c in self.categories])
        def col(name, default=0):
            return pd.to_numeric(frame[name], errors='coerce').fillna(default).to_numpy(np.float32)
        x, z = np.clip(col('plate_x'), -3, 3), np.clip(col('plate_z', 2.5)-2.5, -3, 3)
        bases = col('bases').astype(int)
        numeric = [col('balls')/3, col('strikes')/2, col('outs_when_up')/2,
                   col('inning')/9, (col('home_score')-col('away_score'))/5,
                   (frame.inning_topbot == 'Top').to_numpy(np.float32),
                   (bases & 1).astype(float), ((bases >> 1) & 1).astype(float), ((bases >> 2) & 1).astype(float),
                   x, z, x*x, z*z, x*z]
        if self.variant == 'archetype_no_context':
            # Count and conditional location remain; legal game transitions/WE
            # still receive the complete state outside the response network.
            numeric = numeric[:2] + numeric[9:]
        if self.variant != 'minus_b' and not self.variant.startswith('archetype'):
            numeric += [np.log1p(col('batter_pa_prior'))/8, col('batter_obp_prior', .32), col('batter_k_prior', .23)]
        if self.variant.startswith('archetype'):
            numeric += list(self.archetypes.numeric_features(frame).T)
        return cats, np.column_stack(numeric).astype(np.float32)


class Network(nn.Module):
    def __init__(self, sizes, n_numeric):
        super().__init__()
        dims = [min(12, max(2, int(np.sqrt(n)))) for n in sizes]
        self.embeddings = nn.ModuleList([nn.Embedding(n, d, padding_idx=0) for n, d in zip(sizes, dims)])
        self.layers = nn.Sequential(nn.Linear(sum(dims)+n_numeric, 96), nn.ReLU(),
                                    nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, len(OUTCOMES)))

    def forward(self, cat, numeric):
        return self.layers(torch.cat([e(cat[:, i]) for i, e in enumerate(self.embeddings)] + [numeric], dim=1))


class PitchModel:
    def __init__(self, variant='full', seed=42):
        self.encoder = Encoder(variant)
        self.seed = seed
        self.temperature = 1.

    def fit(self, train, calibration, epochs=5, batch_size=4096):
        torch.manual_seed(self.seed)
        torch.set_num_threads(4)
        self.encoder.fit(train)
        cat, num = self.encoder.transform(train)
        cc, cn = self.encoder.transform(calibration)
        y, cy = outcome_labels(train), outcome_labels(calibration)
        assert (y >= 0).all() and (cy >= 0).all()
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.net = Network([len(self.encoder.maps[c])+1 for c in self.encoder.categories], num.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=.002, weight_decay=.005)
        # Entire small tensors fit on M4 shared memory; one training job only.
        tensors = [torch.as_tensor(a, device=self.device) for a in [cat, num, y]]
        rng = np.random.default_rng(self.seed)
        best, state = np.inf, None
        history = []
        start = time.perf_counter()
        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(len(y))
            total = 0.
            for begin in range(0, len(y), batch_size):
                ix = torch.as_tensor(order[begin:begin+batch_size], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(self.net(tensors[0][ix], tensors[1][ix]), tensors[2][ix])
                loss.backward()
                optimizer.step()
                total += loss.item()*len(ix)
            pred = self.predict(calibration)
            score = metrics(cy, pred)['log_loss']
            record = {'epoch': epoch+1, 'train_log_loss': total/len(y), 'calibration_log_loss': score,
                      'elapsed_seconds': time.perf_counter()-start}
            history.append(record)
            print(record, flush=True)
            if score < best:
                best, state = score, copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(state)
        # Scalar temperature chosen exclusively on May-June calibration.
        from scipy.optimize import minimize_scalar
        logits = self.logits(calibration)
        from scipy.special import logsumexp
        def objective(t):
            v = logits/t
            return float((logsumexp(v, axis=1)-v[np.arange(len(cy)), cy]).mean())
        result = minimize_scalar(objective, bounds=(.6, 2.), method='bounded')
        self.temperature = float(result.x)
        self.training_report = {'history': history, 'temperature': self.temperature, 'n_train': len(train),
                                'n_calibration': len(calibration), 'device': self.device, 'seed': self.seed,
                                'variant': self.encoder.variant, 'seconds': time.perf_counter()-start,
                                'categories': self.encoder.categories, 'numeric_inputs': list(NUMERIC_INPUTS),
                                'conditional_location_model': True}
        if self.encoder.variant.startswith('archetype'):
            self.training_report['archetypes'] = self.encoder.archetypes.report()
            from .archetypes import HISTORY_COLUMNS
            self.training_report['numeric_inputs'] = list(NUMERIC_INPUTS[:-3]) + list(HISTORY_COLUMNS)
            if self.encoder.variant == 'archetype_no_context':
                self.training_report['numeric_inputs'] = ['balls', 'strikes', 'plate_x', 'plate_z', *HISTORY_COLUMNS]
        return self

    def logits(self, frame, batch_size=16384):
        cat, num = self.encoder.transform(frame)
        result = []
        self.net.eval()
        with torch.no_grad():
            for begin in range(0, len(frame), batch_size):
                c = torch.as_tensor(cat[begin:begin+batch_size], device=self.device)
                n = torch.as_tensor(num[begin:begin+batch_size], device=self.device)
                result.append(self.net(c, n).cpu().numpy())
        return np.concatenate(result)

    def predict(self, frame):
        from scipy.special import softmax
        return softmax(self.logits(frame)/self.temperature, axis=1)

    def save(self, destination: Path):
        torch.save({'state_dict': {k: v.cpu() for k, v in self.net.state_dict().items()},
                    'encoder': self.encoder, 'temperature': self.temperature, 'report': self.training_report}, destination)

    @classmethod
    def load(cls, source: Path):
        obj = torch.load(source, map_location='cpu', weights_only=False)
        model = cls(obj['encoder'].variant, obj['report']['seed'])
        model.encoder, model.temperature, model.training_report = obj['encoder'], obj['temperature'], obj['report']
        model.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        n_num = obj['state_dict']['layers.0.weight'].shape[1] - sum(v.shape[1] for k, v in obj['state_dict'].items() if k.startswith('embeddings.'))
        model.net = Network([len(model.encoder.maps[c])+1 for c in model.encoder.categories], n_num).to(model.device)
        model.net.load_state_dict(obj['state_dict'])
        return model


class CountBaseline:
    """Training-only smoothed count/hand outcome frequencies, no location oracle."""
    keys = ['balls', 'strikes', 'stand', 'p_throws']
    def fit(self, train):
        t = train[self.keys].copy()
        t['y'] = outcome_labels(train)
        self.global_p = (np.bincount(t.y, minlength=len(OUTCOMES)) + 1).astype(float)
        self.global_p /= self.global_p.sum()
        self.table = {}
        for key, g in t.groupby(self.keys):
            count = np.bincount(g.y, minlength=len(OUTCOMES))+50*self.global_p
            self.table[key] = count/count.sum()
        return self

    def predict(self, frame):
        return np.array([self.table.get(k, self.global_p) for k in frame[self.keys].itertuples(index=False, name=None)])


class DeliveryDistribution:
    """Observed delivery distribution for TYPE-ONLY pre-pitch prediction.

    Draws are from training pitches only; no claim they are intended targets.
    """
    def fit(self, train, draws=25, seed=42):
        self.draws = draws
        rng = np.random.default_rng(seed)
        self.pools = {}
        for keys in [['pitch_type', 'stand'], ['pitcher', 'pitch_type', 'stand']]:
            for key, g in train.groupby(keys, observed=True):
                if len(keys) == 3 and len(g) < 80:
                    continue
                pts = g[['plate_x', 'plate_z']].to_numpy(float)
                self.pools[key] = pts[rng.choice(len(pts), draws, replace=len(pts)<draws)]
        pts = train[['plate_x', 'plate_z']].to_numpy(float)
        self.fallback = pts[rng.choice(len(pts), draws)]
        return self

    def predict(self, model, frame, chunk_size=1000):
        result = []
        for start in range(0, len(frame), chunk_size):
            piece = frame.iloc[start:start+chunk_size]
            points = np.array([self.pools.get((r.pitcher, r.pitch_type, r.stand),
                                             self.pools.get((r.pitch_type, r.stand), self.fallback))
                               for r in piece[['pitcher', 'pitch_type', 'stand']].itertuples(index=False)])
            expanded = piece.loc[piece.index.repeat(self.draws)].reset_index(drop=True)
            expanded['plate_x'], expanded['plate_z'] = points[:, :, 0].ravel(), points[:, :, 1].ravel()
            p = model.predict(expanded).reshape(len(piece), self.draws, -1).mean(1)
            result.append(p)
        return np.concatenate(result)
