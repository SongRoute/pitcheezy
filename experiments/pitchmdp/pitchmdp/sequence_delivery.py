"""Train-only JOINT current-delivery marginalization for physical sequence models."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


class JointDelivery:
    # Every retained vector was one actual TRAIN delivery, preserving correlations.
    TIERS = [('pitch_type', 'p_throws', 'stand'),
             ('pitch_type', 'p_throws', 'stand', 'balls', 'strikes'),
             ('pitcher', 'pitch_type', 'p_throws', 'stand'),
             ('pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes')]

    def fit(self, train, normalizer, draws=25, seed=42):
        if not train.split.eq('train').all():
            raise ValueError('Delivery fit requires TRAIN only')
        self.draws, self.normalizer = draws, normalizer
        physical = normalizer.transform(train)
        self.pools = {}
        rng = np.random.default_rng(seed)
        for level, keys in enumerate(self.TIERS):
            for key, indices in train.groupby(list(keys), observed=True).indices.items():
                minimum = 20 if 'balls' in keys else 50
                if len(indices) < minimum:
                    continue
                positions = rng.choice(indices, size=draws, replace=len(indices) < draws)
                self.pools[(level, key)] = physical[positions]
        self.fallback = physical[rng.choice(len(physical), draws, replace=len(physical)<draws)]
        self.report = {'draws': draws, 'seed': seed, 'training_rows': len(train),
                       'pool_count': len(self.pools), 'tiers': self.TIERS,
                       'priority': 'pitcher/type/hand/count; pitcher/type/hand; league/type/hand/count; league/type/hand',
                       'sample': 'Joint normalized physical vectors from TRAIN only; no intended target labels.'}
        return self

    def sample(self, frame):
        result, levels = [], []
        columns = list(dict.fromkeys(k for tier in self.TIERS for k in tier))
        for values in frame[columns].itertuples(index=False, name=None):
            row = dict(zip(columns, values))
            points, used = self.fallback, -1
            for level in reversed(range(len(self.TIERS))):
                key = tuple(row[k] for k in self.TIERS[level])
                if (level, key) in self.pools:
                    points, used = self.pools[(level, key)], level
                    break
            result.append(points)
            levels.append(used)
        return np.asarray(result), np.asarray(levels)

    def logits(self, model, store, context_encoder, rows, chunk_size=256):
        predictions = []
        levels_all = []
        for begin in range(0, len(rows), chunk_size):
            selected = np.asarray(rows[begin:begin+chunk_size])
            frame = store.frame.iloc[selected]
            samples, levels = self.sample(frame)
            repeated = np.repeat(selected, self.draws)
            tokens, valid = store.gather(repeated, current=samples.reshape(-1, samples.shape[-1]))
            context = np.repeat(context_encoder.transform(frame), self.draws, axis=0)
            logits = model.logits((tokens, valid, context)).reshape(len(selected), self.draws, -1)
            predictions.append(logits)
            levels_all.append(levels)
        return np.concatenate(predictions), np.concatenate(levels_all)

    def predict(self, model, store, context_encoder, rows):
        logits, _ = self.logits(model, store, context_encoder, rows)
        return softmax(logits/getattr(model, 'delivery_temperature', model.temperature), axis=-1).mean(1)

    def calibrate(self, model, store, context_encoder, rows, labels):
        logits, levels = self.logits(model, store, context_encoder, rows)
        def loss(temperature):
            p = softmax(logits/temperature, axis=-1).mean(1)
            return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1)).mean())
        fitted = minimize_scalar(loss, bounds=(.5, 2.5), method='bounded')
        model.delivery_temperature = float(fitted.x)
        model.report['delivery_temperature'] = float(fitted.x)
        model.report['calibration_integrated_log_loss'] = float(fitted.fun)
        model.report['delivery_calibration_rows'] = len(rows)
        return model
