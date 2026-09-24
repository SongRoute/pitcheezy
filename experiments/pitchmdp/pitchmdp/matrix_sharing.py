"""TRAIN-only pitcher archetypes and explicit predictive partial pooling.

Routing IDs are metadata consumed by the wrapper, never neural input features.
Partial pooling is a fixed probability shrinkage adaptation, not a fitted
Bayesian posterior or a claim about causal pitcher types.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax


CLUSTER_FEATURES = ('effective_speed', 'release_spin_rate', 'pfx_x', 'pfx_z')


def fit_pitcher_clusters(train, *, k=4, seed=20260924):
    if k != 4 or not len(train) or not train.split.eq('train').all():
        raise ValueError('Four-cluster fit requires nonempty TRAIN only')
    if train.pitcher.isna().any():
        raise ValueError('Pitcher identity required')
    grouped = train.groupby('pitcher', sort=True)
    physical = grouped[list(CLUSTER_FEATURES)].mean()
    counts = grouped.size().reindex(physical.index)
    types = pd.crosstab(train.pitcher, train.pitch_type).reindex(physical.index, fill_value=0)
    types = types.div(counts, axis=0)
    hand = grouped.p_throws.agg(lambda s: float(s.eq('L').mean()))
    raw = pd.concat([physical, hand.rename('left_share'), types.add_prefix('type_')], axis=1)
    values = raw.to_numpy(float)
    median = np.nanmedian(values, axis=0)
    median = np.where(np.isfinite(median), median, 0.)
    values = np.where(np.isfinite(values), values, median)
    mean, scale = values.mean(0), values.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    x = (values - mean) / scale
    if len(x) < k:
        raise ValueError('Insufficient TRAIN pitchers for four clusters')
    rng = np.random.default_rng(seed)
    # Deterministic k-means++ with equal weight per pitcher.
    chosen = [int(rng.integers(len(x)))]
    for _ in range(1, k):
        distance = np.square(x[:, None] - x[chosen]).sum(2).min(1)
        distance[chosen] = 0.
        remaining = np.setdiff1d(np.arange(len(x)), chosen)
        chosen.append(int(rng.choice(len(x), p=distance / distance.sum())) if distance.sum() else int(remaining[0]))
    centers = x[chosen].copy()
    labels = np.full(len(x), -1, dtype=int)
    for iteration in range(100):
        new_labels = np.square(x[:, None] - centers).sum(2).argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            if (labels == c).any():
                centers[c] = x[labels == c].mean(0)
    ids = physical.index.to_numpy(np.int64)
    return {'version': 'pitcher_clusters_v1', 'k': k, 'seed': seed,
            'columns': list(raw.columns), 'median': median.tolist(), 'mean': mean.tolist(),
            'scale': scale.tolist(), 'centers': centers.tolist(), 'iterations': iteration + 1,
            'pitcher_cluster': {str(int(pid)): int(label) for pid, label in zip(ids, labels)},
            'pitcher_counts': {str(int(pid)): int(n) for pid, n in zip(ids, counts)},
            'cluster_counts': {str(c): int(counts.to_numpy()[labels == c].sum()) for c in range(k)},
            'cluster_players': {str(c): int((labels == c).sum()) for c in range(k)},
            'scope': 'eligible TRAIN; equal pitcher-weight kmeans, physical means, hand share, pitch-type proportions',
            'unseen_rule': 'unknown and league fallback; no assignment from future player statistics'}


class SharingContext:
    def __init__(self, base, clusters):
        self.base, self.clusters = base, clusters
        self.mapping = {int(pid): label for pid, label in clusters['pitcher_cluster'].items()}

    def transform(self, frame):
        context = self.base.transform(frame)
        cluster = np.array([self.mapping.get(int(pid), -1) for pid in frame.pitcher], dtype=int)
        flags = np.eye(5, dtype=np.float32)[np.where(cluster >= 0, cluster, 4)]
        # IDs fit exactly into float32 for the declared MLB identity range.
        ids = frame.pitcher.to_numpy(np.int64)
        if (np.abs(ids) >= 2**24).any():
            raise ValueError('Routing ID cannot be represented exactly')
        return np.column_stack([context, flags, cluster, ids]).astype(np.float32)


def training_arrays(arrays, cluster_features=False):
    tokens, valid, context = arrays
    if context.shape[1] < 7:
        raise ValueError('Sharing context requires cluster and routing metadata')
    return tokens, valid, context[:, :-2] if cluster_features else context[:, :-7]


class SharingPredictor:
    """Conditional predictor routed before common delivery marginalization."""
    def __init__(self, cell, global_model, clusters, *, feature_model=None,
                 personal_models=None, cluster_models=None, individual_tau=1000., cluster_tau=10000.):
        if cell not in ('G0-global', 'G1-personal', 'G2-feature', 'G3-cluster', 'G4-partial'):
            raise ValueError('Unknown sharing cell')
        self.cell, self.global_model, self.clusters = cell, global_model, clusters
        self.feature_model = feature_model
        self.personal_models, self.cluster_models = personal_models or {}, cluster_models or {}
        self.individual_tau, self.cluster_tau = individual_tau, cluster_tau
        self.temperature = self.delivery_temperature = 1.
        self.report = {'cell': cell, 'partial_pooling': 'fixed conditional probability shrinkage',
                       'individual_tau': individual_tau, 'cluster_tau': cluster_tau,
                       'unknown': 'league fallback', 'routing_metadata_never_a_learned_feature': True}

    def logits(self, arrays):
        original = training_arrays(arrays)
        enriched = training_arrays(arrays, cluster_features=True)
        routing = arrays[2][:, -2:].astype(np.int64)
        cluster_ids, pitchers = routing[:, 0], routing[:, 1]
        global_p = softmax(self.global_model.logits(original).astype(float), axis=1)
        result = global_p.copy()
        if self.cell == 'G0-global':
            return np.log(np.clip(result, 1e-30, 1))
        if self.cell == 'G2-feature':
            if self.feature_model is None:
                raise ValueError('Cluster-feature predictor missing')
            known = cluster_ids >= 0
            if known.any():
                result[known] = softmax(self.feature_model.logits(tuple(a[known] for a in enriched)).astype(float), axis=1)
        if self.cell in ('G3-cluster', 'G4-partial'):
            for cid, model in self.cluster_models.items():
                mask = cluster_ids == int(cid)
                if not mask.any():
                    continue
                probability = softmax(model.logits(tuple(a[mask] for a in enriched)).astype(float), axis=1)
                count = self.clusters['cluster_counts'][str(cid)]
                weight = 1. if self.cell == 'G3-cluster' else count / (count + self.cluster_tau)
                result[mask] = weight * probability + (1 - weight) * result[mask]
        if self.cell in ('G1-personal', 'G4-partial'):
            for pid, model in self.personal_models.items():
                mask = pitchers == int(pid)
                if not mask.any():
                    continue
                probability = softmax(model.logits(tuple(a[mask] for a in original)).astype(float), axis=1)
                count = self.clusters['pitcher_counts'][str(pid)]
                weight = 1. if self.cell == 'G1-personal' else count / (count + self.individual_tau)
                result[mask] = weight * probability + (1 - weight) * result[mask]
        if not np.allclose(result.sum(1), 1., atol=1e-10, rtol=0):
            raise ValueError('Sharing mixture mass differs')
        return np.log(np.clip(result, 1e-30, 1))
