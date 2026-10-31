"""F1 batter-representation bridge: an information-removal control for G0-global.

The full arm is the preserved G0-global model. The masked arm keeps the same
52-channel neural input contract, network shape and initialization but sets
the seventeen existing batter channels ``context[:, 11:28]`` (six style rates,
six reliabilities, five soft memberships) to zero in training, early stopping,
delivery-temperature calibration and every integrated prediction. The seven
routing channels appended by ``SharingContext`` are stripped by the unchanged
``SharingPredictor`` wrapper exactly as for G0-global, so both arms share the
conditional softmax -> log path. H5 physical/type/outcome history is retained;
this is not a removal of every batter-related signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matrix_sharing import SharingPredictor, training_arrays

CONTEXT_WIDTH = 52
ROUTING_WIDTH = 7
BATTER_START, BATTER_STOP = 11, 28
BATTER_CHANNELS = slice(BATTER_START, BATTER_STOP)
BATTER_GROUPS = ('zero', 'low', 'high')


def mask_batter_context(context):
    """Return a copy of a 52-channel neural context with [11:28] set to zero."""
    values = np.asarray(context)
    if values.ndim != 2 or values.shape[1] != CONTEXT_WIDTH:
        raise ValueError('Batter mask requires the 52-channel G0 neural context after routing removal')
    masked = values.copy()
    masked[:, BATTER_CHANNELS] = 0
    return masked


BASE_FEATURES = ['balls', 'strikes', 'outs', 'inning', 'defensive_score_lead', 'defender_is_home',
                 'first_base', 'second_base', 'third_base', 'batter_left', 'pitcher_left',
                 'six_batter_style_rates', 'six_style_reliabilities', 'five_soft_memberships']
BATTER_WIDTH = BATTER_STOP - BATTER_START
PITCHER_WIDTH = CONTEXT_WIDTH - BATTER_STOP


def check_context_layout(base, sharing, probe):
    """Assert 11 game/hand + 17 batter + 23 pitcher profile + 1 count (+7 routing)."""
    if base.report()['features'] != BASE_FEATURES:
        raise ValueError('Frozen base context feature order differs from the F1 mask contract')
    batter = base.archetypes.numeric_features(probe)
    widths = {'base': int(base.transform(probe).shape[1]), 'batter': int(np.asarray(batter).shape[1]),
              'pitcher_profile': len(sharing.clusters['columns']),
              'sharing': int(sharing.transform(probe).shape[1])}
    expected = {'base': BATTER_STOP, 'batter': BATTER_WIDTH, 'pitcher_profile': PITCHER_WIDTH - 1,
                'sharing': CONTEXT_WIDTH + ROUTING_WIDTH}
    if widths != expected:
        raise ValueError(f'Frozen G context widths {widths} differ from F1 layout {expected}')
    return {**widths, 'game_and_hand': [0, BATTER_START], 'batter': [BATTER_START, BATTER_STOP],
            'pitcher': [BATTER_STOP, CONTEXT_WIDTH], 'routing': [CONTEXT_WIDTH, CONTEXT_WIDTH + ROUTING_WIDTH]}


def masked_training_arrays(arrays):
    """SharingContext arrays (59 channels) -> G0 training arrays with batter mask."""
    tokens, valid, context = training_arrays(arrays)
    return tokens, valid, mask_batter_context(context)


class BatterMaskedModel:
    """Inference-time mask around a fitted matrix model; never unmasked."""

    def __init__(self, inner):
        self.inner = inner

    @property
    def kind(self):
        return self.inner.kind

    @property
    def seed(self):
        return self.inner.seed

    @property
    def report(self):
        return self.inner.report

    def logits(self, arrays, **kwargs):
        tokens, valid, context = arrays
        return self.inner.logits((tokens, valid, mask_batter_context(context)), **kwargs)


def masked_g0_predictor(inner, clusters):
    """Same SharingPredictor G0-global wrapper as the full arm, masked inner model."""
    predictor = SharingPredictor('G0-global', BatterMaskedModel(inner), clusters)
    predictor.report = {**predictor.report, 'bridge_arm': 'masked',
                        'masked_context_channels': [BATTER_START, BATTER_STOP],
                        'mask_scope': 'training, early stopping, calibration and inference'}
    return predictor


def fit_masked(model, train_arrays, labels, early_arrays, early_labels, **budget):
    """Fit a MatrixModel on masked arrays built from SharingContext output."""
    return model.fit(masked_training_arrays(train_arrays), labels,
                     masked_training_arrays(early_arrays), early_labels, **budget)


def network_signature(model):
    """Architecture/initialization-shape signature used to pair full and masked."""
    net = model.net
    return {'network': dict(net.config),
            'parameter_count': int(sum(p.numel() for p in net.parameters())),
            'parameter_shapes': {name: list(tensor.shape) for name, tensor in net.state_dict().items()}}


def batter_train_volume(train_batters, *, quantile=.25):
    """D100 TRAIN pitch counts per batter and the linear q25 of positive counts.

    These are supervised TRAIN sample counts, not post-cutoff cumulative
    observation counts. The threshold is fixed from TRAIN only.
    """
    batters = np.asarray(train_batters)
    if batters.ndim != 1 or not len(batters):
        raise ValueError('Nonempty TRAIN batter identities required')
    if pd.isna(batters).any():
        raise ValueError('TRAIN batter identity missing')
    ids, counts = np.unique(batters.astype(np.int64), return_counts=True)
    q = float(np.quantile(counts.astype(float), quantile, method='linear'))
    return {'counts': {str(int(b)): int(n) for b, n in zip(ids, counts)}, 'q25': q,
            'quantile': quantile, 'method': 'linear', 'boundary': 'low includes count == q25',
            'positive_batters': int(len(ids)), 'scope': 'D100 TRAIN supervised pitches per batter'}


def batter_groups(dev_batters, counts, q25):
    """zero: absent from D100 TRAIN; low: 1..q25 inclusive; high: > q25."""
    n = np.array([int(counts.get(str(int(b)), 0)) for b in np.asarray(dev_batters)], dtype=np.int64)
    return np.select([n == 0, n <= q25], ['zero', 'low'], default='high'), n
