"""RealMLP-style robust scale + smooth clip in front of the frozen residual backbone (PREPARED, NOT registered).

Status: author-only preparation (T100, 2026-10-04) of conditional related-work candidate 3, contrast A
(coordinator reports/research-related-work/FEASIBILITY-candidate3-realmlp-preproc-ko.md §4).  NOT selected, NOT
registered, no registry/queue/config wiring; selection happens only after the full EXP-P35-001 family is scored.

Source: Holzmüller, Grinsztajn, Steinwart 2024, "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular
Data" (NeurIPS 2024, arXiv:2407.04491v3) §3 "Data preprocessing": per numerical column
``f(s_j * (x_j - q_1/2))`` with ``s_j = 1 / (q_3/4 - q_1/4)``, else ``2 / (q_1 - q_0)`` when the interquartile range
is 0, else 0, and ``f(x) = x / sqrt(1 + (x / 3)^2)``.  Written here from the paper text (quotes checked against the
arXiv LaTeX in coordinator T89a); no third-party code is copied and no package is installed.  NOT a faithful
reproduction: only this preprocessing step (the learnable diagonal scaling layer, NTP, the paper's MLP and its
hyperparameters are absent); only the periodic / PLE continuous set is transformed (the paper's RealMLP-TD also
transforms one-hot and other numerical columns; its preprocessing ablation leaves one-hot columns alone): past
type/outcome one-hots, the game-state ordinals and flags (context 0-10) and the near-binary left share stay raw;
applied to our TRAIN mean/std standardized inputs, which is equivalent up to float32
rounding because the rule is invariant to a positive affine map of a column; numpy float64 linear quantiles cast to
float32; our residual backbone, ten-class head, AdamW loop and calibration chain, no tuning.

Exact input transform (fixed), on the same scalars the periodic / PLE arms embed:

  x        = cat(t[:, :-1, :8].flatten(1), context[:, continuous])        # 81 scalars for the counts map
  c_j      = median(TRAIN_j)            q_* = numpy float64 quantile(TRAIN_j, method='linear')
  s_j      = 1/(q_.75 - q_.25)  if > 0  else  2/(max - min)  if > 0  else  0
  x'_j     = z / sqrt(1 + (z/3)^2),  z = s_j * (x_j - c_j)                 # float32 buffers c, s
  t, ctx   = x' written back in place; current token, one-hots, mask and raw context columns unchanged
  logits   = ResidualMLPNetwork.forward(t, valid, ctx)                      # frozen; zeroes padded past positions

Physics columns use the rows whose past position is valid (padded values never enter c, s; padded positions are
zeroed by the frozen forward after the transform).  c, s are computed in ``fit`` from the ``train`` argument only
(never early-stop, temperature, DEV or mlb_dev rows), stored as persistent buffers (``rc_center``, ``rc_scale``),
hashed (``rc_stats_sha256``: sha256 over the float32 little-endian bytes of c then s) and never recomputed at predict
or load time; the fit report carries the hash and the per-rule column labels (written before the frozen fit's own
checkpoint save, so one save).  The network is built by the frozen ResidualMLPNetwork constructor and only registers
buffers, so for one seed every parameter equals the ``residual_counts`` arm's initial value.  Training is the frozen
DirectResidualModel.fit, unchanged.  The zero-current-physics guard of the embedding arms is applied (fail closed).
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_residual_model as residual
from pitchmdp.direct_residual_model import DirectResidualModel, ResidualMLPNetwork

KIND = 'robust_clip_residual_mlp'
FORMAT = 'direct_robust_clip_residual_model_v1'
CLIP = 3.
PREPROCESSING = {'type': 'robust_scale_smooth_clip', 'center': 'median_train',
                 'scale': 'inverse_iqr_else_2_over_range_else_0', 'clip': 'x/sqrt(1+(x/3)^2)', 'clip_constant': CLIP,
                 'columns': 'past_physics_valid_rows+context_continuous', 'learnable_scaling_layer': False}
ARCHITECTURE = {'backbone': {'kind': residual.KIND, **residual.ARCHITECTURE}, 'preprocessing': PREPROCESSING}


def feature_labels(length, context_continuous):
    """Transformed-scalar labels in statistics order (as direct_ple_model.feature_labels)."""
    return [f'past{p}.phys{c}' for p in range(length - 1) for c in range(periodic.N_PHYSICAL)] + \
        [f'context{int(i)}' for i in context_continuous]


def _columns(arrays, context_continuous):
    tokens, valid, context = (np.asarray(a) for a in arrays)
    columns = [tokens[valid[:, p], p, c] for p in range(tokens.shape[1] - 1) for c in range(periodic.N_PHYSICAL)]
    return columns + [context[:, int(i)] for i in context_continuous]


def compute_stats(arrays, context_continuous):
    """Per-scalar (center, scale) float32 [F] from ONE split (call with TRAIN only), embedding order.

    Also returns the per-column rule used: 'iqr', 'range' (IQR 0) or 'constant' (s = 0).  Fails closed on empty or
    non-finite columns.
    """
    centers, scales, rules = [], [], []
    for values in _columns(arrays, context_continuous):
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            raise ValueError('Robust-scale column has no rows')
        if not np.isfinite(values).all():
            raise ValueError('Robust-scale values must be finite')
        q0, q25, q50, q75, q1 = np.quantile(values, [0., .25, .5, .75, 1.], method='linear')
        if q75 - q25 > 0:
            scale, rule = 1. / (q75 - q25), 'iqr'
        elif q1 - q0 > 0:
            scale, rule = 2. / (q1 - q0), 'range'
        else:
            scale, rule = 0., 'constant'
        centers.append(q50)
        scales.append(scale)
        rules.append(rule)
    center, scale = np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)
    check_stats(center, scale)
    return center, scale, rules


def check_stats(center, scale):
    center, scale = np.asarray(center), np.asarray(scale)
    if center.dtype != np.float32 or scale.dtype != np.float32 or center.ndim != 1 or center.shape != scale.shape \
            or not len(center):
        raise ValueError('Robust-scale statistics must be aligned non-empty 1-D float32 arrays')
    if not (np.isfinite(center).all() and np.isfinite(scale).all()) or (scale < 0).any():
        raise ValueError('Robust-scale statistics must be finite with non-negative scale')


def stats_sha256(center, scale):
    check_stats(center, scale)
    return hashlib.sha256(np.asarray(center).astype('<f4').tobytes() + np.asarray(scale).astype('<f4').tobytes()).hexdigest()


def smooth_clip(z):
    return z / torch.sqrt(1. + (z / CLIP) ** 2)


class RobustClipResidualNetwork(ResidualMLPNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, center, scale, width=128, hidden=256, blocks=2,
                 dropout=.1, n_classes=10):
        continuous = [int(i) for i in context_continuous]
        if len(set(continuous)) != len(continuous) or not all(0 <= i < n_context for i in continuous) \
                or n_token <= periodic.N_PHYSICAL:
            raise ValueError('Invalid continuous context map')
        check_stats(center, scale)
        if len(center) != (length - 1) * periodic.N_PHYSICAL + len(continuous):
            raise ValueError('Robust-scale statistics do not cover the transformed scalars')
        super().__init__(n_context, n_token, length, width, hidden, blocks, dropout, n_classes)  # same RNG draws
        self.register_buffer('rc_center', torch.as_tensor(np.asarray(center)).clone())
        self.register_buffer('rc_scale', torch.as_tensor(np.asarray(scale)).clone())
        self.register_buffer('continuous', torch.tensor(continuous, dtype=torch.long), persistent=False)
        self.config.update(kind=KIND, context_continuous=continuous, rc_stats_sha256=stats_sha256(center, scale))

    def transform(self, tokens, context):
        n, length = tokens.shape[:2]
        past = tokens[:, :-1, :periodic.N_PHYSICAL]
        x = torch.cat((past.flatten(1), context[:, self.continuous]), dim=1)
        y = smooth_clip(self.rc_scale * (x - self.rc_center))
        split = (length - 1) * periodic.N_PHYSICAL
        past = y[:, :split].reshape(n, length - 1, periodic.N_PHYSICAL)
        tokens = torch.cat((torch.cat((past, tokens[:, :-1, periodic.N_PHYSICAL:]), dim=2), tokens[:, -1:]), dim=1)
        return tokens, context.index_copy(1, self.continuous, y[:, split:])

    def forward(self, tokens, valid, context):
        tokens, context = self.transform(tokens, context)
        return super().forward(tokens, valid, context)  # masked_fill zeroes padded past positions


def network_kwargs(architecture):
    return {k: architecture['backbone'][k] for k in ('width', 'hidden', 'blocks', 'dropout')}


class DirectRobustClipResidualModel(DirectResidualModel):
    """Frozen DirectResidualModel.fit with the robust-clip network; statistics from the TRAIN argument only."""

    KIND, FORMAT = KIND, FORMAT

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None):
        self.architecture = copy.deepcopy(ARCHITECTURE if architecture is None else architecture)
        if self.architecture != ARCHITECTURE:
            raise ValueError('Robust-clip residual architecture differs from the fixed network')
        self.context_continuous = [int(i) for i in context_continuous]
        self.kind, self.seed, self.width, self.n_classes = KIND, seed, self.architecture['backbone']['width'], 10
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.temperature = self.delivery_temperature = 1.
        self.lightgbm_params = {}
        self.rc_stats = None

    def network(self, n_context, n_token, length):
        if self.rc_stats is None:
            raise RuntimeError('Robust-scale statistics are computed in fit from TRAIN only')
        return RobustClipResidualNetwork(n_context, n_token, length, self.context_continuous, *self.rc_stats,
                                         **network_kwargs(self.architecture), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        train, calibration = periodic._no_current_physics(train), periodic._no_current_physics(calibration)
        center, scale, rules = compute_stats(train, self.context_continuous)
        self.rc_stats = (center, scale)
        self.rc_rules = dict(zip(feature_labels(train[0].shape[1], self.context_continuous), rules, strict=True))
        return super().fit(train, labels, calibration, calibration_labels, **kwargs)

    def _base_report(self, n, early_n, weight):
        # Called by the frozen fit before its own checkpoint save, so the archive carries these fields (one save).
        rules = list(self.rc_rules.values())
        return {**super()._base_report(n, early_n, weight), 'preprocessing': self.architecture['preprocessing'],
                'rc_stats_sha256': stats_sha256(*self.rc_stats),
                'rc_rule_counts': {r: rules.count(r) for r in ('iqr', 'range', 'constant')},
                'rc_rule_columns': {r: [c for c, x in self.rc_rules.items() if x == r] for r in ('range', 'constant')}}

    def logits(self, arrays, batch_size=4096):
        return super().logits(periodic._no_current_physics(arrays), batch_size)

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': KIND, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'rc_stats_sha256': self.net.config['rc_stats_sha256'],
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized robust-clip residual model archive')
        config, state = dict(payload['network_config']), payload['state_dict']
        if not isinstance(config, dict) or config.pop('kind', None) != KIND or 'context_continuous' not in config \
                or 'rc_center' not in state or 'rc_scale' not in state:
            raise ValueError('Archived network differs from the archived architecture')
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device)
        expected = {**network_kwargs(model.architecture), 'n_classes': model.n_classes}
        if {k: config.get(k) for k in expected} != expected:
            raise ValueError('Archived network differs from the archived architecture')
        sha = config.pop('rc_stats_sha256', None)
        center, scale = state['rc_center'].numpy().copy(), state['rc_scale'].numpy().copy()
        if stats_sha256(center, scale) != sha or sha != payload.get('rc_stats_sha256'):
            raise ValueError('Archived robust-scale statistics differ from the recorded hash')
        model.net = RobustClipResidualNetwork(**config, center=center, scale=scale).to(model.device)
        model.net.load_state_dict(state)
        model.rc_stats = (center, scale)
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model
