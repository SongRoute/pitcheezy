"""TRAIN-only pitcher-ID embedding on the frozen periodic-residual direct head (EXP-P38-001 DRAFT).

One information change vs the P36/P37 arm periodic_residual_counts: the export already
carries one ``pitcher`` int64 per row (export_cuda_direct.ARRAYS) that no direct-head
model has read.  This arm maps it to a TRAIN-only vocabulary and adds a learned vector:

  idx    = 1 + position of pitcher in the TRAIN vocabulary (pitchers with >= MIN_TRAIN_PITCHES
           TRAIN rows, sorted by id); 0 = UNK for every other pitcher, in TRAIN and later splits
           (cold-start fallback is therefore trained on real rare TRAIN pitchers)
  z      = Linear_in(cat(raw, e)) + P @ T[idx]        # T [n_vocab + 1, 8] zero init, P [128, 8]
  rest   = the frozen PeriodicResidualNetwork backbone (2 residual blocks, output head)

The id index travels as one extra last context column (exact float32 integer); the
network strips it before the frozen raw/continuous column maps, so those maps are the
registered P36 ones.  Initialisation keeps the paired comparison clean: the parent
network is built first with the baseline width (identical modules and global RNG use),
T is zeros and P is drawn from a private generator, so at the same seed the untrained
network returns exactly the baseline logits and the dropout / batch streams match.

Fixed here, not tuned and not chosen on DEV: MIN_TRAIN_PITCHES 100, D_ID 8 (= the frozen
d_embedding), AdamW weight decay .01 like every other parameter.  Vocabulary, counts and
UNK shares come from TRAIN rows only and are recorded; no labels, dates after the TRAIN
cutoff or later-season statistics enter the mapping.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_periodic_residual_model as base

from pitchmdp.direct_pitcher_id_registry import D_ID, MIN_TRAIN_PITCHES, UNK, build_vocab, map_ids  # noqa: F401  torch-free

KIND = 'pitcher_id_periodic_residual_mlp'
FORMAT = 'direct_pitcher_id_periodic_residual_model_v1'
ID_SEED_OFFSET = 7_919  # private generator for P only; keeps the global torch RNG stream equal to the baseline


def id_inputs(inputs, pitcher, ids):
    """(tokens, valid, context + one id-index column) for the arm; returns the UNK row share too."""
    tokens, valid, context = inputs
    index = map_ids(pitcher, ids)
    if len(index) != len(context):
        raise ValueError('Pitcher ids must align with the arm inputs')
    if index.max(initial=0) >= 1 << 24:
        raise ValueError('Vocabulary index is not exact in float32')
    context = np.concatenate([np.asarray(context, dtype=np.float32), index[:, None].astype(np.float32)], axis=1)
    return (tokens, valid, context), float((index == UNK).mean()) if len(index) else 0.


class PitcherIdPeriodicResidualNetwork(base.PeriodicResidualNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, n_ids, id_seed, d_id=D_ID, **kwargs):
        super().__init__(n_context - 1, n_token, length, context_continuous, **kwargs)  # baseline modules first
        width = self.input.out_features
        self.n_ids = int(n_ids)
        self.config.update(kind=KIND, n_context=n_context, n_ids=self.n_ids, id_seed=int(id_seed), d_id=d_id)
        self.id_table = nn.Parameter(torch.zeros(self.n_ids + 1, d_id))  # row 0 = UNK; zeros: no RNG use
        bound = 1 / math.sqrt(d_id)
        g = torch.Generator().manual_seed(int(id_seed))
        self.id_proj = nn.Parameter(torch.empty(width, d_id).uniform_(-bound, bound, generator=g))

    def forward(self, tokens, valid, context):
        index = context[:, -1].long()
        context = context[:, :-1]
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, periodic.N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(),
                         context[:, self.raw]), dim=1)
        z = self.input(torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1))
        z = z + self.id_table[index] @ self.id_proj.T
        for block in self.blocks:
            z = z + block(z)
        return self.output(z)


def _check_index(arrays, n_ids):
    column = np.asarray(arrays[2])[:, -1]
    if np.any(column != np.round(column)) or column.min(initial=0) < 0 or column.max(initial=0) > n_ids:
        raise ValueError('Last context column is not a valid pitcher-id index')


class DirectPitcherIdModel(base.DirectPeriodicResidualModel):
    """Periodic-residual training loop + TRAIN-only pitcher vocabulary (set before fit)."""

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None, vocab_ids=None):
        super().__init__(seed, context_continuous, architecture, device)
        self.kind = KIND
        self.vocab_ids = None if vocab_ids is None else np.asarray(vocab_ids, dtype=np.int64)

    def network(self, n_context, n_token, length):
        if self.vocab_ids is None:
            raise ValueError('Set the TRAIN vocabulary before building the network')
        return PitcherIdPeriodicResidualNetwork(
            n_context, n_token, length, self.context_continuous, len(self.vocab_ids), self.seed + ID_SEED_OFFSET,
            **base.network_kwargs('periodic', self.architecture), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        if self.vocab_ids is None:
            raise ValueError('Set the TRAIN vocabulary before building the network')
        _check_index(train, len(self.vocab_ids)), _check_index(calibration, len(self.vocab_ids))
        return super().fit(train, labels, calibration, calibration_labels, **kwargs)

    def logits(self, arrays, batch_size=4096):
        _check_index(arrays, self.net.n_ids)
        return super().logits(arrays, batch_size)

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': self.kind, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'vocab_ids': torch.as_tensor(self.vocab_ids),
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized pitcher-id model archive')
        config = dict(payload['network_config'])
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device,
                    payload['vocab_ids'].numpy())
        config.pop('kind')
        if config['n_ids'] != len(model.vocab_ids):
            raise ValueError('Archived vocabulary differs from the archived network')
        kwargs = {k: config[k] for k in ('width', 'hidden', 'blocks', 'dropout', 'n_frequencies', 'd_embedding',
                                         'sigma', 'n_classes')}
        model.net = PitcherIdPeriodicResidualNetwork(config['n_context'], config['n_token'], config['length'],
                                                     config['context_continuous'], config['n_ids'],
                                                     config['id_seed'], config['d_id'], **kwargs).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


def vocab_sha256(ids):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()).hexdigest()


def build_pitcher_id(arm, seed, device, vocab_ids):
    """Unfitted candidate for the checked registry arm with its TRAIN vocabulary (the runner builds it from TRAIN)."""
    from pitchmdp.direct_embed_residual_registry import ARMS
    from pitchmdp.direct_pitcher_id_registry import REFERENCE, check_arm
    check_arm(arm)  # exactly the registered candidate entry
    spec = ARMS[REFERENCE]
    return DirectPitcherIdModel(seed, spec['context_continuous'], device=device, vocab_ids=vocab_ids)


def load_pitcher_id(arm, path, device, ids_sha256):
    """Load the candidate; its archived vocabulary must hash to the run-recorded TRAIN vocabulary (fit.json)."""
    from pitchmdp.direct_embed_residual_registry import ARMS
    from pitchmdp.direct_pitcher_id_registry import REFERENCE, check_arm
    check_arm(arm)
    if not ids_sha256:
        raise ValueError('Pitcher-ID load needs the run-recorded vocabulary hash (fit.json next to the archive)')
    model = DirectPitcherIdModel.load(path, device)
    if vocab_sha256(model.vocab_ids) != ids_sha256:
        raise ValueError('Archived vocabulary differs from the run-recorded TRAIN vocabulary')
    if model.context_continuous != list(ARMS[REFERENCE]['context_continuous']):
        raise ValueError('Archived continuous map differs from the registered arm')
    return model
