"""Two id paths (pitcher + batter) on the frozen periodic-residual direct head (EXP-P43-001 DRAFT arm
pitcher_batter_id_periodic_residual_counts).

  z = Linear_in(cat(raw, e)) + P_p @ T_p[pitcher idx] + P_b @ T_b[batter idx]
  rest = the frozen PeriodicResidualNetwork backbone

The pitcher path is the P38 path verbatim (same zero-init table, projection drawn from the private generator
seed + ID_SEED_OFFSET).  The batter path is the same construction with its own private generator
(seed + ID_SEED_OFFSET + BATTER_SEED_OFFSET); the single-column batter arms use DirectPitcherIdModel unchanged with
id_seed = seed + ID_SEED_OFFSET + BATTER_SEED_OFFSET (see batter_id_seed), so at one model seed the batter
projection is identical in all three P43 id arms.  The two index columns travel as the last two context columns
(pitcher, batter), exact float32 integers; the network strips both before the frozen raw / continuous maps.  Both
tables start at zero, so at a seed the untrained network returns exactly the reference logits. P43 has no ID dropout;
P47 adds optional per-column training dropout and a permanent zero-UNK gate without changing P43 defaults.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pitchmdp import direct_periodic_model as periodic
from pitchmdp import direct_periodic_residual_model as base
from pitchmdp.direct_pitcher_id_model import (ID_SEED_OFFSET, ID_DROPOUT_SEED_OFFSET, DirectPitcherIdModel,
                                             PitcherIdPeriodicResidualNetwork, vocab_sha256)
from pitchmdp.direct_pitcher_id_registry import D_ID, UNK

KIND = 'pitcher_batter_id_periodic_residual_mlp'
FORMAT = 'direct_pitcher_batter_id_periodic_residual_model_v1'
BATTER_SEED_OFFSET = 1_009  # batter projection generator = seed + ID_SEED_OFFSET + 1009 (private; global RNG untouched)
COLUMNS = ('pitcher', 'batter')


def batter_id_seed(seed):
    return int(seed) + ID_SEED_OFFSET + BATTER_SEED_OFFSET


def _projection(width, d_id, seed):
    bound = 1 / math.sqrt(d_id)
    g = torch.Generator().manual_seed(int(seed))
    return nn.Parameter(torch.empty(width, d_id).uniform_(-bound, bound, generator=g))


class PitcherBatterIdPeriodicResidualNetwork(base.PeriodicResidualNetwork):
    def __init__(self, n_context, n_token, length, context_continuous, n_ids, id_seeds, d_id=D_ID,
                 id_dropout=0., freeze_unk=False, **kwargs):
        super().__init__(n_context - 2, n_token, length, context_continuous, **kwargs)  # baseline modules first
        width = self.input.out_features
        self.n_ids = [int(n) for n in n_ids]
        if len(self.n_ids) != 2 or len(id_seeds) != 2:
            raise ValueError('Exactly two id paths (pitcher, batter)')
        self.config.update(kind=KIND, n_context=n_context, n_ids=self.n_ids, id_seeds=[int(s) for s in id_seeds],
                           d_id=d_id)
        self.id_dropout, self.freeze_unk = float(id_dropout), bool(freeze_unk)
        if not 0. <= self.id_dropout < 1.:
            raise ValueError('id_dropout must be in [0, 1)')
        if self.id_dropout:  # additive P47 fields only; P43 archive/config unchanged
            self.config['id_dropout'] = self.id_dropout
        if self.freeze_unk:
            self.config['freeze_unk'] = True
        self._id_generators = [None, None]
        self.id_tables = nn.ParameterList(nn.Parameter(torch.zeros(n + 1, d_id)) for n in self.n_ids)  # row 0 = UNK
        self.id_projs = nn.ParameterList(_projection(width, d_id, s) for s in id_seeds)

    def dropout_indices(self, index):
        """Independent per-column private device streams; evaluation always uses true indices."""
        if not self.training or not self.id_dropout:
            return index
        columns = []
        for j in range(2):
            generator = self._id_generators[j]
            if generator is None or generator.device != index.device:
                generator = torch.Generator(device=index.device).manual_seed(
                    self.config['id_seeds'][j] + ID_DROPOUT_SEED_OFFSET)
                self._id_generators[j] = generator
            drop = torch.rand(index[:, j].shape, device=index.device, generator=generator) < self.id_dropout
            columns.append(index[:, j].masked_fill(drop, UNK))
        return torch.stack(columns, dim=1)

    def forward(self, tokens, valid, context):
        index = self.dropout_indices(context[:, -2:].long())
        context = context[:, :-2]
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        raw = torch.cat((tokens[:, :-1, periodic.N_PHYSICAL:].flatten(1), tokens[:, -1], valid.float(),
                         context[:, self.raw]), dim=1)
        z = self.input(torch.cat((raw, self.embed(tokens, valid, context).flatten(1)), dim=1))
        for j, (table, proj) in enumerate(zip(self.id_tables, self.id_projs)):
            embedding = table[index[:, j]]
            if self.freeze_unk:
                # Zero-initialized row 0 has zero gradient; AdamW decay keeps it exactly zero.
                # The gate applies at training AND evaluation, including unknown true IDs.
                embedding = embedding * (index[:, j] != UNK)[:, None]
            z = z + embedding @ proj.T
        for block in self.blocks:
            z = z + block(z)
        return self.output(z)


def _check_indices(arrays, n_ids):
    columns = np.asarray(arrays[2])[:, -2:]
    for j, n in enumerate(n_ids):
        c = columns[:, j]
        if np.any(c != np.round(c)) or c.min(initial=0) < 0 or c.max(initial=0) > n:
            raise ValueError(f'Context column {COLUMNS[j]} is not a valid id index')


class DirectPitcherBatterIdModel(base.DirectPeriodicResidualModel):
    """Periodic-residual training loop + TRAIN-only pitcher and batter vocabularies (set before fit)."""

    def __init__(self, seed=0, context_continuous=(), architecture=None, device=None, vocab_ids=None,
                 id_dropout=0., freeze_unk=False):
        super().__init__(seed, context_continuous, architecture, device)
        self.kind = KIND
        self.vocab_ids = None if vocab_ids is None else [np.asarray(v, dtype=np.int64) for v in vocab_ids]
        self.id_dropout, self.freeze_unk = float(id_dropout), bool(freeze_unk)

    def id_seeds(self):
        return [self.seed + ID_SEED_OFFSET, batter_id_seed(self.seed)]

    def network(self, n_context, n_token, length):
        if self.vocab_ids is None or len(self.vocab_ids) != 2:
            raise ValueError('Set the TRAIN pitcher and batter vocabularies before building the network')
        return PitcherBatterIdPeriodicResidualNetwork(
            n_context, n_token, length, self.context_continuous, [len(v) for v in self.vocab_ids], self.id_seeds(),
            id_dropout=self.id_dropout, freeze_unk=self.freeze_unk,
            **base.network_kwargs('periodic', self.architecture), n_classes=self.n_classes)

    def fit(self, train, labels, calibration, calibration_labels, **kwargs):
        if self.vocab_ids is None:
            raise ValueError('Set the TRAIN vocabularies before building the network')
        n = [len(v) for v in self.vocab_ids]
        _check_indices(train, n), _check_indices(calibration, n)
        return super().fit(train, labels, calibration, calibration_labels, **kwargs)

    def logits(self, arrays, batch_size=4096):
        _check_indices(arrays, self.net.n_ids)
        return super().logits(arrays, batch_size)

    def save(self, path):
        torch.save({'format': FORMAT, 'kind': self.kind, 'seed': self.seed, 'architecture': self.architecture,
                    'temperature': self.temperature, 'delivery_temperature': self.delivery_temperature,
                    'report': self.report, 'network_config': self.net.config,
                    'vocab_ids': [torch.as_tensor(v) for v in self.vocab_ids],
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}}, Path(path))

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(Path(path), map_location='cpu', weights_only=True)
        if payload.get('format') != FORMAT or payload.get('kind') != KIND:
            raise ValueError('Unrecognized pitcher-batter-id model archive')
        config = dict(payload['network_config'])
        model = cls(payload['seed'], config['context_continuous'], payload['architecture'], device,
                    [v.numpy() for v in payload['vocab_ids']], config.get('id_dropout', 0.), config.get('freeze_unk', False))
        if config['n_ids'] != [len(v) for v in model.vocab_ids] or config['id_seeds'] != model.id_seeds():
            raise ValueError('Archived vocabularies / id seeds differ from the archived network')
        kwargs = {k: config[k] for k in ('width', 'hidden', 'blocks', 'dropout', 'n_frequencies', 'd_embedding',
                                         'sigma', 'n_classes')}
        model.net = PitcherBatterIdPeriodicResidualNetwork(config['n_context'], config['n_token'], config['length'],
                                                           config['context_continuous'], config['n_ids'],
                                                           config['id_seeds'], config['d_id'],
                                                           id_dropout=model.id_dropout, freeze_unk=model.freeze_unk,
                                                           **kwargs).to(model.device)
        model.net.load_state_dict(payload['state_dict'])
        if model.freeze_unk and any(torch.count_nonzero(t[UNK]).item() for t in model.net.id_tables):
            raise ValueError('Frozen UNK rows must be exactly zero in the archive')
        model.temperature, model.delivery_temperature = payload['temperature'], payload['delivery_temperature']
        model.report = payload['report']
        return model


def build_pitcher_batter_id(arm, seed, device, vocab_ids):
    from pitchmdp.direct_embed_residual_registry import ARMS
    from pitchmdp.direct_pitcher_id_registry import REFERENCE, FLAG, id_columns
    if id_columns(arm) != list(COLUMNS):
        raise ValueError('Only the registered pitcher+batter arm builds two id paths')
    return DirectPitcherBatterIdModel(seed, ARMS[REFERENCE]['context_continuous'], device=device, vocab_ids=vocab_ids,
                                     id_dropout=arm[FLAG].get('id_dropout', 0.), freeze_unk=arm[FLAG].get('freeze_unk', False))


def load_pitcher_batter_id(arm, path, device, ids_sha256):
    """Load; each archived vocabulary must hash to the run-recorded TRAIN vocabulary of its column (fit.json)."""
    from pitchmdp.direct_embed_residual_registry import ARMS
    from pitchmdp.direct_pitcher_id_registry import REFERENCE, FLAG, id_columns
    if id_columns(arm) != list(COLUMNS):
        raise ValueError('Archive / arm mismatch: pitcher+batter archive needs its registered arm')
    if not isinstance(ids_sha256, dict) or set(ids_sha256) != set(COLUMNS):
        raise ValueError('Pitcher+batter load needs the run-recorded vocabulary hash of both columns')
    model = DirectPitcherBatterIdModel.load(path, device)
    if model.id_dropout != arm[FLAG].get('id_dropout', 0.) or model.freeze_unk != arm[FLAG].get('freeze_unk', False):
        raise ValueError('Archived dropout / UNK gate differs from the registered arm')
    for column, ids in zip(COLUMNS, model.vocab_ids):
        if vocab_sha256(ids) != ids_sha256[column]:
            raise ValueError(f'Archived {column} vocabulary differs from the run-recorded TRAIN vocabulary')
    if model.context_continuous != list(ARMS[REFERENCE]['context_continuous']):
        raise ValueError('Archived continuous map differs from the registered arm')
    return model


def append_indices(inputs, indices):
    """(tokens, valid, context + one exact float32 index column per id column, in COLUMNS order)."""
    tokens, valid, context = inputs
    index = np.stack([np.asarray(i, dtype=np.int64) for i in indices], axis=1)
    if len(index) != len(context):
        raise ValueError('Id indices must align with the arm inputs')
    if index.max(initial=0) >= 1 << 24 or index.min(initial=0) < UNK:
        raise ValueError('Vocabulary index is not exact in float32')
    return tokens, valid, np.concatenate([np.asarray(context, dtype=np.float32), index.astype(np.float32)], axis=1)


class DirectBatterIdModel(DirectPitcherIdModel):
    """Single batter id path: the P38 network and archive format unchanged, only the projection generator seed is the
    batter one (batter_id_seed), so the batter projection matches the batter path of the pitcher+batter arm."""

    def network(self, n_context, n_token, length):
        if self.vocab_ids is None:
            raise ValueError('Set the TRAIN vocabulary before building the network')
        return PitcherIdPeriodicResidualNetwork(
            n_context, n_token, length, self.context_continuous, len(self.vocab_ids), batter_id_seed(self.seed),
            id_dropout=self.id_dropout, **base.network_kwargs('periodic', self.architecture), n_classes=self.n_classes)


def build_batter_id(arm, seed, device, vocab_ids):
    from pitchmdp.direct_embed_residual_registry import ARMS
    from pitchmdp.direct_pitcher_id_registry import BATTER, BATTER_SHUFFLE, REFERENCE, id_columns, id_spec
    if arm.get('name') not in (BATTER, BATTER_SHUFFLE) or id_columns(arm) != ['batter'] or id_spec(arm)[2]:
        raise ValueError('Only the registered single-column batter arms build one batter id path')
    return DirectBatterIdModel(seed, ARMS[REFERENCE]['context_continuous'], device=device, vocab_ids=vocab_ids)


def load_batter_id(arm, path, device, ids_sha256):
    """P38 archive format; vocabulary hash = run-recorded batter vocabulary, id seed = batter_id_seed(archived seed)."""
    from pitchmdp.direct_pitcher_id_model import load_pitcher_id
    if not isinstance(ids_sha256, dict) or set(ids_sha256) != {'batter'}:
        raise ValueError('Batter-ID load needs the run-recorded batter vocabulary hash')
    model = load_pitcher_id(arm, path, device, ids_sha256['batter'])
    if model.net.config['id_seed'] != batter_id_seed(model.seed):
        raise ValueError('Archived batter id seed differs from the registered construction')
    return model
