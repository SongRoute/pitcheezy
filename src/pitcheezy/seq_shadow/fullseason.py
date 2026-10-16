"""Fixed fullseason T CPU shadow. Explicit load only; no fit, feed or service I/O.

Callers own the real-input registration/grant. Source/metadata/state bytes are all
pinned before state deserialization. Prediction takes encoded rows and bounded
past-token provenance, not current realized outcomes or a live feed.
"""
from __future__ import annotations

import importlib
import io
import json
from pathlib import Path
import stat
import sys
import zipfile

import numpy as np
from . import PinMismatch, read_pinned

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'ws-fullseason-shadow-v1'
LEAVES = ('ball', 'foul', 'hit_by_pitch', 'narrow_whiff', 'other_strike',
          'ground_ball_in_play', 'other_in_play')
SEEDS = (0, 1, 2)
ROWS = 100
STATE_CAP = 8 << 20
JSON_CAP = 1 << 20
SOURCE_FILES = (
    'src/pitcheezy/__init__.py', 'src/pitcheezy/seq_shadow/__init__.py',
    'src/pitcheezy/seq_shadow/fullseason.py', 'src/pitcheezy/seq_shadow/adapter.py',
    'src/pitcheezy/seq_shadow/calibration.py', 'experiments/ws_campaign/__init__.py',
    'experiments/ws_campaign/models.py', 'experiments/ws_seq/train_seq.py',
    'experiments/ws_seq/tm002.py', 'experiments/ws_fullseason/data.py',
    'experiments/ws_fullseason/calibrate.py')
TOKEN_FIELDS = ('type', 'out', 'balls', 'strikes', 'same', 'pos')


def read_artifact(binding, cap):
    """Bounded regular file, one verified byte buffer; never reopen to decode."""
    if set(binding) != {'path', 'bytes', 'sha256'}:
        raise PinMismatch('exact artifact binding required')
    p = Path(binding['path']); size = binding['bytes']
    if type(size) is not int or not 0 < size <= cap or not stat.S_ISREG(p.lstat().st_mode):
        raise PinMismatch('artifact size/type')
    raw = read_pinned(p, binding['sha256'], cap=size)
    if len(raw) != size:
        raise PinMismatch('artifact byte length')
    return raw


def _sources(pins):
    if set(pins) != set(SOURCE_FILES):
        raise PinMismatch('exact source closure required')
    for name, digest in pins.items():
        read_pinned(ROOT / name, digest, cap=JSON_CAP)
    # A fixed checkout must not silently reuse modules imported from another tree.
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__name__', '')
        if name.startswith(('experiments.ws_campaign', 'experiments.ws_seq',
                            'experiments.ws_fullseason', 'pitcheezy.seq_shadow')):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(ROOT):
                raise PinMismatch('foreign scientific module already imported')


def _encoder(e):
    try:
        cols = e['continuous']; knots = np.asarray(e['knots'], dtype=float)
        values = [np.asarray(e[k], dtype=float) for k in ('median', 'mean', 'scale')]
        ok = (e['dimension'] == 199 and e.get('z_clip', 8) == 8
              and len(cols) == 36 and len(set(cols)) == 36
              and all(type(i) is int and 0 <= i < 91 for i in cols)
              and knots.shape == (3, 36) and np.isfinite(knots).all()
              and all(v.shape == (91,) and np.isfinite(v).all() for v in values)
              and (values[2] > 0).all())
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok: raise PinMismatch('fixed 91-to-199 encoder geometry')


def load(config):
    """Load three fixed T members. Real calls require the external reviewed grant."""
    if (config['schema'] != SCHEMA or config['seeds'] != list(SEEDS)
            or config['leaf_order'] != list(LEAVES) or config['threads'] != 1):
        raise PinMismatch('fixed shadow identity')
    _sources(config['source_pins'])
    if set(config['state']) != {'0', '1', '2'}:
        raise PinMismatch('exact three seed states required')
    state_raw = {s: read_artifact(config['state'][str(s)], STATE_CAP) for s in SEEDS}
    docs = {name: json.loads(read_artifact(config[name], JSON_CAP))
            for name in ('recipe', 'encoder', 'calibrator')}
    recipe, encoder, calibrator = (docs[k] for k in ('recipe', 'encoder', 'calibrator'))
    bc = recipe['base']; ctx = bc['context']
    if (recipe['arm'] != 'T' or recipe['seeds'] != list(SEEDS)
            or recipe['execution_stage'] != 'FINAL_REFIT'
            or recipe['data_role'] != '2026 EXPOSED DEVELOPMENT'
            or bc['inputs']['feature_dimension'] != 199 or ctx['max_length'] != 32
            or ctx['position_vocab'] != 33 or ctx['same_batter_vocab'] != 2
            or bc['training']['precision'] != 'fp32'):
        raise PinMismatch('fixed final T recipe')
    if ctx != dict(max_length=32, pitch_type_vocab=19, outcome_vocab=10, balls_vocab=4,
                   strikes_vocab=3, same_batter_vocab=2, position_vocab=33):
        raise PinMismatch('fixed token vocabularies')
    model = bc['model']
    dimensions = [*model['trunk_hidden'], model['token_dim'], model['encoder_ff']]
    if (len(model['trunk_hidden']) != 2 or any(type(n) is not int or not 0 < n <= 256 for n in dimensions)
            or type(model['encoder_layers']) is not int or not 0 < model['encoder_layers'] <= 2
            or type(model['encoder_heads']) is not int or not 0 < model['encoder_heads'] <= 4
            or model['token_dim'] % model['encoder_heads']):
        raise PinMismatch('bounded model architecture')
    _encoder(encoder)
    cal = importlib.import_module('experiments.ws_fullseason.calibrate')
    # Validate the stored typed calibrator before any torch.load.
    cal.apply({s: np.full((1, 7), 1 / 7) for s in SEEDS}, calibrator)
    count, parameters = config['expected_state_tensors'], config['expected_parameters']
    if (type(count) is not int or not 0 < count <= 512
            or type(parameters) is not int or not 0 < parameters <= 1_000_000):
        raise PinMismatch('state geometry cap')
    for raw in state_raw.values():
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info = z.infolist()
            if (len(info) > 1024 or len({v.filename for v in info}) != len(info)
                    or sum(v.file_size for v in info) > STATE_CAP
                    or any(v.compress_type != zipfile.ZIP_STORED or v.flag_bits & 1 for v in info)):
                raise PinMismatch('bounded stored state archive required')
    import torch
    from experiments.ws_campaign.models import build_model
    torch.set_num_threads(1)
    nets = {}
    for seed in SEEDS:
        state = torch.load(io.BytesIO(state_raw[seed]), map_location='cpu', weights_only=True)
        if (not isinstance(state, dict) or len(state) != count
                or not all(isinstance(v, torch.Tensor) and v.layout == torch.strided for v in state.values())
                or sum(v.numel() for v in state.values()) != parameters
                or not all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in state.values())):
            raise PinMismatch('state tensor count/dtype/parameters/finiteness')
        net = build_model(recipe, 'T', seed).cpu()
        net.load_state_dict(state, strict=True)
        if sum(v.numel() for v in net.state_dict().values()) != parameters:
            raise PinMismatch('built model geometry')
        nets[seed] = net.eval()
    return Ensemble(nets, recipe, encoder, calibrator, cal)


class Ensemble:
    def __init__(self, nets, recipe, encoder, calibrator, cal):
        self.nets, self.recipe, self.encoder, self.calibrator, self.cal = nets, recipe, encoder, calibrator, cal

    def encode(self, raw):
        from .adapter import encode_for_seq
        raw = np.asarray(raw)
        if raw.ndim != 2 or not 0 < len(raw) <= ROWS or raw.shape[1] != 91 or raw.dtype.kind not in 'fiu':
            raise ValueError('bounded raw91 matrix')
        return encode_for_seq(raw, self.encoder)

    def predict(self, x, tokens, keys):
        import torch
        from experiments.ws_seq.train_seq import seven_from_margins
        x = np.asarray(x); keys = np.asarray(keys)
        if x.ndim != 2 or not 0 < len(x) <= ROWS or x.shape[1] != 199 or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('bounded finite float32 encoded199 matrix')
        b = len(x); self.cal.checked_keys(keys)
        if keys.shape != (b, 3): raise ValueError('query key shape')
        required = set(TOKEN_FIELDS) | {'valid', 'context_keys', 'context_pitcher', 'query_pitcher'}
        if set(tokens) != required: raise ValueError('exact tokens and temporal provenance')
        tok = {k: np.asarray(v) for k, v in tokens.items()}
        valid = tok['valid']; ck = tok['context_keys']; cp = tok['context_pitcher']; qp = tok['query_pitcher']
        if (valid.shape != (b, 32) or valid.dtype != bool or ck.shape != (b, 32, 3)
                or cp.shape != (b, 32) or qp.shape != (b,)
                or any(a.dtype != np.int64 for a in (ck, cp, qp)) or (qp <= 0).any()):
            raise ValueError('token provenance shape/type')
        prior = (ck[:, :, 1] < keys[:, None, 1]) | ((ck[:, :, 1] == keys[:, None, 1]) & (ck[:, :, 2] < keys[:, None, 2]))
        recent = (ck[:, 1:, 1] < ck[:, :-1, 1]) | ((ck[:, 1:, 1] == ck[:, :-1, 1]) & (ck[:, 1:, 2] < ck[:, :-1, 2]))
        if (not ((~valid) | ((ck > 0).all(2) & (ck[:, :, 0] == keys[:, None, 0]) & (cp == qp[:, None]) & prior)).all()
                or (valid[:, 1:] & ~valid[:, :-1]).any() or (valid[:, 1:] & ~recent).any()):
            raise ValueError('strict prior same-game/pitcher recent-first context')
        vocab = self.recipe['base']['context']
        sizes = dict(type=vocab['pitch_type_vocab'], out=vocab['outcome_vocab'], balls=vocab['balls_vocab'],
                     strikes=vocab['strikes_vocab'], same=vocab['same_batter_vocab'], pos=vocab['position_vocab'])
        for name in TOKEN_FIELDS:
            a = tok[name]
            if a.shape != (b, 32) or a.dtype != np.int64 or (a < 0).any() or (a >= sizes[name]).any():
                raise ValueError('token shape/type/vocabulary: ' + name)
        if not (tok['pos'] == np.arange(1, 33)[None, :]).all(): raise ValueError('fixed token position')
        tensors = {k: torch.as_tensor(np.ascontiguousarray(tok[k])) for k in (*TOKEN_FIELDS, 'valid')}
        xt = torch.as_tensor(np.ascontiguousarray(x)); kt = torch.as_tensor(np.ascontiguousarray(keys, dtype=np.int64))
        raw = {}
        with torch.no_grad():
            for seed, net in self.nets.items():
                z = {k: v.double().numpy() for k, v in net(xt, tensors, kt).items()}
                if (set(z) != {'parent', 'narrow_whiff', 'ground_ball_per_pitch'}
                        or z['parent'].shape != (b, 10) or z['narrow_whiff'].shape != (b,)
                        or z['ground_ball_per_pitch'].shape != (b,) or not all(np.isfinite(v).all() for v in z.values())):
                    raise ValueError('finite fixed head margins')
                raw[seed] = seven_from_margins(z['parent'], z['narrow_whiff'], z['ground_ball_per_pitch'])
        return {'raw_by_seed': raw, 'raw_mean': self.cal.mean_seeds(raw), 'calibrated': self.cal.apply(raw, self.calibrator)}
