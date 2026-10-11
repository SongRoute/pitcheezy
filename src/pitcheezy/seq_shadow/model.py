"""EXP-WS-SEQ-001 arm N1, three seeds: hash-checked loader and CPU forward on one thread.

Order of the loader, so that no model byte reaches ``torch.load`` unless everything registered matches:
  1. every registered file is read ONCE into memory and hashed (three ``model-state.pt``, three
     ``temperatures.json``, the architecture config, ``train_seq.py``); any mismatch raises ``PinMismatch``;
  2. torch is imported and set to one thread;
  3. ``train_seq.py`` is imported by file path (numpy at import, torch only inside ``build_model``) and
     ``train_seq.load_config`` validates the architecture config against what that source implements;
  4. ``torch.load(<the verified bytes>, map_location='cpu', weights_only=True)``, tensor count and parameter count
     checked against the registration, strict ``load_state_dict``, ``eval()``.

Forward: ``zp, zw, zg = net(x, tok)`` with x float32 (rows, 199) and tok int64 tensors (rows, L) plus bool ``valid``
exactly as ``train_seq.Data.tokens`` yields them. Margins are returned as float64 numpy arrays.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

from . import PinMismatch, read_pinned

TRAIN_MODULE_NAME = 'pitcheezy_seq_shadow_pinned_train_seq'
TOKEN_FIELDS = ('type', 'out', 'balls', 'strikes', 'same', 'pos')
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')


def _train_module(path, raw):
    """``train_seq.py`` executed from the verified bytes (never re-read from disk after the hash check)."""
    held = sys.modules.get(TRAIN_MODULE_NAME)
    if held is not None and getattr(held, '_pinned_sha256', None) == hashlib.sha256(raw).hexdigest():
        return held
    spec = importlib.util.spec_from_loader(TRAIN_MODULE_NAME, loader=None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)  # noqa: S102  (hash-pinned, reviewed source)
    module._pinned_sha256 = hashlib.sha256(raw).hexdigest()
    sys.modules[TRAIN_MODULE_NAME] = module
    return module


class Ensemble:
    def __init__(self, torch, nets, temperatures, architecture, sha256, load_ms, seeds):
        self.torch, self.nets, self.temperatures, self.architecture = torch, nets, temperatures, architecture
        self.sha256, self.load_ms, self.seeds = sha256, load_ms, list(seeds)
        self.length = architecture['context']['max_length']
        self.dimension = architecture['inputs']['feature_dimension']

    def tensors(self, x, tokens, rows):
        """(x tensor, token tensors) for ``rows`` candidate rows that share one context: the (L,) token arrays are
        tiled to (rows, L)."""
        import numpy as np
        torch = self.torch
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.shape != (rows, self.dimension) or not np.isfinite(x).all():
            raise ValueError('feature matrix shape or finiteness')
        tok = {}
        for name in TOKEN_FIELDS:
            a = np.asarray(tokens[name])
            if a.shape != (self.length,) or a.dtype != np.int64:
                raise ValueError('token array shape or dtype: '+name)
            tok[name] = torch.as_tensor(np.ascontiguousarray(np.tile(a, (rows, 1))), dtype=torch.int64)
        valid = np.asarray(tokens['valid'])
        if valid.shape != (self.length,) or valid.dtype != bool:
            raise ValueError('token valid mask shape or dtype')
        tok['valid'] = torch.as_tensor(np.ascontiguousarray(np.tile(valid, (rows, 1))), dtype=torch.bool)
        vocab = self.architecture['context']
        for name, size in (('type', vocab['pitch_type_vocab']), ('out', vocab['outcome_vocab']),
                           ('balls', vocab['balls_vocab']), ('strikes', vocab['strikes_vocab']),
                           ('same', vocab['same_batter_vocab'])):
            a = np.asarray(tokens[name])[valid]
            if len(a) and (a.min() < 0 or a.max() >= size):
                raise ValueError('token value outside its vocabulary: '+name)
        return torch.as_tensor(x), tok

    def forward(self, x, tok):
        """{seed: {head: float64 numpy}} from prepared tensors, eval mode, no grad."""
        import numpy as np
        out = {}
        with self.torch.no_grad():
            for seed, net in zip(self.seeds, self.nets):
                zp, zw, zg = net(x, tok)
                z = {'parent': zp.double().numpy(), 'narrow_whiff': zw.double().numpy(),
                     'ground_ball_per_pitch': zg.double().numpy()}
                if not all(np.isfinite(v).all() for v in z.values()):
                    raise ValueError('non-finite margin')
                out[seed] = z
        return out


def load_ensemble(config):
    """The registered N1 seeds as an ``Ensemble``. ``config`` is the whole seq shadow configuration."""
    m = config['model']
    seeds = [int(s) for s in m['seeds']]
    start = time.perf_counter()
    # 1. read once + hash everything, before torch exists in this function
    state_raw = {s: read_pinned(m['state'][str(s)]['path'], m['state'][str(s)]['sha256'], m['max_state_bytes'])
                 for s in seeds}
    temp_raw = {s: read_pinned(m['temperatures'][str(s)]['path'], m['temperatures'][str(s)]['sha256']) for s in seeds}
    arch_raw = read_pinned(m['architecture']['path'], m['architecture']['sha256'])
    train_raw = read_pinned(m['train_seq']['path'], m['train_seq']['sha256'])
    # 2. torch, one thread
    import torch
    torch.set_num_threads(config['threads'])
    # 3. pinned source + architecture
    T = _train_module(Path(m['train_seq']['path']), train_raw)
    architecture = json.loads(arch_raw)
    try:
        c, mm, tt = architecture['context'], architecture['model'], architecture['training']
        consistent = [architecture['id'] == T.EXP_ID, m['arm'] in architecture['arms'],
                      architecture['inputs']['feature_dimension'] == m['feature_dimension'],
                      tuple(architecture['labels']['bip_classes']) == T.BIP, architecture['labels']['parent_classes'] == 10,
                      architecture['labels']['whiff_parent_class'] == 1, c['position_vocab'] == c['max_length']+1,
                      mm['token_dim'] % mm['encoder_heads'] == 0, len(mm['trunk_hidden']) == 2,
                      mm['activation'] == 'gelu' and mm['encoder_activation'] == 'gelu', mm['layer_norm'] is True,
                      tt['precision'] == 'fp32', c['same_batter_vocab'] == 2]  # train_seq.load_config lines 77-81
    except (KeyError, TypeError) as exc:
        raise PinMismatch(f'architecture config lacks a field: {exc}') from None
    if not all(consistent):
        raise PinMismatch('architecture config does not match what train_seq implements: '+str(consistent))
    temperatures = {}
    for s in seeds:
        t = json.loads(temp_raw[s])['temperatures']
        if set(t) != set(HEADS) or not all(isinstance(t[h], float) and t[h] > 0 for h in HEADS):
            raise PinMismatch('temperatures.json shape')
        temperatures[s] = {h: t[h] for h in HEADS}
    # 4. weights from the verified bytes
    nets = []
    for s in seeds:
        state = torch.load(io.BytesIO(state_raw[s]), map_location='cpu', weights_only=True)
        tensors = len(state)
        parameters = sum(int(v.numel()) for v in state.values())
        if tensors != m['expected_state_tensors'] or parameters != m['expected_parameters']:
            raise PinMismatch(f'state dict has {tensors} tensors / {parameters} parameters, not the registered '
                              f"{m['expected_state_tensors']} / {m['expected_parameters']}")
        net = T.build_model(torch, architecture, m['arm'])
        net.load_state_dict(state, strict=True)
        net.eval()
        if sum(int(p.numel()) for p in net.parameters()) != m['expected_parameters']:
            raise PinMismatch('built network parameter count differs from the registration')
        nets.append(net)
    del state_raw
    sha = {'state': {str(s): m['state'][str(s)]['sha256'] for s in seeds},
           'temperatures': {str(s): m['temperatures'][str(s)]['sha256'] for s in seeds},
           'architecture': m['architecture']['sha256'], 'train_seq': m['train_seq']['sha256']}
    return Ensemble(torch, nets, temperatures, architecture, sha, (time.perf_counter()-start)*1000, seeds)
