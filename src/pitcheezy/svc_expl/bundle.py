"""EXP-SVC-PRED-001 extension (EXPL): an EXPL-1121 arm (X-GRU-*, X-TAK, X-GBT, X-DATA ...) saved by ``expl.fitw`` as a Bundle.

Interface as ``pitcheezy.svc_p9.predictor`` / ``pitcheezy.svc_lit.bundle``. ``data.py`` / ``models.py`` here are byte copies of
e83c5e7a:experiments/lit_base/{data,models}.py (exp/lit-base-final-1121; build_seq gru path identical to the fitw commits
dc10b68f / eb39468a, plus gru_attn). The predict math is ``expl.final_predict`` (e83c5e7a) on one query batch:
  seq: models.seq_load(spec, model-state-seed{s}.pt) + models.seq_predict_inputs per seed; mean, renormalise, temper(spec tau)
  gbt: TargetEncoder from target-encoder.npz -> data.tabular features -> LightGBM model-seed0.txt (torch-free worker); temper
balls / strikes / stand from state11 with the pinned package Encoder11. ``kind`` 'preq' (X-PREQ, as-of within-season counts)
is not supported here: its inputs (earlier-day labels of the same pitcher) are not part of the W1 v4 live inputs.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from . import data as D
from . import models as M

CLASSES = D.CLASSES
TOKEN_KEYS = ('valid', 'type', 'out', 'balls', 'strikes', 'same')
TE_FIELDS = ('glob', 'pu', 'pc', 'pcu', 'pcc', 'psu', 'psc', 'bu', 'bc')
KINDS = ('seq', 'gbt')


class ExplBundleError(RuntimeError):
    pass


def _need(c, what):
    if not c:
        raise ExplBundleError(what)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def activate_snapshot(*a, **k):
    from pitcheezy.svc_p9 import predictor as P9
    return P9.activate_snapshot(*a, **k)


def check_loaded_modules(*a, **k):
    from pitcheezy.svc_p9 import predictor as P9
    return P9.check_loaded_modules(*a, **k)


def files_for(spec):
    if spec['kind'] == 'seq':
        return ['spec.json', 'vocab.npz'] + [f'model-state-seed{s}.pt' for s in spec['seeds']]
    return ['spec.json', 'model-seed0.txt', 'target-encoder.npz']


def make_pin(fitw_dir, state_encoder):
    """svc-expl-fitw-v1 pin for an expl.fitw directory (sha of every file the Bundle reads)."""
    d = pathlib.Path(fitw_dir); spec = json.loads((d / 'spec.json').read_text())
    _need(spec['kind'] in KINDS, f"kind {spec['kind']} not supported")
    _need((d / 'DONE').exists(), 'fitw directory not DONE')
    return {'format': 'svc-expl-fitw-v1', 'family': spec['family'], 'kind': spec['kind'], 'dir': str(d), 'hp': spec['hp'], 'tau': spec['tau'],
            'seeds': spec['seeds'], 'data': spec.get('data'), 'state_encoder': state_encoder,
            'files_sha256': {n: sha256_file(d / n) for n in files_for(spec)}}


class Bundle:
    def __init__(self, pin, spec, parts, report):
        self.pin, self.spec, self.parts, self.report = pin, spec, parts, report

    @classmethod
    def load(cls, config, threads=2, candidate=None):
        pin = json.loads(pathlib.Path(config).read_text())
        _need(pin.get('format') == 'svc-expl-fitw-v1' and pin.get('kind') in KINDS, 'not an svc-expl-fitw-v1 pin')
        _need(candidate in (None, pin['family']), f"candidate {candidate} differs from the pin family {pin['family']}")
        root = pathlib.Path(pin['dir'])
        for name, want in pin['files_sha256'].items():
            _need(sha256_file(root / name) == want, f'fitw file sha differs: {name}')
        _need(sha256_file(pin['state_encoder']['path']) == pin['state_encoder']['sha256'], 'state encoder sha')
        spec = json.loads((root / 'spec.json').read_text())
        _need((spec['family'], spec['kind'], spec['hp'], spec['tau'], spec['seeds']) == (pin['family'], pin['kind'], pin['hp'], pin['tau'], pin['seeds']),
              'spec.json differs from the pin')
        _need(sorted(pin['files_sha256']) == sorted(files_for(spec)), 'pin file list')
        parts = {}
        if spec['kind'] == 'seq':
            import torch
            torch.set_num_threads(int(threads))
            _need(sorted(f['seed'] for f in spec['fits']) == sorted(spec['seeds']), 'all seeds saved')
            v = np.load(root / 'vocab.npz', allow_pickle=False)
            parts['pv'], parts['bv'] = v['pitcher'].astype(np.int64), v['batter'].astype(np.int64)
            nets = []
            for f in spec['fits']:
                p = root / f"model-state-seed{f['seed']}.pt"
                _need(sha256_file(p) == f['state_sha256'], 'weights sha vs spec')
                nets.append(M.seq_load(spec, p, 'cpu'))
            parts['nets'] = nets
        else:
            from pitcheezy.svc_lit.bundle import GbtWorker
            t = np.load(root / 'target-encoder.npz', allow_pickle=False)
            te = D.TargetEncoder(float(t['alpha']))
            for k in TE_FIELDS:
                setattr(te, k, t[k])
            parts['te'] = te; parts['booster'] = GbtWorker(root / 'model-seed0.txt', threads)
        report = {'candidate': pin['family'], 'fold': 'fullseason', 'recipe_sha256': pin['files_sha256']['spec.json'],
                  'calibration': {'kind': 'single temperature on the renormalised seed mean (EXPL DEV tau)', 'tau': spec['tau']},
                  'components': [{'name': f"{pin['family']}-seed{s}", 'kind': spec['kind'], 'seed': s, 'source_commit': (spec.get('commit') or [None])[0]}
                                 for s in spec['seeds']]}
        return cls(pin, spec, parts, report)

    def raw_q(self, state11, pitcher, batter, tokens):
        state11 = np.asarray(state11, np.float32)
        _need(state11.ndim == 2 and state11.shape[1] == 11 and np.isfinite(state11).all(), 'state11 shape')
        pitcher = np.asarray(pitcher, np.int64); batter = np.asarray(batter, np.int64)
        tok = {k: np.ascontiguousarray(tokens[k]) for k in TOKEN_KEYS}
        if self.spec['kind'] == 'seq':
            pidx = D.index(pitcher, self.parts['pv']); bidx = D.index(batter, self.parts['bv'])
            qs = [M.seq_predict_inputs(n, state11, pidx, bidx, tok, 'cpu') for n in self.parts['nets']]
            known = pidx > 0
        else:
            balls, strikes, stand, _ = D.raw_from_state(state11, self.pin['state_encoder']['path'])
            enc = self.parts['te'].transform(pitcher, batter, balls, strikes, stand)
            X = D.tabular(state11, balls, strikes, enc, tok)
            qs = [np.asarray(self.parts['booster'].predict(X), np.float64)]
            known = np.isin(pitcher, self.parts['te'].pu)
        q = np.mean(qs, 0)
        return q / q.sum(1, keepdims=True), known

    def predict(self, state11, pitcher, batter, tokens):
        qm, known = self.raw_q(state11, pitcher, batter, tokens)
        q = M.temper(qm, self.spec['tau'])
        _need(np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(1), 1.0, atol=1e-9), 'bundle probabilities')
        return {'q': q, 'known': known, 'fallback': ~known}
