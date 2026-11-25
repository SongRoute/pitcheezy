"""EXP-SVC-PRED-001: an EXP-LIT-BASE-001 baseline (B-TAB / B-MLR / B-GBT / B-SEQ / B-TAK) as a predictor Bundle.

Interface of ``pitcheezy.svc_p9.predictor`` (``CLASSES``, ``activate_snapshot``, ``check_loaded_modules``,
``Bundle.load(config=..., threads=..., candidate=<family>)``, ``Bundle.predict(state11, pitcher, batter, tokens) -> {'q', 'fallback'}``)
so ``scripts/svc_r1/predict_stage.run`` (the R4 predict stage) uses it unchanged.

``data.py`` / ``models.py`` in this package are byte copies of eb8e8b1b:experiments/lit_base/{data,models}.py (sha pinned in
the instance config); this package must not import ``experiments`` (``activate_snapshot`` binds ``experiments`` to R's
vendored snapshot so the W1 inputs come from the same pinned code as R4-003).

Every family reproduces the EXP-LIT-BASE-001 final step on one query batch:
  table: TargetEncoder(alpha) fitted on FIT -> 'pc'                         (models.tab_predict)
  mlr:   features = data.tabular(state11, balls, strikes, TargetEncoder(4.0) on FIT, tokens); softmax(((X-mu)/sd) W + b + P[pidx])
  gbt:   same features; LightGBM booster saved after the final rounds
  seq:   models.build_seq nets per seed, softmax, renormalise, mean       (models.seq_predict)
then q = temper(mean over seeds renormalised, family tau). balls / strikes / stand are recovered from state11 with the
pinned package Encoder11 (data.raw_from_state), as EXP-LIT-BASE-001 prepare did. Unknown pitcher/batter id -> index 0 /
encoder fallbacks exactly as EXP-LIT-BASE-001; ``fallback`` = unknown pitcher (descriptive flag only).
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import struct
import subprocess
import sys

import numpy as np

from . import data as D
from . import models as M

CLASSES = D.CLASSES
TOKEN_KEYS = ('valid', 'type', 'out', 'balls', 'strikes', 'same')
FAMILY_KIND = {'B-TAB': 'table', 'B-MLR': 'mlr', 'B-GBT': 'gbt', 'B-SEQ': 'seq', 'B-TAK': 'seq'}
TE_FIELDS = ('glob', 'pu', 'pc', 'pcu', 'pcc', 'psu', 'psc', 'bu', 'bc')


class LitBundleError(RuntimeError):
    pass


def _need(c, what):
    if not c:
        raise LitBundleError(what)


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


# LightGBM and torch each ship libomp; the two cannot run in one process on this Mac (OMP Error #15, either import order).
# The R4 predict stage runs torch, so the booster is evaluated in a torch-free child process that only imports numpy +
# lightgbm and answers float32 feature batches with q (length-prefixed .npy bytes over stdin/stdout).
GBT_WORKER = r"""
import io, struct, sys
import numpy as np
import lightgbm as lgb
b = lgb.Booster(model_file=sys.argv[1]); th = int(sys.argv[2])
inp, out = sys.stdin.buffer, sys.stdout.buffer
while True:
    h = inp.read(8)
    if len(h) < 8:
        break
    X = np.load(io.BytesIO(inp.read(struct.unpack('<Q', h)[0])), allow_pickle=False)
    buf = io.BytesIO(); np.save(buf, np.asarray(b.predict(X, num_threads=th), np.float64)); d = buf.getvalue()
    out.write(struct.pack('<Q', len(d))); out.write(d); out.flush()
"""


class GbtWorker:
    def __init__(self, model_file, threads):
        self.p = subprocess.Popen([sys.executable, '-c', GBT_WORKER, str(model_file), str(int(threads))],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def predict(self, X):
        buf = io.BytesIO(); np.save(buf, np.ascontiguousarray(X)); d = buf.getvalue()
        self.p.stdin.write(struct.pack('<Q', len(d))); self.p.stdin.write(d); self.p.stdin.flush()
        h = self.p.stdout.read(8)
        _need(len(h) == 8, 'lightgbm worker died')
        return np.load(io.BytesIO(self.p.stdout.read(struct.unpack('<Q', h)[0])), allow_pickle=False)

    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close(); self.p.wait(timeout=30)


def save_encoder(path, te):
    np.savez(path, alpha=np.array(te.alpha), **{k: getattr(te, k) for k in TE_FIELDS})


def load_encoder(path):
    z = np.load(path, allow_pickle=False)
    te = D.TargetEncoder(float(z['alpha']))
    for k in TE_FIELDS:
        setattr(te, k, z[k])
    return te


def files_for(kind, seeds):
    base = ['bundle.json', 'vocab.npz']
    if kind == 'table':
        return base + ['tab-encoder.npz']
    if kind == 'mlr':
        return base + ['feature-encoder.npz', 'mlr.npz']
    if kind == 'gbt':
        return base + ['feature-encoder.npz', 'gbt.txt']
    return base + [f'seed{s}.pt' for s in seeds]


def load_pin(config):
    pin = json.loads(pathlib.Path(config).read_text())
    _need(pin.get('format') == 'svc-lit-bundle-v1' and pin.get('family') in FAMILY_KIND, 'not an svc-lit-bundle-v1 pin')
    return pin


class Bundle:
    def __init__(self, family, kind, parts, pv, bv, tau, encoder_path, report):
        self.family, self.kind, self.parts, self.pv, self.bv, self.tau = family, kind, parts, pv, bv, tau
        self.encoder_path, self.report = encoder_path, report

    @classmethod
    def load(cls, config, threads=2, candidate=None):
        pin = load_pin(config)
        fam = pin['family']; kind = FAMILY_KIND[fam]
        _need(candidate in (None, fam), f'candidate {candidate} differs from the pin family {fam}')
        root = pathlib.Path(pin['dir'])
        _need(sorted(pin['files_sha256']) == sorted(files_for(kind, pin['seeds'])), 'pin file list')
        for name, want in pin['files_sha256'].items():
            _need(sha256_file(root / name) == want, f'bundle file sha differs: {name}')
        _need(sha256_file(pin['state_encoder']['path']) == pin['state_encoder']['sha256'], 'state encoder sha')
        meta = json.loads((root / 'bundle.json').read_text())
        _need((meta['family'], meta['hp'], meta['tau'], meta['seeds']) == (fam, pin['hp'], pin['tau'], pin['seeds']), 'bundle.json differs from the pin')
        v = np.load(root / 'vocab.npz', allow_pickle=False)
        pv, bv = v['pitcher'].astype(np.int64), v['batter'].astype(np.int64)
        parts = {}
        if kind == 'table':
            parts['te'] = load_encoder(root / 'tab-encoder.npz')
        elif kind in ('mlr', 'gbt'):
            parts['fe'] = load_encoder(root / 'feature-encoder.npz')
            if kind == 'mlr':
                z = np.load(root / 'mlr.npz', allow_pickle=False); parts.update({k: z[k] for k in z.files})
            else:
                parts['booster'] = GbtWorker(root / 'gbt.txt', threads)
        else:
            import torch
            torch.set_num_threads(int(threads))
            nets = []
            for s in pin['seeds']:
                net = M.build_seq(pin['hp'], meta['n_pitcher'], meta['n_batter'], s)
                net.load_state_dict(torch.load(root / f'seed{s}.pt', map_location='cpu', weights_only=True), strict=True)
                net.eval(); nets.append(net)
            parts['nets'] = nets
        report = {'candidate': fam, 'fold': 'fullseason', 'recipe_sha256': pin['files_sha256']['bundle.json'],
                  'calibration': {'kind': 'single temperature on the renormalised seed mean (EXP-LIT-BASE-001 DEV tau)', 'tau': pin['tau']},
                  'components': [{'name': f'{fam}-seed{s}', 'kind': kind, 'seed': s, 'source_commit': meta.get('commit')} for s in pin['seeds']]}
        return cls(fam, kind, parts, pv, bv, float(pin['tau']), pin['state_encoder']['path'], report)

    def raw_q(self, state11, pitcher, batter, tokens):
        """q before the family temperature (the seed mean, renormalised), as EXP-LIT-BASE-001 final ``qm``."""
        state11 = np.asarray(state11, np.float32)
        _need(state11.ndim == 2 and state11.shape[1] == 11 and np.isfinite(state11).all(), 'state11 shape')
        pitcher = np.asarray(pitcher, np.int64); batter = np.asarray(batter, np.int64)
        tok = {k: np.ascontiguousarray(tokens[k]) for k in TOKEN_KEYS}
        pi = D.index(pitcher, self.pv); bi = D.index(batter, self.bv)
        balls, strikes, stand, _ = D.raw_from_state(state11, self.encoder_path)
        if self.kind == 'table':
            q = self.parts['te'].transform(pitcher, batter, balls, strikes, stand)['pc']
        elif self.kind in ('mlr', 'gbt'):
            X = D.tabular(state11, balls, strikes, self.parts['fe'].transform(pitcher, batter, balls, strikes, stand), tok)
            if self.kind == 'mlr':
                import torch   # same float32 torch arithmetic as models.mlr_fit_predict's evaluation step
                p = self.parts
                with torch.no_grad():
                    z = torch.as_tensor((X - p['mu']) / p['sd'], dtype=torch.float32) @ torch.as_tensor(p['W']) + torch.as_tensor(p['b'])
                    if p['P'].size:
                        z = z + torch.as_tensor(p['P'])[torch.as_tensor(pi, dtype=torch.long)]
                q = M.softmax_np(z.double().numpy())
            else:
                q = np.asarray(self.parts['booster'].predict(X), np.float64)
        else:
            import torch
            data = M.SeqData(torch, torch.device('cpu'), state11, pi, bi, tok)
            qs = [M.seq_predict(torch, n, data, torch.arange(len(state11)), 8192) for n in self.parts['nets']]
            q = np.mean(qs, 0)
        return q / q.sum(1, keepdims=True), pi

    def predict(self, state11, pitcher, batter, tokens):
        qm, pi = self.raw_q(state11, pitcher, batter, tokens)
        q = M.temper(qm, self.tau)
        _need(np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(1), 1.0, atol=1e-9), 'bundle probabilities')
        return {'q': q, 'known': pi > 0, 'fallback': pi == 0}
