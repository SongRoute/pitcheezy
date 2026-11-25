"""EXP-SVC-PRED-001: a combination of predictor Bundles (e.g. R + B-SEQ) with the CANDIDATES-1121 combine rules.

pin (configs/EXP-SVC-PRED-001/<instance>/mix.json): {'format': 'svc-mix-bundle-v1', 'method': 'AM' | 'GM', 'weights': [...],
'temperature': T or null, 'components': [{'name', 'module', 'config', 'config_sha256', 'candidate'}]}
  AM: q = sum_i w_i q_i          GM: q = softmax(sum_i w_i log q_i)        then, if T: q = softmax(log q / T)
Weights must be non-negative and sum to 1 (equal weights = 1/k). Each component is loaded with its own module's
``Bundle.load(config=..., threads=..., candidate=...)`` (svc_p9 R / C, svc_lit baselines) and predicts with its own calibration.
``fallback`` = any component's fallback (descriptive).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import pathlib

import numpy as np

CLASSES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
METHODS = ('AM', 'GM')
REPO = pathlib.Path(__file__).resolve().parents[3]


class MixError(RuntimeError):
    pass


def _need(c, what):
    if not c:
        raise MixError(what)


def activate_snapshot(*a, **k):
    from pitcheezy.svc_p9 import predictor as P9
    return P9.activate_snapshot(*a, **k)


def check_loaded_modules(*a, **k):
    from pitcheezy.svc_p9 import predictor as P9
    return P9.check_loaded_modules(*a, **k)


def combine(qs, weights, method, temperature=None):
    qs = [np.asarray(q, np.float64) for q in qs]; w = np.asarray(weights, np.float64)
    _need(method in METHODS and len(qs) == len(w) >= 1 and (w >= 0).all() and abs(w.sum() - 1) < 1e-9, 'combine arguments')
    _need(temperature is None or float(temperature) > 0, 'temperature must be > 0')
    if method == 'AM':
        q = sum(wi * qi for wi, qi in zip(w, qs))
    else:
        z = sum(wi * np.log(np.clip(qi, 1e-15, 1.0)) for wi, qi in zip(w, qs))
        z = z - z.max(1, keepdims=True); q = np.exp(z)
    q = q / q.sum(1, keepdims=True)
    if temperature is not None:
        z = np.log(np.clip(q, 1e-15, 1.0)) / float(temperature); z = z - z.max(1, keepdims=True); q = np.exp(z); q = q / q.sum(1, keepdims=True)
    return q


class Bundle:
    def __init__(self, pin, parts, report):
        self.pin, self.parts, self.report = pin, parts, report

    @classmethod
    def load(cls, config, threads=2, candidate=None):
        pin = json.loads(pathlib.Path(config).read_text())
        _need(pin.get('format') == 'svc-mix-bundle-v1' and pin['method'] in METHODS, 'not an svc-mix-bundle-v1 pin')
        _need(candidate in (None, pin['name']), 'candidate differs from the mix name')
        parts = []
        for c in pin['components']:
            path = pathlib.Path(c['config']); path = path if path.is_absolute() else REPO / path
            _need(hashlib.sha256(path.read_bytes()).hexdigest() == c['config_sha256'], f"component pin sha differs: {c['name']}")
            _need(c['module'] in ('pitcheezy.svc_lit.bundle', 'pitcheezy.svc_p9.predictor', 'pitcheezy.svc_expl.bundle'),
                  f"component module not allowed: {c['module']}")
            _need(c.get('candidate'), f"component {c['name']}: candidate is required")
            mod = importlib.import_module(c['module'])
            _need(tuple(mod.CLASSES) == CLASSES, f"component classes differ: {c['name']}")
            b = mod.Bundle.load(config=path, threads=threads, candidate=c['candidate'])
            _need(b.report.get('candidate') == c['candidate'], f"component {c['name']} loaded candidate {b.report.get('candidate')}")
            parts.append(b)
        report = {'candidate': pin['name'], 'fold': 'fullseason', 'recipe_sha256': hashlib.sha256(pathlib.Path(config).read_bytes()).hexdigest(),
                  'calibration': {'kind': f"{pin['method']} combine, weights {pin['weights']}, temperature {pin.get('temperature')}"},
                  'components': [{'name': c['name'], 'kind': c['module'], 'seed': None, 'source_commit': None, 'candidate': b.report.get('candidate'),
                                  'fold': b.report.get('fold'), 'pin_sha256': c['config_sha256'], 'recipe_sha256': b.report.get('recipe_sha256')}
                                 for c, b in zip(pin['components'], parts)]}
        return cls(pin, parts, report)

    def predict(self, state11, pitcher, batter, tokens):
        outs = [b.predict(state11, pitcher, batter, tokens) for b in self.parts]
        q = combine([o['q'] for o in outs], self.pin['weights'], self.pin['method'], self.pin.get('temperature'))
        _need(np.isfinite(q).all() and np.allclose(q.sum(1), 1.0, atol=1e-9), 'mix probabilities')
        fb = np.any([np.asarray(o['fallback'], bool) for o in outs], axis=0)
        return {'q': q, 'known': ~fb, 'fallback': fb}
