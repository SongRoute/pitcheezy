"""Generated prediction runtime: exact selected definitions from a hash-verified capture.
Training methods may remain for class identity; the service does not invoke them.
"""
from __future__ import annotations

import numpy as np


def condition_on_legality(p, impossible):
    p = np.asarray(p, dtype=float)
    impossible = np.asarray(impossible, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 10 or impossible.shape != (len(p),):
        raise ValueError('Expected ten-class probabilities and one legality flag per row')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
        raise ValueError('Invalid probability mass')
    constrained = p.copy()
    constrained[impossible, 9] = 0.
    remaining = constrained[impossible].sum(1)
    if (remaining <= 0).any():
        raise ValueError('Cannot condition rows assigning all probability to an impossible event')
    constrained[impossible] /= remaining[:, None]
    return constrained
