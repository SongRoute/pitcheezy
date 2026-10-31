"""Fullseason development adapter; existing campaign feature and label semantics.

No I/O at import. The small named source closure is loaded only by an explicit caller.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = ('first_small_features', 'history_transfer', 'outcome_transfer', 'scale_safe', 'abc_corpus', 'scale_runner')


def closure():
    modules = {}
    for name in SOURCE_NAMES:
        path = ROOT / 'experiments/ws_campaign/source' / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(fsf=modules['first_small_features'], ht=modules['history_transfer'],
                           ot=modules['outcome_transfer'], safe=modules['scale_safe'],
                           cp=modules['abc_corpus'], scale=modules['scale_runner'])


def key_digest(keys):
    keys = np.asarray(keys, dtype='<i8')
    if keys.ndim != 2 or keys.shape[1] != 3:
        raise ValueError('three-column keys required')
    return hashlib.sha256(np.ascontiguousarray(keys).tobytes()).hexdigest()


def project(raw, recipe, M):
    # Check before the reused projector's int64 cast: no fractional/negative identity can collapse.
    for name in M.cp.KEY + ['pitcher', 'batter']:
        values = raw[name].dropna().to_numpy()
        if values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or (values <= 0).any() or (values >= 2**53).any() or (values != np.floor(values)).any():
            raise ValueError('invalid integral identity ' + name)
    g, census = M.cp.project(raw, recipe, M.safe.outcome_labels, M.safe.head_labels)
    if g[M.cp.KEY].duplicated().any() or (g[M.cp.KEY].to_numpy() <= 0).any():
        raise ValueError('nonunique/nonpositive accepted key')
    fate = M.safe.core_fate(g)
    if not np.isin(fate, M.safe.CORE).all():
        raise ValueError('accepted non-core label')
    days = g.game_date.dt.strftime('%Y-%m-%d').to_numpy()
    fit = (days >= recipe['window']['fit'][0]) & (days <= recipe['window']['fit'][1])
    oof = (days >= recipe['window']['oof'][0]) & (days <= recipe['window']['oof'][1])
    if not (fit | oof).all() or (fit & oof).any() or not fit.any() or not oof.any():
        raise ValueError('empty, overlapping or unassigned role')
    if set(g.game_pk[fit]) & set(g.game_pk[oof]):
        raise ValueError('game crosses roles')
    y7 = np.array([int(v[-1]) for v in fate], dtype=np.int64)
    return g, census, fit, oof, y7


def audit_summary(g, census, fit, oof, y7, M):
    summary = {'census': census, 'roles': {}}
    for name, mask in [('FIT', fit), ('OOF', oof)]:
        frame = g.loc[mask]
        summary['roles'][name] = {'rows': int(mask.sum()), 'games': int(frame.game_pk.nunique()),
            'first_date': str(frame.game_date.min().date()) if len(frame) else None, 'last_date': str(frame.game_date.max().date()) if len(frame) else None,
            'ordered_keys_sha256': key_digest(frame[M.cp.KEY].to_numpy()),
            'parent_label_counts': np.bincount(frame.outcome10, minlength=10).tolist(),
            'seven_leaf_counts': np.bincount(y7[mask], minlength=7).tolist(),
            'physics_nonfinite': {k: int((~np.isfinite(frame[k].to_numpy(float))).sum()) for k in M.fsf.PHYSICS},
            'pitch_type_unknown_rows': int(frame.pitch_type.eq('UNKNOWN').sum())}
    return summary
