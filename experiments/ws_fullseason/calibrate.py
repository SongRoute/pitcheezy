"""One pooled temperature after the fixed three-seed seven-leaf mean.

Pure array operations only: callers own registered input custody and real-I/O
gates. No model/seed/family selection or final-FIT recalibration is provided.
"""
import numpy as np
from scipy.special import softmax
from experiments.ws_fullseason.data import key_digest
from experiments.ws_seq import train_seq as base

FOLDS = ('july', 'august', 'september')
SEEDS = (0, 1, 2)
CLIP = 1e-15
MIN_ROWS = 1000
SCHEMA = 'ws-fullseason-mean7-temperature-v1'


def probabilities(q):
    q = np.asarray(q)
    if (q.ndim != 2 or q.shape[1] != 7 or not len(q)
            or q.dtype.kind not in 'fiu' or not np.isfinite(q).all()
            or (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-7, rtol=0)):
        raise ValueError('finite normalized seven-leaf probabilities required')
    return q.astype(np.float64, copy=False)


def mean_seeds(predictions):
    if set(predictions) != set(SEEDS):
        raise ValueError('all three fixed seeds required')
    qs = [probabilities(predictions[seed]) for seed in SEEDS]
    if any(q.shape != qs[0].shape for q in qs):
        raise ValueError('seed row count mismatch')
    return np.mean(np.stack(qs), axis=0)


def checked_keys(keys):
    keys = np.asarray(keys)
    if (keys.ndim != 2 or keys.shape[1] != 3 or not len(keys)
            or keys.dtype.kind not in 'iu' or (keys <= 0).any()
            or (keys > np.iinfo(np.int64).max).any()
            or len(np.unique(keys, axis=0)) != len(keys)):
        raise ValueError('unique positive integral pitch keys required')
    return keys.astype(np.int64, copy=False)


def pool(windows):
    """Each fold supplies expected keys/labels and seed -> (keys, raw q).

    The registered loader must establish expected keys/labels independently of
    predictions and prove their date roles and byte custody before this call.
    """
    if set(windows) != set(FOLDS):
        raise ValueError('all three fixed OOF windows required')
    keys, labels, qs = [], [], []
    for fold in FOLDS:
        window = windows[fold]
        k = checked_keys(window['keys'])
        y = np.asarray(window['labels'])
        if (y.shape != (len(k),) or y.dtype.kind not in 'iu'
                or (y < 0).any() or (y > 6).any()):
            raise ValueError('legal aligned seven-leaf labels required')
        if set(window['seeds']) != set(SEEDS):
            raise ValueError('all three fixed seeds required')
        predictions = {}
        for seed in SEEDS:
            pk, q = window['seeds'][seed]
            if not np.array_equal(checked_keys(pk), k):
                raise ValueError('OOF seed/label key mismatch')
            q = probabilities(q)
            if len(q) != len(k):
                raise ValueError('OOF probability/key row mismatch')
            predictions[seed] = q
        keys.append(k); labels.append(y); qs.append(mean_seeds(predictions))
    keys = checked_keys(np.concatenate(keys))
    labels = np.concatenate(labels); q = np.concatenate(qs)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return keys[order], labels[order], q[order]


def fit(windows):
    keys, labels, q = pool(windows)
    result = {'schema': SCHEMA, 'kind': 'identity', 'temperature': 1.0,
              'clip': CLIP, 'bounds': list(base.T_BOUNDS), 'seeds': list(SEEDS),
              'order': 'mean_raw_seeds_then_temperature', 'weighting': 'equal_pitch',
              'folds': list(FOLDS), 'minimum_rows': MIN_ROWS, 'rows': len(keys),
              'keys_sha256': key_digest(keys), 'optimizer_calls': 0,
              'reason': 'insufficient_support'}
    if len(keys) < MIN_ROWS:
        return result
    result['optimizer_calls'] = 1
    try:
        temperature = base.temperature(np.log(np.clip(q, CLIP, 1.0)), labels)
    except base.Stop:
        # Inputs above are already valid. This is the existing optimizer's
        # nonconvergence/nonfinite/boundary-identification Stop, not I/O failure.
        result['reason'] = 'optimizer_failure'
        return result
    if not np.isfinite(temperature) or not base.T_BOUNDS[0] < temperature < base.T_BOUNDS[1]:
        raise ValueError('unexpected invalid optimizer return')
    result.update(kind='temperature', temperature=float(temperature), reason='fitted')
    return result


def apply(predictions, calibrator):
    """Final predictor uses the same three-seed raw mean before calibration."""
    if (calibrator.get('schema') != SCHEMA or calibrator.get('clip') != CLIP
            or calibrator.get('bounds') != list(base.T_BOUNDS)
            or calibrator.get('seeds') != list(SEEDS)
            or calibrator.get('order') != 'mean_raw_seeds_then_temperature'
            or calibrator.get('weighting') != 'equal_pitch'
            or calibrator.get('folds') != list(FOLDS)
            or calibrator.get('minimum_rows') != MIN_ROWS):
        raise ValueError('calibrator contract mismatch')
    t = calibrator.get('temperature')
    if not isinstance(t, (float, int)) or isinstance(t, bool) or not np.isfinite(t):
        raise ValueError('invalid calibration temperature')
    kind = calibrator.get('kind')
    if kind == 'identity':
        if t != 1.0 or calibrator.get('reason') not in ('insufficient_support', 'optimizer_failure'):
            raise ValueError('invalid identity calibrator')
    elif kind != 'temperature' or not base.T_BOUNDS[0] < t < base.T_BOUNDS[1] or calibrator.get('reason') != 'fitted':
        raise ValueError('invalid fitted calibrator')
    q = mean_seeds(predictions)
    if kind == 'identity':
        return q  # Identity preserves exact zeros; do not clip them away.
    return probabilities(softmax(np.log(np.clip(q, CLIP, 1.0)) / t, axis=1))
