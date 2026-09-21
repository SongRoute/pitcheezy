"""Bounded inference, paired comparisons, and crash-safe stages for the new sweep."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
from scipy.special import softmax
from pitchmdp.data import hash_file
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_pilot import dump

REPRESENTATIONS = ('reference', 'continuous', 'clusters_3', 'clusters_5', 'clusters_10', 'clusters_20')
SEEDS = (42, 43, 44, 45, 46)
YEARS = (2024, 2025)


class NoPastStore:
    """Keep context/current candidate, remove ALL five past tokens and masks."""
    def __init__(self, store):
        self.store = store
        self.frame, self.normalizer = store.frame, store.normalizer

    def gather(self, rows, current=None):
        tokens, valid = self.store.gather(rows, current=current)
        if tokens.shape[1] != 6 or valid.shape != tokens.shape[:2]:
            raise ValueError('Expected five past slots and one current candidate')
        tokens, valid = tokens.copy(), valid.copy()
        tokens[:, :5], valid[:, :5] = 0., False
        return tokens, valid


class BatchedModel(SequenceModel):
    """Only inference batching changes; the frozen training algorithm is reused."""
    inference_batch = 8192

    def fit(self, *args, **kwargs):
        self._inside_fit = True
        try:
            return super().fit(*args, **kwargs)
        finally:
            self._inside_fit = False

    def logits(self, arrays, batch_size=None):
        size = 4096 if getattr(self, '_inside_fit', False) else self.inference_batch
        return super().logits(arrays, batch_size or size)


def integrated_chunks(delivery, model, store, context, rows, chunk_size):
    """Bound the 400-candidate expansion, including prediction probability arrays."""
    if chunk_size < 1 or len(rows) == 0:
        raise ValueError('Nonempty rows and positive chunk size required')
    for begin in range(0, len(rows), chunk_size):
        selected = np.asarray(rows[begin:begin+chunk_size])
        logits, levels = delivery.logits(model, store, context, selected, chunk_size=chunk_size)
        if logits.shape != (len(selected), 400, 10):
            raise ValueError('Frozen 400-draw, ten-outcome delivery contract changed')
        yield begin, logits, levels


def integrated_predict(delivery, model, store, context, rows, chunk_size):
    result = np.empty((len(rows), 10), dtype=np.float32)
    for begin, logits, _ in integrated_chunks(delivery, model, store, context, rows, chunk_size):
        result[begin:begin+len(logits)] = softmax(logits/model.delivery_temperature, axis=-1).mean(1)
    return result


def read_stage(path, contract):
    """A renamed stage is committed; a partial directory is never trusted."""
    if not path.exists():
        return False
    manifest = json.loads((path/'stage.json').read_text())
    if manifest['contract'] != contract:
        raise ValueError('Stage configuration changed: '+str(path))
    if not manifest['files']:
        raise ValueError('Empty committed stage')
    for rel, digest in manifest['files'].items():
        file = (path/rel).resolve()
        if not file.is_relative_to(path.resolve()) or hash_file(file) != digest:
            raise ValueError('Stage artifact changed: '+str(file))
    return True


def commit_stage(path, contract, writer):
    """Publish all files + hashes in one atomic directory rename on the SSD."""
    if read_stage(path, contract):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    # The caller holds the process lock. Remove abandoned writes only for this stage.
    for stale in path.parent.glob('.'+path.name+'-*.partial'):
        shutil.rmtree(stale)
    staging = Path(tempfile.mkdtemp(prefix='.'+path.name+'-', suffix='.partial', dir=path.parent))
    try:
        writer(staging)
        files = {str(p.relative_to(staging)): hash_file(p) for p in staging.rglob('*') if p.is_file()}
        if not files:
            raise ValueError('Cannot commit an empty stage')
        dump(staging/'stage.json', {'contract': contract, 'files': files})
        staging.rename(path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return True


@contextmanager
def execution_lock(path):
    """OS lock releases on exit/interrupt; stale PID text is not a stale lock."""
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another history/batter sweep is running') from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid())+'\n')
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def planned_contrasts():
    history = {name: {name+'_h0': 1, name+'_h5': -1} for name in REPRESENTATIONS}
    representation = {
        f'{name}_h{h}_minus_continuous_h{h}': {name+f'_h{h}': 1, f'continuous_h{h}': -1}
        for h in (0, 5) for name in REPRESENTATIONS if name != 'continuous'
    }
    interaction = {
        name: {name+'_h0': 1, name+'_h5': -1, 'continuous_h0': -1, 'continuous_h5': 1}
        for name in REPRESENTATIONS if name != 'continuous'
    }
    return {'history': history, 'representation': representation, 'interaction': interaction}


def compare_families(y, predictions, games, replicates=2000):
    """Linear contrasts of per-pitch losses, with paired shared game draws."""
    y, games = np.asarray(y, int), np.asarray(games)
    if not len(y) or len(games) != len(y) or replicates < 2:
        raise ValueError('Invalid bootstrap inputs')
    for p in predictions.values():
        classification_metrics(y, p)
    unique, group = np.unique(games, return_inverse=True)
    counts = np.bincount(group)
    draws = np.random.default_rng(42).integers(0, len(unique), size=(replicates, len(unique)))
    denominator = counts[draws].sum(1)
    results = {}
    for metric in ('log_loss', 'brier_multiclass'):
        losses = {}
        for name, p in predictions.items():
            p = np.asarray(p, float)
            losses[name] = (-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
                            if metric == 'log_loss' else ((p-np.eye(10)[y])**2).sum(1))
        for family, contrasts in planned_contrasts().items():
            delta = np.stack([sum(weight*losses[name] for name, weight in terms.items())
                              for terms in contrasts.values()])
            totals = np.stack([np.bincount(group, weights=d, minlength=len(unique)) for d in delta])
            bootstrap = totals[:, draws].sum(2)/denominator[None, :]
            means = delta.mean(1)
            radius = float(np.quantile(np.max(np.abs(bootstrap-means[:, None]), axis=0), .95))
            entries = {name: {'estimate': float(means[i]),
                        'pointwise95': np.quantile(bootstrap[i], [.025, .975]).tolist(),
                        'simultaneous95': [float(means[i]-radius), float(means[i]+radius)]}
                       for i, name in enumerate(contrasts)}
            results.setdefault(family, {})[metric] = entries
    return {'families': results, 'contrasts': planned_contrasts(), 'games': len(unique),
            'replicates': replicates, 'scope': 'Exploratory, fixed fitted models/CAL weights; '
            'simultaneous intervals within each family and year, not across all families/years.'}
