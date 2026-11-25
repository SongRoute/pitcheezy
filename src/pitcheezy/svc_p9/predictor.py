"""W2 of EXP-SVC-P9-001: pin-verified B_PSR9N6 predictor (CPU only, inference only).

What it loads (all pinned in configs/EXP-SVC-P9-001.json ``candidate.pins``; any mismatch raises PinMismatch):
  * 9 ID-side components EXP-MPS-PSR-00{1,2,4} september s0..s8 (pitcher x state residual on pitcher+batter T32)
  * 6 no-ID components N0 (P9B-001 noid_t32 s0), N1/N2 (P9B-002 s1/s2), D0..D2 (P9B-003 noid_t32_lr100_drop020)
  * calibration from the P9-013 receipt, key temperatures_and_mixture.PSR9.*.september (tau_K, tau_N, w_b0..b5)
  * exposure counts from the september entry of the frozen frequency table (baseline-counts.json)

Lineage checks per component (identity + lineage, from the real sealed files; nothing assumed):
  receipt bytes sha256 == pin; receipt status COMPLETED, fold september, seed, variant, input_metadata_sha256;
  receipt outputs_sha256[model-state.pt] == pin == sha256(model-state.pt bytes); the recipe rebuilt here from the
  frozen producer recipe gives object_sha == receipt model_recipe_sha256; identity-vocabularies bytes == pin and
  its sizes == receipt vocabulary_sizes; state dict loads strict into the research builder's model.

Model code is the vendored research snapshot (src/pitcheezy/svc_p9/research_snapshot, byte-identical to every
training commit; SNAPSHOT.json). ``activate_snapshot`` puts it first on sys.path; call it in a fresh process
(the prediction worker, the evaluation script, the dry run) before anything imports ``experiments``.

Bundle formula (nov_posthoc lane.calibrated_mixture 'B' arm as used for B_PSR9N6 in P9-013; no bias term):
  K = mean of the 9 ID-side softmax probabilities, N = mean of the 6 no-ID ones (float64, probability mean)
  Kc = softmax(log(clip(K, 1e-15, 1)) / tau_K), Nc = same with tau_N
  b = exposure bin of the pitcher's FIT_E9 row count (b0 = 0 = unseen; P9-002 bins), q = w_b*Kc + (1 - w_b)*Nc
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
SNAPSHOT = pathlib.Path(__file__).resolve().parent / 'research_snapshot'
CONFIG = REPO / 'configs/EXP-SVC-P9-001.json'
PRODUCER = REPO / 'configs/EXP-WS-FULLSEASON-T-OOF3.json'
PRODUCER_SHA256 = '9ec47ad13c78be1b0a0536a7441dd12baa5075d930e1c3e02d0bc2c125f8580d'
INPUT_METADATA_SHA256 = '5b64a6ccc41e59048f22e9d5ae0dbe68c6e81da6d06d77006ccefa9817994a3d'
T7_RUNS = pathlib.Path(os.environ.get('PITCHEEZY_SVC_P9_RUNS', '${PITCHEEZY_RUNS}'))
CLASSES = ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'OT')
TYPES = ('UNKNOWN', 'CH', 'CS', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'PO', 'SC', 'SI', 'SL', 'ST', 'SV', 'UN')
CONTEXT = 32
CAL_CLIP = 1e-15
EXPOSURE_BINS = ((0, 0), (1, 99), (100, 299), (300, 999), (1000, 2999), (3000, None))  # EXP-POSTHOC-P9-002 mixture.exposure_bins
PSR_DIR = 'MPS-20261110/EXP-MPS-PSR-00{lane}/attempt-001/returned/MPSPSR-pitcher_batter_lr100_drop020_psr-september-s{seed}'
# EXP-SVC-P9-002 (D198): the same 15-component layout retrained on the full 2026 regular season by EXP-MPS-FS-001 (fold 'fullseason', no OOF)
FS_ROOT = 'MPS-20261119/EXP-MPS-FS-001/attempt-001/returned'
FS_PSR_DIR = FS_ROOT + '/MPSFS-psr-fullseason-s{seed}'
FS_LANE = 'EXP-MPS-FS-001'
FOLDS = ('september', 'fullseason')
PSR_LANE = {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 4, 7: 4, 8: 4}
NOID = {'N0': (0, 'noid_t32', 'EXP-MPS-P9B-001'), 'N1': (1, 'noid_t32', 'EXP-MPS-P9B-002'), 'N2': (2, 'noid_t32', 'EXP-MPS-P9B-002'),
        'D0': (0, 'noid_t32_lr100_drop020', 'EXP-MPS-P9B-003'), 'D1': (1, 'noid_t32_lr100_drop020', 'EXP-MPS-P9B-003'),
        'D2': (2, 'noid_t32_lr100_drop020', 'EXP-MPS-P9B-003')}
D_FIELDS = {'base.model.dropout': (0.1, 0.2), 'base.model.encoder_dropout': (0.1, 0.2)}  # configs/EXP-MPS-P9B-003.json recipe_change.fields
P9_013_RECEIPT = 'NOV-POSTHOC-20261110/EXP-POSTHOC-P9-013/attempt-001/receipt.json'
P9_013_SHA256 = '3ebc95b9e485574c2c174370806e75e34fb46a32e91c09a874d6fd781b8fed51'


class PinMismatch(ValueError):
    """A pinned identity or lineage fact differs from the real file: refuse to load."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    """Same canonical form as experiments/ws_fullseason_c/run.py:object_sha (receipt model_recipe_sha256)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _need(cond, what):
    if not cond:
        raise PinMismatch(what)


# ---------------------------------------------------------------- research snapshot
def snapshot_pin(config=CONFIG):
    return json.loads(pathlib.Path(config).read_text())['candidate']['pins']['research_snapshot']['manifest_sha256']


def verify_snapshot(root=SNAPSHOT, manifest_sha256=None):
    """Manifest bytes == the registration pin (code review P2), then every listed file == its manifest sha."""
    raw = (root / 'SNAPSHOT.json').read_bytes()
    _need(sha256_bytes(raw) == (manifest_sha256 or snapshot_pin()), 'snapshot manifest sha256')
    manifest = json.loads(raw)
    for rel, spec in manifest['files'].items():
        p = root / rel
        _need(p.is_file(), f'snapshot file missing: {rel}')
        _need(sha256_file(p) == spec['sha256'], f'snapshot file changed: {rel}')
    return manifest


def activate_snapshot(root=SNAPSHOT, manifest_sha256=None):
    """Make ``experiments`` resolve to the vendored snapshot only. Refuses when another ``experiments`` is loaded."""
    manifest = verify_snapshot(root, manifest_sha256)
    loaded = sys.modules.get('experiments')
    if loaded is not None and pathlib.Path(getattr(loaded, '__file__', '') or '').resolve() != (root / 'experiments/__init__.py').resolve():
        raise PinMismatch('another experiments package is already imported in this process; use a fresh process')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import experiments  # noqa: F401
    _need(pathlib.Path(experiments.__file__).resolve() == (root / 'experiments/__init__.py').resolve(), 'experiments did not resolve to the snapshot')
    return manifest


def check_loaded_modules(root=SNAPSHOT):
    """Every imported experiments.* module must be a snapshot file with the pinned bytes."""
    manifest = json.loads((root / 'SNAPSHOT.json').read_text())
    seen = {}
    for name, mod in list(sys.modules.items()):
        if name == 'experiments' or name.startswith('experiments.'):
            f = getattr(mod, '__file__', None)
            if f is None:
                continue  # namespace sub-package (no file)
            rel = pathlib.Path(f).resolve().relative_to(root.resolve()).as_posix()
            _need(rel in manifest['files'], f'module outside snapshot manifest: {name}')
            _need(sha256_file(f) == manifest['files'][rel]['sha256'], f'module bytes changed: {name}')
            seen[name] = rel
    return seen


# ---------------------------------------------------------------- recipes (rebuilt, then checked against receipts)
def producer():
    _need(sha256_file(PRODUCER) == PRODUCER_SHA256, 'frozen producer recipe')
    return json.loads(PRODUCER.read_text())


def recipes(prod):
    import importlib
    batter = importlib.import_module('experiments.gpu14h_batter.train')
    prepitch = importlib.import_module('experiments.gpu14h_prepitch.train')
    reg = importlib.import_module('experiments.gpu14h_pitcherreg.train')
    psr = importlib.import_module('experiments.' + 'mps' + '_psr.train')
    r = batter.recipe(prod, 'pitcher_batter'); v = reg.VARIANTS['lr100_drop020']          # mps_pb2.run.recipe
    _need((r['base']['training']['lr'], r['base']['model']['dropout'], r['base']['model']['encoder_dropout']) == (.001, .1, .1), 'PB-002 parent recipe')
    r['base']['training']['lr'] = v['lr']; r['base']['model']['dropout'] = v['dropout']; r['base']['model']['encoder_dropout'] = v['dropout']
    r['identity_residual'] = copy.deepcopy(psr.RESIDUAL)                                     # mps_psr.run.recipe
    n = prepitch.recipe(prod, 'state_t32')                                                  # mps_p9b noid arm
    d = copy.deepcopy(n)                                                                     # mps_p9b.recipe3.apply_change
    for path, (ref, new) in D_FIELDS.items():
        node = d
        *head, leaf = path.split('.')
        for k in head:
            node = node[k]
        _need(node[leaf] == ref, f'D recipe reference value {path}')
        node[leaf] = new
    return {'PSR': r, 'N': n, 'D': d}


# ---------------------------------------------------------------- pins
def load_pins(config=CONFIG, candidate='C'):
    """EXP-SVC-P9-001: candidate C only. EXP-SVC-P9-002 and EXP-SVC-P9-003 (same candidates block): C (september fold, same
    pins as P9-001) and R (fullseason fold)."""
    c = json.loads(pathlib.Path(config).read_text())
    if c.get('id') in ('EXP-SVC-P9-002', 'EXP-SVC-P9-003'):
        _need(candidate in ('C', 'R') and candidate in c['candidates'], f"{c['id']} candidate {candidate}")
        pins = c['candidates'][candidate]['pins']
        _need(pins.get('fold') == ('september' if candidate == 'C' else 'fullseason'), f'candidate {candidate} fold')
        return pins, c
    _need(c.get('id') == 'EXP-SVC-P9-001' and c['candidate']['selected'] == 'C' and candidate == 'C', 'registration is not EXP-SVC-P9-001 candidate C')
    return c['candidate']['pins'], c


def pins_fold(pins):
    fold = pins.get('fold', 'september')
    _need(fold in FOLDS, f'fold {fold}')
    return fold


def components(pins, runs=T7_RUNS):
    """(name, kind, seed, directory, pinned receipt sha, pinned model-state sha, pinned vocabulary sha or None)."""
    out = []
    fold = pins_fold(pins)
    id_side = pins['id_side']
    for seed in range(9):
        if fold == 'september':
            key, directory = f'PSR-00{PSR_LANE[seed]} s{seed}', runs / PSR_DIR.format(lane=PSR_LANE[seed], seed=seed)
        else:
            key, directory = f'FS s{seed}', runs / FS_PSR_DIR.format(seed=seed)
        p = id_side[key]
        out.append((f'PS{seed}', 'PSR', seed, directory, p['receipt'], p['model_state'], p['identity_vocabularies']))
    noid = pins['noid_side_N6']
    for name, (seed, arm, lane) in NOID.items():
        p = noid[name]
        if fold == 'september':
            _need(p['path'].startswith(lane + '/') and f'september-s{seed}' in p['path'] and arm in p['path'], f'{name} pinned path lineage')
            directory = runs / 'MPS-20261110' / p['path']
        else:
            _need(p['path'] == f'{FS_ROOT}/MPSFS-{arm}-fullseason-s{seed}', f'{name} pinned path lineage')
            directory = runs / p['path']
        out.append((name, 'D' if name.startswith('D') else 'N', seed, directory, p['receipt'], p['model_state'], None))
    return out


def check_component(name, kind, seed, directory, receipt_pin, state_pin, vocab_pin, recipe_sha, fold='september'):
    raw = (directory / 'receipt.json').read_bytes()
    _need(sha256_bytes(raw) == receipt_pin, f'{name}: receipt sha256')
    r = json.loads(raw)
    _need(r.get('status') == 'COMPLETED' and r.get('fold') == fold and r.get('seed') == seed, f'{name}: receipt status/fold/seed')
    if fold == 'fullseason':
        _need(r.get('lane') == FS_LANE and r.get('OOF_rows') == 0, f'{name}: full-season lane/OOF')
    _need(r.get('input_metadata_sha256') == INPUT_METADATA_SHA256, f'{name}: input metadata')
    _need(r.get('producer_recipe_sha256') == PRODUCER_SHA256, f'{name}: producer recipe')
    _need(r.get('model_recipe_sha256') == recipe_sha, f'{name}: rebuilt recipe sha {recipe_sha[:12]} != receipt {str(r.get("model_recipe_sha256"))[:12]}')
    _need((r.get('outputs_sha256') or {}).get('model-state.pt') == state_pin, f'{name}: receipt outputs_sha256 model-state')
    _need(sha256_file(directory / 'model-state.pt') == state_pin, f'{name}: model-state bytes')
    info = {'name': name, 'kind': kind, 'seed': seed, 'variant': r.get('variant'), 'lane': r.get('lane'), 'source_commit': r.get('source_commit')}
    if kind == 'PSR':
        _need(r.get('variant') == 'pitcher_batter_lr100_drop020_psr' and r.get('trainer_variant') == 'pitcher_t32', f'{name}: variant')
        vb = (directory / 'identity-vocabularies.json').read_bytes()
        _need(sha256_bytes(vb) == vocab_pin == (r.get('outputs_sha256') or {}).get('identity-vocabularies.json'), f'{name}: identity vocabularies')
        vocab = json.loads(vb)
        sizes = {k: len(v) for k, v in vocab['factors'].items()}
        _need(sizes == r.get('vocabulary_sizes') and vocab.get('unknown_index') == 0, f'{name}: vocabulary sizes')
        info['vocabulary'] = vocab
    else:
        _need(r.get('variant') == 'state_t32' and r.get('lane_arm') == ('noid_t32' if kind == 'N' else 'noid_t32_lr100_drop020'), f'{name}: variant')
    return info


def calibration(pins, runs=T7_RUNS):
    cal = pins['calibration']
    raw = (runs / P9_013_RECEIPT).read_bytes()
    _need(sha256_bytes(raw) == P9_013_SHA256, 'P9-013 receipt sha256')
    t = json.loads(raw)['temperatures_and_mixture']['PSR9']
    tau_k, tau_n, mx = t['tau_K']['september'], t['tau_N']['september'], t['mixture_fit']['september']
    _need(tau_k['kind'] == 'temperature' and tau_n['kind'] == 'temperature', 'temperature kinds')
    _need(tau_k['temperature'] == cal['tau_K'] and tau_n['temperature'] == cal['tau_N'], 'tau pins')
    _need(mx['w'] == cal['mixture_w_b0_b5'] and mx['fit_months'] == cal['fit_months'] == ['july', 'august'], 'mixture weight pins')
    _need(len(mx['w']) == len(EXPOSURE_BINS) and mx['w'][0] == 0.0, 'b0 weight fixed 0')
    return {'tau_K': float(tau_k['temperature']), 'tau_N': float(tau_n['temperature']), 'w': np.asarray(mx['w'], np.float64)}


def exposure_table(pins, id_vocabulary):
    spec = pins['frozen_frequency_table']
    path = pathlib.Path(spec['path'])
    _need(sha256_file(path) == spec['sha256'] and path.stat().st_size == spec['bytes'], 'frozen frequency table bytes')
    t = json.loads(path.read_bytes())[spec.get('entry', 'september')]
    _need(t['alpha'] == spec['alpha'] and t['rows'] == spec['rows'] and t['fit9_keys_sha256'].startswith(spec['fit9_keys_sha256_prefix'].split()[0]), 'frequency table september entry')
    counts = {int(k): int(sum(v)) for k, v in t['pitcher'].items()}
    _need(sum(counts.values()) == t['rows'], 'exposure counts add up to FIT_E9 rows')
    _need(sorted(counts) == [int(i) for i in id_vocabulary['factors']['pitcher']], 'exposure pitchers == ID vocabulary pitchers')
    _need(t['fit9_keys_sha256'] == id_vocabulary['FIT_E9_keys_sha256'], 'exposure and vocabulary FIT_E9 keys')
    return counts


def exposure_bin(n):
    n = np.asarray(n, np.int64)
    out = np.full(len(n), -1, np.int64)
    for i, (lo, hi) in enumerate(EXPOSURE_BINS):
        m = n >= lo
        if hi is not None:
            m &= n <= hi
        out[m] = i
    _need((out >= 0).all(), 'exposure bin coverage')
    return out


def indices(values, vocabulary):
    """gpu14h_identity.train.indices: sorted FIT_E9 ids -> 1-based index, unknown 0."""
    ids = np.asarray(vocabulary, np.int64); values = np.asarray(values, np.int64)
    where = np.searchsorted(ids, values)
    known = (where < len(ids)) & (ids[np.minimum(where, len(ids) - 1)] == values)
    return np.where(known, where + 1, 0).astype(np.int64)


def temper(q, tau):
    from scipy.special import softmax
    return softmax(np.log(np.clip(np.asarray(q, np.float64), CAL_CLIP, 1.0)) / tau, axis=1)


# ---------------------------------------------------------------- bundle
class Bundle:
    """Loaded B_PSR9N6. ``predict`` takes model inputs (see live_inputs.ModelInputs fields) and returns q [n, 9]."""

    def __init__(self, models, cal, counts, vocab, report):
        self.models, self.cal, self.counts, self.vocab, self.report = models, cal, counts, vocab, report

    @classmethod
    def load(cls, config=CONFIG, runs=T7_RUNS, threads=2, candidate='C'):
        import torch
        activate_snapshot()
        torch.set_num_threads(min(int(threads), 2))
        pins, cfg = load_pins(config, candidate)
        fold = pins_fold(pins)
        rec = recipes(producer())
        shas = {k: object_sha(v) for k, v in rec.items()}
        import importlib
        psr = importlib.import_module('experiments.' + 'mps' + '_psr.train')
        prepitch = importlib.import_module('experiments.gpu14h_prepitch.train')
        models, infos, vocab = [], [], None
        for name, kind, seed, directory, receipt_pin, state_pin, vocab_pin in components(pins, runs):
            info = check_component(name, kind, seed, directory, receipt_pin, state_pin, vocab_pin, shas[kind], fold)
            if kind == 'PSR':
                v = info.pop('vocabulary')
                _need(vocab is None or v == vocab, f'{name}: vocabulary differs across ID-side components')
                vocab = v
                model = psr.build_model(rec['PSR'], 'pitcher_batter', seed, {k: len(x) for k, x in v['factors'].items()})
            else:
                model = prepitch.build_model(rec[kind], 'state_t32', seed)
            state = torch.load(directory / 'model-state.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            models.append((name, kind, model))
            infos.append(info)
        cal = calibration(pins, runs)
        counts = exposure_table(pins, vocab)
        report = {'candidate': candidate, 'fold': fold, 'components': infos, 'recipe_sha256': shas, 'snapshot_modules': check_loaded_modules(),
                  'calibration': {'tau_K': cal['tau_K'], 'tau_N': cal['tau_N'], 'w': cal['w'].tolist()},
                  'exposure_pitchers': len(counts), 'torch_threads': torch.get_num_threads(), 'device': 'cpu'}
        return cls(models, cal, counts, vocab, report)

    def _forward(self, model, x, tokens):
        import torch
        with torch.no_grad():
            tok = {k: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.bool if k == 'valid' else torch.int64) for k, v in tokens.items()}
            z = model(torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32), tok, None)['leaf'].double().numpy()
        from scipy.special import softmax
        return softmax(z, axis=1)

    def predict(self, state11, pitcher, batter, tokens):
        """state11 float32 [n,11]; pitcher/batter MLBAM ids [n]; tokens {valid,type,out,balls,strikes,same,pos} [n,32]."""
        state11 = np.asarray(state11, np.float32)
        _need(state11.ndim == 2 and state11.shape[1] == 11 and np.isfinite(state11).all(), 'state11 shape')
        pi = indices(pitcher, self.vocab['factors']['pitcher']); bi = indices(batter, self.vocab['factors']['batter'])
        x13 = np.column_stack((state11, pi.astype(np.float32), bi.astype(np.float32)))
        K = np.mean([self._forward(m, x13, tokens) for _, k, m in self.models if k == 'PSR'], axis=0, dtype=np.float64)
        N = np.mean([self._forward(m, state11, tokens) for _, k, m in self.models if k != 'PSR'], axis=0, dtype=np.float64)
        n = np.array([self.counts.get(int(p), 0) for p in pitcher], np.int64)
        b = exposure_bin(n); w = self.cal['w'][b][:, None]
        q = w * temper(K, self.cal['tau_K']) + (1.0 - w) * temper(N, self.cal['tau_N'])
        _need(np.isfinite(q).all() and np.allclose(q.sum(1), 1.0, atol=1e-9), 'bundle probabilities')
        return {'q': q, 'known': pi > 0, 'bin': b, 'fallback': pi == 0}


def synthetic_inputs(n=4, seed=0, pitchers=None, batters=None):
    """Synthetic model inputs only (no game data): random state11 in the encoder range and random earlier tokens."""
    rng = np.random.default_rng(seed)
    valid = np.zeros((n, CONTEXT), bool)
    for i in range(n):
        valid[i, :rng.integers(0, CONTEXT + 1)] = True
    tokens = {'valid': valid, 'type': rng.integers(0, len(TYPES), (n, CONTEXT)) * valid, 'out': rng.integers(0, 10, (n, CONTEXT)) * valid,
              'balls': rng.integers(0, 4, (n, CONTEXT)) * valid, 'strikes': rng.integers(0, 3, (n, CONTEXT)) * valid,
              'same': rng.integers(0, 2, (n, CONTEXT)) * valid, 'pos': np.broadcast_to(np.arange(1, CONTEXT + 1), (n, CONTEXT)).copy()}
    state11 = rng.normal(0, 1, (n, 11)).clip(-8, 8).astype(np.float32)
    return state11, np.asarray(pitchers if pitchers is not None else [1] * n), np.asarray(batters if batters is not None else [1] * n), tokens


def dry_run(config=CONFIG, runs=T7_RUNS):
    """Pin and lineage check on the real sealed files + one synthetic forward. Reads no game data."""
    b = Bundle.load(config, runs)
    known = int(b.vocab['factors']['pitcher'][0])
    s, p, bt, tok = synthetic_inputs(4, pitchers=[known, known, 1, 1], batters=[1, 1, 1, 1])
    out = b.predict(s, p, bt, tok)
    again = b.predict(s, p, bt, tok)
    report = dict(b.report)
    report['synthetic_forward'] = {'rows': 4, 'known': out['known'].tolist(), 'bin': out['bin'].tolist(),
                                   'row_sums_ok': bool(np.allclose(out['q'].sum(1), 1.0)), 'deterministic': bool(np.array_equal(out['q'], again['q']))}
    report['status'] = 'PINS_OK'
    return report


if __name__ == '__main__':
    if sys.argv[1:] != ['--dry-run']:
        raise SystemExit('usage: python -m pitcheezy.svc_p9.predictor --dry-run')
    print(json.dumps(dry_run(), indent=1, default=str))
