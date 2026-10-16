"""EXP-WS-SEQ-002 training: ONE pre-registered grid config x ONE seed of the pitch-sequence model with richer
prior-pitch tokens. Extension (copy, not import) of train_seq.py (EXP-WS-SEQ-001, arm N1).

Model: MLP trunk on the 199 prequential pre-pitch features + transformer encoder over the up-to-L EARLIER pitches of
the same pitcher in the same game; optional (grid) extra token fields of those EARLIER pitches (zone, release_speed /
pfx_x / pfx_z buckets) and an optional second context = up-to-B EARLIER pitches seen by the current batter in the same
game (any pitcher; shared encoder + side embedding, pooled separately). The CURRENT pitch contributes only its 199
pre-pitch features: its own zone / physics / outcome are never gathered (a context row index never equals the row).

2026 EXPOSED DEVELOPMENT; research model; never a winner/adoption claim. This script writes margins and probabilities
only. It computes NO metric on EVAL rows (EVAL labels are used only as tokens of LATER pitches of the same game).

Runs on the A5000 server (one card per process, CUDA_VISIBLE_DEVICES set by the launcher) or on CPU (selfcheck).
Imports: numpy, scipy, torch only (the JSON grid file is authoritative; no pyyaml needed).

  python experiments/ws_seq/train_seq2.py --config-id G04 --seed 0
      [--inputs data/ws-seq/inputs2] [--base-inputs data/ws-seq/inputs] [--grid configs/EXP-WS-SEQ-002.json]
      [--out runs/ws-seq2/EXP-WS-SEQ-002/G04-s0] [--selected-config-hash H   (required for seeds 1,2)]
  python experiments/ws_seq/train_seq2.py --selfcheck [--selfcheck-out FILE]     (synthetic unit checks, CPU)

Outputs under <out>/ (= one run directory <Gxx>-s<seed>): margins.npz, temperatures.json, seven-leaf-evaluation.npy,
curve.json, model-state.pt, receipt.json (written last).

Exit codes (as train_seq.py): 0 ok; 2 usage/config; 3 missing import; 4 reserved (the SEQ-002 stop rule lives in
select_config.py; this script never returns 4); 5 input/code/config hash mismatch; 6 registered STOP (non-finite,
temperature, coherence, layout, context contract); 7 output directory already holds a receipt.
"""
import os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # deterministic cuBLAS (must be set before CUDA init)
import argparse, hashlib, json, time
from pathlib import Path
try:
    import numpy as np
except ImportError as _e:  # pragma: no cover
    print('MISSING IMPORT (nothing is installed by this script): ' + str(_e), file=sys.stderr); sys.exit(3)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP_ID = 'EXP-WS-SEQ-002'
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')
ROLES = ('tuning', 'calibration', 'evaluation')
BIP = (3, 4, 5, 6, 7, 9)
T_BOUNDS = (.05, 20.)
EXIT_OK, EXIT_USAGE, EXIT_IMPORT, EXIT_RESERVED, EXIT_HASH, EXIT_STOP, EXIT_EXISTS = 0, 2, 3, 4, 5, 6, 7
CORPUS_MEMBERS = ('keys', 'y10', 'y7', 'games', 'fit', 'tuning', 'calibration', 'evaluation', 'narrow_whiff', 'ground_ball_per_pitch')
SEQ_MEMBERS = ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'pitch_type', 'outcome10', 'balls', 'strikes', 'stand')
RICH_FIELDS = ('zone', 'release_speed_bucket', 'pfx_x_bucket', 'pfx_z_bucket')          # new token fields of EARLIER pitches
TOKENS2_MEMBERS = ('game_pk', 'at_bat_number', 'pitch_number') + RICH_FIELDS
RICH_VOCAB = {'zone': 'zone', 'release_speed_bucket': 'release_speed', 'pfx_x_bucket': 'pfx_x', 'pfx_z_bucket': 'pfx_z'}
TOKEN_SETS = ('base', 'physics_zone')
GRID_KEYS = ('id', 'context_length', 'tokens', 'batter_context', 'token_dim', 'encoder_layers', 'encoder_ff')
HASH_KEYS = ('id', 'inputs', 'labels', 'vocab', 'buckets', 'model', 'training', 'temperature', 'nll_clip')
OUTPUT_FILES = ('margins.npz', 'temperatures.json', 'seven-leaf-evaluation.npy', 'curve.json', 'model-state.pt')


class Stop(Exception):
    """Registered STOP: no result for this config/seed."""


class HashMismatch(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False, default=str, ensure_ascii=False) + '\n')


def imports():
    """Lazy heavy imports so that a missing package is a clean exit 3."""
    try:
        import scipy, scipy.optimize, scipy.special  # noqa: F401
        import torch
    except ImportError as e:
        print('MISSING IMPORT (nothing is installed by this script): ' + str(e), file=sys.stderr); sys.exit(EXIT_IMPORT)
    return torch, scipy


# ---------------------------------------------------------------- config / grid
def load_config(path):
    cfg = json.loads(Path(path).read_text())
    v, m, t, b, grid = cfg['vocab'], cfg['model'], cfg['training'], cfg['buckets'], cfg['grid']
    ids = [e['id'] for e in grid]
    ok = [cfg['id'] == EXP_ID, cfg['inputs']['feature_dimension'] > 0, tuple(cfg['labels']['bip_classes']) == BIP,
          cfg['labels']['parent_classes'] == 10, cfg['labels']['whiff_parent_class'] == 1, len(m['trunk_hidden']) == 2,
          t['optimizer'] == 'adamw', t['precision'] == 'fp32', tuple(cfg['temperature']['bounds_T']) == T_BOUNDS,
          m['activation'] == 'gelu' and m['encoder_activation'] == 'gelu', m['layer_norm'] is True, all(t['loss_weights'][h] == 1.0 for h in HEADS),
          v['same'] == 2, all(v[f] == b[f] + 1 for f in ('release_speed', 'pfx_x', 'pfx_z')), 1 <= len(grid) <= 12, len(set(ids)) == len(ids), ids == sorted(ids),
          all(tuple(e) == GRID_KEYS for e in grid), all(e['tokens'] in TOKEN_SETS for e in grid), all(e['token_dim'] % m['encoder_heads'] == 0 for e in grid),
          all(0 <= e['batter_context'] <= e['context_length'] and e['context_length'] >= 1 for e in grid), list(cfg['seeds']) == [0, 1, 2]]
    if not all(ok):
        raise SystemExit('BLOCKED: config does not match what this source implements: ' + str(ok))
    return cfg


def grid_entry(cfg, config_id):
    hit = [e for e in cfg['grid'] if e['id'] == config_id]
    return hit[0] if len(hit) == 1 else None


def config_hash(cfg, entry):
    """Stable hash of ONE resolved config: the grid entry + every shared block that changes the fitted model.
    Other grid entries, seeds, selection text and descriptive strings are not part of it."""
    body = {'entry': entry, **{k: cfg[k] for k in HASH_KEYS}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


# ---------------------------------------------------------------- sequence index (strictly earlier pitches, same group, same game)
def build_context_index(group, game, at_bat, pitch_no, length):
    """(n, length) int32 row indices. Slot j of row i = the row of the pitch (j+1) pitches EARLIER in the same
    (group, game) set in (at_bat_number, pitch_number) order; -1 = padding. group = pitcher id (pitcher-side context)
    or batter id (batter-side context: earlier pitches seen by that batter in this game, any pitcher). Never the row
    itself, never a later pitch. Vectorised: lexsort by (group, game, at_bat, pitch_no); position k sees k-1..k-length.
    (Body identical to train_seq.build_context_index; only the first argument was renamed.)"""
    group, game, at_bat, pitch_no = (np.asarray(a, np.int64) for a in (group, game, at_bat, pitch_no))
    n = len(group)
    order = np.lexsort((pitch_no, at_bat, game, group))
    p, g, a, q = group[order], game[order], at_bat[order], pitch_no[order]
    new = np.ones(n, bool)
    if n > 1:
        new[1:] = (p[1:] != p[:-1]) | (g[1:] != g[:-1])
        if (~new[1:] & (a[1:] == a[:-1]) & (q[1:] == q[:-1])).any():
            raise Stop('STOP duplicate (group, game, at_bat_number, pitch_number): the earlier/later order is undefined')
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    earlier = np.arange(n) - start                                    # number of earlier pitches in the group
    back = np.arange(1, length + 1)
    valid = back[None, :] <= earlier[:, None]
    src = np.where(valid, order[np.clip(np.arange(n)[:, None] - back[None, :], 0, None)], -1)
    ctx = np.empty((n, length), np.int32); ctx[order] = src
    return ctx


def brute_context_index(group, game, at_bat, pitch_no, length):
    """Independent O(n * group) python loop used only by the selfcheck."""
    n = len(group); out = np.full((n, length), -1, np.int32)
    for i in range(n):
        earlier = [(int(at_bat[k]), int(pitch_no[k]), k) for k in range(n)
                   if group[k] == group[i] and game[k] == game[i] and (at_bat[k], pitch_no[k]) < (at_bat[i], pitch_no[i])]
        earlier.sort(reverse=True)
        for j, (_, _, k) in enumerate(earlier[:length]):
            out[i, j] = k
    return out


def check_context(ctx, group, game, at_bat, pitch_no):
    """Run-time proof on the real index: every non-padding slot is the same group (pitcher or batter) and game and
    strictly earlier; slots are strictly decreasing in time; padding is a suffix; no row references itself."""
    n, L = ctx.shape; valid = ctx >= 0; safe = np.where(valid, ctx, 0)
    key = np.asarray(at_bat, np.int64) * 100000 + np.asarray(pitch_no, np.int64)
    if (np.asarray(pitch_no) >= 100000).any() or (np.asarray(pitch_no) < 0).any():
        raise Stop('STOP pitch_number outside the order-key range')
    ck = key[safe]
    ok = (((np.asarray(group)[safe] == np.asarray(group)[:, None]) & (np.asarray(game)[safe] == np.asarray(game)[:, None]) & (ck < key[:, None])) | ~valid).all()
    ok = ok and bool(((ck[:, 1:] < ck[:, :-1]) | ~valid[:, 1:]).all()) and bool((valid[:, :-1] | ~valid[:, 1:]).all())
    ok = ok and not bool((ctx == np.arange(n)[:, None]).any())
    if not ok:
        raise Stop('STOP context index violates the strictly-earlier / same group / same game contract')
    return {'rows': int(n), 'length': int(L), 'rows_without_context': int((~valid[:, 0]).sum()), 'rows_full_context': int(valid[:, -1].sum()),
            'mean_context_length': float(valid.sum(1).mean())}


# ---------------------------------------------------------------- leaves / temperature (copies, not imports)
# leaves(): copy of scale_safe.leaves (lines 132-136 of
# /Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-step2-scale-f2-recovery-author-001/scale_safe.py,
# sha256 1f2cd2e221bcff53c9f2b1b1ee7ceae0476f02bf5559a21e31a8b520f94b5234); only the exception type differs.
def leaves(p10, q, r):
    s = p10[:, list(BIP)].sum(1); out = np.column_stack([p10[:, 0], p10[:, 2], p10[:, 8], p10[:, 1] * q, p10[:, 1] * (1 - q), s * r, s * (1 - r)])
    if not np.isfinite(out).all() or np.abs(out.sum(1) - 1).max() > 1e-9:
        raise Stop('STOP seven-leaf coherence')
    return out


# temperature(): verbatim copy of scale_runner.temperature (lines 470-480 of
# /Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-step2-scale-f2-recovery-author-001/scale_runner.py,
# sha256 e25db72841066e28aa28b25bcd4ed07e7b7c57e199dbf9ab33f947f80ba0a436); only the exception type differs.
def temperature(z, y):
    """ONE scalar temperature: minimise mean NLL of softmax(z/T) or sigmoid(z/T) over log T in registered bounds."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    z = np.asarray(z, np.float64); y = np.asarray(y); lo, hi = np.log(T_BOUNDS)
    if len(z) == 0 or len(z) != len(y) or not np.isfinite(z).all():
        raise Stop('STOP temperature input')
    if z.ndim == 2:
        f = lambda u: float(np.mean(logsumexp(z / np.exp(u), axis=1) - z[np.arange(len(y)), y] / np.exp(u)))
    else:
        f = lambda u: float(np.mean(np.logaddexp(0, z / np.exp(u)) - y * z / np.exp(u)))
    r = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10, 'maxiter': 500})
    if not r.success or not np.isfinite(r.fun) or min(r.x - lo, hi - r.x) < 1e-4:
        raise Stop(f'STOP temperature not identified inside registered bounds: T={np.exp(r.x)}')
    return float(np.exp(r.x))


def seven_from_margins(zp, zw, zg, T=None):
    from scipy.special import softmax, expit
    T = T or {h: 1.0 for h in HEADS}
    return leaves(softmax(np.asarray(zp, np.float64) / T['parent'], axis=1), expit(np.asarray(zw, np.float64) / T['narrow_whiff']), expit(np.asarray(zg, np.float64) / T['ground_ball_per_pitch']))


def nll7(q, y7, clip):
    return float(np.mean(-np.log(np.clip(q[np.arange(len(y7)), y7], clip, 1))))


def seven_label(y10, whiff, gb):
    """Same mapping as abc_pilot.seven_label (used only to cross-check the shipped y7)."""
    y = np.full(len(y10), -1); y[y10 == 0] = 0; y[y10 == 2] = 1; y[y10 == 8] = 2; s = y10 == 1; y[s & (whiff == 1)] = 3; y[s & (whiff == 0)] = 4
    b = np.isin(y10, BIP); y[b & (gb == 1)] = 5; y[b & (gb == 0)] = 6
    return y


# ---------------------------------------------------------------- inputs
def verify_package(inputs, base, cfg, config_path):
    """sha256 of every file read by the training (inputs2 files + the SEQ-001 base files they extend), and of this
    script + the grid config, against the bindings of inputs2/manifest.json. Nothing is decoded before this passes."""
    I = cfg['inputs']; mp = Path(inputs) / I['manifest_file']
    if not mp.is_file():
        raise HashMismatch('manifest missing: ' + str(mp))
    man = json.loads(mp.read_text()); seen = {}
    for section, root, names in (('files', inputs, (I['tokens2_file'], I['bucket_edges_file'])), ('base_files', base, (I['features_file'], I['corpus_file'], I['sequence_file']))):
        for name in names:
            p = Path(root) / name
            seen[name] = sha(p) if p.is_file() else None
            if name not in man.get(section, {}) or seen[name] is None or seen[name] != man[section][name]['sha256']:
                raise HashMismatch(f'input sha256 mismatch or missing ({section}): {name}')
    code = {'train_seq2.py': sha(Path(__file__).resolve()), 'config': sha(config_path)}
    for k, v in code.items():
        if man['bound_sources'].get(k) != v:
            raise HashMismatch(f'{k} differs from the sha256 bound in the manifest (prepare inputs2 again after any reviewed code/config change)')
    if man.get('id') != EXP_ID:
        raise HashMismatch('manifest id')
    return man, sha(mp), seen, code


def load_package(inputs, base, cfg):
    """x (n,199) float32, corpus C, token sources S (SEQ-001 integer tokens + the four SEQ-002 fields), bucket edges."""
    I = cfg['inputs']; d = I['feature_dimension']; v = cfg['vocab']
    x = np.load(Path(base) / I['features_file'], allow_pickle=False)
    with np.load(Path(base) / I['corpus_file'], allow_pickle=False) as z:
        C = {k: np.array(z[k]) for k in CORPUS_MEMBERS}
    with np.load(Path(base) / I['sequence_file'], allow_pickle=False) as z:
        S = {k: np.array(z[k]) for k in SEQ_MEMBERS}
    with np.load(Path(inputs) / I['tokens2_file'], allow_pickle=False) as z:
        if tuple(sorted(z.files)) != tuple(sorted(TOKENS2_MEMBERS)):
            raise Stop('STOP seq-inputs2 members: ' + str(sorted(z.files)))
        S2 = {k: np.array(z[k]) for k in TOKENS2_MEMBERS}
    edges = json.loads((Path(inputs) / I['bucket_edges_file']).read_text())
    n = len(C['y10'])
    if x.shape != (n, d) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise Stop(f'STOP feature matrix shape/dtype/non-finite: {x.shape} {x.dtype}')
    if any(len(a) != n for a in list(C.values()) + list(S.values()) + list(S2.values())) or any(a.dtype.kind != 'i' for a in list(S.values()) + list(S2.values())):
        raise Stop('STOP package row counts / integer dtypes')
    same = [np.array_equal(S['game_pk'], C['keys'][:, 0]), np.array_equal(S['at_bat_number'], C['keys'][:, 1]), np.array_equal(S['pitch_number'], C['keys'][:, 2]),
            np.array_equal(S['outcome10'], C['y10']), np.array_equal(C['games'], C['keys'][:, 0]),
            bool(np.all(C['fit'].astype(int) + C['tuning'] + C['calibration'] + C['evaluation'] == 1)),
            np.array_equal(seven_label(C['y10'], C['narrow_whiff'], C['ground_ball_per_pitch']), C['y7']),
            all(np.array_equal(S2[k], S[k]) for k in ('game_pk', 'at_bat_number', 'pitch_number'))]   # new token rows are the base rows, row for row
    rng = [S['pitch_type'].min() >= 0, S['pitch_type'].max() < v['pitch_type'], S['outcome10'].min() >= 0, S['outcome10'].max() < v['outcome'],
           S['balls'].min() >= 0, S['balls'].max() < v['balls'], S['strikes'].min() >= 0, S['strikes'].max() < v['strikes']]
    rng += [bool(S2[f].min() >= 0 and S2[f].max() < v[RICH_VOCAB[f]]) for f in RICH_FIELDS]
    eb = [edges.get('bins') == cfg['buckets'], all(len(edges['edges'][f]) == cfg['buckets'][f] - 1 for f in cfg['buckets']), edges.get('fit_role') == 'fit']
    if not all(same) or not all(bool(r) for r in rng) or not all(eb):
        raise Stop('STOP package identity (token rows vs corpus rows / one role per row / y7), token range or bucket file: ' + str(same) + str([bool(r) for r in rng]) + str(eb))
    for f in RICH_FIELDS:
        S[f] = S2[f]
    return x, C, S, edges


# ---------------------------------------------------------------- model
def build_model(torch, cfg, entry):
    """Module construction order for a 'base' entry without batter context is exactly train_seq.build_model(arm N1):
    trunk, six token embeddings, encoder, no_context, heads (same initial state for the same torch seed)."""
    nn = torch.nn; m = cfg['model']; v = cfg['vocab']; d_in = cfg['inputs']['feature_dimension']; h1, h2 = m['trunk_hidden']
    D = entry['token_dim']; L = entry['context_length']; rich = entry['tokens'] == 'physics_zone'; bat = entry['batter_context'] > 0

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d_in, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(m['dropout']),
                                       nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(m['dropout']))
            self.rich, self.bat = rich, bat
            self.pad_type, self.pad_out = v['pitch_type'], v['outcome']
            self.e_type = nn.Embedding(v['pitch_type'] + 1, D, padding_idx=self.pad_type)
            self.e_out = nn.Embedding(v['outcome'] + 1, D, padding_idx=self.pad_out)
            self.e_balls = nn.Embedding(v['balls'], D); self.e_strikes = nn.Embedding(v['strikes'], D)
            self.e_same = nn.Embedding(v['same'], D); self.e_pos = nn.Embedding(L + 1, D, padding_idx=0)
            layer = nn.TransformerEncoderLayer(d_model=D, nhead=m['encoder_heads'], dim_feedforward=entry['encoder_ff'], dropout=m['encoder_dropout'],
                                               activation='gelu', batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, num_layers=entry['encoder_layers'], enable_nested_tensor=False)
            self.no_context = nn.Parameter(torch.zeros(D)); nn.init.normal_(self.no_context, std=0.02)
            if rich:   # new fields of EARLIER pitches; category 0 = missing, padding id = vocabulary size
                self.e_rich = nn.ModuleDict({f: nn.Embedding(v[RICH_VOCAB[f]] + 1, D, padding_idx=v[RICH_VOCAB[f]]) for f in RICH_FIELDS})
            if bat:    # batter-side context: the SAME encoder, a side embedding (0 pitcher side, 1 batter side), its own no-context vector
                self.e_side = nn.Embedding(2, D)
                self.no_context_bat = nn.Parameter(torch.zeros(D)); nn.init.normal_(self.no_context_bat, std=0.02)
            width = h2 + D + (D if bat else 0)
            self.head_parent = nn.Linear(width, cfg['labels']['parent_classes']); self.head_whiff = nn.Linear(width, 1); self.head_gb = nn.Linear(width, 1)

        def pool(self, tok, side, no_context):
            valid = tok['valid']; z = torch.zeros_like(tok['type'])
            e = (self.e_type(torch.where(valid, tok['type'], z + self.pad_type)) + self.e_out(torch.where(valid, tok['out'], z + self.pad_out))
                 + self.e_balls(torch.where(valid, tok['balls'], z)) + self.e_strikes(torch.where(valid, tok['strikes'], z))
                 + self.e_same(torch.where(valid, tok['same'], z)) + self.e_pos(torch.where(valid, tok['pos'], z)))
            if self.rich:
                for f in RICH_FIELDS:
                    e = e + self.e_rich[f](torch.where(valid, tok[f], z + v[RICH_VOCAB[f]]))
            if self.bat:
                e = e + self.e_side(z + side)
            empty = ~valid.any(1); pad = ~valid; pad = pad & ~(empty[:, None] & (tok['pos'] == 1))  # an all-padding row would give NaN attention
            s = self.encoder(e, src_key_padding_mask=pad)
            w = valid.unsqueeze(-1).to(s.dtype)
            pooled = (s * w).sum(1) / w.sum(1).clamp(min=1.0)
            return torch.where(empty[:, None], no_context.expand_as(pooled), pooled)

        def forward(self, x, tok, btok=None):
            parts = [self.trunk(x), self.pool(tok, 0, self.no_context)]
            if self.bat:
                parts.append(self.pool(btok, 1, self.no_context_bat))
            h = torch.cat(parts, 1)
            return self.head_parent(h), self.head_whiff(h).squeeze(-1), self.head_gb(h).squeeze(-1)

    return Net()


class Data:
    """Everything on the device once; contexts are gathered per batch from the (n, L) / (n, B) row indices.
    Every token field of a context slot is read at the EARLIER row's index s = ctx[idx]; nothing but x, the row's
    batter/pitcher id (for the same-counterpart flag) and the training labels is read at idx itself."""

    def __init__(self, torch, device, x, C, S, ctx, bctx=None):
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.torch = torch; self.x = t(x, torch.float32); self.ctx = t(ctx, torch.int64); self.bctx = None if bctx is None else t(bctx, torch.int64)
        self.type = t(S['pitch_type'], torch.int64); self.out = t(S['outcome10'], torch.int64); self.balls = t(S['balls'], torch.int64)
        self.strikes = t(S['strikes'], torch.int64); self.batter = t(S['batter'], torch.int64); self.pitcher = t(S['pitcher'], torch.int64)
        self.rich = {f: t(S[f], torch.int64) for f in RICH_FIELDS}
        self.y10 = t(C['y10'], torch.int64); self.whiff = t(C['narrow_whiff'], torch.float32); self.gb = t(C['ground_ball_per_pitch'], torch.float32)
        self.is_whiff_row = self.y10 == 1; self.is_bip_row = torch.isin(self.y10, t(np.array(BIP), torch.int64))
        self.pos = torch.arange(1, max(ctx.shape[1], 0 if bctx is None else bctx.shape[1]) + 1, device=device, dtype=torch.int64)[None, :]

    def tokens(self, idx, side=0):
        """side 0: earlier pitches of the same pitcher (flag = same batter as the current row);
        side 1: earlier pitches seen by the same batter (flag = thrown by the same pitcher as the current row)."""
        c = (self.ctx if side == 0 else self.bctx)[idx]; valid = c >= 0; s = c.clamp(min=0)
        counterpart = self.batter if side == 0 else self.pitcher
        tok = {'valid': valid, 'type': self.type[s], 'out': self.out[s], 'balls': self.balls[s], 'strikes': self.strikes[s],
               'same': (counterpart[s] == counterpart[idx][:, None]).to(self.torch.int64), 'pos': self.pos[:, :c.shape[1]].expand_as(c)}
        for f in RICH_FIELDS:
            tok[f] = self.rich[f][s]
        return tok


def forward(model, data, idx):
    return model(data.x[idx], data.tokens(idx, 0), data.tokens(idx, 1) if model.bat else None)


def predict(torch, model, data, rows, batch):
    """Raw margins (float64 numpy) for the given row indices, eval mode, no grad."""
    model.eval(); P, W, G = [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            zp, zw, zg = forward(model, data, rows[s:s + batch])
            P.append(zp.double().cpu().numpy()); W.append(zw.double().cpu().numpy()); G.append(zg.double().cpu().numpy())
    z = {'parent': np.concatenate(P), 'narrow_whiff': np.concatenate(W), 'ground_ball_per_pitch': np.concatenate(G)}
    if not all(np.isfinite(a).all() for a in z.values()):
        raise Stop('STOP non-finite margin')
    return z


def masked_mean(torch, loss, mask):
    k = mask.sum()
    return (loss * mask).sum() / k if k > 0 else loss.sum() * 0.0


def job_env():
    """Launcher metadata, if present (the gpu tool writes job.env next to the job)."""
    out = {k: os.environ.get(k) for k in ('GPU_JOB_ID', 'GPU_JOB_DIR', 'CUDA_VISIBLE_DEVICES')}
    d = os.environ.get('GPU_JOB_DIR')
    try:
        if d and (Path(d) / 'job.env').is_file():
            kv = dict(line.split('=', 1) for line in (Path(d) / 'job.env').read_text().splitlines() if '=' in line)
            out.update({k.lower(): kv.get(k) for k in ('COMMIT', 'BRANCH', 'DIRTY', 'JOB_ID')})
    except Exception as e:  # metadata only
        out['job_env_error'] = repr(e)
    out.setdefault('commit', None)
    if out['commit'] is None:
        out['commit'] = '미측정'
    return out


def train_config(torch, cfg, entry, seed, data_np, device, out, base):
    """Training recipe identical to train_seq.train_arm (AdamW, batch, epochs, early stop on SELECT, scalar T on CAL).
    EVAL rows: margins and calibrated seven-leaf probabilities are written; no EVAL label is compared to a prediction."""
    x, C, S, ctx, bctx = data_np; t = cfg['training']; clip = cfg['nll_clip']; tag = f'{entry["id"]} s{seed}'; out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model = build_model(torch, cfg, entry).to(device).float()
    data = Data(torch, device, x, C, S, ctx, bctx)
    rows = {r: torch.as_tensor(np.flatnonzero(C[r]), dtype=torch.int64, device=device) for r in ('fit',) + ROLES}
    opt = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=t['weight_decay'])
    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    F = torch.nn.functional; y7_tune = C['y7'][C['tuning']]; y10_tune = C['y10'][C['tuning']]
    best = {'epoch': None, 'nll7': float('inf'), 'state': None}; curve = []; wall_cap_hit = False; n_train = len(rows['fit'])
    for epoch in range(1, t['max_epochs'] + 1):
        e0 = time.monotonic(); model.train(); perm = rows['fit'][torch.randperm(n_train, generator=gen).to(device)]
        tot = torch.zeros(4, dtype=torch.float64, device=device); nb = 0
        for s in range(0, n_train, t['batch_size']):
            idx = perm[s:s + t['batch_size']]
            zp, zw, zg = forward(model, data, idx)
            lp = F.cross_entropy(zp, data.y10[idx])
            lw = masked_mean(torch, F.binary_cross_entropy_with_logits(zw, data.whiff[idx], reduction='none'), data.is_whiff_row[idx].float())
            lg = masked_mean(torch, F.binary_cross_entropy_with_logits(zg, data.gb[idx], reduction='none'), data.is_bip_row[idx].float())
            loss = lp + lw + lg
            if not torch.isfinite(loss):
                raise Stop(f'STOP non-finite training loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += torch.stack([loss.detach(), lp.detach(), lw.detach(), lg.detach()]).double(); nb += 1
        z = predict(torch, model, data, rows['tuning'], t['eval_batch_size'])
        q = seven_from_margins(z['parent'], z['narrow_whiff'], z['ground_ball_per_pitch']); v = nll7(q, y7_tune, clip)
        from scipy.special import logsumexp
        v10 = float(np.mean(logsumexp(z['parent'], axis=1) - z['parent'][np.arange(len(y10_tune)), y10_tune]))
        improved = v < best['nll7']
        if improved:
            best = {'epoch': epoch, 'nll7': v, 'state': {k: w.detach().cpu().clone() for k, w in model.state_dict().items()}}
        m = (tot / max(nb, 1)).cpu().tolist()
        curve.append({'epoch': epoch, 'train_loss': m[0], 'train_parent_ce': m[1], 'train_whiff_bce': m[2], 'train_ground_ball_bce': m[3], 'batches': nb,
                      'select_seven_leaf_nll_uncalibrated': v, 'select_parent10_nll_uncalibrated': v10, 'improved': bool(improved), 'seconds': time.monotonic() - e0})
        print(f'[{tag}] epoch {epoch} train {m[0]:.6f} select_nll7 {v:.6f} best {best["nll7"]:.6f}@{best["epoch"]}', flush=True)
        if epoch - best['epoch'] >= t['patience']:
            break
        if time.monotonic() - start > t['wall_cap_seconds']:
            wall_cap_hit = True; break
    model.load_state_dict(best['state'])
    margins = {r: predict(torch, model, data, rows[r], t['eval_batch_size']) for r in ROLES}
    cal = C['calibration']; y10c = C['y10'][cal]; zc = margins['calibration']
    wm = y10c == 1; bm = np.isin(y10c, BIP)
    T = {'parent': temperature(zc['parent'], y10c), 'narrow_whiff': temperature(zc['narrow_whiff'][wm], C['narrow_whiff'][cal][wm].astype(np.float64)),
         'ground_ball_per_pitch': temperature(zc['ground_ball_per_pitch'][bm], C['ground_ball_per_pitch'][cal][bm].astype(np.float64))}
    ze = margins['evaluation']; q_eval = seven_from_margins(ze['parent'], ze['narrow_whiff'], ze['ground_ball_per_pitch'], T).astype(np.float64)
    zt = margins['tuning']; best_check = nll7(seven_from_margins(zt['parent'], zt['narrow_whiff'], zt['ground_ball_per_pitch']), y7_tune, clip)
    if abs(best_check - best['nll7']) > 1e-9:
        raise Stop(f'STOP restored best state does not reproduce its SELECT NLL: {best_check} vs {best["nll7"]}')
    np.savez(out / 'margins.npz', **{f'{r}__{h}': margins[r][h] for r in ROLES for h in HEADS})
    save_json(out / 'temperatures.json', {'temperatures': T, 'bounds_T': list(T_BOUNDS), 'fit_rows': {'parent': int(cal.sum()), 'narrow_whiff': int(wm.sum()), 'ground_ball_per_pitch': int(bm.sum())},
                                          'split': 'calibration', 'rule': 'one scalar T per head; copy of scale_runner.temperature'})
    np.save(out / 'seven-leaf-evaluation.npy', q_eval, allow_pickle=False)
    save_json(out / 'curve.json', {'config_id': entry['id'], 'seed': seed, 'epochs': curve})
    torch.save(best['state'], out / 'model-state.pt')
    cuda = device.type == 'cuda'; prop = torch.cuda.get_device_properties(0) if cuda else None
    receipt = dict(base, seed=seed, epochs_run=len(curve), best_epoch=best['epoch'], best_select_seven_leaf_nll_uncalibrated=best['nll7'],
                   early_stopped=bool(len(curve) < t['max_epochs'] and not wall_cap_hit), wall_cap_hit=wall_cap_hit, elapsed_seconds=time.monotonic() - start,
                   parameters=int(sum(p.numel() for p in model.parameters())), rows={r: int(C[r].sum()) for r in ('fit',) + ROLES},
                   temperatures=T, device=str(device),
                   peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None,
                   gpu_name=prop.name if cuda else None, gpu_uuid=str(getattr(prop, 'uuid', '미측정')) if cuda else None,
                   gpu_total_memory_bytes=int(prop.total_memory) if cuda else None,
                   versions={'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if cuda else None,
                             'numpy': np.__version__, 'scipy': __import__('scipy').__version__},
                   determinism={'torch_manual_seed': seed, 'numpy_seed': seed, 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'use_deterministic_algorithms': 'True (warn_only)',
                                'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'precision': 'fp32',
                                'caveat': 'CUDA/cuDNN runs are not guaranteed bit-reproducible across runs, cards or library versions'},
                   held_out_role_metrics='none: this script writes margins and probabilities only; SELECT NLL is the only metric it computes',
                   outputs_sha256={f: sha(out / f) for f in OUTPUT_FILES}, utc=time.time())
    save_json(out / 'receipt.json', receipt)
    return receipt


def run(args):
    cfg = load_config(args.grid); entry = grid_entry(cfg, args.config_id)
    if entry is None or args.seed not in cfg['seeds']:
        print('usage: --config-id must be a registered grid id and --seed a registered seed', file=sys.stderr); return EXIT_USAGE
    chash = config_hash(cfg, entry)
    if args.seed != 0 and args.selected_config_hash != chash:
        print(f'usage: seeds 1,2 run only for the config selected by select_config.py; pass --selected-config-hash <config hash of {entry["id"]} from SELECTION.json>', file=sys.stderr)
        return EXIT_USAGE
    torch, _ = imports()
    out = Path(args.out) if args.out else REPO / 'runs' / 'ws-seq2' / EXP_ID / f'{entry["id"]}-s{args.seed}'
    if out.name != f'{entry["id"]}-s{args.seed}':
        print(f'usage: --out must be a run directory named {entry["id"]}-s{args.seed}', file=sys.stderr); return EXIT_USAGE
    if (out / 'receipt.json').exists():
        print('BLOCKED: receipt exists (a rerun needs a fresh --out and a recorded reason): ' + str(out), file=sys.stderr); return EXIT_EXISTS
    try:
        man, man_sha, seen, code = verify_package(args.inputs, args.base_inputs, cfg, args.grid)
    except HashMismatch as e:
        print('INPUT HASH MISMATCH: ' + str(e), file=sys.stderr); return EXIT_HASH
    try:
        x, C, S, edges = load_package(args.inputs, args.base_inputs, cfg)
        order = (S['game_pk'], S['at_bat_number'], S['pitch_number'])
        ctx = build_context_index(S['pitcher'], *order, entry['context_length']); report = {'pitcher_side': check_context(ctx, S['pitcher'], *order)}
        bctx = None
        if entry['batter_context'] > 0:
            bctx = build_context_index(S['batter'], *order, entry['batter_context']); report['batter_side'] = check_context(bctx, S['batter'], *order)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = {'id': EXP_ID, 'config_id': entry['id'], 'config_hash': chash, 'config_entry': entry, 'selected_config_hash_argument': args.selected_config_hash,
                'data_role': cfg['data_role'], 'manifest_sha256': man_sha, 'registration_sha256': man.get('registration_sha256'), 'review_sha256': man.get('review_sha256'),
                'inputs_sha256': seen, 'train_seq2_sha256': code['train_seq2.py'], 'config_sha256': code['config'], 'config_path': str(args.grid),
                'context_index': report, 'bucket_edges_sha256': seen[cfg['inputs']['bucket_edges_file']], 'job': job_env(), 'argv': sys.argv[1:]}
        train_config(torch, cfg, entry, args.seed, (x, C, S, ctx, bctx), device, out, base)
        return EXIT_OK
    except Stop as e:
        print(str(e), file=sys.stderr); return EXIT_STOP


# ---------------------------------------------------------------- selfcheck (synthetic unit checks; no package, no real data)
def synthetic_tokens(rng, n, v):
    """Random integer token sources with several pitchers per game and batters that come up more than once."""
    game = rng.integers(1, 4, n); ab = rng.integers(1, 25, n); pn = np.zeros(n, np.int64); seen = {}
    for i in range(n):
        k = (game[i], ab[i]); seen[k] = seen.get(k, 0) + 1; pn[i] = seen[k]
    pitcher = 10 + (ab - 1) // 8 + 10 * (ab % 2); batter = 100 + (ab % 6) + 50 * (ab % 2)   # one pitcher and one batter per at-bat
    S = {'game_pk': game, 'at_bat_number': ab, 'pitch_number': pn, 'pitcher': pitcher, 'batter': batter, 'pitch_type': rng.integers(0, v['pitch_type'], n),
         'outcome10': rng.integers(0, v['outcome'], n), 'balls': rng.integers(0, v['balls'], n), 'strikes': rng.integers(0, v['strikes'], n)}
    for f in RICH_FIELDS:
        S[f] = rng.integers(0, v[RICH_VOCAB[f]], n)
    return S


def selfcheck(args):
    torch, _ = imports(); from scipy.special import softmax, expit
    out = {'id': EXP_ID, 'mode': 'train_seq2 --selfcheck (synthetic, CPU)', 'torch': torch.__version__, 'numpy': np.__version__, 'checks': {}}
    ck = out['checks']; rng = np.random.default_rng(20261011)
    # (1) index builder vs brute force for both group kinds (pitcher side, batter side), shuffled rows, groups longer than L
    n = 1500; S = synthetic_tokens(rng, n, {'pitch_type': 19, 'outcome': 10, 'balls': 4, 'strikes': 3, 'zone': 15, 'release_speed': 9, 'pfx_x': 6, 'pfx_z': 6})
    order = (S['game_pk'], S['at_bat_number'], S['pitch_number']); res = {}
    for side, group in (('pitcher_side', S['pitcher']), ('batter_side', S['batter'])):
        for L in (4, 16, 64):
            fast = build_context_index(group, *order, L); brute = brute_context_index(group, *order, L); rep = check_context(fast, group, *order)
            res[f'{side}_L{L}'] = dict(rep, equal_to_brute_force=bool(np.array_equal(fast, brute)))
            assert np.array_equal(fast, brute) and rep['rows_without_context'] > 0 and (L == 64 or rep['rows_full_context'] > 0)
    ck['context_index_vs_brute_force'] = res
    bfast = build_context_index(S['batter'], *order, 16); v0 = bfast >= 0
    other_pitcher = int((S['pitcher'][np.where(v0, bfast, 0)] != S['pitcher'][:, None])[v0].sum()); assert other_pitcher > 0   # batter side really crosses pitchers
    ck['batter_side_slots_from_another_pitcher'] = other_pitcher
    fast = build_context_index(S['pitcher'], *order, 16); neg = {}
    j = int(np.flatnonzero(fast[:, 0] >= 0)[0])
    for name in ('self_reference', 'later_pitch', 'other_game', 'other_group'):
        arr = fast.copy()
        if name == 'self_reference':
            arr[j, 0] = j
        if name == 'later_pitch':
            arr[fast[j, 0], 0] = j
        if name == 'other_game':
            arr[j, 0] = int(np.flatnonzero(S['game_pk'] != S['game_pk'][j])[0])
        if name == 'other_group':
            arr[j, 0] = int(np.flatnonzero((S['pitcher'] != S['pitcher'][j]) & (S['game_pk'] == S['game_pk'][j]))[0])
        try:
            check_context(arr, S['pitcher'], *order); neg[name] = 'NOT DETECTED'
        except Stop:
            neg[name] = 'detected'
    try:
        build_context_index(np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), 4); neg['duplicate_order_key'] = 'NOT DETECTED'
    except Stop:
        neg['duplicate_order_key'] = 'detected'
    assert all(x == 'detected' for x in neg.values()), neg
    ck['context_index_negative_controls'] = neg
    # (2) leaves closed form, (3) temperature recovers a known scale (as train_seq.py)
    p = np.zeros((1, 10)); p[0] = [.30, .20, .10, .05, .05, .05, .05, .05, .10, .05]
    got = leaves(p, np.array([.25]), np.array([.4]))[0]; want = np.array([.30, .10, .10, .05, .15, .12, .18])
    assert np.allclose(got, want, atol=1e-15); ck['leaves_closed_form_max_abs'] = float(np.abs(got - want).max())
    try:
        leaves(p * 1.1, np.array([.25]), np.array([.4])); raise AssertionError('incoherent leaves accepted')
    except Stop:
        ck['leaves_incoherent_input'] = 'detected'
    z = rng.normal(size=(4000, 10)) * 1.5; y = np.array([rng.choice(10, p=softmax(r)) for r in z]); Tm = temperature(z * 1.7, y)
    zb = rng.normal(size=20000) * 2; yb = (rng.random(20000) < expit(zb)).astype(float); Tb = temperature(zb * .6, yb)
    assert abs(Tm - 1.7) < .12 and abs(Tb - .6) < .05, (Tm, Tb); ck['temperature_recovery'] = {'multiclass_true_1.7': Tm, 'binary_true_0.6': Tb}
    # (4) grid: twins identical, ids, distinct hashes
    cfg = load_config(args.grid); yp = Path(str(args.grid)[:-5] + '.yaml'); hashes = {e['id']: config_hash(cfg, e) for e in cfg['grid']}
    assert len(set(hashes.values())) == len(hashes)
    alt = json.loads(json.dumps(cfg)); alt['training']['lr'] = 0.002; alt2 = json.loads(json.dumps(cfg)); alt2['grid'][-1]['encoder_ff'] += 1
    assert config_hash(alt, alt['grid'][0]) != hashes[cfg['grid'][0]['id']] and config_hash(alt2, alt2['grid'][0]) == hashes[cfg['grid'][0]['id']]
    ck['grid'] = {'configs': len(cfg['grid']), 'config_hashes': hashes, 'hash_changes_with_shared_training_block': True, 'hash_independent_of_other_grid_entries': True}
    try:
        import yaml
        same = yaml.safe_load(yp.read_text()) == json.loads(Path(args.grid).read_text())
        ck['config_twins'] = {'json': str(args.grid), 'yaml': str(yp), 'identical_values': bool(same), 'json_sha256': sha(args.grid), 'yaml_sha256': sha(yp)}
        assert same, 'YAML and JSON configs differ'
    except ImportError:
        ck['config_twins'] = {'identical_values': None, 'note': 'pyyaml missing here: twin comparison NOT run'}
    # (5) per grid config: shapes, and the leak test extended to the new fields
    n = 600; d = cfg['inputs']['feature_dimension']; v = cfg['vocab']; S = synthetic_tokens(rng, n, v)
    C = {'y10': S['outcome10'].copy(), 'narrow_whiff': rng.integers(0, 2, n), 'ground_ball_per_pitch': rng.integers(0, 2, n)}
    x = rng.normal(size=(n, d)).astype(np.float32); dev = torch.device('cpu'); allrows = torch.arange(n); order = (S['game_pk'], S['at_bat_number'], S['pitch_number'])
    key = S['at_bat_number'].astype(np.int64) * 100000 + S['pitch_number']; leak = {}; params = {}
    vocab_of = {'outcome10': v['outcome'], 'pitch_type': v['pitch_type'], **{f: v[RICH_VOCAB[f]] for f in RICH_FIELDS}}

    def margins_of(net, S_, ctx, bctx):
        r = predict(torch, net, Data(torch, dev, x, C, S_, ctx, bctx), allrows, 256); return np.column_stack([r['parent'], r['narrow_whiff'], r['ground_ball_per_pitch']])

    for e in cfg['grid']:
        torch.manual_seed(0); net = build_model(torch, cfg, e); params[e['id']] = int(sum(p.numel() for p in net.parameters()))
        ctx = build_context_index(S['pitcher'], *order, e['context_length']); bctx = build_context_index(S['batter'], *order, e['batter_context']) if e['batter_context'] else None
        base = margins_of(net, S, ctx, bctx); assert base.shape == (n, 12)
        i = int(np.flatnonzero((ctx[:, 0] >= 0) & (ctx == -1).any(1) & np.isin(np.arange(n), ctx))[0])   # has context, has padding, is used by a later row
        users = (ctx == i).any(1) | ((bctx == i).any(1) if bctx is not None else np.zeros(n, bool)); rep = {'changed_row': i, 'rows_using_it_as_context': int(users.sum())}
        for f in ('outcome10', 'pitch_type') + RICH_FIELDS:
            S_alt = {k: a.copy() for k, a in S.items()}; S_alt[f][i] = (S_alt[f][i] + 1) % vocab_of[f]
            alt = margins_of(net, S_alt, ctx, bctx); own = float(np.abs(alt[i] - base[i]).max()); later = float(np.abs(alt[users] - base[users]).max()); other = float(np.abs(alt[~users] - base[~users]).max())
            used = f in ('outcome10', 'pitch_type') or e['tokens'] == 'physics_zone'
            assert own == 0.0 and other == 0.0 and (later > 0) == used, (e['id'], f, own, later, other)
            rep[f] = {'own_margin_change': own, 'users_max_margin_change': later, 'all_other_rows_change': other, 'field_is_a_token_in_this_config': used}
        # stronger: rewrite EVERY token field of the row itself and of every later pitch of its game -> its own margins must not move
        worst = 0.0; moved = 0
        for i2 in rng.choice(n, 12, replace=False):
            at_or_after = (S['game_pk'] == S['game_pk'][i2]) & (key >= key[i2]); S_alt = {k: a.copy() for k, a in S.items()}
            for f in vocab_of:
                S_alt[f][at_or_after] = (S_alt[f][at_or_after] + 1 + rng.integers(0, vocab_of[f] - 1, int(at_or_after.sum()))) % vocab_of[f]
            alt = margins_of(net, S_alt, ctx, bctx); worst = max(worst, float(np.abs(alt[i2] - base[i2]).max())); moved += int((np.abs(alt - base).max(1) > 0).sum())
        assert worst == 0.0 and moved > 0
        rep['current_and_later_pitches_rewritten'] = {'rows_tested': 12, 'own_margin_change_max': worst, 'later_rows_that_moved_total': moved}
        # padded slots: token content must not matter; empty contexts are finite
        empty = np.flatnonzero((ctx[:, 0] < 0) & ((bctx[:, 0] < 0) if bctx is not None else True)); net.eval()
        with torch.no_grad():
            D_ = Data(torch, dev, x, C, S, ctx, bctx); ie = torch.as_tensor(empty); t1 = D_.tokens(ie, 0); t2 = {k: (a if k in ('valid', 'pos') else a * 0 + 1) for k, a in t1.items()}
            b1 = D_.tokens(ie, 1) if bctx is not None else None; b2 = None if b1 is None else {k: (a if k in ('valid', 'pos') else a * 0 + 1) for k, a in b1.items()}
            a = net(D_.x[ie], t1, b1)[0]; b = net(D_.x[ie], t2, b2)[0]
        assert len(empty) > 0 and torch.isfinite(a).all() and torch.equal(a, b)
        rep['empty_context'] = {'rows': int(len(empty)), 'finite': True, 'independent_of_padding_token_content': True}
        leak[e['id']] = rep
    ck['no_current_pitch_leak_per_config'] = leak; ck['parameters_per_config'] = params
    # (6) the base grid entry with L=32, d=64, 2 layers is the SEQ-001 N1 architecture: same initial state and same forward
    old = HERE / 'train_seq.py'; old_cfg = REPO / 'configs' / 'EXP-WS-SEQ-001.json'; ref = [e for e in cfg['grid'] if (e['context_length'], e['tokens'], e['batter_context'], e['token_dim'], e['encoder_layers'], e['encoder_ff']) == (32, 'base', 0, 64, 2, 128)]
    if old.is_file() and old_cfg.is_file() and ref:
        import importlib.util
        sys.dont_write_bytecode = True   # the by-path import of train_seq.py must not leave a __pycache__ in the repo
        spec = importlib.util.spec_from_file_location('ws_seq_train_001', old); T1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T1)
        c1 = T1.load_config(old_cfg); torch.manual_seed(0); n1 = T1.build_model(torch, c1, 'N1'); torch.manual_seed(0); n2 = build_model(torch, cfg, ref[0])
        s1, s2 = n1.state_dict(), n2.state_dict(); same_state = list(s1) == list(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
        ctx = build_context_index(S['pitcher'], *order, 32)
        m1 = T1.predict(torch, n1, T1.Data(torch, dev, x, C, S, ctx), allrows, 256); m2 = predict(torch, n2, Data(torch, dev, x, C, S, ctx), allrows, 256)
        diff = max(float(np.abs(m1[h] - m2[h]).max()) for h in HEADS); same_recipe = all(c1['training'][k] == cfg['training'][k] for k in c1['training']) and c1['model']['trunk_hidden'] == cfg['model']['trunk_hidden']
        assert same_state and diff == 0.0 and same_recipe
        ck['reference_entry_equals_seq001_N1'] = {'entry': ref[0]['id'], 'same_parameter_names_and_initial_values_for_seed_0': bool(same_state), 'forward_max_abs_diff': diff, 'training_block_equal': bool(same_recipe),
                                                  'train_seq_sha256': sha(old), 'note': 'architecture/initialisation/recipe identity on CPU; NOT a claim that a CUDA re-run reproduces the SEQ-001 numbers bit for bit'}
    else:
        ck['reference_entry_equals_seq001_N1'] = {'run': False, 'note': 'train_seq.py / EXP-WS-SEQ-001.json / reference entry not present here'}
    out['verdict'] = 'SELFCHECK_PASS'; out['train_seq2_sha256'] = sha(Path(__file__).resolve())
    if args.selfcheck_out:
        save_json(args.selfcheck_out, out)
    print(json.dumps({'verdict': out['verdict'], 'checks': sorted(ck)}))
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs2')); ap.add_argument('--base-inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs'))
    ap.add_argument('--out', default=None); ap.add_argument('--grid', default=str(REPO / 'configs' / (EXP_ID + '.json')))
    ap.add_argument('--config-id'); ap.add_argument('--seed', type=int); ap.add_argument('--selected-config-hash', default=None)
    ap.add_argument('--selfcheck', action='store_true'); ap.add_argument('--selfcheck-out')
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args)
    if args.config_id is None or args.seed is None:
        ap.print_usage(sys.stderr); return EXIT_USAGE
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
