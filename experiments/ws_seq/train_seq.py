"""EXP-WS-SEQ-001 (E4) training: small pitch-sequence neural model on the prequential 199-feature base.

Arms: N0 = MLP on the 199 pre-pitch features; N1 = N0 trunk + transformer encoder over the up-to-L EARLIER pitches of
the same pitcher in the same game (strictly pre-pitch context). 2026 EXPOSED DEVELOPMENT; research model; never a
winner/adoption claim.

Runs on the A5000 server (one card per process, CUDA_VISIBLE_DEVICES set by the launcher) or on CPU (selfcheck).
Imports: numpy, scipy, torch only. No lightgbm / sklearn / pandas / pyyaml needed (the JSON config is authoritative).

  python experiments/ws_seq/train_seq.py --arm N0+N1 --seed 0 [--stop-rule-reference X]
      [--inputs data/ws-seq/inputs] [--out runs/ws-seq/EXP-WS-SEQ-001] [--config configs/EXP-WS-SEQ-001.json]
  python experiments/ws_seq/train_seq.py --selfcheck [--selfcheck-out FILE]     (synthetic unit checks, CPU)

Outputs per arm/seed under <out>/<arm>-s<seed>/: margins.npz, temperatures.json, seven-leaf-evaluation.npy,
curve.json, model-state.pt, receipt.json (written last).

Exit codes: 0 ok; 2 usage/config; 3 missing import; 4 registered stop rule (seed 0, arm N1, only with
--stop-rule-reference; all outputs are still written); 5 input/code hash mismatch; 6 registered STOP (non-finite,
temperature, coherence, layout); 7 output directory already holds a receipt.
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
EXP_ID = 'EXP-WS-SEQ-001'
ARMS = ('N0', 'N1')
HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')
ROLES = ('tuning', 'calibration', 'evaluation')
BIP = (3, 4, 5, 6, 7, 9)
T_BOUNDS = (.05, 20.)
EXIT_OK, EXIT_USAGE, EXIT_IMPORT, EXIT_STOP_RULE, EXIT_HASH, EXIT_STOP, EXIT_EXISTS = 0, 2, 3, 4, 5, 6, 7
CORPUS_MEMBERS = ('keys', 'y10', 'y7', 'games', 'fit', 'tuning', 'calibration', 'evaluation', 'narrow_whiff', 'ground_ball_per_pitch')
SEQ_MEMBERS = ('game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'pitch_type', 'outcome10', 'balls', 'strikes', 'stand')


class Stop(Exception):
    """Registered STOP: no result for this arm/seed."""


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


# ---------------------------------------------------------------- config
def load_config(path):
    cfg = json.loads(Path(path).read_text())
    c, m, t = cfg['context'], cfg['model'], cfg['training']
    ok = [cfg['id'] == EXP_ID, list(cfg['arms']) == list(ARMS), cfg['inputs']['feature_dimension'] > 0, tuple(cfg['labels']['bip_classes']) == BIP,
          cfg['labels']['parent_classes'] == 10, cfg['labels']['whiff_parent_class'] == 1, c['position_vocab'] == c['max_length'] + 1,
          m['token_dim'] % m['encoder_heads'] == 0, len(m['trunk_hidden']) == 2, t['optimizer'] == 'adamw', t['precision'] == 'fp32',
          tuple(cfg['temperature']['bounds_T']) == T_BOUNDS, m['activation'] == 'gelu' and m['encoder_activation'] == 'gelu', m['layer_norm'] is True,
          all(t['loss_weights'][h] == 1.0 for h in HEADS), c['same_batter_vocab'] == 2]
    if not all(ok):
        raise SystemExit('BLOCKED: config does not match what this source implements: ' + str(ok))
    return cfg


# ---------------------------------------------------------------- sequence index (strictly earlier pitches, same pitcher, same game)
def build_context_index(pitcher, game, at_bat, pitch_no, length):
    """(n, length) int32 row indices. Slot j of row i = the row of the pitch thrown (j+1) pitches EARLIER by the same
    pitcher in the same game, in (at_bat_number, pitch_number) order; -1 = padding. Never the row itself, never a later
    pitch. Vectorised: stable lexsort by (pitcher, game, at_bat, pitch_no); within a group position k sees k-1..k-length."""
    pitcher, game, at_bat, pitch_no = (np.asarray(a, np.int64) for a in (pitcher, game, at_bat, pitch_no))
    n = len(pitcher)
    order = np.lexsort((pitch_no, at_bat, game, pitcher))
    p, g, a, q = pitcher[order], game[order], at_bat[order], pitch_no[order]
    new = np.ones(n, bool)
    if n > 1:
        new[1:] = (p[1:] != p[:-1]) | (g[1:] != g[:-1])
        if (~new[1:] & (a[1:] == a[:-1]) & (q[1:] == q[:-1])).any():
            raise Stop('STOP duplicate (pitcher, game, at_bat_number, pitch_number): the earlier/later order is undefined')
    start = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    earlier = np.arange(n) - start                                    # number of earlier pitches in the group
    back = np.arange(1, length + 1)
    valid = back[None, :] <= earlier[:, None]
    src = np.where(valid, order[np.clip(np.arange(n)[:, None] - back[None, :], 0, None)], -1)
    ctx = np.empty((n, length), np.int32); ctx[order] = src
    return ctx


def brute_context_index(pitcher, game, at_bat, pitch_no, length):
    """Independent O(n * group) python loop used only by the selfcheck."""
    n = len(pitcher); out = np.full((n, length), -1, np.int32)
    for i in range(n):
        earlier = [(int(at_bat[k]), int(pitch_no[k]), k) for k in range(n)
                   if pitcher[k] == pitcher[i] and game[k] == game[i] and (at_bat[k], pitch_no[k]) < (at_bat[i], pitch_no[i])]
        earlier.sort(reverse=True)
        for j, (_, _, k) in enumerate(earlier[:length]):
            out[i, j] = k
    return out


def check_context(ctx, pitcher, game, at_bat, pitch_no):
    """Run-time proof on the real index: every non-padding slot is the same pitcher and game and strictly earlier;
    slots are strictly decreasing in time; padding is a suffix; no row references itself."""
    n, L = ctx.shape; valid = ctx >= 0; safe = np.where(valid, ctx, 0)
    key = np.asarray(at_bat, np.int64) * 100000 + np.asarray(pitch_no, np.int64)
    if (np.asarray(pitch_no) >= 100000).any() or (np.asarray(pitch_no) < 0).any():
        raise Stop('STOP pitch_number outside the order-key range')
    ck = key[safe]
    ok = (((np.asarray(pitcher)[safe] == np.asarray(pitcher)[:, None]) & (np.asarray(game)[safe] == np.asarray(game)[:, None]) & (ck < key[:, None])) | ~valid).all()
    ok = ok and bool(((ck[:, 1:] < ck[:, :-1]) | ~valid[:, 1:]).all()) and bool((valid[:, :-1] | ~valid[:, 1:]).all())
    ok = ok and not bool((ctx == np.arange(n)[:, None]).any())
    if not ok:
        raise Stop('STOP context index violates the strictly-earlier / same pitcher / same game contract')
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
def verify_package(inputs, cfg, config_path):
    """sha256 of every file read by the training, and of this script + the config against the manifest binding."""
    I = cfg['inputs']; mp = Path(inputs) / I['manifest_file']
    if not mp.is_file():
        raise HashMismatch('manifest missing: ' + str(mp))
    man = json.loads(mp.read_text()); seen = {}
    for name in (I['features_file'], I['corpus_file'], I['sequence_file']):
        p = Path(inputs) / name
        seen[name] = sha(p) if p.is_file() else None
        if name not in man['files'] or seen[name] != man['files'][name]['sha256']:
            raise HashMismatch(f'input sha256 mismatch or missing: {name}')
    code = {'train_seq.py': sha(Path(__file__).resolve()), 'config': sha(config_path)}
    for k, v in code.items():
        if man['bound_sources'].get(k) != v:
            raise HashMismatch(f'{k} differs from the sha256 bound in the manifest (prepare inputs again after any reviewed code/config change)')
    if man.get('id') != EXP_ID:
        raise HashMismatch('manifest id')
    return man, sha(mp), seen, code


def load_package(inputs, cfg):
    I = cfg['inputs']; d = cfg['inputs']['feature_dimension']
    x = np.load(Path(inputs) / I['features_file'], allow_pickle=False)
    with np.load(Path(inputs) / I['corpus_file'], allow_pickle=False) as z:
        C = {k: np.array(z[k]) for k in CORPUS_MEMBERS}
    with np.load(Path(inputs) / I['sequence_file'], allow_pickle=False) as z:
        S = {k: np.array(z[k]) for k in SEQ_MEMBERS}
    n = len(C['y10'])
    if x.shape != (n, d) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise Stop(f'STOP feature matrix shape/dtype/non-finite: {x.shape} {x.dtype}')
    if any(len(v) != n for v in list(C.values()) + list(S.values())) or any(v.dtype.kind != 'i' for v in S.values()):
        raise Stop('STOP package row counts / integer dtypes')
    same = [np.array_equal(S['game_pk'], C['keys'][:, 0]), np.array_equal(S['at_bat_number'], C['keys'][:, 1]), np.array_equal(S['pitch_number'], C['keys'][:, 2]),
            np.array_equal(S['outcome10'], C['y10']), np.array_equal(C['games'], C['keys'][:, 0]),
            bool(np.all(C['fit'].astype(int) + C['tuning'] + C['calibration'] + C['evaluation'] == 1)),
            np.array_equal(seven_label(C['y10'], C['narrow_whiff'], C['ground_ball_per_pitch']), C['y7'])]
    v = cfg['context']
    rng = [S['pitch_type'].min() >= 0, S['pitch_type'].max() < v['pitch_type_vocab'], S['outcome10'].min() >= 0, S['outcome10'].max() < v['outcome_vocab'],
           S['balls'].min() >= 0, S['balls'].max() < v['balls_vocab'], S['strikes'].min() >= 0, S['strikes'].max() < v['strikes_vocab']]
    if not all(same) or not all(bool(r) for r in rng):
        raise Stop('STOP package identity (sequence rows vs corpus rows / one role per row / y7) or token range: ' + str(same) + str([bool(r) for r in rng]))
    return x, C, S


# ---------------------------------------------------------------- model
def build_model(torch, cfg, arm):
    nn = torch.nn; m = cfg['model']; c = cfg['context']; d_in = cfg['inputs']['feature_dimension']; h1, h2 = m['trunk_hidden']; D = m['token_dim']

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # trunk is built first so that N0 and N1 share the same trunk initialisation for a given seed
            self.trunk = nn.Sequential(nn.Linear(d_in, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(m['dropout']),
                                       nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(m['dropout']))
            self.seq = arm == 'N1'
            if self.seq:
                self.pad_type, self.pad_out = c['pitch_type_vocab'], c['outcome_vocab']
                self.e_type = nn.Embedding(c['pitch_type_vocab'] + 1, D, padding_idx=self.pad_type)
                self.e_out = nn.Embedding(c['outcome_vocab'] + 1, D, padding_idx=self.pad_out)
                self.e_balls = nn.Embedding(c['balls_vocab'], D); self.e_strikes = nn.Embedding(c['strikes_vocab'], D)
                self.e_same = nn.Embedding(c['same_batter_vocab'], D); self.e_pos = nn.Embedding(c['position_vocab'], D, padding_idx=0)
                layer = nn.TransformerEncoderLayer(d_model=D, nhead=m['encoder_heads'], dim_feedforward=m['encoder_ff'], dropout=m['encoder_dropout'],
                                                   activation='gelu', batch_first=True, norm_first=False)
                self.encoder = nn.TransformerEncoder(layer, num_layers=m['encoder_layers'], enable_nested_tensor=False)
                self.no_context = nn.Parameter(torch.zeros(D)); nn.init.normal_(self.no_context, std=0.02)
            width = h2 + (D if self.seq else 0)
            self.head_parent = nn.Linear(width, cfg['labels']['parent_classes']); self.head_whiff = nn.Linear(width, 1); self.head_gb = nn.Linear(width, 1)

        def forward(self, x, tok=None):
            h = self.trunk(x)
            if self.seq:
                valid = tok['valid']; z = torch.zeros_like(tok['type'])
                e = (self.e_type(torch.where(valid, tok['type'], z + self.pad_type)) + self.e_out(torch.where(valid, tok['out'], z + self.pad_out))
                     + self.e_balls(torch.where(valid, tok['balls'], z)) + self.e_strikes(torch.where(valid, tok['strikes'], z))
                     + self.e_same(torch.where(valid, tok['same'], z)) + self.e_pos(torch.where(valid, tok['pos'], z)))
                empty = ~valid.any(1); pad = ~valid; pad = pad & ~(empty[:, None] & (tok['pos'] == 1))  # an all-padding row would give NaN attention
                s = self.encoder(e, src_key_padding_mask=pad)
                w = valid.unsqueeze(-1).to(s.dtype)
                pooled = (s * w).sum(1) / w.sum(1).clamp(min=1.0)
                pooled = torch.where(empty[:, None], self.no_context.expand_as(pooled), pooled)
                h = torch.cat([h, pooled], 1)
            return self.head_parent(h), self.head_whiff(h).squeeze(-1), self.head_gb(h).squeeze(-1)

    return Net()


class Data:
    """Everything on the device once; contexts are gathered per batch from the (n, L) index (no materialised copies)."""

    def __init__(self, torch, device, x, C, S, ctx):
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.torch = torch; self.x = t(x, torch.float32); self.ctx = t(ctx, torch.int64)
        self.type = t(S['pitch_type'], torch.int64); self.out = t(S['outcome10'], torch.int64); self.balls = t(S['balls'], torch.int64)
        self.strikes = t(S['strikes'], torch.int64); self.batter = t(S['batter'], torch.int64)
        self.y10 = t(C['y10'], torch.int64); self.whiff = t(C['narrow_whiff'], torch.float32); self.gb = t(C['ground_ball_per_pitch'], torch.float32)
        self.is_whiff_row = self.y10 == 1; self.is_bip_row = torch.isin(self.y10, t(np.array(BIP), torch.int64))
        self.pos = torch.arange(1, ctx.shape[1] + 1, device=device, dtype=torch.int64)[None, :]

    def tokens(self, idx):
        c = self.ctx[idx]; valid = c >= 0; s = c.clamp(min=0)
        return {'valid': valid, 'type': self.type[s], 'out': self.out[s], 'balls': self.balls[s], 'strikes': self.strikes[s],
                'same': (self.batter[s] == self.batter[idx][:, None]).to(self.torch.int64), 'pos': self.pos.expand_as(c)}


def predict(torch, model, data, rows, batch):
    """Raw margins (float64 numpy) for the given row indices, eval mode, no grad."""
    model.eval(); P, W, G = [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            idx = rows[s:s + batch]
            zp, zw, zg = model(data.x[idx], data.tokens(idx) if model.seq else None)
            P.append(zp.double().cpu().numpy()); W.append(zw.double().cpu().numpy()); G.append(zg.double().cpu().numpy())
    z = {'parent': np.concatenate(P), 'narrow_whiff': np.concatenate(W), 'ground_ball_per_pitch': np.concatenate(G)}
    if not all(np.isfinite(v).all() for v in z.values()):
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


def train_arm(torch, cfg, arm, seed, data_np, device, out, base, stop_reference):
    x, C, S, ctx = data_np; t = cfg['training']; clip = cfg['nll_clip']; out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        print('BLOCKED: receipt exists (a rerun needs a fresh --out and a recorded reason): ' + str(out), file=sys.stderr); sys.exit(EXIT_EXISTS)
    start = time.monotonic()
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    model = build_model(torch, cfg, arm).to(device).float()
    data = Data(torch, device, x, C, S, ctx)
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
            zp, zw, zg = model(data.x[idx], data.tokens(idx) if model.seq else None)
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
        print(f'[{arm} s{seed}] epoch {epoch} train {m[0]:.6f} select_nll7 {v:.6f} best {best["nll7"]:.6f}@{best["epoch"]}', flush=True)
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
    save_json(out / 'curve.json', {'arm': arm, 'seed': seed, 'epochs': curve})
    torch.save(best['state'], out / 'model-state.pt')
    stop = {'applies': bool(arm == 'N1' and seed == 0 and stop_reference is not None), 'reference_select_nll': stop_reference,
            'rule': 'seed 0, arm N1: triggered iff best uncalibrated SELECT seven-leaf NLL >= reference (not lower); then seeds 1,2 are not run'}
    stop['triggered'] = bool(stop['applies'] and not best['nll7'] < stop_reference)
    cuda = device.type == 'cuda'; prop = torch.cuda.get_device_properties(0) if cuda else None
    files = ('margins.npz', 'temperatures.json', 'seven-leaf-evaluation.npy', 'curve.json', 'model-state.pt')
    receipt = dict(base, arm=arm, seed=seed, epochs_run=len(curve), best_epoch=best['epoch'], best_select_seven_leaf_nll_uncalibrated=best['nll7'],
                   early_stopped=bool(len(curve) < t['max_epochs'] and not wall_cap_hit), wall_cap_hit=wall_cap_hit, elapsed_seconds=time.monotonic() - start,
                   parameters=int(sum(p.numel() for p in model.parameters())), rows={r: int(C[r].sum()) for r in ('fit',) + ROLES},
                   temperatures=T, stop_rule=stop, device=str(device),
                   peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None,
                   gpu_name=prop.name if cuda else None, gpu_uuid=str(getattr(prop, 'uuid', '미측정')) if cuda else None,
                   gpu_total_memory_bytes=int(prop.total_memory) if cuda else None,
                   versions={'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if cuda else None,
                             'numpy': np.__version__, 'scipy': __import__('scipy').__version__},
                   determinism={'torch_manual_seed': seed, 'numpy_seed': seed, 'cudnn_deterministic': True, 'cudnn_benchmark': False, 'use_deterministic_algorithms': 'True (warn_only)',
                                'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'), 'precision': 'fp32',
                                'caveat': 'CUDA/cuDNN runs are not guaranteed bit-reproducible across runs, cards or library versions'},
                   outputs_sha256={f: sha(out / f) for f in files}, utc=time.time())
    save_json(out / 'receipt.json', receipt)
    return receipt


def run(args):
    cfg = load_config(args.config); torch, _ = imports()
    arms = args.arm.split('+')
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms) or args.seed not in cfg['seeds']:
        print('usage: --arm N0 | N1 | N0+N1 and a registered --seed', file=sys.stderr); return EXIT_USAGE
    try:
        man, man_sha, seen, code = verify_package(args.inputs, cfg, args.config)
    except HashMismatch as e:
        print('INPUT HASH MISMATCH: ' + str(e), file=sys.stderr); return EXIT_HASH
    try:
        x, C, S = load_package(args.inputs, cfg)
        ctx = build_context_index(S['pitcher'], S['game_pk'], S['at_bat_number'], S['pitch_number'], cfg['context']['max_length'])
        ctx_report = check_context(ctx, S['pitcher'], S['game_pk'], S['at_bat_number'], S['pitch_number'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = {'id': EXP_ID, 'data_role': cfg['data_role'], 'manifest_sha256': man_sha, 'inputs_sha256': seen, 'train_seq_sha256': code['train_seq.py'], 'config_sha256': code['config'],
                'config_path': str(args.config), 'context_index': ctx_report, 'job': job_env(), 'argv': sys.argv[1:]}
        rc = EXIT_OK
        for arm in arms:
            r = train_arm(torch, cfg, arm, args.seed, (x, C, S, ctx), device, Path(args.out) / f'{arm}-s{args.seed}', base, args.stop_rule_reference)
            if r['stop_rule']['triggered']:
                print(f'STOP_RULE: N1 seed 0 best SELECT NLL {r["best_select_seven_leaf_nll_uncalibrated"]} is not lower than the reference {args.stop_rule_reference}; seeds 1,2 must not run', file=sys.stderr)
                rc = EXIT_STOP_RULE
        return rc
    except Stop as e:
        print(str(e), file=sys.stderr); return EXIT_STOP


# ---------------------------------------------------------------- selfcheck (synthetic unit checks; no package, no real data)
def selfcheck(args):
    torch, _ = imports(); from scipy.special import softmax, expit
    out = {'id': EXP_ID, 'mode': 'train_seq --selfcheck (synthetic, CPU)', 'torch': torch.__version__, 'numpy': np.__version__, 'checks': {}}
    ck = out['checks']; rng = np.random.default_rng(20261011)
    # (1) index builder vs brute force, shuffled row order, groups longer than the context length
    n = 1500; pit = rng.integers(1, 6, n); gm = rng.integers(1, 5, n); ab = rng.integers(1, 30, n); pn = np.zeros(n, np.int64)
    seen = {}
    for i in range(n):
        k = (pit[i], gm[i], ab[i]); seen[k] = seen.get(k, 0) + 1; pn[i] = seen[k]
    perm = rng.permutation(n); pit, gm, ab, pn = pit[perm], gm[perm], ab[perm], pn[perm]
    res = {}
    for L in (4, 32):
        fast = build_context_index(pit, gm, ab, pn, L); brute = brute_context_index(pit, gm, ab, pn, L)
        rep = check_context(fast, pit, gm, ab, pn)
        res[str(L)] = dict(rep, equal_to_brute_force=bool(np.array_equal(fast, brute)))
        assert np.array_equal(fast, brute) and rep['rows_full_context'] > 0 and rep['rows_without_context'] > 0
    ck['context_index_vs_brute_force'] = res
    bad = fast.copy(); i = int(np.flatnonzero(fast[:, 0] >= 0)[0]); bad[i, 0] = i
    neg = {}
    for name, arr in (('self_reference', bad), ('later_pitch', None), ('other_game', None)):
        if name == 'later_pitch':
            arr = fast.copy(); j = int(np.flatnonzero(fast[:, 0] >= 0)[0]); arr[fast[j, 0], 0] = j
        if name == 'other_game':
            arr = fast.copy(); j = int(np.flatnonzero(fast[:, 0] >= 0)[0]); arr[j, 0] = int(np.flatnonzero(gm != gm[j])[0])
        try:
            check_context(arr, pit, gm, ab, pn); neg[name] = 'NOT DETECTED'
        except Stop:
            neg[name] = 'detected'
    assert all(v == 'detected' for v in neg.values()), neg
    try:
        build_context_index(np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), 4); neg['duplicate_order_key'] = 'NOT DETECTED'
    except Stop:
        neg['duplicate_order_key'] = 'detected'
    assert neg['duplicate_order_key'] == 'detected'
    ck['context_index_negative_controls'] = neg
    # (2) leaves closed form
    p = np.zeros((1, 10)); p[0] = [.30, .20, .10, .05, .05, .05, .05, .05, .10, .05]
    got = leaves(p, np.array([.25]), np.array([.4]))[0]; want = np.array([.30, .10, .10, .05, .15, .12, .18])
    assert np.allclose(got, want, atol=1e-15); ck['leaves_closed_form_max_abs'] = float(np.abs(got - want).max())
    try:
        leaves(p * 1.1, np.array([.25]), np.array([.4])); raise AssertionError('incoherent leaves accepted')
    except Stop:
        ck['leaves_incoherent_input'] = 'detected'
    # (3) temperature recovers a known scale
    z = rng.normal(size=(40000, 10)) * 1.5; y = np.array([rng.choice(10, p=softmax(r / 1.0)) for r in z[:4000]]); Tm = temperature(z[:4000] * 1.7, y)
    zb = rng.normal(size=20000) * 2; yb = (rng.random(20000) < expit(zb)).astype(float); Tb = temperature(zb * .6, yb)
    assert abs(Tm - 1.7) < .12 and abs(Tb - .6) < .05, (Tm, Tb); ck['temperature_recovery'] = {'multiclass_true_1.7': Tm, 'binary_true_0.6': Tb}
    # (4) config twins identical (JSON authoritative, YAML for humans)
    cfg = load_config(args.config); yp = Path(str(args.config)[:-5] + '.yaml')
    try:
        import yaml
        twin = yaml.safe_load(yp.read_text()); same = twin == json.loads(Path(args.config).read_text())
        ck['config_twins'] = {'json': str(args.config), 'yaml': str(yp), 'identical_values': bool(same), 'json_sha256': sha(args.config), 'yaml_sha256': sha(yp)}
        assert same, 'YAML and JSON configs differ'
    except ImportError:
        ck['config_twins'] = {'identical_values': None, 'note': 'pyyaml missing here: twin comparison NOT run'}
    # (5) model: shapes, empty context, padding invariance, no dependence on the current row's realised outcome
    n = 400; d = cfg['inputs']['feature_dimension']; L = cfg['context']['max_length']
    pit = rng.integers(1, 4, n); gm = rng.integers(1, 3, n); ab = np.arange(n) // 4 + 1; pn = np.arange(n) % 4 + 1
    S = {'pitch_type': rng.integers(0, 19, n), 'outcome10': rng.integers(0, 10, n), 'balls': rng.integers(0, 4, n), 'strikes': rng.integers(0, 3, n), 'batter': rng.integers(1, 9, n)}
    C = {'y10': S['outcome10'].copy(), 'narrow_whiff': rng.integers(0, 2, n), 'ground_ball_per_pitch': rng.integers(0, 2, n)}
    x = rng.normal(size=(n, d)).astype(np.float32); ctx = build_context_index(pit, gm, ab, pn, L); dev = torch.device('cpu')
    torch.manual_seed(0); net = build_model(torch, cfg, 'N1'); torch.manual_seed(0); net0 = build_model(torch, cfg, 'N0')
    same_trunk = all(torch.equal(a, b) for a, b in zip(net.trunk.state_dict().values(), net0.trunk.state_dict().values()))
    allrows = torch.arange(n); base = predict(torch, net, Data(torch, dev, x, C, S, ctx), allrows, 128)
    assert base['parent'].shape == (n, 10) and base['narrow_whiff'].shape == (n,) and same_trunk
    i = int(np.flatnonzero((ctx[:, 0] >= 0) & (ctx == -1).any(1))[0])  # a row with some context and some padding
    users = np.flatnonzero((ctx == i).any(1)); assert len(users) > 0
    S2 = {k: v.copy() for k, v in S.items()}; S2['outcome10'][i] = (S2['outcome10'][i] + 1) % 10; S2['pitch_type'][i] = (S2['pitch_type'][i] + 1) % 19
    alt = predict(torch, net, Data(torch, dev, x, C, S2, ctx), allrows, 128)
    own = float(np.abs(alt['parent'][i] - base['parent'][i]).max()); later = float(np.abs(alt['parent'][users] - base['parent'][users]).max())
    others = np.setdiff1d(np.arange(n), users); other = float(np.abs(alt['parent'][others] - base['parent'][others]).max())
    assert own == 0.0 and other == 0.0 and later > 0
    ck['no_current_pitch_leak'] = {'changed_row': i, 'own_margin_change': own, 'rows_using_it_as_context': int(len(users)), 'their_max_margin_change': later, 'all_other_rows_change': other}
    empty = np.flatnonzero(ctx[:, 0] < 0); net.eval()   # token content of padded slots must not matter
    with torch.no_grad():
        t1 = Data(torch, dev, x, C, S, ctx).tokens(torch.as_tensor(empty)); t2 = dict(t1); t2['type'] = t1['type'] * 0 + 7; t2['out'] = t1['out'] * 0 + 3
        a = net(torch.as_tensor(x[empty]), t1)[0]; b = net(torch.as_tensor(x[empty]), t2)[0]
    assert len(empty) > 0 and torch.isfinite(a).all() and torch.equal(a, b)
    ck['empty_context'] = {'rows': int(len(empty)), 'finite': True, 'independent_of_padding_token_content': True}
    ck['model'] = {'N0_parameters': int(sum(p.numel() for p in net0.parameters())), 'N1_parameters': int(sum(p.numel() for p in net.parameters())), 'N0_N1_same_trunk_init_for_same_seed': bool(same_trunk)}
    out['verdict'] = 'SELFCHECK_PASS'; out['train_seq_sha256'] = sha(Path(__file__).resolve())
    if args.selfcheck_out:
        save_json(args.selfcheck_out, out)
    print(json.dumps({'verdict': out['verdict'], 'checks': sorted(ck)}))
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs')); ap.add_argument('--out', default=str(REPO / 'runs' / 'ws-seq' / EXP_ID))
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int); ap.add_argument('--config', default=str(REPO / 'configs' / (EXP_ID + '.json')))
    ap.add_argument('--stop-rule-reference', type=float, default=None); ap.add_argument('--selfcheck', action='store_true'); ap.add_argument('--selfcheck-out')
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck(args)
    if args.arm is None or args.seed is None:
        ap.print_usage(sys.stderr); return EXIT_USAGE
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
