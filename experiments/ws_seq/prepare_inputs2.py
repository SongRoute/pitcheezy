"""EXP-WS-SEQ-002 input preparation on the Mac: builds `inputs2/`, the extension of the EXP-WS-SEQ-001 `inputs/` package.
Extension (copy, not import) of prepare_inputs.py.

  inputs2/seq-inputs2.npz    int16/int32 only, one row per accepted row in scale-002 corpus order:
                             game_pk, at_bat_number, pitch_number (row identity), zone (0 = missing, 1..14),
                             release_speed_bucket (0 = missing, 1..8), pfx_x_bucket, pfx_z_bucket (0 = missing, 1..5)
  inputs2/bucket-edges.json  quantile edges fitted on FIT rows only (+ the rule)
  inputs2/manifest.json      sha256 of the two files above, of the three SEQ-001 base files they extend
                             (x-prequential.npy, corpus-subset.npz, seq-inputs.npz: referenced by sha256, NOT copied),
                             of train_seq2.py + the grid config, and the registration / review sha256

The server reads the base files from the SEQ-001 inputs directory (data/ws-seq/inputs) and the new files from
data/ws-seq/inputs2. No parquet and no raw physics value goes to the server (bucket ids only). The batter-side context
needs no new array: train_seq2.py builds it from seq-inputs.npz (batter, game_pk, at_bat_number, pitch_number).

New limited column read (first time in this track): ZONE_COLUMNS = key + zone, straight from the S2 parquet with
pyarrow, joined by key to the projected frame. plate_x / plate_z / sz_top / sz_bot and every other column outside the
29-column projection stay unread. zone and the physics buckets of a row are used by the model ONLY as token fields
of LATER pitches (see train_seq2.py); the current pitch never sees its own.

Runner python: /Users/song/Projects/pitcheezy/.venv-observer/bin/python -I

Modes:
  --plan       sha256 of every registered input and gate state; decodes no array. Writes --plan-out.
  --selfcheck  synthetic rows only (scale_runner.synthetic_frame + a synthetic zone column) under --work; never opens
               the real parquet or the real lock.
  --execute    BLOCKED unless the registration is READY_FOR_REVIEW, free of PENDING_ placeholders, and REVIEW.json
               (verdict PASS, distinct reviewer, attempt id, matching source sha256) exists. Chain: gates -> memory gate
               -> lock -> read_source -> project -> row-for-row identity with corpus.npz and with the SEQ-001 base
               package -> key+zone read -> FIT-only bucket edges -> package -> manifest. Lock released in finally.
"""
import os, sys
sys.dont_write_bytecode = True  # importing the pinned helpers must not write __pycache__ into their directories
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'  # before numpy
import argparse, bisect, copy, fcntl, gc, hashlib, importlib.util, json, re, resource, signal, subprocess, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP_ID = 'EXP-WS-SEQ-002'
READY = 'READY_FOR_REVIEW'
PENDING = 'PENDING_'
CONFIG_JSON = REPO / 'configs' / (EXP_ID + '.json')
CONFIG_YAML = REPO / 'configs' / (EXP_ID + '.yaml')
SOURCES = {'experiments/ws_seq/prepare_inputs2.py': HERE / 'prepare_inputs2.py', 'experiments/ws_seq/train_seq2.py': HERE / 'train_seq2.py',
           'experiments/ws_seq/select_config.py': HERE / 'select_config.py', 'configs/EXP-WS-SEQ-002.json': CONFIG_JSON, 'configs/EXP-WS-SEQ-002.yaml': CONFIG_YAML}
ROLE_KEY = {'FIT': 'FIT', 'SELECT': 'SELECT', 'CAL': 'CAL', 'EVAL': 'TEST'}  # scale_runner.project names the EVAL mask TEST
CORPUS_ROLE = {'FIT': 'fit', 'SELECT': 'tuning', 'CAL': 'calibration', 'TEST': 'evaluation'}
CORPUS_MEMBERS = ('keys', 'y10', 'y7', 'games', 'fit', 'tuning', 'calibration', 'evaluation', 'narrow_whiff', 'ground_ball_per_pitch')
SEQ_DTYPES = {'game_pk': np.int32, 'at_bat_number': np.int16, 'pitch_number': np.int16, 'pitcher': np.int32, 'batter': np.int32, 'pitch_type': np.int16,
              'outcome10': np.int16, 'balls': np.int16, 'strikes': np.int16, 'stand': np.int16}
STAND = {'L': 0, 'R': 1}
KEY = ('game_pk', 'at_bat_number', 'pitch_number')
ZONE_COLUMNS = KEY + ('zone',)                      # the ONLY columns of the new limited parquet read
ZONE_RANGE = (1, 14)                                # Statcast zone; token 0 = missing
PHYSICS = ('release_speed', 'pfx_x', 'pfx_z')       # already in the 29-column projection; bucketed here, never shipped raw
TOKENS2_DTYPES = {'game_pk': np.int32, 'at_bat_number': np.int16, 'pitch_number': np.int16, 'zone': np.int16,
                  'release_speed_bucket': np.int16, 'pfx_x_bucket': np.int16, 'pfx_z_bucket': np.int16}
BASE_INPUTS = {'x-prequential.npy': 'base_x_prequential', 'corpus-subset.npz': 'base_corpus_subset', 'seq-inputs.npz': 'base_seq_inputs'}
CODE_INPUTS = ('scale_registration', 'scale_runner', 'scale_safe')
WATCHDOG_EXIT = 71


class Stop(Exception):
    """A registered STOP: no package (manifest.json is never written)."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False, default=str, ensure_ascii=False) + '\n')


def own_hashes(reg_path):
    return {'REGISTRATION.json': sha(reg_path), **{k: sha(p) for k, p in SOURCES.items()}}


def pending_fields(obj, at=''):
    """Every JSON path whose value is a string starting with the placeholder prefix PENDING_."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in pending_fields(v, f'{at}.{k}' if at else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in pending_fields(v, f'{at}[{i}]')]
    return [at] if isinstance(obj, str) and obj.startswith(PENDING) else []


def load_registration(path):
    reg = json.loads(Path(path).read_text())
    ok = [reg['id'] == EXP_ID, reg['bootstrap']['unit'] == 'game_pk', reg['feature_dimension'] == 199, reg['config']['authoritative'] == 'configs/EXP-WS-SEQ-002.json',
          set(CODE_INPUTS) <= set(reg['inputs']), set(BASE_INPUTS.values()) <= set(reg['inputs']), 'corpus' in reg['inputs'], isinstance(reg['source_rows'], int),
          reg['token']['zone_read_columns'] == list(ZONE_COLUMNS), reg['token']['zone_range'] == list(ZONE_RANGE), list(reg['token']['buckets']) == list(PHYSICS)]
    if not all(ok):
        raise SystemExit('BLOCKED: REGISTRATION.json does not match this source: ' + str(ok))
    return reg


def review_gate(reg, reg_path):
    """--execute precondition (same schema as EXP-WS-SEQ-001). Returns the sha256 of the accepted REVIEW.json."""
    if reg.get('status') != READY:
        raise SystemExit('BLOCKED: registration status is not exactly ' + READY)
    pend = pending_fields(reg)
    if pend:
        raise SystemExit('BLOCKED: registration still carries a ' + PENDING + ' placeholder at: ' + ', '.join(pend))
    p = Path(reg['review']['path'])
    if not p.exists():
        raise SystemExit('BLOCKED: independent REVIEW.json (verdict PASS, matching source sha256) required at ' + str(p))
    r = json.loads(p.read_text())
    if r.get('verdict') != 'PASS' or r.get('distinct_from_author') is not True or r.get('attempt_id') != reg['attempt_id']:
        raise SystemExit('BLOCKED: review verdict / distinct reviewer / attempt id')
    if r.get('sources_sha256') != own_hashes(reg_path):
        raise SystemExit('BLOCKED: reviewed source sha256 differ from the current REGISTRATION.json / scripts / configs')
    return sha(p)


def verify_inputs(reg, names=None, collect=False, section='inputs'):
    """sha256 of the registered inputs. Hash only: nothing is decoded. collect=True returns (observed, bad) instead of blocking."""
    observed = {}; bad = []
    for name, spec in reg[section].items():
        if names is not None and name not in names:
            continue
        if str(spec['sha256']).startswith(PENDING):
            observed[name] = None; bad.append(name + ' (' + spec['sha256'] + ')'); continue
        p = Path(spec['path'])
        observed[name] = sha(p) if p.is_file() else None
        if observed[name] != spec['sha256']:
            bad.append(name)
    if collect:
        return observed, bad
    if bad:
        raise SystemExit('BLOCKED: registered input missing, pending or sha256 mismatch: ' + ', '.join(bad))
    return observed


def load_scale(reg):
    """Import scale_runner by file path (sys.path untouched), validate its registration and build the pinned closure M
    (as prepare_inputs.load_scale)."""
    verify_inputs(reg, CODE_INPUTS)
    spec = importlib.util.spec_from_file_location('scale_runner', reg['inputs']['scale_runner']['path'])
    SR = importlib.util.module_from_spec(spec); sys.modules['scale_runner'] = SR; spec.loader.exec_module(SR)
    c = SR.load(reg['inputs']['scale_registration']['path'])
    M = SR.closure(c)  # sha-checks the pinned helpers, installs the deny finder (torch/lightgbm cannot be imported afterwards)
    I = reg['inputs']
    same = [c['source']['parquet'] == I['s2_parquet']['path'], c['source']['parquet_sha256'] == I['s2_parquet']['sha256'],
            c['source']['manifest'] == I['s2_manifest']['path'], c['source']['manifest_sha256'] == I['s2_manifest']['sha256'],
            all(c['roles'][k] == reg['roles'][k] for k in ROLE_KEY),
            all(I['pinned_' + k]['path'] == p and I['pinned_' + k]['sha256'] == h for k, (p, h, _) in c['pinned'].items()),
            c['lock']['path'] == reg['lock']['path'], c['features']['dimension'] == reg['feature_dimension'],
            list(M.fsf.TYPES) == reg['token']['pitch_type_order'] and len(M.fsf.TYPES) == 19, c['source']['rows'] == reg['source_rows'],
            tuple(M.cp.KEY) == KEY, set(PHYSICS) <= set(M.cp.PROJECT), 'zone' not in M.cp.PROJECT, 'zone' in c['forbidden_feature_columns']]
    if not all(same):
        raise SystemExit('BLOCKED: scale registration differs from the SEQ-002 registration (source/roles/pinned/lock/types/rows/columns): ' + str(same))
    return SR, c, M


def load_train_module():
    """train_seq2.py by file path (numpy only at import time; torch is imported lazily there and never here)."""
    spec = importlib.util.spec_from_file_location('ws_seq_train2', HERE / 'train_seq2.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------- resources (copied from prepare_inputs.py)
def free_memory_bytes(text=None):
    """(free + inactive + speculative pages) x page size from `LC_ALL=C vm_stat`."""
    if text is None:
        text = subprocess.run(['/usr/bin/vm_stat'], capture_output=True, text=True, timeout=20, env=dict(os.environ, LC_ALL='C'), check=True).stdout
    page = int(re.search(r'page size of (\d+) bytes', text).group(1))
    return page * sum(int(re.search(rf'Pages {k}:\s+(\d+)\.', text).group(1)) for k in ('free', 'inactive', 'speculative'))


def memory_gate(minimum):
    free = free_memory_bytes()
    if free < minimum:
        raise Stop(f'STOP memory gate: free+inactive+speculative {free} bytes < required {minimum}')
    return free


def acquire_lock(path, retry_seconds, max_wait_seconds):
    """flock(LOCK_EX|LOCK_NB) on the EXISTING lock file opened read-only. Busy -> retry, then STOP. Never creates/deletes."""
    p = Path(path)
    if not p.is_file():
        raise Stop('STOP shared lock file missing (this runner never creates it): ' + str(p))
    f = p.open('r'); start = time.monotonic(); tries = 0
    while True:
        tries += 1
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f, {'waited_seconds': time.monotonic() - start, 'tries': tries, 'inode': os.fstat(f.fileno()).st_ino}
        except BlockingIOError:
            if time.monotonic() - start + retry_seconds > max_wait_seconds:
                f.close()
                raise Stop(f'STOP shared lock busy for {max_wait_seconds} s ({tries} tries); not contended')
            time.sleep(retry_seconds)


def peak_rss_bytes():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == 'darwin' else r * 1024)


class Watchdog(threading.Thread):
    """Own-process peak RSS and wall cap (this script has no child process). Over a cap -> exit 71; the kernel drops the flock;
    manifest.json does not exist, so the partial package is unusable."""

    def __init__(self, max_rss, deadline, every):
        super().__init__(daemon=True); self.max_rss, self.deadline, self.every = max_rss, deadline, every; self.halt = threading.Event()

    def run(self):
        while not self.halt.wait(self.every):
            rss = peak_rss_bytes()
            if rss > self.max_rss or time.monotonic() > self.deadline:
                print(f'WATCHDOG STOP: peak RSS {rss} (cap {self.max_rss}) or wall cap exceeded', file=sys.stderr, flush=True); os._exit(WATCHDOG_EXIT)


# ---------------------------------------------------------------- population identity (copied from prepare_inputs.py)
def check_population(g, rm, y7, corpus, expected, M):
    """The rebuilt accepted frame must be the scale-002 corpus row for row; registered denominators must match."""
    ev = rm['TEST']
    d = {'accepted_rows': int(len(g)), 'role_rows': {r: int(rm[k].sum()) for r, k in ROLE_KEY.items()},
         'eval_games': int(g.game_pk[ev].nunique()), 'eval_leaf_counts': np.bincount(y7[ev], minlength=7).tolist()}
    if expected is not None:
        bad = {k: {'expected': expected[k], 'observed': d[k]} for k in d if d[k] != expected[k]}
        if bad:
            raise Stop('STOP denominator mismatch: ' + json.dumps(bad))
    if len(corpus['keys']) != len(g) or not np.array_equal(g[M.cp.KEY].to_numpy('<i8'), corpus['keys']):
        raise Stop('STOP rebuilt accepted keys differ from scale-002 corpus.npz keys (row for row)')
    for k, name in CORPUS_ROLE.items():
        if not np.array_equal(rm[k], corpus[name]):
            raise Stop('STOP role mask differs from corpus.npz: ' + name)
    if not (np.array_equal(y7, corpus['y7']) and np.array_equal(g.outcome10.to_numpy(np.int64), corpus['y10']) and np.array_equal(g.game_pk.to_numpy(), corpus['games'])
            and np.array_equal(g.narrow_whiff.to_numpy(np.int8), corpus['narrow_whiff']) and np.array_equal(g.ground_ball_per_pitch.to_numpy(np.int8), corpus['ground_ball_per_pitch'])):
        raise Stop('STOP labels, head labels or games differ from corpus.npz')
    if not np.all(sum(v.astype(int) for v in rm.values()) == 1):
        raise Stop('STOP accepted row without exactly one role')
    return d


def sequence_arrays(g, M):
    """Integer-only per-row token sources of EXP-WS-SEQ-001 (copy of prepare_inputs.sequence_arrays); rebuilt here only
    to prove that the SEQ-001 base package rows are these rows."""
    lookup = {t: i for i, t in enumerate(M.fsf.TYPES)}
    pt = g.pitch_type.map(lookup); st = g.stand.map(STAND)
    if pt.isna().any() or st.isna().any():
        raise Stop('STOP pitch_type outside TYPES or stand outside L/R in the accepted frame')
    src = {'game_pk': g.game_pk, 'at_bat_number': g.at_bat_number, 'pitch_number': g.pitch_number, 'pitcher': g.pitcher, 'batter': g.batter, 'pitch_type': pt,
           'outcome10': g.outcome10, 'balls': g.balls, 'strikes': g.strikes, 'stand': st}
    out = {}
    for k, dt in SEQ_DTYPES.items():
        v = src[k].to_numpy(); w = v.astype(dt)
        if not np.array_equal(w.astype(np.float64), v.astype(np.float64)):
            raise Stop(f'STOP {k} does not fit {np.dtype(dt).name} exactly')
        out[k] = w
    return out


# ---------------------------------------------------------------- new token sources
def read_zone(parquet_path, keys, source_rows):
    """The new limited read: ZONE_COLUMNS only. One row per key (STOP on a duplicate), registered row count BEFORE any
    filtering, every accepted key present. Returns int16 zone tokens in accepted-row order (0 = missing, 1..14) + a report.
    Called inside the lock right after abc_corpus.read_source bound the parquet sha256."""
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    absent = sorted(set(ZONE_COLUMNS) - set(pf.schema_arrow.names))
    if absent:
        raise Stop('STOP zone read: column absent: ' + str(absent))
    t = pf.read(columns=list(ZONE_COLUMNS))
    if t.column_names != list(ZONE_COLUMNS) or t.num_rows != source_rows:
        raise Stop(f'STOP zone read: columns {t.column_names} / rows {t.num_rows} (registered {source_rows} before filtering)')
    zt = t.schema.field('zone').type
    if not (pa.types.is_integer(zt) or pa.types.is_floating(zt)):
        raise Stop('STOP zone read: zone is not a numeric column: ' + str(zt))
    df = t.to_pandas(); del t
    null_key = df[list(KEY)].isna().any(axis=1).to_numpy(); d = df.loc[~null_key]
    kf = d[list(KEY)].to_numpy(np.float64)
    if not np.array_equal(kf, np.round(kf)):
        raise Stop('STOP zone read: non-integer key')
    d = pd.DataFrame({k: kf[:, j].astype(np.int64) for j, k in enumerate(KEY)}).assign(zone=d['zone'].to_numpy(dtype=np.float64, na_value=np.nan))
    if d.duplicated(list(KEY)).any():
        raise Stop('STOP zone read: more than one row for a (game_pk, at_bat_number, pitch_number) key')
    left = pd.DataFrame(np.asarray(keys, np.int64), columns=list(KEY))
    m = left.merge(d, on=list(KEY), how='left', validate='one_to_one', indicator=True)
    if len(m) != len(left) or (m['_merge'] != 'both').any() or not np.array_equal(m[list(KEY)].to_numpy(np.int64), np.asarray(keys, np.int64)):
        raise Stop('STOP zone read: an accepted row has no key match in the zone table (or the join changed the row order)')
    z = m['zone'].to_numpy(np.float64); miss = np.isnan(z); val = z[~miss]
    if (~np.isfinite(val)).any() or (val != np.round(val)).any() or (val < ZONE_RANGE[0]).any() or (val > ZONE_RANGE[1]).any():
        raise Stop(f'STOP zone read: a non-missing zone is not an integer in {ZONE_RANGE[0]}..{ZONE_RANGE[1]}')
    zone = np.zeros(len(z), np.int16); zone[~miss] = val.astype(np.int16)
    return zone, {'columns_read': list(ZONE_COLUMNS), 'source_rows': int(source_rows), 'null_key_rows_dropped': int(null_key.sum()), 'accepted_rows_joined': int(len(zone)),
                  'accepted_rows_missing_zone': int(miss.sum()), 'arrow_type': str(zt)}


def fit_bucket_edges(values, fit, bins):
    """bins-1 interior quantile edges (np.quantile, method 'linear', probabilities k/bins) of the finite values of FIT
    rows ONLY. Rows outside `fit` never enter."""
    v = np.asarray(values, np.float64); src = v[np.asarray(fit, bool) & np.isfinite(v)]
    if len(src) < bins:
        raise Stop('STOP bucket edges: fewer finite FIT values than bins')
    edges = np.quantile(src, np.arange(1, bins) / bins, method='linear')
    if not np.isfinite(edges).all() or (np.diff(edges) < 0).any():
        raise Stop('STOP bucket edges: non-finite or decreasing')
    return edges, int(len(src))


def bucketize(values, edges):
    """0 = missing (NaN / non-finite); otherwise 1 + number of edges <= value (1 .. len(edges)+1)."""
    v = np.asarray(values, np.float64); ok = np.isfinite(v); out = np.zeros(len(v), np.int16)
    out[ok] = 1 + np.searchsorted(edges, v[ok], side='right')
    return out


def new_tokens(g, zone, fit, bins):
    """seq-inputs2 members + the bucket-edges document. Edges come from FIT rows; every row is then bucketed with them."""
    tok = {'game_pk': g.game_pk.to_numpy().astype(np.int32), 'at_bat_number': g.at_bat_number.to_numpy().astype(np.int16), 'pitch_number': g.pitch_number.to_numpy().astype(np.int16), 'zone': zone}
    edges = {}; used = {}
    for f in PHYSICS:
        v = g[f].to_numpy(dtype=np.float64, na_value=np.nan)
        edges[f], used[f] = fit_bucket_edges(v, fit, bins[f]); tok[f + '_bucket'] = bucketize(v, edges[f])
    doc = {'id': EXP_ID, 'fit_role': 'fit', 'bins': {f: int(bins[f]) for f in PHYSICS}, 'quantile_method': 'linear',
           'rule': 'edges = np.quantile(finite values of FIT rows, k/bins for k = 1..bins-1); bucket = 0 if missing else 1 + number of edges <= value',
           'edges': {f: [float(e) for e in edges[f]] for f in PHYSICS}, 'edges_hex': {f: [float(e).hex() for e in edges[f]] for f in PHYSICS},
           'fit_rows_used': used, 'zone_categories': '0 = missing, 1..14 = Statcast zone of that (earlier) pitch',
           'fit_rows_per_category': {k: np.bincount(tok[k][np.asarray(fit, bool)], minlength=1).tolist() for k in ('zone', 'release_speed_bucket', 'pfx_x_bucket', 'pfx_z_bucket')}}
    return tok, doc


def npy_header(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False); return tuple(a.shape), a.dtype


def run_chain(reg, reg_path, SR, c, M, out, config_path, review_sha, inputs_sha):
    """memory gate -> lock -> watchdog -> projection -> identity (corpus + SEQ-001 base package) -> zone read -> FIT-only
    buckets -> package -> manifest (written last)."""
    import pandas, pyarrow
    caps = reg['caps']['prepare']; out = Path(out); I = reg['inputs']; T = load_train_module(); cfg = T.load_config(config_path)
    if out.exists():
        raise SystemExit('BLOCKED: output directory exists (a rerun needs a fresh directory and a recorded reason): ' + str(out))
    start = time.monotonic(); lockf = None; wd = None

    def on_term(signum, _frame):
        raise Stop('STOP signal ' + signal.Signals(signum).name)
    previous = signal.signal(signal.SIGTERM, on_term)
    try:
        free_before = memory_gate(caps['min_free_memory_bytes'])
        lockf, lock_info = acquire_lock(reg['lock']['path'], reg['lock']['retry_seconds'], reg['lock']['max_wait_seconds'])
        free_locked = memory_gate(caps['min_free_memory_bytes']); work = time.monotonic()
        wd = Watchdog(caps['max_rss_bytes'], work + caps['wall_seconds'], caps['rss_sample_seconds']); wd.start()
        raw, binding = M.cp.read_source(c)  # binds manifest + parquet sha256 before decoding the 29 columns
        g, census, rm, sample, info, y7 = SR.project(c, raw, M)
        del raw, sample; gc.collect()
        with np.load(I['corpus']['path'], allow_pickle=False) as z:
            corpus = {k: np.array(z[k]) for k in CORPUS_MEMBERS}
        den = check_population(g, rm, y7, corpus, reg['expected_denominators'], M)
        if census['source_rows'] != reg['source_rows']:
            raise Stop('STOP source rows differ from the registration')
        S = sequence_arrays(g, M)
        with np.load(I['base_seq_inputs']['path'], allow_pickle=False) as z:   # the SEQ-001 package must be these rows, token for token
            if sorted(z.files) != sorted(SEQ_DTYPES) or not all(z[k].dtype == S[k].dtype and np.array_equal(z[k], S[k]) for k in SEQ_DTYPES):
                raise Stop('STOP the SEQ-001 base seq-inputs.npz differs from the rebuilt token sources (row for row)')
        with np.load(I['base_corpus_subset']['path'], allow_pickle=False) as z:
            if sorted(z.files) != sorted(CORPUS_MEMBERS) or not all(np.array_equal(z[k], corpus[k]) for k in CORPUS_MEMBERS):
                raise Stop('STOP the SEQ-001 base corpus-subset.npz differs from scale-002 corpus.npz')
        shape, dtype = npy_header(I['base_x_prequential']['path'])
        if shape != (den['accepted_rows'], reg['feature_dimension']) or dtype != np.float32:
            raise Stop(f'STOP base x-prequential shape/dtype {shape} {dtype}')
        zone, zone_report = read_zone(c['source']['parquet'], corpus['keys'], reg['source_rows'])
        tok, edges_doc = new_tokens(g, zone, corpus['fit'], cfg['buckets'])
        if not all(tok[k].dtype == np.dtype(dt) for k, dt in TOKENS2_DTYPES.items()) or sorted(tok) != sorted(TOKENS2_DTYPES) \
                or not all(np.array_equal(tok[k], S[k]) for k in KEY) or any(int(tok[f + '_bucket'].max()) > cfg['buckets'][f] for f in PHYSICS):
            raise Stop('STOP new token arrays: members / dtypes / key identity / bucket range')
        zone_report['accepted_rows_missing_zone_by_role'] = {r: int((zone[rm[k]] == 0).sum()) for r, k in ROLE_KEY.items()}
        missing_physics = {f: {r: int((tok[f + '_bucket'][rm[k]] == 0).sum()) for r, k in ROLE_KEY.items()} for f in PHYSICS}
        del g; gc.collect()
        out.mkdir(parents=True)
        np.savez(out / 'seq-inputs2.npz', **tok); save_json(out / 'bucket-edges.json', edges_doc)
        files = {'seq-inputs2.npz': {'sha256': sha(out / 'seq-inputs2.npz'), 'bytes': (out / 'seq-inputs2.npz').stat().st_size, 'members': {k: [list(v.shape), str(v.dtype)] for k, v in tok.items()}},
                 'bucket-edges.json': {'sha256': sha(out / 'bucket-edges.json'), 'bytes': (out / 'bucket-edges.json').stat().st_size}}
        base_files = {name: {'sha256': inputs_sha[key], 'bytes': Path(I[key]['path']).stat().st_size, 'registered_path': I[key]['path']} for name, key in BASE_INPUTS.items()}
        if any(v['sha256'] != I[key]['sha256'] for (name, key), v in zip(BASE_INPUTS.items(), base_files.values())):
            raise Stop('STOP base file sha256 differs from the registration')
        manifest = {'id': EXP_ID, 'attempt_id': reg['attempt_id'], 'data_role': reg['data_role'], 'synthetic': 'synthetic_selfcheck' in reg, 'utc': time.time(), 'files': files,
                    'base_files': base_files, 'base_note': 'the three SEQ-001 files are read by train_seq2.py from --base-inputs (default data/ws-seq/inputs) and verified against these sha256; they are not copied',
                    'bound_sources': {'train_seq2.py': sha(HERE / 'train_seq2.py'), 'config': sha(config_path), 'config_path': str(config_path), 'select_config.py': sha(HERE / 'select_config.py')},
                    'grid_config_hashes': {e['id']: T.config_hash(cfg, e) for e in cfg['grid']},
                    'registration_sha256': sha(reg_path), 'review_sha256': review_sha, 'prepare_inputs2_sha256': sha(HERE / 'prepare_inputs2.py'),
                    'source_sha256': {k: inputs_sha.get(k) for k in ('s2_parquet', 's2_manifest', 'corpus', 'scale_registration', 'scale_runner', 'scale_safe')},
                    'source_binding': binding, 'denominators': den, 'source_rows': census['source_rows'], 'excluded': census['excluded'],
                    'zone_read': zone_report, 'missing_physics_by_role': missing_physics,
                    'token_sources': {'row_order': 'scale-002 corpus order = sort by (game_pk, at_bat_number, pitch_number)',
                                      'note': 'integer arrays only; zone / bucket ids of a row are used by the model ONLY as token fields of LATER rows of the same game; raw physics values are not shipped'},
                    'resources': {'free_memory_bytes_before_lock': free_before, 'free_memory_bytes_after_lock': free_locked, 'lock': lock_info, 'peak_rss_bytes': peak_rss_bytes(), 'elapsed_seconds': time.monotonic() - start},
                    'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pandas.__version__, 'pyarrow': pyarrow.__version__}}
        if peak_rss_bytes() > caps['max_rss_bytes'] or time.monotonic() - work > caps['wall_seconds']:
            raise Stop('STOP RSS or wall cap exceeded')
        save_json(out / 'manifest.json', manifest)
        return manifest, tok, corpus, S
    finally:
        if wd is not None:
            wd.halt.set()
        if lockf is not None:
            lockf.close()
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------- plan / execute
def plan(reg, reg_path, config_path, out):
    pend = pending_fields(reg); observed, bad = verify_inputs(reg, collect=True); cal_obs, cal_bad = verify_inputs(reg, collect=True, section='calibration_inputs')
    closure = None
    try:
        SR, c, M = load_scale(reg); closure = sorted(vars(M))
    except SystemExit as e:
        closure = str(e)
    try:
        review_gate(reg, reg_path); review_state = 'PASS bound to the current sources'
    except SystemExit as e:
        review_state = str(e)
    twins = None
    try:
        import yaml
        twins = yaml.safe_load(CONFIG_YAML.read_text()) == json.loads(CONFIG_JSON.read_text())
    except ImportError:
        pass
    T = load_train_module(); cfg = T.load_config(config_path); hashes = {e['id']: T.config_hash(cfg, e) for e in cfg['grid']}
    base_manifest = None
    try:   # metadata only: the SEQ-001 manifest must list the three base files with the registered sha256
        bm = json.loads(Path(reg['inputs']['base_manifest']['path']).read_text())
        base_manifest = all(bm['files'][name]['sha256'] == reg['inputs'][key]['sha256'] for name, key in BASE_INPUTS.items())
    except Exception as e:
        base_manifest = repr(e)
    cal = reg['calibration_stage']['script']; cal_path = Path(cal['path'])
    checks = {'runner_python_exists': Path(reg['runtime']['runner_python']).exists(), 'lock_file_exists': Path(reg['lock']['path']).is_file(), 'output_dir_absent': not Path(out).exists(),
              'status_ready_for_review': reg.get('status') == READY, 'no_pending_placeholder': not pend, 'inputs_all_verified': not bad, 'calibration_inputs_all_verified': not cal_bad,
              'pinned_closure_loaded': isinstance(closure, list), 'config_is_repo_authoritative': Path(config_path).resolve() == CONFIG_JSON.resolve(), 'config_twins_identical': twins is True,
              'grid_config_hashes_equal_registration': hashes == reg['grid']['config_hashes'], 'grid_equals_registration': cfg['grid'] == reg['grid']['configs'],
              'seq001_manifest_lists_registered_base_files': base_manifest is True}
    ready = bool(all(checks.values()) and review_state.startswith('PASS'))
    return {'id': reg['id'], 'attempt_id': reg['attempt_id'], 'utc': time.time(), 'own_sources_sha256': own_hashes(reg_path), 'pending_placeholders': pend,
            'inputs_sha256': observed, 'inputs_not_verified': bad, 'calibration_inputs_sha256': cal_obs, 'calibration_inputs_not_verified': cal_bad,
            'grid_config_hashes': hashes, 'pinned_closure_loaded': closure, 'checks': checks, 'review': review_state, 'review_file_exists': Path(reg['review']['path']).exists(),
            'calibrator_script': {'path': str(cal_path), 'exists': cal_path.is_file(), 'sha256_now': sha(cal_path) if cal_path.is_file() else None,
                                  'note': 'informational; the calibration stage is gated by its own registration/review (EXP-WS-SEQ-CAL-001 conventions), not by this plan'},
            'execute_ready': ready, 'refused': None if ready else 'PLAN REFUSED: ' + ('placeholder ' + PENDING + '* present; ' if pend else '') + 'see checks / review',
            'arrays_decoded': 0, 'output_dir': str(out),
            'stages': ['Mac: prepare_inputs2.py --execute (projection + identity + key+zone read + FIT-only buckets + package)', 'lead: copy inputs2/ next to the SEQ-001 inputs/ on the server (data/ws-seq/inputs2)',
                       'server: 10 seed-0 jobs of train_seq2.py via gpu run (4 cards at a time)', 'Mac: gpu pull, select_config.py (SELECT only; refuses if any EVAL metric exists)',
                       'server: seeds 1,2 of the selected config', 'Mac: gpu pull, calibrate_logits.py (the single registered EVAL read)'],
            'command': f"{reg['runtime']['runner_python']} -I {HERE / 'prepare_inputs2.py'} --execute --registration {reg_path}"}


def execute(reg, reg_path, config_path, out):
    if 'synthetic_selfcheck' in reg:
        raise SystemExit('BLOCKED: synthetic key in the real registration')
    if Path(config_path).resolve() != CONFIG_JSON.resolve():
        raise SystemExit('BLOCKED: --execute binds only the authoritative repo config ' + str(CONFIG_JSON))
    review_sha = review_gate(reg, reg_path)
    if Path(sys.prefix).resolve() != Path(reg['runtime']['runner_python']).parent.parent.resolve():
        raise SystemExit('BLOCKED: run with the registered runner python')
    inputs = verify_inputs(reg); SR, c, M = load_scale(reg)
    T = load_train_module(); cfg = T.load_config(config_path)
    if {e['id']: T.config_hash(cfg, e) for e in cfg['grid']} != reg['grid']['config_hashes'] or cfg['grid'] != reg['grid']['configs']:
        raise SystemExit('BLOCKED: grid / config hashes differ from the registration')
    manifest, _, _, _ = run_chain(reg, reg_path, SR, c, M, out, config_path, review_sha, inputs)
    return {'verdict': 'PACKAGE_READY', 'out': str(out), 'manifest_sha256': sha(Path(out) / 'manifest.json'), 'files': {k: v['sha256'] for k, v in manifest['files'].items()},
            'base_files': {k: v['sha256'] for k, v in manifest['base_files'].items()}, 'denominators': manifest['denominators'], 'zone_read': manifest['zone_read'], 'resources': manifest['resources']}


# ---------------------------------------------------------------- selfcheck (synthetic only)
def expect(fn, exc, text):
    try:
        fn()
    except exc as e:
        if text not in str(e):
            raise AssertionError(f'wrong message: {e!r} (wanted {text!r})')
        return True
    raise AssertionError('not raised: ' + text)


def grouped_brute_context(group, S, length):
    """Independent pandas/python loop: per (group id, game) set sorted by (at_bat_number, pitch_number), earlier rows only."""
    import pandas as pd
    f = pd.DataFrame({'grp': np.asarray(group, np.int64), **{k: S[k].astype(np.int64) for k in KEY}}); out = np.full((len(f), length), -1, np.int32)
    for _, grp in f.groupby(['grp', 'game_pk'], sort=False):
        rows = grp.sort_values(['at_bat_number', 'pitch_number']).index.to_list()
        for pos, i in enumerate(rows):
            prev = rows[max(0, pos - length):pos][::-1]
            out[i, :len(prev)] = prev
    return out


def selfcheck(reg, reg_path, config_path, work):
    """Builds a synthetic world under `work` (must be a directory named selfcheck-tmp) and runs the SAME run_chain as --execute."""
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    work = Path(work)
    if work.name != 'selfcheck-tmp' or (work / 'inputs2').exists():
        raise SystemExit('BLOCKED: --work must be a directory named selfcheck-tmp without an inputs2/ directory')
    work.mkdir(parents=True, exist_ok=True); (work / 'holdout').mkdir(exist_ok=True); out = {'id': EXP_ID, 'mode': 'prepare_inputs2 --selfcheck (synthetic)', 'checks': {}}; ck = out['checks']
    ck['real_registration_pending_placeholders'] = pending_fields(reg)
    expect(lambda: execute(dict(reg, synthetic_selfcheck=True), reg_path, config_path, work / 'x'), SystemExit, 'synthetic key'); ck['execute_refuses_synthetic_registration'] = True
    expect(lambda: review_gate(dict(reg, note='PENDING_X'), reg_path), SystemExit, 'placeholder'); ck['review_gate_refuses_pending_placeholder'] = True
    SR, c, M = load_scale(reg); T = load_train_module(); cfg = T.load_config(config_path)
    # ---- synthetic world: scale_runner.synthetic_frame (29 columns) + a synthetic zone column (nullable integer, like the S2 parquet)
    raw, _ = SR.synthetic_frame(np.random.default_rng(20261011)); date = pd.to_datetime(raw.game_date); zr = np.random.default_rng(20261012)
    raw.loc[(raw.pitcher == 101) & (date >= '2026-08-10'), 'pitcher'] = 901; raw.loc[(raw.pitcher == 103) & (date >= '2026-09-07'), 'pitcher'] = 903
    raw.loc[(raw.batter == 201) & (date >= '2026-08-10'), 'batter'] = 991
    zone_raw = pd.array(np.where(zr.random(len(raw)) < .04, -1, zr.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14], len(raw))), dtype='Int64'); zone_raw[zone_raw == -1] = pd.NA
    raw['zone'] = zone_raw; cols = list(M.cp.PROJECT) + ['zone']

    def write_source(frame, name):
        pp = work / 'holdout' / name; pq.write_table(pa.Table.from_pandas(frame[cols], preserve_index=False), pp)
        bind = {'version': 'synthetic', 'season': 2026, 'file': 'holdout/' + name, 'start': '2026-03-25', 'end': '2026-09-27', 'rows': len(frame), 'columns': len(cols), 'game_type': 'R', 'parquet_sha256': sha(pp), 'fetch_commit': 'synthetic'}
        mp = work / ('manifest-' + name + '.json')
        save_json(mp, {'version': 'synthetic', 'fetch_commit': 'synthetic', 'parameters': {'game_type': 'R', 'season_dates': {'2026': [bind['start'], bind['end']]}},
                       'seasons': {'2026': {'season': 2026, 'file': bind['file'], 'start': bind['start'], 'end': bind['end'], 'rows': len(frame), 'rows_raw': len(frame), 'columns': len(cols), 'duplicate_keys': 0, 'sha256': bind['parquet_sha256'], 'frozen': True}}})
        cs = copy.deepcopy(c); cs['fit_sample']['min_games_per_team'] = 2
        cs['source'] = {'id': 'SYNTHETIC', 'parquet': str(pp), 'parquet_sha256': bind['parquet_sha256'], 'manifest': str(mp), 'manifest_sha256': sha(mp), 'manifest_binding': bind, 'rows': len(frame)}
        return cs, pp, mp

    cs, pp, mp = write_source(raw, 'synthetic.parquet')
    src, _ = M.cp.read_source(cs); g, census, rm, sample, info, y7 = SR.project(cs, src, M); n = len(g)
    assert 'zone' not in g.columns and 'zone' not in src.columns   # the pinned 29-column projection never carries zone
    y10 = g.outcome10.to_numpy(np.int64); rng = np.random.default_rng(7)
    x = (rng.normal(size=(n, reg['feature_dimension'])) * .1).astype(np.float32)   # synthetic stand-in for E1b's x-prequential: pre-pitch columns only
    x[:, 0] += g.balls.to_numpy() * .5; x[:, 1] += g.strikes.to_numpy() * .5
    for j, t in enumerate(('FF', 'SL', 'CH', 'CU', 'UNKNOWN')):
        x[:, 2 + j] += (g.pitch_type == t).to_numpy() * 2.0
    base = work / 'inputs'; base.mkdir()                                             # synthetic SEQ-001 base package
    corpus = dict(keys=g[M.cp.KEY].to_numpy('<i8'), y7=y7, y10=y10, games=g.game_pk.to_numpy(), narrow_whiff=g.narrow_whiff.to_numpy(np.int8),
                  ground_ball_per_pitch=g.ground_ball_per_pitch.to_numpy(np.int8), **{CORPUS_ROLE[k]: rm[k] for k in CORPUS_ROLE})
    np.save(base / 'x-prequential.npy', x, allow_pickle=False); np.savez(base / 'corpus-subset.npz', **{k: corpus[k] for k in CORPUS_MEMBERS}); np.savez(work / 'corpus.npz', sample=sample, **corpus)
    S0 = sequence_arrays(g, M); np.savez(base / 'seq-inputs.npz', **S0)
    save_json(base / 'manifest.json', {'id': 'EXP-WS-SEQ-001', 'synthetic': True, 'files': {k: {'sha256': sha(base / k)} for k in BASE_INPUTS}})
    lock = work / 'fixture.lock'; lock.write_text('selfcheck fixture lock\n')
    rs = copy.deepcopy(reg); rs['synthetic_selfcheck'] = True; rs['source_rows'] = int(len(raw))
    rs['expected_denominators'] = {'accepted_rows': int(n), 'role_rows': {r: int(rm[k].sum()) for r, k in ROLE_KEY.items()}, 'eval_games': int(g.game_pk[rm['TEST']].nunique()), 'eval_leaf_counts': np.bincount(y7[rm['TEST']], minlength=7).tolist()}
    rs['lock'].update(path=str(lock), retry_seconds=.05, max_wait_seconds=.2); rs['caps']['prepare'].update(min_free_memory_bytes=1 << 20, rss_sample_seconds=1)
    rs['review']['path'] = str(work / 'REVIEW.json')
    for k, p in (('corpus', work / 'corpus.npz'), ('s2_parquet', pp), ('s2_manifest', mp), ('base_manifest', base / 'manifest.json'), *((key, base / name) for name, key in BASE_INPUTS.items())):
        rs['inputs'][k] = dict(rs['inputs'][k], path=str(p), sha256=sha(p))
    for k in rs['calibration_inputs']:
        rs['calibration_inputs'][k] = {'path': str(work / 'missing'), 'sha256': 'synthetic: not used by the selfcheck'}
    rs['selection']['stop_rule_reference_select_nll'] = 99.0
    rs['selection']['eval_result_paths_that_must_not_exist'] = [str(work / 'cal'), str(work / 'attempts' / rs['attempt_id'] / 'result.json')]
    rsp = work / 'REGISTRATION.synthetic.json'; save_json(rsp, rs)
    assert not pending_fields(rs)
    # review gate negatives, then a valid synthetic review
    expect(lambda: review_gate(rs, rsp), SystemExit, 'REVIEW.json'); good = {'verdict': 'PASS', 'distinct_from_author': True, 'attempt_id': rs['attempt_id'], 'sources_sha256': own_hashes(rsp)}
    for name, bad in (('verdict', dict(good, verdict='FAIL')), ('not_distinct', dict(good, distinct_from_author=False)), ('attempt', dict(good, attempt_id='x')), ('sha', dict(good, sources_sha256=dict(good['sources_sha256'], **{'REGISTRATION.json': '0'})))):
        save_json(rs['review']['path'], bad); expect(lambda: review_gate(rs, rsp), SystemExit, 'BLOCKED')
    expect(lambda: review_gate(dict(rs, status='DRAFT'), rsp), SystemExit, 'status')
    save_json(rs['review']['path'], good); review_sha = review_gate(rs, rsp); ck['review_gate_negative_controls'] = ['missing', 'verdict', 'not_distinct', 'attempt', 'sha', 'status', 'pending']
    names = ('corpus', 's2_parquet', 's2_manifest', 'base_manifest') + tuple(BASE_INPUTS.values()) + CODE_INPUTS
    inputs = verify_inputs(rs, names)
    bad_in = copy.deepcopy(rs); bad_in['inputs']['base_x_prequential']['sha256'] = '0' * 64
    expect(lambda: verify_inputs(bad_in, ('base_x_prequential',)), SystemExit, 'base_x_prequential'); ck['input_sha_mismatch_blocks'] = True
    # ---- negative controls of the chain (each must STOP before a manifest exists)
    neg = {}

    def chain_stop(name, text, mutate=None, regm=None, hold_lock=False, mutate_base=None, source=None):
        r2 = copy.deepcopy(rs) if regm is None else regm; o = work / ('neg-' + name)
        if mutate is not None:
            with np.load(work / 'corpus.npz') as z:
                cc = {k: np.array(z[k]) for k in z.files}
            mutate(cc); np.savez(work / f'corpus-{name}.npz', **cc); r2['inputs']['corpus']['path'] = str(work / f'corpus-{name}.npz')
        if mutate_base is not None:
            fname, fn = mutate_base
            with np.load(base / fname) as z:
                cc = {k: np.array(z[k]) for k in z.files}
            fn(cc); np.savez(work / f'base-{name}.npz', **cc); r2['inputs'][BASE_INPUTS[fname]]['path'] = str(work / f'base-{name}.npz')
        holder = None
        if hold_lock:
            holder = lock.open('r'); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            expect(lambda: run_chain(r2, rsp, SR, source or cs, M, o, config_path, review_sha, inputs), Stop, text)
        finally:
            if holder is not None:
                holder.close()
        assert not (o / 'manifest.json').exists(); neg[name] = 'STOP: ' + text

    def flip_key(cc): cc['keys'][5, 2] += 1
    def flip_role(cc): cc['tuning'][0] = ~cc['tuning'][0]
    def flip_y(cc): cc['y10'][3] = (cc['y10'][3] + 1) % 10
    def flip_token(cc): cc['balls'][7] = (cc['balls'][7] + 1) % 4
    def flip_base_role(cc): cc['fit'][0] = ~cc['fit'][0]
    chain_stop('keys', 'keys differ', flip_key); chain_stop('role', 'role mask', flip_role); chain_stop('y10', 'labels', flip_y)
    r3 = copy.deepcopy(rs); r3['expected_denominators']['accepted_rows'] += 1; chain_stop('denominator', 'denominator mismatch', regm=r3)
    r4 = copy.deepcopy(rs); r4['caps']['prepare']['min_free_memory_bytes'] = 1 << 60; chain_stop('memory', 'memory gate', regm=r4)
    chain_stop('lock_busy', 'lock busy', hold_lock=True)
    r5 = copy.deepcopy(rs); r5['lock']['path'] = str(work / 'no-such.lock'); chain_stop('lock_missing', 'lock file missing', regm=r5)
    chain_stop('base_seq_inputs_other_rows', 'base seq-inputs.npz differs', mutate_base=('seq-inputs.npz', flip_token))
    chain_stop('base_corpus_subset_other_rows', 'base corpus-subset.npz differs', mutate_base=('corpus-subset.npz', flip_base_role))
    r6 = copy.deepcopy(rs); r6['feature_dimension'] = 198; chain_stop('base_x_shape', 'base x-prequential shape', regm=r6)
    r7 = copy.deepcopy(rs); r7['source_rows'] += 1; chain_stop('source_rows', 'source rows differ', regm=r7)
    bad15 = raw.copy(); bad15.loc[bad15.index[11], 'zone'] = 15; cs15, _, _ = write_source(bad15, 'zone15.parquet'); chain_stop('zone_out_of_range', 'not an integer in 1..14', source=cs15)
    ck['chain_negative_controls'] = neg
    # ---- zone-read negative controls on the function itself (key + zone tables written to temp parquet files)
    zt = raw[list(ZONE_COLUMNS)].copy(); keys = corpus['keys']; zneg = {}

    def zone_stop(name, frame, text, rows=None, k=keys):
        p = work / f'zone-{name}.parquet'; pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), p)
        expect(lambda: read_zone(p, k, len(frame) if rows is None else rows), Stop, text); zneg[name] = 'STOP: ' + text
    dup = zt.copy(); dup.iloc[1, :3] = dup.iloc[0, :3].to_numpy()   # two rows now share one key (row count unchanged)
    zone_stop('duplicate_key', dup, 'more than one row'); zone_stop('row_count', zt, 'rows', rows=len(zt) + 1)
    zone_stop('accepted_key_absent', zt.iloc[5:], 'no key match'); zone_stop('zone_column_absent', zt[list(KEY)], 'column absent')
    half = zt.astype({'zone': 'float64'}); half.iloc[0, 3] = 3.5; zone_stop('non_integer_zone', half, 'not an integer in 1..14', k=keys)
    txt = zt.astype({'zone': 'string'}); zone_stop('non_numeric_zone', txt, 'not a numeric column')
    ck['zone_read_negative_controls'] = zneg
    # ---- the real chain on the synthetic world
    manifest, tok, _, S = run_chain(rs, rsp, SR, cs, M, work / 'inputs2', config_path, review_sha, inputs)
    holder = lock.open('r'); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB); holder.close(); ck['fixture_lock_released'] = True
    expect(lambda: run_chain(rs, rsp, SR, cs, M, work / 'inputs2', config_path, review_sha, inputs), SystemExit, 'exists'); ck['existing_output_blocks'] = True
    with np.load(work / 'inputs2' / 'seq-inputs2.npz', allow_pickle=False) as z:
        P = {k: np.array(z[k]) for k in z.files}
    edges_doc = json.loads((work / 'inputs2' / 'bucket-edges.json').read_text()); fit = corpus['fit']
    assert set(P) == set(TOKENS2_DTYPES) and all(P[k].dtype == np.dtype(TOKENS2_DTYPES[k]) for k in P) and sorted(p.name for p in (work / 'inputs2').iterdir()) == ['bucket-edges.json', 'manifest.json', 'seq-inputs2.npz']
    assert np.array_equal(np.column_stack([P[k] for k in KEY]), corpus['keys']) and manifest['zone_read']['columns_read'] == list(ZONE_COLUMNS)
    # zone token vs an independent dict lookup on the raw synthetic frame
    look = {(int(a), int(b), int(c_)): (0 if pd.isna(zv) else int(zv)) for a, b, c_, zv in zip(raw.game_pk, raw.at_bat_number, raw.pitch_number, raw.zone)}
    want_zone = np.array([look[tuple(int(v) for v in k)] for k in corpus['keys']]); assert np.array_equal(P['zone'], want_zone) and (P['zone'] == 0).any() and P['zone'].max() == 14
    # bucket edges vs an independent pandas quantile on FIT rows; buckets vs a python bisect loop
    edge_report = {}
    for f in PHYSICS:
        col = g[f].astype('float64'); want = col[fit].dropna().quantile(np.arange(1, cfg['buckets'][f]) / cfg['buckets'][f], interpolation='linear').to_numpy()
        got = np.array(edges_doc['edges'][f]); gothex = np.array([float.fromhex(h) for h in edges_doc['edges_hex'][f]])
        assert np.array_equal(got, gothex) and np.abs(got - want).max() < 1e-12 and len(got) == cfg['buckets'][f] - 1
        vals = col.to_numpy(); loop = np.array([0 if not np.isfinite(v) else 1 + bisect.bisect_right(list(got), v) for v in vals]); assert np.array_equal(loop, P[f + '_bucket'])
        cnt = np.bincount(P[f + '_bucket'][fit], minlength=cfg['buckets'][f] + 1)
        edge_report[f] = {'bins': cfg['buckets'][f], 'edges_max_abs_diff_vs_pandas_fit_quantile': float(np.abs(got - want).max()), 'buckets_equal_python_bisect_loop': True,
                          'fit_rows_per_bucket_min_max': [int(cnt[1:].min()), int(cnt[1:].max())], 'missing_rows': int((P[f + '_bucket'] == 0).sum()), 'max_bucket': int(P[f + '_bucket'].max())}
    assert (P['release_speed_bucket'] == 0).any()
    ck['bucket_edges'] = edge_report
    # FIT-only fitting, function level: values outside FIT cannot move an edge; values inside FIT do
    v = g['release_speed'].to_numpy(dtype=np.float64, na_value=np.nan); e0, _ = fit_bucket_edges(v, fit, 8); v1 = v.copy(); v1[~fit] = v1[~fit] * 1.5 + 30; v2 = v.copy(); v2[fit] = v2[fit] * 1.5 + 30
    assert np.array_equal(e0, fit_bucket_edges(v1, fit, 8)[0]) and not np.array_equal(e0, fit_bucket_edges(v2, fit, 8)[0])
    # FIT-only fitting, chain level: rewrite the physics of every non-FIT (SELECT/CAL/EVAL) source row -> identical edges, different non-FIT tokens
    late = raw.copy(); nonfit = (pd.to_datetime(late.game_date) > pd.Timestamp(rs['roles']['FIT'][1])).to_numpy()
    for f in PHYSICS:
        late.loc[nonfit, f] = late.loc[nonfit, f] * 1.5 + 30
    cs2, pp2, mp2 = write_source(late, 'nonfit-physics.parquet'); r8 = copy.deepcopy(rs)
    man2, tok2, _, _ = run_chain(r8, rsp, SR, cs2, M, work / 'inputs2-nonfit-physics', config_path, review_sha, inputs)
    ed2 = json.loads((work / 'inputs2-nonfit-physics' / 'bucket-edges.json').read_text())
    same_edges = ed2['edges_hex'] == edges_doc['edges_hex']; fit_same = all(np.array_equal(tok2[f + '_bucket'][fit], P[f + '_bucket'][fit]) for f in PHYSICS)
    moved = int(sum((tok2[f + '_bucket'][~fit] != P[f + '_bucket'][~fit]).sum() for f in PHYSICS)); assert same_edges and fit_same and moved > 0
    fitrow = raw.copy(); infit = ~nonfit
    for f in PHYSICS:
        fitrow.loc[infit, f] = fitrow.loc[infit, f] * 1.5 + 30
    cs3, _, _ = write_source(fitrow, 'fit-physics.parquet'); run_chain(copy.deepcopy(rs), rsp, SR, cs3, M, work / 'inputs2-fit-physics', config_path, review_sha, inputs)
    ed3 = json.loads((work / 'inputs2-fit-physics' / 'bucket-edges.json').read_text()); assert ed3['edges_hex'] != edges_doc['edges_hex']
    ck['bucket_edges_fit_rows_only'] = {'function_level_non_fit_values_rewritten_edges_identical': True, 'function_level_fit_values_rewritten_edges_change': True,
                                        'chain_level_non_fit_source_physics_rewritten': {'edges_identical': bool(same_edges), 'fit_row_tokens_identical': bool(fit_same), 'non_fit_row_tokens_changed': moved},
                                        'chain_level_fit_source_physics_rewritten_edges_change': True}
    # context indices (pitcher side and batter side) on the projected synthetic frame vs an independent grouped loop
    res = {}; role = np.select([corpus['fit'], corpus['tuning'], corpus['calibration'], corpus['evaluation']], [0, 1, 2, 3]); order = (S['game_pk'], S['at_bat_number'], S['pitch_number'])
    for side, group, lengths in (('pitcher_side', S['pitcher'], (4, 16, 32, 64)), ('batter_side', S['batter'], (4, 16))):
        for length in lengths:
            fast = T.build_context_index(group, *order, length); brute = grouped_brute_context(group, S, length); rep = T.check_context(fast, group, *order)
            vmask = fast >= 0; assert np.array_equal(fast, brute) and (role[np.where(vmask, fast, 0)] == role[:, None])[vmask].all()
            res[f'{side}_L{length}'] = dict(rep, equal_to_grouped_loop=True)
    ck['context_index_on_projected_frame'] = dict(res, contexts_never_cross_roles=True)
    from scipy.special import softmax, expit
    z = rng.normal(size=(3000, 10)); zq = rng.normal(size=3000); zr_ = rng.normal(size=3000)
    a = T.leaves(softmax(z, axis=1), expit(zq), expit(zr_)); b = M.safe.leaves(softmax(z, axis=1), expit(zq), expit(zr_))
    yy = np.array([rng.choice(10, p=r) for r in softmax(z / 1.3, axis=1)]); yb = (rng.random(3000) < expit(zq / .8)).astype(float)
    t_equal = T.temperature(z, yy) == SR.temperature(z, yy) and T.temperature(zq, yb) == SR.temperature(zq, yb)
    lab = np.array_equal(T.seven_label(y10, corpus['narrow_whiff'], corpus['ground_ball_per_pitch']), M.ap.seven_label(y10, corpus['narrow_whiff'], corpus['ground_ball_per_pitch']))
    assert np.array_equal(a, b) and t_equal and lab and T.T_BOUNDS == SR.T_BOUNDS and T.BIP == M.safe.BIP
    ck['train_seq2_copies_equal_pinned'] = {'leaves_vs_scale_safe': True, 'temperature_vs_scale_runner': True, 'seven_label_vs_abc_pilot': True, 'T_BOUNDS': True, 'BIP': True}
    ck['package'] = {'rows': int(n), 'role_rows': rs['expected_denominators']['role_rows'], 'seq_inputs2_members': {k: str(v.dtype) for k, v in P.items()}, 'integer_arrays_only': True,
                     'files_in_inputs2': ['bucket-edges.json', 'manifest.json', 'seq-inputs2.npz'], 'base_files_referenced_not_copied': sorted(manifest['base_files']),
                     'zone_equals_independent_lookup': True, 'zone_read': manifest['zone_read'], 'manifest_sha256': sha(work / 'inputs2' / 'manifest.json'), 'peak_rss_bytes': manifest['resources']['peak_rss_bytes']}
    ck['torch_imported_in_this_process'] = 'torch' in sys.modules
    assert 'torch' not in sys.modules
    out['synthetic_registration'] = str(rsp); out['inputs_dir'] = str(work / 'inputs2'); out['base_inputs_dir'] = str(base); out['verdict'] = 'SELFCHECK_PASS'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--registration', required=True); ap.add_argument('--out', default=str(REPO / 'data' / 'ws-seq' / 'inputs2')); ap.add_argument('--config', default=str(CONFIG_JSON))
    ap.add_argument('--plan-out'); ap.add_argument('--work'); ap.add_argument('--selfcheck-out')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true'); mode.add_argument('--selfcheck', action='store_true'); mode.add_argument('--execute', action='store_true')
    args = ap.parse_args(); reg = load_registration(args.registration)
    if args.plan:
        p = plan(reg, args.registration, args.config, args.out)
        if args.plan_out:
            save_json(args.plan_out, p)
        print(json.dumps({k: p[k] for k in ('execute_ready', 'refused', 'pending_placeholders', 'inputs_not_verified', 'calibration_inputs_not_verified', 'review', 'checks')}, indent=1, ensure_ascii=False))
        return 0 if p['execute_ready'] else 2
    if args.selfcheck:
        r = selfcheck(reg, args.registration, args.config, args.work)
        if args.selfcheck_out:
            save_json(args.selfcheck_out, r)
        print(json.dumps({'verdict': r['verdict'], 'checks': sorted(r['checks'])})); return 0
    try:
        print(json.dumps(execute(reg, args.registration, args.config, args.out), indent=1, ensure_ascii=False)); return 0
    except Stop as e:
        print(str(e), file=sys.stderr); return 6


if __name__ == '__main__':
    sys.exit(main())
