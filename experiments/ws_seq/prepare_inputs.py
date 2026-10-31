"""EXP-WS-SEQ-001 (E4) input preparation on the Mac: builds the minimal server package `inputs/`.

  x-prequential.npy   byte copy of the E1b encoded prequential features (float32 (n,199)), sha256 pinned in the registration
  corpus-subset.npz   keys, y10, y7, games, fit, tuning, calibration, evaluation, narrow_whiff, ground_ball_per_pitch (scale-002 corpus)
  seq-inputs.npz      integer-only per-row tokens from the projected frame g (same projection as scale-002 / E1 / E1b)
  manifest.json       sha256 of the three files + the sha256 of train_seq.py and the JSON config they are bound to

No parquet goes to the server. Runner python: /Users/song/Projects/pitcheezy/.venv-observer/bin/python -I

Modes:
  --plan       sha256 of every registered input and gate state; decodes no array. Refuses (exit 2) while any
               "PENDING_E1B" placeholder is present in the registration. Writes --plan-out.
  --selfcheck  synthetic rows only (scale_runner.synthetic_frame) under --work; never opens the real parquet or lock.
  --execute    BLOCKED unless the registration is READY_FOR_REVIEW, free of placeholders, and REVIEW.json (verdict PASS,
               distinct reviewer, attempt id, matching source sha256) exists. Chain: gates -> memory gate -> lock ->
               read_source -> project -> row-for-row identity with corpus.npz -> package -> manifest. Lock released in finally.
"""
import os, sys
sys.dont_write_bytecode = True  # importing the pinned helpers must not write __pycache__ into their directories
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'  # before numpy
import argparse, copy, fcntl, gc, hashlib, importlib.util, json, re, resource, shutil, signal, subprocess, threading, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP_ID = 'EXP-WS-SEQ-001'
READY = 'READY_FOR_REVIEW'
PENDING = 'PENDING_E1B'
CONFIG_JSON = REPO / 'configs' / (EXP_ID + '.json')
CONFIG_YAML = REPO / 'configs' / (EXP_ID + '.yaml')
SOURCES = {'experiments/ws_seq/prepare_inputs.py': HERE / 'prepare_inputs.py', 'experiments/ws_seq/train_seq.py': HERE / 'train_seq.py',
           'experiments/ws_seq/evaluate_seq.py': HERE / 'evaluate_seq.py', 'configs/EXP-WS-SEQ-001.json': CONFIG_JSON, 'configs/EXP-WS-SEQ-001.yaml': CONFIG_YAML}
ROLE_KEY = {'FIT': 'FIT', 'SELECT': 'SELECT', 'CAL': 'CAL', 'EVAL': 'TEST'}  # scale_runner.project names the EVAL mask TEST
CORPUS_ROLE = {'FIT': 'fit', 'SELECT': 'tuning', 'CAL': 'calibration', 'TEST': 'evaluation'}
CORPUS_MEMBERS = ('keys', 'y10', 'y7', 'games', 'fit', 'tuning', 'calibration', 'evaluation', 'narrow_whiff', 'ground_ball_per_pitch')
SEQ_DTYPES = {'game_pk': np.int32, 'at_bat_number': np.int16, 'pitch_number': np.int16, 'pitcher': np.int32, 'batter': np.int32, 'pitch_type': np.int16,
              'outcome10': np.int16, 'balls': np.int16, 'strikes': np.int16, 'stand': np.int16}
STAND = {'L': 0, 'R': 1}
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
    """Every JSON path whose value is exactly the placeholder string."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in pending_fields(v, f'{at}.{k}' if at else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in pending_fields(v, f'{at}[{i}]')]
    return [at] if obj == PENDING else []


def load_registration(path):
    reg = json.loads(Path(path).read_text())
    ok = [reg['id'] == EXP_ID, reg['bootstrap']['unit'] == 'game_pk', reg['feature_dimension'] == 199, reg['token']['context_length'] == 32,
          reg['config']['authoritative'] == 'configs/EXP-WS-SEQ-001.json', set(CODE_INPUTS) <= set(reg['inputs']), 'x_prequential' in reg['inputs'], 'corpus' in reg['inputs']]
    if not all(ok):
        raise SystemExit('BLOCKED: REGISTRATION.json does not match this source: ' + str(ok))
    return reg


def review_gate(reg, reg_path):
    """--execute precondition (same schema as E1). Returns the sha256 of the accepted REVIEW.json."""
    if reg.get('status') != READY:
        raise SystemExit('BLOCKED: registration status is not exactly ' + READY)
    pend = pending_fields(reg)
    if pend:
        raise SystemExit('BLOCKED: registration still carries ' + PENDING + ' at: ' + ', '.join(pend))
    p = Path(reg['review']['path'])
    if not p.exists():
        raise SystemExit('BLOCKED: independent REVIEW.json (verdict PASS, matching source sha256) required at ' + str(p))
    r = json.loads(p.read_text())
    if r.get('verdict') != 'PASS' or r.get('distinct_from_author') is not True or r.get('attempt_id') != reg['attempt_id']:
        raise SystemExit('BLOCKED: review verdict / distinct reviewer / attempt id')
    if r.get('sources_sha256') != own_hashes(reg_path):
        raise SystemExit('BLOCKED: reviewed source sha256 differ from the current REGISTRATION.json / scripts / configs')
    return sha(p)


def verify_inputs(reg, names=None, collect=False):
    """sha256 of the registered inputs. Hash only: nothing is decoded. collect=True returns (observed, bad) instead of blocking."""
    observed = {}; bad = []
    for name, spec in reg['inputs'].items():
        if names is not None and name not in names:
            continue
        if spec['sha256'] == PENDING:
            observed[name] = None; bad.append(name + ' (' + PENDING + ')'); continue
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
    (adapted from recency_run.load_scale, e1-recency/recency_run.py sha256 af1947c0...f232d)."""
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
            list(M.fsf.TYPES) == reg['token']['pitch_type_order'] and len(M.fsf.TYPES) == 19]
    if not all(same):
        raise SystemExit('BLOCKED: scale registration differs from the E4 registration (source/roles/pinned/lock/types): ' + str(same))
    return SR, c, M


def load_train_module():
    """train_seq.py by file path (numpy only at import time; torch is imported lazily there and never here)."""
    spec = importlib.util.spec_from_file_location('ws_seq_train', HERE / 'train_seq.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------- resources (copied from recency_run.py: free_memory_bytes, memory_gate, acquire_lock)
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


# ---------------------------------------------------------------- population identity (recency_run.check_population, adapted: + head labels)
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
    """Integer-only per-row token sources. pitch_type index follows first_small_features.TYPES (19, UNKNOWN = 0)."""
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
    if not (0 <= out['balls'].min() and out['balls'].max() <= 3 and 0 <= out['strikes'].min() and out['strikes'].max() <= 2 and 0 <= out['outcome10'].min() and out['outcome10'].max() <= 9):
        raise Stop('STOP token range (balls 0..3, strikes 0..2, outcome10 0..9)')
    return out


def npy_header(path):
    a = np.load(path, mmap_mode='r', allow_pickle=False); return tuple(a.shape), a.dtype


def run_chain(reg, reg_path, SR, c, M, out, config_path, review_sha, inputs_sha):
    """memory gate -> lock -> watchdog -> projection -> identity -> package -> manifest (written last)."""
    import pandas, pyarrow
    caps = reg['caps']['prepare']; out = Path(out)
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
        with np.load(reg['inputs']['corpus']['path'], allow_pickle=False) as z:
            corpus = {k: np.array(z[k]) for k in CORPUS_MEMBERS}
        den = check_population(g, rm, y7, corpus, reg['expected_denominators'], M)
        S = sequence_arrays(g, M)
        if not (np.array_equal(S['game_pk'], corpus['keys'][:, 0]) and np.array_equal(S['at_bat_number'], corpus['keys'][:, 1]) and np.array_equal(S['pitch_number'], corpus['keys'][:, 2])
                and np.array_equal(S['outcome10'], corpus['y10'])):
            raise Stop('STOP sequence arrays differ from corpus keys / y10 after the integer cast')
        del g; gc.collect()
        xs = Path(reg['inputs']['x_prequential']['path']); shape, dtype = npy_header(xs)
        if shape != (den['accepted_rows'], reg['feature_dimension']) or dtype != np.float32:
            raise Stop(f'STOP x-prequential shape/dtype {shape} {dtype}')
        out.mkdir(parents=True)
        np.savez(out / 'seq-inputs.npz', **S); np.savez(out / 'corpus-subset.npz', **corpus)
        shutil.copyfile(xs, out / 'x-prequential.npy')
        x_sha = sha(out / 'x-prequential.npy')
        if x_sha != reg['inputs']['x_prequential']['sha256']:
            raise Stop('STOP copied x-prequential.npy sha256 differs from the registration')
        files = {'x-prequential.npy': {'sha256': x_sha, 'bytes': (out / 'x-prequential.npy').stat().st_size, 'shape': list(shape), 'dtype': 'float32'},
                 'corpus-subset.npz': {'sha256': sha(out / 'corpus-subset.npz'), 'bytes': (out / 'corpus-subset.npz').stat().st_size, 'members': {k: [list(v.shape), str(v.dtype)] for k, v in corpus.items()}},
                 'seq-inputs.npz': {'sha256': sha(out / 'seq-inputs.npz'), 'bytes': (out / 'seq-inputs.npz').stat().st_size, 'members': {k: [list(v.shape), str(v.dtype)] for k, v in S.items()}}}
        manifest = {'id': EXP_ID, 'attempt_id': reg['attempt_id'], 'data_role': reg['data_role'], 'synthetic': 'synthetic_selfcheck' in reg, 'utc': time.time(), 'files': files,
                    'bound_sources': {'train_seq.py': sha(HERE / 'train_seq.py'), 'config': sha(config_path), 'config_path': str(config_path)},
                    'registration_sha256': sha(reg_path), 'review_sha256': review_sha, 'prepare_inputs_sha256': sha(HERE / 'prepare_inputs.py'),
                    'source_sha256': {k: inputs_sha.get(k) for k in ('s2_parquet', 's2_manifest', 'corpus', 'x_prequential', 'scale_registration', 'scale_runner', 'scale_safe')},
                    'source_binding': binding, 'denominators': den, 'source_rows': census['source_rows'], 'excluded': census['excluded'],
                    'token_sources': {'pitch_type_order': list(M.fsf.TYPES), 'stand': STAND, 'row_order': 'scale-002 corpus order = sort by (game_pk, at_bat_number, pitch_number)',
                                      'note': 'integer arrays only; no physics, no description/events text; outcome10 of a row is used by the model ONLY as a token of LATER rows'},
                    'resources': {'free_memory_bytes_before_lock': free_before, 'free_memory_bytes_after_lock': free_locked, 'lock': lock_info, 'peak_rss_bytes': peak_rss_bytes(), 'elapsed_seconds': time.monotonic() - start},
                    'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pandas.__version__, 'pyarrow': pyarrow.__version__}}
        if peak_rss_bytes() > caps['max_rss_bytes'] or time.monotonic() - work > caps['wall_seconds']:
            raise Stop('STOP RSS or wall cap exceeded')
        save_json(out / 'manifest.json', manifest)
        return manifest, S, corpus
    finally:
        if wd is not None:
            wd.halt.set()
        if lockf is not None:
            lockf.close()
        signal.signal(signal.SIGTERM, previous)


# ---------------------------------------------------------------- plan / execute
def plan(reg, reg_path, config_path, out):
    pend = pending_fields(reg); observed, bad = verify_inputs(reg, collect=True)
    ev_obs = {}; ev_bad = []
    for name, spec in reg['evaluation_inputs'].items():
        if PENDING in (spec['sha256'], spec['path']):
            ev_obs[name] = None; ev_bad.append(name + ' (' + PENDING + ')'); continue
        p = Path(spec['path']); ev_obs[name] = sha(p) if p.is_file() else None
        if ev_obs[name] != spec['sha256']:
            ev_bad.append(name)
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
    checks = {'runner_python_exists': Path(reg['runtime']['runner_python']).exists(), 'lock_file_exists': Path(reg['lock']['path']).is_file(), 'output_dir_absent': not Path(out).exists(),
              'status_ready_for_review': reg.get('status') == READY, 'no_pending_placeholder': not pend, 'inputs_all_verified': not bad, 'evaluation_inputs_all_verified': not ev_bad,
              'pinned_closure_loaded': isinstance(closure, list), 'config_is_repo_authoritative': Path(config_path).resolve() == CONFIG_JSON.resolve(), 'config_twins_identical': twins is True}
    ready = bool(all(checks.values()) and review_state.startswith('PASS'))
    return {'id': reg['id'], 'attempt_id': reg['attempt_id'], 'utc': time.time(), 'own_sources_sha256': own_hashes(reg_path), 'pending_placeholders': pend,
            'inputs_sha256': observed, 'inputs_not_verified': bad, 'evaluation_inputs_sha256': ev_obs, 'evaluation_inputs_not_verified': ev_bad,
            'pinned_closure_loaded': closure, 'checks': checks, 'review': review_state, 'review_file_exists': Path(reg['review']['path']).exists(), 'execute_ready': ready,
            'refused': None if ready else 'PLAN REFUSED: ' + ('placeholder ' + PENDING + ' present; ' if pend else '') + 'see checks / review',
            'arrays_decoded': 0, 'output_dir': str(out),
            'stages': ['Mac: prepare_inputs.py --execute (projection + identity + package)', 'lead: copy inputs/ to the server path linked by .gpurc',
                       'server: train_seq.py via gpu run (seed 0 N0+N1, then seeds 1,2 only if the stop rule passes)', 'Mac: gpu pull, evaluate_seq.py'],
            'command': f"{reg['runtime']['runner_python']} -I {HERE / 'prepare_inputs.py'} --execute --registration {reg_path}"}


def execute(reg, reg_path, config_path, out):
    if 'synthetic_selfcheck' in reg:
        raise SystemExit('BLOCKED: synthetic key in the real registration')
    if Path(config_path).resolve() != CONFIG_JSON.resolve():
        raise SystemExit('BLOCKED: --execute binds only the authoritative repo config ' + str(CONFIG_JSON))
    review_sha = review_gate(reg, reg_path)
    if Path(sys.prefix).resolve() != Path(reg['runtime']['runner_python']).parent.parent.resolve():
        raise SystemExit('BLOCKED: run with the registered runner python')
    inputs = verify_inputs(reg); SR, c, M = load_scale(reg)
    manifest, _, _ = run_chain(reg, reg_path, SR, c, M, out, config_path, review_sha, inputs)
    return {'verdict': 'PACKAGE_READY', 'out': str(out), 'manifest_sha256': sha(Path(out) / 'manifest.json'), 'files': {k: v['sha256'] for k, v in manifest['files'].items()},
            'denominators': manifest['denominators'], 'resources': manifest['resources']}


# ---------------------------------------------------------------- selfcheck (synthetic only)
def expect(fn, exc, text):
    try:
        fn()
    except exc as e:
        if text not in str(e):
            raise AssertionError(f'wrong message: {e!r} (wanted {text!r})')
        return True
    raise AssertionError('not raised: ' + text)


def grouped_brute_context(S, length):
    """Independent pandas/python loop: per (pitcher, game) group sorted by (at_bat_number, pitch_number), earlier rows only."""
    import pandas as pd
    f = pd.DataFrame({k: S[k].astype(np.int64) for k in ('pitcher', 'game_pk', 'at_bat_number', 'pitch_number')}); out = np.full((len(f), length), -1, np.int32)
    for _, grp in f.groupby(['pitcher', 'game_pk'], sort=False):
        rows = grp.sort_values(['at_bat_number', 'pitch_number']).index.to_list()
        for pos, i in enumerate(rows):
            prev = rows[max(0, pos - length):pos][::-1]
            out[i, :len(prev)] = prev
    return out


def selfcheck(reg, reg_path, config_path, work):
    """Builds a synthetic world under `work` (must be a directory named selfcheck-tmp) and runs the SAME run_chain as --execute."""
    import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    work = Path(work)
    if work.name != 'selfcheck-tmp' or (work / 'inputs').exists():
        raise SystemExit('BLOCKED: --work must be a directory named selfcheck-tmp without an inputs/ directory')
    work.mkdir(parents=True, exist_ok=True); (work / 'holdout').mkdir(exist_ok=True); out = {'id': EXP_ID, 'mode': 'prepare_inputs --selfcheck (synthetic)', 'checks': {}}; ck = out['checks']
    # gates on the REAL registration (hash only)
    ck['real_registration_pending_placeholders'] = pending_fields(reg)
    if pending_fields(reg):
        expect(lambda: review_gate(reg, reg_path), SystemExit, PENDING); ck['review_gate_refuses_pending'] = True
        expect(lambda: verify_inputs(reg), SystemExit, PENDING); ck['verify_inputs_refuses_pending'] = True
    expect(lambda: execute(dict(reg, synthetic_selfcheck=True), reg_path, config_path, work / 'x'), SystemExit, 'synthetic key'); ck['execute_refuses_synthetic_registration'] = True
    SR, c, M = load_scale(reg); T = load_train_module()
    # ---- synthetic world (as recency_run.synthetic_world: season over the registered role windows + debut players)
    raw, _ = SR.synthetic_frame(np.random.default_rng(20261011)); date = pd.to_datetime(raw.game_date)
    raw.loc[(raw.pitcher == 101) & (date >= '2026-08-10'), 'pitcher'] = 901; raw.loc[(raw.pitcher == 103) & (date >= '2026-09-07'), 'pitcher'] = 903
    raw.loc[(raw.batter == 201) & (date >= '2026-08-10'), 'batter'] = 991
    pp = work / 'holdout' / 'synthetic.parquet'; pq.write_table(pa.Table.from_pandas(raw[M.cp.PROJECT], preserve_index=False), pp)
    bind = {'version': 'synthetic', 'season': 2026, 'file': 'holdout/synthetic.parquet', 'start': '2026-03-25', 'end': '2026-09-27', 'rows': len(raw), 'columns': 29, 'game_type': 'R', 'parquet_sha256': sha(pp), 'fetch_commit': 'synthetic'}
    mp = work / 'manifest-source.json'
    save_json(mp, {'version': 'synthetic', 'fetch_commit': 'synthetic', 'parameters': {'game_type': 'R', 'season_dates': {'2026': [bind['start'], bind['end']]}},
                   'seasons': {'2026': {'season': 2026, 'file': bind['file'], 'start': bind['start'], 'end': bind['end'], 'rows': len(raw), 'rows_raw': len(raw), 'columns': 29, 'duplicate_keys': 0, 'sha256': bind['parquet_sha256'], 'frozen': True}}})
    cs = copy.deepcopy(c); cs['fit_sample']['min_games_per_team'] = 2
    cs['source'] = {'id': 'SYNTHETIC', 'parquet': str(pp), 'parquet_sha256': bind['parquet_sha256'], 'manifest': str(mp), 'manifest_sha256': sha(mp), 'manifest_binding': bind, 'rows': len(raw)}
    src, _ = M.cp.read_source(cs); g, census, rm, sample, info, y7 = SR.project(cs, src, M); n = len(g)
    y10 = g.outcome10.to_numpy(np.int64); rng = np.random.default_rng(7)
    x = (rng.normal(size=(n, reg['feature_dimension'])) * .1).astype(np.float32)   # synthetic stand-in for E1b's x-prequential: pre-pitch columns only
    x[:, 0] += g.balls.to_numpy() * .5; x[:, 1] += g.strikes.to_numpy() * .5
    for j, t in enumerate(('FF', 'SL', 'CH', 'CU', 'UNKNOWN')):   # the called pitch type drives the synthetic outcome, so the heads and temperatures are identified
        x[:, 2 + j] += (g.pitch_type == t).to_numpy() * 2.0
    np.save(work / 'x-prequential-src.npy', x, allow_pickle=False)
    np.savez(work / 'corpus.npz', keys=g[M.cp.KEY].to_numpy('<i8'), y7=y7, y10=y10, games=g.game_pk.to_numpy(), sample=sample, narrow_whiff=g.narrow_whiff.to_numpy(np.int8),
             ground_ball_per_pitch=g.ground_ball_per_pitch.to_numpy(np.int8), **{CORPUS_ROLE[k]: rm[k] for k in CORPUS_ROLE})
    lock = work / 'fixture.lock'; lock.write_text('selfcheck fixture lock\n')
    rs = copy.deepcopy(reg); rs['synthetic_selfcheck'] = True
    rs['expected_denominators'] = {'accepted_rows': int(n), 'role_rows': {r: int(rm[k].sum()) for r, k in ROLE_KEY.items()}, 'eval_games': int(g.game_pk[rm['TEST']].nunique()), 'eval_leaf_counts': np.bincount(y7[rm['TEST']], minlength=7).tolist()}
    rs['lock'].update(path=str(lock), retry_seconds=.05, max_wait_seconds=.2); rs['caps']['prepare'].update(min_free_memory_bytes=1 << 20, rss_sample_seconds=1)
    rs['review']['path'] = str(work / 'REVIEW.json')
    for k, p in (('corpus', work / 'corpus.npz'), ('x_prequential', work / 'x-prequential-src.npy'), ('s2_parquet', pp), ('s2_manifest', mp)):
        rs['inputs'][k] = dict(rs['inputs'][k], path=str(p), sha256=sha(p))
    rs['stop_rule']['reference_select_nll'] = 99.0
    for k in rs['evaluation_inputs']:
        rs['evaluation_inputs'][k] = {'path': str(work / 'missing'), 'sha256': 'filled by the job selfcheck'}
    rsp = work / 'REGISTRATION.synthetic.json'; save_json(rsp, rs)
    assert not pending_fields(rs)
    # review gate negatives, then a valid synthetic review
    expect(lambda: review_gate(rs, rsp), SystemExit, 'REVIEW.json'); good = {'verdict': 'PASS', 'distinct_from_author': True, 'attempt_id': rs['attempt_id'], 'sources_sha256': own_hashes(rsp)}
    for name, bad in (('verdict', dict(good, verdict='FAIL')), ('not_distinct', dict(good, distinct_from_author=False)), ('attempt', dict(good, attempt_id='x')), ('sha', dict(good, sources_sha256=dict(good['sources_sha256'], **{'REGISTRATION.json': '0'})))):
        save_json(rs['review']['path'], bad); expect(lambda: review_gate(rs, rsp), SystemExit, 'BLOCKED')
    expect(lambda: review_gate(dict(rs, status='DRAFT'), rsp), SystemExit, 'status')
    save_json(rs['review']['path'], good); review_sha = review_gate(rs, rsp); ck['review_gate_negative_controls'] = ['missing', 'verdict', 'not_distinct', 'attempt', 'sha', 'status']
    inputs = verify_inputs(rs, ('corpus', 'x_prequential', 's2_parquet', 's2_manifest') + CODE_INPUTS)
    bad_in = copy.deepcopy(rs); bad_in['inputs']['x_prequential']['sha256'] = '0' * 64
    expect(lambda: verify_inputs(bad_in, ('x_prequential',)), SystemExit, 'x_prequential'); ck['input_sha_mismatch_blocks'] = True
    # negative controls of the chain (each must STOP before a manifest exists)
    neg = {}

    def chain_stop(name, text, mutate=None, regm=None, hold_lock=False):
        r2 = copy.deepcopy(rs) if regm is None else regm; o = work / ('neg-' + name)
        if mutate is not None:
            with np.load(work / 'corpus.npz') as z:
                cc = {k: np.array(z[k]) for k in z.files}
            mutate(cc); np.savez(work / f'corpus-{name}.npz', **cc); r2['inputs']['corpus']['path'] = str(work / f'corpus-{name}.npz')
        holder = None
        if hold_lock:
            holder = lock.open('r'); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            expect(lambda: run_chain(r2, rsp, SR, cs, M, o, config_path, review_sha, inputs), Stop, text)
        finally:
            if holder is not None:
                holder.close()
        assert not (o / 'manifest.json').exists(); neg[name] = 'STOP: ' + text

    def flip_key(cc): cc['keys'][5, 2] += 1
    def flip_role(cc): cc['tuning'][0] = ~cc['tuning'][0]
    def flip_y(cc): cc['y10'][3] = (cc['y10'][3] + 1) % 10
    def flip_head(cc): cc['narrow_whiff'][np.flatnonzero(cc['y10'] == 1)[0]] ^= 1
    chain_stop('keys', 'keys differ', flip_key); chain_stop('role', 'role mask', flip_role); chain_stop('y10', 'labels', flip_y); chain_stop('head', 'labels', flip_head)
    r3 = copy.deepcopy(rs); r3['expected_denominators']['accepted_rows'] += 1; chain_stop('denominator', 'denominator mismatch', regm=r3)
    r4 = copy.deepcopy(rs); r4['caps']['prepare']['min_free_memory_bytes'] = 1 << 60; chain_stop('memory', 'memory gate', regm=r4)
    chain_stop('lock_busy', 'lock busy', hold_lock=True)
    r5 = copy.deepcopy(rs); r5['lock']['path'] = str(work / 'no-such.lock'); chain_stop('lock_missing', 'lock file missing', regm=r5)
    r6 = copy.deepcopy(rs); r6['feature_dimension'] = 198; chain_stop('x_shape', 'x-prequential shape', regm=r6)
    r7 = copy.deepcopy(rs); r7['inputs']['x_prequential']['sha256'] = '0' * 64; chain_stop('x_copy_sha', 'copied x-prequential', regm=r7)
    ck['chain_negative_controls'] = neg
    # the real chain on the synthetic world
    manifest, S, corpus = run_chain(rs, rsp, SR, cs, M, work / 'inputs', config_path, review_sha, inputs)
    holder = lock.open('r'); fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB); holder.close(); ck['fixture_lock_released'] = True
    expect(lambda: run_chain(rs, rsp, SR, cs, M, work / 'inputs', config_path, review_sha, inputs), SystemExit, 'exists'); ck['existing_output_blocks'] = True
    with np.load(work / 'inputs' / 'seq-inputs.npz', allow_pickle=False) as z:
        P = {k: np.array(z[k]) for k in z.files}
    with np.load(work / 'inputs' / 'corpus-subset.npz', allow_pickle=False) as z:
        Cs = {k: np.array(z[k]) for k in z.files}
    assert set(P) == set(SEQ_DTYPES) and all(P[k].dtype == np.dtype(SEQ_DTYPES[k]) for k in P) and set(Cs) == set(CORPUS_MEMBERS)
    assert np.array_equal(Cs['keys'], g[M.cp.KEY].to_numpy('<i8')) and np.array_equal(P['pitcher'], g.pitcher.to_numpy()) and np.array_equal(P['batter'], g.batter.to_numpy())
    assert np.array_equal(np.array(M.fsf.TYPES)[P['pitch_type']], g.pitch_type.to_numpy()) and np.array_equal(np.array(['L', 'R'])[P['stand']], g.stand.to_numpy())
    assert np.array_equal(np.load(work / 'inputs' / 'x-prequential.npy'), x) and manifest['files']['x-prequential.npy']['sha256'] == rs['inputs']['x_prequential']['sha256']
    ck['package'] = {'rows': int(n), 'role_rows': rs['expected_denominators']['role_rows'], 'seq_members': {k: str(v.dtype) for k, v in P.items()}, 'integer_arrays_only': True,
                     'keys_equal_corpus_keys': True, 'tokens_decode_back_to_frame': True, 'x_byte_copy_equal': True, 'manifest_sha256': sha(work / 'inputs' / 'manifest.json'),
                     'unknown_pitch_type_rows': int((P['pitch_type'] == 0).sum()), 'peak_rss_bytes': manifest['resources']['peak_rss_bytes']}
    # train_seq copies vs the pinned originals (exact) and the context index vs an independent grouped loop on the projected synthetic frame
    L = reg['token']['context_length']; res = {}
    for length in (4, L):
        fast = T.build_context_index(P['pitcher'], P['game_pk'], P['at_bat_number'], P['pitch_number'], length); brute = grouped_brute_context(P, length)
        rep = T.check_context(fast, P['pitcher'], P['game_pk'], P['at_bat_number'], P['pitch_number']); res[str(length)] = dict(rep, equal_to_grouped_loop=bool(np.array_equal(fast, brute)))
        assert np.array_equal(fast, brute)
        v = fast >= 0; role = np.select([Cs['fit'], Cs['tuning'], Cs['calibration'], Cs['evaluation']], [0, 1, 2, 3])
        assert (role[np.where(v, fast, 0)] == role[:, None])[v].all()   # a context never crosses a data role (same game -> same game_date)
    ck['context_index_on_projected_frame'] = dict(res, contexts_never_cross_roles=True)
    from scipy.special import softmax, expit
    z = rng.normal(size=(3000, 10)); zq = rng.normal(size=3000); zr = rng.normal(size=3000)
    a = T.leaves(softmax(z, axis=1), expit(zq), expit(zr)); b = M.safe.leaves(softmax(z, axis=1), expit(zq), expit(zr))
    yy = np.array([rng.choice(10, p=r) for r in softmax(z / 1.3, axis=1)]); yb = (rng.random(3000) < expit(zq / .8)).astype(float)   # labels with a known scale so T is interior
    t_equal = T.temperature(z, yy) == SR.temperature(z, yy) and T.temperature(zq, yb) == SR.temperature(zq, yb)
    lab = np.array_equal(T.seven_label(y10, Cs['narrow_whiff'], Cs['ground_ball_per_pitch']), M.ap.seven_label(y10, Cs['narrow_whiff'], Cs['ground_ball_per_pitch']))
    assert np.array_equal(a, b) and t_equal and lab and T.T_BOUNDS == SR.T_BOUNDS and T.BIP == M.safe.BIP
    ck['train_seq_copies_equal_pinned'] = {'leaves_vs_scale_safe': True, 'temperature_vs_scale_runner': True, 'seven_label_vs_abc_pilot': True, 'T_BOUNDS': True, 'BIP': True}
    ck['torch_imported_in_this_process'] = 'torch' in sys.modules
    assert 'torch' not in sys.modules
    out['synthetic_registration'] = str(rsp); out['inputs_dir'] = str(work / 'inputs'); out['verdict'] = 'SELFCHECK_PASS'
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--registration', required=True); ap.add_argument('--out', default=str(REPO / 'data' / 'ws-seq' / 'inputs')); ap.add_argument('--config', default=str(CONFIG_JSON))
    ap.add_argument('--plan-out'); ap.add_argument('--work'); ap.add_argument('--selfcheck-out')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true'); mode.add_argument('--selfcheck', action='store_true'); mode.add_argument('--execute', action='store_true')
    args = ap.parse_args(); reg = load_registration(args.registration)
    if args.plan:
        p = plan(reg, args.registration, args.config, args.out)
        if args.plan_out:
            save_json(args.plan_out, p)
        print(json.dumps({k: p[k] for k in ('execute_ready', 'refused', 'pending_placeholders', 'inputs_not_verified', 'evaluation_inputs_not_verified', 'review')}, indent=1, ensure_ascii=False))
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
