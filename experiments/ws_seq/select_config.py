"""EXP-WS-SEQ-002 config selection on the Mac (after `gpu pull` of the ten seed-0 runs): picks ONE grid config
STRICTLY by its best uncalibrated SELECT seven-leaf NLL at seed 0 and writes SELECTION.json.

  python -I experiments/ws_seq/select_config.py --registration <e6-seq2/REGISTRATION.json> --runs <pulled .../EXP-WS-SEQ-002>
      [--out <SELECTION.json>] [--grid configs/EXP-WS-SEQ-002.json] [--inputs data/ws-seq/inputs2] [--base-inputs data/ws-seq/inputs]
      [--excluded-stop Gxx ...]

Rule (pre-registered): selected = argmin over the registered grid of receipt.best_select_seven_leaf_nll_uncalibrated
(seed 0; recomputed here from the SELECT margins and required to agree within 1e-9); exact ties -> smaller id.
Stop rule: if the selected value is not lower than the registered reference (E1b arm P SELECT NLL), status STOP_RULE,
exit 4, and seeds 1,2 are not run.

What it reads: receipts, curve.json, the SELECT (tuning) members of margins.npz, SELECT labels of corpus-subset.npz,
and file sha256. It never decodes an EVAL margin, an EVAL probability or an EVAL label, and it REFUSES (exit 1) if any
EVAL metric exists: any file in the runs tree that is not a registered output of a seed-0 run directory, any JSON key
path there that names both a held-out role (eval / evaluation / test) and a metric (nll, brier, ece, ...), or any
registered downstream result path (the calibration stage) that already exists.

Exit codes: 0 selected; 1 BLOCKED (gates, EVAL metric present, output exists); 4 stop rule; 6 registered STOP
(receipt / hash / recomputation inconsistency).
"""
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, importlib.util, json, re, time
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
RUN_FILES = ('margins.npz', 'temperatures.json', 'seven-leaf-evaluation.npy', 'curve.json', 'model-state.pt', 'receipt.json')
HELD_OUT_TOKENS = frozenset({'eval', 'evaluation', 'test'})
METRIC_TOKENS = frozenset({'nll', 'brier', 'ece', 'logloss', 'loss', 'accuracy', 'auc', 'reliability', 'metric', 'metrics'})
EXIT_OK, EXIT_BLOCKED, EXIT_STOP_RULE, EXIT_STOP = 0, 1, 4, 6


class Stop(Exception):
    """Registered STOP: no SELECTION.json."""


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
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in pending_fields(v, f'{at}.{k}' if at else k)]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in pending_fields(v, f'{at}[{i}]')]
    return [at] if isinstance(obj, str) and obj.startswith(PENDING) else []


def review_gate(reg, reg_path):
    """Same gate as prepare_inputs2.review_gate (copy). Returns the sha256 of the accepted REVIEW.json."""
    if reg.get('id') != EXP_ID or reg.get('status') != READY:
        raise SystemExit('BLOCKED: registration id / status is not exactly ' + READY)
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


def load_train_module():
    """train_seq2.py by file path: config_hash, load_config, seven_from_margins, nll7 (numpy/scipy only; torch is not imported)."""
    spec = importlib.util.spec_from_file_location('ws_seq_train2', HERE / 'train_seq2.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


# ---------------------------------------------------------------- EVAL-metric refusal
def key_paths(obj, at=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{at}.{k}' if at else str(k)
            yield p
            yield from key_paths(v, p)
    elif isinstance(obj, list):
        for v in obj:
            yield from key_paths(v, at)


def names_held_out_metric(path_text):
    tok = set(t for t in re.split(r'[^a-z0-9]+', path_text.lower()) if t)
    return bool(tok & HELD_OUT_TOKENS) and bool(tok & METRIC_TOKENS)


def eval_metric_findings(runs, allowed_dirs, forbidden_paths):
    """Every reason to believe an EVAL metric already exists. Empty list = clean. File names and JSON KEYS only are
    inspected (values are not read), plus existence of the registered downstream result paths."""
    runs = Path(runs); found = []; scanned = 0
    for p in sorted(runs.rglob('*')):
        if p.is_dir():
            continue
        scanned += 1; rel = p.relative_to(runs)
        if len(rel.parts) != 2 or rel.parts[0] not in allowed_dirs or rel.parts[1] not in RUN_FILES:
            found.append(f'unregistered file in the runs tree: {rel}'); continue
        if p.suffix == '.json':
            try:
                doc = json.loads(p.read_text())
            except ValueError:
                found.append(f'unreadable JSON: {rel}'); continue
            bad = sorted({k for k in key_paths(doc) if names_held_out_metric(k)})
            if bad:
                found.append(f'held-out metric key in {rel}: {bad[:5]}')
        if p.name == 'margins.npz':
            with np.load(p, allow_pickle=False) as z:
                bad = sorted(k for k in z.files if names_held_out_metric(k))
            if bad:
                found.append(f'held-out metric member in {rel}: {bad[:5]}')
    for q in forbidden_paths:
        if Path(q).exists():
            found.append(f'registered downstream result path already exists: {q}')
    return found, scanned


# ---------------------------------------------------------------- selection
def rank(rows):
    """Sorted table: ascending best SELECT NLL, exact ties -> smaller id. rows = [{'id', 'best_select_seven_leaf_nll_uncalibrated', ...}]."""
    return sorted(rows, key=lambda r: (r['best_select_seven_leaf_nll_uncalibrated'], r['id']))


def read_run(T, cfg, entry, d, y7_tune, expected, clip):
    """One seed-0 run directory -> table row. STOP on any binding / recomputation inconsistency."""
    rid = entry['id']; rp = d / 'receipt.json'
    r = json.loads(rp.read_text()); bad = []
    want = dict(expected, config_id=rid, seed=0, config_hash=T.config_hash(cfg, entry), id=EXP_ID, config_entry=entry)
    bad += [k for k, v in want.items() if r.get(k) != v]
    bad += ['outputs_sha256.' + f for f in RUN_FILES[:-1] if not (d / f).is_file() or r.get('outputs_sha256', {}).get(f) != sha(d / f)]
    if bad:
        raise Stop(f'STOP {rid}: receipt binding differs: {bad}')
    names = sorted(f'{role}__{h}' for role in T.ROLES for h in T.HEADS)
    with np.load(d / 'margins.npz', allow_pickle=False) as z:
        if sorted(z.files) != names:
            raise Stop(f'STOP {rid}: margins.npz members')
        zp, zw, zg = (np.array(z['tuning__' + h]) for h in T.HEADS)        # SELECT members only
    if len(zp) != len(y7_tune) or r['rows']['tuning'] != len(y7_tune):
        raise Stop(f'STOP {rid}: SELECT row count')
    try:
        again = T.nll7(T.seven_from_margins(zp, zw, zg), y7_tune, clip)
    except T.Stop as e:
        raise Stop(f'STOP {rid}: {e}')
    curve = json.loads((d / 'curve.json').read_text())['epochs']; best = r['best_select_seven_leaf_nll_uncalibrated']
    if abs(again - best) > 1e-9 or len(curve) != r['epochs_run'] or curve[r['best_epoch'] - 1]['select_seven_leaf_nll_uncalibrated'] != best \
            or min(c['select_seven_leaf_nll_uncalibrated'] for c in curve) != best:
        raise Stop(f'STOP {rid}: best SELECT NLL is not reproduced from the SELECT margins / curve ({again} vs {best})')
    return {'id': rid, 'config_hash': r['config_hash'], 'best_select_seven_leaf_nll_uncalibrated': best, 'recomputed_from_select_margins_abs_diff': abs(again - best),
            'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'], 'early_stopped': r['early_stopped'], 'wall_cap_hit': r['wall_cap_hit'], 'elapsed_seconds': r['elapsed_seconds'],
            'parameters': r['parameters'], 'peak_gpu_memory_bytes': r['peak_gpu_memory_bytes'], 'device': r['device'], 'gpu_uuid': r.get('gpu_uuid'), 'job': r.get('job'),
            'entry': entry, 'receipt_sha256': sha(rp)}


def select(reg, reg_path, args):
    review_sha = review_gate(reg, reg_path); out = Path(args.out) if args.out else Path(reg_path).resolve().parent / 'attempts' / reg['attempt_id'] / 'SELECTION.json'
    if out.exists():
        raise SystemExit('BLOCKED: SELECTION.json exists (a second selection needs a new attempt id and a recorded reason): ' + str(out))
    synthetic = 'synthetic_selfcheck' in reg
    if not synthetic and Path(args.grid).resolve() != CONFIG_JSON.resolve():
        raise SystemExit('BLOCKED: the selection binds only the authoritative repo config ' + str(CONFIG_JSON))
    T = load_train_module(); cfg = T.load_config(args.grid); grid = cfg['grid']; ids = [e['id'] for e in grid]
    hashes = {e['id']: T.config_hash(cfg, e) for e in grid}
    if not synthetic and (hashes != reg['grid']['config_hashes'] or grid != reg['grid']['configs']):
        raise SystemExit('BLOCKED: grid / config hashes differ from the registration')
    if cfg['selection'] != reg['selection']['config_block'] and not synthetic:
        raise SystemExit('BLOCKED: selection block of the config differs from the registration')
    excluded = sorted(set(args.excluded_stop or []))
    if any(x not in ids for x in excluded):
        raise SystemExit('BLOCKED: --excluded-stop names an unregistered config id')
    runs = Path(args.runs); allowed = {f'{i}-s0' for i in ids}
    findings, scanned = eval_metric_findings(runs, allowed, reg['selection']['eval_result_paths_that_must_not_exist'])
    if findings:
        raise SystemExit('BLOCKED: an EVAL metric (or an unregistered file) exists; selection must precede every EVAL read:\n  ' + '\n  '.join(findings))
    for i in ids:
        has = (runs / f'{i}-s0' / 'receipt.json').is_file()
        if i in excluded and has:
            raise SystemExit(f'BLOCKED: {i} is listed in --excluded-stop but has a receipt')
        if i not in excluded and not has:
            raise SystemExit(f'BLOCKED: seed-0 receipt missing for {i} (all registered configs are required; a run that ended in a registered STOP must be named with --excluded-stop)')
    man_path = Path(args.inputs) / cfg['inputs']['manifest_file']; corpus_path = Path(args.base_inputs) / cfg['inputs']['corpus_file']
    if not man_path.is_file() or not corpus_path.is_file():
        raise SystemExit('BLOCKED: local inputs2 manifest or base corpus-subset.npz missing')
    man = json.loads(man_path.read_text())
    if sha(corpus_path) != man['base_files'][cfg['inputs']['corpus_file']]['sha256'] or man.get('registration_sha256') != sha(reg_path) or man.get('review_sha256') != review_sha:
        raise Stop('STOP local corpus-subset.npz / registration / review differ from the inputs2 manifest binding')
    expected = {'manifest_sha256': sha(man_path), 'registration_sha256': sha(reg_path), 'review_sha256': review_sha, 'train_seq2_sha256': sha(HERE / 'train_seq2.py'), 'config_sha256': sha(args.grid)}
    with np.load(corpus_path, allow_pickle=False) as z:
        y7_tune = np.array(z['y7'])[np.array(z['tuning'])]                   # SELECT labels only
    rows = [read_run(T, cfg, e, runs / f'{e["id"]}-s0', y7_tune, expected, cfg['nll_clip']) for e in grid if e['id'] not in excluded]
    if not rows:
        raise Stop('STOP no config left to select from')
    table = rank(rows); best = table[0]; ref = reg['selection']['stop_rule_reference_select_nll']
    for k, r in enumerate(table):
        r['rank'] = k + 1; r['minus_selected'] = r['best_select_seven_leaf_nll_uncalibrated'] - best['best_select_seven_leaf_nll_uncalibrated']
    triggered = not best['best_select_seven_leaf_nll_uncalibrated'] < ref
    by_id = {r['id']: r for r in table}; rep = reg['selection']['report_only_references']
    cmd = lambda s: (f'cd {REPO} && /Users/song/.local/bin/gpu run -n ws-seq2-{best["id"]}-s{s} -g <card> -- python experiments/ws_seq/train_seq2.py '
                     f'--config-id {best["id"]} --seed {s} --selected-config-hash {best["config_hash"]}')
    doc = {'id': EXP_ID, 'attempt_id': reg['attempt_id'], 'synthetic': synthetic, 'status': 'STOP_RULE' if triggered else 'SELECTED',
           'label': '2026 EXPOSED DEVELOPMENT; selection on SELECT only (seed 0, uncalibrated seven-leaf NLL); no EVAL quantity was read; not a winner or adoption',
           'rule': reg['selection']['rule'], 'selected': {'id': best['id'], 'config_hash': best['config_hash'], 'entry': best['entry'],
                                                          'best_select_seven_leaf_nll_uncalibrated': best['best_select_seven_leaf_nll_uncalibrated'], 'seed_0_run_reused': f'{best["id"]}-s0'},
           'table': table, 'excluded_registered_stop': excluded,
           'stop_rule': {'reference_select_nll': ref, 'triggered': bool(triggered), 'rule': reg['selection']['stop_rule']},
           'report_only': {'note': 'SELECT quantities only; descriptive', 'selected_minus_reference': best['best_select_seven_leaf_nll_uncalibrated'] - ref,
                           'selected_minus_G01': (best['best_select_seven_leaf_nll_uncalibrated'] - by_id['G01']['best_select_seven_leaf_nll_uncalibrated']) if 'G01' in by_id else None,
                           'selected_minus_seq001_N1_seed0': best['best_select_seven_leaf_nll_uncalibrated'] - rep['seq001_N1_seed0_select_nll'], 'references': rep},
           'refusal_scan': {'what': 'unregistered files, held-out metric key names, downstream result paths', 'files_scanned': scanned, 'findings': [], 'forbidden_paths_checked': reg['selection']['eval_result_paths_that_must_not_exist']},
           'bindings': dict(expected, sources_sha256=own_hashes(reg_path), runs=str(runs)),
           'next_commands': [] if triggered else [cmd(1), cmd(2)], 'scientific_decision': None, 'adoption': False, 'utc': time.time()}
    out.parent.mkdir(parents=True, exist_ok=True); save_json(out, doc)
    print(json.dumps({'status': doc['status'], 'selected': doc['selected']['id'], 'config_hash': best['config_hash'], 'best_select_nll': best['best_select_seven_leaf_nll_uncalibrated'],
                      'table': [[r['id'], r['best_select_seven_leaf_nll_uncalibrated']] for r in table], 'out': str(out), 'next_commands': doc['next_commands']}, indent=1))
    return EXIT_STOP_RULE if triggered else EXIT_OK


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--registration', required=True); ap.add_argument('--runs', required=True); ap.add_argument('--out', default=None)
    ap.add_argument('--grid', default=str(CONFIG_JSON)); ap.add_argument('--inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs2')); ap.add_argument('--base-inputs', default=str(REPO / 'data' / 'ws-seq' / 'inputs'))
    ap.add_argument('--excluded-stop', nargs='*', default=[])
    args = ap.parse_args(); reg = json.loads(Path(args.registration).read_text())
    try:
        return select(reg, args.registration, args)
    except SystemExit as e:
        print(str(e), file=sys.stderr); return EXIT_BLOCKED
    except Stop as e:
        print(str(e), file=sys.stderr); return EXIT_STOP


if __name__ == '__main__':
    sys.exit(main())
