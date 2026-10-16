"""Gate P of the sequence-model shadow: do the three N1 networks, loaded on this Mac (CPU, fp32, one thread) from the
hash-pinned ``model-state.pt`` files, reproduce the margins the A5000 server wrote for the CALIBRATION-role rows?

Written by the author; the REAL comparison is run by the lead only, inside the registered run. It reads, for the
calibration rows only (corpus ``calibration`` mask; no feature row, margin or label of another role is indexed,
compared or scored; the small integer .npz members are decoded whole to select the calibration games):
  data/ws-seq/inputs/{manifest.json, corpus-subset.npz, seq-inputs.npz} and x-prequential.npy (memory map),
  RUNS/N1-s{seed}/margins.npz members calibration__parent / calibration__narrow_whiff /
  calibration__ground_ball_per_pitch.
Every file is sha256-checked first (manifest pin from the config, the three inputs against the manifest, the margins
and the model files against the config); any mismatch refuses (rc 2) before an array is decoded.

Tokens are rebuilt exactly as in training: ``train_seq.build_context_index`` over every row of the games that hold a calibration row (context never crosses a
game, so this equals the index over the whole package) and
``train_seq.Data.tokens`` (the pinned ``train_seq.py``). Reported per seed and head: max abs difference, mean abs
difference, fraction of rows within ``parity.max_abs_tolerance``; and the mean seven-leaf NLL of the C0
probabilities on the calibration rows from the Mac margins and from the server margins (two independent code
paths: ``pitcheezy.seq_shadow.calibration.probs_c0`` vs ``train_seq.seven_from_margins`` + ``nll7``).

  python -B scripts/seq_shadow_parity.py --out DIR [--rows N] [--allow-concurrent-with-protected]
  python -B scripts/seq_shadow_parity.py --synthetic --out DIR      (self-test: fabricated tiny package)

Exit codes: 0 PARITY.json written and every head of every seed within tolerance; 1 written, some head outside
tolerance; 2 refused (nothing written); 3 stopped by a cap.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
              'MKL_NUM_THREADS'):
    os.environ[_name] = '1'  # before any numeric import

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO/'src') not in sys.path:
    sys.path.insert(0, str(REPO/'src'))

import pitcheezy.seq_shadow as SS  # noqa: E402

HEADS = ('parent', 'narrow_whiff', 'ground_ball_per_pitch')
RC_OK, RC_OUTSIDE, RC_REFUSED, RC_CAP = 0, 1, 2, 3


def margins_of(torch, T, net, x, C, S, ctx, rows, batch, check=lambda: None):
    """Margins {head: float64 array} of ``net`` for the package row indices ``rows`` (ascending). ``x`` holds ONLY
    those rows (same order); tokens come from ``Data.tokens`` over the full package index."""
    import numpy as np
    data = T.Data(torch, torch.device('cpu'), x, C, S, ctx)
    index = torch.as_tensor(np.asarray(rows), dtype=torch.int64)
    parts = {h: [] for h in HEADS}
    net.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            check()
            zp, zw, zg = net(data.x[start:start+batch], data.tokens(index[start:start+batch]))
            for h, z in zip(HEADS, (zp, zw, zg)):
                parts[h].append(z.double().numpy())
    out = {h: np.concatenate(parts[h]) for h in HEADS}
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ValueError('non-finite margin')
    return out


def compare(mine, server, tolerance):
    """Per head: max / mean abs difference and the fraction of rows whose largest abs difference is within
    ``tolerance``."""
    import numpy as np
    out = {}
    for h in HEADS:
        a, b = np.asarray(mine[h], dtype=np.float64), np.asarray(server[h], dtype=np.float64)
        if a.shape != b.shape:
            raise ValueError(f'margin shape differs for {h}: {a.shape} vs {b.shape}')
        d = np.abs(a-b)
        per_row = d.reshape(len(d), -1).max(1) if len(d) else d.reshape(0)
        out[h] = {'rows': int(len(d)), 'max_abs_difference': float(d.max()) if d.size else None,
                  'mean_abs_difference': float(d.mean()) if d.size else None,
                  'fraction_rows_within_tolerance': float((per_row <= tolerance).mean()) if len(d) else None,
                  'within_tolerance': bool(d.size == 0 or d.max() <= tolerance)}
    return out


def nll_two_ways(T, mine, server, temperatures, y7, clip, tolerance):
    """Mean C0 seven-leaf NLL on the given rows from the Mac margins (this job's ``probs_c0``) and from the server
    margins (``train_seq.seven_from_margins`` + ``nll7``): independent code, same labels."""
    import numpy as np
    from pitcheezy.seq_shadow.calibration import probs_c0
    q = probs_c0({h: np.asarray(mine[h], dtype=np.float64) for h in HEADS}, temperatures, tolerance)
    mac = float(np.mean(-np.log(np.clip(q[np.arange(len(y7)), y7], clip, 1))))
    ref = T.nll7(T.seven_from_margins(server['parent'], server['narrow_whiff'], server['ground_ball_per_pitch'],
                                      temperatures), y7, clip)
    return {'mac_margins_this_job_probs_c0': mac, 'server_margins_train_seq_seven_from_margins': float(ref),
            'abs_difference': abs(mac-float(ref)), 'rows': int(len(y7)), 'clip': clip}


def run_package(config, T, torch, nets, temperatures, inputs_dir, margins_paths, *, rows_cap, check):
    """The comparison over one input package. ``nets`` / ``temperatures`` / ``margins_paths``: {seed: ...}."""
    import numpy as np
    p, arch = config['parity'], json.loads(Path(config['model']['architecture']['path']).read_text())
    names = arch['inputs']
    with np.load(Path(inputs_dir)/names['corpus_file'], allow_pickle=False) as z:
        C = {k: np.array(z[k]) for k in ('y10', 'y7', 'narrow_whiff', 'ground_ball_per_pitch', p['role'])}
    with np.load(Path(inputs_dir)/names['sequence_file'], allow_pickle=False) as z:
        S = {k: np.array(z[k]) for k in T.SEQ_MEMBERS}
    total = len(C['y10'])
    if any(len(v) != total for v in list(C.values())+list(S.values())):
        raise ValueError('package row counts differ')
    role_rows = np.flatnonzero(C[p['role']])
    # Context = earlier pitches of the same pitcher in the SAME GAME, so only the games that hold a calibration row
    # are needed: the index over that subset equals the index over the whole package for these rows (and it keeps
    # the (rows, L) index small). Order is preserved, so subset positions map back to package rows.
    subset = np.flatnonzero(np.isin(S['game_pk'], np.unique(S['game_pk'][role_rows])))
    C = {k: v[subset] for k, v in C.items()}
    S = {k: v[subset] for k, v in S.items()}
    ctx = T.build_context_index(S['pitcher'], S['game_pk'], S['at_bat_number'], S['pitch_number'],
                                arch['context']['max_length'])
    T.check_context(ctx, S['pitcher'], S['game_pk'], S['at_bat_number'], S['pitch_number'])
    keep = len(role_rows) if rows_cap is None else min(rows_cap, len(role_rows))
    rows = np.flatnonzero(C[p['role']])[:keep]  # positions inside the subset
    if not np.array_equal(subset[rows], role_rows[:keep]):
        raise ValueError('subset positions do not map back to the calibration rows')
    x_all = np.load(Path(inputs_dir)/names['features_file'], mmap_mode='r', allow_pickle=False)
    if x_all.shape != (total, names['feature_dimension']) or x_all.dtype != np.float32:
        raise ValueError(f'feature matrix shape/dtype: {x_all.shape} {x_all.dtype}')
    x = np.ascontiguousarray(x_all[role_rows[:keep]])  # only the calibration rows are ever materialised
    del x_all
    if not np.isfinite(x).all():
        raise ValueError('non-finite feature in the calibration rows')
    y7 = C['y7'][rows]
    out = {'role': p['role'], 'role_rows': int(len(role_rows)), 'rows_compared': int(keep), 'batch': p['batch'],
           'context_index_rows': int(len(subset)), 'package_rows': int(total),
           'tolerance': p['max_abs_tolerance'], 'seeds': {}}
    for seed, net in nets.items():
        check()
        t0 = time.perf_counter()
        mine = margins_of(torch, T, net, x, C, S, ctx, rows, p['batch'], check)
        with np.load(margins_paths[seed], allow_pickle=False) as z:
            server = {h: np.array(z[f"{p['role']}__{h}"])[:keep] for h in HEADS}
            stored = int(len(z[f"{p['role']}__parent"]))
        if stored != len(role_rows):
            raise ValueError('server margins row count differs from the calibration rows of the package')
        heads = compare(mine, server, p['max_abs_tolerance'])
        out['seeds'][str(seed)] = {'heads': heads, 'within_tolerance': all(v['within_tolerance'] for v in heads.values()),
                                   'c0_nll': nll_two_ways(T, mine, server, temperatures[seed], y7, p['nll_clip'],
                                                          config['calibration']['row_sum_tolerance']),
                                   'forward_seconds': time.perf_counter()-t0}
    out['all_within_tolerance'] = all(v['within_tolerance'] for v in out['seeds'].values())
    return out


def synthetic(config, out):
    """Self-test of the comparison machinery on a fabricated tiny package: identical networks -> difference 0;
    one perturbed weight -> difference > 0 and outside tolerance. No registered file but the architecture config
    and the pinned train_seq.py is read."""
    import numpy as np
    import torch
    from pitcheezy.seq_shadow import model as MOD
    m = config['model']
    T = MOD._train_module(Path(m['train_seq']['path']), SS.read_pinned(m['train_seq']['path'], m['train_seq']['sha256']))
    arch = json.loads(SS.read_pinned(m['architecture']['path'], m['architecture']['sha256']))
    torch.set_num_threads(config['threads'])
    rng = np.random.default_rng(0)
    n, d, role = 600, arch['inputs']['feature_dimension'], config['parity']['role']
    v = arch['context']
    S = {'game_pk': np.repeat(np.arange(6), 100).astype(np.int32), 'at_bat_number': np.tile(np.repeat(np.arange(1, 21), 5), 6).astype(np.int16),
         'pitch_number': np.tile(np.arange(1, 6), 120).astype(np.int16), 'pitcher': np.tile(np.repeat([7, 8], 50), 6).astype(np.int32),
         'batter': rng.integers(100, 110, n).astype(np.int32), 'pitch_type': rng.integers(0, v['pitch_type_vocab'], n).astype(np.int16),
         'outcome10': rng.integers(0, v['outcome_vocab'], n).astype(np.int16), 'balls': rng.integers(0, v['balls_vocab'], n).astype(np.int16),
         'strikes': rng.integers(0, v['strikes_vocab'], n).astype(np.int16), 'stand': rng.integers(0, 2, n).astype(np.int16)}
    y10 = S['outcome10'].astype(np.int64)
    whiff, gb = rng.integers(0, 2, n).astype(np.int8), rng.integers(0, 2, n).astype(np.int8)
    C = {'y10': y10, 'y7': T.seven_label(y10, whiff, gb), 'narrow_whiff': whiff, 'ground_ball_per_pitch': gb,
         role: (S['game_pk'] % 2 == 1)}
    x = rng.normal(size=(n, d)).astype(np.float32)
    work = Path(out)/'synthetic-package'
    work.mkdir(parents=True)
    names = arch['inputs']
    np.save(work/names['features_file'], x, allow_pickle=False)
    np.savez(work/names['corpus_file'], **C)
    np.savez(work/names['sequence_file'], **S)
    torch.manual_seed(0)
    net = T.build_model(torch, arch, m['arm'])
    net.eval()
    ctx = T.build_context_index(S['pitcher'], S['game_pk'], S['at_bat_number'], S['pitch_number'], v['max_length'])
    rows = np.flatnonzero(C[role])
    z = margins_of(torch, T, net, x[rows], C, S, ctx, rows, 128)
    np.savez(work/'margins.npz', **{f'{role}__{h}': z[h] for h in HEADS})
    temperatures = {0: {h: 1.0 for h in HEADS}}
    cfg = json.loads(json.dumps(config))
    cfg['parity']['batch'] = 128
    same = run_package(cfg, T, torch, {0: net}, temperatures, work, {0: work/'margins.npz'}, rows_cap=None, check=lambda: None)
    with torch.no_grad():
        net.head_parent.bias[0] += 1.0
    moved = run_package(cfg, T, torch, {0: net}, temperatures, work, {0: work/'margins.npz'}, rows_cap=77, check=lambda: None)
    s, mv = same['seeds']['0'], moved['seeds']['0']
    checks = {'identical_network_difference_is_zero': all(s['heads'][h]['max_abs_difference'] == 0.0 for h in HEADS),
              'identical_network_within_tolerance': same['all_within_tolerance'] is True,
              'identical_network_nll_agrees': s['c0_nll']['abs_difference'] < 1e-12,
              'perturbed_weight_is_detected': mv['heads']['parent']['max_abs_difference'] > config['parity']['max_abs_tolerance']
              and moved['all_within_tolerance'] is False and mv['heads']['parent']['fraction_rows_within_tolerance'] == 0.0,
              'perturbed_other_heads_unchanged': mv['heads']['narrow_whiff']['max_abs_difference'] == 0.0,
              'row_cap_respected': moved['rows_compared'] == 77 and same['rows_compared'] == int(C[role].sum()),
              'only_role_rows_compared': same['rows_compared'] == 300}
    return {'schema': 'seq-shadow-parity-v1', 'mode': 'synthetic_selftest', 'checks': checks,
            'verdict': 'SELFTEST_PASS' if all(checks.values()) else 'SELFTEST_FAIL', 'identical': same, 'perturbed': moved}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Gate P: Mac CPU forward vs server margins on calibration rows')
    parser.add_argument('--out', required=True, help='output directory (must not exist); PARITY.json is written there')
    parser.add_argument('--config', default=str(SS.CONFIG_PATH))
    parser.add_argument('--rows', type=int, help='cap on the number of calibration rows (default: all)')
    parser.add_argument('--synthetic', action='store_true', help='self-test on a fabricated tiny package')
    parser.add_argument('--allow-concurrent-with-protected', action='store_true',
                        help='real mode: run although registered protected processes are alive (lead decision)')
    args = parser.parse_args(argv)
    Refused = Stop = None
    created = False
    try:
        config = SS.load_config(args.config)
        out = Path(args.out).resolve()
        if out.exists():
            raise ValueError('output_dir_exists')
        if [w for w in config['forbidden_path_parts'] if w in out.parts]:
            raise ValueError('path_inside_a_protected_directory')
        if args.rows is not None and args.rows < 1:
            raise ValueError('rows_must_be_positive')
        if args.synthetic:
            out.mkdir(parents=False)
            created = True
            result = synthetic(config, out)
            with open(out/'PARITY.json', 'x') as handle:
                json.dump(result, handle, indent=1)
            print(json.dumps({'verdict': result['verdict'], 'checks': result['checks']}, indent=1))
            return RC_OK if result['verdict'] == 'SELFTEST_PASS' else RC_OUTSIDE
        if sys.executable != config['runtime_python']:
            raise ValueError('interpreter_is_not_the_registered_runtime')
        if Path(args.config).resolve() != SS.CONFIG_PATH.resolve():
            raise ValueError('a_real_run_uses_the_registered_config_file')
        mods = SS.load_step5_shadow(config)  # the pinned guard: memory floor, protected processes, caps
        Refused, Stop = mods.G.Refused, mods.G.Stop
        guard = mods.G.Guard(config, allow_concurrent=args.allow_concurrent_with_protected, acquire_heavy_lock=False,
                             stop_marker=str(out)+'.RSS-STOP.json')
        guard.preflight()
        p, arch_spec = config['parity'], config['model']['architecture']
        manifest = json.loads(SS.read_pinned(Path(p['inputs_dir'])/'manifest.json', p['manifest_sha256']))
        arch = json.loads(SS.read_pinned(arch_spec['path'], arch_spec['sha256']))
        names = arch['inputs']
        if manifest['bound_sources']['train_seq.py'] != config['model']['train_seq']['sha256'] \
                or manifest['bound_sources']['config'] != arch_spec['sha256']:
            raise ValueError('input package is bound to another train_seq.py or architecture config')
        seen = {}
        for name in (names['features_file'], names['corpus_file'], names['sequence_file']):
            seen[name] = SS.file_sha256(Path(p['inputs_dir'])/name)
            if seen[name] != manifest['files'][name]['sha256']:
                raise ValueError('input sha256 differs from the manifest: '+name)
        seeds = [int(s) for s in config['model']['seeds']]
        for s in seeds:
            if SS.file_sha256(p['margins'][str(s)]['path']) != p['margins'][str(s)]['sha256']:
                raise ValueError(f'server margins sha256 differs from the pin: seed {s}')
        from pitcheezy.seq_shadow import model as MOD
        ensemble = MOD.load_ensemble(config)  # hash-checks model-state / temperatures / architecture / train_seq
        T = sys.modules[MOD.TRAIN_MODULE_NAME]
        out.mkdir(parents=False)
        created = True
        guard.start()
        try:
            result = run_package(config, T, ensemble.torch, dict(zip(ensemble.seeds, ensemble.nets)),
                                 ensemble.temperatures, p['inputs_dir'],
                                 {s: p['margins'][str(s)]['path'] for s in seeds}, rows_cap=args.rows, check=guard.check)
        except Stop as exc:
            receipt = guard.finish()
            with open(out/'PARITY.json', 'x') as handle:
                json.dump({'schema': 'seq-shadow-parity-v1', 'mode': 'actual', 'status': 'STOPPED_BY_CAP',
                           'stop': str(exc), 'guard': receipt}, handle, indent=1)
            print(json.dumps({'stopped': str(exc)}))
            return RC_CAP
        except BaseException:
            guard.finish()
            raise
        result = {'schema': 'seq-shadow-parity-v1', 'mode': 'actual', 'status': 'COMPLETED'} | result | {
            'sha256': {'inputs': seen, 'manifest': p['manifest_sha256'], 'model': ensemble.sha256,
                       'margins': {str(s): p['margins'][str(s)]['sha256'] for s in seeds},
                       'config': SS.file_sha256(args.config), 'own_source': SS.own_source_sha256()},
            'interpreter': sys.executable, 'torch': ensemble.torch.__version__,
            'torch_num_threads': ensemble.torch.get_num_threads(), 'evaluation_rows_compared_or_scored': 0,
            'note': 'the corpus and sequence .npz members are decoded whole to select the calibration games; no feature '
                    'row, margin or label outside the calibration role is indexed, compared or scored',
            'claims': 'numerical parity of the loaded networks on calibration rows only; not a performance result',
            'guard': guard.finish()}
        with open(out/'PARITY.json', 'x') as handle:
            json.dump(result, handle, indent=1)
        print(json.dumps({k: result[k] for k in ('status', 'rows_compared', 'tolerance', 'all_within_tolerance', 'seeds')}, indent=1))
        return RC_OK if result['all_within_tolerance'] else RC_OUTSIDE
    except (ValueError, OSError, KeyError) as exc:
        if created:  # an error after the output directory was made is a crash, never reported as a refusal
            raise
        print(json.dumps({'refused': f'{type(exc).__name__}: {exc}'}))
        return RC_REFUSED
    except Exception as exc:
        if Refused is not None and isinstance(exc, Refused):
            print(json.dumps({'refused': str(exc)}))
            return RC_REFUSED
        raise


if __name__ == '__main__':
    sys.exit(main())
