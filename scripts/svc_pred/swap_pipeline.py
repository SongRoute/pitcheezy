"""EXP-SVC-PRED-001 swap loop in one command: final-mix.json -> verified display watch for 849799 + bundle folder.

    scripts/svc_pred/swap_pipeline.sh <final-mix.json> [--name NAME] [--ref NAME=WATCH ...]

Steps (each timed, outputs in <bundle-root>/<name>-<UTC>/):
  1 pins (from_final_mix pins) -> commit       2 gate X1/F1/M3 -> M0/M4 (from_final_mix gate) -> commit receipt
  3 instance (build_pred new-instance, REGISTERED + execute) -> commit
  4 849799 predict -> predictions sha -> commit   5 watch build (r3r1 + ring8)   6 check   7 compare vs zo-ring8 (+ ledger rows)
  8 compare vs --ref watches                     9 E_PS mix npz (fan_numbers --pred format, no labels)
 10 fan_numbers (E_PS already scored; descriptive) 11 booth lines to change by hand + revert lines (safety net)
Independent (Fable) verification is a separate step. This script never edits booth files (start_booth.sh, DAY-OF-1124.md) and never
touches booth processes or ports. Commits only the files it creates, with explicit paths; never pushes.
"""
from __future__ import annotations

import os
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
PY = pathlib.Path(sys.executable)
JOB = pathlib.Path(os.path.expandvars('${PITCHEEZY_JOBS}/20261121-svc-gru'))
FAN = REPO / 'scripts/fan_numbers.py'  # public repo: moved in from the private job folder
BOOTH = pathlib.Path(os.path.expandvars('${PITCHEEZY_JOBS}/20261119-loop'))
SAFETY = pathlib.Path(os.path.expandvars('${PITCHEEZY_JOBS}/20261121-safety-net'))
SAFETY_WATCH = 'watch/r4-849799.json'           # = start_booth.sh WATCHPOL_FIXED at the time the safety net was made
T7_RUNS = pathlib.Path(os.path.expandvars('${PITCHEEZY_RUNS}'))
SAFETY_WATCH_T7 = T7_RUNS / 'DEMO-WS-2026/R4-RETRAINED-1119/v4/c_all/r4/watch/849799.json'
CFG = REPO / 'configs/EXP-SVC-PRED-001'
KST = dt.timezone(dt.timedelta(hours=9))
NO_RUN = [((2026, 11, 22), (19, 30), (22, 10), '11/22 19:30~22:10 맥 무거운 작업 금지'),
          ((2026, 11, 24), (0, 0), (23, 59), '11/24 부스 운영일')]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(pathlib.Path(p).resolve().relative_to(REPO.resolve()))


class Run:
    def __init__(self, out):
        self.out, self.steps = out, []

    def step(self, name, fn):
        t0 = time.time(); start = dt.datetime.now(KST).strftime('%H:%M:%S')
        print(f'[{start}] {name} ...', flush=True)
        try:
            r = fn()
        except BaseException as ex:
            self.steps.append({'step': name, 'start_kst': start, 'seconds': round(time.time() - t0, 2), 'ok': False, 'error': str(ex)})
            self.save(); raise
        self.steps.append({'step': name, 'start_kst': start, 'seconds': round(time.time() - t0, 2), 'ok': True})
        print(f'    {self.steps[-1]["seconds"]} s', flush=True)
        self.save()
        return r

    def save(self):
        (self.out / 'timings.json').write_text(json.dumps(self.steps, indent=1, ensure_ascii=False) + '\n')


def sh(*args, capture=True):
    p = subprocess.run([str(a) for a in args], cwd=REPO, capture_output=capture, text=True)
    if p.returncode:
        raise SystemExit(f'FAILED ({p.returncode}): {" ".join(map(str, args))}\n{(p.stdout or "")[-2000:]}\n{(p.stderr or "")[-2000:]}')
    return p.stdout


def last_json(text):
    i = text.rfind('\n{'); i = 0 if i < 0 and text.startswith('{') else i + 1
    return json.loads(text[i:])


def commit(paths, msg):
    paths = [rel(p) for p in paths]
    sh('git', 'add', '--', *paths)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode == 0:
        return None
    sh('git', 'commit', '-q', '-m', f'{msg}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>', '--', *paths)
    return sh('git', 'rev-parse', 'HEAD').strip()


def admission(allow_window):
    now = dt.datetime.now(KST)
    for (y, m, d), (h0, m0), (h1, m1), why in NO_RUN:
        a = dt.datetime(y, m, d, h0, m0, tzinfo=KST); b = dt.datetime(y, m, d, h1, m1, tzinfo=KST)
        if a <= now <= b and not allow_window:
            raise SystemExit(f'refused: {why} (now {now:%m/%d %H:%M})')
    free = shutil.disk_usage('/').free / 2 ** 30
    if free <= 10:
        raise SystemExit(f'refused: internal disk free {free:.1f} GiB <= 10')
    dirty = [l for l in sh('git', 'status', '--porcelain').splitlines() if not l.startswith('??')]
    if dirty:
        raise SystemExit(f'refused: worktree has uncommitted changes: {dirty[:5]}')
    if not T7_RUNS.is_dir():
        raise SystemExit('refused: T7 not mounted')
    return {'now_kst': now.isoformat(timespec='seconds'), 'disk_free_gib': round(free, 1), 'head': sh('git', 'rev-parse', 'HEAD').strip()}


def ledger_check(new_watch, base_watch):
    """Only recommendation content may differ: header row equal; decision rows equal except result.candidate / result.rho_candidate
    and the hash-chain fields (prev, sha256)."""
    def rows(w):
        rd = pathlib.Path(json.loads(pathlib.Path(w).read_text())['paths']['run_dir'])
        return [json.loads(l) for l in open(rd / 'ledger.jsonl')]
    a, b = rows(base_watch), rows(new_watch)
    if len(a) != len(b):
        raise SystemExit(f'ledger rows {len(a)} vs {len(b)}')

    def strip(r):
        r = json.loads(json.dumps(r)); r.pop('prev', None); r.pop('sha256', None)
        if isinstance(r.get('result'), dict):
            r['result'].pop('candidate', None); r['result'].pop('rho_candidate', None)
        return r
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if strip(x) != strip(y)]
    changed = sum(1 for x, y in zip(a, b) if x != y and x.get('kind') == 'decision')
    head_ok = a[0] == b[0] if a and a[0].get('kind') == 'header' else None
    nl = json.loads(pathlib.Path(new_watch).read_text())['ledger']
    ok = not bad and head_ok is not False and b[-1].get('sha256') == nl['ledger_head_sha256']
    return {'rows': len(a), 'decision_rows_changed': changed, 'rows_differing_outside_candidate': bad[:20], 'header_equal': head_ok,
            'last_row_sha_is_head': b[-1].get('sha256') == nl['ledger_head_sha256'], 'pass': bool(ok)}


def booth_lines(new_watch_t7, new_identity):
    """Lines PM changes by hand, current values (read-only) and the safety-net revert values."""
    sb = (BOOTH / 'start_booth.sh').read_text().splitlines()
    cur = {k: next(((i + 1, l) for i, l in enumerate(sb) if l.startswith(f'{k}=')), None) for k in ('WATCHPOL_FIXED', 'SCREEN_FIXED')}
    day = (BOOTH / 'DAY-OF-1124.md').read_text().splitlines()
    dline = next(((i + 1, l) for i, l in enumerate(day) if 'preflight_1124.sh 8876' in l and '정책 id' in l), None)
    sums = dict(reversed(l.split(None, 1)) for l in (SAFETY / 'SHA256SUMS').read_text().splitlines() if l.strip())
    safe_sha = sums.get(f'./{SAFETY_WATCH}') or sums.get(SAFETY_WATCH)
    if safe_sha is None or sha(SAFETY / SAFETY_WATCH) != safe_sha:
        raise SystemExit('safety-net watch missing or differs from its SHA256SUMS')
    if sha(SAFETY_WATCH_T7) != safe_sha:
        raise SystemExit('safety-net T7 original differs from the safety-net copy')
    safe_id = json.loads(SAFETY_WATCH_T7.read_text())['policy']['identity_sha256'][:8]
    return {'start_booth': str(BOOTH / 'start_booth.sh'), 'day_of': str(BOOTH / 'DAY-OF-1124.md'),
            'current': {k: (v and {'line': v[0], 'text': v[1]}) for k, v in cur.items()} | {'DAY-OF': dline and {'line': dline[0], 'text': dline[1]}},
            'swap': {'WATCHPOL_FIXED': f'WATCHPOL_FIXED="{new_watch_t7}"', 'policy_id': new_identity[:8],
                     'SCREEN_FIXED': 'unchanged by this pipeline (the new watch is ring8: the screen must draw zone-outside cells; screen choice is the booth screen session / PM)'},
            'revert': {'WATCHPOL_FIXED': f'WATCHPOL_FIXED="{SAFETY_WATCH_T7}"', 'policy_id': safe_id, 'watch_sha256': safe_sha,
                       'safety_copy': str(SAFETY / SAFETY_WATCH)}}


def booth_text(b):
    c = b['current']; s = b['swap']; r = b['revert']
    L = ['# 부스 설정 — PM이 손으로 바꾼다 (이 파이프라인은 부스 파일을 고치지 않는다)', '',
         f'파일: {b["start_booth"]}']
    for k in ('WATCHPOL_FIXED', 'SCREEN_FIXED'):
        if c[k]:
            L.append(f'- 현재 {c[k]["line"]}행: {c[k]["text"]}')
    L += ['', '## 교체', f'- start_booth.sh {c["WATCHPOL_FIXED"]["line"] if c["WATCHPOL_FIXED"] else "?"}행 →', f'  {s["WATCHPOL_FIXED"]}',
          f'- SCREEN_FIXED: {s["SCREEN_FIXED"]}',
          f'- {b["day_of"]} {c["DAY-OF"]["line"] if c["DAY-OF"] else "?"}행의 기대 정책 id `be5385cf` 자리 → `{s["policy_id"]}`',
          f'- 확인: `grep -n FIXED= {b["start_booth"]}` 후 `./preflight_1124.sh 8876`의 "watch 정책 id" INFO가 {s["policy_id"]}',
          '', '## 되돌리기 (보관본 20261121-safety-net)',
          f'- start_booth.sh WATCHPOL_FIXED 줄 →', f'  {r["WATCHPOL_FIXED"]}',
          f'  (파일 sha256 {r["watch_sha256"][:8]}, 보관본 사본 {r["safety_copy"]}와 같음; T7이 없으면 사본 경로를 써도 된다)',
          f'- DAY-OF 기대 정책 id → `{r["policy_id"]}`', '- SCREEN_FIXED를 바꿨다면 바꾸기 전 줄로 되돌린다(위 "현재" 줄).',
          f'- 확인: `cd {SAFETY} && shasum -a 256 -c SHA256SUMS`, 그다음 preflight INFO가 {r["policy_id"]}']
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('final_mix', type=pathlib.Path)
    ap.add_argument('--name', help='instance name (default: final-mix name)')
    ap.add_argument('--ref', action='append', default=[], help='NAME=WATCH extra paired compare (e.g. GM-w)')
    ap.add_argument('--bundle-root', type=pathlib.Path, default=JOB / 'swap')
    ap.add_argument('--allow-window', action='store_true', help='skip the no-run window check (never for 11/22 19:30~22:10)')
    a = ap.parse_args(argv)
    fm_path = a.final_mix.resolve(); fm = json.loads(fm_path.read_text())
    name = a.name or fm['name']
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = a.bundle_root / f'{name}-{stamp}'; out.mkdir(parents=True)
    run = Run(out)
    ctx = {'final_mix': str(fm_path), 'final_mix_sha256': sha(fm_path), 'name': name, 'bundle': str(out)}
    ctx['admission'] = run.step('0 admission', lambda: admission(a.allow_window))
    fmm = [str(REPO / 'scripts/svc_pred/from_final_mix.py')]
    bp = [str(REPO / 'scripts/svc_pred/build_pred.py')]

    def s_pins():
        r = last_json(sh(PY, '-B', *fmm, 'pins', '--final-mix', fm_path, '--name', name))
        mix = json.loads((REPO / r['mix_pin']).read_text())
        files = [REPO / r['mix_pin']] + [REPO / c['config'] for c in mix['components'] if c['config'].startswith('configs/EXP-SVC-PRED-001/pins/')]
        r['commit'] = commit(files, f'exp(EXP-SVC-PRED-001): pins {name} (swap_pipeline)')
        return r
    ctx['pins'] = run.step('1 pins', s_pins)

    def s_gate():
        eps = []
        for c in fm['components']:
            if c['name'].startswith('X-'):
                eps += ['--eps-pred', f"{c['name']}={c['eps_pred']['path']}"]
        r = last_json(sh(PY, '-B', *fmm, 'gate', '--final-mix', fm_path, '--name', name, '--out', out / 'gate.json', *eps))
        if r.get('passed') is not True:
            raise SystemExit('gate failed (see gate.json)')
        return {'passed': True, 'commit': commit([CFG / 'gates' / f'{name}.json'], f'exp(EXP-SVC-PRED-001): gate receipt {name} (swap_pipeline)'),
                'X1': r['X1'], 'F1': r['F1'], 'M0': r['M0'], 'M4': r['M4']}
    ctx['gate'] = run.step('2 gate', s_gate)

    inst = CFG / f'{name}.json'

    def s_instance():
        last_json(sh(PY, '-B', *bp, 'new-instance', '--name', name, '--predictor-kind', 'mix', '--mix', REPO / ctx['pins']['mix_pin']))
        d = json.loads(inst.read_text()); d['status'] = 'REGISTERED'; d['execute'] = True
        d['review'] = {'instance_check': f"swap_pipeline: final-mix {ctx['final_mix_sha256'][:8]}, mix pin {ctx['pins']['sha256'][:8]}, gates passed "
                                         f"(receipt gates/{name}.json, report {out / 'gate.json'})",
                       'by': 'swap_pipeline; independent verification is a separate step'}
        inst.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
        return {'config': rel(inst), 'commit': commit([inst], f'exp(EXP-SVC-PRED-001): instance {name} REGISTERED (swap_pipeline)')}
    ctx['instance'] = run.step('3 instance', s_instance)

    def s_predict():
        r = last_json(sh(PY, '-B', *bp, 'predict', '--config', inst))
        if (r['rows'], r['predicted']) != (255, 254):
            raise SystemExit(f"predict rows {r['rows']} predicted {r['predicted']} (expected 255 / 254)")
        d = json.loads(inst.read_text()); d['predictions']['sha256'] = r['sha256']
        inst.write_text(json.dumps(d, indent=1, ensure_ascii=False) + '\n')
        r['commit'] = commit([inst], f'exp(EXP-SVC-PRED-001): {name} 849799 predictions sha {r["sha256"][:8]} (swap_pipeline)')
        return r
    ctx['predict'] = run.step('4 predict 849799', s_predict)

    def s_build():
        r = last_json(sh(PY, '-B', *bp, 'build', '--config', inst))
        cp = pathlib.Path(r['copy']); w = json.loads(cp.read_text())
        t7 = pathlib.Path(w['paths']['dataset'])
        if sha(t7) != sha(cp):
            raise SystemExit('watch copy differs from the T7 watch')
        return {'copy': str(cp), 't7': str(t7), 'sha256': sha(cp), 'identity': w['policy']['identity_sha256'], 'coverage': r.get('coverage')}
    ctx['build'] = run.step('5 watch build', s_build)

    def s_check():
        r = json.loads(sh(PY, '-B', *bp, 'check', '--config', inst))
        (out / 'check.json').write_text(json.dumps(r, indent=1, ensure_ascii=False) + '\n')
        if r.get('ok') is not True:
            raise SystemExit('check not ok (see check.json)')
        return {'ok': True}
    ctx['check'] = run.step('6 check', s_check)

    def s_compare():
        sh(PY, '-B', *bp, 'compare', '--config', inst, '--out', out / 'compare-zo-ring8.json')
        c = json.loads((out / 'compare-zo-ring8.json').read_text())
        if set(c['non_recommendation_differences']) - {'top:ledger'} or c['predictor_reference_law_invalid_rows']:
            raise SystemExit('compare: differences outside recommendations (see compare-zo-ring8.json)')
        base = json.loads(inst.read_text())['compare']['base_watch']
        lc = ledger_check(ctx['build']['t7'], base)
        (out / 'ledger-check.json').write_text(json.dumps(lc, indent=1) + '\n')
        if not lc['pass']:
            raise SystemExit('ledger rows differ outside result.candidate / rho_candidate (see ledger-check.json)')
        keys = ('rank1_pitch_type_changed', 'rank1_cell_changed', 'rank1_cell_inside_outside', 'candidate_set_changed', 'recommendation_presence')
        return {k: c[k] for k in keys} | {'ledger': {k: lc[k] for k in ('decision_rows_changed', 'pass')}}
    ctx['compare_zo_ring8'] = run.step('7 compare vs zo-ring8 + ledger', s_compare)

    def s_refs():
        sys.path.insert(0, str(REPO / 'scripts/svc_pred'))
        import build_pred as BP
        res = {}
        nw = json.loads(pathlib.Path(ctx['build']['copy']).read_text())
        for spec in a.ref:
            rn, rp = spec.split('=', 1)
            r = BP.compare(json.loads(pathlib.Path(rp).read_text()), nw)
            (out / f'compare-{rn}.json').write_text(json.dumps(r | {'base': rp, 'base_sha256': sha(rp)}, indent=1, ensure_ascii=False) + '\n')
            res[rn] = {k: r[k] for k in ('rank1_pitch_type_changed', 'candidate_set_changed', 'non_recommendation_differences')}
        return res
    ctx['compare_refs'] = run.step('8 compare vs refs', s_refs)

    def s_npz():
        npz = out / 'eps-mix.npz'
        r = last_json(sh(PY, '-B', REPO / 'scripts/svc_pred/export_eps_mix.py', REPO, REPO / ctx['pins']['mix_pin'], fm_path, npz))
        wn = JOB / 'watch' / name / 'eps-mix.npz'
        if not wn.exists():
            shutil.copy2(npz, wn)
        return {'path': str(npz), 'sha256': r['sha256'], 'rows': r['rows'], 'built': r['built']}
    ctx['eps_npz'] = run.step('9 E_PS mix npz', s_npz)

    def s_fan():
        sh('nice', '-n', '10', PY, '-B', FAN, '--pred', out / 'eps-mix.npz', '--name', name, '--status', 'adopted', '--out', out / 'fan')
        return {'out': str(out / 'fan'), 'files': sorted(p.name for p in (out / 'fan').iterdir())}
    ctx['fan_numbers'] = run.step('10 fan_numbers', s_fan)

    def s_booth():
        b = booth_lines(ctx['build']['t7'], ctx['build']['identity'])
        (out / 'booth-lines.md').write_text(booth_text(b)); (out / 'booth-lines.json').write_text(json.dumps(b, indent=1, ensure_ascii=False) + '\n')
        return b
    ctx['booth'] = run.step('11 booth lines', s_booth)
    shutil.copy2(ctx['build']['copy'], out / '849799.json')
    ctx['timings'] = run.steps; ctx['total_seconds'] = round(sum(s['seconds'] for s in run.steps), 1)
    ctx['next'] = 'Fable independent verification (separate step), then screen check (booth screen session), then PM edits booth lines by hand'
    (out / 'summary.json').write_text(json.dumps(ctx, indent=1, ensure_ascii=False) + '\n')
    print((out / 'booth-lines.md').read_text())
    print(json.dumps({'bundle': str(out), 'watch_sha256': ctx['build']['sha256'], 'identity': ctx['build']['identity'][:8],
                      'compare_zo_ring8': ctx['compare_zo_ring8'], 'compare_refs': ctx['compare_refs'], 'total_seconds': ctx['total_seconds']},
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
