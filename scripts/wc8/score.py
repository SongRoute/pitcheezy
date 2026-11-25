"""WC8-CONFIRM-001 step 5: THE one scoring run (PREREG §5-§6, §8). Labels are read here and nowhere else.

    python -B scripts/wc8/score.py --pred-dir <dir> --out <result.json> [--exploratory]

Refuses unless: <out> does not exist; <dir>/PINS.json exists and every pinned file exists with its sha (exactly the
8 games x 3 models + baselines); the scripts/wc8 code files equal the shas in PINS and HEAD == PINS code_commit with a
clean tracked tree; the copied scoring functions equal eval_prediction.py (sha 1c49e173). Then, per game, the final
feed (sha from registration-facts) gives the actual pitch code of the n-th pitch (x1_run.targets order; the at-bat
of service_key is checked against it).

Rows: every pitch; a target outside the 9 groups (eval_prediction ``label`` None) is dropped as a target. A row that
any of v4 / B-SEQ / R omits is scored uniform 1/9 for ALL probabilistic models (v4, B-SEQ, R, frequency table).
Floor 0.001 on all. Metrics = eval_prediction row_parts / pooled (NLL, Brier, ECE10 top-label, top-1) + top-3.
Primary (one verdict): dNLL = NLL(v4) - NLL(B-SEQ), game-level paired bootstrap 2000, seed 20261110, 2.5/97.5:
  upper < 0 -> "v4가 낫다(와일드카드 8경기)"; lower > 0 -> "v4가 못하다(와일드카드 8경기)"; else "차이 감지 안 됨(와일드카드 8경기)".
Descriptive only: R - v4, v4 - frequency (same bootstrap), per-model metrics, modal top-1, the primary contrast
without 849843 (7 games), per-game dNLL and sign counts. ``--exploratory`` (PREREG §3: a prior scoring record found)
prefixes "탐색 " to the verdict and uses the "탐색" sentence head. Conservative reading (t df=7 + exact sign-flip) is combined with the bootstrap verdict: disagreement -> 차이 감지 안 됨.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evalfns as E  # noqa: E402

VERDICTS = {'better': 'v4가 낫다(와일드카드 8경기)', 'worse': 'v4가 못하다(와일드카드 8경기)',
            'none': '차이 감지 안 됨(와일드카드 8경기)'}
RULE = '95% upper < 0 -> v4가 낫다; lower > 0 -> v4가 못하다; else 차이 감지 안 됨 (PREREG §6)'
PROB_MODELS = ('v4', 'bseq', 'r', 'freq')
T_CRIT_DF7 = 2.364624251592785   # t_{0.975, df=7} (PREREG §6 보수적 읽기)


def conservative_reading(dnll_games):
    """PREREG §6 (xval-ml 보강, 결과 전): per-game ΔNLL (unweighted game means) -> t interval (df=n-1, n=8 -> 2.365)
    and the exact two-sided sign-flip test on the mean (2^n assignments). Verdict 낫다/못하다 only when the t interval
    excludes 0 on that side AND sign-flip p < 0.05; else 차이 감지 안 됨."""
    import itertools
    d = np.asarray(dnll_games, np.float64); n = len(d)
    if n != 8:
        raise ValueError('conservative reading is registered for 8 games')
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    half = T_CRIT_DF7 * sd / np.sqrt(n); lo, hi = mean - half, mean + half
    obs = abs(mean); count = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        if abs(float((d * np.array(signs)).mean())) >= obs - 1e-15:
            count += 1
    p = count / 2 ** n
    text = VERDICTS['better'] if (hi < 0 and p < 0.05) else VERDICTS['worse'] if (lo > 0 and p < 0.05) else VERDICTS['none']
    return {'mean_game_dnll': mean, 'sd': sd, 't_crit_df7': T_CRIT_DF7, 't_ci95': [lo, hi], 'signflip_p_two_sided': p,
            'signflip_assignments': 2 ** n, 'verdict': text,
            'rule': 't 구간이 0을 그쪽으로 제외하고 부호 뒤집기 p < 0.05일 때만 낫다/못하다, 그 밖은 차이 감지 안 됨'}


def verdict_text(ci, exploratory=False):
    lo, hi = ci
    text = VERDICTS['better'] if hi < 0 else VERDICTS['worse'] if lo > 0 else VERDICTS['none']
    return f'탐색 {text}' if exploratory else text


def targets_ab(final):
    """(actual pitch code, at_bat_number) of the n-th pitch, in replay_snapshots' order (x1_run.targets + at-bat)."""
    out = []
    for play in final['liveData']['plays']['allPlays']:
        for event in play['playEvents']:
            if event.get('isPitch'):
                out.append((((event.get('details') or {}).get('type') or {}).get('code'), int(play['about']['atBatIndex']) + 1))
    return out


def assemble_game(pk, docs, base, actual):
    """Rows of one game aligned by (n, service_key) across the three prediction files and the baselines."""
    rows_by = {m: docs[m]['rows'] for m in ('v4', 'bseq', 'r')}
    n_rows = len(rows_by['v4'])
    if any(len(v) != n_rows for v in rows_by.values()) or len(base['rows']) != n_rows or len(actual) != n_rows:
        raise SystemExit(f'{pk}: row counts differ (models / baselines / feed pitches)')
    out = []
    for i in range(n_rows):
        rv, rb, rr, b = rows_by['v4'][i], rows_by['bseq'][i], rows_by['r'][i], base['rows'][i]
        key = rv['service_key']
        if not (rv['n'] == rb['n'] == rr['n'] == b['n'] == i) or not (key == rb['service_key'] == rr['service_key'] == b['service_key']):
            raise SystemExit(f'{pk} row {i}: (n, service_key) misaligned')
        gk, ab, _ = key.split(':')
        if int(gk) != int(pk) or int(ab) != actual[i][1]:
            raise SystemExit(f'{pk} row {i}: service_key at-bat differs from the feed pitch order')
        reasons = {m: r['omit_reason'] for m, r in (('v4', rv), ('bseq', rb), ('r', rr)) if r['q'] is None}
        out.append({'game_pk': int(pk), 'n': i, 'service_key': key, 'actual': actual[i][0],
                    'v4': rv['q'], 'bseq': rb['q'], 'r': rr['q'], 'freq': b['freq_q'], 'modal': b['modal'],
                    'omit_reasons': reasons})
    return out


def top3(q, y):
    order = np.argsort(-np.asarray(q, np.float64), axis=1, kind='stable')[:, :3]
    return (order == np.asarray(y)[:, None]).any(1).astype(np.float64)


def metrics(Fq, y):
    out = E.pooled(E.row_parts(Fq, y))
    out['top3'] = float(top3(Fq, y).mean())
    return out


def score_rows(rows, *, replicates=E.REPLICATES, seed=E.SEED, exploratory=False, drop_game=C.SENSITIVITY_DROP):
    excluded, omitted, omitted_by_model, keep = {}, {}, {m: 0 for m in ('v4', 'bseq', 'r')}, []
    for r in rows:
        y = E.label(r['actual'])
        if y is None:
            excluded[str(r['actual'])] = excluded.get(str(r['actual']), 0) + 1
            continue
        if r['omit_reasons']:
            for m in r['omit_reasons']:
                omitted_by_model[m] += 1
            reason = '|'.join(sorted(set(str(v) for v in r['omit_reasons'].values())))
            omitted[reason] = omitted.get(reason, 0) + 1
            r = dict(r, filled=True, **{m: E.UNIFORM for m in PROB_MODELS})
        keep.append((r, y))
    if not keep:
        raise ValueError('no evaluable rows')
    y = np.array([k[1] for k in keep]); games = np.array([k[0]['game_pk'] for k in keep])
    Q = {m: np.array([k[0][m] for k in keep], np.float64) for m in PROB_MODELS}
    for m, q in Q.items():
        if q.shape != (len(keep), E.K) or not np.isfinite(q).all() or not np.allclose(q.sum(1), 1, atol=1e-9) or (q < 0).any():
            raise ValueError(f'{m} probabilities')
    F = {m: E.floor(q) for m, q in Q.items()}
    present = np.array([not k[0].get('filled') for k in keep])
    modal = np.array([k[0]['modal'] for k in keep], np.int64)

    def contrast(a, b, mask=None):
        if mask is None:
            return E.paired_contrast(F[a], F[b], y, games, replicates=replicates, seed=seed)
        return E.paired_contrast(F[a][mask], F[b][mask], y[mask], games[mask], replicates=replicates, seed=seed)

    primary = contrast('v4', 'bseq')
    per_game = {}
    for g in sorted(set(games.tolist())):
        m = games == g
        per_game[str(g)] = {'rows': int(m.sum()),
                            'dnll_v4_minus_bseq': E.pooled(E.row_parts(F['v4'][m], y[m]))['nll'] - E.pooled(E.row_parts(F['bseq'][m], y[m]))['nll']}
    signs = [v['dnll_v4_minus_bseq'] for v in per_game.values()]
    cons = conservative_reading(signs) if len(signs) == 8 else None
    boot_verdict = verdict_text(primary['ci95']['nll'])
    final = boot_verdict if (cons is not None and cons['verdict'] == boot_verdict) else VERDICTS['none']
    report = {
        'registration': 'WC8-CONFIRM-001', 'rows_scored': len(keep), 'games': int(len(set(games.tolist()))),
        'excluded_targets': excluded, 'omitted_filled_uniform_all_models': omitted,
        'omitted_rows_by_model': omitted_by_model, 'omitted_rows_total': int((~present).sum()), 'floor_eps': E.EPS,
        'bootstrap': {'replicates': replicates, 'seed': seed, 'unit': 'game, paired', 'percentiles': [2.5, 97.5]},
        'primary': {'contrast': 'NLL(v4) - NLL(B-SEQ)', **primary,
                    'bootstrap_verdict': boot_verdict, 'conservative_reading': cons,
                    'verdict': f'탐색 {final}' if exploratory else final,
                    'rule': RULE + '; 최종 = 부트스트랩 판정과 보수적 읽기가 같을 때 그 판정, 갈리면 차이 감지 안 됨 (PREREG §6 보강)',
                    'exploratory': bool(exploratory)},
        'descriptive': {
            'r_minus_v4': contrast('r', 'v4'),
            'v4_minus_freq': contrast('v4', 'freq'),
            'per_model': {m: metrics(F[m], y) for m in PROB_MODELS},
            'modal_top1': {'all_rows': float((modal == y).mean()), 'rows': int(len(y)),
                           'non_omitted_rows': float((modal[present] == y[present]).mean()) if present.any() else None,
                           'non_omitted_n': int(present.sum()),
                           'note': 'point prediction; scored on its own value on every row (it needs no W1 input)'},
            'per_game': per_game,
            'per_game_sign_counts': {'v4_better(neg)': sum(v < 0 for v in signs), 'v4_worse(pos)': sum(v > 0 for v in signs),
                                     'zero': sum(v == 0 for v in signs)}},
        'verdict_note': 'descriptive blocks carry no verdict (PREREG §6)'}
    drop = games == int(drop_game)
    if drop.any() and (~drop).any():
        report['descriptive']['sensitivity_without_849843'] = dict(
            games=int(len(set(games[~drop].tolist()))), rows=int((~drop).sum()), **contrast('v4', 'bseq', ~drop),
            note='sensitivity, not a verdict')
    return report


def sentence(report, pitches, exploratory=False):
    """PREREG §10 (보강 포함). `pitches` is kept for the record; the sentence uses the scored row count."""
    p = report['primary']
    lo, hi = p['ci95']['nll']
    n = report['rows_scored']
    c = p.get('conservative_reading')
    split = '(경기 수가 적어 더 엄한 검정으로는 구분되지 않았습니다)' if (c is not None and c['verdict'] != p['bootstrap_verdict']) else ''
    head = '탐색: 와일드카드 8경기' if exploratory else '저희가 구종 예측 모델을 채점한 적이 없는 와일드카드 8경기'
    pm = report['descriptive']['per_model']
    second = (f"같은 8경기에서 지금 모델 1순위 일치 {pm['v4']['top1'] * 100:.1f}%, 투수×카운트 표 {pm['freq']['top1'] * 100:.1f}%, "
              f"최빈 구종 {report['descriptive']['modal_top1']['all_rows'] * 100:.1f}%.")
    return (f"{head}(채점 투구 {n:,}구)로 한 번만 확인했습니다. 지금 모델과 문헌 방식 GRU의 예측 오차 차이는 "
            f"{p['point']['nll']:+.4f}[95% 구간 {lo:+.4f} ~ {hi:+.4f}]로 '{p['verdict']}'였습니다{split}. 경기 수가 적어 구간이 넓습니다. "
            + second)


def check_pins(d, pins):
    want = C.expected_files()
    if sorted(pins['files']) != want:
        raise SystemExit('PINS.json does not list exactly the expected files')
    for name, sha in pins['files'].items():
        p = d / name
        if not p.exists() or C.sha256_file(p) != sha:
            raise SystemExit(f'{name}: missing or differs from PINS.json')
    for rel, sha in pins['code_sha256'].items():
        if C.sha256_file(C.REPO / rel) != sha:
            raise SystemExit(f'{rel} changed since PINS.json')
    if set(pins['code_sha256']) != set(C.CODE_FILES):
        raise SystemExit('PINS.json code list differs')
    commit = C.require_clean()
    if commit != pins['code_commit']:
        raise SystemExit(f"HEAD {commit[:8]} differs from the pinned code commit {pins['code_commit'][:8]}")
    return commit


def run(pred_dir, out, exploratory=False):
    out = pathlib.Path(out).resolve()
    if out.exists():
        raise SystemExit(f'{out} exists; scoring runs once (PREREG §8)')
    d = pathlib.Path(pred_dir).resolve()
    pins_path = d / 'PINS.json'
    if not pins_path.exists():
        raise SystemExit('PINS.json missing: run scripts/wc8/pin.py first')
    pins = json.loads(pins_path.read_text())
    commit = check_pins(d, pins)
    ref = E.verify_reference(C.EVAL_REFERENCE['path'], C.EVAL_REFERENCE['sha256'])
    f = C.facts()
    rows, pitches = [], 0
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    for pk in C.WC8:
        game = C.game_entry(pk, f=f)
        docs = {m: json.loads((d / C.pred_name(pk, m)).read_text()) for m in C.MODELS}
        base = json.loads((d / C.baseline_name(pk)).read_text())
        final = C.read_feed_verified(game)          # labels: first read here
        actual = targets_ab(final)
        del final
        if len(actual) != game['is_pitch']:
            raise SystemExit(f'{pk}: feed pitches {len(actual)} != is_pitch {game["is_pitch"]}')
        pitches += len(actual)
        rows += assemble_game(pk, docs, base, actual)
    report = score_rows(rows, exploratory=exploratory)
    report['pitches_total'] = pitches
    report['sentence'] = sentence(report, pitches, exploratory)
    report['inputs'] = {'pins': str(pins_path), 'pins_sha256': C.sha256_file(pins_path), 'code_commit': commit,
                        'prediction_files_sha256': pins['files'], 'eval_reference': ref, 'facts_sha256': C.FACTS_SHA,
                        'scored_utc': started, 'python': sys.executable}
    report['view_log_line'] = (f"{started} score.py commit {commit[:8]} PINS {report['inputs']['pins_sha256'][:8]} "
                               f"-> {out.name}; verdict: {report['primary']['verdict']}")
    sha = C.write_new(out, report)
    return report, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--pred-dir', type=pathlib.Path, required=True)
    ap.add_argument('--out', type=pathlib.Path, required=True)
    ap.add_argument('--exploratory', action='store_true', help='PREREG §3: a prior scoring record was found')
    a = ap.parse_args(argv)
    report, sha = run(a.pred_dir, a.out, a.exploratory)
    p, dsc = report['primary'], report['descriptive']
    print(json.dumps({'result_sha256': sha, 'sentence': report['sentence'], 'verdict': p['verdict'],
                      'dnll_v4_minus_bseq': p['point']['nll'], 'ci95': p['ci95']['nll'],
                      'r_minus_v4': [dsc['r_minus_v4']['point']['nll'], dsc['r_minus_v4']['ci95']['nll']],
                      'v4_minus_freq': [dsc['v4_minus_freq']['point']['nll'], dsc['v4_minus_freq']['ci95']['nll']],
                      'per_model': dsc['per_model'], 'modal_top1': dsc['modal_top1'],
                      'sensitivity_without_849843': {k: dsc.get('sensitivity_without_849843', {}).get(k) for k in ('point', 'ci95')},
                      'per_game_sign_counts': dsc['per_game_sign_counts'], 'rows_scored': report['rows_scored'],
                      'omitted': report['omitted_filled_uniform_all_models'], 'excluded_targets': report['excluded_targets'],
                      'view_log_line': report['view_log_line']}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
