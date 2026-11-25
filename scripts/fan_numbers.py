"""FAN-NUMBERS 재계산 도구 (읽기 전용, CPU 1개). 예측기 하나를 넣으면 발표용 팬 숫자와 판정 문장 후보를 다시 뽑는다.

usage: python -B scripts/fan_numbers.py --pred R [--status adopted|posthoc] [--out DIR]
  --pred R | C               : EXP-SVC-P9-003 CR attempt-001 candidate.jsonl 의 cand_R / cand_C
  --pred path/to/file.npz    : 배열 game_pk, n, probs(행×9, PITCH_NAMES 순서) [+ built(bool)]. 없는 행·built=False 행은 1/9
  --pred path/to/file.jsonl  : 줄마다 {"game_pk","n","q"(9개 또는 null)}
  --status adopted : 결과 열람 전에 정한 규칙으로 고른 모델(기본). posthoc : E_PS를 보고 고른 모델(서술 문장도 '단서 필요'로 내림)
  --name 표시 이름(기본: --pred 값)

규칙(FAN-NUMBERS-1121과 같음): E_PS 31경기·8,943구(849799 제외), floor eps 0.001, 누락 행 1/9, top-k는 argsort(-q) 앞 k개,
경기 단위 짝 부트스트랩 2000회 seed 20261110 (eval_prediction.py @ cuda-score-1120). 기준선 = CR baselines.jsonl freq(투수×카운트 빈도표).
판정 규칙은 RULES 절에 코드로 적었다. 화면 카드 비율은 예측기와 무관한 USEFUL-1 S 블록 값을 인용만 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

import os  # noqa: E402
WT = pathlib.Path(__file__).resolve().parents[1]   # public repo root (originally a worktree at the cuda-score-1120 commit)
sys.path.insert(0, str(WT / 'src')); sys.path.insert(0, str(WT / 'scripts/svc_p9'))
import eval_prediction as E  # noqa: E402

CR = pathlib.Path(os.environ.get('PITCHEEZY_CR_DIR', 'data/archive/CR'))   # EXP-SVC-P9-003 CR attempt-001 (not in the public repo)
USEFUL = pathlib.Path(os.environ.get('PITCHEEZY_USEFUL_RESULT', 'data/archive/EXP-USEFUL-1-result.json'))   # optional
K9 = 9

# ---------------------------------------------------------------- RULES (FAN-NUMBERS.md 3절을 코드로)
# 팬 말 후보: "약 d번 중 k번". 분모는 팬이 바로 그릴 수 있는 2·3·4·10만 쓴다.
FRACTIONS = sorted({(k, d) for d in (2, 3, 4, 10) for k in range(1, d)}, key=lambda kd: (kd[0] / kd[1], kd[1]))
SAY_POINT_TOL, SAY_RANGE_TOL = 0.02, 0.03      # 말해도 됨: 점추정이 분수에서 ±0.02 안, 동률 범위 양끝이 ±0.03 안
CAVEAT_POINT_TOL, CAVEAT_RANGE_TOL = 0.035, 0.05  # 단서 필요: 위를 못 넘지만 이 안이면 "약"과 범위를 같이 말함
# 기준선 대비 차이: 등록 비교가 아니므로 '말해도 됨'은 없다. 네 가지 동률 처리 모두 95% 하한 > 0 이면 '단서 필요', 하나라도 0을 포함하면 '쓰지 않음'.
DIFF_VARIANTS = ('rule_as_is', 'ties_all_correct', 'ties_all_wrong', 'built_rows_only')
DIFF_CAVEATS = ['사후 비교다(예측기 대 표는 등록 비교가 아님)',
                '표는 8월(~08-31)까지 자료다. 예측기 학습 기간과 다르면 그 차이를 같이 말한다',
                '구간은 경기 표본의 흔들림만 담고 seed 흔들림은 담지 않는다']
ALWAYS_NO = [('"화면 추천 1순위 적중률 N%" / "화면 1순위가 절반 맞는다"', '이 숫자는 구종 예측 모델의 것이다. 화면 추천의 일치율이 아니다(N23).'),
             ('"N% 확률로 맞힌다" / "이 공이 나올 확률 N%"', '화면의 %는 정책 선택 비중이다. 팬 말은 "N번 중 M번" 빈도로만 쓴다.'),
             ('"5차전에서도 N번 중 M번"', '5차전(849799)은 E_PS 31경기에 없다.')]


def frac_phrase(k, d):
    return f'약 {d}번 중 {k}번'


def nearest_fraction(x):
    return min(FRACTIONS, key=lambda kd: (abs(x - kd[0] / kd[1]), kd[1]))


def verdict_rate(point, lo, hi, status):
    k, d = nearest_fraction(point); f = k / d
    dev_p, dev_r = abs(point - f), max(abs(lo - f), abs(hi - f))
    if dev_p <= SAY_POINT_TOL and dev_r <= SAY_RANGE_TOL:
        v = '말해도 됨'
    elif dev_p <= CAVEAT_POINT_TOL and dev_r <= CAVEAT_RANGE_TOL:
        v = '단서 필요'
    else:
        v = '쓰지 않음'
    why = f'가장 가까운 분수 {k}/{d}={f:.3f}, 점추정 차 {dev_p:.3f}(기준 {SAY_POINT_TOL}), 동률 범위 최대 차 {dev_r:.3f}(기준 {SAY_RANGE_TOL})'
    if v == '말해도 됨' and status == 'posthoc':
        v, why = '단서 필요', why + '; E_PS를 보고 고른 모델이라 한 단계 내림'
    return v, frac_phrase(k, d), why


def verdict_diff(variants):
    los = [variants[n]['ci95'][0] for n in DIFF_VARIANTS]; his = [variants[n]['ci95'][1] for n in DIFF_VARIANTS]
    if all(l > 0 for l in los):
        return '단서 필요', f'네 가지 동률 처리 모두 95% 하한 > 0 (최저 하한 {min(los):+.4f})'
    if all(h < 0 for h in his):
        return '쓰지 않음', '예측기가 표보다 낮다(네 처리 모두 상한 < 0)'
    bad = [n for n in DIFF_VARIANTS if variants[n]['ci95'][0] <= 0]
    return '쓰지 않음', '구간이 0을 포함하는 처리: ' + ', '.join(f"{n} [{variants[n]['ci95'][0]:+.4f}, {variants[n]['ci95'][1]:+.4f}]" for n in bad)


# ---------------------------------------------------------------- data
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_cr():
    rec = json.loads((CR / 'receipt.json').read_text())
    for n in ('candidate', 'baselines'):
        if sha(CR / f'{n}.jsonl') != rec['stage_files_sha256'][n]:
            raise SystemExit(f'FAILED: CR {n}.jsonl sha != receipt')
    cand = [json.loads(l) for l in open(CR / 'candidate.jsonl')]
    base = {(b['game_pk'], b['n']): b for b in (json.loads(l) for l in open(CR / 'baselines.jsonl'))}
    keys, y, g, T, cr = [], [], [], [], {'R': [], 'C': []}
    for c in cand:
        b = base[(c['game_pk'], c['n'])]
        lab = E.label(b['actual'])
        if lab is None:
            raise SystemExit('FAILED: unlabelled row')
        keys.append((int(c['game_pk']), int(c['n']))); y.append(lab); g.append(int(c['game_pk'])); T.append(b['freq'])
        cr['R'].append(c['cand_R']); cr['C'].append(c['cand_C'])
    games = set(g)
    if len(keys) != 8943 or len(games) != 31 or 849799 in games or games != set(E.e_ps_games()):
        raise SystemExit('FAILED: E_PS rows/games')
    return keys, np.array(y), np.array(g), np.array(T, float), cr, {n: rec['stage_files_sha256'][n] for n in ('candidate', 'baselines')}


def load_pred(spec, keys, cr):
    pos = {k: i for i, k in enumerate(keys)}
    Q = np.full((len(keys), K9), 1.0 / K9); built = np.zeros(len(keys), bool); src = {'spec': spec}
    if spec in cr:
        for i, q in enumerate(cr[spec]):
            if q is not None:
                Q[i] = q; built[i] = True
        src['source'] = f'CR candidate.jsonl cand_{spec}'
        return Q, built, src
    p = pathlib.Path(spec)
    if not p.is_file():
        raise SystemExit(f'FAILED: --pred {spec}: R, C 또는 .npz/.jsonl 파일 경로여야 한다')
    src.update(path=str(p.resolve()), sha256=sha(p))
    if p.suffix == '.npz':
        z = np.load(p, allow_pickle=False)
        probs = np.asarray(z['probs'], float); bz = np.asarray(z['built'], bool) if 'built' in z.files else np.ones(len(probs), bool)
        rows = zip(z['game_pk'], z['n'], probs, bz)
    elif p.suffix == '.jsonl':
        rows = ((r['game_pk'], r['n'], r['q'], r['q'] is not None) for r in (json.loads(l) for l in open(p)))
    else:
        raise SystemExit('FAILED: .npz 또는 .jsonl만 받는다')
    seen = 0
    for gk, n, q, b in rows:
        i = pos.get((int(gk), int(n)))
        if i is None:
            continue
        seen += 1
        if b and q is not None:
            q = np.asarray(q, float)
            if q.shape != (K9,) or not np.isfinite(q).all() or abs(q.sum() - 1) > 1e-6 or (q < 0).any():
                raise SystemExit(f'FAILED: bad probability row {(gk, n)}')
            Q[i] = q; built[i] = True
    src['matched_rows'] = seen
    if seen == 0:
        raise SystemExit('FAILED: no E_PS rows matched')
    return Q, built, src


# ---------------------------------------------------------------- metrics
def hits(F, y, k):
    order = np.argsort(-F, axis=1, kind='quicksort')[:, :k]  # compare.py 168행과 같은 np.argsort(-F[i])[:k]
    return (order == y[:, None]).any(1).astype(float)


def boot_diff(a, b, inv, n_games, mask):
    a, b, gi = a[mask], b[mask], inv[mask]
    rng = np.random.default_rng(E.SEED); out = np.empty(E.REPLICATES)
    for r in range(E.REPLICATES):
        w = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)[gi].astype(float)
        out[r] = w @ a / w.sum() - w @ b / w.sum()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--name'); ap.add_argument('--status', choices=('adopted', 'posthoc'), default='adopted')
    ap.add_argument('--out', default=str(pathlib.Path(__file__).parent / 'out'))
    a = ap.parse_args(); t0 = time.time()
    name = a.name or pathlib.Path(a.pred).stem
    keys, y, g, T, cr, in_sha = load_cr()
    Q, built, src = load_pred(a.pred, keys, cr)
    F, FT = E.floor(Q), E.floor(T)
    tie = np.isclose(F.max(1), F.min(1))
    ids, inv = np.unique(g, return_inverse=True)
    res = {'name': name, 'status': a.status, 'pred': src, 'rows': int(len(y)), 'games': int(len(ids)), 'built_rows': int(built.sum()),
           'uniform_rows': int((~built).sum()), 'tie_rows_pred': int(tie.sum()), 'tie_rows_table': int(np.isclose(FT.max(1), FT.min(1)).sum()),
           'bootstrap': {'unit': 'game_pk', 'replicates': E.REPLICATES, 'seed': E.SEED}, 'floor_eps': E.EPS, 'cr_sha256': in_sha, 'topk': {}}
    allm = np.ones(len(y), bool)
    for k in (1, 3):
        h, hT = hits(F, y, k), hits(FT, y, k)
        var = {'rule_as_is': (h, allm), 'ties_all_correct': (np.where(tie, 1.0, h), allm),
               'ties_all_wrong': (np.where(tie, 0.0, h), allm), 'built_rows_only': (h, built)}
        V = {}
        for vn, (hv, m) in var.items():
            per = [(hv[m & (g == x)].mean(), hT[m & (g == x)].mean()) for x in ids if (m & (g == x)).any()]
            V[vn] = {'n': int(m.sum()), 'pred': float(hv[m].mean()), 'table': float(hT[m].mean()), 'diff': float(hv[m].mean() - hT[m].mean()),
                     'ci95': boot_diff(hv, hT, inv, len(ids), m), 'games_pred_higher': int(sum(p > t for p, t in per)),
                     'games_pred_lower': int(sum(p < t for p, t in per))}
        lo, hi = V['ties_all_wrong']['pred'], V['ties_all_correct']['pred']
        vr, phrase, why = verdict_rate(V['rule_as_is']['pred'], lo, hi, a.status)
        vd, whyd = verdict_diff(V)
        res['topk'][f'top{k}'] = {'variants': V, 'tie_range': [lo, hi], 'rate_verdict': vr, 'rate_phrase': phrase, 'rate_why': why,
                                  'diff_verdict': vd, 'diff_why': whyd, 'table_phrase': frac_phrase(*nearest_fraction(V['rule_as_is']['table']))}
    if USEFUL.is_file():
        S = json.loads(USEFUL.read_text())['S']
        res['screen_card'] = {'source': str(USEFUL), 'sha256': sha(USEFUL), 'card_rows': S['rows'], 'games': S['card_any_minus_B_PCOUNT_topk']['games'],
                              'card_contains_actual': S['card_any_minus_B_PCOUNT_topk']['a']['rate'], 'table_top3': S['card_any_minus_B_PCOUNT_topk']['b']['rate'],
                              'diff_ci95': S['card_any_minus_B_PCOUNT_topk']['ci95']['rate'], 'card_rank1_actual': S['card_rank1_minus_B_PCOUNT_top1']['a']['rate'],
                              'note': '예측기와 무관(USEFUL-1 당시 화면 카드). 화면 정책이 바뀌면 이 도구 밖에서 다시 재야 한다. 정확도로 말하지 않는다.'}
    res['seconds'] = round(time.time() - t0, 1)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'fan-{name}.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / f'fan-{name}.md').write_text(render(res))
    print(render(res))


def render(r):
    t1, t3 = r['topk']['top1'], r['topk']['top3']
    L = [f"# 팬용 숫자 — {r['name']} ({'결과 전 규칙으로 고른 모델' if r['status'] == 'adopted' else 'E_PS를 보고 고른 모델'})", '',
         f"자료: E_PS 31경기·{r['rows']:,}구(849799 제외), 예측 있는 행 {r['built_rows']:,}, 1/9로 채운 행 {r['uniform_rows']}(동률 행 {r['tie_rows_pred']}, 표 동률 {r['tie_rows_table']}). "
         f"경기 단위 짝 부트스트랩 {r['bootstrap']['replicates']}회 seed {r['bootstrap']['seed']}. 입력 {r['pred'].get('source') or r['pred'].get('path')}", '', '## 값']
    for k, t in (('top1', t1), ('top3', t3)):
        L.append(f'- {k}')
        for vn in DIFF_VARIANTS:
            v = t['variants'][vn]
            L.append(f"  - {vn}(n={v['n']}): 예측기 {v['pred']:.4f} / 표 {v['table']:.4f}, 차이 {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}], "
                     f"경기 예측기 높음 {v['games_pred_higher']}·낮음 {v['games_pred_lower']}")
        L.append(f"  - 동률 처리 범위: {t['tie_range'][0]:.4f}~{t['tie_range'][1]:.4f}")
    L += ['', '## 판정 문장 후보',
          f"1. **{t1['rate_verdict']}** — \"학습에 안 쓴 올해 포스트시즌 31경기에서, 구종 예측 모델의 1순위가 실제 구종과 같은 것이 {t1['rate_phrase']}이었다.\" ({t1['variants']['rule_as_is']['pred']:.4f}; {t1['rate_why']})",
          f"2. **{t3['rate_verdict']}** — \"3순위 안에 실제 구종이 드는 것은 {t3['rate_phrase']}이었다.\" ({t3['variants']['rule_as_is']['pred']:.4f}; {t3['rate_why']})",
          f"3. **{t1['diff_verdict']}** — \"같은 공들에서 '이 투수가 이 카운트에서 던진 비율' 표는 1순위가 {t1['table_phrase']} 맞았다. 모델이 더 자주 맞혔다(약 {round(t1['variants']['rule_as_is']['diff'] * 100)}%p 차이).\" "
          f"({t1['variants']['rule_as_is']['diff']:+.4f} [{t1['variants']['rule_as_is']['ci95'][0]:+.4f}, {t1['variants']['rule_as_is']['ci95'][1]:+.4f}]; {t1['diff_why']})",
          f"4. **{t3['diff_verdict']}** — \"3순위 안 기준으로도 모델이 표보다 낫다({t3['variants']['rule_as_is']['pred']:.3f} 대 {t3['variants']['rule_as_is']['table']:.3f}).\" ({t3['diff_why']})"]
    if t1['diff_verdict'] == '단서 필요' or t3['diff_verdict'] == '단서 필요':
        L.append('   - 차이 문장의 단서: ' + ' / '.join(DIFF_CAVEATS))
    n = 5
    for s, why in ALWAYS_NO:
        L.append(f'{n}. **쓰지 않음** — {s}: {why}'); n += 1
    sc = r.get('screen_card')
    if sc:
        L += ['', '## 화면 카드 안 비율 (인용, 예측기와 무관)',
              f"- 카드 구종 안에 실제 공: {sc['card_contains_actual']:.3f} vs 표 상위 3개 {sc['table_top3']:.3f}, 차이 구간 [{sc['diff_ci95'][0]:+.4f}, {sc['diff_ci95'][1]:+.4f}] ({sc['card_rows']:,}행·{sc['games']}경기). 카드 1순위 = 실제 {sc['card_rank1_actual']:.3f}.",
              f"- 판정: **쓰지 않음**(자랑할 숫자로). 물으면 \"화면 추천은 예측이 아니라서 실제 공과 덜 겹친다\"의 근거로만. {sc['note']}"]
    L += ['', '## 늘 붙일 말', '- 주어는 "구종 예측 모델"이다. "화면 추천 3칸이 10번 중 9번 맞는다는 뜻은 아니다."',
          '- 같은 31경기를 여러 번 읽었다. "처음 보는 시험", "사전 등록 시험"이라고 하지 않는다(P9-003은 사후 등록).']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
