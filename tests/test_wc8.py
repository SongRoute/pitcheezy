"""WC8-CONFIRM-001 scripts/wc8: scoring logic, baselines, gate comparison, pins and guards on SYNTHETIC rows only.
No feed is opened, no predictor is loaded, nothing is predicted (registration configs are read for the config test)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts/wc8'))

import common as C  # noqa: E402
import evalfns as E  # noqa: E402
import score as S  # noqa: E402
import baselines as B  # noqa: E402
import gate_849799 as G  # noqa: E402
import pin as PN  # noqa: E402

CODES = ['FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'CH', 'FS', 'KN']


def _q(rng, k=9):
    v = rng.random(k) + 0.05
    return list(v / v.sum())


def synth_rows(seed=0, n_per=40, games=C.WC8, omit_every=0, good=None):
    """good: model name whose q puts 0.9 on the true class (others random)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in games:
        for i in range(n_per):
            code = CODES[rng.integers(0, 9)]
            y = E.label(code)
            r = {'game_pk': int(g), 'n': i, 'service_key': f'{g}:{i // 4 + 1}:{i % 4 + 1}', 'actual': code,
                 'freq': _q(rng), 'modal': int(rng.integers(0, 9)), 'omit_reasons': {}}
            for m in ('v4', 'bseq', 'r'):
                if m == good:
                    q = np.full(9, 0.1 / 8); q[y] = 0.9; r[m] = list(q)
                else:
                    r[m] = _q(rng)
            if omit_every and i % omit_every == 0:
                for m in ('v4', 'bseq', 'r'):
                    r[m] = None
                r['omit_reasons'] = {'v4': 'x', 'bseq': 'x', 'r': 'x'}
            rows.append(r)
    return rows


# ---------------------------------------------------------------- copied scoring code
def test_evalfns_identical_to_reference():
    if not pathlib.Path(C.EVAL_REFERENCE['path']).exists():
        pytest.skip('reference worktree absent')
    E.verify_reference(C.EVAL_REFERENCE['path'], C.EVAL_REFERENCE['sha256'])


def test_evalfns_refuses_wrong_sha():
    if not pathlib.Path(C.EVAL_REFERENCE['path']).exists():
        pytest.skip('reference worktree absent')
    with pytest.raises(SystemExit):
        E.verify_reference(C.EVAL_REFERENCE['path'], '0' * 64)


# ---------------------------------------------------------------- verdict
@pytest.mark.parametrize('ci,want', [((-0.02, -0.001), 'v4가 낫다(와일드카드 8경기)'), ((0.001, 0.02), 'v4가 못하다(와일드카드 8경기)'),
                                     ((-0.01, 0.01), '차이 감지 안 됨(와일드카드 8경기)'), ((-0.01, 0.0), '차이 감지 안 됨(와일드카드 8경기)'),
                                     ((0.0, 0.01), '차이 감지 안 됨(와일드카드 8경기)')])
def test_verdict_mapping(ci, want):
    assert S.verdict_text(ci) == want
    assert S.verdict_text(ci, exploratory=True) == '탐색 ' + want


# ---------------------------------------------------------------- scoring
def test_equal_models_give_zero_and_no_difference():
    rows = synth_rows(1)
    for r in rows:
        r['bseq'] = list(r['v4'])
    rep = S.score_rows(rows, replicates=50)
    assert rep['primary']['point']['nll'] == 0
    assert rep['primary']['ci95']['nll'] == [0.0, 0.0]
    assert rep['primary']['verdict'] == '차이 감지 안 됨(와일드카드 8경기)'


def test_good_v4_is_better_and_matches_direct_contrast():
    rows = synth_rows(2, good='v4')
    rep = S.score_rows(rows, replicates=200)
    assert rep['primary']['verdict'] == 'v4가 낫다(와일드카드 8경기)'
    y = np.array([E.label(r['actual']) for r in rows]); g = np.array([r['game_pk'] for r in rows])
    fa = E.floor(np.array([r['v4'] for r in rows])); fb = E.floor(np.array([r['bseq'] for r in rows]))
    direct = E.paired_contrast(fa, fb, y, g, replicates=200, seed=E.SEED)
    assert rep['primary']['point'] == direct['point'] and rep['primary']['ci95'] == direct['ci95']
    assert rep['descriptive']['per_game_sign_counts']['v4_better(neg)'] == 8


def test_good_bseq_makes_v4_worse():
    rep = S.score_rows(synth_rows(3, good='bseq'), replicates=200)
    assert rep['primary']['verdict'] == 'v4가 못하다(와일드카드 8경기)'


def test_bootstrap_deterministic_and_seeded():
    rows = synth_rows(4)
    a = S.score_rows(rows, replicates=100)['primary']['ci95']
    b = S.score_rows(rows, replicates=100)['primary']['ci95']
    c = S.score_rows(rows, replicates=100, seed=1)['primary']['ci95']
    assert a == b and a != c


def test_uniform_fill_for_all_models_including_freq():
    rows = synth_rows(5, omit_every=5)
    rep = S.score_rows(rows, replicates=20)
    assert rep['omitted_rows_total'] == 8 * 8
    assert rep['omitted_filled_uniform_all_models'] == {'x': 64}
    # same rows with uniform written explicitly everywhere (incl. freq) give identical metrics
    rows2 = []
    for r in rows:
        r = dict(r)
        if r['omit_reasons']:
            r.update(v4=E.UNIFORM, bseq=E.UNIFORM, r=E.UNIFORM, freq=E.UNIFORM, omit_reasons={})
        rows2.append(r)
    rep2 = S.score_rows(rows2, replicates=20)
    for m in S.PROB_MODELS:
        assert rep['descriptive']['per_model'][m] == rep2['descriptive']['per_model'][m]
    assert rep['primary']['point'] == rep2['primary']['point']


def test_omission_in_one_model_fills_all():
    rows = synth_rows(6)
    rows[0]['r'] = None; rows[0]['omit_reasons'] = {'r': 'key_mismatch'}
    rep = S.score_rows(rows, replicates=10)
    assert rep['omitted_rows_by_model'] == {'v4': 0, 'bseq': 0, 'r': 1}
    assert rep['omitted_filled_uniform_all_models'] == {'key_mismatch': 1}


def test_excluded_targets_dropped():
    rows = synth_rows(7)
    rows[0]['actual'] = 'PO'; rows[1]['actual'] = None; rows[2]['actual'] = 'UN'
    rep = S.score_rows(rows, replicates=10)
    assert rep['rows_scored'] == len(rows) - 3
    assert rep['excluded_targets'] == {'PO': 1, 'None': 1, 'UN': 1}


def test_sensitivity_drops_849843_only():
    rows = synth_rows(8)
    rep = S.score_rows(rows, replicates=30)
    s = rep['descriptive']['sensitivity_without_849843']
    assert s['games'] == 7 and s['rows'] == 7 * 40
    rows7 = [r for r in rows if r['game_pk'] != 849843]
    y = np.array([E.label(r['actual']) for r in rows7]); g = np.array([r['game_pk'] for r in rows7])
    d = E.paired_contrast(E.floor(np.array([r['v4'] for r in rows7])), E.floor(np.array([r['bseq'] for r in rows7])), y, g,
                          replicates=30, seed=E.SEED)
    assert s['point'] == d['point'] and s['ci95'] == d['ci95']


def test_modal_and_top3():
    rows = synth_rows(9)
    for r in rows:
        r['modal'] = E.label(r['actual'])
    rep = S.score_rows(rows, replicates=10)
    assert rep['descriptive']['modal_top1']['all_rows'] == 1.0
    q = np.array([[0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0], [1 / 9] * 9])
    assert list(S.top3(q, [2, 3])) == [1.0, 0.0]          # ties: stable order -> classes 0,1,2
    assert list(S.top3(q, [3, 2])) == [0.0, 1.0]


def test_sentence_fills_numbers():
    rep = S.score_rows(synth_rows(10), replicates=20)
    s = S.sentence(rep, 2315)
    assert s.startswith(f"저희가 구종 예측 모델을 채점한 적이 없는 와일드카드 8경기(채점 투구 {rep['rows_scored']:,}구)") and rep['primary']['verdict'] in s
    assert '같은 8경기에서 지금 모델 1순위 일치 ' in s and '투수×카운트 표 ' in s and '최빈 구종 ' in s
    assert S.sentence(rep, 2315, exploratory=True).startswith('탐색: 와일드카드 8경기')


def test_conservative_reading_rules():
    import pytest
    allneg = [-0.02, -0.01, -0.03, -0.015, -0.02, -0.025, -0.01, -0.012]
    c = S.conservative_reading(allneg)
    assert c['signflip_p_two_sided'] == 2 / 256 and c['t_ci95'][1] < 0 and c['verdict'] == S.VERDICTS['better']
    assert S.conservative_reading([-x for x in allneg])['verdict'] == S.VERDICTS['worse']
    mixed = [-0.02, 0.01, -0.03, 0.015, -0.02, 0.025, -0.01, 0.012]
    assert S.conservative_reading(mixed)['verdict'] == S.VERDICTS['none']
    one_big = [-0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]   # sign-flip p large even if mean negative
    assert S.conservative_reading(one_big)['verdict'] == S.VERDICTS['none']
    with pytest.raises(ValueError):
        S.conservative_reading(allneg[:7])


def test_final_verdict_disagreement_is_none():
    rep = S.score_rows(synth_rows(10), replicates=20)
    p = rep['primary']
    c = p['conservative_reading']
    want = p['bootstrap_verdict'] if (c and c['verdict'] == p['bootstrap_verdict']) else S.VERDICTS['none']
    assert p['verdict'] == want


# ---------------------------------------------------------------- alignment
def _docs(pk, n=6, omit=None):
    rows = [{'n': i, 'service_key': f'{pk}:{i // 3 + 1}:{i % 3 + 1}', 'w1_key': f'{pk}:{i // 3 + 1}:{i % 3 + 1}',
             'q': None if i == omit else [1 / 9] * 9, 'fallback': False, 'flags': [], 'omit_reason': 'x' if i == omit else None}
            for i in range(n)]
    docs = {m: {'format': 'svc-r1-predictions-v1', 'game_pk': pk, 'final_feed_sha256': 'f', 'classes': list(C.CLASSES),
                'rows': [dict(r) for r in rows]} for m in C.MODELS}
    base = {'format': 'wc8-baselines-v1', 'game_pk': pk, 'final_feed_sha256': 'f', 'labels_read': False,
            'frequency_table': {'sha256': C.FREQ_TABLE['sha256']},
            'rows': [{'n': r['n'], 'service_key': r['service_key'], 'freq_q': [1 / 9] * 9, 'modal': 0} for r in rows]}
    return docs, base


def test_assemble_aligns_and_refuses_misalignment():
    pk = 849841
    docs, base = _docs(pk, omit=2)
    actual = [('FF', i // 3 + 1) for i in range(6)]
    rows = S.assemble_game(pk, docs, base, actual)
    assert [r['n'] for r in rows] == list(range(6)) and rows[2]['omit_reasons'] == {m: 'x' for m in C.MODELS}
    bad = [('FF', 9)] + actual[1:]
    with pytest.raises(SystemExit):
        S.assemble_game(pk, docs, base, bad)
    docs['bseq']['rows'][1]['service_key'] = f'{pk}:9:9'
    with pytest.raises(SystemExit):
        S.assemble_game(pk, docs, base, actual)
    with pytest.raises(SystemExit):
        S.assemble_game(pk, docs, base, actual[:-1])


def test_keys_match_order():
    assert C.keys_match_order(['1:1:1', '1:1:2', '1:2:1'], [1, 1, 2], 1)
    assert not C.keys_match_order(['1:1:1', '1:2:1'], [1, 1], 1)
    assert not C.keys_match_order(['2:1:1'], [1], 1)


def test_pin_validate_game():
    pk = 849841
    docs, base = _docs(pk, omit=1)
    game = {'final_feed_sha256': 'f', 'official_date': '2026-09-30', 'is_pitch': 6}
    rec = {m: {'ok': True, 'as_of_ok': True, 'trained_through': '2026-09-28', 'predictor_pin_sha256': C.MODEL_SPEC[m]['pin'][1],
               'labels_read': False} for m in C.MODELS}
    bad, summ = PN.validate_game(pk, docs, rec, base, game)
    assert bad == [] and summ['omit_identical']
    rec['r']['trained_through'] = '2026-09-30'
    docs['v4']['rows'][0]['q'] = [0.5] * 9
    bad, _ = PN.validate_game(pk, docs, rec, base, game)
    assert any('as-of' in b for b in bad) and any('q format' in b for b in bad)


# ---------------------------------------------------------------- baselines
def _table():
    return {'alpha': 4.0, 'global': [50, 40, 30, 20, 10, 5, 5, 5, 1], 'pitcher': {'1': [0, 7, 0, 7, 0, 0, 0, 0, 0], '2': [3, 0, 0, 0, 0, 0, 9, 0, 0]},
            'context': {'1:0:0': [0, 2, 0, 0, 0, 0, 0, 0, 0]}, 'classes': list(C.CLASSES)}


def test_modal_class_rules():
    t = _table()
    assert B.modal_class(t, 1) == (1, True)     # tie SI vs SL -> SI (higher global count)
    assert B.modal_class(t, 2) == (6, True)
    assert B.modal_class(t, 99) == (0, False)   # unknown -> global modal FF


def test_baseline_rows_frequency_formula():
    t = _table()
    rows = B.baseline_rows(t, [(0, 'k0', 1, 0, 0), (1, 'k1', 99, 1, 2)])
    g = np.array(t['global'], float); gp = (g + 4 / 9) / (g.sum() + 4)
    pp = (np.array(t['pitcher']['1']) + 4 * gp) / (14 + 4)
    want0 = (np.array(t['context']['1:0:0']) + 4 * pp) / (2 + 4)
    assert np.allclose(rows[0]['freq_q'], want0) and np.allclose(rows[1]['freq_q'], gp)
    assert rows[0]['pitcher_in_table'] and not rows[1]['pitcher_in_table'] and rows[1]['modal'] == 0


# ---------------------------------------------------------------- gate comparison
def test_compare_rows():
    docs, _ = _docs(849799, omit=3)
    a, b = docs['v4']['rows'], [dict(r) for r in docs['r']['rows']]
    assert G.compare_rows(a, b) == ([], 0.0)
    b[0] = dict(b[0], q=[v + (1e-13 if k == 0 else 0) for k, v in enumerate(b[0]['q'])])
    bad, worst = G.compare_rows(a, b)
    assert bad == [] and 0 < worst <= 1e-12
    b[1] = dict(b[1], q=[v + (1e-9 if k == 0 else 0) for k, v in enumerate(b[1]['q'])])
    assert G.compare_rows(a, b)[0]
    b2 = [dict(r) for r in a]; b2[2]['w1_key'] = 'other'
    assert G.compare_rows(a, b2)[0]
    assert G.compare_rows(a, a[:-1])[0]


# ---------------------------------------------------------------- guards and per-game configs (pinned configs only)

def _need_facts():
    if not pathlib.Path(C.FACTS).exists():
        pytest.skip('경기 등록 정보 파일(registration-facts.json, MLB 파생)이 이 공개 레포에 없음')


def test_game_entry_guards():
    _need_facts()
    f = C.facts()
    eps = int(f['E_PS']['game_pks'][0])
    with pytest.raises(SystemExit):
        C.game_entry(eps, f=f)
    with pytest.raises(SystemExit):
        C.game_entry(C.DEMO_GAME, f=f)
    with pytest.raises(SystemExit):
        C.game_entry(123, f=f)
    assert C.game_entry(C.DEMO_GAME, allow_demo=True, f=f)['game_pk'] == C.DEMO_GAME
    total = sum(C.game_entry(pk, f=f)['is_pitch'] for pk in C.WC8)
    assert total == 2315   # PREREG §2
    assert {C.game_entry(pk, f=f)['official_date'] for pk in C.WC8} == {'2026-09-29', '2026-09-30'}


@pytest.mark.parametrize('model', C.MODELS)
def test_game_config_only_replaced_blocks(model, tmp_path):
    _need_facts()
    import predict as P
    game = C.game_entry(849843)
    c = P.game_config(model, game, tmp_path, 'deadbeef')
    t = P.template(model)
    assert {k for k in set(c) | set(t) if c.get(k) != t.get(k)} <= P.CONFIG_REPLACED
    assert c['game']['game_pk'] == 849843 and c['predictor']['game_date'] == '2026-09-29'
    assert c['predictor']['trained_through'] < '2026-09-29'
    assert c['predictor']['config_sha256'] == C.MODEL_SPEC[model]['pin'][1]
    assert c['predictions']['path'].startswith(str(tmp_path))
    assert not list(tmp_path.iterdir())          # pure: nothing written


def test_expected_files():
    f = C.expected_files()
    assert len(f) == 8 * (3 * 3 + 1) and C.baseline_name(849843) in f


def test_sentence_marks_split_readings():
    rep = S.score_rows(synth_rows(10), replicates=20)
    p = rep['primary']
    p['bootstrap_verdict'] = S.VERDICTS['better']; p['conservative_reading'] = dict(p['conservative_reading'], verdict=S.VERDICTS['none'])
    p['verdict'] = S.VERDICTS['none']
    assert f"'{S.VERDICTS['none']}'였습니다(경기 수가 적어 더 엄한 검정으로는 구분되지 않았습니다)" in S.sentence(rep, 2315)
    p['conservative_reading'] = dict(p['conservative_reading'], verdict=S.VERDICTS['better']); p['verdict'] = S.VERDICTS['better']
    assert '더 엄한 검정' not in S.sentence(rep, 2315)
