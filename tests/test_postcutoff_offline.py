"""Generated-only fixed-six corrected-retrospective adapter checks."""
from copy import deepcopy

import pytest

from experiments.ws_fullseason.data import closure
from pitcheezy.seq_shadow.fullseason import LEAVES
from pitcheezy.seq_shadow.postcutoff_offline import SIX, evaluate, prepare
from test_ws_fullseason import rows


SHA = 'a' * 64


def source():
    M = closure()
    f = rows().iloc[:3][M.cp.PROJECT].copy()
    f['game_pk'], f['game_date'], f['game_type'] = 849812, SIX[849812], 'L'
    return f, M


def test_prior_context_crosses_pa_and_current_result_never_enters_own_snapshot():
    f, M = source()
    fourth = f.iloc[[0]].copy(); fourth['at_bat_number'] = 2
    f = __import__('pandas').concat([f, fourth], ignore_index=True)
    before = f.copy(deep=True)
    out = prepare(f, SHA, M)
    assert f.equals(before)
    assert out['census']['query_ready'] == 4 and out['census']['query_abstain'] == 0
    q = out['cases'][-1]['snapshot']
    assert q['pitch_type_basis'] == 'observed_actual' and q['prospective_claim'] is False
    assert q['query']['at_bat_number'] == 2
    assert q['tokens']['valid'][0][:3] == [True, True, True]
    assert q['tokens']['context_keys'][0][:3] == [[849812, 1, 3], [849812, 1, 2], [849812, 1, 1]]
    assert not {'description', 'events', 'bb_type', 'release_speed', 'outcome10', 'leaf'} & set(q['query'])
    changed = f.copy(deep=True); changed.loc[3, 'description'] = 'foul'
    other = prepare(changed, SHA, M)
    assert other['cases'][-1]['snapshot']['query'] == q['query']
    assert other['cases'][-1]['snapshot']['tokens'] == q['tokens']
    assert other['cases'][-1]['label'] != out['cases'][-1]['label']


def test_missing_same_pa_predecessor_abstains_and_no_future_context():
    f, M = source(); f = f.drop(index=1).reset_index(drop=True)
    out = prepare(f, SHA, M)
    assert out['cases'][0]['status'] == 'ready'
    assert out['cases'][1]['status'] == 'abstain'
    assert out['cases'][1]['reason'] == 'stored_prefix_incomplete'
    assert out['cases'][0]['snapshot']['tokens']['valid'][0][0] is False


def test_earlier_pa_gap_blocks_later_pa_query():
    f, M = source()
    later = f.iloc[[0]].copy(); later['at_bat_number'] = 2
    f = __import__('pandas').concat([f.drop(index=1), later], ignore_index=True)
    out = prepare(f, SHA, M)
    by_key = {tuple(case['key']): case for case in out['cases']}
    assert by_key[(849812, 1, 1)]['status'] == 'ready'
    assert by_key[(849812, 1, 3)]['reason'] == 'stored_prefix_incomplete'
    assert by_key[(849812, 2, 1)]['reason'] == 'stored_prefix_incomplete'


def test_unknown_prior_needs_32_newer_same_pitcher_rows_and_unkeyed_blocks_game():
    f, M = source()
    f = __import__('pandas').concat([f.iloc[[0]].assign(at_bat_number=n) for n in range(1, 102)], ignore_index=True)
    f.loc[0, 'description'] = 'unmapped_generated_description'
    out = prepare(f, SHA, M)
    by_pa = {case['key'][1]: case for case in out['cases']}
    assert by_pa[2]['reason'] == 'prior_quality_unknown'
    assert by_pa[33]['reason'] == 'prior_quality_unknown'
    assert by_pa[34]['status'] == 'ready'
    assert by_pa[34]['snapshot']['context_dependency_floor'] == [849812, 2, 1]
    small, M = source(); small.loc[0, 'pitcher'] = None
    blocked = prepare(small, SHA, M)
    assert blocked['census']['unkeyed_rows_by_game']['849812'] == 1
    assert all(case['status'] == 'abstain' for case in blocked['cases'])


def test_unknown_realized_type_is_not_a_known_prior_token():
    f, M = source()
    f.loc[0, 'pitch_type'] = 'UNKNOWN'
    out = prepare(f, SHA, M)
    assert out['cases'][0]['reason'] == 'observed_type_unknown'
    assert out['cases'][1]['reason'] == 'prior_quality_unknown'


def test_duplicate_and_wrong_game_date_type_stop_before_scoring():
    f, M = source()
    duplicate = __import__('pandas').concat([f, f.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate physical key'):
        prepare(duplicate, SHA, M)
    for field, value in [('game_pk', 999999), ('game_date', '2026-10-13'), ('game_type', 'R')]:
        bad = f.copy(deep=True); bad.loc[0, field] = value
        with pytest.raises(ValueError):
            prepare(bad, SHA, M)


def test_aggregate_score_fixed_fit_marginal_same_rows_and_no_zero_row_metric():
    f, M = source(); prepared = prepare(f, SHA, M)
    before = deepcopy(prepared)
    def fake(snapshot):
        assert snapshot['status'] == 'ready'
        return {'status': 'scored', 'leaf_order': list(LEAVES), 'probabilities': [1/7]*7}
    out = evaluate(prepared, fake)
    assert prepared == before
    assert out['scored'] == 3 and out['per_game']['849812']['n'] == 3
    assert out['overall']['n'] == 3 and out['overall']['model_nll'] is not None
    assert out['per_game']['849820']['n'] == 0 and out['per_game']['849820']['model_nll'] is None
    assert out['baseline']['total'] == 720505 and out['prospective_claim'] is False
    missing = f.drop(index=1).reset_index(drop=True)
    only = prepare(missing, SHA, M)
    assert evaluate(only, fake)['query_abstain_reasons'] == {'stored_prefix_incomplete': 1}
    none = deepcopy(only)
    for case in none['cases']:
        case.update(status='abstain', reason='generated_hold', snapshot=None)
    zero = evaluate(none, fake)
    assert zero['scored'] == 0 and zero['overall']['model_nll'] is None


def test_predictor_vector_refused():
    f, M = source(); prepared = prepare(f, SHA, M)
    with pytest.raises(ValueError, match='invalid predicted seven-vector'):
        evaluate(prepared, lambda _: {'status': 'scored', 'leaf_order': list(LEAVES),
                                      'probabilities': [1.0]*7})


def test_predictor_cannot_mutate_prepared_population():
    f, M = source(); prepared = prepare(f, SHA, M); before = deepcopy(prepared)
    def mutate(snapshot):
        snapshot['query']['pitcher'] = -1
        return {'status': 'scored', 'leaf_order': list(LEAVES), 'probabilities': [1/7]*7}
    assert evaluate(prepared, mutate)['scored'] == 3
    assert prepared == before
