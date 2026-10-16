"""Generated-only checks for the fixed-action paired population and refusal gates."""
from copy import deepcopy

import pytest

from pitcheezy.seq_shadow.conditional_ff_eval import evaluate
from pitcheezy.seq_shadow.fullseason import LEAVES


SHA = 'a' * 64
ORDER = list(LEAVES)


def attempt(game=1, pitch=1, kind='FF', leaf='ball', native_status='scored', step5_status='scored'):
    row = {'decision_id': f'd-{game}-{pitch}', 'game_pk': game, 'at_bat_number': 1,
           'pitch_number': pitch, 'action': 'FF', 'receipt_seq': 2,
           'query_sha256': SHA, 'context_sha256': SHA, 'leaf_order': ORDER,
           'target': {'game_pk': game, 'at_bat_number': 1, 'pitch_number': pitch,
                      'first_receipt_seq': 3, 'realized_pitch_type': kind,
                      'leaf': leaf, 'leaf_reason': None if leaf is not None else 'unresolved'}}
    for name, status, prob in [('native', native_status, [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
                               ('step5', step5_status, [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])]:
        row[name] = {k: row[k] for k in ('decision_id', 'game_pk', 'at_bat_number', 'pitch_number',
                                         'action', 'receipt_seq', 'query_sha256', 'context_sha256', 'leaf_order')}
        row[name].update(status=status, reason=None if status == 'scored' else 'unsupported_ff',
                         probabilities=prob if status == 'scored' else None)
    return row


def test_fixed_ff_population_game_ci_and_no_input_mutation():
    rows = [attempt(1), attempt(2), attempt(3, kind='SL'),
            attempt(4, native_status='abstain'), attempt(5, step5_status='abstain'),
            attempt(6, leaf=None)]
    rows.append(attempt(7)); rows[-1]['target'] = None
    before = deepcopy(rows)
    out = evaluate(rows)
    assert rows == before
    c = out['counts']
    assert c['attempts'] == 7 and c['observed_targets'] == 6
    assert c['eligible_resolved_ff'] == 4 and c['paired_scored_ff'] == 2 and c['paired_games'] == 2
    assert c['excluded_in_order'] == {'ff_label_unresolved': 1, 'native_abstain': 1,
                                      'realized_non_ff': 1, 'step5_abstain': 1, 'target_absent': 1}
    assert c['arm_scored_resolved_ff'] == {'native': 3, 'step5': 3}
    assert out['nll']['native'] < out['nll']['step5']
    assert out['brier']['native'] < out['brier']['step5']
    assert out['game_cluster_ci95'] is not None and out['superiority_claim'] is False


def test_one_game_point_without_ci_and_nonff_never_in_loss():
    out = evaluate([attempt(), attempt(1, 2, kind='SL', leaf='foul')])
    assert out['counts']['paired_scored_ff'] == 1
    assert out['game_cluster_ci95'] is None and out['superiority_claim'] is False
    empty = evaluate([attempt(kind='SL')])
    assert empty['nll'] is None and empty['counts']['paired_scored_ff'] == 0
    unknown = evaluate([attempt(kind='UNKNOWN', leaf=None)])
    assert unknown['counts']['excluded_in_order'] == {'realized_type_unknown': 1}


@pytest.mark.parametrize('change', [
    lambda r: r.append(deepcopy(r[0])),
    lambda r: r[0]['step5'].__setitem__('query_sha256', 'b'*64),
    lambda r: r[0]['native'].__setitem__('leaf_order', list(reversed(ORDER))),
    lambda r: r[0]['native'].__setitem__('probabilities', [1.0]*7),
    lambda r: r[0]['target'].__setitem__('first_receipt_seq', 2),
    lambda r: r[0]['target'].__setitem__('leaf', 'not_a_leaf'),
    lambda r: r[0]['target'].__setitem__('realized_pitch_type', 'UNKNOWN'),
    lambda r: r[0]['target'].__setitem__('realized_pitch_type', 'UN'),
    lambda r: r[0].__setitem__('action', 'SL'),
])
def test_refuses_identity_vector_label_clock_conflicts(change):
    rows = [attempt()]
    change(rows)
    with pytest.raises(ValueError):
        evaluate(rows)
