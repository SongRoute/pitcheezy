import pytest
from pitchmdp.matrix_confirmation_metrics import finish_decisions


def pair(p=.004):
    return {'status': 'measured', 'nll': {'delta': -.004, 'ci95': [-.006, -.002], 'p_less': p},
            'brier': {'delta': -.001, 'ci95': [-.002, 0]}}


def row():
    return {'paired': pair(), 'low_paired': pair(), 'seed_deltas': [-.004]*4+[.001],
            'low_seed_deltas': [-.004]*4+[.001]}


def test_one_candidate_keeps_four_holm_slots_and_requires_four_of_five():
    records = [row()]
    adjusted = finish_decisions(records)
    assert adjusted[:2] == pytest.approx([.016, .016])
    assert adjusted[2:] == [None, None]
    assert records[0]['N']['status'] == 'predictive_improvement'
    records = [row()]
    records[0]['seed_deltas'] = [-.004]*3+[.001]*2
    finish_decisions(records)
    assert records[0]['N']['status'] == 'inconclusive'


def test_group_improvement_still_requires_overall_noninferiority():
    record = row()
    record['paired']['nll']['ci95'][1] = .002
    finish_decisions([record])
    assert record['G']['low_group']['status'] == 'predictive_improvement'
    assert record['G']['status'] == 'inconclusive'
    assert not record['G']['whole_population_noninferior']


def test_baseline_only_has_no_superiority_test_and_missing_low_stays_null():
    assert finish_decisions([]) == [None]*4
    record = row()
    record['low_paired'] = record['low_seed_deltas'] = None
    adjusted = finish_decisions([record])
    assert adjusted == [.016, None, None, None]
    assert record['G']['status'] == 'unmeasured'
    with pytest.raises(ValueError, match='At most two'):
        finish_decisions([row(), row(), row()])
