import pytest
import numpy as np
import pandas as pd
import pitchmdp.matrix_confirmation_metrics as metrics
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


def test_family_wires_declared_order_five_seeds_and_fixed_robustness_budget(monkeypatch):
    y = np.arange(600) % 10
    control = np.full((600, 10), .1)
    candidate = np.full((600, 10), .8/9)
    candidate[np.arange(600), y] = .2
    predictions = {name: {'primary': p, 'seed_primary': np.stack([p]*5)}
                   for name, p in [('candidate', candidate), ('control', control)]}
    calls = []
    monkeypatch.setattr(metrics, 'paired_game_comparison', lambda *args, **kwargs: pair())
    monkeypatch.setattr(metrics, 'guardrails', lambda *args, **kwargs: calls.append(kwargs) or {'status': 'unconfirmed'})
    monkeypatch.setattr(metrics, 'volume_interaction', lambda *args, **kwargs: {'status': 'descriptive'})
    records, family = metrics.compare_family(y, predictions, np.arange(600)%30,
        pd.DataFrame({'train_volume': ['low']*600}), [('candidate', 'control')])
    assert calls == [{'candidate_family_size': 2, 'draws': 100000}]
    assert family['adjusted'] == pytest.approx([.016, .016, None, None], nan_ok=True)
    assert records[0]['N']['status'] == 'predictive_improvement'
    assert len(records[0]['seed_deltas']) == 5
    predictions['candidate']['seed_primary'] = predictions['candidate']['seed_primary'][:3]
    with pytest.raises(ValueError, match='five-seed'):
        metrics.compare_family(y, predictions, np.arange(600)%30,
            pd.DataFrame({'train_volume': ['low']*600}), [('candidate', 'control')])
