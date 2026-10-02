from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from score_ml_sharing import logical_costs, followup_candidates


def test_composite_costs_count_all_dependencies_without_counting_new_fits():
    units = {'global': {'mode': 'global'}, 'feature': {'mode': 'feature'},
             'cluster0': {'mode': 'cluster'}, 'personal10': {'mode': 'personal'}}
    fits = {str(seed): {name: {'seconds_total': i+1} for i, name in enumerate(units)} for seed in range(3)}
    costs = logical_costs(units, fits)
    assert costs['G0-global']['logical_fits'] == 3
    assert costs['G2-feature']['total_fit_seconds'] == 9
    assert costs['G4-partial']['logical_fits'] == 9
    assert costs['G4-partial']['total_fit_seconds'] == 24
    assert 'feature' not in costs['G4-partial']['units_per_seed']


def test_measured_guardrail_failure_blocks_promotion_before_nll_ranking():
    cells = ['G1-personal', 'G2-feature', 'G3-cluster', 'G4-partial']
    comparisons = [{'candidate': cell, 'N': {'status': 'predictive_improvement'},
                    'G': {'status': 'inconclusive'}, 'robustness': {'status': 'unconfirmed'}} for cell in cells]
    comparisons[0]['robustness']['status'] = 'failed'
    reports = {c: {'primary': {'log_loss': 1. if i == 0 else 1.5}} for i, c in enumerate(cells)}
    costs = {c: {'total_fit_seconds': 4-i} for i, c in enumerate(cells)}
    assert followup_candidates(comparisons, reports, costs) == ['G4-partial', 'G3-cluster']
