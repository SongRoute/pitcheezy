"""Synthetic-only C0 → C1 consumer examples and algebraic boundary checks."""
import copy
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from c_contract_examples import build_examples
from observer_app.event_analysis import analyze_event, failed_event

COMPONENTS = ('strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp')


def test_examples_round_trip_recompute_and_preserve_boundaries():
    packet = json.loads(json.dumps(build_examples(), allow_nan=False))
    assert packet == build_examples()
    assert packet['development_only'] and not packet['use_for_performance_evaluation']
    cases = {}
    for example in packet['examples']:
        fn = failed_event if example['function'] == 'failed_event' else analyze_event
        result = fn(**example['inputs'])
        assert result == example['result']
        assert result['evidence']['development_only'] is True
        assert result['evidence']['use_for_performance_evaluation'] is False
        assert result['components']['outcome_residual_pp']['causal_batter_skill'] is False
        assert result['replacement']['value_pp'] is None
        cases[example['case']] = result
    cancelling = cases['cancelling_contrasts']
    assert cancelling['values']['total_pp'] == 0
    assert cancelling['shares']['denominator_pp'] == pytest.approx(40)
    assert [cancelling['components'][key]['abs_share'] for key in COMPONENTS] == pytest.approx([.25, .5, .25])
    assert cases['zero_contrasts']['shares']['stable'] is False
    for case, reason in [('missing_intent', 'missing_intent'), ('partial_coordinates', 'no_plate_plane_calibration')]:
        result = cases[case]
        assert result['status'] == 'partial' and result['reason'] == reason
        assert result['components']['unallocated_residual_pp'] == pytest.approx(-10)
        assert all(result['components'][key]['value_pp'] is None for key in COMPONENTS)
    assert cases['missing_execution']['status'] == 'partial'
    assert cases['missing_observed']['status'] == 'unavailable'
    assert cases['failed']['status'] == 'failed'
    assert all(cases['failed']['values'][key] is None for key in cases['failed']['values'])


@pytest.mark.parametrize('present', list(itertools.product((False, True), repeat=4)))
def test_each_value_missingness_combination_preserves_accounting(present):
    args = copy.deepcopy(build_examples()['examples'][0]['inputs'])
    for key, exists in zip(('reference', 'plan', 'execution', 'observed'), present):
        if not exists:
            args['values'][key] = None
    reference, plan, execution, observed = present
    if execution and not plan:
        with pytest.raises(ValueError, match='execution path requires plan'):
            analyze_event(**args)
        return
    result = analyze_event(**args)
    complete = all(present)
    total_exists = reference and observed
    assert result['status'] == ('complete' if complete else 'partial' if total_exists else 'unavailable')
    parts = result['components']
    if total_exists:
        assigned = sum(parts[key]['value_pp'] or 0 for key in COMPONENTS)
        assert assigned + parts['unallocated_residual_pp'] == pytest.approx(result['values']['total_pp'])
    else:
        assert result['values']['total_pp'] is None
        assert parts['unallocated_residual_pp'] is None
    if not complete:
        assert result['shares']['denominator_pp'] is None
        assert all(parts[key]['abs_share'] is None for key in COMPONENTS)
