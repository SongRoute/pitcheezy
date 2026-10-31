from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_ml_transfer import chosen_comparison, config_check, CONTROLS


def fixture():
    return {'followup_candidates': [],
            'reports': {cell: {'primary': {'log_loss': 1.5}} for cell in CONTROLS},
            'logical_cell_costs': {cell: {'total_fit_seconds': 4-i} for i, cell in enumerate(CONTROLS)}}


def test_preregistered_promotion_precedes_diagnostic_nll_ranking():
    data = fixture()
    data['followup_candidates'] = ['G2-feature', 'G1-personal']
    data['reports']['G4-partial']['primary']['log_loss'] = 1.0
    assert chosen_comparison(data) == {'candidate': 'G2-feature', 'control': 'G0-global', 'status': 'screen_promoted'}


def test_no_promotion_uses_cost_tie_break_and_explicit_diagnostic_label():
    assert chosen_comparison(fixture()) == {'candidate': 'G4-partial', 'control': 'G2-feature',
                                           'status': 'diagnostic_only_not_promoted'}


def test_transfer_cannot_swap_control_or_parent_identity():
    valid = {'protocol': 'ml_transfer_v1', 'seeds': [0, 1, 2], 'scope': 'Cmlb',
        'candidate': 'G3-cluster', 'control': 'G2-feature',
        'parent_preparation_sha256': 'a'*64, 'parent_analysis_sha256': 'b'*64}
    assert config_check(valid) is valid
    for changed in ({'control': 'G0-global'}, {'seeds': [0]}, {'parent_analysis_sha256': 'x'*64}):
        with pytest.raises(ValueError):
            config_check({**valid, **changed})
