"""Small C1 registration and shared-unit selection tests; no real fits."""
from copy import deepcopy

import pytest

from pitchmdp.matrix_confirmation import (SEEDS, member_identity, required_units,
                                          unit_identity, validate_config)
from pitchmdp.matrix_confirmation_metrics import SCORING


def registration(cells, comparisons, status='candidate_comparison'):
    return {'protocol': 'ml_confirmation_g_v1', 'experiment_id': 'EXP-C1-test',
            'parent_run': '/approved/G', 'parent_preparation_sha256': 'a'*64,
            'parent_analysis_sha256': 'b'*64, 'cells': cells,
            'primary_comparisons': comparisons, 'selection_status': status,
            'seeds': list(SEEDS), 'kind': 'flatten_mlp', 'width': 128,
            'draws': 400, 'device': 'auto',
            'budget': {'epochs': 30, 'patience': 5, 'batch_size': 1024, 'learning_rate': .0005},
            'individual_tau': 1000, 'cluster_tau': 10000,
            'registration': {'c1_scoring': deepcopy(SCORING)}}


def test_baseline_only_and_mapped_contrasts_are_exact():
    base = registration(['G0-global'], [], 'baseline_stability_only')
    assert validate_config(base) == base
    paired = registration(['G0-global', 'G1-personal', 'G2-feature'],
                          [['G1-personal', 'G0-global'], ['G2-feature', 'G0-global']])
    assert validate_config(paired) == paired
    for bad in [registration(['G1-personal'], [], 'baseline_stability_only'),
                registration(['G0-global', 'G3-cluster'], [['G3-cluster', 'G0-global']]),
                registration(['G0-global', 'G1-personal'], [['G1-personal', 'G0-global']]*2)]:
        with pytest.raises(ValueError):
            validate_config(bad)


def test_config_rejects_changed_seed_training_or_posterior_route():
    base = registration(['G0-global'], [], 'baseline_stability_only')
    for name, value in [('seeds', [0, 1, 2, 3]), ('draws', 25),
                        ('individual_tau', 500), ('kind', 'transformer')]:
        modified = deepcopy(base)
        modified[name] = value
        with pytest.raises(ValueError, match='settings'):
            validate_config(modified)
    modified = deepcopy(base)
    modified['registration']['c1_scoring']['required_negative_seeds'] = 3
    with pytest.raises(ValueError, match='scoring family'):
        validate_config(modified)


def test_unit_union_and_new_seed_identity():
    parent = {'global': {'mode': 'global'}, 'feature': {'mode': 'feature'},
              'cluster0': {'mode': 'cluster'}, 'cluster1': {'mode': 'cluster'},
              'personal7': {'mode': 'personal'}}
    chosen = required_units(parent, ['G2-feature', 'G3-cluster', 'G4-partial'])
    assert list(chosen) == list(parent)
    assert list(required_units(parent, ['G0-global'])) == ['global']
    prep = {'units': chosen, 'cells': ['G0-global', 'G4-partial'],
            'parent_preparation_sha256': 'a'*64}
    assert unit_identity(prep, 'global', 3)['seed'] == 3
    assert member_identity(prep, 'G4-partial', 4)['seed'] == 4
    with pytest.raises(ValueError, match='registered'):
        unit_identity(prep, 'global', 2)
