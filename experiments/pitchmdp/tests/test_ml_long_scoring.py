"""Pure F4 identity gates; no datasets, model fitting or scoring."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_long_experiment import _source_hashes
from run_ml_long_history import auxiliary_identity
from score_ml_long_history import validate_comparisons, validate_fit_identity


def fixtures():
    sources = {name: 'a' * 64 for name in _source_hashes()}
    config = {'width': 128, 'device': 'auto',
              'budget': {'epochs': 30, 'patience': 5, 'batch_size': 256, 'learning_rate': .0005},
              'registration': {'control': 'F4-H0', 'primary_candidates': ['F4-32', 'F4-128']}}
    samples = {name: {'n': 5, 'rows_sha256': str(i) * 64, 'path': name + '.parquet'}
               for i, name in enumerate(('train', 'earlystop', 'temperature', 'blend', 'dev'))}
    prep = {'identity': {'source_hashes': {'pitchmdp/' + k: v for k, v in sources.items()}},
            'samples': samples, 'features': {'long_stream': {'long_length': 128, 'max_capacity': 128},
                                            'context': {'frozen_context': 'yes'}},
            'auxiliary_hashes': {'normalizer_sha256': 'b'*64, 'delivery_sha256': 'c'*64}}
    saved = {'parent_preparation_sha256': 'd'*64, 'cell': 'F4-32', 'seed': 0,
             'samples': {k: {'n': v['n'], 'rows_sha256': v['rows_sha256']} for k, v in samples.items()},
             'budget': deepcopy(config['budget']), 'width': 128, 'device_requested': None,
             'features': {'long_length': 32, 'max_capacity': 128}, 'source_hashes': sources,
             'context_sha256': canonical_hash(prep['features']['context']),
             'normalizer_sha256': 'b'*64, 'delivery_sha256': 'c'*64}
    return config, prep, saved


def test_complete_identity_passes_and_every_previously_unchecked_field_is_bound():
    config, prep, saved = fixtures()
    validate_fit_identity(saved, prep, config, 'F4-32', 0, 'd'*64)
    for field, value in [('width', 64), ('device_requested', 'mps'), ('context_sha256', 'x'*64),
                         ('normalizer_sha256', 'x'*64), ('delivery_sha256', 'x'*64),
                         ('source_hashes', {}), ('features', {'long_length': 128, 'max_capacity': 128})]:
        changed = deepcopy(saved); changed[field] = value
        with pytest.raises(ValueError, match='complete fit identity'):
            validate_fit_identity(changed, prep, config, 'F4-32', 0, 'd'*64)
    for extra in ('samples', 'budget'):
        changed = deepcopy(saved); changed[extra]['unregistered_axis'] = True
        with pytest.raises(ValueError):
            validate_fit_identity(changed, prep, config, 'F4-32', 0, 'd'*64)
    changed = deepcopy(saved); changed['unexpected'] = True
    with pytest.raises(ValueError):
        validate_fit_identity(changed, prep, config, 'F4-32', 0, 'd'*64)


def test_member_and_exact_sample_family_and_requested_device():
    config, prep, saved = fixtures()
    config['device'], saved['device_requested'] = 'cpu', 'cpu'
    validate_fit_identity(saved, prep, config, 'F4-32', 0, 'd'*64)
    with pytest.raises(ValueError):
        validate_fit_identity(saved, prep, config, 'F4-32', 1, 'd'*64)
    with pytest.raises(ValueError):
        validate_fit_identity(saved, prep, config, 'F4-32', 0, 'e'*64)
    changed = deepcopy(prep); changed['samples']['other_dev'] = changed['samples']['dev']
    with pytest.raises(ValueError, match='sample family'):
        validate_fit_identity(saved, changed, config, 'F4-32', 0, 'd'*64)


def test_exact_control_and_ordered_two_candidate_family():
    config, _, _ = fixtures()
    validate_comparisons(config)
    for control, candidates in [('G0-global', ['F4-32', 'F4-128']),
                                ('F4-H0', ['F4-32', 'F4-128', 'F4-32']),
                                ('F4-H0', ['F4-128', 'F4-32'])]:
        changed = deepcopy(config)
        changed['registration'] = {'control': control, 'primary_candidates': candidates}
        with pytest.raises(ValueError):
            validate_comparisons(changed)


def test_auxiliary_hash_pins_actual_normalizer_and_delivery_bytes():
    delivery = SimpleNamespace(report={'draws': 400, 'seed': 42},
        pools={(0, ('FF', 'R', 'L')): np.arange(32, dtype=np.float32).reshape(4, 8)},
        fallback=np.arange(32, dtype=np.float32).reshape(4, 8))
    aux = {'normalizer': SimpleNamespace(report=lambda: {'mean': [1., 2.], 'scale': [3., 4.]}),
           'delivery': delivery}
    original = auxiliary_identity(aux)
    changed = deepcopy(delivery)
    changed.pools[(0, ('FF', 'R', 'L'))][0, 0] += 1.
    assert auxiliary_identity({**aux, 'delivery': changed})['delivery_sha256'] != original['delivery_sha256']
    changed = deepcopy(delivery); changed.fallback[-1, -1] += 1.
    assert auxiliary_identity({**aux, 'delivery': changed})['delivery_sha256'] != original['delivery_sha256']
    changed_aux = {**aux, 'normalizer': SimpleNamespace(report=lambda: {'mean': [1., 99.], 'scale': [3., 4.]})}
    assert auxiliary_identity(changed_aux)['normalizer_sha256'] != original['normalizer_sha256']
