"""Torch-free checks of the DRAFT embedded-input residual registry (no config, no runner, no fit)."""
import copy
import json
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from pitchmdp import direct_embed_residual_registry as er  # noqa: E402
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp import direct_periodic_residual_model as pr  # noqa: E402


def earlier(path, name):
    return next(a for a in json.loads((REPO / 'configs' / path).read_text())['arms'] if a['name'] == name)


def family():
    arms = [copy.deepcopy(er.BASELINES[n]) if n in er.BASELINES else
            {'name': n, 'priority': 1, er.FLAG: True, 'workload': er.WORKLOAD, **copy.deepcopy(er.ARMS[n])}
            for n in er.ARM_ORDER]
    return {'arms': arms, 'seeds': list(er.SEEDS), 'fits': er.FITS,
            'comparison': {er.CONTRAST_KEY: copy.deepcopy(er.CONTRASTS)}}


def test_baselines_are_the_exact_earlier_config_entries():
    combo, p30 = 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json', 'EXP-P30-001.json'
    assert er.same(er.BASELINES['residual_counts'], earlier(combo, 'residual_counts'))
    for name in ('control', 'periodic_counts'):
        assert er.same(er.BASELINES[name], earlier(p30, name)) and er.same(er.BASELINES[name], earlier(combo, name))
    for path in (p30, 'EXP-P31-001.json'):
        assert er.same(er.BASELINES['periodic_counts_d10'], earlier(path, 'periodic_counts_d10'))
    assert er.CONTINUOUS == er.BASELINES['periodic_counts']['context_continuous']
    assert {a['embedding'] for a in er.ARMS.values()} == set(pr.KINDS)
    assert {a['model'] for a in er.ARMS.values()} == set(pr.KINDS.values())


def test_seeds_disjoint_from_every_earlier_block_and_family_passes():
    assert not set(er.SEEDS) & {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], 13, 14, 15}
    cfg = family()
    assert er.check_family(cfg) is cfg and er.check_contrasts(cfg) == er.CONTRASTS and er.FITS == 18 and len(er.CONTRASTS) == 6


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][4].update(width=256),
    lambda c: c['arms'][5].update(embedding='periodic'),
    lambda c: c['arms'][4].update(workload=wr.WORKLOADS['block']),
    lambda c: c['arms'][4].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][4].pop(er.FLAG),
    lambda c: c['arms'][4].update(priority=True),
    lambda c: c['arms'][3].update(head_dropout=.2),
    lambda c: c['arms'].pop(3),
    lambda c: c['arms'][1].update(width=145),
    lambda c: c['arms'][2].update(combo=1),
    lambda c: c['arms'].reverse(),
    lambda c: c.update(seeds=[16, 17, 19]),
    lambda c: c.update(repeat_tag='x'),
    lambda c: c.update(fits=15.0),
    lambda c: c['comparison'][er.CONTRAST_KEY][0].update(reference='residual_counts'),
    lambda c: c['comparison'][er.CONTRAST_KEY].pop(),
    lambda c: c['comparison'][er.CONTRAST_KEY][1]['held'].update(dropout_rate=0.),
])
def test_any_edit_is_refused(edit):
    cfg = family()
    edit(cfg)
    with pytest.raises(ValueError):
        er.check_contrasts(cfg)


def test_each_fixed_contrast_changes_exactly_its_declared_factors():
    for c in er.CONTRASTS:
        cand, ref = (dict(zip(er.FACTOR_NAMES, er.FACTORS[c[k]])) for k in ('candidate', 'reference'))
        assert {k for k in er.FACTOR_NAMES if cand[k] != ref[k]} == er.ISOLATES[c['isolates']]
        assert all(cand[k] == ref[k] == v for k, v in c['held'].items())
        assert 'control' not in (c['candidate'], c['reference'])
    assert [c['isolates'] for c in er.CONTRASTS].count('backbone_bundle') == 1


def test_factor_table_matches_the_networks():
    from pitchmdp.direct_residual_model import ARCHITECTURE
    assert ARCHITECTURE['dropout'] == er.FACTORS['residual_counts'][2] == pr.ARCHITECTURES['periodic']['backbone']['dropout']
    assert wr.ARMS['periodic_counts_d10']['head_dropout'] == er.FACTORS['periodic_counts_d10'][2]


def test_review_factor_labels_follow_from_the_arm_specs():
    """Review 040c8b4: input / backbone / dropout labels of FACTORS are derived from the fixed entries, not free text."""
    from pitchmdp.direct_residual_model import ARCHITECTURE
    specs = {**er.BASELINES, **er.ARMS}
    backbone = {'residual_mlp': 'residual', 'periodic_flatten_mlp': 'flatten', 'periodic_head_dropout_mlp': 'flatten',
                **{kind: 'residual' for kind in pr.KINDS.values()}}
    for name, (inp, body, rate, position) in er.FACTORS.items():
        spec = specs[name]
        expected_input = spec.get('embedding') or ('periodic' if 'context_continuous' in spec else 'raw')
        expected_rate = ARCHITECTURE['dropout'] if backbone[spec['model']] == 'residual' else spec.get('head_dropout', 0.)
        expected_position = 'blocks' if body == 'residual' else ('head' if expected_rate else 'none')
        assert (inp, body, rate, position) == (expected_input, backbone[spec['model']], expected_rate, expected_position)
    assert set(er.FACTORS) == set(er.ARM_ORDER) - {'control'}
