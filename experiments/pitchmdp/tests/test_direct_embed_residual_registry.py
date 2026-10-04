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
    assert er.CONTINUOUS == er.BASELINES['periodic_counts']['context_continuous']
    assert {a['embedding'] for a in er.ARMS.values()} == set(pr.KINDS)
    assert {a['model'] for a in er.ARMS.values()} == set(pr.KINDS.values())


def test_seeds_disjoint_from_every_earlier_block_and_family_passes():
    assert not set(er.SEEDS) & {0, 1, 2, *wr.FIRST_SEEDS, *wr.SEED_BLOCKS['repeat2_seeds_8_12'], 13, 14, 15}
    cfg = family()
    assert er.check_family(cfg) is cfg and er.check_contrasts(cfg) == er.CONTRASTS and er.FITS == 15


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][3].update(width=256),
    lambda c: c['arms'][4].update(embedding='periodic'),
    lambda c: c['arms'][3].update(workload=wr.WORKLOADS['block']),
    lambda c: c['arms'][3].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][3].pop(er.FLAG),
    lambda c: c['arms'][3].update(priority=True),
    lambda c: c['arms'][1].update(width=145),
    lambda c: c['arms'][2].update(combo=1),
    lambda c: c['arms'].reverse(),
    lambda c: c.update(seeds=[16, 17, 19]),
    lambda c: c.update(repeat_tag='x'),
    lambda c: c.update(fits=15.0),
    lambda c: c['comparison'][er.CONTRAST_KEY][0].update(reference='residual_counts'),
    lambda c: c['comparison'][er.CONTRAST_KEY].pop(),
])
def test_any_edit_is_refused(edit):
    cfg = family()
    edit(cfg)
    with pytest.raises(ValueError):
        er.check_contrasts(cfg)


def test_each_fixed_contrast_changes_exactly_one_factor():
    changes = {c['name']: {k for k, (a, b) in zip(('input', 'backbone'), zip(er.FACTORS[c['candidate']],
                                                                            er.FACTORS[c['reference']])) if a != b}
               for c in er.CONTRASTS}
    assert all(len(v) == 1 for v in changes.values()) and len(changes) == 4
    assert all('control' not in (c['candidate'], c['reference']) for c in er.CONTRASTS)
