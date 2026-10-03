"""Synthetic CPU tests: periodic workload x head-dropout family (DRAFT-CUDA-WORKLOAD-DROPOUT-20261004)."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cuda_direct as rc  # noqa: E402
import score_cuda_direct as sc  # noqa: E402
import test_cuda_frontier_run as tf  # noqa: E402  shared synthetic export / fake-task helpers
from pitchmdp import direct_workload_dropout_registry as wr  # noqa: E402
from pitchmdp.direct_dropout_model import PeriodicHeadDropoutModel  # noqa: E402
from pitchmdp.direct_periodic_model import DirectPeriodicModel  # noqa: E402

torch.set_num_threads(1)
RUN = REPO / 'configs' / 'DRAFT-CUDA-WORKLOAD-DROPOUT-20261004.json'
COMBO = REPO / 'configs' / 'DRAFT-CUDA-DATA-COMBO-RUN-20261004.json'
DROPOUT_RUN = REPO / 'configs' / 'DRAFT-CUDA-DROPOUT-RUN-20261004.json'
BASE = REPO / 'configs' / 'DRAFT-CUDA-DIRECT-20261004.json'
COLUMNS = ['prior_records_per100', 'thruorder_minus1_over3', 'log_rest_days', 'rest_missing']


def load(registered=False, unpin=True):
    return tf.load(RUN, registered, unpin)


@pytest.fixture
def export(tmp_path):
    return tf.make_export(tmp_path / 'export', tf.lineage(load()))


# ------------------------------------------------------------------ config

def test_config_is_draft_exact_10_arm_family_on_fresh_seeds():
    cfg = load(unpin=False)
    combo = {a['name']: a for a in json.loads(COMBO.read_text())['arms']}
    p29 = json.loads(DROPOUT_RUN.read_text())
    assert cfg['status'] == 'DRAFT' and cfg['execution'] == {'enabled': False} and 'repeat_tag' not in cfg
    assert [a['name'] for a in cfg['arms']] == list(wr.ARM_ORDER) and len(wr.ARM_ORDER) == 10
    assert cfg['seeds'] == [3, 4, 5, 6, 7] and cfg['fits'] == 50 == 10 * len(cfg['seeds'])
    arms = {a['name']: a for a in cfg['arms']}
    for name in ('control', 'periodic', 'periodic_counts', 'periodic_block'):  # exact old entries
        assert arms[name] == combo[name]
    assert {a['name']: a for a in p29['arms']}['periodic'] == arms['periodic']
    assert wr.CONTINUOUS['counts'] == [*wr.CONTINUOUS['none'], 52]
    assert wr.CONTINUOUS['block'] == [*wr.CONTINUOUS['none'], 52, 53, 54, 55]  # rest_missing (55) included as P27
    for name in wr.ARMS:
        level, base = arms[name]['workload_level'], wr.STEM[arms[name]['workload_level']]
        assert arms[name]['workload'] == arms[base]['workload'] and arms[name]['context_continuous'] == \
            arms[base]['context_continuous'] == wr.CONTINUOUS[level]
        assert arms[name]['head_dropout'] in (.1, .2)
    assert cfg['training'] == rc.TRAINING == p29['training'] and cfg['calibration'] == p29['calibration']
    assert cfg['export'] == p29['export'] and cfg['device'] == p29['device']
    q = cfg['queue']
    assert (q['task_timeout_seconds'], q['family_budget_seconds'], q['max_attempts']) == (3600, 36000, 2)
    assert cfg['comparison'][wr.CONTRAST_KEY] == wr.CONTRASTS and len(wr.CONTRASTS) == 12
    assert {k for k in cfg['comparison'] if k.endswith('_contrasts')} == {wr.CONTRAST_KEY}


@pytest.mark.parametrize('edit', [
    lambda c: c['arms'][2].update(head_dropout=.3),
    lambda c: c['arms'][5].update(workload=['prior_records_per100', 'rest_missing']),
    lambda c: c['arms'][6].update(context_continuous=wr.CONTINUOUS['none']),
    lambda c: c['arms'][9].update(context_continuous=wr.CONTINUOUS['block'][:-1]),  # drop the missing indicator
    lambda c: c['arms'][3].update(lr=1e-3),                                          # unchecked extra parameter
    lambda c: c['arms'][4].update(head_dropout=.1),                                  # old arm altered
    lambda c: c['arms'][4].update(workload_dropout=True),
    lambda c: c['arms'][8].update(priority=2),
    lambda c: c['arms'].pop(),
    lambda c: c['arms'].insert(1, c['arms'].pop(2)),
    lambda c: c['arms'].append({**c['arms'][1], 'name': 'periodic_dropout', 'dropout_head': True}),
    lambda c: c.update(seeds=[0, 1, 2]),
    lambda c: c.update(seeds=[3, 4, 5, 6]),
    lambda c: c.update(seeds=[7, 6, 5, 4, 3]),
    lambda c: c.update(seeds=[3, 4, 5, 6, 8]),
    lambda c: c.update(seeds=[8, 9, 10, 11, 12]),                                    # repeat without the tag
    lambda c: c.update(repeat_tag='repeat2_seeds_8_12'),                             # tag with first seeds
    lambda c: c.update(seeds=[13, 14, 15, 16, 17], repeat_tag='repeat3'),
    lambda c: c.update(fits=49),
    lambda c: c.update(training={**c['training'], 'epochs': 40}),
])
def test_config_check_refuses_any_family_or_seed_drift(edit):
    cfg = copy.deepcopy(load())
    edit(cfg)
    with pytest.raises(ValueError):
        rc.config_check(cfg)


def test_second_repetition_only_with_explicit_tag_and_same_arms():
    cfg = copy.deepcopy(load())
    cfg.update(seeds=[8, 9, 10, 11, 12], repeat_tag='repeat2_seeds_8_12')
    assert rc.config_check(cfg)['seeds'] == [8, 9, 10, 11, 12]
    cfg['arms'][2]['head_dropout'] = .15
    with pytest.raises(ValueError):
        rc.config_check(cfg)


@pytest.mark.parametrize('path', [BASE, COMBO, DROPOUT_RUN])
def test_other_protocols_keep_seeds_0_2(path):
    cfg = json.loads(path.read_text())
    assert rc.config_check(copy.deepcopy(cfg))['seeds'] == [0, 1, 2]
    assert wr.check_contrasts(cfg) is None
    for change in ({'seeds': [3, 4, 5, 6, 7]}, {'repeat_tag': 'repeat2_seeds_8_12'},
                   {'seeds': [8, 9, 10, 11, 12], 'repeat_tag': 'repeat2_seeds_8_12'}):
        with pytest.raises(ValueError):
            rc.config_check({**copy.deepcopy(cfg), **change})


def test_contrasts_change_exactly_one_factor_with_candidate_minus_reference():
    cfg = load()
    arms = {a['name']: a for a in cfg['arms']}
    seen = set()
    for c in wr.check_contrasts(cfg):
        (cw, cd, _), (rw, rd, _) = wr.factors(arms[c['candidate']]), wr.factors(arms[c['reference']])
        if c['isolates'] == 'head_dropout':
            assert cw == rw == c['held']['workload'] and rd == 0. and cd > 0.
        else:
            assert cd == rd == c['held']['head_dropout'] and rw == 'none' and cw in ('counts', 'block')
        assert 'control' not in (c['candidate'], c['reference'])  # flatten control only via vs-control block
        assert c['name'] == f"{c['candidate']}_vs_{c['reference']}"
        seen.add((c['candidate'], c['reference']))
    assert len(seen) == 12


@pytest.mark.parametrize('edit', [
    lambda cs: cs.append(dict(cs[0], name='again')),
    lambda cs: cs[0].update(candidate='periodic', reference='periodic_d10'),   # reversed sign
    lambda cs: cs[6].update(reference='control'),                               # architecture change
    lambda cs: cs[8].update(reference='periodic'),                              # workload AND dropout change
    lambda cs: cs[1].update(held={'workload': 'counts'}),
    lambda cs: cs[0].update(winner=True),
    lambda cs: cs.pop(),
])
def test_contrast_list_is_fixed(edit):
    cfg = copy.deepcopy(load(registered=True))
    edit(cfg['comparison'][wr.CONTRAST_KEY])
    with pytest.raises(ValueError):
        wr.check_contrasts(cfg)


def test_modules_pinned_and_config_check_needs_no_torch():
    assert {'pitchmdp/direct_workload_dropout_model.py', 'pitchmdp/direct_workload_dropout_registry.py'} \
        <= set(rc.MODEL_SOURCES)
    assert 'pitchmdp/direct_workload_dropout_registry.py' in sc.SCORER_SOURCES
    code = ('import sys, json; sys.path.insert(0, "scripts"); import run_cuda_direct as rc; '
            f'rc.config_check(json.load(open({str(RUN)!r}))); import score_cuda_direct; '
            'bad = [m for m in ("torch", "pandas", "scipy", "sklearn") if m in sys.modules]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], cwd=PROJECT, check=True)


# ------------------------------------------------------------------ networks / training

def inputs(arm, n, seed):
    a = tf.split_arrays(n, seed, 0)
    return rc.arm_inputs(a, arm, COLUMNS), a['y']


def small_fit(model, arm):
    train, y = inputs(arm, 300, 11)
    early, ey = inputs(arm, 90, 12)
    return model.fit(train, y, early, ey, epochs=3, patience=2, batch_size=64)


@pytest.mark.parametrize('level', ['none', 'counts', 'block'])
def test_zero_dropout_reproduces_old_arm_training_and_logits_bitwise(level):
    cfg = load()
    old = rc.arm_spec(cfg, wr.STEM[level])
    base = small_fit(rc.build_model(old, 5, 'cpu'), old)  # exact old code path (combo / base periodic)
    assert type(base) is DirectPeriodicModel
    ours = small_fit(PeriodicHeadDropoutModel(5, wr.CONTINUOUS[level], 0., 'cpu'), old)
    assert [h['earlystop_conditional_nll'] for h in base.report['history']] == \
        [h['earlystop_conditional_nll'] for h in ours.report['history']]
    assert all(torch.equal(a, b) for a, b in zip(base.net.parameters(), ours.net.parameters()))
    x, _ = inputs(old, 50, 13)
    assert x[2].shape[1] == 52 + len(old['workload'])
    assert np.array_equal(base.logits(x), ours.logits(x))
    for d in (.1, .2):
        spec = rc.arm_spec(cfg, wr.arm_name(level, d))
        drop = small_fit(rc.build_model(spec, 5, 'cpu'), spec)
        assert drop.report['parameter_count'] == base.report['parameter_count']
        assert [type(m).__name__ for m in drop.net.head] == \
            ['Linear', 'GELU', 'Dropout', 'Linear', 'GELU', 'Dropout', 'Linear']
        assert [m.p for m in drop.net.modules() if isinstance(m, torch.nn.Dropout)] == [d, d]  # head only
        assert drop.net.config['n_context'] == base.net.config['n_context']
        assert drop.report['history'][0]['train_weighted_nll'] != base.report['history'][0]['train_weighted_nll']


@pytest.mark.parametrize('name', ['periodic_counts_d10', 'periodic_block_d20'])
def test_checkpoint_roundtrip_and_strict_loader(tmp_path, name):
    cfg = load()
    spec = rc.arm_spec(cfg, name)
    model = small_fit(rc.build_model(spec, 3, 'cpu'), spec)
    model.save(tmp_path / 'm.pt')
    again = rc.load_model(spec, tmp_path / 'm.pt', 'cpu')
    x, _ = inputs(spec, 30, 15)
    assert np.array_equal(model.logits(x), again.logits(x))
    assert again.architecture['head_dropout'] == spec['head_dropout']
    for other in ('periodic_counts_d20', 'periodic_block_d10', 'periodic_d10', 'periodic_counts', 'periodic_block'):
        if other != name:
            with pytest.raises(ValueError):
                rc.load_model(rc.arm_spec(cfg, other), tmp_path / 'm.pt', 'cpu')
    with pytest.raises(ValueError):
        rc.build_model(spec, 3, 'cpu', ['x'] * 52)


def test_runner_smoke_task_cpu_and_seed_refusal(export, tmp_path):
    path, digest, _ = export
    cfg = load()
    dest = rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'periodic_block_d20', 3, 'cpu', {}, smoke=True)
    fit = json.loads((dest / 'fit.json').read_text())
    assert fit['context_width'] == 56
    identity = json.loads((dest / 'identity.json').read_text())
    assert identity['seed'] == 3 and identity['arm']['head_dropout'] == .2
    assert set(identity['source_sha256']) == {rc.SELF, *rc.MODEL_SOURCES}
    for seed in (0, 2, 8):
        with pytest.raises(ValueError, match='Unregistered seed'):
            rc.run_task(cfg, path, digest, tmp_path / 'smoke', 'periodic_block_d20', seed, 'cpu', {}, smoke=True)
    with pytest.raises(ValueError, match='REGISTERED'):
        rc.run_task(cfg, path, digest, tmp_path / 'real', 'periodic_block_d20', 3, 'cpu', {})


# ------------------------------------------------------------------ scorer

@pytest.fixture
def few_draws(monkeypatch):
    monkeypatch.setattr(sc, 'DRAWS', 200)  # synthetic speed only; the config/scorer default stays 10,000


def test_scorer_reports_all_contrasts_with_sign(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    assert sc.DRAWS == 200
    result = tf.run_score(tmp_path, cfg, export, tf.family(tmp_path, cfg, digest, splits, same_as_control=()))
    assert result['decision'] is None and result['ranking'] is None
    assert 'factorial_contrasts' not in result and 'regularization_contrasts' not in result
    assert result['seed_block']['seeds'] == [3, 4, 5, 6, 7] and result['seed_block']['repeat_tag'] is None
    for split in ('dev', 'mlb_dev'):
        arms = result['splits'][split]
        assert list(arms) == list(wr.ARM_ORDER)
        block = result['workload_dropout_contrasts'][split]
        assert list(block) == [c['name'] for c in wr.CONTRASTS]
        vs = lambda a: arms[a]['paired_vs_control']['ensemble']['nll']['delta']  # noqa: E731
        for c in wr.CONTRASTS:
            one = block[c['name']]
            assert (one['isolates'], one['held']) == (c['isolates'], c['held'])
            assert set(one['seeds']) == {'3', '4', '5', '6', '7'}
            assert one['ensemble']['nll']['delta'] == pytest.approx(vs(c['candidate']) - vs(c['reference']), abs=1e-9)
    assert 'periodic_block_d20_vs_periodic_d20' in (tmp_path / 'score' / 'table_ko.md').read_text()


def test_scorer_refuses_incomplete_wrong_config_or_missing_contrasts(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=(), skip=(('periodic_block_d20', 7),))
    with pytest.raises(ValueError, match='Incomplete'):
        tf.run_score(tmp_path, cfg, export, fam)
    repeat = copy.deepcopy(cfg)
    repeat.update(seeds=[8, 9, 10, 11, 12], repeat_tag='repeat2_seeds_8_12')
    other = tf.family(tmp_path, repeat, digest, splits, same_as_control=(), name='repeat')
    with pytest.raises(ValueError, match='Incomplete'):  # seed mismatch: first-block config, repeat tasks
        tf.run_score(tmp_path, cfg, export, other)
    renamed = copy.deepcopy(cfg)
    renamed['seeds'] = [8, 9, 10, 11, 12]
    renamed['repeat_tag'] = 'repeat2_seeds_8_12'
    renamed['purpose'] += ' (edited)'
    with pytest.raises(ValueError, match='identity config_sha256'):  # same seeds, different config digest
        tf.run_score(tmp_path, renamed, export, other)
    bare = copy.deepcopy(cfg)
    del bare['comparison'][wr.CONTRAST_KEY]
    with pytest.raises(ValueError, match='fixed contrasts'):
        tf.run_score(tmp_path, bare, export, tmp_path / 'no-family-needed')
    assert not (tmp_path / 'score').exists()
