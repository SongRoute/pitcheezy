"""Reviewer tests for the P35 PLE wiring (OPUS-review-p35-wiring-6be28eb): mutants that the author's tests missed.

R1  the scorer refuses two bins hashes BEFORE any probability is read (mutant: check moved after score_family).
R2  a contrast labelled ``bundle`` must change several factors and a non-bundle row exactly one, even when the
    name-keyed CHANGED set agrees (mutant: the cardinality clause removed).
Synthetic CPU only; no real data, no GPU.
"""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cuda_ple_wiring as w  # noqa: E402  shared synthetic export / fake-family helpers
from test_cuda_ple_wiring import export, few_draws, one_thread  # noqa: E402,F401  fixtures

sc, tf, reg = w.sc, w.tf, w.reg


def test_scorer_checks_one_bins_hash_before_any_probability_is_read(export, tmp_path, few_draws, monkeypatch):
    path, digest, splits = export
    cfg = w.load(registered=True)
    fam = w.with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg,
                         digests={(w.SINGLE, 31): 'ef' * 32})

    def no_read(*args, **kwargs):
        raise AssertionError('a probability was read before the bins-hash check')
    monkeypatch.setattr(sc, 'member', no_read)
    with pytest.raises(ValueError, match='one bins hash'):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


@pytest.mark.parametrize('row, label', [(0, 'bundle'), (3, 'embedding_type_single')])
def test_bundle_label_must_match_the_number_of_changed_factors(monkeypatch, row, label):
    """Registry and config agree on a relabelled row (name and CHANGED unchanged): the cardinality rule refuses."""
    contrasts = copy.deepcopy(reg.CONTRASTS)
    contrasts[row]['isolates'] = label
    monkeypatch.setattr(reg, 'CONTRASTS', contrasts)
    cfg = copy.deepcopy(w.load())
    cfg['comparison'][reg.CONTRAST_KEY] = copy.deepcopy(contrasts)
    with pytest.raises(ValueError, match='declared factors'):
        reg.check_contrasts(cfg)


# ------------------------------------------------------------------ W2: effective next to allocated parameter count

@pytest.mark.parametrize('name', w.NEW)
def test_runner_records_the_effective_count_next_to_the_allocated_count(export, tmp_path, name):
    path, digest, splits = export
    cfg = w.load()
    spec = w.rc.arm_spec(cfg, name)
    fit = w.read(w.rc.run_task(cfg, path, digest, tmp_path / 'smoke', name, 30, 'cpu', {}, smoke=True) / 'fit.json')
    bins = w.plm.compute_bins(w.train_inputs(splits, spec, w.rc.SMOKE['train']), reg.CONTINUOUS)
    allocated, effective = w.plm.parameter_count(53, 38, 6, reg.CONTINUOUS, bins, tabm=name == w.TABM)
    assert (fit['parameter_count'], fit['ple_effective_parameter_count']) == (allocated, effective)
    assert fit['ple_effective_parameter_count'] == fit['report']['network']['ple_effective_parameter_count']
    assert allocated == w.ALLOCATED[name] and allocated - effective == 8 * (81 * 48 - sum(fit['ple_n_bins']))


def test_effective_record_refuses_missing_or_inconsistent_counts():
    counts = [48] * 80 + [0]
    network = w.network_record(counts)
    assert reg.effective_record(network, w.ALLOCATED[w.SINGLE]) == {
        'ple_effective_parameter_count': w.ALLOCATED[w.SINGLE] - 8 * 48}
    for allocated, edit in ((None, None), (float(w.ALLOCATED[w.SINGLE]), None), (w.ALLOCATED[w.TABM], None),
                            (w.ALLOCATED[w.SINGLE], lambda n: n.pop('ple_effective_parameter_count')),
                            (w.ALLOCATED[w.SINGLE], lambda n: n.update(ple_effective_parameter_count=w.ALLOCATED[w.SINGLE])),
                            (w.ALLOCATED[w.SINGLE], lambda n: n.update(ple_effective_parameter_count=1.5))):
        bad = copy.deepcopy(network)
        if edit:
            edit(bad)
        with pytest.raises(ValueError, match='effective parameter count'):
            reg.effective_record(bad, allocated)


def test_scorer_checks_and_reports_the_effective_count(export, tmp_path, few_draws):
    path, digest, splits = export
    cfg = w.load(registered=True)
    fam = w.with_records(tf.family(tmp_path, cfg, digest, splits, same_as_control=()), cfg)
    tasks = tf.run_score(tmp_path, cfg, export, fam)['provenance']['tasks']
    for name in w.NEW:
        assert tasks[name]['31']['ple_parameters'] == {'allocated': w.ALLOCATED[name],
                                                       'effective': w.ALLOCATED[name] - 8 * 48}
    assert 'ple_parameters' not in tasks['tabm_residual_counts']['31']
    for i, edit in enumerate((lambda f: f.update(ple_effective_parameter_count=f['parameter_count']),
                              lambda f: f.pop('ple_effective_parameter_count'), lambda f: f.pop('parameter_count'))):
        dest = w.rc.task_dir(fam, w.TABM, 30, False)
        fit = w.read(dest / 'fit.json')
        good = copy.deepcopy(fit)
        edit(fit)
        w.rc.write_json_atomic(dest / 'fit.json', fit)
        tf.reseal(dest)
        with pytest.raises(ValueError, match=f'{w.TABM}/30: PLE bins record'):
            tf.run_score(tmp_path, cfg, export, fam, dest=f'bad{i}')
        w.rc.write_json_atomic(dest / 'fit.json', good)
        tf.reseal(dest)
