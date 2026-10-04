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
