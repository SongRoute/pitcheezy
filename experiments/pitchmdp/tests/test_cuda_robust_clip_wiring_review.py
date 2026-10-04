"""Review tests (T104 same_model_review of the robust-clip wiring a23e77b): close two surviving mutants.

R1: the scorer must refuse differing robust-clip statistics records BEFORE any probability is read (score_family /
member never called).  F2: direct_robust_clip_registry.check_family ALONE must refuse a drifted reference arm (the
runner's per-arm loop happens to catch most drifts too, which hid a weakened registry check).  Plus the PREREG §3 m1
column split (transformed = 81 scalars; raw context = 0-10 and 32).  Synthetic CPU only.
same_model_review; Codex_review_pending.
"""
import copy

import pytest

import test_cuda_robust_clip_wiring as w  # noqa: E402  (sys.path set up by that module)
from test_cuda_robust_clip_wiring import export, few_draws, one_thread  # noqa: F401  fixtures
from pitchmdp import direct_robust_clip_registry as reg  # noqa: E402

rc, sc, tf = w.rc, w.sc, w.tf


@pytest.mark.parametrize('field', ['digest', 'train_rows'])
def test_scorer_refuses_two_statistics_records_before_reading_any_probability(export, tmp_path, few_draws,  # noqa: F811
                                                                              monkeypatch, field):
    path, digest, splits = export
    cfg = w.load(registered=True)
    fam = tf.family(tmp_path, cfg, digest, splits, same_as_control=())
    fam = w.with_records(fam, cfg, digests={37: 'ef' * 32}) if field == 'digest' else \
        w.with_records(fam, cfg, train_rows={(w.NEW, 37): 'bb' * 32})

    def boom(*args, **kwargs):
        raise AssertionError('probabilities read before the one-statistics-hash check')
    monkeypatch.setattr(sc, 'score_family', boom)
    monkeypatch.setattr(sc, 'member', boom)
    with pytest.raises(ValueError, match='one statistics hash'):
        tf.run_score(tmp_path, cfg, export, fam)
    assert not (tmp_path / 'score').exists()


@pytest.mark.parametrize('name', ['control', 'residual_counts', 'periodic_residual_counts'])
@pytest.mark.parametrize('edit', [lambda a: a.update(note='x'), lambda a: a.update(priority=2),
                                  lambda a: a.update(width=128.0), lambda a: a.pop('workload')])
def test_registry_family_check_alone_refuses_a_drifted_reference_arm(name, edit):
    cfg = w.load()
    assert reg.check_family(copy.deepcopy(cfg))
    bad = copy.deepcopy(cfg)
    edit(next(a for a in bad['arms'] if a['name'] == name))
    with pytest.raises(ValueError, match='exact registered spec'):
        reg.check_family(bad)


def test_prereg_m1_column_split_matches_the_registry():
    """PREREG §3: 5 past x 8 physics + 41 context = 81 transformed; raw context = 0-10 (game state) and 32 (left_share)."""
    n_context = w.read(w.RUN)['export']['shapes']['n_context'] + len(reg.WORKLOAD)
    assert reg.n_scalars(6) == 81 and len(reg.CONTINUOUS) == 41
    assert sorted(set(range(n_context)) - set(reg.CONTINUOUS)) == [*range(11), 32]
