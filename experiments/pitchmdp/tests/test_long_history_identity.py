"""Synthetic identity-only long-stream exclusion and temporal-boundary tests."""
import numpy as np
import pandas as pd
import pytest

from pitchmdp.matrix_features import MatrixHistoryStore
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from test_matrix_long_history import observations, Context


def identity_frame():
    plan = [
        ('2024-04-01', 10, 1, [7, 7]),
        ('2024-04-01', 11, 1, [8]),
        ('2024-04-01', 11, 2, [7]),
        ('2024-04-02', 20, 1, [7, 8]),  # Excluded date/game prefix.
        ('2024-04-02', 20, 2, [7, 7]),
        ('2024-04-02', 20, 3, [7, 8]),  # Excluded interior PA.
        ('2024-04-02', 20, 4, [8]),
        ('2024-04-02', 20, 5, [7]),
        ('2024-04-02', 21, 1, [7, 8]),  # Other same-date game prefix.
        ('2024-04-02', 21, 2, [7]),
        ('2024-04-03', 30, 1, [7, 7]),
        ('2024-04-03', 30, 2, [8]),
    ]
    template = observations().iloc[0].to_dict()
    rows = []
    for date, game, pa, batters in plan:
        for pitch, batter in enumerate(batters, 1):
            rows.append({**template, 'game_date': pd.Timestamp(date), 'game_pk': game,
                'at_bat_number': pa, 'pitch_number': pitch, 'batter': batter,
                'split': 'train' if game < 20 else 'dev',
                'balls': pitch-1, 'effective_speed': 85.+len(rows),
                'pitch_type': 'FF' if pitch == 1 else 'SL'})
    return pd.DataFrame(rows)


def observed(store, row):
    indices = store.long_indices([row])[0]
    return indices[indices >= 0].tolist()


def test_excluded_prefix_and_interior_links_obey_identity_date_game_and_pa_boundaries():
    frame = identity_frame()
    base = MatrixHistoryStore.from_frame(frame, history_length=5)
    arrays = [base.physical.copy(), base.type_channels.copy(), base.outcome_channels.copy(), base.indices.copy()]
    store = MatrixLongHistoryStore(base, 128)
    assert store.base is base and store.frame is base.frame and store.normalizer is base.normalizer
    assert len(store.frame) == len(frame)
    for before, after in zip(arrays, (base.physical, base.type_channels, base.outcome_channels, base.indices)):
        np.testing.assert_array_equal(before, after)
    assert observed(store, 6) == [0, 1, 3]
    assert observed(store, 7) == [0, 1, 3]  # Current PA's first pitch still excluded.
    assert observed(store, 10) == [2]  # Other batter must also skip interior exclusion.
    assert observed(store, 11) == [0, 1, 3, 6, 7]
    assert observed(store, 14) == [0, 1, 3]  # Never game20 on the same date.
    assert observed(store, 15) == [0, 1, 3, 6, 7, 11, 14]
    assert observed(store, 16) == observed(store, 15)
    assert observed(store, 17) == [2, 10]
    excluded = np.flatnonzero(store.excluded_long_rows)
    np.testing.assert_array_equal(excluded, [4, 5, 8, 9, 12, 13])
    for name in ('previous_global', 'previous_game', 'date_root', 'game_root'):
        links = getattr(store, name)
        assert not np.isin(links, excluded).any()
        assert np.all(links[excluded] == -1)
    for query in np.flatnonzero(~store.excluded_long_rows):
        result = observed(store, query)
        assert len(result) == len(set(result))
        assert not set(result).intersection(excluded)
        assert all(frame.iloc[i].batter == frame.iloc[query].batter for i in result)
    h5, mask, _, _ = store.gather([7, 16])
    direct, direct_mask = base.gather([7, 16])
    np.testing.assert_array_equal(h5, direct)
    np.testing.assert_array_equal(mask, direct_mask)
    report = store.report()
    assert report['version'] == 'batter_dual_stream_v2'
    assert report['source_identity_audit'] == {
        'excluded_pa': 3, 'excluded_rows': 6,
        'by_split': {'dev': {'excluded_pa': 3, 'excluded_rows': 6}}}


def test_ambiguous_queries_rejected_for_all_arms_without_filtering_base_rows():
    store = MatrixLongHistoryStore.from_frame(identity_frame(), long_length=128)
    for length in (0, 32, 128):
        arm = store.with_length(length)
        assert arm.base is store.base and arm.excluded_long_rows is store.excluded_long_rows
        with pytest.raises(ValueError, match='Ambiguous-batter'):
            arm.long_indices([5])
        with pytest.raises(ValueError, match='Ambiguous-batter'):
            LazyPitchBatch(arm, Context(), [6, 8]).gather()
        # Unchanged base/H5 access remains possible; the query guard is LONG-only.
        assert arm.base.gather([5])[0].shape[0] == 1
    assert observed(store.with_length(0), 6) == []


def test_empty_retained_history_and_entirely_ambiguous_source_are_well_defined():
    frame = identity_frame().iloc[4:8].copy().reset_index(drop=True)
    frame['split'] = 'train'
    store = MatrixLongHistoryStore.from_frame(frame, long_length=128)
    assert observed(store, 2) == [] and observed(store, 3) == []
    h5, valid, long, mask = store.gather([3])
    assert valid.sum() == 2  # Current PA H5 retains its own first pitch.
    assert not mask.any() and not long.any()
    assert store.base.indices[3, -1] == 2
    all_excluded = MatrixLongHistoryStore.from_frame(frame.iloc[:2].copy(), long_length=32)
    assert all_excluded.excluded_long_rows.all()
    assert all_excluded.report()['source_identity_audit']['excluded_rows'] == 2
    assert all_excluded.long_indices([]).shape == (0, 128)


def test_links_never_use_supported_pa_or_outcomes_and_later_identity_changes_do_not_reach_backwards():
    frame = identity_frame()
    store = MatrixLongHistoryStore.from_frame(frame, long_length=128)
    changed = frame.copy()
    changed['supported_pa'] = ~changed.supported_pa
    changed['description'] = 'hit_by_pitch'
    changed['events'] = 'home_run'
    other = MatrixLongHistoryStore.from_frame(changed, normalizer=store.normalizer,
        type_vocabulary=store.base.type_vocabulary, long_length=128)
    for name in ('previous_global', 'previous_game', 'date_root', 'game_root', 'excluded_long_rows'):
        np.testing.assert_array_equal(getattr(store, name), getattr(other, name))
    # Repairing a later ambiguous PA must not change an earlier clean query.
    later_changed = frame.copy(); later_changed.loc[9, 'batter'] = 7
    later = MatrixLongHistoryStore.from_frame(later_changed, normalizer=store.normalizer,
        type_vocabulary=store.base.type_vocabulary, long_length=128)
    for row in (0, 1, 2, 3, 6, 7):
        assert observed(later, row) == observed(store, row)
    # Current PA is wholly withheld even when a later pitch changes its ID.
    with pytest.raises(ValueError, match='Ambiguous-batter'):
        store.long_indices([8])
    assert observed(later, 8) == [0, 1, 3, 6, 7]
