"""Point-in-time clocks of the June store: prior-date aggregates vs same-PA history vs current delivery."""
from __future__ import annotations

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import june_calibration_synthetic as syn  # noqa: E402
from pitchmdp.archetypes import STYLE_COLUMNS, HISTORY_COLUMNS  # noqa: E402
from pitchmdp.data import KEY  # noqa: E402
from pitchmdp.matrix_benchmark import predict_streamed  # noqa: E402
from pitchmdp.matrix_features import MatrixHistoryStore  # noqa: E402
from pitchmdp.sequence_data import PhysicalNormalizer  # noqa: E402
from pitchmdp.sequence_delivery import JointDelivery  # noqa: E402
import run_ml_june_calibration as runner  # noqa: E402

CUTOFF = '2025-06-30'


@pytest.fixture(scope='module')
def raw():
    return syn.make_pitches(seed=5)


def june_rows(frame):
    return frame.loc[frame.game_date.dt.strftime('%Y-%m').eq('2025-06')]


def keyed(frame, columns):
    return frame.set_index(KEY)[list(columns)]


def test_truncated_store_is_the_exact_prefix_of_the_full_season_path(raw):
    truncated = runner.build_store_frame(raw, CUTOFF)
    full = syn.full_season_frame(raw)
    assert truncated.game_date.max() <= pd.Timestamp(CUTOFF) and full.game_date.max() > pd.Timestamp(CUTOFF)
    prefix = full.iloc[:len(truncated)].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, prefix)
    assert raw.game_type.eq('S').any() and not truncated.game_type.eq('S').any()


def test_future_rows_cannot_change_june_features(raw):
    base = runner.build_store_frame(raw, CUTOFF)
    changed = raw.copy()
    later = changed.game_date.gt(CUTOFF)
    changed.loc[later, 'description'] = 'hit_into_play'
    changed.loc[later, 'events'] = 'home_run'
    changed.loc[later, 'is_pa_terminal'] = True
    changed.loc[later, 'plate_x'] = 99.0
    pd.testing.assert_frame_equal(runner.build_store_frame(changed, CUTOFF), base)
    dropped = runner.build_store_frame(raw.loc[~later], CUTOFF)
    pd.testing.assert_frame_equal(dropped, base)


def test_same_day_doubleheader_rows_are_excluded_but_earlier_dates_count(raw):
    day = pd.Timestamp('2025-06-05')
    games = sorted(raw.loc[raw.game_date.eq(day), 'game_pk'].unique())
    assert len(games) == 2  # doubleheader
    first, second = games
    base = runner.build_store_frame(raw, CUTOFF)
    same_day = raw.copy()
    rows = same_day.game_pk.eq(second)
    same_day.loc[rows, 'description'] = 'swinging_strike'
    same_day.loc[rows, 'events'] = 'strikeout'
    same_day.loc[rows, 'is_pa_terminal'] = True
    perturbed = runner.build_store_frame(same_day, CUTOFF)
    on_day = base.game_date.eq(day)
    pd.testing.assert_frame_equal(keyed(perturbed.loc[on_day], HISTORY_COLUMNS), keyed(base.loc[on_day], HISTORY_COLUMNS))
    later = base.game_date.gt(day)
    assert not keyed(perturbed.loc[later], STYLE_COLUMNS).equals(keyed(base.loc[later], STYLE_COLUMNS))
    earlier = raw.copy()
    prior = earlier.game_date.eq(pd.Timestamp('2025-06-04'))
    earlier.loc[prior, 'description'] = 'swinging_strike'
    earlier.loc[prior, 'events'] = 'strikeout'
    earlier.loc[prior, 'is_pa_terminal'] = True
    legal = runner.build_store_frame(earlier, CUTOFF)
    assert not keyed(legal.loc[on_day], STYLE_COLUMNS).equals(keyed(base.loc[on_day], STYLE_COLUMNS))


def _normalizer():
    normalizer = PhysicalNormalizer()
    normalizer.fill, normalizer.mean, normalizer.scale = np.zeros(8), np.zeros(8), np.ones(8)
    return normalizer


def _store(frame):
    return MatrixHistoryStore.from_frame(frame, normalizer=_normalizer(), history_length=5, type_vocabulary=('CH', 'FF', 'SL'))


def _pa_rows(frame):
    pa = june_rows(frame).groupby(['game_pk', 'at_bat_number']).head(3).index.to_numpy()[:3]
    assert frame.loc[pa, 'pitch_number'].tolist() == [1, 2, 3]
    return pa


def test_earlier_same_pa_pitch_is_history_but_later_pitch_and_current_outcome_are_not(raw):
    frame = runner.build_store_frame(raw, CUTOFF)
    first, query, later = _pa_rows(frame)
    base = _store(frame).gather([query])[0]
    for row, column, value in ((later, 'plate_x', 42.0), (later, 'description', 'hit_by_pitch'),
                               (query, 'description', 'hit_by_pitch'), (query, 'events', 'home_run')):
        changed = frame.copy()
        changed.loc[row, column] = value
        np.testing.assert_array_equal(_store(changed).gather([query])[0], base)
    changed = frame.copy()
    changed.loc[first, 'plate_x'] = 42.0
    assert not np.array_equal(_store(changed).gather([query])[0], base)
    changed = frame.copy()
    changed.loc[first, 'description'] = 'hit_by_pitch'
    assert not np.array_equal(_store(changed).gather([query])[0], base)
    assert (_store(frame).indices[first] == -1).all()  # the first pitch sees no earlier PA


class LinearModel:
    delivery_temperature = 1.0

    def __init__(self, width):
        self.weights = np.random.default_rng(0).normal(size=(width, 10))

    def logits(self, arrays):
        tokens, valid, context = arrays
        flat = np.column_stack([tokens.reshape(len(tokens), -1), valid, context])
        return flat @ self.weights[:flat.shape[1]]


class ZeroContext:
    def transform(self, frame):
        return np.zeros((len(frame), 2), dtype=np.float32)


def test_current_realized_physics_is_marginalized_by_train_draws(raw):
    frame = runner.build_store_frame(raw, CUTOFF)
    train = frame.loc[frame.split.eq('train')]
    delivery = JointDelivery().fit(train, _normalizer(), draws=7, seed=0)
    first, query, _ = _pa_rows(frame)
    model = LinearModel(2000)
    base = predict_streamed(model, delivery, _store(frame), ZeroContext(), np.asarray([query]))
    changed = frame.copy()
    changed.loc[query, ['plate_x', 'plate_z', 'effective_speed', 'release_spin_rate']] = [9.0, 9.0, 150.0, 5000.0]
    after = predict_streamed(model, delivery, _store(changed), ZeroContext(), np.asarray([query]))
    for left, right in zip(base, after):
        np.testing.assert_array_equal(left, right)
    changed = frame.copy()
    changed.loc[first, 'plate_x'] = 9.0
    earlier = predict_streamed(model, delivery, _store(changed), ZeroContext(), np.asarray([query]))
    assert not np.array_equal(earlier[0], base[0])


def test_store_digest_survives_pickle_and_detects_any_row_change(raw):
    frame = runner.build_store_frame(raw, CUTOFF)
    digest = runner.store_digest(_store(frame))
    assert runner.store_digest(_store(pickle.loads(pickle.dumps(frame, protocol=5)))) == digest
    changed = frame.copy()
    changed.loc[len(frame) - 1, 'plate_z'] += 1e-3
    assert runner.store_digest(_store(changed)) != digest


def test_compact_june_store_keeps_exact_inputs_and_detects_a_broken_history(raw):
    frame = runner.build_store_frame(raw, CUTOFF)
    full = _store(frame)
    queries = june_rows(frame).index.to_numpy()
    compact, positions, kept = runner.compact_store(full, queries)
    assert len(compact.frame) < len(frame) and set(queries) <= set(kept)
    assert not compact.frame.game_date.lt('2025-06-01').any()  # no June PA starts before June
    delivery = JointDelivery().fit(frame.loc[frame.split.eq('train')], _normalizer(), draws=4, seed=0)
    context = ZeroContext()
    report = runner.require_same_inputs(full, compact, queries, positions, context, delivery, chunk=97)
    assert report['rows_checked'] == len(queries)
    np.testing.assert_array_equal(compact.frame[KEY].to_numpy()[positions], frame[KEY].to_numpy()[queries])
    broken = runner.compact_store(full, queries)[0]
    target = positions[np.flatnonzero(full.indices[queries][:, -1] >= 0)[0]]
    broken.indices[target, -1] = broken.indices[target, -1] - 1 if broken.indices[target, -1] > 0 else 1
    with pytest.raises(ValueError, match='history tokens differ'):
        runner.require_same_inputs(full, broken, queries, positions, context, delivery)
