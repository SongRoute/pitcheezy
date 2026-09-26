"""Boundary tests for the fixed small replay; no frozen results are opened."""
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/a_small_eval.py'
spec = importlib.util.spec_from_file_location('a_small_eval', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_chronological_selection_ignores_support_and_future_outcomes():
    rows = []
    for pitcher in (657277, 554430):
        for date, game in [('2025-04-01', 10), ('2025-04-02', 20), ('2025-04-03', 30),
                           ('2025-07-01', 40), ('2025-07-02', 50), ('2025-07-03', 60), ('2025-07-04', 70)]:
            game += pitcher
            for pitch in (1, 2):
                rows.append({'game_date': date, 'game_pk': game, 'pitcher': pitcher, 'starter_pitcher': pitcher,
                             'at_bat_number': 1, 'pitch_number': pitch,
                             'supported_pa': not (date == '2025-07-01'), 'events': 'home_run' if pitch == 2 else ''})
    frame = pd.DataFrame(rows)
    cfg = {'pitchers_train_rank_order': [657277, 554430], 'train_start': '2025-04-01',
           'train_end': '2025-04-30', 'dev_start': '2025-07-01', 'dev_end': '2025-07-31',
           'train_games_per_pitcher': 3, 'dev_games_per_pitcher': 3}
    selection, train, dev = module.select_full_games(frame, cfg)
    assert len(train) == len(dev) == 12
    assert selection['dev']['657277'][0]['game_date'] == '2025-07-01'
    changed = frame.copy()
    changed['events'] = 'double_play'
    changed['supported_pa'] = ~changed['supported_pa']
    assert module.select_full_games(changed, cfg)[0] == selection
    assert set(train.game_pk).isdisjoint(dev.game_pk)


def test_complete_pa_checks_full_history_and_stable_state():
    common = {'game_pk': 1, 'at_bat_number': 2, 'inning': 3, 'inning_topbot': 'Top', 'outs_when_up': 1,
              'bases': 0, 'home_score': 0, 'away_score': 0, 'batter': 2, 'stand': 'L',
              'pitcher': 657277, 'supported_pa': True, 'pitch_type': 'FF', 'plate_x': 0.1, 'plate_z': 2.5,
              'balls': 0, 'strikes': 0, 'description': 'ball', 'events': ''}
    pa = pd.DataFrame([{**common, 'pitch_number': 1, 'is_pa_terminal': False},
                       {**common, 'pitch_number': 2, 'is_pa_terminal': True}])
    assert module.complete_pa_reason(pa) is None
    poisoned = pa.copy()
    poisoned.loc[1, 'home_score'] = 5
    assert module.complete_pa_reason(poisoned) == 'state_changed_inside_pa'
    incomplete = pa.iloc[1:].copy()
    assert module.complete_pa_reason(incomplete) == 'incomplete_start'


def test_saved_baseline_restores_same_predictions():
    from pitchmdp.model import CountBaseline
    train = pd.DataFrame({'balls': [0, 0, 1], 'strikes': [0, 0, 1], 'stand': ['L'] * 3,
                          'p_throws': ['R'] * 3, 'description': ['ball', 'called_strike', 'foul'],
                          'events': [''] * 3})
    query = train[['balls', 'strikes', 'stand', 'p_throws']].copy()
    before = CountBaseline().fit(train).predict(query)
    after = pickle.loads(pickle.dumps(CountBaseline().fit(train))).predict(query)
    np.testing.assert_array_equal(before, after)
    assert np.allclose(after.sum(1), 1)


def test_cohort_grouping_keeps_substitution_rows_visible():
    common = {'game_pk': 1, 'at_bat_number': 2, 'inning': 3, 'inning_topbot': 'Top', 'outs_when_up': 1,
              'bases': 0, 'home_score': 0, 'away_score': 0, 'batter': 2, 'stand': 'L',
              'supported_pa': True, 'pitch_type': 'FF', 'plate_x': 0.1, 'plate_z': 2.5,
              'balls': 0, 'strikes': 0, 'description': 'ball', 'events': ''}
    full = pd.DataFrame([{**common, 'pitcher': 657277, 'pitch_number': 1, 'is_pa_terminal': False},
                         {**common, 'pitcher': 999, 'pitch_number': 2, 'is_pa_terminal': True}])
    kept, reasons, total = module.cohort_pa_rows(full, [657277])
    assert total == 1 and kept.empty
    assert reasons == {'pitcher_change': 1}


def test_legality_projection_is_shared_with_baseline():
    frame = pd.DataFrame({'outs_when_up': [2, 1], 'bases': [1, 0]})
    raw = np.full((2, 10), 0.1)
    legal = module.legal_baseline(raw, frame)
    assert np.allclose(legal.sum(1), 1)
    assert np.all(legal[:, 9] == 0)
