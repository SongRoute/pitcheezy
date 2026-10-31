"""Synthetic guards for the frozen B PA-history sequence."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import b_pa_history_v2 as b

FAMILIES = {'families': {'fastball': ['FF'], 'breaking': ['SL'], 'offspeed': ['CH']}, 'unknown_family': 'other'}


def rows():
    records = []
    for game, date, pitches in [(10, '2025-04-01', ['FF', 'SL']), (20, '2025-04-07', ['FF', 'CH']),
                                (30, '2025-04-13', ['SL', 'FF'])]:
        for i, pitch in enumerate(pitches, 1):
            records.append({'game_pk': game, 'at_bat_number': 1, 'pitch_number': i,
                            'game_date': date, 'split': 'train', 'pitch_type': pitch,
                            'pitcher': 657277,
                            'balls': 0, 'strikes': i-1, 'stand': 'R', 'p_throws': 'L',
                            'description': 'ball' if game == 10 else 'called_strike', 'events': None,
                            'outs_when_up': 0, 'bases': 1})
    return b.add_prior(pd.DataFrame(records), FAMILIES)


def test_prior_is_same_pa_and_complete():
    frame = rows()
    assert frame.previous_family.tolist() == ['NONE', 'fastball', 'NONE', 'fastball', 'NONE', 'breaking']
    broken = frame.drop(index=0)
    with pytest.raises(ValueError, match='Incomplete'):
        b.add_prior(broken, FAMILIES)


def test_parent_fallback_and_disk_checkpoint(tmp_path):
    frame = rows()
    model = b.PAHistoryModel(.5).fit(frame)
    query = frame.copy()
    query.loc[0, 'previous_family'] = 'NONE'
    query.loc[1, 'previous_family'] = 'unseen'
    p, parent, _, seen = model.predict_with_origin(query)
    np.testing.assert_array_equal(p[:2], parent[:2])
    assert not seen[:2].any() and seen[3]
    path = tmp_path / 'checkpoint.json'
    b.write_json(path, b.checkpoint(model))
    restored = b.restore(json.loads(path.read_text()))
    np.testing.assert_array_equal(restored.predict(query), p)


def test_forward_fold_rejects_future_fit():
    frame = rows()
    manifest = {str(i): [{'game_pk': game, 'game_date': date}] for i, (game, date) in enumerate(
        [(10, '2025-04-01'), (20, '2025-04-07'), (30, '2025-04-13')])}
    spec = {'selection': {'train': manifest}, 'training_folds': [
        {'fit_game_ids': [10], 'validation_game_ids': [20]},
        {'fit_game_ids': [10, 20], 'validation_game_ids': [30]}]}
    b.validate_folds(frame, spec)
    spec['training_folds'][0] = {'fit_game_ids': [20], 'validation_game_ids': [10]}
    with pytest.raises(ValueError, match='forward'):
        b.validate_folds(frame, spec)


def test_selection_ties_and_two_block_veto():
    def block(y, a0, a1):
        def matrix(prob):
            p = np.full((len(y), 10), (1-prob)/9)
            p[np.arange(len(y)), y] = prob
            return p
        return {'y': np.asarray(y), 'predictions': {0.0: matrix(a0), .25: matrix(a1)}}
    selected, best, _ = b.choose_alpha([block([0], .5, .5), block([1], .5, .5)], [0.0, .25])
    assert (selected, best) == (0.0, 0.0)
    selected, best, _ = b.choose_alpha([block([0]*10, .4, .6), block([1], .6, .5)], [0.0, .25])
    assert best == .25 and selected == 0.0  # pooled gain, second block worsens
    selected, best, _ = b.choose_alpha([block([0], .4, .6), block([1], .4, .5)], [0.0, .25])
    assert (selected, best) == (.25, .25)


def test_dev_gate_zero_before_dev_read(tmp_path, monkeypatch):
    report = tmp_path / 'report'
    report.mkdir()
    (report / 'train_selection.json').write_text(json.dumps({'experiment_id': 'EXP-B-PAHISTORY-002',
        'phase': 'train_selected', 'selected_alpha': 0.0, 'gate': 'rejected'}))
    (report / 'checkpoint.json').write_text('{}')
    monkeypatch.setattr(b, 'REPORT', report)
    monkeypatch.setattr(b, 'require_committed', lambda path: None)
    with pytest.raises(ValueError, match='no DEV'):
        b.dev_gate({'experiment_id': 'EXP-B-PAHISTORY-002', 'parent_selection_manifest': 'unused',
                    'parent_common_comparison': 'unused'})


def test_committed_gate_rejects_changed_bytes(monkeypatch):
    monkeypatch.setattr(b, 'git_bytes', lambda relative: b'old committed content')
    with pytest.raises(ValueError, match='Uncommitted or changed prerequisite'):
        b.require_committed(b.CONFIG)
