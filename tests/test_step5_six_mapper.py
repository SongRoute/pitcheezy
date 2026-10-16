"""Generated-only corrected Statcast -> frozen Step 5 raw prestate checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from pitcheezy.seq_shadow.ledger import sha
from pitcheezy.seq_shadow.postcutoff_offline import prepare
from pitcheezy.seq_shadow.step5_six_mapper import map_snapshot
from test_postcutoff_offline import SHA, source


FEATURES = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261010-ws-service-006/'
                'step5-2024-runner-lifecycle-f1-author-001/step5/features.py')


def _features():
    spec = importlib.util.spec_from_file_location('frozen_step5_features_generated', FEATURES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(pitch=1):
    f, M = source()
    f['pitch_type'] = 'FF'
    return prepare(f, SHA, M)['cases'][pitch - 1]['snapshot']


def _resign(snapshot):
    snapshot['snapshot_sha256'] = sha({k: v for k, v in snapshot.items() if k != 'snapshot_sha256'})
    return snapshot


def test_exact_frozen_step5_raw_prestate_first_and_previous_pitch():
    features = _features()
    for pitch in (1, 2, 3):
        snap = _snapshot(pitch)
        mapped = map_snapshot(snap, source_sha256=SHA)
        state = mapped['state']
        raw = features.context_raw(state)
        q = snap['query']
        assert mapped['status'] == 'typed_ff_candidate'
        assert state['official_date'] == q['game_date']
        assert state['score'] == (q['home_score'], q['away_score'])
        assert state['bases'] == tuple(q[k] is not None for k in ('on_1b', 'on_2b', 'on_3b'))
        assert list(raw[:4]) == [q['balls']/3, q['strikes']/2, q['outs_when_up']/2, q['inning']/9]
        assert raw[4] == (1 if state['top'] else -1)*(q['home_score']-q['away_score'])/5
        assert raw[49] == int(pitch > 1)
        assert raw[30 + features.TYPES.index('FF')] == int(pitch > 1)
        assert not raw[11:30].any() and not raw[50:].any()


def test_current_realization_is_outside_typed_prestate_and_snapshot_is_immutable():
    f, M = source(); f['pitch_type'] = 'FF'
    before = prepare(f, SHA, M)['cases'][0]['snapshot']
    changed = f.copy(deep=True)
    changed.loc[0, 'description'] = 'foul'
    after = prepare(changed, SHA, M)['cases'][0]['snapshot']
    assert map_snapshot(before, source_sha256=SHA)['state'] == map_snapshot(after, source_sha256=SHA)['state']
    tampered = deepcopy(before); tampered['query']['balls'] = 2
    with pytest.raises(ValueError, match='snapshot changed'):
        map_snapshot(tampered, source_sha256=SHA)


def test_rejects_unreviewed_history_source_and_non_ff_excludes_without_score():
    snap = _snapshot(2)
    for edit in (
        lambda s: s['query'].update(previous_valid=False),
        lambda s: s['query'].update(previous_type='UNKNOWN'),
        lambda s: s['evidence'].update(source_sha256='b'*64),
        lambda s: s.update(prospective_claim=True),
        lambda s: s['query'].update(game_date='2026-10-13'),
    ):
        changed = deepcopy(snap); edit(changed); _resign(changed)
        with pytest.raises(ValueError):
            map_snapshot(changed, source_sha256=SHA)
    with pytest.raises(ValueError):
        map_snapshot(snap, source_sha256='not-a-digest')
    nonff = deepcopy(snap); nonff['query']['pitch_type'] = 'SI'; _resign(nonff)
    assert map_snapshot(nonff, source_sha256=SHA)['status'] == 'excluded_non_FF'
