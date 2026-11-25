"""Catcher-setup estimates API on a small synthetic game (no T7, no video)."""
import hashlib
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.main import create_app
from observer_app.setup_estimates import SCHEMA, SetupEstimates, x_band

GAME = 900001
SOURCE = {'repository': 'org/video-module', 'branch': 'feature/intent-v0', 'commit': 'abc1234',
          'path': 'docs/results/game_900001_intent_v0.jsonl', 'video': 'condensed game',
          'selection': 'pitches the condensed game shows'}


def estimate(at_bat, number, x=None, *, reason=None, deepest='plate_feet'):
    hop = lambda source, target: {'source_frame': source, 'target_frame': target, 'method': 'm', 'version': 'v1',
                                  'error': None, 'error_units': None, 'error_status': 'unmeasured', 'evidence': {}}
    base = {'schema_version': 1, 'pitch_id': f'{GAME}:{at_bat}:{number}', 'clip_id': 'clip', 'clip_sha256': '0' * 64,
            'method': {'kind': 'assistant_visual_estimate', 'version': 'v0'},
            'evidence': {'frame_index': 10, 'frame_time': .5},
            'provenance': {'label_source': 'assistant_visual_estimate', 'review_status': 'unreviewed'},
            'is_intent_proxy': True, 'uncertainty': {'value': 15., 'units': 'pixels', 'basis': 'stated'},
            'claims': {'catcher_intent_verified': False, 'independent_ground_truth': False,
                       'physical_plate_coordinates': False, 'accuracy_estimate': None}}
    if reason is not None:
        return base | {'status': 'unavailable', 'unavailable_reason': reason, 'deepest_frame': None, 'blocked_by': None,
                       'points': {}, 'transform_chain': []}
    points = {'image_pixels': {'x': 600., 'y': 300.}}
    if deepest == 'image_pixels':
        return base | {'status': 'estimated', 'unavailable_reason': None, 'deepest_frame': deepest,
                       'blocked_by': 'no_image_plane_calibration', 'points': points, 'transform_chain': []}
    points['annotated_image_zone'] = {'x': .5, 'y': 1.5, 'inside_annotated_quad': True}
    points['plate_feet'] = {'x': x, 'z': .8, 'x_convention': 'statcast_plate_x_catcher_view'}
    base['claims']['physical_plate_coordinates'] = True
    return base | {'status': 'estimated', 'unavailable_reason': None, 'deepest_frame': 'plate_feet',
                   'blocked_by': 'no_batter_zone_bounds', 'points': points,
                   'transform_chain': [hop('image_pixels', 'annotated_image_zone'), hop('annotated_image_zone', 'plate_feet')]}


def write(directory, estimates, **pin):
    raw = ''.join(json.dumps(e) + '\n' for e in estimates).encode()
    (directory/f'{GAME}.jsonl').write_bytes(raw)
    (directory/f'{GAME}.source.json').write_text(json.dumps(SOURCE | {'sha256': hashlib.sha256(raw).hexdigest()} | pin))
    return SetupEstimates(directory)


LINES = [estimate(2, 3, -.5), estimate(1, 4, .1), estimate(3, 1, reason='glove_not_visible'),
         estimate(4, 2, deepest='image_pixels'), estimate(5, 5, 1.2)]


def test_view_serves_only_the_horizontal_position(tmp_path):
    view = write(tmp_path, LINES).view(GAME)
    assert view['schema'] == SCHEMA and view['display'] == view['display'] | {'horizontal': True, 'vertical': False}
    assert view['coverage'] == {'lines': 5, 'estimated': 4, 'unavailable': 1, 'with_plate_x': 3}
    assert [(p['at_bat_number'], p['pitch_number']) for p in view['pitches']] == [(1, 4), (2, 3), (3, 1), (4, 2), (5, 5)]
    assert [p['x_band'] for p in view['pitches']] == ['middle', 'left', None, None, 'off_right']
    assert view['pitches'][2]['unavailable_reason'] == 'glove_not_visible' and view['pitches'][3]['plate_x_feet'] is None
    assert all(p['is_intent_proxy'] is True and p['review_status'] == 'unreviewed' for p in view['pitches'])
    assert 'z' not in json.dumps(view['pitches']) and view['accuracy']['measured'] is False
    assert view['source']['commit'] == 'abc1234' and view['x_convention'] == 'statcast_plate_x_catcher_view'


def test_x_band_follows_the_catcher_view_columns():
    assert [x_band(x) for x in (-.9, -.83, -.3, 0., .27, .28, .83, .84)] == \
        ['off_left', 'left', 'left', 'middle', 'middle', 'right', 'right', 'off_right']


def test_a_changed_file_or_one_bad_line_refuses_the_whole_game(tmp_path):
    setups = write(tmp_path, LINES)
    assert setups.view(GAME)['coverage']['lines'] == 5
    with pytest.raises(KeyError):
        setups.view(GAME + 1)
    with pytest.raises(ValueError, match='pinned source'):
        write(tmp_path, LINES, sha256='0' * 64).view(GAME)
    observed = estimate(1, 1, .0) | {'status': 'observed'}
    with pytest.raises(ValueError, match='never an observation'):
        write(tmp_path, [observed]).view(GAME)
    with pytest.raises(ValueError, match='repeated pitch'):
        write(tmp_path, [estimate(1, 1, .0), estimate(1, 1, .2)]).view(GAME)
    other = estimate(1, 1, .0) | {'pitch_id': '900002:1:1'}
    with pytest.raises(ValueError, match='another game'):
        write(tmp_path, [other]).view(GAME)


def test_api_returns_setup_or_a_clear_refusal(tmp_path, monkeypatch):
    write(tmp_path, LINES)
    monkeypatch.setenv('PITCHEEZY_INTENT_DIR', str(tmp_path))
    client = TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path))
    body = client.get(f'/api/watch/{GAME}/setup').json()
    assert body['coverage']['with_plate_x'] == 3 and body['game_pk'] == GAME
    assert client.get(f'/api/watch/{GAME + 1}/setup').status_code == 404
    (tmp_path/f'{GAME}.jsonl').write_text('{}\n')
    assert client.get(f'/api/watch/{GAME}/setup').status_code == 503
