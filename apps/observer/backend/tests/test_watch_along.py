"""Watch-along replay API on a small synthetic precomputed game (no T7, no model)."""
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.main import create_app
from observer_app.watch_along import SCHEMA, WatchAlong

GAME = 900001


def decision(index, pa, number, ready=True):
    return {'index': index, 'pa_id': f'{GAME}:{pa}', 'at_bat_number': pa, 'pitch_number': number,
            'situation': {'inning': 1, 'half': 'Top', 'outs': 0, 'balls': number - 1, 'strikes': 0, 'bases': 0,
                          'home_score': 0, 'away_score': 0},
            'pitcher': {'id': 1, 'name': 'P One', 'hand': 'R'}, 'batter': {'id': 2, 'name': 'B Two', 'side': 'L'},
            'status': 'SUPPORTED' if ready else 'UNSUPPORTED_UNKNOWN_PITCHER', 'root_status': None, 'path': 'ledger',
            'pre': {'status': 'ready' if ready else 'unsupported', 'reason': None if ready else '학습 기록이 없는 투수',
                    'recommendation': {'candidates': [{'rank': 1, 'pitch_type': 'SL', 'pitch_label': '슬라이더',
                                                       'zone_id': 'low_left', 'zone_label': '낮은 왼쪽',
                                                       'target': {'x': -.55, 'z': 1.9},
                                                       'detail': {'probability': .5, 'reference_probability': .45,
                                                                  'kernel_mass': .2, 'kernel_ess': 150.}}]} if ready else None},
            'actual': {'pitch_type': 'FF', 'pitch_label': '포심', 'result_label': '볼', 'event_label': None,
                       'play_text': None, 'speed_mph': 95.1, 'x': .9, 'z': 3.1, 'zone_label': '존 바깥'},
            'we': {'home_before': .5, 'home_after': .48, 'home_delta': -.02, 'batting_delta': .02}}


def dataset(**changes):
    data = {'schema': SCHEMA, 'badge': '검증 전 실험 버전 · 위치는 실제 투구 분포 근사',
            'game': {'game_pk': GAME, 'date': '2026-09-29', 'game_type': 'F', 'away_team': 'Away Club',
                     'home_team': 'Home Club', 'final': {'away': 3, 'home': 1}},
            'policy': {'name': 'ARM-B', 'identity_sha256': 'a6dffaea' + '0' * 56, 'tau': .1, 'validated': False,
                       'note': '검증 전'},
            'location': {'basis': 'realized_delivery_proxy', 'evaluated': False, 'note': '근사',
                         'zone_bounds': {'bottom': 1.6, 'top': 3.39}},
            'we': {'note': '표시용'}, 'coverage': {'pitch_decisions': 3, 'ready': 2, 'ready_share_of_pitches': .667},
            'decisions': [decision(0, 1, 1), decision(1, 1, 2), decision(2, 2, 1, ready=False)]}
    return data | changes


def write(tmp_path, data, name=f'{GAME}.json'):
    (tmp_path/name).write_text(json.dumps(data, ensure_ascii=False))
    return tmp_path


def test_timeline_hides_every_post_pitch_field(tmp_path):
    view = WatchAlong(write(tmp_path, dataset())).timeline(GAME)
    assert 'final' not in view['game'] and view['badge'].startswith('검증 전 실험 버전')
    text = json.dumps(view, ensure_ascii=False)
    for leaked in ('"actual"', '"we"', '95.1', 'home_before', '"final"'):
        assert leaked not in text
    assert [d['pre']['status'] for d in view['decisions']] == ['ready', 'ready', 'unsupported']
    assert view['decisions'][2]['pre']['reason'] == '학습 기록이 없는 투수'


def test_api_reveal_one_pitch_and_errors(tmp_path):
    write(tmp_path, dataset())
    with TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path)) as client:
        assert client.get('/api/watch/games').json() == {'games': [{'game_pk': GAME, 'date': '2026-09-29', 'game_type': 'F',
                                                                    'away_team': 'Away Club', 'home_team': 'Home Club',
                                                                    'pitches': 3, 'ready': 2, 'ready_share': .667}]}
        assert 'final' not in client.get('/api/watch/games').text  # the picker must not spoil the score
        assert len(client.get(f'/api/watch/{GAME}').json()['decisions']) == 3
        reveal = client.get(f'/api/watch/{GAME}/reveal/1').json()
        assert set(reveal) == {'index', 'actual', 'we'} and reveal['actual']['pitch_type'] == 'FF'
        assert client.get(f'/api/watch/{GAME}/reveal/3').status_code == 404
        assert client.get('/api/watch/123/reveal/0').status_code == 404


def test_dataset_integrity_is_checked(tmp_path):
    bad = dataset()
    bad['decisions'][1]['index'] = 5
    write(tmp_path, bad)
    with TestClient(create_app(object(), start_worker=False, watch_dir=tmp_path)) as client:
        assert client.get(f'/api/watch/{GAME}').status_code == 503
    write(tmp_path, dataset(schema='other'))
    assert WatchAlong(tmp_path).games() == []
