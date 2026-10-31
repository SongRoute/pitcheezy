"""Inning decisions are explicit historical records, separate from PA replay."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.dataset import DemoDataset
from observer_app.main import create_app
from observer_app.service import ObserverService
from observer_app.store import Store


ROOT = Path(__file__).parents[4]
SOURCE = ROOT / 'results/EXP-C-INNING-001/conditional_keep_777063.json'
ANCHOR = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/C-ROSTER-001/decision_anchor_777063.json')


class NoInference:
    ready = False


@pytest.fixture
def service(tmp_path):
    state = dict(inning=1, half='Top', outs=0, bases=0, home_score=0, away_score=0,
                 balls=0, strikes=0)
    pa = {'id': 1, 'batter_label': '타자', 'pitcher_label': '투수', 'batter_stand': 'R',
          'inning': 1, 'half': 'Top', 'zone_bounds': {'bottom': 1.5, 'top': 3.5},
          'repertoire_counts': {'FF': 100},
          'pitches': [{'id': 'private-pitch', 'pitch_number': 1, 'state': state,
                       'request': {'future_sentinel': 'never-public'}, 'context_notes': [],
                       'actual': {'pitch_type': 'FF', 'x': 0.0, 'z': 2.5,
                                  'speed_mph': 90.0, 'description': 'future_sentinel',
                                  'event': 'field_out'}}],
          'terminal_state': state | {'outs': 1}}
    data = {'schema_version': 1, 'games': [{'id': 10, 'date': '2025-08-17',
            'home_team': 'SF', 'away_team': 'TB', 'title': 'TB @ SF',
            'plate_appearances': [pa]}]}
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(data))
    (tmp_path / 'dataset_manifest.json').write_text(json.dumps({
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    return ObserverService(DemoDataset(path), NoInference(), Store(tmp_path / 'observer.sqlite3'))


@pytest.fixture
def imported(service):
    service.inning_decisions.import_result(SOURCE, ANCHOR)
    with TestClient(create_app(service)) as client:
        catalog = client.get('/api/inning-decisions', params={'game_id': 777063})
        assert catalog.status_code == 200
        decision = catalog.json()['decisions'][0]
    return decision


def test_list_resolve_identity_restart_and_pa_privacy(service, imported):
    context = imported['context']
    linkage = context['linkage']
    identity = {key: linkage[key] for key in ('game_pk', 'anchor_kind', 'anchor_time_utc',
                'anchor_action_index', 'first_observed_pitch_id')}
    canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    expected_id = 'inning-decision-' + hashlib.sha256(canonical.encode()).hexdigest()
    assert imported['decision_id'] == expected_id
    assert imported['revision'] == 1
    assert context['phase'] == 'before_pitching_change'
    assert linkage['anchor_time_utc'].endswith('Z') and '.827000Z' in linkage['anchor_time_utc']

    restarted = ObserverService(service.dataset, NoInference(), Store(service.store.path))
    with TestClient(create_app(restarted)) as client:
        catalog = client.get('/api/inning-decisions', params={'game_id': 777063})
        assert catalog.status_code == 200
        assert catalog.json() == {'schema_version': 'inning-decision-v1',
                                  'mode': 'historical_decision_review', 'decisions': [imported]}
        assert set(imported) == {'decision_id', 'revision', 'context'}
        assert client.get('/api/inning-decisions', params={'game_id': 10}).json()['decisions'] == []
        alternate = deepcopy(context)
        alternate['linkage']['anchor_time_utc'] = linkage['anchor_time_utc'].replace('Z', '+00:00')
        resolved = client.post(f'/api/inning-decisions/{expected_id}/resolve',
                               json={'revision': 1, 'context': alternate})
        assert resolved.status_code == 200
        payload = resolved.json()
        assert set(payload) == {'schema_version', 'mode', 'decision_id', 'revision', 'context', 'result'}
        assert payload['context'] == context
        assert payload['result']['schema_version'] == 'inning-result-v1'
        assert payload['result']['status'] == 'bounded'
        assert payload['result']['linkage']['game_pk'] == 777063

        created = client.post('/api/sessions', json={'game_id': 10, 'pa_id': 1})
        assert created.status_code == 200
        view = created.json()
        assert view['cursor'] == 0 and view['history'] == [] and view['last_pitch'] is None
        assert 'inning_decisions' not in view and 'inning_result' not in view
        assert 'future_sentinel' not in json.dumps(view)
        assert client.get('/api/sessions/' + view['id']).json() == view


@pytest.mark.parametrize('field,value', [
    ('game_pk', 777064), ('official_game_date', '2025-07-20'),
    ('keep_pitcher_id', 123456), ('anchor_action_index', 48),
])
def test_valid_wrong_linkage_is_conflict(service, imported, field, value):
    context = deepcopy(imported['context'])
    context['linkage'][field] = value
    if field == 'game_pk':
        context['linkage']['first_observed_pitch_id'] = context['linkage']['first_observed_pitch_id'].replace('777063:', '777064:', 1)
    elif field == 'official_game_date':
        context['initial_state']['date'] = value
    with TestClient(create_app(service)) as client:
        response = client.post(f"/api/inning-decisions/{imported['decision_id']}/resolve",
                               json={'revision': 1, 'context': context})
    assert response.status_code == 409
    assert set(response.json()) == {'detail'}


def test_wrong_phase_state_time_and_revision_are_conflicts(service, imported):
    original = imported['context']
    changes = []
    phase = deepcopy(original)
    phase['phase'] = 'after_pitching_change'
    changes.append((1, phase))
    state = deepcopy(original)
    state['initial_state']['outs'] = 1
    changes.append((1, state))
    time = deepcopy(original)
    time['linkage']['anchor_time_utc'] = '2025-07-22T00:20:57.827000Z'
    changes.append((1, time))
    changes.append((2, original))
    with TestClient(create_app(service)) as client:
        for revision, context in changes:
            response = client.post(f"/api/inning-decisions/{imported['decision_id']}/resolve",
                                   json={'revision': revision, 'context': context})
            assert response.status_code == 409, response.text
            assert set(response.json()) == {'detail'}


def test_malformed_inputs_unknown_id_and_unavailable_service(service, imported):
    endpoint = f"/api/inning-decisions/{imported['decision_id']}/resolve"
    valid = {'revision': 1, 'context': imported['context']}
    with TestClient(create_app(service)) as client:
        for query in ('/api/inning-decisions', '/api/inning-decisions?game_id=0',
                      '/api/inning-decisions?game_id=true', '/api/inning-decisions?game_id=-1'):
            response = client.get(query)
            assert response.status_code == 400 and set(response.json()) == {'detail'}
        for body in ({}, {'revision': True, 'context': imported['context']},
                     {'revision': 0, 'context': imported['context']},
                     {'revision': 1, 'context': []}, valid | {'extra': 1}):
            response = client.post(endpoint, json=body)
            assert response.status_code == 400 and set(response.json()) == {'detail'}
        for mutation in (
            lambda c: c['linkage'].__setitem__('official_game_date', '2025-02-30'),
            lambda c: c['linkage'].__setitem__('anchor_time_utc', '2025-07-22T09:20:56+09:00'),
            lambda c: c['linkage'].__setitem__('first_observed_pitch_id', 'bad'),
            lambda c: c.__setitem__('extra', 1),
        ):
            context = deepcopy(imported['context'])
            mutation(context)
            response = client.post(endpoint, json={'revision': 1, 'context': context})
            assert response.status_code == 400, response.text
        for decision_id, expected in (('malformed', 400), ('inning-decision-' + '0'*64, 404)):
            response = client.post(f'/api/inning-decisions/{decision_id}/resolve', json=valid)
            assert response.status_code == expected and set(response.json()) == {'detail'}
        client.app.state.service = None
        for response in (client.get('/api/inning-decisions?game_id=777063'),
                         client.post(endpoint, json=valid)):
            assert response.status_code == 503 and set(response.json()) == {'detail'}
