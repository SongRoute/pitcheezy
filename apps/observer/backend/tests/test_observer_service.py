"""Synthetic replay/privacy/CAS/worker tests; no frozen data or model inference."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import threading

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_app.dataset import DemoDataset
from observer_app.main import create_app
from observer_app.service import ObserverService, ServiceError
from observer_app.settings import CONFIG
from observer_app.store import Store
from observer_app.worker import run_once
from observer_app.cv import CVRequest, NoMediaAdapter


class FakeRecommender:
    ready = True

    def __init__(self):
        self.calls = []
        self.fail = False

    def recommend(self, pitch, pa):
        self.calls.append(pitch['id'])
        if self.fail:
            raise RuntimeError('synthetic inference failure')
        return {'id': 'rec-'+pitch['id'], 'status': 'ready', 'mode': 'experimental_location_proxy',
                'model_version': 'test-v1', 'candidates': [{'pitch_type': 'FF', 'pitch_label': '포심',
                'zone_id': 'low_left', 'zone_label': '낮은 왼쪽', 'target': {'x': -.55, 'z': 1.8},
                'value': .5, 'delta_pp': 0., 'support': 50}], 'baseline_value': .5,
                'zone_bounds': pa['zone_bounds'], 'basis': [], 'reason': None}


@pytest.fixture
def service(tmp_path):
    state = dict(inning=1, half='Top', outs=0, bases=0, home_score=0, away_score=0, balls=0, strikes=0)
    pitches = []
    for i in range(2):
        pitches.append({'id': f'p{i+1}', 'pitch_number': i+1, 'state': state | {'strikes': i},
                        'request': {'private_sentinel': 'never-public'}, 'context_notes': [f'이전 투구 {i}개'],
                        'actual': {'pitch_type': 'SL' if i else 'FF', 'x': None if i else -.5,
                                   'z': None if i else 2., 'speed_mph': 88.+i,
                                   'description': 'future_sentinel' if i else 'called_strike',
                                   'event': 'field_out' if i else None}})
    pa = {'id': 1, 'batter_label': '타자', 'pitcher_label': '투수', 'batter_stand': 'R',
          'inning': 1, 'half': 'Top', 'zone_bounds': {'bottom': 1.5, 'top': 3.5},
          'repertoire_counts': {'FF': 100}, 'pitches': pitches,
          'terminal_state': state | {'outs': 1}}
    data = {'schema_version': 1, 'games': [{'id': 10, 'date': '2025-08-17', 'home_team': 'SF',
            'away_team': 'TB', 'title': 'TB @ SF', 'plate_appearances': [pa]}]}
    path = tmp_path/'dataset.json'
    path.write_text(json.dumps(data))
    (tmp_path/'dataset_manifest.json').write_text(json.dumps({'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))
    return ObserverService(DemoDataset(path), FakeRecommender(), Store(tmp_path/'observer.sqlite3'))


def finish(service, view):
    while not view['complete']:
        view = service.advance(view['id'], view['revision'])
    return view


def test_future_hidden_and_saved_pre_pitch_recommendation(service):
    first = service.create(10, 1)
    encoded = json.dumps(first)
    assert 'future_sentinel' not in encoded and 'never-public' not in encoded
    assert first['history'] == [] and first['last_pitch'] is None
    assert first['analysis'] is None and first['summary'] is None
    assert 'pitches' not in first['plate_appearance'] and 'pitch_count' not in encoded
    old_rec = deepcopy(first['recommendation'])
    next_view = service.advance(first['id'], 0)
    assert next_view['last_pitch']['recommendation'] == old_rec
    assert next_view['recommendation']['id'] != old_rec['id']
    assert 'future_sentinel' not in json.dumps(next_view)
    assert 'never-public' not in json.dumps(next_view)
    done = service.advance(first['id'], 1)
    assert done['complete'] and done['recommendation'] is None
    assert done['summary']['pitch_count'] == 2 and done['state']['outs'] == 1
    assert done['last_pitch']['x'] is None and done['last_pitch']['z'] is None
    assert done['history'][0]['recommendation'] == old_rec
    assert service.recommender.calls == ['p1', 'p2']
    service.get(first['id'])
    assert service.recommender.calls == ['p1', 'p2']
    with service.store.transaction() as db:
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            db.execute("UPDATE recommendations SET payload='{}'")


def test_slow_inference_does_not_block_other_session_reads_or_writes(service):
    first = service.create(10, 1)
    other = service.create(10, 1)
    entered, release = threading.Event(), threading.Event()
    original = service.recommender.recommend
    def slow(pitch, pa):
        entered.set()
        assert release.wait(timeout=5), 'test did not release inference'
        return original(pitch, pa)
    service.recommender.recommend = slow
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(service.advance, first['id'], 0)
        assert entered.wait(timeout=2)
        try:
            def independent_work():
                assert service.get(other['id'])['cursor'] == 0
                with service.store.transaction() as db:
                    Store.enqueue(db, 'unrelated-pitch', 'test')
                return True
            assert pool.submit(independent_work).result(timeout=2)
        finally:
            release.set()
        assert pending.result(timeout=2)['cursor'] == 1


def test_cas_allows_only_one_concurrent_advance(service):
    initial = service.create(10, 1)
    def advance():
        try:
            return service.advance(initial['id'], 0)['cursor']
        except ServiceError as exc:
            return exc.status
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: advance(), range(2))) == [1, 409]
    assert service.get(initial['id'])['cursor'] == 1


def test_manual_is_completion_only_and_survives_worker_and_restart(service):
    initial = service.create(10, 1)
    with pytest.raises(ServiceError) as error:
        service.manual_intent(initial['id'], 0, 'low_left')
    assert error.value.status == 409
    done = finish(service, initial)
    before = deepcopy(done['last_pitch']['recommendation'])
    manual = service.manual_intent(done['id'], done['revision'], 'high_right')
    assert manual['analysis']['source'] == 'manual'
    assert manual['analysis']['status'] == 'complete'
    assert manual['analysis']['comparisons']['target_error_zone_units'] is None
    assert manual['analysis']['comparisons']['actual_zone_label'] == '위치 정보 없음'
    assert manual['analysis']['version'] > done['analysis']['version']
    assert run_once(service.store)
    assert not run_once(service.store)
    restarted = ObserverService(service.dataset, FakeRecommender(), Store(service.store.path))
    retained = restarted.get(done['id'])
    assert retained['analysis']['manual_zone_id'] == 'high_right'
    assert retained['analysis']['cv_status'] == 'unavailable:no_media'
    assert retained['last_pitch']['recommendation'] == before
    with pytest.raises(ServiceError) as error:
        service.manual_intent(done['id'], done['revision'], 'low_left')
    assert error.value.status == 409


def test_terminal_job_is_unique_across_sessions_manual_is_not(service):
    one = finish(service, service.create(10, 1))
    two = finish(service, service.create(10, 1))
    assert one['analysis']['id'] == two['analysis']['id']
    service.manual_intent(one['id'], one['revision'], 'low_left')
    assert service.get(two['id'])['analysis']['manual_zone_id'] is None
    with service.store.transaction() as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 1


def test_worker_lease_retry_and_stale_worker_cannot_finish(service):
    finish(service, service.create(10, 1))
    first = service.store.claim(lease_seconds=10, max_attempts=2, now=100.)
    assert first['attempts'] == 1
    assert service.store.claim(lease_seconds=10, max_attempts=2, now=105.) is None
    second = service.store.claim(lease_seconds=10, max_attempts=2, now=111.)
    assert second['attempts'] == 2 and second['lease_token'] != first['lease_token']
    assert not service.store.finish(first['id'], first['lease_token'], now=112.)
    assert service.store.claim(lease_seconds=10, max_attempts=2, now=122.) is None
    with service.store.transaction() as db:
        assert db.execute('SELECT status FROM jobs').fetchone()[0] == 'failed'


def test_inference_failure_remains_replayable(service):
    service.recommender.fail = True
    view = service.create(10, 1)
    assert view['recommendation']['status'] == 'unavailable'
    assert finish(service, view)['complete']


def test_api_contract_errors_and_no_background_advance(service):
    with TestClient(create_app(service)) as client:
        assert client.get('/api/health').json()['model_ready']
        assert client.get('/api/zones').json()['coordinate_frame'] == 'catcher_view'
        assert len(client.get('/api/zones').json()['zones']) == 9
        catalog = client.get('/api/catalog').json()
        assert 'pitches' not in json.dumps(catalog) and 'future_sentinel' not in json.dumps(catalog)
        assert client.post('/api/sessions', json={'game_id': 10, 'pa_id': 999}).status_code == 404
        assert client.post('/api/sessions', json={'game_id': '10', 'pa_id': 1}).status_code == 400
        created = client.post('/api/sessions', json={'game_id': 10, 'pa_id': 1}).json()
        prefix = '/api/sessions/'+created['id']
        assert client.get(prefix).json()['cursor'] == 0
        assert client.post(prefix+'/advance', json={'revision': 9}).status_code == 409
        assert client.post(prefix+'/manual-intent', json={'revision': 0, 'zone_id': 'bad'}).status_code == 400
        assert client.post(prefix+'/advance', json={'revision': -1}).status_code == 400
        assert client.post(prefix+'/advance', json={'revision': 0}).json()['cursor'] == 1
        assert client.get('/api/unknown').status_code == 404


def test_session_refuses_changed_dataset_identity(service):
    created = service.create(10, 1)
    service.dataset.identity = 'different'
    with pytest.raises(ServiceError) as error:
        service.get(created['id'])
    assert error.value.status == 409


def test_lifespan_starts_and_stops_worker_subprocess(service):
    app = create_app(service, start_worker=True)
    with TestClient(app) as client:
        worker = app.state.worker
        assert worker is not None and worker.poll() is None
        assert client.get('/api/runtime').json()['worker_status'] == 'running'
    assert worker.poll() is not None


def test_failed_model_constructor_keeps_catalog_and_replay(service, monkeypatch):
    from observer_app import main, recommender
    def failing_model():
        raise RuntimeError('synthetic model load failure')
    monkeypatch.setattr(main, 'RUN', service.store.path.parent)
    monkeypatch.setattr(main, 'database_path', lambda: service.store.path)
    monkeypatch.setattr(recommender, 'Recommender', failing_model)
    with TestClient(create_app(start_worker=False)) as client:
        health = client.get('/api/health').json()
        assert health['status'] == 'degraded' and health['dataset_ready'] and not health['model_ready']
        assert client.get('/api/catalog').status_code == 200
        created = client.post('/api/sessions', json={'game_id': 10, 'pa_id': 1}).json()
        assert created['recommendation']['status'] == 'unavailable'
        advanced = client.post('/api/sessions/'+created['id']+'/advance', json={'revision': 0})
        assert advanced.status_code == 200 and advanced.json()['cursor'] == 1


def test_no_media_adapter_never_claims_cv_even_with_media_reference():
    adapter = NoMediaAdapter()
    missing = adapter.analyze(CVRequest('p1'))
    assert missing.status == 'unavailable' and missing.cv_status == 'unavailable:no_media'
    supplied = adapter.analyze(CVRequest('p1', media_ref='local-video', setup_before_pitch=100.))
    assert supplied.status == 'unavailable' and supplied.cv_status == 'unavailable:adapter_not_configured'
    assert supplied.source == 'none'


def test_event_result_revealed_only_after_pa_and_versioned_separately(service, monkeypatch):
    initial = service.create(10, 1)
    assert initial['event_analysis'] is None
    assert service.advance(initial['id'], initial['revision'])['event_analysis'] is None
    saved = deepcopy(service.get(initial['id'])['recommendation'])
    calls = []
    def event_result(session_id, pitch, _pa, recommendation, _created_at):
        calls.append(pitch['id'])
        assert recommendation == saved
        return {'revision': 1, 'status': 'partial', 'linkage': {'recommendation_id': recommendation['id']},
                'values': {'total_pp': 1.25}, 'evidence': {'development_only': False}}
    monkeypatch.setattr(service, '_calculate_event', event_result)
    done = service.advance(initial['id'], 1)
    assert done['event_analysis']['status'] == 'partial'
    assert calls == ['p2']
    assert service.get(initial['id'])['event_analysis'] == done['event_analysis']
    assert calls == ['p2']
    manual = service.manual_intent(done['id'], done['revision'], 'low_left')
    assert manual['event_analysis'] == done['event_analysis']
    with service.store.transaction() as db:
        Store.save_event_result(db, done['id'], 'p2', done['event_analysis'] | {'revision': 2, 'status': 'unavailable'})
    revised = service.get(done['id'])
    assert revised['event_analysis']['revision'] == 2
    assert revised['last_pitch']['recommendation'] == saved


def test_event_calculation_failure_has_no_invented_values(service, monkeypatch):
    def broken(*_args):
        raise RuntimeError('synthetic event calculation error')
    monkeypatch.setattr(service, '_calculate_event', broken)
    done = finish(service, service.create(10, 1))
    event = done['event_analysis']
    assert event['status'] == 'failed'
    assert event['values']['total_pp'] is None
    assert event['components']['unallocated_residual_pp'] is None
    assert service.get(done['id'])['event_analysis'] == event
