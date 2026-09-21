"""Exercise the running app against every private demo record; no training.

Creates independent validation sessions and saves a compact report on the SSD.
Does not advance a user's existing session or modify original research assets.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'apps/observer/backend'))
from observer_app.dataset import DemoDataset
from observer_app.settings import RUN, require_storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8766')
    args = parser.parse_args()
    if not args.base_url.startswith(('http://127.0.0.1:', 'http://localhost:')):
        parser.error('Only a local Observer server is supported')
    require_storage()
    data = DemoDataset(RUN/'dataset.json')
    durations, examples = [], []

    def call(path, payload=None, expected=200):
        body = None if payload is None else json.dumps(payload).encode()
        request = Request(args.base_url+'/api'+path, data=body, headers={'Content-Type': 'application/json'})
        started = time.perf_counter()
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            status, result = response.status, json.load(response)
        durations.append((time.perf_counter()-started)*1000)
        assert status == expected, (path, status, result)
        return result

    assert call('/health')['model_ready']
    assert len(call('/catalog')['games']) == len(data.games)
    pitches = 0
    for game in data.games.values():
        for pa in game['plate_appearances']:
            view = call('/sessions', {'game_id': game['id'], 'pa_id': pa['id']})
            session = '/sessions/'+view['id']
            assert view['cursor'] == 0 and not view['history'] and view['last_pitch'] is None
            assert 'pitches' not in view['plate_appearance'] and view['summary'] is None
            initial = view['recommendation']
            for index, pitch in enumerate(pa['pitches']):
                assert view['state'] == pitch['state']
                recommendation = view['recommendation']
                assert recommendation['status'] == 'ready' and len(recommendation['candidates']) == 3
                for candidate in recommendation['candidates']:
                    assert 0 <= candidate['value'] <= 1 and candidate['support'] >= 20
                view = call(session+'/advance', {'revision': view['revision']})
                assert view['cursor'] == index+1 and len(view['history']) == index+1
                last = view['last_pitch']
                assert last['recommendation'] == recommendation
                assert last['id'] == pitch['id'] and last['pre_state'] == pitch['state']
                for field in ('x', 'z', 'speed_mph', 'pitch_type'):
                    assert last[field] == pitch['actual'][field]
                assert view['history'][0]['recommendation'] == initial
                if index+1 < len(pa['pitches']):
                    assert not view['complete'] and view['summary'] is None and view['analysis'] is None
                pitches += 1
            assert view['complete'] and view['recommendation'] is None and view['state'] == pa['terminal_state']
            assert view['analysis']['selected_pitch_id'] == pa['pitches'][-1]['id']
            assert view['summary']['pitch_count'] == len(pa['pitches'])
            old_revision = view['revision']
            view = call(session+'/manual-intent', {'revision': old_revision, 'zone_id': 'low_left'})
            assert view['analysis']['source'] == 'manual' and view['analysis']['manual_zone_id'] == 'low_left'
            call(session+'/advance', {'revision': old_revision}, expected=409)
            assert call(session)['analysis']['manual_zone_id'] == 'low_left'
            examples.append({'game_id': game['id'], 'pa_id': pa['id'], 'session_id': view['id'],
                             'pitches': len(pa['pitches']), 'first_recommendation': initial['candidates'][0]})
    ordered = sorted(durations)
    result = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_identity': data.identity, 'games': len(data.games), 'plate_appearances': len(examples),
              'pitches': pitches, 'http_requests': len(durations),
              'request_latency_ms': {'median': round(statistics.median(durations), 2),
                                     'p95': round(ordered[int(.95*(len(ordered)-1))], 2), 'max': round(max(durations), 2)},
              'checked': ['pre-pitch state', 'hidden future observations', 'immutable recommendation association',
                          'actual coordinates/type/speed', 'terminal state', 'manual intent', 'stale revision'],
              'runtime': call('/runtime'), 'examples': examples}
    name = datetime.now(timezone.utc).strftime('replay-validation-%Y%m%dT%H%M%SZ.json')
    (RUN/name).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({key: result[key] for key in ('status', 'games', 'plate_appearances', 'pitches', 'http_requests', 'request_latency_ms')}))
    print(RUN/name)


if __name__ == '__main__':
    main()
