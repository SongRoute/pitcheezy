"""Exercise the real imported decision via ASGI, without loading a pitch model."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))
from fastapi.testclient import TestClient
from observer_app.main import create_app, UnavailableRecommender
from observer_app.service import ObserverService
from observer_app.store import Store


def verify(database: Path, output: Path):
    if not database.is_file():
        raise ValueError('Import the real source into a new isolated database first')
    output.mkdir(parents=True, exist_ok=False)

    def application():
        # This resource does not use a PA dataset or predictor. Injection is
        # explicit; this verifies ASGI routes, not normal model startup or UI.
        return create_app(ObserverService(None, UnavailableRecommender(), Store(database)), start_worker=False)

    with TestClient(application()) as client:
        response = client.get('/api/inning-decisions', params={'game_id': 777063})
        assert response.status_code == 200, response.text
        catalog = response.json()
        assert len(catalog['decisions']) == 1
        selected = catalog['decisions'][0]
        route = f"/api/inning-decisions/{selected['decision_id']}/resolve"
        body = {'revision': selected['revision'], 'context': selected['context']}
        response = client.post(route, json=body)
        assert response.status_code == 200, response.text
        resolved = response.json()
        assert resolved['result'] == json.loads((ROOT/'results/C-D-INNING-001/bounded.json').read_text())
        mismatches = []
        for case in ('game', 'date', 'time', 'pitcher', 'outs', 'phase', 'revision'):
            wrong = deepcopy(body)
            linkage, state = wrong['context']['linkage'], wrong['context']['initial_state']
            if case == 'game':
                linkage['game_pk'] = 777064
                linkage['first_observed_pitch_id'] = '777064:49:1'
            elif case == 'date':
                linkage['official_game_date'] = state['date'] = '2025-07-20'
            elif case == 'time':
                linkage['anchor_time_utc'] = '2025-07-22T00:22:07.083000Z'
            elif case == 'pitcher':
                linkage['keep_pitcher_id'] = 1
            elif case == 'outs':
                state['outs'] = 1
            elif case == 'phase':
                wrong['context']['phase'] = 'after_pitching_change'
            else:
                wrong['revision'] = 2
            rejected = client.post(route, json=wrong)
            assert rejected.status_code == 409, (case, rejected.status_code, rejected.text)
            assert set(rejected.json()) == {'detail'}
            mismatches.append({'case': case, 'status': rejected.status_code, 'response': rejected.json()})
        assert client.get('/api/inning-decisions', params={'game_id': 777064}).json()['decisions'] == []

    with TestClient(application()) as client:
        response = client.post(route, json=body)
        assert response.status_code == 200 and response.json() == resolved

    outputs = {'catalog.json': catalog, 'resolved.json': resolved, 'mismatches.json': mismatches}
    for name, data in outputs.items():
        with (output/name).open('x') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    audit = {'artifact_id': 'C-D-API-001', 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'database': str(database.resolve()), 'transport': 'FastAPI ASGI TestClient',
             'service_injection': 'No PA dataset or predictor; actual imported C decision only',
             'decision_id': selected['decision_id'], 'match_status': 200,
             'mismatch_cases': len(mismatches), 'restart_preserved': True,
             'original_result_preserved': True, 'model_calls': 0, 'new_data_fetches': 0,
             'http_server_started': False, 'ui_connected': False,
             'outputs_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in outputs}}
    with (output/'audit.json').open('x') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'decision_id': selected['decision_id'], 'matched': 200, 'mismatches_rejected': len(mismatches), 'restart_preserved': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    verify(args.database, args.output_dir)
