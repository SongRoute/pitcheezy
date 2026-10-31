"""Retrospective frozen-model HR/K partial result; no intent or player credit.

This is a reproducibility smoke, not a saved live recommendation or evaluation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps/observer/backend'))

from observer_app.event_analysis import (analyze_event, frozen_observed_value,
    recommendation_sha256, value_point)
from observer_app.settings import BUNDLE
from observer_app.standalone_engine import load_engine

DATA = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-events-v1/event_packets.json')
DEV = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-S0S1-001/dev_full_games.parquet')
OUT = Path('/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-events-v1/c_smoke.json')


def main():
    packets = json.loads(DATA.read_text())['events']
    frame = pd.read_parquet(DEV)
    engine = load_engine(BUNDLE, device='cpu')
    from pitchmdp.game import GameState, terminal_values
    from pitchmdp.planner import solve_pa
    identity = {
        'model_version': 'observer-zone-v1',
        'model_sha256': hashlib.sha256((engine.bundle / 'bundle_manifest.json').read_bytes()).hexdigest(),
        'value_spec_version': 'defense-we-pa-v1',
        'baseline_policy_id': 'frozen_type_frequency_v1',
    }
    for event in packets:
        if event['pitcher_id'] not in (657277, 554430) or not event['pa_supported_in_source'] or not event['next_state_complete']:
            continue
        game, pa, pitch = map(int, event['pitch_id'].split(':'))
        row = frame.loc[(frame.game_pk == game) & (frame.at_bat_number == pa) & (frame.pitch_number == pitch)]
        if len(row) != 1:
            continue
        row = row.iloc[0]
        pre = event['pre_pitch_state']
        request = {'date': event['date'], 'inning': int(pre['inning']), 'topbot': pre['inning_topbot'],
                   'outs': int(pre['outs_when_up']), 'bases': int(pre['bases']),
                   'home_score': int(pre['home_score']), 'away_score': int(pre['away_score']),
                   'balls': int(pre['balls']), 'strikes': int(pre['strikes']),
                   'pitcher_id': int(event['pitcher_id']), 'batter_id': int(event['batter_id']),
                   'batter_stand': str(row.stand), 'top_k': 3}
        try:
            prediction = engine.predict_counts(request)
        except (ValueError, KeyError):
            continue
        types = list(prediction['pitch_types'])
        repertoire = engine.metadata['repertoire_counts'].get(str(event['pitcher_id']), {})
        usage = np.array([repertoire.get(name, 0) for name in types], dtype=float)
        if not types or usage.sum() <= 0:
            continue
        usage /= usage.sum()
        initial = prediction['state']
        terminal = terminal_values(initial, engine.we, engine.advancement)
        plan = solve_pa(prediction['probabilities']['blend'], terminal, [0] * len(types), baseline_policy=usage)
        reference = float(plan.baseline_values[request['balls'], request['strikes'], 0])
        next_state = event['recorded_next_state']
        post = GameState(int(next_state['inning']), next_state['half'], int(next_state['outs']),
                         int(next_state['bases']), int(next_state['home_score']), int(next_state['away_score']))
        observed = frozen_observed_value(engine=engine, initial_state=initial, post_state=post, identity=identity)
        defender = event['initial_defender']
        if defender != observed['initial_defender']:
            raise ValueError('event packet initial defender disagrees with frozen game state')
        recommendation = {'id': hashlib.sha256(event['pitch_id'].encode()).hexdigest(),
                          'baseline_value': reference, 'kind': 'retrospective_type_frequency_smoke'}
        now = datetime.now(timezone.utc).isoformat()
        result = analyze_event(
            linkage={'session_id': 'c-development-smoke', 'pitch_id': event['pitch_id'],
                     'recommendation_id': recommendation['id'],
                     'recommendation_created_at': now,
                     'recommendation_sha256': recommendation_sha256(recommendation),
                     'event_input_revision': 1},
            identity=identity, initial_defender=defender,
            values={'reference': value_point(reference, identity=identity, initial_defender=defender,
                                             source='retrospective frozen type-frequency baseline'),
                    'plan': None, 'execution': None, 'observed': observed},
            evidence={'actual_source': 'A-events-v1 recorded_next_state', 'references': [str(DATA)],
                      'development_only': True, 'use_for_performance_evaluation': False,
                      'pre_pitch_recommendation_was_stored_live': False,
                      'note': 'Retrospective frozen-model same-pitch calculation; no actual intent.'},
            provenance={'received_at': now, 'generated_at': now, 'correction_of_revision': None},
            stored_recommendation=recommendation)
        OUT.write_text(json.dumps({'event': {'pitch_id': event['pitch_id'], 'event': event['event']},
                                   'result': result}, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'path': str(OUT), 'pitch_id': event['pitch_id'], 'event': event['event'],
                          'status': result['status'], 'total_pp': result['values']['total_pp']}))
        return
    raise RuntimeError('No supported complete HR/K case could be evaluated')


if __name__ == '__main__':
    main()
