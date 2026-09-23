"""Experimental type × location adapter around the immutable five-seed model.

Location is a Gaussian reweighting of observed TRAIN joint deliveries, not an
identified intervention on intended target or a measured command-error model.
No replay realization is accepted by the inference computation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from .settings import BUNDLE, CONFIG, RUN, REPO, require_storage
from .domain import ZONES, PITCH_LABELS, target_point
from .explanations import explain_choices
from .recommendation_adapter import (CandidateAction, ObservedDeliveryKernel,
    OUTCOMES, PrePitchEvaluation, PrePitchInput, VALUE_SPEC_VERSION)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def location_weights(xz, targets, sigma):
    """Return normalized local weights and effective sample sizes per action."""
    return ObservedDeliveryKernel().weights(xz, targets, sigma)


def supported_actions(support, mass):
    # ESS alone can be high for equally distant samples. Require non-negligible
    # absolute local probability mass too, at every planned count.
    return ((support.min(axis=(0, 1)) >= CONFIG['minimum_effective_draws']) &
            (mass.min(axis=(0, 1)) >= CONFIG['minimum_kernel_mass']))


class Recommender:
    def __init__(self, execution_distribution=None):
        require_storage()
        source = BUNDLE/'source'
        hashes = json.loads((BUNDLE/'source_hashes.json').read_text())
        for relative, expected in hashes.items():
            if hashlib.sha256((source/relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Frozen source hash mismatch: {relative}')
        # The research bundle captures prediction sources but its CLI imports
        # also require experiment utilities. Use the complete local tree after
        # checking every captured Python source against the immutable capture.
        self.runtime_kind = os.environ.get('PITCHEEZY_OBSERVER_RUNTIME', 'research')
        if self.runtime_kind == 'standalone':
            from .standalone_engine import load_engine
            self.engine = load_engine(BUNDLE, device='cpu')
            utilities = self.engine.runtime_manifest['runtime_files']
        elif self.runtime_kind == 'research':
            project = REPO/'experiments/pitchmdp'
            for relative, expected in hashes.items():
                if relative.endswith('.py') and hashlib.sha256((project/relative).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f'Local source differs from frozen model: {relative}')
            utilities = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for folder in ('scripts', 'pitchmdp') for p in (project/folder).glob('*.py')}
            sys.path[:0] = [str(project/'scripts'), str(project)]
            from minimal_pitch_service import Engine
            self.engine = Engine(BUNDLE, device='cpu')
        else:
            raise ValueError('Unknown observer runtime selection')
        self.identity = digest({'bundle': json.loads((BUNDLE/'bundle_manifest.json').read_text()),
                                'source': hashes, 'utilities': utilities, 'config': CONFIG,
                                'adapter': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                'explanations': hashlib.sha256(Path(__file__).with_name('explanations.py').read_bytes()).hexdigest(),
                                'domain': hashlib.sha256(Path(__file__).with_name('domain.py').read_bytes()).hexdigest()})
        self.execution_distribution = execution_distribution or ObservedDeliveryKernel()
        if self.execution_distribution.identity != ObservedDeliveryKernel.identity:
            raise ValueError('Only the frozen observational execution distribution is validated for current recommendations')
        self.cache = RUN/'recommendation_cache'
        self.cache.mkdir(exist_ok=True)
        self._lock = threading.RLock()
        self.ready = True
        self.last_latency_ms = None
        self.computations = 0
        self.cache_hits = 0

    def unavailable(self, bounds, reason):
        return {'id': digest([self.identity, bounds, reason]), 'status': 'unavailable',
                'mode': 'experimental_location_proxy', 'model_version': CONFIG['model_version'],
                'candidates': [], 'baseline_value': None, 'zone_bounds': bounds, 'basis': [], 'reason': reason}

    def recommend(self, pitch, pa):
        # Deliberately extract the pre-pitch request only. Actual and future rows
        # cannot affect cache identity or predictions.
        inputs = PrePitchInput.from_replay(pitch, pa)
        request = inputs.request
        balls, strikes = request.pop('balls'), request.pop('strikes')
        if type(balls) is not int or type(strikes) is not int or not 0 <= balls <= 3 or not 0 <= strikes <= 2:
            raise ValueError('Invalid pre-pitch count')
        bounds, repertoire = inputs.zone_bounds, inputs.repertoire_counts
        key = digest([self.identity, request, bounds, repertoire])
        path = self.cache/f'{key}.json'
        with self._lock:
            started = time.perf_counter()
            if path.exists():
                cached = json.loads(path.read_text())
                if cached['sha256'] != digest(cached['recommendations']):
                    raise RuntimeError('Recommendation cache integrity mismatch')
                recommendations = cached['recommendations']
                self.cache_hits += 1
            else:
                recommendations = self._compute(request | {'balls': 0, 'strikes': 0}, bounds, repertoire, key)
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps({'sha256': digest(recommendations), 'recommendations': recommendations},
                                                ensure_ascii=False, allow_nan=False))
                temporary.replace(path)
                self.computations += 1
            self.last_latency_ms = round((time.perf_counter()-started)*1000, 2)
            return copy.deepcopy(recommendations[f'{balls}-{strikes}'])

    def evaluate_pre_pitch(self, pitch, pa):
        """Expose every supported candidate at the current pre-pitch count.

        This read-only calculation is not saved in the public recommendation
        cache; it uses the same frozen model and computation as ``recommend``.
        """
        inputs = PrePitchInput.from_replay(pitch, pa)
        request = inputs.request
        balls, strikes = request.pop('balls'), request.pop('strikes')
        if type(balls) is not int or type(strikes) is not int or not 0 <= balls <= 3 or not 0 <= strikes <= 2:
            raise ValueError('Invalid pre-pitch count')
        key = digest([self.identity, request, inputs.zone_bounds, inputs.repertoire_counts])
        recommendations, detail = self._compute(request | {'balls': 0, 'strikes': 0},
            inputs.zone_bounds, inputs.repertoire_counts, key, include_detail=True)
        result = recommendations[f'{balls}-{strikes}']
        if detail is None:
            return PrePitchEvaluation(result['status'], result['reason'], self.identity,
                CONFIG['model_version'], VALUE_SPEC_VERSION, 'observer-repertoire-kernel-v1',
                OUTCOMES, (), None, None, None, None, None, None, result)
        return PrePitchEvaluation(result['status'], result['reason'], self.identity,
            CONFIG['model_version'], VALUE_SPEC_VERSION, 'observer-repertoire-kernel-v1',
            OUTCOMES, detail['actions'], detail['probabilities'], detail['support'],
            detail['mass'], detail['baseline'], detail['values'], detail['baseline_values'], result)

    def _compute(self, request, bounds, repertoire, key, include_detail=False):
        import numpy as np
        import pandas as pd
        from scipy.special import softmax
        from minimal_pitch_service import validate_request, temperature_predictions, condition_on_legality
        from pitchmdp.game import terminal_values
        from pitchmdp.planner import solve_pa

        if request['batter_profile']['as_of'] >= request['date']:
            raise ValueError('Batter profile must precede the entire replay date')
        row = validate_request(request, self.engine.metadata)
        types = [name for name in row['pitch_types'] if repertoire.get(name, 0) >= CONFIG['repertoire_minimum_pitches']]
        def unavailable(reason):
            result = {f'{b}-{s}': self.unavailable(bounds, reason) for b in range(4) for s in range(3)}
            return (result, None) if include_detail else result
        if not types:
            return unavailable('최근 기록에서 지원되는 구종이 부족합니다.')
        records = [{'balls': b, 'strikes': s, 'pitch_type': name, 'inning': row['inning'],
                    'inning_topbot': row['topbot'], 'outs_when_up': row['outs'], 'bases': row['bases'],
                    'home_score': row['home_score'], 'away_score': row['away_score'],
                    'pitcher': row['pitcher_id'], 'p_throws': row['p_throws'], 'stand': row['batter_stand'],
                    **row['profile']} for b in range(4) for s in range(3) for name in types]
        frame = pd.DataFrame(records)
        draws, tiers = self.engine.delivery.sample(frame)
        n, count, physical = draws.shape
        tokens = np.zeros((n*count, 6, physical), dtype=np.float32)
        tokens[:, -1] = draws.reshape(-1, physical)
        valid = np.zeros((n*count, 6), dtype=bool)
        valid[:, -1] = True
        context = np.repeat(self.engine.context.transform(frame), count, axis=0)
        neural = np.zeros((n, count, 10), dtype=np.float64)
        for model in self.engine.models:
            logits = model.logits((tokens, valid, context)).reshape(n, count, 10)
            neural += softmax(logits/model.delivery_temperature, axis=-1)/len(self.engine.models)
        physical_draws = draws*self.engine.delivery.normalizer.scale + self.engine.delivery.normalizer.mean
        targets = np.array([[target_point(z['id'], bounds)[axis] for axis in ('x', 'z')] for z in ZONES])
        weights, support, mass = self.execution_distribution.weights(physical_draws[:, :, 6:8], targets, CONFIG['target_sigma_ft'])
        local = np.einsum('ndz,ndk->nzk', weights, neural)
        frequency = temperature_predictions(self.engine.baseline.predict(frame), self.engine.baseline_temperature)
        probabilities = self.engine.weight*local + (1-self.engine.weight)*frequency[:, None, :]
        flat = probabilities.reshape(-1, 10)
        flat /= flat.sum(axis=-1, keepdims=True)
        flat = condition_on_legality(flat, np.full(len(flat), row['outs'] == 2 or row['bases'] == 0))
        actions = len(types)*len(ZONES)
        probabilities = flat.reshape(4, 3, 1, actions, 10)
        support = support.reshape(4, 3, actions)
        supported = supported_actions(support, mass.reshape(4, 3, actions))
        if not supported.any():
            return unavailable('목표 구역을 비교할 충분한 투구 표본이 없습니다.')
        indices = np.flatnonzero(supported)
        all_mass = mass.reshape(4, 3, actions)
        probabilities = probabilities[:, :, :, supported, :]
        baseline = mass.reshape(4, 3, len(types), len(ZONES)).mean(axis=(0, 1))
        baseline *= np.array([repertoire[name] for name in types])[:, None]
        baseline = baseline.reshape(-1)[supported]
        baseline /= baseline.sum()
        terminal = terminal_values(row['state'], self.engine.we, self.engine.advancement)
        plan = solve_pa(probabilities, terminal, [0]*len(indices), baseline_policy=baseline)
        detail = None
        if include_detail:
            detail = {'actions': tuple(CandidateAction(i, types[action//len(ZONES)],
                       ZONES[action % len(ZONES)]['id'], target_point(ZONES[action % len(ZONES)]['id'], bounds))
                       for i, action in enumerate(indices)),
                      'probabilities': probabilities.copy(), 'support': support[:, :, supported].copy(),
                      'mass': all_mass[:, :, supported].copy(), 'baseline': baseline.copy(),
                      'values': plan.values.copy(), 'baseline_values': plan.baseline_values.copy()}
        results = {}
        for balls in range(4):
            for strikes in range(3):
                candidates = []
                ranked = plan.topk(balls, strikes, 0, 3)
                for entry in ranked:
                    action = int(indices[entry['action_index']])
                    name, zone = types[action//len(ZONES)], ZONES[action % len(ZONES)]
                    candidates.append({'pitch_type': name, 'pitch_label': PITCH_LABELS.get(name, name),
                        'zone_id': zone['id'], 'zone_label': zone['label'], 'target': target_point(zone['id'], bounds),
                        'value': float(entry['value']), 'delta_pp': 100*float(entry['delta_vs_baseline']),
                        'support': round(float(support[balls, strikes, action]), 1)})
                explanation = explain_choices(
                    [probabilities[balls, strikes, 0, entry['action_index']] for entry in ranked],
                    [entry['value'] for entry in ranked], plan.values, terminal, balls, strikes)
                results[f'{balls}-{strikes}'] = {'id': digest([key, balls, strikes]), 'status': 'ready',
                    'mode': 'experimental_location_proxy', 'model_version': CONFIG['model_version'],
                    'candidates': candidates, 'baseline_value': float(plan.baseline_values[balls, strikes, 0]),
                    'explanation': explanation,
                    'zone_bounds': bounds, 'reason': None,
                    'basis': ['주자·아웃·카운트·점수·홈/원정 반영', '경기 전날까지의 타자 성향 반영',
                              '목표 구역은 실제 도달 위치 분포를 활용한 근사 제안',
                              '승률 차이는 모델 내부 추정이며 실제 개선 효과는 미검증']}
        return (results, detail) if include_detail else results
