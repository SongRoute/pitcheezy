"""Versioned, descriptive PA event contrasts against one frozen defensive WE.

This module does not estimate intent, learn a model, or assign causal player credit.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Mapping

from .intent import validate_intent_estimate

SCHEMA_VERSION = 'event-analysis-v1'
VALUE_SPEC_VERSION = 'defense-we-pa-v1'
HORIZON = 'current_pa'
SHARE_EPSILON_PP = 1e-6
IDENTITY_KEYS = ('model_version', 'model_sha256', 'value_spec_version', 'baseline_policy_id')
VALUE_KEYS = ('reference', 'plan', 'execution', 'observed')


def _timestamp(value, name):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a timezone-aware ISO timestamp')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(f'{name} must be a timezone-aware ISO timestamp') from None
    if parsed.tzinfo is None:
        raise ValueError(f'{name} must include timezone')
    return parsed


def _identity(identity):
    if not isinstance(identity, Mapping) or any(not isinstance(identity.get(key), str) or not identity[key]
                                                for key in IDENTITY_KEYS):
        raise ValueError('identity requires nonempty model, bundle, value and baseline policy versions')
    if identity['value_spec_version'] != VALUE_SPEC_VERSION:
        raise ValueError('value_spec_version mismatch')
    return {key: identity[key] for key in IDENTITY_KEYS}


def value_point(value, *, identity, initial_defender, source, horizon=HORIZON):
    """Tag a value with its actual evaluator identity and the fixed initial defender."""
    ident = _identity(identity)
    if initial_defender not in ('home', 'away') or horizon != HORIZON:
        raise ValueError('Only initial home/away defender and current_pa are supported')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('WE value must be finite probability in [0,1]')
    if not isinstance(source, str) or not source:
        raise ValueError('value source is required')
    return {'value': float(value), 'identity': ident, 'horizon': horizon,
            'initial_defender': initial_defender, 'source': source}


def frozen_observed_value(*, engine, initial_state, post_state, identity):
    """Value an observed post-PA HR/K state with the *same* frozen engine WE.

    The caller supplies the observed post state, including actual runner movement.
    No deterministic terminal fallback is silently substituted for that observation.
    """
    from pathlib import Path
    bundle = Path(engine.bundle)
    manifest = bundle / 'bundle_manifest.json'
    if not manifest.is_file():
        raise ValueError('frozen engine has no bundle manifest')
    actual_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ident = _identity(identity)
    if ident['model_sha256'] != actual_sha:
        raise ValueError('model_sha256 differs from loaded frozen evaluator')
    defender = 'home' if initial_state.defender_is_home else 'away'
    if not hasattr(post_state, 'half') or not hasattr(engine, 'we'):
        raise ValueError('post_state and frozen engine.we are required')
    value = engine.we.predict_defense(post_state, initial_state.defender_is_home)
    return value_point(value, identity=ident, initial_defender=defender,
                       source='observed_post_pa_frozen_we')


def _validated_point(point, name, identity, defender):
    if point is None:
        return None
    if not isinstance(point, Mapping):
        raise ValueError(f'{name} must be a tagged value point')
    if _identity(point.get('identity')) != identity or point.get('horizon') != HORIZON or point.get('initial_defender') != defender:
        raise ValueError(f'{name} evaluator identity, horizon or initial defender mismatch')
    return value_point(point.get('value'), identity=identity, initial_defender=defender,
                       source=point.get('source'))['value']


def _component(value, reason=None, *, outcome=False):
    result = {'value_pp': value, 'status': 'available' if value is not None else 'unavailable',
              'reason': None if value is not None else reason, 'abs_share': None}
    if outcome:
        result['causal_batter_skill'] = False
    return result


def recommendation_sha256(recommendation):
    if not isinstance(recommendation, Mapping):
        raise ValueError('stored recommendation must be mapping')
    return hashlib.sha256(json.dumps(recommendation, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()


def _intent_reason(intent, linkage, release_frame_time, clip_pitch_id):
    if intent is None:
        return 'missing_intent'
    validate_intent_estimate(intent)
    if intent['pitch_id'] != linkage['pitch_id']:
        raise ValueError('IntentEstimate pitch_id differs from event pitch_id')
    if intent['status'] == 'unavailable':
        if clip_pitch_id is not None and clip_pitch_id != linkage['pitch_id']:
            raise ValueError('clip-to-pitch evidence does not match event pitch_id')
        return intent['unavailable_reason']
    if clip_pitch_id is None or clip_pitch_id != linkage['pitch_id']:
        raise ValueError('clip-to-pitch evidence does not match event pitch_id')
    if release_frame_time is None or not math.isfinite(release_frame_time):
        raise ValueError('release_frame_time in clip clock is required')
    if intent['evidence']['frame_time'] >= release_frame_time:
        raise ValueError('intent evidence must precede release in the same clip clock')
    if intent['deepest_frame'] != 'zone9' or intent['points']['zone9']['zone_id'] is None:
        return intent.get('blocked_by') or 'intent_action_unavailable'
    return None


def analyze_event(*, linkage, identity, initial_defender, values, evidence, provenance,
                  intent_estimate=None, release_frame_time=None, clip_pitch_id=None,
                  replacement=None, stored_recommendation=None):
    """Return C1 result; numeric inputs must be tagged by their frozen evaluator.

    A valid setup proxy permits a descriptive plan/arrival path. It never
    establishes actual pitcher intent or a causal command/batter score.
    """
    ident = _identity(identity)
    if initial_defender not in ('home', 'away'):
        raise ValueError('initial_defender must be home or away')
    if not isinstance(linkage, Mapping) or any(not linkage.get(k) for k in
        ('session_id', 'pitch_id', 'recommendation_id', 'recommendation_created_at', 'recommendation_sha256')):
        raise ValueError('immutable session, pitch and recommendation linkage required')
    _timestamp(linkage['recommendation_created_at'], 'recommendation_created_at')
    if type(linkage.get('event_input_revision')) is not int or linkage['event_input_revision'] < 1:
        raise ValueError('event_input_revision must be positive integer')
    if stored_recommendation is not None:
        if (stored_recommendation.get('id') != linkage['recommendation_id'] or
                recommendation_sha256(stored_recommendation) != linkage['recommendation_sha256']):
            raise ValueError('stored same-pitch recommendation identity mismatch')
    if not isinstance(evidence, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError('evidence and provenance must be mappings')
    plan_action = evidence.get('plan_action')
    synthetic_input = (intent_estimate is not None and isinstance(intent_estimate, Mapping) and
                       isinstance(intent_estimate.get('method'), Mapping) and
                       intent_estimate['method'].get('kind') == 'synthetic_contract_fixture')
    synthetic_plan = isinstance(plan_action, Mapping) and plan_action.get('source') == 'synthetic_fixture'
    if (synthetic_input or synthetic_plan) and (evidence.get('development_only') is not True or
                                                 evidence.get('use_for_performance_evaluation') is not False):
        raise ValueError('synthetic intent or plan requires development_only and excludes performance evaluation')
    if not evidence.get('development_only') and stored_recommendation is None:
        raise ValueError('production event analysis requires the saved same-pitch recommendation')
    _timestamp(provenance.get('received_at'), 'received_at')
    _timestamp(provenance.get('generated_at'), 'generated_at')
    if not isinstance(values, Mapping):
        raise ValueError('values must be mapping of tagged frozen values')
    numeric = {key: _validated_point(values.get(key), key, ident, initial_defender) for key in VALUE_KEYS}
    if stored_recommendation is not None and numeric['reference'] is not None:
        saved_baseline = stored_recommendation.get('baseline_value')
        if (isinstance(saved_baseline, bool) or not isinstance(saved_baseline, (int, float)) or
                not math.isfinite(saved_baseline) or abs(numeric['reference'] - saved_baseline) > 1e-10):
            raise ValueError('reference must equal saved same-pitch recommendation baseline_value')
    intent_reason = _intent_reason(intent_estimate, linkage, release_frame_time, clip_pitch_id)
    if intent_reason is not None and (numeric['plan'] is not None or numeric['execution'] is not None):
        raise ValueError('plan/execution cannot be assigned without usable linked pre-release intent')
    if numeric['execution'] is not None and numeric['plan'] is None:
        raise ValueError('execution path requires plan value')
    if numeric['plan'] is not None:
        planned = evidence.get('plan_action')
        if (not isinstance(planned, Mapping) or
                planned.get('zone_id') != intent_estimate['points']['zone9']['zone_id'] or
                not planned.get('pitch_type') or
                planned.get('source') not in ('pre_release_signal', 'synthetic_fixture')):
            raise ValueError('plan action must match linked intent zone and pre-release pitch type evidence')
    if numeric['execution'] is not None:
        delivered = evidence.get('execution_action')
        if (not isinstance(delivered, Mapping) or not delivered.get('pitch_type') or
                not delivered.get('zone_id') or delivered.get('source') != 'recorded_delivery'):
            raise ValueError('execution requires mapped recorded delivery action')
    if evidence.get('actual_is_intent') is True:
        raise ValueError('actual delivered action cannot masquerade as intent')

    ref, plan, execution, observed = (numeric[k] for k in VALUE_KEYS)
    total = None if ref is None or observed is None else 100 * (observed - ref)
    strategy = None if plan is None or ref is None else 100 * (plan - ref)
    delivery = None if execution is None or plan is None else 100 * (execution - plan)
    outcome = None if observed is None or execution is None else 100 * (observed - execution)
    assigned = [v for v in (strategy, delivery, outcome) if v is not None]
    unallocated = None if total is None else total - sum(assigned)
    complete = total is not None and all(v is not None for v in (strategy, delivery, outcome))
    if complete and abs(unallocated) > 1e-9:
        raise ValueError('ordered contrast values do not telescope')
    if complete:
        unallocated = 0.0
    status = 'complete' if complete else 'partial' if total is not None else 'unavailable'
    reason = None if complete else (intent_reason or 'missing_compatible_values') if status == 'partial' else 'missing_compatible_total'
    components = {
        'strategy_contrast_pp': _component(strategy, intent_reason or 'missing_plan_or_reference'),
        'execution_contrast_pp': _component(delivery, intent_reason or 'missing_plan_or_execution'),
        'outcome_residual_pp': _component(outcome, intent_reason or 'missing_execution_or_observed', outcome=True),
        'unallocated_residual_pp': unallocated,
    }
    denominator = sum(abs(v) for v in assigned) if complete else None
    stable = bool(complete and denominator > SHARE_EPSILON_PP)
    if not stable:
        denominator = None
    else:
        for name in ('strategy_contrast_pp', 'execution_contrast_pp', 'outcome_residual_pp'):
            components[name]['abs_share'] = abs(components[name]['value_pp']) / denominator
    if replacement is None:
        replacement = {'status': 'unavailable', 'horizon': 'inning_end', 'value_pp': None,
                       'reason': 'missing_contemporaneous_candidates_and_inning_evaluator'}
    validate_replacement(replacement)
    intent_evidence = {
        'intent_source': None if intent_estimate is None else intent_estimate['provenance']['label_source'],
        'intent_is_proxy': None if intent_estimate is None else True,
        'intent_review_status': None if intent_estimate is None else intent_estimate['provenance']['review_status'],
        'intent_uncertainty': None if intent_estimate is None else intent_estimate['uncertainty'],
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'analysis_id': f"{linkage['session_id']}:{linkage['pitch_id']}",
        'revision': linkage['event_input_revision'], 'status': status, 'reason': reason,
        'linkage': dict(linkage), 'identity': ident,
        'scope': {'horizon': HORIZON, 'initial_defender': initial_defender,
                  'unit': 'defense_win_probability', 'difference_unit': 'percentage_points'},
        'values': {**numeric, 'total_pp': total}, 'components': components,
        'interactions': {'status': 'unallocated', 'value_pp': None,
                         'reason': 'single ordered path does not identify order-dependent interaction'},
        'shares': {'basis': 'sum_abs_assigned_components', 'denominator_pp': denominator, 'stable': stable},
        'evidence': {**dict(evidence), **intent_evidence}, 'provenance': dict(provenance),
        'replacement': dict(replacement),
    }


def failed_event(*, linkage, identity, initial_defender, evidence, provenance, error_code,
                 stored_recommendation=None):
    """Produce an explicit failure revision after a calculation error, without values."""
    if not isinstance(error_code, str) or not error_code:
        raise ValueError('failed analysis requires a stable error_code')
    result = analyze_event(linkage=linkage, identity=identity,
        initial_defender=initial_defender, values={key: None for key in VALUE_KEYS},
        evidence=evidence, provenance=provenance, stored_recommendation=stored_recommendation)
    result['status'] = 'failed'
    result['reason'] = error_code
    return result


def unavailable_replacement(*, keep_pitcher_id=None, substitute_pitcher_ids=None,
                            decision_at=None, candidates=None, roster_as_of=None,
                            inning_evaluator=None, workload_evidence=None):
    """C3 boundary; only a contemporary eligible roster can unlock evaluation.

    The actual replacement evaluator is intentionally separate from this PA
    analysis. Merely seeing a later pitcher appear never proves availability.
    """
    if keep_pitcher_id is None or not substitute_pitcher_ids or decision_at is None or candidates is None or roster_as_of is None:
        reason = 'missing_contemporaneous_candidates'
    else:
        decision = _timestamp(decision_at, 'decision_at')
        roster = _timestamp(roster_as_of, 'roster_as_of')
        if roster > decision:
            raise ValueError('replacement availability evidence cannot postdate decision')
        requested = set(substitute_pitcher_ids)
        if keep_pitcher_id in requested:
            raise ValueError('keep pitcher cannot also be a substitute')
        if not isinstance(candidates, list) or not candidates or any(
            not isinstance(candidate, Mapping) or not candidate.get('pitcher_id') or
            candidate.get('eligible') is not True for candidate in candidates
        ) or requested != {candidate['pitcher_id'] for candidate in candidates}:
            reason = 'no_verified_eligible_candidates'
        elif not workload_evidence:
            reason = 'missing_contemporaneous_workload_rest'
        elif inning_evaluator is None:
            reason = 'missing_inning_end_evaluator'
        else:
            reason = 'inning_end_comparison_not_implemented'
    return {'status': 'unavailable', 'horizon': 'inning_end', 'value_pp': None, 'reason': reason}


def validate_replacement(replacement):
    """C3 input boundary: no invented bullpen candidates or PA/inning addition."""
    if not isinstance(replacement, Mapping) or replacement.get('horizon') != 'inning_end':
        raise ValueError('replacement comparison requires separate inning_end horizon')
    if replacement.get('status') != 'unavailable' or replacement.get('value_pp') is not None or not replacement.get('reason'):
        raise ValueError('replacement remains unavailable until contemporaneous roster and inning evaluator exist')
    return replacement
