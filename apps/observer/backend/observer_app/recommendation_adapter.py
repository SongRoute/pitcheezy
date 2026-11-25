"""Pre-pitch candidate boundary for the frozen Observer recommender.

The execution distribution is a replaceable *interface*. The current provider
only reweights observed TRAIN deliveries; it is not a learned intent or command
model and does not estimate the effect of intervening on a target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


VALUE_SPEC_VERSION = 'defense-we-pa-v1'
OUTCOMES = ('ball', 'strike', 'foul', 'out', 'single', 'double', 'triple',
            'home_run', 'hbp', 'double_play')


@dataclass(frozen=True)
class PrePitchInput:
    request: dict
    zone_bounds: dict
    repertoire_counts: dict

    @classmethod
    def from_replay(cls, pitch: Mapping, pa: Mapping) -> 'PrePitchInput':
        # A positive allowlist makes actual/current and future replay fields
        # unreachable by the inference method and its cache key.
        import copy
        return cls(copy.deepcopy(pitch['request']), copy.deepcopy(pa['zone_bounds']),
                   copy.deepcopy(pa['repertoire_counts']))


@dataclass(frozen=True)
class CandidateAction:
    index: int
    pitch_type: str
    zone_id: str
    target: dict


@dataclass(frozen=True)
class PrePitchEvaluation:
    status: str
    reason: str | None
    model_identity: str
    model_sha256: str
    model_version: str
    value_spec_version: str
    baseline_policy_id: str
    outcomes: tuple[str, ...]
    actions: tuple[CandidateAction, ...]
    probabilities: np.ndarray | None  # 4 × 3 × 1 × supported actions × 10
    support_ess: np.ndarray | None      # 4 × 3 × supported actions
    kernel_mass: np.ndarray | None      # 4 × 3 × supported actions
    baseline_policy: np.ndarray | None  # supported action probabilities
    candidate_values: np.ndarray | None # 4 × 3 × 1 × supported actions, PA WE Q
    values: np.ndarray | None           # 4 × 3 × 1, optimized PA WE
    baseline_values: np.ndarray | None  # 4 × 3 × 1, reference PA WE
    recommendation: dict


class ExecutionDistribution(Protocol):
    identity: str

    def weights(self, xz: np.ndarray, targets: np.ndarray, sigma: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


class ObservedDeliveryKernel:
    """Current observational TRAIN-draw Gaussian reweighting (sigma in feet)."""
    identity = 'observed-train-delivery-gaussian-v1'

    def weights(self, xz: np.ndarray, targets: np.ndarray, sigma: float
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_weights = -np.square(xz[:, :, None, :] - targets[None, None, :, :]).sum(-1)/(2*sigma*sigma)
        mass = np.exp(log_weights).mean(axis=1)
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        weights /= weights.sum(axis=1, keepdims=True)
        support = 1/np.square(weights).sum(axis=1)
        return weights, support, mass


ARMB_MODE = 'armb_type_location_proxy'
LOCATION_BASIS = 'realized_delivery_proxy'
LOCATION_NOTE = '위치는 이 투수의 실제 투구 분포 기반 근사이며 평가되지 않았습니다.'


def armb_recommendation(type_probabilities: Mapping[str, float], evaluation: PrePitchEvaluation,
                        balls: int, strikes: int, zone_bounds: dict, *, policy_identity=None,
                        top_k: int = 3) -> dict:
    """ARM-B pitch-type ranking first; each type gets an unevaluated location proxy.

    ``type_probabilities`` is the ARM-B law over pitch types only (KL-regularized
    candidate over the pitcher's TRAIN repertoire). The zone is the supported
    (ESS/mass rule) realized-delivery kernel cell with the most local mass for
    that type at this count. No value or policy claim is made about the zone.
    """
    from .domain import PITCH_LABELS, ZONE_BY_ID
    probs = {str(k): float(v) for k, v in type_probabilities.items()}
    values = np.array(list(probs.values()))
    if not probs or not np.isfinite(values).all() or (values < 0).any() or abs(values.sum()-1) > 1e-6:
        raise ValueError('ARM-B type probabilities must be a finite distribution')
    ranked = sorted((name for name, p in probs.items() if p > 0), key=lambda name: (-probs[name], name))[:top_k]
    mass, ess = evaluation.kernel_mass, evaluation.support_ess
    candidates = []
    for rank, name in enumerate(ranked, 1):
        options = [a for a in evaluation.actions if a.pitch_type == name]
        best = max(options, key=lambda a: (mass[balls, strikes, a.index], a.zone_id)) if options else None
        candidates.append({
            'rank': rank, 'pitch_type': name, 'pitch_label': PITCH_LABELS.get(name, name),
            'zone_id': best.zone_id if best else None,
            'zone_label': ZONE_BY_ID[best.zone_id]['label'] if best else '위치 근사 불가(표본 부족)',
            'target': best.target if best else None,
            'location_basis': LOCATION_BASIS, 'location_evaluated': False,
            # Secondary (D49): numbers are shown only after type and target.
            'detail': {'probability': probs[name],
                       'kernel_mass': float(mass[balls, strikes, best.index]) if best else None,
                       'kernel_ess': round(float(ess[balls, strikes, best.index]), 1) if best else None}})
    return {'status': 'ready', 'mode': ARMB_MODE, 'model_version': evaluation.model_version,
            'policy_identity': policy_identity, 'baseline_policy_id': 'SupportedBC', 'value_spec_version': None,
            'candidates': candidates, 'baseline_value': None, 'zone_bounds': zone_bounds, 'reason': None,
            'location_basis': LOCATION_BASIS, 'location_evaluated': False, 'location_note': LOCATION_NOTE,
            'location_proxy_identity': ObservedDeliveryKernel.identity,
            'basis': ['구종 순위는 ARM-B(구종 전용) 정책 확률', LOCATION_NOTE]}
