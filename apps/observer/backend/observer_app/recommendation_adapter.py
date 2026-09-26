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
