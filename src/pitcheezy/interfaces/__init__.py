"""RE24·텐서·Q 스키마 + 검증. docs/interface-spec.md 의 코드화.

변경 = 계약 테스트(tests/interfaces) + 스펙 변경 이력 + decisions.md 한 줄.

states      count_id / base_out_id / state_id (S = 288 × K)
pitch_types pitch_id 0..8 (Statcast 코드 병합표)
grid        loc_id 0..24, action_id = pitch_id × 25 + loc_id (A = 225)
outcomes    outcome_id 0..10, Statcast 매핑, count_rule, 규칙 마스크
re24        RE24[24], dRE24[8, 24]
tensor      TransitionTensor 저장·로드
value       ValueBundle 저장·로드, lookup()
validate    validate_transition / validate_value
"""

from . import grid, outcomes, pitch_types, re24, states, tensor, validate, value  # noqa: F401
from .grid import N_ACTIONS, N_LOC, action_id, decode_action, loc_id, z_norm  # noqa: F401
from .outcomes import N_OUTCOMES, N_TERMINAL, TERMINAL, next_count, outcome_id, rule_mask  # noqa: F401
from .pitch_types import N_PITCH, PITCH_TYPE_MAP_VERSION, pitch_id  # noqa: F401
from .re24 import RE24Table  # noqa: F401
from .states import N_BASE_OUT, N_COUNTS, base_out_id, count_id, decode_state, n_states, state_id  # noqa: F401
from .tensor import SPEC_VERSION, TransitionTensor  # noqa: F401
from .validate import validate_transition, validate_value  # noqa: F401
from .value import ValueBundle, lookup  # noqa: F401
