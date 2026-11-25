"""EXP-MPS-PSR-001 모델·학습기: EXP-MPS-PB-002 팔 pitcher_batter_lr100_drop020 모델에 '투수 × 상태 잔차'(EXP-GPU14H-010 형식)만 더한다.

잔차(residual, 원 모델 출력에 더하는 작은 보정항) 정의 — EXP-GPU14H-010 experiments/gpu14h_idinteraction/train.py(eb9282df)의
variant pitcher_state_residual에서 그대로 옮겼다:
    logits = control_logits + output( tanh( state(x[:, :11]) ) * pitcher_embedding(x[:, 11]) )
    state  = Linear(11, 16, bias=False)  (PyTorch 기본 초기화, fork_rng 안에서 같은 seed로)
    output = Linear(16, 9,  bias=False)  (0으로 초기화)
    pitcher_embedding = 대조 모델의 기존 투수 embedding(공유; 행 0 = 처음 보는 투수 = padding_idx 영구 0 벡터)
추가 파라미터 11*16 + 16*9 = 320개, 새 dropout 없음, 대조 모델 forward는 한 번만 부른다.

대조(control) 모델은 gpu14h_batter.train.build_model(r, 'pitcher_batter', seed, sizes)가 만든 것 그 자체다(복제하지 않음).
새 모델은 그 하위 모듈(pitcher, batter_embedding, batter_offset)을 같은 이름으로 등록하고 state·output 두 행렬만 더한다.
그래서 state dict 이름 = 대조 이름 + state.weight + output.weight.

학습기 fit_body는 EXP-MPS-PB-001/002 MPS 학습기 experiments/mps_pb/train.py:fit의 코드 객체(__code__) 그 자체를 쓴다.
글자를 옮기지 않았다. 다른 점은 그 코드가 찾는 전역 이름 build_model 하나가 이 모듈의 build_model을 가리킨다는 것뿐이다
(나머지 전역은 원 모듈의 같은 객체; test_run이 확인). 기존 파일은 바꾸지 않는다.
"""
import copy
import types

from experiments.gpu14h_batter import train as batter
from experiments.mps_pb import train as pbtrain

SOURCE = 'experiments/mps_pb/train.py:fit'
DESIGN_SOURCE = 'EXP-GPU14H-010 experiments/gpu14h_idinteraction/train.py (commit eb9282df) variant pitcher_state_residual'
TRAINER_VARIANT = 'pitcher_batter'  # gpu14h_batter.train 변형(투수 + 타자 요인) = 대조 팔 PB-002 학습기 변형
STATE_COLUMNS = 11
HIDDEN = 16
CLASSES = 9
ADDED_PARAMETERS = STATE_COLUMNS * HIDDEN + HIDDEN * CLASSES  # 320
NEW_KEYS = ('state.weight', 'output.weight')
# EXP-GPU14H-010 recipe()가 r['identity_residual']에 넣는 값(pitcher_state_residual 변형) 그대로
RESIDUAL = {'variant': 'pitcher_state_residual', 'state_dimension': 11, 'hidden_dimension': 16, 'output_dimension': 9, 'bias': False,
            'additional_parameters': 320, 'gate': 'existing_pitcher_embedding', 'activation': 'tanh', 'output_initial': 'zero',
            'RNG': 'fork_rng same seed', 'parent_forward_calls': 1, 'new_dropout': 0, 'unknown_ID_interaction': 'zero'}
# 학습기(PB-001 MPS fit)가 receipt에 적는 문구와, 이 모델에 맞게 바꿔 적는 문구
BASE_INTERACTION = 'separate additive factorized offsets; no ID x state or P x B interaction'
INTERACTION = 'separate additive factorized offsets plus pitcher x state11 residual (EXP-GPU14H-010 form, shared pitcher embedding); no P x B interaction'
# receipt에 새로 붙는 칸(상수; 모양·개수는 학습 뒤 model_structure가 저장 파일에서 다시 확인한다)
RECEIPT_RESIDUAL = {'form': 'logits = control_logits + output(tanh(state(state11)) * pitcher_embedding)', 'design_source': 'EXP-GPU14H-010 eb9282df pitcher_state_residual',
                    'state_columns': list(range(STATE_COLUMNS)), 'state_shape': [HIDDEN, STATE_COLUMNS], 'output_shape': [CLASSES, HIDDEN], 'bias': False,
                    'additional_parameters': ADDED_PARAMETERS, 'gate': 'existing pitcher embedding (shared with the pitcher offset factor)',
                    'output_initial': 'zero', 'unknown_pitcher_residual': 'zero', 'new_dropout': 0}


def build_model(r, variant, seed, sizes):
    """대조 모델(같은 seed) + 투수×상태 잔차. 새 행렬은 fork_rng 안에서 만들어 전역 난수를 쓰지 않는다(010과 같은 방식)."""
    import torch
    if variant != TRAINER_VARIANT or r.get('identity_residual') != RESIDUAL: raise ValueError('pitcher x state residual: pitcher_batter trainer variant and the EXP-GPU14H-010 residual recipe only')
    control = batter.build_model(r, variant, seed, sizes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        state = torch.nn.Linear(STATE_COLUMNS, HIDDEN, bias=False)
        output = torch.nn.Linear(HIDDEN, CLASSES, bias=False); torch.nn.init.zeros_(output.weight)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 대조 모델의 하위 모듈을 같은 이름으로 등록(같은 객체). 대조 모듈 자체는 등록하지 않고 forward 호출에만 쓴다.
            self.pitcher = control.pitcher; self.batter_embedding = control.batter_embedding; self.batter_offset = control.batter_offset
            self.state = state; self.output = output; self.seq = control.seq
            object.__setattr__(self, 'control', control)

        def residual(self, x):
            h = torch.tanh(self.state(x[:, :STATE_COLUMNS]))
            return self.output(h * self.pitcher.embedding(x[:, STATE_COLUMNS].long()))

        def forward(self, x, tok=None, row_keys=None):
            return {'leaf': self.control(x, tok, row_keys)['leaf'] + self.residual(x)}

    model = Model()
    a, b = model.state_dict(), control.state_dict()
    if (sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters()) != ADDED_PARAMETERS
            or set(a) - set(b) != set(NEW_KEYS) or set(b) - set(a) or any(a[k] is not b[k] and not torch.equal(a[k], b[k]) for k in b)):
        raise ValueError('exact added parameter budget')
    return model


# PB-001 MPS 학습기 코드 객체 그대로 + 전역 build_model만 이 모듈의 것
_GLOBALS = dict(vars(pbtrain))
_GLOBALS['build_model'] = build_model
fit_body = types.FunctionType(pbtrain.fit.__code__, _GLOBALS, 'fit', pbtrain.fit.__defaults__, pbtrain.fit.__closure__)
fit_body.__kwdefaults__ = pbtrain.fit.__kwdefaults__


def fit(r, seed, package, out, synthetic=False, device='cuda', trace_recipe=False):
    """PB-001 MPS 학습기 본문으로 학습. receipt 문구 identity_interaction만 이 모델에 맞게 바꾸고 pitcher_state_residual 칸을 더한다."""
    if r.get('identity_residual') != RESIDUAL: raise ValueError('pitcher x state residual recipe required')
    result = fit_body(r, TRAINER_VARIANT, seed, package, out, synthetic=synthetic, device=device, trace_recipe=trace_recipe)
    if result.get('identity_interaction') != BASE_INTERACTION or 'pitcher_state_residual' in result: raise ValueError('unexpected trainer record')
    result['identity_interaction'] = INTERACTION
    result['pitcher_state_residual'] = copy.deepcopy(RECEIPT_RESIDUAL)
    return result
