"""전이 텐서 계약: 형상·dtype / NaN 없음 / valid 행 합 1±1e-5 / valid=False 행 합 0 / 규칙 마스크 셀 0 /
룩업 표 크기 = 축 길이 / meta 필수 키 / sha256 일치 (docs/interface-spec.md 검증 계약)."""

import json

import numpy as np
import pytest

from pitcheezy.interfaces import outcomes as O
from pitcheezy.interfaces import tensor as T
from pitcheezy.interfaces import validate as VD
from pitcheezy.interfaces.states import count_id, state_id
from pitcheezy.interfaces.validate import validate_transition

from conftest import make_tensor


def test_expected_shape():
    assert T.expected_shape(3, 2) == (3, 576, 225, 11)


def test_fixture_tensor_passes(small_tensor):
    assert validate_transition(small_tensor) == []
    assert small_tensor.P.dtype == np.float32
    assert small_tensor.valid.dtype == np.bool_
    assert small_tensor.n_obs.dtype == np.int32


def test_save_load_roundtrip_and_hash(tmp_path, small_tensor):
    d = tmp_path / "transition"
    small_tensor.save(d)
    assert {f.name for f in d.iterdir()} == set(T.FILES) | {"sha256.txt"}
    t2 = T.TransitionTensor.load(d)
    assert validate_transition(t2) == []
    assert np.array_equal(t2.P, small_tensor.P)
    assert t2.meta == small_tensor.meta
    assert set(json.loads((d / "meta.json").read_text())) >= set(T.META_REQUIRED_KEYS)


def test_sha256_mismatch_rejected(tmp_path, small_tensor):
    d = tmp_path / "transition"
    small_tensor.save(d)
    (d / "meta.json").write_text("{}")
    with pytest.raises(ValueError, match="sha256"):
        T.TransitionTensor.load(d)
    T.TransitionTensor.load(d, check_hash=False)  # 명시적으로 끄면 통과


def test_shape_dtype_violations():
    t = make_tensor()
    t.P = t.P.astype(np.float64)
    assert any("dtype" in p for p in validate_transition(t))
    t = make_tensor()
    t.P = t.P[:, :-1]
    assert any("형상" in p for p in validate_transition(t))
    t = make_tensor()
    t.n_obs = t.n_obs.astype(np.int64)
    assert any("n_obs dtype" in p for p in validate_transition(t))


def test_nan_rejected():
    t = make_tensor()
    idx = np.argwhere(t.valid)[0]
    t.P[tuple(idx)][O.FOUL] = np.nan
    assert any("NaN" in p for p in validate_transition(t))


def test_row_sum_violations_not_renormalized():
    t = make_tensor()
    idx = tuple(np.argwhere(t.valid)[0])
    t.P[idx] *= 1.01  # 합 1.01
    probs = validate_transition(t)
    assert any("valid 행 합" in p for p in probs)
    # 허용 오차 안이면 통과
    t = make_tensor()
    t.P[idx][O.FOUL] += 5e-6
    assert validate_transition(t) == []


def test_invalid_rows_must_be_zero():
    t = make_tensor()
    idx = tuple(np.argwhere(~t.valid)[0])
    t.P[idx][O.FOUL] = 0.5
    assert any("valid=False" in p for p in validate_transition(t))


def test_rule_mask_cells_must_be_zero():
    t = make_tensor()
    K = t.K
    # 3볼 카운트 상태에 '볼' 확률
    s = state_id(count_id(3, 1), 0, 0, K)
    pitcher, action = 0, 0
    t.valid[pitcher, s, action] = True
    row = np.zeros(O.N_OUTCOMES, dtype=np.float32)
    row[O.BALL] = 0.5
    row[O.FOUL] = 0.5
    t.P[pitcher, s, action] = row
    assert any("규칙 마스크" in p for p in validate_transition(t))
    # 2스트라이크 상태에 '스트라이크'
    t = make_tensor()
    s = state_id(count_id(0, 2), 0, 0, K)
    t.valid[pitcher, s, action] = True
    row = np.zeros(O.N_OUTCOMES, dtype=np.float32)
    row[O.STRIKE] = 1.0
    t.P[pitcher, s, action] = row
    assert any("규칙 마스크" in p for p in validate_transition(t))


def test_lookup_tables_must_match_axes():
    t = make_tensor()
    t.states = t.states.iloc[:-1]
    assert any("states" in p for p in validate_transition(t))
    t = make_tensor()
    t.outcomes = t.outcomes.drop(columns=["statcast"])
    assert any("outcomes" in p for p in validate_transition(t))
    t = make_tensor()
    t.pitchers = t.pitchers.iloc[:1]
    assert any("pitchers" in p for p in validate_transition(t))


def test_meta_required_keys():
    t = make_tensor()
    del t.meta["holdout_nll"]
    del t.meta["excluded_pitchers"]
    probs = validate_transition(t)
    assert any("holdout_nll" in p and "excluded_pitchers" in p for p in probs)


def test_b1_collapse_shape_is_same_family():
    """B1(SmartPitch류)은 K=1 + base_out 붕괴. 텐서 형상은 K=1 과 같다 (D6)."""
    t = make_tensor(K=1)
    assert t.P.shape == (2, 288, 225, 11)
    assert validate_transition(t) == []


def test_legacy_states_table_without_context_id():
    """맥락 이전(v1) 산출물의 states.parquet 에는 context_id 열이 없다 → C=1 이면 그대로 통과 (재검증 가능)."""
    t = make_tensor(K=2)
    t.states = t.states.drop(columns=["context_id"])
    assert validate_transition(t) == []
    t.states = t.states.iloc[:-1]  # 길이가 틀리면 여전히 실패
    assert any("states 표" in p for p in validate_transition(t))


def test_context_tensor_requires_context_id_column():
    """C > 1 이면 5열 states 표를 그대로 요구한다 (맥락을 잃은 표는 실패)."""
    t = make_tensor(n_pitchers=1, K=1, C=4)
    assert t.C == 4 and t.P.shape == (1, 1152, 225, 11)
    assert validate_transition(t) == []
    t.states = t.states.drop(columns=["context_id"])
    assert any("states 표" in p for p in validate_transition(t))


def test_row_sum_and_mask_checks_are_chunked():
    """행 합·마스크 검사는 투수 청크로 돈다 (C=7 에서 전체 float64 복사가 9GB). 청크 경계 너머의 위반도 세야 한다."""
    n_p = VD.P_CHUNK + 1  # 청크 2개 (마지막은 투수 1명)
    t = make_tensor(n_pitchers=n_p, K=1)
    assert validate_transition(t) == []
    last = n_p - 1
    s_v, a_v = (int(x) for x in np.argwhere(t.valid[last])[0])
    t.P[last, s_v, a_v] *= 1.01  # 마지막 청크에 valid 행 합 위반 1개
    s_m, a_m = state_id(count_id(3, 1), 0, 0, 1), 0  # 3볼에 '볼' = 규칙 마스크 위반. 합은 1 로 맞춰 행 합 검사와 분리
    assert (s_m, a_m) != (s_v, a_v)
    row = np.zeros(O.N_OUTCOMES, dtype=np.float32)
    row[O.BALL], row[O.FOUL] = 0.5, 0.5
    t.valid[last, s_m, a_m] = True
    t.P[last, s_m, a_m] = row
    probs = validate_transition(t)
    assert any("valid 행 합" in x and "1개" in x for x in probs)
    assert any("규칙 마스크 셀 ≠ 0: 1개" in x for x in probs)
    # 첫 청크에도 같은 위반을 넣으면 개수가 청크를 가로질러 합산된다
    t.valid[0, s_m, a_m] = True
    t.P[0, s_m, a_m] = row
    assert any("규칙 마스크 셀 ≠ 0: 2개" in x for x in validate_transition(t))
