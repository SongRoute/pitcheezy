# CX-17 — one-key 시계 오차의 유한성 경계

상태: 독립 경계 검토·Root 수용 뒤 단독 method1회 실행. NaN 수용 실패1개 재현, 독립 결과 검토 PASS·Root 최종 수용 완료. 진단 인계 완료이며 제품은 미수정이다.
CLAIM https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249598918 ; base `359e0c657269209278c386551c24b948428d8cc0`. 작성자 coordination_docs, 검토자 coordination_review.

## 입력·기대·실행 경계

CX15의 읽기 전용 `onekey_wrapper-v4.py` SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`를 재사용한다. 원본 L-onekey source/registration/private/자격 파일은 새로 읽거나 변경하지 않는다. snapshot의 validate_pre:106–213 중 clock 검사:122–124가 대상이다. 현재 타입과 abs(value)>5 검사만 있어 NaN 거부가 빠질 가능성을 확인하는 진단이며, 실행 전에는 재현된 결함으로 주장하지 않는다.

기존 `Acceptance`의 import 전 SHA·socket/DNS/process guard, 빈 환경, 가상시각, 임시 폴더를 재사용한다. 새 method는 validate_pre(harness=True)만 부른다. 이 모드에서는 registration 파일 검사 분기가 실행되지 않는다. main/check_client/build_plan_grant/cmd_run/cmd_probe/head/runner는 호출하지 않는다. 합성 localhost URL은 문자열 검증에만 사용한다. 실제 승인·팀원 합의·credentials가 아니라 순수 검사에 필요한 합성 필드다.

새 method 안에 전체 필수 필드를 만족하는 합성 preconditions baseline을 만들었다. runtime pin 값은 검토한 snapshot의 EXPECTED_RUNTIME 상수를 재사용하며 파일을 열지 않는다. 검사 대상 clock의 기대 범위는 source 계약에 맞춰 **유한한 int/float, -5≤offset≤5**로 고정했다. 다른 필드 오류가 없는지 매 입력에서 따로 확인한다.

| 구분 | 고정 값 | 기대 |
|---|---|---|
| 정상 baseline | 0 | 오류0; 이후 matrix의 출발 조건 |
| 허용 matrix3개 | 0, -5, +5 | 오류0 |
| 비유한 matrix3개 | NaN, +Infinity, -Infinity | clock 오류1 |
| 범위 밖 matrix2개 | +5.001, -5.001 | clock 오류1 |
| 타입 matrix3개 | True, False, 문자열 "0" | clock 오류1 |

**새1 method, baseline1회+matrix11 subcases=validate_pre12호출**이다. method 수와 subcase 수를 섞어12tests라고 하지 않는다. 기존7methods/assertions/helper/guard/fixture는 수정하지 않았고 추가 method를 제거한 bytes가 이전 SHA `339f936e110367f9e8aeb448fa3dab20fd36383dbf77ff2e67493c66ae2e9618`로 복원됨을 확인했다. 현재 test SHA `09d55f0142e748e0e6a3f7a588019a4fde1609f3b0d3bb1c41f1a77c76f27e9c`.

## 정확 실행 계획

독립 경계 검토·Root 수용 뒤 기존7개를 반복하지 않고 새 method만1회 실행한다. 실제 shell stdout/stderr는 새 `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX17` 아래 배타 저장한다. 실패는 기대를 완화하거나 원본을 고쳐 숨기지 않는다.

```sh
python3 -I -B tests/coordination/test_onekey_acceptance.py \
  --source /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX15/onekey_wrapper-v4.py \
  --sha256 53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af \
  Acceptance.test_clock_offset_requires_finite_bounded_number -v
```

## 실제 결과와 최소 수정 제안

고정 입력으로 단독1회 실행: exit1, `Ran 1 test in 0.006s / FAILED (failures=1)`. 정상 baseline은 오류0이었다. matrix11개 중 NaN만 기대와 달랐고, 나머지10개는 기대와 일치했다. NaN에서 validate_pre가 오류목록 `[]`를 반환해 clock 거부 assertion이 실패했다. error0/skip0/guard실패0, test/snapshot SHA는 실행 전후 불변이다. 기존7개는 재실행하지 않았다.

`run-001.stderr.txt` SHA `59758ed82031cfc5a600bf95046624dff6583f9b78877bd9a47b5bcdd833c164`; `run-001-exit.json` SHA `3fcf365c70bf262b9592ec1998f4c4609c36c75d846b67c5e86d7c9f03f07176`; stdout은 비어 있다. 실패 영수증과 red assertion을 그대로 보존한다.

원인은 snapshot:122–124에서 `abs(NaN) > 5`가 false이므로 범위 오류로 분류되지 않는 것이다. 소유자에게 `type` 검사를 유지한 채 `not (-MAX_NTP_OFFSET_S <= value <= MAX_NTP_OFFSET_S)`로 범위 검사를 바꾸는 최소 후보를 제안한다. 원본/수정 복사본을 만들거나 이 수정안을 실행하지 않았다. 실제 clock 동기·JSON main 입력·운영/배포 또는 CX09 formal gate 수용을 이번 순수 함수 검사로 대체하지 않는다. 추가 full-suite/main/실host 실행 계획은 없다.

같은 모델 별도 에이전트 coordination_review가 source/결과 SHA 및 실패기록을 대조해 진단 인계 PASS로 검토했고 Root가 수용했다. 검토자의 추가 실행0. tests/coordination 범위에서 graphify update .를1회 실행해 기존4 source AST만 갱신했다(ignored 출력, 코드 의미 검증 근거 아님).
