# CX-18 — deferred probe의 원래 대상 결합

상태: 독립 경계 검토·Root 수용 후 새2methods1회 실행. 동일 대상 control PASS, 변경 대상 첫HEAD전 거부 계약 FAIL. 독립 결과 검토 PASS·Root 최종 수용 완료(진단 인계, 제품 미수정).
CLAIM https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249664017 ; base `f322bf22a0eac0f26940b2d71205801c356d1bd7`. 작성자 coordination_docs, 검토자 coordination_review.

## 요구와 고정 입력

CX15 readonly snapshot SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`를 재사용한다. source:488–630의 cmd_run은 :519에 원래 upload_base/public_origin의 target_sha256을 기록하고, :633–661의 cmd_probe는 본 시험 로그에 후속 증거를 붙여 완료를 다시 판정한다. 다른 대상의401을 원래 대상 토큰 회수 증거로 합칠 수 없어야 한다는 기존 시험 식별 요구를 검사한다. **동적 검증 전에는 target 미결합 결함이 확정됐다고 하지 않는다.**

새 test SHA `bff04493579038cd95d7627923bb9ac8b30090c40c742470cb2105e0593d1209`. 새 helper+2methods를 제거한 bytes는 기존8methods가 담긴 SHA `09d55f0142e748e0e6a3f7a588019a4fde1609f3b0d3bb1c41f1a77c76f27e9c`로 완전히 복원된다. 기존8개 assertion/guard/fixture는 변경하지 않았다.

## 새 경로의 실행 경계

각 method는 기존 scenario(expired=False, answer=SKIP)로 **실제 cmd_run이 쓴 합성 미완 로그**를 새 TemporaryDirectory에 만든다. baseline은 rc4/MAIN_PASS_INCOMPLETE여야 한다. 실제 credentials/운영 로그를 읽거나 임의로 PASS 로그를 쓰지 않는다. 다음 cmd_probe도 실제 함수이며 read_token·token_fingerprint·probe·assess·finish를 유지한다.

추가 대역은 동일45자 fake bearer를 주는 getpass, 가짜 TTY, 401/application-json을 반환하는 head 전송뿐이다. 호출한 target/relative를 메모리에 기록한다. 원래 socket/DNS/process guard·빈 환경·가상시각·임시 폴더 격리를 유지한다. cmd_probe가 읽고 다시 쓰는 wrapper-log/요약/secret scan은 해당 합성 out에 한정된다. main/validate_pre/registration/check_client/build_plan_grant/실제 runner/HTTP는 실행하지 않는다(기존 scenario의 pin/build/child는 대역).

| 새 method | 입력 차이 | 사전 기대 |
|---|---|---|
| deferred_same_target_completes | 원래 synthetic.invalid 대상·같은 run_id·같은 bearer | HEAD1, rc0/PASS/COMPLETE, same_token_as_main_run=true |
| deferred_changed_target_refused_before_head | target 두 URL의 host만 changed.synthetic.invalid로 변경 | HEAD0, nonzero rc, 기존 MAIN_PASS_INCOMPLETE 유지 |

*.invalid는 순수 문자열이며 실제 네트워크 접근0이다. 후속 pre는 cmd_probe가 사용하는 최소 run_id/game_key/target만 생성한다. 실제 main/preconditions 통합 검증을 주장하지 않는다. 최소 helper는 이미 만든 합성 로그를 대상으로 후속 호출1회를 제공하며, 별도 계획된 retry/budget 검사는 아직 구현하지 않았다.

## 정확 실행 계획

독립 검토·Root 수용 뒤 새2methods만1회 실행한다. 기존8개 전체 재실행0. 출력은 신규 `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX18` 아래 배타 보존하고 실패/guard 오류를 그대로 기록한다.

```sh
python3 -I -B tests/coordination/test_onekey_acceptance.py \
  --source /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX15/onekey_wrapper-v4.py \
  --sha256 53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af \
  Acceptance.test_deferred_same_target_completes \
  Acceptance.test_deferred_changed_target_refused_before_head -v
```

원본·수정 복사본·실제 대상/토큰 변경은 하지 않는다. 원래 target/plan 식별을 첫 요청 전에 검증하는 최소 수정 후보를 소유자에게 제안하며 CX09 formal pair/운영 수용과 분리한다.

실패 진단 메시지에는 rc/result/completion scalar만 포함하며 주소·token·전체 로그는 출력하지 않는다. 판정·제어흐름을 바꾸지 않고 잘못된 완료 여부를 stderr에 남기기 위한 보완이다.

## 실제 결과

exit1, `Ran 2 tests in 0.013s / FAILED (failures=1)`. 동일 대상 control은 PASS했다. 변경 대상은 가짜 HEAD1회가 호출됐고 rc0/result=PASS_SYNTHETIC_HTTPS/completion=COMPLETE가 stderr로 관측되어 HEAD0 기대가 실패했다. 두 생성 baseline은 rc4/MAIN_PASS_INCOMPLETE를 만족했다. failure1/error0/skip0/guard실패0, test/snapshot SHA는 실행 전후 불변이다. 기존8개는 재실행하지 않았다.

`CX18/run-001.stderr.txt` SHA `6b201f905b3accf47004f1c85f7aec18866bd6f831373a01897ebbd0666313b4`; `run-001-exit.json` SHA `3fcf365c70bf262b9592ec1998f4c4609c36c75d846b67c5e86d7c9f03f07176`. 실패 기대와 원본 결과를 그대로 보존한다.

현재 snapshot의 cmd_probe는 같은 run_id와 salted token fingerprint를 검사하지만 기록한 target_sha256과 현재 target을 대조하지 않은 채 probe·finish에 도달했다. 최소 수정 후보는 현재 upload_base/public_origin을 원래 canonical 규칙으로 해시하여 target_sha256과 비교하고 불일치 시 **token 입력/첫 HEAD 이전**에 Refuse하는 것이다. 실제 코드/복사본 수정·수정안 실행은0이다. 이번 재현은 cmd_run 대역 환경→cmd_probe 함수 경로에 한정하며 main/등록 검증을 통과한 실제 운영 재현이라고 주장하지 않는다.

같은 모델 별도 에이전트 coordination_review가 source/실행기록 SHA와 결과를 대조해 진단 인계 PASS로 검토했다(추가 실행0). Root 최종 수용 뒤 tests/coordination에서 scoped graphify update .를1회 실행해 기존4 source AST를 갱신했다. ignored 그래프는 탐색 보조이며 재현 근거는 고정 source와 실행기록이다.
