# CX-20 — 후속 확인 재시도·요청 예산

상태: 독립 경계 검토·Root 수용 후 새1method1회 PASS. 독립 결과 검토 PASS·Root 최종 수용 완료, 커밋 대기.
CLAIM https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249730354 ; base `645632ade8c91a7902d1a551ba5626aab6f7993b`. 작성자 coordination_docs, 검토자 coordination_review.

## 고정 범위와 기대

CX15 readonly snapshot SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af` 재사용. 실제 cmd_probe:633–661, probe:343–368, assess:394–433을 검사한다. source는 응답 없는 전송 오류를 missing으로 남기고 동일 kind의 마지막 행을 판정하며, 요청 전 post count를 증가시킨다. POST_BUDGET3을 첫 요청 전에 검사한다(:647–648).

기존 scenario(expired=False, answer=SKIP)로 새 임시 미완 run을 만든다. 같은 target/run_id/45자 합성 bearer를 유지하고 실제 cmd_probe·probe·assess·finish를 호출한다. 기존 deferred_probe helper에 선택적 transport_error=False만 추가해 가짜 HEAD 시도를 메모리에 센 다음 OSError를 낸다. default는 기존401 그대로다. 실제 timeout 대기·실제 HTTP는 없고 **모의 전송 OSError** 경로를 검사한다.

| 순서 | 가짜 응답 | 기대 rc/완료 | post count / 전체 probes |
|---|---|---|---|
| baseline | 기존 scope403+runner영수증, 회수SKIP | 4/INCOMPLETE | 0/1 |
| 후속1 | OSError, 응답 없음 | 4/INCOMPLETE, missing revoked | 1/2 |
| 후속2 | 같은 대상·토큰401 | 0/COMPLETE, missing 없음 | 2/3 |
| 후속3 | 같은 대상·토큰401 | 0/COMPLETE | 3/4 |
| 후속4 | 예산 소진 | nonzero, 첫HEAD0 | 기존 COMPLETE 로그 그대로3/4 |

매 단계 main_requests18을 유지하고 최초 scope행과 실패한 revoked 행을 보존한다. 마지막 실패행은 뒤의 성공으로 판정이 바뀌어도 지우지 않는다. 네 번째 거부가 기존 완료 로그를 실패로 바꾸는 것은 기대하지 않는다.

## 경계·변경 검증

기존 SHA·socket/DNS/process guard, 빈 환경·가짜 자격/TTY·가상시각·TemporaryDirectory 경계를 유지한다. cmd_probe의 파일 읽기/쓰기/secret scan은 생성한 임시 out만 대상으로 한다. main/validate_pre/실registration/private/원본runner/실host 실행0. source 원본·수정 복사본을 만들거나 바꾸지 않는다.
새 test SHA `442bf22c1d93b18f6e7b36d9abfc3637c4d9ee61dcf8617ecf92e59f1db9d647`. 새 method와 helper의 선택적 OSError 세 줄을 제거하면 이전 SHA `bff04493579038cd95d7627923bb9ac8b30090c40c742470cb2105e0593d1209`로 복원된다. 기존10methods와 기대는 그대로며 이번에 재실행하지 않는다.

## 정확 실행 계획

독립 경계 검토·Root 수용 뒤 새1method만1회 실행한다. 결과 stdout/stderr/exit는 신규 `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX20`에 배타 저장한다.

```sh
python3 -I -B tests/coordination/test_onekey_acceptance.py \
  --source /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX15/onekey_wrapper-v4.py \
  --sha256 53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af \
  Acceptance.test_deferred_retry_counts_attempts_and_stops_at_budget -v
```

## 실제 결과

exit0, `Ran 1 test in 0.009s / OK`, failure0/error0/skip0/guard실패0. 위 표의 생성 baseline과4개 후속 단계가 모두 assertion을 통과했다. 모의 HEAD 시도는 후속1~3에서 각1회, 후속4에서0회였으며 실제 네트워크 요청은0이다. post3·전체probes4 및 기존COMPLETE 로그가 네 번째 거부 뒤 그대로 유지됐다. test/snapshot SHA는 실행 전후 불변이며 기존10methods는 재실행하지 않았다.

`CX20/run-001.stderr.txt` SHA `abbe7f27ee43e90beea02356c6826e966bf04e2b4035d779f07731cd518602e4`; `run-001-exit.json` SHA `bfdedf06596faa2042b0fedd9653ec1023d2e96ed56dc40b607985bc8438c1d5`; stdout은 비어 있다.

검사 통과는 이 고정 함수 경로의 합성 예산/재시도 의미에 한정한다. CX17 NaN·CX18 target 결합 실패의 수리 또는 전체11methods PASS, CX09 formal pair, 실제 토큰 회수/운영/배포 GO를 뜻하지 않는다.

같은 모델 별도 에이전트 coordination_review가 source/test·실행기록 SHA와 assertions를 대조해 결과 PASS로 검토했다(추가 실행0). Root가 최종 수용했다. tests/coordination에서 scoped graphify update .1회로 기존4 source AST만 갱신했고 ignored 그래프는 비커밋이다.
