# CX-15 — 현재 v4 snapshot의 기존 합성 회귀 진단

상태: 독립 실행 경계 검토·Root 수용 후 기존 7개 합성 회귀 1회 PASS. 독립 결과 검토 PASS·Root 최종 수용 완료, 커밋 대기.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249493768 . base `9fbccd33191becafbb07d5dee8de8b69d5008cb0`; 작성자 coordination_docs, 독립 검토자 coordination_review.

## 고정 입력과 범위

출력 root: `/Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX15`.
현재 L-onekey 원본 bytes의 SHA를 확인한 뒤 새 `onekey_wrapper-v4.py`로 배타 복사하고 읽기 전용(0444)으로 고정했다. snapshot SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`.
기존 `tests/coordination/test_onekey_acceptance.py` SHA `339f936e110367f9e8aeb448fa3dab20fd36383dbf77ff2e67493c66ae2e9618`는 불변이다. 테스트·원본 제품 소스를 수정하지 않는다.
이 진단은 현재 고정 소스의 F2/F3/F4 경로를 대조한다. 정식 CX09 source/registration/review 쌍 검증, main CLI/preconditions 검증, 실제 HTTPS·runner 실행 또는 v4 운영 수용이 아니다. registration/private/실자료/실제 자격 파일은 읽지 않았다.

## 실행 경계 대조

| 경로 | 기존 시험이 제공하는 경계 | 판단 |
|---|---|---|
| import/top-level :21–72,707–708 | SHA 검사 후 별도 module 이름으로 compile/exec; 실제 main 미호출 | stdlib import·상수·정규식·snapshot 경로 계산뿐, 등록/원장 읽기 없음 |
| check_client/build_plan_grant :214–268 | 두 함수를 가짜 고정 plan/grant로 대체 | Git·source/build 탐색·registration 읽기 없음 |
| cmd_run :488–630 | TemporaryDirectory의 out/ledger, 빈 os.environ, 가짜 TTY/getpass/clock | 실제 자격 접근 없음; JSON·요약·ledger·secret scan은 생성 임시 파일만 |
| head :324–340 | 가짜 HEAD 전송으로 대체; 실제 probe 판정 유지 | socket/connect/DNS 차단, 실제 HTTPS/SSL transport 실행 없음 |
| Popen :543–571 | 메모리 child가 임시 runner receipt만 작성 | 실제 runner/child 실행 없음; 다른 subprocess/os 실행도 ForbiddenIO 차단 |
| 창 재확인 :509,515,539 | 실제 check_window, 두 getpass가 가상시각 이동 | 긴 입력 후 첫 HEAD/runner 호출0 여부 검사 |
| 완료 판정 :394–433,468–485 | 실제 assess/finish·로그 판정 | SKIP/expiry defer가 전체 PASS가 되지 않는지 검사 |
| gateway 판정 :343–378 | 실제 probe/is_gateway_json + 가짜 status/header | JSON403 정상, cf-mitigated/HTML403 거부를 검사 |

ForbiddenIO는 BaseException 계열이라 wrapper의 일반 Exception 처리로 숨겨지지 않고 tearDown에서도 시도 목록0을 요구한다. snapshot은 새 코드 파일이므로 전체 소스를 읽어 위 경로를 확인했다. `os.urandom`은 salt 생성에 사용되며 자격 읽기·외부 실행은 아니다. 취약 입력에 대한 모든 syscall을 격리하는 일반 sandbox라는 뜻은 아니다.
정상 child는 이미 성공한 runner 영수증을 모사한다. 실제 runner 검증·시작창 운영 증빙·서버 정체성·실제 secret 유출 검사를 통과했다고 주장하지 않는다. cmd_probe의 나중 증거 병합·반복 probe·토큰 fingerprint 대조 경로는 이번 7개 범위 밖이다.

## 실행 계획과 사전 기대

기존 7 methods 그대로 1회: 정상 완료·일반 JSON403·cf-mitigated403 control 3개, 필수 revoked SKIP·expiry defer·긴 자격 입력·cf-mitigated 없는 HTML403 계약4개. 현재 소스에서 F2는 INCOMPLETE/rc4, F3는 입력 후 창 재확인, F4는 application/json 요구를 구현하므로 7PASS를 예상하되 실제 결과는 실행 후 따로 기록한다.

```sh
python3 -I -B tests/coordination/test_onekey_acceptance.py \
  --source /Users/song/Projects/pitcheezy-pm-jobs/20261018-codex-continuous/CX15/onekey_wrapper-v4.py \
  --sha256 53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af -v
```

독립 경계 검토·Root 수용 뒤 위 명령을 1회 실행해 stdout/stderr를 새 CX15 출력에 배타 저장했다. 실패/오류/guard 발동은 없었다. 기존 suite의 일반 discovery는 이미 검증됐고 변경되지 않아 이번에 반복하지 않았다.

## 후속 검토 가치

CX06은 고정 v2 wrapper-log 계약의 엄격한 대조기다. 현재 v4는 wrapper-log-v2(:517), completion/required_evidence/같은 토큰 fingerprint, 후속 시도 count와 동일 kind의 마지막 행 우선(:394–433,633–661)을 사용한다. 따라서 v2 checker에 v4를 자동으로 호환시켜 부르는 대신 **별도 CLAIM으로 schema·중복/재시도·post count·same-token 증거 의미를 대조할 가치가 있다**. 이번에는 checker를 읽거나 바꾸거나 실행하지 않았다. CX09 정식 쌍 인계 대기는 그대로다.

현재 N24는 expired/revoked/deferred 요청 전에 post_requests를 증가시킨다(:590,617,657). 과거 응답만 세던 상태를 현재 결함으로 반복하지 않는다. Root가 제시한 clock.ntp_offset_s 비유한 값(NaN) 후보(:122–124)는 별도 순수 검증 후보로만 남긴다. main/validate_pre를 실행하지 않아 이번에 재현하거나 확정한 결함은 아니다.

## 실제 결과

exit0, `Ran 7 tests in 0.028s / OK`: control3개와 계약4개 모두 PASS, failure0/error0/skip0. 기존 tearDown의 ForbiddenIO 시도0 요구도 통과했다. 실행 전후 snapshot/test SHA가 위 값과 같았다. 새로 테스트나 기대값을 고친 결과가 아니다. 정상 완료 control은 실제 cmd_run·assess가 `(0, PASS_SYNTHETIC_HTTPS)`와 probe403/403/401 및 main18을 만족함을 확인했다. SKIP/expiry defer는 전체PASS가 아님, 긴 입력은 HEAD/Popen 각각0, HTML403은 증거로 수용하지 않음을 기존 assertion 그대로 확인했다.

출력 `run-001.stdout.txt`는 비어 있다. `run-001.stderr.txt` SHA `aee73b4e7ec2f1cd8a595dc32ac13e342569efdc80a243f08c0a1320227617a8`, `run-001-exit.json` SHA `bfdedf06596faa2042b0fedd9653ec1023d2e96ed56dc40b607985bc8438c1d5`. 각 시나리오 임시 출력은 기존 suite의 TemporaryDirectory 정리 규칙을 따랐고 영구 실제 로그를 열지 않았다.

같은 모델 별도 에이전트 coordination_review가 source/test와 실행기록 SHA·7개 결과를 대조해 PASS했다. 검토자의 추가 실행은0회다.

실행기/제품 code 변경0이므로 graphify 갱신은 하지 않았다. 이 결과로 이전 고정 v2의4실패 증거를 바꾸지 않으며, 현 snapshot에서 해당4계약을 충족한 좁은 합성 진단으로만 해석한다. 별도 v4 전체검토·실제 공동시험·CX09 gate 또는 배포 GO가 아니다.
