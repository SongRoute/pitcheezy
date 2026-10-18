# CX-22 — wrapper 생성 로그와 오프라인 검사기의 통합 대조

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 실행 경계·결과 읽기 검토 PASS 및 Root 수용; 신규 3개 검사 통과, 별도 검토 재실행 없음. Base `d5ec90342c0d497b84a034bdb41d9bba79ee0896`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254419049 .

## 목적과 경계

[CX-21](cx-21-onekey-v4-receipt-checker-2026-10-19.md)은 직접 생성한 JSON으로 checker를 검증했다. 이번에는 기존 `Acceptance.scenario`와 `deferred_probe` 경계 안에서 **실제 wrapper 함수가 생성한 합성 로그**를 checker에 넘겨 형식·판정이 맞는지 대조한다. 실제 runner·HEAD transport·자격·운영 접속을 검증한 것은 아니다.

원본과 CX15 읽기 전용 사본은 실행 전후 모두 SHA-256 `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`로 같았다. 사본의 source pin 확인 후 기존 guard 아래 `cmd_run`, `cmd_probe`, `probe`, `assess`, 로그/정리 함수만 실행한다. 원본 수정·실행, 실제/private 로그·자격·자료 열람은 없다. 등록 파일·runner·실제 네트워크도 사용하지 않았다.

기존 네트워크/DNS/process 차단, 빈 환경, 가짜 TTY/getpass/clock/HEAD/Popen, 임시폴더 경계는 유지했다. checker import는 신규 method 안에서 guard 설치 후 이루어진다. checker는 stdlib만 사용하는 로컬 코드이며 top-level 외부 I/O가 없다. `REAL_HOST`는 로그 mode 문자열로만 사용하고 transport는 fake HEAD다.

수정은 `test_onekey_acceptance.py`와 이 보고서뿐이다. 기존 11개 test method의 AST·source segment, 기존 helper·guard의 source segment를 기준 커밋과 대조해 동일함을 확인했다. `deferred_probe`는 `http_status=401` 기본 인자와 fake HEAD의 반환 status만 최소 확장했으며 default 동작은 유지한다. 역변환한 기존 prefix도 정확히 같았다. 기존 NaN clock/changed target 두 RED는 실행하거나 기대를 바꾸지 않았다.

## 고정 코드와 실행

| 입력 | SHA-256 |
|---|---|
| acceptance test | `c395c7d42872961361e0923bb0a63f7532927e881e607383d772cf5ce5597d2c` |
| CX21 checker | `43b25c7f2c22e9b934629bafeebebbd4650cdd18c667913e99112ca7e51229c6` |
| CX15 wrapper snapshot와 현재 원본 | `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af` |

독립 에이전트 `queue_1019`가 위 SHA와 기존 guard·11개 검사 보존, 신규 3개 경계를 읽기 전용 검토해 PASS한 뒤 아래 **정확한 신규 3개 method만 1회** 실행했다. 사본 경로는 지정 CX15 인계 위치이며 원본 경로를 대입하지 않는다.

```sh
python3 -B tests/coordination/test_onekey_acceptance.py \
  --source REVIEWED_CX15_READONLY_SNAPSHOT \
  --sha256 53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af -v \
  Acceptance.test_receipt_checker_complete_generated_log \
  Acceptance.test_receipt_checker_deferred_expiry_generated_log \
  Acceptance.test_receipt_checker_generated_retry_recovery
```

CX22 `run-001`: `Ran 3 tests in 0.018s`, `OK`, exit0, failure/error/skip0. 모든 실행에서 tearDown의 forbidden boundary 목록이 비어 있었다. 원본·사본·test·checker의 실행 전후 SHA가 같다. stdout/stderr/exit/pins를 CX22 전용 새 파일로 보존했다.

| method | 실제 생성 경로·대조 |
|---|---|
| complete_generated_log | cmd_run의 scope403 + expired403 + revoked401, fake child1, wrapper rc0 → checker LOCAL_EVIDENCE_CONSISTENT/reasons[] 및 SAME_TARGET_BINDING 미검증 표시 |
| deferred_expiry_generated_log | expiry가 대기 상한 밖이라 확인을 미룸. scope403 + revoked401, fake child1, wrapper rc4/missing expired → checker INCOMPLETE/REQUIRED_EXPIRED_MISSING |
| generated_retry_recovery | cmd_run scope403, SKIP rc4 → deferred OSError rc4 → deferred503 rc2 → deferred401 rc0. checker는 각각 missing / missing / failed / consistent로 대조. main18 고정, post1/2/3, 이전 모든 행 보존, 최종 status열 `[403,None,503,401]`, COMPLETE/failed[]/missing[] 확인 |

위 표의 fake HEAD는 합계 9회, fake child는 3회다. 이는 실제 HTTP/프로세스 실행 수가 아니며 전부 mock 호출이다. 3개 method 안의 단계들을 추가 고유 테스트로 합산하지 않는다. 현재 Acceptance 클래스는 14개 method지만 이번에 기존 11개를 포함한 전체 실행은 하지 않았고 전체 PASS로 주장하지 않는다.

`git diff --check` 통과. scoped `tests/coordination`에서 `graphify update .` AST-only 1회 exit0, 138 nodes / 238 edges / 7 communities. ignored graph 산출물만 갱신했고 광범위 자료 탐색이나 의미 추출은 하지 않았다.

## 인계

현재 v4 wrapper가 생성한 위 합성 정상/미완/실패/복구 로그를 독립 checker가 수용 조건에 맞게 구분한다. 이는 실제 wire·배포·운영 GO, 등록·검토 pair, bearer/대상 동일성의 외부 증명을 대신하지 않는다. CX17/CX18 원본 결함 및 CX09/CX19 대기는 유지한다. 기존 테스트의 실패 기대나 원본을 수정하지 않았다.

COORD-05 단일 Git index 소유자가 독립 결과 검토 후 정확한 2파일만 선별 커밋한다. 작성자의 stage/commit은 없다.
