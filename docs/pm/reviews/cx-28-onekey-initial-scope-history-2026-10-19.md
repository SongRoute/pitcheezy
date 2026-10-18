# CX-28 — 독립 영수증 검사기의 초기 scope 선행성

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 독립 재현·수정 경계·최종 결과 검토 PASS 및 Root 수용. 17method 통과, 검토자의 추가 재실행 없음. Base `3b2b2e789eb54f3cd6e7834f9f22078cb98b7d48`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254752454 .

고정 원본 wrapper SHA `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`의 cmd_run:529–541은 required out_of_scope 검사가 통과하지 않으면 runner 생성 전에 중단한다. 그러나 **우리 offline checker**는 마지막 scope 행만 보아, 초기 행이 없거나 FAIL이어도 뒤의 deferred PASS가 앞선 runner 성공과 공존하는 불가능한 이력을 일관된 것으로 수용했다. 원본 wrapper의 별도 미해결 결함을 고친 작업이 아니다.

신규 `V4ReceiptTests.test_runner_requires_initial_scope_pass` 1method에 불가능 이력2개와 정상 control을 추가했다. scope required/expiry not-selected 상태에서 초기 scope 없음(main17) 또는 초기503 FAIL(main18), runner17 성공, 뒤 deferred403 PASS로 수량·summary까지 일치시켰다. 정상 initial PASS→deferred FAIL→PASS는 미완→일관으로 회복하고, runner 없는 초기 scope 실패 기록은 INCOMPLETE로 유지하도록 검사한다. 기존16method 본문은 바이트 동일하다.

수정 전 `repro-001`은 지정 신규1method만 실행해 **exit1, failures2, 0.000s**를 기록했다. 두 negative 모두 실제 `LOCAL_EVIDENCE_CONSISTENT`를 반환해 INVALID 기대에 실패했다. 정상 controls는 실패하지 않았다. 독립 검토자가 실제 stderr를 대조한 뒤 checker에5줄만 추가했다. nondeferred scope의 이미 검사한 판정을 보존하고, required scope와 runner 객체가 있는데 초기 PASS가 없으면 `RUNNER_WITHOUT_INITIAL_SCOPE_PASS`로 INVALID를 추가한다. 잘못된 expected/type/gateway 등 기존 INVALID 판정은 유지한다. runner 없는 중단은 이 새 조건에 해당하지 않으며 v1 처리와 latest-row summary/후속 복구는 그대로다.

독립 diff 검토 후 `run-001`에서 기존16+신규1, **17method를1회 실행해 exit0/17PASS/0.017s**였다. 최초 실패와 수정 후 실행은 별도 보존하며 subcase·반복 실행을 고유 test 수에 합산하지 않는다.

```sh
python3 -B tests/coordination/test_onekey_receipt.py V4ReceiptTests.test_runner_requires_initial_scope_pass -v
python3 -B tests/coordination/test_onekey_receipt.py -v
```

증거는 20261019-codex-continuous 전용 `CX28/`의 각 stdout/stderr/exit/pins에 있다. 두 실행 모두 test/checker 및 원본·읽기 전용 snapshot의 실행 전후 핀은 동일했다(수정은 실행 사이에 수행). 원본과 snapshot은 읽기만 했고 import/실행하지 않았다. 생성 JSON·stdlib 검사만 수행했으며 실제 로그/자격/네트워크/실자료·운영 접근은 없다.

| 대상 | SHA-256 |
|---|---|
| 수정 전 checker | `43b25c7f2c22e9b934629bafeebebbd4650cdd18c667913e99112ca7e51229c6` |
| 최종 checker | `e8dcd6e18acbfde8cc52ff00e27ad3bc7ee2e77d2aa5a16c28e2f50621726e82` |
| test (두 실행 동일) | `5a1d004272d068e49aaaad1e0d4a4b013c2f51b5d5fccd69d5cec5d72ef5be7e` |
| repro-001.stderr.txt | `49596fbadcad3a9056d9438f2fb5e44059b51b24acf6dc5eb86c24fb93edf9ec` |
| run-001.stderr.txt | `c6b8a0d87b319b5636a24633114715f72774f6992261c3c0788e82cd39a85c9f` |

`git diff --check` 통과. 최종 코드에서 `tests/coordination` scoped AST-only graph update1회(142nodes/247edges/9communities), ignored 출력만 갱신했다. 이것은 단일 로그 내부 이력 모순 검출 개선이며 로그 진위·동일 target 결합·실호스트 운영 GO를 증명하지 않는다. 독립 최종 검토 후 COORD-05 단일 index 소유자에게 정확한 checker/test/보고서3파일을 인계한다. 작성자는 stage/commit하지 않았다.
