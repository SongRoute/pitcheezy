# CX-21 — v4 wrapper 영수증의 오프라인 일관성 검사

작성자 `implementation_1019`, 독립 검토자 `queue_1019`. 상태: 합성 검사 통과, `queue_1019` 독립 읽기 검토 PASS 및 Root 수용. 별도 검토자의 재실행 없음. Base `f821ea2b168c04c1a79f79faf28ccabf581324f6`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6254352364 .

## 범위

`check_onekey_receipt.py`는 기존 v2 wrapper의 `wrapper-log-v1` 경로를 유지하고 v4 wrapper의 `wrapper-log-v2` 경로를 추가한다. 기존 runner 검사를 공통 함수로 추출했으며 기존 8개 unittest 내용은 보존했다. 새 검사는 8개 method다.

근거 source는 읽기 전용 `L-onekey/onekey_wrapper.py`, SHA-256 `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af`다. `cmd_run`, `cmd_probe`, `probe`, `assess`, `finish`의 로그 생성과 판정을 대조했다. 원본 import/실행/수정, 실제/private 로그·자격 열람, 외부 HTTP, 운영 자원 사용은 하지 않았다. 새로운 dependency도 없다.

명령과 출력 상태는 [CX-06](cx-06-onekey-receipt-checker-2026-10-18.md)과 같다. 입력 schema에 따라 경로를 선택하며 알 수 없는 schema는 거부한다.

```sh
python3 -B tests/coordination/check_onekey_receipt.py EXPLICIT_WRAPPER_LOG \
  --scope required --expiry required --mode LOOPBACK_HARNESS
```

`LOCAL_EVIDENCE_CONSISTENT`/exit0, `INCOMPLETE`/exit1, `INVALID`/exit2는 단일 파일의 내적 일관성 판정이다. v4의 `basis`는 `V4_SINGLE_WRAPPER_LOG_EMBEDDED_RUNNER_ONLY`. 기존 unverified 항목은 모든 판정에 유지하고 v4에는 `SAME_TARGET_BINDING`을 추가하며 원시 값·주소·경로를 출력하지 않는다.

## v4 대조

- 외부 scope/expiry 선언과 `required_evidence`의 정확한 bool·키 구성을 대조한다. `runner_pass`는 엄격한 bool이어야 하며 embedded runner와 main 예산으로 재계산한다. `False`가 올바른 실패 기록이면 그 자체를 구조 오류로 분류하지 않는다.
- 각 probe의 기대 status, pass, gateway MIME(`application/json`, parameter 허용), mitigation 표시를 대조한다. 응답 없는 transport/interrupt 행은 `status=None`, `pass=False`, error 문자열 형태이며 응답 meta가 없어도 구조상 허용하고 missing으로 취급한다.
- kind별 마지막 행으로 증거를 판단한다. 앞선 실패에서 후속 PASS로 복구하거나 앞선 PASS가 후속 실패/무응답으로 바뀌는 흐름 모두 대조한다. 과거 행은 보존하고 모든 행의 타입·자기모순은 계속 검사한다.
- revoked 또는 deferred 행의 `same_token_as_main_run`은 true여야 한다. 이는 기록 필드의 대조이며 실제 bearer 동일성 입증이 아니다.
- 최초 non-deferred out_of_scope만 main에 포함하고 모든 deferred 행은 종류와 응답 유무에 관계없이 post에 포함한다. main<=18, post<=3, 합<=21 및 기록 카운트를 대조한다. 미선택 scope/expiry의 deferred 행은 source가 허용하므로 존재만으로 거부하지 않으며 필수 증거로 승격하지 않는다.
- `main_result`, `revocation`, `completion`, `result`, `missing_evidence`, `failed_evidence`를 독립 재계산한다. upload_off 503은 회수 401을 대신하지 않는다. `secret_found`가 true인 완료 실패 및 `SECRET_FOUND_IN_OUTPUT` override를 허용하지만 일관성 통과로 판정하지 않는다.

검사 정책은 source의 `runner_pass`보다 엄격하다. 기존 v1의 고정 17-method 순서(13번째 `DELETE:upload_conflict`), 유한 elapsed `0<=x<120`, runner/wrapper stop_code 없음 요구를 유지한다. 그러므로 source가 PASS를 쓴 로그도 추가 정책에 따라 INCOMPLETE/INVALID일 수 있다. 메서드·타입·요약 자기모순은 INVALID, 유효하게 기록한 미완/실패는 INCOMPLETE다. runner 누락은 INCOMPLETE 사유다. source timeout의 보수적 main 상한 가산으로 18을 넘은 기록은 실제 wire 수 추정 없이 예산 초과 INVALID로 표시한다. 총 시도 수는 로그의 행과 카운트 대조이지 실제 wire 측정이 아니다.

중복 JSON키·비유한 수·bool 수치·256KiB 상한·regular file·symlink 거부 등 기존 입력 경계는 유지한다. 실행 당시의 입력/출력은 전부 코드로 생성한 값이다. `REAL_HOST` unit case는 mode 문자열 대조이며 실호스트 접속이 아니다.

## 검증 기록

명령:

```sh
python3 -B -m unittest discover -s tests/coordination -p test_onekey_receipt.py -v
```

전용 CX21 산출물에 stdout/stderr/exit를 새 파일로 보존했다.

| 실행 | 실제 결과 | 조치 |
|---|---|---|
| run-001 | 16 methods, 0.017s, failures 1 / errors 0, exit1 | 신규 정상 CLI fixture의 미정규화 임시경로가 macOS `/var` symlink를 통과해 기존 거부 경계에 걸림. 기존 v1 fixture처럼 `Path(folder).resolve()` 1줄 수정. 최초 실패 기록 보존 |
| run-002 | 16 methods, 0.017s, failures/errors/skips 0, exit0 | 당시 코드 통과. 기존 8 + 신규 8이며 반복 실행을 고유 검사로 합산하지 않음 |

Root 읽기 검토에서 대상 동일성 미검증을 CLI에서도 명시하도록 요청했다. v4 `unverified`에 `SAME_TARGET_BINDING` 1개를 추가하고 v4 기대값만 수정했다. run-003은 16 methods, 0.017s, failures/errors/skips 0, exit0으로 변경 후 통과했다. run-002와 run-003의 stderr 바이트가 같아 SHA도 같다. 최종 source는 아래 표다.

신규 검사는 정상 선언 조합과 CLI, SKIP·무응답 후 복구, PASS→실패/무응답→PASS, deferred scope와 선택적 증거, 시도 예산·카운트, summary·required·runner bool, gateway·same-token·타입, upload_off·secret override, 엄격한 runner 정책과 이력 모순을 포함한다. CLI 정상 경로도 unittest 내부에서 임시 JSON 하나로 실행했다.

| 파일 | SHA-256 |
|---|---|
| checker | `43b25c7f2c22e9b934629bafeebebbd4650cdd18c667913e99112ca7e51229c6` |
| test | `a243ebedede6b859e1d4572c1a11c7501a19aa9de47e8d91b9dc604ffb6fe037` |
| run-001.stderr.txt | `08ffd34268ba53e93a4c855a30ad17db6ed89e26063073465f9b43a70a35d4a1` |
| run-002 / run-003.stderr.txt | `a458a1d2dd743388fbd43ee61d62f967e843af55f3b3a3b5365ca43872d31ebd` |
| run-002 / run-003.exit.txt | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |

`git diff --check` 통과. `tests/coordination`에서 `graphify update .` AST-only 2회 exit0(코드 변경 후 갱신 포함), 135 nodes / 231 edges / 6 communities. 하위 ignored 출력만 갱신했으며 전체 저장소·실자료·문서 의미 추출은 하지 않았다.

## 한계와 인계

v4 로그에 target hash가 있어도 이 검사기는 실제 대상·preconditions·원격 응답을 교차 입증하지 않는다. source의 deferred target binding 결함(CX18)과 NaN clock 결함(CX17)을 수리하지 않았다. 현재 로그만으로 실제 bearer 동일성, artifact 진위, 실제 wire, 승인/등록/독립 검토, 최초 요청 창, 별도 영수증 identity, 팀원 확인이나 운영 GO를 증명하지 않는다. CX09 formal 수용 또는 CX19 원본 patch 대기를 닫지 않는다.

COORD-05가 독립 검토 후 정확한 3파일만 선별 커밋한다. 작성자는 Git index와 커밋을 조작하지 않았다.
