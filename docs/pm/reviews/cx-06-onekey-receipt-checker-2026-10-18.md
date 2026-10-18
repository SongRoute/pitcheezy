# CX-06 — v2 단일 영수증의 오프라인 일관성 검사

상태: 독립 source 검토 PASS/root 수용 → 합성 unittest8개 PASS → 독립 결과 검토 PASS/root 최종 수용. 작성자 `coordination_docs`, 검토자 `coordination_review`.
CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248803164 . base `dd0209c060226db27a7139b6af2bd4335a0fa43e`.

## 범위와 근거

- 고정 v2 wrapper SHA `479162d6f19fd0c624a0feccd8a6bf1a235e0f5aab8082f387dc5c22c454bb88`, registration v2 SHA `8e0b9fd1febd8e9ff9daca43822e37a2e073b39a8f5990c13cd01ad9c39635ca`의 출력/수용조건 대조기다. 새 운영 정책이 아니다.
- [CX-01](cx-01-onekey-audit-2026-10-18.md)과 [CX-04](cx-04-onekey-regressions-2026-10-18.md)의 고정 근거 및 CX-04 합성17행 receipt 형태를 재사용했다. 현재 Claude가 수정 중인 원본 v3·실제/private 로그·자격·실자료는 읽거나 실행하지 않았다.
- 출력 형식은 CX-04 시작 시 위 SHA를 확인하고 읽은 v2 `cmd_run:392–516` 원문에 근거한다. wrapper는 원 runner의 `result='confirmed'`를 `METHOD:ok`로 요약하고 나머지는 `METHOD:원래result`로 보존한다. 따라서13번째 DELETE만 `upload_conflict`다. 테스트 정상 fixture는 checker 상수를 복사하지 않고 독립 literal17개로 고정했다.
- `tests`와 `apps/observer/scripts`의 source 검색에서 동일 wrapper-log 검사기는 없었다. 최소 stdlib 검사기1개와 생성 JSON unittest1개만 추가하며 network/subprocess/원본 실행은 없다.
- 단일 명시 wrapper-log 내부 **embedded runner**만 읽는다. 별도 runner 파일·standalone probe를 찾아 읽거나 자동 병합하지 않는다. 선언된 mode/scope/expiry에 대해 기록이 일관적인지만 판단한다.

## 입력과 판정

```sh
python3 -B tests/coordination/check_onekey_receipt.py EXPLICIT_WRAPPER_LOG \
  --scope required --expiry required --mode LOOPBACK_HARNESS
```

scope/expiry는 각각 `required|not-selected`, mode는 `LOOPBACK_HARNESS|REAL_HOST`를 **모두 명시**한다. revoked는 항상 필수다. 판정은 `LOCAL_EVIDENCE_CONSISTENT`(exit0), `INCOMPLETE`(exit1), `INVALID`(exit2). 누락/실패 증거는 미완, 잘못된 타입·구조/자기모순은 INVALID가 우선한다. 출력은 고정 reason 코드와 미검증 항목뿐이며 원시 값·주소·입력 경로를 echo하지 않는다.

| 대조 | 조건 |
|---|---|
| wrapper | schema/mode 일치, result PASS, credentials_recorded isFalse, 중단 코드 없음 |
| embedded runner | exit int0, result PASS, requests int17, elapsed 유한수 `0≤x<120`, worker stopped isTrue, pending int0, cleanup `confirmed_tombstone_retained`, stop_code null |
| methods | 17개 고정 순서,13번째 `DELETE:upload_conflict`,나머지 해당 method`:ok` |
| probe | 지원 kind만/중복 금지, 선택된 scope/expiry 및 revoked 필수, 미선택 row 금지; out_of_scope→expired→revoked→upload_off에서 없는 항목만 빠진 순서; expected/status/pass/edge 표시 대조, HTML 거부 |
| upload_off | 선택적503 기록만, revocation_evidence isFalse; revoked401 대체 불가 |
| 요청 | main=`17+실제 scope row 수`이며≤18; post=expiry/revoked/upload_off의 실제 행 수≤3; main+post≤21 |
| trust boundary | 중복 JSON키·NaN/Infinity/overflow float·잘못 container·bool 수치 거부, 명시 regular file1개≤256KiB, 조상/최종 symlink 거부 |

256KiB는 검사기의 입력 안전 상한이며 등록된 실험 예산이 아니다. Content-Type이 없으면 HTML로 판정하지 않지만 실제 gateway/wire를 증명하지도 않는다. 필수 probe가 빠졌을 때 실제 main 요청 수와도 모순되면 INVALID 사유가 함께 남을 수 있다.

모든 결과에 `DEPLOY_GO`, `WHOLE_JOINT_TEST_COMPLETION`, `INDEPENDENT_SECRET_SCAN`, `FIRST_REQUEST_WINDOW`, `STANDALONE_RUN_IDENTITY`, `PARTNER_CONFIRMATION`, `ACTUAL_GATEWAY_OR_WIRE`, `ARTIFACT_AUTHENTICITY`가 미검증으로 출력된다. 시작 창·별도 영수증 identity·팀원 설정/확인·비밀 독립검사는 이 파일만으로 입증할 수 없다. 기존 GATE-06 합의·운영 판정을 대체하지 않는다.

## 검증 계획과 인계

```sh
python3 -B -m unittest discover -s tests/coordination -p test_onekey_receipt.py -v
```

생성 fixture로 선택 조합·mode·upload_off 정상,필수 probe 누락,runner 실패,정확 method/카운트,HTML/edge/상태 모순,bool수치/비유한수/중복키/입력상한/symlink,CLI 필수 선언·원시값 비출력을 대조한다. 실제/private 로그 실행은0이며 `REAL_HOST` 정상 unit case도 문자열 mode 대조일 뿐 실호스트 실행이 아니다.

2026-10-18 위 명령을 작성자가1회 실행했다: exit0, `Ran 8 tests in 0.016s`, `OK`, failure/error/skip0. 8개 unittest method 안의 table-driven subcase를 별도 검사 수로 합산하지 않는다. 실제/private 로그0·v3 원본 열람0·network0·subprocess0이며 생성 임시파일은 정리했다.

| 실행 후 불변 source | SHA-256 |
|---|---|
| check_onekey_receipt.py | `905c923b11ee15181819736c4d4c6bcadcd570ecb50bd5da91fea48e92e9fea3` |
| test_onekey_receipt.py | `5a1f08f566e0674a03de9f039ede93e3c3df239ed8aba470e24b5f0a36b24cbb` |

같은 모델의 별도 에이전트 `coordination_review`가 source SHA와 실행기록·보고서를 대조해 PASS했고 root가 최종 수용했다. 검토자의 추가 실행0이며 독립 재실행으로 주장하지 않는다.

`tests/coordination`에서 `graphify update .`1회 exit0(AST only,108 nodes/162 edges/8 communities). source_file은 기존 CX-04/CX-05와 신규 checker/test의 정확4개다. 하위 ignored `graphify-out/`만 갱신했고 전체 저장소·의미 추출은 하지 않았다. community 변경에 따른 hub 자동 이름은 적용됐고 LLM label 갱신은 미실행이다.

COORD-04 선별 커밋으로 인계한다. v3 수리본의 출력/수용 계약이 달라지면 고정 v2 검사기를 그대로 승격하지 않고 소유자 인계·독립 대조 후 별도 갱신한다.
