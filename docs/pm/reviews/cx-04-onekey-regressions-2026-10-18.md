# CX-04 — 한 키 래퍼 합성 계약 회귀

- 상태: 정적 실행 경계·합성 독립 재실행/결과 검토 **PASS**, root 최종 수용 완료. 회귀 산출물 수용이며 제품 계약 4건은 red·미수리다. 작성자 Codex `coordination_docs`, 검토자 `coordination_review`.
- CLAIM: https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248583813 . base `d10d63484d381fc9a410ef749d7a95723be5992a`, 전용 branch `codex/ws-onekey-regressions-1018`.
- 대상은 Claude 소유 원본 `L-onekey/onekey_wrapper.py`, SHA-256 `479162d6f19fd0c624a0feccd8a6bf1a235e0f5aab8082f387dc5c22c454bb88`. 원본/등록 변경 없음.
- 근거: [CX-01 감사](cx-01-onekey-audit-2026-10-18.md), `REGISTRATION-v2.json` SHA-256 `8e0b9fd1febd8e9ff9daca43822e37a2e073b39a8f5990c13cd01ad9c39635ca`.

## 범위와 실행 경계

`tests/coordination/test_onekey_acceptance.py`는 stdlib unittest 직접 CLI 검사다. 일반 import/discovery에서는 모든 검사가 skip되고 외부 소스도 읽지 않는다. 명시 경로·검토한 SHA를 모두 넣어야 실행하며 활성 실행에서는 xfail/skip으로 결함을 숨기지 않는다.

원본 바이트의 SHA를 먼저 확인하고 `exec(compile(...))`한다. import 전 socket/DNS/subprocess/OS shell 실행을 차단하고 원환경을 읽지 않도록 빈 환경 mapping으로 교체한다. 금지 경계의 예외는 래퍼의 `except Exception`을 통과하지 못하며 검사 실패로 드러난다. getpass/TTY·시계·sleep은 가상이다. 원장·plan/grant·영수증·출력은 매 검사 TemporaryDirectory에만 쓰고 제거한다.

실제 `cmd_run`, `check_window`, `read_secrets`, `probe`, ledger/log/secret scan을 호출한다. `check_client`/`build_plan_grant`는 Git·등록·자산 접근을 막는 고정 fixture이고, HEAD와 Popen은 경계 대역이다. Popen은 합성 성공 영수증 17행을 만들 뿐 실제 runner를 실행하지 않는다. 정상 결과를 강제하는 판정 함수 mock은 없다. `validate_pre`/CLI `main`·실제 TLS·HTTP·runner 검증·운영 승인을 시험한 것이 아니다.

기존 `L-onekey/work/test_onekey_wrapper.py`의 합성 영수증 구조를 참고했다. 그 파일의 pytest/node/openssl/loopback harness는 실행하거나 의존하지 않는다. private/PRECONDITIONS/기존 USED-KEYS/사용자 로그·실자료는 읽지 않는다.

## 검사와 기대

| 검사 | 실제 판정 근거(위 SHA 원본) | 계약상 기대 | 현재 실행 결과 |
|---|---|---|---|
| 정상 완료 | `cmd_run:392–516` | scope403·expiry403·revoked401, 본시험18, 최종 PASS/rc0 | PASS; HEAD3/Popen1 |
| 일반 JSON403 | `probe:339–353` | scope 증거 통과 | PASS |
| cf-mitigated HTML403 | `probe:339–353` | scope 증거 거부 | PASS |
| F2 회수 SKIP | `cmd_run:495–516` | 필수 증거가 없으면 최종 rc/result가 미완을 드러냄 | FAIL; probes=[out_of_scope], 최종 PASS/rc0 |
| F2 만료 연기 | `cmd_run:468–476` | 활성 만료 검사를 >900초로 미루면 최종 rc/result가 미완을 드러냄 | FAIL; probes=[out_of_scope,revoked], 최종 PASS/rc0 |
| F3 입력 지연 | `check_window:263–272`; `cmd_run` 창411/입력415/첫HEAD424 | 실제 read_secrets의 가상 getpass가 창·만료 뒤로 이동하면 HEAD/Popen 모두0 | FAIL; 실제 경계 대역 호출 (1,1) |
| F4 표시 없는 HTML403 | `probe:339–353` | cf-mitigated가 없어도 HTML403은 scope 증거가 아님 | FAIL; probe pass=True |

F2는 **등록 필수 증거 충족을 요구하는 contract regression이며 실호스트 버그 발견이 아니다**. 기존 PASS가 main 시험만 뜻한다면 전체 완료로 승격할 수 없다. 현재 계약 검사는 새 필드명을 강요하지 않고 `rc==0 && result==PASS_SYNTHETIC_HTTPS`인 미완 결과를 거부한다. 소유자가 main_pass와 별도 completion 계약을 채택하면 합의 후 이 assert도 갱신한다. 기존 REVIEW-002 N15/N16 지적과 연결되며 새로운 운영 사고 증거가 아니다.

F2는 안전한 실행 전 조기 거부도 허용하며 특정 probe 실행 순서·횟수를 수리 조건으로 강요하지 않는다. 내부 오류/비밀 값 출력 실패를 계약 수리로 오인하지 않도록 제외하고, 정상 완료 control로 무조건 거부하는 가짜 green을 막는다.

F3는 처음 검사 시각은 유효하되 토큰·쿠키 입력에서 각각 1,801초를 가상 경과시킨다. 실제 창 검사를 mock하지 않는다. F4는 actual probe + fake head의 응답 분류 검사이며 TLS/edge 동작 재현이 아니다. 성공 영수증은 wrapper 수용 경계만 분리하는 fixture이고 실제 runner 성공 증거가 아니다.

## 검토 후 직접 실행

```sh
python3 -I -B tests/coordination/test_onekey_acceptance.py \
  --source /Users/song/Projects/pitcheezy-pm-jobs/20261017-ws-launch-prep/L-onekey/onekey_wrapper.py \
  --sha256 479162d6f19fd0c624a0feccd8a6bf1a235e0f5aab8082f387dc5c22c454bb88 -v
```

기본 수집 확인도 이 파일만 대상으로 한다: `python3 -B -m unittest discover -s tests/coordination -p test_onekey_acceptance.py -v`. 프로젝트 전체 pytest는 실행하지 않는다. 원본 수정 후에는 소유자 검토를 받은 새 파일 경로/SHA를 명시하며, 불일치를 자동 승인하거나 현재 소스를 복사·수정해 green으로 만들지 않는다.

2026-10-18 작성자 실행: 위 전용 worktree에서 명시 CLI 1회, exit **1**, `Ran 7 tests`, **정상3 PASS / 계약4 FAIL / error0 / skip0**. 네 실패 모두 위 계약 assertion이며 금지 IO guard 발동·내부 예외는 없었다. 기본 수집 1회는 exit **0**, `Ran 7 tests`, `OK (skipped=7)`. 원본 SHA는 실행 뒤에도 위 고정값과 일치했다.

실행한 테스트 소스 SHA-256: `f687bc9f4f4efd86a43cdbfc08f63a42cef00e2b782cf75f0fc1e94095f1d855`. 실행 영수증은 이 문서의 명령·SHA·exit·결과표이며 가상 출력 디렉터리는 제거되었다. 실호스트 요청·runner 실행·운영 데이터 검증은 0회다.

독립 검토자 `coordination_review`도 같은 명시 CLI 1회(exit1, 정상3 PASS/계약4 FAIL/error0/skip0/guard실패0)와 단일 파일 discovery 1회(exit0,7skip)를 실행했다. F2 probe 목록·F3(1,1)·F4 True가 작성자 결과와 같고 실행 뒤 test/원본 SHA가 불변임을 확인했다. 동일 모델 별도 에이전트의 결과 검토 **PASS**이며 회귀 인계 품질 판정이다. 제품 수리 PASS·실호스트 검증·운영 GO가 아니다.

graphify([COORD-03 확인](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6248632730)): 신규 worktree 루트에 기존 그래프가 없어 전체 저장소 bootstrap은 미실행했다. root의 후속 범위 승인으로 조정자가 `tests/coordination`에서 `graphify update .` 1회 실행해 exit0(48 nodes/62 edges/4 communities)을 확인했다. source_file은 신규 test 1개뿐이며 생성 그래프는 해당 하위 `graphify-out/`의 ignored·비커밋 산출물이다. 정본 그래프 복사·링크 없음, test SHA 불변.

## 인계

실행 전 경계 PASS → 작성자 합성 실행·기록 → 독립 재실행·결과 검토 PASS → root 수용까지 완료했다. 동일 7검사의 별도 재실행이며 14개 검사로 합산하지 않는다. COORD-03 선별 커밋으로 인계한다. 계약 회귀 4건의 red는 그대로 보존했으며 수리·운영 GO는 미완이다. 기존 CL-09 소유자에게 회귀를 인계하며 신규 원본 수정 점유를 만들지 않는다.
