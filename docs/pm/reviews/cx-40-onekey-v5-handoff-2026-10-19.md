# CX40 — one-key v5 채택과 N23 인계 정정

- 작성: Codex implementation_1019, 2026-10-19 KST. 검토: queue_1019 / Root.
- 상태: 독립 정적 검토 PASS 및 Root 수용. 실행 준비 완료나 GO가 아니다.
- 선점: [CLAIM6258428755](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6258428755), base `950101e34e1ec6d556accedbe40e9cbe52301f30`.

## 실행 담당에게 넘길 현재 상태

**CL02b는 CX19의 두 수정을 별도 v5에 채택했고, 새 source·REGISTRATION-v5·REVIEW-005 쌍이 존재하며 delta 검토는 PASS다.** v4 원본은 보존됐다. 과거 [CX19](cx-19-onekey-isolated-candidate-2026-10-19.md)·[CX38](cx-38-onekey-joint-run-plan-2026-10-19.md)의 당시 대기 기록을 덮어쓰지 않고 이 문서가 새 관측을 연결한다. 최종 실행 client pin·운영 체크리스트·공동 합의·해당 창 GO는 별개로 남는다.

| L-onekey 파일 | 이번 실제 바이트의 SHA-256 |
|---|---|
| `onekey_wrapper-v5.py` | `9c35c5834db9f2e810f0eb030f6031f60509e64eaf97d8a97dc89fc78854eb5b` |
| `REGISTRATION-v5.json` | `6d068fee9a23d004df8f6c96848cb0ce288d0b387cf908c1f245506ef05bbd12` |
| `REVIEW-005.json` | `ecf17fa0e192f340bd558415587b24d743ce217411230b51869508fe087dabea` |
| 보존된 `onekey_wrapper.py` (v4) | `53afba570a3ea03766181397c7880ddeb6dabbf1e2a3858b6e10d1ceb30d35af` |

REVIEW-005에 기재된 v5 source·등록 pin과 REGISTRATION-v5의 wrapper pin을 실제 bytes와 대조해 모두 일치했다. 등록은 `execute:false`다. 등록의 상태 문자열에는 검토 대기가 남아 있으나, 이후 REVIEW-005는 delta를 PASS로 수용했다. 이 문서는 등록 파일이나 실행 권한을 변경하지 않는다.

CX19 격리 후보 SHA `05a9cc2066e80bc08a419baf5882d215cfc408dde7f190e77ae7f1981a90db05`와 v5의 실제 diff는 **40행의 REGISTRATION-v4→v5 파일명과 주석 한 줄뿐**이었다. 따라서 NaN를 거부하는 유한 범위 비교와 deferred probe의 target SHA 결합 두 수정이 새 v5에 포함된 것을 정적으로 확인했다. 원본 v4에 적용됐다고 표현하지 않는다.

## N23: 정상 fixture 대기는 해소, main 입력 거부 구현은 별개

1. [CX10 보고서](cx-10-onekey-token-fixture-2026-10-18.md)와 commit `8f7c257f2971baa63f04e7718e8458505d844cde`는 정상 scenario의 짧은 합성 bearer를 **45자, `[A-Za-z0-9_-]{43,128}`에 맞는 합성 값**으로 바꾼 작업이다. [완료 인계6249075545](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6249075545)가 이미 존재한다.
2. 현재 `tests/coordination/test_onekey_acceptance.py:100`도 같은45자 값이다. 파일 SHA `c395c7d42872961361e0923bb0a63f7532927e881e607383d772cf5ce5597d2c`를 확인하고 literal의 길이·정규식 일치를 정적으로 대조했다. 테스트를 import하거나 실행하지 않았다.
3. REVIEW-005 H8은 **그 c395 파일의 d9ddcff7 사본**과 checker `e8dcd6e18acbfde8cc52ff00e27ad3bc7ee2e77d2aa5a16c28e2f50621726e82`를 사용해 v5의14 acceptance methods가 PASS했다고 보고한다. 이는 REVIEW-005의 검토 결과를 인계하는 것이며 이번 CX40의 새14검사나 원로그 재검증이 아니다.

따라서 REGISTRATION-v5의 `v5_delta.N23`/`open_items_v5[2]`, REVIEW-005의 N26에 남은 **“Codex 정상 fixture 변경 대기”라는 선행조건은 기존 완료 근거로 해소**됐다. CL02에는 이 근거를 반환하면 되며 fixture를 다시 바꾸거나 동일 검사를 반복할 필요가 없다.

반면 v5 **`cmd_run`의 main 입력 경로**는 여전히 `token_shape_ok`를 기록한 뒤 형식이 틀려도 경고만 남긴다(514,526–528행). 첫 scope HEAD 전 강제 거부로 바꾸는 구현은 미완이다. **`cmd_probe`는 이미654–655행에서 형식 불일치를 Refuse**하므로 모든 경로에 거부가 없다고 확대하지 않는다. 실제 코드/등록 변경과 그에 맞는 검토는 CL02 소유이며 CX40은 수리·추가 negative test를 하지 않았다.

## delta 수용 뒤 남은 실행 인계

| 남은 항목 | 현재 근거와 담당 행동 |
|---|---|
| 최종 client commit 선택·source pin 재등록·독립 검토 | REVIEW-005 N27은 당시 `ws-launch-a-1017@3eda8d09`의 publisher SHA `958f1f25…`와 등록 `3bbb9ef4…` 차이로 real-mode 검사2개가 거부됐다고 보고한다. 이는 **그 검토 시점의 drift 증거**이며 현재 tip이나 실행할 commit을 선택한 기록이 아니다. 소유자가 최종 client를 정하고 re-pin delta·검토를 완료해야 한다. scratch re-pin의2 PASS도 실제 등록을 갱신한 것은 아니다 |
| 운영 체크리스트 정합 | 현재 `OPERATOR-CHECKLIST.md` SHA `6e07e2d53ff40be8aadb0d9c69f13c02323fbaf7fe19ccdc42359eaf5b8af4a6`에 v4 wrapper/등록 이름이 남은 것을 확인했다. N28에 따라 실제 사용할 버전과 맞추는 소유자 갱신이 필요하다 |
| N23 main 형식 거부 | fixture 선행 대기만 제거 가능. main 입력의 경고→강제 거부 변경은 별도 소유자 구현·등록/검토 대상이다 |
| GATE-06·비공개 preconditions·해당 창 Song GO | [CX38 공동 실행표](cx-38-onekey-joint-run-plan-2026-10-19.md)는 미합의·미발신 초안이다. 정상/긴급 회수·전체 요청/시간 원장·팀원 확인과 해당 창 GO를 대신하지 않는다 |

**CL02 반환 요약:** 정상 fixture 선행은 CX10/8f7c257f 및 현재 c395로 완료돼 있고 REVIEW-005도 그 파일을 사용했다. N23 잔여를 main 입력 hard-refusal 구현으로 정확히 좁힐 수 있다. v5 delta 수용은 확인됐지만 실제 실행 가능한 최종 쌍은 client re-pin 후 검토·체크리스트 정합과 현행 공동 실행 절차를 기다린다.

이번 변경은 이 인계 문서1개다. 기존 보고서·v4/v5 source·등록·체크리스트·테스트는 수정하지 않았고, wrapper import/실행·테스트 재실행·원로그/자격 열람·실호스트·외부 회신 발신은0이다. 공동큐의 인계 기록은 별도이며 작성자는 stage/commit하지 않는다.
