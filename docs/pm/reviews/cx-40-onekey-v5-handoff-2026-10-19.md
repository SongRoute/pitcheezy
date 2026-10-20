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

## CX38 회신 마지막 문단 교체안 — 미발신

우리 쪽 두 수정은 별도 v5 실행기에 채택됐고 독립 delta 검토를 통과했으며, 기존 v4는 보존했습니다. 최종 실행 client pin 확정, 체크리스트 정합, main 입력의 토큰 형식 강제 거부와 양측 실행표 확정은 남아 있습니다. 실제 공동 시험은 아직 실시하지 않았습니다.

이번 변경은 이 인계 문서1개다. 기존 보고서·v4/v5 source·등록·체크리스트·테스트는 수정하지 않았고, wrapper import/실행·테스트 재실행·원로그/자격 열람·실호스트·외부 회신 발신은0이다. 공동큐의 인계 기록은 별도이며 작성자는 stage/commit하지 않는다.

## CX41 추가 — 2026-10-20 KST, DOC01 체크리스트 delta

- 작성: Codex `checklist_delta`; base `2330f17b20b4a981848f8d77e03eec487cb9b289`. [CLAIM6285080136](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6285080136). 독립 검토 결과는 고정 커밋을 가리키는 공동큐 REVIEW에 기록한다.
- [DOC01 완료6259701071](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6259701071) 이후의 명시된 비밀 없는 파일만 정적 대조했다. 기존 합성 검사·원로그 검증은 반복하지 않았다.

**N28의 v4 파일명 대기는 해소됐다. 운영 체크리스트 전체 정합이나 실행 준비 완료를 뜻하지 않는다.** 현재 `OPERATOR-CHECKLIST.md` SHA-256은 `e4cb579692dd1b5c6d8e947661f804e231fe07a1a04250e1674a9b3b9e887e02`다. 기준 등록과 check·dry-run·run·deferred probe 명령은 v5를 가리키며, v4 wrapper 이름은13행의 이력 설명에만 남는다. 이 파일명은 실제 v5의 `REGISTRATION` 상수(40행) 및 등록의 `invocation`과 일치한다.

| 새 관측 / 재사용 근거 | 범위와 남은 행동 |
|---|---|
| v5 source·REGISTRATION-v5·REVIEW-005의 현재 SHA는 위 CX40 표와 각각 동일 | 기존 delta PASS를 재사용한다. 등록은 여전히 `execute:false`; N27 최종 client 선택·re-pin·독립 검토의 완료 근거가 새로 생긴 것은 아니다 |
| 체크리스트14행은 여전히 “N23은 CX-04 시험 자료 변경 대기” | CX10/`8f7c257f`·c395 fixture·REVIEW-005 H8의 기존 근거로 해소된 선행조건이다. 소유자가 이 문구와 후속 등록 설명을 바로잡으면 되며 fixture 재수정·동일14검사는 필요 없다 |
| [#38 발송 회신6267615609](https://github.com/SongRoute/pitcheezy/issues/38#issuecomment-6267615609)도 fixture 대기 때문에 경고를 남길 수 있다고 설명 | 발송 사실과 설명의 정확성을 구분한다. **현재 `cmd_run`은 형식 불일치여도 경고 뒤 키를 소진하고 첫 scope HEAD로 진행할 수 있다**(514,526–531행). “안전한 쪽으로 실패”가 첫 요청 전 거부를 뜻하지 않는다. `cmd_probe`는654–655행에서 이미 형식 강제 거부한다 |
| [팀원 WS-partner3 완료6270417682](https://github.com/Pitcheezy/transition-models/issues/3#issuecomment-6270417682) | 상대는 v5·14검사를 우리 보고로 구분하고 최종 파일·SHA·client commit은 미수신, N23 설명 차이는 확인 사항으로 보존했다. 상대의 직접 실행본 검증이나 N23 변경 수용으로 해석하지 않는다 |

CL02의 최종 실행본 후속 인계에는 **정상 fixture 대기 제거 → `cmd_run` 첫 요청 전 형식 거부의 구현·변경 범위 검토 → 최종 client pin에 묶인 등록·검토 → 체크리스트 설명 정합**을 연결하면 된다. 이미 거부하는 `cmd_probe`와 보존된 v4를 중복 수정할 요청이 아니다. CL02 소유 파일은 이번에 수정하지 않았고 최종 실행본 버전도 대신 선택하지 않았다.

[Song 결정 인용 계획6284951661](https://github.com/SongRoute/pitcheezy/issues/37#issuecomment-6284951661)에는 공동 시험 **10/22 18:00~19:30 KST, 담당 Song·팀원**이 이미 있다. 이 일정·담당 계획과 GATE-06 전체 요청/시간 원장·정상/긴급 회수 순서·역할 세부의 양측 합의, 비공개 자격 전달·해당 창 Song GO 완료 근거는 별개다. [CX38](cx-38-onekey-joint-run-plan-2026-10-19.md)의 본창18·runner120초 미만과 외부/관리 요청을 구분하며,21 등 부분합을 공동 전체 승인 예산으로 쓰지 않는다.

이번 추가는 기존 보고서만 갱신했다. graphify는 실제 vocab `wrapper token acceptance registration`으로 먼저 query했으나 이 L-onekey 문서 delta 근거를 제공하지 않아 명시 원문을 사용했다. 그래프는 검증 증거로 쓰지 않았으며 갱신하지 않았다. wrapper import/실행·테스트·빌드·원로그/자격 열람·실호스트 요청·제품/원본 수정·외부 회신 발신은0이다.
