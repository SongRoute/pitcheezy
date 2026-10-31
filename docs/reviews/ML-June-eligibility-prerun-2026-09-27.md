# June 적격 준비 실행 전 구현 검토 — Astra

2026-09-27. **구현 검토 PASS, 미해결 구현 차단 사항 0건.** 실제 자료 실행이나 최종 등록 감사의 통과를 뜻하지 않는다. [과학 계약](../contracts/ML-JUNE-ELIGIBILITY-v1.md)에 따라 코드 C·등록 D·실제 config/plan/source/data pin을 고정한 뒤 별도 등록 감사를 마쳐야 새 June 적격 필드에 접근할 수 있다. 이 검토에서는 실제 June 결과 필드·새 준비 산출물을 읽지 않았고 모델·준비 worker를 실행하지 않았다.

## 검토 대상과 바이트 identity

Opus worker 구현 커밋 `3833fd87dcf3431d8042c81b2c26fd8b12d75c77` 및 provenance 수정 커밋 `a3a58f955d81db1cccbe3bd1b9fb99ef2bbbb737`를 검토했다. 아래 해시는 최종 과학 실행 코드 C가 만들어지기 전의 검토 바이트이며, 총괄이 통합 후 동일성을 확인한다.

| 파일/근거 | SHA256 |
|---|---|
| `scripts/prepare_june_eligibility.py` | `98a144908fb1528993dea71427cfc8c78b35ee1a6ef72f884f4c2f68bd171249` |
| `tests/test_june_eligibility.py` | `30020f8982650cf686a4400fdc731875704a1f6bf6c312d40f8f4c8022e738ee` |
| `configs/ML-JUNE-ELIGIBILITY-DRAFT-v1.json` | `7ae0f7c1c438376d997db99b47c311b444e9a668dfd8bb5a9fa30260c3a6dd97` |
| `scripts/supervise_june_eligibility.py` 및 COORD 배포본 | `929ce9f8cf744c03dc0fdd1b16eb9ac10cc97d3a8469db7d4a72625f0e039e81` |
| 이 검토 때의 계약 | `97540749699096d2f6087928515a5d40924fdea517a056e29b31243fbc7396de` |
| archived P4 `source/pitchmdp/model.py` | `0e963b96900adb09705a56afb79e8dedcf31973d253df393ffe205df39b8a150` |

Worker 검토 경로는 `/Users/song/Projects/pitcheezy-worktrees/june-eligibility`다. 감독기 COORD 경로는 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-june-eligibility/supervise_june_eligibility.py`이며 저장소 스냅샷과 SHA256이 정확히 같다. 등록은 실제 실행 경로의 바이트를 pin해야 한다. draft config는 등록 완료 config가 아니다.

## Worker 검토

기존 P4 소스에서 `OUTCOMES`, `outcome_labels`, `eligible`만 AST 추출하며 torch나 모델 모듈을 초기화하지 않는다. 불일치하거나 예상 밖 참조가 있는 소스는 중단한다. 기존 `.notna()`를 finite 검사로 바꾸지 않고 stored `supported_pa`를 재사용한다. 가공 `pitch_outcome`으로 매핑을 대체하지 않는다.

새 허용 필드 8개와 기존 metadata만 projection하고 June 2025/R를 pushdown한다. legacy `calibration`은 날짜에 따른 `blend`로 구분한다. 모든 input/guard hash를 확인한 뒤 디코드하고 쓰기 후 다시 확인한다. 부모·G0 frozen bundle을 변경하지 않는다.

Ordered Cpanel 4,821키와 metadata, 패널 요청에서 제외된 391키의 정확한 차집합을 검증한다. 요청/적격/제외 및 두 종류의 요청 차집합, 정확한 적격 차집합의 union/disjointness를 검사한다. 일곱 제외 bool의 OR와 권위 eligible의 여집합이 같아야 한다. 겹치는 제외 원인과 disjoint bitmask를 구분하며 결과 클래스 빈도는 만들지 않는다.

그룹은 동결 TRAIN mapping과 cutoff를 쓰고 지원은 500구 AND 30경기다. 현재 draft의 missing identity/role/hand 정책은 fail이다. Persisted parquet는 키·metadata·eligible·제외 bool만 포함하며 원래 outcome/type/count/coordinate 값과 label은 제거한다. Manifest는 모든 필수 출력과 게이트 완료 후 마지막에 쓴다.

외부 등록 config 수정에서는 실행 checkout HEAD가 코드 C와 **정확히 일치**하고 tracked source가 clean이어야 한다. Config는 별도 등록 checkout에 있어도 되지만 실제 전달 경로가 tracked/clean이고 현재 HEAD blob 바이트와 같아야 한다. 실행/등록 checkout과 commit, 실제 config 경로와 해시를 결과·manifest에 남긴다. 잘못된 C, 짧은 commit, script hash 불일치, dirty/staged/untracked config를 거절하는 합성 git-layout 검사가 추가됐다.

## 감독기 수정 확인

초기 검토의 다음 문제는 수정됐다.

- Worker 종료 직후 Popen→wait 시간을 고정한다. 이후 manifest 검사 실패가 발생해도 사후 감사 시간이 worker 비용에 섞이지 않는다. 별도 `postflight_seconds`를 기록한다.
- timeout/신호 종료 중 SIGINT/SIGTERM을 무시하는 cleanup 구간에서 child process group을 종료·회수한다. 반복 신호가 회수를 끊지 않게 했다.
- 등록과 계획의 코드·config·source·명령·출력·잠금·600초 상한을 내용으로 교차 검증한다. 완료 manifest에는 config SHA, registered code C, execution HEAD=C 및 complete 상태 검사가 필수다.
- 실행 checkout의 HEAD와 tracked-clean 상태를 검증한다. 시작/종료/refusal 기록이 존재하면 자동 재실행하지 않는다. 로그·실패 출력과 비용을 보존한다.

감독기 preflight는 등록·코드·config metadata/hash만 읽으며 새 June 결과 필드를 디코드하지 않는다. Worker가 공용 heavy lock을 소유하고 supervisor는 queue lock을 소유한다. 출력 완료 검사는 worker wall 이후의 별도 감사 경계다.

## 검사 근거와 한계

총괄은 worker 집중 검사 **16개**, 감독기 합성 검사 **5개**의 통과를 보고했다. Opus는 구현 worktree의 전체 검사 **310개** 통과를 보고했다. Astra는 테스트 소스와 수정 diff를 검토했으며 해당 검사들을 중복 실행하지 않았다. Worker 검사는 고정 outcome 매핑과 독립 표의 동등성, notna/finite 차이, 제외 OR/중복, 경계 500구·30경기, 순서 오류·metadata 오류·pin 오류의 중단, 금지 열 저장 방지, 잠금·git-layout 실패를 포함한다. 감독기 검사는 사후 감사 실패의 비용 분리, 등록 불일치와 completion identity, timeout cleanup 및 중복 시작 거절을 포함한다.

Worker validator가 June 2025/30경기/500구/600초 전체를 상수로 강제하는 것은 아니다. 기간은 등록된 비2026 season, 지원 문턱은 양수 AND 조건으로 검사한다. 정확한 June/30/500 값은 최종 config의 byte pin과 등록 감사로 고정하고, 600초는 supervisor가 상수로 강제한다.

남은 **등록 감사**는 별도다. 실제 config의 June 기간·30/500·600초·허용 열·부모/자료/계약 해시, 통합 소스 C와 배포본, config가 커밋된 D, 실제 argv·환경·fresh output·manifest 기대값을 확인해야 한다. 현재 검토는 실자료 적격 수나 실제 준비 비용을 측정하지 않았고, 미개봉/독립 확인·보정 효과·후속 I2 활성화를 보장하지 않는다.
