# Claude 재개 인수인계 — D92 이후

2026-09-29 작성. 새 세션의 시작점이다. [이전 재개 문서(D91)](CLAUDE-resume-2026-09-28.md)의 §4 첫 작업 묶음은 D92에서 **구현·등록안 작성까지** 끝났으므로 다시 하지 않는다. 과학 규약을 바꾸거나 실행을 승인하지 않는다.

## 1. 현재 위치

- 저장소 `/Users/song/Projects/pitcheezy`, 브랜치 `codex/ml-matrix-execution`, PR [#34](https://github.com/SongRoute/pitcheezy/pull/34)(merge 금지·새 PR 중복 금지). reset하지 말고 현재 HEAD에서 이어간다.
- D92 커밋은 [기계 기록](../../results/ML-policy-materialization-prep-v1.json)의 `commits`에 있다. 재개 시 `git status --short`, `git log -5 --oneline`, ListAgents, `docs/decisions.md` 꼬리를 확인한다.
- D92는 Codex 감독기 없이 Claude가 직접 재개한 세션이다. SSD 작업대장은 `coordination/20260929-policy-materialization/board.json`.

읽기 순서: [AGENTS](../../AGENTS.md)·[CLAUDE](../../CLAUDE.md)·[협업 규칙](../AI_COLLABORATION.md) → 이 문서 → [D92 보고서](../reports/ML-policy-materialization-prep-2026-09-29.md) → [계약](../contracts/ML-POLICY-MATERIALIZATION-v1.md)·[config](../../configs/ML-POLICY-MATERIALIZATION-v1.json) → [검토 패킷](../reviews/COOP-017-policy-materialization-review-packet-2026-09-29.md). D91 이전 맥락은 이전 재개 문서를 본다.

## 2. 이미 끝난 것 — 다시 하지 말 것

| 경로 | 역할 |
|---|---|
| [policy_identity.py](../../experiments/pitchmdp/pitchmdp/policy_identity.py) | `bind_components` → `BoundComponents`: 모든 후보 부품을 pin된 파일에서 조립·대조, 완전한 식별자(ML-POLICY-IDENTITY-v1), `verify`, 연결 probe |
| [policy_runtime.py](../../experiments/pitchmdp/pitchmdp/policy_runtime.py) | 후보는 `components=`로만. 요청별 지원 표 = pool 확인. `verify_components()`는 stage 봉인 전 필수 |
| [test_policy_identity.py](../../experiments/pitchmdp/tests/test_policy_identity.py) | 합성 검사. 정책 경로 = 실제 평가 경로(`predict_streamed` 등) 재현 |

유지할 의미: 로깅 법칙 = 전체 어휘 BC(재정규화 금지). 후보 모드에서 지원 표 누락·빈 마스크인데 pool이 지원하는 요청과 맥락 행 불일치 요청은 FAILED_INTEGRITY다(D92 상태 변경). 맥락 행 SHA는 요청 fingerprint에 들어간다. G0 = member별 보정 → 5확률 평균 → 공통 앙상블 혼합. 원장 분모는 제출된 요청/PA다. 이력 검사는 국소 일관성이다. 식별자는 "같은 부품"만 보장하고 인과 식별은 아니다.

## 3. 다음 작업 — 순서

| 순서 | 작업 | 완료 기준 |
|---|---|---|
| 1 | **독립 검토.** Sol 재현, Astra 핵심 검토 1회(패킷 질문 10개). Codex 협업 환경에서만 가능 | 결과 파일과 실제 응답 모델 기록. 차단사항은 좁은 수정·재검토 |
| 2 | **Song 결정** D-1~D-11(계약 §6) | decisions에 기록. 추정량/분모 변경(D-1·D-2·D-3·D-5·D-6)은 계약 반영 |
| 3 | **실행기·요청 생성기 구현**(실데이터 없음): `experiments/pitchmdp/scripts/run_policy_validation.py` subcommand `census`/`materialize-bc`/`bind-probe`/`profile`/`v5-denominators`. 요청 생성기는 R2·R3(c) 새 상태·R4b 스냅샷·R4c 불변성 검사·R8 | 합성 검사, 결정 뒤 필드를 바꿔도 요청·확률 불변, 무투구 행이 FAILED_INTEGRITY로 멈추지 않음 |
| 4 | **ML-POLICY-VAL-v1 등록**(source C, config D, 환경 = G0 번들 환경, 출력 `…/ML-POLICY-VAL-v1/`) → 사용자 실행 승인 | 등록본 해시, 단일 큐·`.heavy.lock`, stage 상한 |
| 5 | **S0→S4 실행**: census → BC-P/BC-E·지원 표 → 실제 bind·probe(64행, atol 1e-6) → profile → V5 분모 | stage별 manifest, `verify_components` 통과, 실패 보존 |
| 6 | 추정기(R5·R6), V2 세계(D-10), G0 τ 등록(D-9) | 별도 등록 |

## 4. 실행 경계와 협업

- `policy_frozen=false`, `execution.enabled=false`. 2026 추가 열람·수집·fit·추론·OPE, 정책/서비스 승격, 봉인 파일 덮어쓰기, PR merge 금지. 실제 실행은 자료핀·비용·독립 검토·사용자 승인 뒤에만.
- Opus 메인·Sol 서브·Astra 필수 최소, Fable 제외, Claude→Codex 재호출 금지. 호출할 수 없으면 사용한 척하지 말고 패킷과 미완료 상태를 남긴다.
- 코드 변경 후 `graphify update .`(AST만). Ponytail 기본 모드와 연구 예외를 따른다.
