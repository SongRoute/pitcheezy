# Codex–Claude CLI 협업 실행 기록

2026-09-27. 진행 중. 규칙: [AI_COLLABORATION.md](../AI_COLLABORATION.md), D68/D69.

## 실제 연결과 복구

- Codex가 `scripts/run_claude_task.py`를 통해 Claude CLI를 호출했다. 요청·이벤트·오류·종료 상태는 SSD의 `ML-MATRIX-20260924/coordination/20260927-claude/`에 보존한다.
- 첫 두 호출은 모델 작업 시작 전 iTerm `cc-status` SessionStart 훅이 GUI 응답을 기다려 중단했다. 해당 호출 프로세스만 종료하고 실패 시도와 진단을 보존했다. 영구 Claude/iTerm 설정은 바꾸지 않았다. 후속 호출은 호출 범위의 `disableAllHooks`와 명시적 검사를 사용한다.
- Fable 5.1은 CLI 2.1.278에서 응답 모델 `claude-fable-5-1`을 확인했다.
- Opus 5.5 첫 호출은 CLI >=2.1.280 요구로 API 400을 반환했다. 감독기는 실패로 기록했다. Claude CLI를 공식 `claude update`로 **2.1.283**으로 갱신하고 새 attempt에서 `claude-opus-5-5` 응답을 확인했다. 다른 모델로 대체하지 않았다.
- 일반 감독기의 모델 ID 검증도 추가했다. 모델 오류·결과 부재·권한 거절·모델 불일치·시간 초과를 성공으로 세지 않으며, 정상 반환도 `returned_for_review`로 남긴다.

## 현재 작업

| 작업 | 실행/검토 모델 | 상태와 근거 |
|---|---|---|
| COOP-001 F4 캐시 감사 러너 | Fable 5.1 구현, Astra 과학적 검토, Sol 감독기 보완·실행, Codex 통합 | Fable attempt-002/003에서 `3a4d7c6`·`7ca6062` 반환, 합성 검사 71개 통과. [Astra 사전 검토](../reviews/COOP-001-F4-cache-prereg-review.md), [코드 검토](../reviews/COOP-001-Fable-code-review.md) 뒤 최종 감독기 보완·등록 중. 실데이터 측정 전 |
| COOP-002 G0/F1 5seed 연장 | Sol 초안·수정, Opus 5.5 독립 검토·재검토 | **설계 검토 완료**. [수정 초안](../contracts/ML-G0-F1-CONFIRMATION-DRAFT-v1.md), [원검토/재검토](../reviews/COOP-002-Opus-5.5-design-review.md). 새 G0 부모 해시, 비용 대장, 10개 예측 완료 후 채점, 두 번째 DEV 열람의 한계를 보완했다. 실행 등록·학습은 아직 없음 |

작업대장 절대 경로: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-claude/board.json`. `COOP-001/attempt-001`, `COOP-002/attempt-001`은 훅 중단, `COOP-002/attempt-002`는 CLI 버전 실패다. 재시도는 별도 디렉터리다.

## 검증과 해석

- 최초 전체 프로젝트 검사: `.venv/bin/python -m pytest tests -q` → **251 passed, 18.96초**.
- 이후 정확한 응답 모델 검사를 추가한 감독기 집중 검사: **6 passed, 3.03초**. 실제 LLM 성공/실패 이벤트와 별도로 가짜 CLI를 사용해 오류 분류·출력 덮어쓰기 거절을 검사한다.
- 실험 전체 예산·시드·자료·400draw는 바꾸지 않았다. F4 감사는 MPS 실측 전이며, 5seed 초안은 이미 노출된 DEV의 안정성 확인을 독립 확인과 구분한다.
- Astra의 사전 검토는 학습/예측 프로세스별 캐시 생성 비용, 정확한 입력 검사 비용, 이전 실패/감사 비용과 원래 전체 예산을 모두 포함하도록 요구했다. 수치 허용치는 실측 전에 새 config에 동결한다.
- 실제 왕복 협업: Sol 초안 → Opus의 필수 수정2건 및 권고5건 → Sol 수정 → 같은 Opus 세션 재검토에서 `RESOLVED`, 추가 명확화3건 반영. 세션 ID를 지정한 attempt-004 재호출이 동작했으며 응답 모델 검증도 통과했다.
- Fable의 첫 러너 구현을 그대로 실행하지 않고 Astra가 비용 합산, 선행 수치 검사, 보정 목적함수, 양쪽 표본의 입력 순서 검사를 검토했다. Fable은 후속 호출에서 이를 수정했고 CPU 합성·인접 검사 **71 passed, 5.93초**를 기록했다. 추가 검토에서 찾은 선행 manifest 연결 문제는 수정됐고, 감독기 중단 정리와 검증된 AppleDouble 제외는 Sol이 보완한다.
- 난도별 배정은 사용자 지정 순서 **GPT-6 Astra → Claude Fable 5.1 → Claude Opus 5.5 → GPT-6 Sol**이다. 과학적 판단·핵심 검토, 복잡한 학습 경로 구현, 독립 교차 검토, 정해진 수정·실행·기록에 각각 적용했다.

## F4 실측 등록 — 결과 열람 전

- 실행 소스는 별도 worktree `pitcheezy-worktrees/f4-cache-execution`의 `d022711368057a75badaba46c50f728e718c3e52`에 고정한다. 해당 worktree는 실행 중 갱신하지 않는다. 최종 Astra 검토에서 제한된 감사 실행의 차단 사항 없음, root 독립 검사 **79 passed, 6.56초**.
- 완성 config는 주 저장소의 `configs/EXP-P4-002-v3-cache-audit-v2.yaml`이며 SHA256 `3b3bd37d0e7459db323c52f8cedd8db454e06ba6e404cfaabb413c1cdb0d4230`. 30개 소스와 두 계약, 부모 manifest, 표본, 이전 비용 대장을 핀한다. 실행 코드 커밋 C와 등록 config 커밋 D를 분리해 커밋 자기 참조를 피한다.
- fresh 출력은 SSD의 `ML-MATRIX-20260924/EXP-P4-002-v3-cache-audit-v2`, 실행 큐는 `coordination/20260927-claude/execution/f4-cache-audit-queue-v2`. `run_f4_cache_audit_queue.py` SHA256 `489cb62485d8a5480011743064beb3e6febc486feda0b1e4c0f58eaf3bf57de0`.
- prepare → 각 H0/32/128의 stage1·stage2·original/cached stage3·compare → summary, 총 **17명령**. 상한 합계 13,800초, 검증된 과거 16명령 244.084859초. 각 명령 전 실제 누적 비용과 예약 상한을 검사하고 첫 실패에 큐를 종료한다. stage3 각 600초에 종료 처리를 포함한다. 실패 후 자동 재시도나 허용 오차 완화는 없다.
- 내부 감독기 대장과 별도로 외부 큐가 Python 시작부터 전체 프로세스 wall을 기록한다. 둘을 더하지 않는다. 실제 지출은 더 넓은 범위를 측정하는 외부 큐로 보고하고, 봉인된 summary가 자기 실행을 상한으로 예약한 수치와 구분한다. 외부 큐의 성공·실패·프로세스 그룹 timeout·SIGINT를 CPU 모의 실행으로 확인했다.
- 전체 캐시 생성 비용은 이번 표본 감사에서 외삽하지 않는다. 채택·full fit는 별도 조건을 충족하기 전까지 미실행이다.
