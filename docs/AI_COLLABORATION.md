# Codex–Claude Code 협업

2026-09-27. 사용자는 Codex가 Claude CLI를 호출하는 협업과 `claude --dangerously-skip-permissions` 사용을 명시적으로 허용했다. D68. Codex가 작업 배정·검토·통합·진행 보고를 맡고, Claude Code는 지정된 작업을 구현하거나 독립 검토한다.

## 작업과 상태

- 공용 작업대장: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/coordination/20260927-claude/board.json`. 총괄만 갱신한다. worktree의 파일은 자동 동기화되지 않으므로 변경 전달은 고정 커밋 또는 명시된 읽기 전용 경로로 한다.
- 각 호출은 고유 attempt 디렉터리에 `request.md`, `events.jsonl`, `stderr.log`, `state.json`, 반환 시 `result.json`을 남긴다. 요청 해시·기준/최종 커밋·세션 ID·시간·종료 상태를 보관하며 기존 attempt는 덮어쓰지 않는다.
- `scripts/run_claude_task.py`가 CLI를 감독한다. `returned_for_review`는 모델이 결과를 반환했다는 뜻이다. 완료 판정은 실제 diff·필요한 검사·산출물을 검토한 총괄이 작업대장에 별도로 기록한다. CLI 오류·시간 초과·권한 거절·결과 부재는 완료로 세지 않는다.
- 세션 ID로 후속 호출을 이어가되 새 attempt를 만든다. 중단 시 기존 프로세스와 산출물을 먼저 확인하고 중복 실행하지 않는다. 무인 실행은 tmux에서 감독기를 실행해 대화 연결과 분리한다.
- 이 Mac의 사용자 훅은 iTerm `cc-status`, 프로젝트 훅은 편집 후 pytest다. iTerm 훅이 headless 시작을 막는 것이 실측되어 감독기의 `--disable-hooks`로 해당 호출에만 `disableAllHooks`를 적용한다. 영구 설정은 바꾸지 않는다. 필수 검사는 명시적 명령으로 실행·기록한다. 관리자 정책 훅을 우회하지 않는다.

## 역할과 실행 범위

1. 작업 지시에 기준 커밋, 수정 가능 파일, 완료 조건, 검사, 실험 실행 허용 범위, 시간/turn 상한을 명시한다.
2. 구현자는 독립 worktree에서 작업한다. 검토자는 다른 에이전트이며 검토한 커밋만 통합한다. 중요한 의견 차이는 코드·규약·측정으로 판정한다.
3. Claude 권한 옵션은 해당 작업의 실행 허용을 뜻하며, 과제 범위·실험 예산·자료 사용 규칙을 바꾸지 않는다. 이 협업에서 공유 main/PR의 merge와 실행 큐 관리는 총괄만 맡는다.
4. 무거운 학습·추론·MPS 자원 측정은 한 실행 큐에서만 수행한다. 모든 worktree가 같은 artifact_root의 기존 `.heavy.lock`을 쓴다. 학습 중 실행 소스를 바꾸지 않는다.
5. 원본 데이터·봉인된 산출물은 보존한다. 새 시도는 새 출력 경로와 config에 등록한다. 원래 F4 7,200초/member·28,800초/family 예산을 늘리지 않는다. 2026 자료는 추가 열람하지 않는다.
6. 현재 사용자 지정 체제는 **GPT-6 Astra → Claude Opus 5.5 → GPT-6 Sol**이다(D77). Fable은 새 작업에서 제외한다. Astra는 과학적 설계·복잡한 구현 판단·핵심 독립 검토, Opus는 범위를 정한 구현 및 다른 작성자의 교차 검토, Sol은 반복 검사·실행·기록을 맡는다. 같은 산출물의 작성자와 최종 검토자는 분리한다. 실제 호출은 `gpt-6-astra`, `claude-opus-5-5`, `gpt-6-sol`이며 실제 응답 모델을 기록한다. 접근 실패를 조용히 다른 모델의 성공으로 바꾸지 않는다. Claude가 Codex를 다시 호출하는 순환 위임은 하지 않는다. 아래 Fable 참여 기록은 과거 실행 이력으로 보존한다.


## 첫 작업

| ID | 담당 | 산출물 | 완료 기준 |
|---|---|---|---|
| COOP-001 | Claude 구현 → Codex 검토·실행 | F4 실데이터 context cache 감사 러너·회귀 검사·새 실행 등록 | 코드 검토, 실제 동등성/비용 측정 또는 보존된 실패 근거, 채택 여부와 기존 예산 적용 |
| COOP-002 | Codex 설계 → Claude 독립 검토 | G0/F1 5seed 안정성·bridge 연장 및 G0 전체 MLB 평가 설계 | 재사용·보정·다중 비교·비용·기존 DEV 노출 한계를 명시하고 실행 공백 식별 |

## G0/F1 5seed 후속 — D72

사용자가 후속 목록1~4의 실제 수행을 승인했다. 새 작업대장은 SSD의 `ML-MATRIX-20260924/coordination/20260927-confirmation/board.json`이며, 첫 작업대장에서 이 경로를 연결한다. COOP-003은 Fable의 additive 실행기 구현, COOP-004는 Opus·Astra의 독립 검토, COOP-005는 Sol의 단일 실행 큐·보고다. C1과 F1연장 각각7,200초의 비용 대장을 분리하고 총14,400초를 넘기지 않는다. 전체 MLB 단독 평가는 이 묶음의 범위에 포함하지 않는다.

이 문서는 협업 규칙이다. 실험의 실제 상태와 과학적 결론은 각 config·manifest·실행 보고서가 결정한다.

## 전체 June 적격 준비 — D79~D81

Astra가 과학 계약·코드/등록/산출물을 독립 검토하고, COOP-011 Claude Opus5.5가 별도 worktree에서 worker와 합성 검사를 구현했다. 두 CLI attempt는 실제 응답 모델 `claude-opus-5-5`, 성공 반환 후 검토·통합됐으며, 두 번째는 외부 등록 config와 동결 C checkout을 연결하는 수정이다. Sol은 감독기·중단/비용 기록과 단일 실제 실행을 맡았다. Root는 통합315검사·등록·보고를 맡았다. 작업대장은 SSD `coordination/20260927-june-eligibility/board.json`, Claude 원본 로그는 그 아래 `COOP-011/attempt-001`, `attempt-002`다. 신규 Fable 호출은 없다. 실제 준비는104,970구·worker1.219838초이며 새fit/추론은0이다. [보고서](reports/ML-June-eligibility-2026-09-27.md).

## 전체 June 보정 설계·과학 등록 — D82~D83

Astra가 B0/B1/B2·N3/R78·비용 계약을 작성하고, Sol이 재사용 입력/배열 헤더·비용·누락 구현을 조사했다. COOP-012 Opus5.5는 두 번 독립 검토해 출처/해시 연결 보완 후 PASS를 반환했다. Root가 설정·기록을 통합했고, Sol이 최종 등록 해시/구조를 별도 확인했다. 작업대장은 SSD `coordination/20260928-june-calibration/board.json`이다. 실제 응답 모델은 두 호출 모두 `claude-opus-5-5`이며 Fable 호출은 없다. 최초 검토의 ‘이전5추론이한프로세스’ 표현은 별도5worker 비용대장으로 정정했고 원문/정정 이력을 보존했다. 과학 등록 완료와 실행 코드를 구분하며 이번 단계는 새fit/추론0이다. [설계 보고서](reports/ML-June-calibration-design-2026-09-28.md).
