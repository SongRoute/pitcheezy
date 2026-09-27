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
6. 하위 에이전트는 작업 범위를 분리해 배정하고 전체 자원을 총괄이 관리한다. 코드 정리·목록은 낮은 모델, 통계 설계·새 학습 경로·핵심 검토는 높은 모델을 쓴다. Claude가 Codex를 다시 호출하는 순환 위임은 하지 않는다.

## 첫 작업

| ID | 담당 | 산출물 | 완료 기준 |
|---|---|---|---|
| COOP-001 | Claude 구현 → Codex 검토·실행 | F4 실데이터 context cache 감사 러너·회귀 검사·새 실행 등록 | 코드 검토, 실제 동등성/비용 측정 또는 보존된 실패 근거, 채택 여부와 기존 예산 적용 |
| COOP-002 | Codex 설계 → Claude 독립 검토 | G0/F1 5seed 안정성·bridge 연장 및 G0 전체 MLB 평가 설계 | 재사용·보정·다중 비교·비용·기존 DEV 노출 한계를 명시하고 실행 공백 식별 |

이 문서는 협업 규칙이다. 실험의 실제 상태와 과학적 결론은 각 config·manifest·실행 보고서가 결정한다.
