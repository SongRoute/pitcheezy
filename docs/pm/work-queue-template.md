# 작업 기록 양식

상위: [공동 작업큐](work-queue.md) · 절차: [착수·인계](work-queue-protocol.md)

발신 권한이 있으면 추가 전용 댓글에, 발신이 금지되면 로컬 작업 기록에 사용한다. 로컬 기록을 원격 CLAIM이라고 부르지 않는다.

```text
PLAN | CLAIM | REVIEW | BLOCKED | RELEASE | DONE — <작업 ID>
actor: Codex / session: <고유 ID> / at: YYYY-MM-DD HH:mm KST
state: READY | RESERVED | ACTIVE | REVIEW | BLOCKED | DONE
owner: <실제 소유자 또는 추천임을 명시> / prior: <선행 댓글 링크>
goal/done: <검토 가능한 완료 조건>
depends: <입력·결정·선행 작업> / base: <고정 커밋>
worktree/branch: <식별자> / write: <정확한 파일 목록, 없으면 none>
resources/output: <실행 자원·출력 범위, 없으면 none> / excludes: <비대상>
evidence: <커밋·검사·산출물·독립 검토 링크>
next/checkpoint: <다음 행동·필요 입력·확인 시각; 자동 만료 아님>
handoff: <dirty·동작 중 작업·보존 자원·반환 범위; 반환 시 필수>
```


도입 당시 초기 계획은 [2026-10-31 고정 원문](https://github.com/SongRoute/pitcheezy/blob/6e2038fd0010657e6f6f2d1e49820490f6195da4/docs/pm/work-queue.md#도입-당시-초기-작업-계획-이력)에 보존한다. PLAN/READY는 추천이며 소유 수락이나 실행 증거가 아니다.
