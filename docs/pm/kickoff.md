# 새 작업 시작

상위: [운영 진입점](overview.md) · 절차 정본: [workflow](workflow.md)

다음 프롬프트는 새로운 권한이나 상시 실행을 만들지 않는다. 현재 요청에 필요한 경로만 읽는다. 다른 worktree의 문서·로컬 커밋은 자동으로 동기화되지 않는다.

```text
Pitcheezy의 현재 사용자 요청을 수행하라.
1. AGENTS.md와 docs/pm/overview.md를 읽고 현재 요청에 해당하는 계약·상태·고정 입력을 확인하라.
2. docs/pm/work-queue.md에 따라 최신 전체 댓글·양쪽 상황판·열린 mail을 읽어 소유권을 확인하라. 원격 발신은 현재 권한을 따르라.
3. 기준 커밋·dirty·정확한 수정 파일·완료 조건·자원·비대상을 기록하고 별도 worktree를 사용하라. 다른 활성 소유자의 범위를 재배정하지 마라.
4. docs/pm/workflow.md의 등록·검사·검토·PR·종료 절차를 필요한 범위에 적용하라. 승인된 후속 작업은 계속하되 요청 기간·중단 조건을 지켜라.
5. 한 일 / 검사 결과 / 미측정·남은 조건 / 결정 필요 네 줄로 보고하라. 완료·통합·운영·과학 판정을 구분하라.
```

## 읽기 조회

```sh
gh issue view 35 -R SongRoute/pitcheezy --comments
gh issue list -R Pitcheezy/transition-models -l board --state all --limit 100
gh issue list -R SongRoute/pitcheezy -l mail --state open --limit 100
gh issue view 37 -R SongRoute/pitcheezy
gh api --paginate repos/SongRoute/pitcheezy/issues/37/comments --jq '.[] | {id, body}'
```

발견한 board/mail도 본문·댓글을 읽고 한도에 도달하면 페이지를 추가 조회한다. 쓰기 직전 최신 점유를 재확인한다. 조회는 CLAIM·발신·자료 실행·배포를 승인하지 않는다. 실제 명령은 사용 가능한 커넥터/CLI에 맞춘다.
