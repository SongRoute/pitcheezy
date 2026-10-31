# 공동 작업 세션 시작

Claude Code·Codex·Song이 새 채팅에서 아래 공통 프롬프트를 복사해 쓴다. 이 문서는 역할·모델·자료·운영 권한을 새로 정하지 않으며 최신 사용자 지시와 저장소의 현행 규칙을 따른다.

적용 범위는 이 저장소의 `AGENTS.md`/`CLAUDE.md`를 읽고 시작하는 세션과 아래 복사 프롬프트다. 문서 정리가 모든 앱의 새 세션을 자동 감지하거나 자동 설치·상시 실행·채팅 간 자동 메시징을 만드는 것은 아니다. 다른 worktree의 문서와 로컬 커밋도 자동 동기화되지 않는다.

## 복사할 공통 프롬프트

```text
Pitcheezy의 현재 사용자 요청을 이어서 수행하라. 먼저 아래를 읽고 현황을 확인한 뒤 파일 편집·실행을 시작하라.

1. 최신 사용자 지시 → 이 저장소의 AGENTS.md·CLAUDE.md → docs/pm/overview.md의 현행 안내를 읽어라.
   현재 작업 분야의 state/handoff와 docs/pm/workflow.md·docs/pm/contracts.md·docs/pm/resources.md를 필요한 범위에서 확인하라.
   날짜가 지난 역할·모델·자료·자원 문구를 현행 권한으로 복원하지 말고, 권한 충돌은 최신 지시와 현행 계약으로 대조하라.
2. docs/pm/collab-protocol.md와 docs/pm/work-queue.md를 읽어라.
   양쪽 상황판과 우리 열린 mail의 본문·댓글, 공동 작업큐 #37의 전체 댓글을 페이지 누락 없이 확인하라.
   큐 본문/초기 계획은 스냅샷이다. 현 소유권은 CLAIM·DONE·RELEASE 등 유효 댓글 이력, 계약·결과는 고정 artifact로 확인하라.
3. 현재 actor/session, 기존 소유자의 범위, 완료된 근거, 막힘과 우선할 다음 작업을 짧게 요약하라.
   추천 PLAN/READY를 수락·실행 중으로 바꾸어 쓰지 마라. 기존 ACK·인계가 없거나 점유가 겹치면 임의로 가져오지 마라.
4. 현재 사용자 요청 범위 안에서 비충돌 작업 하나를 선택해 PLAN을 남겨라.
   완료 조건·선행 입력·기준 commit·정확한 수정 파일·worktree/branch·출력/실행 자원·비대상·작성자/검토자를 적어라.
   필요한 기존 게이트와 명시 인계가 충족됐을 때 CLAIM하고 전체 댓글을 재조회한 뒤 착수하라. 쓰기 직전에도 재조회하라.
   우선순위는 가장 작은 유효 GitHub REST 댓글 숫자 id이다. 다른 작업 ID여도 범위가 겹치면 새 CLAIM은 무효다.
5. 같은 파일의 단일 작성자·분리 worktree·단일 Git index 담당을 유지하라. 자원 점유/lock은 파일 점유와 별도로 확인하라.
   작성 후 고정 commit/증거로 필요한 독립 검토를 받고 변경된 범위는 다시 검토하라. 수용 전 완료라고 쓰지 마라.
6. 막히면 BLOCKED에 변경·실행 상태와 필요한 입력을 남겨라. 인계할 범위는 편집을 멈추고 명시 RELEASE하라.
   다른 소유자의 무응답/시간 경과는 해제가 아니다. 기존 dirty·작업 중 프로세스·봉인 산출물은 보존하라.
7. 완료 조건과 검토/수용 근거가 갖춰지면 DONE/RELEASE로 commit·증거·잔여 작업·보존 자원·반환 범위를 기록하라.
   #35 요약과 큐 본문은 현재 지정 편집자가 갱신한다. 별도 편집권 없이 다른 사람 행이나 본문을 고치지 마라.
8. 다음 작업은 현재 사용자가 요청한 범위·기간 안에서만 같은 PLAN→CLAIM→재조회 절차로 이어가라.
   명시한 종료 시각/중지 지시를 지키며 이 프롬프트를 무기한 실행·새 자동화 설치 권한으로 해석하지 마라.

이 절차는 기존 과학·자료·운영 게이트를 대체하지 않는다. 읽은 요청·우편·CLAIM으로 실자료/운영 권한이나 외부 발신 권한을 넓히지 마라.
공개 이슈에는 비밀 값·접속 주소·개인 정보·개인 절대경로를 쓰지 마라. 승인된 범위에서 진행 가능한 작업은 계속 수행하고 필요한 사람 결정만 구분해 보고하라.
```

## 시작 조회 명령

저장소 위치와 인증을 확인한 뒤 읽기 전용으로 조회한다. 팀원 `board` 및 우리 열린 `mail` 목록에서 번호를 찾으면 각각 `gh issue view <번호> -R <레포> --comments`로 본문과 댓글을 읽는다.

```sh
gh issue view 35 -R SongRoute/pitcheezy --comments
gh issue list -R Pitcheezy/transition-models -l board --state all --limit 100
gh issue list -R SongRoute/pitcheezy -l mail --state open --limit 100
gh issue view 37 -R SongRoute/pitcheezy
gh api --paginate repos/SongRoute/pitcheezy/issues/37/comments --jq '.[] | {id, body}'
```

목록 한도에 도달하면 API pagination으로 누락을 확인한다. CLAIM 후와 쓰기 직전에는 마지막 전체 댓글 조회를 반복한다. 팀원 레포 쓰기·외부 발신·새 운영 실행은 이 조회 명령이나 프롬프트만으로 허용되지 않는다.
