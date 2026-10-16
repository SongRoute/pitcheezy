# 인수인계 — 2인 협업·월드시리즈 공개 서비스 준비 (2026-10-11~12 세션)

작성: 2026-10-12, Claude PM 세션 종료 시. 다음 세션은 이 문서, 상황판(SongRoute/pitcheezy#35), 팀원 우편(Pitcheezy/transition-models#18)을 먼저 읽는다.

## 한 줄 상태

1차 목표는 2026-10-23~31 월드시리즈 공개 서비스(준비 완료 시한 10-22)다. 발행기와 공개용 화면을 만들고 실제 경기 한 판(849831)을 오프라인으로 재생해 확인했다. 실시간 실경기, 실제 호스팅, 팀원 쪽 연결은 아직이다.

## Song이 정한 것 (이 세션)

- 정본은 Song 레포. 레포는 둘로 유지. 팀원(조현준, `Pitcheezy/transition-models`)은 영상 미트 자료 + 호스팅·운영, Song은 추천 모델에 집중.
- 통합본 본체는 Song 관전 화면. 서비스 코드 기준은 `rc/ws-2026`.
- 공개 경로: 맥미니가 파일을 올리기만 하고 팬의 화면은 그 파일만 읽는다. 지연은 모두에게 같은 값 하나(기본 0). 중계 영상은 넣지 않는다.
- 두 마감: 월드시리즈(1차), 11월 말 전시(2차, 1차 이용 기록 제시).
- 상황판·우편함은 GitHub 이슈. 작업 방식 v1은 `docs/pm/collab-protocol.md` 맨 아래(팀원 설명·합의 전).
- Claude PM 하나가 Codex CLI(astra/sol)를 일손으로 부른다. Claude 주간 사용량이 적을 때는 Codex를 더 쓴다.
- MLB 자료 이용: 허가 요청 메일 + 지도교수 문의를 함께 진행. 초안 `docs/pm/service/mlb-permission-request-draft.md`. Song 발송 전.
- "확인 중" 규칙 두 가지 완화 승인(이미 수락된 투구는 계속 표시 / 경기 단위 게이트는 추천 입력이 바뀔 때만). `rc/ws-2026` 작업 세션에 전달할 문장은 이 세션 대화에 있다. 전달 여부는 Song에게 확인한다.

## 브랜치와 커밋

원격에 있음
- `rc/ws-2026`: 서비스 코드 기준. 다른 세션(pitcheezy-52)이 계속 커밋 중.
- `collab/ws-2026` `c0e2e28`: 헌장, 규약(작업 방식 v1 포함), 공개 발행 계약, COLLAB-001 요청서, 첫 연결 자료 묶음, MLB 조사, intent-spec v1.1.
- `svc/ws-publisher` `d5a1dd9`: 발행기 + 오프라인 재생 도구.
- `svc/ws-static-web` `ef4e4df`: 공개용 화면(발행 파일만 읽음).

로컬에만 있음
- `svc/ws-publisher-upload` `6ca6fc3` (`d5a1dd9` 위): HTTP 업로드(HttpSink·MirrorSink). Codex astra 작성. 백엔드 788 통과 / 1 실패(기존 T7 건, PM이 샌드박스 밖에서 실행). 다른 모델 검토 전.
- `svc/ws-static-web` 로컬은 `da1f686`까지 5커밋 더 있음: 읽기 간격을 발행 간격에 맞추고 health를 먼저 읽는 변경(Codex astra). 브라우저 검사는 Codex 샌드박스에서 못 돌렸다. 아래 "닫을 때 상태" 참고.
- `codex/teammate-candidates-1011` `4eacb36`: 팀원 아이디어 목록·대응·등록 초안 2건·질문 13개. 실행·검토 없음.
- `codex/ml-matrix-execution`(주 작업 폴더): 이 문서와 PM 문서 커밋들. push하지 않았다.

worktree: `pitcheezy-worktrees/` 아래 `ws-publisher`, `ws-static-web`, `collab-ws-2026`, `codex-first-connection`, `codex-publisher-upload`, `codex-teammate-candidates`, `_ref-transition-models`(팀원 레포 읽기용 사본).

## 실자료 실행 기록

등록·결과: `/Users/song/Projects/pitcheezy-pm-jobs/20261012-ws-publisher-replay-001/` (REGISTRATION.md 9판)
- 시도 1 `run-849831-off-001`(정책 끔, 코드 `1006e95`): 1,177주기 완료, 불변 조건 위반 0, 최대 RSS 106,315,776 B, 경기 파일 최대 534,097 B(gzip 27,765).
- 시도 2 `run-849831-on-001`(정책 켬·MPS, 코드 `d5a1dd9`): 1,177주기 완료, 위반 0, 계산 320회·발행된 결정 292·원장 292행, 최대 RSS 1,994,440,704 B, 경기 파일 최대 732,239 B(gzip 53,114), 계산 6건이 5.5~7초(바인딩 6회와 수가 같음, 인과 미확인).
- 한 경기·오프라인이다. 실시간 폴링, 호스팅, 다른 경기는 확인되지 않았다.

드러난 것
- 제공자가 한 버전에서 투구 식별값을 빼고 보내 170구(주기 837)·91구(주기 1031)가 한 주기 동안 미확인이 됐다. 1031에서는 미확인 91구 상태로 추천이 발행됐다 — 규칙 완화 작업에서 판단 필요.
- 거의 모든 투구에 `corrected`가 붙는다(주기 836에 169개 중 159개).
- 스키마가 추천 후보 2곳에서 어긋난다: `/next/rec/candidates/*/detail/probability`, `/next/rec/candidates/*/target`.
- 추천 모델은 이 맥에서 MPS를 쓴다. 이 묶음에 CPU 경로는 없다.

## 다음에 할 일 (우선순위 순)

1. `svc/ws-static-web` 읽기 횟수 변경의 브라우저 검사 결과 확인(아래), 실패면 수정. 통과면 다른 모델 검토 후 push.
2. 발행기 쪽 후속(Codex, `svc/ws-publisher-upload` 위에서): 미지원 투수 예제 추가, 추천 항목 스키마 2곳, health의 `game_checks`(설계 노트: 이 세션 scratchpad `codex-rewrite/result.md` — 요지는 health에 경기별 `checked_at`/`sequence_checked`/`file_sequence`/`feed_ok`를 넣고, 내용이 안 바뀐 진행 중 경기 파일은 다시 쓰지 않는 것. 다시 쓰기 생략은 화면이 먼저 배포된 뒤 마지막에 켠다).
3. 업로드 기능의 독립 검토(작성자와 다른 모델, 실행 포함) 후 push, 팀원 이슈에 업로드 받는 쪽 요구사항과 갱신된 호스팅 조건(ETag·304) 게시.
4. 팀원 일정: 10-13 자료 전달(문서는 올림), 10-14~15 예제 연결, 10-17 실제 발행기 연결. 팀원 답(업로드 API, 접근 문, 미트 자료 처리 시간)을 #18에서 확인.
5. 2단계(실제 진행 중 경기 한 경기) 등록서 초안 → 독립 검토 → Song에게 `결정 필요`. 발행기와 기존 서버가 각자 피드를 읽는 문제, 저장소 분리, MPS·메모리, 다른 세션의 시험과 겹치지 않는 시간을 정해야 한다.
6. `rc/ws-2026`, `svc/ws-publisher(-upload)`, `svc/ws-static-web`을 합치는 순서. 세 갈래가 `service_api.py`, `main.py`, `GameScreen.tsx`, `api.ts`를 함께 건드린다. pitcheezy-52 세션의 계획을 먼저 확인한다.
7. 묵은 화면 검사 정리와 `zone-geometry-check` 회귀 후보(`afe893f`)는 `rc/ws-2026` 세션 몫.
8. 팀원 아이디어 등록 초안 2건은 Claude 쪽 검토 뒤 실행 승인을 Song에게 올린다(GPU 필요 시 `gpu st` 확인).

## Song의 행동을 기다리는 것

- MLB 허가 요청 메일과 지도교수 문의 발송(빈칸: 이름·학교·연락처·허가 없을 때의 계획)
- 팀원에게 작업 방식 v1 설명과 세 가지 질문(즉시 내림 스위치, 접근 문, 하루 한 번 확인)
- `rc/ws-2026` 세션에 전달문 붙이기(확인 중 규칙 완화, 화면 배치 결함, 회귀 후보, 시청자별 지연 기본값)

## 주의

- 실행 중인 서버·기록기(PID 94628·94642, 99202 등)는 건드리지 않는다. 실자료 실행은 등록 + 작성자와 다른 에이전트의 검토 뒤에만.
- Codex를 worktree에서 쓰기 권한으로 부를 때는 `--add-dir /Users/song/Projects/pitcheezy/.git`을 준다. Codex 샌드박스는 소켓·브라우저를 못 열어 테스트 수치가 불완전하다 — 전체 테스트는 PM이 밖에서 다시 돌린다. 기존 테스트 묶음이 외장 드라이브 자료를 읽는다는 점을 Codex 지시문에 미리 적는다.
- 팀원 레포에 새 이슈를 올리는 것은 Song 승인 뒤에만. 이미 있는 #18에 이어 쓰는 것은 진행해도 된다.
- 이슈는 공개다. 비밀 값·접속 주소를 적지 않는다.

## 닫을 때 상태 — `svc/ws-static-web` 읽기 횟수 변경 (로컬 `da1f686`, push 안 함)

- PM이 샌드박스 밖에서 실행: `npm run build` 성공, `npm run build:public` 성공, `node tests/static-publish-smoke.mjs` **370 통과 / 2 실패**.
- 실패 2건(원인 미조사):
  - `late/elapsed: judged from the newest publish time seen + elapsed time (data-publish-clock=elapsed), never the device clock`
  - `final checks: higher file_sequence applies late score correction`
- API 방식 스크립트(`display-state-smoke`, `demo-recovery-smoke`, `studio-layout-check`)는 이 5커밋에 대해 아직 돌리지 않았다.
- 전체 실행에 10분 넘게 걸린다(읽기 간격이 10초가 되어 실시간 대기가 늘었다). 다음 세션은 백그라운드로 돌린다.
- 따라서 원격의 `svc/ws-static-web`은 `ef4e4df`(271 통과 / 0 실패가 확인된 지점) 그대로다. 팀원은 이 커밋을 기준으로 받는다.
- Codex가 적은 요청 수 기대값(검사 안의 단언, 실측 아님): 진행 중 경기 시청자 1명당 분당 60 → 새 화면+지금 파일 13 → 새 health 필드가 있을 때 변경 없는 주기 7, 목록 24 → 14.
