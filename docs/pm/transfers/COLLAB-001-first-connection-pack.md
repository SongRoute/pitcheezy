# COLLAB-001 첫 연결 묶음

상태: 전달용 초안, 2026-10-12. 정본은 Song 저장소다. 이 문서는 첫 연결에 필요한 고정 소스와 파일 계약을 가리킨다. 공개 호스팅의 선택과 운영은 팀원 쪽 몫이다. 근거 경로는 모두 저장소 상대 경로이며, 아래 `svc/` 브랜치의 파일은 해당 브랜치에서 읽는다.

## 1. 한눈에

- 발행기: Song 저장소 `svc/ws-publisher`, `d5a1dd98003672235cc50509391fd4c41b28be32`. `apps/observer/publish_contract/v1/`에 JSON Schema와 상태별 예시가 있고, `apps/observer/backend/observer_app/publisher.py`가 파일을 만든다.
- 공개 화면: Song 저장소 `svc/ws-static-web`, `ef4e4df2f0547331459f9d952c5d76e1090f11e8`. `npm run build:public`로 공용 정적 묶음을 만든다.
- 이 전달 문서: Song 저장소 `codex/first-connection-pack`. 작성 전 기준 커밋은 `ed3a51c60482dbccdb084f437cc3c5e93f2dfc6a`다. 이 문서를 담은 새 커밋은 전달할 때 함께 적는다.
- 두 `svc/` 브랜치는 현재 로컬 작업트리에서 확인했다. 원격에 올라갔는지는 확인하지 못했으므로 팀원이 위 커밋을 받을 수 있다고 전제하지 않는다.

원격 push 여부: (PM이 채움)

## 2. 화면 빌드와 배치

`svc/ws-static-web`의 `apps/observer/README.md` 「공개용(정적 파일 읽기) 화면 빌드」와 `apps/observer/web/package.json` 기준이다. Node.js 요구 범위는 잠긴 Vite `7.3.6`의 `apps/observer/web/package-lock.json`에 있는 `^20.19.0 || >=22.12.0`이다. 프로젝트가 고정한 npm 버전은 **모름**이다. npm과 Node.js가 있어야 `npm ci`를 실행할 수 있다.

```sh
cd apps/observer/web
npm ci
npm run build:public
```

결과는 `apps/observer/web/dist-public/`이다. API 방식과 비교하는 선택 검사 명령은 `npm run build`를 먼저 실행한 뒤 `node tests/static-publish-smoke.mjs`다. 이 검사는 로컬 테스트용이며 첫 연결 절차에서 서버 실행을 요구하지 않는다. 근거: `apps/observer/README.md` 같은 절, `apps/observer/web/tests/static-publish-smoke.mjs` 첫머리.

빌드 시 `apps/observer/web/.env.public`를 읽고 셸 환경변수가 우선한다. 기본값은 다음과 같다. 단위는 이름 끝의 `MS`가 밀리초, `S`가 초다.

- `VITE_PUBLISH_BASE=.`: `v1/`을 찾는 위치. `.`은 `index.html` 옆이다. 다른 출처를 쓰려면 빌드 전에 이 값을 그 공개 파일 위치로 정한다.
- `VITE_PUBLISH_POLL_MS=2000`, `VITE_PUBLISH_PREGAME_POLL_MS=10000`, `VITE_PUBLISH_LIST_POLL_MS=10000`, `VITE_PUBLISH_DATES_POLL_MS=10000`: 진행 중 경기, 경기 전, 현재 목록과 health, 날짜와 manifest의 재조회 간격.
- `VITE_PUBLISH_RETRY_MS=4000`, `VITE_PUBLISH_TIMEOUT_MS=15000`, `VITE_PUBLISH_LOST_MS=30000`, `VITE_PUBLISH_TICK_MS=5000`: 실패 뒤 재시도, 한 요청의 제한 시간, 성공한 읽기가 없을 때 연결 끊김 기준, 시간 안내 재평가 간격.
- `VITE_PUBLISH_STALE_S=30`: **잠정·미측정** 갱신 지연 기준. 리허설에서 정한다. `LOST_MS`와 `STALE_S`는 각 화면 읽기 간격의 `3`배보다 짧게 적용되지 않는다.
- `VITE_PUBLISH_MISSING_RETRY_MS=10000`, `VITE_PUBLISH_MISSING_MAX_MS=60000`: 없는 경기 파일을 다시 찾는 시작·최대 간격. **잠정·미측정**이다.
- `VITE_PUBLISH_PINNED_POLL_MS=10000`: 종료 전 지난 공에 화면을 고정했을 때의 재조회 간격. **잠정·미측정**이다.

근거: `apps/observer/web/.env.public`, `apps/observer/web/src/service/publish.ts::PUBLISH_CONFIG`, `apps/observer/README.md` 같은 절. 같은 출처면 `dist-public/`의 내용과 `v1/`을 같은 폴더에 둔다. 링크는 `?game=<game_pk>`와 `?date=<날짜>` 형태여서 별도 주소 재작성 규칙이 필요 없다. 다른 출처면 아래 CORS 헤더가 필요하다. 호스트의 이용 기록·피드백 코드는 화면 뒤의 `<div id="pitcheezy-host-slot" data-screen="list|game" data-game-pk="…">`와 `window`의 `pitcheezy:host-slot` 이벤트(`detail.element/screen/gamePk`)에 붙인다. 근거: `apps/observer/README.md` 같은 절, `apps/observer/web/src/service/HostSlot.tsx`.

공개 묶음은 `/api/**` 호출, 옛 API 경로 대체, 시연 세션 `?demo=`, 운영자 화면, `?after=` 이어받기, 접속 기록·피드백·영상을 담지 않는다. 화면은 `dates`, `manifest`, `games/<날짜>`, `game/<game_pk>`, `health` 다섯 꼴의 JSON만 요청하며 `setup` 파일은 현재 공개 화면의 요청 대상이 아니다. 근거: `apps/observer/README.md` 같은 절, `apps/observer/web/src/service/publish.ts::publishedPath`, `apps/observer/web/src/service/api.ts::getPublished`.

## 3. 발행 파일과 고정 예시

`svc/ws-publisher`의 `apps/observer/backend/observer_app/publisher.py::layout`, `Publisher.cycle`, `Publisher._lists`, `Publisher._game`, `Publisher._setup`과 `apps/observer/README.md` 「공개 발행기」 기준이다. 모든 파일은 원래 서비스 본문에 최상위 `publish`만 더한다. `publish`에는 `contract: "v1"`, `sequence`, `published_at`이 있고 경기·셋업 파일에만 `final`이 있다. 순번은 파일을 마지막으로 쓴 주기의 번호다. 다른 파일끼리 순번을 맞춰 비교하지 않는다.

- `v1/manifest.json`: 계약, 발행기 코드 `commit`·`commit_dirty`, 발행 시각·간격·고정 지연, `live_dates`, 파일 경로 목록. 매 주기 다시 쓴다. `publish`에는 `final`이 없다.
- `v1/health.json`: 이번 주기의 완료 표식, 순번·시각·간격·쓴 파일 수와 실패 목록. 주기 맨 마지막에 쓴다. `publish`에는 `final`이 없다.
- `v1/dates.json`: 보관된 종료 경기의 날짜 목록. 매 주기 다시 쓴다. 진행 중 날짜는 manifest의 `live_dates`다. `publish`에는 `final`이 없다.
- `v1/games/<YYYY-MM-DD>.json`: 날짜별 경기 카드. 따라가는 날짜는 매 주기, 지난 날짜는 `settled_recheck_s`에 도달하면 다시 만들고 내용이 바뀔 때만 다시 쓴다. `publish`에는 `final`이 없다.
- `v1/game/<game_pk>.json`: 경기 전체 본문. 진행 중이며 따라가는 날짜의 경기는 매 주기 다시 쓴다. 경기 종료·확인 중인 공 없음이 함께 성립하면 `publish.final: true`로 쓰고, 이후 재확인에서 내용이 바뀐 정정만 새 순번으로 다시 쓴다. 경기 전 등 느린 대상은 설정된 재확인 시점에 만든다.
- `v1/setup/<game_pk>.json`: 종료 경기의 미트 가로 위치 추정. 원본 JSONL·출처 핀 검사를 통과했을 때만 생기고, 내용이 바뀌면 다시 쓴다. `publish.final: true`다. 공개 화면은 아직 이 파일을 읽지 않는다.

아래 SHA-256은 `svc/ws-publisher`에서 `shasum -a 256 apps/observer/publish_contract/v1/{examples,schema}/*`로 확인했다. 경로의 공통 앞부분은 `apps/observer/publish_contract/v1/`이다. 이 예시는 합성 경기에서 만든 **독립 스냅샷**이므로 서로 다른 `sequence`와 시각을 하나의 실제 발행 주기로 해석하지 않는다. 근거: `apps/observer/scripts/export_publish_contract.py`, `apps/observer/backend/tests/publish_fixture.py`.

예시 파일과 보여 주는 상태:

- `examples/dates.json` — 보관 날짜 목록 — `0d82da07ff35f63502dabbafbf0c9e19aec007c63e4968a3fbbb657573401113`
- `examples/manifest-live.json` — 따라가는 날짜와 발행기 식별 — `26c7227140e00643bf7d007b69a4b748fedcad1f86646daa59bb85287df8a505`
- `examples/games-live.json` — 진행 중 경기 목록 — `c46eb01f6ea3cf6e9f8d1260f4b6f56ab278850fad3b023442d0b92fa5c8002d`
- `examples/games-archive.json` — 종료 경기 목록 — `83ddab03fe985ff97271a9381dfde5ab253555f747d1ba7559576ab2824d4aac`
- `examples/game-pregame.json` — 경기 전 — `b8ec23572fd042e6cb9edd461d483a4740d536da8200d2a4b012fa74025ce517`
- `examples/game-live.json` — 진행 중, `recommended` — `0de4cd9e8fb92f6a0c0efcfa2a99530988602f960fdbd70d1fbc9a04407cff92`
- `examples/game-live-unconfirmed.json` — 확인 중인 공, `unconfirmed` — `8d2a2ec14b0619202d0c80c3fdd7fca960df3a24a7e83bb9930aeb1c03f23f43`
- `examples/game-between-innings.json` — 공수 교대, `waiting` — `493c68af92f7cb97f6417511b34f6a43226ecced2a9ed57bbd78202fb8db0d84`
- `examples/game-disconnected.json` — 피드 연결 끊김, `disconnected` — `e54eaca9bc5d4eaaeb145c91d9217230adee5429d22fca75c0479b9f9e0a0930`
- `examples/game-final.json` — 라이브 출처 경기 종료 — `25fd54a8c793869848c0fac8b5c0a25302de2f833ba23b4ccca3e691808eb74b`
- `examples/game-final-archive.json` — 보관 출처 경기 종료 — `724112fc4f61627ba63d2ca19a5b55f83474fb6f9b8680044125dfcef84f8bf6`
- `examples/setup.json` — 종료 경기의 검토 전 미트 가로 위치 — `b552f7043abce807607970ee0357b510b21e52a3a22809ae85dcf4a2363713f2`
- `examples/health-live.json` — 발행이 이어지는 상태 — `73fcc38d5d6bff12564ea20286c801a964a33ba954c038b73478acabf8e0b5e9`

JSON Schema 파일:

- `schema/dates.schema.json` — `24c2602ed2b0da117139690e0387cbbcb9fd6f36671e6c547298f751965c05cc`
- `schema/games.schema.json` — `f885e45f093e1d63f8070a12ac0eb20157c66aaf585406a194e9c290f6401d96`
- `schema/game.schema.json` — `78d2f67890b0129861dc473ba4188d0f7d2512b87d09ecf667c7ea32ef5ee37a`
- `schema/setup.schema.json` — `7d07e650137930d0e6065ea2b84541683179dd7da6741fdaa7edb051c1191d1b`
- `schema/manifest.schema.json` — `d4b09e762f2c38e5c9cc638846ffbcaafdab3ae791663e6ed251e8c044404d6c`
- `schema/health.schema.json` — `3c39efa163a4e81c239cefcb70a0f4bac346f44fbd2e031a5c5b5ede4b12524a`

합성 자료에서 추론한 스키마는 실경기 오프라인 재생에서 본 형태와 본문 생성 코드의 허용 타입에 맞춰 넓혔다. 경기 `venue`·사람의 `jersey`는 null 가능, `current.context`의 타자·투수 통계는 null 가능, `pitcher.earned_runs`는 정수 또는 null, 타석의 `batter`·`pitcher`·`first_index`·`last_index`는 null 가능, `batting_order`는 선택 항목이며 정수 또는 null이다. manifest의 `commit`·`commit_dirty`도 null을 허용한다. 근거: `apps/observer/backend/observer_app/publish_schema.py::CONTRACT_TYPES`, `apps/observer/backend/tests/test_publisher.py::test_the_schemas_accept_the_shapes_the_builders_can_produce_beyond_the_fixtures`. 이 스키마는 확인한 빌더 타입의 범위이며 임의의 다른 타입까지 허용한다는 뜻은 아니다.

## 4. 발행·복구 규칙

- 발행기 하나에 주기 순번 하나다. 주기를 시작할 때 순번을 올리고 상태 파일을 파일·디렉터리까지 `fsync`한 다음 공개 파일을 쓴다. 재시작 때 상태 파일과 이미 나간 **모든 계약 경로 파일**의 `publish.sequence` 최대값에서 잇는다. 상태 파일이 깨져 있으면 경고하고 공개 파일을 훑으며, 깨지거나 계약 경로가 아닌 파일은 건너뛴다. 읽기가 가능한 원격 sink는 `paths`·`read_tail`을 제공해야 이 복구 근거가 유지된다. 근거: `publisher.py::Publisher._load_state`, `_published_sequence`, `_save_state`, `cycle`; `apps/observer/README.md` 「파일」.
- 출력 폴더 하나에 발행기 하나만 잠금을 잡는다. 다른 프로세스가 같은 폴더 잠금을 잡고 있으면 시작하지 않는다. 로컬 sink는 출력 폴더 옆의 임시 폴더에 완성본을 쓰고 `os.replace`로 교체하며, 이전 실행이 남긴 자기 패턴의 임시 파일을 잠금 획득 뒤 청소한다. 올리는 쪽도 공개 경로에서 부분 파일을 보이지 않게 하고 점(.)으로 시작하는 임시 파일을 올리지 않는다. 근거: `publisher.py::Publisher._acquire`, `LocalDirSink.write`, `LocalDirSink.cleanup`, `Sink.write`; `apps/observer/README.md` 「파일」.
- 종료 표시는 `game.status=final`, 현재 phase와 표시 상태가 final, 확인 중인 공이 없음이라는 조건을 함께 만족해야 한다. 종료 파일은 안정된 본문이 같으면 순번을 유지하고, 뒤늦은 공식 정정으로 본문이 바뀌면 새 순번으로 교체한다. 시계만 변하는 `feed`와 `display_state.feed`의 연결 상태 일부는 정정으로 치지 않는다. `settled_recheck_s=600`, `final_recheck_for_s=21600`, `final_slow_recheck_s=21600`은 모두 **잠정·미측정** 설정이다. 종료 후 일정 기간이 지나도 느린 재확인은 계속할 수 있고, 로컬 보관 최종 피드가 도착하면 즉시 읽어 바꾼다. 근거: `publisher.py::is_final`, `game_digest`, `Publisher._game`, `load_config`; `apps/observer/publish_config.json`.
- 시연이 잡은 경기는 목록 카드를 가리고 `game`·`setup` 파일을 삭제한다. 상태 파일이 이 파일을 몰라도 삭제하며, 주기 중 시연 시작을 다시 확인한다. 마지막 확인 이후 시연 시작은 다음 주기에 반영되는 틈이 있다. 호스팅은 `DELETE` 또는 동등한 삭제 반영을 지원해야 한다. 삭제 실패는 `withdraw_failed`로 health에 남는다. 근거: `publisher.py::Publisher._withdraw`, `_lists`, `_game`, `cycle`; `apps/observer/README.md` 「발행하지 않는 것」.
- `health.json`은 맨 마지막에 쓰며 `schema`, `sequence`, `published_at`, `publish_interval_s`, `files_written`, `unavailable_games`, `unavailable_dates`, `feed_failing_games`를 싣는다. `unavailable_games`에는 경기 번호, 응답 상태(있을 때), 고정 코드만 넣는다. 코드는 `refused`, `build_failed`, `write_failed`, `setup_failed`, `not_public`, `withdraw_failed`, `archive_unread`이다. 피드 실패 중 마지막 스냅샷을 쓰는 경기는 `feed_failing_games`에 번호를 넣는다. 경로나 오류 문장은 공개하지 않는다. 근거: `publisher.py::Publisher._fail`, `_game`, `cycle`; `apps/observer/README.md` 「파일」.
- 한 경기·날짜 목록의 생성 또는 쓰기 실패는 다른 경기와 health 발행을 막지 않는다. 그 파일은 대체로 마지막 발행본을 유지하지만, 시연 중 목록을 안전하게 다시 만들지 못하면 이전 점수를 남기지 않도록 목록을 삭제한다. 주기 전체의 예외는 로그에 남기고 다음 주기에 재시도하므로 이때 health는 움직이지 않는다. 저장소나 추천 장부가 없으면 시작을 거부하고, 장부가 실행 중 꺼지면 그 주기는 순번을 잡거나 파일을 쓰기 전에 거부한다. 근거: `publisher.py::Publisher._lists`, `_game`, `run`, `cycle`, `unsafe_reason`, `build`; `apps/observer/README.md` 「공개 발행기」.

## 5. 호스팅이 지켜야 할 조건과 관측치

다음 조건은 `docs/pm/transfers/COLLAB-001-ws-public-launch-request-draft.md` §1-1과 `svc/ws-static-web`의 `apps/observer/README.md` 「공개용(정적 파일 읽기) 화면 빌드」를 따른다.

- `v1/*.json`은 매번 원본에 다시 확인하게 한다(`Cache-Control: no-cache` 또는 발행 간격보다 짧은 `max-age`). 종료 표시 `publish.final`이 붙은 경기 파일만 길게 캐시해도 된다.
- 캐시가 대신 응답할 때 `Age`를 붙이고 `Date`를 지우지 않는다. 화면은 이 값으로 갱신 지연을 판정한다.
- 화면과 `v1/`의 출처가 다르면 `Access-Control-Allow-Origin`과 `Access-Control-Expose-Headers: Date, Age`가 필요하다. 헤더가 노출되지 않아도 화면은 동작하지만 이미 멈춘 발행을 늦게 알아챌 수 있다.
- 없는 파일은 404로 답하고 `index.html`을 200으로 대신 주지 않는다.
- 파일 삭제를 반영할 수 있어야 한다. 시연 중인 경기 파일을 내리는 데 필요하다.
- 올릴 때 파일은 통째로 바꿔치기하고 점(.)으로 시작하는 임시 파일은 올리지 않는다.

시청자 한 명의 2026-10-11 로컬 시험 관측 요청률은 진행 중 경기 화면 **60요청/분**, 경기 전 화면 **12요청/분**, 목록 화면 **24요청/분**이었다. 이 수치는 요청에 주어진 관측값이며 별도 측정 결과 파일은 전달받지 못했다. 기본 간격과 요청 파일의 근거는 `svc/ws-static-web`의 `apps/observer/web/.env.public`, `apps/observer/web/src/service/GameScreen.tsx`, `WatchList.tsx`, `api.ts`다. 실제 호스팅의 동시 접속 부하와 대역폭은 **미측정**이다.

실경기 한 경기의 오프라인 재생 요약 `20261012-ws-publisher-replay-001/run-849831-off-001/SUMMARY.md`에 적힌 종류별 최대 파일 크기(바이트, raw / gzip)는 `dates 178 / 158`, `game 534097 / 27765`, `games 686 / 396`, `health 288 / 182`, `manifest 670 / 335`다. `setup`은 그 재생에서 관측되지 않아 **미측정**이다. 이 값은 해당 오프라인 재생의 최대값이지 공개 호스팅 계측값이 아니다.

**OPEN ISSUE:** 현재 진행 중 경기 파일은 매 발행 주기에 다시 쓰고 화면은 기본 `2000 ms`마다 읽는다. 두 주기 모두 검토 중이다. 근거: `svc/ws-publisher`의 `publisher.py::Publisher._game`, `apps/observer/publish_config.json`; `svc/ws-static-web`의 `apps/observer/web/.env.public`, `src/service/publish.ts::PUBLISH_CONFIG`.

## 6. 첫 연결 절차 제안

아래 경로의 예시 파일은 공개 계약 예시를 복사하는 **시험용**이다. 서로 다른 순간의 스냅샷이므로 업로드 순서가 발행기 한 주기의 원자성을 증명하지는 않는다. 실제 운영에서는 위 순번·교체·삭제 규칙을 적용한다. 근거: `apps/observer/scripts/export_publish_contract.py`, `publisher.py::Publisher.cycle`.

1. 두 고정 커밋과 예시·스키마의 SHA-256을 확인하고, 배포 목록에 Song 발행기 커밋, Song 화면 커밋, 팀원 호스팅 커밋, 계약 `v1`, 배포 파일 SHA-256을 기록한다. 팀원 커밋은 현재 **모름**이다. 발행기 예시의 `manifest-live.json`에 든 `commit`은 합성 예시값이므로 실제 배포 식별자로 쓰지 않는다.
2. `dist-public/` 내용을 화면 위치에 둔다. 같은 위치 아래 `v1/`을 만들거나, 다른 출처라면 `VITE_PUBLISH_BASE`를 지정해 다시 빌드하고 CORS·`Date`·`Age` 헤더를 확인한다. `v1/manifest.json`과 `v1/dates.json`을 먼저 올린다.
3. 목록용 `games-live.json`을 `v1/games/2026-10-05.json`에, `games-archive.json`을 `v1/games/2026-09-30.json`에 둔다. 경기 예시는 같은 `game_pk`의 파일을 `game-pregame` → `game-live` → `game-live-unconfirmed` → `game-disconnected` → `game-between-innings` → `game-final` 순서로 `v1/game/900200.json`에 하나씩 교체한다. 이 순서는 예시의 같은 파일 순번이 뒤로 가지 않게 한다. 매 교체가 완전한 JSON으로 보이는지 확인한다. 종료 보관 예시는 `v1/game/900101.json`에, 셋업 예시는 `v1/setup/900101.json`에 둔다. 이 날짜·번호·순번은 예시 본문에서 확인한 값이다.
4. 각 단계의 데이터 파일을 올린 뒤 `health-live.json`을 `v1/health.json`에 **마지막으로** 올린다. 화면에서 `?date=2026-10-05`, `?game=900200`, `?game=900101`을 열어 확인한다. 목록에는 진행 중/종료 카드, 경기 전 예시는 경기 전, `game-live`는 다음 공 추천, `game-live-unconfirmed`는 확인 중, `game-disconnected`는 연결 끊김, `game-between-innings`는 공수 교대 대기, 두 final 예시는 경기 종료가 기본 상태로 보여야 한다. 고정 예시의 발행 시각은 과거이므로 현재 시각의 호스팅에서는 이 기본 상태 위에 "갱신 지연"이 함께 뜰 수 있다. `setup.json`은 현재 화면에서 읽지 않으므로 업로드·검증까지만 확인한다. 화면 상태의 근거는 예시 본문과 `svc/ws-static-web`의 `apps/observer/web/src/service/api.ts::STATE_COPY`, `GameScreen.tsx`다.
5. 발행이 멈춘 상황은 새 `health.json`을 올리지 않은 채 호스팅의 현재 `Date`·`Age`와 임시 기준을 넘겨 확인한다. 진행 중 화면에는 마지막 내용을 남기고 **"갱신 지연"**이 떠야 한다. 고정 예시의 발행 시각은 과거이므로 그대로 올린 시험에서는 즉시 지연으로 보일 수 있다. 시험 시각을 바꾼 파일은 원본 예시 SHA-256과 달라지므로 별도 시험 산출물로 기록한다. 근거: `svc/ws-static-web`의 `apps/observer/README.md` 「갱신 지연 판정」, `src/service/publish.ts::publishLag`, `GameScreen.tsx`.
6. 실제 발행 시에는 `v1/manifest.json`의 `commit`, `commit_dirty`, `contract`, `publish.sequence`와 별도 배포 목록의 두 소스 커밋·계약 버전을 대조해 배포 소스를 식별한다. manifest의 커밋은 발행기 소스의 식별자이며 화면 소스 커밋은 별도 배포 목록에 적어야 한다. 근거: `svc/ws-publisher`의 `publisher.py::code_commit`, `Publisher.cycle`; `docs/pm/collaboration-charter-draft.md` §5.

첫 연결 합격 문장과 확인 항목:

- “합의한 예시를 올린다”: 위 경로와 SHA-256, 404·교체·삭제·캐시 헤더를 확인한다.
- “화면이 읽는다”: 목록·경기 화면을 실제 공개 위치에서 열고 `/api/**` 요청이 없는지 확인한다.
- “갱신 / 미지원 / 발행 중단을 구별한다”: `game-live`의 추천 갱신과 오래된 health의 갱신 지연은 확인할 수 있다. **현재 고정 예시에는 화면의 `display_state.state=unsupported`를 직접 보이는 파일이 없다.** `game-disconnected.json`의 `next.rec.status=unsupported`는 화면 상태 `disconnected`와 다르다. 미지원 합격에는 발행기 출력으로 만든 별도 합의 예시와 SHA-256이 필요하며, 현재 값은 **모름**이다. 근거: `apps/observer/publish_contract/v1/examples/`, `svc/ws-static-web`의 `src/service/api.ts::STATE_COPY`.
- “배포 소스를 식별한다”: 실제 manifest의 발행기 커밋과 팀원 배포 목록의 화면·호스팅 커밋을 대조한다.

## 7. 의도(미트) 자료

검사 경계는 `apps/observer/backend/observer_app/intent.py::validate_intent_estimate`다. 테스트에서 쓰는 정확한 검사 명령은 다음과 같다. 근거: `apps/observer/README.md` 「백엔드 테스트」, `apps/observer/backend/tests/test_intent.py`.

```sh
PYTHONPATH=apps/observer/backend .venv-observer/bin/python -m pytest apps/observer/backend/tests/test_intent.py -q
```

받는 형식은 한 경기의 `PITCHEEZY_INTENT_DIR/<game_pk>.jsonl`에 투구당 `IntentEstimate` v1 한 줄, 옆의 `<game_pk>.source.json`에 `repository`, `branch`, `commit`, `path`, `sha256`, `video`, `selection`이다. `sha256`은 JSONL의 원본 바이트 해시와 일치해야 하고 모든 줄은 검사기를 통과해야 한다. 투구 키가 다른 경기이거나 중복이면 경기 전체를 거부한다. 공개 셋업 파일의 `source`에는 원본의 `commit`과 `sha256`만 남는다. 근거: `apps/observer/backend/observer_app/setup_estimates.py::SetupEstimates._read`, `publisher.py::public_setup`, `Publisher._setup`.

셋업은 투수의 실제 의도가 아닌 **검토 전 추정**이며 공개 본문은 가로 위치만 제공한다. 자료가 없는 경기는 셋업 파일도 없다. 근거: `docs/intent-spec.md` §2·§5, `setup_estimates.py::SetupEstimates._view`, `publisher.py::Publisher._setup`.

## 8. 이용 조건

보고서: `docs/pm/service/mlb-data-terms-research-2026-10-11.md`. 이 브랜치에서 이 경로의 마지막 커밋은 `ed3a51c60482dbccdb084f437cc3c5e93f2dfc6a`다(`git log -1 --format=%H -- docs/pm/service/mlb-data-terms-research-2026-10-11.md`).

보고서 결론은 이렇다. 무료 학생 프로젝트여도 누구나 접속하는 웹사이트에서 MLB의 투구별 실시간 데이터를 표시할 권한은 확인되지 않았다. 공개 표시·자동 수집·보관·모델 활용 범위를 적어 MLBAM의 사전 서면 허가를 받거나 그 범위를 포함한 데이터 라이선스를 확보하는 경로가 가장 분명하다. 지연 표시나 출처 표기만으로 허가가 생기지 않는다. 허가 전에는 초대 이용자와 접근 코드처럼 공개 접근을 제한해야 할 수 있으므로 호스팅 설계에서 접근 게이트를 켤 수 있게 한다. 근거: 위 보고서 「결론」·§3·§4. 공개 여부와 허가 판단은 사람이 한다.

## 9. 아직 정해지지 않은 것 / 우리 쪽에서 곧 바뀔 수 있는 것

- 원격 push 여부, 팀원 호스팅·업로드 방식·호스팅 커밋, 예시 전달 경로와 실제 접근 주소는 **모름**이다. `docs/pm/transfers/COLLAB-001-ws-public-launch-request-draft.md`는 아직 초안이다.
- 발행 간격, 종료 재확인 주기, 화면의 갱신 지연 기준·없는 파일 재시도·고정 화면 재조회, 캐시 시간은 잠정 설정이다. 리허설 측정 후 바뀔 수 있다. 근거: `apps/observer/publish_config.json`, `svc/ws-static-web`의 `apps/observer/web/.env.public`.
- 진행 중 경기 파일의 매 주기 재작성과 화면의 기본 `2000 ms` 조회는 검토 중이다. 화면이 `setup`을 읽는 연결도 아직 없다. 근거: `svc/ws-publisher`의 `publisher.py::Publisher._game`, `svc/ws-static-web`의 `src/service/publish.ts::publishedPath`.
- `unsupported` 화면 상태를 보여 주는 고정 예시와 그 SHA-256은 아직 없다. 첫 연결 합격 전 별도 합의가 필요하다. 근거: `apps/observer/publish_contract/v1/examples/`.
- 동시 접속 수·끝에서 끝까지의 지연·공개 업로드 성능·알림 시간은 **미측정**이다. 공개 자료 이용 허가도 확인되지 않았다. 근거: `docs/pm/transfers/COLLAB-001-ws-public-launch-request-draft.md` 「합격 기준」, `docs/pm/service/mlb-data-terms-research-2026-10-11.md`.
