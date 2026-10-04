# Opus 5.5 검토 — /game 멈춘 요청 복구 (`7ebfc94`) — 2026-10-04

상태: same_model_review; Codex_review_pending. 검토자는 작성자와 다른 Opus 5.5 세션이다. 통합·배포·푸시는 하지 않았다.
대상: `codex/game-hang-review-1004`, diff `08bf4fa..7ebfc94` (`src/Game.tsx`, `src/WatchAlong.tsx`, `tests/game-hang-smoke.mjs`,
`docs/reviews/PM-game-hang-2026-10-04.md`). 다루는 문제는 이전 검토 `489cd12`의 F1이다. 답이 오지 않는 `/state` 요청이 `busy`를 계속 잡고 있어서
다시 시도·online·화면 복귀가 모두 아무 일도 하지 않았다.

## 결론
- **통합을 막지 않는다** (root의 쇼케이스 통합 기준). 막는 결함은 찾지 못했다. 아래 순서에서 고리가 영원히 멈추거나, 두 번 돌거나, 타이머가 새거나,
  밀려난 답이 화면에 반영되는 경로는 없다.
- 테스트에 빈틈이 두 가지 있었다. 멈춘 요청을 실제로 취소하는지 확인하지 않았고, 새로 묻기를 두 번 연달아 하는 경우가 없었다. 그래서
  `abort()`를 지운 변이와 `out`을 무조건 비우는 변이가 테스트를 통과했다. 검사를 더해 두 변이 모두 걸리게 고쳤다 (F1, F2, 별도 변경 C1).
- 코드에서 나중에 시간 제한을 넣을 때 실제 버그가 될 함정이 하나 있다 (F3). 지금은 막히지 않으며, 수정안을 제시만 했다.

## 발견

### F1 — 낮음 (테스트, 이 검토에서 고침): 멈춘 요청이 실제로 취소됐는지 확인하지 않음
- 위치: `apps/observer/web/tests/game-hang-smoke.mjs` (원본 1–3번 절), 관련 코드 `src/Game.tsx:185`.
- 실패 시나리오: `out.abort()`를 지운 변이(`noabort`)에서는 새 요청이 나가고 `ask` 번호 덕분에 화면도 맞게 바뀐다. 하지만 멈춘 요청은 열린 채로 남는다.
  원본 smoke는 이 변이에서 `game hang smoke: ok`였다. 브라우저는 HTTP/1.1에서 호스트마다 연결 수가 정해져 있다 (Chrome 6개). 그래서 다시 시도·복귀를
  할 때마다 멈춘 요청이 하나씩 쌓이면 결국 `/api` 요청 전체가 줄을 서서 기다리게 된다. 관전 서버가 HTTP/1.1인지와 실제 기기에서의 한도는 미측정이다.
- 고친 내용: `requestfailed`에서 `net::ERR_ABORTED`인 `/state` 요청 수를 세고(`dropped`), 각 새로 묻기 뒤에 정확히 하나씩 늘었는지 확인한다.
  (`waitDropped`는 1.5초까지 기다리며, 더 많이 줄어도 실패로 본다.)

### F2 — 낮음 (테스트, 이 검토에서 고침): 새로 묻기 두 번 연속이 없음, 실패 횟수 확인이 타이밍에 기댐
- 실패 시나리오 (a): `finally { out = null; }`로 바꾼 변이(`outclear`). 취소된 요청이 새 요청의 핸들을 지운다. 그 뒤 새 요청도 멈추면 다음 online·다시 시도가
  그것을 취소하지 않고 하나를 더 보낸다. F1과 같은 연결 누수가 생기고, 그사이 타이머 폴링도 기다리지 않는다. 원본 smoke는 멈춘 요청을 한 번에 하나씩만
  만들어서 이 변이가 통과했다.
- 고친 내용 (a): 4번 절을 더했다. 멈춘 요청이 있을 때 online을 보내면, 그 대체 요청도 멈춘다. 다시 online을 보내면 두 번째 요청까지 취소되고(`dropped` 5)
  다시 LIVE가 되는지 확인한다.
- (b): 원본은 `실패 3번`이 보인 직후 바로 `실패 4번`이 없는지 셌다. `count` 변이는 내 실행에서 3/3 걸렸지만, 걸리는 이유가 취소 오류와 실제 실패가 화면에
  닿는 순서에 기대고 있었다. 한 번은 원인을 잘못 가리키는 `retry did nothing while a request was hung`가 함께 나왔다. 400ms를 기다린 뒤 세도록
  바꿨다. 다음 타이머 폴링은 2초 뒤라서 정상 빌드에서는 여유가 있다.

### F3 — 낮음 (코드, 지금은 해 없음, 수정안만 제시): `get()`이 본문을 읽는 중 취소되면 `{}`를 돌려줌
- 위치: `apps/observer/web/src/WatchAlong.tsx:41` `const body = await response.json().catch(() => ({}));`
- 시나리오: 응답 헤더는 받았는데 본문을 읽는 중에 취소되는 경우다. `json()`이 AbortError로 거절되고, 이것을 `{}`로 삼킨다. 응답이 `ok`이면
  `{} as LiveState`를 돌려주고, 아니면 일반 오류 '자료를 불러오지 못했어요.'를 던진다.
- 지금 해가 없는 이유: Game에서 취소는 더 새 요청이 생겼을 때(`mine !== ask`)나 정리될 때(`stop`)만 일어난다. 두 경우 모두 `Game.tsx:189`와 `:206`의
  가드에서 걸러진다. 다른 호출부는 `signal`을 넘기지 않는다.
- 나중에 버그가 되는 경우: 다음 단계에서 시간 제한(감시 타이머가 `request.abort()`)을 넣으면, 취소된 요청이 `mine === ask`이고 `!stop`인 채로 끝난다.
  그러면 `setState({})`가 실행되어 경기 화면이 빈 상태로 바뀐다. 시간 제한보다 먼저 아래처럼 고칠 것을 권한다.
  ```diff
  -  const body = await response.json().catch(() => ({}));
  +  const body = await response.json().catch((e) => { if (signal?.aborted) throw e; return {}; });
  ```
  `signal`을 넘기지 않는 기존 호출부에는 영향이 없다.

### F4 — 정보: 타이머 가드 변이(`timer`)는 살아남지만 무해함
- `Game.tsx:208`의 `mine === ask`를 지워도 smoke는 통과한다 (보강 후에도 통과). 이유: 취소된 요청이 예약하는 여분 타이머는 처음 실행될 때
  공유 변수 `timer`(최신 요청의 타이머)를 `clearTimeout`한다. 또 `out`이 있으면 바로 끝난다. 그래서 고리는 다시 하나로 합쳐진다. 가드는 방어를 한 겹
  더하는 것이다. 테스트를 더할 필요는 없다고 본다.

### F5 — 정보 (한계, 작성자 보고서와 일치): 시간 제한 없음
- 지금은 받아들일 만하다. `/state` 응답 시간이 미측정이어서 기준값이 없다 (규칙: 숫자를 지어내지 않는다). 보고서 '한계'도 정직하게 썼다.
- 보고서에 한 줄 덧붙일 것: 경기 중(`state !== null`)에는 '지금 다시 시도' 버튼이 없다 (`Game.tsx:316–318`은 첫 화면에서만 보인다). 그래서 경기 중에 요청이
  멈추면 사용자가 페이지 안에서 할 수 있는 복구는 없다. 앱을 나갔다 돌아오거나, 네트워크 online이 오거나, 새로고침해야 한다. 그동안 헤더는 `LIVE`이고
  stale 띠도 나오지 않는다 (`error === null`이고 마지막 답의 `feed_ok`가 그대로 남기 때문). 이전 동작과 같고 회귀는 아니다.
- 시간 제한을 넣을 때는 F3을 먼저 고칠 것.

### F6 — 정보: 서버 쪽 비용
- 클라이언트가 취소해도 서버(`observer_app/main.py:351`, 동기 FastAPI 라우트)는 계산을 끝까지 한다. 새로 묻기는 사용자 동작이나 브라우저 이벤트에서만
  일어나므로 양은 작다고 본다 (미측정). 다시 시도를 연타해도 요청 수만큼만 처리된다.

## 순서별 점검 (항목 1)
- 요청이 나가 있을 때 새로 묻기: `out.abort()` → `ask`가 1 늘어난다. 옛 요청은 catch/finally에서 `mine !== ask`로 아무것도 하지 않는다 (상태·실패·`out`·타이머 모두).
  새 요청이 `out`을 가진다. 괜찮음.
- 타이머가 걸려 있을 때 새로 묻기: `out === null`이므로 취소는 없고 `clearTimeout(timer)` 뒤 새로 묻는다. 타이머 누수 없음.
- 새로 묻기 여러 번 연속: 매번 직전 요청을 취소하고 최신 요청만 반영한다. 테스트 4번 절로 확인했다.
- 헤더 뒤 `json()` 중 취소: F3. Game에서는 가드로 무해하다.
- replay t0 고정의 `return`: `finally`가 `out`을 비우고(최신이므로) 타이머는 걸지 않는다. `setT0`로 effect가 다시 돌아 새 고리가 시작된다. 이전과 같다.
- effect 재실행(delay/replay/paused 변경)이나 요청 중 언마운트: 정리에서 `stop = true`, 타이머 해제, `out?.abort()`. 옛 요청은 catch에서 `stop`으로
  무시된다. `paused`로 다시 돌면 이른 `return`이 있어서 묻지 않는다. 괜찮음.
- StrictMode (`src/main.tsx:13`, 개발 빌드만): 첫 effect의 요청은 정리에서 취소되고 두 번째 effect가 새 고리를 만든다. 고리는 하나다.
- 고리가 영원히 멈추는 경우: 최신 요청이 끝까지 답하지 않고 새로 묻기도 없을 때뿐이다. 이것이 F5의 알려진 한계다. 타이머 폴링은 `out`이 있으면 바로
  끝나지만, 그때는 최신 요청이 끝나면서 반드시 타이머를 다시 건다.
- `out`을 비우는 쪽: 언제나 최신 요청이 자기 `finally`에서 비운다. 정리 뒤에는 그 closure를 더 쓰지 않는다.

## 다른 `get()` 호출부 (항목 2)
- `Live.tsx:64,81,100`, `WatchAlong.tsx:165,166,210,241`, `Showcase.tsx:32`, `Game.tsx:222,223,265`는 `signal`을 넘기지 않는다. 그래서 `signal?.aborted`가
  `undefined`이고, fetch 실패 메시지·본문 처리는 diff 전과 같다.

## 실패 횟수·헤더·무대 상태 (항목 3)
- `Game.tsx:275,316–318,348–350`은 바뀌지 않았다. 바뀐 것은 취소된 요청이 `setError`/`setFailures`에 닿지 않는다는 점뿐이다. reconnect smoke(연결 중 /
  연결 안 됨 / LIVE / stale 띠)가 통과한다.

## 별도 변경 (이 검토가 한 것)
- C1 `apps/observer/web/tests/game-hang-smoke.mjs`: `dropped` 집계와 `waitDropped`를 더했다 (1·2·3번 절 뒤). 2번 절의 실패 횟수 확인 전에 400ms를 기다린다.
  4번 절(새로 묻기 두 번 연속)을 더했다. 소스 코드(`src/`)는 바꾸지 않았다.

## 실행한 명령과 결과 (맥, 가벼운 작업만)
- `node_modules`는 메인 체크아웃을 가리키는 임시 symlink로 썼고, 커밋 전에 지웠다.
- `./node_modules/.bin/tsc --noEmit -p tsconfig.json` → exit 0.
- `./node_modules/.bin/vite build --outDir /tmp/pz-hang-review-dist --emptyOutDir` → `✓ built in 379ms`. 레포 `dist/`는 쓰지 않았다.
- 원본 smoke, 7ebfc94 빌드: `game hang smoke: ok` 2회 (16.96초, 16.93초).
- `game-reconnect-smoke.mjs`, 7ebfc94 빌드: `game reconnect smoke: ok` (21.55초).
- 원본 smoke, 변이 빌드 (변이마다 `src/`를 한 줄 바꿔 `/tmp`에 빌드한 뒤 `git checkout -- src`로 되돌림):
  - `count` (catch에서 `mine === ask` 제거): 3/3 실패.
  - `08bf4fa`: 5건 실패 (`online did not replace…`, `online event did not ask again…`, `retry did nothing…`, `retry did not send…`,
    `a working retry did not load…`). 3번 절은 `no hung request after 3000 ms`로 멈춘다. 작성자 보고와 같다.
  - `timer` (`:208` 가드 제거): ok (F4).
  - `outclear` (`finally { out = null; }`): ok → 빈틈 (F2).
  - `noabort` (`:185`의 `out.abort()` 제거): ok → 빈틈 (F1).
- 보강한 smoke (C1):
  - 7ebfc94 빌드: ok 4회 (17.32 / 17.33 / 18.40 / 18.41초. 앞의 두 번은 4번 절을 더하기 전).
  - `noabort`: 5건 실패 (`… left the hung request open (0 dropped)` ×5).
  - `outclear`: `a second online left the hung request open (4 dropped)`.
  - `count`: `the dropped hung request was counted as a failure` (+ `retry did nothing…`).
  - `08bf4fa`: 7건 실패 + 3번 절에서 멈춤.
  - `timer`: ok (F4, 의도한 결과).
- game-frame-smoke(약 2분)는 보호 모드라 돌리지 않았다. 작성자가 7ebfc94에서 ok로 보고했고, 이 검토는 소스를 바꾸지 않았다.
- 테스트 처리기: 멈춘 route는 끝에서 `route.abort('timedout').catch(…)`로 풀린다. 페이지가 이미 취소한 요청이나 닫힌 페이지에서는 catch가 오류를 삼킨다.
  보강 후 정상 빌드에서 처리기가 남거나 프로세스가 걸린 일은 없었다 (매번 18초 안팎에 끝남). 사소한 점: `waitHung`이 시간을 넘기면 `fail`이 아니라
  예외를 던진다. 그래서 그 뒤 검사는 건너뛰지만 종료 코드는 0이 아니다 (08bf4fa에서 확인).
