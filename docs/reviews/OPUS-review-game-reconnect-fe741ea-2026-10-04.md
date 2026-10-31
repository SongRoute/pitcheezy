# /game 재연결 검토 — fe741ea (2026-10-04)

`same_model_review; Codex_review_pending`

- 검토자: Opus 5.5. 작성자와 다른 세션이지만 같은 모델이다. 다른 모델은 호출하지 않았다.
- 대상: `840eeee..fe741ea`
  - `apps/observer/web/src/Game.tsx`
  - `apps/observer/web/tests/game-reconnect-smoke.mjs`
  - `docs/reviews/PM-game-reconnect-2026-10-04.md`
- 경계 확인: API·백엔드·모델·지원 기준·추천 내용 변경 없음. 바뀐 파일은 `Game.tsx` 한 파일과 테스트·문서뿐이다.
  - 새 조회 경로는 기존 `/live/{pk}/state` 요청을 앞당겨 보낼 뿐이다. 응답이 없을 때 추천을 만들어 보여 주는 코드는 없다.
  - 지원하지 않는 투수 화면(840eeee의 사유·마지막으로 받은 공)도 이 diff에서 바뀌지 않았다. `game-frame-smoke` unsupported 단계에서 확인했다. 결과는 아래 표.

## 결론

**쇼케이스 통합 기준 비차단(non-blocking)**

- 조회 루프는 어떤 순서로 이벤트가 겹쳐도 루프가 둘로 늘거나, 타이머가 새거나, 언마운트 뒤 setState가 일어나지 않는다. 자세한 근거는 1번 항목.
- 다만 요청이 응답 없이 멈추는 경우(F1)에는 재시도 버튼과 "2초마다" 문구가 사실과 달라진다.
  - 서버나 Tailscale 경로가 SYN에 응답하지 않으면 이런 일이 생긴다.
  - 같은 hang은 840eeee에도 있었다. 이번 변경은 이를 사용자 눈에 드러냈을 뿐 새로 만들지 않았다.
  - 고치려면 요청 제한 시간이 필요하다. 그런데 `/state` 응답 시간이 미측정이라 이번에는 코드를 바꾸지 않았다.
- 이 검토에서 바꾼 코드는 없다. 이 문서만 추가했다.

## 확인 항목별 답

### 1. 조회 루프

| 겹치는 상황 | 결과 |
|---|---|
| 요청 중에 깨우기(visibilitychange·online·버튼) | `busy`라서 무시된다. 진행 중인 요청이 끝나면 `finally`가 `busy=false`로 풀고, 그 시점의 `document.hidden`으로 다음 간격(2초/10초)을 정한다. 그래서 다시 보인 상태에서 깨우기가 무시돼도 다음 조회는 10초가 아니라 2초 뒤다. 요청은 하나만 나가고 루프도 하나다. |
| 타이머 대기 중에 깨우기 | `tick` 시작 때 `clearTimeout(timer)`로 대기 중 타이머를 지운다. `busy=false`부터 새 타이머 등록까지는 같은 동기 구간(await 없음)이라 사이에 끼어들 수 없다. 타이머는 언제나 최대 1개다. |
| 빠른 숨김/보임 반복 | 보임 이벤트마다 `tick`이 불리지만 `busy`로 하나만 나간다. 스모크 5단계에서 깨우기 6회 뒤 6.1초 동안 3회 조회를 확인했다(루프 1개). |
| 장면 이동·언마운트·의존값 변경 중 요청 | cleanup이 `stop=true`, `clearTimeout`, 리스너 제거, `pollNow` 무효화를 한다. 끝난 옛 요청은 `if (stop) return`과 `catch`의 `!stop` 가드 때문에 setState를 하지 않고 다음 타이머도 걸지 않는다. |
| StrictMode 이중 effect (`main.tsx:13`이 StrictMode) | 개발 모드에서만 첫 요청이 2번 나간다. 첫 클로저는 `stop=true`라 결과를 버리고 다시 예약하지 않는다. 운영 빌드에서는 해당 없음. |
| `busy` 해제 | `finally`라서 예외가 나도 풀린다. fetch가 중단(abort)되어도 마찬가지다. `t0` 분기의 `return`도 `finally`를 거친다. 이 분기는 예약을 건너뛰지만 `setT0`로 effect가 다시 돌기 때문에 기존과 같다. |
| 루프가 영원히 멈추는 경우 | **요청이 끝나지 않을 때만** 멈춘다. F1 참고. |

### 2. 머리말·본문 상태
- 머리말 `연결 안 됨`: `state === null && error !== null`일 때만 나온다. 첫 응답 뒤 실패하면 기존대로 `stale`가 우선해 `연결 끊김`/`기록 멈춤`이 나온다.
  - 클래스는 `offline ? 'stale'`이고 `offline`은 `stale`을 포함하므로 기존 stale 표시는 그대로다.
  - `game-demo-regression`(stale 띠·머리말·복구) 결과는 아래 표.
- 실패 횟수: 응답이 오면 `setFailures(0)`, 장면 이동 때도 0이 된다.
  - 실패 횟수는 `!state`인 본문에서만 쓰인다. 그래서 첫 응답 뒤 쌓이는 횟수는 다음 리셋 전까지 화면에 나오지 않고 해도 없다.
- 첫 응답 전 **로딩 중**(오류 없음)에는 여전히 `LIVE`가 뜬다. F2 참고.

### 3. 접근성·모바일
- 360×560: 버튼 높이 44px이고 화면 안에 들어오며 스크롤도 없다. 스크린샷으로 직접 확인했다. 버튼은 `.gm-wide`(min-height 44px, 전체 폭)이고 오류 문장 바로 아래에 있어 하단 3버튼과 겹치지 않는다.
- 더 큰 화면: `game-frame-smoke`가 3개 크기에서 통과했다. 다만 실패 상태 화면은 새 스모크가 360×560에서만 본다.
- 문구 "2초마다 자동으로 다시 연결하고 있어요"는 실패가 바로 돌아올 때만 맞다. 연결 거부나 DNS 실패가 그런 경우다. 요청이 멈추면 틀린 문구가 된다(F1). 숨긴 페이지는 10초 간격이지만 사용자가 볼 수 없으니 문제없다.
- 실패할 때마다 `role="status"` 문장이 다시 읽힌다(F3).

### 4. 테스트
- 음성 대조를 직접 다시 확인했다.
  - 840eeee 빌드: `FAIL header says LIVE…`와 `실패 2번` 대기 시간 초과로 실패했다.
  - 버튼 `onClick`을 아무 일도 안 하게 바꾼 빌드: 4회 모두 2단계 `다음 공` 1500ms 대기에서 실패했다. 스크린샷 인자를 준 2회도 실패했다.
  - 따라서 테스트는 새 동작과 옛 동작을 구분한다.
- 2단계가 구분되는 이유는 위상 고정이다. 클릭이 "실패 2번" 표시 직후, 즉 다음 자동 조회까지 약 1.6~1.8초 남은 시점에 일어난다(추정, 실측 아님). 이 여유가 1500ms 제한보다 조금 커서 구분된다. 여유는 얇다(F4).
- 600ms·700ms 제한: 고정 응답은 Playwright route 왕복 수 ms라 여유가 크다. 정상 빌드로는 1회만 돌렸고 통과했다. 반복 실행으로 흔들림을 재지는 않았다.
- 빠진 경우
  - 요청 중 hang(F1)
  - 로딩 중 머리말(F2)
  - 요청 중 장면 이동
  - 실패 후 첫 응답이 `unsupported`일 때 안내·버튼이 사라지는지

## 지적 사항

### F1 — 중간: 응답 없는 요청이 루프·버튼·문구를 모두 멈춘다
- 위치: `apps/observer/web/src/Game.tsx:182-185`(`busy` 가드 + 제한 시간 없는 `get`), `:314`(문구), `WatchAlong.tsx:37-39`(`fetch`에 signal·timeout 없음)
- 상황
  - 첫 조회는 바로 실패해 버튼이 나온다.
  - 2초 뒤 재시도는 응답이 오지 않는다. 폰에서 Tailscale이 꺼져 100.x 주소의 SYN이 버려지거나, 서버가 멈춘 경우다.
  - 이때 `busy=true`가 TCP 시간 초과(수십 초, 기기마다 다름, 미측정)까지 유지된다.
  - 그동안 버튼 탭, `online`, 복귀 즉시 조회가 모두 소리 없이 무시되고 다음 타이머도 걸리지 않는다.
  - 화면은 계속 "2초마다 자동으로 다시 연결하고 있어요"라고 말한다.
- 재현: 리뷰어 임시 검사 `/tmp/pz-rr-mut/.../tests/hang-probe.mjs`, 커밋 안 함. 첫 요청은 abort, 이후 요청은 응답하지 않게 했다.
  ```
  fail-then-hang: asks before taps=2, asks in 6074 ms after 3 taps=0, note="2초마다 자동으로 다시 연결하고 있어요"
  ```
- 같은 hang은 840eeee에도 있었다. 이번 변경으로 생긴 것은 "버튼과 문구가 동작을 약속하는데 지키지 못함"이다.
  - iOS에서 백그라운드 전에 보낸 요청이 복귀 뒤까지 매달려 있는 경우도 같은 경로를 탄다. 실기기 미측정.
- 제안: `/state` 응답 시간 p99를 잰 뒤 그보다 넉넉한 제한 시간을 정한다(값 미정). 그 다음 아래처럼 바꾼다. 깨우기가 매달린 요청을 끊고 새로 묻게 하는 것은 선택이다.
  ```diff
  // WatchAlong.tsx
  -export async function get<T>(path: string): Promise<T> {
  +export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
     let response: Response;
  -  try { response = await fetch(`/api${path}`, { cache: 'no-store' }); } catch { throw new Error('관전 서버에 연결하지 못했어요. …'); }
  +  try { response = await fetch(`/api${path}`, { cache: 'no-store', signal }); } catch { throw new Error('관전 서버에 연결하지 못했어요. …'); }
  // Game.tsx tick
  -      try {
  -        const next = await get<LiveState>(`/live/${gamePk}/state?${query}`);
  +      const ac = new AbortController(), cut = window.setTimeout(() => ac.abort(), STATE_TIMEOUT_MS);  // configs/측정값으로 정할 것
  +      try {
  +        const next = await get<LiveState>(`/live/${gamePk}/state?${query}`, ac.signal);
  ...
  -      finally { busy = false; }
  +      finally { busy = false; window.clearTimeout(cut); }
  ```
- 측정 전 임시 대안: 문구를 "자동으로 계속 다시 연결하고 있어요"로 바꿔 숫자 약속을 뺀다. 요청 중에는 버튼을 `disabled` + "연결 시도 중…"으로 보여 탭이 무시되는 것을 감춘다.
- 이번에 고치지 않은 이유: `/state`는 `armb.lookup`을 동기로 부를 수 있다(`backend/observer_app/main.py:361-363`). 응답 시간 분포가 미측정이라, 제한 시간을 잘못 정하면 느리지만 정상인 응답이 실패로 바뀐다. 쇼케이스 직전에 넣기에는 "명백히 올바른" 수정이 아니다.

### F2 — 낮음: 첫 응답 전 로딩 중에는 여전히 `LIVE`(빨간 점)
- 위치: `Game.tsx:345-346`. 주석은 "never say LIVE before the first answer has arrived"라고 하지만, `state === null && error === null`(불러오는 중)이면 `LIVE`가 뜬다.
- 재현(hang-probe `hang-first`, 첫 요청 무응답 3초):
  ```
  hang-first: header = "LIVE" stage chip = "불러오는 중" asks = 1
  ```
  보통은 1초 안에 지나가지만, F1처럼 요청이 매달리면 수십 초 동안 `LIVE`가 뜬다.
- 제안:
  ```diff
  -  const liveLabel = stale ? (…) : offline ? '연결 안 됨' : paused !== null ? … : isReplay ? … : 'LIVE';
  +  const liveLabel = stale ? (…) : offline ? '연결 안 됨' : state === null ? '연결 중' : paused !== null ? … : isReplay ? … : 'LIVE';
  ```
  클래스는 `state === null`일 때 `off`(회색 점)로 한다. 주석을 실제 동작에 맞게 고치는 것만으로도 된다.

### F3 — 낮음(접근성): 실패할 때마다 화면 낭독기가 문장을 다시 읽는다
- 위치: `Game.tsx:314`. `role="status"`(polite live region) 안에 `실패 N번`이 있어 2초마다 텍스트가 바뀐다. VoiceOver 사용자는 2초마다 "2초마다 자동으로 다시 연결하고 있어요, 실패 3번…"을 듣는다.
- 제안: live region에는 바뀌지 않는 문장만 둔다. 횟수는 live region 밖의 별도 `<span aria-hidden>`이나 일반 텍스트로 뺀다. `role="status"`를 칩 `연결 안 됨`으로 옮겨도 된다.

### F4 — 낮음(테스트): 2단계 구분력이 타이밍 위상에 기대고 있다
- 위치: `tests/game-reconnect-smoke.mjs:48-61`
- 버튼이 동작하지 않아도 다음 자동 조회가 1500ms 안에 오면 `다음 공`이 뜨고, 700ms 검사도 우연히 통과할 수 있다.
- 지금은 "실패 2번" 직후 클릭이라 다음 자동 조회까지 약 1.6~1.8초가 남아 구분된다. no-op 빌드 6/6 실패로 확인했다. 그런데 앞에 대기를 추가하거나 기계가 느리면 이 여유가 사라진다.
- 제안: 클릭 직전 마지막 조회 시각 `last = asks.at(-1)`을 잡고, 클릭 뒤 첫 조회가 `last + 1500` 전에 왔는지 확인한다. 즉 자동 주기로는 설명할 수 없는 시각인지 본다. 또는 클릭 전 `asks.length`를 기록하고 300ms 안에 늘었는지 본다.

### F5 — 정보: 숨김 → 보임 복귀 때 이전에 보낸 요청의 응답
- iOS Safari는 백그라운드에서 JS를 얼리고 연결을 끊는 경우가 많다. 그러면 복귀 직후 진행 중이던 요청이 오류로 끝나 `연결 끊김` 띠가 잠깐 보였다가 2초 뒤 복구될 수 있다.
- 이 경우 `busy` 때문에 복귀 즉시 조회는 무시된다. 실기기 미측정이라 기록만 남긴다. F1의 "깨우기가 매달린 요청을 끊고 새로 묻기"가 같이 해결한다.

## 실행한 명령과 결과

모두 워크트리 `apps/observer/web`에서 실행했다. `node_modules`는 원본 레포를 가리키는 임시 symlink를 썼고 커밋 전에 지웠다.
- 레포 `dist/`는 쓰지 않았다.
- 음성 대조 빌드는 `/tmp/pz-rr-old`(840eeee)와 `/tmp/pz-rr-mut`(fe741ea, 버튼 no-op 변형)에 `git archive`로 풀어서 만들었다.

| 명령 | 결과 |
|---|---|
| `./node_modules/.bin/tsc --noEmit -p tsconfig.json` | exit 0 |
| `./node_modules/.bin/vite build --outDir /tmp/pz-reconnect-review-dist --emptyOutDir` | `✓ built in 360ms`, exit 0 |
| `PZ_WEB_DIST=/tmp/pz-reconnect-review-dist node tests/game-reconnect-smoke.mjs /tmp/pz-rr-shots` | `game reconnect smoke: ok` (20.3 s) |
| 같은 스모크, 840eeee 빌드 `/tmp/pz-rr-old-dist` (2회) | `FAIL header says LIVE before any answer arrived` → `getByText('실패 2번')` 6000ms 초과 |
| 같은 스모크, 버튼 no-op 빌드 `/tmp/pz-rr-noop-dist` (4회 + 스크린샷 인자 2회) | 6/6 `locator.waitFor: Timeout 1500ms exceeded … '다음 공'` |
| 리뷰어 임시 `hang-probe.mjs fail-then-hang` (fe741ea 빌드) | `asks in 6074 ms after 3 taps=0` |
| 리뷰어 임시 `hang-probe.mjs hang-first` (fe741ea 빌드) | `header = "LIVE" stage chip = "불러오는 중"` |
| `PZ_WEB_DIST=/tmp/pz-reconnect-review-dist node tests/game-frame-smoke.mjs` | `game frame smoke: ok` (3개 화면 크기, 1:56) |
| `game-demo-regression.mjs` (dist 줄만 `PZ_WEB_DIST`로 바꾼 `/tmp` 사본) | `game demo regression: ok` (1:08) |

## 이 검토에서 바꾼 것
- 코드 변경 없음. 이 문서만 추가했다.
- 리뷰어 임시 검사(`hang-probe.mjs`)는 F1이 고쳐지기 전에는 실패하는 검사라 커밋하지 않았다. F1을 고칠 때 `game-reconnect-smoke.mjs`에 "첫 실패 뒤 요청 무응답 → 제한 시간 뒤 다시 조회" 단계로 옮기길 권한다.
