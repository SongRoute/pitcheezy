# /game 상태 요청 시간 제한 검토 — c35ca85 (2026-10-04)

`same_model_review; Codex_review_pending`

- 검토자: Opus 5.5. 작성자와 다른 세션이지만 같은 모델이다. 다른 모델이나 다른 Claude 프로세스는 부르지 않았다.
- 대상: `9d21dcc..c35ca85`
  - `apps/observer/web/src/Game.tsx`
  - `apps/observer/web/src/game.css`
  - `apps/observer/web/tests/game-timeout-smoke.mjs`
  - `docs/reviews/PM-game-timeout-2026-10-04.md`
- 경계 확인: API·백엔드·모델·지원 기준·추천 내용은 바뀌지 않았다.
  - 시간 초과는 기존 `/live/{pk}/state` 요청을 취소하고 실패 한 번으로 셀 뿐이다.
  - 응답 없이 추천을 만들어 보여 주는 코드는 없다. 화면에 남는 것은 마지막으로 받은 상황(흐리게 처리)과 낡은 상황 띠뿐이다.

## 결론

**쇼케이스 통합 기준 비차단(non-blocking)**

- 감시 타이머(watchdog) 로직은 확인한 모든 이벤트 순서에서 맞게 동작한다.
  - 타이머가 새지 않는다.
  - 다른 요청에 밀려 취소된 요청은 실패로 세지 않는다.
  - 루프가 멈추거나 둘로 늘지 않는다.
  - 근거는 1번 항목.
- 새 smoke는 9d21dcc와 c35ca85를 구별한다. 기존 hang·reconnect smoke도 통과한다.
- 15초는 측정값에서 정직하게 나왔다. 재계산한 수치가 작성자 보고와 일치한다. 다만 코드 주석의 "median about 0.9 s"는 합친 중앙값이 아니다(F3).
- 고치면 좋은 것: 띠의 `다시 시도` 버튼이 70×24 px라서, 이 페이지의 다른 모든 버튼이 지키는 44 px 규칙보다 작다(F1).
  - 실기기(iPhone) 확인 전에 고치기를 권한다. CSS 한 줄이다.
- 이 검토에서 바꾼 코드: `game-timeout-smoke.mjs`에 검사 한 줄(아래 "검토자 변경"). 앱 코드는 바꾸지 않았다.

## 지적 사항

### F1 — 중간·비차단 — 띠의 재시도 버튼이 44 px 탭 크기 규칙보다 작다
- 위치: `apps/observer/web/src/game.css:106`, `apps/observer/web/src/Game.tsx:372`
- 측정 (vite 빌드, Playwright, 상태 응답 후 503으로 띠를 띄움): 버튼 크기 **70×24 px**, 띠 높이 36 px.
  - 같은 값이 나온 크기: 402×681, 360×620, 360×560, 320×560.
- 기준: `tests/game-frame-smoke.mjs:38-48`은 `.gm-page` 안의 모든 버튼·링크가 44 px 미만이면 실패로 본다.
  - `gm-top`·`gm-live`·`gm-name`·`gm-dots`·`gm-bar` 모두 44 px로 맞춰져 있다.
  - frame smoke에는 "상태 있음 + 오류" 단계가 없어서 이 버튼을 재지 않는다. 그래서 작성자 보고의 `game-frame-smoke ok`와 모순되지 않는다.
- 실패 상황: 경기 중 연결이 끊긴 사람이 폰으로 높이 24 px 버튼을 눌러야 한다.
  - 띠는 `overflow:hidden`이라 버튼 바깥으로 누를 수 있는 영역을 넓힐 수도 없다.
- 제안 diff (/tmp 사본에서 측정함, 커밋 안 함):
  ```css
  -.gm-stale button{flex:none;margin-left:auto;padding:2px 10px;...}
  +.gm-stale{align-items:center}
  +.gm-stale button{flex:none;min-height:44px;margin:-4px -4px -4px auto;padding:0 12px;...}
  ```
  - `align-items` 없이 `min-height:44px`만 넣은 버전을 쟀다.
    - 띠 48 px, 버튼 74×44 px.
    - 위 네 크기 × 상태 두 개에서 stage 잘림 0, 가로 넘침 0, 페이지 스크롤 0.
    - 글자가 띠 위쪽에 붙어 보여서(`baseline` 정렬 때문) `align-items:center`를 같이 넣기를 권한다. 이 조합 자체는 재지 않았다.
  - 함께 할 일: frame smoke에 "상태 있음 + 오류" 단계를 넣는다. 그래야 44 px 검사와 stage 잘림 검사가 이 띠까지 덮는다.

### F2 — 낮음·PLAUSIBLE — 백그라운드에서 돌아올 때 "연결 끊김"이 잠깐 잘못 뜰 수 있다
- 위치: `Game.tsx:192` (watchdog), `Game.tsx:218` (`wake`)
- 상황:
  1. 요청이 오가는 중에 사용자가 중계 앱으로 넘어간다.
  2. iOS Safari나 Chrome Android가 페이지를 동결한다.
  3. 15초 넘게 지나서 돌아온다.
- 미뤄졌던 watchdog이 `visibilitychange`보다 먼저 실행되면, 그 요청은 "15초 동안 답하지 않았어요"인 실패로 세진다.
  - 그 뒤 `wake`가 새로 묻는다. 그래도 답이 올 때까지(측정 중앙값 약 0.9초) 띠와 헤더에 "연결 끊김"이 보인다.
- 9d21dcc에서는 `wake`가 그 요청을 대체해서 세지 않았다. 그래서 이 깜빡임은 이번 변경에서 새로 생길 수 있는 경로다.
- 검증 안 됨: 동결이 풀릴 때 타이머와 `visibilitychange` 중 어느 쪽이 먼저 실행되는지는 브라우저마다 다르다. Playwright로는 재현하지 못했다.
  - 참고: iOS가 동결 중에 요청을 네트워크 오류로 끊는 경우에는 같은 깜빡임이 9d21dcc에도 있었다.
- 제안 (선택): watchdog이 예정보다 한참 늦게 실행되면 페이지가 멈춰 있었던 것으로 보고 실패로 세지 않는다.
  ```ts
  -      const watchdog = window.setTimeout(() => { late = true; request.abort(); }, STATE_TIMEOUT_MS);
  +      const started = performance.now();  // a timer far past due means the page was frozen, not that the server was silent
  +      const watchdog = window.setTimeout(() => { late = performance.now() - started < STATE_TIMEOUT_MS + 5000; request.abort(); }, STATE_TIMEOUT_MS);
  ```
  - 이렇게 하면 `late`가 false인 취소는 `(e as Error).message`(AbortError 문구)로 실패로 세진다.
  - 그래서 catch에서도 `!late && request.signal.aborted && !stop`을 대체된 요청처럼 건너뛰게 해야 한다.
  - 실기기 의견을 본 뒤 정해도 된다.

### F3 — 낮음·문서 — 측정 요약 문구 두 곳이 원자료보다 강하다
- `Game.tsx:40` "13,106 answers … median about 0.9 s"
  - 13,106회를 합친 중앙값은 계산된 적이 없다.
  - 8769 인스턴스(지연 0, n=4,982)의 중앙값은 **10.4 ms**다. 그래서 합친 중앙값은 0.9초보다 낮다.
  - 출처별 중앙값: ds 3경기 906.45 ms, 8766 929.4 ms, 8769 10.4 ms.
  - 제안: `median 0.9 s on the delayed instances (10 ms at delay 0), max 4.6 s`.
- `PM-game-timeout-2026-10-04.md:16` "8769 … 오류 4 (모두 서버 재시작 시각)"
  - 요약 JSON에 시각이 남은 오류는 2건뿐이다: `fixed_code_8769.restart_errors` = 01:15:49Z, 01:15:51Z.
  - 나머지 2건이 재시작 때 난 것인지는 이 파일로 확인할 수 없다.
- `PM-game-timeout-2026-10-04.md:29` "9d21dcc 빌드에서 14건 실패": 내 실행에서는 **15건**이었다(아래 명령 기록).
  - 대조 빌드에서 실패하는 사실은 같다. 연쇄 실패라 실행마다 1건쯤 달라질 수 있다.

### F4 — 낮음·시험 — 시험이 덮지 못하는 경로
- 답의 헤더는 왔는데 본문 읽기가 멈추는 경로(`response.json()` 중 watchdog 실행)는 어떤 smoke도 실행하지 않는다.
  - `route.fulfill`로는 본문을 중간에 멈출 수 없다.
  - 코드를 읽어 맞다고 판단했다(1번 항목).
- watchdog을 지우지 않는 변이(m2)는 살아남는다. 동작이 같은 변이라서 괜찮다.
  - 끝난 요청에 `abort()`를 불러도 아무 일도 없다.
  - 남는 타이머는 2초 폴링 기준 최대 8개쯤이고, 각각 15초 뒤 사라진다.
- 숨김 페이지(10초 폴링)와 일시정지된 다시 보기에서의 watchdog도 시험하지 않는다. 코드를 읽어 확인했다.
- 버튼 위치 검사가 화면 폭(360)과 비교하고 있었다. 띠는 자기 넘침을 잘라내므로 실제 잘리는 경계는 띠의 오른쪽 끝(344 px)이다. → 검사 한 줄을 추가했다(아래).

### F5 — 참고 — `role="status"` 안의 버튼
- `Game.tsx:370-372`: 띠 전체가 라이브 영역이라 띠가 뜰 때 "다시 시도"까지 상태 문장으로 읽힌다.
  - ARIA 위반은 아니고, 버튼은 키보드·스크린리더로 도달할 수 있다.
- 다듬으려면 `role="status"`를 `<b>`·`<span>`을 감싼 안쪽 요소로 옮긴다. 낮은 우선순위다.

## 확인 항목별 답

### 1. watchdog 정확성 (코드 읽기 + smoke)
- **새 요청이 대체한 뒤**: 대체할 때 `out.abort()`가 옛 요청을 즉시 거절한다. 그러면 옛 catch에서 `mine !== ask`라 세지 않는다.
  - 옛 finally가 옛 watchdog을 지운다.
  - 그 사이 옛 watchdog이 먼저 실행돼도 문제없다. 옛 클로저의 `late`와 `request`만 건드리고(이미 취소된 요청이라 효과 없음), catch에서 여전히 세지 않는다.
  - 새 요청의 `request`는 건드릴 수 없다.
- **답이 와서 `response.json()`을 읽는 중**: watchdog의 abort로 본문 읽기가 거절된다.
  - `get()`(`WatchAlong.tsx:41`)이 `signal.aborted`를 보고 다시 던지므로 `{}`가 되지 않는다.
  - 그다음 tick의 catch에서 `late=true`이므로 시간 초과로 한 번 센다. 맞다.
  - 본문을 이미 다 읽었다면 abort는 효과가 없고 정상 경로로 간다.
- **다시 보기 t0 고정의 이른 `return`**: `try` 안의 return이라 finally가 실행되어 watchdog이 지워지고 `out=null`이 된다.
  - 다음 타이머는 걸리지 않는다. 의도대로다: `setT0`로 effect가 다시 돈다.
- **정리/언마운트(stop=true)**: 정리 함수가 `out?.abort()`를 부르므로 요청이 거절되고 finally에서 watchdog이 지워진다.
  - 그 사이 watchdog이 실행돼도 catch의 `!stop`에서 걸러진다. setState도, 새 타이머도 없다.
- **StrictMode 이중 effect**: 첫 effect의 요청은 정리 함수의 abort로 위와 같이 정리된다. 두 번째 effect는 변수를 새로 만든다.
  - 프로덕션 빌드에서는 이중 실행 자체가 없다.
- **숨김 페이지·일시정지**:
  - 숨김 상태에서는 타이머가 늦어질 수 있다. 일찍 실행될 수는 없다.
  - 돌아오면 `wake`가 대체한다(F2의 순서 예외만 있음).
  - 일시정지 중에는 effect가 요청을 만들지 않고, 이전 effect 정리가 진행 중인 요청을 취소한다.
- **루프 수**: 다음 타이머는 `mine === ask`인 최신 tick만 건다. 새로 묻는 tick은 `clearTimeout(timer)`로 시작한다.
  - 그래서 루프는 늘 하나다. 시간 초과 뒤에도 finally에서 `out=null`이 되고 타이머가 다시 걸려 루프가 멈추지 않는다.

### 2. 헤더·stage 상태
- **상태 없음**: smoke 1단계에서 확인. "연결 안 됨" 칩, 15초 문구, `지금 다시 시도`가 보이고 다음 폴링이 스스로 복구한다.
- **상태 있음**: smoke 2·3단계에서 확인.
  - 낡은 상황 띠와 헤더 "연결 끊김"이 뜨고 받은 시각이 표시된다.
  - 버튼을 누르거나 그대로 두면 LIVE로 돌아온다.
- **레이아웃**: 402/360/320 폭에서 가로 넘침·stage 잘림·페이지 스크롤 모두 0.
  - 360 px에서는 띠 문구가 "…받은 상황 · …"로 줄어든다. 320 px에서는 받은 시각의 초 단위까지 잘린다(320은 목표 크기 밖).
  - 버튼은 늘 띠 안에 보인다. 크기 문제는 F1.

### 3. 15초의 근거 — 원자료로 재계산 (읽기만 함)
- 849829·849828·849835 `*_polls.jsonl` (849830 파일은 열지 않음):

  | 경기 | 행 | http 200 | `error` 키 | 중앙값 ms | 최대 ms | >2 s | >5 s |
  |---|---|---|---|---|---|---|---|
  | 849829 | 2,024 | 2,024 | 0 | 901.7 | 2,312.1 | 4 | 0 |
  | 849828 | 1,971 | 1,971 | 0 | 894.9 | 2,338.8 | 4 | 0 |
  | 849835 | 1,819 | 1,819 | 0 | 919.1 | 2,413.4 | 9 | 0 |
  | 합계 | 5,814 | 5,814 | 0 | 906.45 | 2,413.4 | 17 | 0 |

  - 합계 p99 1,423.96 ms, p99.9 2,176.0 ms (선형 보간). 작성자 수치와 일치한다.
  - 모든 행이 `delay_s=30`이다. 조회 간격은 중앙값 5.9초, 최대 8.0초다.
    - 기록기의 20초 제한에 걸린 조회가 있었다면 간격이 26초 이상으로 벌어졌을 것이다. 그런 간격이 없으므로 "시간 초과 0"이 원자료로 뒷받침된다.
  - 기록기(`scripts/live_rehearsal.py:50-52`)는 오류가 나면 `error` 키를 남긴다. 세 파일 어디에도 이 키가 없다.
- `LIVE-rehearsal-849844.json`
  - 8766: 2,310회, 오류 0, 중앙값 929.4 ms, 최대 4,322.8 ms.
  - 8769: 4,986회, 오류 4, 응답 4,982회, 중앙값 10.4 ms, 최대 4,562.8 ms.
  - 작성자 수치와 일치한다. 오류 4건의 원인은 F3 참고.
- 합계 응답 5,814 + 2,310 + 4,982 = **13,106**, 최대 4.56초. 일치한다.
- 판단: 15초 ≈ 측정 최대의 3.3배로, 근거가 정직하다.
  - 한계도 작성자가 적은 그대로다. 모든 측정이 같은 맥에서 Tailscale 주소로 잰 것이라, 휴대폰 망에서 더해지는 시간은 미측정이다.

### 4. 시험 적합성
- 변이 시험: `/tmp` 사본에서 빌드해 `game-timeout-smoke`를 실행했다. FAIL 줄 수에서 마지막 요약 줄은 뺐다.
  - 대조 9d21dcc: FAIL 15
  - m1 (시간 초과를 실패로 세지 않음): FAIL 13 — 잡힘
  - m2 (finally에서 watchdog을 지우지 않음): 통과 — 동작이 같은 변이(F4)
  - m3 (띠 버튼 없음): FAIL 8 — 잡힘
  - m5 (watchdog이 `late`를 표시하지 않음): FAIL 1 (`no timeout message`) — 잡힘
- 시간 경계:
  - 1단계 "13초까지 취소 없음"은 제한까지 2초 여유가 있다. "취소 시각 ≥ 14.5초"는 0.5초 여유가 있다.
  - 2단계 "띠는 14초 이후"는 `t`를 요청 감지 뒤에 재므로 늘 15초보다 작게 잰다. 그래서 1초 여유가 있다.
  - 4단계 "12초에 1–3번"은 5초 답 + 2초 간격 기준으로 2번 근처다.
  - c35ca85 빌드에서 두 번(검사 보강 전·후) 실행해 둘 다 ok였다. 흔들릴 만한 경계는 보이지 않는다.
- 끝날 때 남는 라우트 핸들러:
  - 멈춘 요청은 `hung.splice(0)`로 풀어 주고, 이어지는 `route.abort`는 `.catch`로 막힌다.
  - 4단계의 마지막 느린 요청은 최대 5초짜리 `setTimeout`이 남는다. 컨텍스트가 닫힌 뒤 `route.fulfill`이 거절되어도 `.catch`로 막힌다. 프로세스는 정상 종료한다.
- 버튼 위치 검사가 띠 경계보다 느슨했다 → 보강(아래).

### 5. 회귀
- `game-hang-smoke` ok (18.4초)
- `game-reconnect-smoke` ok (21.6초)
- `game-frame-smoke`는 보호 모드라 다시 돌리지 않았다. 작성자 보고는 ok (117초).

## 검토자 변경 (각각 따로)
1. `apps/observer/web/tests/game-timeout-smoke.mjs:83-84`: 재시도 버튼의 오른쪽 끝이 띠의 오른쪽 끝을 넘으면 실패로 보는 검사를 추가했다.
   - 띠가 넘친 부분을 잘라내므로 화면 폭 검사만으로는 잘린 버튼을 놓친다.
   - 현재 빌드에서 통과한다(아래 기록). 이 검사만 따로 실패하는 변이는 만들지 않았다.

앱 코드(`Game.tsx`, `game.css`)는 바꾸지 않았다. F1·F2·F3의 diff는 제안일 뿐이다.

## 실행한 명령과 결과
`node_modules`는 메인 체크아웃을 가리키는 임시 symlink로 썼고 커밋 전에 지웠다. 레포 `dist/`는 쓰지 않았다. 포트·tmux·ssh·서버는 건드리지 않았다.

```
$ ./node_modules/.bin/tsc --noEmit -p tsconfig.json                         → exit 0
$ ./node_modules/.bin/vite build --outDir /tmp/pz-timeout-review-dist --emptyOutDir
  index-DPm4HSt2.css 100.86 kB · index-BNekkkEe.js 362.33 kB · built in 378ms   → exit 0
$ PZ_WEB_DIST=/tmp/pz-timeout-review-dist perl -e 'alarm 240; exec @ARGV' node tests/game-timeout-smoke.mjs
  game timeout smoke: ok   (1:07.98)                                            → exit 0
$ … node tests/game-hang-smoke.mjs        game hang smoke: ok       (18.41 s)   → exit 0
$ … node tests/game-reconnect-smoke.mjs   game reconnect smoke: ok  (21.55 s)   → exit 0
$ (검사 보강 뒤) … node tests/game-timeout-smoke.mjs   game timeout smoke: ok (67.91 s)
$ git archive 9d21dcc apps/observer/web → /tmp 빌드 → game-timeout-smoke
  FAIL the hung first request was not cancelled … FAIL no retry button in the stale band … (15 FAIL) → exit 1
$ 변이 m1/m2/m3/m5 (/tmp 사본, perl 한 줄 치환) → 위 4번 결과
$ 임시 측정 스크립트 (상태 → 503, 띠 버튼·띠·stage 측정, 402/360/360/320 × 상태 2개; 커밋 안 함, 삭제함)
  c35ca85:  btn 70×24, band 36 px, clipped 0, wide 0, pageScroll 0
  F1 제안:  btn 74×44, band 48 px, clipped 0, wide 0, pageScroll 0
$ python3 (읽기만): ds 3경기 polls.jsonl + LIVE-rehearsal-849844.json 재계산 → 3번 표
```
