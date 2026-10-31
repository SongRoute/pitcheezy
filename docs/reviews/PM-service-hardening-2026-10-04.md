# 영상 시연(/showcase) 오류 복구 보강 — 2026-10-04

작업자: Claude Opus 5.5 (서비스 작업자, 하위 위임 없음) · 브랜치 `codex/showcase-hardening-1004` · 기준 `f054f73`
범위: `apps/observer/web/src/Showcase.tsx`, `src/WatchAlong.tsx`, `tests/showcase-smoke.mjs`. 서버·모델·연구·configs·2026 산출물은 바꾸지 않았다.

## 바뀐 것 (재현 가능한 실패 3건)

1. **영상 불러오기 실패 뒤 복구 수단 없음** (`Showcase.tsx`)
   - 증상: 영상 요청이 한 번 실패하면 "영상 재생을 확인하지 못했어요" 경고가 뜨고 "이 공 다시 보기"가 계속 비활성. "불러오기" 버튼도 사라져, 일시적 네트워크(Tailscale) 끊김이어도 페이지 새로고침 말고는 방법이 없었다.
   - 수정: 경고 안에 "영상 다시 불러오기" 버튼. 오류·준비 상태를 지우고 `video.load()`로 같은 주소를 다시 요청한다. 메타데이터가 오면 기존 `onLoadedMetadata`가 선택 장면 시작점(release−3초)으로 다시 맞춘다.
2. **실제 투구 공개 실패 오류가 성공 뒤에도 남고, 화면 읽기 프로그램에 알려지지 않음** (`WatchAlong.tsx`)
   - 증상: 임베드 화면에서 공개 요청이 실패하면 오류 문구가 뜨는데, 다시 눌러 성공해도 문구가 그대로 남았다(오류는 공 이동 `go()`에서만 지워지고, 임베드에서는 이동이 막혀 있음). 오류 문단에 `role="alert"`가 없어 보조기기에 알려지지 않았다.
   - 수정: 공개 시도 시작 때 이전 오류를 지운다. `WatchAlong.tsx`의 `wa-error` 문단 4곳에 `role="alert"`. (`Live.tsx`는 주기적으로 다시 시도하며 같은 오류를 반복 렌더하므로 일부러 제외.)
3. **경기 자료(타임라인) 실패 뒤 복구 수단 없음** (`Showcase.tsx`)
   - 증상: `/api/watch/849843` 실패 시 비교 영역에 오류만 표시되고 다시 시도할 방법이 없었다.
   - 수정: 오류 옆 "다시 시도" 버튼이 같은 요청을 다시 보낸다(`attempt` 상태를 effect 의존성으로). 성공하면 선택 장면의 투구로 비교 iframe이 새로 열린다.

세 경우 모두 장면 선택(`?scene=`)·투구 식별(`#i=`)은 유지되고, 추천은 공개 전 상태로 남는다(테스트가 단언).

## 테스트 (`tests/showcase-smoke.mjs`, 1440·390 폭 각각)

- 영상 실패 → 요청 복구 → "영상 다시 불러오기" → 다시 보기 활성, 경고 사라짐, 장면 0 시작점에서 정지, iframe `#i=2` 유지, 공개 없음.
- 공개 503 → iframe 안 `role=alert` "합성 공개 실패", 실제 투구 표시 없음 → 복구 후 다시 공개 → 합성 실제 투구 2 표시, 공개 오류 사라짐.
- 경기 자료 503 → 복구 → "다시 시도" → iframe `/watch?game=849843&embed=1#i=2`, 추천 없음 문구 표시, 공개 없음, 바깥 경고 사라짐.
- 수정 전 실패 확인: 기준 코드에서 ① "영상 다시 불러오기" 버튼 없음으로 실패. `WatchAlong.tsx`만 기준으로 되돌리면 ② `role=alert` 없음으로 실패, 오류 지우기만 빼면 `successful retry clears the reveal error`로 실패.

## 확인한 결과 (증거)

- `npm run build` 통과(tsc + vite 7.3.6). 의존성은 `codex-showcase-1006`의 node_modules를 심볼릭 링크로 읽기만 함(커밋 안 함). dist는 이 워크트리 것만 빌드.
- `node tests/showcase-smoke.mjs` → `showcase smoke: ok …` 연속 3회 통과(각 약 3.7초). Playwright 1.63.0, Chromium 헤드리스, Node 25.9.0.
- `node tests/watch-summary-panel.mjs` → `PASS watch-summary-panel (9 mocked requests)`.
- `watch-along-smoke.mjs` 등 실서버(8766) 대상 테스트는 실행하지 않음(운영 포트 사용 금지, 이 변경의 빌드를 검사하지도 않음).

## 한계 (미검증)

- 영상은 기존과 같이 합성 무음 WAV로 대체. 실제 MLB 영상·Safari·iPhone에서 `load()` 재요청 동작은 **미검증**.
- "불러오는 중"에서 오류 이벤트 없이 멈추는(stalled) 경우는 다루지 않았다. 그때도 "공식 영상 따로 열기"는 오류 경고에서만 보인다.
- 임베드 화면에서 경기 자료 자체가 실패하면 iframe 안에 "경기 목록으로" 링크가 보여 iframe이 목록으로 이동할 수 있다(장면 잠금 이탈). 이번에는 고치지 않음 — 바깥 "다시 시도"는 바깥 타임라인 실패에만 해당.
- 실기기 키보드·보조기기 낭독은 **미검증**(`role=alert` 존재만 확인).
