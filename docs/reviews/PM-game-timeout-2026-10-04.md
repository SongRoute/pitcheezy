# /game 상태 요청 시간 제한과 경기 중 다시 시도 — 2026-10-04

상태: self-tested; same_model_review 대기; Codex_review_pending. 통합·배포 안 함 (8766/8770/8771, pitcheezy-demo, 녹화 창 손대지 않음).
기준: `codex/game-hang-1004` `9d21dcc` 위 (`08bf4fa` → 쇼케이스 `840eeee`). `PM-game-hang-2026-10-04.md`의 한계 1·2와 검토 F5를 다룬다.

## 근거: 실제 `/state` 응답 시간 (읽기만 함, 새 부하 없음)
기록기 `scripts/live_rehearsal.py record`가 휴대폰처럼 `/api/live/{game}/state`를 묻고 매 조회의 `ms`를 남긴다 (Tailscale 주소 `100.108.252.111:8766`, 같은 맥, 기록기 자체 제한 20초).
- 849829·849828·849835 (디비전시리즈, `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/DEMO-WS-2026/live_rehearsal/ds/*_polls.jsonl`, 10/4 10:4x KST에 계산): 조회 5,814회, 오류 0. 중앙값 906.5 ms, p99 1,424 ms, p99.9 2,176 ms, 최대 2,413 ms. 2초 넘은 조회 17회, 5초 넘은 조회 0회.
  - 경기별 최대: 849829 2,312 ms (2,024회), 849828 2,339 ms (1,971회), 849835 2,413 ms (1,819회).
- 849844 (10/2, `pitcheezy-demo/results/checks/LIVE-rehearsal-849844.json`): 8766 2,310회 오류 0, 중앙값 929 ms, 최대 4,323 ms. 8769 4,986회 중 오류 4 (모두 서버 재시작 시각), 최대 4,563 ms.
- 합계: 응답 13,106회 중 시간 초과 0회, 최대 4.56초.
- 849830은 기록 중이라 넣지 않았다.
- 미측정: 휴대폰 망(LTE·와이파이·Tailscale 중계)에서 더해지는 시간. 실제 iPhone 측정은 아직 없다.

## 바꾼 것 (`apps/observer/web/src/Game.tsx`, `game.css`)
- `STATE_TIMEOUT_MS = 15000`: 위 최대 4.56초의 약 3배. 15초 안에 답이 없으면 요청을 취소하고 실패 한 번으로 센다. 화면에는 "관전 서버가 15초 동안 답하지 않았어요."가 나오고, 2초 뒤 평소처럼 다시 묻는다.
  - 새 요청(다시 시도·online·화면 복귀)에 밀려 취소된 요청은 전처럼 실패로 세지 않는다. 최신 요청만 센다.
- 경기 중: 시간 초과가 나면 기존 낡은 상황 띠가 뜬다 ("연결 끊김 · HH:MM:SS에 받은 상황 · 다시 연결 중", 헤더도 "연결 끊김"). 띠 안에 **다시 시도** 버튼을 새로 넣었다 (검토 F5). 
- 바꾸지 않은 것: API, 백엔드, 모델, 지원 게이트, 추천 내용, 폴링 간격(2초, 숨김 10초).

## 검사 (맥, 가벼운 작업만)
- `tsc --noEmit -p tsconfig.json`: exit 0.
- `vite build --outDir /tmp/pz-timeout-1004-dist --emptyOutDir`: 성공. 레포 `dist/`는 건드리지 않았다.
- 새 `tests/game-timeout-smoke.mjs` (360×560, 미리 정한 응답, 서버 없음, 실제 15초 대기): `game timeout smoke: ok` (68초).
  1. 첫 요청이 답하지 않고 아무것도 누르지 않음 → 13초까지 취소 없음·요청 쌓임 없음 → 약 15초에 취소, "연결 안 됨" + 15초 문구 → 다음 폴링이 스스로 경기 표시.
  2. 경기 중 요청 멈춤 → 15초 전에는 띠 없음, 뒤에는 낡은 상황 띠·헤더 "연결 끊김"·받은 시각 표시, 띠의 다시 시도 버튼이 화면 안(360px)에 있고 누르면 바로 다시 물어 LIVE로 돌아옴.
  3. 다시 멈춤, 아무것도 누르지 않음 → 시간 초과 뒤 다음 폴링이 스스로 복구.
  4. 5초 걸리는 느린 답(측정 최대보다 김) → 취소 없음, 띠 없음, 12초에 요청 1–3번.
- 음성 대조: `9d21dcc` 빌드에서 새 시험이 14건 실패 (`the hung first request was not cancelled`, `no retry button in the stale band` 등). 시험의 모든 대기에 시간 상한이 있다 (처음 버전은 상한 없는 대기 때문에 대조 빌드에서 멈춰 중단했고, 상한을 넣어 다시 돌렸다).
- 회귀 (새 빌드): `game-hang-smoke` ok (19초), `game-reconnect-smoke` ok (21초), `game-frame-smoke` ok (117초).
- `node_modules`는 메인 체크아웃을 가리키는 임시 symlink를 썼고 삭제했다.
- 스크린샷: `/tmp/pz-timeout-shots/timeout-1-first.png`, `timeout-2-stale.png` (커밋 안 함).

## 한계
- 15초는 맥 안에서 잰 응답 시간에 여유를 둔 값이다. 휴대폰 망 시간은 미측정이라, 실제 iPhone에서 느린 망이면 너무 짧거나 길 수 있다.
- 0–15초 사이 멈춘 요청은 여전히 "LIVE"로 보인다 (측정상 정상 응답의 99.9%가 2.2초 안).
- 실제 iPhone Safari 확인은 Song의 기기 작업으로 남는다.
