# /game 멈춘 요청 복구 (리뷰 F1 남은 부분) — 2026-10-04

상태: self-tested; same_model_review_not_run; Codex_review_pending. 통합·배포 안 함 (8766/8770/8771 손대지 않음).
기준: `codex/game-reconnect-1004` `08bf4fa` (쇼케이스 `840eeee` 위). 같은 모델 검토 `489cd12`의 F1(답이 안 오는 요청이 `busy`를 잡고 있어 다시 시도·online·화면 복귀가 모두 아무 일도 안 함)을 다룬다.

## 바꾼 것
- `src/WatchAlong.tsx` `get(path, signal?)`: 선택 인자 `signal`만 추가. 취소된 요청은 원래 오류를 그대로 던지고, 나머지 호출부는 동작이 같다.
- `src/Game.tsx` 폴링: `busy` 대신 요청 번호(`ask`)와 `AbortController`.
  - 타이머 폴링은 전처럼 이미 나간 요청을 기다린다 (요청이 겹치지 않음).
  - 사용자가 누른 다시 시도, `online`, 화면 복귀는 나간 요청을 취소하고 새로 묻는다.
  - 취소된 요청은 화면을 바꾸지 않고, 실패 횟수에 들어가지 않고, 다음 폴링 타이머도 걸지 않는다 (번호가 최신인 요청만).
  - 화면을 떠날 때(effect 정리) 나간 요청을 취소한다.
- 바꾸지 않은 것: API·백엔드·모델·지원 게이트·추천 내용. 요청 시간 제한(watchdog)은 넣지 않았다. `/state` 응답 시간이 미측정이라 기준값을 정할 근거가 없다.

## 검사 (맥, 가벼운 작업만)
- `tsc --noEmit -p tsconfig.json`: exit 0.
- `vite build --outDir /tmp/pz-hang-1004-dist --emptyOutDir`: 성공. 레포 `dist/`는 건드리지 않았다.
- 새 `tests/game-hang-smoke.mjs` (360×560, 미리 정한 응답, 서버 없음): `game hang smoke: ok` (17.2초).
  1. 첫 요청이 답하지 않음 → 2.5초 동안 요청이 쌓이지 않음 → `online`에서 새로 묻고 경기 표시.
  2. 첫 연결 실패 뒤 자동 폴링이 멈춤 → 다시 시도가 바로 묻고, 취소된 요청은 실패로 세지 않음 (2번 → 3번) → 성공하면 경기 표시.
  3. 경기 중 폴링이 멈춘 채 화면을 떠났다 돌아옴 → 바로 새로 묻고, 그 뒤 6초에 2–4번(폴링 고리 하나).
- 음성 대조:
  - `08bf4fa` 빌드에서는 1·2번이 실패한다 (`online did not replace the hung first request`, `retry did nothing while a request was hung` 등 5건). 3번에서는 요청을 기다리다 시간 초과로 멈춘다.
  - 변이 `wake`(화면 복귀·online이 새로 묻지 않음): 4건 실패.
  - 변이 `count`(취소도 실패로 셈): `the dropped hung request was counted as a failure` 실패.
- 회귀: `game-reconnect-smoke` ok, `game-frame-smoke` ok (3크기, 1:57), `game-demo-regression` ok (dist 경로만 env로 바꾼 임시 사본, 실행 뒤 삭제, 1:08).
- `node_modules`는 메인 체크아웃을 가리키는 임시 symlink를 썼고 삭제했다.

## 한계
- 아무 동작 없이 기다리기만 하면 멈춘 요청은 여전히 풀리지 않는다. 시간 제한은 `/state` 응답 시간을 잰 뒤 설정값으로 넣어야 한다 (미측정).
- 경기 중 요청이 멈추면 `error`가 없어서 오래된 화면이 '지금'처럼 보일 수 있다. 위 시간 제한과 같이 다뤄야 한다.
- 실제 iPhone Safari에서 백그라운드 복귀, Tailscale 끊김은 미측정.
