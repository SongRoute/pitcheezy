# /game 멈춘 요청 복구 — 검토 처리 (2026-10-04)

상태: same_model_reviewed; Codex_review_pending. 통합·배포 안 함.
검토: 다른 Opus 5.5 세션, `5c3112d` (`codex/game-hang-review-1004`). 결론은 통합을 막지 않음.

## 발견별 처리
- F1 (테스트가 실제 취소를 확인하지 않음): 받아들임. 검토자의 보강 smoke를 cherry-pick했다 (`0a2229c`). `requestfailed`의 `ERR_ABORTED` 수를 센다.
- F2 (새로 묻기 두 번 연속 없음, 실패 횟수 확인이 타이밍에 기댐): 받아들임. 같은 cherry-pick에 4번 절과 400ms 대기가 들어 있다.
- F3 (본문을 읽다 취소되면 `get()`이 `{}`를 돌려줌): 고쳤다. `WatchAlong.tsx`에서 `signal`이 취소됐으면 `json()` 오류를 그대로 던진다. `signal`을 넘기지 않는 호출부는 그대로다. 지금은 Game의 가드 때문에 해가 없었다. 나중에 시간 제한을 넣을 때 화면이 빈 상태로 바뀌는 것을 미리 막는다. 이 경로는 지금 브라우저 smoke로 재현할 수 없어서 테스트를 더하지 않았다.
- F4 (타이머 가드 변이는 살아남음): 정보. 방어용 가드로 둔다.
- F5 (경기 중에는 화면 안 복구 방법이 없음): 보고서 한계에 한 줄 더했다. 시간 제한은 `/state` 응답 시간을 잰 뒤에 넣는다 (미측정).
- F6 (서버는 취소된 요청도 끝까지 계산): 정보. 새로 묻기는 사용자 동작이나 이벤트에서만 일어난다.

## 다시 한 검사 (맥, 가벼운 작업)
- `tsc --noEmit -p tsconfig.json`: exit 0.
- `vite build --outDir /tmp/pz-hang-1004-r2-dist --emptyOutDir`: 성공.
- 보강 `game-hang-smoke` (r2 빌드): ok 2회. `game-reconnect-smoke` (r2 빌드): ok.
- 음성 대조: 같은 보강 smoke를 `08bf4fa` 빌드에 돌리면 FAIL 7건.
- game-frame-smoke·game-demo-regression은 다시 돌리지 않았다. 바뀐 것이 `get()`의 본문 오류 처리 한 줄뿐이고, `signal`이 없는 호출에서는 동작이 같다. `7ebfc94`에서는 둘 다 통과했다.
- `node_modules`는 임시 symlink를 썼고 삭제했다.
