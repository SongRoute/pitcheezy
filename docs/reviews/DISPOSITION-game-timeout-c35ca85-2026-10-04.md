# 처리 — /game 시간 제한 검토 (c35ca85) — 2026-10-04

상태: same_model_reviewed; Codex_review_pending. 통합·배포 안 함.
검토: `b8c650b` (`codex/game-timeout-review-1004`, `OPUS-review-game-timeout-c35ca85-2026-10-04.md`), 이 브랜치에 cherry-pick `f9294aa`. 결론 비차단.

- F1 (중간) 띠 재시도 버튼 70×24 px → 고침. `game.css`: `min-height:44px`, 음수 margin으로 띠 높이 증가 최소화, 버튼 있는 띠만 `align-items:center` (`:has`). `game-timeout-smoke`에 44 px 검사 추가. 360×560 스크린샷에서 버튼 44 px 높이, 띠 안, stage 잘림 없음 (`/tmp/pz-timeout-shots/timeout-2-stale.png`, 커밋 안 함). frame smoke에 "상태 있음 + 오류" 단계 추가는 하지 않음 (다음 작업 후보).
- F2 (낮음, PLAUSIBLE) 동결된 페이지에서 돌아올 때 "연결 끊김"이 잠깐 뜰 수 있음 → 이번에는 고치지 않음. 브라우저별 순서가 재현되지 않았고 실제 iPhone 확인이 필요하다. 한계로 기록.
- F3 (낮음, 문서) 중앙값·오류 4건·대조 실패 수 문구 → 고침 (`Game.tsx` 주석, `PM-game-timeout-2026-10-04.md`).
- F4 (시험 공백) 본문 읽는 중 시간 초과 경로는 코드 읽기로만 확인 → 한계로 둠. 검토자의 띠 경계 검사(cherry-pick)는 채택.
- F5 (참고) `role="status"` 안의 버튼 → 바꾸지 않음.

검사 (고친 뒤): `tsc --noEmit` exit 0; `vite build --outDir /tmp/pz-timeout-1004-dist` 성공; `game-timeout-smoke` ok (68초); `game-hang-smoke` ok (18초). 임시 node_modules symlink 삭제.
