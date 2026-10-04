# 처분 — 모델 카드 섞기 문장 리뷰 (c84edc1 of 3a3690d)

상태: `same_model_reviewed; Codex_review_pending`. 리뷰어는 다른 Opus 세션(helper 45498, exit 0, 권한 거절 0).

## 처분
- F2 (리뷰어 수정, 서비스 경로 고정 `blend.servicePath`·smoke 검사): 수락, ff 3a3690d → c84edc1.
- F1 (6월 자료는 개발 패널 일부 투수): 수락. 문장에 "일부 투수들의 4,821구"를 `blend.servicePath.fitSample.pitches`(smoke가 P10 `coverage.blend.eligible_pitches`와 대조)로 표시.
- F3 (비율 읽기): 수락. "신경망 5개의 평균(약 75%)과 상황 빈도 기준(약 25%)을 섞은 값". 75·25 모두 `modelWeight`에서 계산.
- F4 둘째 항목 (시드별 점수의 정체): 수락. "시드 5/5개" → "시드별 신경망 하나씩도 5/5개". 첫째 항목(5월 보정 생략)은 그대로 둠.
- F5: 정보, 변경 없음 (시연은 LEGACY=0).

## 확인 (`apps/observer/web`, OMP/UV threads 1, alarm 300)
- `npx tsc -b` → 0
- `npx vite build --outDir dist-t46b --emptyOutDir` → 0; index.html 55763be8…, js `index-BqXM-igR.js` 6ceb01ea…, css `index-CbbIe769.css` 그대로
- `DIST=dist-t46b MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs` → 0 (`source identity: ok`, 1440/390/360)
- `DIST=dist-t46b CV_M3_RECEIPT=…/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → 0 (`receipt identity: ok`)
- showcase-smoke (임시 dist 복사, 끝나고 삭제) → 0
- 음성 대조: 3a3690d 빌드 `dist-t45`로 model smoke → exit 1 (`detail: … 시드별 신경망 하나씩도 5/5개 …`)
- 화면: `disposition-open-390.png`, `disposition-open-1440.png` (390폭 줄바꿈 정상, 눈으로 확인)
