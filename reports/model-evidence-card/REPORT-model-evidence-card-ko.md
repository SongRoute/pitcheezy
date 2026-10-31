# 예측 모델·추천 검증 카드 (/showcase) — 작업 기록

작성 2026-10-04 13:54 KST, 위임 작업자 (cycle-004, T32). 상태: 직접 검사 통과, 같은 모델(Opus) 검토 전, Codex 검토 전, 8771 미반영.

## 왜
시연 3:30–4:30 구간에서 "예측 정확도 / 추천 정책 효과 / CV 좌표"를 나눠 설명해야 한다(DEMO-1006-rehearsal). `/showcase`에는 CV 카드와 정책 한 문장만 있었다. CV 카드와 같은 방식으로 접힌 카드를 하나 더 둔다.

## 바꾼 것 (기준: showcase `7b268a3`)
- 새 `apps/observer/web/src/model-evidence.json`: 메인 레포 `b38bea6`의 커밋된 결과에서 Python으로 그대로 복사(반올림 없음). 파일 sha256을 함께 고정.
  - `results/EXP-P11-001.json` 44dae07c…230b: 서비스 G0 vs 상황 빈도 기준, 2025-07-01~09-30 전체 MLB 311,721구·1,161경기, NLL 1.490408 vs 1.504081, Δ −0.013673 [−0.014415, −0.012924], 시드 5/5, ECE10 0.008710 vs 0.001660.
  - `results/EXP-P16-001.json` 9d372da2…8597: 연구 혼합 − G0 Δ −0.004062 [−0.004383, −0.003752], 5/5, 서비스 미적용(D145).
  - `results/MLB-2026-OPE-v1.json` 0bb8fcd0…09e9: 2,429경기·184,288타석, 미지 투수 시작 제외 34,763, L1-R Δ [−0.001205, +0.000921], NO_EVIDENCE_OF_IMPROVEMENT, 문턱 0.001(D131).
  - `docs/ML_EXPERIMENT_MATRIX.md` 38a72509…5155 26행: DEV 기간·학습/보정 기간.
- 새 `ModelEvidenceCard.tsx`: 요약 2줄 + 펼침(예측 / 추천 / 한계 / 출처). 링크·버튼·영상 없음. 10/4 탐색 실험(P30–P33) 수치는 독립 검토 전이라 넣지 않음.
- `Showcase.tsx`: CV 카드 바로 앞에 배치. `showcase.css`: CV 카드 스타일을 `:is(.cv-card,.model-card)`로 공유(CV 카드 클래스는 그대로).
- 새 `tests/model-evidence-smoke.mjs`: `MODEL_EVIDENCE_REPO`가 주어지면 네 원본 파일의 sha256과 모든 수치를 정확히 대조, 그 뒤 1440/390/360 폭에서 접힘 기본·요약·펼친 문구·링크 없음·가로 넘침 없음·합성 호스트 밖 요청 0.

## 검사 (맥, 가벼운 검사만; 8766/8770/8771 손대지 않음)
- `node_modules/.bin/tsc --noEmit -p .` → exit 0.
- `node_modules/.bin/vite build` (이 worktree 자신의 `dist/`, 8771이 서빙하는 경로 아님) → index.html 48fea38e…, `index-D7itW42y.js` acdd8b57…, `index-CbbIe769.css` 02cbd371….
- `MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs reports/model-evidence-card` → exit 0, `source identity: ok`, `model evidence smoke: ok` (`model.log`).
- `CV_M3_RECEIPT=…/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → exit 0, `receipt identity: ok` (`cv.log`).
- `node tests/showcase-smoke.mjs` → exit 0 (`sc.log`).
- 음성 대조: 배포 중인 `7b268a3` 빌드(`codex-showcase-1006/.../dist`)에 새 검사 → exit 1, TimeoutError(카드 없음) (`negative-control-7b268a3.log`).
- 기록된 실패: 첫 시도에서 카드에 `cv-card` 클래스를 같이 붙여 CV 검사가 strict-mode 위반(요소 2개)으로 실패 → 클래스 분리 후 통과. 또 첫 시도는 `| tail` 때문에 종료 코드가 0으로 보였음 → 로그 파일로 다시 실행해 실제 종료 코드를 기록.
- 화면: `model-card-{closed,open}-{1440,390,360}.png`.

## 하지 않은 것
- 다른 Opus 검토, 8771 반영, 실제 아이폰/Safari 확인. 새 계산·새 평가 없음. 2026 자료를 새로 읽지 않음(이미 커밋된 D131 결과 JSON을 표시용으로만 복사).
