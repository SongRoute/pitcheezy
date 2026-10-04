# 처분: 예측 모델·추천 검증 카드 검토 (`63d4127`, 대상 `3a73f06`)

`same_model_reviewed; Codex_review_pending` — 검토자는 작성자와 다른 Opus 5.5 인스턴스.

- 처분 시각: 2026-10-04 14:08 KST (`date`), v3 cycle-006 (T34).
- 반영: `63d4127`을 `codex/model-evidence-card-1004`에 cherry-pick → `aabd012`.

## 항목별
1. BLOCKING (2026 "학습·선택에는 쓰지 않음" 과장) — **수용**. 직접 대조함:
   - 메인 `docs/decisions.md:120` D87: "과거 P0 12건/P1 8건 OPE 및 2026 propensity 교차 적합·지원 집합·평가 정책 적응·반복 선택을 노출 이력으로 보존".
   - `docs/decisions.md:208` D131: "노출 이력이 있는 구간(D87)을 포함한 가정 조건부 regime 추정이며 인과 효과가 아니다".
   - `results/MLB-2026-OPE-v1.json` `interpretation`: "single registered 2026 OPE (exposed-window labels in the prereg) …".
   - 수정 문구("한 번 실행 … 노출 이력이 있어(D87), 처음 보는 자료에서의 평가는 아닙니다")가 세 출처와 맞다.
2–5. 평가 타석 149,525/184,288, 적격 311,721/341,941구, "빈도 기준보다 낮음", 위쪽 끝 vs 문턱 — **수용** (D131 문장 "E0 149,525, 미지 투수로 시작 전 제외 34,763"과 일치).
6. 스모크 보강(방향·의미 문장·지운 과장 재등장 금지) — **수용**. 검토자 변이 M3b·M4가 exit 1로 잡힘(음성 대조).
7. 제안 "던질 구종이 정해졌을 때 …" — **보류(적용 안 함)**. 고정 원본(P11 JSON·매트릭스 l.26)에 평가가 실제 구종 조건부라는 명시가 없어 카드에 넣으면 출처 없는 문장이 된다. 시연 중 질문이 나오면 말로 "구종 후보별 결과 확률을 비교한다"고 보충. 원본 확인 뒤 별도 작은 변경으로.
8. 인과 한계 문장 — **현행 유지** (OPE `interpretation`과 같은 문장; 검토자 추천과 같음).

## 처분 뒤 확인 (`apps/observer/web`, 각 `perl -e 'alarm 300'`, 로그 `disposition/`)
- `tsc --noEmit -p .` → 0
- `vite build` (이 worktree 자신의 dist, 8771 서빙 경로 아님) → 0. index.html 2db4feab…, `index-BpZyo5LN.js` 4064e103…, `index-CbbIe769.css` 02cbd371… (검토자 빌드와 파일명 같음)
- `MODEL_EVIDENCE_REPO=/Users/song/Projects/pitcheezy node tests/model-evidence-smoke.mjs` → 0, `source identity: ok`
- `CV_M3_RECEIPT=…/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → 0, `receipt identity: ok`
- `node tests/showcase-smoke.mjs` → 0
