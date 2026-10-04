# T45 — 모델 카드: "서비스 모델 확률 = 신경망 5개 평균 + 상황 빈도 기준 약 25%" 문장

상태: 작성·자체 검사 끝, same-model(Opus) 검토 대기. Codex 검토 없음. 8771 미배포.

## 왜
검토 `f07a6ed`의 N1. 교수 질문 예상: "1.4904가 신경망 단독 점수인가? 비교 대상(빈도 기준)이 모델 안에 섞여 있지 않나?"
비교 자체는 유효하다(섞는 비율은 평가 기간 전 6월 자료로 정함). 다만 카드에 적혀 있지 않았다.

## 바꾼 것 (기준 showcase `80624ca`)
- `ModelEvidenceCard.tsx` "무엇을 예측하나"에 한 문장:
  "서비스 모델의 확률은 신경망 5개의 평균에 상황 빈도 기준을 약 25% 섞은 값입니다. 섞는 비율은 평가 기간보다 앞선 2025년 6월 자료로 정했습니다."
  숫자는 JSON에서 계산(25 = round((1−0.74678)·100), 6월 = blendWindow).
- `model-evidence.json`에 `blend` 블록: modelWeight 0.7467754577062221(그대로, 반올림 없음), members 5, blendWindow 2025-06-01~06-30, 원본 3개 sha256 + 앵커(b38bea6).
- `tests/model-evidence-smoke.mjs`: `git show b38bea6:<file>` sha256·앵커 확인, P10 `reports.G0-global.selection.model_weight`와 정확히 같음, model+baseline=1, P11 설정이 이 P10 파일 sha를 고정, 혼합식 앵커 유지, members = P11 seeds 수, 문서의 blend `06-01~06-30`가 DEV 앞. 카드 문장 2개 텍스트 검사.

## 원본 사슬 (b38bea6에서 읽음)
- `matrix_g0_whole_metrics.py`: `primary = weights[0]*calibrated + (1-weights[0])*frequency['dev']`, `evaluate_whole`의 `g0` = `primary`.
- `score_ml_g0_whole.py`: `frozen_five_predictions(members, baseline, read_json(c1_results_path)['reports']['G0-global'])`.
- `configs/EXP-P11-001.yaml`: `analysis_results_sha256` = 82768e20… = `results/EXP-P10-001.json`; `june_weights` 그대로 재사용.
- `results/EXP-P10-001.json`: model_weight 0.7467754577062221, baseline_weight 0.2532245422937779.
- `docs/ML_EXPERIMENT_MATRIX.md`: blend `06-01~06-30`, DEV `07-01~09-30`.

## 검사 (web 폴더, threads 1)
- `npx tsc -b` → 0
- `npx vite build --outDir dist-t45 --emptyOutDir` → 0 (js `index-C4GX1afL.js`, css `index-CbbIe769.css` 기존과 같음)
- `DIST=dist-t45 MODEL_EVIDENCE_REPO=<main> node tests/model-evidence-smoke.mjs` → `source identity: ok`, ok (1440/390/360)
- `DIST=dist-t45 CV_M3_RECEIPT=<main>/results/checks/CV-M3-receipt-recalculation-20261004.json node tests/cv-evidence-smoke.mjs` → `receipt identity: ok`, ok
- `showcase-smoke` (임시 dist 복사 후 삭제) → ok
- 음성 대조 1: 현재 배포 빌드 `t44b-build`(80624ca) → exit 1 (`detail: 서비스 모델의 확률은 …`)
- 음성 대조 2: modelWeight 끝자리 1 바꿈 → exit 1 (`archived June weight (exact)`), JSON 복원 cmp 일치

## 한계
- 같은 모델(Opus) 작성. 검토 전.
- "신경망"은 flatten_mlp(기존 conditioning 앵커). 빈도 기준이 실제 구종을 조건으로 쓴다는 사실은 JSON에만 있고 카드에는 넣지 않음(N1 선택 문장 보류).
- 실제 iPhone/Safari 미확인.
