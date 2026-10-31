# OLD–fit100 비교 범위 (월드시리즈 적응)

소유: 연구 PM · 갱신: Root가 새 수용 결정·검토를 넘길 때 · 마지막 확인: Root 동결 상태 2026-10-11T00:11:58Z(09:11:58 KST).

이 쪽은 [측정 정의](measurement-schema.md)에서 주제별로 옮겨 온 OLD–fit100 비교만 다룬다(2026-10-11 주제 분리, 문구·수치·증거 링크는 그대로 옮김). 값은 Root 동결 상태 receipt [ROOT-FINAL-STATUS.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-final-status-checkpoint-001/ROOT-FINAL-STATUS.json)(SHA-256 `12a439268b442a4562a15847c697a3a97a30e8e694e45b619717ec89f66df6fa`)에서 옮겨 적었고, 작성자는 배열·raw·모델·결과 payload를 열지 않았다. 아래 어느 항목도 고정된 199-feature source, label, live-support, 서비스 측정 계약을 바꾸지 않는다. Step 2 자료 역할은 **2026 EXPOSED DEVELOPMENT**이며 독립 TEST, 2024 대비 우위, 물리적 투구 전체 커버리지, 자동 서비스 채택은 없다.

## 단계별 처분

| 단계 | Root 처분 | 모집단 | 말하는 것 / 말하지 않는 것 | 증거 |
|---|---|---|---|---|
| Stage 1 key recovery | `ACCEPT_BOUNDED_STAGE1_ALIGNMENT` | CAL 56,275 중 공통 56,272·역사적 OLD-unsupported 3; EVAL 80,584 중 공통 80,559·OLD-unsupported 25 | source-flow/key/header 정렬만. key/role/label 불일치 없음, source 직접 rc0·소유 자식 부재. raw margin 순서는 역사적 provenance 한계. 수치 우위 아님 | [ROOT-STAGE1-ACCEPTANCE.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-key-recovery-001/ROOT-STAGE1-ACCEPTANCE.json) `99df98caf0bfe4dc79aeb956b3bdc98dd281bf1a36e07dea83d71dc0ac5e1561`, [MANAGEMENT-POSTFLIGHT-RECEIPT.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-key-recovery-001/MANAGEMENT-POSTFLIGHT-RECEIPT.json) `a153fada4df60da0dca87bd3ef0d30e60e19ccab207bb5aa0bf298c0f138f516` |
| eop-001 수치 | `ACCEPT_BOUNDED_RECORDED_EOP_RESULT` | 공통 EVAL 80,559행/275경기. OLD-unsupported 25행은 paired 비교 밖 | 기록된 결과의 제한 수용. 독립 raw metric·bootstrap 재계산 없음, 선수별·action-support 패널 PENDING. **승자·채택으로 부르지 않는다** | [ROOT-RESULT-DECISION.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-numeric-prereg-001/ROOT-RESULT-DECISION.json) `0344bda6533d2d110b439a0b9c143f2d62f7d5067d317be1c8e1c71d950661a2`, [result.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-numeric-prereg-001/author-002/attempts/eop-001/result.json) `e498fa74d4878915e70b118dce63097fba121bf49fea65ef6db2d2d0481915ce` |

## eop-001 기록 차이

eop-001에서 Root가 제한 수용한 보고 차이(fit100 − OLD, paired 95% CI; Root receipt 값을 반올림 없이 옮김):

| 지표 | 차이 | paired CI95 |
|---|---|---|
| NLL | -0.0012849726597854438 | [-0.001987740672541339, -0.0006110915861636338] |
| Brier | -0.00038700551426285035 | [-0.0006543897698058032, -0.00012561801296268141] |
| ECE | 0.000790963565581216 | [-0.0032101103479615038, 0.004531989724685142] |

- **구간 계약(기록된 값·계약, 새 재계산 아님):** paired whole-game bootstrap 10,000 resamples, seed 20261211, 추정량은 pitch-weighted `sum(loss)/sum(pitches)`, percentile 95% 구간. 2026-10-11 주제 분리 작업 지시(Root)가 이미 기록된 계약으로 넘긴 내용을 옮긴 것이며, 이 문서 작성자는 배열·raw·모델·result payload를 다시 읽거나 지표·bootstrap을 재계산하지 않았다.
- **인과 해석 금지:** OLD와 fit100은 학습 시기(era), feature, history, calibration이 서로 다르다. 따라서 위 수치는 학습 자료 크기의 인과 효과 추정이 아니다.
- 검토 범위: 지정 JSON 산술 검사 46건과 pre/post manifest 연결([REVIEW.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-numeric-prereg-001/output-review-001/REVIEW.json) `b628332db61eb67db964be6c89ef77917188314a57f351667aec1b1959b1a0da`, 보충 [SUPPLEMENT-REVIEW.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-numeric-prereg-001/review-005-supplement-clean/SUPPLEMENT-REVIEW.json) `b934b7cc15debb0cfedd336f204a8f1d378547cb60e7021d03857225ac79a49e`).
- 넓은 1,201개 텍스트 파일 hash scan은 수용 증거에서 **제외**되어 별도 사고 기록에 남았다([OUTPUT-REVIEW-SCOPE-INCIDENT.md](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-numeric-prereg-001/OUTPUT-REVIEW-SCOPE-INCIDENT.md) `bb5711ac3f3b1e7febaf5e06602e24a734e6f18c02bffe8262af3da68160d078`). 깨끗한 검토 수행으로 쓰지 않는다.
- key recovery 이력: 최초 합성 31/3 실패는 [INTERIM-31-3-EVIDENCE.json](/Users/song/Projects/pitcheezy-pm-jobs/20261011-ws-fit100-old-key-recovery-001/INTERIM-31-3-EVIDENCE.json)(SHA-256 `a34989582b004e7500144de40b53b93aecbd8685644b152f2bbf08f529aecf8f`)에 보존됐다. 최종 31/31은 작성자 단독 결과였고, 그 뒤 별개 source/data 검토·운영 검토와 수용된 정렬 실행 1회가 이어졌다. v2 C1 REWORK는 이력이다.
