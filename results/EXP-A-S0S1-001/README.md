# A S0/S1 실제 실행 결과

공통 기준 `5289d90`, 결과 전 선정/평가 코드 기준 `ccda09a`. 계약 C0-v1, 가치 defense-we-pa-v1. 코드·config·자료 해시와 시드는 JSON 원본을 따른다. 최종 평가나 모델 채택 실험이 아닌 **노출된 2025년 7월 소규모 개발 평가**다.

| S1 예측기 | NLL ↓ | Multiclass Brier ↓ | Top-label ECE10 ↓ |
|---|---:|---:|---:|
| 고정 서비스 blend(구종 주변화 경로) | 1.521566 | 0.726403 | 0.009218 |
| 고정 서비스 frequency | 1.555541 | 0.734695 | 0.047247 |
| 작은 4월 count/좌우 빈도 기준선 | 1.589641 | 0.746602 | 0.018373 |

동일 563구·151타석·6경기. blend−작은 기준선 NLL 차 −0.068076, 경기 paired bootstrap 95% CI [−0.107074, −0.033128]. blend−고정 frequency −0.033975 [−0.049475, −0.009452]. 1,000회, seed 20260923. 표본이 작고 이미 노출된 개발 자료다. 모델의 학습량/정보가 다르므로 이 차이를 새 구조의 효과로 해석하지 않는다. ECE 역시 563구 수준 보정의 증명은 아니다.

S0는 같은 고정 목록의 투수별 첫 2경기·211구. 기존 전처리 자료를 검증/선정한 뒤 작은 기준선 적합→저장/복원→고정 신경망/WE 복원→구종 및 목표 추천→평가를 연결했다. 독립 Engine 재로드의 neural/frequency/blend 텐서는 정확히 일치했고 CountBaseline 저장/복원도 정확히 일치했다. 이 실행의 Engine은 해시를 검증한 research 경로이며, C0 실제 예시는 별도로 standalone 경로에서 실행했다.

자료 보존: TRAIN 6경기·429타석·1,655구(4/1~13), DEV 6경기·439타석·1,690구(7/5~21). 다른 선수·미지원 타석까지 원래 전체 경기 행을 보관했다. 점수 계산은 TRAIN 선정 투수 149타석 중 145타석, DEV 160타석 중 151타석이다. DEV 제외 사유는 종료 공 없음 1, 타석 중 상태 변화 3, 미지원 전이 5. S1 triple 0개, S0 triple/hbp 0개이므로 해당 희귀 클래스 상세 평가는 null이다.

`observer-zone-v1`은 S0 각 경기의 첫 지원 타석 1개씩, **2건 중 2건** 추천 ready를 확인했다. 이는 전체 563구의 위치 추천 지원률이 아니다. 범위를 고정한 연결 검사이며 4월 자료만으로 만든 league 존 높이·레퍼토리를 썼다. 서비스의 전체 과거/최근90일 프로필 경로와 입력 범위가 다르며 config에 명시했다. 구역별 후보 support는 커널 ESS이고 OPE ESS가 아니다. 구종-only WE 진단은 151타석에서 기록했고, 둘의 기준 정책을 분리했다.

**정책 가치/목표 개입 OPE는 미측정(null).** 목표 의도와 logging propensity가 없어 식별되지 않는다. 내부 WE·%p는 실제 승률 개선이 아니다. 선택 경기 전체 홈런 13개·삼진 102개는 사전에 정한 규칙으로 `event_cases.json`에 남겼다. 비코호트 사건은 고정 모델 미지원일 수 있으며 실제 의도/가용 불펜 명단은 없다.

실측 평가 12.077초, 최대 RSS 421,773,312 bytes(macOS). 모든 결과·선정 자료·동결 번들 파일 해시 검증 통과. 원자료 클라우드 재해시는 timeout이라 수행 완료로 주장하지 않는다. 로컬 처리자료 SHA256은 기존 승인된 `data_quality.json`과 일치하며 원자료의 기존 해시 attestations를 보존했다. 2026 파일을 새로 열지 않았다.

## 재현·파일

루트에서 최초 실행(동일 경로가 비어 있는 환경):

```sh
.venv-observer-standalone/bin/python scripts/a_small_eval.py prepare
# 선정 manifest를 확인/커밋한 뒤, 평가
.venv-observer-standalone/bin/python scripts/a_small_eval.py evaluate
```

기존 실험을 덮어쓰지 않도록 prepare/evaluate는 완료 경로의 재실행을 거부한다. 현재 산출물의 읽기 전용 재검증:

```sh
.venv-observer-standalone/bin/python scripts/a_verify_run.py
```

새 데이터/설정 실험은 별도 experiment ID/config·출력 경로·보고서로 만들어야 한다. 재현 전에는 현재 config 및 결과에 기록된 evaluation code hash를 확인한다.

- Git: `selection_manifest.json`, `results.json`, `verification.json`.
- SSD `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/A-S0S1-001/`: `train_full_games.parquet`, `dev_full_games.parquet`, `selected_pitch_keys.json`, `april_count_baseline.pkl`, `predictions.npz`, `internal_we_diagnostics.json`, `observer_zone_diagnostics.json`, `event_cases.json`, `evaluate.log`.
- B의 작은 모델 학습에서는 **4/30 프로필 스냅샷을 4월 TRAIN 공에 쓰지 않는다**. 이 스냅샷은 이번 7월 DEV 추론에만 사용했다. B는 TRAIN 각 공 이전의 프로필을 기존 strictly-prior-date helper로 구성하고 같은 TRAIN/DEV 목록·지원 마스크에서 요소 한 축을 비교해야 한다.
