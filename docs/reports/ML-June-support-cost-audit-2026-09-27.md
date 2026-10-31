# June 보정 지원·비용 메타데이터 감사 (ML-JUNE-SUPPORT-AUDIT-v1, 수정 attempt 2)

**결과:** 2025년 6월 정규시즌 요청 모집단은 **115,816구·397경기·552투수**다. June Cpanel은 패널 투수 요청 **5,212구·115경기** 중 적격 **4,821구·110경기**이고, 알려진 부적격은 **391구**다. 차집합은 두 가지로 따로 보고한다. **패널 밖 투수 요청 110,604구**(115,816−5,212)와 **적격 Cpanel key의 정확한 차집합 110,995구**(115,816−4,821, 알려진 부적격 391구 포함)다. 전체 6월 **적격** key 목록은 감사한 등록 입력에서 확인되지 않아 **미측정**이므로 지원은 상·하한으로만 판정했다. TRAIN-volume 가운데 전체 6월 적격 기준으로 **지원이 확정된 것은 high 하나**다. zero·low·middle은 **불확정**, 비-Cpanel 적격 기준으로는 지원이 확정된 그룹이 없다. 이 감사는 예측·보정·채점을 하지 않았고 I2 비활성 판정을 바꾸지 않는다. 비용은 모두 **감사 전용**이며 후보 실험 비용이 아니다.

- 작성: Claude Opus 5.5(구현). 독립 검토: Astra. attempt 1에 대한 수정 요청 4건을 반영한 뒤 [Astra 최종 재검토](../reviews/G0-freeze-data-audits-Astra-2026-09-27.md)에서 PASS, 미해결 차단 사항0건을 확인했다.
- 실행 코드 커밋: `c4b31073b8bc7785bada72d2d8ffc86ca6546c76`(실제 감사 전에 커밋).
- 실행 스크립트 SHA256: `5f5da015a1470db0911354091533a7fd453c359211e1f6ddc869362c131e2fae`. 설정 `implementation.script_sha256`과 같은지 실행 시 검사하고, 스크립트나 설정이 커밋되지 않은 상태면 중단한다.
- 설정 SHA256: `b1c21f3f1ada028ea48c6514347b2c96964c5107a0c5865d1aa20820884c0050`.
- 결과 파일: `results/ML-JUNE-SUPPORT-AUDIT-v1.json`, SHA256 `6c98069e78ca2f3bdbbfb07eb88c7e9d6049f366ffaf15ae6b12c5fccde8259f`. SSD `coordination/20260927-g0-freeze-audits/COOP-009/corrected-audit-attempt-002/result.json`과 바이트가 같다.
- 명령: `/usr/bin/time -p /Users/song/Projects/pitcheezy/.venv/bin/python scripts/audit_june_calibration_support.py --output <attempt-002>/result.json`.
- 시드: 결정적 메타데이터 집계라 해당 없음.

## attempt 1 보존과 수정 사항

첫 감사의 설정(`cf77ed77…`), 스크립트, 검사, 결과(`4ec64b40…`), 보고서, 명령을 원본 커밋(`7f27fea`, `9f787b3`)의 바이트 그대로 SSD `COOP-009/first-audit-evidence/`에 복사했다. SHA256은 같은 폴더의 `SHA256SUMS`에 있다. attempt 1의 개수와 지원 판정은 attempt 2에서도 모두 같다. 바뀐 것은 이름, 비용 출처, 검증 범위, 실패 처리다.

1. **차집합 이름.** attempt 1의 `complement_requested` 110,604구는 투수 소속 기준이었다. 이를 적격 key의 정확한 차집합이라고 부르지 않는다. 이제 `outside_cpanel_pitcher_requested`(110,604)와 `complement_of_eligible_cpanel_keys`(110,995)를 둘 다 보고한다. 전체 적격의 **촘촘한 상한**은 알려진 부적격 391구를 뺀 `full_june_tight_eligible_upper` **115,425구**다.
2. **비용 출처.** 공식 감독 큐 `status.json`(SHA256 `c784953d…aed70`)의 worker wall을 권위 값으로 쓴다. 실행 폴더의 내부 ledger(1,772.66초)는 교차 확인용일 뿐 권위 값이 아니다. 감사 비용에서는 검증 시간과 외부 명령 wall을 분리했다.
3. **출처 검증.** 선언된 출처 9개(`p11_preparation`, 공식 큐 status 포함)의 SHA256을 모두 읽기 전에 검증한다. 스크립트 SHA는 설정에 고정했고, 실행 커밋과 스크립트 바이트 해시를 결과에 기록한다.
4. **실패 처리.** key 순서 불일치, 저장된 volume 라벨과 동결 cutoff의 불일치, 기대 개수 불일치는 이제 기록만 하지 않고 감사를 **중단**한다. 기대값이 없는 항목은 통과로 처리하지 않고 `unknown`으로 남긴다. 이번 실행의 전체 판정은 `unknown_items_present`인데, 전체 6월 적격 수가 없기 때문이다.

## 입력 경계

- **원천 데이터:** SSD `processed/pitches.parquet`. SHA256이 EXP-P3-001 `dataset_identity.processed_sha256`와 같다. key 3개와 `game_date`, `game_type`, `split`, `pitcher`, `batter`, `starter_pitcher`, `p_throws`만 읽었다. 날짜 pushdown으로 115,816행만 읽었고 모두 `R`(정규시즌)이다.
- **Cpanel:** EXP-P4-001 `blend_keys`/`blend_metadata`의 허용 열만 읽었다. `panel.json`의 동결 q25=170, q75=1514와 1,137명 라벨을 썼고, 6월 분위수는 다시 계산하지 않았다. TRAIN에 없는 투수는 zero다.
- **그룹 정의:** 게임 역할은 `pitcher == starter_pitcher`이면 starter, 아니면 relief다. 손은 `p_throws`다.
- **읽지 않은 것:** events, description, 라벨, 확률, 손실, 현재 투구 좌표, `supported_pa`, DEV 점수·예측, aux.pkl, 체크포인트, 2026 데이터. 새 fetch와 새 profile도 없다.

## 정렬·중복·분할

- **Cpanel 정렬:** `blend_keys`와 `blend_metadata`의 key 순서가 같고 중복 key는 0이다. 4,821개 blend key가 모두 6월 요청 모집단에 있다. pitcher, batter, in_cpanel, train_volume, game_role, throwing_hand 불일치는 모두 **0**이다.
- **기대 개수:** P4 `coverage.blend`(요청 5,212, 적격 4,821), `samples.blend.games`(110), attempt 1 개수와 모두 일치한다.
- **정확한 key 분할:** 아래 세 식이 모두 성립한다.
  - 5,212 + 110,604 = 115,816
  - 4,821 + 110,995 = 115,816
  - 110,604 + 391 = 110,995
- **알려진 부적격 391구:** 50경기, 17투수에 걸쳐 있다.
- **경기 겹침:** Cpanel 적격 110경기는 모두 두 차집합과 경기를 공유한다. 패널 요청 115경기도 모두 패널 밖 요청과 겹친다. 패널 밖 투수만 나온 경기는 282개다.

## 적격 목록의 확인 범위와 상·하한

동결된 전체 June 적격 목록은 이번에 감사한 등록 입력에서 확인되지 않았다. 저장소 전체에서 그러한 목록이 존재하지 않음을 입증한 것은 아니다. 봉인된 결과의 부재 설명도 이 검색 범위로 한정해 해석한다.

EXP-P4-001과 EXP-P11-001의 whole-MLB 적격 목록(`mlb_dev`, 311,721구)은 **2025-07~09만** 포함한다. P11 preparation의 coverage도 P4 `mlb_dev`와 같음을 검증했다. 6월 비-Cpanel 적격성은 결과·좌표 필드가 있어야 만들 수 있으므로 이 감사에서는 만들지 않았다.

- **전체 6월 적격:** 하한은 `cpanel_eligible_known_keys`다. P11에서 Cpanel DEV 12,334구가 whole-MLB 적격의 in_cpanel 부분과 정확히 일치했으므로 같은 적격 규칙이다. 상한은 `full_june_tight_eligible_upper`다.
- **비-Cpanel 적격:** 두 차집합의 적격 부분은 같은 집합이다. 하한 0, 상한은 `outside_cpanel_pitcher_requested`다.
- **판정 규칙:** 하한이 규칙을 통과하면 supported, 상한이 실패하면 unsupported, 그 밖에는 indeterminate다.

## 지원표 (구/경기, 규칙: ≥30경기 AND ≥500구)

| 그룹 | 전체 요청 | 전체 적격 상한(촘촘) | Cpanel 적격 | 패널 밖 요청 | 적격 key 차집합 | 전체 적격 판정 | 비-Cpanel 적격 판정 |
|---|---|---|---|---|---|---|---|
| zero | 5,154/132 | 5,154/132 | 0/0 | 5,154/132 | 5,154/132 | indeterminate | indeterminate |
| low | 5,923/191 | 5,895/191 | 635/12 | 5,260/180 | 5,288/184 | indeterminate | indeterminate |
| middle | 43,376/395 | 43,311/395 | 1,014/19 | 42,297/395 | 42,362/395 | indeterminate | indeterminate |
| high | 61,363/389 | 61,065/389 | 3,172/86 | 57,893/385 | 58,191/388 | **supported** | indeterminate |
| starter | 67,318/397 | 67,158/397 | 3,333/39 | 63,825/397 | 63,985/397 | **supported** | indeterminate |
| relief | 48,498/396 | 48,267/396 | 1,488/81 | 46,779/396 | 47,010/396 | **supported** | indeterminate |
| L | 31,858/376 | 31,697/376 | 2,538/67 | 29,159/372 | 29,320/374 | **supported** | indeterminate |
| R | 83,958/397 | 83,728/397 | 2,283/54 | 81,445/397 | 81,675/397 | **supported** | indeterminate |
| starter×zero | 2,411/32 | 2,411/32 | 0/0 | 2,411/32 | 2,411/32 | indeterminate | indeterminate |
| starter×low | 1,744/24 | 1,716/24 | 499/6 | 1,217/18 | 1,245/22 | **unsupported** | **unsupported** |
| starter×middle | 14,206/164 | 14,159/164 | 889/10 | 13,270/155 | 13,317/162 | indeterminate | indeterminate |
| starter×high | 48,957/364 | 48,872/364 | 1,945/23 | 46,927/356 | 47,012/361 | indeterminate | indeterminate |
| relief×zero | 2,743/107 | 2,743/107 | 0/0 | 2,743/107 | 2,743/107 | indeterminate | indeterminate |
| relief×low | 4,179/175 | 4,179/175 | 136/7 | 4,043/169 | 4,043/169 | indeterminate | indeterminate |
| relief×middle | 29,170/395 | 29,152/395 | 125/9 | 29,027/395 | 29,045/395 | indeterminate | indeterminate |
| relief×high | 12,406/334 | 12,193/331 | 1,227/66 | 10,966/312 | 11,179/323 | **supported** | indeterminate |

- Cpanel 적격 volume 지원(0/635/1,014/3,172구, 0/12/19/86경기)은 EXP-P11-002의 I2 비활성 기록과 같다.
- hand×volume 8칸, 패널 요청 열, 부적격 391구의 그룹별 분포는 결과 JSON에 있다.
- starter×low는 촘촘한 상한도 24경기라 **지원 불가**다.
- indeterminate 칸을 지원 가능으로 읽으면 안 된다.

## 비용 (측정, 외삽, 미측정 분리)

**측정: 권위 값, 공식 큐 status.json**
- 5멤버 400 draw predict worker wall(멤버당 311,721행): 354.2867696660105 / 356.4457342501264 / 354.83392454194836 / 355.8842616248876 / 356.2332737080287초.
- 관측 범위는 멤버당 **354.29~356.45초**, 합 **1,777.68초**다.
- 같은 평가 단계의 prepare 2.78초, profile 13.48초, score 27.30초. 권위 평가 단계 합계는 1,821.25초다.

**측정: profile.json**
- 400 draw, 8,192행, 추론 9.09초로 멤버당 행당 **0.0011091초**다.

**교차 확인: 내부 ledger, 권위 아님**
- predict 합 1,772.66초로, 공식 값보다 약 5초 작다.

**외삽: 선형, 보장 없음**

| 모집단 | 행 | profile 추론 전용(5멤버) | full-worker 선형 proxy |
|---|---|---|---|
| 전체 적격 상한(촘촘) | 115,425 | 640초 | 658초 |
| 전체 요청 | 115,816 | 642초 | 660초 |
| 패널 밖 요청 | 110,604 | 613초 | 631초 |
| Cpanel 적격 | 4,821 | 27초 | 27초 |

- profile 추론 전용 값에는 멤버별 인터프리터·import·모델 load·검증 같은 고정 overhead가 빠져 있다.
- full-worker proxy는 공식 wall 합을 311,721행으로 나눈 비율이라 고정 overhead까지 행 수에 비례시킨다. 그래서 행 수가 상한이어도 **비용 상한이 보장되지 않는다**. 작은 모집단일수록 고정 overhead를 과소 반영할 수 있다.

**미측정:** 6월 적격 key·특징 준비, 새 fit(이 감사는 가정하지 않음), 보정 fit, 채점·bootstrap.

**감사 전용 비용: 후보 실험 아님**
- attempt 2 프로세스 안 측정: 출처 9개 SHA256 검증 **0.041초**, 읽기·집계·비용 계산 **0.249초**, 결과 쓰기 전까지 합 **0.312초**. 인터프리터 시작, import, 쓰기는 빠져 있다.
- attempt 2 외부 명령 wall(`/usr/bin/time -p`): **real 0.54초**, user 0.50초, sys 0.07초.
- attempt 1: 프로세스 안 값 0.211초만 있다. 이 값은 읽은 파일의 해시는 포함하지만 `p11_preparation` 검증은 포함하지 않는다. 외부 wall은 **미측정**이다.
- 위 값을 외삽 비용과 더하지 않는다.

## 해석 경계와 한계

- EXP-P11-001(전체 G0 평가)과 EXP-P11-002(보정)는 이미 노출된 근거다. 이 감사는 그 점수와 예측을 읽지 않았다. 6월 데이터의 독립성은 주장하지 않으며, 노출 감사는 Astra 담당이다.
- 적격 지원은 상·하한까지만 측정할 수 있다. 정확한 값을 얻으려면 별도로 등록한 6월 준비 단계가 필요하다.
- 게임 역할은 Cpanel 4,821구에서만 P4 `game_role`과 교차 확인했다. 비-Cpanel 행은 `starter_pitcher` 기록에 의존한다.
- 내부 `ledger.jsonl`의 해시는 이 감사가 처음 고정했다. 권위 비용에는 쓰지 않는다.
