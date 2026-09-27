# June 보정 지원·비용 메타데이터 감사 (ML-JUNE-SUPPORT-AUDIT-v1)

**결과:** 2025년 6월 정규시즌 전체 요청 모집단은 **115,816구·397경기·552투수**, 기존 June Cpanel 보정 집합은 요청 **5,212구·115경기** 중 적격 **4,821구·110경기**다. 전체 6월의 **적격** 집합은 동결된 key 목록이 없어 **미측정**이고, 지원은 상·하한으로만 판정했다. TRAIN-volume 네 그룹 중 전체 6월 적격 기준 **지원 확정은 high 하나**이고 zero·low·middle은 **불확정**(요청 수로는 30경기·500구를 넘지만 적격 수를 모름)이다. 비-Cpanel 차집합은 적격 하한이 0이므로 **지원 확정 그룹이 없다**. 이 감사는 예측·보정·채점을 하지 않았고 I2 활성화 여부를 바꾸지 않는다.

작성: Claude Opus 5.5(구현). 독립 검토: Astra(미완료). 결과 파일 `results/ML-JUNE-SUPPORT-AUDIT-v1.json` SHA256 `4ec64b401bb3972d51a3f004d914d730d5233c9f483bac09d40069f100b96fb5`, 설정 SHA256 `cf77ed772df6022bc51870212f2348d8c62abd39402b4355fa99da31947dc72b`, 실행 코드 커밋 `7f27fea`. 실행 명령: `/Users/song/Projects/pitcheezy/.venv/bin/python scripts/audit_june_calibration_support.py`. 결정적 메타데이터 집계라 시드는 해당 없음.

## 입력 경계

- 원천: SSD `processed/pitches.parquet`(SHA256 `9f65d1db…e8de27`, EXP-P3-001 `dataset_identity.processed_sha256`와 일치). 읽은 열은 key 3개·`game_date`·`game_type`·`split`·`pitcher`·`batter`·`starter_pitcher`·`p_throws`뿐이고 날짜 pushdown(2025-06-01~30)으로 115,816행만 읽었다(모두 `R`).
- Cpanel: EXP-P4-001 `blend_keys`/`blend_metadata`(key, pitcher, batter, `in_cpanel`, `train_volume`, `game_role`, `throwing_hand`, `month`만). 해시는 P4 preparation 및 EXP-P11-002 설정의 고정값과 일치.
- TRAIN-volume: `panel.json`의 동결 q25=170, q75=1514와 1,137명 `train_players` 라벨을 그대로 썼다. 저장 라벨과 동결 cutoff 불일치 0. June 분위수는 다시 계산하지 않았다. TRAIN에 없는 투수 = zero.
- 게임 역할: `pitcher == starter_pitcher` → starter, 아니면 relief. 손: `p_throws`.
- 읽지 않은 것: events/description/라벨/확률/손실/현재 투구 좌표, `supported_pa`, DEV 점수·예측, aux.pkl·체크포인트, 2026 데이터. 새 fetch·profile 없음.

## 정렬·중복·차집합

`blend_keys`와 `blend_metadata` key 순서 동일, 중복 key 0. 4,821개 blend key 전부가 6월 요청 모집단에 있고 pitcher·batter·in_cpanel·train_volume·game_role·throwing_hand 불일치 모두 **0**. Cpanel 요청(패널 48명 중 6월 등판 20명) 5,212구는 P4 `coverage.blend.requested_pitches`와 일치, 요청→적격 제외 **391구**(P4 기록 사유: unsupported PA 391 등, 사유 중복 집계)는 보존만 하고 재계산하지 않았다. 비-Cpanel 차집합은 **110,604구·397경기·532투수**이고 key 분할은 정확하다(5,212+110,604=115,816). Cpanel 적격 110경기 전부, Cpanel 요청 115경기 전부가 차집합과 경기를 공유하며 차집합 전용 경기는 282개다.

## 적격 모집단이 없는 이유

EXP-P4-001/EXP-P11-001의 whole-MLB 적격 목록(`mlb_dev`, 311,721구)은 **2025-07~09만** 포함한다. 6월 비-Cpanel 적격성은 결과·좌표 필드가 있어야 재현되므로 이 감사 범위에서 만들지 않았다. 따라서 전체 6월 적격 지원은 **하한 = Cpanel 적격 부분집합**(같은 적격 규칙: P11에서 Cpanel DEV 12,334구가 whole-MLB 적격의 `in_cpanel` 부분과 정확히 일치), **상한 = 요청 수**다. 차집합은 하한 0, 상한 = 요청 수. 판정: 하한 통과 → supported, 상한 실패 → unsupported, 그 외 → indeterminate.

## 지원표 (구/경기, 규칙: ≥30경기 AND ≥500구)

| 그룹 | 전체 6월 요청 | Cpanel 적격 | 비-Cpanel 요청 | 전체 적격 판정 | 차집합 적격 판정 |
|---|---|---|---|---|---|
| zero | 5,154/132 | 0/0 | 5,154/132 | indeterminate | indeterminate |
| low | 5,923/191 | 635/12 | 5,260/180 | indeterminate | indeterminate |
| middle | 43,376/395 | 1,014/19 | 42,297/395 | indeterminate | indeterminate |
| high | 61,363/389 | 3,172/86 | 57,893/385 | **supported** | indeterminate |
| starter | 67,318/397 | 3,333/39 | 63,825/397 | **supported** | indeterminate |
| relief | 48,498/396 | 1,488/81 | 46,779/396 | **supported** | indeterminate |
| L | 31,858/376 | 2,538/67 | 29,159/372 | **supported** | indeterminate |
| R | 83,958/397 | 2,283/54 | 81,445/397 | **supported** | indeterminate |
| starter×zero | 2,411/32 | 0/0 | 2,411/32 | indeterminate | indeterminate |
| starter×low | 1,744/24 | 499/6 | 1,217/18 | **unsupported** | **unsupported** |
| starter×middle | 14,206/164 | 889/10 | 13,270/155 | indeterminate | indeterminate |
| starter×high | 48,957/364 | 1,945/23 | 46,927/356 | indeterminate | indeterminate |
| relief×zero | 2,743/107 | 0/0 | 2,743/107 | indeterminate | indeterminate |
| relief×low | 4,179/175 | 136/7 | 4,043/169 | indeterminate | indeterminate |
| relief×middle | 29,170/395 | 125/9 | 29,027/395 | indeterminate | indeterminate |
| relief×high | 12,406/334 | 1,227/66 | 10,966/312 | **supported** | indeterminate |

Cpanel 적격 volume 지원(0/635/1,014/3,172구, 0/12/19/86경기)은 EXP-P11-002 I2 비활성 기록과 같다. hand×volume 8칸과 Cpanel 요청 열은 결과 JSON에 있다. starter×low는 6월 요청 전체도 24경기라 어느 적격 규칙이든 **지원 불가**다. 요청 수는 적격 수의 상한일 뿐이므로 indeterminate 칸을 지원 가능으로 읽으면 안 된다.

## 비용 (측정과 외삽 분리)

**측정(기존 기록):** EXP-P11-001 `profile.json`(400 draw, 8,192행, wall 12.50초, 추론 9.09초, 멤버당 행당 0.0011091초)과 `ledger.jsonl`(prepare 1.85초, profile 12.55초, predict 5멤버 353.3/355.4/353.8/354.9/355.2초, 합 1,772.66초, 멤버당 311,721행). G0 보고서의 평가 단계 권위 worker wall 1,821.25초(외부 감독기 기록)는 단계 전체 값이라 위 구성요소와 별도로 둔다.

**외삽(추론만, 선형, 보장 없음):** 5멤버 400 draw 추론을 행 수에 비례시켰다. 전체 6월 요청 115,816행(적격 상한) **642초(profile 비율)/659초(ledger 비율)**, 차집합 요청 110,604행 **613/629초**, Cpanel 적격 4,821행 **27/27초**. 적격 행은 요청보다 적으므로 상한 성격이지만 장치 상태·I/O·행 구성에 따라 달라질 수 있다.

**미측정:** 6월 적격 key·특징 준비, 새 fit(이 감사는 fit을 가정하지 않음), 보정 fit, 채점·bootstrap 비용.

**감사 자체 wall:** 0.211초(해시 검증·투영 읽기·집계). 위 외삽과 더하지 않는다.

## 해석 경계와 한계

- EXP-P11-001(전체 G0 평가)·EXP-P11-002(보정)는 이미 노출된 근거이며 이 감사는 그 점수·예측을 읽지 않았다. 6월 데이터의 독립성은 주장하지 않는다(노출 감사는 Astra 담당).
- 적격 지원은 상·하한만 측정 가능하다. 정확한 6월 적격 지원을 얻으려면 별도 등록된 준비 단계가 필요하다.
- 게임 역할은 `starter_pitcher` 기록에 의존하며 P4 `game_role`과 Cpanel 4,821구에서 불일치 0으로 확인했을 뿐 비-Cpanel 행에서는 따로 교차 확인하지 않았다.
- 원본 `ledger.jsonl` 해시는 이 감사가 처음 고정했다(기존 manifest에 없음).
