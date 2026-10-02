# ML-JUNE-ELIGIBILITY-v1 — 전체 June 적격 목록 준비

2026-09-27. **실행 등록용 과학 계약**이다. 기준 저장소 HEAD는 `0b445a41cec7ce40c68051bab167405be554d21c`이며, 이 문서만으로 실행 등록이 완료되지는 않는다. 사용자의 “좋아 진행해”는 전체 June 적격성 준비, 정확한 지원 수와 실제 준비 비용 측정을 승인했다. 총괄이 검토된 코드 C와 실행 설정·자료·소스 pin·새 출력 경로를 등록 D에 고정한 뒤 한 번 실행한다. 모델 추론·학습·보정 적합은 이번 범위가 아니다.

## 1. 목적과 자료 접근 변경

[이전 June 감사](../reports/ML-June-support-cost-audit-2026-09-27.md)는 요청 메타데이터만 읽었으므로 전체 적격 목록을 확인하지 못했다. 이번에는 **2025년 6월 정규시즌에 한하여 적격 판정에 필요한 결과·지원·구종·좌표 필드를 새로 읽는다.** 이전의 metadata-only 접근을 그대로 유지했다고 표현하지 않는다. 노출 원장에 시점·파일·행 창·열·용도와 출력 해시를 추가한다.

목표는 동결된 P4 적격 규칙으로 전체 June eligible key inventory를 만들고, Cpanel을 정확히 재현하며, 그룹별 실제 적격 투구/경기 수를 기록하는 것이다. 결과 클래스 빈도·좌표 분포·정답 예시·예측·품질을 탐색하지 않는다. 2026, 새로운 DEV 행/정답/성능, 원자료, 새 자료 수집, aux/checkpoint 역직렬화, 신경망·frequency 추론, optimizer, 보정 적합, bootstrap, 서비스 변경은 범위 밖이다. 이미 봉인된 부모의 metadata/source/등록 해시를 읽는 것은 허용한다.

지원 충족은 데이터 준비 사실이지 보정 효과나 독립 확인이 아니다. 기존 I1 악화와 I2 비활성 판정을 변경하지 않는다. 후속 보정 후보가 필요하면 그 알고리즘·평가·예산을 새로 등록한다.

## 2. 입력 창과 최소 열

기존 `processed/pitches.parquet`만 자료 원천으로 사용한다. SHA256은 `9f65d1dbf28374d6879dbf0c19c5c98834a2e1e95a6218c83e213c955e8ede27`이다. 원자료의 현재 바이트를 재검증했다고 주장하지 않는다. 전체 파일의 opaque SHA 검증은 허용하지만 parquet를 디코드할 때는 **명시된 열 projection 및 2025-06-01~2025-06-30 날짜 pushdown**을 적용한다. 디코드된 자료가 날짜 창을 벗어나거나 2026을 포함하면 즉시 중단하며 행 내용을 출력하지 않는다. 정규시즌 `game_type == 'R'`만 요청 모집단이다.

| 목적 | 허용 열 |
|---|---|
| 키·기간·역할 | `game_pk`, `at_bat_number`, `pitch_number`, `game_date`, `game_type`, `split`, `pitcher`, `batter`, `starter_pitcher`, `p_throws` |
| 이번에 새로 허용하는 적격 판정 | `description`, `events`, `balls`, `strikes`, `supported_pa`, `pitch_type`, `plate_x`, `plate_z` |

현재 저장 processed `split`은 옛 `calibration` 구간(5~6월)이다. **원천 `split == 'blend'`를 요구하지 않는다.** 날짜로 동결 temporal 2025 fold의 `blend`(6/1~30)를 부여하고 원천 calibration/date 관계를 검증한다. 저장 출력은 source split과 연구 split을 명확히 구분한다. 전체 temporal/data loader를 호출해 다른 달·raw·physics sidecar까지 읽지 않는다.

부모 입력은 P4의 `blend_keys.parquet`, `blend_metadata.parquet`, `panel.json`, `preparation.json`, archived `source/pitchmdp/model.py` 및 이전 June 감사의 config/result다. 각 바이트 해시는 실행 설정에 별도로 고정한다. 기존 Cpanel keys와 metadata 순서가 일치해야 한다. 필요한 frozen TRAIN mapping은 `panel.train_players`에서 읽으며 TRAIN 자료를 재로드하거나 분위수를 다시 계산하지 않는다.

## 3. 적격 규칙 — P4와 동일

권위 소스는 `EXP-P4-001/source/pitchmdp/model.py`의 `outcome_labels`와 `eligible`이다. 소스 SHA256은 `0e963b96900adb09705a56afb79e8dedcf31973d253df393ffe205df39b8a150`이다. 기준 HEAD의 같은 파일도 이 해시다.

```text
eligible = mapped_outcome
           & supported_pa.fillna(False)
           & pitch_type.notna()
           & plate_x.notna() & plate_z.notna()
           & balls.between(0, 3) & strikes.between(0, 2)
mapped_outcome = outcome_labels(frame) >= 0
```

`between`은 양 끝을 포함한다. `.notna()`를 유한성 검사나 좌표 범위 검사로 바꾸지 않는다. 새로운 구종 화이트리스트·선수 역할 필터를 추가하지 않는다. `supported_pa`는 기존 처리 값 그대로 쓰며 PA 결과·완결성 판정을 다시 만들지 않는다. 이는 후향적 지원 판정이고 pre-pitch 가용성을 뜻하지 않는다.

권위 함수는 description/events 결측을 빈 문자열로 치환한 뒤 다음을 매핑한다. 아래는 데이터 빈도가 아니라 고정 코드 정의다.

- ball: `ball`, `blocked_ball`, `pitchout`, `intent_ball`.
- strike: `called_strike`, `swinging_strike`, `swinging_strike_blocked`, `missed_bunt`, `foul_tip`, `bunt_foul_tip`.
- foul: `foul`, `foul_bunt`; 단 `foul_bunt`이며 strikes=2면 strike.
- hbp: `hit_by_pitch`.
- description=`hit_into_play`일 때 events `single`, `double`, `triple`, `home_run`은 해당 클래스.
- 같은 in-play 조건에서 `field_out`, `force_out`, `fielders_choice_out`, `sac_fly`, `sac_bunt`는 out.
- 같은 in-play 조건에서 `double_play`, `grounded_into_double_play`, `sac_fly_double_play`, `sac_bunt_double_play`는 double_play.
- 나머지는 -1(미매핑).

처리 자료의 `pitch_outcome`을 대체 정답으로 사용하지 않는다. 그 매핑은 권위 `outcome_labels`와 다를 수 있다. 전체 모델 모듈을 import해 torch/모델을 초기화할 필요도 없다. 구현은 권위 소스의 순수 두 함수와 상수를 제한 추출하거나, 동등한 순수 판정을 구현하되 **합성 fixture에서 권위 함수와의 동일성**을 검사한다. 실제 새 자료에서는 적격 mask와 제외 원인 bool만 사용한다. 클래스 label 배열이 임시로 생기더라도 저장·출력·빈도 집계하지 않는다.

## 4. 순서·Cpanel·정확한 차집합 게이트

KEY는 `(game_pk, at_bat_number, pitch_number)`다. 키 결측·중복은 실패다. 행은 `(game_date, game_pk, at_bat_number, pitch_number)` 오름차순으로 고정한다. 날짜 하나당 게임과 키가 일관되는지도 검증한다. keys/metadata/exclusion flags는 같은 순서를 유지하고 int64 KEY의 C-contiguous bytes SHA256 및 실제 파일 SHA256을 모두 기록한다.

기존 감사에서 알려진 요청 모집단은 **115,816구·397경기·552투수**다. 동결 패널 pitcher 소속 요청은 **5,212구·115경기·20투수**, 그 적격 집합은 **4,821구·110경기**다. 총계뿐 아니라 새 eligible 중 panel 소속인 **ordered KEY 배열 전체가 P4 blend_keys와 정확히 같아야 한다.** pitcher/batter/TRAIN-volume/role/hand metadata도 기존 Cpanel과 일치해야 한다. 틀리면 새 전체 적격 수를 성공 결과로 공개하지 않고 차단 상태와 비수치 원인을 기록한다.

패널 요청에서 새 mask가 제외한 keys가 `panel_requested_keys − frozen_blend_keys`와 정확히 같아야 한다. 그 수는 **391구·50경기·17투수**다. 이는 기존 frozen eligible keys와 요청 metadata로 구성한 참조 차집합이며, 이전에 별도 391-key 파일을 봉인했다고 주장하지 않는다.

다음을 구분해 저장·보고한다.

1. 전체 요청, 전체 적격, 전체 부적격.
2. 패널 투수 요청, 패널 적격(기존 4,821 keys), 패널 부적격(391 keys).
3. 패널 밖 투수 요청 **110,604구**와 그중 적격·부적격.
4. 요청 전체에서 frozen eligible Cpanel keys를 뺀 정확한 차집합 **110,995구**. 여기에는 알려진 부적격 391구가 포함된다.
5. **전체 적격에서 frozen eligible Cpanel keys를 뺀 정확한 적격 차집합**. 이는 패널 밖 투수 요청 중 적격과 정확히 같은 keys여야 한다.

모든 분할의 key disjointness/union equality를 검증한다. 전체 eligible 수는 4,821 이상, 이전 촘촘한 상한 **115,425 이하**여야 한다. 정확한 전체 수 자체는 사전 기대값으로 지정하지 않는다. Cpanel/적격 차집합의 game overlap과 각 집합의 고유 게임·투수 수를 보고한다. 행 차집합을 독립 경기 집합으로 부르지 않는다.

## 5. 제외·지원 집계

요청 key마다 일곱 bool 원인(`unmapped_outcome`, `unsupported_pa`, `missing_type`, `missing_plate_x`, `missing_plate_z`, `invalid_balls`, `invalid_strikes`)과 `eligible`를 만든다. 각 원인의 OR가 정확히 `~eligible`인지 검사한다. 원인별 투구 수와 고유 경기 수는 겹칠 수 있음을 명시하고, `missing_coordinates = missing_plate_x OR missing_plate_z`의 합계도 기존 P4 coverage와 비교한다. **원인 합을 유일 제외 수로 취급하지 않는다.** 각 원인의 실제 description/events 값이나 클래스별 구성은 보고하지 않는다.

그룹은 이전 감사의 동결 정의를 유지한다.

- TRAIN-volume: 원래 P4 `train_players[].train_volume`; q25=170, q75=1514와 저장 라벨 일치 검사. TRAIN mapping에 없는 투수는 zero. 재추정·재선정 없음.
- game role: `pitcher == starter_pitcher`이면 starter, 아니면 relief. 결측 starter ID는 임의로 relief 처리하지 않고 실패하거나 명시적 unknown으로 기록한다. 등록 설정에서 이 처리 방식을 고정한다.
- hand: `p_throws`; L/R 외 값과 결측을 숨기지 않는다. zero/low/middle/high, starter/relief, L/R 및 role×volume·hand×volume을 고정한다. unknown이 존재하면 별도 진단 집계로 남기고 기존 그룹에 합치지 않는다.

각 요청/적격/부적격·차집합의 투구 수와 distinct game 수를 보고한다. 각 고정 그룹의 적격 지원은 **500구 이상 AND 30경기 이상**으로만 판정한다. 0건 그룹도 유지한다. 그룹들은 같은 경기를 공유하므로 그룹 게임 수를 합산해 전체 경기 수로 읽지 않는다. 사후 그룹 경계 변경이나 최소 지원 문턱 완화는 금지한다. 새로운 지원 충족을 기존 I2 활성화나 보정 성공으로 소급 해석하지 않는다.

## 6. 최소 출력과 봉인

새 고유 attempt 디렉터리에만 쓴다. 부모 run, 원자료, 기존 June 감사 및 G0 frozen bundle은 변경하지 않는다.

- 요청 eligibility inventory: KEY·날짜/연구 split·pitcher/batter·동결 그룹·panel 소속·eligible 및 일곱 bool 원인. source split은 필요하면 별도 명명한다.
- 전체 eligible ordered keys/metadata 및 정확한 non-Cpanel eligible ordered keys/metadata. 하나의 inventory에서 명시적 mask로 재구성하는 동등한 저장 형식도 등록 설정에 고정하면 허용한다.
- preparation/result JSON: 기간·열 접근 기록, 표본/그룹/제외·overlap 집계, 게이트 결과, source/data/config/code identity, 환경 버전, 준비 단계 내부 시간과 완료 상태.
- manifest: 모든 산출물의 SHA256, ordered-key 해시, 행·경기 수, 부모 해시. 성공 completion marker는 필수 검사·모든 출력이 끝난 뒤 마지막에 작성한다. 감독기가 프로세스 종료와 manifest 바이트를 다시 확인한다.

**저장하지 않는 값:** description/events, 클래스 label, 원래 supported_pa/type/balls/strikes/plate_x/plate_z 값, 예측·loss, 클래스별 빈도. 허용된 제외 bool과 적격 key 자체가 결과 의존적 자료 노출이라는 사실은 숨기지 않는다. 에러 로그에서 frame head/값 샘플을 출력하지 않는다.

## 7. 비용·실패·실행 순서

이번 준비 family의 권위 비용 한도는 **600초**다. 외부 감독기의 monotonic **worker Popen 직전→wait 종료**를 잰다. worker interpreter/import, 내부 source/data 검증, projection, 적격 판정, 정렬·집계·출력과 정상 종료까지 포함한다. worker 내부 timer는 단계 교차 확인용이다. supervisor 자신의 시작/preflight와 종료 후 manifest 감사는 별도 outer/audit wall로 보고하고 worker 비용에 이중 합산하지 않는다. 별도 독립 사후 감사 비용도 분리한다.

최초 실행은 한 번이며 실패·중단도 family 600초에 포함한다. 자동 재시도나 실패 디렉터리 덮어쓰기는 금지한다. 실패 후 수정은 기존 시도·비용·이미 발생한 필드 노출을 보존하고, 새로운 config/attempt·잔여 예산을 검토한 뒤에만 수행한다. 시간초과 종료 지연도 실제 wall로 보고하며 초과를 성공으로 취급하지 않는다. 숨은 profile이나 표본만의 대체 실행은 없다.

실행 전 순서:

1. 구현·합성 경계 검사·독립 검토를 끝내고 코드 C를 커밋한다. 새 June 결과 필드는 아직 읽지 않는다.
2. 코드 C의 모든 실행 소스(순수 적격 함수/worker/감독기 저장소 원본 포함; 동일 바이트를 COORD에 배치), 실제 실행 config 파일 bytes/path, 계약, 기존 data/parent/reference bytes, 열 projection, 기간·분할·정렬, 지원·비용 문턱, output 경로, Python/의존성 정보를 등록 D에서 pin한다. `--config`를 받으면 **실제 전달된 경로의 해시**를 검증하고 다른 기본 config의 해시로 대신하지 않는다.
3. 소스·등록 파일이 clean이고 hash가 일치하는 실행 체크아웃에서 단일 supervised worker를 시작한다. 기존 heavy 작업과 중복하지 않으며 공용 실행 잠금 정책을 따른다.
4. 모든 입력 pin과 날짜/열 경계를 검증한 뒤에만 새 필드를 읽는다. 실제 접근 열·기간·시각·purpose를 기록한다.
5. Cpanel exact replay·분할·집계 게이트와 산출물 봉인·600초 게이트를 통과한 경우에만 준비 완료로 기록한다.
6. 작성자와 다른 검토자가 소스/해시/집계 산술·지원 판정·비용 및 노출 원장 연결을 감사한다. 추가 결과 필드 재열람은 필요성과 비용을 따로 기록한다.

## 8. 해석과 완료 조건

완료는 전체 June 적격 목록·정확한 차집합·지원 수·제외 수·실제 준비 비용을 재현 가능하게 봉인하고 독립 검토를 마친 상태다. 연구 G0의 weights, May temperature, June blend weights, 예측 결과와 후보 상태는 바뀌지 않는다. 이 준비가 독립 미개봉 확인셋을 만들거나 2026 접근을 허용하지 않는다. 후속 실행 여부는 새로 드러난 적격 지원과 비용을 근거로 별도 검토한다.
