# 의도(목표 위치) 스펙 v1

기준일 2026-09-22 (D46·D47·D48). 코드화 `apps/observer/backend/observer_app/intent.py`·`intent_store.py`,
계약 테스트 `apps/observer/backend/tests/test_intent.py`.
변경하면 이 파일의 변경 이력에 한 줄, `docs/decisions.md`에 한 줄, 계약 테스트를 같이 고친다.

## 0 — 왜 있나, 누가 무엇을 하나

`docs/roadmap.md` 의 기능 3(G4, 사건별 기여 분해)에는 "투수가 원래 어디를 노렸나"라는 기준점이 필요하다.
Statcast 에 의도가 없고(`design.md` §0-f), 기준점이 없으면 계획 오류와 실행 오류를 가를 수 없다.
이 경로가 M1 에서 가장 오래 걸리고, 그래서 가장 먼저 연다.

**영상 모듈은 별도 담당자가 맡는다 (D46).** 그래서 이 문서는 라벨을 모으는 방법이 아니라
**그 모듈의 산출물이 꽂힐 자리의 모양**이다. 순서는 계약 먼저, 구현은 나중에 합친다.

| 쪽 | 하는 일 |
|---|---|
| 영상 모듈 (담당자) | 클립·투구 → 투구 전 셋업 프레임에서 미트 위치 추정, 카메라 보정, 품질·기권 판정 |
| 이쪽 (서비스) | 좌표계 사슬, `IntentEstimate` 검사, 검토 라벨 보관, 소비(설명·분해) |
| 경계 | `validate_intent_estimate` 하나. 여기를 통과하지 못하면 저장되지 않는다 |

## 1 — 좌표계 사슬

좌표는 네 프레임을 **건너뛰지 않고 순서대로만** 지난다.

```
image_pixels  ──①──▶  annotated_image_zone  ──②──▶  plate_feet  ──③──▶  zone9
   픽셀                    단위정사각형              피트              9구역 id
```

홉마다 `{source_frame, target_frame, method, version, error, error_units, error_status, evidence}` 를
남긴다. `error=None` 은 **미측정**이며 0 이나 추정치로 대신하지 않는다 (`error_status='unmeasured'`).

| 홉 | 버전 | 필요한 입력 | 오차 | 현재 상태 |
|---|---|---|---|---|
| ① image_pixels → annotated_image_zone | `annotated_quad_homography_v1` | `calibration_corners` 네 점 (TL,TR,BR,BL) | **미측정** | 구현 있음. 저장된 라벨에는 네 점이 없다 |
| ② annotated_image_zone → plate_feet | 담당자가 정함 | `plate_calibration` (3×3 행렬, method, x_convention, rms_error_feet, fit_evidence) | 담당자가 실측하거나 명시적으로 미측정 | **구현·측정 모두 없음.** 담당자가 채울 홉 |
| ③ plate_feet → zone9 | `domain_zone9_v1` | 타자별 `zone_bounds{top,bottom}` | 오차 대신 `boundary_margin_feet` (양자화라 오차가 아니라 경계까지의 여유) | 구현 있음 |

**홉이 없으면 거기서 멈추고 이유에 이름을 붙인다.** `deepest_frame` 과 `blocked_by`
(`no_image_plane_calibration` / `no_plate_plane_calibration` / `no_batter_zone_bounds`) 가 같이 나간다.
그래서 **② 가 없는 지금, 9구역 의도는 어떤 경로로도 나오지 않는다.** 이게 이 계약의 핵심 장치다.

좌우 뒤집힘은 추론하지 않는다. `plate_calibration.x_convention` 은 `statcast_plate_x_catcher_view`
하나만 받는다. 9구역 id·부호 대응은 `apps/observer/CONTRACT.md` 의 `GET /api/zones`
(`column0 = 포수 시점 왼쪽`, 타자 손에 따라 뒤집지 않음)와 `domain.py::ZONES` 를 그대로 따른다.
`PLATE_HALF_WIDTH_FEET = .83` 도 `domain.py` 와 같은 값이며, 한쪽만 바꾸면 두 곳이 갈린다.

## 2 — IntentEstimate

`SERVICE_ARCHITECTURE_DRAFT.md` §5 의 `IntentEstimate` 를 코드로 좁힌 것이다.

```json
{
  "schema_version": 1,
  "pitch_id": "…|null", "clip_id": "…", "clip_sha256": "…",
  "status": "estimated | unavailable",
  "unavailable_reason": "…|null",
  "method":     {"kind": "…", "version": "…"},
  "evidence":   {"frame_index": 12, "frame_time": 7.4},
  "provenance": {"label_source": "video_module|assistant_visual_estimate|human_manual_annotation",
                 "review_status": "unreviewed"},
  "deepest_frame": "annotated_image_zone", "blocked_by": "no_plate_plane_calibration",
  "points": {"image_pixels": {"x": …, "y": …},
             "annotated_image_zone": {"x": …, "y": …, "inside_annotated_quad": true}},
  "transform_chain": [ …홉 기록… ],
  "uncertainty": {"value": 10.0, "units": "pixels", "basis": "…"},
  "is_intent_proxy": true,
  "claims": {"catcher_intent_verified": false, "independent_ground_truth": false,
             "physical_plate_coordinates": false, "accuracy_estimate": null}
}
```

검사가 강제하는 것 (전부 계약 테스트에 있다):

1. `status='observed'` 는 없다. 셋업 추정은 관측이 아니다.
2. `is_intent_proxy` 는 항상 `true`이고 끌 수 없다. **포수 셋업은 의도의 대리값이지 투수의 진짜 의도가 아니다.**
3. `catcher_intent_verified`·`independent_ground_truth` 는 `false` 고정, `accuracy_estimate` 는
   검토 표본에서 재기 전까지 `null` 고정.
4. `points` 의 키는 실제로 도달한 프레임들과 **정확히 같고 순서도 같다**. 홉을 건너뛴 사슬은 거부된다.
5. `physical_plate_coordinates` 는 실제로 `plate_feet` 이상에 도달했을 때만 `true`.
6. `zone9` 는 `deepest_frame='zone9'` 일 때만 존재한다.
7. `review_status` 는 `unreviewed` 만 받는다. **검토 상태를 스스로 주장할 수 없다.**
8. `status='unavailable'` 이면 `points`·`transform_chain` 이 비어 있고 이유가 있어야 한다.
   기권은 기권으로 기록되며, 좌표를 곁들인 기권은 거부된다.
9. `deepest_frame` 이 `zone9` 가 아니면 `blocked_by` 가 반드시 있다.

## 3 — 사람 검토 라벨

**미검토 추정과 검토 라벨은 다른 타입이고 다른 표에 있다. 승격 함수는 없다** (`intent_store.py`).

```json
{
  "schema_version": 1, "label_type": "reviewed_setup_label",
  "source_estimate_id": "<추정의 sha256>",
  "decision": "accepted | corrected | rejected",
  "reviewer_id": "…", "reviewed_at": "2026-09-22T10:00:00+09:00",
  "evidence": {"frame_index": 12, "frame_time": 7.4},
  "uncertainty": {"value": 8.0, "units": "pixels", "basis": "…"},
  "frame": "image_pixels|null", "point": {"x": …, "y": …}, "notes": "…"
}
```

- `reviewer_id` 와 `decision` 에 **기본값이 없다**. 사람이 고르지 않으면 이 레코드는 존재할 수 없다.
- `corrected` 만 자기 좌표를 들고 있다. `accepted`·`rejected` 는 좌표를 들 수 없다.
- `uncertainty.value=null` 은 허용하되 `basis` 에 미측정임을 적어야 한다.
- `rejected` 는 검토 라벨을 만들지 않는다. 추정은 여전히 미검토로 남는다.
- 한 추정에 검토자 여러 명이 붙을 수 있다 (일치도 측정용). 추정 행과 검토 행 모두 불변이다.
- `reviewed_labels()` 가 검토 라벨을 얻는 **유일한 경로**이고, 결과에는 항상 검토자와 결정이 붙는다.

## 4 — 미트-공 차이

`setup_actual_difference` 는 **같은 프레임 안에서만** 뺀다. 프레임이 다르면 `unavailable` 로 거절한다.
결과에는 `is_command_error=false` 와 아직 아무도 분리하지 않은 성분들의 이름이 함께 나간다:

- `setup_is_a_proxy_not_the_pitcher_intent`
- `glove_detection_error`
- `coordinate_transform_error`
- `glove_movement_between_setup_frame_and_release`
- `pitch_movement_and_measurement_error`

이 다섯이 분리되기 전에는 이 거리를 제구 오차라고 부르지 않는다.

## 5 — 담당자가 내야 하는 것 (적합성 검사)

산출물 JSON 을 `validate_intent_estimate` 에 넣어 통과하면 저장할 수 있다.
`tests/test_intent.py` 가 그 검사의 전체 목록이다. 특히:

1. **투구 전 셋업 프레임만.** `evidence.frame_time` 이 릴리스 이전이어야 한다
   (`video_lab.track_frames` 가 이미 릴리스 이후 프레임을 거부한다).
2. **기권 경로가 반드시 있어야 한다.** 가려짐·장면 전환·저품질에서 좌표를 내는 대신 `unavailable` 을 낸다.
3. **② 홉을 채우면 9구역이 열린다.** `plate_calibration` 에 3×3 행렬·`method`·`x_convention`·
   `fit_evidence` 를 넣고, `rms_error_feet` 는 실측하거나 `null`(미측정)로 명시한다.
   이것이 G4 를 막고 있는 단 하나의 기술 항목이다.
4. **표본 구성과 선택 편향을 같이 낸다.** 클립 선정 규칙을 문서로 남긴다. 결과로 고른 표본
   (예: 삼진 모음)만으로 학습·평가하지 않는다 (`roadmap.md` M1).
5. **실제 검출기를 붙이는 날 풀 장치는 `store.finish` 한 곳이다.** 지금은 `status` 가
   `unavailable`/`failed` 가 아니면 `ValueError` 를 낸다. 그 전까지 이 장치를 풀지 않는다.

라벨 목표 개수와 표본 설계, 검토자 배정은 **미정**이다. `roadmap.md` M1 의 순서대로
첫 라벨의 일치도와 목표 오차 폭을 재고 나서 정한다.

## 6 — 현재 실측 (2026-09-22)

`runs/observer-improvement-v2/video_annotations.sqlite3` 를 직접 세었다. 추정하지 않았다.

| 항목 | 실측 |
|---|---|
| 저장된 라벨 | **5** (구간 4개 + `40`/`40.0` JSON 표기 검증 때 생긴 중복 1) |
| `label_source` | 5개 모두 `assistant_visual_estimate` (AI 시각 추정) |
| `review_status` | 5개 모두 `unreviewed`. **사람 검토 라벨 0개** |
| `calibration_corners` | 5개 모두 `null` → ① 홉조차 없어 **전부 `image_pixels` 에서 멈춘다** |
| `pitch_id` | 5개 모두 `null`. 실제 투구와 연결된 라벨 0개 |
| 추적 결과 | 4건, 전부 `abstained` (`low_match_confidence_or_occlusion`), `normalized_target=null` |
| 9구역 의도 | **0개** (② 홉 없음) |
| 클립 | 1개 (`seven_strikeouts`, 1280×720). 공식 Logan Webb 삼진 모음 2025-08-17 |
| 표본 구성 | 한 투수·한 경기·한 카메라·삼진만. **선택 편향 있음**, 일반화 근거 아님 |
| 검토자 | 없음 |

광류 후보가 같은 4구간을 끝까지 추적해 AI 참고값과 1.07~8.59px 차이였다는 기록은
`apps/observer/IMPROVEMENT_LOG.md` 에 있다. 탐색적 비교이며 서비스 기본 추적기는 고정 템플릿 그대로다.
저장된 추적 결과 4건은 모두 고정 템플릿의 기권이다.

`IntentLabels.sample_composition()` 이 이 표를 저장된 것만 세어 다시 만든다.

## 변경 이력

- v1 (2026-09-22, D46·D47·D48) 신설. 좌표계 사슬 4프레임, `IntentEstimate`, 검토 라벨, 미트-공 차이 규칙.
