# CV 의도 모듈 v0 작업 지시서 (팀원 Claude Code용, 2026-10-01)

Song이 팀원에게 전달하는 문서입니다. 아래 "붙여 넣을 프롬프트"를 팀원의 Claude Code 세션에 그대로 넣으면 됩니다.
배경은 [확인 결과](CV-intent-module-request-2026-10-01.md)에 있습니다.
요약하면 지금 `transition-models`에는 포수 미트·의도 모듈이 없습니다. 있는 것은 747139 한 경기의 판단 프레임 297구뿐입니다.

## 목표 한 줄

**먼저 "한 경기에서 투구별 포수 셋업 위치를 우리 계약(IntentEstimate v1) JSON으로 내는, 끝까지 돌아가는 v0"를 만들고, 그 뒤에 정확도와 규모를 올린다.**
정확도보다 연결이 먼저입니다. v0이 있어야 서비스(Pitcheezy 관전 화면)에 꽂아 보고 다음 우선순위를 정할 수 있습니다.

## 마일스톤과 도달 기준

| 단계 | 할 일 | 도달 기준(완료 판정) |
|---|---|---|
| **M0 계약 고정** (반나절) | 새 브랜치 `feature/intent-v0`(main 직접 수정 금지). 출력 스키마를 IntentEstimate v1로 고정하고, 우리 검사 함수를 복사한 `validate_intent_estimate`로 자체 테스트 | 가짜 입력 3개(정상·기권·보정 없음)가 검사를 통과하고, 잘못된 입력은 거부하는 테스트가 있다 |
| **M1 한 경기 v0** (1–2일) | 747139의 판단 프레임 297구에서 미트 점을 찍는다(수동, 반자동, 간단한 검출 모두 가능. 방법은 `method.kind`에 정직하게 기록). 홈플레이트 네 모서리로 ① 픽셀→정규화 좌표 보정. 한 줄 명령: `python -m intent.run --game 747139 --out intent_747139.jsonl` | 297줄 JSONL. 모든 줄이 검사를 통과한다. 기권은 `status=unavailable` + 이유. 소요 시간을 기록한다. `review_status`는 항상 `unreviewed` |
| **M2 plate 좌표(홉 ②)** (2–3일) | 정규화 좌표 → 홈플레이트 피트 좌표(`plate_feet`) 변환과 그 오차 측정. `x_convention = statcast_plate_x_catcher_view` 고정 | 변환 행렬과 방법, `rms_error_feet`를 기록한다. 실측하지 않았으면 `null`(추정치를 넣지 않음). 10구 이상을 사람이 확인한 표본에서 오차를 보고한다 |
| **M3 정확도 표** (2–3일) | 사람이 직접 찍은 라벨 50구 이상을 만들고, 다른 경기 2개 이상(가능하면 다른 방송사)에서 픽셀·피트 오차와 기권률을 잰다 | `accuracy_report.json`(경기별 n, 중앙값·90분위 오차, 기권률). 개발에 쓴 경기와 평가 경기를 분리 |
| **M4 규모** (이후, 별도 합의) | 여러 경기 자동 처리, 처리량, 대량 클립 수집 경로의 사용 권리 확인 | 투구당 처리 초와 장비, 사용 가능한 영상 범위 문서 |

**M1까지만 되어도 바로 알려 주세요.** 저희가 관전 화면의 "추천 위치 vs 포수 셋업 vs 실제 공" 비교에 연결합니다.

## 지켜 주실 것

- 팀원 레포의 `main`은 그대로 두고 새 브랜치에서 작업합니다. 저희 레포는 수정하지 않습니다.
- 포수 셋업은 **의도의 대리값**입니다. `is_intent_proxy=true`, `catcher_intent_verified=false`로 고정하고, 정확도는 재기 전까지 `null`로 둡니다.
- 좌표 홉은 건너뛰지 않습니다. ②가 없으면 `blocked_by="no_plate_plane_calibration"`으로 멈춥니다.
- 영상은 MLB 권리입니다. 내려받아 저장하는 범위와 공개 여부는 먼저 확인합니다.
- 숫자를 지어내지 않습니다. 없으면 "미측정"으로 둡니다.

## 붙여 넣을 프롬프트 (팀원 Claude Code 세션용)

```text
이 레포(Pitcheezy/transition-models)에서 포수 미트 셋업 위치를 투구별로 추정하는 "의도 모듈 v0"를 만든다.
main은 수정하지 말고 `feature/intent-v0` 브랜치를 codex/fix-point-label-alignment(3fc3e5e) 위에서 만든다.
이미 있는 747139 경기의 판단 프레임·릴리스 시각·pitch_id 연결(docs/MLB_P0.md, src/vision/*)을 재사용한다.

순서와 완료 기준:
M0) 출력 스키마를 IntentEstimate v1로 고정한다(아래 스키마). 검사 함수 validate_intent_estimate를 만들고
    정상·기권·보정 없음 3가지 가짜 입력과 잘못된 입력 거부 테스트를 추가한다.
M1) 747139의 297구 판단 프레임에서 미트 중심 픽셀을 찍는다(수동·반자동·간단한 검출 중 하나. method.kind에 정직하게 기록).
    홈플레이트 네 모서리(TL,TR,BR,BL)로 호모그래피를 만들어 image_pixels → annotated_image_zone으로 변환한다.
    한 줄 명령 `python -m intent.run --game 747139 --out intent_747139.jsonl`이 297줄을 내고
    모든 줄이 검사를 통과해야 한다. 판정 못 하면 status=unavailable과 이유를 쓴다. 걸린 시간을 기록한다.
    여기까지 되면 멈추고 결과(줄 수, 기권 수, 샘플 5줄, 소요 시간)를 보고한다.
M2) annotated_image_zone → plate_feet 변환(3×3 행렬, method, x_convention=statcast_plate_x_catcher_view,
    rms_error_feet, fit_evidence)을 만들고, 사람이 확인한 10구 이상에서 오차를 잰다. 못 재면 null.
M3) 다른 경기 2개 이상에서 사람이 찍은 50구 이상으로 픽셀·피트 오차와 기권률을 재서 accuracy_report.json을 만든다.
    개발 경기와 평가 경기를 분리한다.

IntentEstimate v1 스키마(필수 규칙):
{"schema_version":1,"pitch_id":"<game_pk>:<at_bat_number>:<pitch_number>","clip_id":"…","clip_sha256":"…",
 "status":"estimated|unavailable","unavailable_reason":null,
 "method":{"kind":"…","version":"…"},"evidence":{"frame_index":…,"frame_time":…},
 "provenance":{"label_source":"video_module|assistant_visual_estimate|human_manual_annotation","review_status":"unreviewed"},
 "deepest_frame":"annotated_image_zone|plate_feet","blocked_by":"no_plate_plane_calibration|null",
 "points":{"image_pixels":{"x":…,"y":…},"annotated_image_zone":{"x":…,"y":…,"inside_annotated_quad":true}},
 "transform_chain":[{"source_frame","target_frame","method","version","error","error_units","error_status","evidence"}…],
 "uncertainty":{"value":…,"units":"pixels","basis":"…"},
 "is_intent_proxy":true,
 "claims":{"catcher_intent_verified":false,"independent_ground_truth":false,"physical_plate_coordinates":false,"accuracy_estimate":null}}
- status에 observed는 없다. is_intent_proxy는 항상 true, catcher_intent_verified·independent_ground_truth는 false.
- points의 키는 실제로 도달한 좌표계와 정확히 같아야 한다. 홉을 건너뛰지 않는다. error를 모르면 null + error_status="unmeasured".
- 숫자를 지어내지 않는다. 영상은 MLB 권리이므로 저장·공개 범위를 README에 적는다.
```

## 받은 뒤 저희가 할 일

- M1 JSONL을 Pitcheezy `validate_intent_estimate`로 한 번 더 검사하고 관전 화면에 연결합니다.
- M2 이후에는 ARM-A 준비(의도 위치 로깅 법칙 κ)에 필요한 규모와 식별 조건(BLK-04)을 함께 검토합니다.
