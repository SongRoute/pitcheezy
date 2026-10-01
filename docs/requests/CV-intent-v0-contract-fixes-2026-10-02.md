# CV 의도 모듈 v0 — 계약 맞춤 수정 요청 (2026-10-02)

대상: `Pitcheezy/transition-models` 브랜치 `feature/intent-v0`, 커밋 `6d27197` 기준.
전송은 Song이 한다. 팀원 레포는 읽기만 했고 아무것도 수정하지 않았다.

## 확인한 것

- `docs/results/mlb_p0/game_747139_intent_v0.jsonl` 297줄을 서비스 쪽 검사 함수
  (`apps/observer/backend/observer_app/intent.py`의 `validate_intent_estimate`)에 그대로 넣었다.
- 결과: **0/297 통과.** 아래 필수 3건만 고친 사본으로는 **297/297 통과**(메모리 안에서만 바꿔 확인).
- 줄 구성: 추정 233(`plate_feet`까지 153, `image_pixels`에서 멈춤 80), 기권 64.

## 원인은 우리 작업 지시서다

2·3·4번은 지시서(`docs/INTENT_V0_WORK_ORDER.md`)의 스키마 요약이 빠뜨린 부분이다.
요약에는 `plate_feet` 점의 키가 없었고, `blocked_by`를 `no_plate_plane_calibration|null`로만 적었다.
팀원 구현은 지시서를 그대로 따랐다. 계약 원본(`docs/intent-spec.md`)의 규칙은 아래와 같다.

## 필수 수정 3건

1. **`evidence.frame_index`를 0 이상 정수로** (297줄 모두 지금 `null`)
   - 뜻: 원본 영상에서 그 프레임의 번호(0부터). 우리 쪽 시각 기준은 "고정 프레임률, 번호 ÷ fps"다.
   - 추출할 때 번호를 알면 그 값을, 시각으로만 뽑았다면 `round(frame_time × 60000/1001)`을 넣는다(747139는 59.94 fps).
     시각에서 유도했다는 사실은 `.run.json`의 `caveats`에 한 줄 적는다.
   - 위치: `intent/schema.py` `_base(... frame_index=None)`와 이를 부르는 `intent/run.py`. 검사(`schema.py:111–121`)도 `null`을 거부하도록 바꾼다.

2. **`points.plate_feet`의 키를 `x, z`로** (153줄, 지금 `x, y`)
   - Statcast의 `plate_x`, `plate_z`와 같은 이름이다. `x_convention`은 그대로 둔다.
   - 위치: `intent/schema.py:293–297`. `plate_feet`의 `"y"`를 읽는 다른 곳(`intent/calibrate.py`, `intent/human_labels.py`, 테스트)도 같이 바꾼다.
   - `image_pixels`와 `annotated_image_zone`은 `x, y` 그대로다.

3. **`plate_feet`에서 멈춘 줄의 `blocked_by`를 `"no_batter_zone_bounds"`로** (153줄, 지금 `null`)
   - 규칙: `deepest_frame`이 `zone9`가 아니면 무엇이 막았는지 반드시 적는다. `plate_feet` 다음 홉(9구역)은 타자별 존 상·하한이 필요하고 그것이 없어서 멈춘 것이다.
   - 위치: `intent/schema.py:305`, 검사 `schema.py:151–152`(허용 목록)와 `206–214`.

## 권장 1건 (검사는 지금도 통과)

4. **`image_pixels`에서 멈춘 80줄의 `blocked_by`를 `"no_image_plane_calibration"`으로**
   - 지금은 `no_plate_plane_calibration`인데, 이 줄들은 앞선 모서리가 없어 홉 ①을 못 간 경우라 계약상 이름이 다르다.
   - 정리하면 `blocked_by`는 멈춘 프레임에 따라 정해진다:
     `image_pixels` → `no_image_plane_calibration`, `annotated_image_zone` → `no_plate_plane_calibration`, `plate_feet` → `no_batter_zone_bounds`, 기권 → `null`.

## 완료 기준

- 팀원 레포의 `validate_intent_estimate`와 테스트가 위 규칙을 강제한다(`frame_index` null 거부, `plate_feet` 키 `x,z`, 멈춘 프레임별 `blocked_by`).
- `python -m intent.run --game 747139 --out docs/results/mlb_p0/game_747139_intent_v0.jsonl`을 다시 돌려 297줄을 커밋한다. 좌표 값 자체는 바뀌지 않아야 한다.
- 커밋 해시를 알려 주면 우리가 서비스 쪽 검사로 다시 확인하고 관전 화면에 연결한다.

## 팀원이 할 일이 아닌 것 (서비스 쪽 숙제)

- 홉 ①을 네 모서리 호모그래피에서 "플레이트 앞선 기준 유사변환 v1"으로 바꾼 것은 그대로 둔다(미트가 플레이트 평면 위에 없다는 이유가 타당하다).
  다만 우리 스펙은 `annotated_image_zone`을 단위 정사각형으로 적고 있어, v가 0~3까지 가는 새 정의에 맞춰 스펙을 고칠지 우리가 정한다.
- 9구역(홉 ③)은 타자별 존 상·하한을 우리가 붙여서 계산한다.

## 보낼 메시지 (메신저용)

```text
intent v0 M0~M2 잘 받았습니다. 297구 결과를 저희 서비스 검사 함수에 넣어 보니 형식 3가지가 달라서 0/297 통과였고, 그 3가지만 고치면 297/297 통과합니다.
2·3번은 저희 작업 지시서 요약에 빠져 있던 내용이라 저희 쪽 누락입니다.

1) evidence.frame_index: null → 0 이상 정수(원본 영상 프레임 번호, 시각으로 뽑았으면 round(frame_time × 60000/1001))
2) points.plate_feet 키: x, y → x, z (Statcast plate_x/plate_z와 같은 이름)
3) plate_feet에서 멈춘 줄의 blocked_by: null → "no_batter_zone_bounds"
(권장) image_pixels에서 멈춘 80줄의 blocked_by: "no_image_plane_calibration"

좌표 값은 그대로 두고 형식만 바꿔 다시 뽑아 커밋해 주시면, 저희가 다시 검사해서 관전 화면에 연결하겠습니다. 아래 프롬프트를 Claude Code에 그대로 넣으셔도 됩니다.
```

## 붙여 넣을 프롬프트 (팀원 Claude Code 세션용)

```text
feature/intent-v0 브랜치에서 IntentEstimate v1 출력을 서비스 쪽 계약에 맞춘다. 좌표 계산은 바꾸지 않고 형식만 바꾼다.

1) evidence.frame_index는 0 이상 정수여야 한다(null 금지). 원본 영상의 프레임 번호(0부터)이며,
   시각으로만 프레임을 뽑았다면 round(frame_time × 60000/1001)을 쓴다. 유도 방식은 .run.json caveats에 한 줄 적는다.
   intent/schema.py의 _base와 검사, intent/run.py의 호출부를 고친다.
2) points.plate_feet의 키는 x, z, x_convention이다(y가 아니라 z). image_pixels와 annotated_image_zone은 x, y 그대로.
   plate_feet의 "y"를 읽는 intent/calibrate.py, intent/human_labels.py, 테스트도 같이 고친다.
3) blocked_by는 멈춘 프레임으로 정한다:
   image_pixels → "no_image_plane_calibration", annotated_image_zone → "no_plate_plane_calibration",
   plate_feet → "no_batter_zone_bounds", status=unavailable → null.
   deepest_frame이 zone9가 아닌 추정 줄에서 blocked_by가 비어 있으면 검사가 거부해야 한다.

validate_intent_estimate와 테스트가 위 세 규칙을 강제하게 고친 뒤,
`python -m intent.run --game 747139 --out docs/results/mlb_p0/game_747139_intent_v0.jsonl`을 다시 실행한다.
297줄, 추정 233(plate_feet 153, image_pixels 80), 기권 64가 그대로이고 좌표 값이 이전과 같은지 확인하고,
테스트를 돌린 뒤 커밋·푸시하고 커밋 해시를 보고한다. main은 수정하지 않는다.
```
