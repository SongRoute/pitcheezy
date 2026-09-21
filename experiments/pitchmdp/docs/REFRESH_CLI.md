# 날짜별 프로필·구종 사이드카

기존 `minimal-pitch-service-v1` 번들을 수정하지 않고, 날짜별 타자 프로필과 지원 구종 빈도를 별도 JSON으로 저장한다. 승인된 2023–2025 원본과 처리 데이터의 해시를 읽기 전용으로 확인한다. 학습·다운로드·물리 특성 캐시 갱신은 하지 않는다.

프로젝트 루트에서 기존 가상환경을 사용한다. 출력은 마운트된 T7의 `runs` 바로 아래 **새 디렉터리**여야 한다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/refresh_pitch_service.py refresh \
  --as-of 2025-08-16 \
  --output '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/pitch-refresh-20250816-v1'

.venv/bin/python experiments/pitchmdp/scripts/refresh_pitch_service.py recommend \
  --snapshot '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/pitch-refresh-20250816-v1' \
  --request '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/my-input/request.json'
```

요청 형식은 기존 서비스와 같다. `date`는 이제 필수이며 모델·보정 자료 마감일인 `2025-06-30` 이후, 사이드카 `effective_date` 이상이어야 한다. `batter_profile`을 직접 제공하면 `rates`·`reliabilities` 각 6개와 요청일보다 이른 `as_of`가 필수다. 직접 제공한 자료의 실제 출처는 독립적으로 검증하지 않았다는 플래그를 반환한다.

`--as-of D`는 `game_date < D`인 전체 승인 투구 풀을 사용한다. 마지막 관측일의 모든 투구를 포함하도록 기존 `add_batter_style_history`에 관측량 0인 조회 행을 추가한다. 여섯 누적 수축 비율과 신뢰도 정의는 기존 파이프라인을 그대로 따른다. 알려지지 않은 타자는 이전 날짜 리그 비율과 신뢰도 0으로 처리하고 누락·낮은 신뢰도를 표시한다. `profile_cutoff`와 개별 프로필 `as_of`는 포함된 마지막 자료일이다.

투수 목록은 원래 지원 6명을 유지한다. 후보 구종은 원래 모델의 지원 구종 중 `[D - 90일, D)`에 20개 이상 관측된 것만 허용한다. 이 기간의 구종 빈도로 비교 정책을 갱신한다. 새로운 구종은 모델 지원에 추가하지 않는다. 조건에 맞는 구종이 없거나 투구 손 정보가 없거나 원래 지원과 충돌하면 해당 투수는 사용할 수 없으며 추천 요청이 오류를 반환한다. 오래된 구종으로 자동 복구하지 않는다. `pitchers`의 `flags`, `masked_pitch_types`, `unmodeled_recent_pitch_types`에 이유를 남긴다.

`snapshot.json`과 `snapshot_manifest.json`은 원래 메타데이터 및 자료 출처 해시, 자료 마감일, 조회 가능일을 기록한다. 다시 읽을 때 해시와 모델 지원 범위를 검사한다. 실행 중에만 엔진 메타데이터를 교체하며 성공·오류 모두 원상 복구한다. `recommend --output`을 지정하면 요청·응답을 별도 새 실행 디렉터리에 저장한다.

CLI `refresh`는 사용한 `refresh_pitch_service.py`와 `archetypes.py`의 사본을 실행 디렉터리의 `source/`에 보존하고 `source_hashes.json`에 SHA256을 기록한다. 같은 해시는 스냅샷의 `source_identity.refresh_code_sha256`에도 포함되어 자료와 프로필 계산 코드의 출처를 함께 고정한다.

독립 평가에서 채택한 10개 결과 절편 보정이 있을 때만 `recommend --calibration /absolute/path/calibration.json`을 사용할 수 있다. JSON 필드는 `accepted`, `bias`(10개), `available_from`, `original_metadata_sha256`이다. 채택되지 않은 후보는 적용하지 않는다. 채택된 보정은 지정 가용일 이후 요청에만 적용하며, 가용일은 보정 채택 게이트 다음 날인 `2025-08-16`보다 빠를 수 없다. 혼합 결과 확률에만 적용하고 구조적 확률 0을 보존한다.

2026년 날짜로 스냅샷을 만들어도 자료는 2025년까지다. 최근 관측 조건을 만족하지 못하는 투수는 사용할 수 없다. 실시간 서비스 성능이나 현실 승률 향상을 주장하지 않는다.
