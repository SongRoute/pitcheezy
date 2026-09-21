# 최소 구종 추천 서비스

타자 성향, 카운트, 주자·아웃, 이닝, 점수, 홈원정을 입력하면 전체 타석의 수비 승리 확률을 기준으로 구종 후보를 반환한다. 기존 연속 성향 MLP 5개와 빈도 모델을 재사용한다. 이전 투구 이력은 사용하지 않는다. 대상은 TRAIN 이닝으로 선정한 6명이고 구종 목록은 `/metadata`에서 확인한다.

저장 번들: `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1`

## 실행

프로젝트 루트 `/Users/song/Projects/pitcheezy`에서 기존 가상환경으로 실행한다. 추가 설치·학습은 필요 없다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/minimal_pitch_service.py \
  --bundle '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1'
```

다른 터미널에서 실제 데이터 예제를 전송한다.

```sh
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/metadata
curl -s http://127.0.0.1:8765/recommend \
  -H 'Content-Type: application/json' \
  --data-binary '@/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1/example_request.json'
```

서버 없이 같은 요청을 처리할 수도 있다.

```sh
.venv/bin/python experiments/pitchmdp/scripts/minimal_pitch_service.py \
  --bundle '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1' \
  --request '/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/minimal-pitch-service-v1/example_request.json'
```

## 입력·출력

필수 입력은 `inning`, `topbot`(Top/Bot), `outs`(0–2), `bases`, `home_score`, `away_score`, `balls`(0–3), `strikes`(0–2), `pitcher_id`, `batter_stand`(L/R)다. `bases`는 1루=1, 2루=2, 3루=4를 더한다. 예: 1·3루=5, 만루=7.

타자는 `batter_id`로 TRAIN 시점 프로필을 조회하거나 `batter_profile`의 `rates`·`reliabilities` 각 6개를 직접 제공한다. 순서는 `/metadata`의 `profile_order`를 따른다. 조회되지 않는 타자는 신뢰도 0의 기본 프로필을 쓰고 응답에 표시한다. 직접 제공한 프로필은 반드시 요청 시점 이전 데이터로 계산해야 한다. 선택 입력 `date`가 있는 스냅샷 조회에서는 프로필 날짜 이후 요청인지 검사한다. 번들 스냅샷은 실시간 갱신되지 않는다.

응답의 `recommendations`에는 구종, 모델 내부 수비 WE, TRAIN 구종 빈도 정책 대비 차이, 다음 공의 10개 결과 확률이 포함된다. 후속 카운트에서도 추천 정책을 따른다는 가정이다. `top_k` 기본값은 3이다. 지원하지 않는 투수나 잘못된 입력은 HTTP 400이다.

## 해석과 범위

동결된 역사 데이터 모델을 로컬에서 호출하는 연구 MVP다. 구종만 추천하며 목표 위치, 실시간 수집, 타석 도중 도루·교체, 피로 변화는 모델링하지 않는다. 기본 CPU 추론이며 `--device mps`는 선택 가능하지만 별도 성능 검증 대상이다. API는 localhost에만 바인딩한다.

`evaluation.json`에 기존 관측 예측 성능, 서비스 확률 재현 오차, WE 최소 점검, RE/WE×한 구/타석 계획 비교가 있다. `example_response.json`에 실제 추천 예시를 저장한다. **모델 내부 WE 차이는 현실의 승률 향상 검증이 아니다.** 빈도 예측기로 다시 평가한 결과도 독립적인 정책 평가로 간주하지 않는다.

설계 범위는 [실행 프로토콜](docs/MINIMAL_SERVICE_PROTOCOL.md), 최종 결과는 [결과 보고서](MINIMAL_SERVICE_RESULTS.md)를 따른다. 기존 완료 실험과 번들은 수정하지 않는다.
