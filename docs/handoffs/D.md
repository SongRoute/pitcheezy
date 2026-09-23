# D — Observer 첫 실제 연동 증분

2026-09-23. 작업 브랜치 `codex/d-observer-v1`, 시작 기준 `16b30cf`. C1 계약 `7075730`, C 계산 구현 `ef8c9e9`에 맞춰 D1 화면과 사건 카드·저장을 연결했다. D 브랜치는 C 파일을 복제하거나 cherry-pick하지 않았다. **통합 시 C 계약·구현을 먼저 반영해야 한다.**

## 화면·API

- 투구 전에는 저장된 1순위 **구종+목표 위치**를 큰 제안으로 보여준다. 2·3순위, 근거, 모델 내부 확률/차이는 펼치기에서 확인한다.
- 실제 공을 공개한 뒤에는 선택된 그 공에 저장된 사전 추천과 실제 구종·존 위치를 같은 줄과 포수 시점 존에서 비교한다. 위치 결측은 `위치 정보 없음`; 미래 공 정보/총 길이는 계속 숨긴다.
- 타석 완료 뒤 `event_analysis`는 CV `analysis`와 독립된 `event-analysis-v1` 결과다. 실제 의도 없는 자료에서는 고정된 수비 승률 모델의 부호 있는 전체 차이만 `partial`로 보여주고 선택/실행/타자 몫과 비중은 비운다. 남은 차이는 미배분이다. 개발용 합성 수치는 실제 사건 기여로 표시하지 않는다.
- 추천 생성시 UTC 저장시각을 별도 기록한다. 사건 결과는 `(session_id,pitch_id,revision)`로 별도 저장하며 recommendation 행은 그대로 유지한다. 수동 목표 메모는 사건 의도로 소비하지 않는다. CV no-media 워커 상태는 WE 계산의 성공/실패 조건이 아니다.
- 새 추천에는 동결 모델 SHA, 추천 어댑터 identity, WE 가치·기준 정책 버전을 함께 저장한다. 사건 계산 시 현재 엔진과 같지 않거나 예전 추천에 이 정보가 없으면 저장 기준값을 새 엔진 값으로 오인하지 않고 전체 비교를 unavailable로 둔다. 사건 결과 저장은 세션/투구/추천 ID·SHA·양의 revision을 확인하며 동일 revision의 서로 다른 결과를 거부한다. 계산 실패 결과는 C의 표준 `failed_event`가 만든다.

## 실제 연결 확인

독립 실행 위치 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/D-observer-v1/`; 기존 Observer dataset/manifest를 복사하고 D 전용 DB/추천 캐시를 사용했다. standalone frozen model과 C `event_analysis.py`를 Python 패키지 검색 경로에 추가해 통합 전 실제 계산 경로를 점검했다. `2025-08-17` 경기 `776703`, 타석 3의 삼진 3구를 pre→reveal→event까지 진행했다. 첫 사전 추천 `ready`, 마지막 공과 사건에 연결된 저장 추천 ID 일치, `partial` 전체 `+1.2021670371 %p` (초기 수비팀 기준), 선택·실행·결과 잔차 null, 미배분 잔차 `+1.2021670371 %p`, 비중 null. 이는 결과 모델 비교이며 기여/성능 검증이 아니다.

identity 보강 뒤 같은 실제 타석을 새 세션으로 다시 재생해 `partial +1.2021670371 %p`, `reference_status=compatible`, 마지막 공 추천 ID 일치를 확인했다. 마지막 공의 저장 추천 뒤 현재 추천 어댑터 identity를 바꾼 별도 재생에서는 `unavailable`, reference/total null, `reference_status=saved_evaluator_identity_missing_or_changed`를 확인했다.

검증: `npm run build` 통과; `PYTHONPATH=.:apps/observer/backend /Users/song/Projects/pitcheezy/.venv-observer-standalone/bin/python -m pytest apps/observer/backend/tests/test_observer_service.py -q`는 D 단독 브랜치에서 13 통과·2 건너뜀(C 모듈 없음). C 모듈을 패키지 검색 경로에 추가해 같은 15개 전부 통과했다. 테스트는 사전 비공개·동일 공 저장 추천·CAS·재시작·수동 메모/워커 격리, 사건 버전/링크 보존·실패 표준형·identity 변경을 확인한다. 루트 통합 체크아웃의 브라우저 실제 시각 검증은 A가 수행한다.

## 남은 범위

실제 CV 의도는 아직 없고, 교체/이닝 종료 비교 자료·평가기 또한 없다. 기존 세션은 사전 추천의 실제 저장시각을 복원할 수 없어 사건 결과가 null이다. 새 사건 입력 정정을 제공하는 CV/원자료 어댑터는 미연결이며, DB는 새 분석 revision을 추천 변경 없이 받을 수 있다. 새 추천 모델 학습/품질 우위, 2026 평가, 원본 모델·동결 자료 변경은 이번 D 증분 밖이다.
