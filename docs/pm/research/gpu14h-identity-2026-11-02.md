# 14시간 GPU 연구 005 — 투수 ID의 증분가치

현재 source 사전등록 DRAFT이며 실자료 실행0이다. 원003의 MLP/T32/C32에 FIT_E9에서만 만든16차원 투수 ID를 추가한다. 새 부분은 원9종 logits에 더하는 저차원 투수별 offset이며 초기0/RNG보존, unknown0이다. 직접적인 ID×상태 상호작용 모델로 해석하지 않는다.

주대조는 같은 architecture/fold/seed의 원003 no-ID 예측이다. 새로운3조건×3foldseed0=9, 완결 paired ΔNLL<0/Brier≤0 중 고정순위 최대1에 seed1/2 6개를 추가한다. 원003이 선정하지 않았고 해당원task claim도 없는 no-ID 대조만005에서 정확원trainer로 최대6개 완성한다. 기존/선정/실행중/실패 원대조의 복제·재시도0, 총신규21fit상한이다. 003동결빈도는 보조대조로 재사용하며 새countfit0이다.

state11/strictprior32/9종 map/원package/30epoch/optimizer는 고정이다. 원 Q_pkg는 과거 outcome 적격으로 선택된 분모이고 mapped E9_pkg 채점이며 전체MLB투구Q가 아니다. known/cold-start/미매핑 census를 보존한다. 2026 EXPOSED DEVELOPMENT/최초수신가용성UNKNOWN, seed선정편향·다중후보노출을 유지한다. CAL/FINAL/고정6경기/봉인/OPE/서비스교체0이다.

[전체 사전등록](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch5/REGISTRATION.md), [작성 계획](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch5/PLAN.md), [생성 검사](/Users/song/Projects/pitcheezy-pm-jobs/20261102-gpu14h/author-batch5/synthetic.log). 새소스3개 외 원003 source50은 불변이고 shared14h 5GPU/총6GiB/24GiBfree/마감·기록기 우선 한도 안에서 별도운영등록 후 실행한다.
