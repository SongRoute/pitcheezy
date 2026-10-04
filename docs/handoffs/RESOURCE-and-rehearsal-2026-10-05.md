# 자원 활용·리허설 계획 (2026-10-05)

작성: Claude Opus5.5 자원 감사 하위 에이전트(root 위임). 2026-10-05 00:31~00:40 KST 읽기 전용 관측, 같은 날 root 검토에 따라 정정. 이전 [RESOURCE-next-queue-2026-10-04.md](RESOURCE-next-queue-2026-10-04.md)를 이 문서가 이어받는다. 상세 근거: `/Users/song/Projects/pitcheezy-pm-jobs/20261005-resource-plan/REPORT-001.md`, `evidence-001.json`.

## Song 새 지시 (2026-10-05, 최신 지시가 우선)

- **10/5 밤:** 맥북·아이폰 접속 연결과 5분 리허설을 한다(정확한 시각은 정하지 않음).
- **10/6 오전~16:00 시연 전:** 리허설에서 발견한 문제를 고치는 버퍼로 쓴다. 미리 할 수 있는 일은 그 전에 끝낸다. 기존 "10/6 14:00 내부 고정"은 점검 목표 시점이며, 필요한 수정을 막는 금지선이 아니다. 시연본 보존·되돌리기 보호는 그대로 둔다.
- **이메일:** 급하지 않음, 보류. 권한 재연결 요청·재시도·다른 경로 사용을 하지 않으며 막힘 항목이 아니다.
- 모바일 UI 작성은 root가 별도 격리 작업트리로 관리한다. 연속 coordinator는 UI 파일을 중복 작성하거나 점유하지 않는다.

## 현재 개발 방식 (요약)

- root PM은 관리 전용, 조사·구현·실험·테스트·문서는 Opus5.5 하위 에이전트가 한다(358702e).
- GPU 실험은 newsrv **GPU3·GPU4만**, 등록 config·소스 핀·예산(36,000 worker-s, 작업당 최대 2회 시도)·`max_parallel=2`·GPU 단독 점유 lock 안에서 큐로 돈다. 2026 자료는 학습·선택에 쓰지 않는다.
- 맥미니는 서비스(8766/8771)·경기 기록·동기화를 우선하고, 무거운 계산은 공통 `.heavy.lock`을 잡고 기록·동기화 gate를 통과한 뒤에만 한다.

## 관측 (읽기 전용, 00:31~00:40 KST)

- **P35(EXP-P35-001, 25 fit):** 00:31에 10 완료(rc 0)·2 실행·13 대기. 완료 10개 실측 145/80(control), 260/185(periodic), 270/220(ple), 420/585(ple_tabm), 635/650(tabm)초. 큐 상태 기록상 작업 사이 공백은 0~수초.
- **GPU3·4 짧은 표본:** `nvidia-smi dmon` 1초×8회 SM 8~9%, 메모리 1.8GiB/24GiB, 73/71W, 학습 프로세스 CPU 각 ~100%. 표본 시각에 두 worker는 시작 후 26~36초였다. **시작 직후 8초 표본이므로 학습 정상 구간의 병목을 입증하지 않는다.**
- **P34 이후 시각 기록:** P34 채점 완료 10/4 17:34:12, P30-002는 약 19:13 종료(root 검토 기준), P35 시간 측정 19:53~19:58, 20:17 GPU3·4 idle 관측, P35 큐 생성 00:02:49. 20:17→00:02는 idle 관측과 시작 기록 사이의 구간이며, 그동안 계속 idle이었는지는 연속 확인하지 않았다. P35 시작이 "10/5 00:00 이후 첫 사이클" 날짜 가드(387acd7)에 묶여 있었던 것은 기록으로 확인된다.
- **맥미니:** Mac16,10 Apple M4, CPU 10코어(성능 4+효율 6), GPU 10코어, 메모리 24GB, macOS 26.6.2. load 1.9, 메모리 여유 67%, 내장 69GiB·T7 3.3TiB 여유. 로컬 torch 2.14.0 MPS 사용 가능. 맥 GPU 사용률은 미측정.
- **맥 공통 lock:** 경기 후 큐(`pm_postgame_queue.py`, manifest `PM-POSTGAME-FULL-20261004.json`)의 작업 경로로 본 공유 lock은 `/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/.heavy.lock`이다. 파일은 있고, 00:39 `lsof`에 여는 프로세스가 없었다. 이것은 **그 순간의 표본일 뿐 예약이나 사용 허가가 아니다.** 무거운 작업은 이 lock을 `flock`으로 직접 잡고 큐 gate(기록 프로세스 없음, `/api/health` 정상, 다음 동기화까지 여유)를 통과해야 한다.
- **기록·동기화 창:** `/tmp/pitcheezy-demo-sync.log` 수정 00:17, 30분 주기 규칙이면 다음 동기화는 00:47 이후. 다음 경기 기록 일정은 읽을 수 있는 설정에서 찾지 못해 **미확인**. DS 4경기 `all.done`은 지난 기록이 끝났다는 뜻일 뿐 무거운 작업 허가가 아니다.

## 병목 가설 (소스 근거, 미검증)

현재 P35가 쓰는 서버 `code-8864389`의 `direct_residual_model.py`·`direct_tabm_model.py`·`direct_ple_model.py`·`direct_periodic_residual_model.py` sha256이 main `8864389`와 같음을 확인했다. 이 학습 루프는 학습 자료 전체를 GPU에 올려 두고 미니배치마다 ① numpy 순열 조각을 `torch.as_tensor(ids, device)`로 복사하고 ② `loss.item()`으로 CPU-GPU 동기화한다(ple·periodic 계열은 상속). periodic-29는 1,252,824행·배치 1024·12 epoch ≈ 14,700 step을 203.5초에 돌았다(≈13.9 ms/step). "step마다 동기화와 실행 지시 부담"은 이 소스와 맞는 **가설**이며, 효과 크기는 parity·시간 측정으로만 확인한다. DataLoader가 없으므로 `pin_memory`·비동기 복사 제안은 이 루프에 해당하지 않는다.

## Root 결정 (2026-10-05)

- **먼저:** 학습 인덱스 복사·손실 동기화 부담 제거(①). 별도 throughput 작성자가 runtime-overhead parity/처리량 프로토타입을 지금 준비하고, P35가 GPU와 공통 lock을 놓은 뒤 측정한다. 다음 준비된 연구 작업을 불필요하게 막지 않을 때만 끼운다. Song 일반 승인 대기는 없다.
- **나중 후보(측정 뒤에만):** `torch.compile`·AMP·배치 변경·한 GPU 동시 fit·MPS. 속도 수치를 미리 약속하지 않는다.
- **유지:** 현재 P35 설정, 총 2 fit 동시 상한, GPU 단독 점유 lock.
- 한 GPU 2 fit 측정은 다른 GPU를 비워 두면 "총 2 fit"을 지킬 수 있다. 운영에서 4 fit으로 늘리려면 별도 검토된 새 프로토콜이 필요하며 이번 감사 범위가 아니다.
- 측정 계획(고정 비용)은 작성자가 실행 전에 확정한다.

## 다음 측정 방법

SM %를 목표로 삼지 않는다. 같은 소스·같은 자료·같은 GPU에서 warmup을 두고 **작업 전체 소요 시간(fit wall·step 시간)**과 결과 동일성(같은 시드 state_dict·조기종료 epoch·검증 NLL, 또는 기존 재실행 간 차이 이하)을 함께 본다.

| 후보 | 기대 | 위험 | 통과 조건 |
|---|---|---|---|
| ① 동기화 제거(손실 GPU 누적·epoch 순열 1회 복사) | step 시간 감소(크기 미측정) | train 손실 합산 순서의 부동소수 차이 | 위 결과 동일성 + 작업 소요 시간 비교 |
| ② `torch.compile(mode="reduce-overhead")` | 실행 지시 부담 감소 | 수치 차이·컴파일 시간·torch 2.5.1 호환 | ①과 같은 방법. 차이가 있으면 새 실험 조건으로 등록 |
| ③ 한 GPU에 2 fit(다른 GPU 비움) | 유휴 감소(피크 1.6GB) | 서로 느려짐 | 합산 처리량이 단독보다 높을 때만, 운영 반영은 새 프로토콜 |

큰 배치·AMP는 학습 결과가 달라지는 **새 실험 조건**이다. 맥 MPS는 별도 짧은 parity·속도 측정 뒤에만 쓰고 CUDA 시드 비교와 섞지 않는다.

## 지금 할 수 있는 일

1. **P35 → 채점 → P36 공백 줄이기:** P36 등록 pack·smoke·채점 경로를 P35 실행 중에 검토·핀 확인까지 끝내고, P35 보고가 §7 규칙을 만족하면 바로 투입(coordinator).
2. **맥 가벼운 작업으로 시연 준비 앞당기기:** 10/5 밤 리허설 전에 8771 시연본 빌드·브라우저 폭 검사·장면 자료 점검·회귀 테스트. 맥 무거운 연구 작업은 공통 lock과 기록·동기화 gate를 통과한 뒤에만. GPU fit을 맥에서 중복 실행해 사용률만 올리지 않는다.
3. **① 프로토타입 준비:** root가 배정한 throughput 작성자가 별도 브랜치에서 작성·CPU 검사. GPU 측정은 P35 종료 뒤.

P35 예상 종료: 완료 10개 평균으로 남은 15 fit ≈ 5,200 worker-s ÷ 2 GPU → 약 01:15~01:35 KST(추정).

## 공식 문서

- PyTorch Performance Tuning Guide — 불필요한 동기화(`.item()`) 피하기, 대상 장치에서 바로 텐서 만들기, CUDA Graphs(`torch.compile` reduce-overhead), AMP: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Apple — PyTorch MPS(베타): https://developer.apple.com/metal/pytorch/
