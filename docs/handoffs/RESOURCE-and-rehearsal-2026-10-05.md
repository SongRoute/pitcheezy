# 자원 활용·리허설 계획 (2026-10-05)

작성: Claude Opus5.5 자원 감사 하위 에이전트(root 위임), 2026-10-05 00:31~00:40 KST 읽기 전용 관측. 이전 [RESOURCE-next-queue-2026-10-04.md](RESOURCE-next-queue-2026-10-04.md)를 이 문서가 이어받는다. 상세 근거: `/Users/song/Projects/pitcheezy-pm-jobs/20261005-resource-plan/REPORT-001.md`, `evidence-001.json`.

## Song 새 지시 (2026-10-05)

- **10/5 밤:** 맥북·아이폰 접속 연결과 5분 리허설을 한다.
- **10/6 오전~16:00 시연 전:** 리허설에서 발견한 문제를 고치는 버퍼로 쓴다. 미리 할 수 있는 일은 전부 그 전에 끝낸다. 기존 PM-state의 "10/6 14:00 내부 고정" 기준과의 조정은 root가 정한다.
- **이메일:** 급하지 않음, 보류. 권한 재연결 요청·재시도·다른 경로 사용을 하지 않으며 이메일은 더 이상 막힘 항목이 아니다.
- 모바일 UI 작성은 root가 별도 격리 작업트리로 관리한다. 연속 coordinator는 UI 파일을 중복 작성하거나 점유하지 않는다.

## 현재 개발 방식 (요약)

- root PM은 관리 전용, 조사·구현·실험·테스트·문서는 Opus5.5 하위 에이전트가 한다(358702e).
- GPU 실험은 newsrv **GPU3·GPU4만**, 등록 config·소스 핀·예산(36,000 worker-s, 작업당 최대 2회 시도)·`max_parallel=2`·GPU 단독 점유 lock 안에서 큐로 돈다. 2026 자료는 학습·선택에 쓰지 않는다.
- 맥미니는 서비스(8766/8771)·경기 기록·동기화를 우선하고 무거운 계산은 공통 `.heavy.lock` 아래에서만 한다.

## 관측 (읽기 전용, 00:31~00:36 KST)

- **P35(EXP-P35-001, 25 fit):** 10 완료(rc 0)·2 실행·13 대기. 완료 10개 실측 145/80(control), 260/185(periodic), 270/220(ple), 420/585(ple_tabm), 635/650(tabm)초. 큐 안에서 작업 사이 공백은 상태 기록상 0~수초.
- **GPU3·4:** `nvidia-smi dmon` 8회 SM 8~9%, 메모리 1.8GiB/24GiB, 73/71W. 학습 프로세스 CPU 각 ~100%(단일 코어 포화), RSS 5.0/6.5GB. 표본 시각에 두 프로세스는 시작 26~36초 차이였으므로 적재 단계와 섞였을 수 있다.
- **P34 끝→P35 시작 공백:** 10/4 20:17 GPU idle 기록 → P35 큐 생성 00:02:49. 약 3시간 45분 GPU 유휴. 원인은 "10/5 00:00 이후 첫 사이클" 날짜 가드(387acd7).
- **맥미니:** Mac16,10 Apple M4, CPU 10코어(성능 4+효율 6), GPU 10코어, 메모리 24GB, macOS 26.6.2. load 1.9, 메모리 여유 67%, 내장 69GiB·T7 3.3TiB 여유. `.heavy.lock` 없음(무거운 맥 작업 없음). DS 4경기 기록 `all.done` 존재. 로컬 torch 2.14.0 MPS 사용 가능. 맥 GPU의 실제 사용률은 미측정(powermetrics 미사용).

## 병목 가설 (소스 근거, 미검증)

학습 루프(`direct_residual_model.py`, `direct_tabm_model.py` fit; ple·periodic 계열은 이를 상속)는 학습 자료 전체를 GPU에 올려 두고 미니배치마다 ① numpy 순열 조각을 `torch.as_tensor(ids, device)`로 복사하고 ② `loss.item()`으로 CPU-GPU 동기화한다. periodic-29는 1,252,824행·배치 1024·12 epoch = 약 14,700 step을 203.5초에 돌았다(약 13.9 ms/step). 작은 모델에서 GPU SM 9%·CPU 단일 코어 100%와 맞는 설명은 "step마다 동기화와 Python·커널 실행 지시 부담"이다. 단, profiler로 확인하지 않았으므로 **가설**이다. DataLoader가 없으므로 `pin_memory`·비동기 복사 제안은 이 루프에 해당하지 않는다.

## 지금 할 수 있는 일 (P35 진행과 무관)

1. **P35 끝→채점→P36 대기 공백 제거:** P36 등록 pack·smoke·채점 경로를 P35 실행 중에 검토·핀 확인까지 끝낸다(진행 중). P35 보고가 사전 규칙(§7)을 만족하면 바로 투입. 날짜 가드처럼 GPU를 묶는 시각 조건은 새로 만들지 않는 쪽을 제안한다.
2. **맥 CPU로 시연 준비 전부 앞당기기:** 10/5 밤 리허설 전에 8771 시연본 빌드·브라우저 폭 검사·장면 자료 점검·회귀 테스트를 끝낸다. `.heavy.lock`이 비어 있으므로 맥 CPU 연구 작업(저장 확률 채점·자료 품질 검사)도 lock을 잡고 가능하다. 경기 기록·서비스가 우선이며, GPU fit을 맥에서 중복 실행해 사용률만 올리지 않는다.
3. **동기화 제거 성능 측정 초안(코드 작성만):** 아래 ①의 변경을 별도 브랜치에 작성하고 CPU 단위 검사까지 한다. 실제 GPU 측정은 P35가 끝난 뒤.

## P35 종료 후 측정할 후보 (새 조건, 등록·검토 뒤에만)

| 후보 | 기대 | 위험 | 통과 조건 |
|---|---|---|---|
| ① 동기화 제거: 손실 합을 GPU에서 누적해 epoch 끝에 한 번 읽기, epoch 순열을 한 번에 GPU로 복사 | step 시간 감소(크기 미측정) | 기록되는 train 손실의 부동소수 합 순서 차이 | 같은 시드 1 fit에서 기존 코드와 state_dict·조기종료 epoch·검증 NLL 동일(또는 기존 재실행 간 차이 이하), fit 시간 비교 |
| ② `torch.compile(mode="reduce-overhead")`(CUDA Graphs) | 커널 실행 지시 부담 감소 | 수치 차이, 컴파일 시간, torch 2.5.1 호환 | ①과 같은 parity·시간 측정. 차이가 있으면 새 실험 조건으로 따로 등록 |
| ③ 한 GPU에 2 fit | GPU 유휴 감소(메모리 1.6GB 피크) | 서로 느려짐, lock·예산 규칙 변경 필요 | 짧은 시간 측정으로 두 fit 합산 처리량이 단독 대비 높을 때만, 등록 변경으로 |

큰 배치·AMP는 학습 결과가 달라지는 **새 실험 조건**이다. 속도 개선으로 섞지 않는다. 맥 MPS는 별도 parity·속도 측정(짧고 제한된)을 거친 뒤에만 쓰고, CUDA 시드 비교와 같은 표에 넣지 않는다.

## 누가·언제

- root: 이 문서 읽고 우선순위 결정. 10/5 밤 리허설 일정 관리.
- 연속 coordinator(v5): P35 완료 확인 → 채점·보고 → §7 규칙이면 P36. ① 초안은 Opus 하위 에이전트에 위임. UI 파일 점유 금지.
- P35 예상 종료: 완료 10개 평균으로 남은 15 fit ≈ 5,200 worker-s ÷ 2 GPU → **약 01:15~01:35 KST(추정)**.

## 공식 문서

- PyTorch Performance Tuning Guide — 불필요한 동기화(`.item()`) 피하기, 대상 장치에서 바로 텐서 만들기, CUDA Graphs(`torch.compile` reduce-overhead), AMP: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Apple — PyTorch MPS(베타): https://developer.apple.com/metal/pytorch/
