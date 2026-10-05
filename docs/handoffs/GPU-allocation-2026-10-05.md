# GPU 배정 — 2026-10-05 (Song 최신 지시, 1장만)

Root 위임으로 Claude Opus5.5(job 20261005-single-gpu/author-001)가 작성. Root PM은 관리자 전용이고 실행은 서브에이전트가 한다. Opus5.5 우선, 실제 Opus 사용량 소진 때만 GPT Sol 대체 규칙은 그대로다. 이 문서는 자원 규칙만 바꾸며 과학 판정·정책 채택은 없다.

## 갱신 — 맥미니 MPS 병행 (Song 승인, Root가 2026-10-05 19:07 KST 전달; Root 위임으로 Claude Opus5.5 mps-lane/author-001 작성)
- Song이 Root 추천을 승인했다: **P38은 newsrv GPU3에 그대로 두고, 맥미니 MPS로 별도 추가 실험을 한다.** 아래 "현재 한도"의 "Song 승인 없이 다시 늘리지 않는다"는 이 승인 범위만큼만 바뀐다.
- **newsrv**: 여전히 GPU3(UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`) 한 장, 공통 배타 lock 필수. GPU4(`GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`)는 반납 상태 그대로(할당·probe·대체 금지). 현재 P38은 동시 1fit 그대로다.
- **맥미니 MPS**: newsrv GPU3 fit 1개와 **동시에 MPS fit 1개**를 돌릴 수 있다. 조건: 경기 기록기 우선(기록 중이거나 상태를 확인할 수 없으면 무거운 MPS 작업·대용량 전송 금지), 공통 `.heavy.lock`(`/Volumes/T7 Shield/pitcheezy/pitchmdp/runs/ML-MATRIX-20260924/.heavy.lock`) flock, MPS 명시·CPU 대체 금지, 8766/8770/8771·시연·기록기 보호. 실행기는 `mps_direct_v1`(별도 신원, `codex/mps-research-lane-1005`).
- MPS 결과는 MPS 안에서만 비교한다. CUDA 동시 실행·처리량 검증 근거로 쓰지 않고, CUDA와 MPS 차이를 모델 개선으로 읽지 않는다.
- **같은 GPU3에 2fit**은 앞으로도 검토된 스케줄러(소유권·정리·시간 제한·신원 통제)와 실측 bench를 통과한 뒤에만 허용된다. 그 전에는 GPU3 동시 1fit.
- 그 밖의 확장(다른 newsrv 카드, 맥 MPS 2fit 등)은 여전히 Song 승인이 필요하다.

## 현재 한도 (이전 2fit·GPU3/4 허용을 대체)
- 다른 프로젝트가 GPU 한 장이 필요하다(Song, 2026-10-05). Pitcheezy는 **GPU 1장, 동시 fit 최대 1개**만 쓴다.
- **유지**: 물리 GPU3, UUID `GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9`. 기존 GPU3 공통 배타 lock은 계속 필수다.
- **반납**: 물리 GPU4, UUID `GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99`. Pitcheezy는 할당·probe·smoke·예약·대체 사용을 하지 않는다. GPU3가 바쁘면 GPU4로 넘어가지 않고 기다린다.
- 장치 번호(nvidia-smi index, CUDA ordinal)는 바뀔 수 있다. 실행 전 UUID로 GPU3인지 확인하고 `CUDA_VISIBLE_DEVICES`는 그 장치 하나만 가리킨다.
- Song 승인 없이 다시 2장으로 늘리지 않는다. 이전 2장 런북·큐 설정은 새 2장 사용 권한이 아니다.
- 끝난 실험(P30–P36 등)의 2장 기록·config는 당시 이력이며 고치지 않는다.

## 최근 관측 (읽기 전용, 2026-10-05 14:15:32–14:15:38 KST 서버 시각)
- GPU3 `GPU-c334d905…`: 사용률 0 %, 메모리 2 MiB, compute 프로세스 없음.
- GPU4 `GPU-ae803a24…`: 사용률 0 %, 메모리 2 MiB, compute 프로세스 없음.
- Pitcheezy 큐·fit 프로세스 없음. GPU0–2는 다른 프로젝트 학습이 사용 중이며 손대지 않았다.
- 순간 표본일 뿐 예약이 아니다. GPU4는 이제 다른 프로젝트 몫이므로 "비어 있음"을 Pitcheezy 가용으로 읽지 않는다.

## 다음
- EXP-P37-001 DRAFT(`codex/next-research-1005`)를 GPU3 단독·동시 1fit으로 고친 뒤 별도 검토 → Root 수용 → smoke → 등록. 실행 중 아님.
- 거절된 code-f429318 검증과 멈춘 v5는 이 변경과 무관하게 계속 막혀 있다.
