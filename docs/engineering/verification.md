# 검사와 PR 증거

상위: [작업 절차](../pm/workflow.md) · 원칙: [구현](implementation.md)

worktree의 편집 설치가 다른 checkout을 가리킬 수 있으므로 현재 소스를 `PYTHONPATH` 앞에 둔다. 기존 환경을 재사용하고 누락 의존성을 임의 설치하거나 설정을 바꾸지 않는다.

```sh
# 저장소 루트. 환경 경로는 실제 사용 가능한 환경으로 지정한다.
PYTHONPATH=src:.:apps/observer/backend .venv/bin/pytest tests -q
PYTHONPATH=src:.:apps/observer/backend .venv/bin/pytest apps/observer/ops/tests -q
# 웹 디렉터리
cd apps/observer/web
npm run build
npm run build:public
node tests/<관련-검사>.mjs
```

observer 백엔드는 필요하면 `.venv-observer`를 사용한다. 실제 사용한 Python·Node·소스 커밋을 기록한다. 코드/CSS에 `merge=union`을 적용하지 않는다.

## 검사의 범위

변경한 경로와 의미 있는 회귀를 먼저 검사한다. 합친 소스에서는 영향받는 통합 검사도 수행한다. 실자료·모델이 필요한 검사와 합성 검사를 구분하고, 등록 없이 실자료 실행으로 넓히지 않는다. 모든 검사를 무조건 반복하지 않되 필수 게이트는 생략하지 않는다.

웹 릴리스의 기본 확인은 notice-hide, ws-launch-layout, zone-geometry, zone-label-clip, 관련 static/replay 스모크다. 구체 실행 명령·최종 수용은 [통합 담당 안내](../handoffs/OPUS-github-manager-prompt-2026-10-31.md)를 대조한다. 브라우저 모의 화면 통과를 실제 iPhone 검수로 부르지 않는다.

## PR에 남길 것

문제와 최종 동작, 기준/최종 커밋, 실제 검사 명령·통과/실패/skip 이유, 미측정·제한과 복구 조건을 쓴다. 부분 재검사로 과거 전체 실패를 지우지 않는다. 파일·ZIP 해시는 실제 원본 바이트에서 계산한다. 커밋은 `fix(observer): …`, `docs(pm): …`처럼 모듈 scope를 사용한다.

## 연구 증거

실험 설정은 `configs/`로 고정하고 파라미터를 하드코딩하지 않는다. 결과에는 seed·데이터 버전·소스 커밋을 남긴다. 기본 기록 경로는 config → runs의 ID/seed → results의 결과 JSON → 실험 원장이다. 탐색 {0,1,2}·채택 {0..4} 시드와 시드별 체크포인트는 해당 고정 등록을 대조하며, 문서 정리로 등록된 예외를 바꾸지 않는다.

공통 성능 비교는 재구현으로 하고 논문 보고값은 출처 있는 참고로 구분한다. notebook 출력만 결과 근거로 쓰지 않는다. 예측 후보 선정(NLL·Brier·보정)과 추천 정책 채택을 구분하며, 서비스 정책 채택은 e2e 정책 평가로 판단한다. 결과 열람 전에 분할·지표·확장 규칙을 고정하고 작은 실제 경로 검증 후 등록된 단계로 확장한다. 이 절은 실험 실행 승인이 아니다.
