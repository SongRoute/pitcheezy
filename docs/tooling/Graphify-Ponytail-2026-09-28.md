# Graphify·Ponytail 적용 기록 — 2026-09-28

공식 README의 Codex 절을 읽고 CLI 도움말과 설치 범위를 확인했다. Graphify0.9.70을 `uv tool install graphifyy`로 격리 설치하고 `graphify install --project --platform codex`로 이 저장소에 Codex 스킬을 등록했다. 프로젝트 .codex/hooks.json의 PreToolUse는 공식 구현의 no-op이다. 전역 AGENTS.md나 실험 Python 환경은 변경하지 않았다.

공식 출처: [Graphify README](https://github.com/Graphify-Labs/graphify#install), [Ponytail README](https://github.com/DietrichGebert/ponytail#codex). 확인한 commit·명령·해시는 함께 있는 graphify-installation-2026-09-28.json에 기록했다.

## 도구 적용으로 추가·수정한 파일

- `.gitignore`: 사용자 승인에 따라 `graphify-out/` 제외.
- `AGENTS.md`: 기존 프로젝트 지침을 보존하고 공식 Graphify 탐색 규칙을 추가. 사용자 승인 후 Ponytail 기본 규칙과 연구 예외를 병합.
- `.graphifyignore`: 데이터·미디어·실험 산출물·의미 분석 문서 제외.
- `.codex/hooks.json`: 공식 Graphify Codex hook 등록.
- `.codex/skills/graphify/SKILL.md`와 `references/`의 공식 참고 파일 8개: 프로젝트 Codex 스킬.
- `docs/tooling/AGENTS-ponytail-proposal.md`: 기존 지침 + Graphify + 공식 Ponytail 기본 규칙 + 연구 예외를 합친 정확한 병합안.
- `docs/tooling/Graphify-Ponytail-2026-09-28.md`, `docs/tooling/graphify-installation-2026-09-28.json`: 적용 보고서와 출처·설치·검증 기록.

`graphify-out/`의 그래프·AST 캐시·HTML·GRAPH_REPORT.md·진단 JSON은 로컬 생성물로 보존한다. 공식 README 원문 사본은 저장소 밖 `../pitcheezy-tooling-references/20260928/`에 보관했다. 함께 진행 중인 보정 실험의 새 소스·검사는 이 도구 설치의 변경 목록에 포함하지 않는다.

## 실행 범위와 생성물

.graphifyignore로 데이터·Statcast/물리캐시·미디어·라벨·가중치·추적/실험산출물과 문서/PDF를 제외했다. code-only AST 추출 뒤 LLM 라벨링을 비활성화한 클러스터링을 수행했다. 이후 감독기 코드 변경을 반영해 `graphify update .`로 AST만 갱신했다. 모델/영상/정답 자료를 읽거나 의미 분석하지 않았으며 API token cost는 input0/output0이다. 코드·지원되는 JSON/패키지 설정464파일을 처리했다. 전체탐지517파일 중 YAML52개와 HTML1개는 Graphify가 document로 분류해 code-only에서 제외했다. YAML 설정 관계는 아직 분석하지 않았으며 이를 위해 의미 분석을 임의로 활성화하지 않았다.

출력 graphify-out/graph.json은 5,890노드·16,753연결·228커뮤니티다. 보고서97% EXTRACTED,3% INFERRED(557연결)이며 import cycle은 탐지되지 않았다. 5,000노드를 넘어 HTML은228커뮤니티·1,336커뮤니티간연결로 축약했다. 총생성물은 약27MB이며 사용자 승인 후 graphify-out/을 .gitignore에 추가했다. 로컬 생성물은 보존했다. 작업중 소스와 검사도 포함된 스냅샷으로, 보고서의 commit1cf88804만으로 clean committed source 그래프라고 해석하지 않는다.

## 요청한 모듈 지도

| 모듈 | 확인한 구현과 의존 관계 |
|---|---|
| 데이터 전처리 | src/pitcheezy/data/prepare.py 및 experiments/pitchmdp/pitchmdp/data.py·sequence_data.py·matrix_data.py → 공통 KEY/라벨/물리특징·과거투구 store. 실험 준비/모델/검증이 이를 공유한다. |
| 전이확률 NN | src/pitcheezy/transition/neural.py가 count_transitions·TransitionTensor·공통 action grid를 공유한다. 별도 연구모델은 pitchmdp/model.py·sequence_model.py·matrix_models.py이며 과거투구와 JointDelivery로 추론한다. 두 구현군을 동일 모델로 해석하지 않는다. |
| RL 정책(MDP+VI) | src/pitcheezy/policy/vi.py는 공통 상태/action grid·전이텐서를 소비한다. 연구 planner.py·sequence_planner.py·matrix_policy.py와 offline RL인 matrix_offline_rl.py는 frozen outcome 모델/game 가치/동일 action 정의에 의존한다. VI와 offline RL은 별도 실험 경로다. |
| 이상 탐지 | 별도 완성된 anomaly detector는 그래프와 확인한 소스에서 식별하지 못했다. diagnose_adaptive_location.py, diagnose_sequence_legality.py, robustness/stress·동등성 검사 경로가 현재 진단 기능이다. 이를 독립 이상탐지 모델이라고 부르지 않는다. |
| CV | apps/observer/research/optical_flow_tracker.py·location_evaluator.py, backend intent.py·video_lab.py·intent_store.py가 존재한다. intent의 좌표/근거/검토 상태 검증이 event_analysis로 연결된다. src/pitcheezy/cv/__init__.py는 파일럿/후순위 표지다. |
| 책임 분해 | backend event_analysis.py가 동일 frozen WE·추천identity·IntentEstimate 검증을 사용해 계획/실행/관측 대비를 계산한다. service.py가 추천·자료·저장·이벤트를 조정한다. src/pitcheezy/decomp/__init__.py 자체는 후순위 표지다. 현재 대비를 인과적 선수책임으로 확장하지 않는다. |
| LLM 해석 | backend explanations.py의 explain_choices는 planner와 동일한 확률·continuation value를 산술 분해하고 재구성 오차를 검사한다. 이 구현에는 외부 LLM API 호출이 없다. 현재 사용자 설명은 결정적 계산이며 완성된 LLM 해석 모듈로 보고하지 않는다. |
| 프론트엔드 | apps/observer/web/src/App.tsx·types.ts·CatalogPicker·ObservedContext가 추천/사건/주자·카운트 화면을 조립한다. backend main.py·ObserverService API와 연결된다. HTTP/동적 import 경계는 순수 AST 연결만으로 완전히 표현되지 않으므로 대응 소스를 확인했다. |

## 강한 결합과 주의할 추론

1. 가장 큰 허브가 NN 자체보다 실험 증거 처리다: hash_file316연결, read_json192, assert_hashes118, dump114, canonical_hash111. 이는 재현성·산출물 무결성 경로가 다양한 실험에 공유되는 구조다. 높은 degree만으로 과설계나 삭제 대상으로 판단하지 않는다.
2. ObserverService–Store–추천–이벤트/교체 기록이 서로 연결된다. Store는37연결로 여러 경로를 잇는 노드다. 저장형식·추천identity 변경은 replay와 event 결과 검증까지 영향을 줄 수 있다.
3. 관전 recommender는 연구용 source와 standalone runtime_src 양쪽의 frozen 모델·planner·전달분포에 연결된다. 동적 import는 AST에서 누락/모호해질 수 있으나 source hash 검증과 sys.path 격리를 실제 소스에서 확인했다. runtime_src 복제는 배포 동등성 경계이며 자동 중복삭제 대상이 아니다.
4. 설명과 책임 분해가 추천의 값/확률/identity를 공유한다. 설명 문구·이벤트 결과를 독립 추정치처럼 갱신하면 동일평가기 검사가 깨질 수 있다.
5. Surprising Connections의 inferred `slow→original`, `launch→Child`는 다른 파일의 동일 이름으로 잘못 연결됐을 가능성이 높다. 그래프 탐색 후보이지 확인된 생산 코드 의존이라고 보고하지 않는다. EXTRACTED import/call과 실제 소스 흐름을 우선했다.

보고서가 제안한 유용한 다음 탐색은 “hash_file이 왜 수십 실험 커뮤니티를 잇는가”, “Store 변경이 어떤 replay/이벤트 경로에 영향을 주는가”다. 갱신 뒤 커뮤니티 cohesion 예시는 .0443037975와 .0589506173이다. 낮은 cohesion은 공용 유틸·모호한 import 해석·실험 변형 때문일 수도 있으므로 모듈분할 근거로 단독 사용하지 않는다.

## Ponytail 범위와 충돌

공식 Codex 플러그인 명령에는 확인한 도움말상 project-install 옵션이 없다. 사용자 요청의 repo-only 범위를 위해 공식 AGENTS.md instruction-only 경로를 선택해 사용자 승인된 병합안을 적용했다. 기본모드는full이고 ultra는 사용하지 않는다. 기존 지침과 Graphify 항목을 보존한 채 연구 예외를 우선하도록 병합했다. 기존 코드의 일괄 리팩터링은 수행하지 않았다.

Graphify의 query-first와 Ponytail의 기존 구현 재사용은 양립한다. 다만 Graphify는 코드 변경 뒤 AST갱신을 권장하고 Ponytail은 최소작업을 선호하므로, 변경을 묶어 갱신하는 비용을 관리한다. Ponytail의 deletion/minimal-test 문구와 연구 재현성은 충돌할 수 있어 비교변형/ablation·설정/seed/logging·평가/누수/재현검사·봉인source를 보호하는 예외를 명시했다. 현재의 연구 검증·실험 변형을 줄이는 작업은 요청받지 않았으며 수행하지 않았다.

실제 graph source423개를 검사해 제외 경로/확장자 위반0개를 확인했다. Graphify read-only 진단은 missing/dangling endpoint0, exact duplicate0, self-loop17을 보고했다. 자기 연결은 재귀/해석 결과일 수 있어 삭제하지 않았고, 이 진단은 이미 생성된 그래프 이전의 edge 축약이나 추론 오류를 배제하지 못한다.
