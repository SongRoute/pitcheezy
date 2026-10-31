# 졸업작품 산출물 (충북대 2026 개정 양식 D01~D08)

저장소 기록(docs/decisions.md, docs/pm/*, results/, tests/)을 바탕으로 한글(HWP) 양식을 **스크립트로 채운** 산출물이다.
사람만 아는 값(학번, 개발기간 등)은 본문에 `[학번]`, `[확인]`, `[확인자]`로 남겨 두었다.

| 폴더 | 내용 |
|---|---|
| `templates/` | 학교 배포 빈 양식(D01~D08, 양식1~6). 수정하지 않는다. |
| `build/` | 생성 도구. `hwpfill.py`(양식 채우기 도구), `builders.py`(문서별 빌더), `content/`(문서별 내용), `diagrams/`(SVG 원본), `gen_class_svgs.py`(클래스 다이어그램 생성), `make.py`(진입점), `verify.py`(전수 검증) |
| `diagrams/` | 렌더링한 PNG(개발 절차, 사용사례, 배포, 패키지, 스프린트별 클래스)와 공개 화면 캡처(`shots/`, 고정 자료 849843 기준 390×844) |
| `out/` | 생성 결과. 문서마다 `.hwpx`(한글 2014+ 표준, 라이브러리가 실한컴 열림을 검증한 형식), `.hwp`(HWP 5.0 변환본), `.md`(검토용 내보내기) |

## 다시 만들기

```bash
# python-hwpx·pyhwp·pillow·lxml이 있는 가상환경에서 (예: uv venv .venv-hwp -p 3.12 && uv pip install python-hwpx pyhwp pillow lxml six)
cd docs/capstone/build
python make.py                 # 내용 모듈이 있는 문서 전부
python make.py D04 D06-3       # 일부만
python verify.py               # 열림 안전 검사·재파싱·잔여 자리표시자 점검
```

내용을 고치려면 `build/content/*.py`(D01~D08, `sprints.py`가 1~4차 스프린트 공통)를 수정하고 다시 만든다. 양식 구조(앵커 문장)는 `builders.py`가 안다.

## 검증 수준

- 저장할 때마다 python-hwpx의 열림 안전 검사(`validate_editor_open_safety`)를 통과해야 한다(실패하면 생성 중단).
- `verify.py`가 `.hwp`를 python-hwpx와 pyhwp(hwp5txt)로 다시 열어 표·그림 수와 잔여 자리표시자를 확인한다.
- 실제 한/글 프로그램에서 여는 확인은 사람이 한다. 문제가 있으면 `.hwpx`를 먼저 열고(검증된 형식), 필요하면 한/글에서 `.hwp`로 다른 이름 저장한다.
- 목차 페이지의 쪽 번호는 양식 그대로이므로 한/글에서 열어 갱신한다.

## 가정

- 스프린트는 저장소 이력 기준 4개(1차 9/7~9/20, 2차 9/21~10/3, 3차 10/4~10/12, 4차 10/13~10/22)로 나눴다. 구분을 바꾸려면 `content/common.py`의 `SPRINTS`와 `content/sprints.py`를 고친다.
- 시스템 이름은 `AI 투구 추천 야구 관전 보조 시스템(Pitcheezy)`, 문서 번호는 `2026-Pitcheezy-Doc-00N(-차수)`다.
- 수치는 저장소에 기록된 값만 썼고, 없는 값은 `미측정`·`미확인`으로 적었다.
