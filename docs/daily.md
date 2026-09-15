# 하루 명령어 모음

맥미니 `songui-Macmini`, 사용자 `song`, 셸 zsh(oh-my-zsh + starship), 터미널 iTerm. tmux prefix는 **Ctrl-a** (기본 Ctrl-b 아님), 마우스 켜짐, 창 번호 1부터.

## 아침 (맥미니 앞이거나 맥북 SSH)

```bash
ssh song@songui-Macmini.local        # 맥북에서만. 맥미니 앞이면 생략
~/Projects/pitcheezy/scripts/dev.sh  # tmux 세션 pz에 붙기 (없으면 만들고 Claude 띄움)
```

Claude 창에서:

```
/clear                # 어제 대화를 잇지 않을 때 (= 아침 새 세션)
밤사이 결과 확인        # 루틴 시작. results/·runs/_logs/ 읽고 experiments.md 채우고 한 가지 제안
```

아이폰: Claude 앱 → 맥미니 세션이 목록에 보임 (Remote Control 자동 시작). 붙어서 같은 대화를 이어감.

## tmux (prefix = Ctrl-a)

| 하고 싶은 것 | 키 |
|---|---|
| Claude 창 ↔ runs 창 | `Ctrl-a 1` / `Ctrl-a 2` |
| 다음 창 | `Ctrl-a n` |
| 세션에서 떨어지기 (Claude는 계속 돎) | `Ctrl-a d` |
| 창 세로/가로 분할 | `Ctrl-a \|` / `Ctrl-a -` |
| 분할 창 이동 | `Ctrl-a h/j/k/l` |
| 세션 목록 | `tmux ls` |
| 다시 붙기 | `tmux attach -t pz` |

## 저녁: 다음 밤 실험 걸기 (runs 창)

```bash
cd ~/Projects/pitcheezy
cp configs/_template.yaml configs/EXP-P0-001.yaml   # 처음 한 번. 이후엔 Claude가 config를 만듦
nohup python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --seed 0 \
  > runs/_logs/EXP-P0-001_s0.log 2>&1 &
tail -f runs/_logs/EXP-P0-001_s0.log                 # 잘 도는지 잠깐 확인. Ctrl-c로 빠져도 실험은 계속
```

시드 3개 동시에:

```bash
for s in 0 1 2; do
  nohup python scripts/run_experiment.py --config configs/EXP-P0-001.yaml --seed $s \
    > runs/_logs/EXP-P0-001_s$s.log 2>&1 &
done
```

걸어둔 뒤 docs/plan.md "다음 아침 확인" 줄에 로그 경로를 적는다. 없으면 "없음".

## 확인·정리

```bash
ps aux | grep run_experiment | grep -v grep   # 돌고 있는 실험
tail -n 20 runs/_logs/*.log                    # 로그 끝
ls runs/                                       # 산출물
git status && git log --oneline -5             # 어제 커밋 상태
.venv/bin/pytest tests -q                      # 테스트 (편집 시 훅이 자동으로도 돌림)
```

## 데이터 (Drive 설치 후 한 번)

```bash
brew install --cask google-drive && open -a "Google Drive"
# 화면에서 로그인 → Finder에서 My Drive/pitcheezy/data/raw 우클릭 → "오프라인 사용 가능"
# 그다음 Claude에게 "Drive 설치했어" → 경로 연결 + 해시 대조
```

## 가끔

```bash
source ~/.zshrc            # 환경변수 바꾼 직후 현재 셸에만
claude --resume            # 이전 대화 골라서 되살리기
tmux kill-session -t pz    # 세션 완전히 끄기 (거의 안 씀)
```
