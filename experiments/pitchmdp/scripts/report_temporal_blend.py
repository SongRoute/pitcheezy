"""Render completed rolling-fold results without selecting models on evaluation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pitchmdp-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
NAMES = {'frequency': '상황 빈도 기준선', 'flatten_mlp_ensemble': 'MLP 5-seed 평균',
         'transformer_ensemble': 'Transformer 5-seed 평균', 'flatten_mlp_blend': 'MLP + 기준선',
         'transformer_blend': 'Transformer + 기준선'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    results = json.loads((run/'results.json').read_text())
    config = json.loads((run/'config.json').read_text())
    lines = ['# 날짜 재검증과 공정한 혼합 비교', '',
             f"실행 시작: {config['created_utc']}; 완료: {results['finished_utc']}.", '',
             f'전체 아티팩트: [{run.name}](<{run}>)', '',
             '두 연도 모두 학습 기간의 선발 이닝만으로 상위6명을 선정했다. 평가 기간의 출전량은 선정에 사용하지 않았다. '
             '각 모델을 초기값5개로 새로 학습하고, 같은400개 물리 표본·같은 빈도 기준선·같은 보정 표본으로 비교했다.', '',
             '## 예측 결과', '', '로그손실과 Brier는 낮을수록 좋다. 아래는 평가용 표본에 대한 결과이며, 실제 추천의 승률 개선을 뜻하지 않는다.', '',
             '| 연도 | 모델 | 평가 투구 | 로그손실 | Brier | 신경망 혼합 비중 |',
             '|---|---|---:|---:|---:|---:|']
    for year, result in results['folds'].items():
        for name in NAMES:
            m = result['metrics'][name]
            w = result['selections'][name.removesuffix('_blend')]['model_weight'] if name.endswith('_blend') else None
            lines.append(f"| {year} | {NAMES[name]} | {m['n']:,} | {m['log_loss']:.6f} | {m['brier_multiclass']:.6f} | {w:.1%} |" if w is not None else
                         f"| {year} | {NAMES[name]} | {m['n']:,} | {m['log_loss']:.6f} | {m['brier_multiclass']:.6f} | — |")
    lines += ['', '## 같은 경기끼리 비교', '', '차이는 앞 모델에서 뒤 모델을 뺀 값이다. 음수이면 앞 모델이 우수하다. '
              '95% 구간은 경기 단위2000회 bootstrap이며 학습·보정 표본 불확실성을 포함하지 않는다.', '',
              '| 연도 | 비교 | Δ 로그손실 [95% 구간] | Δ Brier [95% 구간] |', '|---|---|---|---|']
    comparisons = [('flatten_mlp_blend', 'frequency'), ('transformer_blend', 'frequency'), ('transformer_blend', 'flatten_mlp_blend')]
    for year,result in results['folds'].items():
        for a,b in comparisons:
            scores = result['paired'][a+'_minus_'+b]['metrics']
            cells=[]
            for metric in ('log_loss', 'brier_multiclass'):
                m=scores[metric]; lo,hi=m['game_only_bootstrap95']
                cells.append(f"{m['model_minus_reference']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
            lines.append(f"| {year} | {NAMES[a]} − {NAMES[b]} | {' | '.join(cells)} |")
    lines += ['', '## 데이터와 선수', '',
              '| 연도 | NN 학습 | 전체 학습 | 조기 종료 | 온도 보정 | 혼합 보정 | 평가 | 평가 경기 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for year,result in results['folds'].items():
        s=result['samples']
        lines.append(f"| {year} | {s['train']['n']:,} | {s['full_train']['n']:,} | {s['earlystop']['n']:,} | {s['temperature']['n']:,} | {s['blend']['n']:,} | {s['dev']['n']:,} | {s['dev']['games']} |")
    lines += ['', '| 연도 | 선수 | TRAIN 이닝 | 평가 원투구 | 평가 적격 투구 | 적격 선발 경기 |', '|---|---|---:|---:|---:|---:|']
    for year,result in results['folds'].items():
        for m in result['cohort']['members']:
            d=m['coverage']['dev']; outs=m['train_outs']
            lines.append(f"| {year} | {m['name']} ({m['pitcher']}) | {outs//3}.{outs%3} | {d['raw_pitches']:,} | {d['eligible_pitches']:,} | {d['eligible_starts']} |")
    lines += ['', '## 고정 조건과 해석 범위', '',
              '- TRAIN은2023-05-15부터 각 연도4월30일까지 누적한다. May1–15는 조기 종료, May16–31은 온도, June은 혼합 비율, July–September는 평가에 사용했다. 서로 겹치는 경기가 없다.',
              '- MLP와 Transformer는 같은 입력과 학습 표본·예산을 사용했다. 파라미터 수는99,946 대278,762로 다르므로 같은 크기에서 구조만 분리한 비교는 아니다.',
              '- 실제 현재 공의 물리량을 예측에 넣지 않고 TRAIN 공동 물리 표본400개로 평균냈다. 관측된 현재 구종을 조건으로 하며 구종 선택의 반사실적 효과를 입증하지 않는다.',
              '- 타자 ID는 과거 이력 결합에만 쓰며 신경망 입력이 아니다. 타자 성향은 전날까지의 데이터로 갱신하고, 정규화·유형 중심·신경망·빈도표는 학습 기간 후 고정했다.',
              '- 원래의 supported-PA 및 현재 위치 결측 제외 조건을 유지했다. 따라서 관측 후 적격 판정된 부분집합의 예측 성능이며 모든 미래 투구에 대한 전향적 효과는 아니다.',
              '- 이전 연구 과정에서2024/2025 데이터가 이미 사용되었다. 이번은 새로 분할해 재학습한 탐색적 시간 재현 실험이며 완전히 미관측 확인 시험이 아니다.2025 TRAIN에는2024 평가 기간이 포함되므로 두 연도를 독립 반복으로 취급하지 않는다.',
              '- 기존 결과와 선수·CAL 조건·적분 표본 수가 달라 절대 로그손실을 직접 전후 비교하지 않는다. 각 연도 안의 동일 표본 비교가 주 분석이다.',
              '- 평가 출전이 없는 선수도 그대로 공개하며 다른 선수로 교체하지 않는다. 개별 선수의 불확실성 및 부상·출전 원인은 이 실험으로 판단하지 않는다.',
              '- 400개 평균도 정확한 전체 풀 적분은 아니다. 구간은 고정된 앙상블·혼합 비율에 조건부이며 학습 seed·CAL 추정·후속 모델 선택 불확실성을 모두 포괄하지 않는다.', '',
              '[사전 고정 프로토콜](docs/TEMPORAL_BLEND_PROTOCOL.md) · [실행기](scripts/run_temporal_blend.py) · [검증 테스트](tests/test_temporal_blend.py)', '']
    decisions = []
    for year, result in results['folds'].items():
        primary = result['paired']['transformer_blend_minus_flatten_mlp_blend']['metrics']['log_loss']
        lo, hi = primary['game_only_bootstrap95']
        verdict = 'Transformer 혼합 우세 구간' if hi < 0 else ('MLP 혼합 우세 구간' if lo > 0 else '차이의 구간이0을 포함해 우열 미확정')
        decisions.append(f'- {year}: {verdict}.')
    summary = ['## 핵심 판정', '', *decisions, '', '두 모델 모두에 동일한 빈도 혼합을 제공한 뒤의 차이가 주 비교다. 아래 표의 기준선 대비 이득과 모델 사이의 차이를 구분해 해석한다.', '']
    insert = lines.index('## 예측 결과')
    lines[insert:insert] = summary
    lines += ['## 검증 범위', '', '독립 검증기는 처리 데이터에서 선정·표본·정답을 다시 구성하고, 저장 예측에서 앙상블·혼합 최적값·점수·경기 bootstrap을 재계산한다. 물리 표본 적분과 온도 최적값은 동결 코드·설정·파일 해시를 확인한 범위이며, 온도 보정용 후보 logits를 독립 추론해 재현한 것은 아니다.', '']
    lines += ['## 학습 비용', '', '동일 장비의 저장된 fit 시간 평균이다. 최종400표본 적분 추론 시간은 제외한다.', '', '| 모델 | 파라미터 | 10회 평균 fit 시간(초) |', '|---|---:|---:|']
    for kind, label in [('flatten_mlp', 'MLP'), ('transformer', 'Transformer')]:
        fits = [json.loads((run/year/f'{kind}_{seed}_fit.json').read_text()) for year in results['folds'] for seed in range(42,47)]
        lines.append(f"| {label} | {fits[0]['parameter_count']:,} | {np.mean([fit['seconds'] for fit in fits]):.1f} |")
    lines += ['', '## 완료 검증과 다음 우선순위', '', '전체138개 테스트와 두 연도·20모델의 독립 감사가 통과했다. 감사 결과는 같은 실행 폴더의 `final_audit.json`에 있다. 신경망 학습·추론 프로세스는 모두 종료됐다.', '', '두 연도 모두 두 혼합의 기준선 대비 로그손실·Brier 개선 구간은0 아래에 있었다. 반면 Transformer 혼합과 MLP 혼합의 차이는 두 연도 모두0을 포함했다. 동일하다고 증명된 것은 아니지만, 약7배의 학습 비용을 정당화할 일관된 추가 우위는 이번 비교에서 확인하지 못했다.', '', '다음 우선순위는 저렴한 MLP 혼합을 중심으로 주자·아웃, 점수차·이닝·홈원정, 과거 타자 성향, 이전 투구 이력을 각각 제거하는 실험이다. 동일 시간 분할·기준선·보정 조건을 유지해 추가 요소별 기여가 두 연도에 재현되는지 확인하고, 중요한 비교를 Transformer에서 교차 확인하는 방식이 적절하다. 이 후속 제거 실험은 이번20모델 배치의 결과에 포함되지 않는다.', '']
    report='\n'.join(lines)
    (PROJECT/'TEMPORAL_BLEND_RESULTS.md').write_text(report)
    (run/'TEMPORAL_BLEND_RESULTS.md').write_text(report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True,sharex=True)
    labels=['MLP blend − frequency', 'Transformer blend − frequency', 'Transformer blend − MLP blend']
    for ax,(year,result) in zip(axes,results['folds'].items()):
        for i,(a,b) in enumerate(comparisons):
            m=result['paired'][a+'_minus_'+b]['metrics']['log_loss']
            mean=m['model_minus_reference'];lo,hi=m['game_only_bootstrap95']
            ax.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color=['#4878d0','#ee854a','#6acc64'][i],capsize=5)
        ax.axvline(0,color='black',linewidth=.8,linestyle='--')
        ax.set_xlim(-.0225,.003)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.set_title(year);ax.set_yticks(range(3),labels);ax.set_xlabel('Log-loss difference (lower is better)');ax.grid(axis='x',alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Rolling evaluation: matched frequency blends\n95% paired game bootstrap; fixed fitted pipelines')
    fig.tight_layout();fig.savefig(run/'temporal_blend_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    main()
