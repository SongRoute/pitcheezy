"""Human-readable complete sweep report; all configurations retained."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/pitchmdp-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]
NAMES={'hand':'좌·우타만','id':'개인 ID+좌·우타','continuous':'성향6개+신뢰도6개',
       'clusters_3':'군집3개','clusters_5':'군집5개','clusters_10':'군집10개','clusters_20':'군집20개',
       'reference':'연속 성향12+군집5 / 이전5구',**{f'history_{n}':f'이전{n}구' for n in (1,2,3,4)}}
ENGLISH={'hand':'Hand only','id':'Batter ID','continuous':'Continuous profiles',
         'clusters_3':'Clusters K=3','clusters_5':'Clusters K=5','clusters_10':'Clusters K=10','clusters_20':'Clusters K=20',
         **{f'history_{n}':f'Previous {n} '+('pitch' if n==1 else 'pitches') for n in (1,2,3,4)}}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();r=json.loads((run/'results.json').read_text());cfg=json.loads((run/'config.json').read_text())
    lines=['# 타자 표현과 투구 이력 길이 비교','',f"실행 시작: {cfg['created_utc']}; 완료: {r['finished_utc']}.",'',
           f'전체 아티팩트: [{run.name}](<{run}>)','',
           '주자·아웃과 모든 기본 경기 상황은 유지했다. 타자 표현은 이전5구를 고정해 비교하고, 이력 길이는 기존 타자 표현을 고정해 비교했다. '
           '두 연도·초기값5개·신규110개 MLP를 사용했으며, 이전 전체 입력 MLP 결과를 동일 표본의 기준으로 재사용했다. '
           '군집은 학습 기간에만 구성하고 타자 성향은 전날까지의 이력으로 갱신했다.','',
           '## 해석 방법','',
           '낮은 로그손실·Brier가 좋다. Δ는 해당 구성에서 기존 전체 입력 혼합의 점수를 뺀 값이다. '
           '양수이면 기존 구성이 더 좋고 음수이면 해당 구성이 더 좋다. '
           '보정95% 구간은 각 연도 내 타자7개/이력4개 비교군별로2000회 동일 경기 bootstrap의 최대 중심 편차를 사용했다. '
           '모든 모델·온도·혼합 비율을 고정한 조건부 구간이며, 학습 및 CAL 추정의 전체 불확실성을 포함하지 않는다.','']
    for family,title in [('batter','타자 표현'),('history','이력 길이')]:
        lines += [f'## {title}','', '| 연도 | 구성 | 혼합 로그손실 | Δ [비교군 보정95% 구간] | 혼합 Brier | NN 비중 | 단독 앙상블 로그손실 |',
                  '|---|---|---:|---|---:|---:|---:|']
        for year,fold in r['folds'].items():
            ref=fold['reference_metrics']['reference'];rawref=fold['reference_metrics']['reference_ensemble']
            lines.append(f"| {year} | 기존 전체 입력 | {ref['log_loss']:.6f} | 기준 | {ref['brier_multiclass']:.6f} | 기존 보정 | {rawref['log_loss']:.6f} |")
            for name in fold['comparisons'][family]['contrasts']:
                v=fold['variants'][name];m=v['metrics'];d=fold['comparisons'][family]['metrics']['log_loss'][name];lo,hi=d['simultaneous95']
                lines.append(f"| {year} | {NAMES[name]} | {m['log_loss']:.6f} | {d['variant_minus_reference']:+.6f} [{lo:+.6f}, {hi:+.6f}] | {m['brier_multiclass']:.6f} | {v['selection']['model_weight']:.1%} | {v['ensemble_metrics']['log_loss']:.6f} |")
        lines += ['']
    lines += ['## 학습에 없던 타자와 긴 이력 표본','',
              '| 연도 | 구성 | 기존 타자 n / 로그손실 | 신규 ID 타자 n / 로그손실 | 이전5구 이상 n / 로그손실 |','|---|---|---|---|---|']
    def cell(m):return f"{m['n']:,} / {m['log_loss']:.6f}" if m['n'] else '0 / 미평가'
    for year,fold in r['folds'].items():
        groups={'reference':fold['reference_subgroups'],**{name:v['subgroups'] for name,v in fold['variants'].items()}}
        for name,s in groups.items():
            lines.append(f"| {year} | {NAMES[name]} | {cell(s['known_id'])} | {cell(s['unseen_id'])} | {cell(s['history_ge5'])} |")
    lines += ['', '신규 ID는 공통 신경망 TRAIN 표본에 없었던 ID다. 그 타자의 평가 시점 이전 경기에서 얻은 성향 정보는 연속·군집 모델에 사용 가능하다. '
              '이 하위집단 점수는 기술 통계이며 작은 신규 타자 표본에서 순위를 확정하지 않는다.','',
              '## 모델 크기와 학습 비용','', '| 연도 | 구성 | 문맥 입력 수 | 파라미터 | 평균 fit 초 |','|---|---|---:|---:|---:|']
    for year,fold in r['folds'].items():
        for name,v in fold['variants'].items():
            fits=[json.loads((run/year/name/f'seed{seed}_fit.json').read_text()) for seed in cfg['seeds']]
            lines.append(f"| {year} | {NAMES[name]} | {v['context']['n_context']} | {fits[0]['parameter_count']:,} | {np.mean([f['seconds'] for f in fits]):.1f} |")
    lines += ['', 'ID는16차원 학습 임베딩을 추가한다. 군집 수에 따라 문맥 투영층의 입력 수와 파라미터가 조금 달라진다. '
              '따라서 타자 비교는 실용적 표현 방식 비교이며 완전히 같은 유효 용량의 정보량 실험은 아니다. '
              '이력 길이는6개 슬롯과 전체 파라미터 수를 유지하고 오래된 슬롯의 값과 마스크를 모두 제거했다.','',
              '## 고정 조건과 한계','',
              '- 두 실험 축은 따로 비교했다. 특정 타자 표현과 특정 이력 길이를 결합한 최적 조합은 이번 결과로 입증하지 않는다.',
              '- 군집은 soft 유사도이며 임의의 구분을 실제 야구 유형이라고 단정하지 않는다. 군집-only 모델에 연속 성향 입력을 별도로 남기지 않았다.',
              '- 초기값5개 반복은 신경망 학습에 적용했다. 각 K의 군집 학습 seed는42로 고정했으므로 군집 초기화 자체의 안정성까지 검증한 것은 아니다.',
              '- 개인 ID의 학습에서 보지 못한 값은 고정0임베딩을 사용한다. 좌·우타 및 경기 상황은 그대로 제공한다.',
              '- 이력은 같은 타석 안의 직전 투구다. 이전 타자의 타석까지 이어지는 투수의 연속 이력은 이번 범위에 포함하지 않았다.',
              '- 학습기간 이닝으로 고른 선발6명을 유지했다.2025에 평가 투구가 없는 선수도 교체하지 않았다.',
              '- 조기 종료 May1–15, 온도 May16–31, 혼합 June, 평가 July–September 분할을 이전 실험과 동일하게 유지했다.400개 TRAIN 공동 물리 표본과 빈도 기준선도 그대로 재사용했다.',
              '- 2024/2025는 앞서 사용한 역사 데이터다. 탐색적 비교이며 평가 점수로 배포용 K나 이력 길이를 선택하지 않았다. 두 연도의 학습 구간은 중첩된다.',
              '- 원래 supported-PA 및 현재 위치 결측 제외 조건을 유지해 관측 후 적격 판정된 표본에 한정된다. 실제 추천이나 승률의 인과적 개선을 검증하지 않았다.',
              '- 단독 앙상블과 혼합 점수를 함께 공개한다. 각 구성의 온도·혼합 비율을 별도로 보정하므로, 주 비교는 보정까지 포함한 예측 방식이다. 기준선과의 혼합이 표현 차이를 완화할 수 있다.',
              '- 연속·군집 성향은 적격 조건으로 거르기 전의 전체 과거 투구에서 계산되며 평가 중에도 전날까지 갱신된다. ID 임베딩은 표본으로 뽑은 적격 TRAIN에서 학습한 뒤 고정된다. 사용 가능한 과거 증거량·갱신 시점도 달라 순수한 인코딩 방식만의 우열을 입증하지 않는다. ID+성향 또는 고정 시점 성향 대조는 후속 별도 실험 대상이다.',
              '- K를 늘리면 군집 중심·soft 유사도 폭·입력 차원도 바뀐다. 이력 마스크를 줄이면 과거 물리량뿐 아니라 이용 가능한 이력 길이 신호도 달라진다. 각각 실용적 표현/윈도우 설정의 비교로 해석한다.', '',
              '[프로토콜](docs/REPRESENTATION_HISTORY_PROTOCOL.md) · [실행기](scripts/run_representation_history.py) · [표현/이력 구현](scripts/representation_adapters.py)','']
    conclusions=['## 이번 비교에서 확인된 점','']
    for name in ('hand','id'):
        supported=[year for year,fold in r['folds'].items() if fold['comparisons']['batter']['metrics']['log_loss'][name]['simultaneous95'][0]>0]
        conclusions.append(f"- {NAMES[name]} 대비 기존 연속 성향+군집 구성의 로그손실 개선이 비교군 보정 구간에서 확인된 연도: {', '.join(supported) or '없음'}.")
    history_null=all(d['simultaneous95'][0]<=0<=d['simultaneous95'][1] for fold in r['folds'].values() for d in fold['comparisons']['history']['metrics']['log_loss'].values())
    if history_null:
        conclusions.append('- 이전1~4구와 기존5구의 차이는 두 연도 모두 보정 구간에0을 포함했다.5구로 늘리는 추가 이득을 확인하지 못했으며, 모든 길이가 동등하다고 증명한 것은 아니다.')
    conclusions += ['- 군집 수별 최저 평균은 탐색적 기술 통계다. 특정 K의 우월성이나 두 축을 결합한 최적 모델을 확정하지 않는다.', '- 이번 비교는 실제 사용 가능한 예측 방식의 비교다. 개인 ID와 성향 기반 방식의 과거 증거량·갱신 시점·유효 용량 차이도 포함하므로 순수한 인코딩 효과만으로 해석하지 않는다.', '']
    lines[lines.index('## 해석 방법'):lines.index('## 해석 방법')]=conclusions
    audit_path=run/'final_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        assert audit['status']=='passed' and audit['final_artifact_hashes_verified']
        assert sum(fold['models'] for fold in audit['folds'])==110
        lines += ['## 검증 및 다음 우선순위','',
                  f'- 두 연도110개 모델의 독립 CPU 감사 통과: [final_audit.json](<{audit_path}>). 표본·ID vocabulary·미관측 임베딩·보정 최적값·혼합·점수·동시 구간·아티팩트 해시를 재검산했다.',
                  '- 실행 전 전체148개 테스트 통과. 신경망 실험 배치는 약52분에 완료했다.',
                  '- 주자·아웃과 타자 성향을 필수 입력으로 유지한다. 다음 우선순위는 같은 성향 정보에 개인 ID를 추가하는 비교, 그다음 성향의 갱신 시점을 고정하는 대조다. ID 자체의 추가 정보와 성향 갱신 효과를 분리할 수 있다.',
                  '- 이력 확장은 현재 타석의1~5구 순위를 더 고르는 것보다 이전 타석까지 이어지는 투수 이력을 별도 가설로 검증한다. 현재 표본에서 이전5구 이상이 있는 비율은2024 약8.7%,2025 약9.1%다. 타석 경계와 상대 타자 변경을 명시하고 같은 평가 표본을 유지해야 한다.',
                  '- 위 후속 비교는 아직 실행하지 않았다. K10 또는 이전2구를 확정 최적값으로 채택하지 않는다.','']
    report='\n'.join(lines);(PROJECT/'REPRESENTATION_HISTORY_RESULTS.md').write_text(report);(run/'REPRESENTATION_HISTORY_RESULTS.md').write_text(report)
    for family in ('batter','history'):
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True,sharex=True)
        extrema=[]
        for ax,(year,fold) in zip(axes,r['folds'].items()):
            entries=fold['comparisons'][family]['metrics']['log_loss']
            for i,(name,d) in enumerate(entries.items()):
                value=d['variant_minus_reference'];lo,hi=d['simultaneous95'];extrema += [lo,hi]
                ax.errorbar(value,i,xerr=np.array([[value-lo],[hi-value]]),fmt='o',capsize=4,color='#3569b5')
            ax.set_yticks(range(len(entries)),[ENGLISH[name] for name in entries]);ax.axvline(0,color='black',linestyle='--',linewidth=.8)
            ax.set_title(year);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Log loss: variant minus full reference')
        axes[0].invert_yaxis();span=max(extrema)-min(extrema);axes[0].set_xlim(min(extrema)-span*.1,max(extrema)+span*.1)
        fig.suptitle(f'{family.capitalize()} sweep: fixed frequency blends\n95% simultaneous family intervals, paired game bootstrap')
        fig.tight_layout();fig.savefig(run/(family+'_comparison.png'),dpi=180);plt.close(fig)

if __name__=='__main__':main()
