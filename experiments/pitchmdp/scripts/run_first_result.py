"""Train, evaluate and recommend on approved real data. Outputs only to mounted SSD."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ['WANDB_MODE'] = 'offline'
os.environ.setdefault('MPLCONFIGDIR', '/Volumes/T7 Shield/pitcheezy/pitchmdp/cache/matplotlib')

import numpy as np
import pandas as pd

from pitchmdp.model import PitchModel, CountBaseline, DeliveryDistribution, eligible, metrics, outcome_labels
from pitchmdp.recommend import recommend, supported_actions


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def code_hashes():
    paths = sorted(PROJECT.glob('pitchmdp/*.py')) + sorted(PROJECT.glob('scripts/*.py')) + sorted(PROJECT.glob('configs/*.json'))
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def binary_metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-8, 1-1e-8)
    bins = []
    for lo in np.arange(0, 1, .1):
        m = (p >= lo) & (p < lo+.1)
        if m.any():
            bins.append({'low': float(lo), 'n': int(m.sum()), 'predicted': float(p[m].mean()), 'observed': float(y[m].mean())})
    return {'n': len(y), 'log_loss': float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),
            'brier': float(((p-y)**2).mean()), 'calibration_bins': bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='Reuse completed checkpoints in this run directory')
    parser.add_argument('--config', type=Path, default=PROJECT/'configs/first_result.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root'])
    volume = Path('/Volumes/T7 Shield')
    if not volume.is_mount() or not root.resolve().is_relative_to(volume.resolve()):
        raise SystemExit('Mounted T7 Shield required; no internal fallback.')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise SystemExit('Use configured existing Python environment.')
    start = time.perf_counter()
    run = args.resume or root/'runs'/datetime.now(timezone.utc).strftime('first-%Y%m%dT%H%M%SZ')
    if not run.resolve().is_relative_to(root.resolve()):
        raise SystemExit('Run directory must be on configured SSD.')
    if args.resume:
        saved_config = json.loads((run/'config.json').read_text())
        saved_hashes = json.loads((run/'code_hashes_start.json').read_text())
        if saved_config != config or saved_hashes != code_hashes():
            raise SystemExit('Resume requires unchanged config and source. Start a new run for changed code.')
        saved_data = json.loads((run/'data_quality.json').read_text())
        current_data = json.loads((root/'reports/data_quality.json').read_text())
        if saved_data['processed_sha256'] != current_data['processed_sha256']:
            raise SystemExit('Resume refused: processed data identity changed.')
    else:
        run.mkdir(parents=True, exist_ok=False)
        dump(run/'config.json', config)
        dump(run/'code_hashes_start.json', code_hashes())
        for rel in code_hashes():
            destination = run/'source'/rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT/rel, destination)
    print(f'RUN_DIR={run}', flush=True)
    frame = pd.read_parquet(root/'processed/pitches.parquet')
    assert frame.game_date.max() < pd.Timestamp('2026-01-01')
    assert frame[['game_pk', 'at_bat_number', 'pitch_number']].duplicated().sum() == 0
    cohort_path = Path(config.get('cohort_manifest_path', root/'reports/cohort_manifest.json'))
    cohort = json.loads(cohort_path.read_text())
    if config.get('cohort_scope', 'LAD') == 'MLB':
        cohort_mask = frame.pitcher.isin(cohort['pitcher_ids']) & frame.pitcher.eq(frame.starter_pitcher)
    else:
        cohort_mask = frame.cohort_pitcher & frame.is_lad_start
    model_variant = config.get('model_variant', 'full')
    if model_variant == 'archetype':
        from pitchmdp.archetypes import add_batter_style_history
        print('Constructing strictly prior-date batting style histories', flush=True)
        add_batter_style_history(frame)
    fit_mask = eligible(frame)
    train_all = frame[(frame.split == 'train') & fit_mask].copy()
    calibration_all = frame[(frame.split == 'calibration') & fit_mask].copy()
    dev_cohort = frame[(frame.split == 'dev') & cohort_mask].copy()
    dev = dev_cohort[eligible(dev_cohort)].copy()
    assert train_all.game_date.max() < calibration_all.game_date.min() < dev.game_date.min()
    assert not set(train_all.game_pk) & set(dev.game_pk)
    train = pd.concat([train_all.sample(min(len(train_all), config['training_random_rows']), random_state=config['seed']),
                       train_all[cohort_mask.reindex(train_all.index)]]).drop_duplicates(
                           ['game_pk', 'at_bat_number', 'pitch_number']).sort_index()
    calibration = calibration_all.sample(min(len(calibration_all), config['calibration_random_rows']), random_state=config['seed'])
    dev_sample = dev.sample(min(len(dev), config['dev_max_rows']), random_state=config['seed']).sort_index()
    dump(run/'cohort_manifest.json', cohort)
    shutil.copyfile(root/'reports/data_quality.json', run/'data_quality.json')
    coverage = {'all_rows': len(frame), 'train_eligible': len(train_all), 'train_used': len(train),
                'calibration_eligible': len(calibration_all), 'calibration_used': len(calibration),
                'dev_cohort_rows': len(dev_cohort), 'dev_eligible_rows': len(dev), 'dev_used': len(dev_sample),
                'dev_eligible_row_fraction': len(dev)/max(1, len(dev_cohort)),
                'dev_cohort_pa': int(dev_cohort.groupby(['game_pk','at_bat_number']).ngroups),
                'dev_eligible_pa': int(dev.groupby(['game_pk','at_bat_number']).ngroups),
                'dev_dates': [str(dev.game_date.min().date()), str(dev.game_date.max().date())],
                'dev_games': int(dev.game_pk.nunique()), 'dev_batters': int(dev.batter.nunique()),
                'per_pitcher': {str(int(p)): {'all_pitches': len(g), 'eligible_pitches': int(eligible(g).sum()),
                                             'all_pa': int(g.groupby(['game_pk','at_bat_number']).ngroups),
                                             'games': int(g.game_pk.nunique())}
                                for p, g in dev_cohort.groupby('pitcher')}}
    coverage['selected_without_dev_starts'] = [x for x in cohort['selected'] if str(x['pitcher']) not in coverage['per_pitcher']]
    dump(run/'coverage.json', coverage)
    print('COVERAGE', json.dumps(coverage), flush=True)
    checkpoint = run/'pitch_model.pt'
    if checkpoint.exists():
        model = PitchModel.load(checkpoint)
    else:
        model = PitchModel(variant=model_variant, seed=config['seed']).fit(train, calibration, epochs=config['epochs'])
        model.save(checkpoint)
    dump(run/'training.json', model.training_report)
    baseline = CountBaseline().fit(train_all)
    delivery = DeliveryDistribution().fit(train_all, draws=config['delivery_draws'], seed=config['seed'])
    pred = delivery.predict(model, dev_sample)
    labels = outcome_labels(dev_sample)
    prediction = {'primary_prepitch_type_conditional': metrics(labels, pred),
                  'count_hand_baseline': metrics(labels, baseline.predict(dev_sample)),
                  'retrospective_actual_location_diagnostic': metrics(labels, model.predict(dev_sample)),
                  'per_pitcher': {str(int(p)): metrics(labels[dev_sample.pitcher.to_numpy() == p],
                                                      pred[dev_sample.pitcher.to_numpy() == p])
                                  for p in dev_sample.pitcher.unique()},
                  'delivery_note': 'Training-only empirical location mixture conditioned on pitcher/type/stand; not observed targets.',
                  'evaluation_note': 'Temporal development set, previously used in broader project; not untouched confirmation.'}
    dump(run/'prediction_metrics.json', prediction)
    print('PREDICTION', json.dumps({k: {m: v for m, v in d.items() if m in ['n','log_loss','brier_multiclass']}
                                    for k, d in prediction.items() if isinstance(d, dict) and 'n' in d}), flush=True)
    from pitchmdp.game import WinExpectancy, EmpiricalAdvancement
    game_checkpoint = run/'game_models.pkl'
    if game_checkpoint.exists():
        with game_checkpoint.open('rb') as f:
            we, advancement = pickle.load(f)
    else:
        we = WinExpectancy().fit(frame[frame.split == 'train'])
        advancement = EmpiricalAdvancement().fit(frame[frame.split == 'train'])
        with game_checkpoint.open('wb') as f:
            pickle.dump((we, advancement), f)
    # W uses real game win labels, one observation per PA, held-out July+ games.
    we_dev = frame[(frame.split == 'dev') & frame.complete_game & (frame.pitch_number == 1)].copy()
    we_dev = we_dev.sample(min(30000, len(we_dev)), random_state=config['seed']).sort_index()
    we_p = we.predict_home(we_dev)
    we_eval = binary_metrics(we_dev.final_home_win, we_p)
    we_eval['games'] = int(we_dev.game_pk.nunique())
    we_eval['dates'] = [str(we_dev.game_date.min().date()), str(we_dev.game_date.max().date())]
    we_eval['constant_0_5'] = binary_metrics(we_dev.final_home_win, np.full(len(we_dev), .5))
    we_eval['fit_report'] = we.training_report
    we_eval['advancement_report'] = advancement.report
    dump(run/'we_metrics.json', we_eval)
    print('WE', json.dumps({k: v for k, v in we_eval.items() if k in ['n','games','log_loss','brier']}), flush=True)
    pa = dev[(dev.pitch_number == 1) & (dev.balls == 0) & (dev.strikes == 0) & (dev.inning <= 8)]
    cases = pd.concat([g.sample(min(len(g), config['evaluation_pa_per_game']), random_state=config['seed'])
                       for _, g in pa.groupby('game_pk')]).sort_index()
    results, failures = [], []
    actions_by_key = {}
    for n, (_, row) in enumerate(cases.iterrows()):
        key = (int(row.pitcher), str(row.stand))
        if key not in actions_by_key:
            actions_by_key[key] = supported_actions(train_all, *key)
        actions = actions_by_key[key]
        if not actions:
            failures.append({'pitcher': key[0], 'game_pk': int(row.game_pk), 'reason': 'No supported target actions'})
            continue
        result = recommend(model, row, actions, we, advancement, config['control_sigma_ft'])
        results.append(result)
        if n % 10 == 0:
            print(f'PLANNED {n+1}/{len(cases)} {result["top_k"][0]}', flush=True)
            dump(run/'recommendations_partial.json', results)
    sensitivity = []
    representatives = pa.groupby('pitcher', sort=True).head(1)
    representative_results = []
    for _, row in representatives.iterrows():
        actions = actions_by_key.get((int(row.pitcher), str(row.stand)))
        if actions is None:
            actions = supported_actions(train_all, int(row.pitcher), str(row.stand))
            actions_by_key[(int(row.pitcher), str(row.stand))] = actions
        if actions:
            representative_results.append(recommend(model, row, actions, we, advancement, config['control_sigma_ft']))
        for sigma in config['control_sensitivity_ft']:
            if actions:
                sensitivity.append(recommend(model, row, actions, we, advancement, sigma))
    dump(run/'recommendations.json', results)
    dump(run/'representative_recommendations.json', representative_results)
    dump(run/'control_sensitivity.json', sensitivity)
    advantages = np.array([r['model_internal_advantage_pp'] for r in results])
    summary = {'n_pa': len(results), 'n_games': len({r['input']['game_pk'] for r in results}),
               'sample_rule': config['example_rule'], 'failures': failures,
               'mean_internal_advantage_pp': float(advantages.mean()),
               'pitcher_equal_weight_internal_advantage_pp': float(np.mean([
                   np.mean([r['model_internal_advantage_pp'] for r in results if r['input']['pitcher'] == p])
                   for p in {r['input']['pitcher'] for r in results}])),
               'first_action_differs_from_myopic_fraction': float(np.mean([not r['planned_first_action_equals_myopic'] for r in results])),
               'median_recommendation_seconds': float(np.median([r['seconds'] for r in results])),
               'no_causal_or_independent_policy_evaluation': True}
    dump(run/'strategy_summary.json', summary)
    # Store everything necessary for an independent single recommendation rerun.
    with (run/'recommendation_context.pkl').open('wb') as f:
        pickle.dump({'we': we, 'advancement': advancement, 'rows': representatives,
                     'actions': actions_by_key, 'delivery': delivery}, f)
    runtime = {'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'pipeline_seconds': time.perf_counter()-start,
               'elapsed_since_development_start_seconds': (datetime.now(timezone.utc)-datetime.fromisoformat(config['started_at_utc'])).total_seconds(),
               'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'code_hashes_end': code_hashes(), 'python': sys.version, 'pid': os.getpid()}
    dump(run/'runtime.json', runtime)
    report(run, coverage, prediction, we_eval, representative_results, sensitivity, summary, runtime)
    print(f'COMPLETE {run}', flush=True)


def report(run, coverage, prediction, we_eval, results, sensitivity, summary, runtime):
    config = json.loads((run/'config.json').read_text())
    cohort = json.loads((run/'cohort_manifest.json').read_text())
    absent = coverage.get('selected_without_dev_starts', [])
    scope_note = ("MLB 전체 팀의 선발 이닝을 선수 ID로 연결했고, DEV 50이닝 이상을 사전 조건으로 사용했다. "
                  "이는 평가 기간 생존·출장량을 조건으로 한 회고적 집합이며 미래 예측용 선정과 다르다."
                  if config.get('cohort_scope') == 'MLB' else 'LAD 선발 등판만 평가했다.')
    batter_note = ('타자 ID는 과거 기록 연결에만 사용했다. 예측 입력은 과거 선구안·컨택·장타 등 연속 성향, 신뢰도와 5개 연성 유형 가중치다.'
                   if config.get('model_variant') == 'archetype' else '최초 모델은 타자 ID 임베딩과 과거 비율을 포함했다.')
    lines = ['# 실제 데이터 첫 실행 결과', '',
             f"완료: {runtime['finished_at_utc']}. 착수 후 {runtime['elapsed_since_development_start_seconds']/60:.1f}분.", '',
             scope_note, '', batter_note, '',
             '평가 표본 없는 선정 선수: '+(', '.join(x['player_name'] for x in absent) if absent else '없음')+'.', '',
             '## 관측 데이터 예측 평가', '',
             f"DEV {coverage['dev_dates'][0]}–{coverage['dev_dates'][1]}: {coverage['dev_used']:,}구, "
             f"{coverage['dev_games']}경기, {coverage['dev_batters']}타자. "
             f"대상 선수 실제 투구 중 평가 적격 비율 {coverage['dev_eligible_row_fraction']:.1%}.", '',
             '| 평가 | N | Log loss ↓ | Brier ↓ |', '|---|---:|---:|---:|']
    for title, key in [('투구 전: 위치 분포 적분', 'primary_prepitch_type_conditional'),
                       ('카운트·좌우 빈도 기준선', 'count_hand_baseline'),
                       ('실제 도달 위치를 아는 사후 진단', 'retrospective_actual_location_diagnostic')]:
        m = prediction[key]
        lines.append(f"| {title} | {m['n']:,} | {m['log_loss']:.5f} | {m['brier_multiclass']:.5f} |")
    lines += ['', f"독립 날짜 경기 승패로 평가한 WE: {we_eval['n']:,} 타석 시작, {we_eval['games']}경기, "
              f"log loss {we_eval['log_loss']:.5f}, 이진 Brier {we_eval['brier']:.5f}. "
              '투구 결과 Brier는 다중분류 합계이므로 WE Brier와 직접 비교하지 않는다.', '',
              '## 추천과 모델 내부 비교', '',
              f"사전 규칙으로 선택한 {summary['n_pa']}타석/{summary['n_games']}경기에서 기준 정책 대비 "
              f"평균 {summary['mean_internal_advantage_pp']:+.4f}pp. 경기당 최대 3타석 표본의 평균으로, 전체 적격 타석 가중 평균과 다르다. "
              '동일 모델을 최적화하고 동일 모델로 평가한 진단이며 실제 승률 개선의 증거가 아니다.', '',
              '기준 정책: 학습 기간 구종 사용 비중 × 구종 내 지원 목표 셀 균등 분포. 실제 목표 선택 정책을 복원했다는 의미가 아니다.', '',
              '| 투수 ID | 날짜 / 타자 ID | 상황 | 첫 추천 | 타석 종료 수비 WE | 기준 WE |', '|---|---|---|---|---:|---:|']
    seen = set()
    for r in results:
        inp, a = r['input'], r['top_k'][0]
        if inp['pitcher'] in seen:
            continue
        seen.add(inp['pitcher'])
        g = inp['state']
        lines.append(f"| {inp['pitcher']} | {inp['game_date']} / {inp['batter']} | {g['inning']}{g['half']}, {g['outs']}아웃, 주자mask {g['bases']}, {g['home_score']}:{g['away_score']} | "
                     f"{a['pitch_type']} ({a['target_x_ft']:+.2f}, {a['target_z_ft']:.2f})ft | {r['planned_defense_we']:.5f} | {r['reference_defense_we']:.5f} |")
    lines += ['', '## 범위와 미완료', '',
              '- 학습 2023-05-15–2025-04-30, 보정/epoch 선택 2025-05-01–06-30, DEV 2025-07-01 이후. 2026 원본은 읽지 않았다.',
              '- 타자 개인 최소 표본 제한 없음. 선수 일별 통계는 같은 날 전체 결과를 제외한 과거만 사용한다.',
              '- 실제 목표 라벨 없음. 목표별 도달 위치는 독립 Gaussian σ=0.30ft 가정, 0.20/0.45ft 민감도 별도 JSON.',
              '- 목표 셀은 포수 시점 물리 좌표(ft). 같은 공의 sz_top/sz_bot나 실현 구속을 투구 전 입력으로 사용하지 않았다.',
              '- 중간 주루·점수·아웃 변화, 오류/특수 이벤트 및 불완전 연결 타석은 제외. 결과를 본 후 제외하는 선택 편향이 있다. 상세 내역은 data_quality.json.',
              '- 실제 경기 종료·끝내기 타석과 연장 타석은 이번 예측/추천 평가에서 제외했다. 솔버의 종료 경계 자체는 별도 테스트했다.',
              '- 제구 모델과 결과 모델의 분포 변화, 타구별 진루, 희귀 사건, 후기 접전의 보정은 추가 검증 필요.',
              '- 독립 심판 모델, sequential OPE, 다중 시드/정책 효과 신뢰구간, RE·neutral 비교는 아직 없다.',
              '- 이 DEV 기간은 프로젝트 전체에서 완전 미사용 평가셋이 아니다. 이번 결과는 실행·예측 검증 파일럿이다.', '',
              '## 재실행', '', '```sh',
              '.venv/bin/python experiments/pitchmdp/scripts/preflight.py',
              '.venv/bin/python experiments/pitchmdp/scripts/prepare_data.py',
              '.venv/bin/python experiments/pitchmdp/scripts/run_first_result.py',
              f'.venv/bin/python experiments/pitchmdp/scripts/recommend_one.py --run "{run}"', '```', '',
              f"추천 계산 중앙값 {summary['median_recommendation_seconds']:.3f}s; 프로세스 peak RSS {runtime['peak_process_rss_bytes']/1024**3:.2f}GiB."]
    (run/'FIRST_RESULT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
