"""Build a compact historical recommendation bundle and bounded end-to-end evaluation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from pitchmdp.archetypes import add_batter_style_history, STYLE_COLUMNS, RELIABILITY_COLUMNS, INITIAL_RATES
from pitchmdp.data import KEY, hash_file
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.game import GameState, WinExpectancy, EmpiricalAdvancement, terminal_values
from pitchmdp.planner import solve_pa, TERMINALS
from run_sequence_pilot import dump
from run_temporal_blend import assign_fold, select_cohort, samples_for, check_sources
from run_representation_history import verify_manifest
from run_sequence_frequency_blend import paired_fixed_predictors
from pitchmdp.sequence_model import classification_metrics
from diagnose_sequence_legality import condition_on_legality


def fit_run_expectancy(train):
    """TRAIN PA-start remaining runs; exclude halves with no observed third out."""
    keys = ['game_pk', 'inning', 'inning_topbot']
    ordered = train.sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
    ends = ordered.groupby(keys, sort=False).tail(1).copy()
    ends = ends.loc[(ends.outs_when_up+ends.outs_recorded).eq(3)]
    ends['half_final_runs'] = np.where(ends.inning_topbot.eq('Top'), ends.post_away_score, ends.post_home_score)
    starts = ordered.drop_duplicates(['game_pk', 'at_bat_number']).copy()
    starts = starts.merge(ends[keys+['half_final_runs']], on=keys, how='inner', validate='many_to_one')
    starts['remaining'] = starts.half_final_runs-np.where(starts.inning_topbot.eq('Top'), starts.away_score, starts.home_score)
    if starts.remaining.lt(0).any():
        raise ValueError('Negative observed remaining runs')
    means = starts.groupby(['outs_when_up', 'bases']).remaining.mean()
    if len(means) != 24:
        raise ValueError('Expected all24 TRAIN run-expectancy states')
    return {f'{int(outs)}:{int(bases)}': float(value) for (outs, bases), value in means.items()}


def re_terminals(state, advancement, table):
    values = {}
    for event in TERMINALS:
        value = 0.
        for probability, nxt in advancement.distribution(state, event):
            runs = (nxt.home_score-state.home_score if state.half == 'Bot' else nxt.away_score-state.away_score)
            continuation = (table[f'{nxt.outs}:{nxt.bases}']
                            if nxt.winner is None and (nxt.inning, nxt.half) == (state.inning, state.half) else 0.)
            value -= probability*(runs+continuation)
        values[event] = float(value)
    return values


def request_for(row):
    return {'inning': int(row.inning), 'topbot': str(row.inning_topbot), 'outs': int(row.outs_when_up),
            'bases': int(row.bases), 'home_score': int(row.home_score), 'away_score': int(row.away_score),
            'balls': int(row.balls), 'strikes': int(row.strikes), 'pitcher_id': int(row.pitcher),
            'batter_stand': str(row.stand), 'batter_id': int(row.batter), 'top_k': 3,
            'batter_profile': {'rates': [float(row[k]) for k in STYLE_COLUMNS],
                               'reliabilities': [float(row[k]) for k in RELIABILITY_COLUMNS]}}


def select_cases(dev):
    """Pre-state only: one start/game, prefer late close runner situations; <=12games/pitcher."""
    starts = dev.loc[dev.balls.eq(0) & dev.strikes.eq(0) & dev.pitch_number.eq(1)].copy()
    starts['priority'] = np.where(starts.inning.ge(5) & (starts.home_score-starts.away_score).abs().le(2) & starts.bases.gt(0), 0,
                                 np.where(starts.inning.ge(5), 1, 2))
    selected = starts.sort_values(['priority', 'game_date', *KEY]).drop_duplicates(['pitcher', 'game_pk'])
    return selected.sort_values(['game_date', *KEY]).groupby('pitcher', sort=False).head(12)


def policy_values(p, frequency, we_values, re_values, usage, balls, strikes):
    next_prev = [0]*p.shape[3]
    plans = {'we': solve_pa(p, we_values, next_prev, usage), 're': solve_pa(p, re_values, next_prev, usage)}
    root = (balls, strikes, 0)
    values = {}
    for evaluator_name, evaluator_p in [('mixed_model', p), ('frequency_stress', frequency)]:
        reference = solve_pa(evaluator_p, we_values, next_prev, usage)
        scores = {'frequency': float(reference.baseline_values[root])}
        for objective, plan in plans.items():
            one_action = int(np.argmax(plan.myopic_q_values[root]))
            scores[objective+'_one'] = float(reference.myopic_q_values[root][one_action])
            policy = np.eye(p.shape[3])[plan.policy]
            evaluation = solve_pa(evaluator_p, we_values, next_prev, policy)
            scores[objective+'_full'] = float(evaluation.baseline_values[root])
        values[evaluator_name] = scores
    return values, {name: {'one': int(np.argmax(plan.myopic_q_values[root])), 'full': int(plan.policy[root])}
                    for name, plan in plans.items()}


def summarize_policy(cases):
    games, inverse = np.unique([row['game_pk'] for row in cases], return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(42).integers(0, len(games), (2000, len(games)))
    result = {}
    for evaluator in ('mixed_model', 'frequency_stress'):
        result[evaluator] = {}
        for left, right in [('we_full', 'frequency'), ('we_full', 're_full'), ('we_full', 'we_one')]:
            delta = 100*np.array([row['values'][evaluator][left]-row['values'][evaluator][right] for row in cases])
            boot = np.bincount(inverse, weights=delta)[draws].sum(1)/counts[draws].sum(1)
            result[evaluator][left+'_minus_'+right] = {'mean_we_percentage_points': float(delta.mean()),
                'game_bootstrap95': np.quantile(boot, [.025, .975]).tolist()}
    return {'cases': len(cases), 'games': len(games), 'comparisons': result,
            'scope': 'Selected historical states; fixed-model descriptive intervals, no multiplicity correction. '
                     'Frequency stress shares TRAIN and WE model; NOT independent or causal policy evaluation.'}


def build(root, output):
    started = time.perf_counter()
    history = root/'runs/history-batter-validation-v1'
    temporal = root/'runs/temporal-blend-20260921T050400Z'
    representation = root/'runs/representation-history-20260921T054642Z'
    for source in (history, temporal, representation):
        verify_manifest(source)
        check_sources(source/'setup' if source == history else source)
    if output.exists():
        raise ValueError('Use a new output directory; completed bundles are immutable')
    local = json.loads((PROJECT/'configs/local.json').read_text())
    raw = add_batter_style_history(prepare_frame(local))
    if raw.attrs['sequence_data_identity'] != json.loads((history/'data_identity.json').read_text()):
        raise ValueError('Frozen data identity mismatch')
    frame = assign_fold(raw, 2025)
    ids, cohort = select_cohort(frame)
    full, parts = samples_for(frame, ids)
    with (temporal/'2025/fitted_preprocessors.pkl').open('rb') as stream:
        prep = pickle.load(stream)
    with (representation/'2025/continuous/context.pkl').open('rb') as stream:
        context = pickle.load(stream)
    result = json.loads((history/'results.json').read_text())['folds']['2025']['variants']['continuous_h0']
    baseline_temp = json.loads((temporal/'2025/baseline_calibration.json').read_text())['temperature']
    pitchers = {}
    usage = {}
    for pid in ids:
        rows = full.loc[full.pitcher.eq(pid)]
        counts = rows.pitch_type.value_counts()
        types = counts[counts.ge(50)].index.astype(str).tolist()
        if not types:
            raise ValueError('No adequately supported repertoire')
        pitchers[str(pid)] = {'p_throws': str(rows.p_throws.mode().iloc[0]), 'pitch_types': types,
                             'name': str(rows.player_name.iloc[-1])}
        usage[str(pid)] = {name: int(counts[name]) for name in types}
    # Preserve exact candidate vectors, retaining only keys this restricted service can query.
    required = set()
    for pid, info in pitchers.items():
        for pitch_type in info['pitch_types']:
            for stand in ('L', 'R'):
                for b in range(4):
                    for s in range(3):
                        row = dict(pitcher=int(pid), pitch_type=pitch_type, p_throws=info['p_throws'], stand=stand, balls=b, strikes=s)
                        required.update((level, tuple(row[key] for key in keys)) for level, keys in enumerate(prep['delivery'].TIERS))
    prep['delivery'].pools = {key: value for key, value in prep['delivery'].pools.items() if key in required}
    profiles = {}
    for _, row in frame.loc[frame.split.eq('train')].sort_values(['game_date', *KEY]).drop_duplicates('batter', keep='last').iterrows():
        profiles[str(int(row.batter))] = {'rates': [float(row[k]) for k in STYLE_COLUMNS],
            'reliabilities': [float(row[k]) for k in RELIABILITY_COLUMNS],
            'as_of': str(pd.Timestamp(row.game_date).date())}
    train = frame.loc[frame.split.eq('train')]
    print('FIT_LIGHTWEIGHT_GAME_VALUES', flush=True)
    we, advancement = WinExpectancy().fit(train), EmpiricalAdvancement().fit(train)
    re_table = fit_run_expectancy(train)
    we_evaluation = we.evaluate(frame.loc[frame.split.eq('dev')])
    labeled = we.labeled_states(frame.loc[frame.split.eq('dev')])
    base_probability = float(we.labeled_states(train).final_home_win.mean())
    targets = labeled.final_home_win.to_numpy(float)
    we_evaluation['train_constant_brier'] = float(np.mean((targets-base_probability)**2))
    if not np.isfinite(we_evaluation['brier']) or we_evaluation['brier'] >= we_evaluation['train_constant_brier']:
        raise ValueError('WE model fails the minimal held-out Brier gate')
    output.mkdir(parents=True)
    (output/'models').mkdir()
    model_names = []
    source_artifacts = []
    for seed in range(42, 47):
        source = history/f'2025/continuous/seed{seed}/calibration/model.pt'
        destination = f'models/seed{seed}.pt'
        shutil.copyfile(source, output/destination)
        model_names.append(destination)
        source_artifacts.append(source)
    for filename, obj in [('preprocessor.pkl', {'context': context, 'delivery': prep['delivery'], 'baseline': prep['baseline']}),
                           ('game_values.pkl', {'we': we, 'advancement': advancement, 'run_expectancy': re_table})]:
        with (output/filename).open('wb') as stream:
            pickle.dump(obj, stream)
    metadata = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'neural_weight': result['selection']['model_weight'], 'baseline_temperature': baseline_temp,
        'models': model_names, 'pitchers': pitchers, 'profiles': profiles, 'profile_cutoff': '2025-04-30',
        'default_profile': {'rates': INITIAL_RATES.tolist(), 'reliabilities': [0.]*6, 'as_of': '2023-01-01'},
        'repertoire_counts': usage, 'style_names': list(STYLE_COLUMNS), 'training_cutoff': '2025-04-30',
        'scope': 'Historical local MVP; pitch types only; fixed TRAIN delivery and profile snapshots. '
                 'Game state fixed within supported PA until terminal. No live feed or causal win gain.',
        'provenance': {str(p): hash_file(p) for p in source_artifacts}}
    dump(output/'metadata.json', metadata)
    files = ['metadata.json', 'preprocessor.pkl', 'game_values.pkl', *model_names]
    dump(output/'bundle_manifest.json', {'sha256': {name: hash_file(output/name) for name in files}})
    print('BUNDLE_READY', output, flush=True)
    from minimal_pitch_service import Engine
    engine = Engine(output)
    selected = select_cases(parts['dev'])
    selected[KEY+['pitcher', 'game_date']].to_parquet(output/'evaluation_cases.parquet', index=False)
    requests = [request_for(row) for _, row in selected.iterrows()]
    dump(output/'example_request.json', requests[0])
    # Real observed-pitch prediction evidence from the EXACT weights/mixture packaged here.
    with np.load(history/'2025/predictions.npz') as archive:
        y, games = archive['y'], archive['game_pk']
        model_p, base_p = archive['continuous_h0'], archive['frequency']
        keys = {tuple(key): i for i, key in enumerate(archive['pitch_keys'])}
        prediction_evidence = {'model': classification_metrics(y, model_p), 'baseline': classification_metrics(y, base_p),
            'paired': paired_fixed_predictors(y, model_p, base_p, games),
            'source': 'Existing fixed2025 historical evaluation, not a new independent test',
            'note': 'Raw observational forecasts before service legality projection; policy effect not measured.'}
        archived_p = model_p.copy()
    cases, latencies, replay_errors = [], [], []
    for (_, row), request in zip(selected.iterrows(), requests):
        start = time.perf_counter()
        data = engine.predict_counts(request)
        latency = time.perf_counter()-start
        latencies.append(latency)
        actions = data['pitch_types']
        weights = np.array([usage[str(int(row.pitcher))][name] for name in actions], float)
        weights /= weights.sum()
        state = data['state']
        we_values = terminal_values(state, we, advancement)
        values, choices = policy_values(data['probabilities']['blend'], data['probabilities']['frequency'],
                                       we_values, re_terminals(state, advancement, re_table), weights,
                                       int(row.balls), int(row.strikes))
        if row.pitch_type in actions:
            expected = archived_p[keys[tuple(row[KEY])]].copy()
            expected = condition_on_legality(expected[None], np.array([state.outs == 2 or state.bases == 0]))[0]
            actual = data['probabilities']['blend'][int(row.balls), int(row.strikes), 0, actions.index(row.pitch_type)]
            error = float(np.max(np.abs(expected-actual)))
            replay_errors.append(error)
            if error > 1e-5:
                raise ValueError(f'Serving inference differs from archived forecast: {error}')
        cases.append({'game_pk': int(row.game_pk), 'pitcher': int(row.pitcher), 'request': request,
                      'pitch_types': actions, 'choices': choices, 'values': values, 'seconds': latency})
        print('CASE', len(cases), '/', len(selected), 'seconds', round(latency, 3), flush=True)
    example = engine.recommend(requests[0])
    dump(output/'example_response.json', example)
    final = {'prediction_evidence': prediction_evidence, 'we_evaluation': we_evaluation,
        'policy_pilot': summarize_policy(cases), 'cases': cases, 'cohort': cohort,
        'case_selection': 'One eligible0-0 PA per pitcher/game; prefer inning>=5, close<=2runs, runners; '
                          'else first inning>=5, else first. Earliest12 games/pitcher. Selection uses pre-pitch state.',
        'serving_replay_max_probability_error': max(replay_errors), 'replayed_observed_pitches': len(replay_errors),
        'latency_seconds': {'median': float(np.median(latencies)), 'p95': float(np.quantile(latencies, .95))},
        'bundle_bytes': sum((output/name).stat().st_size for name in files),
        'seconds': time.perf_counter()-started, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Minimal service readiness + existing historical prediction gain + model-internal policy pilot. '
                 'Frequency stress is not independent evaluation. No real-world win-probability claim.'}
    dump(output/'evaluation.json', final)
    source_files = [Path(__file__).resolve(), PROJECT/'scripts/minimal_pitch_service.py',
                    PROJECT/'docs/MINIMAL_SERVICE_PROTOCOL.md']
    source_files += list((PROJECT/'pitchmdp').glob('*.py'))
    source_files += [PROJECT/'scripts'/name for name in ('representation_adapters.py', 'run_sequence_context_frequency.py',
                       'run_sequence_frequency_baselines.py', 'run_temporal_blend.py', 'run_sequence_frequency_blend.py',
                       'diagnose_sequence_legality.py')]
    for source in source_files:
        target = output/'source'/source.relative_to(PROJECT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    dump(output/'source_hashes.json', {str(p.relative_to(PROJECT)): hash_file(p) for p in source_files})
    dump(output/'artifact_hashes.json', {str(p.relative_to(output)): hash_file(p) for p in output.rglob('*')
                                       if p.is_file() and p.name != 'artifact_hashes.json'})
    print('COMPLETE', output, 'seconds', round(final['seconds'], 1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    local = json.loads((PROJECT/'configs/local.json').read_text())
    root = Path(local['artifact_root']).resolve()
    output = (args.output or root/'runs/minimal-pitch-service-v1').resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not root.is_relative_to(Path('/Volumes/T7 Shield')):
        raise ValueError('Mounted approved SSD required')
    if not output.is_relative_to(root/'runs') or output == root/'runs':
        raise ValueError('Use a new SSD run directory')
    if Path(sys.executable).absolute() != Path(local['python']).absolute():
        raise ValueError('Use the existing configured Python')
    build(root, output)


if __name__ == '__main__':
    main()
