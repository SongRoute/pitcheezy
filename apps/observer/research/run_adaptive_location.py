"""Post-hoc coverage repair and independent common-distribution policy audit.

This is a new exploratory candidate after the first narrow-kernel failure.
No causal intended-target policy improvement is claimed or auto-deployed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'apps/observer/backend'), str(Path(__file__).parent),
                str(REPO/'experiments/pitchmdp/scripts'), str(REPO/'experiments/pitchmdp')]
from observer_app.settings import ARTIFACT_ROOT, BUNDLE
from observer_app.recommender import Recommender, location_weights
from observer_app.domain import ZONES, target_point
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.archetypes import add_batter_style_history, STYLE_COLUMNS, RELIABILITY_COLUMNS
from pitchmdp.model import eligible
from pitchmdp.game import GameState, terminal_values
from pitchmdp.planner import solve_pa
from minimal_pitch_service import evaluate_policy, temperature_predictions
from location_evaluator import legal, row_scores, paired_games
from run_location_evaluation import deterministic_subset, write

ROOT = ARTIFACT_ROOT/'runs/observer-improvement-v2'
SPEC = {'schema_version': 1, 'candidate': 'sigma030_if_supported_else_sigma045_else_type_only',
        'reason': 'First narrow-only candidate lost7.41pp support and failed guard CI; fixed coverage repair',
        'post_hoc': True, 'data_reuse': 'Previously inspected retrospective periods; not confirmatory holdout',
        'sigma_narrow': .30, 'sigma_broad': .45, 'ESS_minimum': 20, 'kernel_mass_minimum': .01,
        'policy_period': ['2025-08-16', '2025-09-30'], 'maximum_pas_per_game': 16,
        'reference_execution_sigmas': [.30, .45, .65], 'primary_reference_sigma': .45,
        'reference_narrow_fallback': 'Use broad distribution on unsupported narrow count-actions',
        'independent_judge': 'July-only observed-location residual logistic; guard gate must pass',
        'paired_unit': 'game', 'bootstrap_replicates': 2000,
        'adoption': 'No automatic serving change; requires support, policy, conditional prediction and approximation caveat review'}


def repair_predictions(source, output):
    summary = {}
    for partition in ('diagnosis', 'guard', 'evaluation'):
        arrays = np.load(source/f'{partition}_proxy_predictions.npz')
        adaptive = np.where(arrays['sigma030_supported'][:, None], arrays['sigma030'], arrays['sigma045'])
        support = arrays['sigma030_supported'] | arrays['sigma045_supported']
        base, candidate = row_scores(arrays['y'], arrays['sigma045']), row_scores(arrays['y'], adaptive)
        summary[partition] = {'rows': len(adaptive), 'games': len(np.unique(arrays['game_ids'])),
            'support_fraction': float(support.mean()), 'baseline_support_fraction': float(arrays['sigma045_supported'].mean()),
            'metrics': {metric: float(values.mean()) for metric, values in candidate.items()},
            'versus_baseline': {metric: paired_games(candidate[metric]-base[metric], arrays['game_ids']) for metric in candidate}}
        np.savez_compressed(output/f'{partition}_adaptive_predictions.npz', p=adaptive, supported=support,
                            y=arrays['y'], game_ids=arrays['game_ids'])
    write(output/'conditional_validation.json', summary)
    print('ADAPTIVE_CONDITIONAL', summary, flush=True)


def action_grid(engine, judge, row, bounds, repertoire):
    types = [name for name in engine.metadata['pitchers'][str(int(row.pitcher))]['pitch_types'] if repertoire.get(name, 0) >= 20]
    records = [{'balls': b, 'strikes': s, 'pitch_type': kind, 'inning': int(row.inning),
                'inning_topbot': row.inning_topbot, 'outs_when_up': int(row.outs_when_up), 'bases': int(row.bases),
                'home_score': int(row.home_score), 'away_score': int(row.away_score), 'pitcher': int(row.pitcher),
                'p_throws': row.p_throws, 'stand': row.stand,
                **{column: float(row[column]) for column in (*STYLE_COLUMNS, *RELIABILITY_COLUMNS)}}
               for b in range(4) for s in range(3) for kind in types]
    grid = pd.DataFrame(records)
    if not len(grid):
        return None
    draws, _ = engine.delivery.sample(grid)
    n, count, physical = draws.shape
    tokens = np.zeros((n*count, 6, physical), np.float32)
    tokens[:, -1] = draws.reshape(-1, physical)
    valid = np.zeros((n*count, 6), bool)
    valid[:, -1] = True
    context = np.repeat(engine.context.transform(grid), count, axis=0)
    neural = np.zeros((n, count, 10), float)
    for model in engine.models:
        logits = model.logits((tokens, valid, context)).reshape(n, count, 10)
        neural += softmax(logits/model.delivery_temperature, axis=-1)/len(engine.models)
    physical_draws = draws*engine.delivery.normalizer.scale+engine.delivery.normalizer.mean
    coordinates = physical_draws[:, :, 6:8]
    targets = np.array([[target_point(zone['id'], bounds)[axis] for axis in ('x', 'z')] for zone in ZONES])
    broad_w, broad_support, broad_mass = location_weights(coordinates, targets, .45)
    narrow_w, narrow_support, narrow_mass = location_weights(coordinates, targets, .30)
    narrow_ok = (narrow_support >= 20) & (narrow_mass >= .01)
    adaptive_w = np.where(narrow_ok[:, None, :], narrow_w, broad_w)
    actions = len(types)*len(ZONES)
    mask = ((broad_support.reshape(4, 3, actions).min(axis=(0, 1)) >= 20) &
            (broad_mass.reshape(4, 3, actions).min(axis=(0, 1)) >= .01))
    if not mask.any():
        return None
    repeated = grid.loc[grid.index.repeat(count)].reset_index(drop=True)
    repeated['plate_x'], repeated['plate_z'] = coordinates[:, :, 0].ravel(), coordinates[:, :, 1].ravel()
    independent = judge.predict(repeated).reshape(n, count, 10)
    frequency = temperature_predictions(engine.baseline.predict(grid), engine.baseline_temperature)
    legal_frame = grid.loc[grid.index.repeat(len(ZONES))].reset_index(drop=True)

    def shape(probabilities):
        return legal(probabilities.reshape(-1, 10), legal_frame).reshape(4, 3, 1, actions, 10)[:, :, :, mask, :]

    probabilities = {name: shape(engine.weight*np.einsum('ndz,ndk->nzk', w, neural)+(1-engine.weight)*frequency[:, None, :])
                     for name, w in [('baseline', broad_w), ('adaptive', adaptive_w)]}
    references = {}
    for sigma in SPEC['reference_execution_sigmas']:
        w, support, mass = location_weights(coordinates, targets, sigma)
        ok = (support >= 20) & (mass >= .01)
        w = np.where(ok[:, None, :], w, broad_w)
        references[str(sigma)] = shape(np.einsum('ndz,ndk->nzk', w, independent))
    baseline_policy = broad_mass.reshape(4, 3, len(types), len(ZONES)).mean(axis=(0, 1))
    baseline_policy *= np.array([repertoire[name] for name in types])[:, None]
    baseline_policy = baseline_policy.ravel()[mask]
    baseline_policy /= baseline_policy.sum()
    center_rows = legal_frame.copy()
    center_rows['plate_x'] = np.tile(targets[:, 0], n)
    center_rows['plate_z'] = np.tile(targets[:, 1], n)
    judge_support = judge.local_support(center_rows).reshape(4, 3, actions)[:, :, mask]
    labels = [(kind, zone['id']) for kind in types for zone in ZONES]
    return probabilities, references, baseline_policy, judge_support, [labels[i] for i in np.flatnonzero(mask)]


def policy_audit(source, output):
    gate = json.loads((source/'evaluator_validation.json').read_text())['policy_judge_gate']
    if not gate:
        raise RuntimeError('Independent evaluator did not pass predictive gate')
    with (source/'evaluators.pkl').open('rb') as stream:
        judge = pickle.load(stream)['location']
    frame = add_batter_style_history(prepare_frame(REPO/'experiments/pitchmdp/configs/local.json'))
    dates = pd.to_datetime(frame.game_date).dt.normalize()
    engine = Recommender().engine
    cohort = [int(value) for value in engine.metadata['pitchers']]
    candidates = frame.loc[dates.between(*SPEC['policy_period']) & frame.pitcher.isin(cohort) &
                           frame.pitch_number.eq(1) & frame.balls.eq(0) & frame.strikes.eq(0) & eligible(frame)]
    selected = pd.concat([deterministic_subset(part, SPEC['maximum_pas_per_game']) for _, part in candidates.groupby('game_pk', sort=True)])
    selected = selected.sort_values(['game_date', 'game_pk', 'at_bat_number'])
    bounds_cache, repertoire_cache, records = {}, {}, []
    cohort_frame = frame.loc[frame.pitcher.isin(cohort)]
    started = time.perf_counter()
    for counter, (_, row) in enumerate(selected.iterrows()):
        date = pd.Timestamp(row.game_date).normalize()
        if date not in bounds_cache:
            prior = frame.loc[dates.lt(date), ['batter', 'sz_bot', 'sz_top']]
            bounds_cache[date] = (prior.groupby('batter')[['sz_bot', 'sz_top']].median(), prior[['sz_bot', 'sz_top']].median())
        table, fallback = bounds_cache[date]
        zone = table.loc[row.batter].fillna(fallback) if row.batter in table.index else fallback
        bounds = {'bottom': float(zone.sz_bot), 'top': float(zone.sz_top)}
        key = (date, int(row.pitcher))
        if key not in repertoire_cache:
            pool = cohort_frame.loc[cohort_frame.pitcher.eq(row.pitcher) & pd.to_datetime(cohort_frame.game_date).lt(date) &
                                    pd.to_datetime(cohort_frame.game_date).ge(date-pd.Timedelta(days=90))]
            repertoire_cache[key] = pool.pitch_type.value_counts().to_dict()
        common = action_grid(engine, judge, row, bounds, repertoire_cache[key])
        item = {'game_id': int(row.game_pk), 'pa_id': int(row.at_bat_number), 'pitcher': int(row.pitcher),
                'date': date.date().isoformat(), 'inning': int(row.inning), 'outs': int(row.outs_when_up), 'bases': int(row.bases)}
        if common is None:
            records.append(item | {'status': 'unavailable'})
            continue
        probabilities, references, baseline_policy, support, labels = common
        state = GameState(int(row.inning), row.inning_topbot, int(row.outs_when_up), int(row.bases), int(row.home_score), int(row.away_score))
        terminal = terminal_values(state, engine.we, engine.advancement)
        policies = {name: solve_pa(p, terminal, [0]*len(labels), baseline_policy=baseline_policy) for name, p in probabilities.items()}
        evaluated = {sigma: {name: float(evaluate_policy(p, terminal, plan.policy)[0, 0, 0]) for name, plan in policies.items()}
                     for sigma, p in references.items()}
        actions = {name: int(plan.policy[0, 0, 0]) for name, plan in policies.items()}
        item.update(status='ready', action_count=len(labels),
                    chosen={name: {'type': labels[action][0], 'zone': labels[action][1], 'judge_observed_cell_count': int(support[0, 0, action])}
                            for name, action in actions.items()}, evaluated=evaluated,
                    policy_different=bool(np.any(policies['baseline'].policy != policies['adaptive'].policy)),
                    first_action_different=actions['baseline'] != actions['adaptive'])
        records.append(item)
        if counter % 25 == 0:
            print('POLICY_PA', counter, len(selected), round(time.perf_counter()-started, 2), flush=True)
            write(output/'policy_progress.json', {'complete': False, 'processed': len(records), 'total': len(selected)})
    ready = [item for item in records if item['status'] == 'ready']
    summary = {'requested_pas': len(records), 'ready_pas': len(ready), 'games': len({item['game_id'] for item in ready}),
               'elapsed_seconds': time.perf_counter()-started,
               'first_action_change_fraction': float(np.mean([item['first_action_different'] for item in ready])),
               'any_count_policy_change_fraction': float(np.mean([item['policy_different'] for item in ready])),
               'paired_adaptive_minus_baseline_pp': {sigma: paired_games(
                   np.array([100*(item['evaluated'][sigma]['adaptive']-item['evaluated'][sigma]['baseline']) for item in ready]),
                   [item['game_id'] for item in ready]) for sigma in ready[0]['evaluated']},
               'scope': 'Common independently predicted outcome probabilities and reference execution distributions; no causal target efficacy claim'}
    high_support = [item for item in ready if all(choice['judge_observed_cell_count'] >= 20 for choice in item['chosen'].values())]
    summary['current_action_judge_support_ge20'] = {'pas': len(high_support), 'games': len({item['game_id'] for item in high_support})}
    if high_support:
        summary['current_action_judge_support_ge20']['paired_primary_pp'] = paired_games(
            np.array([100*(item['evaluated']['0.45']['adaptive']-item['evaluated']['0.45']['baseline']) for item in high_support]),
            [item['game_id'] for item in high_support])
    write(output/'policy_rows.json', records)
    write(output/'policy_summary.json', summary)
    write(output/'policy_progress.json', {'complete': True, 'processed': len(records), 'total': len(selected)})
    print('POLICY_SUMMARY', summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['conditional', 'policy'])
    parser.add_argument('--source', type=Path, default=ROOT/'location-evaluation-v1')
    parser.add_argument('--output', type=Path, default=ROOT/'adaptive-location-v1')
    args = parser.parse_args()
    if not Path('/Volumes/T7 Shield').is_mount() or not args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Approved mounted v2 research output required')
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output/'spec.json'
    if existing.exists() and json.loads(existing.read_text()) != SPEC:
        raise ValueError('Specification differs from existing run')
    write(existing, SPEC)
    write(args.output/'source_hashes.json', {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    completed = args.output/('conditional_validation.json' if args.stage == 'conditional' else 'policy_summary.json')
    if completed.exists():
        raise ValueError('Completed stage already exists; refusing overwrite')
    (repair_predictions if args.stage == 'conditional' else policy_audit)(args.source, args.output)


if __name__ == '__main__':
    main()
