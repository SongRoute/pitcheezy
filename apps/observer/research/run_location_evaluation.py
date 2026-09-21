"""Bounded independent evaluator and observed-location proxy diagnostics.

All candidate choices are made on Aug1–7, guard on Aug8–15, then descriptive
Aug16–Sep30 evaluation. Does not measure causal value of intended targets.
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
from observer_app.recommender import Recommender
from pitchmdp.sequence_data import prepare_frame
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.model import eligible, outcome_labels
from regularized_policy_evaluator import RegularizedPolicyEvaluator
from location_evaluator import LocationEvaluator, legal, row_scores, paired_games

SPEC = {'schema_version': 1, 'fit': ['2025-07-01', '2025-07-21'], 'calibration': ['2025-07-22', '2025-07-31'],
        'diagnosis': ['2025-08-01', '2025-08-07'], 'guard': ['2025-08-08', '2025-08-15'],
        'evaluation': ['2025-08-16', '2025-09-30'], 'maximum_residual_training_rows': 80000,
        'maximum_proxy_rows_per_partition': 6000, 'sampling_seed': 20260921, 'bootstrap_replicates': 2000,
        'candidates': ['type_only', 'sigma030', 'sigma045', 'sigma065', 'sigma045_shrink25'],
        'baseline': 'sigma045', 'ESS_minimum': 20, 'kernel_mass_minimum': .01,
        'scope': 'Retrospective observed type/location conditional prediction; not pre-pitch target policy efficacy',
        'acceptance': 'Choose min diagnosis logloss among .30/.65/shrink25 and baseline; guard paired NLL CI upper<0, Brier delta<=.001, support fraction drop<=.02; final descriptive only'}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))


def load_frame():
    frame = add_batter_style_history(prepare_frame(REPO/'experiments/pitchmdp/configs/local.json'))
    identity = frame.attrs['sequence_data_identity']
    frame = frame.loc[eligible(frame)].copy()
    return frame, identity


def parts(frame):
    dates = pd.to_datetime(frame.game_date)
    return {name: frame.loc[dates.between(*SPEC[name])].copy() for name in ('diagnosis', 'guard', 'evaluation')}


def deterministic_subset(frame, maximum):
    keys = frame[['game_pk', 'at_bat_number', 'pitch_number']].astype(str).agg(':'.join, axis=1)
    ranks = keys.map(lambda key: hashlib.sha256((str(SPEC['sampling_seed'])+key).encode()).hexdigest())
    return frame.loc[ranks.sort_values(kind='stable').index[:maximum]].sort_values(['game_date', 'game_pk', 'at_bat_number', 'pitch_number']).copy()


def fit_evaluator(output):
    frame, identity = load_frame()
    dates = pd.to_datetime(frame.game_date)
    july = frame.loc[dates.between('2025-07-01', '2025-07-31')]
    reports, models = {}, {}
    for name, cls in [('no_location', RegularizedPolicyEvaluator), ('location', LocationEvaluator)]:
        print('FIT', name, len(july), flush=True)
        started = time.perf_counter()
        model = cls(max_training_rows=SPEC['maximum_residual_training_rows']).fit(july)
        reports[name] = model.report | {'elapsed_seconds': time.perf_counter()-started}
        models[name] = model
    with (output/'evaluators.pkl').open('wb') as stream:
        pickle.dump(models, stream)
    write(output/'fit_report.json', reports)
    write(output/'data_identity.json', identity)
    diagnostics = {}
    cohorts = {int(x) for x in json.loads((BUNDLE/'metadata.json').read_text())['pitchers']}
    for name, part in parts(frame).items():
        if name == 'evaluation':
            part = part.loc[part.pitcher.isin(cohorts)]
        y = outcome_labels(part)
        scores = {key: row_scores(y, legal(model.predict(part), part)) for key, model in models.items()}
        diagnostics[name] = {'rows': len(part), 'games': int(part.game_pk.nunique()),
            'metrics': {key: {metric: float(values.mean()) for metric, values in model_scores.items()} for key, model_scores in scores.items()},
            'location_minus_no_location': {metric: paired_games(scores['location'][metric]-scores['no_location'][metric], part.game_pk)
                                           for metric in ('log_loss', 'brier')}}
        print('EVALUATOR', name, diagnostics[name]['metrics'], flush=True)
    diagnostics['policy_judge_gate'] = diagnostics['guard']['location_minus_no_location']['log_loss']['ci95'][1] < 0
    write(output/'evaluator_validation.json', diagnostics)
    print('EVALUATOR_GATE', diagnostics['policy_judge_gate'], flush=True)


def predict_proxies(frame, engine, batch_size=64):
    from minimal_pitch_service import temperature_predictions
    predictions = {name: [] for name in SPEC['candidates']}
    support = {name: [] for name in SPEC['candidates']}
    for start in range(0, len(frame), batch_size):
        piece = frame.iloc[start:start+batch_size]
        draws, _ = engine.delivery.sample(piece)
        n, count, physical = draws.shape
        tokens = np.zeros((n*count, 6, physical), np.float32)
        tokens[:, -1] = draws.reshape(-1, physical)
        valid = np.zeros((n*count, 6), bool)
        valid[:, -1] = True
        context = np.repeat(engine.context.transform(piece), count, axis=0)
        neural = np.zeros((n, count, 10), float)
        for model in engine.models:
            logits = model.logits((tokens, valid, context)).reshape(n, count, 10)
            neural += softmax(logits/model.delivery_temperature, axis=-1)/len(engine.models)
        frequency = temperature_predictions(engine.baseline.predict(piece), engine.baseline_temperature)
        broad = neural.mean(axis=1)
        coordinates = (draws*engine.delivery.normalizer.scale+engine.delivery.normalizer.mean)[:, :, 6:8]
        actual = piece[['plate_x', 'plate_z']].to_numpy(float)
        distance2 = np.square(coordinates-actual[:, None, :]).sum(-1)
        for name in SPEC['candidates']:
            if name == 'type_only':
                localized, supported = broad, np.ones(n, bool)
            else:
                sigma = {'sigma030': .30, 'sigma045': .45, 'sigma065': .65, 'sigma045_shrink25': .45}[name]
                logw = -distance2/(2*sigma*sigma)
                mass = np.exp(logw).mean(axis=1)
                w = np.exp(logw-logw.max(axis=1, keepdims=True))
                w /= w.sum(axis=1, keepdims=True)
                ess = 1/np.square(w).sum(axis=1)
                supported = (ess >= SPEC['ESS_minimum']) & (mass >= SPEC['kernel_mass_minimum'])
                localized = np.einsum('nd,ndk->nk', w, neural)
                if name == 'sigma045_shrink25':
                    localized = .75*localized+.25*broad
                localized[~supported] = broad[~supported]
            p = engine.weight*localized+(1-engine.weight)*frequency
            predictions[name].append(legal(p, piece))
            support[name].append(supported)
        if start % 1024 == 0:
            print('PROXY_ROWS', start, len(frame), flush=True)
    return {name: np.concatenate(values) for name, values in predictions.items()}, {name: np.concatenate(values) for name, values in support.items()}


def evaluate_proxies(output):
    frame, identity = load_frame()
    if identity != json.loads((output/'data_identity.json').read_text()):
        raise RuntimeError('Source data identity changed between fitting and evaluation')
    recommender = Recommender()
    metadata = recommender.engine.metadata
    allowed = np.zeros(len(frame), dtype=bool)
    for pitcher, details in metadata['pitchers'].items():
        allowed |= (frame.pitcher.eq(int(pitcher)) & frame.pitch_type.isin(details['pitch_types'])).to_numpy()
    selected = parts(frame.loc[allowed])
    summaries, score_tables, support_tables = {}, {}, {}
    for name, part in selected.items():
        part = deterministic_subset(part, SPEC['maximum_proxy_rows_per_partition'])
        predictions, support = predict_proxies(part, recommender.engine)
        y = outcome_labels(part)
        scores = {candidate: row_scores(y, p) for candidate, p in predictions.items()}
        scores['game_ids'] = part.game_pk.to_numpy()
        score_tables[name], support_tables[name] = scores, support
        metrics = {candidate: {metric: float(value.mean()) for metric, value in scores[candidate].items()} |
                   {'support_fraction': float(support[candidate].mean())} for candidate in SPEC['candidates']}
        summaries[name] = {'rows': len(part), 'games': int(part.game_pk.nunique()), 'metrics': metrics,
            'versus_sigma045': {candidate: {metric: paired_games(scores[candidate][metric]-scores['sigma045'][metric], part.game_pk)
                                            for metric in ('log_loss', 'brier')} for candidate in SPEC['candidates'] if candidate != 'sigma045'}}
        np.savez_compressed(output/f'{name}_proxy_predictions.npz', y=y, game_ids=part.game_pk.to_numpy(),
                            at_bat_number=part.at_bat_number.to_numpy(), pitch_number=part.pitch_number.to_numpy(),
                            **predictions, **{key+'_supported': value for key, value in support.items()})
        part[['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'pitch_type', 'stand', 'balls', 'strikes',
              'outs_when_up', 'bases', 'plate_x', 'plate_z']].to_parquet(output/f'{name}_proxy_rows.parquet', index=False)
        print('PROXY_METRICS', name, metrics, flush=True)
    candidates = ['sigma030', 'sigma045', 'sigma065', 'sigma045_shrink25']
    chosen = min(candidates, key=lambda candidate: summaries['diagnosis']['metrics'][candidate]['log_loss'])
    decision = {'selected_on_diagnosis': chosen, 'deployed_baseline': 'sigma045', 'accepted': False,
                'reason': 'Baseline retained; intent-target policy efficacy not established by conditional prediction.'}
    if chosen != 'sigma045':
        comparison = summaries['guard']['versus_sigma045'][chosen]
        support_delta = summaries['guard']['metrics'][chosen]['support_fraction']-summaries['guard']['metrics']['sigma045']['support_fraction']
        passed = comparison['log_loss']['ci95'][1] < 0 and comparison['brier']['difference'] <= .001 and support_delta >= -.02
        decision.update(guard_predictive_gate_passed=bool(passed), guard_support_delta=support_delta,
                        reason='Predictive candidate requires separate target-policy/support audit before service adoption.' if passed else 'Guard predictive/support gate failed; baseline retained.')
    summaries['decision'] = decision
    summaries['model_identity'] = recommender.identity
    summaries['scope'] = SPEC['scope']
    write(output/'proxy_validation.json', summaries)
    print('DECISION', decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'proxy'])
    parser.add_argument('--output', type=Path, default=ARTIFACT_ROOT/'runs/observer-improvement-v2/location-evaluation-v1')
    args = parser.parse_args()
    if not Path('/Volumes/T7 Shield').is_mount() or not args.output.resolve().is_relative_to(ARTIFACT_ROOT/'runs'):
        raise ValueError('Mounted approved SSD output required')
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output/'spec.json'
    if config.exists() and json.loads(config.read_text()) != SPEC:
        raise ValueError('Protocol changed; use a new output')
    write(config, SPEC)
    write(args.output/'source_hashes.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__), Path(__file__).with_name('location_evaluator.py')]})
    if args.stage == 'fit':
        if (args.output/'evaluators.pkl').exists():
            raise ValueError('Refusing to overwrite fitted evaluators')
        fit_evaluator(args.output)
    else:
        if (args.output/'proxy_validation.json').exists():
            raise ValueError('Refusing to overwrite completed evaluation')
        evaluate_proxies(args.output)


if __name__ == '__main__':
    main()
