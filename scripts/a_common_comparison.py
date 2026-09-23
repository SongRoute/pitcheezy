"""Frozen common April-to-July prediction and model-internal PA comparison."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT/'experiments/pitchmdp'
sys.path[:0] = [str(ROOT/'scripts'), str(PROJECT/'scripts'), str(PROJECT)]

import numpy as np
import pandas as pd
from pitchmdp.data import KEY
from pitchmdp.game import terminal_values
from pitchmdp.model import CountBaseline, OUTCOMES, outcome_labels
from pitchmdp.planner import solve_pa
from run_sequence_frequency_baselines import HierarchicalFrequencyBaseline, COUNT_KEYS, TYPE_KEYS, PITCHER_KEYS
from minimal_pitch_service import Engine, evaluate_policy
from a_small_eval import cohort_pa_rows, legal_baseline, score_metrics, paired_bootstrap, digest, frozen_source_check

CONFIG = ROOT/'configs/EXP-A-COMMON-001.json'
REPORT = ROOT/'results/EXP-A-COMMON-001'
NAMES = ('count_hand', 'known_type', 'pitcher_type', 'frozen_frequency', 'frozen_blend')
JUDGES = ('frozen_frequency', 'frozen_blend')


def pa_request(start):
    return {'inning': int(start.inning), 'topbot': str(start.inning_topbot),
            'outs': int(start.outs_when_up), 'bases': int(start.bases),
            'home_score': int(start.home_score), 'away_score': int(start.away_score),
            'balls': 0, 'strikes': 0, 'pitcher_id': int(start.pitcher),
            'batter_stand': str(start.stand), 'batter_id': int(start.batter),
            'date': str(pd.Timestamp(start.game_date).date()), 'top_k': 3}


def grid(start, choices):
    records = []
    for balls in range(4):
        for strikes in range(3):
            for pitch_type in choices:
                records.append({'balls': balls, 'strikes': strikes, 'pitch_type': pitch_type,
                                'stand': start.stand, 'p_throws': start.p_throws,
                                'pitcher': start.pitcher,
                                'outs_when_up': start.outs_when_up, 'bases': start.bases})
    return pd.DataFrame(records)


def game_bootstrap(values, games, seed, reps):
    frame = pd.DataFrame({'game': games, 'value': values})
    groups = frame.groupby('game').value.agg(['sum', 'count'])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), (reps, len(groups)))
    draws = groups['sum'].to_numpy()[picks].sum(1)/groups['count'].to_numpy()[picks].sum(1)
    return {'mean': float(np.mean(values)), 'game_bootstrap95': np.quantile(draws, [.025, .975]).tolist(),
            'games': len(groups), 'n_pa': len(values), 'seed': seed, 'replicates': reps}


def policy_matrix(rows, judge_names, seed, reps):
    games = np.array([row['game_pk'] for row in rows])
    result = {}
    for judge in judge_names:
        base = np.array([row['values'][judge]['repertoire'] for row in rows])
        result[judge] = {'repertoire_mean_we': float(base.mean()),
                         'policies': {}}
        for policy in NAMES:
            v = np.array([row['values'][judge][policy] for row in rows])
            result[judge]['policies'][policy] = {
                'defensive_we': game_bootstrap(v, games, seed, reps),
                'delta_vs_repertoire_pp': game_bootstrap(100*(v-base), games, seed, reps)}
    return result


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text())
    assert tuple(cfg['models']) == NAMES and tuple(cfg['policy_judges']) == JUDGES
    assert cfg['small_train_hierarchy'] == {'global_additive_count': 1, 'count_strength': 50,
                                           'type_strength': 50, 'pitcher_strength': 100}
    manifest_path = ROOT/cfg['parent_manifest']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['experiment_id'] == 'EXP-A-S0S1-001'
    dest = Path(cfg['artifact_dir'])
    if dest.exists() or REPORT.exists():
        raise ValueError('Immutable common comparison output already exists')
    paths = {kind: Path(manifest['dataset_paths'][kind]) for kind in ('train', 'dev')}
    for kind, path in paths.items():
        assert digest(path) == manifest['sha256'][kind+'_full_games.parquet']
    a_prediction = paths['dev'].parent/'predictions.npz'
    a_result = json.loads((ROOT/'results/EXP-A-S0S1-001/results.json').read_text())
    assert digest(a_prediction) == a_result['output_sha256']['predictions']
    bundle = Path(cfg['bundle_dir'])
    assert digest(bundle/'bundle_manifest.json') == manifest['sha256']['bundle_manifest']
    assert digest(bundle/'metadata.json') == manifest['sha256']['bundle_metadata']
    source_hash = frozen_source_check(bundle)
    engine = Engine(bundle)
    train, train_reasons, train_pas = cohort_pa_rows(pd.read_parquet(paths['train']), cfg['pitchers'])
    dev, dev_reasons, dev_pas = cohort_pa_rows(pd.read_parquet(paths['dev']), cfg['pitchers'], engine.metadata['pitchers'])
    assert (len(train), len(dev), train.groupby(KEY[:2]).ngroups, dev.groupby(KEY[:2]).ngroups) == (578, 563, 145, 151)
    parent_support = a_result['support']
    assert (train_pas, dev_pas, train_reasons, dev_reasons) == (
        parent_support['train_cohort_pas'], parent_support['dev_cohort_pas'],
        parent_support['train_excluded_pa_reasons'], parent_support['dev_excluded_pa_reasons'])
    models = {'count_hand': CountBaseline().fit(train),
              'known_type': HierarchicalFrequencyBaseline(False).fit(train),
              'pitcher_type': HierarchicalFrequencyBaseline(True).fit(train)}
    dest.mkdir(parents=True)
    with (dest/'small_models.pkl').open('wb') as stream:
        pickle.dump(models, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (dest/'small_models.pkl').open('rb') as stream:
        restored = pickle.load(stream)
    for name in models:
        np.testing.assert_array_equal(models[name].predict(dev), restored[name].predict(dev))
    with np.load(a_prediction) as archive:
        np.testing.assert_array_equal(dev[KEY].to_numpy(dtype=int), archive['pitch_keys'])
        np.testing.assert_array_equal(outcome_labels(dev), archive['y'])
        y = archive['y'].copy()
        saved = {name: archive[source].copy() for name, source in
                 [('count_hand', 'april_count'), ('frozen_frequency', 'frozen_frequency'),
                  ('frozen_blend', 'frozen_blend')]}
    np.testing.assert_array_equal(legal_baseline(restored['count_hand'].predict(dev), dev), saved['count_hand'])
    predictions = {'count_hand': saved['count_hand'],
                   'known_type': legal_baseline(restored['known_type'].predict(dev), dev),
                   'pitcher_type': legal_baseline(restored['pitcher_type'].predict(dev), dev),
                   'frozen_frequency': saved['frozen_frequency'], 'frozen_blend': saved['frozen_blend']}
    origins = {name: restored[name].predict_with_origin(dev)[1] for name in ('known_type', 'pitcher_type')}
    prediction_rows = {name: [] for name in NAMES}
    policies = []
    action_support = Counter()
    for (game, pa), rows in dev.groupby(KEY[:2], sort=False):
        rows = rows.sort_values('pitch_number')
        start = rows.iloc[0]
        inference = engine.predict_counts(pa_request(start))
        assert inference['profile_as_of'] < str(pd.Timestamp(start.game_date).date())
        choices = list(inference['pitch_types'])
        assert set(rows.pitch_type).issubset(choices)
        grid_rows = grid(start, choices)
        tensor = {name: legal_baseline(restored[name].predict(grid_rows), grid_rows).reshape(4, 3, 1, len(choices), 10)
                  for name in ('count_hand', 'known_type', 'pitcher_type')}
        tensor.update(frozen_frequency=inference['probabilities']['frequency'],
                      frozen_blend=inference['probabilities']['blend'])
        for name in NAMES:
            assert tensor[name].shape == (4, 3, 1, len(choices), 10)
            np.testing.assert_allclose(tensor[name].sum(-1), 1, atol=1e-8)
        for row in rows.itertuples(index=False):
            action = choices.index(row.pitch_type)
            action_support[(int(row.pitcher), row.pitch_type)] += 1
            prediction_rows['count_hand'].append(tensor['count_hand'][row.balls, row.strikes, 0, action])
            prediction_rows['known_type'].append(tensor['known_type'][row.balls, row.strikes, 0, action])
            prediction_rows['pitcher_type'].append(tensor['pitcher_type'][row.balls, row.strikes, 0, action])
            prediction_rows['frozen_frequency'].append(tensor['frozen_frequency'][row.balls, row.strikes, 0, action])
            prediction_rows['frozen_blend'].append(tensor['frozen_blend'][row.balls, row.strikes, 0, action])
        usage_counts = engine.metadata['repertoire_counts'][str(int(start.pitcher))]
        usage = np.asarray([usage_counts.get(name, 0) for name in choices], float)
        assert (usage >= 0).all() and usage.sum() > 0
        usage /= usage.sum()
        terminal = terminal_values(inference['state'], engine.we, engine.advancement)
        fitted = {name: solve_pa(tensor[name], terminal, [0]*len(choices), baseline_policy=usage) for name in NAMES}
        values = {}
        for judge in JUDGES:
            values[judge] = {'repertoire': float(fitted[judge].baseline_values[0, 0, 0])}
            for candidate in NAMES:
                policy = fitted[candidate].policy if candidate != 'count_hand' else None
                values[judge][candidate] = (values[judge]['repertoire'] if policy is None else
                    float(evaluate_policy(tensor[judge], terminal, policy)[0, 0, 0]))
        policies.append({'game_pk': int(game), 'at_bat_number': int(pa), 'pitcher': int(start.pitcher),
                         'action_count': len(choices), 'pitch_types': choices,
                         'values': values})
    for name in NAMES:
        np.testing.assert_array_equal(np.asarray(prediction_rows[name]), predictions[name])
    games = dev.game_pk.to_numpy(dtype=int)
    slices = {'all': np.ones(len(dev), bool), 'two_strikes': dev.strikes.to_numpy() == 2}
    assert int(slices['two_strikes'].sum()) == 159
    scores = {}
    for label, mask in slices.items():
        scores[label] = {'pitches': int(mask.sum()), 'games': int(len(np.unique(games[mask]))),
                         'models': {name: score_metrics(y[mask], predictions[name][mask]) for name in NAMES},
                         'paired_nll_vs_count_hand': {name: paired_bootstrap(y[mask], predictions[name][mask],
                             predictions['count_hand'][mask], games[mask], cfg['bootstrap_seed'], cfg['bootstrap_replicates'])
                             for name in NAMES if name != 'count_hand'}}
    np.savez_compressed(dest/'predictions.npz', pitch_keys=dev[KEY].to_numpy(dtype=int), game_pk=games, y=y, **predictions)
    result = {'experiment_id': cfg['experiment_id'], 'phase': 'evaluated_exploratory_dev',
              'scope': 'small April frequency tables versus preexisting large-TRAIN/CAL frozen models; exposed July DEV',
              'scores': scores, 'policy_internal': policy_matrix(policies, JUDGES, cfg['bootstrap_seed'], cfg['bootstrap_replicates']),
              'policy_causal_we': None, 'ope_ess': None,
              'policy_limit': 'Same frozen judges are model-internal, not independent; optimized diagonal mechanically advantaged.',
              'support': {'train_pas': train.groupby(KEY[:2]).ngroups, 'dev_pas': dev.groupby(KEY[:2]).ngroups,
                          'train_pitches': len(train), 'dev_pitches': len(dev),
                          'hierarchy_groups': {name: models[name].report for name in ('known_type', 'pitcher_type')},
                          'dev_origins': {name: {str(k): int(v) for k, v in sorted(Counter(origins[name]).items())}
                                          for name in origins},
                          'observed_action_counts': {f'{pitcher}:{pitch_type}': count for (pitcher,pitch_type),count in sorted(action_support.items())},
                          'pa_action_counts': dict(sorted(Counter(row['action_count'] for row in policies).items()))},
              'sha256': {'config': digest(CONFIG), 'script': digest(__file__), 'contract': digest(ROOT/'docs/contracts/common-comparison-v1.md'),
                         'selection_manifest': digest(manifest_path), 'train_parquet': digest(paths['train']),
                         'dev_parquet': digest(paths['dev']), 'a_predictions': digest(a_prediction),
                         'bundle_manifest': digest(bundle/'bundle_manifest.json'),
                         'bundle_metadata': digest(bundle/'metadata.json'), 'frozen_source_hashes': source_hash,
                         'small_models': digest(dest/'small_models.pkl'), 'predictions': digest(dest/'predictions.npz')},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'elapsed_seconds': time.perf_counter()-started,
              'maxrss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (dest/'policy_pa.json').write_text(json.dumps(policies, indent=2, allow_nan=False)+'\n')
    result['sha256']['policy_pa'] = digest(dest/'policy_pa.json')
    REPORT.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n'
    (dest/'results.json').write_text(payload)
    (REPORT/'results.json').write_text(payload)
    print(json.dumps({'results': str(REPORT/'results.json'), 'seconds': result['elapsed_seconds']}))


if __name__ == '__main__':
    run()
