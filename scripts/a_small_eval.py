"""Fixed, exploratory April-to-July replay for the frozen observer WE model.

Prepare freezes complete game IDs and pitch rows without calculating outcomes or
scores. Evaluate is a separate, explicit command after the manifest is reviewed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'experiments/pitchmdp'
sys.path[:0] = [str(PROJECT / 'scripts'), str(PROJECT)]

import numpy as np
import pandas as pd

from pitchmdp.data import KEY
from pitchmdp.model import CountBaseline, OUTCOMES, eligible, outcome_labels
from diagnose_sequence_legality import condition_on_legality
from pitchmdp.game import terminal_values
from pitchmdp.planner import solve_pa

CONFIG_PATH = ROOT / 'configs/EXP-A-S0S1-001.json'
REPORT_DIR = ROOT / 'results/EXP-A-S0S1-001'
PA_KEY = ['game_pk', 'at_bat_number']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def config():
    value = json.loads(CONFIG_PATH.read_text())
    if value['pitchers_train_rank_order'] != [657277, 554430]:
        raise ValueError('Frozen TRAIN rank changed')
    if (value['train_end'], value['dev_start'], value['dev_end']) != ('2025-04-30', '2025-07-01', '2025-07-31'):
        raise ValueError('Chronological cutoffs changed')
    return value


def chronological_games(frame, pitcher, start, end, limit):
    """Select starts from pre-outcome IDs, before checking any support field."""
    dates = pd.to_datetime(frame.game_date)
    mask = dates.between(start, end) & frame.pitcher.eq(pitcher) & frame.starter_pitcher.eq(pitcher)
    games = frame.loc[mask, ['game_date', 'game_pk']].drop_duplicates()
    games = games.sort_values(['game_date', 'game_pk'], kind='stable').head(limit)
    if len(games) != limit:
        raise ValueError(f'Insufficient chronological games for {pitcher} in {start}..{end}')
    return [{'game_date': str(pd.Timestamp(row.game_date).date()), 'game_pk': int(row.game_pk)}
            for row in games.itertuples(index=False)]


def select_full_games(frame, cfg):
    dates = pd.to_datetime(frame.game_date)
    if dates.isna().any() or not dates.dt.year.isin((2023, 2024, 2025)).all():
        raise ValueError('Unapproved year in source')
    if frame.duplicated(KEY).any():
        raise ValueError('Duplicate source pitch IDs')
    selected = {'train': {}, 'dev': {}}
    for pitcher in cfg['pitchers_train_rank_order']:
        selected['train'][str(pitcher)] = chronological_games(frame, pitcher, cfg['train_start'], cfg['train_end'], cfg['train_games_per_pitcher'])
        selected['dev'][str(pitcher)] = chronological_games(frame, pitcher, cfg['dev_start'], cfg['dev_end'], cfg['dev_games_per_pitcher'])
    train_ids = {g['game_pk'] for group in selected['train'].values() for g in group}
    dev_ids = {g['game_pk'] for group in selected['dev'].values() for g in group}
    if train_ids & dev_ids or len(train_ids) != 6 or len(dev_ids) != 6:
        raise ValueError('Selected games overlap or are duplicated across pitchers')
    return selected, frame.loc[frame.game_pk.isin(train_ids)].copy(), frame.loc[frame.game_pk.isin(dev_ids)].copy()


def complete_pa_reason(rows):
    rows = rows.sort_values('pitch_number')
    start, end = rows.iloc[0], rows.iloc[-1]
    if int(start.pitch_number) != 1 or int(start.balls) != 0 or int(start.strikes) != 0:
        return 'incomplete_start'
    if not bool(end.is_pa_terminal):
        return 'missing_terminal'
    if rows.pitcher.nunique() != 1:
        return 'pitcher_change'
    for name in ('inning', 'inning_topbot', 'outs_when_up', 'bases', 'home_score', 'away_score', 'batter', 'stand'):
        if rows[name].nunique(dropna=False) != 1:
            return 'state_changed_inside_pa'
    if not rows.supported_pa.fillna(False).all():
        return 'unsupported_pa'
    if not eligible(rows).all():
        return 'unsupported_pitch_or_outcome'
    return None


def cohort_pa_rows(full, pitcher_ids, repertoire=None):
    """Return complete supported pitcher PAs, leaving every source row untouched."""
    kept, reasons = [], Counter()
    cohort_pas = 0
    for (_, _), rows in full.groupby(PA_KEY, sort=False):
        if not rows.pitcher.isin(pitcher_ids).any():
            continue
        cohort_pas += 1
        reason = complete_pa_reason(rows)
        if reason is None and int(rows.iloc[0].pitcher) not in pitcher_ids:
            reason = 'cohort_pitcher_not_pa_start'
        if reason is None and repertoire is not None:
            pid = str(int(rows.iloc[0].pitcher))
            allowed = set(repertoire[pid]['pitch_types'])
            if not rows.pitch_type.isin(allowed).all():
                reason = 'outside_frozen_repertoire'
        if reason:
            reasons[reason] += 1
        else:
            kept.append(rows)
    return (pd.concat(kept, ignore_index=True) if kept else full.iloc[:0].copy()), dict(sorted(reasons.items())), cohort_pas


def git_commit():
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def frozen_source_check(bundle):
    captured = json.loads((bundle / 'source_hashes.json').read_text())
    for relative, expected in captured.items():
        path = PROJECT / relative
        if path.suffix == '.py' and digest(path) != expected:
            raise ValueError(f'Frozen source differs: {relative}')
    return digest(bundle / 'source_hashes.json')


def legal_baseline(p, rows):
    impossible = rows.outs_when_up.eq(2).to_numpy() | rows.bases.eq(0).to_numpy()
    return condition_on_legality(p / p.sum(axis=1, keepdims=True), impossible)


def prior_zone_inputs(train, pitcher, bundle_types):
    """April-only zone and repertoire inputs for exploratory Observer adapter."""
    before = train.loc[train.pitcher.eq(pitcher)]
    counts = before.pitch_type.value_counts()
    repertoire = {name: int(counts.get(name, 0)) for name in bundle_types}
    valid = train[['sz_bot', 'sz_top']].dropna()
    if valid.empty:
        raise ValueError('No prior zone bounds')
    bounds = {'bottom': float(valid.sz_bot.median()), 'top': float(valid.sz_top.median())}
    if not bounds['top'] > bounds['bottom']:
        raise ValueError('Invalid prior zone bounds')
    return bounds, repertoire


def manifest_ids(frame):
    return {'games': int(frame.game_pk.nunique()), 'pas': int(frame.groupby(PA_KEY).ngroups),
            'pitches': int(len(frame)), 'pitch_key_sha256': hashlib.sha256(
            frame.sort_values(KEY)[KEY].to_csv(index=False).encode()).hexdigest()}


def load_verified_processed(source_cfg):
    """Use the pinned local processed file; retain prior raw hash attestations."""
    root = Path(source_cfg['artifact_root'])
    quality_path = root / 'reports/data_quality.json'
    quality = json.loads(quality_path.read_text())
    approved = source_cfg['raw_allowlist']
    if [item['file'] for item in quality['sources']] != approved:
        raise ValueError('Raw source manifest changed')
    processed = root / 'processed/pitches.parquet'
    if digest(processed) != quality['processed_sha256']:
        raise ValueError('Processed hash differs from pinned source manifest')
    frame = pd.read_parquet(processed)
    if len(frame) != quality['rows']:
        raise ValueError('Processed row count differs from pinned manifest')
    frame.attrs['sequence_data_identity'] = {'processed_sha256': quality['processed_sha256'],
                                              'raw_sources': quality['sources'],
                                              'data_quality_sha256': digest(quality_path),
                                              'raw_rehashed_this_run': False}
    return frame


def prepare():
    started = time.perf_counter()
    cfg = config()
    dest = Path(cfg['artifact_dir'])
    if dest.exists() or (REPORT_DIR / 'selection_manifest.json').exists():
        raise ValueError('Selection already exists; this experiment is immutable')
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text())
    if source_cfg['raw_allowlist'] != ['statcast_2023.parquet', 'statcast_2024.parquet', 'statcast_2025.parquet']:
        raise ValueError('Only approved 2023-25 inputs allowed')
    frame = load_verified_processed(source_cfg)
    selection, train, dev = select_full_games(frame, cfg)
    if any(pd.to_datetime(train.game_date).gt(cfg['train_end'])) or any(pd.to_datetime(dev.game_date).lt(cfg['dev_start'])):
        raise ValueError('Date boundary violated')
    dest.mkdir(parents=True)
    train_path, dev_path = dest / 'train_full_games.parquet', dest / 'dev_full_games.parquet'
    train.to_parquet(train_path, index=False)
    dev.to_parquet(dev_path, index=False)
    bundle = Path(cfg['bundle_dir'])
    bundle_metadata = json.loads((bundle / 'metadata.json').read_text())
    if bundle_metadata['training_cutoff'] != cfg['train_end'] or bundle_metadata['profile_cutoff'] != cfg['train_end']:
        raise ValueError('Unexpected frozen model/profile cutoff')
    manifest = {'experiment_id': cfg['experiment_id'], 'phase': 'prepared_unscored',
                'selection': selection, 'selected_full_rows': {'train': manifest_ids(train), 'dev': manifest_ids(dev)},
                'source_identity': frame.attrs['sequence_data_identity'],
                'sha256': {'config': digest(CONFIG_PATH), 'script': digest(__file__),
                           'source_config': digest(ROOT / cfg['source_config']),
                           'bundle_manifest': digest(bundle / 'bundle_manifest.json'),
                           'bundle_metadata': digest(bundle / 'metadata.json'),
                           'train_full_games.parquet': digest(train_path), 'dev_full_games.parquet': digest(dev_path)},
                'dataset_paths': {'train': str(train_path), 'dev': str(dev_path)},
                'elapsed_seconds': round(time.perf_counter() - started, 3),
                'maxrss_kib_or_bytes_platform': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'note': 'Selection used only game date, pitcher/starter IDs, and game_pk; no prediction or metric computed.'}
    write_json(dest / 'selection_manifest.json', manifest)
    write_json(REPORT_DIR / 'selection_manifest.json', manifest)
    print(json.dumps({'phase': manifest['phase'], 'selection': selection,
                      'full_rows': manifest['selected_full_rows'], 'artifact_dir': str(dest)}, indent=2))


def score_metrics(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if p.shape != (len(y), 10) or len(y) == 0 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid predictions')
    n = np.bincount(y, minlength=10)
    onehot = np.eye(10)[y]
    conf, pred = p.max(1), p.argmax(1)
    bins = np.minimum((10 * conf).astype(int), 9)
    ece = sum((bins == i).mean() * abs(float((pred[bins == i] == y[bins == i]).mean()) - float(conf[bins == i].mean()))
              for i in range(10) if (bins == i).any())
    return {'n': len(y), 'class_counts': dict(zip(OUTCOMES, map(int, n))),
            'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'brier_multiclass': float(np.square(p - onehot).sum(1).mean()),
            'top_label_ece10': float(ece),
            'rare_class_metrics': {name: None if n[i] == 0 else {'observed_count': int(n[i]), 'predicted_mean': float(p[:, i].mean())}
                                   for i, name in enumerate(OUTCOMES)}}


def paired_bootstrap(y, left, right, games, seed, reps):
    delta = -np.log(np.clip(left[np.arange(len(y)), y], 1e-12, 1)) + np.log(np.clip(right[np.arange(len(y)), y], 1e-12, 1))
    groups = pd.DataFrame({'game': games, 'delta': delta}).groupby('game').agg(total=('delta', 'sum'), n=('delta', 'size'))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(groups), (reps, len(groups)))
    values = groups.total.to_numpy()[draw].sum(1) / groups.n.to_numpy()[draw].sum(1)
    return {'mean_log_loss_delta': float(delta.mean()), 'game_bootstrap95': np.quantile(values, [.025, .975]).tolist(),
            'games': len(groups), 'seed': seed, 'replicates': reps}


def evaluate():
    started = time.perf_counter()
    cfg = config()
    dest = Path(cfg['artifact_dir'])
    manifest = json.loads((dest / 'selection_manifest.json').read_text())
    if manifest != json.loads((REPORT_DIR / 'selection_manifest.json').read_text()):
        raise ValueError('Prepared manifest copy differs')
    if digest(CONFIG_PATH) != manifest['sha256']['config']:
        raise ValueError('Evaluation config differs from reviewed amendment')
    for name, path in (("source_config", ROOT / cfg['source_config']),
                       ("bundle_manifest", Path(cfg['bundle_dir']) / 'bundle_manifest.json'),
                       ("bundle_metadata", Path(cfg['bundle_dir']) / 'metadata.json')):
        if digest(path) != manifest['sha256'][name]:
            raise ValueError(f'Frozen hash changed: {name}')
    for split in ('train', 'dev'):
        if digest(manifest['dataset_paths'][split]) != manifest['sha256'][split + '_full_games.parquet']:
            raise ValueError(f'Selected {split} data changed')
    if (dest / 'results.json').exists():
        raise ValueError('Evaluation already exists; no tuning/overwrite')
    from minimal_pitch_service import Engine
    source_hash = frozen_source_check(Path(cfg['bundle_dir']))
    engine = Engine(cfg['bundle_dir'])
    if [member.seed for member in engine.models] != cfg['model_seeds']:
        raise ValueError('Frozen model seeds changed')
    if cfg['baseline_global_additive_count'] != 1 or cfg['baseline_cell_global_pseudocount'] != 50:
        raise ValueError('Baseline hyperparameters differ from CountBaseline')
    train = pd.read_parquet(manifest['dataset_paths']['train'])
    dev = pd.read_parquet(manifest['dataset_paths']['dev'])
    pitchers = cfg['pitchers_train_rank_order']
    train_rows, train_reasons, train_pas = cohort_pa_rows(train, pitchers)
    dev_rows, dev_reasons, dev_pas = cohort_pa_rows(dev, pitchers, engine.metadata['pitchers'])
    if (train_rows.groupby(PA_KEY).ngroups < cfg['minimum_supported_pa'] or
            dev_rows.groupby(PA_KEY).ngroups < cfg['minimum_supported_pa']):
        raise ValueError('No supported complete PAs')
    baseline = CountBaseline().fit(train_rows)
    with (dest / 'april_count_baseline.pkl').open('wb') as stream:
        pickle.dump(baseline, stream)
    with (dest / 'april_count_baseline.pkl').open('rb') as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(baseline.predict(dev_rows), restored.predict(dev_rows))
    write_json(dest / 'selected_pitch_keys.json', {
        'train': train.sort_values(KEY)[KEY].astype(int).values.tolist(),
        'dev': dev.sort_values(KEY)[KEY].astype(int).values.tolist()})
    # One pre-pitch tensor per complete PA; use only the pitch type/count actually observed.
    predictions = {'april_count': [], 'frozen_frequency': [], 'frozen_blend': []}
    labels, games, keys = [], [], []
    pa_diagnostics = []
    s0_ids = {group[0]['game_pk'] for group in manifest['selection']['dev'].values()}
    zone_candidates = {}
    restored_engine_checked = False
    for (game, pa), rows in dev_rows.groupby(PA_KEY, sort=False):
        rows = rows.sort_values('pitch_number')
        start = rows.iloc[0]
        request = {'inning': int(start.inning), 'topbot': str(start.inning_topbot), 'outs': int(start.outs_when_up),
                   'bases': int(start.bases), 'home_score': int(start.home_score), 'away_score': int(start.away_score),
                   'balls': 0, 'strikes': 0, 'pitcher_id': int(start.pitcher), 'batter_stand': str(start.stand),
                   'batter_id': int(start.batter), 'date': str(pd.Timestamp(start.game_date).date()), 'top_k': 3}
        inference = engine.predict_counts(request)
        if int(game) in s0_ids and not restored_engine_checked:
            independent = Engine(cfg['bundle_dir'])
            replay = independent.predict_counts(request)
            for kind in ('neural', 'frequency', 'blend'):
                np.testing.assert_array_equal(inference['probabilities'][kind], replay['probabilities'][kind])
            del independent
            restored_engine_checked = True
        if inference['profile_as_of'] >= request['date']:
            raise ValueError('Future or same-date profile')
        if int(game) in s0_ids and int(game) not in zone_candidates:
            zone_candidates[int(game)] = (int(pa), request.copy())
        choices = list(inference['pitch_types'])
        for row in rows.itertuples(index=False):
            action = choices.index(row.pitch_type)
            for name, tensor in [('frozen_frequency', inference['probabilities']['frequency']),
                                 ('frozen_blend', inference['probabilities']['blend'])]:
                predictions[name].append(tensor[int(row.balls), int(row.strikes), 0, action])
            one_row = pd.DataFrame([row._asdict()])
            predictions['april_count'].append(legal_baseline(restored.predict(one_row), one_row)[0])
            labels.append(int(outcome_labels(one_row)[0]))
            games.append(int(game))
            keys.append((int(game), int(pa), int(row.pitch_number)))
        # WE is model-internal only; reuse the same tensor without another model pass.
        counts = engine.metadata['repertoire_counts'].get(str(request['pitcher_id']), {})
        usage = np.asarray([counts.get(name, 0) for name in choices], float)
        usage = usage / usage.sum() if usage.sum() > 0 else np.full(len(choices), 1 / len(choices))
        terminal = terminal_values(inference['state'], engine.we, engine.advancement)
        plan = solve_pa(inference['probabilities']['blend'], terminal, [0] * len(choices), baseline_policy=usage)
        best = plan.topk(0, 0, 0, 1)[0]
        pa_diagnostics.append({'game_pk': int(game), 'at_bat_number': int(pa),
                               'baseline_defensive_we': float(plan.baseline_values[0, 0, 0]),
                               'best_defensive_we': best['value'],
                               'delta_vs_baseline_policy': best['delta_vs_baseline'],
                               'kind': 'frozen_model_internal_not_causal'})
    y = np.asarray(labels, int)
    p = {name: np.asarray(values, float) for name, values in predictions.items()}
    if keys != sorted(keys):
        order = sorted(range(len(keys)), key=keys.__getitem__)
        y, games = y[order], np.asarray(games)[order]
        p = {name: value[order] for name, value in p.items()}
        keys = [keys[i] for i in order]
    tiers = {}
    for tier, mask in [('S0', np.isin(games, list(s0_ids))), ('S1', np.ones(len(y), bool))]:
        yy = y[mask]
        pp = {name: value[mask] for name, value in p.items()}
        tiers[tier] = {'games': int(len(np.unique(np.asarray(games)[mask]))), 'pitches': int(len(yy)),
                       'models': {name: score_metrics(yy, value) for name, value in pp.items()},
                       'paired_blend_minus_april_count': paired_bootstrap(yy, pp['frozen_blend'], pp['april_count'],
                              np.asarray(games)[mask], cfg['bootstrap_seed'], cfg['bootstrap_replicates']),
                       'paired_blend_minus_frozen_frequency': paired_bootstrap(yy, pp['frozen_blend'], pp['frozen_frequency'],
                              np.asarray(games)[mask], cfg['bootstrap_seed'], cfg['bootstrap_replicates'])}
    np.savez_compressed(dest / 'predictions.npz', y=y, game_pk=np.asarray(games), pitch_keys=np.asarray(keys), **p)
    os.environ['PITCHEEZY_OBSERVER_RUN'] = str(dest)
    os.environ['PITCHEEZY_OBSERVER_RUNTIME'] = 'research'
    sys.path.insert(0, str(ROOT / 'apps/observer/backend'))
    from observer_app.recommender import Recommender
    zone = Recommender()
    zone_report = []
    for game in sorted(s0_ids):
        if game not in zone_candidates:
            zone_report.append({'game_pk': game, 'status': 'no_complete_supported_pa'})
            continue
        pa, request = zone_candidates[game]
        pitcher = request['pitcher_id']
        bounds, repertoire = prior_zone_inputs(train, pitcher, engine.metadata['pitchers'][str(pitcher)]['pitch_types'])
        snapshot = engine.metadata['profiles'].get(str(request['batter_id']), engine.metadata['default_profile'])
        zone_request = request | {'batter_profile': {'rates': snapshot['rates'],
                           'reliabilities': snapshot['reliabilities'], 'as_of': cfg['train_end']}}
        result = zone.recommend({'request': zone_request}, {'zone_bounds': bounds, 'repertoire_counts': repertoire})
        zone_report.append({'game_pk': game, 'at_bat_number': pa, 'status': result['status'],
                            'candidate_count': len(result['candidates']), 'reason': result.get('reason'),
                            'recommendation': result,
                            'prior_zone_bounds': bounds, 'prior_repertoire_counts': repertoire,
                            'model_version': result['model_version'],
                            'scope': 'observer_zone_v1_model_internal_not_causal'})
    write_json(dest / 'observer_zone_diagnostics.json', {'eligible_s0_games': len(s0_ids),
                 'attempted_s0_games': len(zone_candidates), 'cases': zone_report,
                 'adapter_identity': zone.identity})
    write_json(dest / 'internal_we_diagnostics.json', {'scope': 'model_internal_only', 'pa': pa_diagnostics})
    events = {name: int(dev.events.fillna('').eq(name).sum()) for name in ('home_run', 'strikeout', 'strikeout_double_play')}
    event_rows = dev.loc[dev.events.fillna('').isin(events),
                         [*KEY, 'game_date', 'pitcher', 'batter', 'events']].copy()
    event_rows['game_date'] = pd.to_datetime(event_rows.game_date).dt.strftime('%Y-%m-%d')
    write_json(dest / 'event_cases.json', {
        'selection': 'All HR/K terminal rows in all six preselected DEV games; no model-score selection',
        'scope': 'Full games including non-cohort pitchers; not all cases supported by frozen model',
        'cases': event_rows.to_dict('records'),
        'intended_location': None, 'available_bullpen_roster': None,
        'missing_reason': 'No accepted CV labels or contemporaneous available-roster evidence in this input',
    })
    substitution_pas = sorted([list(map(int, pair)) for pair, rows in dev.groupby(PA_KEY)
                                if rows.pitcher.nunique() > 1])
    results = {'experiment_id': cfg['experiment_id'], 'phase': 'evaluated_exploratory_dev', 'tiers': tiers,
               'support': {'train_cohort_pas': train_pas, 'train_supported_complete_pas': int(train_rows.groupby(PA_KEY).ngroups),
                           'train_excluded_pa_reasons': train_reasons, 'dev_cohort_pas': dev_pas,
                           'dev_supported_complete_pas': int(dev_rows.groupby(PA_KEY).ngroups),
                           'dev_excluded_pa_reasons': dev_reasons,
                           'dev_full_games': manifest['selected_full_rows']['dev'],
                           'dev_event_counts_in_full_games': events,
                           'dev_pitcher_change_pa_keys': substitution_pas,
                           'observer_zone_s0': {'attempted_games': len(zone_candidates),
                               'supported_recommendations': sum(case['status'] == 'ready' for case in zone_report)}},
               'policy_causal_we': None, 'policy_causal_we_reason': 'Intent/target intervention and behavior propensities are not identified; internal WE is not causal policy value.',
               'source_manifest_sha256': digest(dest / 'selection_manifest.json'),
               'evaluation_code_sha256': digest(__file__), 'evaluation_config_sha256': digest(CONFIG_PATH),
               'git_commit': git_commit(), 'frozen_source_hashes_sha256': source_hash,
               'observer_adapter_sha256': digest(ROOT / 'apps/observer/backend/observer_app/recommender.py'),
               'output_sha256': {'baseline': digest(dest / 'april_count_baseline.pkl'),
                                 'predictions': digest(dest / 'predictions.npz'),
                                 'internal_we': digest(dest / 'internal_we_diagnostics.json'),
                                 'observer_zone': digest(dest / 'observer_zone_diagnostics.json'),
                                 'event_cases': digest(dest / 'event_cases.json'),
                                 'selected_pitch_keys': digest(dest / 'selected_pitch_keys.json')},
               'elapsed_seconds': round(time.perf_counter() - started, 3),
               'maxrss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(dest / 'results.json', results)
    write_json(REPORT_DIR / 'results.json', results)
    print(json.dumps({'phase': results['phase'], 'tiers': tiers, 'support': results['support']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'evaluate'))
    args = parser.parse_args()
    {'prepare': prepare, 'evaluate': evaluate}[args.phase]()


if __name__ == '__main__':
    main()
