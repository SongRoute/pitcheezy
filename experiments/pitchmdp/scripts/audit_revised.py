"""Replay the archived MLB/archetype pilot and bootstrap paired losses by game.

No model training, network access, raw-file reads, or 2026 data. The count/hand
reference is reconstructed from the fixed league training rows. Output stays on
the mounted SSD; the original LAD audit script and artifacts are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--bootstrap-replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=7419)
    args = parser.parse_args()
    run = args.run.resolve()
    volume = Path('/Volumes/T7 Shield')
    if not volume.is_mount() or not run.is_relative_to(volume.resolve()):
        raise SystemExit('Run and audit output must be on mounted T7 Shield')
    if args.bootstrap_replicates < 2:
        raise SystemExit('At least two bootstrap replicates required')
    read = lambda name: json.loads((run / name).read_text())
    config, quality = read('config.json'), read('data_quality.json')
    recorded, cohort = read('prediction_metrics.json'), read('cohort_manifest.json')
    start_hashes, runtime = read('code_hashes_start.json'), read('runtime.json')
    assert config['cohort_scope'] == 'MLB' and config['model_variant'] == 'archetype'
    end_hashes = runtime['code_hashes_end']
    changed_sources = [rel for rel in sorted(set(start_hashes) | set(end_hashes))
                       if start_hashes.get(rel) != end_hashes.get(rel)]
    source_drift = {'changed_sources': changed_sources, 'start_end_equal': not changed_sources}
    if changed_sources:
        assert changed_sources == ['pitchmdp/recommend.py'], 'Unexpected source drift'
        old = (run / 'source/pitchmdp/recommend.py').read_text()
        original = "getattr(model.encoder, 'variant', '')"
        replacement = "getattr(getattr(model, 'encoder', None), 'variant', '')"
        assert old.count(original) == 1
        safe_guard = old.replace(original, replacement)
        assert hashlib.sha256(safe_guard.encode()).hexdigest() == end_hashes['pitchmdp/recommend.py']
        source_drift.update(
            exact_end_hash_reconstructed=True, old_expression=original, new_expression=replacement,
            interpretation='Only a safe-access guard for encoder-less test doubles changed during the run; real PitchModel behavior is identical. Replay uses the unchanged startup source archive.',
            start_hash=start_hashes['pitchmdp/recommend.py'], end_hash=end_hashes['pitchmdp/recommend.py'])
    assert all(sha256(run / 'source' / rel) == digest for rel, digest in start_hashes.items())
    processed = Path(quality['output'])
    assert sha256(processed) == quality['processed_sha256'] == cohort['processed_sha256']
    # Import this run's archived code, not the working tree's latest revision.
    sys.path.insert(0, str(run / 'source'))
    from pitchmdp.model import CATEGORICAL, NUMERIC_INPUTS, CountBaseline, PitchModel, eligible, metrics, outcome_labels
    from pitchmdp.archetypes import HISTORY_COLUMNS, RELIABILITY_COLUMNS, STYLE_COLUMNS, add_batter_style_history
    from pitchmdp.recommend import probability_tensor, recommend
    from pitchmdp.game import _we_features

    columns = sorted(set(CATEGORICAL) | set(NUMERIC_INPUTS) | {
        'split', 'description', 'events', 'supported_pa', 'game_pk', 'game_date',
        'at_bat_number', 'pitch_number', 'starter_pitcher', 'is_pa_terminal', 'launch_angle',
    })
    frame = pd.read_parquet(processed, columns=columns)
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all()
    add_batter_style_history(frame)  # Complete pitch pool, before outcome/support selection.
    train = frame[(frame.split == 'train') & eligible(frame)]
    cohort_mask = frame.pitcher.isin(cohort['pitcher_ids']) & frame.pitcher.eq(frame.starter_pitcher)
    dev = frame[(frame.split == 'dev') & cohort_mask]
    dev = dev[eligible(dev)]
    dev = dev.sample(min(len(dev), config['dev_max_rows']), random_state=config['seed']).sort_index()
    assert set(train.game_pk).isdisjoint(dev.game_pk)
    assert train.game_date.max() < dev.game_date.min()
    assert len(dev) == read('coverage.json')['dev_used']
    model = PitchModel.load(run / 'pitch_model.pt')
    with (run / 'recommendation_context.pkl').open('rb') as stream:
        context = pickle.load(stream)
    assert model.encoder.variant == 'archetype'
    assert 'batter' not in model.encoder.categories and 'batter' not in model.encoder.maps
    archetypes = model.encoder.archetypes
    assert pd.Timestamp(archetypes.fit_date_max) <= pd.Timestamp(config['train_end'])
    assert archetypes.report()['membership'].startswith('similarity weights, not calibrated probabilities')

    # Inference is independent of IDs even when the column is entirely absent.
    probe = dev.iloc[:100].copy()
    cats, nums = model.encoder.transform(probe)
    dropped_cats, dropped_nums = model.encoder.transform(probe.drop(columns='batter'))
    np.testing.assert_array_equal(cats, dropped_cats)
    np.testing.assert_array_equal(nums, dropped_nums)
    changed = probe.copy()
    changed['batter'] = -999999
    for col in ('batter_pa_prior', 'batter_obp_prior', 'batter_k_prior'):
        changed[col] = 999999
    np.testing.assert_array_equal(model.encoder.transform(changed)[1], nums)
    # Cold profiles retain league shrinkage and have exactly uniform memberships.
    cold = probe.copy()
    cold[list(RELIABILITY_COLUMNS)] = 0.
    np.testing.assert_allclose(archetypes.transform(cold), 1 / archetypes.n_clusters)
    centers_before = archetypes.centers.copy()
    archetypes.transform(dev)
    np.testing.assert_array_equal(archetypes.centers, centers_before)

    # Independent chronological fixture: future/current outcomes cannot enter
    # the earlier date, including a new player's fallback and same-day rows.
    fixture = pd.DataFrame({
        'batter': [1, 1, 1, 2, 1],
        'game_date': pd.to_datetime(['2024-04-01', '2024-04-02', '2024-04-02', '2024-04-02', '2024-04-03']),
        'events': ['walk', 'home_run', 'strikeout', 'single', 'double'],
        'description': ['ball', 'hit_into_play', 'swinging_strike', 'hit_into_play', 'hit_into_play'],
        'is_pa_terminal': [True] * 5, 'launch_angle': [np.nan, 35., np.nan, 0., 20.],
    })
    expected = add_batter_style_history(fixture.copy())
    altered = fixture.copy()
    altered.loc[altered.game_date.ge('2024-04-02'), ['events', 'description', 'launch_angle']] = ['home_run', 'hit_into_play', 45.]
    altered = add_batter_style_history(altered)
    np.testing.assert_array_equal(expected.loc[:3, list(HISTORY_COLUMNS)], altered.loc[:3, list(HISTORY_COLUMNS)])
    np.testing.assert_array_equal(expected.loc[1, list(HISTORY_COLUMNS)], expected.loc[2, list(HISTORY_COLUMNS)])
    assert expected.loc[3, list(RELIABILITY_COLUMNS)].eq(0).all()

    baseline = CountBaseline().fit(train)
    pred = context['delivery'].predict(model, dev)
    reference = baseline.predict(dev)
    labels = outcome_labels(dev)
    got, base = metrics(labels, pred), metrics(labels, reference)
    for actual, saved in [(got, recorded['primary_prepitch_type_conditional']),
                           (base, recorded['count_hand_baseline'])]:
        assert actual['n'] == saved['n']
        for key in ['log_loss', 'brier_multiclass', 'accuracy', 'top_label_ece_10']:
            assert np.isclose(actual[key], saved[key], atol=1e-7, rtol=0), (key, actual[key], saved[key])
    changed = probe.copy()
    changed['plate_x'], changed['plate_z'] = 999., -999.
    np.testing.assert_allclose(context['delivery'].predict(model, probe),
                               context['delivery'].predict(model, changed), atol=1e-7, rtol=0)
    row = context['rows'].iloc[0].copy()
    actions = context['actions'][(int(row.pitcher), str(row.stand))]
    tensor = probability_tensor(model, row, actions, config['control_sigma_ft'])[0]
    changed_row = row.copy()
    changed_row['plate_x'], changed_row['plate_z'] = 999., -999.
    changed_row['pitch_type'], changed_row['batter'] = 'IMPOSSIBLE', -999999
    changed_tensor = probability_tensor(model, changed_row, actions, config['control_sigma_ft'])[0]
    np.testing.assert_array_equal(tensor, changed_tensor)
    recommendation = recommend(model, row, actions, context['we'], context['advancement'], config['control_sigma_ft'])
    recorded_recommendation = read('representative_recommendations.json')[0]
    for key in ['input', 'n_actions', 'top_k', 'myopic_action', 'terminal_values',
                'planned_defense_we', 'reference_defense_we', 'model_internal_advantage_pp']:
        assert recommendation[key] == recorded_recommendation[key], ('Recommendation replay', key)
    # Confirm meaningful game context reaches both pitch and continuation models.
    context_checks = {}
    for name, replacement in [('bases', int(probe.bases.iloc[0]) ^ 7),
                               ('outs_when_up', (int(probe.outs_when_up.iloc[0]) + 1) % 3),
                               ('home_score', int(probe.home_score.iloc[0]) + 1),
                               ('inning_topbot', 'Bot' if probe.inning_topbot.iloc[0] == 'Top' else 'Top')]:
        original = probe.iloc[[0]].copy()
        perturbation = original.copy()
        perturbation[name] = replacement
        assert not np.array_equal(model.encoder.transform(original)[1], model.encoder.transform(perturbation)[1])
        assert not np.array_equal(_we_features(original), _we_features(perturbation))
        context_checks[name] = {
            'pitch_encoder_changes': True, 'we_features_change': True,
            'home_we_before': float(context['we'].predict_home(original)[0]),
            'home_we_after': float(context['we'].predict_home(perturbation)[0]),
        }

    onehot = np.eye(pred.shape[1])[labels]
    paired = pd.DataFrame({
        'game_pk': dev.game_pk.to_numpy(), 'count': 1,
        'log_loss_delta': -np.log(np.maximum(pred[np.arange(len(labels)), labels], 1e-12))
                          + np.log(np.maximum(reference[np.arange(len(labels)), labels], 1e-12)),
        'brier_delta': ((pred - onehot) ** 2).sum(axis=1) - ((reference - onehot) ** 2).sum(axis=1),
    })
    groups = paired.groupby('game_pk', sort=True).sum()
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(groups), size=(args.bootstrap_replicates, len(groups)))
    denominator = groups['count'].to_numpy()[indices].sum(axis=1)
    differences = {}
    for column in ['log_loss_delta', 'brier_delta']:
        estimates = groups[column].to_numpy()[indices].sum(axis=1) / denominator
        differences[column] = {
            'point_estimate_full_minus_baseline': float(paired[column].mean()),
            'percentile_95_ci': np.quantile(estimates, [.025, .975]).tolist(),
            'bootstrap_standard_error': float(estimates.std(ddof=1)),
        }
    absent = [r for r in cohort['selected'] if int(r['pitcher']) not in set(dev.pitcher)]
    per_pitcher = {}
    for pitcher in sorted(dev.pitcher.unique()):
        mask = dev.pitcher.to_numpy() == pitcher
        per_pitcher[str(int(pitcher))] = {'archetype': metrics(labels[mask], pred[mask]),
                                       'baseline': metrics(labels[mask], reference[mask])}
    result = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'n_pitches': len(dev), 'n_games': len(groups), 'n_pitchers': int(dev.pitcher.nunique()),
        'n_batters': int(dev.batter.nunique()),
        'dates': [str(dev.game_date.min().date()), str(dev.game_date.max().date())],
        'bootstrap_replicates': args.bootstrap_replicates, 'seed': args.seed,
        'method': 'paired game-cluster percentile bootstrap; resample games, aggregate pitch loss sums / resampled pitch counts',
        'direction': 'Archetype minus league-train count/hand baseline; negative favors archetype',
        'uncertainty_scope': 'game sampling only; fixed trained model and cohort; no training, selection, or causal-policy uncertainty',
        'primary_prepitch': got, 'count_hand_baseline': base, 'paired_differences': differences,
        'per_pitcher': per_pitcher,
        'archived_source_hashes_match': True, 'start_end_source_hashes_equal': not changed_sources,
        'source_drift': source_drift, 'archived_representative_recommendation_reproduced': True,
        'processed_input_hash_matches': True, 'saved_metrics_reproduced': True,
        'batter_id_removed_inference_verified': True, 'cold_player_uniform_membership_verified': True,
        'current_and_future_date_history_exclusion_verified': True,
        'target_and_delivery_actual_location_invariance_verified': True,
        'context_feature_checks': context_checks, 'centroid_train_end': archetypes.fit_date_max,
        'frozen_cohort_without_eligible_dev': absent,
        'audit_script_sha256': sha256(Path(__file__)),
        'metadata_caveat': 'Archived training_report.numeric_inputs is a legacy raw-column list, not the actual archetype feature contract; encoder and archetypes report determine actual inputs.',
    }
    (run / 'prediction_uncertainty.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    (run / 'audit_source_drift.json').write_text(json.dumps(source_drift, indent=2) + '\n')
    lines = [
        '# Revised MLB / batter-archetype independent audit', '',
        'The saved run passes archived-code prediction replay and the input-boundary checks below. '
        'This is a retrospective six-starter development pilot; it does not establish a policy effect or a stable taxonomy of batters.', '',
        '| Contract | Audit evidence |', '|---|---|',
        '| No batter-ID response feature | Saved encoder excludes batter from categories/maps. Removing the ID column leaves encoded inputs identical; altering ID and legacy batter prior rates changes no numeric inputs. IDs remain history joins and output labels. |',
        '| Prior-date profiles | Histories reconstructed on the complete processed pool before filters. Independent fixture verifies that current/future outcomes cannot alter earlier/same-day profiles; new batters have zero personal evidence. |',
        f'| Frozen TRAIN-only geometry | Archived runner fits centers on the eligible TRAIN subset; fit enforces train labels; saved maximum date is {archetypes.fit_date_max}. DEV transformation leaves centers unchanged. |',
        '| Cold-player and membership semantics | Zero evidence produces uniform five-way membership. These are similarity weights attenuated by history reliability, not calibrated posterior class probabilities. Six continuous style rates and six reliability measures also enter the encoder. |',
        '| Game context | Bases, outs, score difference and half-inning each change response features and WE features in direct perturbation checks. Top means the home team defends; terminal WE retains the initial defending team after half-inning changes. |',
        '| No realized-location oracle | Changing logged coordinates leaves saved delivery-integrated predictions unchanged. Changing coordinates, logged pitch type and batter ID leaves the candidate target tensor identical. Target Gaussian integration is still an assumed execution model. |',
        '| Provenance | Archived startup source hashes, processed data hash and cohort data hash verified. Start/end source hashes differ: one safe getattr guard in recommend.py changed for encoder-less test doubles. Reconstructing that exact edit reproduces the end hash; real PitchModel behavior is unchanged. Startup-archive prediction metrics and a saved representative recommendation reproduce exactly. See audit_source_drift.json. All loaded data dates are 2023–2025; no models retrained. |',
        '| Cohort interpretation | All-team starts are selected by persistent pitcher ID; six pitchers ranked by TRAIN workload among those with at least 50 DEV innings. This conditions on future survival/workload and is not a prospective or MLB-representative cohort. |',
        '| Exclusions | Whole-PA support filtering uses observed trajectories, and excludes unsupported events and unlinked game-ending PAs. The resulting selected-population losses do not cover all pitches or all batting situations. |',
        '| Metadata caveat | training.json numeric_inputs retains a legacy column list. Actual archetype features are evidenced by archived Encoder.transform, categories and archetypes report. This labeling issue does not affect the saved predictions. |', '',
        '## Prediction replay and paired uncertainty', '',
        f'{len(dev):,} pitches, {len(groups)} games, {dev.pitcher.nunique()} pitchers and {dev.batter.nunique()} batters. '
        f'{args.bootstrap_replicates:,} paired game-cluster bootstrap replicates; negative differences favor the archetype model.', '',
        '| Metric | Archetype | Count/hand | Difference | 95% interval |', '|---|---:|---:|---:|---:|',
    ]
    for metric, column in [('log_loss', 'log_loss_delta'), ('brier_multiclass', 'brier_delta')]:
        item = differences[column]
        lo, hi = item['percentile_95_ci']
        lines.append(f"| {metric} | {got[metric]:.6f} | {base[metric]:.6f} | {item['point_estimate_full_minus_baseline']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ['',
        'Intervals quantify game sampling conditional on the fixed trained model, frozen selected cohort and observed support restrictions. '
        'They do not include training-seed, calibration, model-selection, cohort-selection or policy-effect uncertainty. '
        'A comparison against the original LAD/ID pilot would confound cohort and representation; a same-cohort ID comparison is required for that separate question.', '',
        '## Per-pitcher paired point estimates', '',
        '| Pitcher | Pitches | Log-loss difference | Brier difference |', '|---|---:|---:|---:|',
    ]
    names = {str(r['pitcher']): r['player_name'] for r in cohort['selected']}
    for pitcher, item in per_pitcher.items():
        a, b = item['archetype'], item['baseline']
        lines.append(f"| {names[pitcher]} ({pitcher}) | {a['n']:,} | {a['log_loss']-b['log_loss']:+.6f} | {a['brier_multiclass']-b['brier_multiclass']:+.6f} |")
    lines += ['', 'All six selected pitchers have eligible DEV samples.' if not absent else f'Absent eligible DEV pitchers: {absent}', '',
        'Model-internal policy advantages compare strategies using the same response and WE models. '
        'They remain optimization diagnostics, with no independent causal or observational policy evaluation.', '',
        f'Reproduce: `{sys.executable} experiments/pitchmdp/scripts/audit_revised.py --run "{run}"`.', '']
    (run / 'REVISED_AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps({'run': str(run), 'n_pitches': len(dev), 'n_games': len(groups),
                      'differences': differences, 'all_contract_checks_passed': True}, indent=2))


if __name__ == '__main__':
    main()
