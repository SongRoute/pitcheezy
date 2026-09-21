"""Read-only model audit; writes an appendix, never trains or selects models.

Checks exact source snapshots, processed-data identity, deterministically rebuilt
split keys, outcome labels, saved probability mass, metrics and game bootstrap.
Only the existing processed parquet is read; no raw-season files are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd


KEY = ['game_pk', 'at_bat_number', 'pitch_number']


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def key_hash(frame):
    return hashlib.sha256(frame[KEY].to_numpy(np.int64).tobytes()).hexdigest()


def paired_bootstrap(y, model, reference, game_ids, replicates):
    indices = np.arange(len(y))
    delta = np.log(np.clip(reference[indices, y], 1e-12, 1)) - np.log(np.clip(model[indices, y], 1e-12, 1))
    groups = pd.DataFrame({'game': game_ids, 'delta': delta}).groupby('game', sort=True).delta.agg(['sum', 'count'])
    rng = np.random.default_rng(42)
    # Each draw retains the full sampled game and its original pitch multiplicity.
    results = np.empty(replicates)
    for i in range(replicates):
        sampled = groups.iloc[rng.integers(0, len(groups), len(groups))]
        results[i] = sampled['sum'].sum()/sampled['count'].sum()
    return {'model_minus_reference_log_loss': float(delta.mean()),
            'bootstrap95': np.quantile(results, [.025, .975]).tolist(), 'games': len(groups)}


def summarize(frame):
    return {'pitches': len(frame), 'games': int(frame.game_pk.nunique()),
            'pitcher_game_appearances': int(frame.groupby(['game_pk', 'pitcher']).ngroups),
            'starter_pitcher_games': int(frame.loc[frame.pitcher.eq(frame.starter_pitcher)].groupby(['game_pk', 'pitcher']).ngroups),
            'pas': int(frame.groupby(['game_pk', 'at_bat_number']).ngroups),
            'batters': int(frame.batter.nunique()), 'row_key_sha256': key_hash(frame),
            'date_min': str(frame.game_date.min()), 'date_max': str(frame.game_date.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--replay-binary', action='store_true', help='After runtime completion, replay saved binary checkpoints without fitting')
    parser.add_argument('--replay-all-count', action='store_true', help='Replay all saved ten-class models, reporting both observation regimes before/after calibration')
    args = parser.parse_args()
    run = args.run.resolve()
    root = run.parent.parent
    config = json.loads((run/'config.json').read_text())
    samples = json.loads((run/'samples.json').read_text())
    expected_sources = json.loads((run/'source_hashes.json').read_text())
    actual_sources = {name: digest(run/'source'/name) for name in expected_sources}
    assert actual_sources == expected_sources, 'Frozen source snapshot hash mismatch'
    runtime = json.loads((run/'runtime.json').read_text()) if (run/'runtime.json').exists() else None
    if runtime is not None:
        assert runtime['source_hashes_end'] == expected_sources, 'Source changed during execution'
    # Import the exact archived label function, not the current workspace version.
    sys.path.insert(0, str(run/'source'))
    from pitchmdp.model import eligible, outcome_labels
    quality = json.loads((root/'reports/data_quality.json').read_text())
    processed = root/'processed/pitches.parquet'
    assert digest(processed) == quality['processed_sha256'], 'Processed data identity changed'
    columns = KEY + ['game_date', 'split', 'pitcher', 'starter_pitcher', 'batter',
                     'description', 'events', 'strikes', 'balls', 'supported_pa',
                     'pitch_type', 'plate_x', 'plate_z', 'is_pa_terminal']
    frame = pd.read_parquet(processed, columns=columns).sort_values(['game_date', *KEY], ignore_index=True)
    assert len(frame) == quality['rows']
    assert frame.game_date.dt.year.isin([2023, 2024, 2025]).all()
    assert not frame.duplicated(KEY).any()
    cohort = json.loads((run/'cohort_manifest.json').read_text())
    target = frame.pitcher.isin(cohort['pitcher_ids']) & frame.pitcher.eq(frame.starter_pitcher)
    supported = eligible(frame)
    train_all = frame[frame.split.eq('train') & supported]
    train = pd.concat([train_all.sample(min(len(train_all), config['train_random_rows']), random_state=config['seed']),
                       train_all[target.reindex(train_all.index)]]).drop_duplicates(KEY).sort_index()
    cal_all = frame[frame.split.eq('calibration') & supported]
    cal = cal_all.sample(min(len(cal_all), config['calibration_rows']), random_state=config['seed']).sort_index()
    dev = frame[frame.split.eq('dev') & supported & target]
    parts = {'train': train, 'calibration': cal, 'dev': dev}
    assert {name: key_hash(part) for name, part in parts.items()} == samples['rows_hash']
    for name, bounds in [('train', ('2023-05-15', '2025-04-30')),
                         ('calibration', ('2025-05-01', '2025-06-30')),
                         ('dev', ('2025-07-01', '2025-09-30'))]:
        assert parts[name].game_date.between(*bounds).all(), name+' date boundary violated'
    for left, right in [('train', 'calibration'), ('train', 'dev'), ('calibration', 'dev')]:
        assert set(parts[left].game_pk).isdisjoint(parts[right].game_pk)
    binary = supported & frame.is_pa_terminal & frame.strikes.eq(2) & (
        frame.description.eq('hit_into_play') |
        (frame.events.eq('strikeout') & frame.description.isin(['swinging_strike', 'swinging_strike_blocked'])))
    binary_parts = {}
    for name, limit in [('train', config['train_random_rows']), ('calibration', config['calibration_rows']), ('dev', None)]:
        selected = frame[binary & frame.split.eq(name) & (target if name == 'dev' else True)]
        if limit is not None:
            selected = selected.sample(min(len(selected), limit), random_state=config['seed']).sort_index()
        binary_parts[name] = selected
    coverage = {}
    for name in parts:
        candidate = frame[frame.split.eq(name) & (target if name == 'dev' else True)]
        selected = binary_parts[name]
        coverage[name] = {'candidate_pitches': len(candidate),
                          'candidate_pitcher_game_appearances': int(candidate.groupby(['game_pk', 'pitcher']).ngroups),
                          'candidate_starter_pitcher_games': int(candidate.loc[candidate.pitcher.eq(candidate.starter_pitcher)].groupby(['game_pk', 'pitcher']).ngroups),
                          'all_count_eligible': int(supported[candidate.index].sum()),
                          'binary_used': summarize(selected),
                          'binary_used_description_counts': selected.description.value_counts().to_dict(),
                          'binary_class_1_inplay': int(selected.description.eq('hit_into_play').sum())}
    raw_dev = frame[frame.split.eq('dev') & target]
    pitcher_names = {int(item['pitcher']): item['player_name'] for item in cohort['selected']}
    supported_starts = pd.MultiIndex.from_frame(dev[['game_pk', 'pitcher']].drop_duplicates())
    missing = raw_dev.loc[~pd.MultiIndex.from_frame(raw_dev[['game_pk', 'pitcher']]).isin(supported_starts)]
    coverage['dev']['starts_without_eligible_pitch'] = [
        {'game_pk': int(game), 'pitcher': int(pitcher), 'player_name': pitcher_names[int(pitcher)], 'pitches': len(part),
         'date': str(part.game_date.iloc[0]), 'supported_pa_pitches': int(part.supported_pa.sum())}
        for (game, pitcher), part in missing.groupby(['game_pk', 'pitcher'])]
    report = {'status': 'awaiting_predictions', 'source_snapshot_verified': True,
              'source_unchanged_at_completion': runtime is not None,
              'processed_sha256': quality['processed_sha256'], 'raw_data_read': False,
              'cohort_ids': cohort['pitcher_ids'],
              'all_count_samples': {name: summarize(part) for name, part in parts.items()},
              'task_coverage': coverage,
              'limits': ['Binary keys/coverage reconstructed after fitting, not a separately timestamped pre-fit manifest.',
                         'Binary per-row predictions were not saved by this runner; binary metric replay requires archived checkpoints and encoders.',
                         'Count/hand baseline uses all eligible train rows; neural architectures use matched capped plus cohort-enriched rows.',
                         'Calibration split supplies both checkpoint selection and temperatures; no independent calibration score is claimed.',
                         'Single seed and previously inspected DEV; bootstrap captures game sampling uncertainty only.']}
    supplement = run/'audit_supplemental'
    supplement.mkdir(exist_ok=True)
    with (run/'encoders.pkl').open('rb') as stream:
        encoders = pickle.load(stream)
    (supplement/'normalizer_report.json').write_text(json.dumps(encoders['normalizer'].report(), indent=2)+'\n')
    shutil.copyfile(root/'reports/data_quality.json', supplement/'data_quality.json')
    shutil.copyfile(root/'cache/sequence_physics.json', supplement/'sequence_physics.json')
    protocol = Path(__file__).resolve().parents[1]/'docs/SEQUENCE_PROTOCOL.md'
    shutil.copyfile(protocol, supplement/'SEQUENCE_PROTOCOL.md')
    report['supplemental_provenance'] = {
        'note': 'Copied during post-start audit; protocol was authored before training, copy time is not a pretraining hash freeze.',
        'files': {path.name: digest(path) for path in supplement.iterdir() if path.is_file()}}
    if args.replay_binary or args.replay_all_count:
        assert runtime is not None, 'Wait for fitting to finish before replaying checkpoints'
        from pitchmdp.archetypes import add_batter_style_history
        from pitchmdp.sequence_data import HistoryStore, join_physics
        from pitchmdp.sequence_model import SequenceModel, classification_metrics
        from scipy.special import softmax
        physics_metadata = json.loads((supplement/'sequence_physics.json').read_text())
        physics_path = root/'cache/sequence_physics.parquet'
        assert digest(physics_path) == physics_metadata['sha256']
        assert physics_metadata['identity']['processed_sha256'] == quality['processed_sha256']
        additional = ['stand', 'p_throws', 'outs_when_up', 'inning', 'home_score', 'away_score',
                      'bases', 'inning_topbot', 'launch_angle', 'release_spin_rate', 'pfx_x', 'pfx_z']
        full = pd.read_parquet(processed, columns=columns+additional)
        full = join_physics(full, pd.read_parquet(physics_path))
        full.sort_values(['game_date', *KEY], inplace=True, ignore_index=True)
        add_batter_style_history(full)
        store = HistoryStore.from_frame(full, encoders['normalizer'])
    if args.replay_binary:
        selected = binary_parts['dev']
        tokens, valid = store.gather(selected.index.to_numpy())
        context = encoders['context'].transform(full.iloc[selected.index])
        by = selected.description.eq('hit_into_play').to_numpy(np.int64)
        binary_results = json.loads((run/'binary_results.json').read_text())
        replay, predictions = {}, {}
        for kind in config['binary_models']:
            model = SequenceModel.load(run/f'binary_{kind}.pt')
            logits = model.logits((tokens, valid, context))
            probabilities = softmax(logits/model.temperature, axis=1)
            predictions[kind] = probabilities
            calibrated = classification_metrics(by, probabilities)
            for metric in ['log_loss', 'brier_multiclass', 'accuracy', 'auc']:
                np.testing.assert_allclose(calibrated[metric], binary_results[kind]['metrics_conditional_current_physics'][metric], rtol=1e-5, atol=1e-6)
            replay[kind] = {'calibrated': calibrated, 'uncalibrated': classification_metrics(by, softmax(logits, axis=1))}
        np.savez_compressed(supplement/'binary_heldout_predictions.npz', y=by,
                            pitch_keys=selected[KEY].to_numpy(), game_pk=selected.game_pk.to_numpy(), **predictions)
        comparison = paired_bootstrap(by, predictions['transformer'], predictions['flatten_mlp'], selected.game_pk.to_numpy(), 2000)
        expected = binary_results['paired_transformer_minus_flatten']
        for metric in ['model_minus_reference_log_loss', 'bootstrap95']:
            np.testing.assert_allclose(comparison[metric], expected[metric], rtol=1e-5, atol=1e-6)
        report['binary_checkpoint_replay'] = replay
        report['binary_paired_comparison'] = comparison
        report['limits'][1] = 'Binary per-row predictions replayed after fitting from archived checkpoints/encoders; labels and selected keys reconstructed independently.'
    if args.replay_all_count:
        results = json.loads((run/'all_count_results.json').read_text())
        rows = dev.index.to_numpy()
        tokens, valid = store.gather(rows)
        context = encoders['context'].transform(full.iloc[rows])
        y = outcome_labels(dev)
        replay, probabilities = {}, {}
        with np.load(run/'heldout_predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['pitch_keys'], dev[KEY].to_numpy())
            for kind in config['all_count_models']:
                print('REPLAY_ALL_COUNT '+kind, flush=True)
                model_path = run/f'all_{kind}.pt'
                model = SequenceModel.load(model_path)
                conditional_logits = model.logits((tokens, valid, context))
                delivery_logits, levels = encoders['delivery'].logits(model, store, encoders['context'], rows)
                variants = {
                    'conditional_uncalibrated': softmax(conditional_logits, axis=-1),
                    'conditional_calibrated': softmax(conditional_logits/model.temperature, axis=-1),
                    'delivery_integrated_uncalibrated': softmax(delivery_logits, axis=-1).mean(1),
                    'delivery_integrated_calibrated': softmax(delivery_logits/model.delivery_temperature, axis=-1).mean(1)}
                np.testing.assert_allclose(variants['delivery_integrated_calibrated'], saved[kind], rtol=1e-5, atol=1e-6)
                metrics = {name: classification_metrics(y, p) for name, p in variants.items()}
                for variant, saved_key in [('conditional_calibrated', 'conditional_current_physics_diagnostic'),
                                           ('delivery_integrated_calibrated', 'primary_delivery_integrated')]:
                    for metric in ['log_loss', 'brier_multiclass', 'accuracy']:
                        np.testing.assert_allclose(metrics[variant][metric], results[kind][saved_key][metric], rtol=1e-5, atol=1e-6)
                replay[kind] = {
                    'model_sha256': digest(model_path), 'conditional_temperature': model.temperature,
                    'delivery_temperature': model.delivery_temperature, 'metrics': metrics,
                    'calibrated_probability_max_abs_difference': float(np.max(np.abs(variants['delivery_integrated_calibrated']-saved[kind]))),
                    'delivery_pool_level_counts': {str(int(level)): int((levels == level).sum()) for level in np.unique(levels)}}
                probabilities.update({kind+'__'+name: p for name, p in variants.items()})
        replay_manifest = {
            'scope': 'All fixed architectures and both fixed observation regimes, before and after saved calibration; no fitting or selection.',
            'n': len(dev), 'row_key_sha256': key_hash(dev), 'delivery_draws': encoders['delivery'].draws,
            'encoder_sha256': digest(run/'encoders.pkl'), 'frozen_source_hashes': expected_sources,
            'models': replay}
        (supplement/'all_count_calibration_replay.json').write_text(json.dumps(replay_manifest, indent=2)+'\n')
        np.savez_compressed(supplement/'all_count_calibration_predictions.npz', y=y,
                            pitch_keys=dev[KEY].to_numpy(), game_pk=dev.game_pk.to_numpy(), **probabilities)
        report['all_count_checkpoint_replay'] = replay_manifest
    predictions_file = run/'heldout_predictions.npz'
    if predictions_file.exists():
        results = json.loads((run/'all_count_results.json').read_text())
        with np.load(predictions_file, allow_pickle=False) as saved:
            y, games = saved['y'], saved['game_pk']
            np.testing.assert_array_equal(saved['pitch_keys'], dev[KEY].to_numpy())
            np.testing.assert_array_equal(y, outcome_labels(dev))
            np.testing.assert_array_equal(games, dev.game_pk.to_numpy())
            report['probability_metrics'] = {}
            for name in ['current_only', 'flatten_mlp', 'transformer', 'count_hand']:
                probability = saved[name]
                assert probability.shape == (len(y), 10)
                assert np.isfinite(probability).all() and (probability >= 0).all()
                np.testing.assert_allclose(probability.sum(1), 1., atol=1e-5)
                observed = {'log_loss': float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-12, 1)).mean()),
                            'brier_multiclass': float(((probability-np.eye(10)[y])**2).sum(1).mean())}
                expected = results['count_hand_baseline'] if name == 'count_hand' else results[name]['primary_delivery_integrated']
                for metric, value in observed.items():
                    np.testing.assert_allclose(value, expected[metric], rtol=1e-7, atol=1e-8)
                report['probability_metrics'][name] = observed
            report['paired_comparisons'] = {}
            for name in ['flatten_mlp', 'current_only', 'count_hand']:
                comparison = paired_bootstrap(y, saved['transformer'], saved[name], games, config['bootstrap_replicates'])
                result_key = 'count_hand_baseline' if name == 'count_hand' else name
                expected = results['paired_comparisons'][result_key]
                for metric in ['model_minus_reference_log_loss', 'bootstrap95']:
                    np.testing.assert_allclose(comparison[metric], expected[metric], rtol=1e-6, atol=1e-8)
                report['paired_comparisons'][name] = comparison
        report['status'] = 'passed'
        report['prediction_sha256'] = digest(predictions_file)
    shutil.copyfile(Path(__file__).resolve(), supplement/'audit_sequence_run.py')
    report['supplemental_provenance']['files'] = {
        path.name: digest(path) for path in supplement.iterdir() if path.is_file()}
    output = run/'sequence_audit.json'
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(output),
                      'binary_samples': {name: len(part) for name, part in binary_parts.items()}}, indent=2))


if __name__ == '__main__':
    main()
