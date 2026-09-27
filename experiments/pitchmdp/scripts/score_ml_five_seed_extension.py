"""Score the complete five-seed F1 extension (full G0 minus masked G0) on the exposed Cpanel.

Opens only after the frozen C1 scorer has produced its analysis and all ten
ordered member predictions verify. Requires the extension baseline archive to
have the same SHA256 as C1's ``parent_baseline_predictions.npz`` and the
full-arm five-seed primary/calibrated/raw/seed_primary arrays to be byte-equal
to the C1 ``G0-global`` analysis arrays computed by the same frozen
summarization. One primary NLL hypothesis with the frozen five-seed rule
(four of five same-seed deltas negative), the unchanged 24-bound R family and
descriptive batter-volume summaries. The original three-seed result is
reported alongside; nothing here is independent or held-out confirmation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))

import numpy as np
import pandas as pd

from pitchmdp.data import KEY, hash_file
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_metrics import prediction_metrics, pitch_losses, paired_game_comparison
from pitchmdp.matrix_five_seed_extension import (BOOTSTRAP, DECISION, MULTIPLICITY, NEW_SEEDS, PARENT_CELL, REUSED_SEEDS,
    ROBUSTNESS, SEEDS, STAGE_TEXT, check_full_arrays_equal, five_seed_decision, seed_deltas)
import run_ml_bridge as bridge
import run_ml_five_seed_extension as runner
from run_ml_benchmark import read_json, dump, validate_native_runtime
from run_ml_matrix import assert_hashes, heavy_lock
from score_ml_bridge import batter_descriptive, robustness
from score_ml_matrix import archive, assert_aligned, summarize_cell


def analyze(full_members, masked_members, baseline, metadata, volume, *, c1_arrays, g_stored, g_report, f1_stored, f1_report):
    """Pure scoring core: reconstruct both three-seed references, bind full arm to C1, compare five seeds."""
    if len(full_members) != len(SEEDS) or len(masked_members) != len(SEEDS):
        raise ValueError('Five ordered full and five ordered masked members required')
    for member in (*full_members, *masked_members):
        assert_aligned(member, baseline)
    bridge.verify_full_reconstruction(full_members[:3], baseline, g_stored, g_report)
    three_report, three_values = summarize_cell(masked_members[:3], baseline)
    for kind in ('primary', 'calibrated', 'raw', 'seed_primary'):
        if not np.array_equal(f1_stored['masked_' + kind], three_values[kind]):
            raise ValueError('Preserved masked three-seed reconstruction differs: ' + kind)
    if runner._normalized(three_report['selection']) != runner._normalized(f1_report['selection']):
        raise ValueError('Preserved masked three-seed June weight differs')
    full_report, full = summarize_cell(full_members, baseline)
    check_full_arrays_equal(full, c1_arrays)
    masked_report, masked = summarize_cell(masked_members, baseline)
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    if not np.array_equal(metadata[KEY].to_numpy(np.int64), baseline['dev_keys']):
        raise ValueError('Unpaired DEV metadata')
    primary = paired_game_comparison(y, full['primary'], masked['primary'], games, draws=BOOTSTRAP['draws'], seed=BOOTSTRAP['seed'])
    deltas = seed_deltas(y, full['seed_primary'], masked['seed_primary'], pitch_losses)
    return {'reports': {'full': full_report, 'masked': masked_report},
            'primary': {'candidate': 'full', 'control': 'masked', 'paired': primary, 'decision': five_seed_decision(primary, deltas),
                        'rule': DECISION, 'multiplicity': MULTIPLICITY},
            'robustness': robustness(y, full['primary'], masked['primary'], games, metadata, rule=ROBUSTNESS),
            'batter_volume': batter_descriptive(y, full['primary'], masked['primary'], games, metadata.batter.to_numpy(), volume)}, full, masked


def score(bundle_path, local_path):
    bundle = runner.load_bundle(bundle_path, real=True)
    runner.verify_bundle_pins(bundle)
    c1_config, ext = runner.load_configs(bundle)
    local = read_json(local_path)
    validate_native_runtime()
    started = time.perf_counter()
    ten = runner.verify_ten_predictions(bundle, ext, c1_config, local_path, local)
    prep, live, output = ten['prep'], ten['c1'], ten['ext_output']
    destination = output / 'analysis' / 'five_seed'
    if destination.exists():
        raise ValueError('Preserve completed or interrupted five-seed scoring output')
    c1_out = live['run']
    c1_analysis = c1_out / 'analysis'
    if not (c1_analysis / 'manifest.json').is_file():
        raise ValueError('C1 score must complete before the F1-extension score')
    c1_manifest = read_json(c1_analysis / 'manifest.json')
    for name, suffix in (('results', '.json'), ('predictions', '.npz')):
        if hash_file(c1_analysis / (name + suffix)) != c1_manifest[name + '_sha256']:
            raise ValueError('C1 analysis archive changed')
    for path, digest in c1_manifest['inputs'].items():
        if hash_file(Path(path)) != digest:
            raise ValueError('C1 analysis input changed: ' + path)
    c1_results = read_json(c1_analysis / 'results.json')
    if c1_results['selection_status'] != 'baseline_stability_only' or c1_results['cells'] != [PARENT_CELL] or c1_results['comparisons']:
        raise ValueError('C1 analysis is not the baseline-stability G0 registration')
    baseline_sha = hash_file(output / 'baseline_predictions.npz')
    if baseline_sha != hash_file(c1_out / 'parent_baseline_predictions.npz'):
        raise ValueError('Extension baseline archive differs from C1 parent_baseline_predictions.npz')
    inputs = {**ten['hashes'], str(output / 'baseline_predictions.npz'): baseline_sha,
              str(c1_out / 'parent_baseline_predictions.npz'): baseline_sha,
              **{str(output / name): hash_file(output / name) for name in ('preparation.json', 'dev_metadata.parquet', 'batter_train_volume.json')},
              **{str(c1_analysis / name): hash_file(c1_analysis / name) for name in ('manifest.json', 'results.json', 'predictions.npz')},
              **prep['external_hashes']}
    g_run = Path(prep['parent_run'])
    g_analysis = g_run / 'analysis' / 'panel'
    f1_analysis = Path(prep['parent_f1_run']) / 'analysis' / 'bridge'
    for folder in (g_analysis, f1_analysis):
        manifest = read_json(folder / 'manifest.json')
        for name, suffix in (('results', '.json'), ('predictions', '.npz')):
            if hash_file(folder / (name + suffix)) != manifest[name + '_sha256']:
                raise ValueError('Frozen parent analysis changed: ' + str(folder))
    full_members = [archive(Path(ten['full'][seed]['predictions_path'])) for seed in SEEDS]
    for seed, member in zip(SEEDS, prep['full_reuse']['members']):
        if Path(member['member_dir']) != Path(ten['full'][seed]['directory']):
            raise ValueError('Full-arm member order differs from preparation')
    masked_members = [archive(ten['masked'][seed] / 'predictions.npz') for seed in SEEDS]
    baseline = archive(output / 'baseline_predictions.npz')
    metadata = pd.read_parquet(output / 'dev_metadata.parquet')
    volume = read_json(output / 'batter_train_volume.json')
    c1_arrays = archive(c1_analysis / 'predictions.npz')
    for name in ('keys', 'y', 'game_pk', 'pitcher'):
        if not np.array_equal(c1_arrays[name], baseline['dev_' + name]):
            raise ValueError('C1 analysis metadata differs: ' + name)
    core, full, masked = analyze(full_members, masked_members, baseline, metadata, volume, c1_arrays=c1_arrays,
                                 g_stored=archive(g_analysis / 'predictions.npz'),
                                 g_report=read_json(g_analysis / 'results.json')['reports'][PARENT_CELL],
                                 f1_stored=archive(f1_analysis / 'predictions.npz'),
                                 f1_report=read_json(f1_analysis / 'results.json')['reports']['masked'])
    c1_report = c1_results['reports'][PARENT_CELL]
    if runner._normalized(core['reports']['full']['selection']) != runner._normalized(c1_report['selection']):
        raise ValueError('Full-arm five-seed June weight differs from the C1 analysis')
    y, games = baseline['dev_y'], baseline['dev_game_pk']
    fits = {'new_masked': [read_json(ten['masked'][seed] / 'fit.json')['seconds_total'] for seed in NEW_SEEDS],
            'reused_masked': [m['logical_fit_seconds'] for m in prep['masked_reuse']['members']],
            'full': [m['logical_fit_seconds'] for m in prep['full_reuse']['members']]}
    result = {'experiment_id': ext['experiment_id'], 'scope': 'Cpanel', 'stage': STAGE_TEXT, 'n': len(y),
              'games': len(np.unique(games)), 'seeds': list(SEEDS), 'reused_seeds': list(REUSED_SEEDS), 'new_seeds': list(NEW_SEEDS),
              'comparison': 'five-seed full G0-global minus five-seed batter-masked G0-global', 'frequency': prediction_metrics(y, baseline['dev']),
              **core,
              'three_seed_result': {'source': str(f1_analysis / 'results.json'), 'sha256': hash_file(f1_analysis / 'results.json'),
                                    'primary': prep['three_seed_result'],
                                    'note': 'Original three-seed screen retained; the five-seed result neither replaces nor independently confirms it'},
              'c1_binding': {'analysis_manifest_sha256': hash_file(c1_analysis / 'manifest.json'), 'baseline_sha256_equal': True,
                             'full_arrays_byte_equal': True, 'c1_experiment_id': c1_results['experiment_id']},
              'costs': {'logical_fit_seconds': fits, 'new_fits': len(NEW_SEEDS), 'reused_fits': len(REUSED_SEEDS) * 2 + len(NEW_SEEDS),
                        'authoritative_ledger': 'external supervisor ledger; runner-internal timings are logical costs only'},
              'input_hashes': inputs, 'config_sha256': canonical_hash(ext), 'bundle_sha256': canonical_hash(bundle),
              'policy_effect': None, 'whole_mlb_robustness': None, 'independent_confirmation': None, 'held_out_confirmation': False,
              'f4_transfer_claim': None,
              'limits': ['Five paired seeds on the previously exposed Cpanel DEV; development stability, not confirmation',
                         'Second look at the same hypothesis and games after a passing three-seed screen; no fresh alpha guarantee',
                         'Masked arm removes the 17 existing batter channels only; H5 history retains batter-related signal',
                         'Effective input capacity differs by design (information-removal control)',
                         'Five-seed June ensemble weights are new for both arms; three-seed results are retained separately',
                         'Conditional bootstrap excludes training/calibration/selection uncertainty',
                         'Unobserved R groups remain unconfirmed; zero-TRAIN volume is structurally absent on Cpanel'],
              'seconds': time.perf_counter() - started, 'scored_utc': datetime.now(timezone.utc).isoformat(),
              'scoring_sources': {name: hash_file(PROJECT / name) for name in runner.SOURCES}}
    destination.mkdir(parents=True)
    np.savez_compressed(destination / 'predictions.npz',
        **{name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')},
        **{arm + '_' + name: value for arm, values in (('full', full), ('masked', masked)) for name, value in values.items()})
    dump(destination / 'results.json', result)
    dump(destination / 'manifest.json', {'results_sha256': hash_file(destination / 'results.json'),
         'predictions_sha256': hash_file(destination / 'predictions.npz'), 'inputs': inputs})
    for name in ('predictions.npz', 'results.json', 'manifest.json'):
        runner._freeze(destination / name)
    print({'full': core['reports']['full']['primary']['log_loss'], 'masked': core['reports']['masked']['primary']['log_loss'],
           'decision': core['primary']['decision']['status'], 'R': core['robustness']['status']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--local-config', type=Path, required=True)
    args = parser.parse_args()
    local = read_json(args.local_config)
    with heavy_lock(Path(local['artifact_root']).resolve()):
        score(args.bundle, args.local_config)


if __name__ == '__main__':
    main()
