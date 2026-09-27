"""G0/F1 five-seed bundle: synthetic CPU contract checks only; no real data, parents, fits or DEV scores."""
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
from pitchmdp.data import hash_file
from pitchmdp.matrix_bridge import fit_masked, network_signature
from pitchmdp.matrix_data import canonical_hash
from pitchmdp.matrix_metrics import pitch_losses, paired_game_comparison
from pitchmdp.matrix_models import MatrixModel
from pitchmdp.matrix_policy_artifacts import APPLEDOUBLE_MAGIC
import pitchmdp.matrix_five_seed_extension as ext_module
from pitchmdp.matrix_five_seed_extension import (DECISION, SEQUENCE, STAGE_CAPS, SHARE_SECONDS, arm_obligation,
    check_full_arrays_equal, check_same_seed_pair, check_signature_matches_report, effective_cap, five_seed_decision,
    launch_gate, ledger_totals, member_fit_seconds, project_costs, seed_deltas, third_parent_identity,
    validate_bundle, validate_extension_config, validate_c1_config_shape)
import run_ml_bridge as bridge
import run_ml_five_seed_extension as runner
import score_ml_five_seed_extension as scorer
import supervise_ml_five_seed_extension as supervisor
from run_ml_benchmark import dump, read_json
from score_ml_matrix import summarize_cell
from bridge_synthetic import synthetic_family, synthetic_metadata

CONFIGS = REPO / 'configs'


def draft(name):
    return json.loads((CONFIGS / name).read_text())


# ---------------------------------------------------------------- decision rule

def comparison(delta_nll, ci, p, brier_ci):
    return {'status': 'measured', 'nll': {'delta': delta_nll, 'ci95': ci, 'p_less': p},
            'brier': {'delta': -.0001, 'ci95': brier_ci, 'p_less': .01}}


def test_five_seed_decision_requires_every_criterion_and_four_of_five_seeds():
    good = comparison(-.005, [-.007, -.003], .001, [-.0005, .0005])
    passed = five_seed_decision(good, [-.004, -.005, -.006, -.001, -.002])
    assert passed['status'] == 'development_stability_pass' and passed['negative_seeds'] == 5
    assert passed['held_out_confirmation'] is False and passed['independent_confirmation'] is None
    assert passed['stage'] == 'five-seed development stability, exposed Cpanel' and 'no fresh alpha' in passed['multiplicity']
    four = five_seed_decision(good, [-.004, -.005, -.006, -.001, .002])
    assert four['status'] == 'development_stability_pass' and four['negative_seeds'] == 4
    three = five_seed_decision(good, [-.004, -.005, -.006, .001, .002])
    assert three['status'] == 'inconclusive' and three['criteria']['seed_direction_stable'] is False
    assert five_seed_decision(comparison(-.002, [-.004, -.001], .01, [-.0005, .0005]), [-.001] * 5)['status'] == 'inconclusive'
    worse = five_seed_decision(comparison(.004, [.001, .007], .9, [-.0005, .0005]), [.001] * 5)
    assert worse['status'] == 'worse_or_guardrail_failure' and worse['reverse_point_estimate'] is True and worse['reverse_note']
    guard = five_seed_decision(comparison(-.005, [-.007, -.003], .001, [.002, .004]), [-.004] * 5)
    assert guard['status'] == 'worse_or_guardrail_failure' and guard['reverse_point_estimate'] is False
    assert five_seed_decision({'status': 'insufficient_games', 'nll': {'p_less': None}}, [-.1] * 5)['status'] == 'inconclusive'
    for bad in ([-.1] * 3, [-.1] * 4, [-.1] * 6, [-.1, -.1, -.1, -.1, float('nan')]):
        with pytest.raises(ValueError):
            five_seed_decision(good, bad)
    assert DECISION['required_negative_seeds'] == 4 and DECISION['seeds'] == 5


def test_seed_deltas_require_five_pairs_and_pair_by_index():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 50)
    full = [rng.dirichlet(np.ones(10), 50) for _ in range(5)]
    masked = [rng.dirichlet(np.ones(10), 50) for _ in range(5)]
    deltas = seed_deltas(y, full, masked, pitch_losses)
    assert len(deltas) == 5
    assert deltas[2] == pytest.approx(float((pitch_losses(y, full[2]) - pitch_losses(y, masked[2]))[:, 0].mean()))
    with pytest.raises(ValueError):
        seed_deltas(y, full[:3], masked[:3], pitch_losses)


# ---------------------------------------------------------------- configuration drafts

def test_draft_configs_validate_as_drafts_and_real_mode_rejects_null_pins():
    ext = validate_extension_config(draft('EXP-P9-002.yaml'), real=False)
    assert ext['decision'] == DECISION and ext['held_out_confirmation'] is False and ext['independent_confirmation'] is None
    with pytest.raises(ValueError, match='frozen before a real command'):
        validate_extension_config(draft('EXP-P9-002.yaml'), real=True)
    bundle = validate_bundle(draft('ML-G0-F1-FIVE-SEED-v1.json'), real=False)
    assert set(bundle['sources']) == set(runner.SOURCES) and 'scripts/score_ml_confirmation.py' in bundle['sources']
    assert 'scripts/score_ml_bridge.py' in bundle['sources'] and bundle['sequence'] == [list(s) for s in SEQUENCE]
    with pytest.raises(ValueError, match='repo_commit'):
        validate_bundle(draft('ML-G0-F1-FIVE-SEED-v1.json'), real=True)
    from pitchmdp.matrix_confirmation import validate_config
    c1 = validate_c1_config_shape(validate_config(draft('EXP-P10-001.yaml')))
    assert c1['registration']['batch_wall_budget_seconds'] == 7200 and c1['cells'] == ['G0-global'] and c1['primary_comparisons'] == []
    assert ext['parent_c1']['config_sha256'] == canonical_hash(c1)
    assert ext['parent_g']['preparation_sha256'] == c1['parent_preparation_sha256']
    assert ext['parent_g']['analysis_manifest_sha256'] == c1['parent_analysis_sha256']


def test_frozen_settings_budgets_caps_and_sequence_cannot_change():
    ext = draft('EXP-P9-002.yaml')
    for key, value in (('budget', {**ext['budget'], 'epochs': 20}), ('decision', {**DECISION, 'required_negative_seeds': 3}),
                       ('draws', 100), ('seeds', [0, 1, 2]), ('device', 'cpu'), ('family_wall_budget_seconds', 7201),
                       ('held_out_confirmation', True), ('mask', {'start': 11, 'stop': 27})):
        with pytest.raises(ValueError):
            validate_extension_config({**ext, key: value}, real=False)
    bad = deepcopy(ext); bad['parent_c1']['run'] = ext['parent_g']['run']
    with pytest.raises(ValueError, match='distinct'):
        validate_extension_config(bad, real=False)
    bundle = draft('ML-G0-F1-FIVE-SEED-v1.json')
    for mutate in (lambda b: b['arms']['c1'].update(share_seconds=7201), lambda b: b.update(combined_seconds=14401),
                   lambda b: b['caps'].update(profile=7200), lambda b: b.update(termination_grace_seconds=600),
                   lambda b: b['sequence'].reverse(), lambda b: b['arms'].__setitem__('f1ext', {**b['arms']['f1ext'], 'output': b['arms']['c1']['output']}),
                   lambda b: b['sources'].pop('scripts/score_ml_confirmation.py')):
        broken = deepcopy(bundle); mutate(broken)
        with pytest.raises(ValueError):
            validate_bundle(broken, real=False)
    c1 = draft('EXP-P10-001.yaml')
    for mutate in (lambda c: c['registration'].update(batch_wall_budget_seconds=14400), lambda c: c.update(cells=['G0-global', 'G2-feature']),
                   lambda c: c.update(selection_status='candidate_comparison')):
        broken = deepcopy(c1); mutate(broken)
        with pytest.raises(ValueError):
            validate_c1_config_shape(broken)


# ---------------------------------------------------------------- ledger accounting

def job(arm, stage, seed=None, *, ended=True, elapsed=10., cap=7200.):
    return {'job_id': f'{arm}-{stage}-{seed}-{elapsed}', 'arm': arm, 'stage': stage, 'seed': seed, 'cap_seconds': cap,
            'ended': ended, 'elapsed_seconds': elapsed if ended else None}


def test_ledger_totals_charge_ended_wall_and_reserve_unresolved_caps_per_arm_without_double_counting():
    jobs = [job('c1', 'prepare', elapsed=5.), job('c1', 'fit', 3, elapsed=100.), job('c1', 'fit', 3, elapsed=50.),
            job('f1ext', 'profile', ended=False, cap=600.), job('f1ext', 'fit', 3, elapsed=90.)]
    c1 = ledger_totals(jobs, 'c1')
    assert c1['ended_seconds'] == 155. and c1['reserved_seconds'] == 0. and c1['unresolved_jobs'] == []
    f1 = ledger_totals(jobs, 'f1ext')
    assert f1['ended_seconds'] == 90. and f1['reserved_seconds'] == 600. and len(f1['unresolved_jobs']) == 1
    assert c1['charged_seconds'] + f1['charged_seconds'] == 845.
    assert member_fit_seconds(jobs, 'c1', 3) == 150. and member_fit_seconds(jobs, 'c1', 4) == 0.
    assert member_fit_seconds([job('c1', 'fit', 4, ended=False, cap=7000.)], 'c1', 4) == 7000.


def test_effective_cap_is_limited_by_stage_share_member_and_grace():
    assert effective_cap('fit', 7200., grace=30) == 7200.
    assert effective_cap('fit', 1000., grace=30) == 1000.
    assert effective_cap('predict', 5000., member_remaining=800., grace=30) == 800.
    assert effective_cap('profile', 7200., grace=30) == 600.
    with pytest.raises(RuntimeError, match='full 600-second cap'):
        effective_cap('profile', 599., grace=30)
    with pytest.raises(RuntimeError, match='fail closed'):
        effective_cap('fit', 30., grace=30)
    with pytest.raises(RuntimeError, match='fail closed'):
        effective_cap('predict', 7200., member_remaining=10., grace=30)


def test_launch_gate_refuses_unresolved_projection_and_obligation_overruns():
    totals = ledger_totals([job('c1', 'prepare', elapsed=100.)], 'c1')
    ok = launch_gate(totals, SHARE_SECONDS, 'fit', projected_command=1000., remaining_obligation=5000.)
    assert ok['remaining_share_seconds'] == 7100.
    with pytest.raises(RuntimeError, match='remaining projected obligation'):
        launch_gate(totals, SHARE_SECONDS, 'fit', projected_command=1000., remaining_obligation=7101.)
    with pytest.raises(RuntimeError, match='exceeds remaining arm share'):
        launch_gate(totals, SHARE_SECONDS, 'fit', projected_command=7101., remaining_obligation=7101.)
    with pytest.raises(RuntimeError, match='include this command'):
        launch_gate(totals, SHARE_SECONDS, 'fit', projected_command=10., remaining_obligation=5.)
    with pytest.raises(RuntimeError, match='Unresolved'):
        launch_gate(ledger_totals([job('c1', 'fit', 3, ended=False)], 'c1'), SHARE_SECONDS, 'predict', projected_command=1., remaining_obligation=1.)
    with pytest.raises(RuntimeError, match='Finite'):
        launch_gate(totals, SHARE_SECONDS, 'fit', projected_command=float('nan'), remaining_obligation=1.)


def test_projection_covers_fit_predict_scoring_and_gates_member_and_arm():
    measured = {'command_overhead_seconds': 2., 'load_seconds': 4., 'fit_seconds': 2., 'calibration_seconds': .2,
                'inference_seconds': .1, 'scoring_probe_seconds': 3.}
    samples = {'train': 1252824, 'earlystop': 16000, 'temperature': 2603, 'blend': 4821, 'dev': 12334}
    full = project_costs(measured, samples, arm='c1')
    masked = project_costs(measured, samples, arm='f1ext')
    fit = 1. * (1252824 + 16000) / 67584 * 30
    assert full['fit_command_seconds'] == pytest.approx(6. + fit)
    assert full['predict_command_seconds'] == pytest.approx(6. + .2 / 64 * 2603 + .1 / 64 * (4821 + 12334))
    assert masked['predict_command_seconds'] == pytest.approx(full['predict_command_seconds'] + .1 / 64 * 64)
    assert full['score_seconds'] == pytest.approx(5.) and full['member_gate'] and full['arm_gate']
    assert full['arm_total_seconds'] == pytest.approx(2 * full['member_seconds'] + 5.)
    assert masked['arm_total_seconds'] == pytest.approx(2 * masked['member_seconds'] + 5. + 2 * masked['prepare_seconds'])
    heavy = project_costs({**measured, 'fit_seconds': 30.}, samples, arm='c1')
    assert heavy['member_gate'] is False and heavy['arm_gate'] is False
    projection = {'prepare_seconds': 10., 'fit_command_seconds': 100., 'predict_command_seconds': 20., 'score_seconds': 5.}
    assert arm_obligation(projection, set(), 'c1') == 245.
    assert arm_obligation(projection, {('fit', 3), ('predict', 3)}, 'c1') == 125.
    assert arm_obligation(projection, {('prepare', None), ('freeze', None), ('fit', 3), ('fit', 4), ('predict', 3), ('predict', 4)}, 'f1ext') == 5.
    assert arm_obligation(projection, set(), 'f1ext') == 265.


# ---------------------------------------------------------------- pairing and identity helpers

def fit_record(base_seed, **over):
    record = {'report': {'seed': base_seed, 'device': 'mps', 'network': {'kind': 'flatten_mlp', 'n_context': 52, 'n_token': 38,
                                                                    'length': 6, 'width': 128, 'n_classes': 10}, 'parameter_count': 141834},
              'train_rows_sha256': 'a' * 64, 'earlystop_rows_sha256': 'b' * 64}
    for key, value in over.items():
        (record['report'] if key in record['report'] else record)[key] = value
    return record


def test_same_seed_pair_rejects_device_network_rows_and_seed_mismatch():
    assert check_same_seed_pair(fit_record(3), fit_record(3), seed=3)['device'] == 'mps'
    for over, match in ((dict(device='cpu'), 'devices differ'), (dict(parameter_count=1), 'network signature'),
                        (dict(train_rows_sha256='c' * 64), 'train_rows_sha256'), (dict(earlystop_rows_sha256='c' * 64), 'earlystop_rows'),
                        (dict(seed=4), 'seed differs')):
        with pytest.raises(ValueError, match=match):
            check_same_seed_pair(fit_record(3), fit_record(3, **over), seed=3)
    with pytest.raises(ValueError, match='seed differs'):
        check_same_seed_pair(fit_record(3), fit_record(3), seed=4)
    signature = {'network': fit_record(3)['report']['network'], 'parameter_count': 141834}
    assert check_signature_matches_report(signature, fit_record(3)['report'])
    with pytest.raises(ValueError, match='signature differs'):
        check_signature_matches_report({**signature, 'parameter_count': 5}, fit_record(3)['report'])


def test_full_arrays_must_be_byte_equal_to_c1_analysis():
    full, _, baseline = synthetic_family(games=10, per_game=8)
    _, values = summarize_cell(full, baseline)
    c1 = {f'G0-global_{k}': v.copy() for k, v in values.items()}
    assert check_full_arrays_equal(values, c1)
    for kind, mutate in (('primary', lambda a: a + 1e-12), ('raw', lambda a: a.astype(np.float32)), ('seed_primary', lambda a: a[:2])):
        broken = dict(c1); broken['G0-global_' + kind] = mutate(c1['G0-global_' + kind])
        with pytest.raises(ValueError, match=kind):
            check_full_arrays_equal(values, broken)
    with pytest.raises(ValueError, match='calibrated'):
        check_full_arrays_equal(values, {k: v for k, v in c1.items() if k != 'G0-global_calibrated'})


def manifest(declared):
    hashes = lambda names: {name: 'd' * 64 for name in names}
    return {'c1_experiment_id': declared['experiment_id'], 'c1_run': declared['run'], 'c1_config_sha256': declared['config_sha256'],
            'c1_source_hashes': declared['source_hashes'], 'c1_preparation_sha256': 'e' * 64, 'c1_identity': {}, 'frozen_utc': 'now',
            'fits': {str(s): hashes(['state.json', 'model.pt', 'fit.json']) for s in (3, 4)},
            'members': {str(s): hashes(['prediction_state.json', 'predictions.npz', 'calibration.json', 'prediction_runtime.json']) for s in (3, 4)}}


def test_third_parent_manifest_binds_declaration_and_complete_hash_family():
    declared = {'experiment_id': 'EXP-P10-001', 'run': '/runs/EXP-P10-001', 'config_sha256': 'c' * 64,
                'source_hashes': {'scripts/run_ml_confirmation.py': 'f' * 64}, 'third_parent_manifest': 'third_parent/manifest.json'}
    good = manifest(declared)
    assert third_parent_identity(good, declared)
    assert third_parent_identity(good, {**declared, 'source_hashes': None})
    for mutate, match in ((lambda m: m.update(c1_run='/runs/other'), 'run'), (lambda m: m.update(c1_config_sha256='0' * 64), 'config identity'),
                          (lambda m: m.update(c1_source_hashes={}), 'source hashes'), (lambda m: m['fits']['4'].pop('model.pt'), 'seed 4'),
                          (lambda m: m['members'].pop('3'), 'seed 3'), (lambda m: m['fits']['3'].update({'model.pt': 'zz'}), 'hex'),
                          (lambda m: m.pop('c1_identity'), 'incomplete')):
        broken = deepcopy(good); mutate(broken)
        with pytest.raises(ValueError, match=match):
            third_parent_identity(broken, declared)


# ---------------------------------------------------------------- tiny CPU toy fit: same-seed signature pairing

def toy_arrays(n, seed):
    rng = np.random.default_rng(seed)
    tokens = rng.normal(size=(n, 6, 21)).astype(np.float32)
    tokens[:, -1, -11:] = 0
    valid = np.ones((n, 6), dtype=bool)
    context = rng.normal(size=(n, 59)).astype(np.float32)
    return tokens, valid, context


def test_toy_masked_fit_matches_full_signature_and_report_binding():
    from pitchmdp.matrix_sharing import training_arrays
    train, early = toy_arrays(64, 0), toy_arrays(16, 1)
    y, ey = np.arange(64) % 10, np.arange(16) % 10
    budget = dict(epochs=1, patience=1, batch_size=32, learning_rate=.001)
    full = MatrixModel('flatten_mlp', seed=3, width=8, device='cpu').fit(training_arrays(train), y, training_arrays(early), ey, **budget)
    masked = fit_masked(MatrixModel('flatten_mlp', seed=3, width=8, device='cpu'), train, y, early, ey, **budget)
    assert network_signature(full) == network_signature(masked)
    assert check_signature_matches_report(network_signature(masked), full.report)
    other = fit_masked(MatrixModel('flatten_mlp', seed=3, width=16, device='cpu'), train, y, early, ey, **budget)
    with pytest.raises(ValueError, match='signature differs'):
        check_signature_matches_report(network_signature(other), full.report)
    pair = check_same_seed_pair({'report': full.report, 'train_rows_sha256': 'a' * 64, 'earlystop_rows_sha256': 'b' * 64},
                                {'report': masked.report, 'train_rows_sha256': 'a' * 64, 'earlystop_rows_sha256': 'b' * 64}, seed=3)
    assert pair['parameter_count'] == full.report['parameter_count']


# ---------------------------------------------------------------- scorer core on synthetic families

def five_seed_family(seed=0, **kw):
    full, masked, baseline = synthetic_family(seed=seed, **kw)
    rng = np.random.default_rng(seed + 100)
    def extra(sharpness):
        members = []
        for _ in range(2):
            member = {k: v.copy() for k, v in full[0].items()}
            for name in ('blend', 'dev'):
                y = baseline[name + '_y']
                for suffix in ('', '_raw'):
                    p = rng.dirichlet(np.ones(10), len(y))
                    p[np.arange(len(y)), y] += sharpness
                    member[name + suffix] = p / p.sum(1, keepdims=True)
            members.append(member)
        return members
    return full + extra(.6), masked + extra(.2), baseline


def test_scorer_core_binds_full_arm_to_c1_reconstructs_three_seed_references_and_keeps_24_R_slots():
    full, masked, baseline = five_seed_family(games=40, per_game=40)
    metadata = synthetic_metadata(baseline)
    volume = {'counts': {'100': 5, '101': 20, '102': 40}, 'q25': 12.5, 'method': 'linear', 'boundary': 'low includes count == q25', 'positive_batters': 3}
    g_report, g_values = summarize_cell(full[:3], baseline)
    g_stored = {**{k: baseline['dev_' + k] for k in ('keys', 'y', 'game_pk', 'pitcher')}, **{'G0-global_' + k: v for k, v in g_values.items()}}
    f1_report, f1_values = summarize_cell(masked[:3], baseline)
    f1_stored = {'masked_' + k: v for k, v in f1_values.items()}
    _, five_values = summarize_cell(full, baseline)
    c1_arrays = {'G0-global_' + k: v for k, v in five_values.items()}
    core, out_full, out_masked = scorer.analyze(full, masked, baseline, metadata, volume, c1_arrays=c1_arrays, g_stored=g_stored,
                                                g_report=g_report, f1_stored=f1_stored, f1_report=f1_report)
    decision = core['primary']['decision']
    assert decision['status'] == 'development_stability_pass' and len(decision['seed_deltas']) == 5 and decision['negative_seeds'] == 5
    assert core['robustness']['family_size'] == 24 and core['robustness']['groups']['volume_zero']['structural_missing'] is True
    assert core['robustness']['status'] == 'unconfirmed' and out_full['seed_primary'].shape[0] == 5
    five_report, _ = summarize_cell(full, baseline)
    assert core['reports']['full']['selection'] == five_report['selection']
    tampered = dict(c1_arrays); tampered['G0-global_primary'] = c1_arrays['G0-global_primary'] * .999 + .0001
    with pytest.raises(ValueError, match='Full-arm five-seed array'):
        scorer.analyze(full, masked, baseline, metadata, volume, c1_arrays=tampered, g_stored=g_stored, g_report=g_report,
                       f1_stored=f1_stored, f1_report=f1_report)
    broken = dict(f1_stored); broken['masked_raw'] = f1_stored['masked_raw'] + 1e-9
    with pytest.raises(ValueError, match='masked three-seed reconstruction'):
        scorer.analyze(full, masked, baseline, metadata, volume, c1_arrays=c1_arrays, g_stored=g_stored, g_report=g_report,
                       f1_stored=broken, f1_report=f1_report)
    with pytest.raises(ValueError, match='Preserved G0'):
        scorer.analyze([*full[:2], full[4], *full[3:]], masked, baseline, metadata, volume, c1_arrays=c1_arrays, g_stored=g_stored,
                       g_report=g_report, f1_stored=f1_stored, f1_report=f1_report)
    with pytest.raises(ValueError, match='Five ordered'):
        scorer.analyze(full[:4], masked, baseline, metadata, volume, c1_arrays=c1_arrays, g_stored=g_stored, g_report=g_report,
                       f1_stored=f1_stored, f1_report=f1_report)


# ---------------------------------------------------------------- supervisor: synthetic bundle in tmp

def bundle_fixture(tmp_path, *, grace=.4):
    root = tmp_path / 'artifacts'
    protocol = root / 'runs' / 'ML-MATRIX-20260924'
    protocol.mkdir(parents=True)
    local = tmp_path / 'local.json'; dump(local, {'artifact_root': str(root)})
    c1 = draft('EXP-P10-001.yaml'); ext = draft('EXP-P9-002.yaml'); bundle = draft('ML-G0-F1-FIVE-SEED-v1.json')
    c1['registration']['output'] = str(protocol / 'EXP-P10-001')
    ext['registration']['output'] = str(protocol / 'EXP-P9-002')
    ext['parent_c1']['run'] = str(protocol / 'EXP-P10-001')
    ext['parent_c1']['source_hashes'] = {name: hash_file(PROJECT / name) for name in runner.c1.SOURCES}
    ext['parent_c1']['config_file'] = str(tmp_path / 'EXP-P10-001.yaml')
    for record in (ext['parent_g'], ext['parent_f1']):
        record['run'] = str(protocol / record['experiment_id'])
    c1['parent_run'] = ext['parent_g']['run']
    ext['parent_c1']['config_sha256'] = canonical_hash(c1)
    ext['parent_f1']['config_file'] = str(CONFIGS / 'EXP-P9-001-v2.yaml')
    dump(tmp_path / 'EXP-P10-001.yaml', c1); dump(tmp_path / 'EXP-P9-002.yaml', ext)
    bundle['repo_commit'] = runner.repo_head()
    bundle['sources'] = runner.source_hashes()
    bundle['termination_grace_seconds'] = grace
    for name, path in runner.CONTRACTS.items():
        bundle['contracts'][name] = {'path': path, 'sha256': hash_file(REPO / path)}
    bundle['configs']['c1'] = {'path': str(tmp_path / 'EXP-P10-001.yaml'), 'sha256': hash_file(tmp_path / 'EXP-P10-001.yaml')}
    bundle['configs']['f1ext'] = {'path': str(tmp_path / 'EXP-P9-002.yaml'), 'sha256': hash_file(tmp_path / 'EXP-P9-002.yaml')}
    bundle['configs']['f1_parent'] = {'path': str(CONFIGS / 'EXP-P9-001-v2.yaml'), 'sha256': hash_file(CONFIGS / 'EXP-P9-001-v2.yaml')}
    bundle['arms']['c1']['output'] = str(protocol / 'EXP-P10-001'); bundle['arms']['f1ext']['output'] = str(protocol / 'EXP-P9-002')
    bundle['ledger_dir'] = str(protocol / 'coordination' / 'five-seed-ledger')
    bundle_path = tmp_path / 'bundle.json'; dump(bundle_path, bundle)
    return SimpleNamespace(bundle=bundle_path, local=local, root=root, protocol=protocol, ledger=Path(bundle['ledger_dir']),
                           c1=protocol / 'EXP-P10-001', ext=protocol / 'EXP-P9-002', config=bundle, c1_config=c1, ext_config=ext)


def complete(fx, *steps):
    paths = supervisor.completion_paths(fx.config, read_json(fx.local))
    for step in steps:
        path = paths[step]; path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('{}')
        command, seed = step
        _, arm, stage, _ = next(item for item in SEQUENCE if item[0] == command and item[3] == seed)
        job_id = f'synthetic-{command}-seed{seed}'
        (fx.ledger / 'jobs').mkdir(parents=True, exist_ok=True)
        (fx.ledger / 'ends').mkdir(parents=True, exist_ok=True)
        dump(fx.ledger / 'jobs' / (job_id + '.json'),
             {'protocol': supervisor.LEDGER_PROTOCOL, 'job_id': job_id, 'command': command,
              'arm': arm, 'stage': stage, 'seed': seed, 'cap_seconds': STAGE_CAPS[stage], 'argv': [],
              'bundle_sha256': canonical_hash(fx.config), 'refused': False})
        dump(fx.ledger / 'ends' / (job_id + '.json'),
             {'job_id': job_id, 'outcome': 'completed', 'exit_code': 0, 'elapsed_seconds': .001,
              'cap_seconds': STAGE_CAPS[stage], 'within_cap': True,
              'artifact_path': str(path), 'artifact_sha256': hash_file(path)})


def fake_popen(code, seen, artifact=None):
    def popen(argv):
        seen.append(argv)
        def wait(timeout=None):
            if code == 0 and artifact is not None:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text('{}')
            return code
        return SimpleNamespace(pid=2**30, wait=wait)
    return popen


def test_supervisor_verifies_pins_records_durable_start_and_end_and_builds_parseable_worker_argv(tmp_path):
    fx = bundle_fixture(tmp_path)
    seen = []
    outcome, code = supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=fake_popen(0, seen, fx.c1 / 'preparation.json'))
    assert (outcome, code) == ('completed', 0)
    assert seen[0][1].endswith('run_ml_confirmation.py') and seen[0][-1] == 'prepare' and str(fx.c1) in seen[0]
    state = supervisor.ledger_state(fx.ledger, {'c1': 7200, 'f1ext': 7200})
    assert len(state['jobs']) == 1 and state['jobs'][0]['ended'] and state['jobs'][0]['cap_seconds'] == 7200
    assert state['arms']['c1']['charged_seconds'] > 0 and state['arms']['f1ext']['charged_seconds'] == 0.
    start = read_json(next((fx.ledger / 'jobs').glob('*.json')))
    end = read_json(fx.ledger / 'ends' / (start['job_id'] + '.json'))
    assert start['arm'] == 'c1' and start['stage'] == 'prepare' and end['plan']['remaining_obligation_seconds'] == 0.
    with pytest.raises(RuntimeError, match='never overwrite'):
        supervisor.write_end(fx.ledger, start['job_id'], 'completed', 0, 1., 7200, '')
    tampered = read_json(fx.bundle); tampered['sources']['scripts/score_ml_confirmation.py'] = '0' * 64; dump(fx.bundle, tampered)
    with pytest.raises(ValueError, match='bundle identity'):
        supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=fake_popen(0, seen))
    for command, seed in (('c1-fit', 3), ('f1-predict', 4), ('profile-masked', None), ('c1-score', None), ('f1-score', None), ('freeze-third-parent', None)):
        argv = supervisor.worker_argv(fx.bundle, fx.local, fx.config, read_json(fx.local), command, seed)
        if 'run_ml_five_seed_extension.py' in argv[1] or 'score_ml_five_seed_extension.py' in argv[1]:
            parser = runner.build_parser() if 'run_ml' in argv[1] else None
            if parser:
                args = parser.parse_args(argv[2:]); assert args.bundle == fx.bundle
    assert supervisor.build_parser().parse_args(['--bundle', 'b', '--local-config', 'l', 'c1-predict', '--seed', '4']).seed == 4
    with pytest.raises(SystemExit):
        supervisor.build_parser().parse_args(['--bundle', 'b', '--local-config', 'l', 'c1-fit', '--seed', '2'])


def test_supervisor_enforces_frozen_sequence_and_records_refusals(tmp_path):
    fx = bundle_fixture(tmp_path)
    outcome, code = supervisor.supervise(fx.bundle, fx.local, 'c1-fit', 3, popen=fake_popen(0, []))
    assert (outcome, code) == ('refused', supervisor.EXIT_REFUSED)
    jobs = supervisor.ledger_jobs(fx.ledger)
    assert jobs[0]['outcome'] == 'refused' and jobs[0]['exit_code'] == 2 and jobs[0]['ended']
    reason = read_json(fx.ledger / 'ends' / (jobs[0]['job_id'] + '.json'))['note']
    assert 'c1-prepare:None' in reason and 'profile-masked:None' in reason
    complete(fx, ('c1-prepare', None))
    assert supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=fake_popen(0, []))[0] == 'refused'
    assert 'already complete' in read_json(fx.ledger / 'ends' / (supervisor.ledger_jobs(fx.ledger)[1]['job_id'] + '.json'))['note']
    # All ten predictions are required before C1 score: masked seed 4 missing is a refusal, never a partial score.
    complete(fx, *[(c, s) for c, _, _, s in SEQUENCE if c not in ('c1-score', 'f1-score') and (c, s) != ('f1-predict', 4)])
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-score', None, popen=fake_popen(0, []))
    assert outcome == 'refused' and 'f1-predict:4' in read_json(fx.ledger / 'ends' / (supervisor.ledger_jobs(fx.ledger)[2]['job_id'] + '.json'))['note']


def test_unresolved_start_reserves_cap_and_blocks_every_launch(tmp_path):
    fx = bundle_fixture(tmp_path)
    (fx.ledger / 'jobs').mkdir(parents=True)
    dump(fx.ledger / 'jobs' / 'x.json', {'protocol': supervisor.LEDGER_PROTOCOL, 'job_id': 'x', 'command': 'c1-prepare', 'arm': 'c1',
                                        'stage': 'prepare', 'seed': None, 'cap_seconds': 7200., 'argv': [],
                                        'bundle_sha256': canonical_hash(fx.config)})
    state = supervisor.ledger_state(fx.ledger, {'c1': 7200, 'f1ext': 7200})
    assert state['arms']['c1']['reserved_seconds'] == 7200. and state['arms']['c1']['remaining_seconds'] == 0.
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=fake_popen(0, []))
    assert outcome == 'refused'
    jobs = supervisor.ledger_jobs(fx.ledger)
    refused = next(j for j in jobs if j['outcome'] == 'refused')
    reason = read_json(fx.ledger / 'jobs' / (refused['job_id'] + '.json'))['reason']
    assert 'fail closed' in reason or 'Unresolved' in reason
    with pytest.raises(ValueError, match='Foreign ledger'):
        dump(fx.ledger / 'jobs' / 'y.json', {'protocol': 'other'}); supervisor.ledger_jobs(fx.ledger)


def test_ledger_skips_only_verified_appledouble_and_fresh_output_preserves_unknown_files(tmp_path):
    fx = bundle_fixture(tmp_path)
    (fx.ledger / 'jobs').mkdir(parents=True)
    (fx.ledger / 'jobs' / '._sidecar.json').write_bytes(APPLEDOUBLE_MAGIC + b'\x00' * 20)
    assert supervisor.ledger_jobs(fx.ledger) == []
    (fx.ledger / 'jobs' / '._notreally.json').write_text('not json')
    with pytest.raises(json.JSONDecodeError):
        supervisor.ledger_jobs(fx.ledger)
    out = tmp_path / 'out'; out.mkdir()
    (out / '._meta').write_bytes(APPLEDOUBLE_MAGIC + b'\x00' * 20)
    runner._fresh(out)
    (out / '._payload').write_bytes(b'ordinary payload with sidecar-like name')
    with pytest.raises(ValueError, match='Preserve existing'):
        runner._fresh(out)
    (out / '._payload').unlink(); (out / 'third_parent').mkdir(); (out / 'third_parent' / 'manifest.json').write_text('{}')
    runner._fresh(out, allowed=('third_parent/manifest.json',))
    with pytest.raises(ValueError, match='Preserve existing'):
        runner._fresh(out)


def test_supervisor_reserves_effective_cap_from_remaining_share_and_kills_process_group_on_timeout(tmp_path):
    fx = bundle_fixture(tmp_path, grace=.4)
    (fx.ledger / 'jobs').mkdir(parents=True); (fx.ledger / 'ends').mkdir()
    # Earlier spending leaves 2.0 s in the C1 share; the prepare cap must shrink to that remainder.
    dump(fx.ledger / 'jobs' / 'spent.json', {'protocol': supervisor.LEDGER_PROTOCOL, 'job_id': 'spent', 'command': 'c1-prepare', 'arm': 'c1',
                                            'stage': 'prepare', 'seed': None, 'cap_seconds': 7200., 'argv': [],
                                            'bundle_sha256': canonical_hash(fx.config)})
    dump(fx.ledger / 'ends' / 'spent.json', {'job_id': 'spent', 'outcome': 'failed', 'exit_code': 1,
                                            'elapsed_seconds': 7198., 'cap_seconds': 7200., 'within_cap': True})
    pid_file = tmp_path / 'grandchild.pid'
    program = (f"import subprocess, time, pathlib; child = subprocess.Popen(['sleep', '60']); "
               f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); time.sleep(60)")
    popen = lambda argv: subprocess.Popen([sys.executable, '-c', program], start_new_session=True)
    outcome, code = supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=popen)
    assert outcome == 'timeout' and code in (-15, -9)
    jobs = supervisor.ledger_jobs(fx.ledger)
    mine = [j for j in jobs if j['job_id'] != 'spent'][0]
    assert mine['cap_seconds'] == pytest.approx(2.) and mine['ended'] and mine['outcome'] == 'timeout'
    assert 1.2 <= mine['elapsed_seconds'] < 8.
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 3.
    alive = True
    while alive and time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0); time.sleep(.05)
        except ProcessLookupError:
            alive = False
    assert not alive, 'grandchild in the worker process group must not survive the cap'
    # The share is exhausted afterwards: the next launch is refused, not silently transferred from the other arm.
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=fake_popen(0, []))
    assert outcome == 'refused'
    assert supervisor.ledger_state(fx.ledger, {'c1': 7200, 'f1ext': 7200})['arms']['f1ext']['charged_seconds'] == 0.


def test_supervisor_interrupt_after_launch_reaps_child_before_ending_job(tmp_path):
    fx = bundle_fixture(tmp_path, grace=.4)

    class Child:
        def __init__(self):
            self.process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
            self.pid = self.process.pid
            self.first = True

        def wait(self, timeout=None):
            if self.first:
                self.first = False
                raise KeyboardInterrupt()
            return self.process.wait(timeout=timeout)

    launched = []
    def launch(argv):
        child = Child(); launched.append(child); return child
    with pytest.raises(KeyboardInterrupt):
        supervisor.supervise(fx.bundle, fx.local, 'c1-prepare', None, popen=launch)
    assert launched[0].process.poll() is not None
    jobs = supervisor.ledger_jobs(fx.ledger)
    assert len(jobs) == 1 and jobs[0]['ended'] and jobs[0]['outcome'] == 'failed'


def profiles_fixture(fx, *, fit_seconds=2.):
    """Write both matched profiles, the C1 mandatory profile and its loader evidence with synthetic timings."""
    local = read_json(fx.local)
    expected = runner.identity(fx.ext_config, fx.local, fx.config)
    measured = {'command_overhead_seconds': 2., 'load_seconds': 4., 'fit_seconds': fit_seconds, 'calibration_seconds': .2,
                'inference_seconds': .1, 'scoring_probe_seconds': 3.}
    samples = {'train': 1252824, 'earlystop': 16000, 'temperature': 2603, 'blend': 4821, 'dev': 12334}
    for arm, name in (('c1', 'full'), ('f1ext', 'masked')):
        folder = runner.profile_dir(fx.config, local, name); folder.mkdir(parents=True, exist_ok=True)
        dump(folder / 'profile.json', {'arm': name, 'measured': measured, 'projection': project_costs(measured, samples, arm=arm)})
        dump(folder / 'state.json', {'identity': expected, 'arm': name, 'artifact_hashes': {'profile.json': hash_file(folder / 'profile.json')}})
    profile = fx.c1 / 'profile'; profile.mkdir(parents=True, exist_ok=True)
    dump(fx.c1 / 'preparation.json', {'synthetic': True})
    dump(profile / 'profile.json', {'rough_per_seed_fit_projection_seconds': 300., 'parent_full_cpanel_prediction_seconds': {'G0-global': [25., 26., 25.]}})
    dump(profile / 'state.json', {'preparation_sha256': hash_file(fx.c1 / 'preparation.json'), 'artifact_hashes': {'profile.json': hash_file(profile / 'profile.json')}})
    evidence = fx.ledger / 'profiles' / 'c1_mandatory_loader.json'
    dump(evidence, {'c1_profile_state_sha256': hash_file(profile / 'state.json')})
    return expected


def test_fit_launch_requires_all_profiles_and_gates_the_remaining_obligation(tmp_path):
    fx = bundle_fixture(tmp_path)
    local = read_json(fx.local)
    complete(fx, ('c1-prepare', None), ('c1-profile', None))
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-fit', 3, popen=fake_popen(0, []))
    assert outcome == 'refused'  # matched profiles missing (sequence)
    profiles_fixture(fx)
    complete(fx, ('c1-prepare', None), ('c1-profile', None), ('profile-full', None), ('profile-masked', None))
    seen = []
    fit_artifact = fx.c1 / 'fits' / 'seed3' / 'global' / 'state.json'
    outcome, code = supervisor.supervise(fx.bundle, fx.local, 'c1-fit', 3, popen=fake_popen(0, seen, fit_artifact))
    assert (outcome, code) == ('completed', 0) and seen[0][-3:] == ['c1-fit', '--seed', '3']
    successful = next(j for j in supervisor.ledger_jobs(fx.ledger) if j['command'] == 'c1-fit' and j['outcome'] == 'completed')
    plan = read_json(fx.ledger / 'ends' / (successful['job_id'] + '.json'))['plan']
    assert plan['stage'] == 'fit' and 7199. < plan['cap_seconds'] <= 7200. and plan['projected_command_seconds'] > 0
    assert plan['remaining_obligation_seconds'] == pytest.approx(arm_obligation(read_json(runner.profile_dir(fx.config, local, 'full') / 'profile.json')['projection'], set(), 'c1'))
    # A completed fit cannot run again under the same output identity.
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-fit', 3, popen=fake_popen(0, []))
    refused = [j for j in supervisor.ledger_jobs(fx.ledger) if j['command'] == 'c1-fit' and j['outcome'] == 'refused'][-1]
    assert outcome == 'refused' and 'already complete' in read_json(fx.ledger / 'ends' / (refused['job_id'] + '.json'))['note']
    # Without restricted-loader evidence, the first fit is refused.
    fx_gate = bundle_fixture(tmp_path / 'gate')
    profiles_fixture(fx_gate)
    complete(fx_gate, ('c1-prepare', None), ('c1-profile', None), ('profile-full', None), ('profile-masked', None))
    (fx_gate.ledger / 'profiles' / 'c1_mandatory_loader.json').unlink()
    outcome, _ = supervisor.supervise(fx_gate.bundle, fx_gate.local, 'c1-fit', 3, popen=fake_popen(0, []))
    refused = [j for j in supervisor.ledger_jobs(fx_gate.ledger) if j['outcome'] == 'refused'][-1]
    assert outcome == 'refused' and 'restricted-loader' in read_json(fx_gate.ledger / 'ends' / (refused['job_id'] + '.json'))['note']
    # A projection whose four-fit obligation exceeds the share is refused before any fit.
    fx2 = bundle_fixture(tmp_path / 'two')
    complete(fx2, ('c1-prepare', None), ('c1-profile', None))
    profiles_fixture(fx2, fit_seconds=15.)
    complete(fx2, ('c1-prepare', None), ('c1-profile', None), ('profile-full', None), ('profile-masked', None))
    outcome, _ = supervisor.supervise(fx2.bundle, fx2.local, 'c1-fit', 3, popen=fake_popen(0, []))
    refused = [j for j in supervisor.ledger_jobs(fx2.ledger) if j['outcome'] == 'refused'][-1]
    reason = read_json(fx2.ledger / 'ends' / (refused['job_id'] + '.json'))['note']
    assert outcome == 'refused' and ('exceeds' in reason or 'obligation' in reason)


def test_predict_cap_is_limited_by_member_fit_wall_and_profile_only_needs_its_full_cap(tmp_path):
    fx = bundle_fixture(tmp_path)
    complete(fx, ('c1-prepare', None), ('c1-profile', None), ('c1-fit', 3), ('c1-fit', 4))
    profiles_fixture(fx)
    complete(fx, ('c1-prepare', None), ('c1-profile', None), ('profile-full', None), ('profile-masked', None))
    fit3_end = fx.ledger / 'ends' / 'synthetic-c1-fit-seed3.json'
    dump(fit3_end, {**read_json(fit3_end), 'elapsed_seconds': 6000.})
    seen = []
    prediction = fx.c1 / 'members' / 'G0-global' / 'seed3' / 'prediction_state.json'
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-predict', 3, popen=fake_popen(0, seen, prediction))
    assert outcome == 'completed' and seen[0][-5:] == ['predict', '--cell', 'G0-global', '--seed', '3']
    predicted = next(j for j in supervisor.ledger_jobs(fx.ledger) if j['command'] == 'c1-predict' and j['outcome'] == 'completed')
    plan = read_json(fx.ledger / 'ends' / (predicted['job_id'] + '.json'))['plan']
    assert 1199. < plan['cap_seconds'] < 1200.  # 7200 member limit minus fit wall and charged predecessors
    fit4_end = fx.ledger / 'ends' / 'synthetic-c1-fit-seed4.json'
    dump(fit4_end, {**read_json(fit4_end), 'elapsed_seconds': 7190.})
    outcome, _ = supervisor.supervise(fx.bundle, fx.local, 'c1-predict', 4, popen=fake_popen(0, []))
    assert outcome == 'refused'
    fx3 = bundle_fixture(tmp_path / 'three')
    complete(fx3, ('c1-prepare', None), ('c1-profile', None))
    prep_end = fx3.ledger / 'ends' / 'synthetic-c1-prepare-seedNone.json'
    dump(prep_end, {**read_json(prep_end), 'elapsed_seconds': 6700.})
    assert supervisor.supervise(fx3.bundle, fx3.local, 'profile-full', None, popen=fake_popen(0, []))[0] == 'refused'
    dump(prep_end, {**read_json(prep_end), 'elapsed_seconds': 6500.})
    seen = []
    artifact = fx3.ledger / 'profiles' / 'full' / 'state.json'
    assert supervisor.supervise(fx3.bundle, fx3.local, 'profile-full', None, popen=fake_popen(0, seen, artifact))[0] == 'completed'
    assert seen[0][-3:] == ['profile', '--arm', 'full']
    assert [read_json(p) for p in sorted((fx3.ledger / 'jobs').glob('*.json'))][-1]['cap_seconds'] == 600.


# ---------------------------------------------------------------- ten-prediction gate with a synthetic F1 parent

def test_ten_prediction_gate_rejects_missing_masked_seed_four_and_tampered_states(tmp_path, monkeypatch):
    fx = bundle_fixture(tmp_path)
    local = read_json(fx.local)
    old = Path(fx.ext_config['parent_f1']['run']); old.mkdir(parents=True)
    old_prep = {'samples': {'train': {'rows_sha256': 'a' * 64}}, 'synthetic': True}
    dump(old / 'preparation.json', old_prep)
    ext = deepcopy(fx.ext_config); ext['parent_f1']['preparation_sha256'] = hash_file(old / 'preparation.json')
    for seed in (0, 1, 2):
        dest = bridge.member_dir(old, seed); dest.mkdir(parents=True)
        for name in ('fit.json', 'model.pt', 'predictions.npz', 'calibration.json', 'prediction_runtime.json'):
            (dest / name).write_bytes(f'{name}{seed}'.encode())
        dump(dest / 'fit_state.json', {'identity': bridge.member_identity(old_prep, seed), 'artifact_hashes': {'model.pt': hash_file(dest / 'model.pt'), 'fit.json': hash_file(dest / 'fit.json')}})
        dump(dest / 'prediction_state.json', {'identity': bridge.member_identity(old_prep, seed), 'fit_state_sha256': hash_file(dest / 'fit_state.json'),
                                              'artifact_hashes': {'predictions.npz': hash_file(dest / 'predictions.npz')}})
    manifest_body = {'fits': {str(s): {'state.json': 'f' * 64} for s in (3, 4)}, 'members': {str(s): {} for s in (3, 4)}, 'c1_preparation_sha256': 'e' * 64}
    prep = {'samples': {'train': {'rows_sha256': 'a' * 64}}, 'parent_f1_run': str(old), 'third_parent': {'manifest': manifest_body}}
    monkeypatch.setattr(runner, 'verify', lambda output, expected: prep)
    monkeypatch.setattr(runner, 'verify_c1_parent', lambda *a, **k: {'run': fx.c1, 'prep': {}, 'body': manifest_body})
    monkeypatch.setattr(runner.c1, 'resolve_member', lambda *a, **k: {'hashes': {}, 'directory': 'x', 'predictions_path': 'x'})
    with pytest.raises(ValueError, match=r'incomplete.*\[3, 4\]'):
        runner.verify_ten_predictions(fx.config, ext, fx.c1_config, fx.local, local)
    for seed in (3, 4):
        dest = runner.member_dir(fx.ext, seed); dest.mkdir(parents=True)
        for name in ('fit.json', 'model.pt', 'predictions.npz', 'calibration.json', 'prediction_runtime.json'):
            (dest / name).write_bytes(f'new{name}{seed}'.encode())
        identity = ext_module.extension_member_identity(prep, seed, c1_fit_state_sha256='f' * 64)
        dump(dest / 'fit_state.json', {'identity': identity, 'artifact_hashes': {'model.pt': hash_file(dest / 'model.pt')}})
        if seed == 3:
            dump(dest / 'prediction_state.json', {'identity': identity, 'fit_state_sha256': hash_file(dest / 'fit_state.json'),
                                                  'artifact_hashes': {'predictions.npz': hash_file(dest / 'predictions.npz')}})
    with pytest.raises(ValueError, match=r'incomplete.*\[4\]'):
        runner.verify_ten_predictions(fx.config, ext, fx.c1_config, fx.local, local)
    dest = runner.member_dir(fx.ext, 4)
    dump(dest / 'prediction_state.json', {'identity': ext_module.extension_member_identity(prep, 4, c1_fit_state_sha256='f' * 64),
                                          'fit_state_sha256': hash_file(dest / 'fit_state.json'), 'artifact_hashes': {'predictions.npz': hash_file(dest / 'predictions.npz')}})
    ten = runner.verify_ten_predictions(fx.config, ext, fx.c1_config, fx.local, local)
    assert set(ten['masked']) == {0, 1, 2, 3, 4} and set(ten['full']) == {0, 1, 2, 3, 4}
    (dest / 'predictions.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Artifact identity changed'):
        runner.verify_ten_predictions(fx.config, ext, fx.c1_config, fx.local, local)
    (dest / 'predictions.npz').write_bytes(b'newpredictions.npz4')
    wrong = dict(manifest_body, fits={str(s): {'state.json': '9' * 64} for s in (3, 4)})
    monkeypatch.setattr(runner, 'verify_c1_parent', lambda *a, **k: {'run': fx.c1, 'prep': {}, 'body': wrong})
    with pytest.raises(ValueError, match='changed after preparation'):
        runner.verify_ten_predictions(fx.config, ext, fx.c1_config, fx.local, local)


def test_load_configs_cross_binds_bundle_c1_and_extension(tmp_path):
    fx = bundle_fixture(tmp_path)
    bundle = read_json(fx.bundle)
    c1_config, ext = runner.load_configs(bundle)
    assert ext['parent_c1']['config_sha256'] == canonical_hash(c1_config)
    broken = deepcopy(fx.c1_config); broken['registration']['output'] = str(fx.protocol / 'elsewhere')
    dump(tmp_path / 'EXP-P10-001.yaml', broken)
    with pytest.raises(ValueError, match='config identity differs'):
        runner.load_configs(bundle)
    dump(tmp_path / 'EXP-P10-001.yaml', fx.c1_config)
    bad = deepcopy(fx.ext_config); bad['parent_g']['preparation_sha256'] = '1' * 64
    dump(tmp_path / 'EXP-P9-002.yaml', bad)
    with pytest.raises(ValueError, match='same frozen G parent'):
        runner.load_configs(bundle)
