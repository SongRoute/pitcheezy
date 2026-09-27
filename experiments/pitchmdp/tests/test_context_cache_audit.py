"""Cache audit runner: synthetic CPU guards only; no real data, MPS or fits of registered size."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'scripts'))
from pitchmdp.data import hash_file
from pitchmdp.matrix_long_history import MatrixLongHistoryStore, LazyPitchBatch
from pitchmdp.matrix_observed_context_cache import FrozenObservedContext
from pitchmdp.matrix_sharing import ContinuousPitcherContext
from pitchmdp.sequence_model import SequenceContext
import pitchmdp.matrix_context_cache_audit as audit
import audit_ml_context_cache as runner
from test_observed_context_cache import fixture
from test_lazy_delivery_reuse import FrozenSyntheticPool

COUNTS = {'train': 24, 'train_evaluation': 6, 'earlystop': 8, 'temperature': 3}
SCRIPT = PROJECT/'scripts'/'audit_ml_context_cache.py'


def batches(length):
    """Seventy chronological synthetic rows: five date-shifted copies of the shared 14-row fixture."""
    base_frame, _ = fixture()
    copies = []
    for k in range(5):
        part = base_frame.copy()
        part['game_date'] = part['game_date']+pd.Timedelta(days=10*k); part['game_pk'] = part['game_pk']+1000*k
        copies.append(part)
    frame = pd.concat(copies, ignore_index=True)
    frame['split'] = 'train'; frame.loc[40:, 'split'] = 'earlystop'; frame.loc[60:, 'split'] = 'temperature'
    clusters = {'columns': [f'profile{i}' for i in range(23)], 'pitcher_cluster': {'11': 0},
                'pitcher_profiles': {'11': np.arange(23).tolist()}, 'pitcher_counts': {'11': len(frame)}}
    base = ContinuousPitcherContext(SequenceContext().fit(frame.loc[frame.split.eq('train')]), clusters)
    store = MatrixLongHistoryStore.from_frame(frame, long_length=length)
    return {name: LazyPitchBatch(store, base, np.flatnonzero(frame.split.eq(name).to_numpy()))
            for name in ('train', 'earlystop', 'temperature')}


def draft():
    return json.loads((PROJECT.parents[1]/'configs'/'EXP-P4-002-v3-cache-audit-v2.yaml').read_text())


def test_draft_config_accepted_only_as_draft_and_no_relaxation_or_reduction():
    config = draft()
    assert runner.config_check(config, real=False)['device'] == 'mps'
    with pytest.raises(ValueError, match='Null identity pin'): runner.config_check(config, real=True)
    assert set(config['sources']) == set(runner.SOURCES)
    for mutate in [lambda c: c['stage2']['tolerances']['gradients'].update(atol=1e-3),
                   lambda c: c['stage3']['tolerances']['probability'].update(atol=1e-3),
                   lambda c: c['samples']['train'].update(n=8192),
                   lambda c: c['stage3']['budget'].update(epochs=1),
                   lambda c: c.update(device='cpu'),
                   lambda c: c.update(arms=['F4-H0']),
                   lambda c: c['limits']['command_seconds'].update(stage3=9999),
                   lambda c: c.update(adoption=True),
                   lambda c: c.update(extra_axis=1),
                   lambda c: c['samples'].update(dev_or_june_cache='allowed')]:
        changed = deepcopy(config); mutate(changed)
        with pytest.raises(ValueError): runner.config_check(changed, real=False)
    frozen = deepcopy(config); frozen['repo_commit'] = 'a'*40; frozen['prior_cost_ledger'] = {'path': '/x', 'sha256': 'b'*64}
    frozen['sources'] = {name: 'c'*64 for name in runner.SOURCES}
    assert runner.config_check(frozen, real=True)


def test_cli_help_works():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and 'stage3' in result.stdout and 'summary' in result.stdout
    common = ['--config', 'c', '--local-config', 'l', '--output', 'o']
    args = runner.build_parser().parse_args(['--worker', *common, 'stage3', '--arm', 'F4-H0', '--path', 'cached', '--attempt', '2'])
    assert args.worker and args.command == 'stage3' and args.path == 'cached' and args.attempt == 2
    with pytest.raises(SystemExit): runner.build_parser().parse_args([*common, 'stage3', '--arm', 'F4-H0', '--path', 'cached', '--attempt', '2', '--worker'])
    with pytest.raises(SystemExit): runner.build_parser().parse_args([*common, 'stage3', '--arm', 'F4-H0', '--attempt', '1'])


def test_selectors_reject_dev_and_track_exact_identity():
    parts = batches(0)
    selected, warm = audit.select_samples(parts, COUNTS, warmup_train=8)
    records = audit.sample_records(selected)
    assert records['train']['n'] == 24 and len(warm) == 8
    np.testing.assert_array_equal(selected['train_evaluation'].rows, warm.rows[np.linspace(0, 7, 6, dtype=np.int64)])
    audit.check_sample_identity(records, records)
    with pytest.raises(ValueError, match='selector identity'):
        audit.check_sample_identity(records, {**records, 'train': {'n': 24, 'rows_sha256': 'x'*64}})
    with pytest.raises(ValueError, match='DEV'):
        audit.select_samples({**parts, 'dev': parts['train']}, COUNTS, warmup_train=8)
    with pytest.raises(ValueError, match='do not shrink'):
        audit.select_samples(parts, {**COUNTS, 'train': 1000}, warmup_train=8)


class Tampered(FrozenObservedContext):
    def transform(self, frame):
        values = super().transform(frame)
        if len(values): values = values.copy(); values[0, 0] += np.float32(1e-7)
        return values


@pytest.mark.parametrize('length', [0, 128])
def test_stage1_bitwise_guards_timing_and_rejects_unequal_cache(length, monkeypatch):
    selected, _ = audit.select_samples(batches(length), COUNTS, warmup_train=8)
    result = audit.stage1_context(selected, chunk_size=5, repetitions=2, uneven_chunk_sizes=(1, 4), timing_batch_size=4)
    assert result['all_bitwise_identical'] and result['all_guards_rejected']
    assert set(result['checks']) >= set(audit.STAGE1_ORDERS) and result['checks']['uneven_chunks']['chunks'] == 10
    assert all(result['must_fail'][name]['rejected'] for name in audit.STAGE1_MUST_FAIL)
    assert result['construction']['rng_state_unchanged'] and result['construction']['construction_rows'] == len(np.union1d(selected['train'].rows, selected['train_evaluation'].rows))
    assert [len(v) for kind in result['timing']['seconds'].values() for v in kind.values()] == [2]*4
    assert [row[2] for row in result['timing']['schedule'][:4]] == ['original', 'cached', 'cached', 'original']
    assert result['long_length'] == length and result['dev_scores_read'] is False
    monkeypatch.setattr(audit, 'FrozenObservedContext', Tampered)
    with pytest.raises(ValueError, match='differs bitwise'):
        audit.stage1_context(selected, chunk_size=5, repetitions=2, uneven_chunk_sizes=(1, 4), timing_batch_size=4)


def test_stage2_equivalence_rng_and_unequal_context_rejected(monkeypatch):
    selected, _ = audit.select_samples(batches(32), COUNTS, warmup_train=8)
    torch.set_num_threads(1)
    numpy_state = np.random.get_state()[1].copy()
    result = audit.stage2_backend(selected['train'], device='cpu', minibatches=2, batch_size=4, width=8, chunk_size=3)
    assert result['equivalent'] and result['identical_initialization'] and result['rng_state_unchanged_by_steps']
    assert len(result['steps']) == 2 and result['tolerances'] == audit.STAGE2_TOLERANCES
    assert all(step['updated_weights']['max_absolute_difference'] == 0. for step in result['steps'])
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state)
    with pytest.raises(ValueError, match='exceeds the fixed TRAIN sample'):
        audit.stage2_backend(selected['train'], device='cpu', minibatches=7, batch_size=4, width=8)
    monkeypatch.setattr(audit, 'FrozenObservedContext', Tampered)
    with pytest.raises(ValueError, match='differs bitwise'):
        audit.stage2_backend(selected['train'], device='cpu', minibatches=1, batch_size=4, width=8)


def test_difference_rejects_nonfinite_and_beyond_tolerance():
    assert audit._difference([1., 2.], [1., 2.+5e-7], {'atol': 1e-6, 'rtol': 0.})['pass']
    assert not audit._difference([1., 2.], [1., 2.+5e-6], {'atol': 1e-6, 'rtol': 0.})['pass']
    assert not audit._difference([1., np.nan], [1., np.nan], {'atol': 1e-6, 'rtol': 0.})['pass']
    with pytest.raises(ValueError, match='shape'): audit._difference([1.], [1., 2.], {'atol': 0, 'rtol': 0})


def test_stage3_paired_fit_compare_and_perturbed_outputs_rejected():
    selected, warm = audit.select_samples(batches(32), COUNTS, warmup_train=8)
    torch.set_num_threads(1)
    budget = {'epochs': 2, 'patience': 2, 'batch_size': 4, 'learning_rate': .0005}
    original, oa = audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget)
    rows = np.unique(np.concatenate([warm.rows, *[b.rows for b in selected.values()]]))
    cache, construction = audit.build_cache(selected['train'].context, selected['train'].store.frame, rows, chunk_size=7)
    cached, ca = audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget, cache=cache, path='cached')
    assert construction['construction_rows'] == len(rows) and construction['rng_state_unchanged']
    assert original['profile_weights_reused_as_full_member'] is False and 'log_loss' not in original
    comparison = audit.compare_stage3(original, oa, cached, ca)
    assert comparison['equivalent'] and comparison['selected_state']['max_absolute_difference'] == 0.
    assert comparison['history']['pass'] and comparison['fallback_tiers_equal']
    with pytest.raises(ValueError, match='in that order'): audit.compare_stage3(cached, ca, original, oa)
    with pytest.raises(ValueError, match='Path label'):
        audit.stage3_fit(selected, warm, FrozenSyntheticPool(), device='cpu', width=8, budget=budget, cache=cache)
    perturbed = deepcopy(ca); perturbed['probability'][0, 0] += 1e-3
    assert not audit.compare_stage3(original, oa, cached, perturbed)['equivalent']
    perturbed = deepcopy(ca); key = next(iter(perturbed['state'])); perturbed['state'][key] = perturbed['state'][key]+1e-4
    assert not audit.compare_stage3(original, oa, cached, perturbed)['equivalent']
    drifted = deepcopy(cached); drifted['history'][0]['earlystop_conditional_nll'] += 1e-4
    assert not audit.compare_stage3(original, oa, drifted, ca)['equivalent']
    shorter = deepcopy(cached); shorter['best_epoch'] = original['best_epoch']+1
    assert not audit.compare_stage3(original, oa, shorter, ca)['equivalent']


def test_fresh_output_allows_only_appledouble_and_sealing_detects_changes(tmp_path):
    target = tmp_path/'stage1'/'F4-H0'/'attempt1'
    runner.fresh(target)
    (target/'._results.json').write_bytes(bytes.fromhex('00051607')+b'meta')
    runner.fresh(target)
    (target/'results.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh attempt'): runner.fresh(target)
    identity = runner.stage_identity('reg', 'stage1', 'F4-H0', None, 1)
    runner.seal(target, identity)
    manifest = json.loads((target/'manifest.json').read_text())
    assert set(manifest['artifact_hashes']) == {'results.json'}
    assert runner.sealed(target, identity) == hash_file(target/'manifest.json')
    with pytest.raises(ValueError, match='identity'): runner.sealed(target, runner.stage_identity('other', 'stage1', 'F4-H0', None, 1))
    (target/'extra.json').write_text('{}')
    with pytest.raises(ValueError): runner.sealed(target, identity)
    (target/'extra.json').unlink(); (target/'results.json').write_text('{"changed": 1}')
    with pytest.raises(ValueError, match='Artifact identity changed'): runner.sealed(target, identity)


def test_completed_attempt_requires_exactly_one_sealed_attempt(tmp_path):
    with pytest.raises(ValueError, match='Exactly one sealed'): runner.completed_attempt(tmp_path, 'reg', 'stage3', 'F4-32', 'cached')
    for attempt in (1, 2):
        directory = runner.fresh(tmp_path/'stage3'/'F4-32'/'cached'/f'attempt{attempt}')
        (directory/'failure.json').write_text('{}')
    with pytest.raises(ValueError, match='found 0'): runner.completed_attempt(tmp_path, 'reg', 'stage3', 'F4-32', 'cached')
    directory = tmp_path/'stage3'/'F4-32'/'cached'/'attempt2'; (directory/'failure.json').unlink()
    (directory/'results.json').write_text('{}')
    runner.seal(directory, runner.stage_identity('reg', 'stage3', 'F4-32', 'cached', 2))
    found, digest = runner.completed_attempt(tmp_path, 'reg', 'stage3', 'F4-32', 'cached')
    assert found == directory and digest == hash_file(directory/'manifest.json')
    (tmp_path/'stage3'/'F4-32'/'cached'/'attempt1'/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='found 2'): runner.completed_attempt(tmp_path, 'reg', 'stage3', 'F4-32', 'cached')


def parent_fixture(tmp_path, config):
    root = tmp_path/'artifacts'; parent = root/'runs'/'ML-MATRIX-20260924'/'EXP-P4-002-v3'; parent.mkdir(parents=True)
    (parent/'source').mkdir(); (parent/'source'/'frozen.py').write_text('x')
    (parent/'features.json').write_text('{}')
    prep = {'identity': {'source_hashes': {}}, 'artifact_hashes': {'features.json': hash_file(parent/'features.json')},
            'features': {'long_stream': {'version': 'batter_dual_stream_v2'}}, 'dev_scores_read': False,
            'samples': {name: {'n': n} for name, n in [('train', 100000), ('earlystop', 5000), ('temperature', 640), ('blend', 1000), ('dev', 2000)]}}
    runner.dump(parent/'preparation.json', prep)
    repo = tmp_path/'repo'; (repo/'configs').mkdir(parents=True); (repo/'configs'/'EXP-P4-002-v3.yaml').write_text('{}')
    config['parent'].update(run=str(parent), config_sha256=hash_file(repo/'configs'/'EXP-P4-002-v3.yaml'),
                            preparation_sha256=hash_file(parent/'preparation.json'))
    for cell, length in runner.CELLS.items():
        directory = parent/'profiles'/cell; directory.mkdir(parents=True)
        profile = {'long_length': length, 'dev_scores_read': False, 'projection': {'load_fit_temperature_prediction_seconds': 5000.},
                   'samples': {name: {'n': config['samples'][name]['n'], 'rows_sha256': config['samples'][name]['rows_sha256']} for name in ('train', 'earlystop', 'temperature')},
                   'warmup': {'samples': {'train': {'n': 8192, 'rows_sha256': 'w'*64},
                                          'evaluation_train': {'n': 2048, 'rows_sha256': config['samples']['train_evaluation']['rows_sha256']}}}}
        runner.dump(directory/'profile.json', profile)
        runner.dump(directory/'manifest.json', {'identity': {'preparation_sha256': config['parent']['preparation_sha256'], 'cell': cell, 'profile_version': 'f4_resource_profile_v2'},
                                               'artifact_hashes': {'profile.json': hash_file(directory/'profile.json')}})
        config['parent']['profiles'][cell] = {'profile_sha256': hash_file(directory/'profile.json'), 'manifest_sha256': hash_file(directory/'manifest.json')}
    return {'artifact_root': str(root)}, repo, parent


def test_verify_parent_binds_preparation_profiles_and_registered_selectors(tmp_path):
    config = draft(); local, repo, parent = parent_fixture(tmp_path, config)
    _, prep, profiles = runner.verify_parent(config, local, repo=repo)
    assert set(profiles) == set(runner.CELLS) and prep['dev_scores_read'] is False
    wrong = deepcopy(config); wrong['parent']['preparation_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='preparation changed'): runner.verify_parent(wrong, local, repo=repo)
    wrong = deepcopy(config); wrong['samples']['train']['rows_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='selectors differ'): runner.verify_parent(wrong, local, repo=repo)
    wrong = deepcopy(config); wrong['parent']['config_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='configuration file changed'): runner.verify_parent(wrong, local, repo=repo)
    (parent/'profiles'/'F4-32'/'profile.json').write_text('{"tampered": true}')
    with pytest.raises(ValueError, match='profile evidence changed'): runner.verify_parent(config, local, repo=repo)


def test_prepare_refuses_existing_output_before_touching_parent(tmp_path, monkeypatch):
    output = tmp_path/'audit'; output.mkdir(); (output/'old.json').write_text('{}')
    monkeypatch.setattr(runner, 'verify_parent', lambda *a, **k: pytest.fail('Existing output must be refused first'))
    with pytest.raises(ValueError, match='Preserve the existing audit directory'): runner.prepare(draft(), {}, output, {})


def test_cost_summary_keeps_full_population_cache_unknown_and_lower_bound_honest():
    def result(fit):
        return {'warmup_seconds': 3., 'fit_seconds': fit, 'optimizer_updates': 1024, 'temperature_seconds': .2,
                'inference_seconds': .3, 'load_seconds': 2.}
    populations = {'train': 100000, 'earlystop': 3000, 'temperature': 640, 'blend': 1000, 'dev': 2000}
    profiles = {cell: {'projection': {'load_fit_temperature_prediction_seconds': 5000.}} for cell in runner.CELLS}
    stage3 = {cell: {'original': result(40.), 'cached': result(20.)} for cell in runner.CELLS}
    constructions = {cell: {'stage3_fit_process_rows': {'construction_seconds': 1.5, 'construction_rows': 77792}} for cell in runner.CELLS}
    summary = audit.cost_summary(profiles, stage3, constructions, populations, prior_seconds=1000., audit_seconds=500.)
    arm = summary['arms']['F4-H0']
    assert arm['cached_member_total_seconds'] is None and arm['full_population_cache_construction_seconds']['fit_process'] == 'unmeasured'
    assert summary['family_projection_seconds'] is None and summary['decision'] == 'not_adopted'
    cached = arm['audit_cached_projection_seconds_excluding_cache_construction']
    assert summary['family_lower_bound_seconds_excluding_cache_construction'] == pytest.approx(1500.+9*cached)
    assert arm['audit_original_projection_seconds'] > cached
    exceeded = audit.cost_summary(profiles, stage3, constructions, populations, prior_seconds=28000., audit_seconds=500.)
    assert not exceeded['family_lower_bound_within_budget'] and 'exceeds' in exceeded['reason']
    with pytest.raises(ValueError, match='update-count'):
        audit.cost_summary(profiles, {c: {p: {**r, 'optimizer_updates': 64} for p, r in v.items()} for c, v in stage3.items()},
                           constructions, populations, prior_seconds=0., audit_seconds=0.)
