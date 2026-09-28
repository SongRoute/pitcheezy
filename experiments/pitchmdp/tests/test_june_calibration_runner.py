"""End-to-end synthetic runs of every ML-JUNE-CALIBRATION-v1 stage through ``run()``.

Only git provenance is disabled (temporary parents are not a registration
checkout) and the native-library check is stubbed; every other pin, stage
order, fresh-output, manifest and failure rule is the production code path.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import june_calibration_synthetic as syn  # noqa: E402
from pitchmdp import matrix_june_calibration as mj  # noqa: E402
from pitchmdp.matrix_policy_artifacts import APPLEDOUBLE_MAGIC  # noqa: E402
import run_ml_june_calibration as runner  # noqa: E402


@pytest.fixture(scope='module')
def family(tmp_path_factory):
    return syn.build(tmp_path_factory.mktemp('june-family'))


@pytest.fixture
def patched(family, monkeypatch):
    for name, values in family['constants'].items():
        for key, value in values.items():
            monkeypatch.setitem(getattr(mj, name), key, value)
    monkeypatch.setattr(mj, 'DRAWS', syn.DRAWS)
    monkeypatch.setattr(mj, 'PROFILE_ROWS', syn.PROFILE_ROWS)
    monkeypatch.setattr(mj, 'PROBE_ROWS', syn.PROBE_ROWS)
    monkeypatch.setattr(runner, 'NETWORK', family['network'])
    monkeypatch.setattr(runner, 'validate_native_runtime', lambda: None)
    return family


def run(config, output, stage, seed=None):
    return runner.run(config, output, stage, seed, git=False)


def manifest(output, rel):
    return json.loads((output / rel / 'manifest.json').read_text())


def complete_family(config, output):
    run(config, output, 'prepare')
    run(config, output, 'profile')
    for seed in mj.SEEDS:
        run(config, output, 'predict', seed)
    run(config, output, 'fit')
    run(config, output, 'apply')
    return run(config, output, 'score')


def test_all_stages_complete_with_sealed_manifests_and_additive_contrasts(patched):
    config, output = syn.write_configs(patched, runner, 'attempt-complete')
    summary = complete_family(config, output)
    assert summary == {'contrast_additivity_passed': True}
    prepared = manifest(output, 'prepared')
    assert prepared['stage'] == 'prepare' and prepared['status'] == 'complete'
    assert prepared['config_sha256'] == syn.sha(config) and prepared['registered_code_commit'] == '0' * 40
    assert set(prepared['outputs']) == {'june_store.pkl', 'june_population.npz', 'june_labels.npz',
                                        'june_frequency.npz', 'prepared.json'}
    report = json.loads((output / 'prepared' / 'prepared.json').read_text())
    assert report['store_date_range'][1] <= '2025-06-30' and report['decoded_cache_date_max'] > '2025-06-30'
    assert all(r['passed'] for r in report['cpanel_reports'])
    profile = json.loads((output / 'profile' / 'profile.json').read_text())
    assert profile['rows'] == syn.PROFILE_ROWS and profile['probe_rows'] == syn.PROBE_ROWS
    assert profile['labels_read'] is False and profile['profile_probabilities_discarded'] is True
    assert profile['same_store_as_full_member'] == prepared['store_digest']
    for seed in mj.SEEDS:
        member = manifest(output, f'members/seed{seed}')
        assert member['seed'] == seed and member['store_digest'] == prepared['store_digest']
        runtime = json.loads((output / 'members' / f'seed{seed}' / 'prediction_runtime.json').read_text())
        assert all(r['passed'] for r in runtime['cpanel_replay'])  # truncated store reproduces full-season archive
        with np.load(output / 'members' / f'seed{seed}' / 'predictions.npz') as saved:
            assert set(saved.files) == set(runner.MEMBER_FIELDS) and int(saved['seed']) == seed
    fit = manifest(output, 'fit')
    assert fit['call_counts'] == {'replay': 6, 'candidate': 30, 'total': 36} and fit['dev_scores'] is None
    parameters = json.loads((output / 'fit' / 'parameters.json').read_text())
    assert [c['kind'] for c in parameters['call_ledger']['calls']] == ['replay'] * 6 + ['candidate'] * 30
    assert parameters['dev_separation']['dev_keys_in_fit'] == 0
    application = json.loads((output / 'apply' / 'application.json').read_text())
    assert application['label_archive_reference']['labels_decoded'] is False
    assert all(r['passed'] for r in application['b0_replay'])
    analysis = manifest(output, 'analysis')
    assert analysis['contrast_additivity_passed'] is True and analysis['R_family_size'] == 78
    assert analysis['independent_confirmation'] is None and analysis['held_out_confirmation'] is False
    assert analysis['policy_effect'] is None and analysis['service_adoption'] is None
    results = json.loads((output / 'analysis' / 'results.json').read_text())
    assert list(results['N']) == ['B1-B0', 'B2-B0', 'B2-B1'] and len(results['R']['slots']) == 78
    assert set(analysis['upstream_manifests']) == {'prepared/manifest.json', 'profile/manifest.json', 'fit/manifest.json',
                                                   'apply/manifest.json',
                                                   *(f'members/seed{s}/manifest.json' for s in mj.SEEDS)}


def test_stage_order_is_enforced_and_failures_leave_no_manifest(patched):
    config, output = syn.write_configs(patched, runner, 'attempt-order')
    with pytest.raises(ValueError, match='prepare must complete'):
        run(config, output, 'fit')
    assert list((output / 'failures').glob('fit-*.json'))
    run(config, output, 'prepare')
    with pytest.raises(ValueError, match='Completed profile'):
        run(config, output, 'predict', 0)
    assert (output / 'members' / 'seed0' / 'failure.json').is_file()
    assert not (output / 'members' / 'seed0' / 'manifest.json').exists()
    with pytest.raises(ValueError, match='exists; preserve'):
        run(config, output, 'predict', 0)  # no retry into a failed stage directory
    with pytest.raises(ValueError, match='Completed profile'):
        run(config, output, 'score')  # score demands every upstream stage, profile first
    assert not (output / 'analysis' / 'manifest.json').exists()
    with pytest.raises(ValueError, match='already has stage content'):
        run(config, output, 'prepare')


def test_fit_failure_preserves_call_ledger_and_blocks_apply(patched, monkeypatch):
    config, output = syn.write_configs(patched, runner, 'attempt-fit-failure')
    run(config, output, 'prepare')
    run(config, output, 'profile')
    for seed in mj.SEEDS:
        run(config, output, 'predict', seed)
    original = mj.fit_scalar_blend
    seen = []

    def injected(*args, **kwargs):
        seen.append(kwargs['label'])
        if len(seen) == 9:
            def refuse(fun, **options):
                fun(0.5)
                from scipy.optimize import OptimizeResult
                return OptimizeResult(x=0.5, fun=fun(0.5), success=False, status=1, nfev=2, nit=1, message='injected')
            kwargs['minimizer'] = refuse
        return original(*args, **kwargs)
    monkeypatch.setattr(mj, 'fit_scalar_blend', injected)
    with pytest.raises(mj.FamilyStop, match='scipy_success_false'):
        run(config, output, 'fit')
    failure = json.loads((output / 'fit' / 'failure.json').read_text())
    assert failure['optimizer_call_ledger']['total_calls'] == 9
    assert (failure['optimizer_call_ledger']['replay_calls'], failure['optimizer_call_ledger']['candidate_calls']) == (6, 3)
    assert failure['retry'] is False and not (output / 'fit' / 'manifest.json').exists()
    monkeypatch.setattr(mj, 'fit_scalar_blend', original)
    with pytest.raises(ValueError, match='Completed fit'):
        run(config, output, 'apply')


def test_score_waits_for_the_complete_application(patched):
    config, output = syn.write_configs(patched, runner, 'attempt-score-early')
    run(config, output, 'prepare')
    run(config, output, 'profile')
    for seed in mj.SEEDS:
        run(config, output, 'predict', seed)
    run(config, output, 'fit')
    with pytest.raises(ValueError, match='Completed apply'):
        run(config, output, 'score')  # both candidates and all five seeds must be applied first
    assert (output / 'analysis' / 'failure.json').is_file() and not (output / 'analysis' / 'manifest.json').exists()


def test_tampered_member_output_and_changed_parent_are_refused(patched):
    config, output = syn.write_configs(patched, runner, 'attempt-tamper')
    run(config, output, 'prepare')
    run(config, output, 'profile')
    for seed in mj.SEEDS:
        run(config, output, 'predict', seed)
    path = output / 'members' / 'seed3' / 'prediction_runtime.json'
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='predict output changed'):
        run(config, output, 'fit')
    path.write_bytes(original)
    lock = patched['lock']
    with lock.open('rb') as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another heavy job'):
            run(config, output, 'apply')
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)


def test_pins_are_checked_before_any_value(patched, tmp_path):
    config, output = syn.write_configs(patched, runner, 'attempt-pins',
                                       exec_changes=lambda c: c['data']['processed_pitches'].update(sha256='f' * 64))
    with pytest.raises(ValueError, match='data.processed_pitches'):
        run(config, output, 'prepare')
    assert not (output / 'prepared').exists()
    config, output = syn.write_configs(patched, runner, 'attempt-env',
                                       exec_changes=lambda c: c['environment'].update(numpy='0.0'))
    with pytest.raises(ValueError, match='numpy'):
        run(config, output, 'prepare')
    config, output = syn.write_configs(patched, runner, 'attempt-sources',
                                       exec_changes=lambda c: c['source_hashes'].pop(
                                           str((runner.PROJECT / 'pitchmdp/matrix_june_calibration.py').resolve())))
    with pytest.raises(ValueError, match='lacks required'):
        run(config, output, 'prepare')
    config, output = syn.write_configs(patched, runner, 'attempt-disabled', exec_changes=lambda c: c.update(enabled=False))
    with pytest.raises(ValueError, match='not enabled'):
        run(config, output, 'prepare')
    config, output = syn.write_configs(patched, runner, 'attempt-c1', sci_changes=lambda s: s['B0_provenance']
                                       ['recorded_c1_versions'].update(scipy='1.0.0'))
    with pytest.raises(ValueError, match='recorded C1'):
        run(config, output, 'prepare')
    config, output = syn.write_configs(patched, runner, 'attempt-output')
    with pytest.raises(ValueError, match='registered absolute output_dir'):
        runner.run(config, output.parent / 'elsewhere', 'prepare', git=False)


def test_dev_rows_can_never_enter_a_june_fit(patched):
    config, output = syn.write_configs(patched, runner, 'attempt-dev')
    ctx = runner.verify_everything(json.loads(config.read_text()), config, output, git=False)
    frame = patched['full_frame']
    june = frame.loc[frame.split.eq('blend')].iloc[:20]
    dev = frame.loc[frame.split.eq('dev')].iloc[:1]
    population = {'june_keys': june[runner.KEY].to_numpy(np.int64), 'june_game_pk': june.game_pk.to_numpy(np.int64)}
    assert runner.dev_separation(ctx, population, june.game_date)['dev_keys_in_fit'] == 0
    leaked = {'june_keys': np.vstack([population['june_keys'], dev[runner.KEY].to_numpy(np.int64)]),
              'june_game_pk': np.r_[population['june_game_pk'], dev.game_pk.to_numpy(np.int64)]}
    with pytest.raises(ValueError, match='DEV pitch key'):
        runner.dev_separation(ctx, leaked, june.game_date)
    with pytest.raises(ValueError, match='not separated'):
        runner.dev_separation(ctx, population, dev.game_date)


def test_execution_schema_rejects_drafts():
    good = {'protocol': mj.EXECUTION_PROTOCOL, 'family_id': mj.FAMILY_ID, 'enabled': True, 'code_commit_c': 'a' * 40,
            'scientific_config': {'path': '/x/s.json', 'sha256': 'b' * 64}, 'contract': {'path': '/x/c.md', 'sha256': 'c' * 64},
            'source_hashes': {'/x/a.py': 'd' * 64}, 'environment': {k: 'v' for k in runner.ENV_KEYS},
            'data': {k: {'path': f'/x/{k}', 'sha256': 'e' * 64} for k in runner.DATA_KEYS},
            'output_dir': '/x/out', 'heavy_lock': '/x/.heavy.lock'}
    assert runner.validate_execution_config(good) is good
    for mutate, message in ((lambda c: c.update(code_commit_c='abc'), '40-hex'),
                            (lambda c: c['scientific_config'].update(path='rel.json'), 'absolute'),
                            (lambda c: c.update(extra=1), 'exactly'),
                            (lambda c: c['data'].pop('physics_cache'), 'data must pin'),
                            (lambda c: c['environment'].pop('scipy'), 'environment must pin'),
                            (lambda c: c.update(output_dir='out'), 'absolute')):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        with pytest.raises(ValueError, match=message):
            runner.validate_execution_config(bad)


def test_fresh_outputs_exclusive_writes_and_appledouble(tmp_path):
    target = runner.claim_stage_dir(tmp_path, 'fit')
    runner.write_json_exclusive(target / 'a.json', {'x': 1})
    with pytest.raises(FileExistsError):
        runner.write_json_exclusive(target / 'a.json', {'x': 2})
    assert json.loads((target / 'a.json').read_text()) == {'x': 1}
    (target / '._a.json').write_bytes(APPLEDOUBLE_MAGIC + b'\0' * 60)
    (target / '._fake').write_bytes(b'not appledouble')
    assert runner.content_files(target) == {'a.json', '._fake'}
    with pytest.raises(ValueError, match='exists; preserve'):
        runner.claim_stage_dir(tmp_path, 'fit')
    with pytest.raises(ValueError, match='--seed'):
        runner.claim_stage_dir(tmp_path, 'predict')
