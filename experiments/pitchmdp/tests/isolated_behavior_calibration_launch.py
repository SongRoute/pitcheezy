"""Registered preflight -> smoke through ``launch`` (review finding 1). The config is the committed
EXP-BC-CAL-01 contract in its REGISTERED form (every registered value unchanged, well-formed synthetic
pins, runner paths in a temporary directory); the preflight is the real PREFLIGHT_TESTS run, pointed at
that REGISTERED config (A3_CONFIG_PATH), so it shows the preflight never depends on the DRAFT status.
The smoke then runs on synthetic data through the production A3 readers; the synthetic B0 is not the
registered BC_P, so the smoke must end FAILED_INTEGRITY on exactly that check, with a terminal
final.json and no read past the TRAIN end. Not collected by default; run by
test_behavior_calibration_runner.test_registered_preflight_then_smoke_isolated in a fresh interpreter."""
from datetime import datetime
import json
import os
import sys

from behavior_calibration_synthetic import (NESTED, RUNNER_SHA256, SyntheticSource, as_registered,
                                            digest_entries_for, extra_rows, load_config, make_world)
import run_behavior_calibration as runner
from pitchmdp import behavior_calibration_registry as reg


def test_registered_preflight_passes_and_the_smoke_runs(tmp_path, monkeypatch):
    output_root = tmp_path / 'EXP-BC-CAL-01'
    config = as_registered(load_config(), runner={'output_root': str(output_root),
                                                 'heavy_lock': str(tmp_path / '.heavy.lock')})
    assert reg.config_check(config) is config and reg.registration_state(config) == 'REGISTERED'
    path = tmp_path / 'EXP-BC-CAL-01.registered.json'
    path.write_text(json.dumps(config))
    monkeypatch.setenv('A3_CONFIG_PATH', str(path))  # the preflight child reads the REGISTERED config
    monkeypatch.setenv(NESTED, '1')  # ... and does not start these isolated runs again
    source = SyntheticSource(make_world(0), None, tmp_path / 'data', extras=extra_rows())
    git = {'head': 'a' * 40, 'scope': list(reg.REVIEWED_CODE_SCOPE), 'dirty': False, 'commit_is_ancestor': True,
           'changed_since_commit': [], 'config_committed': True}
    env = runner.LaunchEnv(git_state=lambda commit, p=None: dict(git), runner_sha256=lambda: RUNNER_SHA256,
                           native_check=runner.native_and_dependencies,  # real OpenMP rule and research-a3 pins
                           memory_command=(sys.executable, '-c', 'print("System-wide memory free percentage: 80%")'),
                           preflight_command=runner.default_preflight_command,
                           preflight_env=lambda: dict(os.environ),  # keeps the two test switches set above
                           source_factory=lambda c, local, p, e: source,
                           now=lambda: datetime(2026, 10, 27, 15, 0, tzinfo=runner.KST), alarm=False,
                           digest_entries=digest_entries_for)
    destination = output_root / 'attempt-1'
    code = runner.launch(path, tmp_path / 'local.json', destination, 'smoke', env)
    preflight = json.loads((destination / 'preflight-smoke.json').read_text())
    assert preflight['tests']['returncode'] == 0, preflight
    assert 'passed' in preflight['tests']['summary'] and 'failed' not in preflight['tests']['summary']
    assert preflight['tests']['files'] == list(runner.PREFLIGHT_TESTS)
    assert preflight['tests']['switches'] == {'A3_CONFIG_PATH': str(path), NESTED: '1'}
    assert 'A3_CONFIG_PATH' not in runner.default_preflight_env() and NESTED not in runner.default_preflight_env()
    assert preflight['dependencies']['installed'] == reg.DEPENDENCY_PINS
    assert code == 2
    record = json.loads((runner.Attempt(destination).stage_path('smoke') / 'stage.json').read_text())
    assert record['status'] == 'FAILED_INTEGRITY' and 'b0_artifact_sha256_pinned' in record['error']['detail']
    smoke = json.loads((runner.Attempt(destination).stage_path('smoke') / 'smoke.json').read_text())
    failed = [name for name, ok in smoke['checks'].items() if not ok]
    assert failed == ['b0_artifact_sha256_pinned']  # the synthetic B0 is not the registered BC_P
    assert smoke['rows_decoded_after_train_end'] == 0 and smoke['metrics_computed'] is None
    reads = [e for e in record['reads'] if e['kind'] == 'read']
    assert reads and all(e['rows_at_or_after']['2025-05-01'] == 0 and not e['dev_bearing'] for e in reads)
    assert json.loads((runner.Attempt(destination).stage_path('smoke') / 'layout.json').read_text())['split_aligned']
    assert preflight['peak_child_tree_rss_bytes']['preflight'] > 0
    final = json.loads((destination / 'final.json').read_text())
    assert final['label'] == 'FAILED_INTEGRITY' and final['citable'] is False
    assert [r['event'] for r in runner.read_chain(destination / runner.BUDGET)] == ['start', 'end']
    assert source.revalidations == ['smoke'] and source.opens == 1
