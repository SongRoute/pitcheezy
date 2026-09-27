"""Synthetic review regressions; no frozen artifacts or real-data fits are opened."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(Path(__file__).parent)]
import run_ml_five_seed_extension as runner
from bridge_synthetic import synthetic_family
from score_ml_matrix import summarize_cell


def test_declared_c1_id_mismatch_fails_before_artifact_access(monkeypatch):
    config = {'experiment_id': 'C1'}
    extension = {'parent_c1': {'experiment_id': 'wrong'}}
    bundle = {'configs': {'c1': {'path': 'c1'}, 'f1ext': {'path': 'ext'}},
              'arms': {'c1': {'experiment_id': 'C1'}}}
    monkeypatch.setattr(runner, 'read_json', lambda path: config if path.name == 'c1' else extension)
    monkeypatch.setattr(runner, 'c1_validate_config', lambda value: value)
    monkeypatch.setattr(runner, 'validate_c1_config_shape', lambda value: value)
    monkeypatch.setattr(runner, 'validate_extension_config', lambda value, **kwargs: value)
    with pytest.raises(ValueError, match='Declared C1 experiment identity'):
        runner.load_configs(bundle)


@pytest.fixture
def reference_replay(monkeypatch, tmp_path):
    full, _, baseline = synthetic_family(games=4, per_game=8)
    report, values = summarize_cell(full, baseline)
    g_run, g_analysis, f1_analysis = tmp_path / 'g', tmp_path / 'g/analysis', tmp_path / 'f1/analysis'
    g = {'run': g_run, 'analysis': g_analysis,
         'reuse': [{'member_dir': str(tmp_path / f'g/member{seed}')} for seed in range(3)]}
    old = {'analysis': f1_analysis}
    metadata = {name: baseline['dev_' + name] for name in ('keys', 'y', 'game_pk', 'pitcher')}
    archives = {g_run / 'baseline_predictions.npz': baseline,
                g_analysis / 'predictions.npz': {**metadata, **{'G0-global_' + k: v for k, v in values.items()}},
                f1_analysis / 'predictions.npz': {'full_' + k: v.copy() for k, v in values.items()},
                **{Path(item['member_dir']) / 'predictions.npz': member for item, member in zip(g['reuse'], full)}}
    reports = {g_analysis / 'results.json': {'reports': {'G0-global': report}},
               f1_analysis / 'results.json': {'reports': {'full': deepcopy(report)}}}
    calls = []
    monkeypatch.setattr(runner, 'verify_g_parent', lambda *args: g)
    def old_verify(*args):
        calls.append('masked_replay')
        return old
    monkeypatch.setattr(runner, 'verify_f1_parent', old_verify)
    monkeypatch.setattr(runner, 'archive', lambda path: archives[path])
    monkeypatch.setattr(runner, 'read_json', lambda path: reports[path])
    return g, old, archives, reports, calls


def test_reuse_replay_checks_g_and_f1_full_reference(reference_replay):
    g, old, archives, reports, calls = reference_replay
    assert runner.verify_reused_references({}, Path('local'), {}) == (g, old)
    assert calls == ['masked_replay']
    archives[old['analysis'] / 'predictions.npz']['full_primary'][0, 0] += .001
    with pytest.raises(ValueError, match='F1 full-arm three-seed reconstruction'):
        runner.verify_reused_references({}, Path('local'), {})


def test_g_reference_drift_stops_before_masked_replay(reference_replay):
    g, old, archives, reports, calls = reference_replay
    stored = archives[g['analysis'] / 'predictions.npz']
    stored['G0-global_primary'] = stored['G0-global_primary'].copy()
    stored['G0-global_primary'][0, 0] += .001
    with pytest.raises(ValueError, match='G0 prediction reconstruction'):
        runner.verify_reused_references({}, Path('local'), {})
    assert calls == []


def test_full_summary_drift_is_rejected(reference_replay):
    _, old, _, reports, _ = reference_replay
    reports[old['analysis'] / 'results.json']['reports']['full']['primary']['log_loss'] += .001
    with pytest.raises(ValueError, match='full-arm three-seed summary'):
        runner.verify_reused_references({}, Path('local'), {})


@pytest.mark.parametrize('broken', ['profile-full', 'profile-masked', 'replay', None])
def test_c1_fit_waits_for_both_profiles_and_both_reference_replays(monkeypatch, tmp_path, broken):
    events = []
    monkeypatch.setattr(runner, 'identity', lambda *args: {})
    def profile(*args):
        stage = 'profile-' + args[2]
        events.append(stage)
        if broken == stage:
            raise ValueError(stage)
    def replay(*args):
        events.append('replay')
        if broken == 'replay':
            raise ValueError('replay')
    monkeypatch.setattr(runner, 'load_profile', profile)
    monkeypatch.setattr(runner, 'verify_reused_references', replay)
    monkeypatch.setattr(runner, 'arm_output', lambda *args: tmp_path)
    monkeypatch.setattr(runner.c1, 'identity', lambda *args: {})
    monkeypatch.setattr(runner.c1, 'verify', lambda *args: {})
    monkeypatch.setattr(runner.c1, 'fit', lambda *args: events.append('fit'))
    if broken:
        with pytest.raises(ValueError, match=broken):
            runner.c1_fit({}, {}, {}, Path('local'), {}, 3)
        assert 'fit' not in events
    else:
        runner.c1_fit({}, {}, {}, Path('local'), {}, 3)
        assert events == ['profile-full', 'profile-masked', 'replay', 'fit']


def test_c1_fit_cli_permits_only_two_new_seed_ids():
    parser = runner.build_parser()
    args = parser.parse_args(['--bundle', 'bundle', '--local-config', 'local', 'c1-fit', '--seed', '4'])
    assert (args.command, args.seed) == ('c1-fit', 4)
    with pytest.raises(SystemExit):
        parser.parse_args(['--bundle', 'bundle', '--local-config', 'local', 'c1-fit', '--seed', '0'])
