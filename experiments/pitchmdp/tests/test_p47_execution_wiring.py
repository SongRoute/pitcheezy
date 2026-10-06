"""P47 execution wiring tests use generated records and deny non-TRAIN payload access."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'scripts'), str(PROJECT / 'docs/launch')]
import p47_probe as probe
import p47_probe_child as child
import make_p47_probe_spec
import p47_launch as launch
import run_p47_pipeline as pipeline
import run_cuda_direct as runner
from pitchmdp import direct_pitcher_id_registry as pid


def draft():
    return json.loads(launch.UNDATED.read_text())


def probe_result(task):
    full, arm = task['full'], task['arm']
    vocabs = {c: {'n_vocab': (920 if c == 'pitcher' else 706) if full else 2,
                  'ids_sha256': c + ('full' if full else 'smoke'), 'train_max_date': 20250430}
              for c in ('pitcher', 'batter')}
    block = None if arm == pid.REFERENCE else {'id_columns': ['pitcher', 'batter'], 'vocabs': vocabs,
             'index_source': 'identity', 'shuffle_seed': None,
             **({'id_dropout': pid.ID_DROPOUT} if arm != pid.PITCHER_BATTER else {}),
             **({'freeze_unk': True, 'unk_rows_exact_zero': {'pitcher': True, 'batter': True}}
                if arm == pid.PITCHER_BATTER_ZERO_UNK else {})}
    return {'protocol': probe.PROTOCOL, 'arm': arm, 'seed': task['seed'], 'full_train': full,
            'runtime': {'bound_uuid': pid.RETAINED_GPU, 'cuda_visible_devices': pid.RETAINED_GPU},
            'splits_loaded': ['train', 'earlystop'], 'train_only': True, 'dev_used_for_fit_or_selection': False,
            'epochs_run': 2, 'training_rows': 9000 if full else 2, 'earlystop_rows': 2,
            'selected_rows_expected': {'train': 9000 if full else 2, 'earlystop': 2},
            'source_sha256': {'model': 'a'}, 'export_manifest_sha256': 'manifest',
            'sidecar_sha256': 'sidecar', 'fit_seconds': 1.0, 'peak_device_bytes': 123,
            'train_inputs_sha256': 'a', 'earlystop_inputs_sha256': 'b', 'train_keys_sha256': 'c',
            'earlystop_keys_sha256': 'd', 'state_dict_sha256': 'e', 'id_record': block,
            'verified_payload_sha256': {s: {'export': {'keys': s}, 'batter': s}
                                        for s in ('train', 'earlystop')}}


def run_record():
    return {'completed': 17, 'max_live': 1, 'foreign_gpu_apps': [], 'wall_seconds': 123,
            'tasks': [{'task': t, 'returncode': 0, 'cuda_visible_devices': pid.RETAINED_GPU,
                       'result': probe_result(t)} for t in probe.tasks()]}


def spec():
    return {'pins': {'source_sha256': {'model': 'a'}, 'export_manifest_sha256': 'manifest',
                     'sidecar_sha256': 'sidecar', 'config_sha256': launch.UNDATED_SHA,
                     'probe_code_sha256': {'probe': 'a'}}}


def test_probe_checks_all_seventeen_and_rejects_partial_or_wrong_mode():
    assert [t['arm'] for t in probe.tasks()[:4]] == list(pid.P47_ARM_ORDER)
    assert probe.assess(spec(), run_record())['valid']
    for change in (lambda r: r['tasks'].pop(),
                   lambda r: r['tasks'][0]['result'].update(splits_loaded=['train', 'dev']),
                   lambda r: r['tasks'][-1]['result']['id_record']['vocabs']['batter'].update(n_vocab=705),
                   lambda r: r['tasks'][-1]['result']['id_record']['unk_rows_exact_zero'].update(batter=False),
                   lambda r: r.update(max_live=2)):
        bad = run_record()
        change(bad)
        assert not probe.assess(spec(), bad)['valid']


def test_frozen_probe_spec_has_only_gpu3_serial_and_absent_registration_paths(tmp_path):
    value = make_p47_probe_spec.build(launch.UNDATED, tmp_path / 'export', tmp_path / 'probe-001',
                                      '/venv/bin/python', 'ledger-audit', 'distinct-review')
    assert probe.check_spec(value) == value
    assert value['tasks'] == probe.tasks() and len(value['tasks']) == 17
    assert value['gpu_uuid'] == pid.RETAINED_GPU and value['concurrent_slots'] == 1
    assert value['lock_dir'] == probe.LOCK_DIR and value['absent_before']
    for change in (lambda s: s.update(concurrent_slots=2),
                   lambda s: s['tasks'].pop(),
                   lambda s: s['pins'].update(probe_code_sha256={}),
                   lambda s: s.update(absent_before=[])):
        bad = copy.deepcopy(value)
        change(bad)
        with pytest.raises(ValueError):
            probe.check_spec(bad)


def test_gpu_probe_is_uuid_scoped(monkeypatch):
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        return type('Completed', (), {'stdout': ''})()
    monkeypatch.setattr(probe.subprocess, 'run', fake_run)
    assert probe.gpu3_apps() == []
    assert commands == [['nvidia-smi', f'--id={pid.RETAINED_GPU}', '--query-compute-apps=gpu_uuid,pid',
                         '--format=csv,noheader,nounits']]


def test_probe_child_denies_non_train_payload_hash_and_load(tmp_path, monkeypatch):
    config_path = tmp_path / 'draft.json'
    config_path.write_text(json.dumps(draft()))
    source_map = {'model': 'fixed'}
    monkeypatch.setattr(runner, 'own_hashes', lambda: source_map)
    monkeypatch.setattr(runner, 'bind_gpu', lambda *_: {'bound_uuid': pid.RETAINED_GPU})
    monkeypatch.setattr(runner, 'needs_context_names', lambda _: False)
    monkeypatch.setattr(runner, 'train_subset', lambda cfg, arm, arrays: (arrays, None))
    monkeypatch.setattr(runner, 'pitcher_vocab', lambda *_: None)
    monkeypatch.setattr(runner, 'arm_inputs', lambda *_: (np.zeros((2, 1)), np.zeros((2, 1)), np.zeros((2, 1))))
    monkeypatch.setattr(runner, 'build_model', lambda *_: object())
    monkeypatch.setattr(runner, 'peak_device_bytes', lambda _: 4)
    monkeypatch.setattr(child.bench, 'fit_and_record', lambda *_, **kw: {'seed': kw['seed'], 'epochs_run': 2})
    manifest = {'splits': {s: {'files': {}} for s in ('train', 'earlystop', 'temperature', 'blend', 'dev', 'mlb_dev')}}
    sidecar = {'splits': {s: {'path': s + '.npy', 'sha256': s} for s in manifest['splits']}}
    monkeypatch.setattr(child.bench, 'verify_train_only', lambda *args: manifest)
    monkeypatch.setattr(runner, 'batter_sidecar', lambda *args, **kw: (tmp_path, sidecar))
    real_sha = runner.sha256_file
    seen = []

    def hash_only_allowed(path):
        if Path(path) == config_path:
            return real_sha(path)
        name = Path(path).stem
        assert name in ('train', 'earlystop'), f'forbidden payload hash: {path}'
        seen.append(('hash', name))
        return name

    def load_only_allowed(export, manifest, split, limit):
        assert split in ('train', 'earlystop'), f'forbidden payload load: {split}'
        seen.append(('export', split))
        return {'keys': np.array([1, 2]), 'y': np.array([0, 1])}

    def sidecar_only_allowed(directory, record, split, limit):
        assert split in ('train', 'earlystop'), f'forbidden sidecar load: {split}'
        seen.append(('sidecar', split))
        return np.array([3, 4])

    monkeypatch.setattr(runner, 'sha256_file', hash_only_allowed)
    monkeypatch.setattr(runner, 'load_split', load_only_allowed)
    monkeypatch.setattr(child.bs, 'load', sidecar_only_allowed)
    monkeypatch.setattr(child.np, 'load', lambda *args, **kw: pytest.fail('np.load must be selected via load_split'))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', pid.RETAINED_GPU)
    value = child.fit_task({'protocol': child.PROTOCOL, 'gpu_uuid': pid.RETAINED_GPU,
                            'config': str(config_path), 'export': str(tmp_path),
                            'pins': {'config_sha256': real_sha(config_path), 'source_sha256': source_map,
                                     'export_manifest_sha256': 'manifest'},
                            'workload': {'train_rows': 8192, 'earlystop_rows': 2048}}, pid.REFERENCE, 80, False)
    assert value['splits_loaded'] == ['train', 'earlystop'] and len(seen) == 6
    assert {split for _, split in seen} == {'train', 'earlystop'}


def test_selected_payload_verification_denies_other_opens_and_np_load(tmp_path, monkeypatch):
    manifest = {'splits': {}}
    sidecar = {'splits': {}}
    for split in ('train', 'earlystop', 'temperature', 'blend', 'dev', 'mlb_dev'):
        files = {}
        for name, data in {'keys': np.array([1, 2], np.int64), 'y': np.array([0, 1], np.int64),
                           'date': np.array([20250401, 20250402], np.int64)}.items():
            path = f'{split}-{name}.npy'
            np.save(tmp_path / path, data)
            files[name] = {'path': path, 'shape': list(data.shape), 'dtype': str(data.dtype),
                           'sha256': runner.sha256_file(tmp_path / path)}
        manifest['splits'][split] = {'n': 2, 'rows_sha256': hashlib.sha256(np.array([1, 2], np.int64).tobytes()).hexdigest(),
                                     'files': files}
        path = f'{split}-batter.npy'
        np.save(tmp_path / path, np.array([3, 4], np.int64))
        sidecar['splits'][split] = {'n': 2, 'path': path, 'sha256': runner.sha256_file(tmp_path / path)}
    def metadata(export, config, pin, hash_arrays=True):
        assert hash_arrays is False
        return manifest
    def sidecar_metadata(export, config, record, hash_arrays=True):
        assert hash_arrays is False
        return tmp_path, sidecar
    monkeypatch.setattr(runner, 'verify_export', metadata)
    monkeypatch.setattr(runner, 'batter_sidecar', sidecar_metadata)
    actual_hash, actual_load = runner.sha256_file, np.load
    accessed = []
    def guarded_hash(path):
        split = Path(path).name.split('-')[0]
        assert split in ('train', 'earlystop'), f'forbidden hash/open {path}'
        accessed.append(('hash', split))
        return actual_hash(path)
    def guarded_load(path, *args, **kwargs):
        split = Path(path).name.split('-')[0]
        assert split in ('train', 'earlystop'), f'forbidden np.load {path}'
        accessed.append(('load', split))
        return actual_load(path, *args, **kwargs)
    monkeypatch.setattr(runner, 'sha256_file', guarded_hash)
    monkeypatch.setattr(np, 'load', guarded_load)
    selected, directory, selected_sidecar = child.load_train_only({'export': str(tmp_path),
             'pins': {'export_manifest_sha256': 'manifest'}}, draft())
    for split in ('train', 'earlystop'):
        runner.load_split(tmp_path, selected, split)
        child.bs.load(directory, selected_sidecar, split)
    assert len(accessed) == 2 * (3 + 1 + 3 + 1)
    assert {split for _, split in accessed} == {'train', 'earlystop'}


def test_legacy_smoke_refused_and_dated_registration_requires_budget_and_ledger(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='P47 legacy'):
        runner.require_executable(draft(), smoke=True)
    check = {'protocol': probe.PROTOCOL, 'valid': True, 'errors': [], 'config_sha256': launch.UNDATED_SHA,
             'tasks': 17, 'bound_uuids': [pid.RETAINED_GPU], 'preflight_seconds_measured': 123,
             'source_sha256': {'source': 'a'}, 'export_manifest_sha256': launch.MANIFEST_SHA,
             'sidecar_sha256': draft()[pid.BATTER_SIDECAR_KEY]['sidecar_sha256'],
             'probe_code_sha256': {'probe': 'b'}}
    value = launch.dated(launch.UNDATED.read_text(), '2026-10-06', '10:00', check)
    text = json.dumps(value)
    assert launch.dated_check(text, check) == value
    q = value['queue']
    assert launch.at(q['date'], q['aggregate_by']) - launch.at(q['date'], q['launch_by']) >= \
           launch.timedelta(seconds=4894 + 900 + 1200 + 123)
    monkeypatch.setattr(runner, 'own_hashes', lambda: check['source_sha256'])
    go = tmp_path / 'go.txt'; go.write_text('accepted')
    ledger = tmp_path / 'ledger.json'; ledger.write_text(json.dumps({'experiment_id': 'EXP-P47-001',
                  'seeds': pid.P47_SEEDS, 'reserved': True, 'checked_kst': '2026-10-06T10:00:00+09:00'}))
    registered = launch.build_registered(text, check, 'a' * 64, '10:00', 'abcdef0', str(go), str(ledger),
                                         launch.at('2026-10-06', '10:00'))
    assert registered['status'] == 'REGISTERED' and registered['registration']['seed_id_ledger_review']['sha256']
    monkeypatch.setattr(runner, 'verify_export', lambda *args, **kwargs: pytest.fail('payload touched in identity gate'))
    check_bytes = b'check bytes'
    registered['registration']['preflight']['check_sha256'] = hashlib.sha256(check_bytes).hexdigest()
    pipeline.preflight(registered, check, text.encode(), json.dumps(registered).encode(), check_bytes, 'abcdef0')
    with pytest.raises(ValueError, match='ledger'):
        launch.build_registered(text, check, 'a' * 64, '10:00', 'abcdef0', str(go), 'missing',
                                launch.at('2026-10-06', '10:00'))
    bad = copy.deepcopy(value); bad['queue']['aggregate_by'] = '10:05'
    with pytest.raises(ValueError):
        launch.dated_check(json.dumps(bad), check)


def test_pipeline_seal_gate_requires_all_sixteen(monkeypatch, tmp_path):
    config = draft(); seen = []
    monkeypatch.setattr(runner, 'verify_task', lambda dest, expected: seen.append((dest, expected)))
    monkeypatch.setattr(runner, 'expected_identity', lambda *args: args)
    assert pipeline.sealed_family(config, tmp_path, 'manifest', {'source': 'a'}) == 16
    assert len(seen) == 16
    config['seeds'] = [80, 81]
    with pytest.raises(ValueError, match='arm/seed'):
        pipeline.sealed_family(config, tmp_path, 'manifest', {'source': 'a'})
