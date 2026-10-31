"""Synthetic stage-3 runner boundaries; no real experimental data or scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitchmdp.data import KEY
from pitchmdp.matrix_g0_calibration import fixed_spec
from scripts import run_ml_g0_calibration as runner


def _hex(value='0'):
    return value * 64


def _config(tmp_path):
    spec = fixed_spec()
    spec['fit_population'] = {'ordered_keys_sha256': _hex('1'), 'labels_sha256': _hex('2')}
    return {'experiment_id': 'EXP-P11-002', 'output': str(tmp_path/'stage3'),
        'stage2': {'experiment_id': 'EXP-P11-001', 'run': str(tmp_path/'stage2'),
            'config_file': str(tmp_path/'stage2.json'), 'config_sha256': _hex('3'),
            'source_commit': '4'*40, 'required_files': list(runner.STAGE2_FILES)},
        'c1': {'run': str(tmp_path/'c1'), 'files': {name:_hex('7') for name in runner.C1_FILES}},
        'g': {'run': str(tmp_path/'g'), 'files': {name:_hex('8') for name in runner.G_FILES}},
        'calibration': spec, 'source_commit': '4'*40,
        'sources': {name: _hex('5') for name in runner.REQUIRED_SOURCES},
        'contract': {'path': 'docs/contracts/ML-G0-WHOLE-MLB-DRAFT-v1.md', 'sha256': _hex('6')},
        'budget_seconds': 3600, 'scoring':runner.SCORING}


def test_config_requires_two_stage_parent_and_disjoint_output(tmp_path):
    cfg = _config(tmp_path)
    assert runner.validate_config(cfg) is cfg
    cfg['stage2']['required_files'].remove('analysis/predictions.npz')
    with pytest.raises(ValueError, match='artifact inventory'):
        runner.validate_config(cfg)
    cfg = _config(tmp_path)
    cfg['output'] = cfg['stage2']['run'] + '/subdir'
    with pytest.raises(ValueError, match='disjoint'):
        runner.validate_config(cfg)
    cfg = _config(tmp_path)
    cfg['calibration']['fit_population']['labels_sha256'] = None
    with pytest.raises(ValueError, match='sha256'):
        runner.validate_config(cfg)
    cfg = _config(tmp_path)
    cfg['scoring'] = json.loads(json.dumps(runner.SCORING))
    cfg['scoring']['R']['family_size'] = 50
    with pytest.raises(ValueError, match='scoring settings'):
        runner.validate_config(cfg)


def _small_whole(tmp_path):
    cfg = _config(tmp_path)
    g = Path(cfg['g']['run']); stage2 = Path(cfg['stage2']['run'])
    g.mkdir(); (stage2/'analysis').mkdir(parents=True)
    n = 4
    keys = np.array([[1,1,1],[1,1,2],[2,1,1],[2,1,2]], dtype=np.int64)
    games = keys[:,0]
    pitchers = np.array([7,7,8,8], dtype=np.int64)
    uniform = np.full((n,10), .1, dtype=np.float64)
    # Deliberately omit DEV labels: application must remain label-blind.
    np.savez_compressed(stage2/'analysis/predictions.npz', keys=keys, game_pk=games,
        pitcher=pitchers, primary=uniform, calibrated=uniform, raw=uniform,
        seed_primary=np.stack([uniform]*5))
    np.savez_compressed(g/'baseline_predictions.npz', mlb_dev_keys=keys,
        mlb_dev_game_pk=games, mlb_dev_pitcher=pitchers, mlb_dev=uniform)
    pd.DataFrame({**{key:keys[:,i] for i,key in enumerate(KEY)}, 'pitcher':pitchers,
                  'train_volume':['low','low','zero','zero']}).to_parquet(g/'mlb_dev_metadata.parquet')
    (g/'panel.json').write_text(json.dumps({'train_players':[{'pitcher':7,'train_volume':'low'}]}))
    return cfg, keys


def test_apply_input_loader_does_not_open_dev_labels_and_uses_train_lookup(tmp_path):
    cfg, keys = _small_whole(tmp_path)
    saved, frequency, metadata, groups = runner._whole_inputs(cfg, include_labels=False)
    assert 'y' not in saved
    assert np.array_equal(saved['keys'], keys)
    assert groups.tolist() == ['low','low','zero','zero']
    assert frequency.shape == (4,10)
    metadata.loc[0, 'train_volume'] = 'high'
    metadata.to_parquet(Path(cfg['g']['run'])/'mlb_dev_metadata.parquet')
    with pytest.raises(ValueError, match='TRAIN-volume grouping'):
        runner._whole_inputs(cfg, include_labels=False)


def test_volume_activation_inputs_come_from_stage2_summaries_and_june_counts():
    june = {'groups':np.array(['zero','low','low','middle','high']),
            'game_pk':np.array([1,1,2,3,4])}
    names = ('zero','low','middle','high')
    result = {'R':{'groups':{}}, 'slices':{}}
    for name in names:
        result['R']['groups']['volume_'+name] = {'reporting': {'games':35,'n':600}}
        result['slices']['volume_'+name] = {'n':600,'g0':{'log_loss':1.2},
                                                'frequency':{'log_loss':1.1}}
    rows = runner._volume_diagnostics(result,june)
    assert [r['group'] for r in rows] == list(names)
    assert rows[0]['june_pitches'] == 1 and rows[0]['june_games'] == 1
    assert rows[1]['june_pitches'] == 2 and rows[1]['june_games'] == 2
    assert all(abs(r['dev_g0_minus_frequency_nll']-.1) < 1e-12 for r in rows)


def test_scoring_refuses_incomplete_active_candidate_before_metrics(tmp_path,monkeypatch):
    cfg = _config(tmp_path)
    output = Path(cfg['output']); output.mkdir()
    monkeypatch.setattr(runner,'verify_prepared',lambda *_: ({},{'I1':True,'I2':False}))
    monkeypatch.setattr(runner,'verify_fits',lambda *_: ({},{}))
    keys = np.array([[1,1,1]],dtype=np.int64)
    p = np.full((1,10),.1)
    saved = {'keys':keys,'y':np.array([0]),'game_pk':np.array([1]),
             'primary':p,'seed_primary':np.stack([p]*5)}
    monkeypatch.setattr(runner,'_whole_inputs',lambda *a,**k:(saved,p,pd.DataFrame(),np.array(['zero'])))
    monkeypatch.setattr(runner,'evaluate_candidates',lambda *a,**k:pytest.fail('scorer must not run'))
    (output/'preparation.json').write_text('{}')
    (output/'fits').mkdir(); (output/'fits/manifest.json').write_text('{}')
    with pytest.raises(FileNotFoundError):
        runner.score(cfg,tmp_path/'config.json')


def test_fixed_apply_command_records_two_inactive_slots_without_opening_dev(tmp_path,monkeypatch):
    cfg = _config(tmp_path)
    output = Path(cfg['output']); (output/'fits').mkdir(parents=True)
    (output/'activation.json').write_text('{}')
    (output/'fits/manifest.json').write_text('{}')
    inactive = {'I1':False,'I2':False}
    monkeypatch.setattr(runner,'verify_prepared',lambda *_: ({},inactive))
    monkeypatch.setattr(runner,'verify_fits',lambda *_: ({},{}))
    monkeypatch.setattr(runner,'_whole_inputs',lambda *a,**k:pytest.fail('inactive slots open DEV'))
    runner.apply(cfg,tmp_path/'config.json')
    manifest = runner.verify_applications(output,inactive)
    assert list(manifest['slots']) == ['I1','I2']
    assert all(item['status']=='not_activated' for item in manifest['slots'].values())
    assert not (output/'candidates').exists()
    with pytest.raises(ValueError,match='Preserve'):
        runner.apply(cfg,tmp_path/'config.json')


def test_june_reconstruction_uses_frozen_five_weights_and_pinned_labels(tmp_path):
    cfg = _config(tmp_path)
    g, c1 = Path(cfg['g']['run']), Path(cfg['c1']['run'])
    g.mkdir(); (c1/'analysis').mkdir(parents=True)
    n = 4821
    keys = np.column_stack((np.arange(1,n+1),np.ones(n,dtype=np.int64),np.ones(n,dtype=np.int64)))
    labels = np.zeros(n,dtype=np.int64)
    pitchers = np.full(n,7,dtype=np.int64)
    frequency = np.full((n,10),.1,dtype=np.float64)
    np.savez_compressed(c1/'parent_baseline_predictions.npz',blend_keys=keys,blend_y=labels,
        blend_game_pk=keys[:,0],blend_pitcher=pitchers,blend=frequency)
    pd.DataFrame({**{key:keys[:,i] for i,key in enumerate(KEY)},'pitcher':pitchers,
                  'train_volume':['low']*n}).to_parquet(g/'blend_metadata.parquet')
    (g/'panel.json').write_text(json.dumps({'train_players':[{'pitcher':7,'train_volume':'low'}]}))
    inputs = {}
    for seed in range(5):
        member = (g if seed < 3 else c1)/'members/G0-global'/f'seed{seed}'
        member.mkdir(parents=True)
        p = np.full((n,10),.1,dtype=np.float64)
        p[:,0] += .01*seed
        p[:,1] -= .01*seed
        path = member/'predictions.npz'
        np.savez_compressed(path,blend_keys=keys,blend_y=labels,blend_game_pk=keys[:,0],
                            blend_pitcher=pitchers,blend=p)
        inputs[str(path)] = runner.hash_file(path)
    (c1/'analysis/manifest.json').write_text(json.dumps({'inputs':inputs}))
    report = {'selection':{'model_weight':.75},
              'seeds':[{'blend_selection':{'model_weight':.5}} for _ in range(5)]}
    (c1/'analysis/results.json').write_text(json.dumps({'reports':{'G0-global':report}}))
    cfg['calibration']['fit_population'] = {
        'ordered_keys_sha256':runner.array_sha256(keys,np.int64),
        'labels_sha256':runner.array_sha256(labels,np.int64)}
    row = runner._june(cfg,{})
    assert row['primary'].shape == (n,10) and row['seed_primary'].shape == (5,n,10)
    assert row['groups'].tolist() == ['low']*n
    assert np.allclose(row['primary'][:,0],.1+.75*.02)
    cfg['calibration']['fit_population']['labels_sha256'] = _hex('f')
    with pytest.raises(ValueError,match='June population'):
        runner._june(cfg,{})


def test_mutating_cli_command_holds_shared_heavy_lock(tmp_path,monkeypatch):
    from contextlib import contextmanager
    cfg = _config(tmp_path)
    state = {'locked': False, 'called': False}
    monkeypatch.setattr(runner,'validate_config',lambda value:value)
    monkeypatch.setattr(runner,'read_json',lambda path:cfg if str(path).endswith('config.json') else {'artifact_root':str(tmp_path)})
    monkeypatch.setattr(runner,'check_location',lambda local,output:tmp_path)
    @contextmanager
    def lock(root):
        assert root == tmp_path
        state['locked'] = True
        try:yield
        finally:state['locked'] = False
    def fake_prepare(*args):
        assert state['locked']
        state['called'] = True
    monkeypatch.setattr(runner,'heavy_lock',lock)
    monkeypatch.setattr(runner,'prepare',fake_prepare)
    monkeypatch.setattr(runner.sys,'argv',['runner','--config',str(tmp_path/'config.json'),
        '--local-config',str(tmp_path/'local.json'),'prepare'])
    runner.main()
    assert state == {'locked':False,'called':True}


def test_fit_manifest_links_june_activation_and_rejects_seed_permutation(tmp_path):
    output = tmp_path/'stage3'; fits = output/'fits'; fits.mkdir(parents=True)
    (output/'activation.json').write_text('activation')
    (output/'june_inputs.npz').write_bytes(b'june')
    activation = {'I1':True,'I2':False}
    rows = [{'predictor':name,'fit_success':True,'optimizer_report':{'scipy_success':True}}
            for name in runner.PREDICTORS]
    (fits/'I1.json').write_text(json.dumps({'predictors':list(runner.PREDICTORS),'fits':rows}))
    manifest = {'status':'sealed_before_dev_apply','files':{'I1':runner.hash_file(fits/'I1.json')},
        'inactive_slots':['I2'], 'activation_sha256':runner.hash_file(output/'activation.json'),
        'june_inputs_sha256':runner.hash_file(output/'june_inputs.npz')}
    (fits/'manifest.json').write_text(json.dumps(manifest))
    assert list(runner.verify_fits(output,activation)[1]) == ['I1']
    manifest['june_inputs_sha256'] = _hex('0')
    (fits/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='sealed activation/June'):
        runner.verify_fits(output,activation)
    manifest['june_inputs_sha256'] = runner.hash_file(output/'june_inputs.npz')
    rows[1]['predictor'], rows[2]['predictor'] = rows[2]['predictor'], rows[1]['predictor']
    (fits/'I1.json').write_text(json.dumps({'predictors':list(runner.PREDICTORS),'fits':rows}))
    manifest['files']['I1'] = runner.hash_file(fits/'I1.json')
    (fits/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='permuted'):
        runner.verify_fits(output,activation)
