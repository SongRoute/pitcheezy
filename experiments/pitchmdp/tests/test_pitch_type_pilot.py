"""Synthetic checks only: no registered raw file is opened here."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pitchmdp.pitch_type_pilot import prepare, predict, evaluate_frame, load_registered

ROOT = Path(__file__).resolve().parents[3]


def config():
    return json.loads((ROOT/'configs/EVAL-PITCH-TYPE-004.json').read_text())


def fixture():
    rows = []
    for game, date, n in ((1, '2024-04-10', 240), (2, '2024-05-10', 240),
                           (3, '2024-06-02', 3), (4, '2024-06-10', 3),
                           (5, '2024-06-16', 4), (6, '2024-06-17', 4)):
        for j in range(n):
            rows.append(dict(pitch_type='FF' if j % 2 == 0 else 'SL', game_date=pd.Timestamp(date),
                game_type='R', game_pk=game, at_bat_number=1+j//3, pitch_number=1+j%3,
                pitcher=11 if j%2 == 0 else 22, stand='R', p_throws='R', balls=j%4,
                strikes=j%3, on_1b=np.nan, on_2b=np.nan, on_3b=np.nan,
                outs_when_up=j%3, bat_score=2, fld_score=3))
    return pd.DataFrame(rows)


def test_rosters_and_source_allowlist():
    fit, query, pop, ledger = prepare(fixture(), config())
    assert len(fit) == 480 and len(query) == 8 and ledger == []
    assert set(fit.game_pk) == {1, 2} and set(query.game_pk) == {5, 6}
    assert (query.bases == 0).all() and (query.score_diff == 1).all()
    with pytest.raises(ValueError, match='allowlist'):
        prepare(fixture().assign(release_speed=90), config())


def test_manual_smoothing_and_frozen_eval_labels():
    fit, query, _, _ = prepare(fixture(), config())
    fit = fit.iloc[:2].copy(); fit['label'] = 0
    p, support, tree = predict(fit, query, config())
    expected = (np.array([2]+[0]*8) + 4/9) / 6
    assert np.allclose(tree['global_prior'], expected)
    assert np.allclose(p['pitcher_reference'][0], (np.array([1]+[0]*8)+4*expected)/5)
    assert np.allclose(p['tree_candidate'][0], (np.array([2]+[0]*8)+4*expected)/6)
    assert (p['tree_candidate'] > 0).all()
    changed = query.copy(); changed['label'] = 8
    other, _, _ = predict(fit, changed, config())
    for name in p: np.testing.assert_array_equal(p[name], other[name])


def test_unseen_pitcher_and_context_fallback():
    fit, query, _, _ = prepare(fixture(), config())
    query['pitcher'] = 999
    p, support, tree = predict(fit, query, config())
    assert all(s['effective_level'] == 'global' for s in support)
    assert np.allclose(p['context_control'], tree['global_prior'])
    assert (p['tree_candidate'] > 0).all()


def test_unused_labels_do_not_influence_fit():
    frame = fixture(); changed = frame.copy()
    changed.loc[changed.game_pk.isin([3,4]), 'pitch_type'] = 'NEVER_REGISTERED'
    a = evaluate_frame(frame, config()); b = evaluate_frame(changed, config())
    assert a[0] == b[0] and a[1] == b[1]
    pd.testing.assert_frame_equal(a[2], b[2])


def test_shuffle_determinism():
    a = evaluate_frame(fixture(), config())
    b = evaluate_frame(fixture().sample(frac=1, random_state=71), config())
    assert a[0] == b[0] and a[1] == b[1]
    pd.testing.assert_frame_equal(a[2], b[2])


@pytest.mark.parametrize('label', [None, 'UN', 'PO', 'IN', 'OTHER', 'unknown'])
def test_unknown_not_merged(label):
    f = fixture(); idx = f.index[f.game_pk == 5][0]; f.loc[idx,'pitch_type'] = label
    _, query, pop, ledger = prepare(f, config())
    assert len(query) == 7 and pop['EVAL']['label_eligible'] == 7
    assert ledger[0]['reason'] == 'unregistered_or_missing_pitch_type'


@pytest.mark.parametrize('column,value', [('balls', 4),('strikes',3),('outs_when_up',3),
    ('pitcher',np.nan),('stand','S'),('p_throws',None),('fld_score',np.nan),('on_1b',-1)])
def test_input_errors_remain_eligible(column, value):
    f = fixture(); idx = f.index[f.game_pk == 5][0]; f.loc[idx,column] = value
    result, ledger, audit = evaluate_frame(f, config())
    assert len(audit) == 7
    for cov in result['coverage'].values():
        assert cov['eligible'] == 8 and cov['ok'] == 7 and cov['error'] == 1
        assert cov['late'] is None
    assert ledger[0]['reason'] == 'invalid_pre_pitch_input'


def test_key_conflict_and_game_role_conflict():
    f = fixture()
    with pytest.raises(ValueError, match='duplicate'):
        prepare(pd.concat([f, f.iloc[:1]]), config())
    f.loc[f.index[-1], 'game_pk'] = 1
    f.loc[f.index[-1], 'at_bat_number'] = 10000
    with pytest.raises(ValueError, match='conflicting dates'):
        prepare(f, config())


def test_midnight_and_budget_abort():
    f = fixture(); f.loc[0, 'game_date'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='midnight'): prepare(f, config())
    c = config(); c['budget']['max_fit_rows'] = 2
    with pytest.raises(ValueError, match='budget'): prepare(fixture(), c)


def test_role_boundaries_and_label_map():
    f = fixture().iloc[:4].copy()
    f['game_pk'] = [1,2,3,4]
    f['game_date'] = pd.to_datetime(['2024-04-01','2024-06-01','2024-06-15','2024-07-01'])
    f['pitch_type'] = ['FA','UN','SV','UN']
    fit, query, pop, ledger = prepare(f, config())
    assert fit.label.tolist() == [0] and query.label.tolist() == [3] and not ledger


def test_identity_mismatch_does_not_decode(tmp_path, monkeypatch):
    p = tmp_path/'statcast_2024.parquet'; p.write_bytes(b'not parquet')
    monkeypatch.setattr(pd,'read_parquet',lambda *a,**kw: pytest.fail('decoded before identity check'))
    with pytest.raises(ValueError, match='identity mismatch'): load_registered(p, config())


def test_single_game_uncertainty_and_no_claim():
    f = fixture(); f = f.loc[f.game_pk != 6]
    result, _, _ = evaluate_frame(f, config())
    assert result['comparison']['status'] == 'insufficient_games'
    assert result['comparison']['nll']['ci95'] is None
    assert result['scientific_winner'] is None and result['timeliness'] == 'unmeasured_retrospective'


def test_runner_exclusive_output_and_cpu_lock(tmp_path, monkeypatch):
    import fcntl
    import importlib.util
    import sys
    from pitchmdp.pitch_type_pilot import digest
    script = ROOT/'experiments/pitchmdp/scripts/run_pitch_type_pilot.py'
    spec = importlib.util.spec_from_file_location('pilot_runner', script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    cpath = ROOT/'configs/EVAL-PITCH-TYPE-004.json'
    output, lockpath = tmp_path/'attempt', tmp_path/'cpu.lock'
    argv = [str(script),'--config',str(cpath),'--config-sha256',digest(cpath),
        '--input',str(tmp_path/'never-open'),'--output',str(output),
        '--source-commit','synthetic-check','--cpu-lock',str(lockpath)]
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setattr(runner,'load_registered',lambda *a: pytest.fail('raw access before guards'))
    monkeypatch.setattr(runner.resource,'setrlimit',lambda *a: None)
    monkeypatch.setattr(runner.signal,'alarm',lambda *a: None)
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): runner.main()
    with pytest.raises(FileExistsError): runner.main()
    assert runner.os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not list(output.iterdir())


def test_complete_runner_writes_provenance_without_absolute_relative_mismatch(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pitchmdp.pitch_type_pilot import digest
    script = ROOT/'experiments/pitchmdp/scripts/run_pitch_type_pilot.py'
    spec = importlib.util.spec_from_file_location('pilot_runner_complete',script)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    source = tmp_path/'experiments/pitchmdp/mock.py'; source.parent.mkdir(parents=True)
    source.write_text('# source-only synthetic runner regression\n')
    cpath = ROOT/'configs/EVAL-PITCH-TYPE-004.json'; out = tmp_path/'attempt'
    monkeypatch.setattr(sys,'argv',[str(script),'--config',str(cpath),'--config-sha256',digest(cpath),
        '--input',str(tmp_path/'not-real'),'--output',str(out),'--source-commit','synthetic',
        '--cpu-lock',str(tmp_path/'cpu.lock')])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner.resource,'setrlimit',lambda *a: None)
    monkeypatch.setattr(runner.signal,'alarm',lambda *a: None)
    monkeypatch.setattr(runner,'load_registered',lambda *a: None)
    monkeypatch.setattr(runner,'evaluate_frame',lambda *a: ({'synthetic':True},[],pd.DataFrame({'synthetic':[1]})))
    runner.main()
    result = json.loads((out/'metrics.json').read_text())
    assert result['provenance']['source_sha256'] == {'experiments/pitchmdp/mock.py':digest(source)}
    assert sorted(p.name for p in out.iterdir()) == ['exclusions.jsonl','metrics.json','predictions.parquet']


def test_actual_evaluation_payload_serializes_and_nofit_recovery_matches(monkeypatch):
    import importlib.util
    from sklearn.tree import DecisionTreeClassifier
    original, ledger, audit = evaluate_frame(fixture(), config())
    json.dumps(original, allow_nan=False)  # Real tree.leaves, not mocked payload.
    assert type(original['tree']['leaves']) is int
    spec = importlib.util.spec_from_file_location('recovery',ROOT/'experiments/pitchmdp/scripts/recover_pitch_type_pilot.py')
    recovery = importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    monkeypatch.setattr(DecisionTreeClassifier,'fit',lambda *a,**k: pytest.fail('recovery must not fit'))
    restored = recovery.recover(fixture(),audit.sample(frac=1,random_state=3),ledger,config())
    json.dumps(restored,allow_nan=False)
    for name in ('metrics_common_valid_inputs','comparison','populations','coverage','fallback_counts','unseen_pitcher_queries'):
        assert original[name] == restored[name]
    partial=json.dumps(original,indent=2,sort_keys=True).split('    "leaves":')[0]+'    "leaves": '
    assert recovery.completed_field(partial,'comparison') == original['comparison']
    with pytest.raises(json.JSONDecodeError): recovery.completed_field(partial,'tree')
    bad=audit.copy();bad.iloc[0,bad.columns.get_loc('tree_candidate_FF')]=np.nan
    with pytest.raises(ValueError,match='finite'):recovery.recover(fixture(),bad,ledger,config())


def test_complete_recovery_with_actual_synthetic_payload(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pitchmdp.pitch_type_pilot import digest
    spec=importlib.util.spec_from_file_location('recovery_complete',ROOT/'experiments/pitchmdp/scripts/recover_pitch_type_pilot.py')
    recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
    result,ledger,audit=evaluate_frame(fixture(),config())
    result['provenance']={'source_commit':'0f3ec92d6137e6ce390373920392ecb7f03303a7',
        'config_sha256':recovery.CONFIG_SHA,'data_sha256':config()['source']['sha256']}
    result['tree']['leaves']=np.int64(result['tree']['leaves'])
    attempt=tmp_path/'attempt002';attempt.mkdir()
    with (attempt/'metrics.json').open('w') as f:
        with pytest.raises(TypeError):json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    (attempt/'exclusions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ledger))
    audit.to_parquet(attempt/'predictions.parquet',index=False)
    pins={name:digest(attempt/name) for name in recovery.PINS}
    monkeypatch.setattr(recovery,'PINS',pins)
    monkeypatch.setattr(recovery,'load_registered',lambda *a:fixture())
    monkeypatch.setattr(recovery.resource,'setrlimit',lambda *a:None)
    monkeypatch.setattr(recovery.signal,'alarm',lambda *a:None)
    out=tmp_path/'metrics-recovered.json'
    monkeypatch.setattr(sys,'argv',['recovery','--config',str(ROOT/'configs/EVAL-PITCH-TYPE-004.json'),
        '--input',str(tmp_path/'never-real'),'--attempt',str(attempt),'--output',str(out),
        '--cpu-lock',str(tmp_path/'cpu.lock'),'--recovery-commit','synthetic'])
    recovery.main()
    recovered=json.loads(out.read_text())
    assert recovered['metrics_common_valid_inputs']==result['metrics_common_valid_inputs']
    assert recovered['tree'] is None and recovered['recovery']['fit_count']==0
    assert {name:digest(attempt/name) for name in pins}==pins
    with pytest.raises(FileExistsError):recovery.main()
