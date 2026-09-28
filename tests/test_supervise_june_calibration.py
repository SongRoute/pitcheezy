"""Synthetic processes and metadata only; no calibration worker or model."""
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/supervise_june_calibration.py'
spec = importlib.util.spec_from_file_location('supervise_june_calibration', SCRIPT)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def fixture(tmp_path):
    coord = tmp_path / 'coord'
    coord.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    source = tmp_path / 'worker.py'
    source.write_text('# synthetic only')
    plan = {key: '' for key in s.COMMON_FIELDS}
    plan.update(protocol=s.PROTOCOL, coord_root=str(coord), code_root=str(tmp_path), code_commit='a'*40,
                scientific_config_path=str(tmp_path/'scientific.json'), scientific_config_sha256='b'*64,
                config_path=str(tmp_path/'execution.json'), config_sha256='c'*64,
                supervisor_sha256=s.sha(SCRIPT), source_hashes={str(source):s.sha(source)},
                python_executable='/Users/song/Projects/pitcheezy/.venv/bin/python', output_dir=str(output),
                worker_heavy_lock=str(tmp_path/'.heavy.lock'), registration_path=str(coord/'registration.json'),
                registration_sha256='d'*64, stages=[])
    for stage, seed in s.ORDER:
        jobid=stage if seed is None else f'predict{seed}'
        checks={'status':'complete', 'stage':stage, 'config_sha256':plan['config_sha256'],
                'registered_code_commit':plan['code_commit'], 'execution_head':plan['code_commit']}
        if seed is not None:
            checks['seed']=seed
        if stage == 'fit':
            checks.update(optimizer_calls=36,replay_complete=True)
        if stage == 'score':
            checks['contrast_additivity_passed']=True
        argv=[plan['python_executable'], str(source),'--config', plan['config_path'],'--output-dir',str(output),'--stage',stage]
        if seed is not None:
            argv += ['--seed',str(seed)]
        plan['stages'].append(dict(id=jobid,stage=stage,seed=seed,argv=argv,
                                  completion_path=str(output/jobid/'manifest.json'),
                                  completion_checks=checks,artifact_hashes_key='outputs'))
    path=coord/'plan.json'
    path.write_text(json.dumps(plan))
    return plan,path


def seal(plan, job):
    path=Path(job['completion_path'])
    path.parent.mkdir(parents=True,exist_ok=True)
    artifact=path.parent/'payload.json'
    artifact.write_text('{}')
    path.write_text(json.dumps({**job['completion_checks'],'outputs':{artifact.name:s.sha(artifact)}}))


def test_schema_and_fit_identity_fail_closed(tmp_path):
    plan,path=fixture(tmp_path)
    s.validate(plan,path)
    plan['stages'][2]['seed']=True
    with pytest.raises(ValueError):
        s.validate(plan,path)
    plan['stages'][2]['seed']=0
    del plan['stages'][7]['completion_checks']['replay_complete']
    with pytest.raises(ValueError,match='replay_complete'):
        s.validate(plan,path)


def test_official_profile_scaling_and_member_budget():
    ends=[{'stage':'prepare','popen_wait_seconds':10}, {'stage':'profile','popen_wait_seconds':10}]
    cap,gate=s.budget({'stage':'predict'},ends)
    assert cap==600 and gate['estimate_seconds']==10*104970/8192
    ends[1]['popen_wait_seconds']=30
    with pytest.raises(ValueError,match='gate refused'):
        s.budget({'stage':'predict'},ends)
    ends[1]['popen_wait_seconds']=10
    ends.append({'stage':'predict','popen_wait_seconds':301})
    with pytest.raises(ValueError,match='gate refused'):
        s.budget({'stage':'predict'},ends)
    ends[2]['popen_wait_seconds']=float('nan')
    with pytest.raises(ValueError,match='invalid recorded'):
        s.budget({'stage':'predict'},ends)
    with pytest.raises(ValueError,match='full stage cap'):
        s.budget({'stage':'score'},[{'stage':'prepare','popen_wait_seconds':3350}])


def test_artifact_tamper_and_semantic_identity(tmp_path):
    plan,_=fixture(tmp_path)
    job=plan['stages'][0]
    seal(plan,job)
    assert s.completion(plan,job)==s.sha(Path(job['completion_path']))
    payload=Path(job['completion_path']).parent/'payload.json'
    payload.write_text('tampered')
    with pytest.raises(ValueError,match='artifact changed'):
        s.completion(plan,job)
    seal(plan,job)
    value=s.read(Path(job['completion_path']))
    value['execution_head']='b'*40
    Path(job['completion_path']).write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completion mismatch'):
        s.completion(plan,job)


def invoke(tmp_path,monkeypatch,child,ticks,completion=None):
    plan,path=fixture(tmp_path)
    coord=Path(plan['coord_root'])
    (coord/'ledger').mkdir()
    monkeypatch.setattr(s.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(s.subprocess,'Popen',child)
    if completion is not None:
        monkeypatch.setattr(s,'completion',completion)
    return s.run_worker(plan,plan['stages'][0],coord,300,s.sha(path),{})


class Child:
    pid=2_000_000_000
    def wait(self,timeout=None):
        return 0
    def poll(self):
        return 0


def test_postflight_failure_preserves_full_worker_time(tmp_path,monkeypatch):
    result=invoke(tmp_path,monkeypatch,lambda *a,**k:Child(),iter([0,0,2,20]),
                  lambda *a:(_ for _ in ()).throw(ValueError('tampered')))
    assert result['outcome']=='failed' and result['popen_wait_seconds']==2
    assert result['postflight_seconds']==18


def test_launch_failure_charged_and_no_clock_free_pass(tmp_path,monkeypatch):
    def fail(*a,**kw):
        raise OSError('launch failed')
    result=invoke(tmp_path,monkeypatch,fail,iter([0,3,3]))
    assert result['popen_wait_seconds']==3 and result['outcome']=='failed'


def test_invalid_clock_charges_reserved_cap(tmp_path,monkeypatch):
    result=invoke(tmp_path,monkeypatch,lambda *a,**k:Child(),iter([5,5,4,4]),lambda *a:'a'*64)
    assert result['popen_wait_seconds']==300 and not result['clock_valid']
    assert result['outcome']=='failed' and result['completion_sha256'] is None


@pytest.mark.parametrize('kind',['timeout','signal'])
def test_cleanup_reaps_and_defers_repeated_signals(tmp_path,monkeypatch,kind):
    class Pending(Child):
        def wait(self,timeout=None):
            if kind=='timeout':
                raise subprocess.TimeoutExpired('synthetic',timeout)
            raise s.Interrupted()
    reaped=[]
    def terminate(child):
        os.kill(os.getpid(),signal.SIGINT)
        os.kill(os.getpid(),signal.SIGTERM)
        reaped.append(child.pid)
        return -15
    monkeypatch.setattr(s,'terminate',terminate)
    result=invoke(tmp_path,monkeypatch,lambda *a,**k:Pending(),iter([0,0,4,4]))
    assert reaped==[2_000_000_000]
    assert result['popen_wait_seconds']==4
    assert result['outcome']==('timeout' if kind=='timeout' else 'interrupted')


def test_refusal_durable_and_no_retry(tmp_path,monkeypatch):
    plan,path=fixture(tmp_path)
    monkeypatch.setattr(s,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('source changed')))
    monkeypatch.setattr(s.subprocess,'Popen',lambda *a,**k:pytest.fail('launched'))
    result=s.run(path)
    assert result['outcome']=='failed' and result['jobs']==[]
    assert s.read(Path(plan['coord_root'])/'ledger/refusal.json')['reason'].endswith('source changed')
    with pytest.raises(ValueError,match='single attempt'):
        s.run(path)


def test_ten_stage_queue_and_completion_gate(tmp_path,monkeypatch):
    plan,path=fixture(tmp_path)
    monkeypatch.setattr(s,'preflight',lambda *a,**k:None)
    ticks=iter(range(1000))
    monkeypatch.setattr(s.time,'monotonic',lambda:next(ticks))
    launched=[]
    def popen(argv,**kw):
        job=plan['stages'][len(launched)]
        assert argv==job['argv'] and kw['start_new_session'] is True
        seal(plan,job)
        launched.append(job['id'])
        return Child()
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    result=s.run(path)
    assert result['outcome']=='completed' and len(launched)==10
    assert result['popen_wait_seconds']==20
    # The final status is evidence-bound and manifest is written after all workers.
    assert all(job['completion_sha256'] for job in result['jobs'])


def test_fit_missing_replay_stops_before_apply_score(tmp_path,monkeypatch):
    plan,path=fixture(tmp_path)
    monkeypatch.setattr(s,'preflight',lambda *a,**k:None)
    ticks=iter(range(1000))
    monkeypatch.setattr(s.time,'monotonic',lambda:next(ticks))
    launched=[]
    def popen(argv,**kw):
        job=plan['stages'][len(launched)]
        seal(plan,job)
        if job['stage']=='fit':
            m=s.read(Path(job['completion_path']))
            m['replay_complete']=False
            Path(job['completion_path']).write_text(json.dumps(m))
        launched.append(job['id'])
        return Child()
    monkeypatch.setattr(s.subprocess,'Popen',popen)
    result=s.run(path)
    assert result['outcome']=='failed' and len(launched)==8
    assert launched[-1]=='fit'

def test_preflight_registration_source_and_scientific_binding(tmp_path,monkeypatch):
    from types import SimpleNamespace
    plan,path=fixture(tmp_path)
    science=Path(plan['scientific_config_path'])
    science.write_text(json.dumps({'budget':{'family_seconds':3600,'profile_gate':{'tail_reserve_seconds':900}}}))
    plan['scientific_config_sha256']=s.sha(science)
    execution=Path(plan['config_path'])
    execution.write_text(json.dumps({'enabled':True,'code_commit_c':plan['code_commit'],
                                    'scientific_config':{'path':str(science),'sha256':plan['scientific_config_sha256']}}))
    plan['config_sha256']=s.sha(execution)
    Path(plan['worker_heavy_lock']).touch()
    registration={key:plan[key] for key in s.COMMON_FIELDS}
    registration.update(protocol=s.REGISTRATION_PROTOCOL,registration_commit_d='e'*40,created_utc='2026-09-28T00:00:00Z')
    reg=Path(plan['registration_path'])
    reg.write_text(json.dumps(registration))
    plan['registration_sha256']=s.sha(reg)
    monkeypatch.setattr(s.subprocess,'run',lambda argv,**kw:SimpleNamespace(stdout=plan['code_commit'] if 'rev-parse' in argv else ''))
    s.preflight(plan,fresh=True)
    registration['output_dir']=str(tmp_path/'other-output')
    reg.write_text(json.dumps(registration))
    plan['registration_sha256']=s.sha(reg)
    with pytest.raises(ValueError,match='registration mismatch: output_dir'):
        s.preflight(plan)
    registration['output_dir']=plan['output_dir']
    reg.write_text(json.dumps(registration))
    plan['registration_sha256']=s.sha(reg)
    science.write_text('{}')
    with pytest.raises(ValueError,match='scientific_config_path changed'):
        s.preflight(plan)


def test_launch_startup_is_subtracted_from_wait_cap(tmp_path,monkeypatch):
    waits=[]
    class SlowLaunch(Child):
        def wait(self,timeout=None):
            waits.append(timeout)
            return 0
    result=invoke(tmp_path,monkeypatch,lambda *a,**k:SlowLaunch(),iter([0,7,9,9]),lambda *a:'a'*64)
    assert waits==[293] and result['popen_wait_seconds']==9


def test_swapped_member_identity_rejected(tmp_path):
    plan,_=fixture(tmp_path)
    job=plan['stages'][2]
    seal(plan,job)
    manifest=Path(job['completion_path'])
    value=s.read(manifest)
    value['seed']=1
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completion mismatch: seed'):
        s.completion(plan,job)


@pytest.mark.parametrize('flag', [None, False])
def test_score_additivity_required_in_registration_and_manifest(tmp_path, flag):
    plan,path=fixture(tmp_path)
    job=plan['stages'][-1]
    seal(plan,job)
    manifest=Path(job['completion_path'])
    value=s.read(manifest)
    if flag is None:
        del value['contrast_additivity_passed']
    else:
        value['contrast_additivity_passed']=flag
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='contrast_additivity_passed'):
        s.completion(plan,job)
    if flag is None:
        del job['completion_checks']['contrast_additivity_passed']
    else:
        job['completion_checks']['contrast_additivity_passed']=flag
    with pytest.raises(ValueError,match='missing completion identity: contrast_additivity_passed'):
        s.validate(plan,path)


@pytest.mark.parametrize('name', ['', '../outside.json', 'sub/../../outside.json', '/tmp/outside.json', './payload.json', 'sub//payload.json', 'manifest.json'])
def test_unsafe_relative_artifacts_rejected(tmp_path,name):
    plan,_=fixture(tmp_path)
    job=plan['stages'][0]
    seal(plan,job)
    manifest=Path(job['completion_path'])
    value=s.read(manifest)
    value['outputs']={name:'a'*64}
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='artifact'):
        s.completion(plan,job)


def test_relative_nested_artifact_and_symlink_escape(tmp_path):
    plan,_=fixture(tmp_path)
    job=plan['stages'][0]
    seal(plan,job)
    manifest=Path(job['completion_path'])
    nested=manifest.parent/'nested'
    nested.mkdir()
    payload=nested/'payload.json'
    payload.write_text('{}')
    value=s.read(manifest)
    value['outputs']={'nested/payload.json':s.sha(payload)}
    manifest.write_text(json.dumps(value))
    assert s.completion(plan,job)==s.sha(manifest)
    payload.unlink()
    outside=tmp_path/'outside.json'
    outside.write_text('{}')
    payload.symlink_to(outside)
    with pytest.raises(ValueError,match='invalid artifact reference'):
        s.completion(plan,job)


def test_old_nested_execution_schema_rejected(tmp_path,monkeypatch):
    from types import SimpleNamespace
    plan,_=fixture(tmp_path)
    science=Path(plan['scientific_config_path'])
    science.write_text(json.dumps({'budget':{'family_seconds':3600,'profile_gate':{'tail_reserve_seconds':900}}}))
    plan['scientific_config_sha256']=s.sha(science)
    config=Path(plan['config_path'])
    config.write_text(json.dumps({'execution':{'enabled':True,'code_commit_c':plan['code_commit']},
                                  'scientific_config':{'path':str(science),'sha256':plan['scientific_config_sha256']}}))
    plan['config_sha256']=s.sha(config)
    registration={key:plan[key] for key in s.COMMON_FIELDS}
    registration.update(protocol=s.REGISTRATION_PROTOCOL,registration_commit_d='e'*40,created_utc='2026-09-28T00:00:00Z')
    reg=Path(plan['registration_path'])
    reg.write_text(json.dumps(registration))
    plan['registration_sha256']=s.sha(reg)
    with pytest.raises(ValueError,match='top-level enabled'):
        s.preflight(plan)
