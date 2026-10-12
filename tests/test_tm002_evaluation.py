"""Small owned synthetic fixtures only. No real payload, service or GPU reads."""
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
import torch
from experiments.ws_seq import train_seq as base
from experiments.ws_seq import tm002
from experiments.ws_seq import train_tm002 as trainer
from experiments.ws_seq import evaluate_tm002 as evaluator

ROOT=Path(__file__).resolve().parents[1]


def dump(path,obj):
    base.save_json(path,obj)
    return base.sha(path)


def make_bundle(root):
    """Six real tiny CPU optimizer/CAL passes, shared artificial rows, normal files."""
    root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    cfg,bc=tm002.load_config(ROOT/'configs/EXP-WS-TM-002.json')
    cfg=copy.deepcopy(cfg);bc=copy.deepcopy(bc)
    sizes={'fit':256,'tuning':128,'calibration':1024,'evaluation':128}
    cfg['expected_rows']={r:sizes[k] for r,k in evaluator.ROLE_MAP.items()};cfg['bootstrap']['draws']=40;cfg['expected_eval_games']=16
    bc['model'].update(trunk_hidden=[32,16],token_dim=8,encoder_ff=16)
    bc['context'].update(max_length=4,position_vocab=5)
    bc['training'].update(max_epochs=14,patience=5,batch_size=64,eval_batch_size=256,lr=.02)
    n=sum(sizes.values());rng=np.random.default_rng(20261012)
    y10=np.arange(n)%10;whiff=(np.arange(n)//10)%2;gb=(np.arange(n)//20)%2
    x=np.zeros((n,199),np.float32);x[np.arange(n),y10]=3;x[:,10]=whiff*2-1;x[:,11]=gb*2-1
    x[:,12:]=rng.normal(0,.03,(n,187))
    C={'keys':np.column_stack([np.arange(n)//8+100,np.ones(n,int),np.arange(n)%8+1]).astype(np.int64)}
    C['games']=C['keys'][:,0];at=0
    for k,size in sizes.items():C[k]=(np.arange(n)>=at)&(np.arange(n)<at+size);at+=size
    # Noisy CAL makes temperatures identifiable inside the existing fixed bounds.
    cal=np.flatnonzero(C['calibration']);noise=rng.random(len(cal))<.18
    y10=y10.copy();y10[cal[noise]]=rng.integers(0,10,noise.sum())
    whiff=whiff.copy();gb=gb.copy()
    for labels in (whiff,gb):
        flip=cal[rng.random(len(cal))<.18];labels[flip]=1-labels[flip]
    C.update(y10=y10,narrow_whiff=whiff,ground_ball_per_pitch=gb)
    C['y7']=base.seven_label(y10,whiff,gb)
    S={'game_pk':C['keys'][:,0],'at_bat_number':C['keys'][:,1],'pitch_number':C['keys'][:,2],
       'pitcher':np.arange(n)//8%4+1,'batter':np.arange(n)%7+1,'pitch_type':np.arange(n)%19,
       'outcome10':y10,'balls':np.arange(n)%4,'strikes':np.arange(n)%3,'stand':np.arange(n)%2}
    inputs=root/'inputs';inputs.mkdir();np.save(inputs/'x-prequential.npy',x)
    np.savez(inputs/'corpus-subset.npz',**C);np.savez(inputs/'seq-inputs.npz',**S)
    files={f:{'sha256':base.sha(inputs/f)} for f in ('x-prequential.npy','corpus-subset.npz','seq-inputs.npz')}
    manifest={'id':'EXP-WS-SEQ-001','synthetic':True,'files':files,'registration_sha256':'synthetic-registration',
              'denominators':{'role_rows':cfg['expected_rows']},'bound_sources':{'train_seq.py':base.sha(Path(base.__file__)),'config':base.sha(ROOT/cfg['base_config'])}}
    manifest_sha=dump(inputs/'manifest.json',manifest)
    exposure=root/'exposure.json';dump(exposure,{**{k:cfg[k] for k in ('data_version','data_role','roles','expected_rows')},
        'input_manifest_sha256':manifest_sha,'role_dates_verified':True,'source_registration_sha256':'synthetic-registration',
        'prepare_inputs_sha256':base.sha(ROOT/'experiments/ws_seq/prepare_inputs.py')})
    slots={f'{a}-s{s}':{'attempt_id':f'synthetic-{a}-s{s}','output':str(root/f'{a}-s{s}')} for a in cfg['arms'] for s in cfg['seeds']}
    attempt=root/'attempts.json';dump(attempt,{'id':cfg['id'],'bundle_id':'synthetic-bundle','input_manifest_sha256':manifest_sha,
                                             'scoring_root':str(root/'score-open'),'slots':slots})
    common={'id':cfg['id'],'execute':True,'verdict':'PASS','distinct_from_author':True,'author':'synthetic-author','reviewer':'synthetic-reviewer',
            'sources_sha256':trainer.source_hashes(),'input_manifest_sha256':manifest_sha,'data_version':cfg['data_version'],
            'data_exposure_ledger':str(exposure),'data_exposure_ledger_sha256':base.sha(exposure),
            'attempt_ledger':str(attempt),'attempt_ledger_sha256':base.sha(attempt),'inputs':str(inputs)}
    grant=dict(common,scope='TM002_EVALUATE',output=str(root/'result'),runs={})
    ctx=base.build_context_index(S['pitcher'],S['game_pk'],S['at_bat_number'],S['pitch_number'],4)
    for arm in cfg['arms']:
        for seed in cfg['seeds']:
            name=f'{arm}-s{seed}';out=root/name;out.mkdir()
            g=dict(common,arm=arm,seed=seed,attempt_id=slots[name]['attempt_id'],output=str(out))
            gp=out/'grant.json';grantsha=dump(gp,g)
            metadata={'synthetic':True,'id':cfg['id'],'data_role':cfg['data_role'],'data_version':cfg['data_version'],
                      'commit':'synthetic','bundle_id':'synthetic-bundle','scoring_root':str(root/'score-open'),'attempt_id':slots[name]['attempt_id'],
                      'source_sha256':trainer.source_hashes(),'input_manifest_sha256':manifest_sha,
                      'inputs_sha256':{f:v['sha256'] for f,v in files.items()},'grant':g,'grant_sha256':grantsha,
                      'data_exposure_ledger_sha256':base.sha(exposure),'config_sha256':base.sha(ROOT/'configs/EXP-WS-TM-002.json')}
            receipt=trainer.train(torch,bc,arm,seed,x,C,S,ctx,out,metadata,device=torch.device('cpu'))
            rsha=dump(out/'receipt.json',receipt)
            ssha=dump(out/'SUPERVISOR.json',{'status':'COMPLETED','returncode':0,'reason':None,'receipt_sha256':rsha,'grant_sha256':grantsha,'synthetic_fixture':True})
            grant['runs'][name]={'directory':str(out),'receipt_sha256':rsha,'supervisor_sha256':ssha}
    return cfg,bc,grant,inputs,root/'result'


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    return make_bundle(tmp_path_factory.mktemp('tm002-synthetic'))


def test_six_tiny_fit_to_paired_evaluation(bundle,monkeypatch):
    cfg,bc,grant,inputs,out=bundle
    def forbidden_refit(*args,**kwargs):raise AssertionError('evaluator must not refit CAL')
    monkeypatch.setattr(base,'temperature',forbidden_refit)
    result=evaluator.evaluate(cfg,bc,grant,inputs,out,synthetic=True)
    assert result['synthetic'] is True and result['calibration_optimizations_during_evaluation']==0
    assert set(result['paired'])=={'R-T','R-s0-T-s0','R-s1-T-s1','R-s2-T-s2'}
    assert result['primary']['difference']==pytest.approx(result['arms']['R']['nll']-result['arms']['T']['nll'])
    assert result['bootstrap']['unit']=='game_pk'
    another=dict(grant,output=str(out.parent/'duplicate'))
    with pytest.raises(FileExistsError):evaluator.evaluate(cfg,bc,another,inputs,Path(another['output']),synthetic=True)
    assert not Path(another['output']).exists()


def test_invalid_metadata_blocks_before_open(bundle,tmp_path):
    cfg,bc,grant,inputs,_=bundle
    for mutation in ('missing','source','supervisor'):
        g=copy.deepcopy(grant);g['output']=str(tmp_path/mutation)
        if mutation=='missing':del g['runs']['R-s2']
        elif mutation=='source':g['sources_sha256']['experiments/ws_seq/tm002.py']='0'*64
        else:g['runs']['R-s2']['supervisor_sha256']='0'*64
        with pytest.raises(base.Stop):evaluator.evaluate(cfg,bc,g,inputs,Path(g['output']),synthetic=True)
        assert not Path(g['output']).exists()


def test_config_copy_cannot_change_roles_or_data_label(tmp_path):
    cfg=json.loads((ROOT/'configs/EXP-WS-TM-002.json').read_text())
    cfg['data_role']='ALTERED';cfg['expected_rows']['EVAL']=1
    path=tmp_path/'config.json';dump(path,cfg)
    with pytest.raises(ValueError,match='unregistered config bytes'):tm002.load_config(path)


@pytest.mark.parametrize('damage',['ordered_keys','missing_column','mixed_prediction','cal_bounds'])
def test_corrupt_arrays_are_rejected(bundle,tmp_path,damage):
    cfg,bc,grant,inputs,_=bundle
    _,_,found=evaluator.metadata_gate(cfg,grant)
    x,C,S=base.load_package(inputs,bc)
    name='R-s0';d=tmp_path/name;shutil.copytree(found[name]['directory'],d)
    r=copy.deepcopy(found[name]['receipt']);found[name]={'directory':d,'receipt':r}
    if damage in ('ordered_keys','missing_column'):
        with np.load(d/'margins.npz',allow_pickle=False) as z:M={k:np.array(z[k]) for k in z.files}
        if damage=='ordered_keys':M['evaluation__keys']=M['evaluation__keys'][::-1]
        else:del M['calibration__parent']
        np.savez(d/'margins.npz',**M);r['outputs_sha256']['margins.npz']=base.sha(d/'margins.npz')
    elif damage=='mixed_prediction':
        shutil.copyfile(found['T-s1']['directory']/'seven-leaf-evaluation.npy',d/'seven-leaf-evaluation.npy')
        r['outputs_sha256']['seven-leaf-evaluation.npy']=base.sha(d/'seven-leaf-evaluation.npy')
    else:
        cal=json.loads((d/'temperatures.json').read_text());cal['temperatures']['parent']=30
        r['calibration']=cal;r['outputs_sha256']['temperatures.json']=dump(d/'temperatures.json',cal)
    with pytest.raises(base.Stop):evaluator.load_predictions(cfg,bc,found,C,grant['input_manifest_sha256'],r['inputs_sha256'])


def test_receipt_identity_not_just_its_hash(bundle,tmp_path):
    cfg,bc,grant,inputs,_=bundle;g=copy.deepcopy(grant)
    d=tmp_path/'wrong-seed';shutil.copytree(Path(g['runs']['T-s0']['directory']),d)
    r=json.loads((d/'receipt.json').read_text());r['seed']=1
    rsha=dump(d/'receipt.json',r)
    sup=json.loads((d/'SUPERVISOR.json').read_text());sup['receipt_sha256']=rsha
    ssha=dump(d/'SUPERVISOR.json',sup)
    g['runs']['T-s0']={'directory':str(d),'receipt_sha256':rsha,'supervisor_sha256':ssha}
    with pytest.raises(base.Stop,match='receipt/source'):evaluator.metadata_gate(cfg,g)


def test_owned_external_supervisor_wall_cap(tmp_path):
    # Starts exactly one owned child. No arbitrary PID or system service signals.
    rc=trainer.supervise_command([sys.executable,'-c','import time;time.sleep(30)'],tmp_path,.15,10**9,'synthetic')
    report=json.loads((tmp_path/'SUPERVISOR.json').read_text())
    assert rc==87 and report['status']=='INCOMPLETE' and report['reason']=='wall'
    assert report['elapsed_seconds']<8


def test_one_shot_failed_retry_is_recorded(tmp_path):
    root=tmp_path/'open';out=tmp_path/'first'
    attempts={'scoring_root':str(root),'bundle_id':'synthetic'}
    grant={'output':str(out),'runs':{}}
    marker=evaluator.claim_open(grant,attempts,out)
    failure=out/'STOP.json';dump(failure,{'status':'INCOMPLETE','reason':'synthetic corruption'})
    retry=dict(grant,output=str(tmp_path/'retry'),retry={'index':1,'reason':'fixed synthetic corruption',
        'previous_marker_sha256':base.sha(marker),'failure_file':str(failure),'failure_sha256':base.sha(failure)})
    second=evaluator.claim_open(retry,attempts,Path(retry['output']))
    assert marker.exists() and second.name=='EVAL-OPENED-RETRY-1.json'
    with pytest.raises(FileExistsError):evaluator.claim_open(dict(retry,output=str(tmp_path/'another')),attempts,tmp_path/'another')
    dump(out/'receipt.json',{'status':'COMPLETED'})
    with pytest.raises(base.Stop,match='completed scoring cannot be retried'):
        evaluator.claim_open(dict(retry,output=str(tmp_path/'forbidden')),attempts,tmp_path/'forbidden')


def test_roles_and_scoring_root_cannot_be_rebound(bundle,tmp_path):
    cfg,bc,grant,inputs,_=bundle
    _,C,_=base.load_package(inputs,bc)
    C['games']=C['games'].copy()
    C['games'][np.flatnonzero(C['evaluation'])[0]]=C['games'][np.flatnonzero(C['fit'])[0]]
    with pytest.raises(base.Stop,match='split across roles'):tm002.check_role_partition(C)
    changed=copy.deepcopy(grant)
    attempts=json.loads(Path(grant['attempt_ledger']).read_text());attempts['scoring_root']=str(tmp_path/'new-open')
    p=tmp_path/'changed-attempts.json';changed['attempt_ledger_sha256']=dump(p,attempts);changed['attempt_ledger']=str(p)
    with pytest.raises(base.Stop,match='receipt/source'):evaluator.metadata_gate(cfg,changed)


def test_role_date_and_author_gates(tmp_path):
    cfg,_=tm002.load_config(ROOT/'configs/EXP-WS-TM-002.json')
    with pytest.raises(base.Stop):trainer.check_grant({'reviewer':'x'},cfg,'T',0,tmp_path/'inputs',tmp_path/'out')
    exposure={**{k:cfg[k] for k in ('data_version','data_role','roles','expected_rows')},
              'input_manifest_sha256':'synthetic','role_dates_verified':False,
              'source_registration_sha256':'synthetic','prepare_inputs_sha256':'synthetic'}
    p=tmp_path/'exposure.json';pin=dump(p,exposure)
    grant={'data_exposure_ledger':str(p),'data_exposure_ledger_sha256':pin,'input_manifest_sha256':'synthetic'}
    with pytest.raises(base.Stop,match='role proof'):tm002.check_contract(grant,cfg)
