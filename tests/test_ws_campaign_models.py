"""Owned generated arrays only: variant semantics and all job-kind paths."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.ws_seq import train_seq as base
from experiments.ws_campaign import models, train

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    torch.set_num_threads(1)
    cfg = json.loads((ROOT / 'configs/EXP-WS-CAMPAIGN-001.json').read_text())
    cfg['base']['model'].update(trunk_hidden=[32, 16], token_dim=8, encoder_ff=16)
    cfg['base']['training'].update(max_epochs=14, patience=5, batch_size=64, eval_batch_size=256, lr=.02)
    cfg['pretraining'].update(max_steps=3, finetune_epochs=8)
    cfg['variants']['history_columns'] = [12, 13]
    sizes = {'fit':256, 'tuning':128, 'calibration':1024, 'evaluation':128}
    n = sum(sizes.values()); rng = np.random.default_rng(20261013)
    y10 = np.arange(n) % 10; whiff = np.arange(n) // 10 % 2; gb = np.arange(n) // 20 % 2
    x = rng.normal(0, .03, (n,199)).astype(np.float32)
    x[np.arange(n), y10] = 3; x[:,10] = whiff*2-1; x[:,11] = gb*2-1
    keys = np.column_stack([np.arange(n)//8+100, np.ones(n,int), np.arange(n)%8+1]).astype(np.int64)
    c = {'keys':keys, 'games':keys[:,0]}; at=0
    for k,size in sizes.items():
        c[k] = (np.arange(n)>=at)&(np.arange(n)<at+size); at+=size
    cal = np.flatnonzero(c['calibration']); noisy = cal[rng.random(len(cal)) < .2]
    y10 = y10.copy(); y10[noisy] = rng.integers(0,10,len(noisy))
    for y in (whiff, gb):
        flip = cal[rng.random(len(cal)) < .2]; y[flip] = 1-y[flip]
    c.update(y10=y10,narrow_whiff=whiff,ground_ball_per_pitch=gb,y7=base.seven_label(y10,whiff,gb))
    s = dict(game_pk=keys[:,0],at_bat_number=keys[:,1],pitch_number=keys[:,2],pitcher=np.arange(n)//8%4+1,
             batter=np.arange(n)%7+1,pitch_type=np.arange(n)%19,outcome10=y10,balls=np.arange(n)%4,strikes=np.arange(n)%3)
    counts={};s['fit_prior_support']=np.zeros(n,dtype=np.int64)
    for i in np.flatnonzero(c['fit']):
        pid=int(s['pitcher'][i]);s['fit_prior_support'][i]=counts.get(pid,0);counts[pid]=counts.get(pid,0)+1
    ctx = base.build_context_index(s['pitcher'],*keys.T,32)
    return cfg,(x,c,s,ctx)


def test_inventory_and_orders_padding_causality():
    cfg,arrays=fixture();x,c,s,ctx=arrays
    assert len(cfg['jobs'])==87 and sum(j['calibrations'] for j in cfg['jobs'])==228
    data=base.Data(torch,torch.device('cpu'),*arrays)
    idx=torch.tensor([0,1,7,8]);tok=data.tokens(idx);keys=torch.tensor(c['keys'][idx])
    for arm in ['T','R','C','C-shuf','G','K','N0','T-recent4','T-leaf7','T-tokdrop','T-histmask']:
        model=models.build_model(cfg,arm,0).eval()
        z=model(data.x[idx],tok,keys)
        assert all(torch.isfinite(v).all() for v in z.values())
        if arm in ('C','G','K'):
            tt=model.prepare_tokens(tok,keys)
            assert tt['pos'][2,:7].tolist()==list(range(7,0,-1))
        if arm=='C-shuf':
            tt=model.prepare_tokens(tok,keys);again=model.prepare_tokens(tok,keys)
            assert all(torch.equal(tt[k],again[k]) for k in tt)
            for i in range(len(idx)):
                assert sorted(tt['pos'][i][tt['valid'][i]].tolist())==sorted(tok['pos'][i][tok['valid'][i]].tolist())
    encoder=models.CausalTCN(8,cfg['variants']['tcn']).eval()
    a=torch.randn(2,32,8);b=a.clone();b[:,16:]+=5;pad=torch.zeros(2,32,dtype=torch.bool)
    torch.testing.assert_close(encoder(a,pad)[:,:16],encoder(b,pad)[:,:16],rtol=0,atol=0)
    t=models.build_model(cfg,'T',0).eval();r=models.build_model(cfg,'R',0).eval()
    for key,value in t.state_dict().items():
        if not key.startswith('net.encoder.'):
            torch.testing.assert_close(value,r.state_dict()[key],rtol=0,atol=0)
    # EVAL current-target corruption cannot alter tokens of that same row.
    altered=copy.deepcopy(s);altered['outcome10'][7]=(s['outcome10'][7]+1)%10
    other=base.Data(torch,torch.device('cpu'),x,c,altered,ctx)
    assert torch.equal(data.tokens(torch.tensor([7]))['out'],other.tokens(torch.tensor([7]))['out'])


def test_all_training_kinds_and_no_eval(tmp_path):
    cfg,arrays=fixture();deps={}
    ids=['A-T-s0','A-T-s1','A-T-s2','A-R-s0','A-C-s0','A-G-s0','A-K-s0','A-N0-s0','A-T-recent4-s0','A-C-shuf-s0',
         'A-T-leaf7-s0','A-T-tokdrop-s0','A-T-histmask-s0','B-mask-pre-s0','B-contrast-pre-s0',
         'B-T-mask-pre-s0','B-T-mask-scratch-s0','B-T-contrast-pre-s0','B-T-contrast-scratch-s0',
         'B-T-dro-s0','B-teacher-s0','B-teacher-s1','B-teacher-s2','B-N0-distill-s0']
    results={}
    for id in ids:
        job=next(j for j in cfg['jobs'] if j['id']==id);out=tmp_path/id
        result=train.run_job(job,cfg,None,out,{'synthetic':True,'synthetic_arrays':arrays,'dependencies':deps})
        train.save_json(out/'receipt.json',result);deps[id]=str(out);results[id]=result
        assert result['status']=='COMPLETED' and result['eval_scores_opened'] is False
        assert result['evaluation_predictions_computed'] is False
        assert not any('eval' in name for name in result['outputs_sha256'])
    for objective in ['mask','contrast']:
        a=results[f'B-T-{objective}-pre-s0']['paired_compute'];b=results[f'B-T-{objective}-scratch-s0']['paired_compute']
        assert a['total_target_steps']==b['total_target_steps']
    bad=copy.deepcopy(arrays);bad[3][np.flatnonzero(bad[1]['fit'])[0],0]=np.flatnonzero(bad[1]['evaluation'])[0]
    job=next(j for j in cfg['jobs'] if j['id']=='B-mask-pre-s0')
    with pytest.raises(ValueError,match='outside FIT'):
        train.run_job(job,cfg,None,tmp_path/'bad',{'synthetic':True,'synthetic_arrays':bad})


def test_eval_values_do_not_change_fit_and_cal(tmp_path):
    cfg,arrays=fixture();job=next(j for j in cfg['jobs'] if j['id']=='A-N0-s0')
    changed=copy.deepcopy(arrays);ev=changed[1]['evaluation']
    changed[0][ev]=999
    changed[1]['y7'][ev]=(changed[1]['y7'][ev]+1)%7
    changed[1]['y10'][ev]=(changed[1]['y10'][ev]+1)%10
    receipts=[]
    for name,data in [('base',arrays),('changed-eval',changed)]:
        receipts.append(train.run_job(job,cfg,None,tmp_path/name,{'synthetic':True,'synthetic_arrays':data}))
    assert receipts[0]['best_select_nll7']==receipts[1]['best_select_nll7']
    assert (tmp_path/'base'/'temperatures.json').read_bytes()==(tmp_path/'changed-eval'/'temperatures.json').read_bytes()
    a=torch.load(tmp_path/'base'/'model-state.pt',weights_only=True)
    b=torch.load(tmp_path/'changed-eval'/'model-state.pt',weights_only=True)
    assert all(torch.equal(a[k],b[k]) for k in a)
    c=copy.deepcopy(arrays[2]);del c['fit_prior_support']
    with pytest.raises(ValueError,match='chronological FIT support'):
        train.support_groups(arrays[1],c,cfg['dro'])
