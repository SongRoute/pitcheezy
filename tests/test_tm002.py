"""Synthetic CPU regression; no corpus, checkpoint or service access."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.ws_seq import tm002 as tm
from experiments.ws_seq import train_seq as base

ROOT = Path(__file__).resolve().parents[1]


def test_shared_initialization_and_original_transformer():
    _, cfg = tm.load_config(ROOT / 'configs/EXP-WS-TM-002.json')
    torch.set_num_threads(1)
    for seed in (0, 1, 2):
        before = torch.get_rng_state().clone()
        t = tm.build_model(torch, cfg, 'T', seed)
        r = tm.build_model(torch, cfg, 'R', seed)
        assert torch.equal(before, torch.get_rng_state())
        assert tm.shared_state(t).keys() == tm.shared_state(r).keys()
        assert all(torch.equal(v, tm.shared_state(r)[k]) for k,v in tm.shared_state(t).items())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            original = base.build_model(torch,cfg,'N1')
        assert all(torch.equal(v,original.state_dict()[k]) for k,v in t.state_dict().items())


def test_padding_empty_order_gradient_and_probabilities():
    _, cfg = tm.load_config(ROOT / 'configs/EXP-WS-TM-002.json')
    torch.set_num_threads(1)
    net = tm.build_model(torch,cfg,'R',0).eval()
    e = torch.randn(3,5,64)
    mask = torch.tensor([[False]*5,[False,False,True,True,True],[False,True,True,True,True]])
    out=net.encoder(e,mask)
    for i,n in enumerate((5,2,1)):
        alone=net.encoder(e[i:i+1,:n],torch.zeros((1,n),dtype=torch.bool))
        torch.testing.assert_close(out[i,:n],alone[0])
        assert not out[i,n:].count_nonzero()
    altered=e.clone(); altered[mask]=1e6
    torch.testing.assert_close(out,net.encoder(altered,mask))
    assert not torch.allclose(out[0],net.encoder(e.flip(1),mask)[0].flip(0))
    with pytest.raises(base.Stop):
        net.encoder(e,torch.tensor([[False,True,False,True,True]]*3))
    valid=~mask; valid[2]=False
    tok={k:torch.zeros((3,5),dtype=torch.long) for k in ('type','out','balls','strikes','same')}
    tok.update(valid=valid,pos=torch.arange(1,6).expand(3,5))
    x=torch.randn(3,199)
    logits=net(x,tok)
    changed={k:v.clone() for k,v in tok.items()}
    for k in ('type','out','balls','strikes','same'):
        changed[k][~valid]=100000
    for a,b in zip(logits,net(x,changed)): torch.testing.assert_close(a,b)
    hidden=torch.cat([net.trunk(x[2:3]),net.no_context[None]],1)
    torch.testing.assert_close(logits[0][2],net.head_parent(hidden)[0])
    q=base.seven_from_margins(*(z.detach().numpy() for z in logits))
    assert np.isfinite(q).all() and (q>=0).all()
    np.testing.assert_allclose(q.sum(1),1)
    sum(z.square().mean() for z in logits).backward()
    assert net.no_context.grad is not None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_strict_prior_and_label_mutation():
    pit=np.array([2,1,1,1,1]); game=np.array([1,1,1,1,2]); ab=np.array([1,1,1,2,1]); pitch=np.array([1,2,1,1,1])
    ctx=base.build_context_index(pit,game,ab,pitch,4)
    np.testing.assert_array_equal(ctx,base.brute_context_index(pit,game,ab,pitch,4))
    base.check_context(ctx,pit,game,ab,pitch)
    assert list(ctx[3])==[1,2,-1,-1]
    values=np.arange(5); query=1
    original=values[ctx[query][ctx[query]>=0]]
    values[[0,1,3,4]]=999
    np.testing.assert_array_equal(original,values[ctx[query][ctx[query]>=0]])
    bad=ctx.copy(); bad[1,0]=3
    with pytest.raises(base.Stop): base.check_context(bad,pit,game,ab,pitch)


def test_nonfinite_config_rejected(tmp_path):
    cfg=json.loads((ROOT/'configs/EXP-WS-TM-002.json').read_text())
    cfg['budgets']['host_rss_bytes_per_fit']=float('nan')
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='non-finite'):tm.load_config(p)


def test_grant_rejects_missing_and_score_selected_seed_gate(tmp_path):
    from experiments.ws_seq import train_tm002 as runner
    cfg,_=tm.load_config(ROOT/'configs/EXP-WS-TM-002.json')
    inputs=tmp_path/'inputs';out=tmp_path/'out'
    with pytest.raises(base.Stop):runner.check_grant({},cfg,'T',0,inputs,out)
    ledger=tmp_path/'ledger.json';ledger.write_text('{}')
    grant={'id':cfg['id'],'execute':True,'verdict':'PASS','distinct_from_author':True,
           'reviewer':'reviewer','author':'author','sources_sha256':runner.source_hashes(),
           'arm':'T','seed':0,'inputs':str(inputs),'output':str(out),
           'data_exposure_ledger':str(ledger),'data_exposure_ledger_sha256':base.sha(ledger),
           'commit':'synthetic','attempt_id':'synthetic','gpu_uuid':'synthetic','common_lock':'synthetic',
           'input_manifest_sha256':'synthetic','budget':cfg['budgets'],'data_version':cfg['data_version']}
    runner.check_grant(grant,cfg,'T',0,inputs,out)
    out.mkdir()
    with pytest.raises(base.Stop,match='already exists'):runner.check_grant(grant,cfg,'T',0,inputs,out)
    out.rmdir();grant['seed']=1
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'verdict':'PASS','distinct_from_author':True,
        'sources_sha256':runner.source_hashes(),'eval_scores_opened':True,
        'receipts_sha256':{'T-s0':'synthetic','R-s0':'synthetic'}}))
    grant.update(seed0_gate=str(gate),seed0_gate_sha256=base.sha(gate))
    with pytest.raises(base.Stop,match='seed0 integrity'):runner.check_grant(grant,cfg,'T',1,inputs,out)


def test_watchdog_exits_only_owned_subprocess(tmp_path):
    import subprocess,sys
    # Invalid self-RSS source is itself a STOP; no real profile/data/GPU access.
    code='''import threading,time
from pathlib import Path
from experiments.ws_seq.train_tm002 import watchdog
threading.Thread(target=watchdog,args=(Path(__import__('sys').argv[1]),0.01,10**12,threading.Event()),daemon=True).start()
time.sleep(5)
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,timeout=10)
    assert result.returncode==87
    assert json.loads((tmp_path/'STOP.json').read_text())['status']=='INCOMPLETE'
