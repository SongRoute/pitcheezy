"""Generated state/encoder/rows only. Never open any accepted research artifact."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import softmax

from pitcheezy.seq_shadow import fullseason as fs
from experiments.ws_campaign import models, train
from experiments.ws_fullseason import calibrate as cal

ROOT = Path(__file__).resolve().parents[1]


def binding(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write_doc(path, body):
    path.write_text(json.dumps(body, allow_nan=False)); return binding(path)


def generated_config(root):
    torch.set_num_threads(1)
    recipe = json.loads((ROOT / 'configs/EXP-WS-FULLSEASON-T-001.json').read_text())
    recipe['execution_stage'] = 'FINAL_REFIT'
    recipe['base']['model'].update(trunk_hidden=[16, 8], token_dim=8, encoder_ff=16)
    states = {}; nets = {}
    for seed in fs.SEEDS:
        net = models.build_model(recipe, 'T', seed).eval(); nets[seed] = net
        path = root / f'state-{seed}.pt'; torch.save(net.state_dict(), path); states[str(seed)] = binding(path)
    encoder = {'dimension':199, 'median':[0.1]*91, 'mean':[0.2]*91, 'scale':[0.7]*91,
               'continuous':list(range(36)), 'knots':[[-1.]*36, [0.]*36, [1.]*36]}
    calibrator = dict(schema=cal.SCHEMA, kind='temperature', temperature=1.4, reason='fitted', clip=cal.CLIP,
        bounds=list(cal.base.T_BOUNDS), seeds=list(cal.SEEDS), order='mean_raw_seeds_then_temperature',
        weighting='equal_pitch', folds=list(cal.FOLDS), minimum_rows=cal.MIN_ROWS)
    config = dict(schema=fs.SCHEMA,seeds=list(fs.SEEDS),leaf_order=list(fs.LEAVES),threads=1,state=states,
        recipe=write_doc(root/'recipe.json',recipe),encoder=write_doc(root/'encoder.json',encoder),
        calibrator=write_doc(root/'calibrator.json',calibrator),
        expected_state_tensors=len(nets[0].state_dict()),
        expected_parameters=sum(v.numel() for v in nets[0].state_dict().values()),
        source_pins={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in fs.SOURCE_FILES})
    return config, nets


def generated_inputs():
    rng = np.random.default_rng(20261016); b = 4
    x = rng.normal(size=(b,199)).astype(np.float32)
    keys = np.array([[10+i, 3, 20] for i in range(b)],np.int64)
    tok = {n: np.zeros((b,32),np.int64) for n in fs.TOKEN_FIELDS}
    tok.update(valid=np.zeros((b,32),bool),context_keys=np.zeros((b,32,3),np.int64),
               context_pitcher=np.zeros((b,32),np.int64),query_pitcher=np.arange(1,b+1,dtype=np.int64))
    tok['pos'][:] = np.arange(1,33)
    for i,n in enumerate([0,1,7,32]):
        tok['valid'][i,:n] = True
        tok['context_keys'][i,:n,0] = keys[i,0]
        tok['context_keys'][i,:n,1] = 2
        tok['context_keys'][i,:n,2] = np.arange(n,0,-1)
        tok['context_pitcher'][i,:n] = tok['query_pitcher'][i]
        for name,size in [('type',19),('out',10),('balls',4),('strikes',3),('same',2)]:
            tok[name][i,:n] = rng.integers(0,size,n)
    return x,tok,keys


@pytest.fixture
def candidate(tmp_path):
    config,nets=generated_config(tmp_path)
    return config,nets,fs.load(config)


def test_generated_forward_matches_campaign_reference_and_calibration(candidate):
    config,nets,ensemble=candidate;x,tokens,keys=generated_inputs()
    got=ensemble.predict(x,tokens,keys)
    tt={k:torch.from_numpy(v.copy()) for k,v in tokens.items() if k in (*fs.TOKEN_FIELDS,'valid')}
    qs={}
    with torch.no_grad():
        for seed,net in nets.items():
            z={k:v.double().numpy() for k,v in net(torch.from_numpy(x),tt,torch.from_numpy(keys)).items()}
            qs[seed]=train.probabilities(z)
            np.testing.assert_allclose(got['raw_by_seed'][seed],qs[seed],atol=1e-7,rtol=1e-6)
    mean=np.mean([qs[s] for s in fs.SEEDS],axis=0)
    expected=softmax(np.log(np.clip(mean,1e-15,1))/1.4,axis=1)
    np.testing.assert_allclose(got['raw_mean'],mean,atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(got['calibrated'],expected,atol=1e-7,rtol=1e-6)
    wrong=np.mean([softmax(np.log(np.clip(qs[s],1e-15,1))/1.4,axis=1) for s in fs.SEEDS],axis=0)
    assert np.max(np.abs(wrong-expected))>1e-5
    # C/F storage is not a semantic distinction.
    np.testing.assert_array_equal(ensemble.predict(np.asfortranarray(x),tokens,keys)['calibrated'],got['calibrated'])


def test_stored_encoder_matches_frozen_helper(candidate):
    from experiments.ws_fullseason.data import closure
    _,_,ensemble=candidate
    raw=np.random.default_rng(7).normal(size=(4,91));raw[0,0]=np.nan;raw[1,1]=np.inf;raw[:,2]=1
    expected=closure().safe.apply_encoder(raw,ensemble.encoder)
    np.testing.assert_array_equal(ensemble.encode(raw),expected)
    assert ensemble.encode(raw).dtype==np.float32
    with pytest.raises(ValueError):ensemble.encode(np.zeros((101,91)))


def test_identity_preserves_zero_and_mean_order(candidate):
    _,_,ensemble=candidate
    identity=dict(ensemble.calibrator,kind='identity',temperature=1.,reason='insufficient_support')
    qs={s:np.array([[0,0,0,0,0,0,1.]]) for s in fs.SEEDS}
    np.testing.assert_array_equal(cal.apply(qs,identity),qs[0])


def test_hash_and_source_mismatch_precede_deserialization(tmp_path,monkeypatch):
    config,_=generated_config(tmp_path)
    def forbidden(*a,**k):raise AssertionError('torch.load must not run')
    monkeypatch.setattr(torch,'load',forbidden)
    bad=copy.deepcopy(config);bad['state']['2']['sha256']='0'*64
    with pytest.raises(fs.PinMismatch):fs.load(bad)
    bad=copy.deepcopy(config);bad['source_pins'].pop(next(iter(bad['source_pins'])))
    with pytest.raises(fs.PinMismatch):fs.load(bad)
    bad=copy.deepcopy(config);bad['encoder']['bytes']=fs.JSON_CAP+1
    with pytest.raises(fs.PinMismatch):fs.load(bad)


def test_invalid_metadata_precedes_deserialization(tmp_path,monkeypatch):
    config,_=generated_config(tmp_path)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('premature deserialize'))
    for field,mutate in [('encoder',lambda x:x.update(continuous=[0]*36)),
                         ('calibrator',lambda x:x.update(order='calibrate_then_mean')),
                         ('recipe',lambda x:x['base']['model'].update(token_dim=10**9))]:
        bad=copy.deepcopy(config);value=json.loads(Path(config[field]['path']).read_text());mutate(value)
        bad[field]=write_doc(tmp_path/(field+'-bad.json'),value)
        with pytest.raises((fs.PinMismatch,ValueError)):fs.load(bad)
    bad=copy.deepcopy(config);bad['seeds']=[0,1]
    with pytest.raises(fs.PinMismatch):fs.load(bad)


def test_state_finiteness_shape_and_count_fail_closed(tmp_path):
    config,_=generated_config(tmp_path)
    raw=fs.read_artifact(config['state']['0'],fs.STATE_CAP)
    import io
    state=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    first=next(iter(state));state[first].view(-1)[0]=float('nan')
    path=tmp_path/'nan.pt';torch.save(state,path);bad=copy.deepcopy(config);bad['state']['0']=binding(path)
    with pytest.raises(fs.PinMismatch):fs.load(bad)
    bad=copy.deepcopy(config);bad['expected_parameters']+=1
    with pytest.raises(fs.PinMismatch):fs.load(bad)
    state[first]=torch.zeros(1);path=tmp_path/'shape.pt';torch.save(state,path)
    bad=copy.deepcopy(config);bad['state']['0']=binding(path)
    with pytest.raises((fs.PinMismatch,RuntimeError)):fs.load(bad)


def test_context_and_vocab_fail_before_forward(candidate,monkeypatch):
    _,_,ensemble=candidate;x,tok,keys=generated_inputs()
    for net in ensemble.nets.values():monkeypatch.setattr(net,'forward',lambda *a,**k:pytest.fail('invalid context forwarded'))
    changes=[('context_keys',(1,0,0),999),('context_keys',(1,0,1),4),
             ('context_pitcher',(1,0),999),('valid',(2,1),False),
             ('context_keys',(2,1,2),99),('out',(1,0),10),('pos',(1,0),2)]
    for field,index,value in changes:
        bad={k:v.copy() for k,v in tok.items()};bad[field][index]=value
        with pytest.raises(ValueError):ensemble.predict(x,bad,keys)
    with pytest.raises(ValueError):ensemble.predict(x.astype(np.float64),tok,keys)
    wrong=keys.copy();wrong[1]=wrong[0]
    with pytest.raises(ValueError):ensemble.predict(x,tok,wrong)


def test_read_binding_rejects_symlink_and_size(tmp_path):
    p=tmp_path/'data';p.write_bytes(b'generated');b=binding(p)
    link=tmp_path/'link';link.symlink_to(p);b['path']=str(link)
    with pytest.raises(fs.PinMismatch):fs.read_artifact(b,100)
    b=binding(p);b['bytes']-=1
    with pytest.raises(fs.PinMismatch):fs.read_artifact(b,100)
