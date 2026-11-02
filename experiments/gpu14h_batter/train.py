"""FIT-only 타자 offset의 T32 2×2 확장; 원003/005 source는 불변."""
from pathlib import Path
import math
import time
from experiments.gpu14h_identity import train as identity
from experiments.gpu14h_prepitch import train as parent

VARIANTS={'batter_only':{'factors':['batter']},'pitcher_batter':{'factors':['pitcher','batter']}}
CONTROL='pitcher_t32'


def vocabularies(seq,fit9,variant):
    return {k:identity.vocabulary(seq[k],fit9) for k in VARIANTS[variant]['factors']}


def features(x,seq,ids,variant):
    import numpy as np
    return np.column_stack((parent.features(x),*[identity.indices(seq[k],ids[k]).astype(np.float32) for k in VARIANTS[variant]['factors']]))


def recipe(producer,variant):
    if variant not in VARIANTS:raise ValueError('fixed two batter conditions')
    r=parent.recipe(producer,'state_t32')
    r['identity_factors']={'factors':VARIANTS[variant]['factors'],'embedding_dim':16,'vocabulary':'separate sorted FIT_E9 only','unknown':0,'unknown_vector':'permanent zero','fusion':'separate additive factorized logit offsets','initial_offset':'zero','initial_rng':'preserved; each factor same seed as standalone'}
    return r


def build_model(r,variant,seed,sizes):
    import torch
    first=VARIANTS[variant]['factors'][0]
    single=identity.build_model(r,CONTROL,seed,sizes[first])
    if variant=='batter_only':return single
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        embedding=torch.nn.Embedding(sizes['batter']+1,16,padding_idx=0)
        offset=torch.nn.Linear(16,9,bias=False);torch.nn.init.zeros_(offset.weight)
    class Both(torch.nn.Module):
        def __init__(self):
            super().__init__();self.pitcher=single;self.batter_embedding=embedding;self.batter_offset=offset;self.seq=True
        def forward(self,x,tok=None,row_keys=None):
            if x.ndim!=2 or x.shape[1]!=13:raise ValueError('state11 plus two categorical transports')
            return {'leaf':self.pitcher(x[:,:12],tok,row_keys)['leaf']+self.batter_offset(self.batter_embedding(x[:,12].long()))}
    return Both()


def unknown_zero(model,variant):
    import torch
    layers=[model.embedding] if variant=='batter_only' else [model.pitcher.embedding,model.batter_embedding]
    return all(torch.equal(m.weight[0],torch.zeros_like(m.weight[0])) for m in layers)


def run(producer,variant,seed,package,out,synthetic=False):
    return fit(recipe(producer,variant),variant,seed,package,out,synthetic=synthetic)


def run_control(producer,seed,package,out,synthetic=False):
    return identity.run(producer,CONTROL,seed,package,out,synthetic=synthetic)


def fit(r,variant,seed,package,out,synthetic=False):
    import numpy as np
    import torch
    from experiments.ws_campaign import train as campaign
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_fullseason.train import check_epoch
    if seed not in (0,1,2) or variant not in VARIANTS:raise ValueError('고정후보/seed')
    if not synthetic and not torch.cuda.is_available():raise ValueError('CUDA required')
    rule=r['training_rule'];t=r['base']['training']
    if rule['early_stopping'] or rule['SELECT_rows'] or rule['CAL_rows'] or rule['temperature']!=1:raise ValueError('noSELECT/noCAL')
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    campaign.configure_backend({'arm':'T'},{});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not synthetic:torch.cuda.manual_seed_all(seed)
    device=torch.device('cpu' if synthetic else 'cuda')
    x,c,s=package;y=parent.targets(s['pitch_type']);ids=vocabularies(s,c['fit']&(y>=0),variant);x=features(x,s,ids,variant)
    if c['tuning'].any() or c['calibration'].any() or (c['fit']&c['evaluation']).any() or not (c['fit']|c['evaluation']).all():raise ValueError('원역할 경계')
    if len(np.unique(c['keys'],axis=0))!=len(x):raise ValueError('고유키')
    ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],32)
    base.check_context(ctx,s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'])
    valid=ctx>=0
    if not (c['fit'][np.maximum(ctx,0)]==c['fit'][:,None])[valid].all():raise ValueError('context 역할 경계')
    fit=c['fit']&(y>=0);oof=c['evaluation'];fitrows=np.flatnonzero(fit);oofrows=np.flatnonzero(oof)
    if not len(fitrows) or not len(oofrows):raise ValueError('mapped FIT/OOF 없음')
    # Data.tokens gathers only ctx<query. Its current outcome fields are never
    # read by the new loss; the target below is a separate 9-class tensor.
    data=base.Data(torch,device,x,c,s,ctx);keys=torch.as_tensor(c['keys'],device=device,dtype=torch.long);target=torch.as_tensor(y,device=device,dtype=torch.long)
    model=build_model(r,variant,seed,{k:len(v) for k,v in ids.items()}).to(device);opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    trace=[];steps=0;start=time.monotonic()
    for epoch in range(1,rule['epochs']+1):
        model.train();seen=np.zeros(len(x),np.int32);order=fitrows[torch.randperm(len(fitrows)).numpy()]
        for offset in range(0,len(order),t['batch_size']):
            actual=order[offset:offset+t['batch_size']];idx=torch.as_tensor(actual,device=device,dtype=torch.long)
            z=model(data.x[idx],data.tokens(idx) if model.seq else None,keys[idx])
            loss=torch.nn.functional.cross_entropy(z['leaf'],target[idx]);campaign.update(loss,model,opt)
            np.add.at(seen,actual,1);steps+=1
        evidence=check_epoch(seen,fit,c['keys']);evidence.pop('positive_parent_weight_every_row');evidence.update(epoch=epoch,positive_type_weight_every_row=True,optimizer_steps=math.ceil(len(fitrows)/t['batch_size']));trace.append(evidence)
    out=Path(out)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('비유한 가중치')
    if not unknown_zero(model,variant):raise ValueError('unknown vector changed')
    torch.save(model.state_dict(),out/'model-state.pt')
    from experiments.ws_fullseason_c.run import write_json,object_sha
    vocab={'factors':{k:v.tolist() for k,v in ids.items()},'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'unknown_index':0,'embedding_dim':16}
    write_json(out/'identity-vocabularies.json',vocab)
    z=campaign.predict(model,data,keys,torch.as_tensor(oofrows,device=device,dtype=torch.long),t['eval_batch_size']);q=campaign.probabilities(z)
    if q.shape!=(len(oofrows),9) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-10,rtol=0):raise ValueError('9종 확률')
    np.savez(out/'oof-predictions.npz',keys=c['keys'][oofrows],probabilities=q)
    census={role:{'Q_pkg':int(mask.sum()),'E9_pkg':int((mask&(y>=0)).sum()),'raw_type_counts':dict(zip(parent.TYPES,np.bincount(s['pitch_type'][mask],minlength=19).astype(int).tolist()))} for role,mask in [('FIT',c['fit']),('OOF',oof)]}
    return {'status':'COMPLETED','arm':'PREPITCH9_BATTER','variant':variant,'seed':seed,'epochs':rule['epochs'],'consumption':trace,'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oof]),'census':census,'temperature':1.,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch','metrics_computed':False,'elapsed_compute_seconds':time.monotonic()-start,'backend':{'deterministic':True,'warn_only':True,'tf32':False,'dtype':'fp32'},'model_input':'state11, strictly earlier tokens, separate FIT-only additive ID factors','vocabulary_sha256':object_sha(vocab),'vocabulary_sizes':{k:len(v) for k,v in ids.items()},'OOF_unknown':{k:int((identity.indices(s[k][oof],v)==0).sum()) for k,v in ids.items()},'unknown_vector_zero':True,'identity_interaction':'separate additive factorized offsets; no ID x state or P x B interaction'}
