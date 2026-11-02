"""FIT-only 투수 ID의 저차원 logit offset; 원003 trunk/head/RNG를 보존한다."""
from pathlib import Path
import math
import time
from experiments.gpu14h_prepitch import train as parent

VARIANTS={'pitcher_mlp':{'architecture':'state_mlp','embedding_dim':16},'pitcher_t32':{'architecture':'state_t32','embedding_dim':16},'pitcher_c32':{'architecture':'state_c32','embedding_dim':16}}
ARCH={v:s['architecture'] for v,s in VARIANTS.items()}


def vocabulary(pitcher,fit9):
    import numpy as np
    pitcher=np.asarray(pitcher);fit9=np.asarray(fit9)
    if pitcher.ndim!=1 or pitcher.dtype.kind not in 'iu' or (pitcher<=0).any() or fit9.dtype!=bool or fit9.shape!=pitcher.shape:raise ValueError('pitcher identity integrity')
    ids=np.unique(pitcher[fit9])
    if not len(ids) or len(ids)>=2**24:raise ValueError('FIT-only vocabulary bound')
    return ids


def indices(pitcher,ids):
    import numpy as np
    ids=np.asarray(ids);pitcher=np.asarray(pitcher)
    if ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu' or (ids<=0).any() or not (np.diff(ids)>0).all():raise ValueError('sorted positive vocabulary')
    where=np.searchsorted(ids,pitcher);known=(where<len(ids))&(ids[np.minimum(where,len(ids)-1)]==pitcher)
    return np.where(known,where+1,0).astype(np.int64)


def features(x,pitcher,ids):
    import numpy as np
    return np.column_stack((parent.features(x),indices(pitcher,ids).astype(np.float32)))


def recipe(producer,variant):
    if variant not in VARIANTS:raise ValueError('fixed identity variant')
    r=parent.recipe(producer,ARCH[variant]);r['pitcher_identity']={'variant':variant,'embedding_dim':16,'vocabulary':'sorted FIT_E9 only','unknown':0,'unknown_vector':'permanent zero','fusion':'additive low-rank pitcher logit offset; no state interaction','initial_offset':'zero','initial_rng':'preserved'}
    return r


def build_model(r,variant,seed,vocabulary_size):
    import torch
    backbone=parent.build_model(r,ARCH[variant],seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        embedding=torch.nn.Embedding(vocabulary_size+1,16,padding_idx=0)
        offset=torch.nn.Linear(16,9,bias=False);torch.nn.init.zeros_(offset.weight)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=backbone;self.embedding=embedding;self.offset=offset;self.seq=backbone.seq
        def forward(self,x,tok=None,row_keys=None):
            if x.ndim!=2 or x.shape[1]!=12:raise ValueError('state11 plus categorical index transport')
            code=x[:,11].long()
            return {'leaf':self.backbone(x[:,:11],tok,row_keys)['leaf']+self.offset(self.embedding(code))}
    return Model()


def run(producer,variant,seed,package,out,synthetic=False):
    return fit(recipe(producer,variant),variant,seed,package,out,synthetic=synthetic)


def run_control(producer,variant,seed,package,out,synthetic=False):
    # 원003 함수를 그대로 호출한다. 초기화/학습/예측/RNG를 복제하지 않는다.
    return parent.run(producer,ARCH[variant],seed,package,out,synthetic=synthetic)


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
    campaign.configure_backend({'arm':parent.VARIANTS[ARCH[variant]]['arm']},{});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not synthetic:torch.cuda.manual_seed_all(seed)
    device=torch.device('cpu' if synthetic else 'cuda')
    x,c,s=package;y=parent.targets(s['pitch_type']);ids=vocabulary(s['pitcher'],c['fit']&(y>=0));x=features(x,s['pitcher'],ids)
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
    model=build_model(r,variant,seed,len(ids)).to(device);opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
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
    if not torch.equal(model.embedding.weight[0],torch.zeros_like(model.embedding.weight[0])):raise ValueError('unknown vector changed')
    torch.save(model.state_dict(),out/'model-state.pt')
    from experiments.ws_fullseason_c.run import write_json,object_sha
    vocab={'ids':ids.tolist(),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'unknown_index':0,'embedding_dim':16}
    write_json(out/'pitcher-vocabulary.json',vocab)
    z=campaign.predict(model,data,keys,torch.as_tensor(oofrows,device=device,dtype=torch.long),t['eval_batch_size']);q=campaign.probabilities(z)
    if q.shape!=(len(oofrows),9) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-10,rtol=0):raise ValueError('9종 확률')
    np.savez(out/'oof-predictions.npz',keys=c['keys'][oofrows],probabilities=q)
    census={role:{'Q_pkg':int(mask.sum()),'E9_pkg':int((mask&(y>=0)).sum()),'raw_type_counts':dict(zip(parent.TYPES,np.bincount(s['pitch_type'][mask],minlength=19).astype(int).tolist()))} for role,mask in [('FIT',c['fit']),('OOF',oof)]}
    return {'status':'COMPLETED','arm':'PREPITCH9_ID','variant':variant,'seed':seed,'epochs':rule['epochs'],'consumption':trace,'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oof]),'census':census,'temperature':1.,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch','metrics_computed':False,'elapsed_compute_seconds':time.monotonic()-start,'backend':{'deterministic':True,'warn_only':True,'tf32':False,'dtype':'fp32'},'model_input':'state11, strictly earlier tokens, FIT-only pitcher ID offset','vocabulary_sha256':object_sha(vocab),'vocabulary_size':len(ids),'OOF_unknown_pitchers':int((indices(s['pitcher'][oof],ids)==0).sum()),'unknown_vector_zero':True,'identity_interaction':'additive logit offset only'}
