"""새 9종 표적. 원 패키지는 불변이며 모델에는 state11과 엄격한 이전 token만 전달한다."""
from copy import deepcopy
from pathlib import Path
import math
import time

TYPES=('UNKNOWN','CH','CS','CU','EP','FA','FC','FF','FO','FS','KC','KN','PO','SC','SI','SL','ST','SV','UN')
CLASSES=('FF','SI','FC','SL','ST','CU','CH','FS','OT')
GROUPS={'FF':('FF','FA'),'SI':('SI',),'FC':('FC',),'SL':('SL','SV'),'ST':('ST',),'CU':('CU','KC','CS'),'CH':('CH',),'FS':('FS','FO'),'OT':('KN','SC','EP')}
VARIANTS={'state_mlp':{'arm':'N0'},'state_t32':{'arm':'T'},'state_c32':{'arm':'C'}}


def targets(codes):
    import numpy as np
    codes=np.asarray(codes)
    if codes.dtype.kind not in 'iu' or codes.min()<0 or codes.max()>=len(TYPES):raise ValueError('고정19코드 범위')
    lookup=np.array([next((i for i,g in enumerate(CLASSES) if t in GROUPS[g]),-1) for t in TYPES],np.int64)
    return lookup[codes]


def features(x):
    import numpy as np
    if x.ndim!=2 or x.shape[1]!=199 or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('원199 feature 계약')
    return np.ascontiguousarray(x[:,:11])


def recipe(producer,variant):
    if variant not in VARIANTS or producer['arm']!='T' or producer['seeds']!=[0,1,2] or producer['training_rule']['epochs']!=30 or producer['base']['inputs']['feature_dimension']!=199 or producer['base']['context']['max_length']!=32:raise ValueError('원T producer 및고정후보')
    r=deepcopy(producer);r['base']['inputs']['feature_dimension']=11
    r['prepitch']={'classes':list(CLASSES),'variant':variant,'state_columns':list(range(11)),'loss':'9class cross entropy','target':'current pitch type; historical outcome-eligible package only'}
    return r


def build_model(r,variant,seed):
    import torch
    from experiments.ws_campaign.models import build_model as accepted_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone=accepted_model(r,VARIANTS[variant]['arm'],seed)
        del backbone.net.head_parent,backbone.net.head_whiff,backbone.net.head_gb
        head=torch.nn.Linear(r['base']['model']['trunk_hidden'][-1]+(r['base']['model']['token_dim'] if backbone.seq else 0),9)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=backbone;self.head=head;self.seq=backbone.seq
        def forward(self,x,tok=None,row_keys=None):
            h=self.backbone.net.trunk(x)
            if self.seq:
                _,p=self.backbone.encode(self.backbone.prepare_tokens(tok,row_keys));h=torch.cat((h,p),1)
            return {'leaf':self.head(h)}
    return Model()


def run(producer,variant,seed,package,out,synthetic=False):
    return fit(recipe(producer,variant),variant,seed,package,out,synthetic=synthetic)


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
    campaign.configure_backend({'arm':VARIANTS[variant]['arm']},{});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not synthetic:torch.cuda.manual_seed_all(seed)
    device=torch.device('cpu' if synthetic else 'cuda')
    x,c,s=package;x=features(x);y=targets(s['pitch_type'])
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
    model=build_model(r,variant,seed).to(device);opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
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
    torch.save(model.state_dict(),out/'model-state.pt')
    z=campaign.predict(model,data,keys,torch.as_tensor(oofrows,device=device,dtype=torch.long),t['eval_batch_size']);q=campaign.probabilities(z)
    if q.shape!=(len(oofrows),9) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-10,rtol=0):raise ValueError('9종 확률')
    np.savez(out/'oof-predictions.npz',keys=c['keys'][oofrows],probabilities=q)
    census={role:{'Q_pkg':int(mask.sum()),'E9_pkg':int((mask&(y>=0)).sum()),'raw_type_counts':dict(zip(TYPES,np.bincount(s['pitch_type'][mask],minlength=19).astype(int).tolist()))} for role,mask in [('FIT',c['fit']),('OOF',oof)]}
    return {'status':'COMPLETED','arm':'PREPITCH9','variant':variant,'seed':seed,'epochs':rule['epochs'],'consumption':trace,'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oof]),'census':census,'temperature':1.,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch','metrics_computed':False,'elapsed_compute_seconds':time.monotonic()-start,'backend':{'deterministic':True,'warn_only':True,'tf32':False,'dtype':'fp32'},'model_input':'state11 and strictly earlier tokens only'}
