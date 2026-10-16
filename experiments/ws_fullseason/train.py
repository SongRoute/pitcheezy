"""Fixed-epoch T only. The separately gated GPU runner owns all real I/O.

OOF is predicted only after the final epoch; no SELECT, calibration, or checkpoint
choice. Every optimizer batch records its actual indices for exact epoch coverage.
"""
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments.ws_campaign import train as campaign
from experiments.ws_campaign.models import build_model
from experiments.ws_seq import train_seq as base
from experiments.ws_fullseason.data import key_digest


def check_epoch(seen, fit, keys):
    if not np.array_equal(seen, fit.astype(np.int32)):
        raise ValueError('incomplete/duplicate/out-of-FIT optimizer consumption')
    return {'rows':int(fit.sum()),'ordered_keys_sha256':key_digest(keys[fit]),
            'every_key_once':True,'positive_parent_weight_every_row':True}


def run(recipe, seed, package, out, *, synthetic=False):
    if seed not in recipe['seeds'] or recipe['arm']!='T': raise ValueError('fixed T seed')
    if not synthetic and not torch.cuda.is_available(): raise ValueError('CUDA required')
    rule=recipe['training_rule']; t=recipe['base']['training']
    stage=recipe.get('execution_stage','OOF')
    if stage not in ('OOF','FINAL_REFIT'): raise ValueError('registered execution stage')
    final_refit=stage=='FINAL_REFIT'
    if rule['early_stopping'] or rule['SELECT_rows'] or rule['CAL_rows'] or rule['temperature']!=1:
        raise ValueError('fixed-epoch uncalibrated recipe required')
    torch.set_num_threads(2); torch.manual_seed(seed); np.random.seed(seed)
    campaign.configure_backend({'arm':'T'}, {})
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available() and not synthetic: torch.cuda.manual_seed_all(seed)
    device=torch.device('cpu' if synthetic else 'cuda')
    x,c,s=package
    if c['tuning'].any() or c['calibration'].any() or not np.array_equal(c['fit']|c['evaluation'],np.ones(len(x),bool)) or (c['fit']&c['evaluation']).any():
        raise ValueError('FIT/OOF only disjoint complete package')
    if len(np.unique(c['keys'],axis=0))!=len(x): raise ValueError('unique keys')
    ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],recipe['base']['context']['max_length'])
    base.check_context(ctx,s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'])
    valid=ctx>=0
    if not (c['fit'][np.maximum(ctx,0)]==c['fit'][:,None])[valid].all(): raise ValueError('context crosses role')
    tensors=base.Data(torch,device,x,c,s,ctx)
    keys=torch.as_tensor(c['keys'],device=device,dtype=torch.long)
    y7=torch.as_tensor(c['y7'],device=device,dtype=torch.long)
    fitrows=np.flatnonzero(c['fit']); oofrows=np.flatnonzero(c['evaluation'])
    if not len(fitrows) or (not final_refit and not len(oofrows)) or (final_refit and len(oofrows)):
        raise ValueError('OOF requires two roles; FINAL_REFIT requires FIT only')
    model=build_model(recipe,'T',seed).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
    trace=[]; start=time.monotonic(); steps=0
    for epoch in range(1,rule['epochs']+1):
        model.train(); seen=np.zeros(len(x),np.int32)
        order=fitrows[torch.randperm(len(fitrows)).numpy()]
        for offset in range(0,len(order),t['batch_size']):
            actual=order[offset:offset+t['batch_size']]
            idx=torch.as_tensor(actual,device=device,dtype=torch.long)
            z=model(tensors.x[idx],tensors.tokens(idx),keys[idx])
            loss,_=campaign.losses(z,tensors,idx,y7)
            campaign.update(loss,model,opt)
            np.add.at(seen,actual,1); steps+=1
        evidence=check_epoch(seen,c['fit'],c['keys'])
        evidence.update(epoch=epoch,optimizer_steps=math.ceil(len(fitrows)/t['batch_size']),
            whiff_head_eligible_rows=int((c['fit']&(c['y10']==1)).sum()),
            ground_head_eligible_rows=int((c['fit']&np.isin(c['y10'],base.BIP)).sum()))
        trace.append(evidence)
    out=Path(out)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('nonfinite final checkpoint')
    torch.save(model.state_dict(),out/'model-state.pt')
    if not final_refit:
        oof=torch.as_tensor(oofrows,device=device,dtype=torch.long)
        z=campaign.predict(model,tensors,keys,oof,t['eval_batch_size'])
        q=campaign.probabilities(z)
        if q.shape!=(len(oofrows),7) or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1,atol=1e-7):
            raise ValueError('complete finite normalized OOF probabilities')
        np.savez(out/'oof-predictions.npz',keys=c['keys'][oofrows],probabilities=q)
    return {'status':'COMPLETED','arm':'T','seed':seed,'epochs':rule['epochs'],
        'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),
        'consumption':trace,'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oofrows]),
        'temperature':1.0,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch',
        'metrics_computed':False,'execution_stage':stage,'elapsed_compute_seconds':time.monotonic()-start,
        'backend':{'deterministic':True,'warn_only':True,'cudnn_benchmark':False,'tf32':False,'dtype':'fp32'},
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}}
