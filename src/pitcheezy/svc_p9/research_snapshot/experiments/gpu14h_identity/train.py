"""FIT-only 투수 ID의 저차원 logit offset; 원003 trunk/head/RNG를 보존한다."""
from pathlib import Path
import math
import time
from experiments.gpu14h_prepitch import train as parent

VARIANTS={'pitcher_mlp':{'architecture':'state_mlp','embedding_dim':16},'pitcher_t32':{'architecture':'state_t32','embedding_dim':16},'pitcher_c32':{'architecture':'state_c32','embedding_dim':16}}
ARCH={v:s['architecture'] for v,s in VARIANTS.items()}
# 학습 시 투수 ID 드롭아웃(EXP-MPS-P9B-001). p=0(기본)이면 아래 경로는 전혀 실행되지 않는다(원 학습과 비트 동치).
# 마스크는 CPU 전용 generator에서만 뽑는다: 전역 torch/np 난수(배치 순서 randperm, 초기화, nn.Dropout)를 소비하지 않는다.
ID_DROPOUT_SALT=7437
ID_DROPOUT_FORMULA='torch.Generator(device=cpu).manual_seed(seed*1000003+7437)'


def id_dropout_seed(seed):
    return seed*1000003+ID_DROPOUT_SALT


def hide_identity(xb,generator,p,idcol,known):
    """배치의 행마다 독립으로 확률 p로 투수 index(열 11)를 unknown 0으로 바꾼 새 텐서(torch.where; 입력 불변)를 만든다.
    반환: (가린 텐서, CPU 마스크 행 수, CPU 기대값 = 가린 행 중 원래 index>0, 장치에서 실제로 >0→0이 된 행 수(장치 텐서; 동기화 없음)).
    known은 이 배치 행의 원래 index>0 여부(CPU numpy)라서 기대값 계산에 장치 동기화가 필요 없다."""
    import torch
    hide=torch.rand(len(xb),generator=generator)<p
    out=torch.where(hide.to(xb.device)[:,None]&idcol[None,:],torch.zeros((),dtype=xb.dtype,device=xb.device),xb)
    applied=((xb[:,11]>0)&(out[:,11]==0)).sum()
    return out,int(hide.sum()),int((hide.numpy()&known).sum()),applied


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


def run(producer,variant,seed,package,out,synthetic=False,device='cuda',id_dropout=0.0,trace_recipe=False):
    return fit(recipe(producer,variant),variant,seed,package,out,synthetic=synthetic,device=device,id_dropout=id_dropout,trace_recipe=trace_recipe)


def run_control(producer,variant,seed,package,out,synthetic=False):
    # 원003 함수를 그대로 호출한다. 초기화/학습/예측/RNG를 복제하지 않는다.
    return parent.run(producer,ARCH[variant],seed,package,out,synthetic=synthetic)


def fit(r,variant,seed,package,out,synthetic=False,device='cuda',id_dropout=0.0,trace_recipe=False):
    import numpy as np
    import torch
    from experiments.ws_campaign import train as campaign
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_fullseason.train import check_epoch,resolve_device
    if seed not in (0,1,2) or variant not in VARIANTS:raise ValueError('고정후보/seed')
    if isinstance(id_dropout,bool) or not isinstance(id_dropout,(int,float)) or not 0<=id_dropout<=1:raise ValueError('id_dropout in [0,1]')
    id_dropout=float(id_dropout)
    requested=device;device=resolve_device(device,synthetic)  # 'cuda': 원 계약(synthetic→CPU, 아니면 CUDA 필수); 'mps': MPS 강제·fallback 거부
    rule=r['training_rule'];t=r['base']['training']
    if rule['early_stopping'] or rule['SELECT_rows'] or rule['CAL_rows'] or rule['temperature']!=1:raise ValueError('noSELECT/noCAL')
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    campaign.configure_backend({'arm':parent.VARIANTS[ARCH[variant]]['arm']},{});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if requested=='cuda' and not synthetic:torch.cuda.manual_seed_all(seed)
    if device.type=='mps':torch.mps.manual_seed(seed)
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
    if id_dropout:
        mask_generator=torch.Generator().manual_seed(id_dropout_seed(seed));masked=[];expected=[];on_device=[]
        idcol=torch.arange(x.shape[1],device=device)==11;known_row=x[:,11]>0
    for epoch in range(1,rule['epochs']+1):
        model.train();seen=np.zeros(len(x),np.int32);order=fitrows[torch.randperm(len(fitrows)).numpy()]
        if id_dropout:masked.append(0);expected.append(0);applied=torch.zeros((),dtype=torch.long,device=device)
        for offset in range(0,len(order),t['batch_size']):
            actual=order[offset:offset+t['batch_size']];idx=torch.as_tensor(actual,device=device,dtype=torch.long)
            xb=data.x[idx]
            if id_dropout:
                xb,hidden,want,got=hide_identity(xb,mask_generator,id_dropout,idcol,known_row[actual]);masked[-1]+=hidden;expected[-1]+=want;applied+=got
            z=model(xb,data.tokens(idx) if model.seq else None,keys[idx])
            loss=torch.nn.functional.cross_entropy(z['leaf'],target[idx]);campaign.update(loss,model,opt)
            np.add.at(seen,actual,1);steps+=1
        if id_dropout:
            on_device.append(int(applied.item()))  # epoch당 1회 장치→CPU 읽기
            if on_device[-1]!=expected[-1]:raise ValueError(f'id dropout not applied on device: epoch {epoch} expected {expected[-1]} got {on_device[-1]}')
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
    if device.type=='mps':torch.mps.synchronize()  # MPS 비동기: 시계 읽기 전 대기열 완료
    result={'status':'COMPLETED','arm':'PREPITCH9_ID','variant':variant,'seed':seed,'epochs':rule['epochs'],'consumption':trace,'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oof]),'census':census,'temperature':1.,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch','metrics_computed':False,'elapsed_compute_seconds':time.monotonic()-start,'backend':{'deterministic':True,'warn_only':True,'tf32':False,'dtype':'fp32'},'model_input':'state11, strictly earlier tokens, FIT-only pitcher ID offset','vocabulary_sha256':object_sha(vocab),'vocabulary_size':len(ids),'OOF_unknown_pitchers':int((indices(s['pitcher'][oof],ids)==0).sum()),'unknown_vector_zero':True,'identity_interaction':'additive logit offset only'}
    if id_dropout:
        # p>0일 때만 덧붙인다(p=0 receipt 스키마 불변). 평가(OOF 예측)는 가리지 않는다.
        result['id_dropout']={'p':id_dropout,'generator':{'device':'cpu','kind':'torch.Generator','seed':id_dropout_seed(seed),'formula':ID_DROPOUT_FORMULA},
                              'draw':'per optimizer step, before forward: torch.rand(batch_rows, generator=g) < p; hidden rows use unknown index 0 via torch.where (new tensor; data.x untouched)',
                              'device_check':'per epoch: rows with index>0 that are 0 after masking, counted on device, must equal CPU expectation (hidden & index>0)',
                              'masked_rows_by_epoch':masked,'masked_rows_total':int(sum(masked)),'masked_known_rows_expected_by_epoch':expected,'masked_rows_on_device_by_epoch':on_device,'presented_rows_total':int(len(fitrows)*rule['epochs']),'evaluation_masking':False}
    if trace_recipe:result['trained_recipe']=parent.trained_recipe(model,opt,rule,t,variant=variant,id_dropout=id_dropout if id_dropout else None)
    if device.type=='mps':
        # MPS 결과는 별도 차선: CUDA 결과와 합치지 않는다. 기본(cuda)/synthetic 스키마는 불변.
        import platform
        result['backend'].update(device='mps',mps_fallback_env=False,deterministic=False,deterministic_algorithms_requested=True,deterministic_guaranteed=False)
        result['versions']={'torch':torch.__version__,'numpy':np.__version__,'mps':{'macos':platform.mac_ver()[0],'machine':platform.machine(),'driver_allocated_bytes':int(torch.mps.driver_allocated_memory()),'current_allocated_bytes':int(torch.mps.current_allocated_memory())}}
    return result
