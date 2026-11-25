"""EXP-MPS-PB-001: gpu14h_batter.train.fit(원 008 타자 ID 학습기)의 Apple MPS 판. 원 파일은 바꾸지 않는다.

원 학습기는 CUDA 전용이다(`if not synthetic and not torch.cuda.is_available(): raise`, `device=torch.device('cpu' if synthetic else 'cuda')`).
아래 fit은 원 fit 본문을 글자 그대로 옮기고, 투수 ID 학습기 gpu14h_identity.train.fit이 MPS 차선용으로 이미 쓰는 장치 줄만
그대로 가져다 바꿨다(test_run이 두 원문과 글자 단위로 대조한다):
  1. 서명에 device='cuda', trace_recipe=False 추가(identity.fit과 같은 서명 꼬리)
  2. check_epoch import 줄에 resolve_device 추가
  3. 'CUDA required' 줄 -> identity.fit의 resolve_device 줄('mps': MPS 강제, PYTORCH_ENABLE_MPS_FALLBACK 거부, CPU 대체 없음)
  4. cuda seed 줄 + device 줄 -> identity.fit의 cuda seed 줄 + torch.mps.manual_seed 줄
  5. return {...} -> identity.fit의 torch.mps.synchronize 줄 + result={...}, 뒤에 trained_recipe(trace_recipe일 때)와
     identity.fit의 MPS backend/versions 블록 다섯 줄, return result
모델·자료·어휘·초기화·학습 순서·예측·저장 파일은 원 fit과 같다(VARIANTS, vocabularies, features, build_model, unknown_zero,
recipe, identity, parent는 원 모듈 객체 그 자체를 import한다).
"""
from pathlib import Path
import math
import time
from experiments.gpu14h_batter.train import VARIANTS, CONTROL, vocabularies, features, recipe, build_model, unknown_zero, identity, parent

SOURCE = 'experiments/gpu14h_batter/train.py:fit'
DEVICE_LINES_FROM = 'experiments/gpu14h_identity/train.py:fit'


def fit(r,variant,seed,package,out,synthetic=False,device='cuda',trace_recipe=False):
    import numpy as np
    import torch
    from experiments.ws_campaign import train as campaign
    from experiments.ws_seq import train_seq as base
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_fullseason.train import check_epoch,resolve_device
    if seed not in (0,1,2) or variant not in VARIANTS:raise ValueError('고정후보/seed')
    requested=device;device=resolve_device(device,synthetic)  # 'cuda': 원 계약(synthetic→CPU, 아니면 CUDA 필수); 'mps': MPS 강제·fallback 거부
    rule=r['training_rule'];t=r['base']['training']
    if rule['early_stopping'] or rule['SELECT_rows'] or rule['CAL_rows'] or rule['temperature']!=1:raise ValueError('noSELECT/noCAL')
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    campaign.configure_backend({'arm':'T'},{});torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if requested=='cuda' and not synthetic:torch.cuda.manual_seed_all(seed)
    if device.type=='mps':torch.mps.manual_seed(seed)
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
    if device.type=='mps':torch.mps.synchronize()  # MPS 비동기: 시계 읽기 전 대기열 완료
    result={'status':'COMPLETED','arm':'PREPITCH9_BATTER','variant':variant,'seed':seed,'epochs':rule['epochs'],'consumption':trace,'optimization_steps':steps,'sample_presentations':int(len(fitrows)*rule['epochs']),'FIT_E9_keys_sha256':key_digest(c['keys'][fit]),'OOF_rows':len(oofrows),'OOF_keys_sha256':key_digest(c['keys'][oof]),'census':census,'temperature':1.,'calibration_fits':0,'SELECT_queries':0,'checkpoint':'final epoch','metrics_computed':False,'elapsed_compute_seconds':time.monotonic()-start,'backend':{'deterministic':True,'warn_only':True,'tf32':False,'dtype':'fp32'},'model_input':'state11, strictly earlier tokens, separate FIT-only additive ID factors','vocabulary_sha256':object_sha(vocab),'vocabulary_sizes':{k:len(v) for k,v in ids.items()},'OOF_unknown':{k:int((identity.indices(s[k][oof],v)==0).sum()) for k,v in ids.items()},'unknown_vector_zero':True,'identity_interaction':'separate additive factorized offsets; no ID x state or P x B interaction'}
    if trace_recipe:result['trained_recipe']=parent.trained_recipe(model,opt,rule,t,variant=variant)
    if device.type=='mps':
        # MPS 결과는 별도 차선: CUDA 결과와 합치지 않는다. 기본(cuda)/synthetic 스키마는 불변.
        import platform
        result['backend'].update(device='mps',mps_fallback_env=False,deterministic=False,deterministic_algorithms_requested=True,deterministic_guaranteed=False)
        result['versions']={'torch':torch.__version__,'numpy':np.__version__,'mps':{'macos':platform.mac_ver()[0],'machine':platform.machine(),'driver_allocated_bytes':int(torch.mps.driver_allocated_memory()),'current_allocated_bytes':int(torch.mps.current_allocated_memory())}}
    return result
