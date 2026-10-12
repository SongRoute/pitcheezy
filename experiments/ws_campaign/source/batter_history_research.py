"""Full-pool batter history and fixed type-conditioned shrinkage; exposed TRAIN."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from first_small_features import raw_features,profiles,encode
from history_transfer import KEYS,POOL_COLUMNS,load_queries_and_pool,richer_physics
from outcome_transfer import load_outcome_pool,transfer_features,NAMES,SETTINGS
from frequency_control import digest,apply_encoder
from support_analysis import checked_reference,verify_full
from first_small_models import additive,additive_logits,network,neural_logits,score,finite
from learning_diagnostic import train_curve,calibrate_and_score,paired,fit_budget,select_epoch,nll
from matched_additive import CPU
HISTORY_COLUMNS=KEYS+['game_date','batter','pitch_type','description','bb_type']
RATE_SPECS=[('swing',1,0),('contact_given_swing',2,1),('ground_given_bip',4,3)]
SHRINKAGE=20.

def behavior_values(f):
 return {'n':np.ones(len(f)),'swing':f.description.isin(['swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play','missed_bunt','foul_bunt','bunt_foul_tip']).to_numpy(float),'contact':f.description.isin(['foul','foul_tip','hit_into_play','foul_bunt','bunt_foul_tip']).to_numpy(float),'bip':f.description.eq('hit_into_play').to_numpy(float),'gb':f.bb_type.eq('ground_ball').to_numpy(float)}

def history_features(queries,pool,cutoff):
 if list(pool.columns)!=HISTORY_COLUMNS or pool.duplicated(KEYS).any():raise ValueError('History allowlist/keys')
 if pool.game_date.max()>pd.Timestamp(cutoff) or pool.game_date.min()<pd.Timestamp('2023-05-15'):raise ValueError('History date boundary')
 group=['game_date','batter','pitch_type'];combined=pd.concat([pool[group],queries[group]],ignore_index=True)
 if combined[group].isna().any().any():raise ValueError('Missing history grouping key')
 values={k:np.r_[v,np.zeros(len(queries))] for k,v in behavior_values(pool).items()};train=np.arange(len(combined))<len(pool)
 full=profiles(combined,train,['batter'],values)[len(pool):];typed=profiles(combined,train,['batter','pitch_type'],values)[len(pool):]
 a=[np.log1p(full[:,0])];b=[np.log1p(typed[:,0])];names=['batter_type_prior_log_count']
 for name,num,den in RATE_SPECS:
  prior=np.divide(full[:,num],full[:,den],out=np.full(len(queries),np.nan),where=full[:,den]>0)
  smoothed=(typed[:,num]+SHRINKAGE*prior)/(typed[:,den]+SHRINKAGE)
  a.extend([prior,full[:,den]==0]);b.extend([smoothed,typed[:,den]==0]);names.extend(['batter_type_prior_'+name,'batter_type_prior_'+name+'_missing'])
 return np.column_stack(a),np.column_stack(b),names,full,typed

def raw_variants(original,names,full,typed,type_names):
 columns=[i for i,n in enumerate(names) if n.startswith('batter_prior_')]
 if len(columns)!=7 or original.shape[1]!=91:raise ValueError('Original batter/core schema')
 f1=original.copy();f1[:,columns]=full;keep=np.setdiff1d(np.arange(91),columns)
 if digest(f1[:,keep])!=digest(original[:,keep]):raise ValueError('Other84 raw bytes changed')
 f2=np.column_stack([f1,typed])
 if digest(f2[:,:91])!=digest(f1):raise ValueError('F2 core changed')
 return {'F0':original,'F1':f1,'F2':f2},{'F0':names,'F1':names,'F2':names+type_names},{'changed_columns':columns,'other84_sha256':digest(original[:,keep]),'F1_core91_sha256':digest(f1)}

def build_cache(data,p,out):
 import pyarrow.parquet as pq
 from run_first_small_m4 import load,save,sha,source_hashes
 refs={k:checked_reference(v) for k,v in p['references'].items()};q=refs['43'][0]['protocol'];out=Path(out);out.mkdir(exist_ok=False);manifest={'projection_manifest_sha256':p['projection_manifest_sha256'],'projection_file_sha256':sha(data),'folds':{},'artifacts':{},'source_sha256':sha(__file__),'source_closure_sha256':source_hashes(),'classifier_fits':0,'temperature_fits':0,'fit_only_encoders':4,'inference':0};start=time.monotonic()
 for fold,c in q['folds'].items():
  cfg=dict(q,**c);f,_=load_queries_and_pool(data,cfg);pool=load_outcome_pool(data,cfg);tr=f.game_date.le(c['train_end']).to_numpy();tu=((f.game_date>c['train_end'])&(f.game_date<=c['tuning_end'])).to_numpy();ca=((f.game_date>=c['calibration_start'])&(f.game_date<=c['calibration_end'])).to_numpy();ev=(f.game_date>c['calibration_end']).to_numpy();masks=dict(train=tr,tuning=tu,calibration=ca,evaluation=ev)
  if not np.all(sum(m.astype(int) for m in masks.values())==1):raise ValueError('Disjoint exhaustive role masks')
  y=f.outcome10.to_numpy(dtype='<i8');keyproof={k:digest(f.loc[m,KEYS].to_numpy(dtype='<i8')) for k,m in masks.items()}
  if keyproof!=c['split_keys_sha256'] or digest(y)!=c['query_y_sha256']:raise ValueError('Frozen rows/labels changed')
  raw,names=raw_features(f,tr);rich=richer_physics(raw,names,f,pool[POOL_COLUMNS]);arrays,_=transfer_features(f,pool,SETTINGS);joined=np.column_stack([rich,arrays['generic']]);names=names+NAMES
  if digest(joined)!=c['raw_input_sha256'] or {k:digest(v) for k,v in arrays.items()}!=c['transfer_array_sha256']:raise ValueError('F0 raw/transfer identity')
  hist=pq.read_table(data,columns=HISTORY_COLUMNS,filters=[('game_date','<=',pd.Timestamp(c['train_end']))]).to_pandas();poolkey=digest(hist.sort_values(KEYS)[KEYS].to_numpy(dtype='<i8'))
  if len(hist)!=c['history_pool']['rows'] or poolkey!=c['history_pool']['keys_sha256']:raise ValueError('Historical outcome pool identity')
  full,typed,type_names,fc,tc=history_features(f,hist,c['train_end']);variants,vnames,invariance=raw_variants(joined,names,full,typed,type_names);folder=out/fold;folder.mkdir();np.savez(folder/'roles.npz',y=y,games=f.game_pk.to_numpy(),**masks);support={};designs={}
  for role,m in masks.items():
   support[role]={'rows':int(m.sum()),'games':int(f.loc[m,'game_pk'].nunique()),'class_counts':np.bincount(y[m],minlength=10).tolist(),'full_count_bins':np.histogram(fc[m,0],[-.5,.5,19.5,99.5,np.inf])[0].tolist(),'typed_count_bins':np.histogram(tc[m,0],[-.5,.5,19.5,99.5,np.inf])[0].tolist(),'full_rate_missing':(full[m][:,[2,4,6]]).sum(0).astype(int).tolist(),'type_rate_missing':(typed[m][:,[2,4,6]]).sum(0).astype(int).tolist()}
  for arm,v in variants.items():
   if arm=='F0':
    olddesign=load(refs['A'][1]/fold/'design.json')
    if olddesign!=load(refs['43'][1]/fold/'design.json'):raise ValueError('A/B reference design mismatch')
    enc=olddesign['encoder'];x=apply_encoder(v,enc)
    if {k:digest(x[m]) for k,m in masks.items()}!=c['encoded_input_sha256']:raise ValueError('F0 encoded identity')
   else:x,enc=encode(v,tr)
   finite(x);np.save(folder/f'{arm}.npy',x,allow_pickle=False);design={'features':vnames[arm],'encoder':enc,'encoded_columns':x.shape[1]};save(folder/f'{arm}-design.json',design);designs[arm]={'raw_sha256':digest(v),'encoded_split_sha256':{k:digest(x[m]) for k,m in masks.items()},'raw_columns':v.shape[1],'encoded_columns':x.shape[1]};del x
  value={'split_keys_sha256':keyproof,'y_sha256':digest(y),'pool_keys_sha256':poolkey,'pool_rows':len(hist),'cutoff':c['train_end'],'invariance':invariance,'arms':designs,'support':support};save(folder/'proof.json',value);manifest['folds'][fold]=value
  del variants,raw,rich,joined,arrays,pool,hist,f,full,typed,fc,tc
 manifest['elapsed_seconds']=time.monotonic()-start
 manifest['artifacts']={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()};save(out/'manifest.json',manifest);return manifest

def continuation_gate(folds):
 out={}
 for contrast in ['F1-F0','F2-F1']:
  out[contrast]={}
  for model in ['A','B']:
   cells=[folds[f]['contrasts'][contrast][model] for f in ['early','late']]
   # Paired helper reports mean; inequalities are fixed before scores.
   checks={'both_fold_nll_negative':all(v['nll']['mean']<0 for v in cells),'both_fold_brier_nonpositive':all(v['brier']['mean']<=0 for v in cells)}
   if contrast=='F2-F1':
    original=[folds[f]['contrasts']['F2-F0'][model] for f in ['early','late']]
    checks['F2_vs_F0_both_metrics_nonworse']=all(v['nll']['mean']<=0 and v['brier']['mean']<=0 for v in original)
   out[contrast][model]={'checks':checks,'eligible':all(checks.values()),'automatic_launch':False}
 return out

def check_contract(p):
 if p.get('bootstrap_seed')!=20261020:raise ValueError('Fixed analysis seed required')
 from run_first_small_m4 import sha,load,source_hashes
 if p['stage']!='batter_history_research' or p['experiment_id']!='EXP-M4-BATTER-HISTORY-001':raise ValueError('Identity')
 if p['gpu']!={'epochs':20,'batch_size':512,'width':64,'learning_rate':.001,'l2':.1,'seed':20261050} or p['scientific_seed']!=20261043 or p['cpu']!=CPU:raise ValueError('Fixed algorithms')
 if p['fits']!={'CPU':4,'MPS':4,'temperature':8} or p['wall_seconds']!=3000 or p['fit_timeout_seconds']!=300 or p['shrinkage']!=SHRINKAGE:raise ValueError('Scope/caps')
 if p['history_columns']!=HISTORY_COLUMNS or p['arms']!=['F1','F2'] or list(p['references'])!=['A','43']:raise ValueError('Data/arm contract')
 refs={k:checked_reference(v) for k,v in p['references'].items()};q=refs['43'][0]['protocol']
 if p['projection_manifest']!=q['projection_manifest'] or p['projection_manifest_sha256']!=q['projection_manifest_sha256']:raise ValueError('Projection identity')
 cache=Path(p['cache_manifest']);m=load(cache)
 if sha(cache)!=p['cache_manifest_sha256'] or m['source_sha256']!=sha(__file__) or m['source_closure_sha256']!=source_hashes():raise ValueError('Cache/source identity')
 if m['projection_manifest_sha256']!=p['projection_manifest_sha256'] or m['projection_file_sha256']!=load(p['projection_manifest'])['file_sha256']:raise ValueError('Cache projection identity')
 if list(m['folds'])!=['early','late']:raise ValueError('Cache fold inventory')
 for fold,cfg in q['folds'].items():
  proof=m['folds'][fold]
  if proof['split_keys_sha256']!=cfg['split_keys_sha256'] or proof['y_sha256']!=cfg['query_y_sha256'] or proof['pool_rows']!=cfg['history_pool']['rows'] or proof['pool_keys_sha256']!=cfg['history_pool']['keys_sha256'] or proof['cutoff']!=cfg['train_end']:raise ValueError('Cache fold reference identity')
  if proof['arms']['F0']['raw_sha256']!=cfg['raw_input_sha256'] or proof['arms']['F0']['encoded_split_sha256']!=cfg['encoded_input_sha256']:raise ValueError('Cache baseline input identity')
 for n,h in m['artifacts'].items():
  if Path(n).is_absolute() or '..' in Path(n).parts or sha(cache.parent/n)!=h:raise ValueError('Cache artifact changed')
 return refs,cache.parent,m

def run(data,p,out,trace):
 import torch
 from run_first_small_m4 import save,load
 refs,cache,manifest=check_contract(p);pending={};repro={};torch.set_num_threads(2);fits={'CPU':0,'MPS':0,'temperature':0};start=time.monotonic()
 if not torch.backends.mps.is_available():raise ValueError('No MPS fallback')
 # All four original models pass before the first new fit.
 for fold in ['early','late']:
  roles=np.load(cache/fold/'roles.npz',allow_pickle=False);y=roles['y'];ev=roles['evaluation'];x=np.load(cache/fold/'F0.npy',mmap_mode='r');a=refs['A'][2]['folds'][fold]['A'];b=refs['43'][2]['folds'][fold]['arms']['decay']['metrics'];w=np.load(refs['A'][1]/fold/'A.npy',allow_pickle=False);az=additive_logits(x[ev],w);am,al,ab=score(y[ev],az/a['temperature']);net=network(x.shape[1],64);net.load_state_dict(torch.load(refs['43'][1]/fold/'decay-B.pt',map_location='cpu',weights_only=True));net=net.to('mps');bz=neural_logits(net,x[ev]);bm,bl,bb=score(y[ev],bz/b['temperature']);repro[fold]={'A':verify_full(am,a),'B':verify_full(bm,b)};pending[fold]={'A':(a,al,ab,score(y[ev],az)[0]),'B':(b,bl,bb,score(y[ev],bz)[0])};del net,x;torch.mps.empty_cache()
 save(out/'F0-reproduction.json',repro);results={}
 try:
  for fold in ['early','late']:
   roles=np.load(cache/fold/'roles.npz',allow_pickle=False);y=roles['y'];tr,tu,ca,ev=[roles[k] for k in ['train','tuning','calibration','evaluation']];games=roles['games'][ev];folder=out/fold;folder.mkdir();values={'F0':{k:{'calibrated':v[0],'raw':v[3],'frozen_reuse':True} for k,v in pending[fold].items()}};losses={'F0':pending[fold]}
   for arm in p['arms']:
    x=np.load(cache/fold/f'{arm}.npy');design=load(cache/fold/f'{arm}-design.json');save(folder/f'{arm}-design.json',design);values[arm]={};losses[arm]={}
    for model in ['A','B']:
     if time.monotonic()-start>=p['wall_seconds']:raise TimeoutError('Overall cap')
     event=lambda v:trace({'fold':fold,'arm':arm,'model':model,**v});event({'event':model+'_start'})
     with fit_budget(p['fit_timeout_seconds']):
      if model=='A':w=additive(x[tr],y[tr],p['cpu'],event)
      else:net,curve=train_curve(x,y,tr,tu,dict(p['gpu'],seed=p['scientific_seed']),event)
     fits['CPU' if model=='A' else 'MPS']+=1
     if model=='A':np.save(folder/f'{arm}-A.npy',w,allow_pickle=False);cz=additive_logits(x[ca],w);ez=additive_logits(x[ev],w);extra={'train_nll':nll(y[tr],additive_logits(x[tr],w)),'tuning_nll':nll(y[tu],additive_logits(x[tu],w))}
     else:torch.save({k:v.cpu() for k,v in net.state_dict().items()},folder/f'{arm}-B.pt');save(folder/f'{arm}-curve.json',curve);cz=neural_logits(net,x[ca]);ez=neural_logits(net,x[ev]);extra={'selected_epoch':select_epoch(curve)}
     metric,ll,bs=calibrate_and_score(cz,ez,y[ca],y[ev]);rawmetric,_,_=score(y[ev],ez);fits['temperature']+=1;values[arm][model]={'calibrated':metric,'raw':rawmetric,**extra};losses[arm][model]=(metric,ll,bs)
     if model=='B':del net;torch.mps.empty_cache()
     event({'event':'model_complete'})
    del x
   contrasts={}
   for new,old in [('F1','F0'),('F2','F0'),('F2','F1')]:
    contrasts[new+'-'+old]={model:{'nll':paired(losses[new][model][1]-losses[old][model][1],games,p['bootstrap_seed']),'brier':paired(losses[new][model][2]-losses[old][model][2],games,p['bootstrap_seed'])} for model in ['A','B']}
   value={'models':values,'contrasts':contrasts};save(folder/'result.json',value);results[fold]=value
 except BaseException as exc:save(out/'partial-failure.json',{'error':repr(exc),'fits':fits,'completed_folds':list(results),'automatic_retry':False});raise
 if fits!=p['fits']:raise ValueError('Incomplete inventory')
 result={'folds':results,'fits':fits,'continuation':continuation_gate(results),'scientific_decision':None,'policy_decision':None,'automatic_next_stage':False};save(out/'result.json',result);return result
