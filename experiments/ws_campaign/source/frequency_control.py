"""Direct strict-past frequency controls; no classifier or encoder fit."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.special import logsumexp
from first_small_features import raw_features,PHYSICS
from first_small_models import additive_logits,network,neural_logits,score,finite
from history_transfer import load_queries_and_pool,richer_physics,POOL_COLUMNS
from outcome_transfer import load_outcome_pool,transfer_features,NAMES
from temporal_sharing import fold_protocol,check_contract as check_temporal
from learning_diagnostic import calibrate_and_score,paired

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def check_contract(p):
 if p['experiment_id']!='EXP-M4-FREQUENCY-CONTROL-001' or p['gpu']!={'epochs':0,'seed':20261038} or p['cpu']!={'maxiter':0}:raise ValueError('Frequency identity; classifiers forbidden')
 if p['wall_seconds']!=600 or p['fits']!={'CPU':0,'MPS':0,'temperature':4} or p['controls']!=['weighted','pooled']:raise ValueError('Fixed frequency inventory/budget')

def apply_encoder(raw,e):
 # Apply only stored statistics; do not call encode() or refit normalization.
 median=np.array(e['median']);mean=np.array(e['mean']);scale=np.array(e['scale']);cols=np.array(e['continuous'],int)
 z=np.clip((np.where(np.isfinite(raw),raw,median)-mean)/scale,-8,8)
 return np.column_stack([z]+[np.maximum(z[:,cols]-k,0) for k in np.array(e['knots'])]).astype(np.float32)

def pooled_frequencies(queries,pool):
 """Same complete-row/context/other-pitcher set; pooled counts, total prior1."""
 pool=pool.loc[np.isfinite(pool[list(PHYSICS)].to_numpy(float)).all(1)]
 context=['pitch_type','p_throws','stand','balls','strikes'];days={d:g for d,g in pool.groupby('game_date',sort=False)};dates=sorted(days);cursor=0;totals={};own={};out=np.empty((len(queries),10));support=np.zeros(len(queries),np.int64)
 for date,indices in queries.groupby('game_date',sort=True).indices.items():
  while cursor<len(dates) and dates[cursor]<date:
   for key,h in days[dates[cursor]].groupby(context+['pitcher'],sort=False):
    counts=np.bincount(h.outcome10.to_numpy(int),minlength=10);totals[key[:-1]]=totals.get(key[:-1],np.zeros(10))+counts;own[key]=own.get(key,np.zeros(10))+counts
   cursor+=1
  cache={}
  for i in indices:
   q=queries.iloc[i];key=(q.pitch_type,q.p_throws,q.stand,int(q.balls),int(q.strikes));receiver=key+(int(q.pitcher),)
   if receiver not in cache:
    counts=totals.get(key,np.zeros(10))-own.get(receiver,np.zeros(10));n=counts.sum()
    if n<0 or np.any(counts<0):raise ValueError('Negative donor support')
    cache[receiver]=((counts+.1)/(n+1),int(n))
   out[i],support[i]=cache[receiver]
 finite(out);return out,support

def frequency_inputs(frame,pool,transfer):
 weighted=transfer['generic'][:,12:22].copy();missing=transfer['generic'][:,23]==1;weighted[missing]=.1
 pooled,n=pooled_frequencies(frame,pool)
 expected=np.rint(np.expm1(transfer['generic'][:,22])).astype(np.int64)
 if not np.array_equal(n,expected) or not np.array_equal(missing,n==0):raise ValueError('Frequency controls have different eligible donors')
 for v in [weighted,pooled]:
  if not np.isfinite(v).all() or np.any(v<=0) or not np.allclose(v.sum(1),1,atol=1e-12,rtol=0):raise ValueError('Invalid frequency distribution')
 if not np.array_equal(weighted[~missing],transfer['generic'][~missing,12:22]):raise ValueError('Weighted identity')
 return {'weighted':weighted,'pooled':pooled},n

def support_bins(n):
 return {'0':int((n==0).sum()),'1-19':int(((n>=1)&(n<20)).sum()),'20-99':int(((n>=20)&(n<100)).sum()),'100+':int((n>=100).sum())}

def verify_base(p):
 from run_first_small_m4 import load,sha,ROOT
 b=p['base'];reg=load(b['registration']);base=Path(b['attempt']);terminal=load(base/'terminal.json')
 if sha(b['registration'])!=b['registration_sha256'] or sha(base/'terminal.json')!=b['terminal_sha256']:raise ValueError('Base registration/terminal changed')
 if not terminal['success'] or terminal['returncode']!=0:raise ValueError('Base not complete')
 for name,h in terminal['artifacts'].items():
  if Path(name).is_absolute() or '..' in Path(name).parts or sha(base/name)!=h:raise ValueError('Base artifact changed')
 if reg['source_commit']!='47af8e6494b289d3fa44773539ad26613dfbf4c8' or reg['status']!='REGISTERED':raise ValueError('Wrong base source')
 for name,h in reg['source_sha256'].items():
  if sha(Path(reg['stage'])/name)!=h:raise ValueError('Frozen base source changed')
  if not name.endswith('run_first_small_m4.py') and sha(ROOT/name)!=h:raise ValueError('Scientific helper changed')
 reviewed=load(b['result_review'])
 if sha(b['result_review'])!=b['result_review_sha256'] or reviewed['verdict']!='PASS' or reviewed['terminal_sha256']!=b['terminal_sha256']:raise ValueError('Base independent result review')
 q=reg['protocol'];check_temporal(q)
 if p['calibration_keys_sha256']!={f:q['folds'][f]['split_keys_sha256']['calibration'] for f in ['early','late']}:raise ValueError('Calibration identity changed')
 if p['projection_manifest']!=q['projection_manifest'] or p['projection_manifest_sha256']!=q['projection_manifest_sha256']:raise ValueError('Different data')
 return q,base,load(base/'result.json')

def run(datafile,p,out,trace):
 import torch
 from run_first_small_m4 import save,load
 check_contract(p);basep,base,original=verify_base(p);results={};temperatures=0
 try:
  for fold in ['early','late']:
   cfg=fold_protocol(basep,fold);frame,_=load_queries_and_pool(datafile,cfg);pool=load_outcome_pool(datafile,cfg)
   train=frame.game_date.le(cfg['train_end']).to_numpy();tune=((frame.game_date>cfg['train_end'])&(frame.game_date<=cfg['tuning_end'])).to_numpy();cal=((frame.game_date>=cfg['calibration_start'])&(frame.game_date<=cfg['calibration_end'])).to_numpy();ev=(frame.game_date>cfg['calibration_end']).to_numpy();masks=dict(train=train,tuning=tune,calibration=cal,evaluation=ev);y=frame.outcome10.to_numpy(dtype='<i8')
   if digest(y)!=cfg['query_y_sha256']:raise ValueError('Query labels changed')
   for role,m in masks.items():
    if digest(frame.loc[m,['game_pk','at_bat_number','pitch_number']].to_numpy(dtype='<i8'))!=cfg['split_keys_sha256'][role]:raise ValueError('Split changed')
   raw,names=raw_features(frame,train);rich=richer_physics(raw,names,frame,pool[POOL_COLUMNS]);transfer,_=transfer_features(frame,pool,cfg['transfer_settings'])
   for arm,a in transfer.items():
    if digest(a)!=cfg['transfer_array_sha256'][arm]:raise ValueError('Transfer input changed')
   probs,support=frequency_inputs(frame,pool,transfer);games=frame.loc[ev,'game_pk'].to_numpy();losses={};metrics={};saved_probs={};reproduction={}
   # Reproduce every old model before new scalar calibration in this fold.
   for arm in ['own','generic']:
    design=load(base/fold/(arm+'-design.json'))
    if design['features']!=names+NAMES or design['removed']:raise ValueError('Feature names changed')
    x=apply_encoder(np.column_stack([rich,transfer[arm]])[ev],design['encoder'])
    if digest(x)!=cfg['encoded_input_sha256'][arm]['evaluation']:raise ValueError('Encoded rows changed')
    a=np.load(base/fold/(arm+'-A.npy'),allow_pickle=False);torch.set_num_threads(2)
    if not torch.backends.mps.is_available():raise ValueError('MPS required')
    b=network(x.shape[1],cfg['gpu']['width']);b.load_state_dict(torch.load(base/fold/(arm+'-B.pt'),map_location='cpu',weights_only=True));b=b.to('mps')
    for model,z in [('A',additive_logits(x,a)),('B',neural_logits(b,x))]:
     old=original['folds'][fold]['variants'][arm]['models'][model];zz=z/old['temperature'];metric,ll,bs=score(y[ev],zz);key=arm+'_'+model
     if any(abs(metric[k]-old[k])>1e-6 for k in ['nll','brier']):raise ValueError('Original metric mismatch')
     metric['temperature']=old['temperature'];metrics[key]=metric;losses[key]=(ll,bs);saved_probs[key]=np.exp(zz-logsumexp(zz,axis=1,keepdims=True));reproduction[key]={k:float(metric[k])-float(old[k]) for k in ['nll','brier']}
    del b;torch.mps.empty_cache()
   for control,v in probs.items():
    rawmetric,ll,bs=score(y[ev],np.log(v[ev]));metrics[control+'_raw']=rawmetric;losses[control+'_raw']=(ll,bs)
    trace({'fold':fold,'event':'temperature_start','control':control,'calibration_rows':int(cal.sum()),'calibration_key_sha256':cfg['split_keys_sha256']['calibration']})
    metric,ll,bs=calibrate_and_score(np.log(v[cal]),np.log(v[ev]),y[cal],y[ev]);temperatures+=1;metrics[control+'_calibrated']=metric;losses[control+'_calibrated']=(ll,bs)
    trace({'fold':fold,'event':'temperature_complete','control':control,'temperature':metric['temperature']})
   contrasts={}
   for left,right in [('weighted_raw','pooled_raw'),('weighted_calibrated','pooled_calibrated'),('generic_A','weighted_calibrated'),('generic_B','weighted_calibrated')]:
    contrasts[left+'_minus_'+right]={k:paired(losses[left][i]-losses[right][i],games,basep['bootstrap_seed']) for i,k in enumerate(['nll','brier'])}
   wlog=np.log(probs['weighted'][ev])/metrics['weighted_calibrated']['temperature'];wlog-=logsumexp(wlog,axis=1,keepdims=True);kl={k:float(np.sum(v*(np.log(np.maximum(v,1e-300))-wlog),axis=1).mean()) for k,v in saved_probs.items() if k.startswith('generic_')}
   value={'models':metrics,'paired_contrasts':contrasts,'mean_KL_saved_generic_to_weighted_calibrated':kl,'old_reproduction':reproduction,'evaluation_rows':int(ev.sum()),'evaluation_games':len(np.unique(games)),'donor_absent':int((support[ev]==0).sum()),'donor_support_bins':{role:support_bins(support[m]) for role,m in [('calibration',cal),('evaluation',ev)]},'calibration_key_sha256':cfg['split_keys_sha256']['calibration'],'frequency_probability_sha256':{k:{role:digest(v[m]) for role,m in [('calibration',cal),('evaluation',ev)]} for k,v in probs.items()},'scientific_decision':None,'policy_decision':None}
   save(out/(fold+'-result.json'),value);results[fold]=value;del frame,pool,raw,rich,transfer,probs
 except BaseException as exc:
  save(out/'partial-failure.json',{'error':repr(exc),'completed_folds':list(results),'scalar_temperature_fits':temperatures,'classifier_fits':0,'followup_stopped':True});raise
 result={'folds':results,'fits':{'CPU':0,'MPS':0,'temperature':temperatures},'classifiers_replayed':8,'scientific_decision':None,'policy_decision':None,'automatic_next_stage':False,'exposure':'Repeated exposed TRAIN; explanation control, not independent confirmation'}
 if temperatures!=4:raise ValueError('Incomplete scalar inventory')
 save(out/'result.json',result);return result
