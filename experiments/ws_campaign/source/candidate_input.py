"""Small pre-pitch input reconstruction census; no prediction or action selection."""
from pathlib import Path
import hashlib,resource
import numpy as np
import pandas as pd
from first_small_features import TYPES,PHYSICS,raw_features
from history_transfer import KEYS,POOL_COLUMNS,richer_physics,validate_pool
from outcome_transfer import EXTRA,SETTINGS,NAMES,transfer_features
from batter_history_research import HISTORY_COLUMNS,history_features
from frequency_control import apply_encoder,digest
from learning_diagnostic import split
CONTEXT=KEYS+['game_date','pitcher','batter','pitch_type','balls','strikes','outs_when_up','inning','inning_topbot','home_score','away_score','on_1b','on_2b','on_3b','stand','p_throws']
DONOR=list(dict.fromkeys(POOL_COLUMNS+EXTRA+HISTORY_COLUMNS))
SCHEMA=CONTEXT+['previous_type','previous_valid']
ACTIONS=[x for x in TYPES if x not in ('UNKNOWN','UN')]
VARIABLE_RAW=list(range(11,30))+list(range(50,59))+list(range(66,91))

def schema_frame(query):
 if list(query.columns)!=SCHEMA:raise ValueError('Narrow pre-pitch query schema only')
 if query[KEYS+['game_date','pitcher','batter','pitch_type','stand','p_throws']].isna().any().any():raise ValueError('Missing query identifiers')
 if not query.previous_valid.isin([True,False]).all() or not query.pitch_type.isin(TYPES).all() or not query.previous_type.isin(TYPES).all():raise ValueError('Query type/previous schema')
 f=query[CONTEXT].copy()
 for c in PHYSICS:f[c]=np.nan
 f['description']='';f['bb_type']=''
 raw,names=raw_features(f,np.zeros(len(f),bool));lookup={t:i for i,t in enumerate(TYPES)}
 raw[:,30:49]=np.eye(len(TYPES))[query.previous_type.map(lookup).to_numpy(int)]*query.previous_valid.to_numpy()[:,None];raw[:,49]=query.previous_valid.to_numpy()
 return f,raw,names

def construct(query,pool,cutoff,encoder):
 f,raw,names=schema_frame(query)
 if list(pool.columns)!=DONOR or pool.game_date.max()>pd.Timestamp(cutoff):raise ValueError('Frozen donor allowlist/cutoff')
 raw=richer_physics(raw,names,f,pool[POOL_COLUMNS]);arrays,_=transfer_features(f,pool[POOL_COLUMNS+EXTRA],SETTINGS)
 joined=np.column_stack([raw,arrays['generic']]);full,*_=history_features(f,pool[HISTORY_COLUMNS],cutoff);joined[:,59:66]=full
 return apply_encoder(joined,encoder),joined

def varying_columns(encoder):
 continuous=encoder['continuous'];columns=VARIABLE_RAW.copy()
 for block in range(3):columns += [91+block*len(continuous)+i for i,k in enumerate(continuous) if k in VARIABLE_RAW]
 return sorted(columns)

def query_context(frame):
 f=frame.copy();keys=list(map(tuple,f[KEYS].to_numpy()));lookup={k:t for k,t in zip(keys,f.pitch_type)}
 prev=[lookup.get((k[0],k[1],k[2]-1),'UNKNOWN') for k in keys];valid=[(k[0],k[1],k[2]-1) in lookup for k in keys]
 f['previous_type']=prev;f['previous_valid']=valid
 return f[SCHEMA]

def select_rows(query,evaluation):
 rows=[]
 for valid in (False,True):
  eligible=np.flatnonzero(evaluation & (query.previous_valid.to_numpy()==valid) & ((query.pitch_number.to_numpy()==1) if not valid else True));rank=sorted(eligible,key=lambda i:hashlib.sha256((':'.join(map(str,query.iloc[i][KEYS]))).encode()).hexdigest());used=set();picked=[]
  for i in rank:
   game=int(query.iloc[i].game_pk)
   if game not in used:picked.append(int(i));used.add(game)
   if len(picked)==8:break
  rows.extend(picked)
 if len(rows)!=16:raise ValueError('Insufficient fixed first/previous strata')
 return np.asarray(rows,dtype=np.int64)

def check_contract(p):
 from run_first_small_m4 import load,sha
 if p['stage']!='candidate_input' or p['experiment_id']!='PREP-M4-CANDIDATE-INPUT-001' or p['gpu']!={'epochs':0,'seed':20261096} or p['fits']!={'CPU':0,'MPS':0,'temperature':0} or p['cpu']!={'maxiter':0}:raise ValueError('Input-only inventory')
 if p['context_columns']!=CONTEXT or p['donor_columns']!=DONOR or p['query_schema']!=SCHEMA or p['actions']!=ACTIONS or p['rows_per_fold']!=16 or p['wall_seconds']!=600 or p['fit_only_encoders']!=0:raise ValueError('Fixed input scope')
 if p['launch_by']!='2026-10-08T08:10:00+09:00' or p['kill_by']!='2026-10-08T08:25:00+09:00':raise ValueError('Clock')
 if sha(p['reference_registration'])!=p['reference_registration_sha256'] or sha(p['cache_manifest'])!=p['cache_manifest_sha256']:raise ValueError('Frozen source reference')
 manifest=load(p['projection_manifest'])
 if not set(CONTEXT+DONOR).issubset(manifest['schema']):raise ValueError('Columns outside approved projection')
 q=load(p['reference_registration'])['protocol'];m=load(p['cache_manifest']);root=Path(p['cache_manifest']).parent
 if m['projection_manifest_sha256']!=p['projection_manifest_sha256']:raise ValueError('Projection/cache identity')
 for fold in ('early','late'):
  for n in ('F1.npy','F1-design.json','proof.json'):
   key=fold+'/'+n
   if sha(root/key)!=m['artifacts'][key]:raise ValueError('F1 cache changed')
 return q,root,m

def run(data,p,out,trace):
 import pyarrow.parquet as pq
 from run_first_small_m4 import save,load
 q,cache,manifest=check_contract(p);out=Path(out);meta=pq.read_table(data,columns=CONTEXT).to_pandas()
 if len(meta)!=1252824 or meta.duplicated(KEYS).any():raise ValueError('Approved rows/keys')
 folds={}
 for fold,c in q['folds'].items():
  frame,*masks=split(meta,dict(q,**c));query=query_context(frame);ev=masks[-1]
  if digest(frame.loc[ev,KEYS].to_numpy(dtype='<i8'))!=c['split_keys_sha256']['evaluation']:raise ValueError('Ordered EVAL role')
  proof=load(cache/fold/'proof.json')
  if any(digest(frame.loc[mask,KEYS].to_numpy(dtype='<i8'))!=proof['split_keys_sha256'][role] for role,mask in zip(('train','tuning','calibration','evaluation'),masks)):raise ValueError('All selected FIT/role keys changed')
  idx=select_rows(query,ev);small=query.iloc[idx].reset_index(drop=True);pool=pq.read_table(data,columns=DONOR,filters=[('game_date','<=',pd.Timestamp(c['train_end']))]).to_pandas();validate_pool(pool[POOL_COLUMNS],dict(q,**c))
  if not pool.outcome10.between(0,9).all() or not pool.balls.between(0,3).all() or not pool.strikes.between(0,2).all() or not pool.stand.isin(['L','R']).all() or not pool.p_throws.isin(['L','R']).all():raise ValueError('Historical donor schema')
  design=load(cache/fold/'F1-design.json');encoder=design['encoder'];observed,raw=construct(small,pool,c['train_end'],encoder);old=np.load(cache/fold/'F1.npy',mmap_mode='r')[idx]
  if observed.shape!=(16,199) or digest(observed)!=digest(old):raise ValueError('Observed narrow query199 byte reconstruction failed')
  changed=varying_columns(encoder);fixed=np.setdiff1d(np.arange(199),changed)
  if len(changed)!=134 or len(fixed)!=65:raise ValueError('Candidate coordinate geometry')
  candidates=pd.concat([small.assign(pitch_type=t) for t in ACTIONS],ignore_index=True);x,_=construct(candidates,pool,c['train_end'],encoder)
  if digest(x[:,fixed])!=digest(np.tile(observed[:,fixed],(len(ACTIONS),1))):raise ValueError('Candidate invariant65 changed')
  folder=out/fold;folder.mkdir();np.savez(folder/'inputs.npz',keys=small[KEYS].to_numpy(dtype='<i8'),observed=observed,candidates=x,actions=np.asarray(ACTIONS),varying=np.asarray(changed),fixed=fixed,selected_indices=idx)
  proof={'rows':16,'candidate_rows':len(x),'ordered_eval_keys_sha256':c['split_keys_sha256']['evaluation'],'selected_keys_sha256':digest(small[KEYS].to_numpy(dtype='<i8')),'observed199_sha256':digest(observed),'sealed199_sha256':digest(old),'fixed65_sha256':digest(x[:,fixed]),'pool_rows':len(pool),'pool_cutoff':c['train_end'],'candidate_possible_columns':len(changed),'candidate_invariant_columns':len(fixed),'observed_reconstruction_exact':True,'schema':SCHEMA}
  save(folder/'proof.json',proof);folds[fold]=proof;trace({'event':'input_reconstruction_complete','fold':fold});del pool,x,observed,old,raw,frame,query
 result={'folds':folds,'fits':p['fits'],'encoders':0,'inference':0,'policy_scores':0,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'live_ready':False,'interpretation':'Input construction only; observed pitch type versus intended action and actual arrival timestamps remain unverified.'};save(out/'result.json',result);return result
