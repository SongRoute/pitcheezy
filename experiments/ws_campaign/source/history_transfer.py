"""Cutoff-bound physics history only; no new donor outcomes or dynamic evaluation."""
import hashlib
import numpy as np
import pandas as pd
from first_small_features import PHYSICS,profiles
KEYS=['game_pk','at_bat_number','pitch_number']
POOL_COLUMNS=KEYS+['game_date','pitcher','pitch_type']+list(PHYSICS)

def load_queries_and_pool(path,p):
 import pyarrow.parquet as pq
 from learning_diagnostic import split
 meta=pq.read_table(path,columns=KEYS+['game_date']).to_pandas()
 if len(meta)!=1252824:raise ValueError('Approved projection cardinality')
 selected,*_=split(meta,p)
 # All selected groups are whole games. Filter before materializing outcome columns.
 queries=pq.read_table(path,filters=[('game_pk','in',selected.game_pk.unique().tolist())]).to_pandas()
 if not np.array_equal(queries[KEYS].to_numpy(),selected[KEYS].to_numpy()):raise ValueError('Exact query keys/order')
 pool=pq.read_table(path,columns=POOL_COLUMNS,filters=[('game_date','<=',pd.Timestamp(p['history_pool']['cutoff']))]).to_pandas()
 validate_pool(pool,p)
 return queries,pool

def validate_pool(pool,p):
 cfg=p['history_pool']
 if list(pool.columns)!=POOL_COLUMNS or pool.duplicated(KEYS).any():raise ValueError('Physics-only pool schema/keys')
 if len(pool)!=cfg['rows'] or pool.game_date.max()>pd.Timestamp(cfg['cutoff']) or pool.game_date.min()<pd.Timestamp('2023-05-15'):raise ValueError('Pool scope')
 h=hashlib.sha256(pool.sort_values(KEYS)[KEYS].to_numpy(dtype='<i8').tobytes()).hexdigest()
 if h!=cfg['keys_sha256']:raise ValueError('Pool key identity')

def richer_physics(raw,names,queries,pool):
 combined=pd.concat([pool,queries[POOL_COLUMNS]],ignore_index=True);mask=np.arange(len(combined))<len(pool)
 values={'n':np.ones(len(combined))}
 for c in PHYSICS:
  v=combined[c].to_numpy(float);good=np.isfinite(v);values[c+'_sum']=np.where(good,v,0);values[c+'_n']=good.astype(float)
 pp=profiles(combined,mask,['pitcher','pitch_type'],values)[len(pool):]
 result=raw.copy();result[:,names.index('pitcher_type_prior_log_count')]=np.log1p(pp[:,0])
 for i,c in enumerate(PHYSICS):
  count=pp[:,2+2*i];result[:,names.index('prior_'+c)]=np.divide(pp[:,1+2*i],count,out=np.full(len(queries),np.nan),where=count>0);result[:,names.index('prior_'+c+'_missing')]=count==0
 return result
