"""Strict-date same-pool outcome sharing; current physics/labels never queries."""
import numpy as np
import pandas as pd
from history_transfer import POOL_COLUMNS,KEYS,validate_pool
from first_small_features import PHYSICS
SETTINGS={'prior_total':1.,'support_k':20.,'half_life_days':365.,'bandwidth':1.,'scale_floor':1e-8}
EXTRA=['p_throws','stand','balls','strikes','outcome10']
NAMES=['own_outcome_'+str(i) for i in range(10)]+['own_outcome_log_n','own_outcome_missing']+['donor_outcome_'+str(i) for i in range(10)]+['donor_outcome_log_n','donor_outcome_missing','similarity_abstention']

def load_outcome_pool(path,p):
 import pyarrow.parquet as pq
 pool=pq.read_table(path,columns=POOL_COLUMNS+EXTRA,filters=[('game_date','<=',pd.Timestamp(p['history_pool']['cutoff']))]).to_pandas()
 validate_pool(pool[POOL_COLUMNS],p)
 if not pool.outcome10.between(0,9).all() or not pool.balls.between(0,3).all() or not pool.strikes.between(0,2).all() or not pool.p_throws.isin(['L','R']).all() or not pool.stand.isin(['L','R']).all():raise ValueError('Outcome donor schema/support')
 return pool

def transfer_features(queries,pool,settings=None):
 """Uniform pseudocount1; constants frozen, no outcome-based neighbor selection."""
 cfg=SETTINGS if settings is None else settings
 if cfg!=SETTINGS:raise ValueError('Fixed transfer settings; no sweep')
 # ponytail: daily sufficient statistics and query-key caching; up to all same-stratum pitchers, not top-k search.
 pool=pool.loc[np.isfinite(pool[list(PHYSICS)].to_numpy(float)).all(axis=1)].copy()
 n=len(queries);arms={a:np.zeros((n,25),float) for a in ['own','generic','similar']};stats={};physical={};scale={};audit={'query_rows':n,'donor_absent':0,'similarity_abstention':0,'evaluated_pairs':0}
 dates=sorted(pool.game_date.unique());days={d:g for d,g in pool.groupby('game_date',sort=False)};cursor=0
 for date,indices in queries.groupby('game_date',sort=True).indices.items():
  while cursor<len(dates) and dates[cursor]<date:
   g=days[pd.Timestamp(dates[cursor])];day=np.datetime64(dates[cursor],'D').astype(int)
   for key,h in g.groupby(['pitch_type','p_throws','pitcher'],sort=False):
    z=h[list(PHYSICS)].to_numpy(float);ok=np.isfinite(z);s=np.where(ok,z,0).sum(0);count=ok.sum(0);sq=np.where(ok,z*z,0).sum(0)
    old=physical.get(key,(np.zeros(4),np.zeros(4)));physical[key]=(old[0]+s,old[1]+count)
    sk=key[:2];old=scale.get(sk,(np.zeros(4),np.zeros(4),np.zeros(4)));scale[sk]=(old[0]+s,old[1]+sq,old[2]+count)
   for key,h in g.groupby(['pitch_type','p_throws','stand','balls','strikes','pitcher'],sort=False):
    group=stats.setdefault(key[:-1],{});old=group.get(key[-1],(np.zeros(10),day));group[key[-1]]=(old[0]+np.bincount(h.outcome10.to_numpy(int),minlength=10),day)
   cursor+=1
  today=np.datetime64(date,'D').astype(int);cache={};answers={}
  for i in indices:
   q=queries.iloc[i];sk=(q.pitch_type,q.p_throws);key=sk+(q.stand,int(q.balls),int(q.strikes));pid=int(q.pitcher);answer_key=(key,pid)
   if answer_key in answers:
    for arm in arms:arms[arm][i]=answers[answer_key][arm]
    continue
   context=stats.get(key,{})
   own=context.get(pid,(np.zeros(10),today))[0];own_n=own.sum();ownvec=np.r_[(own+cfg['prior_total']/10)/(own_n+cfg['prior_total']),np.log1p(own_n),float(own_n==0)]
   if key not in cache:
    donors=[]
    for dp,(counts,last) in context.items():
     phys=physical.get(sk+(dp,))
     if phys is not None and np.all(phys[1]>0):donors.append((dp,counts,last,phys[0]/phys[1]))
    if donors:
     ids=np.array([z[0] for z in donors]);cnt=np.array([z[1] for z in donors]);last=np.array([z[2] for z in donors]);means=np.array([z[3] for z in donors]);dn=cnt.sum(1);rates=(cnt+cfg['prior_total']/10)/(dn[:,None]+cfg['prior_total']);weights=dn/(dn+cfg['support_k'])*np.exp2(-(today-last)/cfg['half_life_days'])
    else:ids=np.empty(0,int);dn=np.empty(0);rates=np.empty((0,10));means=np.empty((0,4));weights=np.empty(0)
    cache[key]=(ids,dn,rates,means,weights)
   ids,dn,rates,means,weights=cache[key];valid=ids!=pid;dn=dn[valid];rates=rates[valid];means=means[valid];weights=weights[valid];audit['evaluated_pairs']+=len(weights)
   raw=physical.get(sk+(pid,));sc=scale.get(sk);std=None
   if sc is not None:
    avg=np.divide(sc[0],sc[2],out=np.zeros(4),where=sc[2]>0);variance=np.divide(sc[1],sc[2],out=np.zeros(4),where=sc[2]>0)-avg**2;std=np.sqrt(np.maximum(variance,0))
   abstain=raw is None or np.any(raw[1]<=0) or std is None or np.any(std<=cfg['scale_floor'])
   def donor(w):
    total=w.sum()
    if total<=0:return np.r_[np.zeros(10),0.,1.]
    return np.r_[(w[:,None]*rates).sum(0)/total,np.log1p(dn.sum()),0.]
   generic=donor(weights)
   if abstain:similar=generic.copy()
   else:
    delta=(means-raw[0]/raw[1])/std;simw=weights*np.exp(-.5*np.sum(delta*delta,axis=1)/cfg['bandwidth']**2);similar=donor(simw)
    # Underflow has no supported similarity estimate: use exact generic fallback.
    if len(weights) and simw.sum()<=0:similar=generic.copy();abstain=True
   answer={'own':np.r_[ownvec,np.zeros(10),0.,1.,float(abstain)],'generic':np.r_[ownvec,generic,float(abstain)],'similar':np.r_[ownvec,similar,float(abstain)]};answers[answer_key]=answer
   for arm in arms:arms[arm][i]=answer[arm]
  for i in indices:
   audit['donor_absent']+=int(arms['generic'][i,23]);audit['similarity_abstention']+=int(arms['similar'][i,24])
 for a in arms:
  if not np.isfinite(arms[a]).all():raise ValueError('Nonfinite transfer features')
 return arms,audit
