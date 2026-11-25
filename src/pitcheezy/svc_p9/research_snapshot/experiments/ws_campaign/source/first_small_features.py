"""Shared, causal A/B features; frozen evaluation profiles, no current realization."""
import numpy as np
import pandas as pd
TYPES=('UNKNOWN','CH','CS','CU','EP','FA','FC','FF','FO','FS','KC','KN','PO','SC','SI','SL','ST','SV','UN')
PHYSICS=('release_speed','release_spin_rate','pfx_x','pfx_z')

def nested_games(frame,fraction,seed):
 import hashlib
 games=frame[['game_pk','game_date']].drop_duplicates('game_pk').copy()
 games['month']=games.game_date.dt.strftime('%Y-%m')
 games['rank']=[hashlib.sha256(f'{seed}:{int(g)}'.encode()).hexdigest() for g in games.game_pk]
 picked=[]
 for _,g in games.groupby('month',sort=True):picked.extend(g.sort_values('rank').head(max(1,int(np.ceil(len(g)*fraction)))).game_pk)
 return frame.game_pk.isin(picked).to_numpy()

def profiles(frame,training_mask,group,values):
 """Only selected training dates < query date; evaluation never updates profiles."""
 base=frame.loc[training_mask,group+['game_date']].copy()
 for name,value in values.items():base[name]=np.asarray(value)[training_mask]
 names=list(values)
 sums=base.groupby(group+['game_date'],sort=True)[names].sum().reset_index()
 sums[names]=sums.groupby(group,sort=False)[names].cumsum()
 out=np.zeros((len(frame),len(names)),float)
 indexes=frame.groupby(group,sort=False).indices
 for key,history in sums.groupby(group,sort=False):
  idx=indexes.get(key[0] if len(group)==1 and isinstance(key,tuple) else key)
  if idx is None:continue
  history=history.sort_values('game_date');dates=history.game_date.to_numpy()
  at=np.searchsorted(dates,frame.iloc[idx].game_date.to_numpy(),side='left')-1
  good=at>=0;out[np.asarray(idx)[good]]=history[names].to_numpy()[at[good]]
 return out

def raw_features(frame,training_mask):
 f=frame.reset_index(drop=True);n=len(f)
 score=np.where(f.inning_topbot.eq('Top'),1.,-1.)*(f.home_score-f.away_score)
 x=[f.balls/3,f.strikes/2,f.outs_when_up/2,f.inning/9,score/5,f.inning_topbot.eq('Top')]
 names=['balls','strikes','outs','inning','defensive_lead','home_defense']
 for c in ['on_1b','on_2b','on_3b']:x.append(f[c].notna());names.append(c)
 x.extend([f.stand.eq('L'),f.p_throws.eq('L')]);names.extend(['batter_left','pitcher_left'])
 lookup={t:i for i,t in enumerate(TYPES)};typ=f.pitch_type.map(lookup).fillna(0).to_numpy(int)
 current=np.eye(len(TYPES))[typ]
 prev=np.zeros(n,int);valid=np.zeros(n,bool)
 # Projected records sorted by exact key; gaps yield unknown rather than treating older record as previous.
 order=np.lexsort((f.pitch_number,f.at_bat_number,f.game_pk));a=order[1:];b=order[:-1]
 ok=(f.game_pk.to_numpy()[a]==f.game_pk.to_numpy()[b])&(f.at_bat_number.to_numpy()[a]==f.at_bat_number.to_numpy()[b])&(f.pitch_number.to_numpy()[a]==f.pitch_number.to_numpy()[b]+1)
 prev[a[ok]]=typ[b[ok]];valid[a[ok]]=True
 history=np.eye(len(TYPES))[prev]*valid[:,None]
 x=np.column_stack(x).astype(float);names+=['type_'+t for t in TYPES]+['previous_'+t for t in TYPES]+['previous_valid']
 x=np.column_stack([x,current,history,valid])
 values={'n':np.ones(n)}
 for c in PHYSICS:
  v=f[c].to_numpy(float);good=np.isfinite(v);values[c+'_sum']=np.where(good,v,0);values[c+'_n']=good.astype(float)
 pp=profiles(f,training_mask,['pitcher','pitch_type'],values)
 extra=[np.log1p(pp[:,0])];extra_names=['pitcher_type_prior_log_count']
 for i,c in enumerate(PHYSICS):
  count=pp[:,2+2*i];extra.extend([np.divide(pp[:,1+2*i],count,out=np.full(n,np.nan),where=count>0),count==0]);extra_names.extend(['prior_'+c,'prior_'+c+'_missing'])
 swing=f.description.isin(['swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play','missed_bunt','foul_bunt','bunt_foul_tip']).to_numpy()
 contact=f.description.isin(['foul','foul_tip','hit_into_play','foul_bunt','bunt_foul_tip']).to_numpy()
 bip=f.description.eq('hit_into_play').to_numpy();gb=f.bb_type.eq('ground_ball').to_numpy()
 bp=profiles(f,training_mask,['batter'],{'n':np.ones(n),'swing':swing.astype(float),'contact':contact.astype(float),'bip':bip.astype(float),'gb':gb.astype(float)})
 extra.append(np.log1p(bp[:,0]));extra_names.append('batter_prior_log_count')
 for name,num,den in [('swing',1,0),('contact_given_swing',2,1),('ground_given_bip',4,3)]:
  extra.extend([np.divide(bp[:,num],bp[:,den],out=np.full(n,np.nan),where=bp[:,den]>0),bp[:,den]==0]);extra_names.extend(['batter_prior_'+name,'batter_prior_'+name+'_missing'])
 return np.column_stack([x,*extra]).astype(np.float64),names+extra_names

def encode(raw,train):
 """Identical train-fit univariate bases for both models; no interaction products."""
 median=np.nanmedian(raw[train],axis=0);median=np.where(np.isfinite(median),median,0)
 filled=np.where(np.isfinite(raw),raw,median);mean=filled[train].mean(0);scale=filled[train].std(0);scale=np.where(scale>1e-8,scale,1)
 z=np.clip((filled-mean)/scale,-8,8)
 # Shared hinge features let additive A learn nonlinearity in each numeric column.
 continuous=np.flatnonzero([len(np.unique(filled[train,j]))>2 for j in range(raw.shape[1])])
 knots=np.quantile(z[train][:,continuous],[.25,.5,.75],axis=0)
 terms=[z]+[np.maximum(z[:,continuous]-k,0) for k in knots]
 result=np.column_stack(terms).astype(np.float32)
 assert np.isfinite(result).all()
 return result,{'median':median.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'continuous':continuous.tolist(),'knots':knots.tolist(),'dimension':result.shape[1]}
