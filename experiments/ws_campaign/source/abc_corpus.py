"""Job-owned 2026 S2 projection, roles, sample, FIT-only encoder and causal C physical similarity (A/B/C pilot).

One shared corpus for all arms. Reuses pinned helpers unchanged: project_first_small_train.labels(),
event_heads.labels(), candidate_input.query_context/schema_frame pieces (richer_physics,
transfer_features, history_features), first_small_features.encode/profiles.
Nothing here reads parquet unless read_source() is called by the gated runner.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_pk','at_bat_number','pitch_number']
# Exact projected source columns. Physics/description/events/bb_type are labels or strictly-prior
# history only; plate_x/plate_z/sz_top/sz_bot/zone and launch/effective/spin_axis are never read.
PROJECT=KEY+['game_date','game_type','home_team','away_team','pitcher','batter','pitch_type','description','events','bb_type','release_speed','release_spin_rate','pfx_x','pfx_z','balls','strikes','outs_when_up','inning','inning_topbot','home_score','away_score','on_1b','on_2b','on_3b','stand','p_throws']
NEVER_READ=('plate_x','plate_z','sz_top','sz_bot','zone','launch_speed','launch_angle','effective_speed','spin_axis','des','type','hit_distance_sc','estimated_woba_using_speedangle','delta_run_exp')
ORDER=('FIT','SELECT','CAL','TEST');CHILD={'FIT':'train','SELECT':'tuning','CAL':'calibration','TEST':'evaluation'}
REASONS=('excluded_null_game_key','excluded_null_or_invalid_key','excluded_non_registered_game_type','excluded_date_outside_source_range','excluded_nonpositive_order_key','excluded_duplicate_key','quarantined_multi_date_game','excluded_invalid_state','excluded_missing_handedness','excluded_unmapped_label','excluded_bip_missing_bb_type','excluded_nonbip_with_bb_type','excluded_bip_unknown_bb_type')
MISSING_BB=('None','nan','')  # event_heads.labels missing-bb_type spelling after fillna('None').astype(str)

class ProjectionStop(ValueError):
 """Row-quality STOP that carries the aggregate census so the caller persists it before re-raising."""
 def __init__(self,msg,census):super().__init__(msg);self.census=census

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def check_manifest(m,b,parquet_path,manifest_path):
 """Same structural binding as R1 raw_support_audit_r1.manifest_gate."""
 s=m.get('seasons',{}).get(str(b['season']));p=m.get('parameters',{})
 ok=isinstance(s,dict) and [m.get('version')==b['version'],m.get('fetch_commit')==b['fetch_commit'],p.get('game_type')==b['game_type'],p.get('season_dates',{}).get(str(b['season']))==[b['start'],b['end']],s.get('season')==b['season'],s.get('file')==b['file'],s.get('start')==b['start'],s.get('end')==b['end'],s.get('rows')==b['rows'],s.get('rows_raw')==b['rows'],s.get('columns')==b['columns'],s.get('duplicate_keys')==0,s.get('sha256')==b['parquet_sha256'],s.get('frozen') is True,str(Path(manifest_path).parent/b['file'])==str(parquet_path)]
 if ok is False or not all(ok):raise ValueError('STOP manifest binding')

def read_source(c):
 """Gated runner only: bind manifest + parquet bytes BEFORE decoding any row, then read PROJECT columns."""
 import pyarrow.parquet as pq
 s=c['source']
 if sha(s['manifest'])!=s['manifest_sha256'] or sha(s['parquet'])!=s['parquet_sha256']:raise ValueError('STOP source bytes changed')
 check_manifest(json.loads(Path(s['manifest']).read_text()),s['manifest_binding'],s['parquet'],s['manifest'])
 pf=pq.ParquetFile(s['parquet'])
 if pf.metadata.num_rows!=s['rows'] or len(pf.schema_arrow.names)!=s['manifest_binding']['columns']:raise ValueError('STOP footer rows/columns')
 missing=sorted(set(PROJECT)-set(pf.schema_arrow.names))
 if missing:raise ValueError('STOP projected columns absent: '+str(missing))
 return pf.read(columns=PROJECT).to_pandas(),{'parquet_sha256':s['parquet_sha256'],'manifest_sha256':s['manifest_sha256'],'footer_rows':pf.metadata.num_rows,'columns_read':PROJECT}

def project(raw,c,labels10,head_labels):
 """Row policy in registered order; every source row gets exactly one fate. Returns accepted frame + census."""
 r=c['row_policy'];n=len(raw);fate=np.full(n,'',object);f=raw.reset_index(drop=True).copy()
 def mark(mask,reason):
  m=np.asarray(mask,bool)&(fate=='');fate[m]=reason
 date=pd.to_datetime(f.game_date,errors='coerce');f['game_date']=date
 mark(f.game_pk.isna()|date.isna(),'excluded_null_game_key')
 seg=set(r['segregation_game_pks'])&set(f.game_pk.dropna().astype('int64'))
 if seg:raise ValueError('STOP segregated postseason game present')
 # Multi-date evidence from every row with game key, before any other filter (R1 rule).
 ev=f.loc[fate=='',['game_pk','game_date']].drop_duplicates();multi=set(ev.game_pk[ev.game_pk.duplicated()])
 mark(f[['at_bat_number','pitch_number','pitcher','batter']].isna().any(axis=1)|~f.inning_topbot.isin(['Top','Bot']),'excluded_null_or_invalid_key')
 mark(f.game_type.ne(r['game_type']),'excluded_non_registered_game_type')
 mark((date<pd.Timestamp(r['start']))|(date>pd.Timestamp(r['end'])),'excluded_date_outside_source_range')
 mark((f.at_bat_number.fillna(0)<=0)|(f.pitch_number.fillna(0)<=0),'excluded_nonpositive_order_key')
 live=fate=='';dup=np.zeros(n,bool);dup[np.flatnonzero(live)]=f.loc[live].duplicated(KEY,keep=False).to_numpy()
 if dup.any():
  d=f.loc[dup,[x for x in PROJECT if x!='game_date']+['game_date']].astype(str)
  if d.drop_duplicates().duplicated(KEY,keep=False).any():raise ValueError('STOP non-identical duplicate key content')
  first=np.zeros(n,bool);first[np.flatnonzero(live)]=f.loc[live].duplicated(KEY,keep='first').to_numpy();mark(first,'excluded_duplicate_key')
 mark(f.game_pk.isin(multi),'quarantined_multi_date_game')
 mark(~f.balls.between(0,3)|~f.strikes.between(0,2)|~f.outs_when_up.between(0,2)|f.inning.isna()|f.home_score.isna()|f.away_score.isna(),'excluded_invalid_state')
 mark(~f.stand.isin(['L','R'])|~f.p_throws.isin(['L','R']),'excluded_missing_handedness')
 y=np.full(n,-1,np.int64);ok=fate=='';y[ok]=labels10(f.loc[ok])
 unmapped=ok&(y<0);cats=f.loc[unmapped,['description','events']].fillna('MISSING').astype(str).value_counts().sort_index()
 mark(unmapped,'excluded_unmapped_label')
 # bb_type trust boundary: registered vocabulary only; null/''/'None'/'nan' is missing; nothing unknown becomes non-GB.
 d=f.description.fillna('').astype(str).to_numpy();b=f.bb_type.fillna('None').astype(str).to_numpy()
 step=fate=='';bip=d=='hit_into_play';missing=np.isin(b,MISSING_BB);known=np.isin(b,r['bb_types'])
 lit=np.where(f.bb_type.isna().to_numpy(),'MISSING',f.bb_type.astype(str).to_numpy())  # raw identity: '' / 'None' / 'nan' literals stay visible
 bbcat=pd.Series(np.where(bip,'BIP','nonBIP'))[step].str.cat(pd.Series(lit)[step],sep='|').value_counts().sort_index()
 mark(bip&missing,'excluded_bip_missing_bb_type');mark(~bip&~missing,'excluded_nonbip_with_bb_type');mark(bip&~missing&~known,'excluded_bip_unknown_bb_type')
 acc=fate=='';census={'source_rows':n,'accepted_rows':int(acc.sum()),'excluded':{k:int((fate==k).sum()) for k in REASONS},'multi_date_games':len(multi),'unmapped_label_categories':{f'{a}|{e}':int(v) for (a,e),v in cats.items()},
  'bb_type_vocabulary':list(r['bb_types']),'bb_type_categories':{k:int(v) for k,v in bbcat.items()},'bb_type_category_rows':int(step.sum()),'bb_type_category_scope':'rows reaching the bb_type step (after excluded_unmapped_label); key = BIP|nonBIP (description==hit_into_play) + literal bb_type or MISSING (null)'}
 if census['accepted_rows']+sum(census['excluded'].values())!=n or sum(census['bb_type_categories'].values())!=census['bb_type_category_rows']:raise ValueError('STOP row reconciliation')
 denom=n-sum(census['excluded'][k] for k in REASONS[:6]);census['stop_share_denominator']=denom
 stops=[f'STOP {k} {census["excluded"][k]}/{denom} exceeds registered share {cap}' for k,cap in r['stop_share'].items() if census['excluded'][k]>cap*max(denom,1)]
 census['stop']=stops
 if stops:raise ProjectionStop('; '.join(stops),census)
 if not (np.isin(b[acc&bip],r['bb_types']).all() and np.isin(b[acc&~bip],MISSING_BB).all() and np.array_equal(np.isin(y[acc],(3,4,5,6,7,9)),bip[acc])):raise ValueError('STOP accepted-row bb_type invariant')
 g=f.loc[acc].copy();g['outcome10']=y[acc]
 for k in KEY+['pitcher','batter']:g[k]=g[k].astype('int64')
 g['pitch_type_source']=g.pitch_type.fillna('MISSING').astype(str);g['pitch_type']=g.pitch_type.where(g.pitch_type.isin(r['types']),'UNKNOWN')
 census['pitch_type_mapped_unknown']={k:int(v) for k,v in g.loc[g.pitch_type_source.ne(g.pitch_type),'pitch_type_source'].value_counts().sort_index().items()}
 g=g.sort_values(KEY,kind='stable').reset_index(drop=True)
 hl=head_labels(g.description.fillna('').to_numpy(),g.bb_type.fillna('None').to_numpy())
 for h,v in hl.items():g[h]=v
 return g,census

def roles(g,c):
 """Whole game_date blocks; projection already quarantined multi-date games."""
 w={k:(pd.Timestamp(c['roles'][k][0]),pd.Timestamp(c['roles'][k][1])) for k in ORDER}
 if any(w[ORDER[i]][1]>=w[ORDER[i+1]][0] for i in range(3)):raise ValueError('Roles must be ordered, disjoint')
 m={k:((g.game_date>=a)&(g.game_date<=b)).to_numpy() for k,(a,b) in w.items()}
 for i in range(4):
  for j in range(i+1,4):
   if set(g.game_pk[m[ORDER[i]]])&set(g.game_pk[m[ORDER[j]]]):raise ValueError('Game spans roles')
 return m

def sample_games(g,fit,c):
 """Outcome-blind whole-game FIT subsample: sha256(seed:game_pk) as uniform [0,1) < fraction."""
 s=c['fit_sample'];games=np.unique(g.game_pk[fit])
 u=np.array([int(hashlib.sha256(f"{s['seed']}:{int(x)}".encode()).hexdigest()[:16],16)/2**64 for x in games])
 picked=set(games[u<s['fraction']].tolist());train=fit&g.game_pk.isin(picked).to_numpy()
 teams=pd.concat([g.loc[fit,['game_pk','home_team']].rename(columns={'home_team':'t'}),g.loc[fit,['game_pk','away_team']].rename(columns={'away_team':'t'})]).drop_duplicates()
 cover=teams[teams.game_pk.isin(picked)].t.value_counts().reindex(sorted(teams.t.unique()),fill_value=0)
 if (cover<s['min_games_per_team']).any():raise ValueError('STOP sample team coverage '+json.dumps({k:int(v) for k,v in cover[cover<s['min_games_per_team']].items()}))
 return train,{'fit_games':len(games),'sampled_games':len(picked),'teams':len(cover),'min_sampled_games_per_team':int(cover.min()),'sampled_game_keys_sha256':digest(np.sort(np.array(sorted(picked),dtype='<i8')))}

def features(g,fit_pool,query,cutoff,ci,fsf):
 """candidate_input.construct pieces with a NEW 2026 encoder fit on sampled FIT-train rows only.
 Pool = full eligible FIT (<= cutoff); history strictly prior by date, frozen at cutoff for later roles."""
 from history_transfer import POOL_COLUMNS,richer_physics
 from outcome_transfer import EXTRA,SETTINGS,transfer_features
 from batter_history_research import HISTORY_COLUMNS,history_features
 q=ci.query_context(g.loc[query].reset_index(drop=True));pool=g.loc[fit_pool,ci.DONOR].reset_index(drop=True)
 if pool.game_date.max()>pd.Timestamp(cutoff):raise ValueError('Pool beyond history cutoff')
 f,raw,names=ci.schema_frame(q)
 raw=richer_physics(raw,names,f,pool[POOL_COLUMNS]);arrays,audit=transfer_features(f,pool[POOL_COLUMNS+EXTRA],SETTINGS)
 joined=np.column_stack([raw,arrays['generic']]);full,*_=history_features(f,pool[HISTORY_COLUMNS],cutoff);joined[:,59:66]=full
 return joined,audit

PHYS_GROUP=['pitcher','pitch_type','p_throws']
S_REASONS=('s_zero_unknown_pitch_type','s_zero_query_no_prior_same_type_hand','s_zero_query_missing_physics_component','s_zero_no_target_prior_same_type_hand_physics')

def prior_physics(pool,queries,physics):
 """Strictly-prior pitcher x pitch_type x p_throws means/counts of PHYSICS, the history_transfer.richer_physics
 mechanics (profiles side='left': donor game_date < query date, same day excluded; only pool rows update).
 Query rows contribute zero values, so current-pitch physics never enters its own or any summary."""
 from first_small_features import profiles
 comb=pd.concat([pool[PHYS_GROUP+['game_date']],queries[PHYS_GROUP+['game_date']]],ignore_index=True);mask=np.arange(len(comb))<len(pool)
 values={'n':np.r_[np.ones(len(pool)),np.zeros(len(queries))]}
 for c in physics:
  v=np.r_[pool[c].to_numpy(float),np.zeros(len(queries))];good=np.isfinite(v)&mask;values[c+'_sum']=np.where(good,v,0);values[c+'_n']=good.astype(float)
 pp=profiles(comb,mask,PHYS_GROUP,values)[len(pool):]
 cnt=pp[:,2::2];mean=np.divide(pp[:,1::2],cnt,out=np.full(cnt.shape,np.nan),where=cnt>0)
 return pp[:,0],mean,(cnt>0).all(1)

def similarity(pool,queries,targets,sim):
 """C physical similarity (pre-registered engineering hypothesis, NOT a learned or optimal similarity):
 s_i = max over cohort pitchers P != pitcher_i of exp(-0.5 * sum_k ((mu_i,k - mu_P,k) / scale_k)^2), where mu are the
 strictly-prior same pitch_type, same p_throws FIT-pool means of PHYSICS (fixed scales in sim['scales']; nothing is
 estimated from data). Missing query/target history or any physics component, or UNKNOWN pitch type -> s=0 (counted)."""
 phys=tuple(sim['physics']);scale=np.array([sim['scales'][c] for c in phys],float)
 q=queries[PHYS_GROUP+['game_date']].reset_index(drop=True);n_q,mu_q,ok_q=prior_physics(pool,q,phys)
 tp=pool.loc[pool.pitcher.isin(targets),PHYS_GROUP].drop_duplicates()
 grid=tp.merge(q[['pitch_type','p_throws','game_date']].drop_duplicates(),on=['pitch_type','p_throws']).reset_index(drop=True)
 if len(grid):_,mu_t,ok_t=prior_physics(pool,grid,phys)
 else:mu_t=np.zeros((0,len(phys)));ok_t=np.zeros(0,bool)
 s=np.zeros(len(q));reason=np.full(len(q),'',object)
 unk=(q.pitch_type=='UNKNOWN').to_numpy();reason[unk]=S_REASONS[0]
 reason[(reason=='')&(n_q==0)]=S_REASONS[1];reason[(reason=='')&~ok_q]=S_REASONS[2]
 gk=grid[['pitch_type','p_throws','game_date']].apply(tuple,axis=1) if len(grid) else pd.Series([],dtype=object)
 tidx=gk.groupby(gk).indices if len(grid) else {}
 for key,rows in q.loc[reason==''].groupby(['pitch_type','p_throws','game_date']).groups.items():
  rows=np.asarray(rows);t=np.asarray(tidx.get(key,np.empty(0,int)));t=t[ok_t[t]] if len(t) else t
  if not len(t):reason[rows]=S_REASONS[3];continue
  d2=np.square((mu_q[rows][:,None,:]-mu_t[t][None,:,:])/scale).sum(2)
  d2[q.pitcher.to_numpy()[rows][:,None]==grid.pitcher.to_numpy()[t][None,:]]=np.inf
  best=d2.min(1);reason[rows[~np.isfinite(best)]]=S_REASONS[3];s[rows]=np.clip(np.exp(-.5*best),0,1)
 if not np.isfinite(s).all():raise ValueError('STOP non-finite similarity')
 census={'rows':len(q),'positive':int((reason=='').sum()),**{k:int((reason==k).sum()) for k in S_REASONS}}
 if census['positive']+sum(census[k] for k in S_REASONS)!=len(q):raise ValueError('STOP similarity reconciliation')
 return s,census

def arm_raw(arm,target,s,alpha,beta):
 target=np.asarray(target,bool);s=np.asarray(s,float)
 if not np.isfinite(s).all() or np.any((s<0)|(s>1)):raise ValueError('Similarity outside [0,1]')
 return {'A':np.ones(len(target)),'B':1+alpha*target,'C':1+alpha*target+beta*(~target)*s}[arm].astype(np.float64)

def head_weights(arm,target,s,train,w):
 """Raw weight on the head's TRAIN rows, registered caps, then mean 1; zero outside TRAIN."""
 raw=arm_raw(arm,target[train],s[train],w['alpha'],w['beta'])
 lo,hi=w['raw_bounds']
 if raw.min()<lo-1e-12 or raw.max()>hi+1e-12 or raw.max()/raw.min()>w['max_min_ratio']+1e-12:raise ValueError('STOP raw weight caps')
 out=np.zeros(len(train));out[train]=raw/raw.mean();ess=float(raw.sum()**2/np.square(raw).sum()/len(raw))
 if ess<w['min_ess_fraction']:raise ValueError(f'STOP ESS/N {ess} < {w["min_ess_fraction"]}')
 return out,ess

def mass(w,train,target,pitcher,batter,pitch_type):
 x=w[train];t=x.sum()
 def share(ids):
  v=pd.Series(x).groupby(np.asarray(ids)[train]).sum()/t;return {'max':float(v.max()),'top10':float(v.nlargest(10).sum()),'argmax':int(v.idxmax())}
 return {'ess_fraction':float(t**2/np.square(x).sum()/len(x)),'target_mass':float(x[target[train]].sum()/t),'nontarget_mass':float(x[~target[train]].sum()/t),'target_rows':int(target[train].sum()),'pitcher':share(pitcher),'batter':share(batter),'pitch_type':{str(k):float(v) for k,v in (pd.Series(x).groupby(np.asarray(pitch_type)[train]).sum()/t).sort_index().items()},'max_min_ratio':float(x.max()/x.min())}
