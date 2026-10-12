"""Job-owned SAFE module for the Step 2 fixed-scale runner: minimal pure copies from individually allowed source.

candidate_input.py, batter_history_research.py and frequency_control.py import the denied learning_diagnostic at
module import time, so they are NEVER imported or executed. Only the pure definitions below are copied, verbatim
except where a DIFF note says otherwise. Each block names its origin file, sha256 and line range.
The three closure-clean pinned helpers (first_small_features, history_transfer, outcome_transfer) are imported
unchanged; scale_runner.closure() sha-checks and registers them in sys.modules before this module is loaded.
Nothing here reads a file. No denied file (project_first_small_train.py, event_heads.py, learning_diagnostic.py,
coherence/run.py) was read, imported or copied.
"""
import hashlib
import numpy as np
import pandas as pd
from first_small_features import TYPES,PHYSICS,raw_features,profiles
from history_transfer import KEYS,POOL_COLUMNS,richer_physics
from outcome_transfer import EXTRA,SETTINGS,transfer_features

ORIGINS={
 'candidate_input.py':{'sha256':'bc2af7fd8db9d1509f3bc019611a71e9c842676faae9f991552f9d20bfb69abc','lines':'12-14 (CONTEXT, DONOR, SCHEMA), 18-27 (schema_frame), 41-45 (query_context)','diff':'none'},
 'batter_history_research.py':{'sha256':'04db389f33e8f6c3f6537971d4c10323224122afb194e99f18374dd86768f666','lines':'14-16 (HISTORY_COLUMNS, RATE_SPECS, SHRINKAGE), 18-19 (behavior_values), 21-33 (history_features)','diff':'none'},
 'frequency_control.py':{'sha256':'2319128d354042ec5414a5f28665a8cc1f5e729bacb7e6022d0a1012fa9bb05b','lines':'13 (digest), 19-23 (apply_encoder)','diff':'none'},
 'pitchmdp/model.py':{'sha256':'0e963b96900adb09705a56afb79e8dedcf31973d253df393ffe205df39b8a150','lines':'18-19 (OUTCOMES), 26-41 (outcome_labels)','diff':'OUTCOMES.index(name) loop kept; module not imported (it imports torch)'},
 'abc_pilot.py':{'sha256':'a6b4de8413bedd883036ce315f9fbd508a1fe1d2df736ce67719b7839ad7f5f9','lines':'305 (LEAF_MAP text) -> leaves()','diff':'NEW array implementation of the registered LEAF_MAP; the pinned coherence/run.py leaves() is denied and was not read, so identity to it is NOT claimed'},
 'REGISTRATION-STEP2-TAXONOMY-02.json':{'lines':'label_schema 72-117 -> core_fate(), head_labels()','diff':'NEW vectorised transcription of the accepted descriptive taxonomy rules; aggregate-level provenance only'},
}

# ---- candidate_input.py 12-14 ----
HISTORY_COLUMNS=KEYS+['game_date','batter','pitch_type','description','bb_type']  # batter_history_research.py 14
CONTEXT=KEYS+['game_date','pitcher','batter','pitch_type','balls','strikes','outs_when_up','inning','inning_topbot','home_score','away_score','on_1b','on_2b','on_3b','stand','p_throws']
DONOR=list(dict.fromkeys(POOL_COLUMNS+EXTRA+HISTORY_COLUMNS))
SCHEMA=CONTEXT+['previous_type','previous_valid']

# ---- candidate_input.py 18-27 ----
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

# ---- candidate_input.py 41-45 ----
def query_context(frame):
 f=frame.copy();keys=list(map(tuple,f[KEYS].to_numpy()));lookup={k:t for k,t in zip(keys,f.pitch_type)}
 prev=[lookup.get((k[0],k[1],k[2]-1),'UNKNOWN') for k in keys];valid=[(k[0],k[1],k[2]-1) in lookup for k in keys]
 f['previous_type']=prev;f['previous_valid']=valid
 return f[SCHEMA]

# ---- batter_history_research.py 15-16, 18-19, 21-33 ----
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

# ---- frequency_control.py 13, 19-23 ----
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def apply_encoder(raw,e):
 # Apply only stored statistics; do not call encode() or refit normalization.
 median=np.array(e['median']);mean=np.array(e['mean']);scale=np.array(e['scale']);cols=np.array(e['continuous'],int)
 z=np.clip((np.where(np.isfinite(raw),raw,median)-mean)/scale,-8,8)
 return np.column_stack([z]+[np.maximum(z[:,cols]-k,0) for k in np.array(e['knots'])]).astype(np.float32)

# ---- pitchmdp/model.py 18-19, 26-41: parent-10 ----
OUTCOMES=('ball','strike','foul','out','single','double','triple','home_run','hbp','double_play')

def outcome_labels(frame):
 d,e=frame.description.fillna(''),frame.events.fillna('')
 labels=np.full(len(frame),-1,dtype=np.int64)
 labels[d.isin(['ball','blocked_ball','pitchout','intent_ball'])]=0
 labels[d.isin(['called_strike','swinging_strike','swinging_strike_blocked','missed_bunt','foul_tip','bunt_foul_tip'])]=1
 labels[d.isin(['foul','foul_bunt'])]=2
 labels[(d=='foul_bunt')&(frame.strikes==2)]=1
 labels[d=='hit_by_pitch']=8
 for name in ['single','double','triple','home_run']:
  labels[(d=='hit_into_play')&(e==name)]=OUTCOMES.index(name)
 labels[(d=='hit_into_play')&e.isin(['field_out','force_out','fielders_choice_out','sac_fly','sac_bunt'])]=3
 labels[(d=='hit_into_play')&e.isin(['double_play','grounded_into_double_play','sac_fly_double_play','sac_bunt_double_play'])]=9
 return labels

# ---- accepted descriptive taxonomy (REGISTRATION-STEP2-TAXONOMY-02 label_schema): two binary heads + row fate ----
WHIFF=('swinging_strike','swinging_strike_blocked');GROUND_BALL='ground_ball';BB_TYPES=('ground_ball','line_drive','fly_ball','popup');MISSING_BB=('None','nan','')
AUTOMATIC=('automatic_ball','automatic_strike');UNUSUAL_PITCHOUT=('swinging_pitchout','foul_pitchout')
LEAF_BY_DESCRIPTION={'ball':0,'blocked_ball':0,'pitchout':0,'intent_ball':0,'foul':1,'hit_by_pitch':2,'swinging_strike':3,'swinging_strike_blocked':3,'called_strike':4,'foul_tip':4,'bunt_foul_tip':4,'missed_bunt':4}
BIP_CORE_EVENTS=('single','double','triple','home_run','field_out','force_out','fielders_choice_out','sac_fly','sac_bunt','double_play','grounded_into_double_play','sac_fly_double_play','sac_bunt_double_play')
CORE=tuple(f'LEAF{i}' for i in range(7))
EXCLUDED_FATES=('HYPOTHESIS_LEAF5_GB_BIP_OUTCOME_DEPENDENT','HYPOTHESIS_LEAF6_NONGB_BIP_OUTCOME_DEPENDENT','ABSTAIN_BIP_CATCHER_INTERF','ABSTAIN_BIP_MISSING_BB_TYPE','ABSTAIN_BIP_UNREGISTERED_BB_TYPE','ABSTAIN_NONBIP_WITH_BB_TYPE','ABSTAIN_UNUSUAL_PITCHOUT','NOT_A_PITCH_AUTOMATIC','ABSTAIN_AUTOMATIC_WITH_TRACKING','ABSTAIN_UNKNOWN_DESCRIPTION')

def head_labels(description,bb_type):
 """Signature of the callable abc_corpus.project expects. narrow_whiff = LEAF3 descriptions; ground_ball = bb_type == ground_ball."""
 d=pd.Series(description).fillna('').astype(str).to_numpy();b=pd.Series(bb_type).fillna('None').astype(str).to_numpy()
 return {'narrow_whiff':np.isin(d,WHIFF).astype(np.int8),'ground_ball_per_pitch':(b==GROUND_BALL).astype(np.int8)}

def core_fate(f):
 """Exactly one fate per row from description/events/bb_type and tracking PRESENCE only (first match).
 Field error, fielder's choice, triple play and other non-core BIP events stay HYPOTHESIS; nothing is relabelled."""
 n=len(f);d=f.description.fillna('').astype(str).to_numpy();e=f.events.fillna('').astype(str).to_numpy();b=f.bb_type.fillna('None').astype(str).to_numpy()
 track=(f.pitch_type.notna()|f.release_speed.notna()).to_numpy();missing=np.isin(b,MISSING_BB);known=np.isin(b,BB_TYPES)
 out=np.full(n,'ABSTAIN_UNKNOWN_DESCRIPTION',object)
 auto=np.isin(d,AUTOMATIC);out[auto&~track]='NOT_A_PITCH_AUTOMATIC';out[auto&track]='ABSTAIN_AUTOMATIC_WITH_TRACKING'
 out[np.isin(d,UNUSUAL_PITCHOUT)]='ABSTAIN_UNUSUAL_PITCHOUT'
 leaf=np.full(n,-1)
 for k,v in LEAF_BY_DESCRIPTION.items():leaf[d==k]=v
 fb=d=='foul_bunt';leaf[fb]=np.where((f.strikes==2).to_numpy()[fb],4,1)
 for i in range(5):out[(leaf==i)&missing]=f'LEAF{i}'
 out[(leaf>=0)&~missing]='ABSTAIN_NONBIP_WITH_BB_TYPE'
 bip=d=='hit_into_play';ci=bip&(e=='catcher_interf');core=np.isin(e,BIP_CORE_EVENTS);gb=b==GROUND_BALL
 out[bip&known&core&gb]='LEAF5';out[bip&known&core&~gb]='LEAF6'
 out[bip&known&~core&gb]='HYPOTHESIS_LEAF5_GB_BIP_OUTCOME_DEPENDENT';out[bip&known&~core&~gb]='HYPOTHESIS_LEAF6_NONGB_BIP_OUTCOME_DEPENDENT'
 out[bip&~missing&~known]='ABSTAIN_BIP_UNREGISTERED_BB_TYPE';out[bip&missing]='ABSTAIN_BIP_MISSING_BB_TYPE';out[ci]='ABSTAIN_BIP_CATCHER_INTERF'
 return out

# ---- abc_pilot.py 305 LEAF_MAP -> seven coherent leaves ----
BIP=(3,4,5,6,7,9)
def leaves(p10,q,r):
 s=p10[:,list(BIP)].sum(1);out=np.column_stack([p10[:,0],p10[:,2],p10[:,8],p10[:,1]*q,p10[:,1]*(1-q),s*r,s*(1-r)])
 if not np.isfinite(out).all() or np.abs(out.sum(1)-1).max()>1e-9:raise ValueError('STOP seven-leaf coherence')
 return out
