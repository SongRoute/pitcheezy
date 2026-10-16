"""Date-prior sufficient statistics for the frozen fullseason raw91 recipe.

Export owns donor access; queries only use compact statistics and explicit
previous-type state. No online donor updates or request-time data scans.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import stat
import re

import numpy as np
import pandas as pd

SCHEMA = 'ws-fullseason-date-profiles-v1'
TABLES = {'pitcher_type': (2, 9), 'batter': (1, 5), 'physical': (3, 8),
          'scale': (2, 12), 'outcome': (6, 11)}
MAX_ROWS = 750000
FILE_CAP = 128 << 20
ORIGIN_CAP = 4


def closure():
    from experiments.ws_fullseason.data import closure as existing
    return existing()


def origin_date(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):raise ValueError('canonical origin date')
    date=pd.Timestamp(value)
    if str(date.date())!=value:raise ValueError('canonical origin date')
    return date


def day_number(date):
    return int(np.datetime64(pd.Timestamp(date).date(), 'D').astype(int))


def _table(items, widths):
    items = list(items)
    if not items:
        return np.empty((0, widths[0]), np.int64), np.empty((0, widths[1]), np.float64)
    return np.asarray([k if isinstance(k, tuple) else (k,) for k, _ in items], np.int64), np.asarray([v for _, v in items], np.float64)


def _daily(frame, groups, values):
    # Same pandas daily sums and group cumsum as frozen first_small_features.profiles.
    base = frame[groups + ['game_date']].copy()
    for name, value in values.items(): base[name] = value
    names = list(values)
    sums = base.groupby(groups + ['game_date'], sort=True)[names].sum().reset_index()
    sums[names] = sums.groupby(groups, sort=False)[names].cumsum()
    return sums, names


def export_profiles(pool, origins, cutoff, M=None):
    """Exact-origin snapshots; final origin must be cutoff+1 day. No raw rows saved."""
    M = M or closure(); sf = M.safe
    origins = [origin_date(d) for d in origins]; cutoff = origin_date(cutoff)
    if (not 1 <= len(origins) <= ORIGIN_CAP or origins != sorted(set(origins))
            or origins[-1] != cutoff + pd.Timedelta(days=1) or list(pool.columns) != sf.DONOR
            or len(pool) > MAX_ROWS or pool.empty or pool.duplicated(sf.KEYS).any()
            or pool.game_date.isna().any() or pool.game_date.max() > cutoff):
        raise ValueError('fixed profile export scope')
    type_id = {t:i for i,t in enumerate(M.fsf.TYPES)}; hand = {'L':0, 'R':1}
    if (not pool.pitch_type.isin(type_id).all() or not pool.p_throws.isin(hand).all()
            or not pool.stand.isin(hand).all() or not pool.outcome10.between(0,9).all()):
        raise ValueError('profile donor categories')
    physics = pool[list(M.fsf.PHYSICS)].to_numpy(float)
    values = {'n':np.ones(len(pool))}
    for i,c in enumerate(M.fsf.PHYSICS):
        good=np.isfinite(physics[:,i]); values[c+'_sum']=np.where(good,physics[:,i],0);values[c+'_n']=good.astype(float)
    pp,pnames=_daily(pool,['pitcher','pitch_type'],values)
    bp,bnames=_daily(pool,['batter'],sf.behavior_values(pool))
    complete=pool.loc[np.isfinite(physics).all(1)]
    groups=list(complete.groupby('game_date',sort=True));cursor=0
    physical={};scale={};stats={};snapshots={}
    for origin in origins:
        while cursor<len(groups) and groups[cursor][0]<origin:
            date,g=groups[cursor]; day=day_number(date)
            # Operation/group/insertion order matches frozen transfer_features.
            for key,h in g.groupby(['pitch_type','p_throws','pitcher'],sort=False):
                z=h[list(M.fsf.PHYSICS)].to_numpy(float);ok=np.isfinite(z)
                sums=np.where(ok,z,0).sum(0);count=ok.sum(0);sq=np.where(ok,z*z,0).sum(0)
                key=(type_id[key[0]],hand[key[1]],int(key[2]));old=physical.get(key,(np.zeros(4),np.zeros(4)))
                physical[key]=(old[0]+sums,old[1]+count)
                sk=key[:2];old=scale.get(sk,(np.zeros(4),np.zeros(4),np.zeros(4)))
                scale[sk]=(old[0]+sums,old[1]+sq,old[2]+count)
            for key,h in g.groupby(['pitch_type','p_throws','stand','balls','strikes','pitcher'],sort=False):
                k=(type_id[key[0]],hand[key[1]],hand[key[2]],int(key[3]),int(key[4]),int(key[5]))
                old=stats.get(k,(np.zeros(10),day));stats[k]=(old[0]+np.bincount(h.outcome10.to_numpy(int),minlength=10),day)
            cursor+=1
        p=pp.loc[pp.game_date<origin].groupby(['pitcher','pitch_type'],sort=False).tail(1)
        b=bp.loc[bp.game_date<origin].groupby(['batter'],sort=False).tail(1)
        tables={
            'pitcher_type':_table((((int(row.pitcher),type_id[row.pitch_type]),row[pnames].to_numpy(float)) for _,row in p.iterrows()),TABLES['pitcher_type']),
            'batter':_table(((int(row.batter),row[bnames].to_numpy(float)) for _,row in b.iterrows()),TABLES['batter']),
            'physical':_table(((k,np.r_[v[0],v[1]]) for k,v in physical.items()),TABLES['physical']),
            'scale':_table(((k,np.r_[v[0],v[1],v[2]]) for k,v in scale.items()),TABLES['scale']),
            'outcome':_table(((k,np.r_[v[0],v[1]]) for k,v in stats.items()),TABLES['outcome'])}
        snapshots[str(origin.date())]=Snapshot(tables)
    return ProfileBundle(snapshots,str(cutoff.date()),M)


class Snapshot:
    def __init__(self,tables):
        if set(tables)!=set(TABLES): raise ValueError('exact profile tables')
        self.tables=tables;self.index={};self.context={}
        for name,(keys,values) in tables.items():
            kw,vw=TABLES[name]
            if (keys.ndim!=2 or keys.shape[1]!=kw or values.shape!=(len(keys),vw) or len(keys)>MAX_ROWS
                    or keys.dtype!=np.int64 or values.dtype!=np.float64 or not np.isfinite(values).all()
                    or len(np.unique(keys,axis=0))!=len(keys)):
                raise ValueError('bounded unique profile geometry')
            self.index[name]={tuple(k):v for k,v in zip(keys,values)}
        for key,value in self.index['outcome'].items():
            self.context.setdefault(key[:-1],{})[key[-1]]=value


class ProfileBundle:
    def __init__(self,snapshots,cutoff,M=None):
        self.snapshots=snapshots;self.cutoff=origin_date(cutoff);self.M=M or closure()
        for origin in snapshots:origin_date(origin)
        self.final_origin=str((self.cutoff+pd.Timedelta(days=1)).date())
        if (not 1<=len(snapshots)<=ORIGIN_CAP or self.final_origin not in snapshots
                or any(pd.Timestamp(d)>self.cutoff+pd.Timedelta(days=1) for d in snapshots)):
            raise ValueError('profile origin contract')

    def raw91(self,query):
        """Explicit safe.SCHEMA with previous_type/valid owned by stream state."""
        if not 0<len(query)<=100: raise ValueError('bounded query batch')
        sf=self.M.safe
        for name in sf.KEYS+['pitcher','batter','balls','strikes','outs_when_up','inning','home_score','away_score']:
            v=query[name].to_numpy()
            if v.dtype.kind not in 'iuf' or not np.isfinite(v).all() or (v!=np.floor(v)).any() or (np.abs(v)>=2**53).any():raise ValueError('integral query identity/state')
        if ((query[sf.KEYS+['pitcher','batter','inning']]<=0).any().any()
                or (query[['home_score','away_score']]<0).any().any()
                or not query.balls.between(0,3).all() or not query.strikes.between(0,2).all()
                or not query.outs_when_up.between(0,2).all() or not query.inning_topbot.isin(['Top','Bot']).all()
                or not all(isinstance(v,(bool,np.bool_)) for v in query.previous_valid)):
            raise ValueError('query prestate domain')
        # Frozen SCHEMA uses NaN for known-empty bases. Unknown occupancy must
        # abstain in the receipt-aware normalizer, never be encoded as empty here.
        for name in ('on_1b','on_2b','on_3b'):
            v=query[name].dropna().to_numpy()
            if len(v) and (v.dtype.kind not in 'iuf' or not np.isfinite(v).all() or (v!=np.floor(v)).any() or (v<=0).any() or (v>=2**53).any()):raise ValueError('known-empty or positive runner identity')
        for value in query.game_date:
            if isinstance(value,str):origin_date(value)
            else:
                date=pd.Timestamp(value)
                if pd.isna(date) or date.tz is not None or date!=date.normalize():raise ValueError('official date without clock/timezone')
        f,raw,names=sf.schema_frame(query)
        if not f.p_throws.isin(['L','R']).all() or not f.stand.isin(['L','R']).all(): raise ValueError('query handedness')
        result=np.zeros((len(query),91),np.float64);result[:,:66]=raw
        type_id={t:i for i,t in enumerate(self.M.fsf.TYPES)};hand={'L':0,'R':1};cfg=self.M.ot.SETTINGS
        for i,q in f.reset_index(drop=True).iterrows():
            date=pd.Timestamp(q.game_date).normalize();origin=str(date.date())
            if date>=pd.Timestamp(self.final_origin): origin=self.final_origin
            if origin not in self.snapshots: raise ValueError('unregistered historical profile origin')
            snap=self.snapshots[origin];idx=snap.index;t=type_id[q.pitch_type];pid=int(q.pitcher)
            pp=idx['pitcher_type'].get((pid,t),np.zeros(9));result[i,50]=np.log1p(pp[0])
            for j in range(4):
                count=pp[2+2*j];result[i,51+2*j]=pp[1+2*j]/count if count>0 else np.nan;result[i,52+2*j]=count==0
            bp=idx['batter'].get((int(q.batter),),np.zeros(5));result[i,59]=np.log1p(bp[0])
            for j,(_,num,den) in enumerate(sf.RATE_SPECS):
                result[i,60+2*j]=bp[num]/bp[den] if bp[den]>0 else np.nan;result[i,61+2*j]=bp[den]==0
            sk=(t,hand[q.p_throws]);key=sk+(hand[q.stand],int(q.balls),int(q.strikes));today=day_number(date)
            context=snap.context.get(key,{})
            own=context.get(pid,np.r_[np.zeros(10),today])[:10];own_n=own.sum()
            ownvec=np.r_[(own+cfg['prior_total']/10)/(own_n+cfg['prior_total']),np.log1p(own_n),float(own_n==0)]
            donors=[]
            for dp,value in context.items():
                phys=idx['physical'].get(sk+(dp,))
                if phys is not None and np.all(phys[4:]>0): donors.append((dp,value[:10],value[10],phys[:4]/phys[4:]))
            if donors:
                ids=np.array([v[0] for v in donors]);cnt=np.array([v[1] for v in donors]);last=np.array([v[2] for v in donors])
                dn=cnt.sum(1);rates=(cnt+cfg['prior_total']/10)/(dn[:,None]+cfg['prior_total'])
                weights=dn/(dn+cfg['support_k'])*np.exp2(-(today-last)/cfg['half_life_days'])
                keep=ids!=pid;dn=dn[keep];rates=rates[keep];weights=weights[keep]
            else: ids=np.empty(0,np.int64);dn=np.empty(0);rates=np.empty((0,10));weights=np.empty(0)
            total=weights.sum()
            generic=np.r_[(weights[:,None]*rates).sum(0)/total,np.log1p(dn.sum()),0.] if total>0 else np.r_[np.zeros(10),0.,1.]
            phys=idx['physical'].get(sk+(pid,));scale=idx['scale'].get(sk);std=None
            if scale is not None:
                avg=np.divide(scale[:4],scale[8:],out=np.zeros(4),where=scale[8:]>0)
                variance=np.divide(scale[4:8],scale[8:],out=np.zeros(4),where=scale[8:]>0)-avg**2
                std=np.sqrt(np.maximum(variance,0))
            abstain=phys is None or np.any(phys[4:]<=0) or std is None or np.any(std<=cfg['scale_floor'])
            # Generic arm still carries similarity-abstention. Frozen helper can
            # set it on similarity-weight underflow even though generic q stays unchanged.
            if not abstain:
                means=np.array([v[3] for v in donors]).reshape(-1,4)[ids!=pid]
                delta=(means-phys[:4]/phys[4:])/std
                simw=weights*np.exp(-.5*np.sum(delta*delta,axis=1)/cfg['bandwidth']**2)
                if len(weights) and simw.sum()<=0: abstain=True
            result[i,66:]=np.r_[ownvec,generic,float(abstain)]
        return result


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save_bundle(bundle,path,provenance):
    path=Path(path)
    if path.exists(): raise ValueError('fresh profile directory')
    path.mkdir();files={}
    for origin,snapshot in bundle.snapshots.items():
        name=origin+'.npz';values={}
        for table,(keys,stats) in snapshot.tables.items():values[table+'_keys']=keys;values[table+'_values']=stats
        np.savez(path/name,**values)
        if (path/name).stat().st_size>FILE_CAP:raise ValueError('profile output cap')
        files[origin]={'file':name,'bytes':(path/name).stat().st_size,'sha256':_sha(path/name),
                       'rows':{n:len(v[0]) for n,v in snapshot.tables.items()}}
    manifest={'schema':SCHEMA,'cutoff':str(bundle.cutoff.date()),'final_origin':bundle.final_origin,'origins':files,
              'provenance':provenance,'raw_donor_rows_saved':0,'arrival':'UNKNOWN','frozen_stats_only':True}
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


def load_bundle(path,manifest_sha256,M=None):
    """Bounded verified profile archive; external caller owns real-I/O grant."""
    from experiments.ws_fullseason.calibrate_worker import header_check
    path=Path(path);mp=path/'manifest.json'
    if not stat.S_ISREG(mp.lstat().st_mode) or mp.stat().st_size>1<<20 or _sha(mp)!=manifest_sha256:raise ValueError('profile manifest pin')
    from . import read_pinned
    manifest=json.loads(read_pinned(mp,manifest_sha256,cap=1<<20))
    if manifest['schema']!=SCHEMA or not 1<=len(manifest['origins'])<=ORIGIN_CAP:raise ValueError('profile schema')
    snapshots={}
    members={n+'_'+kind for n in TABLES for kind in ('keys','values')}
    for origin,spec in manifest['origins'].items():
        origin_date(origin)
        name=origin+'.npz';p=path/name
        if (spec['file']!=name or not stat.S_ISREG(p.lstat().st_mode) or not 0<spec['bytes']<=FILE_CAP
                or p.stat().st_size!=spec['bytes'] or _sha(p)!=spec['sha256']):raise ValueError('profile archive pin')
        selected={}
        for n,(kw,vw) in TABLES.items():
            rows=spec['rows'][n]
            if type(rows)is not int or not 0<=rows<=MAX_ROWS:raise ValueError('profile row cap')
            selected[n+'_keys']=((rows,kw),'i');selected[n+'_values']=((rows,vw),'f')
        with io.BytesIO(read_pinned(p,spec['sha256'],cap=spec['bytes'])) as stream:
            # Same128MiB aggregate decode bound as the existing header helper.
            header_check(stream,members,selected)
            with np.load(stream,allow_pickle=False) as z:
                tables={n:(z[n+'_keys'],z[n+'_values']) for n in TABLES}
        snapshots[origin]=Snapshot(tables)
    return ProfileBundle(snapshots,manifest['cutoff'],M)
