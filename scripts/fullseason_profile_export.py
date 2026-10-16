"""Gated fixed-origin profile export and predetermined raw91/encoded199 parity."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
from experiments.ws_fullseason.audit import gates,sha
from experiments.ws_fullseason import data,final_source
from experiments.ws_fullseason.final_worker import read_source
from pitcheezy.seq_shadow import fullseason_profiles as profiles
from pitcheezy.seq_shadow.fullseason import read_artifact

ORIGINS=['2026-03-25','2026-07-01','2026-10-11','2026-10-12']
PARITY_DATES=ORIGINS[:-1]
RAW_ATOL=1e-12
RAW_RTOL=1e-12
ENC_ATOL=1e-7
ENC_RTOL=1e-6


def selected_rows(rows):
    count=min(30,rows)
    if count<=0:raise ValueError('registered origin has no query rows')
    return np.array([k*(rows-1)//max(1,count-1) for k in range(count)],np.int64)


def frozen_reference(pool,query,cutoff,M):
    """Original scale.features operations after explicit full-day previous lookup."""
    f,raw,names=M.safe.schema_frame(query)
    raw=M.ht.richer_physics(raw,names,f,pool[M.ht.POOL_COLUMNS])
    arms,_=M.ot.transfer_features(f,pool[M.ht.POOL_COLUMNS+M.ot.EXTRA],M.ot.SETTINGS)
    joined=np.column_stack([raw,arms['generic']])
    full,*_=M.safe.history_features(f,pool[M.safe.HISTORY_COLUMNS],cutoff)
    joined[:,59:66]=full
    return joined


def compare(raw,ref,encoder,M):
    if raw.shape!=ref.shape or raw.shape[1]!=91 or not np.array_equal(np.isnan(raw),np.isnan(ref)):
        raise ValueError('raw schema/missingness mismatch')
    if not np.allclose(raw,ref,atol=RAW_ATOL,rtol=RAW_RTOL,equal_nan=True):raise ValueError('raw91 parity tolerance')
    x=M.safe.apply_encoder(raw,encoder);expected=M.safe.apply_encoder(ref,encoder)
    if x.shape!=(len(raw),199) or not np.allclose(x,expected,atol=ENC_ATOL,rtol=ENC_RTOL):raise ValueError('encoded199 parity tolerance')
    good=np.isfinite(raw)&np.isfinite(ref)
    return {'rows':len(raw),'raw_max_absolute_difference':float(np.max(np.abs(raw[good]-ref[good]),initial=0)),
            'raw_exact_including_nan':bool(np.array_equal(raw,ref,equal_nan=True)),
            'encoded_max_absolute_difference':float(np.max(np.abs(x-expected),initial=0)),
            'encoded_exact':bool(np.array_equal(x,expected))}


def run(regpath,grantpath):
    reg,recipe=gates(regpath,grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.alarm(reg['bounds']['wall_seconds']+15)
    if reg['origins']!=ORIGINS or reg['parity_dates']!=PARITY_DATES or reg['query_rule']!='min30 per exact date; floor(k*(N-1)/(m-1)); full-date previous lookup first':
        raise ValueError('fixed origins/query rule')
    if recipe['execution_stage']!='FINAL_REFIT' or recipe['window']!={'fit':['2026-03-25','2026-10-11'],'oof':None}:raise ValueError('fixed donor source role')
    M=data.closure()
    if reg['columns']!=M.cp.PROJECT or set(reg['sources'])!=set(final_source.SOURCES):raise ValueError('exact source projection')
    for name,s in reg['sources'].items():
        if {'version':s['version'],'sha256':s['parquet_sha256'],'manifest_sha256':s['manifest_sha256']}!=final_source.SOURCES[name]:raise ValueError('source binding')
    frames={name:read_source(name,reg['sources'][name],reg['columns'],reg['bounds']) for name in ['S2','I3']}
    print('phase=profile-quality',flush=True)
    g,census,fit,oof,y=final_source.project(frames,recipe,M,reg['final_authority']);del frames
    observed=data.audit_summary(g,census,fit,oof,y,M)
    if not 0<reg['expected_rows']<=profiles.MAX_ROWS or len(g)!=reg['expected_rows'] or data.key_digest(g[M.cp.KEY].to_numpy())!=reg['expected_keys_sha256']:raise ValueError('accepted donor identity')
    out=Path(reg['output']);(out/'quality.json').write_text(json.dumps(observed,indent=2,allow_nan=False)+'\n')
    pool=g[M.safe.DONOR].copy()
    print('phase=profile-statistics',flush=True)
    bundle=profiles.export_profiles(pool,ORIGINS,'2026-10-11',M)
    manifest=profiles.save_bundle(bundle,out/'bundle',{'recipe_sha256':sha(reg['recipe']),
        'source_bindings':final_source.SOURCES,'donor_rows':len(g),'ordered_keys_sha256':reg['expected_keys_sha256'],
        'source_commit':reg['source_commit'],'role':'2026 EXPOSED DEVELOPMENT','quality_sha256':sha(out/'quality.json')})
    print('phase=profile-roundtrip-parity',flush=True)
    restored=profiles.load_bundle(out/'bundle',sha(out/'bundle/manifest.json'),M)
    encoder=json.loads(read_artifact(reg['encoder'],1<<20));reports={}
    for date in PARITY_DATES:
        rows=g.loc[g.game_date==pd.Timestamp(date)].reset_index(drop=True)
        query=M.safe.query_context(rows).iloc[selected_rows(len(rows))].reset_index(drop=True)
        ref=frozen_reference(pool,query,'2026-10-11',M);raw=restored.raw91(query)
        report=compare(raw,ref,encoder,M)
        if not np.array_equal(raw,bundle.raw91(query),equal_nan=True):raise ValueError('profile serialization parity')
        report['query_keys_sha256']=data.key_digest(query[M.safe.KEYS].to_numpy())
        report['eligible_date_rows']=len(rows);reports[date]=report
    # Same inode mutation after the first source hash is rejected before publishing.
    for s in reg['sources'].values():
        for field in ('parquet','manifest'):
            if sha(s[field])!=s[field+'_sha256']:raise ValueError('source changed during export')
    receipt={'status':'COMPLETED','stage':'FIXED_ORIGIN_PROFILE_EXPORT_AND_PARITY',
        'registration_sha256':sha(regpath),'grant_sha256':sha(grantpath),'source_commit':reg['source_commit'],
        'donor_rows':len(g),'donor_keys_sha256':reg['expected_keys_sha256'],
        'bundle_manifest_sha256':sha(out/'bundle/manifest.json'),'files':manifest['origins'],
        'quality_sha256':sha(out/'quality.json'),'quality_counts':{'source_rows':sum(s['rows'] for s in reg['sources'].values()),'accepted_rows':len(g),'excluded_rows':sum(s['rows'] for s in reg['sources'].values())-len(g)},
        'origins':ORIGINS,'final_donor_cutoff':'2026-10-11','final_future_query_min':'2026-10-12',
        'parity':reports,'tolerance':{'raw_atol':RAW_ATOL,'raw_rtol':RAW_RTOL,'encoded_atol':ENC_ATOL,'encoded_rtol':ENC_RTOL},
        'fits':0,'calibration_fits':0,'encoder_fits':0,'performance_metrics':False,'raw_donor_rows_saved':0,
        'actual_final_future_query_parity':False,'actual_historical_origins_only':True,
        'arrival':'UNKNOWN; corrected retrospective source snapshots','role':'2026 EXPOSED DEVELOPMENT'}
    (out/'receipt.pending.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--grant',required=True)
    a=p.parse_args();run(Path(a.registration),Path(a.grant))
