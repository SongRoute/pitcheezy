"""Registered S2+I3 final-candidate quality and FIT-only package; no model fit."""
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.ws_fullseason.audit import gates,checked_json,sha
from experiments.ws_fullseason import data,prepare,final_source


def read_source(name,source,columns,bounds):
    import pyarrow.parquet as pq
    for field,cap in [('manifest',bounds['manifest_bytes']),('parquet',bounds['parquet_bytes'])]:
        p=Path(source[field])
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size>cap or sha(p)!=source[field+'_sha256']:
            raise ValueError('source byte custody '+name+' '+field)
    body=checked_json(source['manifest'],bounds['manifest_bytes'])
    node=body['seasons']['2026'] if name=='S2' else body['incremental']
    # Reuse the earlier registered metadata worker's structural manifest contract.
    if (body.get('dry_run') is not False or body.get('version')!=source['version']
            or Path(source['manifest']).parent.name!=source['version']
            or [node['start'],node['end']]!=source['covers']
            or (Path(source['manifest']).parent/node['file']).resolve()!=Path(source['parquet']).resolve()
            or node['sha256']!=source['parquet_sha256']):
        raise ValueError('source manifest binding '+name)
    pf=pq.ParquetFile(source['parquet'])
    if (pf.metadata.num_rows!=source['rows'] or len(pf.schema_arrow.names)!=source['columns']
            or not set(columns)<=set(pf.schema_arrow.names)):
        raise ValueError('source footer/schema '+name)
    # All rows in the exact two snapshots are accounted for, including exclusions.
    return pf.read(columns=columns,use_threads=False).to_pandas()


def run(regpath,grantpath):
    reg,recipe=gates(regpath,grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.alarm(reg['bounds']['wall_seconds']+15)
    if recipe.get('execution_stage')!='FINAL_REFIT' or recipe['window']!={'fit':['2026-03-25','2026-10-11'],'oof':None}:
        raise ValueError('final fixed cutoff/role')
    M=data.closure()
    if reg['columns']!=M.cp.PROJECT or set(reg['sources'])!=set(final_source.SOURCES):raise ValueError('exact final sources/columns')
    for name,s in reg['sources'].items():
        binding=final_source.SOURCES[name]
        if {'version':s['version'],'sha256':s['parquet_sha256'],'manifest_sha256':s['manifest_sha256']}!=binding:
            raise ValueError('fixed source identity '+name)
    frames={name:read_source(name,reg['sources'][name],reg['columns'],reg['bounds']) for name in ['S2','I3']}
    print('phase=final-project',flush=True)
    g,census,fit,oof,y=final_source.project(frames,recipe,M,reg['final_authority']);del frames
    observed=data.audit_summary(g,census,fit,oof,y,M)
    out=Path(reg['output']);quality=observed|{'status':'COMPLETED','stage':'FINAL_QUALITY_NOT_FIT_OR_RELEASE',
        'registration_sha256':sha(regpath),'recipe_sha256':sha(reg['recipe']),
        'source_bindings':reg['sources'],'retained_window':recipe['window'],
        'physical_decode_scope':'whole two pinned files hashed; exactly29 selected columns across all source rows decoded',
        'arrival':'UNKNOWN; corrected retrospective reconstruction',
        'completeness':'known fixed snapshot; prior accepted cancellation interpretation gives2429/2429 regular pitched and26/26 postseason throughOct11; no latest/beyond-cutoff completeness claim'}
    (out/'quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False)+'\n');qh=sha(out/'quality.json')
    built=prepare.arrays(g,fit,oof,y,recipe,M)
    print('phase=save-package',flush=True)
    manifest=prepare.save_package(out/'inputs',built,sha(reg['recipe']),{'sources':final_source.SOURCES,'quality_receipt_sha256':qh},sha)
    receipt={'status':'COMPLETED','stage':'FINAL_FEATURE_PACKAGE_NOT_FIT_OR_RELEASE','fold':'FINAL',
        'registration_sha256':sha(regpath),'recipe_sha256':sha(reg['recipe']),
        'manifest_sha256':sha(out/'inputs/manifest.json'),'roles':manifest['roles'],
        'quality_sha256':qh,'quality_summary':observed,'proof':manifest['proof'],'files':manifest['files'],
        'fits':0,'encoders':1,'optimizer_eligible_rows':int(fit.sum()),
        'data_role':recipe['data_role'],'arrival':'UNKNOWN; retrospective reconstruction'}
    (out/'receipt.pending.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registration',required=True);p.add_argument('--grant',required=True)
    a=p.parse_args();run(a.registration,a.grant)
