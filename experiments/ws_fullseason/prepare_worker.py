"""Separate registered real feature production stage; no model fit or scores."""
import argparse
import json
import os
from pathlib import Path
import signal
import stat
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.ws_fullseason.audit import gates, checked_json, sha
from experiments.ws_fullseason import data, prepare


def run(regpath, grantpath):
    reg,recipe=gates(regpath,grantpath)
    os.fstat(int(os.environ['PITCHEEZY_FULLSEASON_LOCK_FD']))
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124)); signal.alarm(reg['bounds']['wall_seconds']+15)
    bundled=reg.get('quality_mode')=='bundled'
    quality=None
    if not bundled:
        for name in ['quality_receipt','quality_acceptance']:
            binding=reg[name]
            if sha(binding['path'])!=binding['sha256']: raise ValueError(name+' pin')
        quality=checked_json(reg['quality_receipt']['path'])
        if quality['status']!='COMPLETED' or quality['recipe_sha256']!=sha(reg['recipe']):
            raise ValueError('accepted quality recipe')
    import pandas as pd
    import pyarrow.parquet as pq
    M=data.closure(); source=reg['source']
    for field,cap in [('manifest',reg['bounds']['manifest_bytes']),('parquet',reg['bounds']['parquet_bytes'])]:
        p=Path(source[field])
        if not stat.S_ISREG(p.lstat().st_mode) or p.stat().st_size>cap or sha(p)!=source[field+'_sha256']:
            raise ValueError('input byte cap/hash '+field)
    M.cp.check_manifest(checked_json(source['manifest']),source['manifest_binding'],source['parquet'],source['manifest'])
    if reg['columns']!=M.cp.PROJECT: raise ValueError('column allowlist')
    footer=pq.ParquetFile(source['parquet'])
    if footer.metadata.num_rows!=source['rows'] or len(footer.schema_arrow.names)!=source['manifest_binding']['columns']:
        raise ValueError('footer rows/columns')
    raw=pq.read_table(source['parquet'],columns=reg['columns'],filters=[
        ('game_date','>=',pd.Timestamp(recipe['window']['fit'][0])),
        ('game_date','<=',pd.Timestamp(recipe['window']['oof'][1]))],use_threads=False).to_pandas()
    print('phase=project',flush=True)
    g,census,fit,oof,y=data.project(raw,recipe,M); del raw
    observed=data.audit_summary(g,census,fit,oof,y,M)
    out=Path(reg['output'])
    if bundled:
        quality=observed|{'status':'COMPLETED','stage':'QUALITY_NOT_PACKAGE_OR_FIT',
            'registration_sha256':sha(regpath),'recipe_sha256':sha(reg['recipe']),
            'source_sha256':source['parquet_sha256'],'retained_window':recipe['window'],
            'physical_decode_scope':'whole S2 bytes hashed; selected29 columns may decode outside predicate within rowgroups',
            'arrival':'UNKNOWN; corrected retrospective reconstruction'}
        (out/'quality.json').write_text(json.dumps(quality,indent=2,allow_nan=False)+'\n')
        quality_sha=sha(out/'quality.json')
    else:
        if observed['census']!=quality['census'] or observed['roles']!=quality['roles']:
            raise ValueError('quality/key/label denominators changed')
        quality_sha=reg['quality_receipt']['sha256']
    built=prepare.arrays(g,fit,oof,y,recipe,M)
    print('phase=save-package',flush=True)
    out=Path(reg['output']); manifest=prepare.save_package(out/'inputs',built,sha(reg['recipe']),
        {'source_sha256':source['parquet_sha256'],'quality_receipt_sha256':quality_sha},sha)
    receipt={'status':'COMPLETED','stage':'FEATURE_PACKAGE_NOT_FIT_OR_RELEASE','fold':recipe.get('fold','OOF1'),
        'registration_sha256':sha(regpath),'recipe_sha256':sha(reg['recipe']),
        'manifest_sha256':sha(out/'inputs/manifest.json'),'roles':manifest['roles'],
        'quality_sha256':quality_sha,'quality_summary':observed,
        'proof':manifest['proof'],'files':manifest['files'],'fits':0,'encoders':1,
        'optimizer_eligible_rows':int(fit.sum()),'donor_filter_does_not_drop_optimizer_rows':True,
        'data_role':recipe['data_role'],'arrival':'UNKNOWN; retrospective reconstruction'}
    (out/'receipt.pending.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--registration',required=True); p.add_argument('--grant',required=True)
    a=p.parse_args(); run(a.registration,a.grant)
