"""Coherent worker checks with two generated parquets, never real artifacts."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from experiments.ws_fullseason import data,final_source
from experiments.ws_fullseason.audit import sha
from test_ws_fullseason import rows
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('profile_worker',ROOT/'scripts/fullseason_profile_export.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


def run_case(tmp_path,bad=False):
    recipe=json.loads((ROOT/'configs/EXP-WS-FULLSEASON-T-FINAL.json').read_text());M=data.closure()
    regular=rows().iloc[:9][M.cp.PROJECT].copy();regular['game_date']=np.repeat(['2026-03-25','2026-07-01','2026-10-11'],3)
    post=regular.iloc[6:].copy();post['game_type']='L';regular=regular.iloc[:6]
    frames={'S2':regular,'I3':post};sources={};pins={}
    for name,frame in frames.items():
        d=tmp_path/name;d.mkdir();p=d/'source.parquet';frame.to_parquet(p,index=False)
        node={'file':p.name,'start':'2026-03-25','end':'2026-10-11','sha256':sha(p)}
        body={'dry_run':False,'version':name}
        if name=='S2':body['seasons']={'2026':node}
        else:body['incremental']=node
        mp=d/'manifest.json';mp.write_text(json.dumps(body))
        sources[name]={'parquet':str(p),'manifest':str(mp),'parquet_sha256':sha(p),'manifest_sha256':sha(mp),'version':name,'rows':len(frame),'columns':len(frame.columns),'covers':[node['start'],node['end']]}
        pins[name]={'version':name,'sha256':sha(p),'manifest_sha256':sha(mp)}
    authority={'mode':'PRODUCTION_FIT','root_approved':True,'sealed_assets_read':False,'sources':pins}
    with patch.object(final_source,'SOURCES',pins):g,*_=final_source.project(frames,recipe,M,authority)
    rp=tmp_path/'recipe.json';rp.write_text(json.dumps(recipe));regpath=tmp_path/'reg.json';regpath.write_text('{}');grant=tmp_path/'grant.json';grant.write_text('{}')
    encoder={'median':[0.]*91,'mean':[0.]*91,'scale':[1.]*91,'continuous':list(range(36)),'knots':[[0.]*36]*3,'dimension':199}
    ep=tmp_path/'encoder.json';ep.write_text(json.dumps(encoder));out=tmp_path/'output';out.mkdir()
    reg={'origins':w.ORIGINS,'parity_dates':w.PARITY_DATES,'query_rule':'min30 per exact date; floor(k*(N-1)/(m-1)); full-date previous lookup first',
         'recipe':str(rp),'sources':sources,'columns':M.cp.PROJECT,'final_authority':authority,
         'expected_rows':len(g),'expected_keys_sha256':data.key_digest(g[M.cp.KEY].to_numpy()),'source_commit':'generated',
         'encoder':{'path':str(ep),'bytes':ep.stat().st_size,'sha256':sha(ep)},
         'bounds':{'wall_seconds':60,'manifest_bytes':1<<20,'parquet_bytes':1<<20},'output':str(out)}
    if bad:reg['sources']['I3']['covers']=['2026-01-01','2026-10-11']
    with (tmp_path/'lock').open('w') as lock,patch.dict(os.environ,{'PITCHEEZY_FULLSEASON_LOCK_FD':str(lock.fileno())}),patch.object(w,'gates',return_value=(reg,recipe)),patch.object(final_source,'SOURCES',pins):
        try:
            if bad:
                with pytest.raises(ValueError):w.run(regpath,grant)
                assert not (out/'bundle').exists();return
            w.run(regpath,grant)
        finally:signal.alarm(0)
    receipt=json.loads((out/'receipt.pending.json').read_text())
    assert receipt['donor_rows']==9 and receipt['fits']==0 and receipt['encoder_fits']==0
    assert len(receipt['origins'])==4 and sum(v['rows'] for v in receipt['parity'].values())==9
    assert all(v['encoded_exact'] for v in receipt['parity'].values())
    assert receipt['bundle_manifest_sha256']==sha(out/'bundle/manifest.json')


def test_generated_coherent_export_roundtrip_parity(tmp_path):run_case(tmp_path)
def test_manifest_stop_before_export(tmp_path):run_case(tmp_path,True)


def test_fixed_query_selection_and_mismatch_stop():
    q=w.selected_rows(300);assert len(q)==30 and q[0]==0 and q[-1]==299
    with pytest.raises(ValueError):w.selected_rows(0)
    raw=np.zeros((1,91));other=raw.copy();other[0,0]=.01
    with pytest.raises(ValueError):w.compare(raw,other,{},None)
