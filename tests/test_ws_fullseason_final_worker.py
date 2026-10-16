"""Generated two-parquet final pipeline; no real source/assets accessed."""
import copy,json,os,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from experiments.ws_fullseason import data,prepare,final_source,final_worker as worker
from experiments.ws_fullseason.audit import sha
from test_ws_fullseason import rows

class FinalWorker(unittest.TestCase):
 def check_case(self,bad=None):
  root=Path(__file__).resolve().parents[1];recipe=json.loads((root/'configs/EXP-WS-FULLSEASON-T-FINAL.json').read_text());M=data.closure()
  regular=rows()[M.cp.PROJECT];post=regular.iloc[:3].copy();post['game_pk']+=100;post['game_type']='F';post['game_date']='2026-10-01'
  frames={'S2':regular,'I3':post};authority={'mode':'PRODUCTION_FIT','root_approved':True,'sealed_assets_read':False,'sources':copy.deepcopy(final_source.SOURCES)}
  g,_,fit,oof,y=final_source.project(frames,recipe,M,authority);raw,_=prepare.raw_features(g,fit,recipe,M);recipe['base']['inputs']['feature_dimension']=prepare.fit_encoder(raw,fit)['dimension']
  if bad=='quality':post.loc[post.index[0],'description']='invalid-generated-label'
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);sources={};pins={}
   for name,frame in frames.items():
    folder=d/name;folder.mkdir();p=folder/'source.parquet';frame.to_parquet(p,index=False)
    node={'file':p.name,'start':'2026-03-25','end':'2026-10-11','sha256':sha(p)}
    body={'dry_run':False,'version':name}
    if name=='S2':body['seasons']={'2026':node}
    else:body['incremental']=node
    mp=folder/'manifest.json';mp.write_text(json.dumps(body))
    sources[name]={'parquet':str(p),'manifest':str(mp),'parquet_sha256':sha(p),'manifest_sha256':sha(mp),'version':name,'rows':len(frame),'columns':len(frame.columns),'covers':[node['start'],node['end']]}
    pins[name]={'version':name,'sha256':sha(p),'manifest_sha256':sha(mp)}
   if bad=='manifest':sources['I3']['covers']=['2026-01-01','2026-10-11']
   rp=d/'recipe.json';rp.write_text(json.dumps(recipe));regpath=d/'reg.json';regpath.write_text('{}');out=d/'out';out.mkdir();authority['sources']=pins
   reg={'recipe':str(rp),'sources':sources,'columns':M.cp.PROJECT,'final_authority':authority,'bounds':{'wall_seconds':60,'manifest_bytes':1<<20,'parquet_bytes':1<<20},'output':str(out)}
   with (d/'lock').open('w') as lock,patch.dict(os.environ,{'PITCHEEZY_FULLSEASON_LOCK_FD':str(lock.fileno())}),patch.object(worker,'gates',return_value=(reg,recipe)),patch.object(worker.data,'closure',return_value=M),patch.object(final_source,'SOURCES',pins):
    try:
     if bad:
      with self.assertRaises((ValueError,M.cp.ProjectionStop)):worker.run(regpath,d/'grant')
      self.assertFalse((out/'inputs').exists());return
     worker.run(regpath,d/'grant')
    finally:signal.alarm(0)
   r=json.loads((out/'receipt.pending.json').read_text());q=json.loads((out/'quality.json').read_text())
   self.assertEqual(r['roles'],{'FIT':15,'OOF':0});self.assertEqual(r['fits'],0)
   self.assertEqual(q['roles']['OOF']['first_date'],None);self.assertEqual(q['census']['accepted_game_types']['F'],3)
   self.assertFalse(r['proof']['sequence']['earlier_oof_outcomes_used'])
   self.assertEqual(r['quality_sha256'],sha(out/'quality.json'));self.assertEqual(r['manifest_sha256'],sha(out/'inputs/manifest.json'))
 def test_generated_package(self):self.check_case()
 def test_manifest_mismatch_prevents_package(self):self.check_case('manifest')
 def test_quality_stop_prevents_package(self):self.check_case('quality')
if __name__=='__main__':unittest.main()
