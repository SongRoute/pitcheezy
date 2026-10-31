"""Generated parquet/temp package roundtrip for the coherent gate; no real data."""
import copy,json,os,signal,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from experiments.ws_fullseason import data,prepare,prepare_worker as worker
from experiments.ws_fullseason.audit import sha
from test_ws_fullseason import rows

class Bundled(unittest.TestCase):
 def run_case(self,fold,bad=False):
  root=Path(__file__).resolve().parents[1];recipe=json.loads((root/f'configs/EXP-WS-FULLSEASON-T-OOF{fold}.json').read_text());M=data.closure()
  frame=rows();mapping=dict(zip(['2026-03-25','2026-06-30','2026-07-01','2026-07-31'],['2026-03-25',recipe['window']['fit'][1],*recipe['window']['oof']]))
  frame['game_date']=pd.to_datetime(frame.game_date.map(mapping))
  g,_,fit,oof,y=data.project(frame,recipe,M);raw,_=prepare.raw_features(g,fit,recipe,M);recipe['base']['inputs']['feature_dimension']=prepare.fit_encoder(raw,fit)['dimension']
  if bad:frame.loc[0,'description']='invalid-generated-label'
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);source=d/'s.parquet';frame.to_parquet(source,index=False);manifest=d/'manifest.json';manifest.write_text('{}');rp=d/'recipe.json';rp.write_text(json.dumps(recipe));regpath=d/'reg.json';regpath.write_text('{}');out=d/'out';out.mkdir()
   reg={'quality_mode':'bundled','recipe':str(rp),'source':{'parquet':str(source),'manifest':str(manifest),'parquet_sha256':sha(source),'manifest_sha256':sha(manifest),'manifest_binding':{'columns':len(frame.columns)},'rows':len(frame)},'columns':M.cp.PROJECT,'bounds':{'wall_seconds':60,'manifest_bytes':1<<20,'parquet_bytes':1<<20},'output':str(out)}
   with (d/'lock').open('w') as lock,patch.dict(os.environ,{'PITCHEEZY_FULLSEASON_LOCK_FD':str(lock.fileno())}),patch.object(worker,'gates',return_value=(reg,recipe)),patch.object(worker.data,'closure',return_value=M),patch.object(M.cp,'check_manifest'):
    try:
     if bad:
      with self.assertRaises(M.cp.ProjectionStop):worker.run(regpath,d/'unused-grant.json')
      self.assertFalse((out/'inputs').exists());self.assertFalse((out/'quality.json').exists());return
     worker.run(regpath,d/'unused-grant.json')
    finally:signal.alarm(0)
   receipt=json.loads((out/'receipt.pending.json').read_text());quality=json.loads((out/'quality.json').read_text())
   self.assertEqual(receipt['quality_sha256'],sha(out/'quality.json'));self.assertEqual(receipt['roles'],{'FIT':6,'OOF':6});self.assertEqual(receipt['fits'],0)
   self.assertEqual(quality['roles']['OOF']['last_date'],recipe['window']['oof'][1]);self.assertEqual(receipt['fold'],f'OOF{fold}')
 def test_aug_and_sept_generated_roundtrip(self):
  for f in [2,3]:
   with self.subTest(fold=f):self.run_case(f)
 def test_quality_stop_prevents_assets(self):self.run_case(2,True)
if __name__=='__main__':unittest.main()
