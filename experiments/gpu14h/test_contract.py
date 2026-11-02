"""Generated checks only; no registered research artifact is read."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.gpu14h import run, train


class Contract(unittest.TestCase):
    def test_draft_before_grant(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft.json';p.write_text('{"status":"DRAFT","execute":false}')
            with self.assertRaisesRegex(ValueError,'DRAFT'):
                run.gates(p,Path(d)/'never-exists','', 'H14-lr030-july-s0')

    def test_exact_recipe_deltas_and_geometry(self):
        import torch
        from experiments.ws_campaign.models import build_model
        torch.set_num_threads(1)
        original=json.loads((run.ROOT/'configs/EXP-WS-FULLSEASON-T-001.json').read_text())
        saved=copy.deepcopy(original)
        for name,patch in train.VARIANTS.items():
            r=train.recipe(original,name)
            restored=copy.deepcopy(r)
            for key in patch:
                group,field=key.split('.');restored['base'][group][field]=original['base'][group][field]
            self.assertEqual(restored,original)
            net=build_model(r,'T',0)
            self.assertGreater(sum(p.numel() for p in net.parameters()),0)
            self.assertLess(sum(p.numel()*p.element_size() for p in net.parameters())+11_000_000,64<<20)
        self.assertEqual(original,saved)
        with self.assertRaises(ValueError):train.recipe(original,'baseline-duplicate')

    def test_selection_and_incomplete(self):
        def row(v,n,b,status='COMPLETE'):
            return {'variant':v,'status':status,'pooled':{'delta':{'nll':n,'brier':b}}}
        rows=[row('z',-.01,-.01),row('a',-.0099995,0),row('bad',-.03,1e-8),row('null',-.5,-.5,'INCOMPLETE_NOT_SELECTED'),row('equal',0,-1)]
        self.assertEqual(run.select_variants(rows),['a','z'])
        self.assertEqual(run.select_variants([row('equal',0,0)]),[])

    def test_matched_probability_mean_and_metrics(self):
        q=np.array([[.7,.1,.05,.05,.04,.03,.03],[.1,.7,.05,.05,.04,.03,.03]])
        y=np.array([0,1]); same=run.comparison(q,q,y)
        self.assertEqual(same['delta'],dict(nll=0.,brier=0.,ece10=0.))
        self.assertAlmostEqual(same['candidate']['nll'],-np.log(.7))
        seeds=np.stack([q,np.roll(q,1,axis=1),q]); mean=seeds.mean(0)
        self.assertNotEqual(run.metrics(mean,y)['nll'],np.mean([run.metrics(p,y)['nll'] for p in seeds]))
        self.assertTrue(all(os.environ[k]=='1' for k in run.THREAD_ENV))

    def test_deadline_admission(self):
        cfg={'deadline_utc':run.DEADLINE,'recovery_grace_seconds':900,'minimum_fit_seconds':900,'limits':{'fit_wall_seconds':7200}}
        end=run.datetime.fromisoformat(run.DEADLINE.replace('Z','+00:00')).timestamp()
        self.assertEqual(run.effective_wall(cfg,end-10000),7200)
        self.assertEqual(run.effective_wall(cfg,end-2000),1100)
        with self.assertRaises(ValueError):run.effective_wall(cfg,end-1799)


if __name__=='__main__':unittest.main()
