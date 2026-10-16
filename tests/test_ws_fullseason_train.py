"""Tiny generated CPU model checks; no real model or data read."""
import copy
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.ws_fullseason import data,prepare,train,run_gpu
from test_ws_fullseason import rows


class FixedTrain(unittest.TestCase):
    def test_exact_granted_gpu_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'recipe.json').write_text(json.dumps({'id':'generated','training_rule':{'epochs':30}}));(root/'job.env').write_text('COMMIT=abcdef1\n')
            g={'execute':True,'verdict':'PASS','authorized_by':'root','author':'research_readiness','reviewer':'other','seeds':[0,1,2],
                'recipe':'recipe.json','recipe_id':'generated','source_pins':{'recipe.json':run_gpu.sha(root/'recipe.json')},'commit':'abcdef123',
                'wall_seconds_per_seed':3600,'rss_bytes':8<<30,'vram_bytes':16<<30,'output_bytes_per_seed':1<<29,'aggregate_wall_seconds':120,'aggregate_output_bytes':1<<20}
            with patch.object(run_gpu,'ROOT',root),patch.object(run_gpu.sys,'platform','linux'):
                for device in [0,3]:
                    g['gpu_index']=device;(root/'grant.json').write_text(json.dumps(g))
                    with patch.dict(os.environ,{'GPU_JOB_ID':'generated','GPU_JOB_DIR':str(root),'CUDA_VISIBLE_DEVICES':str(device)}):
                        self.assertEqual(run_gpu.checked_grant(root/'grant.json')[0]['gpu_index'],device)
                    with patch.dict(os.environ,{'GPU_JOB_ID':'generated','GPU_JOB_DIR':str(root),'CUDA_VISIBLE_DEVICES':'2'}):
                        with self.assertRaises(ValueError):run_gpu.checked_grant(root/'grant.json')
                g['gpu_index']=2;(root/'grant.json').write_text(json.dumps(g))
                with self.assertRaises(ValueError):run_gpu.checked_grant(root/'grant.json')

    def test_final_aggregate_receipt_gate(self):
        keys=np.arange(6).reshape(2,3);labels=np.array([0,1]);q=np.eye(7)[labels]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir();out=root/'outputs';out.mkdir()
            (root/'recipe.json').write_text('{}')
            np.savez(inputs/'corpus-subset.npz',keys=keys,y7=labels,evaluation=np.ones(2,bool))
            (inputs/'manifest.json').write_text(json.dumps({'files':{'corpus-subset.npz':{'sha256':run_gpu.sha(inputs/'corpus-subset.npz')}}}))
            g={'inputs':'inputs','recipe':'recipe.json','input_manifest_sha256':run_gpu.sha(inputs/'manifest.json'),
                'expected_rows':{'OOF':2},'expected_key_sha256':{'OOF':data.key_digest(keys)},'commit':'synthetic','aggregate_output_bytes':1<<20}
            records=[]
            for seed in [0,1,2]:
                d=out/f'seed{seed}';d.mkdir();np.savez(d/'oof-predictions.npz',keys=keys,probabilities=q)
                r={'status':'COMPLETED','seed':seed,'grant_sha256':'g','recipe_sha256':run_gpu.sha(root/'recipe.json'),
                    'input_manifest_sha256':g['input_manifest_sha256'],'source_commit':'synthetic',
                    'outputs_sha256':{'oof-predictions.npz':run_gpu.sha(d/'oof-predictions.npz')}}
                (d/'receipt.json').write_text(json.dumps(r));rh=run_gpu.sha(d/'receipt.json')
                supervisor={'status':'COMPLETED','child_reaped':True,'returncode':0,'receipt_sha256':rh,'grant_sha256':'g'}
                (d/'SUPERVISOR.json').write_text(json.dumps(supervisor))
                records.append({'seed':seed,'receipt_sha256':rh,'supervisor_sha256':run_gpu.sha(d/'SUPERVISOR.json')})
            with patch.object(run_gpu,'ROOT',root):
                self.assertEqual(run_gpu.final_aggregate(g,out,records,'g'),run_gpu.sha(out/'aggregate.json'))
                with self.assertRaises(ValueError):run_gpu.final_aggregate(g,out,records,'wrong-grant')
                supervisor['child_reaped']=False;(d/'SUPERVISOR.json').write_text(json.dumps(supervisor))
                records[-1]['supervisor_sha256']=run_gpu.sha(d/'SUPERVISOR.json')
                with self.assertRaises(ValueError):run_gpu.final_aggregate(g,out,records,'g')

    def test_ensemble_only_and_exact_keys(self):
        keys=np.arange(6).reshape(2,3);labels=np.array([0,1]);q=np.eye(7)[labels]
        r=run_gpu.ensemble_metrics([(keys,q)]*3,labels,keys)
        self.assertEqual((r['nll'],r['brier'],r['ece_top_label_10bin']),(0,0,0))
        uniform=np.full((2,7),1/7)
        r=run_gpu.ensemble_metrics([(keys,uniform)]*3,labels,keys)
        self.assertAlmostEqual(r['nll'],np.log(7));self.assertAlmostEqual(r['brier'],6/7)
        with self.assertRaises(ValueError):run_gpu.ensemble_metrics([(keys,q)]*2,labels,keys)
        with self.assertRaises(ValueError):run_gpu.ensemble_metrics([(keys,q),(keys,q),(keys[::-1],q)],labels,keys)

    def test_consumption_rejects_missing_duplicate_and_nonfit(self):
        fit=np.array([True,True,False]); keys=np.arange(9).reshape(3,3)
        for seen in [np.array([1,0,0]),np.array([2,1,0]),np.array([1,1,1])]:
            with self.assertRaises(ValueError):train.check_epoch(seen,fit,keys)
        self.assertEqual(train.check_epoch(np.array([1,1,0]),fit,keys)['rows'],2)

    def test_final_epoch_and_no_oof_optimizer_leakage(self):
        recipe=json.loads((Path(__file__).resolve().parents[1]/'configs/EXP-WS-FULLSEASON-T-001.json').read_text())
        M=data.closure();g,_,fit,oof,y=data.project(rows(),recipe,M)
        raw,_=prepare.raw_features(g,fit,recipe,M)
        _,enc=M.fsf.encode(raw[fit],np.ones(fit.sum(),bool))
        recipe['base']['inputs']['feature_dimension']=enc['dimension']
        recipe['base']['model'].update(trunk_hidden=[8,4],token_dim=4,encoder_layers=1,encoder_heads=2,encoder_ff=8)
        recipe['base']['training'].update(batch_size=4,eval_batch_size=4)
        recipe['training_rule']['epochs']=2
        built=prepare.arrays(g,fit,oof,y,recipe,M);x,c,s=built[:3]
        altered_c={k:v.copy() for k,v in c.items()};altered_s={k:v.copy() for k,v in s.items()}
        altered_c['y10'][oof]=2;altered_c['y7'][oof]=1;altered_s['outcome10'][oof]=2
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir()
            r=train.run(recipe,0,(x,c,s),a,synthetic=True)
            train.run(recipe,0,(x,altered_c,altered_s),b,synthetic=True)
            left=torch.load(a/'model-state.pt',weights_only=True);right=torch.load(b/'model-state.pt',weights_only=True)
            self.assertTrue(all(torch.equal(left[k],right[k]) for k in left))
            self.assertEqual(r['sample_presentations'],12);self.assertEqual(r['optimization_steps'],4)
            self.assertEqual(len(r['consumption']),2);self.assertEqual(r['SELECT_queries'],0)
            with np.load(a/'oof-predictions.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['keys'],c['keys'][oof]);self.assertEqual(z['probabilities'].shape,(6,7))
            self.assertFalse(r['metrics_computed'])


if __name__=='__main__':unittest.main()
