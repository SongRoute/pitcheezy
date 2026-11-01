"""Small generated checks only; no real registration, arrays, or model states."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'tests')]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[name] = '1'
import numpy as np
import torch
from experiments.ws_fullseason_c import run, train
from experiments.ws_fullseason import calibrate as cal, data, prepare
from experiments.ws_campaign.models import build_model
from test_ws_fullseason import rows
TMP = Path('/Users/song/Projects/pitcheezy-pm-jobs/20261101-fullseason-c/tmp')
CONFIG = ROOT / 'configs/EXP-WS-FULLSEASON-C-001.json'


def windows():
    out = {}
    for i, fold in enumerate(run.FOLDS):
        k = np.array([[100+i,1,1],[100+i,1,2]],np.int64); y = np.array([0,1]); q = np.full((2,7),1/7)
        out[fold] = {'keys':k,'labels':y,'seeds':{s:(k.copy(),q.copy()) for s in run.SEEDS}}
    return out


class Contract(unittest.TestCase):
    def test_draft_stops_before_grant_or_payload(self):
        cfg=json.loads(CONFIG.read_text());cfg.update(status='DRAFT',execute=False)
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            path=Path(temp)/'draft.json';path.write_text(json.dumps(cfg))
            with patch.object(run,'sha',side_effect=AssertionError('no hash allowed')):
                with self.assertRaisesRegex(ValueError,'DRAFT'): run.gates(path,Path('/absent'), 'absent','C-july-s0')

    def test_recipe_only_changes_registered_arm_identity(self):
        cfg=json.loads(CONFIG.read_text())
        for fold in cfg['folds']: self.assertEqual(run.same_recipe(cfg,fold)['arm'],'C')
        cfg['folds']['july']['recipe']['base']['training']['lr'] *= 2
        with self.assertRaisesRegex(ValueError,'exact T'): run.same_recipe(cfg,'july')

    def test_chronological_C_tokens_and_geometry(self):
        cfg=json.loads(CONFIG.read_text());r=cfg['folds']['final']['recipe'];m=build_model(r,'C',0)
        tok={k:torch.tensor([[4,3,2,1],[0,0,0,0]]) for k in ('type','out','balls','strikes','same','pos')};tok['valid']=torch.tensor([[True,True,True,False],[False]*4])
        got=m.prepare_tokens(tok)
        self.assertEqual(got['type'][0].tolist(),[2,3,4,1]);self.assertFalse(got['valid'][1].any())
        self.assertIsInstance(m.net.encoder.lstm,torch.nn.LSTM)
        geometry={'tensors':len(m.state_dict()),'parameters':sum(x.numel() for x in m.state_dict().values())}
        self.assertEqual(geometry,cfg['expected_C_state_geometry'])

    def test_tiny_fit_no_oof_leakage_and_final_only(self):
        cfg=json.loads(CONFIG.read_text());r=copy.deepcopy(cfg['folds']['july']['recipe']);M=data.closure();g,_,fit,oof,y=data.project(rows(),r,M)
        raw,_=prepare.raw_features(g,fit,r,M);_,enc=M.fsf.encode(raw[fit],np.ones(fit.sum(),bool));r['base']['inputs']['feature_dimension']=enc['dimension']
        r['base']['model'].update(trunk_hidden=[8,4],token_dim=4,encoder_layers=1,encoder_heads=2,encoder_ff=8);r['base']['training'].update(batch_size=4,eval_batch_size=4);r['training_rule']['epochs']=2
        x,c,s=prepare.arrays(g,fit,oof,y,r,M)[:3];otherc=copy.deepcopy(c);others=copy.deepcopy(s);otherc['y10'][oof]=2;otherc['y7'][oof]=1;others['outcome10'][oof]=2
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            a,b,f=[Path(temp)/v for v in ('a','b','f')]
            for p in (a,b,f):p.mkdir()
            receipt=train.run(r,0,(x,c,s),a,synthetic=True);train.run(r,0,(x,otherc,others),b,synthetic=True)
            sa=torch.load(a/'model-state.pt',weights_only=True);sb=torch.load(b/'model-state.pt',weights_only=True)
            self.assertTrue(all(torch.equal(sa[k],sb[k]) for k in sa));self.assertEqual(receipt['sample_presentations'],12)
            fr=copy.deepcopy(r);fr['execution_stage']='FINAL_REFIT';fc=copy.deepcopy(c);fc['fit'][:]=True;fc['evaluation'][:]=False
            final=train.run(fr,0,(x,fc,s),f,synthetic=True);self.assertEqual(final['OOF_rows'],0);self.assertEqual({p.name for p in f.iterdir()},{'model-state.pt'})
            with self.assertRaises(ValueError):train.check_epoch(np.ones(len(x),np.int32),c['fit'],c['keys'])

    def test_oof_pair_identity_and_separate_calibration_score(self):
        w=windows();identity=cal.fit(w);r=run.oof_comparison(w,copy.deepcopy(w),identity,identity)
        self.assertEqual(r['comparison']['pooled']['metrics']['raw_OOF']['C_minus_T']['nll'],0)
        self.assertIn('own_pooled_calibration_resubstitution',r['comparison']['pooled']['metrics']);self.assertFalse(r['final_weights_evaluated'])
        changed=copy.deepcopy(w);changed['july']['labels'][0]=3
        with self.assertRaisesRegex(ValueError,'paired'):run.oof_comparison(w,changed,identity,identity)
        changed=copy.deepcopy(w);del changed['july']['seeds'][2]
        with self.assertRaises(ValueError):cal.pool(changed)
        changed=copy.deepcopy(w);changed['july']['seeds'][0]=(changed['july']['keys'][::-1],changed['july']['seeds'][0][1])
        with self.assertRaises(ValueError):cal.pool(changed)

    def test_one_pooled_optimizer_and_identity_on_registered_failure(self):
        w={}
        for i,fold in enumerate(run.FOLDS):
            k=np.column_stack((np.full(400,100+i),np.arange(1,401),np.ones(400))).astype(np.int64);y=np.arange(400)%7;q=np.full((400,7),1/7)
            w[fold]={'keys':k,'labels':y,'seeds':{s:(k,q) for s in run.SEEDS}}
        with patch.object(cal.base,'temperature',return_value=1.2) as optimizer:
            result=cal.fit(w);self.assertEqual(optimizer.call_count,1);self.assertEqual(result['rows'],1200)
        with patch.object(cal.base,'temperature',side_effect=cal.base.Stop('generated')):
            self.assertEqual(cal.fit(w)['kind'],'identity')
        w['july']['seeds'][0][1][0,0]=np.nan
        with self.assertRaises(ValueError):cal.fit(w)

    def test_primary_final_exact_keys_and_clip(self):
        records=[]
        for game,n in zip((849812,849820,849813,849811,849810,849818),(217,273,133,295,168,237)):
            records += [{'key':[game,1,p+1],'status':'scored','reason':None,'label':0,'observed_pitch_type':'FF','probabilities':[0.,1.,0.,0.,0.,0.,0.]} for p in range(n)]
        records += [{'key':[849812,2,p+1],'status':'abstain','reason':'prior_quality_unknown','label':None,'observed_pitch_type':None,'probabilities':None} for p in range(336)]
        result=run.paired_final_records(records,copy.deepcopy(records));self.assertAlmostEqual(result['comparison']['pooled']['C']['nll'],-np.log(1e-12));self.assertEqual(result['nll_clip'],1e-12)
        altered=copy.deepcopy(records);altered[0]['key'][2]=400
        with self.assertRaisesRegex(ValueError,'physical'):run.paired_final_records(records,altered)
        altered=copy.deepcopy(records);altered[0]['probabilities'][0]=1e-4
        with self.assertRaisesRegex(ValueError,'simplex'):run.paired_final_records(altered,records)

    def test_generated_C_loader_encoder_predict_composition(self):
        from test_seq_shadow_fullseason import generated_inputs
        cfg=json.loads(CONFIG.read_text());recipe=cfg['folds']['final']['recipe']
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            root=Path(temp);states={}
            def bound(p): return {'path':str(p),'bytes':p.stat().st_size,'sha256':run.sha(p)}
            for seed in run.SEEDS:
                p=root/f'state-{seed}.pt';torch.save(build_model(recipe,'C',seed).state_dict(),p);states[str(seed)]=bound(p)
            e=root/'encoder.json';e.write_text(json.dumps({'dimension':199,'median':[.1]*91,'mean':[.2]*91,'scale':[.7]*91,'continuous':list(range(36)),'knots':[[-1.]*36,[0.]*36,[1.]*36]}))
            c=root/'calibrator.json';c.write_text(json.dumps(cal.fit(windows())))
            ensemble=run.c_ensemble(cfg,states,bound(e),bound(c));_,tokens,keys=generated_inputs()
            encoded=ensemble.encode(np.zeros((4,91)));got=ensemble.predict(encoded,tokens,keys)
            self.assertEqual(encoded.shape,(4,199));self.assertEqual(set(got['raw_by_seed']),set(run.SEEDS))
            self.assertEqual(got['calibrated'].shape,(4,7));np.testing.assert_allclose(got['calibrated'].sum(1),1,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(got['calibrated'],got['raw_mean'])
            np.testing.assert_allclose(got['raw_mean'],np.mean(list(got['raw_by_seed'].values()),axis=0),atol=0,rtol=0)
            bad=copy.deepcopy(states);bad['0']['sha256']='0'*64
            with patch.object(torch,'load',side_effect=AssertionError('must not deserialize bad bytes')):
                with self.assertRaises(ValueError):run.c_ensemble(cfg,bad,bound(e),bound(c))

    def test_fresh_output_preserves_only_expected_appledouble(self):
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            root=Path(temp);(root/'tmp').mkdir();(root/'worker.log').touch();(root/'._tmp').write_bytes(b'generated');(root/'._worker.log').write_bytes(b'generated')
            run.fresh_worker_output(root);(root/'receipt.json').write_text('{}')
            with self.assertRaises(ValueError):run.fresh_worker_output(root)
            claim=root/'claim.json';run.write_json(claim,{'task':'generated'})
            with self.assertRaises(FileExistsError):run.write_json(claim,{'task':'duplicate'})

    def test_completed_chain_rejects_missing_or_wrong_recipe(self):
        cfg=json.loads(CONFIG.read_text())
        with tempfile.TemporaryDirectory(dir=TMP) as temp:
            root=Path(temp);r=root/'receipt.json';s=root/'supervisor.json';r.write_text(json.dumps({'status':'COMPLETED','arm':'T'}));s.write_text('{}')
            record={'receipt':{'path':str(r),'bytes':r.stat().st_size,'sha256':run.sha(r)},'supervisor':{'path':str(s),'bytes':s.stat().st_size,'sha256':run.sha(s)},'predictions':{'path':'not-read','bytes':1,'sha256':'a'*64}}
            with self.assertRaisesRegex(ValueError,'receipt chain'):run.completed_C(record,cfg,'july',0,'oof-predictions.npz')


if __name__=='__main__': unittest.main(verbosity=2)
