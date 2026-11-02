"""생성 자료만 사용한 표적·입력·분모·학습 경계 검사."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from experiments.gpu14h_prepitch import run,train
from experiments.ws_fullseason import data,prepare
from experiments.ws_seq import train_seq as base
sys.path.insert(0,str(run.ROOT/'tests'))
from test_ws_fullseason import rows


def tiny():
    r=json.loads((run.ROOT/'configs/EXP-WS-FULLSEASON-T-001.json').read_text())
    M=data.closure();g,_,fit,oof,y=data.project(rows(),r,M);raw,_=prepare.raw_features(g,fit,r,M);_,enc=M.fsf.encode(raw[fit],np.ones(fit.sum(),bool));r['base']['inputs']['feature_dimension']=enc['dimension']
    x,c,s=prepare.arrays(g,fit,oof,y,r,M)[:3]
    padded=np.zeros((len(x),199),np.float32);padded[:,:11]=x[:,:11]
    r['base']['inputs']['feature_dimension']=199
    r['base']['model'].update(trunk_hidden=[8,4],token_dim=4,encoder_layers=1,encoder_heads=2,encoder_ff=8)
    r['base']['training'].update(batch_size=4,eval_batch_size=4)
    return r,(padded,copy.deepcopy(c),copy.deepcopy(s))


class Contract(unittest.TestCase):
    def test_draft_before_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft';p.write_text('{"status":"DRAFT","execute":false}')
            with self.assertRaisesRegex(ValueError,'DRAFT'):run.gates(p,Path(d)/'absent','','x')

    def test_mapping_and_only_state11(self):
        labels=train.targets(np.arange(19,dtype=np.int64));self.assertEqual((labels<0).sum(),3)
        self.assertEqual([train.TYPES[i] for i in np.flatnonzero(labels<0)],['UNKNOWN','PO','UN'])
        for i,g in enumerate(train.CLASSES):self.assertEqual({train.TYPES[j] for j in np.flatnonzero(labels==i)},set(train.GROUPS[g]))
        x=np.arange(398,dtype=np.float32).reshape(2,199);other=x.copy();other[:,11:]=-999
        np.testing.assert_array_equal(train.features(x),train.features(other));self.assertEqual(train.features(x).shape,(2,11))

    def test_current_truth_and_dropped_physics_do_not_enter_prediction(self):
        r,(x,c,s)=tiny();ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],32)
        query=int(np.flatnonzero(c['evaluation'])[-1]);self.assertNotIn(query,ctx[query]);self.assertTrue((ctx[query]>=0).any())
        other_s=copy.deepcopy(s);other_c=copy.deepcopy(c);other_x=x.copy()
        other_s['pitch_type'][query]=1;other_s['outcome10'][query]=7;other_c['y10'][query]=7;other_c['y7'][query]=6;other_c['narrow_whiff'][query]=1;other_c['ground_ball_per_pitch'][query]=1;other_x[query,11:]=123
        for v in train.VARIANTS:
            recipe=train.recipe(r,v);m=train.build_model(recipe,v,0).eval()
            a=base.Data(torch,torch.device('cpu'),train.features(x),c,s,ctx);b=base.Data(torch,torch.device('cpu'),train.features(other_x),other_c,other_s,ctx)
            idx=torch.tensor([query]);keys=torch.as_tensor(c['keys'][[query]])
            with torch.no_grad():
                aa=m(a.x[idx],a.tokens(idx) if m.seq else None,keys)['leaf'];bb=m(b.x[idx],b.tokens(idx) if m.seq else None,keys)['leaf']
            torch.testing.assert_close(aa,bb,atol=0,rtol=0)

    def test_each_model_trains_only_mapped_fit_and_predicts_all_Q(self):
        r,(x,c,s)=tiny();s['pitch_type'][0]=train.TYPES.index('PO');s['pitch_type'][-1]=train.TYPES.index('UNKNOWN')
        for v in train.VARIANTS:
            recipe=train.recipe(r,v);recipe['training_rule']['epochs']=2
            with tempfile.TemporaryDirectory() as d:
                receipt=train.fit(recipe,v,0,(x,c,s),Path(d),synthetic=True)
                self.assertEqual(receipt['census']['FIT']['Q_pkg'],6);self.assertEqual(receipt['census']['FIT']['E9_pkg'],5)
                self.assertEqual(receipt['census']['OOF']['Q_pkg'],6);self.assertEqual(receipt['census']['OOF']['E9_pkg'],5);self.assertEqual(receipt['sample_presentations'],10)
                self.assertTrue(all(t['rows']==5 and t['every_key_once'] for t in receipt['consumption']))
                with np.load(Path(d)/'oof-predictions.npz') as z:self.assertEqual(z['probabilities'].shape,(6,9));run.probabilities(z['probabilities'])

    def test_frequency_FIT_only_and_unseen_fallback(self):
        p={'fit9':np.array([1,1,0,0],bool),'oof':np.array([0,0,1,1],bool),'y':np.array([0,1,0,-1]),'fit9_sha256':'generated','seq':{'pitcher':np.array([1,1,1,2]),'balls':np.array([0,0,3,0]),'strikes':np.array([0,0,2,0])}}
        table=run.fit_frequency(p);other=copy.deepcopy(p);other['y'][2]=8;other['y'][3]=4
        self.assertEqual(run.fit_frequency(other),table)
        q,counts=run.frequency(table,p);globalp=(np.array(table['global'])+4/9)/(2+4)
        np.testing.assert_allclose(q[1],globalp,atol=1e-15,rtol=0)
        self.assertEqual(counts['Q_pkg']['unseen_pitcher_global'],1);self.assertEqual(counts['E9_pkg']['unseen_pitcher_global'],0)
        self.assertTrue((q>0).all());self.assertEqual(table['rows'],2)

    def test_9metric_floor_and_empty_class_semantics(self):
        q=np.array([[0.,1.,0.,0.,0.,0.,0.,0.,0.]])
        got=run.comparison9(q,np.ones((1,9))/9,np.array([0]))
        self.assertAlmostEqual(got['candidate']['nll'],-np.log(1e-12));self.assertEqual(got['candidate']['zero_truth_probability'],1)
        rows=[{'variant':'state_t32','status':'INCOMPLETE_NOT_SELECTED','pooled':None}];self.assertEqual(run.select_variants(rows),[])


if __name__=='__main__':unittest.main(verbosity=2)
