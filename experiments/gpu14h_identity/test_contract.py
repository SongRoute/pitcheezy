"""실자료 없는 ID/RNG/분모/누락 대조 생성 검사."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.gpu14h_identity import run,train
from experiments.gpu14h_prepitch.test_contract import tiny
from experiments.ws_seq import train_seq as base


class Contract(unittest.TestCase):
    def test_draft_before_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft';p.write_text('{"status":"DRAFT","execute":false}')
            with self.assertRaisesRegex(ValueError,'DRAFT'):run.gates(p,Path(d)/'absent','','x')

    def test_fit_only_vocabulary_and_unknown(self):
        p=np.array([80,10,80,99,20]);fit=np.array([1,1,1,0,0],bool)
        ids=train.vocabulary(p,fit);np.testing.assert_array_equal(ids,[10,80]);np.testing.assert_array_equal(train.indices(p,ids),[2,1,2,0,0])
        changed=p.copy();changed[~fit]=1234;np.testing.assert_array_equal(ids,train.vocabulary(changed,fit))
        with self.assertRaisesRegex(ValueError,'integrity'):train.vocabulary(np.array([0,1]),np.array([1,0],bool))

    def test_initial_logits_and_training_rng_exact_for_all_architectures(self):
        r,(x,c,s)=tiny();fit9=c['fit']&(train.parent.targets(s['pitch_type'])>=0);ids=train.vocabulary(s['pitcher'],fit9)
        ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],32)
        state=base.Data(torch,torch.device('cpu'),train.parent.features(x),c,s,ctx);ident=base.Data(torch,torch.device('cpu'),train.features(x,s['pitcher'],ids),c,s,ctx)
        idx=torch.arange(len(x));keys=torch.as_tensor(c['keys'])
        for v in train.VARIANTS:
            torch.manual_seed(123);before=torch.get_rng_state().clone();m=train.build_model(train.recipe(r,v),v,0,len(ids)).eval();self.assertTrue(torch.equal(before,torch.get_rng_state()))
            b=train.parent.build_model(train.parent.recipe(r,train.ARCH[v]),train.ARCH[v],0).eval()
            with torch.no_grad():a=m(ident.x,ident.tokens(idx) if m.seq else None,keys)['leaf'];bb=b(state.x,state.tokens(idx) if b.seq else None,keys)['leaf']
            torch.testing.assert_close(a,bb,atol=0,rtol=0)
            altered=ident.x.clone();altered[:,11]=0
            with torch.no_grad():torch.testing.assert_close(m(altered,ident.tokens(idx) if m.seq else None,keys)['leaf'],bb,atol=0,rtol=0)

    def test_unknown_zero_after_update_and_fit_consumption(self):
        r,(x,c,s)=tiny();s['pitch_type'][0]=train.parent.TYPES.index('PO');s['pitcher'][c['evaluation']]=9999
        for v in train.VARIANTS:
            rcp=train.recipe(r,v);rcp['training_rule']['epochs']=2
            with tempfile.TemporaryDirectory() as d:
                z=train.fit(rcp,v,0,(x,c,s),Path(d),synthetic=True)
                self.assertEqual(z['sample_presentations'],10);self.assertEqual(z['OOF_unknown_pitchers'],6);self.assertTrue(z['unknown_vector_zero'])
                weights=torch.load(Path(d)/'model-state.pt',weights_only=True);torch.testing.assert_close(weights['embedding.weight'][0],torch.zeros(16),atol=0,rtol=0)
                vocab=json.loads((Path(d)/'pitcher-vocabulary.json').read_text());self.assertNotIn(9999,vocab['ids'])
                with np.load(Path(d)/'oof-predictions.npz') as q:self.assertEqual(q['probabilities'].shape,(6,9));run.parent.probabilities(q['probabilities'])

    def test_no_id_completion_exact_original_trainer_and_rng(self):
        r,package=tiny()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            aa=train.run_control(r,'pitcher_mlp',1,copy.deepcopy(package),Path(a),synthetic=True);rng=torch.get_rng_state().clone()
            bb=train.parent.run(r,'state_mlp',1,copy.deepcopy(package),Path(b),synthetic=True);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(aa['consumption'],bb['consumption']);self.assertEqual(aa['sample_presentations'],bb['sample_presentations'])
            wa=torch.load(Path(a)/'model-state.pt',weights_only=True);wb=torch.load(Path(b)/'model-state.pt',weights_only=True)
            for key in wa:torch.testing.assert_close(wa[key],wb[key],atol=0,rtol=0)
            with np.load(Path(a)/'oof-predictions.npz') as qa,np.load(Path(b)/'oof-predictions.npz') as qb:np.testing.assert_array_equal(qa['probabilities'],qb['probabilities'])

    def test_no_duplicate_control_and_complete_selection(self):
        j={'kind':'control','variant':'pitcher_t32','fold':'july','seed':1}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'never duplicated'):run.completion_gate(j,['state_t32'],d)
            old=Path(run.completion_gate(j,['state_mlp'],d));old.write_text('{}')
            with self.assertRaisesRegex(ValueError,'no copy'):run.completion_gate(j,['state_mlp'],d)
        self.assertEqual(run.control_origin('pitcher_t32',0,[]),'003')
        self.assertEqual(run.control_origin('pitcher_t32',1,['state_t32']),'003')
        self.assertEqual(run.control_origin('pitcher_t32',1,[]),'005')
        rows=[{'variant':v,'status':'COMPLETE','pooled':{'delta':{'nll':-.01,'brier':-.001}}} for v in reversed(list(train.VARIANTS))]
        self.assertEqual(run.select_variants(rows),['pitcher_c32'])
        rows[0]['status']='INCOMPLETE_NOT_SELECTED';rows[0]['pooled']=None;self.assertNotIn(rows[0]['variant'],run.select_variants(rows))


if __name__=='__main__':unittest.main(verbosity=2)
