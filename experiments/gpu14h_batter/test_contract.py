"""생성 자료만: ID 두 축·초기 공정성·누수·원 control·선정 경계."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.gpu14h_batter import run,train
from experiments.gpu14h_prepitch.test_contract import tiny
from experiments.ws_seq import train_seq as base

class Contract(unittest.TestCase):
    def test_draft_before_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'draft';p.write_text('{"status":"DRAFT","execute":false}')
            with self.assertRaisesRegex(ValueError,'DRAFT'):run.gates(p,Path(d)/'absent','','x')

    def test_separate_fit_only_vocab_and_query_exclusions(self):
        r,(x,c,s)=tiny();fit=c['fit']&(train.parent.targets(s['pitch_type'])>=0);other=copy.deepcopy(s);other['pitcher'][~fit]=99991;other['batter'][~fit]=99992
        for v in train.VARIANTS:
            ids=train.vocabularies(s,fit,v);changed=train.vocabularies(other,fit,v)
            for k in ids:np.testing.assert_array_equal(ids[k],changed[k])
            xx=x.copy();xx[:,11:]=12345;ss=copy.deepcopy(s);ss['pitch_type'][:]=0;ss['outcome10'][:]=9
            np.testing.assert_array_equal(train.features(x,s,ids,v),train.features(xx,ss,ids,v))
            for k in ids:self.assertTrue((train.identity.indices(other[k][~fit],ids[k])==0).all())

    def test_zero_offsets_matching_factor_initialization_rng_and_logits(self):
        r,(x,c,s)=tiny();fit=c['fit']&(train.parent.targets(s['pitch_type'])>=0);ids=train.vocabularies(s,fit,'pitcher_batter');sizes={k:len(v) for k,v in ids.items()}
        ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],32);keys=torch.as_tensor(c['keys']);idx=torch.arange(len(x))
        state=base.Data(torch,torch.device('cpu'),train.parent.features(x),c,s,ctx);n=train.parent.build_model(train.parent.recipe(r,'state_t32'),'state_t32',0).eval()
        models={}
        for v in train.VARIANTS:
            torch.manual_seed(123);before=torch.get_rng_state().clone();m=train.build_model(train.recipe(r,v),v,0,sizes).eval();self.assertTrue(torch.equal(before,torch.get_rng_state()));models[v]=m
            data=base.Data(torch,torch.device('cpu'),train.features(x,s,ids,v),c,s,ctx)
            with torch.no_grad():torch.testing.assert_close(m(data.x,data.tokens(idx),keys)['leaf'],n(state.x,state.tokens(idx),keys)['leaf'],atol=0,rtol=0)
        p=train.identity.build_model(train.identity.recipe(r,train.CONTROL),train.CONTROL,0,sizes['pitcher'])
        torch.testing.assert_close(models['pitcher_batter'].pitcher.embedding.weight,p.embedding.weight,atol=0,rtol=0)
        torch.testing.assert_close(models['pitcher_batter'].batter_embedding.weight,models['batter_only'].embedding.weight,atol=0,rtol=0)
        self.assertTrue(train.unknown_zero(models['pitcher_batter'],'pitcher_batter'))

    def test_generated_learning_unknown_zero_and_mapped_consumption(self):
        r,(x,c,s)=tiny();s['pitch_type'][0]=train.parent.TYPES.index('PO');s['pitcher'][c['evaluation']]=99991;s['batter'][c['evaluation']]=99992
        for v in train.VARIANTS:
            recipe=train.recipe(r,v);recipe['training_rule']['epochs']=2
            with tempfile.TemporaryDirectory() as d:
                got=train.fit(recipe,v,0,(x,c,s),Path(d),synthetic=True);self.assertEqual(got['sample_presentations'],10);self.assertTrue(got['unknown_vector_zero']);self.assertTrue(all(z==6 for z in got['OOF_unknown'].values()))
                vocab=json.loads((Path(d)/'identity-vocabularies.json').read_text());self.assertTrue(all(99991 not in ids and 99992 not in ids for ids in vocab['factors'].values()))
                with np.load(Path(d)/'oof-predictions.npz',allow_pickle=False) as q:self.assertEqual(q['probabilities'].shape,(6,9));run.parent.probabilities(q['probabilities'])

    def test_control_exact005_weights_probabilities_rng(self):
        r,package=tiny()
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            got=train.run_control(r,1,copy.deepcopy(package),Path(a),synthetic=True);rng=torch.get_rng_state().clone();ref=train.identity.run(r,train.CONTROL,1,copy.deepcopy(package),Path(b),synthetic=True)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(got['consumption'],ref['consumption']);self.assertEqual(got['vocabulary_sha256'],ref['vocabulary_sha256'])
            aa=torch.load(Path(a)/'model-state.pt',weights_only=True);bb=torch.load(Path(b)/'model-state.pt',weights_only=True)
            self.assertTrue(all(torch.equal(aa[k],bb[k]) for k in aa))
            with np.load(Path(a)/'oof-predictions.npz',allow_pickle=False) as qa,np.load(Path(b)/'oof-predictions.npz',allow_pickle=False) as qb:np.testing.assert_array_equal(qa['probabilities'],qb['probabilities'])

    def test_three_comparisons_raw_rank_and_no_duplicate_control(self):
        def row(v,n):return {'variant':v,'status':'COMPLETE','pooled':{'candidate':{'nll':n},'delta':{'nll':-.01,'brier':0.}},'vs_frozen_frequency':{'delta':{'nll':-.1,'brier':-.01}},'vs_noid_T':{'delta':{'nll':-.005,'brier':0.}}}
        rows=[row('batter_only',1.5),row('pitcher_batter',1.4)];self.assertEqual(run.select_variants(rows),['pitcher_batter']);rows[1]['vs_noid_T']['delta']['nll']=0;self.assertEqual(run.select_variants(rows),['batter_only']);rows[0]['vs_frozen_frequency']['delta']['brier']=1e-12;self.assertEqual(run.select_variants(rows),[])
        j={'kind':'control','variant':train.CONTROL,'fold':'july','seed':1}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'never duplicated'):run.completion_gate(j,[train.CONTROL],d)
            old=Path(run.completion_gate(j,['pitcher_mlp'],d));old.write_text('{"state":"FAILED"}')
            with self.assertRaisesRegex(ValueError,'no copy/retry'):run.completion_gate(j,['pitcher_mlp'],d)
        records=[dict(fold=f,seed=0) for f in run.FOLDS];self.assertEqual(len(run.factor_partition(records,[],[0])),3)
        with self.assertRaises(ValueError):run.factor_partition(records[:2],[],[0])

if __name__=='__main__':unittest.main(verbosity=2)
