"""Generated-only strictly-prior producer checks; no payload reads."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from experiments.ws_fullseason.audit import sha
import numpy as np
from experiments.ws_fullseason import data, prepare
from experiments.ws_seq.prepare_inputs import sequence_arrays
from experiments.ws_seq import train_seq as base
from test_ws_fullseason import rows


class Producer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M=data.closure()
        cls.recipe=json.loads((Path(__file__).resolve().parents[1]/'configs/EXP-WS-FULLSEASON-T-001.json').read_text())

    def test_future_oof_changes_leave_fit_encoder_and_all_date_profiles_identical(self):
        g,_,fit,oof,y=data.project(rows(),self.recipe,self.M)
        raw,_=prepare.raw_features(g,fit,self.recipe,self.M)
        changed=g.copy()
        changed.loc[oof,'outcome10']=3; changed.loc[oof,'description']='hit_into_play'
        changed.loc[oof,'bb_type']='ground_ball'
        for col in self.M.fsf.PHYSICS: changed.loc[oof,col]=777
        raw2,_=prepare.raw_features(changed,fit,self.recipe,self.M)
        np.testing.assert_array_equal(raw,raw2)
        x,e=self.M.fsf.encode(raw[fit],np.ones(fit.sum(),bool))
        x2,e2=self.M.fsf.encode(raw2[fit],np.ones(fit.sum(),bool))
        self.assertEqual(e,e2); np.testing.assert_array_equal(x,x2)

    def test_same_day_realized_values_do_not_affect_that_days_features(self):
        g,_,fit,_,_=data.project(rows(),self.recipe,self.M)
        raw,_=prepare.raw_features(g,fit,self.recipe,self.M)
        changed=g.copy(); pivot=changed.game_date.eq('2026-06-30')
        for col in self.M.fsf.PHYSICS: changed.loc[pivot,col]=777
        other,_=prepare.raw_features(changed,fit,self.recipe,self.M)
        np.testing.assert_array_equal(raw[g.game_date<='2026-06-30'],other[g.game_date<='2026-06-30'])
        self.assertFalse(np.array_equal(raw[g.game_date>'2026-06-30'],other[g.game_date>'2026-06-30'],equal_nan=True))

    def test_sequence_current_and_future_excluded_but_prior_oof_used(self):
        g,_,fit,oof,y=data.project(rows(),self.recipe,self.M); s=sequence_arrays(g,self.M)
        ctx=base.build_context_index(s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'],32)
        base.check_context(ctx,s['pitcher'],s['game_pk'],s['at_bat_number'],s['pitch_number'])
        idx=np.flatnonzero(oof)[1]; valid=ctx[idx]>=0; previous=ctx[idx][valid]
        self.assertTrue(len(previous)>0 and oof[previous].all())
        changed=s['outcome10'].copy(); changed[idx:]=9
        np.testing.assert_array_equal(s['outcome10'][previous],changed[previous])
        self.assertTrue((previous<idx).all())

    def test_generated_package_roundtrip_and_effective_donor_proof(self):
        frame=rows(); frame.loc[0,'release_speed']=np.nan
        g,_,fit,oof,y=data.project(frame,self.recipe,self.M)
        raw,_=prepare.raw_features(g,fit,self.recipe,self.M)
        _,enc=self.M.fsf.encode(raw[fit],np.ones(fit.sum(),bool))
        # Only this synthetic fixture uses its FIT-derived dimension; real recipe stays199.
        cfg=copy.deepcopy(self.recipe); cfg['base']['inputs']['feature_dimension']=enc['dimension']
        built=prepare.arrays(g,fit,oof,y,cfg,self.M)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'package'
            man=prepare.save_package(folder,built,'synthetic-only',{'mode':'generated'},sha)
            x,c,s=base.load_package(folder,cfg['base'])
            self.assertTrue(c['fit'].sum()==6 and c['evaluation'].sum()==6)
            self.assertEqual(c['tuning'].sum()+c['calibration'].sum(),0)
            np.testing.assert_array_equal(c['keys'],g[self.M.cp.KEY].to_numpy())
            for name,binding in man['files'].items(): self.assertEqual(sha(folder/name),binding['sha256'])
            proof=man['proof']; self.assertEqual(proof['date_profile_pool']['rows'],6)
            self.assertEqual(proof['effective_donor_subsets']['outcome_transfer_complete_physics']['rows'],5)
            self.assertEqual(proof['effective_donor_subsets']['physics_finite_by_column']['release_speed']['rows'],5)
            self.assertEqual(proof['encoder_fit']['keys_sha256'],data.key_digest(c['keys'][fit]))
            self.assertTrue(np.isfinite(x).all())

    def test_bounded_encoder_exactly_matches_existing_matrix_reductions(self):
        rng=np.random.default_rng(42); raw=rng.normal(size=(513,91)); raw[:,0]=3;raw[:,1]=np.nan
        raw[::7,2:8]=np.nan;raw[::13,9]=np.inf;fit=np.arange(len(raw))%4!=0
        expected_x,expected=self.M.fsf.encode(raw[fit],np.ones(fit.sum(),bool))
        observed=prepare.fit_encoder(raw,fit)
        self.assertEqual(expected,observed)
        np.testing.assert_array_equal(expected_x,self.M.safe.apply_encoder(raw[fit],observed))
        full=self.M.safe.apply_encoder(raw,observed)
        chunked=np.concatenate([self.M.safe.apply_encoder(raw[i:i+17],observed) for i in range(0,len(raw),17)])
        np.testing.assert_array_equal(full,chunked)

    def test_whole_date_query_chunks_match_original_features(self):
        g,_,fit,_,_=data.project(rows(),self.recipe,self.M)
        expected,audit=self.M.scale.features(g.loc[fit,self.M.safe.DONOR].reset_index(drop=True),g,self.recipe['window']['fit'][1],self.M)
        actual,actual_audit=prepare.raw_features(g,fit,self.recipe,self.M,chunk_rows=3)
        np.testing.assert_array_equal(expected,actual);self.assertEqual(audit,actual_audit)

    def test_whole_game_context_proof_matches_full_index(self):
        g,_,fit,oof,_=data.project(rows(),self.recipe,self.M);seq=sequence_arrays(g,self.M)
        full=base.build_context_index(*[seq[n] for n in ['pitcher','game_pk','at_bat_number','pitch_number']],32)
        proof=prepare.context_proof(g,seq,fit,oof,32,chunk_rows=3)
        self.assertEqual(proof['edges'],int((full>=0).sum()))
        self.assertEqual(proof['oof_edges'],int((full[oof]>=0).sum()))
        for indices in prepare.group_batches(g,'game_pk',3):
            local=base.build_context_index(*[seq[n][indices] for n in ['pitcher','game_pk','at_bat_number','pitch_number']],32)
            mapped=np.where(local>=0,indices[np.maximum(local,0)],-1)
            np.testing.assert_array_equal(mapped,full[indices])

    def test_dimension_change_stops_before_package(self):
        g,_,fit,oof,y=data.project(rows(),self.recipe,self.M)
        bad=copy.deepcopy(self.recipe); bad['base']['inputs']['feature_dimension']=1
        with self.assertRaisesRegex(ValueError,'FIT-only encoder dimension'):
            prepare.arrays(g,fit,oof,y,bad,self.M)


if __name__=='__main__': unittest.main()
