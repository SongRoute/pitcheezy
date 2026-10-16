"""Generated-only parity contract tests; no accepted artifacts or data fixtures."""
import importlib.util
import hashlib
import json
import os
import signal
from unittest.mock import patch
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from experiments.ws_seq import train_seq as base
from experiments.ws_campaign import train as campaign
from experiments.ws_campaign.models import build_model

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('parity_worker', ROOT/'scripts/fullseason_shadow_parity.py')
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def synthetic_sequence(n=180):
    i=np.arange(n)
    return {'game_pk': (100+i//60).astype(np.int32), 'pitcher': (200+(i%60)//45).astype(np.int32),
        'at_bat_number': (1+(i%60)//5).astype(np.int16), 'pitch_number': (1+i%5).astype(np.int16),
        'batter': (300+i//5).astype(np.int32), 'pitch_type': (i%19).astype(np.int16),
        'outcome10': (i%10).astype(np.int16), 'balls': (i%4).astype(np.int16), 'strikes': (i%3).astype(np.int16)}


class ParityContracts(unittest.TestCase):
    def test_fixed_endpoints_and_bounds(self):
        rows=p.fixed_indices(720505)
        self.assertEqual(len(rows),100); self.assertEqual(rows[0],0); self.assertEqual(rows[-1],720504)
        self.assertEqual(len(np.unique(rows)),100)
        for n in (0,720506,True):
            with self.assertRaises(ValueError): p.fixed_indices(n)

    def test_context_matches_original_full_index_and_remap(self):
        s=synthetic_sequence(); q=p.fixed_indices(len(s['pitcher']))
        full=base.build_context_index(*(s[n] for n in ('pitcher','game_pk','at_bat_number','pitch_number')),32)
        ctx=p.query_context(s,q); np.testing.assert_array_equal(ctx,full[q])
        x=np.zeros((len(q),199),np.float32)
        lx,c,ls,lc,lk,lq=p.reference_subset(x,s,q,ctx)
        valid=lc[lq]>=0
        original_keys=np.column_stack([s[n] for n in ('game_pk','at_bat_number','pitch_number')])
        np.testing.assert_array_equal(lk[lc[lq][valid]],original_keys[ctx[ctx>=0]])
        self.assertLessEqual(len(lx),3300)

    def test_future_current_outcome_not_used_in_query_tokens(self):
        s=synthetic_sequence(); q=np.array([35]); ctx=p.query_context(s,q)
        before=p.make_tokens(s,q,ctx)
        s['outcome10'][35:]=(s['outcome10'][35:]+1)%10
        after=p.make_tokens(s,q,ctx)
        np.testing.assert_array_equal(before['out'],after['out'])

    def test_generated_same_cpu_reference_tolerance(self):
        import json
        torch.set_num_threads(1)
        recipe=json.loads((ROOT/'configs/EXP-WS-FULLSEASON-T-FINAL.json').read_text())
        s=synthetic_sequence(); q=p.fixed_indices(180); ctx=p.query_context(s,q)
        x=np.random.default_rng(4).normal(size=(len(q),199)).astype(np.float32)
        tokens=p.make_tokens(s,q,ctx)
        tx={k:torch.as_tensor(tokens[k]) for k in ('valid','type','out','balls','strikes','same','pos')}
        lx,c,ls,lc,lk,lq=p.reference_subset(x,s,q,ctx)
        data=base.Data(torch,torch.device('cpu'),lx,c,ls,lc)
        keys=np.column_stack([s[n][q] for n in ('game_pk','at_bat_number','pitch_number')])
        for seed in (0,1,2):
            model=build_model(recipe,'T',seed).eval()
            with torch.no_grad(): z={k:v.double().numpy() for k,v in model(torch.as_tensor(x),tx,torch.as_tensor(keys)).items()}
            actual=campaign.probabilities(z)
            reference=campaign.probabilities(campaign.predict(model,data,torch.as_tensor(lk),torch.as_tensor(lq),len(q)))
            self.assertLessEqual(p.compare(actual,reference),p.ATOL)

    def test_feature_header_c_f_and_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'x.npy'
            for arr in (np.zeros((8,199),np.float32),np.asfortranarray(np.ones((8,199),np.float32))):
                np.save(f,arr)
                with f.open('rb') as stream:
                    view=p.feature_view(stream,8,f.stat().st_size);np.testing.assert_array_equal(view[[0,7]],arr[[0,7]]);del view
            np.save(f,np.zeros((8,198),np.float32))
            with f.open('rb') as stream:
                with self.assertRaises(ValueError): p.feature_view(stream,8,f.stat().st_size)

    def test_reference_uses_verified_held_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'state.pt'; torch.save({'x':torch.ones(2)},f)
            raw=f.read_bytes(); binding={'path':str(f),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            original=torch.load
            def mutate_then_load(stream,**kw):
                f.write_bytes(b'changed after verified read')
                return original(stream,**kw)
            with patch('torch.load',side_effect=mutate_then_load):
                result=p.reference_state(binding)
            self.assertTrue(torch.equal(result['x'],torch.ones(2)))

    def test_same_inode_mutation_is_detected_before_receipt(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            f=Path(d)/'data';f.write_bytes(b'abc')
            binding={'path':str(f),'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
            stream=p.pinned(binding,stack,10);f.write_bytes(b'xyz')
            with self.assertRaises(ValueError): p.unchanged(stream,binding)

    def test_generated_worker_complete_without_corpus(self):
        helper_spec=importlib.util.spec_from_file_location('generated_shadow_helpers',ROOT/'tests/test_seq_shadow_fullseason.py')
        helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); config,_=helper.generated_config(root)
            recipe=json.loads(Path(config['recipe']['path']).read_text())
            cfg=root/'config.json';cfg.write_text(json.dumps(config))
            seq=synthetic_sequence();seq['stand']=np.zeros(180,np.int16)
            sf=root/'seq.npz';np.savez(sf,**seq)
            xf=root/'x.npy';np.save(xf,np.random.default_rng(3).normal(size=(180,199)).astype(np.float32))
            keys=np.column_stack([seq[n] for n in ('game_pk','at_bat_number','pitch_number')])
            reg={'expected_rows':180,'query_rule':'floor(k*(N-1)/99), k=0..99',
                 'tolerance':{'atol':p.ATOL,'rtol':p.RTOL},'adapter_config':str(cfg),
                 'source_pins':{str(cfg):helper.binding(cfg)['sha256']},'recipe':config['recipe']['path'],
                 'inputs':{'features':helper.binding(xf),'sequence':helper.binding(sf)},
                 'expected_keys_sha256':p.key_digest(keys),'bounds':{'wall_seconds':300},
                 'output':str(root),'source_commit':'generated'}
            rp=root/'reg.json';rp.write_text(json.dumps(reg));gp=root/'grant.json';gp.write_text('{}')
            with (root/'lock').open('w') as lock:
                try:
                    with patch.object(p,'ROW_CAP',180),patch.object(p,'gates',return_value=(reg,recipe)), \
                         patch.dict(os.environ,{'PITCHEEZY_FULLSEASON_LOCK_FD':str(lock.fileno())}), \
                         patch('torch.set_num_interop_threads'):
                        p.run(rp,gp)
                finally: signal.alarm(0)
            receipt=json.loads((root/'receipt.pending.json').read_text())
            self.assertEqual(receipt['queries'],100);self.assertFalse(receipt['performance_metrics'])
            self.assertEqual(receipt['fits'],0);self.assertEqual(receipt['calibration_fits'],0)
            self.assertLessEqual(max(receipt['maximum_absolute_differences'].values()),p.ATOL)
            self.assertFalse((root/'corpus-subset.npz').exists())

    def test_tolerance_is_fail_closed(self):
        with self.assertRaises(ValueError): p.compare(np.zeros((1,7)),np.full((1,7),1e-3))
        with self.assertRaises(ValueError): p.compare(np.zeros((1,7)),np.full((1,7),np.nan))


if __name__=='__main__': unittest.main()
