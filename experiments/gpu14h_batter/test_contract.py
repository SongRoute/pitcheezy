"""생성 자료만: ID 두 축·초기 공정성·누수·원 control·선정 경계."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import torch
from experiments.gpu14h_batter import run,train
from experiments.gpu14h_prepitch.test_contract import tiny
from experiments.ws_seq import train_seq as base

AMENDED='configs/EXP-GPU14H-008.score-attempt-002.json'


def _bind(path):
    path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':run.sha(path)}


def _seal(root,name,receipt,config_sha,manifest_sha,predictions=None,**fields):
    """worker/supervise가 남기는 receipt·SUPERVISOR 사슬과 같은 모양의 생성 기록."""
    d=Path(root)/name;d.mkdir();outputs={}
    if predictions is not None:shutil.copy(Path(predictions)/'oof-predictions.npz',d/'oof-predictions.npz');outputs['oof-predictions.npz']=run.sha(d/'oof-predictions.npz')
    run.write_json(d/'receipt.json',dict(receipt,status='COMPLETED',source_manifest_sha256=manifest_sha,config_sha256=config_sha,grant_sha256='synthetic-grant',outputs_sha256=dict(receipt.get('outputs_sha256',{}),**outputs)))
    run.write_json(d/'SUPERVISOR.json',{'status':'COMPLETED','returncode':0,'child_reaped':True,'receipt_sha256':run.sha(d/'receipt.json'),'grant_sha256':'synthetic-grant'})
    record=dict(fields,config_sha256=config_sha,receipt=_bind(d/'receipt.json'),supervisor=_bind(d/'SUPERVISOR.json'))
    if predictions is not None:record['predictions']=_bind(d/'oof-predictions.npz')
    return record


def scoring_fixture(root,cfg):
    """실제 population/frequency/prediction 경로를 그대로 타는 생성 패키지·fit 기록·선행 screen.

    003 noID T, 005 pitcher T s0, 008 control s1/2, 008 batter 후보 s0을 생성 자료에서 실제
    trainer로 학습하고 각 fit_worker가 receipt에 더하는 필드를 그대로 붙인다.
    """
    from experiments.ws_fullseason.calibrate_worker import CORPUS
    from experiments.ws_fullseason.data import key_digest
    from experiments.ws_seq.train_seq import SEQ_MEMBERS
    from experiments.gpu14h_identity import run as identity_run
    root=Path(root);r,(x,c,s)=tiny();y=train.parent.targets(s['pitch_type']);fit9=c['fit']&(y>=0)
    e9=np.flatnonzero(c['evaluation']&(y>=0));assert len(e9)>=4
    s['pitcher'][e9[0]]=990001;s['batter'][e9[1]]=990002;s['pitcher'][e9[2]]=990003;s['batter'][e9[2]]=990004
    assert np.isin(s['pitcher'][e9[3]],s['pitcher'][fit9]) and np.isin(s['batter'][e9[3]],s['batter'][fit9])
    pk=root/'package';pk.mkdir()
    np.savez(pk/'corpus-subset.npz',**{k:c[k] for k in sorted(CORPUS)});np.savez(pk/'seq-inputs.npz',**{k:s[k] for k in SEQ_MEMBERS})
    files=[_bind(pk/'corpus-subset.npz'),_bind(pk/'seq-inputs.npz')]
    expected={'rows':len(x),'FIT':int(c['fit'].sum()),'OOF':int(c['evaluation'].sum()),'key_sha256':{'FIT':key_digest(c['keys'][c['fit']]),'OOF':key_digest(c['keys'][c['evaluation']])}}
    meta={'training':{'packages':{f:{'files':files,'expected':expected,'manifest':{'recorded_sha256':'synthetic-input-manifest'},'producer_recipe':{'recorded_sha256':'synthetic-producer'}} for f in run.FOLDS}}}
    original=run.ROOT/'configs/EXP-GPU14H-008.json';fit008=(run.sha(original),json.loads(original.read_text())['source_manifest']['sha256'])
    ref003=(run.identity.REFERENCE['config_sha256'],run.identity.REFERENCE['source_manifest_sha256']);ref005=(run.REFERENCE005['config_sha256'],run.REFERENCE005['source_manifest_sha256'])
    package=(x,c,s);trained={}
    def fitted(key,call):
        if key not in trained:
            d=root/'fits'/'-'.join(map(str,key));d.mkdir(parents=True);trained[key]=(call(d),d)
        return trained[key]
    def augment(got,fold,**fields):return dict(got,fold=fold,producer_recipe_sha256='synthetic-producer',input_manifest_sha256='synthetic-input-manifest',**fields)
    noid,pitchers,results={0:[],1:[],2:[]},{0:[],1:[],2:[]},[]
    for f in run.FOLDS:
        prod=run.producer(cfg,f)
        for seed in (0,1,2):
            got,d=fitted(('003',seed),lambda d,seed=seed:run.parent.train.run(r,'state_t32',seed,copy.deepcopy(package),d,synthetic=True))
            task=f'H14-B3-state_t32-{f}-s{seed}'
            noid[seed].append(_seal(root,task,augment(got,f,task=task,variant='state_t32',model_recipe_sha256=run.object_sha(run.parent.train.recipe(prod,'state_t32'))),*ref003,d,fold=f,seed=seed))
        got,d=fitted(('005',0),lambda d:run.identity.train.run(r,'pitcher_t32',0,copy.deepcopy(package),d,synthetic=True))
        task=identity_run.task_id('id','pitcher_t32',f,0)
        pitchers[0].append(_seal(root,task,augment(got,f,task=task,variant='pitcher_t32',architecture='state_t32',kind='id',model_recipe_sha256=run.object_sha(run.identity.train.recipe(prod,'pitcher_t32'))),*ref005,d,fold=f,seed=0,origin='005'))
        for seed in (1,2):
            got,d=fitted(('control',seed),lambda d,seed=seed:train.run_control(r,seed,copy.deepcopy(package),d,synthetic=True))
            task=run.task_id('control',train.CONTROL,f,seed)
            pitchers[seed].append(_seal(root,task,augment(got,f,task=task,variant=train.CONTROL,architecture='state_t32',kind='control',model_recipe_sha256=run.object_sha(train.identity.recipe(prod,train.CONTROL))),*fit008,d,fold=f,seed=seed,origin='008'))
        for v in train.VARIANTS:
            got,d=fitted(('008',v),lambda d,v=v:train.run(r,v,0,copy.deepcopy(package),d,synthetic=True))
            task=run.task_id('id',v,f,0)
            results.append(_seal(root,task,augment(got,f,task=task,variant=v,architecture='state_t32',kind='id',model_recipe_sha256=run.object_sha(train.recipe(prod,v))),*fit008,d,fold=f,seed=0,variant=v))
    counts=root/'baseline-counts.json';run.write_json(counts,{f:run.parent.fit_frequency(run.parent.population(meta,f)) for f in run.FOLDS})
    screen003=_seal(root,'score-003',{'task':'score_screen','candidates':[],'selected_variants':[],'seeds':[0],'selection_applied':True,'count_fits':3,'outputs_sha256':{'baseline-counts.json':run.sha(counts)}},*ref003)
    screen005=_seal(root,'score-005',{'task':'score_screen','candidates':[],'selected_variants':[],'seeds':[0],'selection_applied':True},*ref005)
    common={'baseline':_bind(counts),'prepitch_screen':screen003,'identity_screen':screen005,'unavailable_tasks':[],'noid_unavailable':[],'pitcher_unavailable':[]}
    grants={'score_screen':dict(common,task='score_screen',results=results,noid_results=noid[0],pitcher_results=pitchers[0]),
            'score_control':dict(common,task='score_control',results=[],noid_results=noid[0]+noid[1]+noid[2],pitcher_results=pitchers[0]+pitchers[1]+pitchers[2])}
    return meta,grants,{'e9':e9,'y':y}


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

    def test_synthetic_score_screen_end_to_end(self):self.score_end_to_end('score_screen',list(train.VARIANTS))

    def test_synthetic_score_control_end_to_end(self):self.score_end_to_end('score_control',[train.CONTROL])

    def test_amended_score_screen_and_control_end_to_end(self):
        for task,variants in (('score_screen',list(train.VARIANTS)),('score_control',[train.CONTROL])):self.score_end_to_end(task,variants,AMENDED)

    def test_score_amendment_changes_only_source_and_deadline(self):
        cfg=json.loads((run.ROOT/AMENDED).read_text());self.assertEqual(run.score_amendment(cfg)['amends'],run.FIT008)
        self.assertEqual(run.FIT008['config_sha256'],run.sha(run.ROOT/run.FIT008['config']))
        self.assertEqual(run.FIT008['source_manifest_sha256'],json.loads((run.ROOT/run.FIT008['config']).read_text())['source_manifest']['sha256'])
        old=json.loads((run.ROOT/run.FIT008['config']).read_text());self.assertIsNone(run.score_amendment(old));self.assertIs(run.fit008_config(old),old)
        fit=run.fit008_config(cfg);self.assertEqual((fit['_config_sha256'],fit['source_manifest']['sha256']),(run.FIT008['config_sha256'],run.FIT008['source_manifest_sha256']))
        for key,value in (('selection',{}),('metrics',{}),('baseline',{}),('jobs',[]),('variants',{}),('deadline_utc','2026-12-01T00:00:00Z'),('limits',{})):
            with self.assertRaisesRegex(ValueError,'only source/deadline'):run.score_amendment(dict(cfg,**{key:value}))
        with self.assertRaisesRegex(ValueError,'frozen fit008'):run.score_amendment(dict(cfg,score_amendment=dict(cfg['score_amendment'],extra=1)))
        with self.assertRaisesRegex(ValueError,'only source/deadline'):run.score_amendment(dict(cfg,source_manifest=old['source_manifest']))

    def score_end_to_end(self,task,variants,config='configs/EXP-GPU14H-008.json'):
        """population→FIT-only P/B known·cold→prediction→comparison9→4집단→receipt JSON까지 생성 자료로 끝까지."""
        cfg=json.loads((run.ROOT/config).read_text());cfg['_config_sha256']=run.sha(run.ROOT/config)
        with tempfile.TemporaryDirectory() as d:
            meta,grants,info=scoring_fixture(d,cfg)
            if True:
                out=Path(d)/task;out.mkdir()
                got=run.score_worker(cfg,grants[task],meta,out)
                self.assertEqual([row['variant'] for row in got['candidates']],variants);self.assertTrue(all(row['status']=='COMPLETE' for row in got['candidates']))
                n=sum(row['metrics'] is not None for row in got['candidates'][0]['classwise'].values());self.assertGreater(n,0)
                for row in got['candidates']:
                    cells=row['known_cold_four'];self.assertEqual(sorted(cells),['cold_pitcher__cold_batter','cold_pitcher__known_batter','known_pitcher__cold_batter','known_pitcher__known_batter'])
                    self.assertEqual(sum(v['rows'] for v in cells.values()),len(run.FOLDS)*len(info['e9']))
                    self.assertTrue(all(v['rows']>0 and v['metrics'] is not None for v in cells.values()))
                self.assertEqual(got['seeds'],[0] if task=='score_screen' else [0,1,2]);self.assertEqual(got['control_only'],task=='score_control')
                if task=='score_control':self.assertEqual(got['selected_variants'],[])
                else:self.assertEqual(got['selected_variants'],run.select_variants(got['candidates']))
                run.write_json(out/'receipt.json',got);self.assertEqual(json.loads((out/'receipt.json').read_text())['candidates'][0]['variant'],variants[0])

    def test_score_known_batter_uses_training_vocabulary(self):
        """채점 known 타자 = 학습 identity-vocabularies.json의 batter 어휘(FIT_E9)와 정확히 같다."""
        cfg=json.loads((run.ROOT/'configs/EXP-GPU14H-008.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            meta,grants,info=scoring_fixture(d,cfg);p=run.population(meta,'july')
            self.assertIn('batter',p['seq']);self.assertEqual(p['seq']['batter'].dtype.kind,'i')
            rec=next(r for r in grants['score_screen']['results'] if r['variant']=='pitcher_batter')
            vocab=json.loads((Path(rec['receipt']['path']).parent.parent/'fits'/'008-pitcher_batter'/'identity-vocabularies.json').read_text())
            for role in ('pitcher','batter'):
                np.testing.assert_array_equal(train.identity.vocabulary(p['seq'][role],p['fit9']),vocab['factors'][role])
            known=train.identity.indices(p['seq']['batter'][p['oof']],np.asarray(vocab['factors']['batter']))>0
            oof=np.flatnonzero(p['oof']);self.assertFalse(known[np.searchsorted(oof,info['e9'][1])]);self.assertTrue(known[np.searchsorted(oof,info['e9'][3])])

if __name__=='__main__':unittest.main(verbosity=2)
