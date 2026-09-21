"""Controlled MLP batter-representation and history-resolution sweep."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import torch
from pitchmdp.data import KEY, hash_file
from pitchmdp.archetypes import add_batter_style_history
from pitchmdp.model import outcome_labels
from pitchmdp.sequence_data import prepare_frame, HistoryStore
from pitchmdp.sequence_model import SequenceModel, classification_metrics
from run_sequence_pilot import dump, arrays
from run_sequence_ablations import rows_hash
from run_temporal_blend import assign_fold, select_cohort, samples_for, check_sources
from run_sequence_calibration import fit_blend
from representation_adapters import BatterContext, WindowStore, IDModel

SEEDS = [42,43,44,45,46]
VARIANTS = [dict(name='hand', mode='hand', history=5), dict(name='id', mode='id', history=5),
            dict(name='continuous', mode='continuous', history=5)]
VARIANTS += [dict(name=f'clusters_{k}', mode='clusters', clusters=k, history=5) for k in (3,5,10,20)]
VARIANTS += [dict(name=f'history_{h}', mode='reference', history=h) for h in (1,2,3,4)]


def verify_manifest(root, name='artifact_hashes.json'):
    for rel, digest in json.loads((root/name).read_text()).items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root.resolve()) or hash_file(path) != digest:
            raise ValueError('Artifact identity mismatch: '+str(path))


def family_bootstrap(y, probabilities, reference, games, replicates=2000):
    names = list(probabilities)
    unique, group = np.unique(games, return_inverse=True)
    counts = np.bincount(group)
    rng = np.random.default_rng(42)
    draws = rng.integers(0,len(unique),size=(replicates,len(unique)))
    denominator = counts[draws].sum(1)
    result = {}
    def scores(p, metric):
        p = np.asarray(p, float)
        if metric == 'log_loss':
            return -np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
        return ((p-np.eye(10)[y])**2).sum(1)
    for metric in ('log_loss','brier_multiclass'):
        delta = np.stack([scores(probabilities[name],metric)-scores(reference,metric) for name in names])
        sums = np.stack([np.bincount(group,weights=d,minlength=len(unique)) for d in delta])
        estimates = sums[:,draws].sum(2)/denominator[None,:]
        means = delta.mean(1)
        radius = float(np.quantile(np.max(np.abs(estimates-means[:,None]),axis=0),.95))
        result[metric] = {name: {'variant_minus_reference': float(means[i]),
            'pointwise95': np.quantile(estimates[i],[.025,.975]).tolist(),
            'simultaneous95': [float(means[i]-radius),float(means[i]+radius)]} for i,name in enumerate(names)}
    return {'contrasts': names, 'games':len(unique), 'replicates':replicates,
            'method':'shared whole-game bootstrap; simultaneous max absolute centered deviation within family',
            'scope':'fixed fitted ensemble/temperature/CAL weight; exploratory; training and calibration estimation uncertainty excluded',
            'metrics':result}


def fit_temperature_logits(logits, y):
    def loss(t):
        p=softmax(logits/t,axis=-1).mean(1)
        return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
    fit=minimize_scalar(loss,bounds=(.5,2.5),method='bounded')
    return float(fit.x),float(fit.fun)


def run_fold(raw, year, base, output):
    started=time.perf_counter()
    dest=output/str(year);dest.mkdir(exist_ok=True)
    frame=assign_fold(raw,year)
    ids,cohort=select_cohort(frame)
    full,parts=samples_for(frame,ids)
    original=json.loads((base/str(year)/'samples.json').read_text())
    for name,part in parts.items():
        if rows_hash(part)!=original[name]['rows_sha256']:
            raise ValueError('Reference sample mismatch: '+name)
    if rows_hash(full)!=original['full_train']['rows_sha256']:
        raise ValueError('Full TRAIN mismatch')
    with (base/str(year)/'fitted_preprocessors.pkl').open('rb') as stream:
        preprocess=pickle.load(stream)
    store=HistoryStore.from_frame(frame,normalizer=preprocess['normalizer'])
    delivery=preprocess['delivery']
    if delivery.draws!=400:
        raise ValueError('Expected frozen shared400delivery')
    for name,part in parts.items():
        part[KEY+['game_date','pitcher','batter']].to_parquet(dest/(name+'_keys.parquet'),index=False)
    with np.load(base/str(year)/'predictions.npz') as saved:
        devref={k:saved[k].copy() for k in saved.files}
    with np.load(base/str(year)/'calibration_predictions.npz') as saved:
        calref={k:saved[k].copy() for k in saved.files}
    ys={name:outcome_labels(part) for name,part in parts.items()}
    for name,archive in [('dev',devref),('blend',calref)]:
        np.testing.assert_array_equal(archive['pitch_keys'],parts[name][KEY].to_numpy())
        np.testing.assert_array_equal(archive['y'],ys[name])
    bp={'blend':calref['baseline'],'dev':devref['frequency']}
    seen=set(parts['train'].batter.astype(int))
    unknown={name:~part.batter.isin(seen).to_numpy() for name,part in parts.items()}
    history_available=(store.indices[parts['dev'].index.to_numpy()]>=0).sum(1)
    train_exposure=parts['train'].groupby('batter').size()
    exposure=parts['dev'].batter.map(train_exposure).fillna(0).to_numpy(int)
    subgroups={'known_id':~unknown['dev'],'unseen_id':unknown['dev'],'history_ge5':history_available>=5,
               'train_exposure_1_99':(exposure>0)&(exposure<100),'train_exposure_ge100':exposure>=100}
    dump(dest/'data.json',{'samples':original,'cohort':cohort,
        'unseen':{name:{'n':int(v.sum()),'total':len(v)} for name,v in unknown.items()},
        'history_available_counts':{str(h):int((history_available==h).sum()) for h in range(6)},
        'id_vocabulary_training':'same neural TRAIN sample, not future exposure',
        'subgroup_sizes':{name:int(mask.sum()) for name,mask in subgroups.items()}})
    all_dev={'reference':devref['flatten_mlp_blend'],'reference_ensemble':devref['flatten_mlp_ensemble'],'frequency':bp['dev']}
    results={}
    for cfg in VARIANTS:
        name=cfg['name'];variant=dest/name;variant.mkdir(exist_ok=True)
        context=BatterContext(preprocess['context'],cfg['mode'],clusters=cfg.get('clusters')).fit(full,id_train=parts['train'])
        view=WindowStore(store,cfg['history'])
        with (variant/'context.pkl').open('wb') as stream:
            pickle.dump(context,stream)
        dump(variant/'context.json',context.report())
        training=arrays(view,context,parts['train'].index.to_numpy())
        early=arrays(view,context,parts['earlystop'].index.to_numpy())
        member=[]
        for seed in SEEDS:
            modelpath=variant/f'seed{seed}.pt'
            archive=variant/f'seed{seed}_predictions.npz'
            identityfile=variant/f'seed{seed}_hashes.json'
            identity=json.loads(identityfile.read_text()) if identityfile.exists() else {}
            if archive.exists():
                for key,path in [('model',modelpath),('predictions',archive),('temperature_logits',variant/f'seed{seed}_temperature_logits.npy')]:
                    if identity.get(key)!=hash_file(path):
                        raise ValueError('Completed member identity mismatch')
                with np.load(archive) as saved:
                    for part in ('blend','dev'):
                        np.testing.assert_array_equal(saved[part+'_keys'],parts[part][KEY].to_numpy())
                    member.append({part:saved[part].copy() for part in ('blend','dev')})
                continue
            print('MODEL_START',year,name,seed,flush=True)
            if modelpath.exists():
                if identity.get('model')!=hash_file(modelpath) or identity.get('temperature_logits')!=hash_file(variant/f'seed{seed}_temperature_logits.npy'):
                    raise ValueError('Model resume hash mismatch')
                model=(IDModel if cfg['mode']=='id' else SequenceModel).load(modelpath)
            else:
                model=IDModel(context.vocab_size,seed=seed,width=128) if cfg['mode']=='id' else SequenceModel('flatten_mlp',seed=seed,width=128)
                model.fit(training,ys['train'],early,ys['earlystop'],epochs=30,patience=5,batch_size=1024,
                          learning_rate=.0005,checkpoint=variant/f'seed{seed}_checkpoint.pt')
                logits,levels=delivery.logits(model,view,context,parts['temperature'].index.to_numpy())
                model.delivery_temperature,score=fit_temperature_logits(logits,ys['temperature'])
                model.report.update(delivery_temperature=model.delivery_temperature,
                                    calibration_integrated_log_loss=score,delivery_calibration_rows=len(ys['temperature']))
                np.save(variant/f'seed{seed}_temperature_logits.npy',logits)
                np.save(variant/f'seed{seed}_temperature_levels.npy',levels)
                del logits
                model.save(modelpath)
                dump(variant/f'seed{seed}_fit.json',model.report)
                dump(identityfile,{'model':hash_file(modelpath),
                     'temperature_logits':hash_file(variant/f'seed{seed}_temperature_logits.npy')})
            prediction={part:delivery.predict(model,view,context,parts[part].index.to_numpy()) for part in ('blend','dev')}
            np.savez_compressed(archive,**prediction,**{part+'_keys':parts[part][KEY].to_numpy() for part in ('blend','dev')})
            dump(identityfile,{'model':hash_file(modelpath),'predictions':hash_file(archive),
                              'temperature_logits':hash_file(variant/f'seed{seed}_temperature_logits.npy')})
            member.append(prediction)
            print('MODEL_DONE',year,name,seed,model.report['seconds'],flush=True)
            del model;gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
        stacked={part:np.stack([m[part] for m in member]) for part in ('blend','dev')}
        ensemble={part:p.mean(0) for part,p in stacked.items()}
        selection=fit_blend(ys['blend'],ensemble['blend'],bp['blend'],'log_loss')
        weight=selection['model_weight']
        p=weight*ensemble['dev']+(1-weight)*bp['dev']
        all_dev[name]=p;all_dev[name+'_ensemble']=ensemble['dev']
        result={'config':cfg,'context':context.report(),'selection':selection,
                'metrics':classification_metrics(ys['dev'],p),'ensemble_metrics':classification_metrics(ys['dev'],ensemble['dev']),
                'seed_metrics':[classification_metrics(ys['dev'],q) for q in stacked['dev']],
                'subgroups':{key:classification_metrics(ys['dev'][mask],p[mask]) for key,mask in subgroups.items()}}
        results[name]=result
        np.savez_compressed(variant/'ensemble_predictions.npz',blend=ensemble['blend'],dev=ensemble['dev'],mixed_dev=p)
        dump(variant/'results.json',result)
        print('VARIANT_DONE',year,name,result['metrics']['log_loss'],flush=True)
        del training,early,context,view,member,stacked,ensemble;gc.collect()
    families={family:[cfg['name'] for cfg in VARIANTS if (cfg['mode']=='reference')==(family=='history')]
              for family in ('batter','history')}
    comparisons={family:family_bootstrap(ys['dev'],{name:all_dev[name] for name in names},all_dev['reference'],parts['dev'].game_pk.to_numpy())
                 for family,names in families.items()}
    np.savez_compressed(dest/'predictions.npz',y=ys['dev'],game_pk=parts['dev'].game_pk.to_numpy(),
        pitch_keys=parts['dev'][KEY].to_numpy(),batter=parts['dev'].batter.to_numpy(dtype=np.int64),unknown_id=unknown['dev'],
        history_available=history_available,train_exposure=exposure,**all_dev)
    final={'year':year,'variants':results,'comparisons':comparisons,
           'reference_metrics':{key:classification_metrics(ys['dev'],all_dev[key]) for key in ('reference','reference_ensemble','frequency')},
           'reference_subgroups':{key:classification_metrics(ys['dev'][mask],all_dev['reference'][mask]) for key,mask in subgroups.items()},
           'seconds':time.perf_counter()-started}
    check_sources(output);dump(dest/'results.json',final)
    print('FOLD_DONE',year,flush=True)
    return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    local=json.loads((PROJECT/'configs/local.json').read_text())
    root,base,output=Path(local['artifact_root']).resolve(),args.base.resolve(),args.output.resolve()
    if not Path('/Volumes/T7 Shield').is_mount() or not all(p.is_relative_to(root) and p!=root for p in (base,output)):
        raise ValueError('Mounted configured SSD and child run directories required')
    if output==base or base.is_relative_to(output) or output.is_relative_to(base):
        raise ValueError('Use a distinct sibling experiment directory')
    if Path(sys.executable).absolute()!=Path(local['python']).absolute():raise ValueError('Configured Python required')
    verify_manifest(base);check_sources(base)
    if output.exists() and any(output.iterdir()):
        config=json.loads((output/'config.json').read_text())
        if config.get('experiment')!='representation_history' or config.get('base')!=str(base) or config.get('variants')!=VARIANTS:
            raise ValueError('Existing output is not this exact sweep')
        if config['base_artifact_manifest_sha256']!=hash_file(base/'artifact_hashes.json'):
            raise ValueError('Reference artifact manifest changed')
    else:
        output.mkdir(parents=True,exist_ok=True)
        protocol=PROJECT/'docs/REPRESENTATION_HISTORY_PROTOCOL.md'
        rels=set(json.loads((base/'source_hashes.json').read_text()))
        rels.update(['scripts/run_representation_history.py','scripts/representation_adapters.py',str(protocol.relative_to(PROJECT))])
        hashes={rel:hash_file(PROJECT/rel) for rel in sorted(rels)}
        for rel in hashes:
            target=output/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/rel,target)
        dump(output/'source_hashes.json',hashes)
        dump(output/'config.json',{'experiment':'representation_history','created_utc':datetime.now(timezone.utc).isoformat(),
            'base':str(base),'base_artifact_manifest_sha256':hash_file(base/'artifact_hashes.json'),
            'variants':VARIANTS,'seeds':SEEDS,'years':[2024,2025],'protocol_sha256':hash_file(protocol)})
    check_sources(output)
    print('RUN',str(output),flush=True)
    raw=add_batter_style_history(prepare_frame(local))
    if raw.attrs.get('sequence_data_identity')!=json.loads((base/'data_identity.json').read_text()):
        raise ValueError('Raw/processed identity differs from reference')
    dump(output/'data_identity.json',raw.attrs['sequence_data_identity'])
    results={str(year):run_fold(raw,year,base,output) for year in (2024,2025)}
    verify_manifest(base);check_sources(base);check_sources(output)
    dump(output/'results.json',{'folds':results,'finished_utc':datetime.now(timezone.utc).isoformat()})
    dump(output/'artifact_hashes.json',{str(p.relative_to(output)):hash_file(p) for p in output.rglob('*')
        if p.is_file() and 'source' not in p.relative_to(output).parts and p.name!='artifact_hashes.json'})

if __name__=='__main__':main()
