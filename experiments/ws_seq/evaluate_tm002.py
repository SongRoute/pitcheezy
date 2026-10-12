"""One-shot TM-002 six-fit evaluator; no CLI path without independent grant.

Pure metrics/bootstrap come from frozen evaluate_seq. No temperature refit.
All six completed metadata bindings precede the exclusive EVAL-OPENED marker;
only then may prediction arrays be opened. A failed open requires a reviewed,
recorded retry and a fresh output directory, never overwriting the first marker.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import sys
import time

import numpy as np
from experiments.ws_seq import train_seq as base
from experiments.ws_seq import evaluate_seq as metrics
from experiments.ws_seq import tm002
from experiments.ws_seq import train_tm002 as runner

ROOT = Path(__file__).resolve().parents[2]
RUN_FILES = ('seven-leaf-evaluation.npy','margins.npz','model-state.pt','temperatures.json','curve.json')
ROLE_MAP = {'FIT':'fit','SELECT':'tuning','CAL':'calibration','EVAL':'evaluation'}


def require(ok, message):
    if not ok:raise base.Stop('STOP '+message)


def checked_json(path, pin):
    path=Path(path)
    require(path.is_file() and base.sha(path)==pin,'metadata file hash '+str(path))
    return json.loads(path.read_text())


def metadata_gate(cfg, grant):
    """Read receipt/supervisor/ledger JSON, never any prediction payload."""
    require(grant.get('id')==cfg['id'] and grant.get('scope')=='TM002_EVALUATE'
            and grant.get('execute') is True and grant.get('verdict')=='PASS'
            and grant.get('distinct_from_author') is True
            and isinstance(grant.get('author'),str) and bool(grant['author'].strip())
            and isinstance(grant.get('reviewer'),str) and bool(grant['reviewer'].strip())
            and grant['author'].strip()!=grant['reviewer'].strip()
            and grant.get('sources_sha256')==runner.source_hashes(), 'evaluation grant/review/source')
    exposure,attempts=tm002.check_contract(grant,cfg)
    require(set(grant.get('runs',{}))==set(attempts['slots']),'exactly six bound runs required')
    require(len({v['directory'] for v in grant['runs'].values()})==6,'distinct six run directories')
    found={}; commits=set()
    expected_rows={key:cfg['expected_rows'][role] for role,key in ROLE_MAP.items()}
    for name,slot in attempts['slots'].items():
        location=grant['runs'][name]; directory=Path(location['directory'])
        require(directory.is_absolute(),'absolute run location')
        r=checked_json(directory/'receipt.json',location['receipt_sha256'])
        sup=checked_json(directory/'SUPERVISOR.json',location['supervisor_sha256'])
        arm,seed=name.split('-s');seed=int(seed)
        g=r.get('grant',{})
        require(r.get('id')==cfg['id'] and r.get('status')=='COMPLETED'
                and r.get('arm')==arm and type(r.get('seed')) is int and r['seed']==seed
                and r.get('data_version')==cfg['data_version'] and r.get('data_role')==cfg['data_role']
                and r.get('input_manifest_sha256')==grant['input_manifest_sha256']
                and r.get('source_sha256')==runner.source_hashes()
                and r.get('config_sha256')==runner.source_hashes()['configs/EXP-WS-TM-002.json']
                and r.get('attempt_id')==slot['attempt_id'] and r.get('bundle_id')==attempts['bundle_id']
                and r.get('scoring_root')==attempts['scoring_root']
                and r.get('rows')==expected_rows and r.get('eval_scores_opened') is False
                and r.get('data_exposure_ledger_sha256')==grant['data_exposure_ledger_sha256']
                and set(r.get('outputs_sha256',{}))==set(RUN_FILES)
                and g.get('arm')==arm and g.get('seed')==seed and g.get('attempt_id')==slot['attempt_id']
                and g.get('output')==slot['output'] and g.get('input_manifest_sha256')==grant['input_manifest_sha256']
                and g.get('sources_sha256')==runner.source_hashes()
                and g.get('verdict')=='PASS' and g.get('execute') is True
                and g.get('distinct_from_author') is True
                and bool(g.get('author')) and bool(g.get('reviewer')) and g['author']!=g['reviewer'],
                name+' receipt/source/config/input/attempt/rows/grant')
        require(sup.get('status')=='COMPLETED' and sup.get('returncode')==0 and sup.get('reason') is None
                and sup.get('receipt_sha256')==location['receipt_sha256']
                and sup.get('grant_sha256')==r.get('grant_sha256'),name+' supervisor binding')
        commits.add(r.get('commit'));found[name]={'receipt':r,'directory':directory}
    require(len(commits)==1 and None not in commits,'mixed/missing source commit')
    for seed in cfg['seeds']:
        a,b=(found[f'{arm}-s{seed}']['receipt'].get('shared_initial_sha256') for arm in cfg['arms'])
        require(isinstance(a,str) and len(a)==64 and a==b,'paired shared initialization')
    return exposure,attempts,found


def claim_open(grant, attempts, out):
    """Exclusive per-bundle access marker; retry requires explicit failure evidence."""
    require(not out.exists(),'fresh evaluation output required')
    require(Path(grant['output']).resolve()==out.resolve(),'evaluation output grant')
    root=Path(attempts['scoring_root']);root.mkdir(parents=True,exist_ok=True)
    retry=grant.get('retry')
    if retry is None:
        marker=root/'EVAL-OPENED.json'
    else:
        index=retry.get('index')
        require(type(index) is int and index>=1 and bool(retry.get('reason')),'retry index/reason')
        previous=root/('EVAL-OPENED.json' if index==1 else f'EVAL-OPENED-RETRY-{index-1}.json')
        old=checked_json(previous,retry['previous_marker_sha256'])
        failure=Path(retry['failure_file'])
        require(failure.parent.resolve()==Path(old['output']).resolve(),'retry failure output identity')
        failed=checked_json(failure,retry['failure_sha256'])
        require(failed.get('status')=='INCOMPLETE','retry needs failed attempt evidence')
        previous_receipt=Path(old['output'])/'receipt.json'
        require(not previous_receipt.exists() or json.loads(previous_receipt.read_text()).get('status')!='COMPLETED',
                'completed scoring cannot be retried')
        marker=root/f'EVAL-OPENED-RETRY-{index}.json'
    out.mkdir(parents=True,exist_ok=False)
    temporary=None
    try:
        # Reuse policy_artifacts.write_exclusive's hard-link pattern, but NEVER
        # its exFAT O_EXCL fallback: a partial access marker cannot be recovered.
        fd,temporary=tempfile.mkstemp(dir=root,prefix='.'+marker.name+'.')
        with os.fdopen(fd,'w') as handle:
            json.dump({'status':'OPENED','bundle_id':attempts['bundle_id'],'output':str(out.resolve()),
                       'sources_sha256':runner.source_hashes(),'runs':grant['runs'],
                       'grant_content_sha256':base.hashlib.sha256(json.dumps(grant,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                       'retry':retry,'utc':time.time()},handle,indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic no-overwrite publication only after the complete file is closed.
        os.link(temporary,marker)
    except BaseException:
        # Pre-publication failure: remove only our still-empty output.
        try: out.rmdir()
        except OSError: pass
        raise
    finally:
        if temporary is not None:
            # A cleanup failure must not turn successful publication into failure.
            try: os.unlink(temporary)
            except OSError: pass
    return marker


def load_predictions(cfg, bc, found, C, manifest_sha, input_sha):
    Q={}; checks={}; ev=C['evaluation']; y=C['y7'][ev]
    for name,item in found.items():
        d=item['directory'];r=item['receipt']
        require(r['input_manifest_sha256']==manifest_sha and r['inputs_sha256']==input_sha,
                name+' package identity')
        for filename in RUN_FILES:
            require(base.sha(d/filename)==r['outputs_sha256'][filename],name+' output hash '+filename)
        with np.load(d/'margins.npz',allow_pickle=False) as z:
            want={f'{role}__{h}' for role in base.ROLES for h in base.HEADS}|{f'{role}__keys' for role in base.ROLES}
            require(set(z.files)==want,name+' margin columns')
            M={role:{h:np.array(z[f'{role}__{h}']) for h in base.HEADS} for role in base.ROLES}
            for role in base.ROLES:
                keys=np.array(z[f'{role}__keys'])
                require(keys.dtype.kind in 'iu' and np.array_equal(keys,C['keys'][C[role]]),name+' ordered '+role+' keys')
        for role in base.ROLES:
            n=int(C[role].sum())
            for head in base.HEADS:
                values=M[role][head];shape=(n,10) if head=='parent' else (n,)
                require(values.shape==shape and values.dtype==np.float64 and np.isfinite(values).all(),name+' margin shape/finiteness')
        cal=json.loads((d/'temperatures.json').read_text());temp=cal.get('temperatures',{})
        ncal=int(C['calibration'].sum());yc=C['y10'][C['calibration']]
        require(cal==r.get('calibration') and cal.get('role')=='CAL'
                and cal.get('bounds_T')==list(base.T_BOUNDS)
                and set(temp)==set(base.HEADS)
                and all(type(v) in (int,float) and np.isfinite(v) and base.T_BOUNDS[0]<v<base.T_BOUNDS[1] for v in temp.values())
                and cal.get('fit_rows')=={'parent':ncal,'narrow_whiff':int((yc==1).sum()),'ground_ball_per_pitch':int(np.isin(yc,base.BIP).sum())}
                and cal.get('calibration_keys_sha256')==base.hashlib.sha256(C['keys'][C['calibration']].astype('<i8').tobytes()).hexdigest(),
                name+' CAL source/rows/keys/bounds/receipt')
        q=np.load(d/'seven-leaf-evaluation.npy',allow_pickle=False)
        require(q.dtype==np.float64 and q.shape==(len(y),7) and np.isfinite(q).all()
                and (q>=0).all() and np.max(np.abs(q.sum(1)-1))<=1e-9,name+' probability simplex')
        require(r.get('evaluation_keys_sha256')==base.hashlib.sha256(C['keys'][ev].astype('<i8').tobytes()).hexdigest(),name+' EVAL ordered key fingerprint')
        recomputed=base.seven_from_margins(*(M['evaluation'][h] for h in base.HEADS),T=temp)
        gap=float(np.max(np.abs(q-recomputed)))
        require(gap<=1e-12,name+' calibrated probabilities do not reproduce')
        select=base.nll7(base.seven_from_margins(*(M['tuning'][h] for h in base.HEADS)),C['y7'][C['tuning']],bc['nll_clip'])
        require(abs(select-r['best_select_nll7'])<=1e-9,name+' SELECT does not reproduce')
        curve=json.loads((d/'curve.json').read_text())
        require(isinstance(curve,list) and bool(curve)
                and [v['epoch'] for v in curve]==list(range(1,len(curve)+1))
                and all(np.isfinite(v['select_nll7']) for v in curve),name+' finite epoch curve')
        earliest=min(curve,key=lambda v:v['select_nll7'])
        require(earliest['epoch']==r['best_epoch'] and abs(earliest['select_nll7']-select)<=1e-9,name+' SELECT earliest argmin')
        Q[name]=q;checks[name]={'leaves_max_abs':gap,'calibration_optimizations_during_evaluation':0}
    for arm in cfg['arms']:
        Q[arm]=np.mean([Q[f'{arm}-s{s}'] for s in cfg['seeds']],axis=0)
    return Q,checks


def score(cfg,bc,Q,C,S):
    ev=C['evaluation'];y=C['y7'][ev];games=C['games'][ev]
    pairs=[('R','T')]+[(f'R-s{s}',f'T-s{s}') for s in cfg['seeds']]
    b=cfg['bootstrap'];bcfg={'seed':b['seed'],'chunk':100}
    arms,paired,unit=metrics.paired_block(Q,y,games,pairs,bcfg,b['draws'])
    for name,q in Q.items():arms[name]['reliability_per_leaf']=metrics.reliability(q,y,bc['labels']['leaf_order'])
    strata={}
    for kind,ids,bins in (
        ('pitcher',S['pitcher'],[('S0',0,0),('S1',1,199),('S2',200,999),('S3',1000,None)]),
        ('batter',S['batter'],[('B0',0,0),('B1',1,None)])):
        counts=metrics.fit_rows_of(ids[C['fit']],ids[ev]);labels=metrics.assign_bins(counts,bins);strata[kind]={}
        for name,_,_ in bins:
            use=labels==name;n=int(use.sum())
            if not n:
                strata[kind][name]={'rows':0,'games':0,'players':0,'metrics':None,'note':'unmeasured empty cell'};continue
            vals,pair,units=metrics.paired_block({a:Q[a][use] for a in cfg['arms']},y[use],games[use],[('R','T')],bcfg,b['draws'])
            strata[kind][name]={'rows':n,'games':units['games'],'players':int(len(np.unique(ids[ev][use]))),'arms':vals,'paired':pair}
    p=paired['R-T'];lo,hi=p['nll']['ci95'];bhi=p['brier']['ci95'][1]
    label=('FOLLOW_UP_WORTHWHILE' if hi<0 and bhi<=0.0005 else 'EVIDENCE_AGAINST' if lo>0 else 'INCONCLUSIVE')
    return {'id':cfg['id'],'data_version':cfg['data_version'],'data_role':cfg['data_role'],
            'claim_ceiling':'2026 EXPOSED DEVELOPMENT, not independent TEST; no winner/adoption/policy/latency claim',
            'seeds':cfg['seeds'],'ensemble':'arithmetic mean of three calibrated seven-leaf probabilities',
            'primary':{'estimand':'R ensemble minus T ensemble pitch-weighted EVAL seven-leaf NLL',**p['nll']},
            'paired':paired,'arms':arms,'strata':strata,'bootstrap':unit,
            'denominators':{'rows':int(ev.sum()),'games':int(len(np.unique(games))),
                            'pitchers':int(len(np.unique(S['pitcher'][ev]))),'batters':int(len(np.unique(S['batter'][ev]))),
                            'leaf_counts':np.bincount(y,minlength=7).tolist(),
                            'pitch_type_counts':{str(v):int((S['pitch_type'][ev]==v).sum()) for v in np.unique(S['pitch_type'][ev])},
                            'unknown_pitch_type_rows':int((S['pitch_type'][ev]==0).sum()),
                            'feature_missing_fallback_counts':'UNKNOWN: encoded input package does not carry these raw masks'},
            'advisory':{'label':label,'brier_margin':0.0005,'nll_noninferiority_margin':0.0010,
                        'nll_noninferiority':hi<=0.0010,'brier_safety_concern':bhi>0.0005,'ece_concern':p['ece_top_label_10bin']['ci95'][0]>0},
            'calibration_optimizations_during_evaluation':0,'adoption':False,'automatic_followup':False}


def evaluate(cfg,bc,grant,inputs,out,*,synthetic=False):
    exposure,attempts,found=metadata_gate(cfg,grant)
    require(Path(grant['inputs']).resolve()==inputs.resolve(),'input path grant')
    require(base.sha(inputs/'manifest.json')==grant['input_manifest_sha256'],'input manifest grant hash')
    marker=claim_open(grant,attempts,out)
    try:
        man,manifest_sha,input_sha,_=base.verify_package(inputs,bc,ROOT/cfg['base_config'])
        if not synthetic:tm002.check_manifest_contract(man,exposure,cfg)
        else:require(man.get('synthetic') is True,'synthetic package required')
        x,C,S=base.load_package(inputs,bc);del x
        tm002.check_role_partition(C)
        require({k:int(C[k].sum()) for k in ROLE_MAP.values()}=={k:cfg['expected_rows'][r] for r,k in ROLE_MAP.items()},'corpus role counts')
        require(len(np.unique(C['keys'],axis=0))==len(C['keys']),'unique corpus keys')
        require(len(np.unique(C['games'][C['evaluation']]))==cfg['expected_eval_games'],'registered EVAL game count')
        Q,checks=load_predictions(cfg,bc,found,C,manifest_sha,input_sha)
        result=score(cfg,bc,Q,C,S)
        result.update(synthetic=synthetic,checks=checks,bindings={'sources_sha256':runner.source_hashes(),
                    'input_manifest_sha256':manifest_sha,'runs':grant['runs'],'marker_sha256':base.sha(marker),
                    'attempt_ledger_sha256':grant['attempt_ledger_sha256']})
        for arm in cfg['arms']:np.save(out/f'{arm}-ensemble.npy',Q[arm],allow_pickle=False)
        base.save_json(out/'result.json',result)
        base.save_json(out/'receipt.json',{'status':'COMPLETED','id':cfg['id'],'bundle_id':attempts['bundle_id'],
                                         'eval_scores_opened':True,'result_sha256':base.sha(out/'result.json'),
                                         'marker_sha256':base.sha(marker),'synthetic':synthetic})
        return result
    except Exception as exc:
        base.save_json(out/'STOP.json',{'status':'INCOMPLETE','reason':repr(exc),'marker_sha256':base.sha(marker)})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grant',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=p.parse_args()
    cfg,bc=tm002.load_config(ROOT/'configs/EXP-WS-TM-002.json')
    grant=json.loads(args.grant.read_text())
    try:
        if args.worker:evaluate(cfg,bc,grant,args.inputs,args.out)
        else:
            metadata_gate(cfg,grant)
            require(not args.out.exists(),'fresh evaluation output required')
            return runner.supervise_command([sys.executable,'-m','experiments.ws_seq.evaluate_tm002',
                                             *sys.argv[1:],'--worker'],args.out,
                                            cfg['budgets']['evaluation_seconds'],cfg['budgets']['host_rss_bytes_per_fit'],
                                            base.sha(args.grant))
    except (base.Stop,base.HashMismatch,ValueError,KeyError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 2
    return 0


if __name__=='__main__':sys.exit(main())
