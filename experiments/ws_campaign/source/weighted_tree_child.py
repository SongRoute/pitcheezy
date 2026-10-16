"""Job-owned weighted LightGBM child for the A/B/C pilot (runs only under the pinned lightgbm-env python).

Two modes, each a line-for-line mirror of its pinned source plus exactly one semantic change,
a FIT-row sample weight on the TRAIN Dataset. Tuning (SELECT early stop) stays unweighted.
- binary: event_tree_child.run (stage-d08c276, sha a479fd39...): int8 0/1 labels, roles disjoint
  (not exhaustive: TRAIN/tuning restricted to the parent class), 200 rounds, first argmin.
- multiclass: multiclass_tree_child.fit_fixed (stage-14f23dd, sha d62e0163...):
  199 columns, four exhaustive disjoint roles, all ten TRAIN classes, 10 trees/iteration census,
  selected-TUNE-loss replay, saved-model replay on tuning/CAL/EVAL, margins.npz with ycal/yeval.
Runtime check from multiclass_tree_child.run_request: python 3.12, lightgbm 4.7.0, loaded native == pinned.
--synthetic-parity fits both pinned originals and this child with w=1 on synthetic rows and requires
identical margins; then a non-uniform weight must reach the native Dataset and change the margins.
"""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np

BINARY={'objective':'binary','metric':'binary_logloss','learning_rate':.05,'num_leaves':15,'max_bin':63,'min_data_in_leaf':100,'lambda_l2':1.0,'feature_fraction':1.0,'bagging_fraction':1.0,'seed':20261043,'deterministic':True,'force_col_wise':True,'num_threads':2,'verbosity':-1}
MULTICLASS={'objective':'multiclass','num_class':10,'metric':'multi_logloss','learning_rate':.05,'num_leaves':15,'max_bin':63,'min_data_in_leaf':100,'lambda_l2':1.,'feature_fraction':1.,'bagging_fraction':1.,'seed':20261043,'deterministic':True,'force_col_wise':True,'num_threads':2,'verbosity':-1}
ROLES=('train','tuning','calibration','evaluation')
KEYS={'out','x_path','x_sha256','roles_path','roles_sha256','weights_path','weights_sha256','source_sha256','native_sha256','native_path','max_rounds','params'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def check_weights(w,train):
 # Raw weights already capped by the runner; here: finite, positive on TRAIN, zero elsewhere, TRAIN mean 1.
 if w.shape!=train.shape or w.dtype!=np.float64 or not np.isfinite(w).all():raise ValueError('Weight schema')
 if np.any(w[~train]!=0) or np.any(w[train]<=0) or abs(w[train].mean()-1)>1e-9:raise ValueError('Weight support/normalization')
 return w[train]

def runtime(native_path,native_sha):
 import lightgbm as lgb
 if sys.version_info[:2]!=(3,12) or lgb.__version__!='4.7.0' or Path(lgb.basic._LIB._name).resolve()!=Path(native_path).resolve() or sha(native_path)!=native_sha:raise ValueError('Loaded native runtime mismatch')
 return lgb

def fit_binary(lgb,x,y,masks,w,out):
 train,tuning,cal,ev=(masks[k] for k in ROLES)
 if x.ndim!=2 or x.shape[1]!=199 or y.shape!=(len(x),) or y.dtype!=np.int8 or not np.isin(y,(0,1)).all() or any(m.shape!=(len(x),) or m.dtype!=bool for m in (train,tuning,cal,ev)) or np.any(train.astype(int)+tuning.astype(int)+cal.astype(int)+ev.astype(int)>1):raise ValueError('Tree child role/label schema')
 out.mkdir();start=time.monotonic()
 train_set=lgb.Dataset(x[train],label=y[train],weight=check_weights(w,train),free_raw_data=True)
 tune_set=lgb.Dataset(x[tuning],label=y[tuning],reference=train_set,free_raw_data=True)
 curve={};booster=lgb.train(BINARY,train_set,num_boost_round=200,valid_sets=[tune_set],valid_names=['tuning'],callbacks=[lgb.record_evaluation(curve)])
 values=curve['tuning']['binary_logloss']
 if len(values)!=200 or not np.isfinite(values).all():raise ValueError('Incomplete tree curve')
 selected=int(np.argmin(values))+1  # first minimum wins ties
 preds={name:np.asarray(booster.predict(x[m],num_iteration=selected,raw_score=True),dtype=np.float64) for name,m in [('tuning',tuning),('calibration',cal),('evaluation',ev)]}
 if any(not np.isfinite(v).all() for v in preds.values()):raise ValueError('Nonfinite tree margin')
 model=out/'model.txt';booster.save_model(str(model),num_iteration=selected);loaded=lgb.Booster(model_file=str(model))
 if not np.allclose(preds['evaluation'],loaded.predict(x[ev],raw_score=True),rtol=0,atol=1e-10):raise ValueError('Tree save/reload mismatch')
 np.savez_compressed(out/'margins.npz',**preds);(out/'curve.json').write_text(json.dumps(values)+'\n')
 return {'objective':'binary','selected_iteration':selected,'tuning_logloss':float(values[selected-1]),'num_rounds':len(values),'no_split':bool(booster.feature_importance().sum()==0),'train_rows':int(train.sum()),'train_weight_sum':float(w[train].sum()),'tuning_rows':int(tuning.sum()),'calibration_rows':int(cal.sum()),'evaluation_rows':int(ev.sum()),'lightgbm_version':lgb.__version__,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes_macos':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

def fit_multiclass(lgb,x,y,masks,w,out):
 from scipy.special import softmax,logsumexp
 if x.ndim!=2 or x.shape[1]!=199 or not np.isfinite(x).all() or y.shape!=(len(x),) or y.dtype.kind not in 'iu' or not np.isin(y,np.arange(10)).all():raise ValueError('199-column/ten-class input')
 if set(masks)!=set(ROLES) or any(m.shape!=(len(x),) or m.dtype!=bool or not m.any() for m in masks.values()) or not np.all(sum(m.astype(int) for m in masks.values())==1):raise ValueError('Four exhaustive disjoint roles')
 if not np.all(np.bincount(y[masks['train']],minlength=10)>0):raise ValueError('All ten TRAIN classes required')
 out.mkdir();start=time.monotonic();train=lgb.Dataset(x[masks['train']],label=y[masks['train']],weight=check_weights(w,masks['train']),free_raw_data=True);tune=lgb.Dataset(x[masks['tuning']],label=y[masks['tuning']],reference=train,free_raw_data=True);curve={}
 booster=lgb.train(MULTICLASS,train,num_boost_round=200,valid_sets=[tune],valid_names=['tuning'],callbacks=[lgb.record_evaluation(curve)])
 values=curve['tuning']['multi_logloss']
 if len(values)!=200 or not np.isfinite(values).all() or booster.num_model_per_iteration()!=10 or booster.num_trees()>2000:raise ValueError('Multiclass round/tree census')
 selected=int(np.argmin(values))+1;preds={}
 for role in ROLES[1:]:
  margin=np.asarray(booster.predict(x[masks[role]],num_iteration=selected,raw_score=True),dtype=np.float64)
  if margin.shape!=(int(masks[role].sum()),10) or not np.isfinite(margin).all() or not np.allclose(softmax(margin,axis=1).sum(1),1,rtol=0,atol=1e-12):raise ValueError('n by10 finite margins')
  preds[role]=margin
 # Unweighted tuning loss replay: the tuning Dataset carries no weight in any arm.
 tune_loss=float(np.mean(logsumexp(preds['tuning'],axis=1)-preds['tuning'][np.arange(int(masks['tuning'].sum())),y[masks['tuning']]]))
 if abs(tune_loss-values[selected-1])>1e-10:raise ValueError('Selected TUNE loss replay')
 model=out/'model.txt';booster.save_model(str(model),num_iteration=selected);loaded=lgb.Booster(model_file=str(model))
 if loaded.num_trees()!=selected*10:raise ValueError('Saved selected class-tree census')
 for role in ROLES[1:]:
  if not np.allclose(preds[role],loaded.predict(x[masks[role]],raw_score=True),rtol=0,atol=1e-10):raise ValueError('Saved selected-model replay')
 np.savez(out/'margins.npz',**preds,ycal=y[masks['calibration']],yeval=y[masks['evaluation']]);(out/'curve.json').write_text(json.dumps(values)+'\n')
 return {'objective':'multiclass','rounds':200,'maximum_class_trees':2000,'actual_class_trees':booster.num_trees(),'selected_iteration':selected,'selected_model_class_trees':loaded.num_trees(),'selected_tune_nll':values[selected-1],'rows':len(x),'train_rows':int(masks['train'].sum()),'train_weight_sum':float(w[masks['train']].sum()),'columns':199,'classes':10,'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes_macos':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'lightgbm_version':lgb.__version__,'params':MULTICLASS}

def run(request_path):
 q=json.loads(Path(request_path).read_text());out=Path(q['out'])
 if out.exists():raise ValueError('Fresh child output required')
 if set(q)!=KEYS or q['max_rounds']!=200 or q['params'] not in (BINARY,MULTICLASS) or q['source_sha256']!=sha(__file__):raise ValueError('Weighted child request/source changed')
 lgb=runtime(q['native_path'],q['native_sha256'])
 if sha(q['x_path'])!=q['x_sha256'] or sha(q['roles_path'])!=q['roles_sha256'] or sha(q['weights_path'])!=q['weights_sha256']:raise ValueError('Weighted child input changed')
 x=np.load(q['x_path'],mmap_mode='r',allow_pickle=False);w=np.load(q['weights_path'],allow_pickle=False)
 with np.load(q['roles_path'],allow_pickle=False) as z:y=np.array(z['y'],copy=True);masks={k:np.array(z[k],copy=True) for k in ROLES}
 report=(fit_multiclass if q['params']==MULTICLASS else fit_binary)(lgb,x,y,masks,w,out)
 report['files']={n:sha(out/n) for n in ('model.txt','margins.npz','curve.json')};(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');return report

def synthetic_parity(out,native_path,native_sha,multiclass_src,binary_src):
 """Synthetic rows only. Original pinned children vs this child at w=1, then a non-uniform weight."""
 import importlib.util
 lgb=runtime(native_path,native_sha);out=Path(out);out.mkdir()
 def module(p):
  s=importlib.util.spec_from_file_location(Path(p).stem+'_pinned',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
 mc=module(multiclass_src);ev=module(binary_src)
 if mc.PARAMS!=MULTICLASS or ev.PARAMS!=BINARY:raise ValueError('Params drift from pinned children')
 rng=np.random.default_rng(101);n=1200;y=np.tile(np.arange(10),120).astype(np.int64);x=rng.normal(size=(n,199)).astype(np.float32);x[:,0]=y/2+rng.normal(size=n)*.2
 b=[0,600,800,1000,1200];masks={r:(np.arange(n)>=b[i])&(np.arange(n)<b[i+1]) for i,r in enumerate(ROLES)}
 one=np.where(masks['train'],1.,0.);res={}
 mc.fit_fixed(x,y,masks,out/'mc-original',synthetic_only=True);fit_multiclass(lgb,x,y,masks,one,out/'mc-weighted-w1')
 a=np.load(out/'mc-original/margins.npz');c=np.load(out/'mc-weighted-w1/margins.npz')
 res['multiclass_w1_max_abs_margin_diff']=max(float(np.abs(a[k]-c[k]).max()) for k in ('tuning','calibration','evaluation'))
 res['multiclass_w1_model_txt_identical']=(out/'mc-original/model.txt').read_bytes()==(out/'mc-weighted-w1/model.txt').read_bytes()
 # Binary: run the pinned event_tree_child through its own request path (native/source hash checks).
 yb=(y>=5).astype(np.int8);roles=out/'bin-roles.npz';np.savez(roles,y=yb,**masks);xp=out/'x.npy';np.save(xp,x,allow_pickle=False)
 req={'out':str(out/'bin-original'),'x_path':str(xp),'x_sha256':sha(xp),'roles_path':str(roles),'roles_sha256':sha(roles),'source_sha256':sha(binary_src),'native_sha256':native_sha,'native_path':native_path,'max_rounds':200,'params':BINARY}
 (out/'bin-request.json').write_text(json.dumps(req));ev.run(out/'bin-request.json');fit_binary(lgb,x,yb,masks,one,out/'bin-weighted-w1')
 a=np.load(out/'bin-original/margins.npz');c=np.load(out/'bin-weighted-w1/margins.npz')
 res['binary_w1_max_abs_margin_diff']=max(float(np.abs(a[k]-c[k]).max()) for k in ('tuning','calibration','evaluation'))
 # Non-uniform weights (raw [1,2], TRAIN mean 1) must reach the native Dataset and change the fit.
 raw=np.where(masks['train'],1.+(np.arange(n)%3==0),0.);w=np.where(masks['train'],raw/raw[masks['train']].mean(),0.)
 ds=lgb.Dataset(x[masks['train']],label=y[masks['train']],weight=check_weights(w,masks['train']),free_raw_data=False).construct()
 res['native_weight_roundtrip']=bool(np.allclose(ds.get_weight(),w[masks['train']],rtol=0,atol=1e-6))
 fit_multiclass(lgb,x,y,masks,w,out/'mc-weighted-w');d=np.load(out/'mc-weighted-w/margins.npz')
 res['multiclass_weight_changes_margins']=bool(np.abs(d['evaluation']-np.load(out/'mc-weighted-w1/margins.npz')['evaluation']).max()>1e-6)
 res['tuning_unweighted_replay']=True  # fit_multiclass already replays the unweighted TUNE loss at the selected iteration
 res['lightgbm_version']=lgb.__version__;res['classifier_fits_synthetic']=5
 if res['multiclass_w1_max_abs_margin_diff']!=0 or res['binary_w1_max_abs_margin_diff']!=0 or not res['native_weight_roundtrip'] or not res['multiclass_weight_changes_margins']:raise ValueError('Parity failure '+json.dumps(res))
 (out/'parity.json').write_text(json.dumps(res,indent=2)+'\n');return res

if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--request');g.add_argument('--synthetic-parity',nargs=5,metavar=('OUT','NATIVE','NATIVE_SHA','MULTICLASS_SRC','BINARY_SRC'))
 args=a.parse_args()
 print(json.dumps(run(args.request) if args.request else synthetic_parity(*args.synthetic_parity)))
