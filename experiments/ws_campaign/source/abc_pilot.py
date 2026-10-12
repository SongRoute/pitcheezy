"""A/B/C 2026 limited Mac pilot on the existing seven-leaf path (parent10 + two conditional heads).

Modes: --selfcheck (synthetic only), --plan (config + source gates, no rows), --execute --stage audit|full.
--execute is the real chain and stays locked until a distinct REVIEW-PASS binds this exact source/config/
registration; it never reads S2 rows before every gate passes. Runner python: .venv-observer (pandas/pyarrow);
classifier children: pinned lightgbm-env python via weighted_tree_child.py, one fit at a time.
"""
import argparse,ast,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=('A','B','C');HEADS=('parent','narrow_whiff','ground_ball_per_pitch');BIP=(3,4,5,6,7,9);ROLES=('train','tuning','calibration','evaluation')
OWN=('abc_pilot.py','abc_corpus.py','weighted_tree_child.py','ABC-PILOT-CONFIG.json','PILOT-REGISTRATION-DRAFT.json')
REALIZED_CURRENT={'description','bb_type','outcome10','release_speed','release_spin_rate','pfx_x','pfx_z','events'}
INVENTORY={'classifier':9,'temperature':9}
PHYSICS=('release_speed','release_spin_rate','pfx_x','pfx_z');SIM_SCALES={'release_speed':2.0,'release_spin_rate':200.0,'pfx_x':.25,'pfx_z':.25}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n')

def load(path):
 c=json.loads(Path(path).read_text())
 for name,(p,h) in c['helpers'].items():
  if sha(p)!=h:raise ValueError('Pinned helper changed: '+name)
 w=c['weights']
 if c['arms']!=list(ARMS) or list(c['heads'])!=list(HEADS) or c['roles']['exposure']!='ALL EXPOSED DEVELOPMENT':raise ValueError('Registered arm/head/exposure scope')
 if c['history_cutoff']!=c['roles']['FIT'][1]:raise ValueError('History must freeze at FIT end')
 if (w['alpha'],w['beta'],w['raw_bounds'],w['max_min_ratio'],w['min_ess_fraction'])!=(1,1,[1,2],2,.85) or not w['same_alpha_B_C']:raise ValueError('Registered weight design changed')
 s=w['similarity']
 if s['physics']!=list(PHYSICS) or s['scales']!=SIM_SCALES or s['group']!=['pitcher','pitch_type','p_throws']:raise ValueError('Registered physical similarity changed')
 return c

def helper_module(c,name):
 path=c['helpers'][name][0]
 if name in sys.modules:return sys.modules[name]
 d=str(Path(path).parent)
 if d not in sys.path:sys.path.insert(0,d)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def pinned_function(c,name,func):
 """Exact pinned function source (sha-checked file) compiled alone; avoids the module's unrelated duckdb import."""
 path=c['helpers'][name][0];node=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==func)
 ns={'np':np};exec(compile(ast.Module([node],[]),path,'exec'),ns);return ns[func]

def corpus():
 if 'abc_corpus' in sys.modules:return sys.modules['abc_corpus']  # one module instance: ProjectionStop identity is shared
 spec=importlib.util.spec_from_file_location('abc_corpus',HERE/'abc_corpus.py');m=importlib.util.module_from_spec(spec);sys.modules['abc_corpus']=m;spec.loader.exec_module(m);return m

# Source-level column allowlist: string literals reachable from the helper constants, no import/execution.
def symbols(paths):
 table={}
 for p in paths:
  for node in ast.parse(Path(p).read_text()).body:
   if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):table.setdefault(node.targets[0].id,node.value)
 return table
def literals(expr,table,seen=()):
 out=set()
 for n in ast.walk(expr):
  if isinstance(n,ast.Constant) and isinstance(n.value,str):out.add(n.value)
  elif isinstance(n,ast.Name) and n.id in table and n.id not in seen:out|=literals(table[n.id],table,seen+(n.id,))
 return out
def feature_gate(c,extra_donor=(),extra_project=()):
 t=symbols([c['helpers'][k][0] for k in ('candidate_input','first_small_features','history_transfer','outcome_transfer','batter_history_research')])
 query=literals(t['CONTEXT'],t);donor=literals(t['DONOR'],t)|set(extra_donor);forbidden=set(c['forbidden_feature_columns'])
 project=set(literals(symbols([HERE/'abc_corpus.py'])['PROJECT'],symbols([HERE/'abc_corpus.py'])))|set(extra_project)
 if forbidden&(query|donor|project):raise ValueError('Forbidden column reaches projection/features: '+str(sorted(forbidden&(query|donor|project))))
 if REALIZED_CURRENT&query:raise ValueError('Realized current-pitch column in query schema')
 return {'query_columns':sorted(query),'prior_only_donor_columns':sorted(donor-query),'projected_columns':sorted(project)}

def fit_inventory():
 return {'per_arm':{'classifier':3,'temperature':3},'total':dict(INVENTORY),'order':'9 classifier fits -> frozen uncalibrated SELECT-DECISION.json -> 9 unweighted CAL temperatures -> CAL/TEST safety metrics','fits':[{'arm':a,'head':h,'classifier':1,'temperature':1} for a in ARMS for h in HEADS]}

def plan(c):
 return {'feature_gate':feature_gate(c),'inventory':fit_inventory(),'weights':c['weights'],'fit_sample':c['fit_sample'],'budgets':c['budgets'],'source_hashes':{n:sha(HERE/n) for n in OWN}}

# ---------- gates (all before any S2 byte is decoded) ----------
def review_gate(c,stage,review_path=None):
 p=Path(review_path or c['review']['path'])
 if not p.exists():raise SystemExit('BLOCKED: distinct science/data REVIEW-PASS required at '+str(p))
 r=json.loads(p.read_text())
 if r.get('verdict')!='PASS' or r.get('distinct_from_author') is not True or stage not in r.get('approved_stages',[]):raise SystemExit('BLOCKED: review verdict/distinctness/stage')
 if r.get('sources_sha256')!={n:sha(HERE/n) for n in OWN}:raise SystemExit('BLOCKED: reviewed source/config/registration hashes differ from current files')
 return sha(p)

def runtime_gate(c):
 rt=c['runtime']
 if sha(rt['native_path'])!=rt['native_sha256'] or not Path(rt['tree_python']).exists():raise SystemExit('BLOCKED: pinned LightGBM runtime')
 if sha(c['roster']['path'])!=c['roster']['sha256']:raise SystemExit('BLOCKED: roster changed')

def cohort(c):
 d=json.loads(Path(c['roster']['path']).read_text());ids={int(p['id']) for t in d['teams'].values() for p in t['players']}
 if len(ids)!=c['roster']['players']:raise SystemExit('BLOCKED: cohort size')
 return ids

def claim_attempt(c,stage):
 """Stage-specific registered attempt ID (audit-001 / full-001); never reused, never overwritten."""
 ids=c['outputs']['attempt_ids'];aid=ids.get(stage)
 if len(set(ids.values()))!=len(ids) or not aid or not aid.startswith(stage+'-'):raise SystemExit('BLOCKED: registered stage attempt id')
 root=Path(c['outputs']['attempt_root']);att=root/aid
 if att.exists():raise SystemExit('BLOCKED: attempt exists (fresh output only)')
 if root.exists():
  for o in root.iterdir():
   if o.is_dir() and (o/'CLAIM').exists() and not (o/'terminal.json').exists():raise SystemExit('BLOCKED: stale/overlapping attempt '+o.name)
 att.mkdir(parents=True)
 with (att/'CLAIM').open('x') as f:f.write(json.dumps({'pid':os.getpid(),'utc':time.time(),'stage':stage,'attempt_id':aid}))
 return att

def preflight(c,att,stage,fs,counts):
 """Lock then recorder/resource gate inside the owned attempt. On failure write a terminal STOP receipt there;
 no retry, and never release or touch another process's lock (only this runner's own fd is closed)."""
 lock=Lock(c)
 try:lock.__enter__()
 except BaseException as e:save(att/'terminal.json',{'success':False,'stage':stage,'phase':'lock_preflight','counts':counts,'error':repr(e),'retry':False});raise
 try:before=fs.resource();save(att/'resource-before.json',before)
 except BaseException as e:
  lock.__exit__();save(att/'terminal.json',{'success':False,'stage':stage,'phase':'resource_preflight','counts':counts,'error':repr(e),'retry':False});raise
 return lock,before

class Lock:
 """Existing shared .heavy.lock only. mode runner: LOCK_EX|LOCK_NB here. mode supervisor: the supervisor
 already holds it and passes the fd in ABC_SUPERVISOR_LOCK_FD; no second flock (no self-deadlock/double flock)."""
 def __init__(self,c):self.c=c['lock'];self.f=None
 def __enter__(self):
  p=Path(self.c['path'])
  if not p.is_file():raise SystemExit('BLOCKED: shared lock file missing (never created by this runner)')
  if self.c['mode']=='supervisor':
   fd=int(os.environ['ABC_SUPERVISOR_LOCK_FD'])
   if os.fstat(fd).st_ino!=p.stat().st_ino:raise SystemExit('BLOCKED: supervisor fd is not the shared lock')
   return self
  if self.c['mode']!='runner':raise SystemExit('BLOCKED: lock mode')
  self.f=p.open('r')
  try:fcntl.flock(self.f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:self.f.close();self.f=None;raise SystemExit('BLOCKED: shared lock busy')
  return self
 def __exit__(self,*a):
  if self.f:self.f.close()

def alarm(seconds):
 def expired(*_):raise TimeoutError('Registered stage time cap')
 signal.signal(signal.SIGALRM,expired);signal.alarm(max(1,int(seconds)))

# ---------- shared evaluation pieces ----------
def seven_label(y10,whiff,gb):
 y=np.full(len(y10),-1);y[y10==0]=0;y[y10==2]=1;y[y10==8]=2;s=y10==1;y[s&(whiff==1)]=3;y[s&(whiff==0)]=4;b=np.isin(y10,BIP);y[b&(gb==1)]=5;y[b&(gb==0)]=6
 if (y<0).any():raise ValueError('Seven-leaf label coverage')
 return y
def seven_metrics(q,y7):
 p=np.clip(q[np.arange(len(y7)),y7],1e-15,1);ll=-np.log(p);oh=np.eye(7)[y7];bs=np.square(q-oh).sum(1)
 return {'rows':len(y7),'nll':float(ll.mean()),'brier':float(bs.mean())},ll,bs

def head_roles(y10,yh,m,head):
 parent=np.ones(len(y10),bool) if head=='parent' else (y10==1) if head=='narrow_whiff' else np.isin(y10,BIP)
 masks={r:(m[r]&parent if r in ('train','tuning') else m[r]) for r in ROLES}
 y=y10.astype(np.int64) if head=='parent' else yh[head].astype(np.int8)
 return y,masks,parent

# ---------- runner lifecycle: cooperative stop; catchable signals only record, check_stop raises at safe points (SIGKILL is NOT covered) ----------
class RunnerSignal(BaseException):pass
_PENDING=[]
def _on_signal(n,_):_PENDING.append(signal.Signals(n).name)  # record only: never raises mid-Popen, receipt write, finally or lock release
def check_stop():
 """Deliberate safe checkpoint: a pending SIGTERM/SIGINT/SIGHUP becomes RunnerSignal here and nowhere else."""
 if _PENDING:raise RunnerSignal(_PENDING[0])
class Signals:
 """SIGTERM/SIGINT/SIGHUP record a pending stop (no raise, no sys.exit/os._exit, no lock release in the handler); previous handlers restored."""
 SIGS=(signal.SIGTERM,signal.SIGINT,signal.SIGHUP)
 def __enter__(self):_PENDING.clear();self.prev={s:signal.signal(s,_on_signal) for s in self.SIGS};return self
 def __exit__(self,*a):
  for s,h in self.prev.items():signal.signal(s,h)
def durable(p,x):
 p=Path(p);t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:f.write(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p);fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def os_identity(pid):
 """Narrow exact-PID ps: OS start time (lstart) + pgid + argv; start time is the PID-reuse discriminator, argv supplemental."""
 try:r=subprocess.run(['/bin/ps','-o','lstart=','-o','pgid=','-o','command=','-p',str(pid)],capture_output=True,text=True,timeout=5);return {'ps_rc':r.returncode,'ps_line':r.stdout.strip()}
 except Exception as e:return {'error':repr(e)}
def group_gone(pgid):
 try:os.killpg(pgid,0);return False
 except ProcessLookupError:return True

def child_fit(c,d,x_path,roles_path,w_path,params,guard,argv=None,hook=None):
 rt=c['runtime'];q={'out':str(d/'child-output'),'x_path':str(x_path),'x_sha256':sha(x_path),'roles_path':str(roles_path),'roles_sha256':sha(roles_path),'weights_path':str(w_path),'weights_sha256':sha(w_path),'source_sha256':sha(HERE/'weighted_tree_child.py'),'native_sha256':rt['native_sha256'],'native_path':rt['native_path'],'max_rounds':200,'params':params}
 if (argv is not None or hook is not None) and c.get('synthetic_lifecycle') is not True:raise SystemExit('BLOCKED: injected child command/test hook outside the synthetic lifecycle check')
 hook=hook or (lambda phase:None)
 check_stop();save(d/'request.json',q);cs=time.monotonic();last=cs;proc=None;ident=None;gone=None
 try:
  with (d/'child.log').open('x') as log:
   proc=subprocess.Popen(argv or [rt['tree_python'],'-I',str(HERE/'weighted_tree_child.py'),'--request',str(d/'request.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   hook('spawned')  # synthetic-only: deterministic signal at the spawn-to-identity boundary
   ident={'child_pid':proc.pid,'child_pgid':proc.pid,'child_os':os_identity(proc.pid),'runner_pid':os.getpid(),'runner_os':os_identity(os.getpid()),'attempt':d.parent.name,'head':d.name,'popen_utc':time.time(),'terminal':None}
   durable(d/'child-identity.json',ident)
   while proc.poll() is None:
    check_stop()  # <=0.5 s stop latency while the child runs
    if time.monotonic()-cs>c['budgets']['child_seconds']:raise TimeoutError('Classifier time cap')
    if time.monotonic()-last>c['budgets']['guard_interval_seconds']:guard();last=time.monotonic()
    time.sleep(.5)
  if proc.returncode:raise RuntimeError('Classifier failed '+str(proc.returncode))
 finally:  # handler never raises, so this block runs to completion; only the exact owned child PGID is signaled
  hook('cleanup')  # synthetic-only: deterministic signal during (normal-success) cleanup
  how='exit'
  if proc is not None and proc.poll() is None:
   how='TERM';os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:how='KILL';os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  if proc is not None:  # bounded final group check (one-child path: group == reaped leader) before control returns to the lock owner
   for _ in range(25):
    gone=group_gone(proc.pid)
    if gone:break
    os.killpg(proc.pid,signal.SIGKILL);time.sleep(.2)
  if ident is not None:durable(d/'child-identity.json',dict(ident,terminal={'how':how,'returncode':proc.returncode,'group_gone':gone,'utc':time.time(),'pending_signals':list(_PENDING)}))
 if not gone:raise RuntimeError('STOP child group not confirmed gone')
 check_stop()  # a stop recorded during normal-success cleanup is consumed here, before the receipt returns and before any next head
 return json.loads((d/'child-output'/'receipt.json').read_text())

def role_label_census(g,rm):
 """Aggregate per temporal role (FIT = full eligible FIT pool) label-eligibility counts; no row identifiers."""
 out={}
 for r,v in rm.items():
  y10=g.outcome10.to_numpy()[v];y7=seven_label(y10,g.narrow_whiff.to_numpy()[v],g.ground_ball_per_pitch.to_numpy()[v])
  out[r]={'accepted_rows':int(v.sum()),'parent_label_eligible':int(np.isin(y10,np.arange(10)).sum()),'whiff_head_eligible':int((y10==1).sum()),'gb_head_eligible':int(np.isin(y10,BIP).sum()),'seven_leaf_eligible':int(len(y7)),'seven_leaf_counts':np.bincount(y7,minlength=7).tolist()}
 return out

def project_census(c,raw,out):
 """Projection + aggregate census, persisted to projection.json BEFORE any row-quality STOP is re-raised."""
 cp=corpus()
 try:g,census=cp.project(raw,c,pinned_function(c,'project_first_small_train','labels'),helper_module(c,'event_heads').labels)
 except cp.ProjectionStop as e:
  save(out/'projection.json',dict(e.census,role_label_census={'available':False,'reason':'row-quality STOP raised before the accepted frame and head labels were built; overall reason/bb_type category counts above are complete'}));raise
 rm=cp.roles(g,c);census['role_rows']={r:int(v.sum()) for r,v in rm.items()};census['role_label_census']={'available':True,'roles':role_label_census(g,rm)}
 save(out/'projection.json',census);return g,census,rm

def build_corpus(c,raw,cohort_ids,out,deadline):
 """One shared projected/encoded corpus + all nine weight vectors and diagnostics, before any fit."""
 cp=corpus();fsf=helper_module(c,'first_small_features');ci=helper_module(c,'candidate_input')
 g,census,rm=project_census(c,raw,out);check_stop()
 fit_train,sample=cp.sample_games(g,rm['FIT'],c)
 query=fit_train|rm['SELECT']|rm['CAL']|rm['TEST'];q=g.loc[query].reset_index(drop=True)
 m={'train':fit_train[query],'tuning':rm['SELECT'][query],'calibration':rm['CAL'][query],'evaluation':rm['TEST'][query]}
 if not np.all(sum(v.astype(int) for v in m.values())==1):raise ValueError('Exhaustive disjoint query roles')
 denominators={'source_rows':census['source_rows'],'accepted_rows':census['accepted_rows'],'fit_history_pool_rows':int(rm['FIT'].sum()),'fit_history_pool_games':int(g.game_pk[rm['FIT']].nunique()),'fit_train_sample_rows':int(fit_train.sum()),'fit_train_sample':sample,**{r+'_rows':int(rm[r].sum()) for r in ('SELECT','CAL','TEST')},**{r+'_games':int(g.game_pk[rm[r]].nunique()) for r in ('SELECT','CAL','TEST')},'note':'FIT-train is a 25% whole-game sample of the eligible FIT pool; the FULL FIT pool is the strictly-prior history donor pool'}
 save(out/'denominators.json',denominators)
 if time.monotonic()>deadline:raise TimeoutError('Projection cap')
 check_stop();alarm(c['budgets']['feature_seconds'])
 joined,audit=cp.features(g,rm['FIT'],query,c['history_cutoff'],ci,fsf);signal.alarm(0);check_stop()
 x,enc=fsf.encode(joined,m['train'])  # FIT-train only, unweighted, one encoder for all arms
 if x.shape!=(len(q),199) or not np.isfinite(x).all():raise ValueError(f'STOP encoded dimension {x.shape[1]} != registered 199')
 np.save(out/'x.npy',x,allow_pickle=False);save(out/'encoder.json',enc)
 y10=q.outcome10.to_numpy(np.int64);yh={h:q[h].to_numpy(np.int8) for h in HEADS[1:]};games=q.game_pk.to_numpy()
 target=(q.pitcher.isin(cohort_ids)|q.batter.isin(cohort_ids)).to_numpy()  # union target; overlap row counted once
 s=np.zeros(len(q));tr=m['train'];nt=tr&~target  # C similarity only for non-target FIT-train rows; target rows keep the B weight
 check_stop();s[nt],s_census=cp.similarity(g.loc[rm['FIT']],q.loc[nt],cohort_ids,c['weights']['similarity']);check_stop()
 np.savez(out/'corpus.npz',y10=y10,games=games,target=target,similarity=s,keys=q[cp.KEY].to_numpy('<i8'),**{h:yh[h] for h in yh},**m)
 diag={};paths={}
 for head in HEADS:
  check_stop();y,hm,_=head_roles(y10,yh,m,head);rp=out/f'roles-{head}.npz';np.savez(rp,y=y,**hm);paths[head]={'roles':rp}
  for arm in ARMS:
   w,ess=cp.head_weights(arm,target,s,hm['train'],c['weights']);wp=out/f'weights-{arm}-{head}.npy';np.save(wp,w,allow_pickle=False);paths[head][arm]=wp
   diag[f'{arm}/{head}']=cp.mass(w,hm['train'],target,q.pitcher.to_numpy(),q.batter.to_numpy(),q.pitch_type.to_numpy())
 save(out/'weight-diagnostics.json',{'diagnostics':diag,'single_player_mass':'REPORT-ONLY in this pilot; post-run review action required before any expansion','transfer_audit':audit,'similarity_nontarget_train':{'census':s_census,'rows':int((tr&~target).sum()),'zero_share':float((s[tr&~target]==0).mean()) if (tr&~target).any() else None,'mean':float(s[tr&~target].mean()) if (tr&~target).any() else None}})
 return {'y10':y10,'yh':yh,'m':m,'games':games,'target':target,'keys':q[cp.KEY].to_numpy('<i8'),'pitcher':q.pitcher.to_numpy(),'batter':q.batter.to_numpy(),'paths':paths,'x':out/'x.npy'}

def post_cal_diagnostics(q,y7,pitcher,batter):
 """REPORT-ONLY after all nine CAL temperatures: seven-leaf reliability in ten fixed bins and per-pitcher /
 per-batter n, NLL, Brier on the same eligible rows. Feeds nothing (no SELECT, temperature or policy)."""
 ll=-np.log(np.clip(q[np.arange(len(y7)),y7],1e-15,1));bs=np.square(q-np.eye(7)[y7]).sum(1);rel={}
 for leaf in range(7):
  p=q[:,leaf];o=(y7==leaf).astype(float);b=np.minimum((p*10).astype(int),9);rel[str(leaf)]=[]
  for i in range(10):
   s=b==i;k=int(s.sum());rel[str(leaf)].append({'bin':[i/10,(i+1)/10],'count':k,'mean_predicted':float(p[s].mean()) if k else None,'observed_frequency':float(o[s].mean()) if k else None})
 def group(ids):
  rows=[{'id':int(u),'n':int(s.sum()),'nll':float(ll[s].mean()),'brier':float(bs[s].mean())} for u in np.unique(ids) for s in [ids==u]]
  n=np.array([x['n'] for x in rows])
  if n.sum()!=len(y7):raise ValueError('STOP player denominator does not reconcile')
  return {'players':len(rows),'rows':int(n.sum()),'n_min':int(n.min()),'n_median':float(np.median(n)),'n_max':int(n.max()),'players_by_n':{'1':int((n==1).sum()),'2-9':int(((n>=2)&(n<=9)).sum()),'10-99':int(((n>=10)&(n<=99)).sum()),'100+':int((n>=100).sum())},'per_player':rows}
 return {'reliability_seven_leaf':rel,'per_pitcher':group(np.asarray(pitcher)),'per_batter':group(np.asarray(batter)),'rows':int(len(y7)),
  'caution':'Per-player NLL/Brier are unweighted means over that player\'s n eligible rows; players_by_n is descriptive only. Small n gives high-variance values; no pass threshold, ranking or confidence claim is made.',
  'scope':'all eligible accepted pitches of this role (no abstention in this pilot); source/row exclusions are in projection.json; REPORT-ONLY, never used for SELECT, temperature fitting, adoption or service policy'}

def evaluate(c,k,margins,temps,out,role='evaluation'):
 """Coherent seven leaves via the pinned coherence leaves(); unweighted metrics on CAL or development TEST;
 game-paired B-A, C-A on TEST only. Safety reporting only: nothing here feeds SELECT or reselects an arm."""
 from scipy.special import softmax,expit
 run=helper_module(c,'coherence_run');ld=helper_module(c,'learning_diagnostic');ev=k['m'][role]
 y7=seven_label(k['y10'][ev],k['yh']['narrow_whiff'][ev],k['yh']['ground_ball_per_pitch'][ev]);tgt=k['target'][ev];games=k['games'][ev];res={};loss={}
 for arm in ARMS:
  p=softmax(margins[arm]['parent'][role]/temps[arm]['parent'],axis=1);qw=expit(margins[arm]['narrow_whiff'][role]/temps[arm]['narrow_whiff']);rg=expit(margins[arm]['ground_ball_per_pitch'][role]/temps[arm]['ground_ball_per_pitch'])
  q=run.leaves(p,qw,rg);np.save(out/f'seven-leaf-{arm}-{role}.npy',q,allow_pickle=False)
  allm,ll,bs=seven_metrics(q,y7);tm,_,_=seven_metrics(q[tgt],y7[tgt]);loss[arm]=(ll,bs)
  res[arm]={'overall':allm,'target':tm,'max_sum_error':float(np.abs(q.sum(1)-1).max()),'temperatures':temps[arm],'post_cal_diagnostics':post_cal_diagnostics(q,y7,k['pitcher'][ev],k['batter'][ev])}
 res['denominators']={'rows':int(ev.sum()),'games':int(len(np.unique(games))),'target_rows':int(tgt.sum()),'target_games':int(len(np.unique(games[tgt])))}
 if role=='evaluation':
  seed=c['bootstrap_seed'];res['paired']={f'{a}-A':{sub:{met:ld.paired((loss[a][i]-loss['A'][i])[sel],games[sel],seed) for i,met in enumerate(('nll','brier'))} for sub,sel in (('overall',np.ones(len(y7),bool)),('target',tgt))} for a in ('B','C')}
 return res

# ---------- preregistered SELECT (uncalibrated) frozen before any CAL temperature ----------
SELECT_RULE='advisory_selected = arm with the lowest uncalibrated SELECT target seven-leaf NLL among arms whose overall SELECT seven-leaf NLL <= arm A overall; ties -> A,B,C order. Advisory record only: no adoption, no policy, no TEST reselection.'
LEAF_MAP={'0 ball':'y10==0 -> p10[0]','1 foul':'y10==2 -> p10[2]','2 HBP':'y10==8 -> p10[8]','3 whiff':'y10==1 & narrow_whiff==1 -> p10[1]*q','4 strike non-whiff':'y10==1 & narrow_whiff==0 -> p10[1]*(1-q)','5 GB':'y10 in BIP & ground_ball==1 -> sum p10[BIP]*r','6 non-GB BIP':'y10 in BIP & ground_ball==0 -> sum p10[BIP]*(1-r)'}

def select_decision(k,margins):
 """Uncalibrated true-leaf seven-leaf NLL on the identical eligible SELECT rows for A/B/C. Reads only SELECT
 (tuning) margins and SELECT labels; conditional heads have tuning margins only on their true parent-class rows,
 which is exactly where the observed leaf needs them. STOP on any label/margin inconsistency."""
 from scipy.special import softmax,expit
 tu=k['m']['tuning'];y10=k['y10'][tu];wf=k['yh']['narrow_whiff'][tu];gb=k['yh']['ground_ball_per_pitch'][tu];tgt=k['target'][tu];games=k['games'][tu]
 strike=y10==1;bip=np.isin(y10,BIP)
 if not tu.any() or not tgt.any():raise ValueError('STOP SELECT overall/target rows empty')
 if not (np.isin(y10,np.arange(10)).all() and np.isin(wf,(0,1)).all() and np.isin(gb,(0,1)).all()):raise ValueError('STOP SELECT label domain')
 if ((wf==1)&~strike).any() or ((gb==1)&~bip).any():raise ValueError('STOP SELECT head label set outside its parent class')
 y7=seven_label(y10,wf,gb);arms={};losses={'games':games,'target':tgt,'y7':y7}
 for arm in ARMS:
  zp,zw,zg=(margins[arm][h]['tuning'] for h in HEADS)
  if zp.shape!=(len(y10),10) or zw.shape!=(int(strike.sum()),) or zg.shape!=(int(bip.sum()),):raise ValueError(f'STOP SELECT {arm} tuning margins do not match parent/head label availability')
  if not (np.isfinite(zp).all() and np.isfinite(zw).all() and np.isfinite(zg).all()):raise ValueError('STOP SELECT non-finite margins')
  p=softmax(zp,axis=1);q=np.full(len(y10),np.nan)
  for leaf,cls in ((0,0),(1,2),(2,8)):q[y7==leaf]=p[y7==leaf,cls]
  w=expit(zw);q[strike]=p[strike,1]*np.where(wf[strike]==1,w,1-w)
  r=expit(zg);q[bip]=p[bip][:,list(BIP)].sum(1)*np.where(gb[bip]==1,r,1-r)
  if not np.isfinite(q).all():raise ValueError('STOP SELECT true-leaf coverage')
  ll=-np.log(np.clip(q,1e-15,1));losses[arm]=ll
  arms[arm]={'target_nll':float(ll[tgt].mean()),'overall_nll':float(ll.mean())}
 safe=[a for a in ARMS if arms[a]['overall_nll']<=arms['A']['overall_nll']]
 rank=sorted(ARMS,key=lambda a:(arms[a]['target_nll'],ARMS.index(a)))
 dec={'stage':'SELECT uncalibrated (raw margins, T=1), frozen before any CAL temperature is fit or read','primary':'target-cohort seven-leaf NLL','safety':'overall seven-leaf NLL','leaf_map':LEAF_MAP,'arms':arms,
  'advisory_rank_by_primary':rank,'safety_eligible':safe,'advisory_selected':min(safe,key=lambda a:(arms[a]['target_nll'],ARMS.index(a))),'rule':SELECT_RULE,'adoption':False,'automatic_policy':False,'cal_or_test_read':False,
  'denominators':{'rows':int(tu.sum()),'games':int(len(np.unique(games))),'target_rows':int(tgt.sum()),'target_games':int(len(np.unique(games[tgt]))),'head_tuning_rows':{'parent':int(tu.sum()),'narrow_whiff':int(strike.sum()),'ground_ball_per_pitch':int(bip.sum())},'leaf_counts':{'overall':np.bincount(y7,minlength=7).tolist(),'target':np.bincount(y7[tgt],minlength=7).tolist()}},
  'identity':{'row_keys_sha256':corpus().digest(k['keys'][tu]),'game_pks_sha256':corpus().digest(np.unique(games).astype('<i8')),'per_game_rows':'select-losses.npz (games, target, y7, per-row NLL per arm)'}}
 return dec,losses

def select_then_calibrate(k,margins,out,calibrate,guard=lambda:None):
 """Write the frozen SELECT-DECISION.json first, then exactly nine unweighted CAL temperatures. The decision
 file is hash-checked before every temperature call and after the last one."""
 dec,losses=select_decision(k,margins);np.savez(out/'select-losses.npz',**losses);dec['identity']['select_losses_array_sha256']={n:corpus().digest(v) for n,v in losses.items()}
 dp=out/'SELECT-DECISION.json'
 with dp.open('x') as f:f.write(json.dumps(dec,indent=2,allow_nan=False)+'\n')
 frozen=sha(dp);temps={a:{} for a in ARMS};ca=k['m']['calibration'];ev=k['m']['evaluation']
 for arm in ARMS:
  for head in HEADS:
   if sha(dp)!=frozen:raise ValueError('STOP SELECT decision changed')
   guard();y,_,parent=head_roles(k['y10'],k['yh'],k['m'],head);z=margins[arm][head]
   if head=='parent':temps[arm][head]=calibrate(head,z['calibration'],y[ca],z['evaluation'],y[ev])
   else:temps[arm][head]=calibrate(head,z['calibration'][parent[ca]],y[ca][parent[ca]],None,None)  # CAL rows within parent class, unweighted
 if sha(dp)!=frozen:raise ValueError('STOP SELECT decision changed')
 return temps,frozen

def execute(c,stage):
 if stage not in ('audit','full'):raise SystemExit('BLOCKED: stage')
 if 'synthetic_lifecycle' in c:raise SystemExit('BLOCKED: synthetic lifecycle key in a real config')  # test-only child/hook injection is unreachable from execute
 review_sha=review_gate(c,stage);runtime_gate(c);ids=cohort(c);fs=helper_module(c,'run_first_small_m4')
 if Path(c['lock']['path'])!=fs.LOCK:raise SystemExit('BLOCKED: lock path is not the shared .heavy.lock')
 att=claim_attempt(c,stage);start=time.monotonic();b=c['budgets'];counts={'classifier_attempted':0,'classifier_completed':0,'temperature':0}
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
 def size():return sum(f.stat().st_size for f in att.rglob('*') if f.is_file())
 def guard():
  check_stop()
  if time.monotonic()-start>b['wall_seconds']:raise TimeoutError('Overall cap')
  if size()>b['max_output_bytes']:raise RuntimeError('Output cap')
  r=fs.resource()  # raises if the recorder queue/lease is busy: recorder keeps priority
  if r['memory_free_percent']<b['min_memory_free_percent'] or r['swap_used_mib']-before['swap_used_mib']>b['max_swap_delta_mib']:raise RuntimeError('Resource guard')
  save(att/'last-resource.json',r)
 sig=Signals().__enter__()  # SIGTERM/SIGINT/SIGHUP record a pending stop; check_stop raises it at safe points; restored after lock release
 try:lock,before=preflight(c,att,stage,fs,counts)  # lock is already held here; never entered twice
 except BaseException:sig.__exit__();raise
 try:
  try:
   guard();save(att/'gates.json',{'review_sha256':review_sha,'sources_sha256':{n:sha(HERE/n) for n in OWN},'helpers':c['helpers'],'stage':stage,'attempt_id':att.name})
   check_stop();alarm(b['projection_seconds']);cp=corpus();raw,binding=cp.read_source(c);save(att/'source-binding.json',binding);check_stop()  # source/manifest/hash failures raise here: no census is claimed
   if stage=='audit':
    project_census(c,raw,att);signal.alarm(0);check_stop()  # aggregate only: no features, fits, row identifiers or row export
    save(att/'terminal.json',{'success':True,'stage':'audit','counts':counts,'features_built':False,'elapsed_seconds':time.monotonic()-start,'artifacts':{f.name:sha(f) for f in att.iterdir() if f.is_file() and f.name!='terminal.json'}});return
   k=build_corpus(c,raw,ids,att,start+b['projection_seconds']+b['feature_seconds']);del raw;guard()
   margins={a:{} for a in ARMS};receipts={};ld=helper_module(c,'learning_diagnostic');eh=helper_module(c,'event_heads');wc=helper_module_path(HERE/'weighted_tree_child.py')
   for arm in ARMS:  # phase 1: all nine classifier fits; no temperature yet
    for head in HEADS:
     guard();d=att/f'{arm}-{head}';d.mkdir();counts['classifier_attempted']+=1;save(att/'counts.json',counts)
     receipts[f'{arm}/{head}']=child_fit(c,d,k['x'],k['paths'][head]['roles'],k['paths'][head][arm],wc.MULTICLASS if head=='parent' else wc.BINARY,guard)
     counts['classifier_completed']+=1;save(att/'counts.json',counts)
     z=np.load(d/'child-output'/'margins.npz');margins[arm][head]={r:z[r] for r in ('tuning','calibration','evaluation')}
   if counts!={'classifier_attempted':9,'classifier_completed':9,'temperature':0}:raise ValueError('Incomplete 9 classifier inventory before SELECT')
   def calibrate(head,zc,yc,ze,ye):  # phase 3 only, after SELECT-DECISION.json is frozen
    alarm(b['temperature_seconds'])
    t=ld.calibrate_and_score(zc,ze,yc,ye)[0]['temperature'] if head=='parent' else eh.temperature(yc,zc)
    signal.alarm(0);counts['temperature']+=1;save(att/'counts.json',counts);return t
   guard();temps,select_sha=select_then_calibrate(k,margins,att,calibrate,guard)  # phase 2 SELECT frozen, then phase 3 CAL
   check_stop()
   if counts!={'classifier_attempted':9,'classifier_completed':9,'temperature':9}:raise ValueError('Incomplete 9+9 inventory')
   result={'stage':'full','label':'EXPOSED DEVELOPMENT limited pilot; live arrival UNKNOWN','select_decision_sha256':select_sha,'select_decision':json.loads((att/'SELECT-DECISION.json').read_text()),'calibration_safety':evaluate(c,k,margins,temps,att,'calibration'),'evaluation':evaluate(c,k,margins,temps,att),'receipts':receipts,'counts':counts,'scientific_decision':None,'automatic_followup':False,'test_reselection':False}
   save(att/'result.json',result);guard()
   save(att/'terminal.json',{'success':True,'stage':'full','counts':counts,'elapsed_seconds':time.monotonic()-start,'artifacts':{str(f.relative_to(att)):sha(f) for f in att.rglob('*') if f.is_file() and f.name!='terminal.json'}})
  except BaseException as e:
   signal.alarm(0);save(att/'terminal.json',{'success':False,'stage':stage,'counts':counts,'error':repr(e),'signal':str(e) if isinstance(e,RunnerSignal) else None,'pending_signals':list(_PENDING),'projection_census_saved':(att/'projection.json').exists(),'retry':False})
   raise
  finally:
   try:save(att/'resource-after.json',dict(fs.resource(),pending_signals_at_exit=list(_PENDING)))  # records a stop that arrived after the last checkpoint
   except Exception as e:save(att/'resource-after.json',{'error':repr(e),'pending_signals_at_exit':list(_PENDING)})
 finally:
  lock.__exit__();sig.__exit__()

def helper_module_path(path):
 spec=importlib.util.spec_from_file_location(Path(path).stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

# ---------- ONE extended synthetic check (no real rows) ----------
def synthetic_frame(c,rng):
 """Synthetic statcast-shaped rows: 40 games over the 2026 role windows, 6 teams, injected defects."""
 import pandas as pd
 rows=[];teams=['AAA','BBB','CCC','DDD','EEE','FFF'];dates=pd.date_range('2026-03-25','2026-09-27',freq='5D')
 desc=[('ball',None,None),('called_strike',None,None),('swinging_strike',None,None),('foul',None,None),('hit_into_play','field_out','ground_ball'),('hit_into_play','single','line_drive'),('hit_into_play','home_run','fly_ball'),('hit_by_pitch','hit_by_pitch',None),('hit_into_play','double','ground_ball'),('hit_into_play','grounded_into_double_play','ground_ball'),('hit_into_play','triple','fly_ball'),('swinging_strike_blocked',None,None)]
 gid=1000
 for i,d in enumerate(dates):
  gid+=1;h,a=teams[i%6],teams[(i+1+i//6)%6]
  for ab in range(1,9):
   for pn in range(1,5):
    k=(i*31+ab*7+pn)%len(desc);dd,e,bb=desc[k];p=100+(ab%4)+(i%3)*10;bt=200+(ab%5)
    rows.append(dict(game_pk=gid,at_bat_number=ab,pitch_number=pn,game_date=d.strftime('%Y-%m-%d'),game_type='R',home_team=h,away_team=a,pitcher=p,batter=bt,pitch_type=['FF','SL','CH','CU'][(p+pn)%4],description=dd,events=e,bb_type=bb,release_speed=90.+rng.normal(),release_spin_rate=2200.,pfx_x=.1,pfx_z=1.,balls=min(pn-1,3),strikes=min(pn-1,2),outs_when_up=ab%3,inning=1+ab//3,inning_topbot='Top' if ab%2 else 'Bot',home_score=0,away_score=0,on_1b=None,on_2b=None,on_3b=None,stand='LR'[ab%2],p_throws='LR'[p%2]))
 f=pd.DataFrame(rows)
 # defects: identical duplicate; non-R copy; null game key; unmapped label; unknown pitch type; second date for game 1001.
 bad=[dict(f.iloc[0]),dict(f.iloc[42],game_type='S'),dict(f.iloc[43],game_pk=None),dict(f.iloc[40],description='automatic_ball',events=None,at_bat_number=99,pitch_number=1),dict(f.iloc[41],at_bat_number=98,pitch_number=1,pitch_type='ZZ'),dict(f.iloc[5],at_bat_number=97,pitch_number=1,game_date='2026-03-30')]
 # bb_type defects: BIP null; BIP '' (normalized missing); non-BIP with type; BIP unknown non-null; non-BIP literal 'nan' (accepted as missing).
 bip=dict(description='hit_into_play',events='field_out',pitch_number=1);non=dict(description='ball',events=None,pitch_number=1)
 bad+=[dict(f.iloc[44],**bip,bb_type=None,at_bat_number=96),dict(f.iloc[48],**bip,bb_type='',at_bat_number=92),dict(f.iloc[45],**non,bb_type='fly_ball',at_bat_number=95),dict(f.iloc[46],**bip,bb_type='bunt_grounder',at_bat_number=94),dict(f.iloc[47],**non,bb_type='nan',at_bat_number=93)]
 return pd.concat([f,pd.DataFrame(bad)],ignore_index=True)

def selfcheck(c):
 import pandas as pd
 cp=corpus();rng=np.random.default_rng(0);checks=[];tmp=Path(tempfile.mkdtemp(prefix='abc-selfcheck-'))
 labels10=pinned_function(c,'project_first_small_train','labels');eh=helper_module(c,'event_heads');fsf=helper_module(c,'first_small_features');ci=helper_module(c,'candidate_input')
 cs=dict(c,row_policy=dict(c['row_policy'],stop_share=dict(c['row_policy']['stop_share'],excluded_unmapped_label=.05,quarantined_multi_date_game=.2,excluded_duplicate_key=.05,excluded_bip_missing_bb_type=.01)))
 # 1 projection schema, fates, unknown-label counts, duplicate/multi-date/non-R/null policy, reconciliation.
 raw=synthetic_frame(c,rng);g,census=cp.project(raw,cs,labels10,eh.labels)
 ex=census['excluded'];assert census['accepted_rows']+sum(ex.values())==len(raw)
 assert ex['excluded_duplicate_key']==1 and ex['excluded_non_registered_game_type']==1
 assert ex['excluded_null_game_key']==1 and ex['excluded_unmapped_label']>=1 and 'automatic_ball|MISSING' in census['unmapped_label_categories'],census
 assert ex['quarantined_multi_date_game']>0 and census['multi_date_games']==1 and census['pitch_type_mapped_unknown']=={'ZZ':1}
 assert set(cp.PROJECT).isdisjoint(cp.NEVER_READ) and not set(c['forbidden_feature_columns'])&set(cp.PROJECT)
 # bb_type trust boundary: three separate counted reasons, literal category census, accepted-row invariant.
 assert (ex['excluded_bip_missing_bb_type'],ex['excluded_nonbip_with_bb_type'],ex['excluded_bip_unknown_bb_type'])==(2,1,1) and 'excluded_bip_bbtype_mismatch' not in ex,ex
 bc=census['bb_type_categories'];assert (bc['BIP|MISSING'],bc['BIP|'],bc['nonBIP|fly_ball'],bc['BIP|bunt_grounder'],bc['nonBIP|nan'])==(1,1,1,1,1) and sum(bc.values())==census['bb_type_category_rows'],bc
 assert set(bc)-{'nonBIP|MISSING','nonBIP|nan','nonBIP|fly_ball','BIP|MISSING','BIP|','BIP|bunt_grounder'}<={'BIP|'+t for t in c['row_policy']['bb_types']},bc
 gbip=(g.description=='hit_into_play').to_numpy();gb_s=g.bb_type.fillna('None').astype(str).to_numpy()
 assert np.isin(gb_s[gbip],c['row_policy']['bb_types']).all() and np.isin(gb_s[~gbip],cp.MISSING_BB).all() and np.array_equal(g.ground_ball_per_pitch.to_numpy()==1,gb_s=='ground_ball') and census['stop']==[]
 for reason in ('excluded_bip_missing_bb_type','excluded_nonbip_with_bb_type','excluded_bip_unknown_bb_type'):  # each STOP share is enforced and the census persists first
  d=tmp/('stop-'+reason);d.mkdir();c1=dict(cs,row_policy=dict(cs['row_policy'],stop_share=dict(cs['row_policy']['stop_share'],**{reason:0})))
  try:project_census(c1,raw,d);raise AssertionError('bb_type STOP not enforced '+reason)
  except cp.ProjectionStop as e:
   pj=json.loads((d/'projection.json').read_text());assert len(pj['stop'])==1 and reason in pj['stop'][0] and pj['excluded']==ex and pj['bb_type_categories']==bc and pj['role_label_census']['available'] is False and 'reason' in pj['role_label_census'],pj
 d=tmp/'audit-clean';d.mkdir();_,_,_=project_census(cs,raw,d);pj=json.loads((d/'projection.json').read_text());rl=pj['role_label_census']
 assert rl['available'] is True and list(rl['roles'])==list(cp.ORDER) and sum(v['accepted_rows'] for v in rl['roles'].values())==census['accepted_rows'],rl
 for v in rl['roles'].values():assert v['parent_label_eligible']==v['seven_leaf_eligible']==v['accepted_rows']==sum(v['seven_leaf_counts']) and v['whiff_head_eligible']==sum(v['seven_leaf_counts'][3:5]) and v['gb_head_eligible']==sum(v['seven_leaf_counts'][5:7])
 try:cp.project(raw,c,labels10,eh.labels);raise AssertionError('unmapped/quarantine STOP not enforced')
 except ValueError as e:assert str(e).startswith('STOP')
 dup=pd.concat([raw,raw.iloc[[10]].assign(balls=3)],ignore_index=True)
 try:cp.project(dup,cs,labels10,eh.labels);raise AssertionError('non-identical duplicate accepted')
 except ValueError as e:assert 'duplicate' in str(e)
 checks.append('projection')
 # 2 whole-game ordered roles, outcome-blind seeded sample, team-coverage STOP, exhaustive query roles.
 rm=cp.roles(g,c);assert sum(int(v.sum()) for v in rm.values())==len(g)
 for a in rm:
  for b in rm:assert a==b or not set(g.game_pk[rm[a]])&set(g.game_pk[rm[b]])
 sc=dict(c,fit_sample=dict(c['fit_sample'],fraction=.5,min_games_per_team=1));tr,info=cp.sample_games(g,rm['FIT'],sc);tr2,_=cp.sample_games(g.assign(outcome10=0),rm['FIT'],sc)
 assert (tr==tr2).all() and tr.any() and (tr<=rm['FIT']).all() and info['sampled_games']<info['fit_games']
 assert g.loc[tr].groupby('game_pk').size().eq(g.loc[rm['FIT']&g.game_pk.isin(g.game_pk[tr])].groupby('game_pk').size()).all()  # whole games
 try:cp.sample_games(g,rm['FIT'],dict(c,fit_sample=dict(c['fit_sample'],fraction=.01,min_games_per_team=1)));raise AssertionError('coverage STOP missing')
 except ValueError as e:assert 'coverage' in str(e)
 checks.append('roles_sample')
 # 3 features: strict-prior FIT-only history frozen at cutoff; FIT-train-only unweighted encoder; 199 cols.
 query=tr|rm['SELECT']|rm['CAL']|rm['TEST'];q=g.loc[query].reset_index(drop=True);mtr=tr[query]
 joined,_=cp.features(g,rm['FIT'],query,c['history_cutoff'],ci,fsf);x,enc=fsf.encode(joined,mtr)
 assert joined.shape==(len(q),91) and np.isfinite(x).all()
 later=g.copy();late=~rm['FIT'];later.loc[late,'release_speed']=999.;later.loc[late,'description']='hit_into_play';j2,_=cp.features(later,rm['FIT'],query,c['history_cutoff'],ci,fsf)
 assert np.array_equal(np.nan_to_num(joined,nan=-7),np.nan_to_num(j2,nan=-7))  # post-cutoff rows never enter history
 cur=g.copy();cur.loc[query,'release_speed']=cur.loc[query,'release_speed']+50;cur.loc[query&~rm['FIT'],'description']='ball';j3,_=cp.features(cur,rm['FIT'],query,c['history_cutoff'],ci,fsf)
 first_day=(q.game_date==q.game_date.min()).to_numpy()
 assert np.array_equal(np.nan_to_num(joined[first_day],nan=-7),np.nan_to_num(j3[first_day],nan=-7))  # current physics/outcome never a query input
 x_alt,enc_alt=fsf.encode(joined,mtr|(q.game_date>pd.Timestamp(c['history_cutoff'])).to_numpy());assert enc_alt['mean']!=enc['mean']  # encoder statistics depend only on registered TRAIN rows
 g2=feature_gate(c)
 for bad in ({'extra_donor':('plate_x',)},{'extra_project':('zone',)}):
  try:feature_gate(c,**bad);raise AssertionError('coordinate accepted')
  except ValueError:pass
 checks.append('features_encoder_strict_prior')
 # 4 union target, causal C similarity incl. synthetic future-donor exclusion and no-history s=0.
 targets={100,204};tgt=(q.pitcher.isin(targets)|q.batter.isin(targets)).to_numpy();both=int(((q.pitcher==100)&(q.batter==204)).sum())
 assert both>0 and tgt.sum()==(q.pitcher==100).sum()+(q.batter==204).sum()-both  # pitcher OR batter; overlap row counted once
 # C physical similarity: strictly-prior same pitch_type/hand physics; fixed scales; s=0 rules counted.
 sim=c['weights']['similarity'];T={9,8,7,6,5};A,B=(95.,2300.,-.5,1.3),(88.,2000.,.5,.6)
 def row(p,d,t,h,v):return dict(pitcher=p,game_date=pd.Timestamp(d),pitch_type=t,p_throws=h,**dict(zip(PHYSICS,v)))
 P=pd.DataFrame([row(9,'2026-04-01','FF','R',A),row(8,'2026-04-01','FF','R',(90.,2100.,0.,1.)),row(7,'2026-04-05','FF','R',B),row(6,'2026-04-01','SL','R',B),row(5,'2026-04-01','FF','L',B),
  row(1,'2026-04-01','FF','R',A),row(1,'2026-04-01','CU','R',A),row(2,'2026-04-01','FF','R',B),row(3,'2026-04-01','FF','R',A[:3]+(np.nan,)),row(9,'2026-04-10','FF','R',(80.,1500.,1.,0.))])
 Q=pd.DataFrame([row(1,'2026-04-05','FF','R',(0,)*4),row(2,'2026-04-05','FF','R',(0,)*4),row(1,'2026-04-01','FF','R',(0,)*4),row(3,'2026-04-05','FF','R',(0,)*4),row(1,'2026-04-05','CU','R',(0,)*4),row(1,'2026-04-05','UNKNOWN','R',(0,)*4),row(2,'2026-04-15','FF','R',(0,)*4)])
 s,sc=cp.similarity(P,Q,T,sim)
 assert s[0]==1. and 0<s[1]<s[0] and s[2]==s[3]==s[4]==s[5]==0,s  # identical prior profile -> 1 > dissimilar same-type profile > 0
 assert sc=={'rows':7,'positive':3,'s_zero_unknown_pitch_type':1,'s_zero_query_no_prior_same_type_hand':1,'s_zero_query_missing_physics_component':1,'s_zero_no_target_prior_same_type_hand_physics':1},sc
 d8=np.square((np.array(B)-np.array((90.,2100.,0.,1.)))/np.array([sim['scales'][x] for x in PHYSICS])).sum();assert abs(s[1]-np.exp(-.5*d8))<1e-15  # nearest valid donor is 8: same-day 7, SL 6, left-handed 5 excluded
 for bad,fix in ((7,{'game_date':pd.Timestamp('2026-04-04')}),(6,{'pitch_type':'FF'}),(5,{'p_throws':'R'})):
  P2=P.copy();P2.loc[P2.pitcher==bad,list(fix)]=list(fix.values());assert cp.similarity(P2,Q,T,sim)[0][1]==1.,bad  # the exclusion has teeth
 fut=P.copy();fut.loc[fut.game_date>=pd.Timestamp('2026-04-05'),list(PHYSICS)]=(99.,3000.,-2.,2.)  # same-day + future target rows perturbed
 fut=pd.concat([fut,pd.DataFrame([row(9,'2026-04-06','FF','R',(99.,3000.,-2.,2.))])],ignore_index=True);s_f,_=cp.similarity(fut,Q,T,sim)
 early=(Q.game_date<=pd.Timestamp('2026-04-05')).to_numpy();assert np.array_equal(s_f[early],s[early]) and s[6]==1. and s_f[6]<s[6],(s,s_f)  # later FIT rows never move an earlier query; they do move a later one
 assert np.array_equal(cp.arm_raw('C',np.zeros(7,bool),s_f,1,1)[early],cp.arm_raw('C',np.zeros(7,bool),s,1,1)[early])
 Qc=Q.copy();Qc[list(PHYSICS)]=500.;assert np.array_equal(cp.similarity(P,Qc,T,sim)[0],s)  # current-pitch physics is never read
 checks.append('union_target_causal_similarity')
 # 5 weights: alpha=beta=1, raw [1,2], max/min<=2, mean 1 per head TRAIN, zero elsewhere, ESS guard, mass report.
 y10=q.outcome10.to_numpy(np.int64);yh={h:q[h].to_numpy(np.int8) for h in HEADS[1:]};m={'train':mtr,'tuning':rm['SELECT'][query],'calibration':rm['CAL'][query],'evaluation':rm['TEST'][query]}
 sim=np.where(mtr,np.linspace(0,1,len(q)),0)
 for head in HEADS:
  y,hm,parent=head_roles(y10,yh,m,head)
  if head!='parent':assert (hm['train']<=parent).all() and (hm['calibration']==m['calibration']).all()
  for arm in ARMS:
   w,ess=cp.head_weights(arm,tgt,sim,hm['train'],c['weights']);wc=helper_module_path(HERE/'weighted_tree_child.py');wc.check_weights(w,hm['train'])
   raw_w=cp.arm_raw(arm,tgt[hm['train']],sim[hm['train']],1,1);assert raw_w.min()>=1 and raw_w.max()<=2 and ess>=.85
   d=cp.mass(w,hm['train'],tgt,q.pitcher.to_numpy(),q.batter.to_numpy(),q.pitch_type.to_numpy());assert abs(d['target_mass']+d['nontarget_mass']-1)<1e-12 and abs(sum(d['pitch_type'].values())-1)<1e-12
 try:cp.head_weights('B',tgt,sim,m['train'],dict(c['weights'],alpha=3));raise AssertionError('cap not enforced')
 except ValueError as e:assert 'caps' in str(e)
 try:cp.head_weights('B',tgt,sim,m['train'],dict(c['weights'],min_ess_fraction=.999));raise AssertionError('ESS guard not enforced')
 except ValueError as e:assert 'ESS' in str(e)
 checks.append('weights_caps_ess_mass')
 # 6 coherent seven leaves via pinned leaves(), CAL temperatures via pinned helpers, metrics + game-paired path.
 run=helper_module(c,'coherence_run');ld=helper_module(c,'learning_diagnostic');ev=m['evaluation'];n=int(ev.sum());ca=m['calibration']
 margins={a:{'parent':{'calibration':rng.normal(size=(int(ca.sum()),10)),'evaluation':rng.normal(size=(n,10))},'narrow_whiff':{'evaluation':rng.normal(size=n)},'ground_ball_per_pitch':{'evaluation':rng.normal(size=n)}} for a in ARMS}
 T=ld.calibrate_and_score(margins['A']['parent']['calibration'],margins['A']['parent']['evaluation'],y10[ca],y10[ev])[0]['temperature']
 Tb=eh.temperature(np.r_[np.zeros(20),np.ones(20)].astype(int),rng.normal(size=40));assert np.isfinite(T) and np.isfinite(Tb) and T>0 and Tb>0
 temps={a:{'parent':T,'narrow_whiff':Tb,'ground_ball_per_pitch':1.} for a in ARMS}
 k={'y10':y10,'yh':yh,'m':m,'target':tgt,'games':q.game_pk.to_numpy(),'pitcher':q.pitcher.to_numpy(),'batter':q.batter.to_numpy()};res=evaluate(c,k,margins,temps,tmp)
 assert all(res[a]['max_sum_error']<1e-12 for a in ARMS) and set(res['paired'])=={'B-A','C-A'} and 'target' in res['paired']['C-A']
 assert all(res[a]['post_cal_diagnostics'][g_]['rows']==n and sum(x['n'] for x in res[a]['post_cal_diagnostics'][g_]['per_player'])==n for a in ARMS for g_ in ('per_pitcher','per_batter'))
 # post-CAL report-only diagnostics on a tiny known-probability/count case.
 Z=np.zeros((3,7));Z[0,:2]=(.05,.95);Z[1,:2]=(.95,.05);Z[2,:2]=(.5,.5);pd_=post_cal_diagnostics(Z,np.array([1,0,1]),np.array([1,1,2]),np.array([5,6,6]))
 r0=pd_['reliability_seven_leaf']['0'];r2=pd_['reliability_seven_leaf']['2']
 assert [b_['count'] for b_ in r0]==[1,0,0,0,0,1,0,0,0,1] and r0[0]['mean_predicted']==.05 and r0[0]['observed_frequency']==0. and r0[9]['observed_frequency']==1. and r0[1]['mean_predicted'] is None and r0[1]['observed_frequency'] is None,r0
 assert r2[0]['count']==3 and r2[0]['mean_predicted']==0. and all(b_['count']==0 for b_ in r2[1:])
 pp={x['id']:x for x in pd_['per_pitcher']['per_player']};bb={x['id']:x for x in pd_['per_batter']['per_player']}
 assert (pp[1]['n'],pp[2]['n'],bb[5]['n'],bb[6]['n'])==(2,1,1,2) and abs(pp[1]['nll']+np.log(.95))<1e-12 and abs(pp[2]['nll']+np.log(.5))<1e-12 and abs(pp[1]['brier']-.005)<1e-12 and abs(pp[2]['brier']-.5)<1e-12
 assert abs(bb[6]['nll']+(np.log(.95)+np.log(.5))/2)<1e-12 and pd_['per_pitcher']['rows']==pd_['per_batter']['rows']==3 and pd_['per_pitcher']['players_by_n']=={'1':1,'2-9':1,'10-99':0,'100+':0}
 checks.append('seven_leaf_cal_metrics_paired')
 # 7 gates: review missing/stale hash, fresh attempt, stale-claim overlap, helper/source hash failure, inventory.
 try:review_gate(c,'full',tmp/'none.json');raise AssertionError('missing review accepted')
 except SystemExit as e:assert 'BLOCKED' in str(e)
 save(tmp/'review.json',{'verdict':'PASS','distinct_from_author':True,'approved_stages':['audit'],'sources_sha256':{n:sha(HERE/n) for n in OWN}})
 try:review_gate(c,'full',tmp/'review.json');raise AssertionError('unapproved stage accepted')
 except SystemExit as e:assert 'stage' in str(e)
 assert review_gate(c,'audit',tmp/'review.json')
 stale={n:sha(HERE/n) for n in OWN};stale['abc_pilot.py']='0'*64;save(tmp/'review2.json',{'verdict':'PASS','distinct_from_author':True,'approved_stages':['audit','full'],'sources_sha256':stale})
 try:review_gate(c,'full',tmp/'review2.json');raise AssertionError('stale source accepted')
 except SystemExit as e:assert 'hashes' in str(e)
 ids={'audit':'audit-001','full':'full-001'};oc=dict(c,outputs={'attempt_root':str(tmp/'attempts'),'attempt_ids':ids});a1=claim_attempt(oc,'audit')
 assert a1.name=='audit-001' and json.loads((a1/'CLAIM').read_text())['stage']=='audit'
 for o,st in ((oc,'audit'),(oc,'full')):  # reuse of audit-001; full-001 while audit-001 has CLAIM without terminal
  try:claim_attempt(o,st);raise AssertionError('stale/overlapping attempt accepted')
  except SystemExit as e:assert 'BLOCKED' in str(e)
 for bad_ids in ({'audit':'x-001','full':'x-001'},{'audit':'full-001','full':'audit-001'},{'audit':'audit-001'}):
  try:claim_attempt(dict(c,outputs={'attempt_root':str(tmp/'attempts-bad'),'attempt_ids':bad_ids}),'full' if 'full' not in bad_ids else 'audit');raise AssertionError('bad stage attempt id accepted')
  except SystemExit as e:assert 'BLOCKED' in str(e)
 # lock / resource preflight failure -> terminal receipt in the owned attempt; other holder's lock untouched; no retry.
 lp=tmp/'heavy.lock';lp.write_text('');pc=dict(c,lock={'path':str(lp),'mode':'runner'},outputs={'attempt_root':str(tmp/'attempts-pre'),'attempt_ids':ids});holder=lp.open('r');fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
 class FS:
  def __init__(self,fail):self.fail=fail
  def resource(self):
   if self.fail:raise RuntimeError('recorder busy')
   return {'memory_free_percent':50,'swap_used_mib':0}
 att=claim_attempt(pc,'audit')
 try:preflight(pc,att,'audit',FS(False),{});raise AssertionError('busy lock accepted')
 except SystemExit as e:assert 'busy' in str(e)
 t=json.loads((att/'terminal.json').read_text());assert t['phase']=='lock_preflight' and t['success'] is False and t['retry'] is False and not (att/'resource-before.json').exists()
 probe=lp.open('r')
 try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('holder lock was released by the runner')
 except BlockingIOError:pass
 holder.close();att=claim_attempt(pc,'full')
 try:preflight(pc,att,'full',FS(True),{});raise AssertionError('busy recorder accepted')
 except RuntimeError as e:assert 'recorder' in str(e)
 t=json.loads((att/'terminal.json').read_text());assert t['phase']=='resource_preflight' and t['stage']=='full' and t['retry'] is False
 fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);probe.close()  # runner released only its own fd
 try:claim_attempt(pc,'full');raise AssertionError('full-001 reused')
 except SystemExit as e:assert 'exists' in str(e)
 bad=json.loads(json.dumps(c));bad['helpers']['first_small_features'][1]='0'*64;save(tmp/'bad.json',bad)
 try:load(tmp/'bad.json');raise AssertionError('helper hash drift accepted')
 except ValueError:pass
 try:execute(c,'full');raise AssertionError('execute not blocked')
 except SystemExit as e:assert str(e).startswith('BLOCKED')
 assert fit_inventory()['total']==INVENTORY and len(fit_inventory()['fits'])==9
 checks.append('gates_fresh_output_hash_inventory')
 # 8 isolated synthetic LightGBM under the pinned runtime: parent/binary parity with pinned children at w=1, native weights.
 rt=c['runtime'];r=subprocess.run([rt['tree_python'],'-I',str(HERE/'weighted_tree_child.py'),'--synthetic-parity',str(tmp/'parity'),rt['native_path'],rt['native_sha256'],c['helpers']['multiclass_tree_child'][0],c['helpers']['event_tree_child'][0]],capture_output=True,text=True,timeout=600)
 if r.returncode:raise AssertionError('parity child failed: '+r.stderr[-2000:])
 parity=json.loads(r.stdout.strip().splitlines()[-1]);checks.append('lightgbm_parity_native_weights')
 # 9 SELECT before CAL: frozen uncalibrated decision written before any temperature; CAL/TEST changes cannot move it; fail-closed.
 tu=m['tuning'];strike=y10==1;bip=np.isin(y10,BIP);assert (tu&strike).any() and (tu&bip).any() and (tu&tgt).any()
 def mk(seed,base=None):
  r=np.random.default_rng(seed);sub={'parent':None,'narrow_whiff':strike,'ground_ball_per_pitch':bip}
  return {a:{h:{role:(base[a][h][role] if base and role=='tuning' else r.normal(size=(int((m[role]&(sub[h] if role=='tuning' and sub[h] is not None else True)).sum()),)+((10,) if h=='parent' else ()))) for role in ('tuning','calibration','evaluation')} for h in HEADS} for a in ARMS}
 k9={'y10':y10,'yh':yh,'m':m,'target':tgt,'games':q.game_pk.to_numpy(),'keys':q[cp.KEY].to_numpy('<i8')};calls=[]
 def cal(d):
  def f(head,zc,yc,ze,yev):
   assert (d/'SELECT-DECISION.json').is_file();calls.append(head)
   return ld.calibrate_and_score(zc,ze,yc,yev)[0]['temperature'] if head=='parent' else eh.temperature(yc,zc)
  return f
 mg=mk(1);(tmp/'sel1').mkdir();t1,h1=select_then_calibrate(k9,mg,tmp/'sel1',cal(tmp/'sel1'));assert len(calls)==9
 perm=np.flatnonzero(ca)[rng.permutation(int(ca.sum()))];y10c=y10.copy();y10c[ca]=y10[perm];yhc={h:v.copy() for h,v in yh.items()}
 for h in yhc:yhc[h][ca]=yh[h][perm]  # CAL labels changed jointly (still consistent); CAL and TEST margins redrawn
 (tmp/'sel2').mkdir();t2,h2=select_then_calibrate(dict(k9,y10=y10c,yh=yhc),mk(2,mg),tmp/'sel2',cal(tmp/'sel2'))
 d1=json.loads((tmp/'sel1'/'SELECT-DECISION.json').read_text())
 assert h1==h2 and t1!=t2 and len(calls)==18 and d1['advisory_selected'] in d1['safety_eligible'] and d1['adoption'] is False and d1['denominators']['target_rows']==int((tu&tgt).sum())
 for bad_k,bad_m in ((dict(k9,yh=dict(yh,narrow_whiff=np.where(tu&(y10==0),1,yh['narrow_whiff']).astype(np.int8))),mg),(k9,{**mg,'B':{**mg['B'],'narrow_whiff':{**mg['B']['narrow_whiff'],'tuning':mg['B']['narrow_whiff']['tuning'][:-1]}}})):
  d=Path(tempfile.mkdtemp(dir=tmp))
  try:select_then_calibrate(bad_k,bad_m,d,cal(d));raise AssertionError('inconsistent SELECT labels/margins accepted')
  except ValueError as e:assert str(e).startswith('STOP') and len(calls)==18 and not (d/'SELECT-DECISION.json').exists(),e
 d=tmp/'sel3';d.mkdir()
 def tamper(*a):(d/'SELECT-DECISION.json').write_text('{}');return 1.
 try:select_then_calibrate(k9,mg,d,tamper);raise AssertionError('changed SELECT decision accepted')
 except ValueError as e:assert 'changed' in str(e)
 checks.append('select_before_cal_frozen')
 return {'status':'PASS','checks':checks,'synthetic_rows_only':True,'real_rows_read':0,'synthetic_lightgbm':parity,'synthetic_projection_census':census,'scratch':str(tmp)}

# ---------- ONE synthetic lifecycle check: production Signals/Lock/child_fit, temp lock, sleep child, no rows ----------
def lifecycle_runner(spec):
 """Synthetic runner: same Signals -> Lock -> check_stop/child_fit per head -> lock release order as execute(); temp lock only.
 Two heads in sequence so a swallowed stop would show up as a second launched head."""
 s=json.loads(Path(spec).read_text());w=Path(s['dir']);c={'synthetic_lifecycle':True,'lock':{'path':s['lock'],'mode':'runner'},'runtime':{'tree_python':sys.executable,'native_sha256':'synthetic','native_path':'synthetic'},'budgets':{'child_seconds':120,'guard_interval_seconds':60}}
 for n in ('x','r','w'):(w/n).write_bytes(b'synthetic')
 def hook(phase):
  if phase==s['hook']:os.kill(os.getpid(),getattr(signal,s['hook_signal']))
 launched=[];sig=Signals().__enter__();lock=Lock(c).__enter__()
 try:
  for head in ('A-parent','A-narrow_whiff'):
   check_stop();d=w/head;d.mkdir();launched.append(head)
   child_fit(c,d,w/'x',w/'r',w/'w',{},lambda:None,argv=[a.replace('{out}',str(d/'child-output')) for a in s['argv']],hook=hook if s['hook'] else None)
 finally:
  durable(w/'lock-release.json',{'utc':time.time(),'heads_launched':launched,'pending_signals':list(_PENDING)});lock.__exit__();sig.__exit__()

def lifecycle_check(out):
 out=Path(out);tmp=Path(tempfile.mkdtemp(prefix='abc-lifecycle-'));lp=tmp/'synthetic.heavy.lock';lp.write_text('');res={'scratch':str(tmp),'cases':[]}
 sentinel=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(120)'],start_new_session=True);s_os=os_identity(sentinel.pid)
 sleep='import time;time.sleep(120)';ign='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'
 okchild='import sys,time,pathlib;time.sleep(1);o=pathlib.Path(sys.argv[1]);o.mkdir();(o/"receipt.json").write_text("{}")'
 # (name, signal, test-only hook phase or None for an external signal, child code, repeat signal during cleanup, expected child terminal)
 cases=(('spawn_boundary_SIGTERM','SIGTERM','spawned',sleep,False,'TERM'),('normal_cleanup_SIGTERM','SIGTERM','cleanup',okchild,False,'exit'),
  ('poll_SIGINT','SIGINT',None,sleep,False,'TERM'),('poll_SIGHUP_term_ignoring_child','SIGHUP',None,ign,True,'KILL'))
 for name,sg,hook,code,repeat,want in cases:
  w=tmp/name;w.mkdir();d=w/'A-parent';sp=w/'spec.json';save(sp,{'dir':str(w),'lock':str(lp),'argv':[sys.executable,'-I','-c',code,'{out}'],'hook':hook,'hook_signal':sg})
  with (w/'runner.log').open('w') as rl:runner=subprocess.Popen([sys.executable,'-I',str(HERE/'abc_pilot.py'),'--lifecycle-runner',str(sp)],stdout=rl,stderr=subprocess.STDOUT,start_new_session=True)
  t0=time.monotonic();probe=lp.open('r');held_before=None;sent=None;repeats=0;acq=None;child_dead_at_acq=None
  while not (d/'child-identity.json').exists() and time.monotonic()-t0<20:time.sleep(.05)
  ident=json.loads((d/'child-identity.json').read_text());cpgid=ident['child_pgid']
  if hook is None:  # external signal while the runner polls the child under the held lock
   time.sleep(.3)
   try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);held_before=False;fcntl.flock(probe,fcntl.LOCK_UN)
   except BlockingIOError:held_before=True
   os.kill(runner.pid,getattr(signal,sg));sent=time.time()
  rep_due=time.monotonic()+1
  while time.monotonic()-t0<40:
   if repeat and repeats<2 and time.monotonic()>rep_due and runner.poll() is None:os.kill(runner.pid,getattr(signal,sg));repeats+=1;rep_due=time.monotonic()+1
   try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);acq=time.time();child_dead_at_acq=group_gone(cpgid);break
   except BlockingIOError:time.sleep(.02)
  rc=runner.wait(timeout=20)
  if acq is not None:fcntl.flock(probe,fcntl.LOCK_UN)
  probe.close()
  term=json.loads((d/'child-identity.json').read_text())['terminal'];rel=json.loads((w/'lock-release.json').read_text())
  ok=(held_before is not False and (hook is not None or held_before) and acq is not None and child_dead_at_acq and term is not None and term['how']==want and term['group_gone'] is True
   and term['pending_signals'][:1]==[sg] and term['utc']<=rel['utc']<=acq and rel['heads_launched']==['A-parent'] and not (w/'A-narrow_whiff').exists()
   and ident['child_os'].get('ps_rc')==0 and str(cpgid) in ident['child_os']['ps_line'] and rc!=0 and group_gone(cpgid) and group_gone(runner.pid)
   and (want!='exit' or (term['returncode']==0 and (d/'child-output'/'receipt.json').exists())))
  res['cases'].append({'case':name,'signal':sg,'test_hook_phase':hook,'repeated_signals_during_cleanup':repeats,'runner_pid':runner.pid,'runner_pgid':runner.pid,'child_identity_at_popen':ident,'lock_held_before_signal':held_before,'external_signal_sent_utc':sent,'child_terminal':term,'runner_lock_release':rel,
   'second_head_dir_exists':(w/'A-narrow_whiff').exists(),'external_lock_acquired_utc':acq,'child_group_gone_at_acquisition':child_dead_at_acq,'runner_returncode':rc,'runner_log_tail':(w/'runner.log').read_text()[-300:],'ok':bool(ok)})
 res['sentinel']={'pid':sentinel.pid,'os_before':s_os,'alive_after_cases':sentinel.poll() is None,'os_after':os_identity(sentinel.pid)}
 res['no_surviving_test_processes']={str(x['child_identity_at_popen']['child_pgid']):group_gone(x['child_identity_at_popen']['child_pgid']) for x in res['cases']}|{str(x['runner_pid']):group_gone(x['runner_pid']) for x in res['cases']}
 sentinel.kill();sentinel.wait()  # the check's own sentinel only
 res['status']='CHECK-OK (not a review PASS)' if all(x['ok'] for x in res['cases']) and all(res['no_surviving_test_processes'].values()) and res['sentinel']['alive_after_cases'] and res['sentinel']['os_before']['ps_line'].split()[:6]==res['sentinel']['os_after']['ps_line'].split()[:6] else 'CHECK-FAILED'
 res['limitations']=['execute() itself is not driven (its review/runtime/S2 gates block synthetic use); the runner uses the same Signals/Lock/check_stop/child_fit objects in the same order','SIGKILL of the runner is not covered; exact-owned-child-first external emergency supervisor still required','bounded stage functions (projection/features/similarity/weights/temperatures) are not interrupted by a stop; it is consumed at the next checkpoint']
 save(out,res);return res

if __name__=='__main__':
 if sys.argv[1:2]==['--lifecycle-runner']:lifecycle_runner(sys.argv[2]);sys.exit(0)
 if sys.argv[1:2]==['--lifecycle-check']:r=lifecycle_check(sys.argv[2]);print(json.dumps({'status':r['status']}));sys.exit(0 if r['status'].startswith('CHECK-OK') else 1)
 a=argparse.ArgumentParser();a.add_argument('--config',default=str(HERE/'ABC-PILOT-CONFIG.json'));m=a.add_mutually_exclusive_group(required=True)
 m.add_argument('--selfcheck',action='store_true');m.add_argument('--plan',action='store_true');m.add_argument('--execute',action='store_true');a.add_argument('--stage',choices=['audit','full'])
 args=a.parse_args();c=load(args.config)
 if args.selfcheck:print(json.dumps(selfcheck(c),indent=2,default=str))
 elif args.plan:print(json.dumps(plan(c),indent=2))
 else:execute(c,args.stage)
