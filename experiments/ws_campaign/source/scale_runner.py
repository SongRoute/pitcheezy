"""Step 2 fixed FIT-scale comparison: 25% whole-game FIT sample vs 100% FIT games, same 10+2 backbone.

Modes: --plan (registration + source closure gates, no rows), --synthetic (temp data/lock only, the same run()
chain with a numpy stand-in child), --execute (real chain; BLOCKED until a distinct hash-bound REVIEW-PASS names
the fresh attempt id and a reviewed recorder probe is registered; never reads S2 before every gate passes).
Small additive adaptation of research-abc-trace/abc_pilot.py: lock, signals, attempt claim, child lifecycle,
seven-leaf label/metrics and report diagnostics are imported unchanged; projection, roles and the game sample are
abc_corpus unchanged. Runner python: .venv-observer; classifier children: weighted_tree_child.py, one at a time.
"""
import argparse,fcntl,hashlib,importlib.abc,importlib.util,json,os,re,shutil,signal,subprocess,sys,tempfile,threading,time,types
from pathlib import Path
import numpy as np

sys.dont_write_bytecode=True  # loading the pinned helpers must not write __pycache__ into their directories
HERE=Path(__file__).resolve().parent
ARMS=('fit25','fit100');HEADS=('parent','narrow_whiff','ground_ball_per_pitch');BIP=(3,4,5,6,7,9);ROLES=('train','tuning','calibration','evaluation')
OWN=('scale_runner.py','scale_safe.py','synthetic_child.py','SCALE-REGISTRATION.json')
INVENTORY={'classifier':6,'temperature':6}
CAPS={'projection_seconds':600,'feature_seconds':2400,'child_seconds':600,'temperature_seconds':120,'wall_seconds':6000}
T_BOUNDS=(.05,20.)
# Focused rework R1/R2: registered lifecycle bounds (a real registration must carry exactly these).
LIFE={'monitor_interval_seconds':15,'monitor_grace_seconds':20,'child_term_grace_seconds':10,'child_hard_grace_seconds':60,'cleanup_margin_seconds':120}
RECORDER_KEYS={'db','pid','ppid','lstart','health_url'}
ABC_CONFIG=('/Users/song/Projects/pitcheezy-pm-jobs/20261009-ws-batch1-001/research-abc-trace/ABC-PILOT-CONFIG.json','b4b7c6eca43207f57246344bd7f5eae83229c3e940df177e97dbfbb0cf987ad8')
SOURCE_BINDING=('/Users/song/Projects/pitcheezy-pm-jobs/20261009-ws-batch1-001/research-abc-trace/attempts/audit-001/source-binding.json','cf07db6d10da19a99082a0b643c56af7a4d368d4c8ed8166111fe00f33327797')
# The four denied files plus every sibling that is not individually allowed, and heavy runtimes the runner must not load.
DENIED=frozenset({'project_first_small_train','event_heads','learning_diagnostic','coherence','run','run_first_small_m4','candidate_input','batter_history_research','frequency_control','first_small_models','temporal_sharing','support_analysis','matched_additive','event_tree_child','multiclass_tree_child','lightgbm','torch','duckdb'})
IMPORT_ORDER=('first_small_features','history_transfer','outcome_transfer','abc_corpus','abc_pilot','weighted_tree_child')
SELECT_RULE='advisory_selected = arm with the lower uncalibrated SELECT overall seven-leaf NLL on the identical SELECT rows; exact tie -> fit25. Advisory record only: no adoption, no policy, no CAL/EVAL reselection.'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n')

class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in DENIED:raise ImportError('DENIED import: '+name)

def load(path):
 c=json.loads(Path(path).read_text());b=c['budgets'];s=c['fit_sample']
 if list(c['arms'])[:2]!=list(ARMS) or list(c['heads'])!=list(HEADS) or {k:c['inventory'][k] for k in INVENTORY}!=INVENTORY or c['roles']['exposure']!='ALL EXPOSED DEVELOPMENT':raise ValueError('Registered arm/head/inventory/exposure scope')
 if c['history_cutoff']!=c['roles']['FIT'][1] or (s['fraction'],s['seed'],s['unit'])!=(.25,20261109,'game_pk'):raise ValueError('History cutoff / registered sample')
 floors(b)
 if set(c['resource']['recorder'])!=RECORDER_KEYS or set(c['expected_action_support'])<{'when_accepted_rows','unknown_pitch_type_rows_total'} or any(b[k]!=v for k,v in {**CAPS,**LIFE}.items()) or c['features']['dimension']!=199 or c['projection']['count']!=29 or len(c['projection']['columns'])!=29:raise ValueError('Registered caps / 199 features / 29 columns')
 if c['temperature']['bounds_T']!=list(T_BOUNDS) or not c['select_rule'].startswith('advisory_selected = arm with the lower uncalibrated') or 'tie -> fit25' not in c['select_rule']:raise ValueError('Registered temperature bounds / SELECT rule')
 return c

def closure(c):
 """Sha-check every pinned source, load the closure-clean modules by exact path under a deny finder (no directory
 is added to sys.path), then prove no denied name was imported."""
 if not any(isinstance(f,Deny) for f in sys.meta_path):sys.meta_path.insert(0,Deny())
 for name,(p,h,_) in c['pinned'].items():
  if sha(p)!=h:raise SystemExit('BLOCKED: pinned source changed: '+name)
 def module(name,path):
  if name in sys.modules:return sys.modules[name]
  spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
 mods={n:module(n,c['pinned'][n][0]) for n in IMPORT_ORDER};mods['scale_safe']=module('scale_safe',HERE/'scale_safe.py')
 M=types.SimpleNamespace(fsf=mods['first_small_features'],ht=mods['history_transfer'],ot=mods['outcome_transfer'],cp=mods['abc_corpus'],ap=mods['abc_pilot'],wc=mods['weighted_tree_child'],safe=mods['scale_safe'])
 bad=sorted(n for n in sys.modules if n.split('.')[0] in DENIED)
 if bad:raise SystemExit('BLOCKED: denied module imported: '+str(bad))
 for f,o in M.safe.ORIGINS.items():
  hit=[h for p,h,_ in c['pinned'].values() if Path(p).name==Path(f).name]
  if 'sha256' in o and hit!=[o['sha256']]:raise SystemExit('BLOCKED: scale_safe attribution sha differs from pinned: '+f)
 if M.cp.PROJECT!=c['projection']['columns'] or set(c['forbidden_feature_columns'])&set(M.cp.PROJECT+M.safe.CONTEXT+M.safe.DONOR):raise SystemExit('BLOCKED: 29-column projection / forbidden column')
 if {'description','bb_type','outcome10','events',*M.fsf.PHYSICS}&set(M.safe.CONTEXT):raise SystemExit('BLOCKED: realized current-pitch column in query schema')
 if M.wc.MULTICLASS['seed']!=M.wc.BINARY['seed'] or tuple(c['row_policy']['types'])!=M.fsf.TYPES or tuple(c['row_policy']['bb_types'])!=M.safe.BB_TYPES:raise SystemExit('BLOCKED: seed / type vocabulary')
 return M

def source_hashes(c):return {**{n:sha(HERE/n) for n in OWN},**{'pinned:'+n:sha(p) for n,(p,_,_) in c['pinned'].items()}}

def plan(c):
 M=closure(c)
 for p,h in (ABC_CONFIG,SOURCE_BINDING):
  if sha(p)!=h:raise SystemExit('BLOCKED: reviewed ABC binding changed: '+p)
 if json.loads(Path(SOURCE_BINDING[0]).read_text())['columns_read']!=c['projection']['columns']:raise SystemExit('BLOCKED: projection differs from reviewed 29 columns')
 a=json.loads(Path(ABC_CONFIG[0]).read_text())
 if any(c[k].get(f)!=v for k in ('source','runtime','lock') for f,v in a[k].items()) or c['row_policy']['stop_share']!=a['row_policy']['stop_share'] or any(c['budgets'][k]!=a['budgets'][k] for k in CAPS):raise SystemExit('BLOCKED: source/runtime/lock/caps differ from the reviewed ABC config')
 return {'inventory':INVENTORY,'fits':[{'arm':x,'head':h,'classifier':1,'temperature':1} for x in ARMS for h in HEADS],'projection_columns':len(M.cp.PROJECT),'query_columns':M.safe.CONTEXT,'prior_only_donor_columns':sorted(set(M.safe.DONOR)-set(M.safe.CONTEXT)),'classifier_params':{'parent':M.wc.MULTICLASS,'binary_heads':M.wc.BINARY},'source_hashes':source_hashes(c),'denied_modules_loaded':[],'recorder':c['resource']['recorder'],'execute_ready':False}

# ---------- gates ----------
def review_gate(c):
 p=Path(c['review']['path'])
 if not p.exists():raise SystemExit('BLOCKED: distinct hash-bound REVIEW-PASS required at '+str(p))
 r=json.loads(p.read_text())
 if r.get('verdict')!='PASS' or r.get('distinct_from_author') is not True or 'scale' not in r.get('approved_stages',[]) or r.get('attempt_id')!=c['outputs']['attempt_ids']['scale']:raise SystemExit('BLOCKED: review verdict/distinctness/stage/fresh attempt id')
 if r.get('sources_sha256')!=source_hashes(c):raise SystemExit('BLOCKED: reviewed source/registration hashes differ from current files')
 return sha(p)

GIB=1<<30
CAPTURE={'min_free_bytes':14*GIB,'max_run_bytes':7*GIB}  # live Service capture: --min-free-bytes 15032385536, --max-run-bytes 7516192768 (review R7)
FLOORS={'max_tree_rss_bytes':4*GIB,'capture_headroom_bytes':4*GIB,'min_disk_free_launch_bytes':25*GIB,'min_disk_free_bytes':sum(CAPTURE.values())+1}

def floors(b):
 """Real registrations only: finite 4 GiB tree-RSS cap, >= 4 GiB protected capture headroom, runtime disk floor STRICTLY above
 capture min-free + capture possible write (> 21 GiB), launch disk >= 25 GiB and >= output cap + runtime floor."""
 if b['max_tree_rss_bytes']!=FLOORS['max_tree_rss_bytes'] or any(b[k]<FLOORS[k] for k in FLOORS) or b['min_disk_free_launch_bytes']<b['max_output_bytes']+b['min_disk_free_bytes']:raise SystemExit('BLOCKED: registered RSS/headroom/disk floors below the protected minimum')

PS='/bin/ps'  # the synthetic swaps in a temp proxy to exercise the fail-closed listing
OWNED_PGIDS=set()  # classifier child sessions started by this runner whose group is not yet confirmed gone
SHIM='import os,signal,sys;signal.alarm(int(sys.argv[1]));os.execv(sys.argv[2],sys.argv[2:])'  # OS alarm survives exec: finite child hard deadline without a living runner

def table():
 """Fail-closed process table {pid:(ppid,pgid,rss_kib,lstart)} without the listing process itself. A failing ps or a
 listing without this runner's own row raises; it never yields RSS=0 or an empty child list (R3)."""
 try:
  p=subprocess.Popen([PS,'-axo','pid=,ppid=,pgid=,rss=,lstart='],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
  try:out,_=p.communicate(timeout=30)
  except subprocess.TimeoutExpired:p.kill();p.communicate();raise
 except Exception as e:raise RuntimeError('STOP process listing failed (fail closed): '+repr(e))
 rows={}
 for l in out.splitlines():
  f=l.split(None,4)
  if len(f)==5 and f[0].isdigit():rows[int(f[0])]=(int(f[1]),int(f[2]),int(f[3]),' '.join(f[4].split()))
 rows.pop(p.pid,None);me=rows.get(os.getpid())
 if p.returncode or me is None or me[2]<=0:raise RuntimeError(f'STOP process listing failed (fail closed): rc {p.returncode}, own pid listed {me is not None}')
 return rows

def lstarts(pids):
 """{pid: OS start identity} of the live pids among `pids`. This process's own row must come back, so a failing ps is never read as 'gone'."""
 me=os.getpid()
 try:out=subprocess.run([PS,'-o','pid=,lstart=','-p',','.join(str(p) for p in {me,*pids})],capture_output=True,text=True,timeout=10).stdout
 except Exception as e:raise RuntimeError('STOP process listing failed (fail closed): '+repr(e))
 t={int(l.split()[0]):' '.join(l.split()[1:]) for l in out.splitlines() if l.split() and l.split()[0].isdigit()}
 if not t.get(me):raise RuntimeError('STOP process listing failed (fail closed): own pid not listed')
 return {p:t[p] for p in pids if p in t}

def descendants(root,rows=None):
 """Every live descendant pid of root plus every member of a not-yet-confirmed owned child session (a classifier child
 keeps this parent; its own descendants keep its process group after it exits) and the fail-closed table."""
 rows=table() if rows is None else rows;kids={};found=[];stack=[root]
 for p,v in rows.items():kids.setdefault(v[0],[]).append(p)
 while stack:
  for q in kids.get(stack.pop(),[]):
   if q not in found and q!=root:found.append(q);stack.append(q)
 return sorted(set(found)|{p for p,v in rows.items() if v[1] in OWNED_PGIDS and p!=root}),rows

def cleanup_children(term_grace=5.,confirm=15.):
 """Child-first and bounded, before the lock fd is closed: TERM every owned descendant, KILL what is left after the
 grace, reap, and re-list until none remains. Raises (terminal STOP) when the listing fails or disappearance is not
 confirmed. Returns the pid + OS start identity of everything that had to be signalled."""
 mine={};t0=time.monotonic()
 while True:
  rows=table();cur=descendants(os.getpid(),rows)[0];age=time.monotonic()-t0
  if not cur:OWNED_PGIDS.clear();return [{'pid':p,'lstart':s} for p,s in sorted(mine.items())]
  if age>term_grace+confirm:raise RuntimeError('STOP owned descendants not confirmed gone before lock release: '+json.dumps({str(p):rows[p][3] for p in cur}))
  for p in cur:
   if p not in mine or age>term_grace:
    try:os.kill(p,signal.SIGKILL if p in mine else signal.SIGTERM)
    except (ProcessLookupError,PermissionError):pass
   mine.setdefault(p,rows[p][3])
   try:os.waitpid(p,os.WNOHANG)
   except OSError:pass
  time.sleep(.2)

def outsize(att):
 n=0
 for f in att.rglob('*'):
  try:n+=f.stat().st_size if f.is_file() else 0
  except FileNotFoundError:pass  # a durable() temp file replaced between listing and stat
 return n

def child_fit(c,d,x_path,roles_path,w_path,params,guard,stopcheck,lock,argv=None):
 """abc_pilot.child_fit (sha256 a6b4de84..., lines 189-221) with the R1 adaptation only. The child inherits the common
 lock fd (pass_fds), so the kernel keeps the flock for the child's whole life even when this runner is SIGKILLed; it is
 exec'd through SHIM, an OS alarm set before exec (finite hard deadline needing no living runner); child-identity.json
 holds PID, PGID, OS start identity, lock inode and that deadline before the first poll; the TERM grace is registered.
 Nothing here catches SIGKILL: after a dead runner the supervisor must call reap()."""
 ap=sys.modules['abc_pilot'];rt=c['runtime'];b=c['budgets'];wc=c['pinned']['weighted_tree_child'][0]
 q={'out':str(d/'child-output'),'x_path':str(x_path),'x_sha256':sha(x_path),'roles_path':str(roles_path),'roles_sha256':sha(roles_path),'weights_path':str(w_path),'weights_sha256':sha(w_path),'source_sha256':sha(wc),'native_sha256':rt['native_sha256'],'native_path':rt['native_path'],'max_rounds':200,'params':params}
 if argv is not None and c.get('synthetic_lifecycle') is not True:raise SystemExit('BLOCKED: injected child command outside the synthetic lifecycle check')
 hard=int(b['child_seconds']+b['child_hard_grace_seconds']);cmd=[sys.executable,'-I','-S','-c',SHIM,str(hard),*(argv or [rt['tree_python'],'-I',wc,'--request',str(d/'request.json')])]
 stopcheck();save(d/'request.json',q);cs=time.monotonic();last=cs;proc=None;ident=None;gone=None;rounds=0;fd=lock.f.fileno()
 try:
  with (d/'child.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,));OWNED_PGIDS.add(proc.pid)
   ident={'child_pid':proc.pid,'child_pgid':proc.pid,'child_lstart':None,'child_os':ap.os_identity(proc.pid),'runner_pid':os.getpid(),'runner_os':ap.os_identity(os.getpid()),'attempt':d.parent.name,'head':d.name,'popen_utc':time.time(),'hard_deadline_seconds':hard,'hard_deadline_utc':time.time()+hard,'lock_fd_inherited':fd,'lock_inode':os.fstat(fd).st_ino,'terminal':None}
   ident['child_lstart']=lstarts([proc.pid]).get(proc.pid);ap.durable(d/'child-identity.json',ident)
   if not ident['child_lstart']:raise RuntimeError('STOP child OS start identity not established')
   while proc.poll() is None:
    stopcheck()  # <=0.5 s stop latency while the child runs
    if time.monotonic()-cs>b['child_seconds']:raise TimeoutError('Classifier time cap')
    if time.monotonic()-last>b['guard_interval_seconds']:guard();last=time.monotonic()
    time.sleep(.5)
  if proc.returncode:raise RuntimeError('Classifier failed '+str(proc.returncode))
 finally:  # only the exact owned child PGID is signalled; TERM -> registered grace -> KILL, then bounded group confirmation
  how='exit'
  if proc is not None and proc.poll() is None:
   how='TERM';os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=b['child_term_grace_seconds'])
   except subprocess.TimeoutExpired:how='KILL';os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  if proc is not None:
   for _ in range(25):
    gone=ap.group_gone(proc.pid)
    if gone:OWNED_PGIDS.discard(proc.pid);break
    rounds+=1
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    time.sleep(.2)
  if ident is not None:ap.durable(d/'child-identity.json',dict(ident,terminal={'how':how,'returncode':proc.returncode,'group_gone':gone,'group_kill_rounds_after_leader_exit':rounds,'utc':time.time(),'pending_signals':list(ap._PENDING)}))
 if not gone:raise RuntimeError('STOP child group not confirmed gone')
 stopcheck()
 return json.loads((d/'child-output'/'receipt.json').read_text())

def reap(att,term_grace=10.,confirm=15.):
 """R1 supervisor side, for a runner that died without unwinding (SIGKILL/OOM/group kill). For every child-identity.json
 without a terminal: signal ONLY the exact owned child (same PID and OS start identity) through its own process group,
 TERM then KILL, and confirm the group is gone. A PID whose start identity differs is never signalled (fail closed, ok
 false if its group id is live). Writes reap-receipt.json and, when the attempt has none, a terminal STOP. The lock is not
 touched: the kernel drops the inherited flock when the last owned holder is gone."""
 att=Path(att);kids=[]
 def gone(pgid):
  try:os.killpg(pgid,0);return False
  except ProcessLookupError:return True
 for p in sorted(att.glob('*/child-identity.json')):
  i=json.loads(p.read_text())
  if i.get('terminal'):continue
  pid,pgid=i['child_pid'],i['child_pgid'];same=lambda:bool(i['child_lstart']) and lstarts([pid]).get(pid)==i['child_lstart'];how='leader_not_running';t0=time.monotonic()
  if same():
   how='TERM'
   while not gone(pgid) and time.monotonic()-t0<term_grace+confirm:
    try:os.killpg(pgid,signal.SIGKILL if how=='KILL' else signal.SIGTERM)
    except ProcessLookupError:pass
    time.sleep(.2)
    if time.monotonic()-t0>term_grace and not gone(pgid):how='KILL'
  kids.append({'head':p.parent.name,'child_pid':pid,'child_pgid':pgid,'child_lstart':i['child_lstart'],'how':how,'exact_child_gone':not same(),'group_gone':gone(pgid),'seconds':time.monotonic()-t0})
 ok=all(k['exact_child_gone'] and k['group_gone'] for k in kids);rec={'utc':time.time(),'attempt':att.name,'ok':ok,'children':kids,'reaper_pid':os.getpid(),'note':'only exact PID + OS start identity matches were signalled; lock not touched by the reaper'}
 save(att/'reap-receipt.json',rec)
 if not (att/'terminal.json').exists():save(att/'terminal.json',{'success':False,'phase':'supervisor_reap','error':'runner ended without a terminal; owned children handled by reap()','children_confirmed_gone':ok,'retry':False,'inference_prohibited':True})
 return rec

class Monitor(threading.Thread):
 """R2: one job-owned sampler thread for one attempt (not a scheduler). Every second: pending signals and the internal /
 external deadlines. Every monitor_interval_seconds (<=15): output cap and resource() (recorder/capture priority, tree
 RSS, headroom, disk, swap, health), through every phase. A failure is first offered to the main thread (stopcheck at
 its safe points). If the main thread has not started unwinding within monitor_grace_seconds (blocked projection /
 feature / encoder phase), the monitor runs the child-first confirmed cleanup, writes the durable terminal STOP and
 os._exit(70)s; only that exit drops this runner's lock fd. It signals owned descendants only; the Service capture and
 the recorder are read, never signalled."""
 def __init__(self,c,att,before,t0,ext,state,counts,ap):
  super().__init__(daemon=True);self.c,self.att,self.before,self.t0,self.ext,self.state,self.counts,self.ap=c,att,before,t0,ext,state,counts,ap
  self.why=None;self.done=threading.Event();self.unwinding=threading.Event()
 def run(self):
  b=self.c['budgets'];nxt=0.
  while not self.done.wait(1.):
   why=None
   try:
    if self.ap._PENDING:why='signal '+self.ap._PENDING[0]
    elif time.monotonic()-self.t0>b['wall_seconds']:why='Overall internal wall cap'
    elif time.time()>self.ext-b['cleanup_margin_seconds']:why='External supervisor deadline minus cleanup margin'
    elif time.monotonic()>=nxt:
     nxt=time.monotonic()+b['monitor_interval_seconds']
     if outsize(self.att)>b['max_output_bytes']:why='Output cap'
     else:
      r=resource(self.c,self.before)
      with (self.att/'monitor.jsonl').open('a') as f:f.write(json.dumps({'t':time.monotonic()-self.t0,'phase':self.state['phase'],**{k:r[k] for k in ('tree_rss_bytes','tree_children','headroom_free_inactive_speculative_bytes','disk_free_bytes','recorder_busy','memory_free_percent')}})+'\n')
   except BaseException as e:why=repr(e)
   if why is None or self.done.is_set():continue
   self.why=why
   if self.unwinding.wait(b['monitor_grace_seconds']):return  # the main thread owns terminal + cleanup
   self.hard(why)
 def hard(self,why):
  killed=err=None
  try:killed=cleanup_children(self.c['budgets']['child_term_grace_seconds'])
  except BaseException as e:err=repr(e)
  try:self.ap.durable(self.att/'terminal.json',{'success':False,'counts':self.counts,'phase':self.state['phase'],'error':'STOP monitor hard stop (main thread blocked): '+why,'cleanup':{'children_signalled':killed,'error':err,'confirmed_gone':err is None},'lock':'runner fd dropped by process exit after this cleanup; an unconfirmed child would still hold the inherited flock','pending_signals':list(self.ap._PENDING),'elapsed_seconds':time.monotonic()-self.t0,'retry':False,'inference_prohibited':True})
  finally:os._exit(70)

def resource(c,before=None):
 """Recorder identity, recorder queue/lease, memory pressure, swap and service health: the probe logic of
 run_first_small_m4.py (sha256 a342eac9531c344291abc652dc46f83b4c80a429f66bb456db9867e1dd60f3bd) lines 113-138, copied
 with the registered resource.recorder constants in place of its module globals (that module is never imported).
 Added: sampled aggregate process-tree RSS, free+inactive+speculative headroom (purgeable not counted) and disk.
 before=None is the launch gate. Any failure raises; the caller writes a terminal STOP and never retries."""
 import contextlib,sqlite3,urllib.request
 rc=c['resource']['recorder'];b=c['budgets']
 f=subprocess.check_output([PS,'-p',str(rc['pid']),'-o','ppid=','-o','lstart=','-o','command='],text=True).split(None,6)  # R5: pid + parent + OS start + exact command tail
 if len(f)!=7 or int(f[0])!=rc['ppid'] or ' '.join(f[1:6])!=' '.join(rc['lstart'].split()) or not f[6].strip().endswith('-m observer_app.worker --database '+rc['db']):raise RuntimeError('Recorder process identity changed (pid/parent/OS start/command): fresh review required')
 with contextlib.closing(sqlite3.connect(Path(rc['db']).as_uri()+'?mode=ro',uri=True,timeout=.2)) as db:  # live read-only connection sees WAL
  db.execute('PRAGMA query_only=ON');busy=int(db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running') OR lease_until > ?",(time.time(),)).fetchone()[0])
 if busy:raise RuntimeError('Recorder queue or lease active: service capture keeps priority')
 free=int(re.search(r'free percentage: (\d+)%',subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True)).group(1))
 used=float(re.search(r'used = ([\d.]+)M',subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)).group(1))
 start=time.monotonic()
 with urllib.request.urlopen(rc['health_url'],timeout=3) as response:
  if response.status!=200:raise ValueError('Service health failed')
  health=json.load(response)
 hs=time.monotonic()-start
 if health.get('model_ready') is not True or health.get('dataset_ready') is not True:raise RuntimeError('Service readiness unavailable')
 vm=subprocess.check_output(['/usr/bin/vm_stat'],text=True);page=int(re.search(r'page size of (\d+) bytes',vm).group(1))
 head=page*sum(int(re.search(rf'Pages {x}:\s+(\d+)\.',vm).group(1)) for x in ('free','inactive','speculative'))
 kids,rows=descendants(os.getpid());tree=1024*sum(rows[p][2] for p in (os.getpid(),*kids))  # table() raised if ps failed or own pid is missing
 root=Path(c['outputs']['attempt_root']);disk=shutil.disk_usage(root if root.exists() else root.parent).free
 need=b['capture_headroom_bytes']+(b['max_tree_rss_bytes'] if before is None else max(0,b['max_tree_rss_bytes']-tree));floor=b['min_disk_free_launch_bytes'] if before is None else b['min_disk_free_bytes']
 r={'utc':time.time(),'phase':'launch' if before is None else 'runtime','memory_free_percent':free,'swap_used_mib':used,'health_seconds':hs,'recorder_busy':busy,'headroom_free_inactive_speculative_bytes':head,'headroom_required_bytes':need,'tree_rss_bytes':tree,'tree_children':len(kids),'disk_free_bytes':disk,'disk_floor_bytes':floor}
 stops=[t for ok,t in ((free>=b['min_memory_free_percent'],'memory free percent'),(tree<=b['max_tree_rss_bytes'],'aggregate process-tree RSS over cap'),(head>=need,'memory headroom below RSS cap + protected capture'),(disk>=floor,'disk free below protected floor'),(hs<=b['max_health_seconds'],'service health latency'),(before is None or used-before['swap_used_mib']<=b['max_swap_delta_mib'],'swap growth')) if not ok]
 if stops:raise RuntimeError('STOP resource gate: '+'; '.join(stops)+' '+json.dumps(r))
 return r

# ---------- population, features, arms ----------
def core_guard(g,M):
 """Both arms draw only from this frame: every accepted row must be a core leaf that equals its parent-10 + head label."""
 fate=M.safe.core_fate(g);bad=~np.isin(fate,M.safe.CORE)
 if bad.any():raise ValueError('STOP unsupported label in the core population: '+json.dumps({str(k):int(v) for k,v in zip(*np.unique(fate[bad],return_counts=True))}))
 y7=M.ap.seven_label(g.outcome10.to_numpy(),g.narrow_whiff.to_numpy(),g.ground_ball_per_pitch.to_numpy())
 if not np.array_equal(y7,np.array([int(x[-1]) for x in fate])):raise ValueError('STOP core leaf differs from parent-10 + head label')
 return y7

def project(c,raw,M):
 """abc_corpus projection/roles/sample unchanged + exclusive core-fate census. Returns frame, census, roles, sample."""
 pc=dict(c,roles=dict(c['roles'],TEST=c['roles']['EVAL']));src=M.safe.core_fate(raw)
 g,census=M.cp.project(raw,pc,M.safe.outcome_labels,M.safe.head_labels);y7=core_guard(g,M);acc=M.safe.core_fate(g)
 fates=M.safe.CORE+M.safe.EXCLUDED_FATES;s={k:int((src==k).sum()) for k in fates};a={k:int((acc==k).sum()) for k in fates}
 if sum(s.values())!=len(raw) or sum(a.values())!=len(g):raise ValueError('STOP core fate reconciliation')
 census['core_fate_census']={'source_rows_by_fate':s,'accepted_rows_by_fate':a,'not_accepted_by_fate':{k:s[k]-a[k] for k in fates},'scope':'one fate per source row; LEAF rows not accepted were removed by a structural/state/handedness old fate; excluded fates never get a parent class'}
 rm=M.cp.roles(g,pc)
 if not np.all(sum(v.astype(int) for v in rm.values())==1):raise ValueError('STOP accepted row without exactly one role')
 sample,info=M.cp.sample_games(g,rm['FIT'],pc)
 census['role_rows']={r:int(v.sum()) for r,v in rm.items()};census['seven_leaf_counts_by_role']={r:np.bincount(y7[v],minlength=7).tolist() for r,v in rm.items()}
 return g,census,rm,sample,info,y7

def features(pool,rows,cutoff,M):
 """abc_corpus.features operations (lines 121-132) with the safe copies: query_context/schema_frame, strictly-prior
 richer_physics, transfer_features, history_features. Only SCHEMA columns of `rows` survive query_context."""
 import pandas as pd
 sf=M.safe;q=sf.query_context(rows.reset_index(drop=True))
 if list(pool.columns)!=sf.DONOR or pool.game_date.max()>pd.Timestamp(cutoff):raise ValueError('Frozen donor allowlist/cutoff')
 f,raw,names=sf.schema_frame(q)
 raw=M.ht.richer_physics(raw,names,f,pool[M.ht.POOL_COLUMNS]);arrays,audit=M.ot.transfer_features(f,pool[M.ht.POOL_COLUMNS+M.ot.EXTRA],M.ot.SETTINGS)
 joined=np.column_stack([raw,arrays['generic']]);full,*_=sf.history_features(f,pool[sf.HISTORY_COLUMNS],cutoff);joined[:,59:66]=full
 return joined,audit

def leak_check(g,fit,cutoff,M,feat=features):
 """Rewrite every realized outcome/physics value on and after a mid-FIT date; features of rows dated <= that date
 must not move (strictly-prior, same day excluded), while later rows must move (the check is sensitive)."""
 dates=np.sort(g.game_date[fit].unique());D=dates[len(dates)//2];donor=lambda x:x.loc[fit,M.safe.DONOR].reset_index(drop=True)
 base,_=feat(donor(g),g,cutoff,M);h=g.copy();late=(h.game_date>=D).to_numpy()
 for c in M.fsf.PHYSICS:h.loc[late,c]=h.loc[late,c]*1.5+3
 h.loc[late,'description']='hit_into_play';h.loc[late,'bb_type']='ground_ball';h.loc[late,'outcome10']=3
 alt,_=feat(donor(h),h,cutoff,M);early=(g.game_date<=D).to_numpy()
 if not np.array_equal(base[early],alt[early],equal_nan=True):raise ValueError('STOP future feature leakage: realized same-day/future values changed a query feature')
 if np.array_equal(base[~early],alt[~early],equal_nan=True):raise ValueError('STOP leak check insensitive')
 return {'pivot_date':str(D)[:10],'rows_checked_unchanged':int(early.sum()),'later_rows_changed':True}

def encoder_guard(enc_rows,m25):
 if not np.array_equal(enc_rows,m25['train']) or any((enc_rows&m25[r]).any() for r in ROLES[1:]):raise ValueError('STOP encoder fit rows are not exactly the sampled 25% FIT rows (SELECT/CAL/EVAL or unsampled row present)')

def assert_expected(observed,expected):
 bad={k:{'expected':expected[k],'observed':observed[k]} for k in observed if k in expected and expected[k]!=observed[k]}
 if bad:raise ValueError('STOP expected real-run assertion mismatch: '+json.dumps(bad))

def action_support(g,rm,sample,census):
 """R4: known vs UNKNOWN pitch_type (the action label) per role and per arm on the label-supported core rows. UNKNOWN
 rows stay in every denominator and every fit. This is label eligibility, NOT action or service coverage."""
 unk=(g.pitch_type=='UNKNOWN').to_numpy();src=census['pitch_type_mapped_unknown'];four=('FIT100_TRAIN','SELECT','CAL','EVAL')
 cell=lambda v:{'label_supported_rows':int(v.sum()),'known_pitch_type_rows':int((v&~unk).sum()),'unknown_pitch_type_rows':int((v&unk).sum())}
 roles={'FIT100_TRAIN':cell(rm['FIT']),'FIT25_TRAIN':cell(sample),'SELECT':cell(rm['SELECT']),'CAL':cell(rm['CAL']),'EVAL':cell(rm['TEST'])};total=cell(np.ones(len(g),bool))
 if sum(roles[r]['label_supported_rows'] for r in four)!=len(g) or sum(roles[r]['unknown_pitch_type_rows'] for r in four)!=total['unknown_pitch_type_rows'] or sum(src.values())!=total['unknown_pitch_type_rows'] or (sample&~rm['FIT']).any():raise ValueError('STOP action-support denominators do not reconcile to the label-supported rows')
 shared={r:roles[r] for r in four[1:]}
 return {'all_label_supported_rows':total,'by_role':roles,'by_arm':{'fit25':dict(TRAIN=roles['FIT25_TRAIN'],**shared),'fit100':dict(TRAIN=roles['FIT100_TRAIN'],**shared)},'unknown_by_source_literal':src,'scope':'label-supported core rows only; UNKNOWN = pitch_type null or outside the registered vocabulary, kept and fit as its own category; not a service/action coverage claim'}

def expected_action(act,accepted,e):
 """Assert the earlier accepted UNKNOWN total only where the old population definition matches exactly (same accepted-row count)."""
 same=accepted==e['when_accepted_rows'];got=act['all_label_supported_rows']['unknown_pitch_type_rows']
 if same and got!=e['unknown_pitch_type_rows_total']:raise ValueError('STOP expected real-run assertion mismatch: UNKNOWN pitch_type rows '+json.dumps({'expected':e['unknown_pitch_type_rows_total'],'observed':got}))
 return {'population_definition_matches':same,'asserted':same,'expected_total':e['unknown_pitch_type_rows_total'],'observed_total':got}

def arm_contract(k):
 """Before any fit: the arms differ ONLY in FIT training rows. Same encoder, features, seed/parameters, labels,
 SELECT/CAL/EVAL rows; fit25 TRAIN = encoder rows = the sample; every row is core-supported."""
 a,b=(k['arms'][x] for x in ARMS);x25=np.load(a['x'],mmap_mode='r');x100=np.load(b['x'],mmap_mode='r');n=len(k['y7'])
 if x25.shape!=(len(a['idx']),199) or x100.shape!=(n,199) or not np.array_equal(b['idx'],np.arange(n)):raise ValueError(f'STOP 199-feature contract: {x25.shape} {x100.shape}')
 if not np.array_equal(x25,x100[a['idx']]) or a['encoder_sha256']!=b['encoder_sha256'] or a['params']!=b['params']:raise ValueError('STOP unequal encoder/features/seed/parameters across arms')
 for x in (a,b):
  if not np.all(sum(x['m'][r].astype(int) for r in ROLES)==1):raise ValueError('STOP arm roles not exhaustive/disjoint')
  for r in ROLES[1:]:
   if not np.array_equal(x['idx'][x['m'][r]],np.flatnonzero(k['role'][r])):raise ValueError('STOP role/split mismatch: '+r+' rows differ from the registered role or between arms')
  for h in HEADS:
   w=np.load(x['paths'][h]['weights']);z=np.load(x['paths'][h]['roles'])
   if not np.array_equal(w,z['train'].astype(np.float64)) or (h=='parent' and not np.array_equal(z['train'],x['m']['train'])):raise ValueError('STOP non-uniform weight or head TRAIN mask')
 t25=a['idx'][a['m']['train']];t100=b['idx'][b['m']['train']]
 if not np.array_equal(t100,np.flatnonzero(k['fit'])) or not np.array_equal(t25,np.flatnonzero(k['sample'])) or not np.isin(t25,t100).all() or (k['sample']&~k['fit']).any():raise ValueError('STOP role/split mismatch: TRAIN rows are not the registered FIT / sampled FIT rows')
 if not np.array_equal(np.flatnonzero(k['encoder_rows']),t25):raise ValueError('STOP encoder fit rows differ from fit25 TRAIN (CAL/EVAL or extra FIT row in encoder fit)')
 if not k['supported'].all() or k['y7'].min()<0 or k['y7'].max()>6:raise ValueError('STOP unsupported label entering an arm')
 dg=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
 return {'encoder_sha256':a['encoder_sha256'],'features':199,'params_sha256':dg(np.frombuffer(json.dumps(a['params'],sort_keys=True).encode(),np.uint8)),'seed':a['params']['parent']['seed'],'train_rows':{'fit25':int(len(t25)),'fit100':int(len(t100))},'role_keys_sha256':{r:dg(k['keys'][k['role'][r]]) for r in ROLES[1:]},'x_sha256':{'fit25':sha(a['x']),'fit100':sha(b['x'])},'fit25_x_is_row_subset_of_fit100':True}

ENCODE_CHUNK_ROWS=16384  # scale-002 memory rework: rows per frozen-transform slice (about 26 MiB float64 + 13 MiB float32 at 199 columns)

def encode_save(joined,enc,fitx,enc_rows,targets,apply,chunk=ENCODE_CHUNK_ROWS):
 """scale-002 memory rework. The unchanged scale_safe.apply_encoder on bounded row-order slices of `joined`; each
 float32 slice is appended to every (path,row mask) target that selects it. File bytes equal
 np.save(path,apply(joined,enc)[np.flatnonzero(mask)]) (same v1.0 header, values, row order); the full float64
 intermediates and the full encoded matrix are never held. Same STOPs as before: 199 columns, finite, and the frozen
 transform of the encoder-fit rows equal to the encoder's own fit transform. A STOP can leave partial x files in
 the attempt; the terminal STOP covers them."""
 n=len(joined);bad=pos=0;files=[]
 if chunk<1 or len(enc_rows)!=n or any(len(v)!=n for _,v in targets):raise ValueError('STOP encoder slice/mask length')
 try:
  for p,v in targets:
   f=Path(p).open('wb');files.append(f);np.lib.format.write_array_header_1_0(f,{'descr':'<f4','fortran_order':False,'shape':(int(v.sum()),199)})
  for a in range(0,n,chunk):
   z=apply(joined[a:a+chunk],enc);e=enc_rows[a:a+chunk];k=int(e.sum())
   if z.shape!=(len(e),199) or z.dtype!=np.float32:raise ValueError(f'STOP common encoder cannot encode all rows under the 199-feature contract: shape {(n,)+z.shape[1:]} {z.dtype}, raw {joined.shape}, continuous {len(enc["continuous"])}')
   if not np.isfinite(z).all():bad+=int((~np.isfinite(z)).sum())
   elif not np.array_equal(z[e],fitx[pos:pos+k]):raise ValueError('STOP frozen transform differs from the encoder fit transform')
   pos+=k
   for f,(_,v) in zip(files,targets):f.write(np.ascontiguousarray(z[v[a:a+chunk]]).tobytes())
 finally:
  for f in files:f.close()
 if bad:raise ValueError(f'STOP common encoder cannot encode all rows under the 199-feature contract: shape {(n,199)}, raw {joined.shape}, continuous {len(enc["continuous"])}, nonfinite {bad}')
 if pos!=len(fitx):raise ValueError('STOP frozen transform differs from the encoder fit transform')

def build_corpus(c,raw,out,M,deadline,state):
 """One shared projection, feature matrix and FIT25-fit encoder; both arms' role/weight files; no fit.
 scale-002 memory rework: `raw` may be a one-item list; the source frame is then released right after projection."""
 cp,ap,sf=M.cp,M.ap,M.safe;box=raw if isinstance(raw,list) else [raw];raw=box.pop()
 try:g,census,rm,sample,info,y7=project(c,raw,M)
 except cp.ProjectionStop as e:save(out/'projection.json',e.census);raise
 del raw
 save(out/'projection.json',census);ap.check_stop()
 observed={'accepted_rows':len(g),'role_rows':{('EVAL' if r=='TEST' else r):int(v.sum()) for r,v in rm.items()},'fit_games':info['fit_games'],'sampled_games':info['sampled_games'],'fit25_train_rows':int(sample.sum()),'fit100_train_rows':int(rm['FIT'].sum()),'sampled_game_keys_sha256':info['sampled_game_keys_sha256'],'seven_leaf_counts_by_role':{('EVAL' if r=='TEST' else r):v for r,v in census['seven_leaf_counts_by_role'].items()}}
 act=action_support(g,rm,sample,census);act['expected']=expected_action(act,len(g),c['expected_action_support'])
 save(out/'denominators.json',dict(observed,action_support=act,source_rows=census['source_rows'],fit_sample=info,**{r+'_games':int(g.game_pk[rm[r]].nunique()) for r in rm},note='fit25 TRAIN is the outcome-blind 25% whole-game sample; the FULL FIT pool is the strictly-prior history donor pool for both arms'))
 assert_expected(observed,c['expected_real_run_assertions'])
 if time.monotonic()>deadline:raise TimeoutError('Projection cap')
 ap.check_stop();ap.alarm(c['budgets']['feature_seconds']);state['phase']='features'
 if c.get('synthetic_lifecycle') is True:time.sleep(c.get('synthetic_slow_feature_seconds',0))  # temp-only blocked-phase stand-in for the R2 monitor check
 joined,audit=features(g.loc[rm['FIT'],sf.DONOR].reset_index(drop=True),g,c['history_cutoff'],M);signal.alarm(0);ap.check_stop();state['phase']='encoder_save'
 m={'fit25':{'train':sample,'tuning':rm['SELECT'],'calibration':rm['CAL'],'evaluation':rm['TEST']}};m['fit100']=dict(m['fit25'],train=rm['FIT'])
 enc_rows=sample.copy();encoder_guard(enc_rows,m['fit25'])
 fitx,enc=M.fsf.encode(joined[enc_rows],np.ones(int(enc_rows.sum()),bool))  # only sampled 25% FIT rows are ever passed to the fit
 if len(joined)!=len(g):raise ValueError(f'STOP common encoder cannot encode all rows under the 199-feature contract: raw {joined.shape}, rows {len(g)}')
 am={arm:sum(v.astype(int) for v in m[arm].values())==1 for arm in ARMS}
 encode_save(joined,enc,fitx,enc_rows,[(out/f'x-{arm}.npy',am[arm]) for arm in ARMS],sf.apply_encoder)  # frozen statistics applied to every row of both arms, slice by slice
 del joined,fitx
 save(out/'encoder.json',enc);esha=sha(out/'encoder.json')
 y10=g.outcome10.to_numpy(np.int64);yh={h:g[h].to_numpy(np.int8) for h in HEADS[1:]};params={'parent':M.wc.MULTICLASS,'narrow_whiff':M.wc.BINARY,'ground_ball_per_pitch':M.wc.BINARY};arms={}
 for arm in ARMS:
  ap.check_stop();idx=np.flatnonzero(am[arm]);rmask={r:v[idx] for r,v in m[arm].items()};xp=out/f'x-{arm}.npy';paths={}
  for head in HEADS:
   y,hm,_=ap.head_roles(y10[idx],{h:v[idx] for h,v in yh.items()},rmask,head);rp=out/f'roles-{arm}-{head}.npz';np.savez(rp,y=y,**hm)
   wp=out/f'weights-{arm}-{head}.npy';np.save(wp,hm['train'].astype(np.float64),allow_pickle=False);paths[head]={'roles':rp,'weights':wp}
  arms[arm]={'idx':idx,'m':rmask,'x':xp,'paths':paths,'encoder_sha256':esha,'params':json.loads(json.dumps(params))}
 keys=g[cp.KEY].to_numpy('<i8');np.savez(out/'corpus.npz',y10=y10,y7=y7,keys=keys,games=g.game_pk.to_numpy(),fit=rm['FIT'],sample=sample,**{h:yh[h] for h in yh},**{r:m['fit25'][r] for r in ROLES[1:]})
 k={'y10':y10,'yh':yh,'y7':y7,'supported':np.isin(sf.core_fate(g),sf.CORE),'keys':keys,'games':g.game_pk.to_numpy(),'pitcher':g.pitcher.to_numpy(),'batter':g.batter.to_numpy(),'fit':rm['FIT'],'sample':sample,'encoder_rows':enc_rows,'role':{r:m['fit25'][r] for r in ROLES[1:]},'arms':arms,'unknown_type':(g.pitch_type=='UNKNOWN').to_numpy(),'action_support':act}
 save(out/'arm-contract.json',dict(arm_contract(k),transfer_audit=audit,pitch_type_mapped_unknown=census['pitch_type_mapped_unknown']));return k

# ---------- SELECT, temperature, report ----------
def select_decision(k,margins):
 """Uncalibrated true-leaf seven-leaf NLL on the identical SELECT rows of both arms. Reads only SELECT margins/labels."""
 from scipy.special import softmax,expit
 tu=k['role']['tuning'];y10=k['y10'][tu];wf=k['yh']['narrow_whiff'][tu];gb=k['yh']['ground_ball_per_pitch'][tu];y7=k['y7'][tu];games=k['games'][tu];strike=y10==1;bip=np.isin(y10,BIP)
 if not tu.any() or ((wf==1)&~strike).any() or ((gb==1)&~bip).any():raise ValueError('STOP SELECT rows/labels')
 arms={};losses={'games':games,'y7':y7}
 for arm in ARMS:
  zp,zw,zg=(margins[arm][h]['tuning'] for h in HEADS)
  if zp.shape!=(len(y10),10) or zw.shape!=(int(strike.sum()),) or zg.shape!=(int(bip.sum()),) or not (np.isfinite(zp).all() and np.isfinite(zw).all() and np.isfinite(zg).all()):raise ValueError(f'STOP SELECT {arm} tuning margins do not match parent/head label availability')
  p=softmax(zp,axis=1);q=np.full(len(y10),np.nan)
  for leaf,cls in ((0,0),(1,2),(2,8)):q[y7==leaf]=p[y7==leaf,cls]
  w=expit(zw);q[strike]=p[strike,1]*np.where(wf[strike]==1,w,1-w);r=expit(zg);q[bip]=p[bip][:,list(BIP)].sum(1)*np.where(gb[bip]==1,r,1-r)
  if not np.isfinite(q).all():raise ValueError('STOP SELECT true-leaf coverage')
  losses[arm]=-np.log(np.clip(q,1e-15,1));arms[arm]={'overall_nll':float(losses[arm].mean())}
 dec={'stage':'SELECT uncalibrated (raw margins, T=1), frozen before any CAL temperature is fit or read','primary':'overall seven-leaf NLL','arms':arms,'advisory_selected':min(ARMS,key=lambda a:(arms[a]['overall_nll'],ARMS.index(a))),'rule':SELECT_RULE,'adoption':False,'automatic_policy':False,'cal_or_eval_read':False,
  'denominators':{'rows':int(tu.sum()),'games':int(len(np.unique(games))),'head_tuning_rows':{'parent':int(tu.sum()),'narrow_whiff':int(strike.sum()),'ground_ball_per_pitch':int(bip.sum())},'leaf_counts':np.bincount(y7,minlength=7).tolist()}}
 return dec,losses

def temperature(z,y):
 """ONE scalar temperature for every arm/head: minimise mean NLL of softmax(z/T) or sigmoid(z/T) over log T in registered bounds."""
 from scipy.optimize import minimize_scalar
 from scipy.special import logsumexp
 z=np.asarray(z,np.float64);y=np.asarray(y);lo,hi=np.log(T_BOUNDS)
 if len(z)==0 or len(z)!=len(y) or not np.isfinite(z).all():raise ValueError('STOP temperature input')
 if z.ndim==2:f=lambda u:float(np.mean(logsumexp(z/np.exp(u),axis=1)-z[np.arange(len(y)),y]/np.exp(u)))
 else:f=lambda u:float(np.mean(np.logaddexp(0,z/np.exp(u))-y*z/np.exp(u)))
 r=minimize_scalar(f,bounds=(lo,hi),method='bounded',options={'xatol':1e-10,'maxiter':500})
 if not r.success or not np.isfinite(r.fun) or min(r.x-lo,hi-r.x)<1e-4:raise ValueError(f'STOP temperature not identified inside registered bounds: T={np.exp(r.x)}')
 return float(np.exp(r.x))

def paired(d,games,seed,reps=2000):
 """Whole-game paired bootstrap of the mean per-row difference (ratio of game sums), percentile 95% CI."""
 u,inv=np.unique(games,return_inverse=True);s=np.bincount(inv,weights=d);n=np.bincount(inv).astype(float)
 pick=np.random.default_rng(seed).integers(0,len(u),(reps,len(u)));boot=s[pick].sum(1)/n[pick].sum(1)
 return {'mean_difference':float(d.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'games':int(len(u)),'rows':int(len(d)),'replicates':reps,'seed':seed}

def ece(q,y7):
 conf=q.max(1);hit=q.argmax(1)==y7;b=np.minimum((conf*10).astype(int),9)
 return float(sum(abs(conf[b==i].mean()-hit[b==i].mean())*(b==i).mean() for i in range(10) if (b==i).any()))

def evaluate(c,k,margins,temps,out,role,M):
 """Calibrated coherent seven leaves on CAL or EVAL; REPORT-ONLY. EVAL adds the whole-game paired fit100 - fit25 CI."""
 from scipy.special import softmax,expit
 ev=k['role'][role];y7=k['y7'][ev];games=k['games'][ev];res={};loss={}
 for arm in ARMS:
  z=margins[arm];t=temps[arm];q=M.safe.leaves(softmax(z['parent'][role]/t['parent'],axis=1),expit(z['narrow_whiff'][role]/t['narrow_whiff']),expit(z['ground_ball_per_pitch'][role]/t['ground_ball_per_pitch']))
  if len(q)!=len(y7):raise ValueError('STOP role rows differ from margins')
  np.save(out/f'seven-leaf-{arm}-{role}.npy',q,allow_pickle=False);allm,ll,bs=M.ap.seven_metrics(q,y7);loss[arm]=(ll,bs)
  unk=k['unknown_type'][ev];split={n:({'rows':int(s.sum()),'nll':float(ll[s].mean()),'brier':float(bs[s].mean())} if s.any() else {'rows':0,'nll':None,'brier':None}) for n,s in (('known_pitch_type',~unk),('unknown_pitch_type',unk))}
  res[arm]={'overall':allm,'by_action_support_report_only':split,'ece_top_label':ece(q,y7),'temperatures':t,'post_cal_diagnostics':M.ap.post_cal_diagnostics(q,y7,k['pitcher'][ev],k['batter'][ev])}
 res['denominators']={'rows':int(ev.sum()),'games':int(len(np.unique(games))),'leaf_counts':np.bincount(y7,minlength=7).tolist()}
 if role=='evaluation':res['paired']={'fit100-fit25':{met:paired(loss['fit100'][i]-loss['fit25'][i],games,c['bootstrap_seed']) for i,met in enumerate(('nll','brier'))}}
 return res

# ---------- the chain ----------
def run(c,synthetic=False,external_deadline=None):
 """Gates -> claim -> lock/resource -> monitor -> source -> corpus -> 6 classifiers -> frozen SELECT -> 6 temperatures -> report -> terminal."""
 M=closure(c);ap=M.ap;b=c['budgets']
 if synthetic:
  tmp=Path(tempfile.gettempdir()).resolve();root=Path(c.get('synthetic_root','/nonexistent')).resolve()
  if c.get('synthetic_lifecycle') is not True or tmp not in root.parents or any(root not in Path(p).resolve().parents for p in (c['source']['parquet'],c['source']['manifest'],c['lock']['path'],c['outputs']['attempt_root'],c['review']['path'],c['resource']['recorder']['db'])) or not c['resource']['recorder']['health_url'].startswith('http://127.0.0.1:') or ':8766' in c['resource']['recorder']['health_url']:raise SystemExit('BLOCKED: synthetic run must live entirely under one temp root')
 elif any(str(x).startswith('synthetic') for x in c):raise SystemExit('BLOCKED: synthetic key in a real registration')
 review_sha=review_gate(c)
 if synthetic and external_deadline is None:external_deadline=time.time()+b['wall_seconds']+2*b['cleanup_margin_seconds']
 if c['lock']['mode']!='runner' or external_deadline is None or not np.isfinite(external_deadline) or external_deadline<time.time()+b['wall_seconds']+b['cleanup_margin_seconds']:raise SystemExit('BLOCKED: runner-held lock and a finite external supervisor deadline >= now + wall_seconds + cleanup_margin_seconds are required (--external-deadline-utc)')
 if not synthetic:
  floors(b)
  plan(c);rt=c['runtime']
  if sha(rt['native_path'])!=rt['native_sha256'] or not Path(rt['tree_python']).exists():raise SystemExit('BLOCKED: pinned LightGBM runtime')
 argv=(lambda d:[sys.executable,'-I',str(HERE/'synthetic_child.py'),'--request',str(d/'request.json')]) if synthetic else (lambda d:None)
 att=ap.claim_attempt(c,'scale');start=time.monotonic();counts={'classifier_attempted':0,'classifier_completed':0,'temperature':0}
 for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='2'
 def event(name,**kw):
  with (att/'events.jsonl').open('a') as f:f.write(json.dumps({'t':time.monotonic()-start,'event':name,**kw})+'\n')
 def stopcheck():
  ap.check_stop()
  if mon.why:raise RuntimeError('STOP monitor: '+mon.why)
 def guard():
  stopcheck()
  if time.monotonic()-start>b['wall_seconds']:raise TimeoutError('Overall internal wall cap')
  if outsize(att)>b['max_output_bytes']:raise RuntimeError('Output cap')
  save(att/'last-resource.json',resource(c,before))  # raises a STOP on recorder/capture priority, RSS, headroom, disk, swap or health
 sig=ap.Signals().__enter__();t_lock=time.time()
 try:lock,before=ap.preflight(c,att,'scale',types.SimpleNamespace(resource=lambda:resource(c)),counts)  # common lock held from here to the outer finally
 except BaseException:sig.__exit__();raise
 state={'phase':'gates'};mon=Monitor(c,att,before,start,external_deadline,state,counts,ap);confirmed=False
 try:
  try:
   st=os.fstat(lock.f.fileno());hard=int(b['child_seconds']+b['child_hard_grace_seconds'])
   ap.durable(att/'lock-receipt.json',{'path':c['lock']['path'],'lock_inode':st.st_ino,'lock_device':st.st_dev,'runner_fd':lock.f.fileno(),'acquired_not_before_utc':t_lock,'acquired_not_after_utc':before['utc'],'runner_pid':os.getpid(),'runner_lstart':lstarts([os.getpid()])[os.getpid()],'internal_wall_deadline_utc':time.time()-(time.monotonic()-start)+b['wall_seconds'],'external_deadline_utc':external_deadline,'cleanup_margin_seconds':b['cleanup_margin_seconds'],'child_hard_deadline_seconds_each':hard,'monitor_interval_seconds':b['monitor_interval_seconds'],'monitor_grace_seconds':b['monitor_grace_seconds'],'released':None})
   mon.start();guard();save(att/'gates.json',{'review_sha256':review_sha,'sources_sha256':source_hashes(c),'attempt_id':att.name,'synthetic':synthetic})
   state['phase']='source_projection';ap.check_stop();ap.alarm(b['projection_seconds']);raw,binding=M.cp.read_source(c);save(att/'source-binding.json',binding);ap.check_stop()
   box=[raw];del raw;k=build_corpus(c,box,att,M,start+b['projection_seconds'],state);signal.alarm(0);event('corpus_ready');guard()
   margins={a:{} for a in ARMS};receipts={}
   for arm in ARMS:  # phase 1: six classifier fits, no temperature yet
    for head in HEADS:
     state['phase']=f'classifier:{arm}/{head}';guard();d=att/f'{arm}-{head}';d.mkdir();counts['classifier_attempted']+=1;save(att/'counts.json',counts);x=k['arms'][arm]
     receipts[f'{arm}/{head}']=child_fit(c,d,x['x'],x['paths'][head]['roles'],x['paths'][head]['weights'],x['params'][head],guard,stopcheck,lock,argv=argv(d))
     counts['classifier_completed']+=1;save(att/'counts.json',counts);event('classifier',arm=arm,head=head)
     z=np.load(d/'child-output'/'margins.npz');margins[arm][head]={r:z[r] for r in ROLES[1:]}
   if counts!={'classifier_attempted':6,'classifier_completed':6,'temperature':0}:raise ValueError('Incomplete 6 classifier inventory before SELECT')
   state['phase']='select_temperature_report';guard();dec,losses=select_decision(k,margins);np.savez(att/'select-losses.npz',**losses);dp=att/'SELECT-DECISION.json'  # phase 2
   with dp.open('x') as f:f.write(json.dumps(dec,indent=2,allow_nan=False)+'\n')
   frozen=sha(dp);event('select_frozen',sha256=frozen);temps={a:{} for a in ARMS};ca=k['role']['calibration']
   for arm in ARMS:  # phase 3: six CAL temperatures, decision hash-checked before each
    for head in HEADS:
     if sha(dp)!=frozen:raise ValueError('STOP SELECT decision changed')
     guard();parent=np.ones(int(ca.sum()),bool) if head=='parent' else (k['y10'][ca]==1) if head=='narrow_whiff' else np.isin(k['y10'][ca],BIP)
     y=k['y10'][ca] if head=='parent' else k['yh'][head][ca];ap.alarm(b['temperature_seconds'])
     temps[arm][head]=temperature(margins[arm][head]['calibration'][parent],y[parent]);signal.alarm(0);counts['temperature']+=1;save(att/'counts.json',counts);event('temperature',arm=arm,head=head)
   if sha(dp)!=frozen or counts!={'classifier_attempted':6,'classifier_completed':6,'temperature':6}:raise ValueError('Incomplete 6+6 inventory / SELECT decision changed')
   ap.check_stop()
   save(att/'result.json',{'study':c['study_id'],'label':'EXPOSED DEVELOPMENT restricted scale comparison; EVAL is not an independent TEST','claim_ceiling':c['claim_ceiling'],'select_decision_sha256':frozen,'select_decision':dec,'calibration':evaluate(c,k,margins,temps,att,'calibration',M),'evaluation_report_only':evaluate(c,k,margins,temps,att,'evaluation',M),'action_support':k['action_support'],'receipts':receipts,'counts':counts,'scientific_decision':None,'automatic_followup':False,'eval_reselection':False})
   guard();save(att/'terminal.json',{'success':True,'counts':counts,'elapsed_seconds':time.monotonic()-start,'artifacts':{str(f.relative_to(att)):sha(f) for f in att.rglob('*') if f.is_file() and f.name not in ('terminal.json','monitor.jsonl','lock-receipt.json','last-resource.json')}})
  except BaseException as e:
   mon.unwinding.set();signal.alarm(0);save(att/'terminal.json',{'success':False,'counts':counts,'phase':state['phase'],'error':repr(e),'pending_signals':list(ap._PENDING),'elapsed_seconds':time.monotonic()-start,'retry':False,'inference_prohibited':True});raise
  finally:
   mon.unwinding.set();mon.done.set();killed=err=None  # the sampler stops before cleanup so none of its probe processes is listed
   if mon.is_alive():mon.join(60)
   try:killed=cleanup_children(b['child_term_grace_seconds']);confirmed=True  # child-first, confirmed gone, before this runner's lock fd is closed
   except BaseException as e:err=repr(e)
   try:after=resource(c,before)
   except Exception as e:after={'error':repr(e)}
   save(att/'resource-after.json',dict(after,children_killed_before_lock_release=killed,cleanup_error=err))
   if err:  # disappearance not confirmed: the terminal is forced to a STOP whatever the chain result was
    tp=att/'terminal.json';save(tp,dict(json.loads(tp.read_text()) if tp.exists() else {'counts':counts},success=False,retry=False,inference_prohibited=True,cleanup_error=err,cleanup_confirmed=False));raise RuntimeError(err)
 finally:
  try:ap.durable(att/'lock-receipt.json',dict(json.loads((att/'lock-receipt.json').read_text()),released={'utc':time.time(),'owned_descendants_confirmed_gone':confirmed,'note':'only this runner fd is closed; a surviving owned child would still hold the inherited flock'}))
  except Exception:pass
  lock.__exit__();sig.__exit__()
 return att

# ---------- ONE synthetic check through run() (temp data, temp lock, numpy stand-in child) ----------
def synthetic_frame(rng):
 """Statcast-shaped synthetic season over the registered role windows with injected non-core rows."""
 import pandas as pd
 teams=['AAA','BBB','CCC','DDD','EEE','FFF'];types_=['FF','SL','CH','CU'];rows=[];gid=5000
 non=['ball','called_strike','swinging_strike','foul','hit_by_pitch','swinging_strike_blocked','foul_tip','blocked_ball','hit_into_play'];base=np.array([.33,.16,.08,.18,.012,.02,.02,.02,.178])
 evs=['field_out','single','double','triple','home_run','grounded_into_double_play','force_out','sac_fly'];pe=[.47,.2,.08,.04,.06,.07,.05,.03];bbs=['ground_ball','line_drive','fly_ball','popup']
 for d in pd.date_range('2026-03-25','2026-09-27',freq='2D'):
  for j in range(3):
   gid+=1;hi,ai=gid%6,(gid+1+j)%6;sc=[0,0]
   for ab in range(1,11):
    top=ab%2==1;di,oi=(hi,ai) if top else (ai,hi);pit=100+di*2+int(ab>6);bat=200+oi*5+ab%5;balls=strikes=0
    for pn in range(1,8):
     t=types_[int(rng.integers(4))];p=base.copy();p[2]*={'SL':5,'CU':2.5,'CH':1,'FF':.25}[t];p[0]*=1+.3*(t=='CU')+.2*balls;p[8]*=1+.5*(t=='FF');p/=p.sum();de=non[int(rng.choice(9,p=p))];e=bb=None
     if de=='hit_into_play':e=evs[int(rng.choice(8,p=pe))];bb='fly_ball' if e in ('home_run','sac_fly') else 'ground_ball' if e=='grounded_into_double_play' else bbs[int(rng.choice(4,p=[.75,.1,.1,.05] if t!='FF' else [.1,.25,.5,.15]))]
     if de=='hit_by_pitch':e='hit_by_pitch'
     rows.append(dict(game_pk=gid,at_bat_number=ab,pitch_number=pn,game_date=d.strftime('%Y-%m-%d'),game_type='R',home_team=teams[hi],away_team=teams[ai],pitcher=pit,batter=bat,pitch_type=None if rng.random()<.02 else t,description=de,events=e,bb_type=bb,
      release_speed=np.nan if rng.random()<.01 else {'FF':94,'SL':86,'CH':85,'CU':79}[t]+(pit%7)*.4+rng.normal()*.8,release_spin_rate=2200+(pit%5)*40+rng.normal()*60,pfx_x=(pit%3-1)*.4+rng.normal()*.1,pfx_z={'FF':1.3,'SL':.2,'CH':.6,'CU':-.8}[t]+rng.normal()*.1,
      balls=balls,strikes=strikes,outs_when_up=(ab//2)%3,inning=1+(ab-1)//2,inning_topbot='Top' if top else 'Bot',home_score=sc[0],away_score=sc[1],on_1b=300.+ab if rng.random()<.3 else None,on_2b=300.+ab if rng.random()<.15 else None,on_3b=300.+ab if rng.random()<.08 else None,stand='LR'[bat%2],p_throws='LR'[pit%2]))
     if de in ('ball','blocked_ball'):balls+=1
     elif de=='foul':strikes=min(2,strikes+1)
     elif de in ('called_strike','swinging_strike','swinging_strike_blocked','foul_tip'):strikes+=1
     if de in ('hit_into_play','hit_by_pitch') or balls==4 or strikes==3:break
    sc[0 if not top else 1]+=int(rng.random()<.25)
 f=pd.DataFrame(rows);hip=dict(description='hit_into_play',pitch_number=1);inj=[]
 defects=[('HYPOTHESIS_LEAF5_GB_BIP_OUTCOME_DEPENDENT',dict(hip,events='field_error',bb_type='ground_ball')),('HYPOTHESIS_LEAF6_NONGB_BIP_OUTCOME_DEPENDENT',dict(hip,events='fielders_choice',bb_type='line_drive')),('HYPOTHESIS_LEAF5_GB_BIP_OUTCOME_DEPENDENT',dict(hip,events='triple_play',bb_type='ground_ball')),('ABSTAIN_BIP_CATCHER_INTERF',dict(hip,events='catcher_interf',bb_type=None)),
  ('NOT_A_PITCH_AUTOMATIC',dict(description='automatic_ball',events=None,bb_type=None,pitch_type=None,release_speed=np.nan,pitch_number=1)),('ABSTAIN_AUTOMATIC_WITH_TRACKING',dict(description='automatic_strike',events=None,bb_type=None,pitch_number=1)),('ABSTAIN_UNUSUAL_PITCHOUT',dict(description='swinging_pitchout',events='strikeout',bb_type=None,pitch_number=1)),('ABSTAIN_UNKNOWN_DESCRIPTION',dict(description='weird_thing',events=None,bb_type=None,pitch_number=1)),
  ('ABSTAIN_NONBIP_WITH_BB_TYPE',dict(description='ball',events=None,bb_type='fly_ball',pitch_number=1)),('ABSTAIN_BIP_MISSING_BB_TYPE',dict(hip,events='field_out',bb_type=None)),('ABSTAIN_BIP_UNREGISTERED_BB_TYPE',dict(hip,events='field_out',bb_type='bunt_grounder')),('LEAF0',dict(description='ball',events=None,bb_type=None,pitch_type='ZZ',pitch_number=1))]
 first=f.groupby('game_pk').head(1);hosts=[first.iloc[i] for i in (3,len(first)//3,len(first)-40,len(first)-4)]  # early FIT, mid FIT, SELECT/CAL, EVAL games
 for hnum,host in enumerate(hosts):
  for i,(fate,over) in enumerate(defects):inj.append((fate,dict(host,balls=0,strikes=0,**over,at_bat_number=80+i)))
 return pd.concat([f,pd.DataFrame([r for _,r in inj])],ignore_index=True),inj

def expect(fn,exc,text):
 try:fn()
 except exc as e:
  if text not in str(e):raise AssertionError(f'wrong failure: {e!r} lacks {text!r}')
  return str(e)[:160]
 raise AssertionError('expected failure did not happen: '+text)

def synthetic(reg,keep=None):
 import pandas as pd,pyarrow as pa,pyarrow.parquet as pq,copy
 c0=load(reg);M=closure(c0);res={'started_utc':time.time(),'python':sys.version.split()[0]};root=Path(tempfile.mkdtemp(prefix='scale-synth-')).resolve();rng=np.random.default_rng(20261011)
 raw,inj=synthetic_frame(rng);(root/'holdout').mkdir();pp=root/'holdout'/'synthetic.parquet';pq.write_table(pa.Table.from_pandas(raw[M.cp.PROJECT],preserve_index=False),pp);(root/'.heavy.lock').write_text('synthetic temp lock\n')
 bind={'version':'synthetic','season':2026,'file':'holdout/synthetic.parquet','start':'2026-03-25','end':'2026-09-27','rows':len(raw),'columns':29,'game_type':'R','parquet_sha256':sha(pp),'fetch_commit':'synthetic'}
 mp=root/'manifest.json';save(mp,{'version':'synthetic','fetch_commit':'synthetic','parameters':{'game_type':'R','season_dates':{'2026':[bind['start'],bind['end']]}},'seasons':{'2026':{'season':2026,'file':bind['file'],'start':bind['start'],'end':bind['end'],'rows':len(raw),'rows_raw':len(raw),'columns':29,'duplicate_keys':0,'sha256':bind['parquet_sha256'],'frozen':True}}})
 import sqlite3,http.server,threading
 rdb=root/'observer.sqlite3';db=sqlite3.connect(rdb);db.execute('CREATE TABLE jobs(status TEXT,lease_until REAL)');db.commit()
 class Health(http.server.BaseHTTPRequestHandler):
  def do_GET(self):
   body=json.dumps({'model_ready':True,'dataset_ready':True}).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def log_message(self,*a):pass
 srv=http.server.ThreadingHTTPServer(('127.0.0.1',0),Health);threading.Thread(target=srv.serve_forever,daemon=True).start()
 # Temp stand-in recorder: a detached sleeper (not a descendant of this process) whose argv carries the worker/database identity.
 def detached(*tail):return int(subprocess.check_output([sys.executable,'-I','-c',"import subprocess,sys;print(subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(900)']+sys.argv[1:],start_new_session=True,stdout=subprocess.DEVNULL).pid)",*tail],text=True))
 def wait_for(f,secs,what):
  end=time.monotonic()+secs
  while time.monotonic()<end:
   v=f()
   if v:return v
   time.sleep(.1)
  raise AssertionError('timeout waiting for '+what)
 def osid(pid):
  f=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','ppid=','-o','lstart='],text=True).split();return int(f[0]),' '.join(f[1:])
 rpid=detached('-m','observer_app.worker','--database',str(rdb));wait_for(lambda:osid(rpid)[0]==1,10,'stand-in recorder reparented');rppid,rstart=osid(rpid)
 sent=detached('unrelated-sentinel');sstart=osid(sent)[1]  # unrelated process that no lifecycle path may touch
 def cfg(aid,**budget):
  c=copy.deepcopy(c0);c.update(synthetic_lifecycle=True,synthetic_root=str(root),source={'id':'SYNTHETIC','parquet':str(pp),'parquet_sha256':bind['parquet_sha256'],'manifest':str(mp),'manifest_sha256':sha(mp),'manifest_binding':bind,'rows':len(raw)})
  c['lock']={'path':str(root/'.heavy.lock'),'mode':'runner'};c['outputs']={'attempt_root':str(root/'attempts'),'attempt_ids':{'scale':aid}};c['review']={'path':str(root/f'REVIEW-PASS-{aid}.json')}
  c['fit_sample']['min_games_per_team']=2;c['expected_real_run_assertions']={'status':'synthetic: no numeric expectation'};c['budgets'].update(dict(max_tree_rss_bytes=GIB,capture_headroom_bytes=0,min_disk_free_launch_bytes=2*GIB,min_disk_free_bytes=GIB,monitor_interval_seconds=1,monitor_grace_seconds=30,child_term_grace_seconds=3),**budget);c['resource']['recorder']={'db':str(rdb),'pid':rpid,'ppid':rppid,'lstart':rstart,'health_url':f'http://127.0.0.1:{srv.server_address[1]}/api/health'};return c
 def passfile(c,**over):save(c['review']['path'],dict({'verdict':'PASS','distinct_from_author':True,'approved_stages':['scale'],'attempt_id':c['outputs']['attempt_ids']['scale'],'sources_sha256':source_hashes(c),'note':'SYNTHETIC temp review file written by the synthetic check itself; not a review'},**over))
 c=cfg('scale-001');g={}
 # 1. gates before any row: real registration blocked; synthetic key refused by the real path; review binding.
 g['real_execute_blocked']=expect(lambda:run(c0),SystemExit,'REVIEW-PASS required');g['synthetic_key_in_real_path']=expect(lambda:run(c),SystemExit,'synthetic key in a real registration')
 g['review_missing']=expect(lambda:run(c,True),SystemExit,'REVIEW-PASS required');passfile(c,sources_sha256={});g['review_hash_mismatch']=expect(lambda:run(c,True),SystemExit,'hashes differ')
 passfile(c,attempt_id='scale-000');g['review_wrong_attempt']=expect(lambda:run(c,True),SystemExit,'fresh attempt id');passfile(c)
 if (root/'attempts').exists():raise AssertionError('a blocked gate created an attempt')
 # 2. the actual chain.
 att=run(c,True);t=json.loads((att/'terminal.json').read_text());r=json.loads((att/'result.json').read_text());pj=json.loads((att/'projection.json').read_text());dn=json.loads((att/'denominators.json').read_text());ac=json.loads((att/'arm-contract.json').read_text());z=np.load(att/'corpus.npz')
 ev=[json.loads(l) for l in (att/'events.jsonl').read_text().splitlines()];names=[e['event'] for e in ev]
 assert t['success'] and t['counts']=={'classifier_attempted':6,'classifier_completed':6,'temperature':6},t
 assert names==['corpus_ready']+['classifier']*6+['select_frozen']+['temperature']*6 and ev[7]['sha256']==sha(att/'SELECT-DECISION.json')==r['select_decision_sha256'],names
 fc=pj['core_fate_census'];want={}
 for fate,_ in inj:want[fate]=want.get(fate,0)+1
 assert all(fc['not_accepted_by_fate'][k]==v for k,v in want.items() if not k.startswith('LEAF')) and all(fc['accepted_rows_by_fate'][k]==0 for k in M.safe.EXCLUDED_FATES),fc
 keys={tuple(x) for x in z['keys'].tolist()};gone=[(int(x['game_pk']),int(x['at_bat_number']),1) in keys for f_,x in inj if not f_.startswith('LEAF')];kept=[(int(x['game_pk']),int(x['at_bat_number']),1) in keys for f_,x in inj if f_.startswith('LEAF')]
 assert not any(gone) and all(kept) and pj['pitch_type_mapped_unknown'].get('ZZ')==4 and pj['pitch_type_mapped_unknown'].get('MISSING',0)>0,pj['pitch_type_mapped_unknown']
 picked=sorted({int(x) for x in np.unique(z['games'][z['fit']]) if int(hashlib.sha256(f'20261109:{int(x)}'.encode()).hexdigest()[:16],16)/2**64<.25})
 assert sorted(set(z['games'][z['sample']].tolist()))==picked and dn['sampled_game_keys_sha256']==hashlib.sha256(np.array(picked,'<i8').tobytes()).hexdigest() and .1<dn['fit25_train_rows']/dn['fit100_train_rows']<.45
 enc=json.loads((att/'encoder.json').read_text());assert enc['dimension']==199 and len(enc['continuous'])==36 and ac['train_rows']=={'fit25':int(z['sample'].sum()),'fit100':int(z['fit'].sum())}
 for arm in ARMS:
  for h in HEADS:
   q=json.loads((att/f'{arm}-{h}'/'request.json').read_text());assert q['x_sha256']==ac['x_sha256'][arm] and q['params']==(M.wc.MULTICLASS if h=='parent' else M.wc.BINARY)
  assert all(T_BOUNDS[0]<v<T_BOUNDS[1] for v in r['calibration'][arm]['temperatures'].values()) and np.isfinite(r['evaluation_report_only'][arm]['overall']['nll'])
  assert r['evaluation_report_only'][arm]['post_cal_diagnostics']['per_pitcher']['rows']==r['evaluation_report_only']['denominators']['rows']==int(z['evaluation'].sum())
 pr=r['evaluation_report_only']['paired']['fit100-fit25']['nll'];assert pr['ci95'][0]<=pr['mean_difference']<=pr['ci95'][1] and r['select_decision']['advisory_selected'] in ARMS and r['select_decision']['cal_or_eval_read'] is False
 with open(c['lock']['path']) as lf:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB)  # lock was released by run()
 # R4 action support, R7 lock/deadline receipt and R1 child identity of the normal chain.
 asu=dn['action_support'];unk_total=sum(pj['pitch_type_mapped_unknown'].values());role4=('FIT100_TRAIN','SELECT','CAL','EVAL')
 assert asu==r['action_support'] and asu['all_label_supported_rows']=={'label_supported_rows':pj['accepted_rows'],'known_pitch_type_rows':pj['accepted_rows']-unk_total,'unknown_pitch_type_rows':unk_total} and unk_total>0 and asu['expected']['asserted'] is False,asu
 assert sum(asu['by_role'][x]['unknown_pitch_type_rows'] for x in role4)==unk_total and all(v['known_pitch_type_rows']+v['unknown_pitch_type_rows']==v['label_supported_rows'] for v in asu['by_role'].values())
 assert [asu['by_arm'][a]['TRAIN']['label_supported_rows'] for a in ARMS]==[ac['train_rows'][a] for a in ARMS] and all(asu['by_arm'][a][x]==asu['by_role'][x] and asu['by_role'][x]['label_supported_rows']==dn['role_rows'][x] for a in ARMS for x in role4[1:]) and asu['by_role']['FIT100_TRAIN']['label_supported_rows']==dn['role_rows']['FIT']
 for a in ARMS:
  sp=r['evaluation_report_only'][a]['by_action_support_report_only'];assert sp['known_pitch_type']['rows']==asu['by_role']['EVAL']['known_pitch_type_rows'] and sp['unknown_pitch_type']['rows']==asu['by_role']['EVAL']['unknown_pitch_type_rows']
 assert expected_action({'all_label_supported_rows':{'unknown_pitch_type_rows':367}},713021,c0['expected_action_support'])['asserted'] is True
 lr=json.loads((att/'lock-receipt.json').read_text());ids=[json.loads(p.read_text()) for p in sorted(att.glob('*/child-identity.json'))]
 assert lr['lock_inode']==os.stat(c['lock']['path']).st_ino and lr['acquired_not_before_utc']<=lr['acquired_not_after_utc']<lr['internal_wall_deadline_utc']<lr['external_deadline_utc'] and lr['released']['owned_descendants_confirmed_gone'] is True and lr['runner_lstart'],lr
 assert len(ids)==6 and all(i['child_lstart'] and i['child_pgid']==i['child_pid'] and i['lock_inode']==lr['lock_inode'] and i['hard_deadline_utc']>i['popen_utc'] and i['terminal']['group_gone'] and i['terminal']['how']=='exit' for i in ids),ids
 mo=[json.loads(l) for l in (att/'monitor.jsonl').read_text().splitlines()];assert mo and all(x['tree_rss_bytes']>0 for x in mo),mo
 res['action_support']=asu;res['lock_receipt']=lr;res['monitor_samples_normal_chain']={'samples':len(mo),'phases':sorted({x['phase'] for x in mo})}
 res['chain']={'attempt':att.name,'source_rows':pj['source_rows'],'accepted_core_rows':pj['accepted_rows'],'role_rows':dn['role_rows'],'fit_games':dn['fit_games'],'sampled_games':dn['sampled_games'],'train_rows':ac['train_rows'],'excluded_by_reason':{k:v for k,v in fc['not_accepted_by_fate'].items() if not k.startswith('LEAF')},'old_fate_excluded':pj['excluded'],'pitch_type_mapped_unknown':pj['pitch_type_mapped_unknown'],'encoded_dimension':enc['dimension'],'counts':t['counts'],'event_order':names,'select':r['select_decision']['arms'],'advisory_selected':r['select_decision']['advisory_selected'],'temperatures':{a:r['calibration'][a]['temperatures'] for a in ARMS},'eval_nll':{a:r['evaluation_report_only'][a]['overall']['nll'] for a in ARMS},'paired_nll':pr,'elapsed_seconds':t['elapsed_seconds'],'classifier':'numpy stand-in child (synthetic_child.py); LightGBM NOT loaded or fit'}
 # 3. negative controls on the production guards, from the same projected frame.
 raw2=pq.ParquetFile(pp).read(columns=M.cp.PROJECT).to_pandas();gf,_,rm,sample,_,y7=project(c,raw2,M);n={};cut=c['history_cutoff']
 res['leak_check']=leak_check(gf,rm['FIT'],cut,M)
 n['future_feature_leakage']=expect(lambda:leak_check(gf,rm['FIT'],cut,M,lambda pool,rows,co,M_:features(pool.assign(game_date=pool.game_date-pd.Timedelta(days=1)),rows,co,M_)),ValueError,'future feature leakage')
 bad=gf.copy();i=int(np.flatnonzero((bad.description=='hit_into_play').to_numpy())[0]);bad.loc[i,'events']='field_error';n['unsupported_label_field_error']=expect(lambda:core_guard(bad,M),ValueError,'unsupported label')
 bad=gf.copy();bad.loc[int(np.flatnonzero((bad.description=='ball').to_numpy())[0]),'description']='automatic_ball';n['unsupported_label_automatic']=expect(lambda:core_guard(bad,M),ValueError,'unsupported label')
 bad=gf.copy();bad.loc[bad.game_pk==bad.game_pk.iloc[0],'game_date']=[pd.Timestamp('2026-08-02')]*int((bad.game_pk==bad.game_pk.iloc[0]).sum()-1)+[pd.Timestamp('2026-08-03')];n['game_spans_roles']=expect(lambda:M.cp.roles(bad,dict(c,roles=dict(c['roles'],TEST=c['roles']['EVAL']))),ValueError,'Game spans roles')
 m25={'train':sample,'tuning':rm['SELECT'],'calibration':rm['CAL'],'evaluation':rm['TEST']}
 for role,name in (('calibration','cal_row_in_encoder_fit'),('evaluation','eval_row_in_encoder_fit'),('tuning','select_row_in_encoder_fit')):
  e=sample.copy();e[np.flatnonzero(m25[role])[0]]=True;n[name]=expect(lambda:encoder_guard(e,m25),ValueError,'encoder fit rows')
 e=sample.copy();e[np.flatnonzero(rm['FIT']&~sample)[0]]=True;n['unsampled_fit_row_in_encoder_fit']=expect(lambda:encoder_guard(e,m25),ValueError,'encoder fit rows')
 def arm(a):
  idx=np.flatnonzero(z['fit']|z['tuning']|z['calibration']|z['evaluation']) if a=='fit100' else np.flatnonzero(z['sample']|z['tuning']|z['calibration']|z['evaluation'])
  return {'idx':idx,'m':dict({r:z[r][idx] for r in ROLES[1:]},train=(z['fit'] if a=='fit100' else z['sample'])[idx]),'x':att/f'x-{a}.npy','paths':{h:{'roles':att/f'roles-{a}-{h}.npz','weights':att/f'weights-{a}-{h}.npy'} for h in HEADS},'encoder_sha256':ac['encoder_sha256'],'params':{'parent':dict(M.wc.MULTICLASS),'narrow_whiff':dict(M.wc.BINARY),'ground_ball_per_pitch':dict(M.wc.BINARY)}}
 def kk():return {'y7':z['y7'].copy(),'supported':np.ones(len(z['y7']),bool),'keys':z['keys'],'fit':z['fit'].copy(),'sample':z['sample'].copy(),'encoder_rows':z['sample'].copy(),'role':{r:z[r].copy() for r in ROLES[1:]},'arms':{a:arm(a) for a in ARMS}}
 assert arm_contract(kk())['x_sha256']==ac['x_sha256']
 def tamper(name,text,f):
  k=kk();f(k);n[name]=expect(lambda:arm_contract(k),ValueError,text)
 tamper('unequal_seed','unequal encoder/features/seed/parameters',lambda k:k['arms']['fit100']['params']['parent'].update(seed=1))
 tamper('unequal_parameters','unequal encoder/features/seed/parameters',lambda k:k['arms']['fit100']['params']['narrow_whiff'].update(num_leaves=31))
 tamper('unequal_encoder','unequal encoder/features/seed/parameters',lambda k:k['arms']['fit25'].update(encoder_sha256='0'*64))
 def xt(k):
  x=np.load(att/'x-fit25.npy');x[0,0]+=1;np.save(root/'x-tampered.npy',x);k['arms']['fit25']['x']=root/'x-tampered.npy'
 tamper('unequal_features','unequal encoder/features/seed/parameters',xt)
 def rolet(k):
  m=k['arms']['fit100']['m'];i=np.flatnonzero(m['calibration'])[0];m['calibration']=m['calibration'].copy();m['evaluation']=m['evaluation'].copy();m['calibration'][i]=False;m['evaluation'][i]=True
 tamper('role_split_mismatch','role/split mismatch',rolet)
 def traint(k):
  m=k['arms']['fit25']['m'];m['train']=m['train'].copy();m['tuning']=m['tuning'].copy();i=np.flatnonzero(m['tuning'])[0];m['train'][i]=True;m['tuning'][i]=False
 tamper('select_row_in_fit25_train','role/split mismatch',traint)
 tamper('encoder_rows_not_fit25_train','encoder fit rows differ',lambda k:k['encoder_rows'].__setitem__(np.flatnonzero(k['role']['evaluation'])[0],True))
 tamper('unsupported_label_in_arm','unsupported label entering an arm',lambda k:k['supported'].__setitem__(5,False))
 n['expected_assertion_mismatch']=expect(lambda:assert_expected({'fit_games':dn['fit_games'],'sampled_game_keys_sha256':dn['sampled_game_keys_sha256']},c0['expected_real_run_assertions']),ValueError,'expected real-run assertion mismatch')
 # 4. known-probability temperature recovery (multiclass and binary) and bound STOP.
 zt=rng.normal(size=(20000,10))*1.5;u=rng.random(20000);pm=np.exp(zt);pm/=pm.sum(1,keepdims=True);ym=(pm.cumsum(1)<u[:,None]).sum(1).clip(0,9);zb=rng.normal(size=20000)*2;yb=(rng.random(20000)<1/(1+np.exp(-zb))).astype(np.int8)
 res['temperature_known_probability']={'multiclass_T0_1.7':temperature(zt*1.7,ym),'binary_T0_0.6':temperature(zb*.6,yb)}
 assert abs(res['temperature_known_probability']['multiclass_T0_1.7']/1.7-1)<.05 and abs(res['temperature_known_probability']['binary_T0_0.6']/.6-1)<.05
 n['temperature_at_bound']=expect(lambda:temperature(np.where(yb==1,5.,-5.),yb),ValueError,'temperature not identified')
 # 5. lifecycle: fresh-attempt reuse, internal wall stop mid-inventory, busy common lock.
 n['attempt_reuse']=expect(lambda:run(c,True),SystemExit,'attempt exists')
 done=[e['t'] for e in ev if e['event']=='classifier'];c2=cfg('scale-002',wall_seconds=(done[1]+done[2])/2);passfile(c2);n['internal_wall_stop']=expect(lambda:run(c2,True),(TimeoutError,RuntimeError),'Overall internal wall cap')
 a2=root/'attempts'/'scale-002';t2=json.loads((a2/'terminal.json').read_text());assert not t2['success'] and t2['retry'] is False and t2['inference_prohibited'] and t2['counts']['classifier_completed']<6 and t2['counts']['temperature']==0 and not (a2/'SELECT-DECISION.json').exists() and not (a2/'result.json').exists(),t2
 left=[json.loads(p.read_text())['terminal'] for p in a2.glob('*/child-identity.json')];assert all(x and x['group_gone'] for x in left)
 res['wall_stop']={'counts':t2['counts'],'error':t2['error'],'children_confirmed_gone':len(left)}
 c3=cfg('scale-003');passfile(c3)
 with open(c['lock']['path']) as lf:
  fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB);n['common_lock_busy']=expect(lambda:run(c3,True),SystemExit,'shared lock busy')
 t3=json.loads((root/'attempts'/'scale-003'/'terminal.json').read_text());assert t3['phase']=='lock_preflight' and t3['retry'] is False
 def stop(aid,name,text,pre=lambda x:None,**budget):
  x=cfg(aid,**budget);pre(x);passfile(x);n[name]=expect(lambda:run(x,True),RuntimeError,text);tt=json.loads((root/'attempts'/aid/'terminal.json').read_text());assert tt['success'] is False and tt['retry'] is False and tt['phase']=='resource_preflight',tt
 db.execute("INSERT INTO jobs VALUES('queued',0)");db.commit();stop('scale-004','recorder_queue_busy','Recorder queue or lease active');db.execute('DELETE FROM jobs');db.execute("INSERT INTO jobs VALUES('done',?)",(time.time()+600,));db.commit();stop('scale-005','recorder_lease_active','Recorder queue or lease active');db.execute('DELETE FROM jobs');db.commit()
 stop('scale-006','recorder_identity_changed','identity changed',pre=lambda x:x['resource']['recorder'].update(pid=os.getpid()))
 stop('scale-007','tree_rss_over_cap','aggregate process-tree RSS over cap',max_tree_rss_bytes=1<<20);stop('scale-008','capture_headroom_short','memory headroom below',capture_headroom_bytes=1<<50);stop('scale-009','disk_below_floor','disk free below protected floor',min_disk_free_launch_bytes=1<<60)
 stop('scale-014','recorder_start_identity_changed','identity changed',pre=lambda x:x['resource']['recorder'].update(lstart='Wed Sep 23 22:25:17 2026'));stop('scale-015','recorder_parent_changed','identity changed',pre=lambda x:x['resource']['recorder'].update(ppid=94628))
 x=cfg('scale-016');passfile(x);n['external_deadline_required']=expect(lambda:run(x,True,time.time()+5),SystemExit,'external supervisor deadline');assert not (root/'attempts'/'scale-016').exists()
 n['real_runtime_floor_14gib_refused']=expect(lambda:floors(dict(c0['budgets'],min_disk_free_bytes=14*GIB)),SystemExit,'below the protected minimum');n['real_runtime_floor_exactly_21gib_refused']=expect(lambda:floors(dict(c0['budgets'],min_disk_free_bytes=21*GIB)),SystemExit,'below the protected minimum');n['real_launch_floor_20gib_refused']=expect(lambda:floors(dict(c0['budgets'],min_disk_free_launch_bytes=20*GIB)),SystemExit,'below the protected minimum')
 n['unknown_action_total_mismatch']=expect(lambda:expected_action({'all_label_supported_rows':{'unknown_pitch_type_rows':366}},713021,c0['expected_action_support']),ValueError,'UNKNOWN pitch_type rows')
 # 6. R6 known-answer report metrics and paired whole-game bootstrap, bound to this file's hash.
 q2=np.array([[.5,.2,.1,.05,.05,.05,.05],[.1,.6,.1,.05,.05,.05,.05]]);m2,_,_=M.ap.seven_metrics(q2,np.array([0,2]));assert m2['rows']==2 and abs(m2['nll']-(np.log(2)+np.log(10))/2)<1e-12 and abs(m2['brier']-.75)<1e-12,m2
 N=200000;pk=rng.dirichlet(np.full(7,.8),N);yk=(pk.cumsum(1)<rng.random(N)[:,None]).sum(1).clip(0,6);mk,_,_=M.ap.seven_metrics(pk,yk);ent=float(-(pk*np.log(np.clip(pk,1e-300,1))).sum(1).mean());eb=float(1-(pk**2).sum(1).mean())
 sharp=pk**3;sharp/=sharp.sum(1,keepdims=True);e_cal=ece(pk,yk);e_bad=ece(sharp,yk);pcd=M.ap.post_cal_diagnostics(pk,yk,rng.integers(0,50,N),rng.integers(0,80,N))
 gaps=[abs(x['mean_predicted']-x['observed_frequency']) for leaf in pcd['reliability_seven_leaf'].values() for x in leaf if x['count']>=2000]
 assert abs(mk['nll']-ent)<.02 and abs(mk['brier']-eb)<.01 and e_cal<.01<.05<e_bad and len(gaps)>=20 and max(gaps)<.04 and all(sum(x['count'] for x in leaf)==N for leaf in pcd['reliability_seven_leaf'].values()) and pcd['per_pitcher']['rows']==pcd['per_batter']['rows']==N and pcd['per_pitcher']['players']==50,(mk,ent,eb,e_cal,e_bad,max(gaps))
 gm=np.repeat(np.arange(300),rng.integers(5,40,300));pc=paired(np.full(len(gm),.3),gm,7);assert abs(pc['mean_difference']-.3)<1e-12 and max(abs(v-.3) for v in pc['ci95'])<1e-12 and (pc['games'],pc['rows'],pc['replicates'])==(300,len(gm),2000),pc
 dd=rng.normal(.05,.2,300)[gm]+rng.normal(0,.5,len(gm));pe=paired(dd,gm,c0['bootstrap_seed']);S=[float(dd[gm==i].sum()) for i in range(300)];Nn=[int((gm==i).sum()) for i in range(300)];pick=np.random.default_rng(c0['bootstrap_seed']).integers(0,300,(2000,300))
 ind=np.array([sum(S[j] for j in row)/sum(Nn[j] for j in row) for row in pick.tolist()]);se=float(np.std(np.array(S)-dd.mean()*np.array(Nn),ddof=1)*np.sqrt(300)/len(gm))
 assert np.allclose(pe['ci95'],[np.quantile(ind,.025),np.quantile(ind,.975)],atol=1e-12) and pe['ci95'][0]<dd.mean()<pe['ci95'][1] and abs(pe['mean_difference']-dd.mean())<1e-12 and .8<(pe['ci95'][1]-pe['ci95'][0])/(2*1.96*se)<1.2,(pe,se)
 res['known_answer_metrics']={'two_row_hand_nll_brier':m2,'known_probability_rows':N,'nll':mk['nll'],'mean_entropy':ent,'brier':mk['brier'],'expected_brier':eb,'ece_calibrated':e_cal,'ece_sharpened_cubed':e_bad,'reliability_bins_checked':len(gaps),'reliability_max_gap':max(gaps),'reliability_counts_per_leaf':N,'player_rows_reconcile':N,'paired_constant_0.3':pc,'paired_planted':pe,'paired_equals_independent_loop':True,'paired_width_over_analytic':(pe['ci95'][1]-pe['ci95'][0])/(2*1.96*se)}
 # 7. R1-R3 lifecycle through the production spawn/monitor/cleanup path (temp lock, temp children, separate runner processes).
 def lock_free():
  with open(c['lock']['path']) as lf:
   try:fcntl.flock(lf,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
   except BlockingIOError:return False
 def spawn(x,**env):
  p=root/f"cfg-{x['outputs']['attempt_ids']['scale']}.json";save(p,x);passfile(x)
  return subprocess.Popen([sys.executable,'-I','-B',str(HERE/'scale_runner.py'),'--synthetic-runner',str(p)],env=dict(os.environ,**env),stdout=subprocess.DEVNULL,stderr=(root/(p.name+'.err')).open('w'),start_new_session=True)
 def live(p):
  if p.poll() is not None:raise AssertionError(f'synthetic runner exited early with {p.returncode}: '+''.join(f.read_text()[-600:] for f in root.glob('*.err')))
  return True
 dl=subprocess.Popen([sys.executable,'-I','-S','-c',SHIM,'1',sys.executable,'-I','-c','import time;time.sleep(60)']);assert dl.wait(15)==-signal.SIGALRM  # hard deadline fires through exec with nobody supervising
 # 7a. R1: SIGKILL the runner while its first classifier child is running.
 c10=cfg('scale-010');pr=spawn(c10,SCALE_SYNTH_CHILD_SLEEP='600');a10=root/'attempts'/'scale-010';ip=a10/'fit25-parent'/'child-identity.json'
 idn=wait_for(lambda:live(pr) and ip.exists() and json.loads(ip.read_text()),180,'child identity of the spawned runner');cpid=idn['child_pid'];l10=json.loads((a10/'lock-receipt.json').read_text())
 assert idn['terminal'] is None and idn['child_pgid']==cpid and idn['runner_pid']==pr.pid and idn['lock_inode']==l10['lock_inode']==os.stat(c['lock']['path']).st_ino and lstarts([cpid]).get(cpid)==idn['child_lstart'] and not lock_free(),idn
 os.kill(pr.pid,signal.SIGKILL);code=pr.wait(30);time.sleep(1.5);alive=lstarts([cpid]).get(cpid)==idn['child_lstart'];busy=[not lock_free() for _ in range(3)]
 assert code==-signal.SIGKILL and alive and all(busy) and not (a10/'terminal.json').exists(),(code,alive,busy)
 x=cfg('scale-011');passfile(x);n['overlapping_attempt_blocked_while_child_lives']=expect(lambda:run(x,True),SystemExit,'stale/overlapping attempt')
 rp=reap(a10,3.);gone_after=cpid not in lstarts([cpid]);free_after=lock_free();t10=json.loads((a10/'terminal.json').read_text())
 assert rp['ok'] and len(rp['children'])==1 and rp['children'][0]['how'] in ('TERM','KILL') and rp['children'][0]['group_gone'] and gone_after and free_after and t10['success'] is False and t10['phase']=='supervisor_reap' and t10['retry'] is False and osid(sent)[1]==sstart and osid(rpid)[1]==rstart,(rp,gone_after,free_after)
 res['runner_sigkill_mid_child']={'runner_returncode':code,'child_pid':cpid,'child_lstart':idn['child_lstart'],'child_hard_deadline_seconds':idn['hard_deadline_seconds'],'child_alive_after_runner_kill':alive,'lock_busy_while_child_alive':busy,'terminal_absent_before_reap':True,'reap':rp,'child_gone_after_reap':gone_after,'lock_acquirable_after_reap':free_after,'sentinel_untouched':True,'terminal_after_reap':t10}
 # 7b. R2: blocked feature phase; the temp recorder queue turns busy; the monitor must hard-stop the runner.
 c12=cfg('scale-012',monitor_grace_seconds=2);c12['synthetic_slow_feature_seconds']=300;pr=spawn(c12);a12=root/'attempts'/'scale-012';mj=a12/'monitor.jsonl'
 wait_for(lambda:live(pr) and mj.exists() and '"features"' in mj.read_text(),180,'monitor sample inside the feature phase');db.execute("INSERT INTO jobs VALUES('queued',0)");db.commit();t0=time.monotonic();code=pr.wait(60);took=time.monotonic()-t0;db.execute('DELETE FROM jobs');db.commit()
 t12=json.loads((a12/'terminal.json').read_text());assert code==70 and t12['success'] is False and t12['phase']=='features' and 'Recorder queue or lease active' in t12['error'] and t12['cleanup']['confirmed_gone'] and t12['retry'] is False and took<20 and not list(a12.glob('*/child-identity.json')) and not (a12/'result.json').exists() and lock_free() and osid(sent)[1]==sstart and osid(rpid)[1]==rstart,(code,t12,took)
 res['monitor_blocked_feature_phase']={'runner_returncode':code,'seconds_from_busy_queue_to_exit':took,'terminal':t12,'monitor_samples':len(mj.read_text().splitlines()),'lock_acquirable_after':True,'recorder_stand_in_and_sentinel_untouched':True}
 # 7c. R3: fail-closed listing and natural parent-exit descendant through the actual cleanup path.
 flag=root/'ps-fail';proxy=root/'ps-proxy';proxy.write_text(f'#!/bin/sh\n[ -e "{flag}" ] && exit 1\nexec /bin/ps "$@"\n');proxy.chmod(0o755);noself=root/'ps-noself';noself.write_text('#!/bin/sh\nexec /bin/ps -p 1 -o pid=,ppid=,pgid=,rss=,lstart=\n');noself.chmod(0o755);G=globals()
 stray=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(600)'],start_new_session=True);got=cleanup_children(1.);assert [v['pid'] for v in got]==[stray.pid] and got[0]['lstart'] and stray.pid not in lstarts([stray.pid]),got
 G['PS']=str(noself);n['own_pid_missing_from_listing']=expect(lambda:descendants(os.getpid()),RuntimeError,'process listing failed');G['PS']=str(proxy);assert descendants(os.getpid())[1][os.getpid()][2]>0
 flag.touch();n['process_listing_failed_rss_gate']=expect(lambda:descendants(os.getpid()),RuntimeError,'process listing failed');n['process_listing_failed_cleanup']=expect(cleanup_children,RuntimeError,'process listing failed');n['process_listing_failed_identity']=expect(lambda:lstarts([1]),RuntimeError,'process listing failed');flag.unlink()
 c13=cfg('scale-013');passfile(c13);a13=root/'attempts'/'scale-013';os.environ['SCALE_SYNTH_CHILD_ORPHAN']='1'
 threading.Thread(target=lambda:(wait_for(lambda:(a13/'fit25-narrow_whiff').exists(),180,'second head'),flag.touch()),daemon=True).start()
 try:n['process_listing_failed_mid_run']=expect(lambda:run(c13,True),RuntimeError,'process listing failed')
 finally:G['PS']='/bin/ps';del os.environ['SCALE_SYNTH_CHILD_ORPHAN'];flag.unlink()
 t13=json.loads((a13/'terminal.json').read_text());i13=[json.loads(p.read_text()) for p in sorted(a13.glob('*/child-identity.json'))];orph=[int(p.read_text()) for p in sorted(a13.glob('*/orphan.pid'))];first=json.loads((a13/'fit25-parent'/'child-identity.json').read_text())['terminal']
 assert t13['success'] is False and t13['retry'] is False and t13['cleanup_confirmed'] is False and 'process listing failed' in t13['cleanup_error'] and t13['counts']['temperature']==0 and not (a13/'result.json').exists(),t13
 assert orph and not lstarts(orph) and first['how']=='exit' and first['group_kill_rounds_after_leader_exit']>=1 and all(i['terminal'] and i['terminal']['group_gone'] for i in i13) and not OWNED_PGIDS and lock_free() and cleanup_children(1.)==[] and osid(sent)[1]==sstart,(first,orph)
 res['fail_closed_listing_and_orphan']={'terminal':t13,'orphans_started_by_children':len(orph),'orphans_alive_after':0,'first_child_terminal':first,'children_confirmed_gone':len(i13),'stray_descendant_cleanup':got,'lock_acquirable_after':True}
 os.kill(sent,signal.SIGKILL)
 floors(c0['budgets']);n['real_disk_floor_8gib_refused']=expect(lambda:floors(dict(c0['budgets'],min_disk_free_bytes=8*GIB)),SystemExit,'below the protected minimum');n['synthetic_floors_refused_for_real']=expect(lambda:floors(c['budgets']),SystemExit,'below the protected minimum')
 ra=json.loads((att/'resource-after.json').read_text());assert ra['children_killed_before_lock_release']==[] and ra['tree_children']<=1,ra
 res['resource_probe']={k:ra[k] for k in ('phase','memory_free_percent','tree_rss_bytes','headroom_free_inactive_speculative_bytes','disk_free_bytes','recorder_busy','children_killed_before_lock_release')}
 os.kill(rpid,signal.SIGKILL);srv.shutdown();db.close()
 bad=sorted(x for x in sys.modules if x.split('.')[0] in DENIED);assert not bad,bad
 n['denied_import']=expect(lambda:__import__('learning_diagnostic'),ImportError,'DENIED import')
 res.update(gates=g,negative_controls=n,denied_modules_loaded=bad,source_hashes=source_hashes(c0),temp_root=str(root),finished_utc=time.time(),s2_bytes_read=0,lightgbm_loaded=False,common_lock_touched=False,verdict='SYNTHETIC_PASS')
 if keep is None:shutil.rmtree(root);res['temp_root_removed']=True
 return res

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--registration',default=str(HERE/'SCALE-REGISTRATION.json'));g=a.add_mutually_exclusive_group(required=True)
 g.add_argument('--plan',action='store_true');g.add_argument('--synthetic',action='store_true');g.add_argument('--execute',action='store_true');a.add_argument('--keep-temp',action='store_true')
 g.add_argument('--reap',metavar='ATTEMPT_DIR',help='supervisor side after a dead runner: stop the exact owned children of that attempt and confirm them gone')
 g.add_argument('--synthetic-runner',metavar='TEMP_CFG',help='synthetic lifecycle only: run() on a temp-root config in its own process');a.add_argument('--external-deadline-utc',type=float);args=a.parse_args()
 if args.synthetic or args.synthetic_runner:tempfile.tempdir=str(HERE/'synthetic-tmp');Path(tempfile.tempdir).mkdir(exist_ok=True)  # scale-002 rework: every synthetic temp file stays in this job-owned directory
 if args.plan:print(json.dumps(plan(load(args.registration)),indent=2))
 elif args.reap:out=reap(args.reap);print(json.dumps(out,indent=2));sys.exit(0 if out['ok'] else 3)
 elif args.synthetic_runner:run(json.loads(Path(args.synthetic_runner).read_text()),True)
 elif args.synthetic:
  out=synthetic(args.registration,keep=True if args.keep_temp else None);save(HERE/'SYNTHETIC-RECEIPT.json',out);print(json.dumps({k:out[k] for k in ('verdict','wall_stop','runner_sigkill_mid_child','monitor_blocked_feature_phase','fail_closed_listing_and_orphan','known_answer_metrics','negative_controls')},indent=1))
 else:print(json.dumps({'attempt':str(run(load(args.registration),external_deadline=args.external_deadline_utc))}))
