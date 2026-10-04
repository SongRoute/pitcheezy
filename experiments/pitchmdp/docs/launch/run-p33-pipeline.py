import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# EXP-P33-001 (EMA weight averaging) bounded pipeline; adapted from run-p32-pipeline.py (adds the F6 raw-trajectory check before scoring). Launched by the delegated Opus PM worker.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
code=b/'code-b4b9243/experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
configsha='e803ce31b8451c8c70c58cc7513b739e2d4123a5a512751ca4ea446311976d88'
draftsha='a6cf64eb2698a47c3a9302bd07abdecbdcf33e6057b14654a1bd947058ce8de7'
state=b/'score-p33-state.json'
if state.exists():raise SystemExit('already started')
child=None

def save(status,**kw):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));tmp.replace(state)

def interrupted(signum,frame):raise RuntimeError('pipeline interrupted by signal '+str(signum))
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)

def execute(name,cmd,timeout):
 global child
 with (b/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
  save(name,child_pid=child.pid,command=cmd)
  ret=child.wait(timeout=timeout)
  if ret:raise RuntimeError(name+' failed, exit '+str(ret))
 child=None

def queue(config,sha,output,queue_dir):
 return [python,str(code/'scripts/cuda_direct_queue.py'),'--config',str(config),'--config-sha256',sha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--output',str(output),'--queue-dir',str(queue_dir),'--lock-dir','/home/sds733/pitcheezy-autonomy/gpu-locks','--gpu','GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9','--gpu','GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99','run']

try:
 save('verifying_completed_smoke')
 draft=b/'code-b4b9243/configs/DRAFT-CUDA-EMA-20261004.json';cfg=b/'EXP-P33-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert json.loads((b/'queue-ema-smoke-001/summary.json').read_text())['counts']=={'done':20},'all20 smoke tasks must finish'
 assert json.loads((b/'queue-p30-001/summary.json').read_text())['counts']=={'done':50},'P30-001 must be terminal'
 assert not (b/'queue-p30-002').exists(),'EXP-P30-002 present: start_after not satisfied'
 assert json.loads((b/'queue-p31-001/summary.json').read_text())['counts']=={'done':15},'P31-001 must be terminal'
 assert json.loads((b/'queue-p32-001/summary.json').read_text())['counts']=={'done':18},'P32-001 must be terminal'
 assert not (b/'queue-p33-001').exists() and not (b/'runs/EXP-P33-001').exists(),'P33 output already present'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py'],capture_output=True).stdout.strip(),'another cuda_direct queue is running: start_after not satisfied'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 for arm in old['arms']:
  for seed in old['seeds']:
   runner.verify_task(runner.task_dir(b/'runs/DRAFT-EMA-smoke',arm['name'],seed,True),runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',runner.own_hashes()))
 execute('p33-full-001',queue(cfg,configsha,b/'runs/EXP-P33-001',b/'queue-p33-001'),40000)
 assert json.loads((b/'queue-p33-001/summary.json').read_text())['counts']=={'done':20},'full family incomplete'
 # F6: raw-trajectory identity on the full family BEFORE any score exists (reads TRAIN / early-stop losses only)
 save('trajectory_check')
 with (b/'ema-trajectory-p33-001.json').open('x') as out:
  traj=subprocess.run([python,str(code/'scripts/ema_trajectory_check.py'),'--config',str(cfg),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P33-001')],cwd=code,stdout=out,stderr=subprocess.PIPE,text=True,timeout=600,env={**os.environ,'OMP_NUM_THREADS':'4','PYTHONPATH':str(code),'PYTHONDONTWRITEBYTECODE':'1'})
 try:report=json.loads((b/'ema-trajectory-p33-001.json').read_text())
 except ValueError:report=None
 # exit 1 is a valid report only when it parsed and says all_exact false; an uncaught exception also exits 1
 if not (isinstance(report,dict) and report.get('experiment_id')=='EXP-P33-001' and report.get('all_exact') is (traj.returncode==0) and traj.returncode in (0,1)):
  raise RuntimeError('trajectory check error, exit '+str(traj.returncode)+': '+traj.stderr[-2000:])
 trajectory_exit=traj.returncode
 score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P33-001'),'--destination',str(b/'scores/EXP-P33-001')]
 execute('score-p33-001',score,900)
 result=json.loads((b/'scores/EXP-P33-001/score.json').read_text());assert result['family']=='EXP-P33-001' and result['decision'] is None
 save('complete',score=str(b/'scores/EXP-P33-001/score.json'),trajectory=str(b/'ema-trajectory-p33-001.json'),trajectory_exit=trajectory_exit)
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
