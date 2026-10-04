import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# EXP-P32-001 (embedded-input residual) bounded pipeline; adapted from run-p31-pipeline.py. Launched by the delegated Opus PM worker.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
code=b/'code-ed5ae1a/experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
configsha='e8b60d820477b222cfe7d3a61f654735a3952aba97e521bd85846dcba6374fdd'
draftsha='bb0ce6209db7b5b6bbc9243cb57e7ba4913dc1cf567756c67feacf2f508ac14e'
state=b/'score-p32-state.json'
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
 draft=b/'code-ed5ae1a/configs/DRAFT-CUDA-EMBED-RESIDUAL-20261004.json';cfg=b/'EXP-P32-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert json.loads((b/'queue-embed-smoke-001/summary.json').read_text())['counts']=={'done':18},'all18 smoke tasks must finish'
 assert json.loads((b/'queue-p30-001/summary.json').read_text())['counts']=={'done':50},'P30-001 must be terminal'
 assert not (b/'queue-p30-002').exists(),'EXP-P30-002 present: start_after not satisfied'
 assert json.loads((b/'queue-p31-001/summary.json').read_text())['counts']=={'done':15},'P31-001 must be terminal'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py'],capture_output=True).stdout.strip(),'another cuda_direct queue is running: start_after not satisfied'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 for arm in old['arms']:
  for seed in old['seeds']:
   runner.verify_task(runner.task_dir(b/'runs/DRAFT-EMBED-RESIDUAL-smoke',arm['name'],seed,True),runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',runner.own_hashes()))
 execute('p32-full-001',queue(cfg,configsha,b/'runs/EXP-P32-001',b/'queue-p32-001'),40000)
 assert json.loads((b/'queue-p32-001/summary.json').read_text())['counts']=={'done':18},'full family incomplete'
 score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P32-001'),'--destination',str(b/'scores/EXP-P32-001')]
 execute('score-p32-001',score,900)
 result=json.loads((b/'scores/EXP-P32-001/score.json').read_text());assert result['family']=='EXP-P32-001' and result['decision'] is None
 save('complete',score=str(b/'scores/EXP-P32-001/score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
