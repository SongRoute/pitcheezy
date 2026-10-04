import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# PREPARED TEMPLATE (T75), NOT LAUNCHED. EXP-P35-001 (PLE embedding, 25 fits) bounded pipeline adapted from run-p34-pipeline.py.
# Runs the smoke-verified code-8864389 (frozen files identical to probe commit 3b317f2); checks the P35 timing-probe gates
# (ple_arms_within + every_arm_within + family_within + complete/zero exit); OMP/MKL/OPENBLAS 2 (same env as the probe).
# configsha must be filled from make_p35_config.py output at registration (after Song's explicit go); the placeholder fails closed.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
code=b/'code-8864389/experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
configsha='FILL_AT_REGISTRATION'
assert len(configsha)==64,'template: configsha not filled'
draftsha='f0ae4e9116d3efe90d119fcc84616172bde183a981206caacc4804d68a31b9b2'
probe=b/'runs/DRAFT-PLE-RESIDUAL-probe-3b317f2-timing/summary.json'
probesha='311f0d4f5dff318320a6da0d8944e6d7b50fdc1a1eb8ee4cd7a95d754fb67ff5'
state=b/'score-p35-state.json'
if state.exists():raise SystemExit('already started')
child=None

def save(status,**kw):
 tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));tmp.replace(state)

def interrupted(signum,frame):raise RuntimeError('pipeline interrupted by signal '+str(signum))
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)

def execute(name,cmd,timeout):
 global child
 with (b/(name+'.log')).open('x') as log:
  child=subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'},start_new_session=True)
  save(name,child_pid=child.pid,command=cmd)
  ret=child.wait(timeout=timeout)
  if ret:raise RuntimeError(name+' failed, exit '+str(ret))
 child=None

def queue(config,sha,output,queue_dir):
 return [python,str(code/'scripts/cuda_direct_queue.py'),'--config',str(config),'--config-sha256',sha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--output',str(output),'--queue-dir',str(queue_dir),'--lock-dir','/home/sds733/pitcheezy-autonomy/gpu-locks','--gpu','GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9','--gpu','GPU-ae803a24-0d38-791d-ed7c-3f233fcecf99','run']

try:
 save('verifying_completed_smoke_and_probe')
 draft=b/'code-8864389/configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json';cfg=b/'EXP-P35-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256((b/'code-3b317f2/configs/DRAFT-CUDA-PLE-RESIDUAL-20261004.json').read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules','exposure'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert hashlib.sha256(probe.read_bytes()).hexdigest()==probesha
 p=json.loads(probe.read_text())
 assert p['complete'] is True and set(p['exit_codes'].values())=={0},'timing probe incomplete'
 assert p['preflight']['config_sha256']==draftsha and p['preflight']['export_manifest_sha256']==manifest
 g=p['gate'];assert g['ple_arms_within'] is True and g['every_arm_within'] is True and g['family_within'] is True,'timing gate not passed'
 assert json.loads((b/'queue-ple-smoke-001/summary.json').read_text())['counts']=={'done':25},'all25 smoke tasks must finish'
 for q,n in (('p30-001',50),('p31-001',15),('p32-001',18),('p33-001',20),('p34-001',25),('p30-002',50)):
  assert json.loads((b/('queue-'+q+'/summary.json')).read_text())['counts']=={'done':n},q+' must be terminal'
 assert not (b/'queue-p35-001').exists() and not (b/'runs/EXP-P35-001').exists() and not (b/'scores/EXP-P35-001').exists(),'P35 output already present'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py|timing_probe_'],capture_output=True).stdout.strip(),'another queue/probe is running: start_after not satisfied'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 for arm in old['arms']:
  for seed in old['seeds']:
   runner.verify_task(runner.task_dir(b/'runs/DRAFT-PLE-smoke',arm['name'],seed,True),runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',runner.own_hashes()))
 execute('p35-full-001',queue(cfg,configsha,b/'runs/EXP-P35-001',b/'queue-p35-001'),40000)
 assert json.loads((b/'queue-p35-001/summary.json').read_text())['counts']=={'done':25},'full family incomplete'
 score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P35-001'),'--destination',str(b/'scores/EXP-P35-001')]
 execute('score-p35-001',score,900)
 result=json.loads((b/'scores/EXP-P35-001/score.json').read_text());assert result['family']=='EXP-P35-001' and result['decision'] is None
 save('complete',score=str(b/'scores/EXP-P35-001/score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
