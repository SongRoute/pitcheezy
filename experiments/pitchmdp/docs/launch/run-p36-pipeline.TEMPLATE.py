import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# PREPARED TEMPLATE (T107, author_only), NOT LAUNCHED. EXP-P36-001 (robust-clip, 20 fits) bounded pipeline derived from run-p35-pipeline.TEMPLATE.py.
# Runs the smoke-verified fresh code dir code-<codecommit>; checks the P36 timing-probe gates (GATE_KEYS == make_p36_config.py, every
# gate boolean true + complete/zero exit); P35 terminal; OMP/MKL/OPENBLAS 2 (same env as the probe). Server-side script only.
# configsha and every other FILL_AT_REGISTRATION value are filled at registration (after the go); any placeholder fails closed.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
configsha='FILL_AT_REGISTRATION'
assert len(configsha)==64,'template: configsha not filled'
codecommit='FILL_AT_REGISTRATION'  # smoke-verified P36 archive commit (= registration.source_commit)
probedir='FILL_AT_REGISTRATION'  # P36 probe output dir under runs/ (…probe…-timing)
probesha='FILL_AT_REGISTRATION'  # = registration.timing_probe.summary_sha256
smokeq='FILL_AT_REGISTRATION'  # P36 smoke queue dir (20/20 done)
smokeruns='FILL_AT_REGISTRATION'  # P36 smoke runs dir
assert 'FILL_AT_REGISTRATION' not in (codecommit,probedir,probesha,smokeq,smokeruns),'template: placeholder not filled'
code=b/('code-'+codecommit)/'experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
draftsha='644239f3b2aaa4410391211ab1e4d972f13e1a6b025a026b058f0dff319c470a'
modulesha='a3d6cbcef8350586a23d1a6d35213e36ba8e0ccf21288288464f8ce7dcbf52fa'
GATE_KEYS=('robust_clip_arms_within','every_arm_within','family_within')
probe=b/'runs'/probedir/'summary.json'
state=b/'score-p36-state.json'
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
 save('verifying_p35_terminal_smoke_and_probe')
 draft=b/('code-'+codecommit)/'configs/DRAFT-CUDA-ROBUST-CLIP-20261005.json';cfg=b/'EXP-P36-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules','exposure'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert new['registration']['source_commit']==codecommit and new['registration']['timing_probe']['summary_sha256']==probesha
 assert hashlib.sha256((b/'EXP-P35-001.json').read_bytes()).hexdigest()==new['source_configs']['ple_registered']['sha256'],'EXP-P35-001.json differs from ple_registered'
 assert hashlib.sha256(probe.read_bytes()).hexdigest()==probesha
 p=json.loads(probe.read_text())
 assert p['complete'] is True and set(p['exit_codes'].values())=={0} and len(p['exit_codes'])==2,'timing probe incomplete'
 assert p['preflight']['config_sha256']==draftsha and p['preflight']['export_manifest_sha256']==manifest
 g=p['gate'];assert all(g[k] is True for k in GATE_KEYS) and all(v is True for v in g.values() if isinstance(v,bool)),'timing gate not passed'
 assert json.loads((b/smokeq/'summary.json').read_text())['counts']=={'done':20},'all 20 smoke tasks must finish'
 for q,n in (('p30-001',50),('p31-001',15),('p32-001',18),('p33-001',20),('p34-001',25),('p30-002',50),('p35-001',25)):
  assert json.loads((b/('queue-'+q+'/summary.json')).read_text())['counts']=={'done':n},q+' must be terminal'
 assert json.loads((b/'score-p35-state.json').read_text())['status']=='complete','P35 pipeline (scorer) not complete'
 assert not (b/'queue-p36-001').exists() and not (b/'runs/EXP-P36-001').exists() and not (b/'scores/EXP-P36-001').exists(),'P36 output already present'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py|timing_probe_|score_cuda_direct'],capture_output=True).stdout.strip(),'another queue/probe/scorer is running: start_after not satisfied'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 from pitchmdp import direct_robust_clip_registry as rcreg
 assert rcreg.MODULE_SHA256==modulesha and rcreg.check_module()==modulesha,'robust-clip module pin differs'
 sources=runner.own_hashes();assert sources['pitchmdp/direct_robust_clip_model.py']==modulesha
 uuids,stats=set(),set()
 for arm in old['arms']:
  for seed in old['seeds']:
   d=runner.task_dir(b/smokeruns,arm['name'],seed,True)
   runner.verify_task(d,runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',sources))
   fit=json.loads((d/'fit.json').read_text());uuids.add(fit['runtime']['bound_uuid'])
   if arm['name']=='robust_clip_residual_counts':
    assert fit['rc_module_sha256']==modulesha,'smoke robust-clip module sha differs';stats.add(fit['rc_stats_sha256'])
 assert uuids==set(old['device']['allowed_gpu_uuids']),'smoke did not use exactly both pinned GPUs'
 assert len(stats)==1,'smoke robust-clip statistics hash differs across seeds/GPUs'
 execute('p36-full-001',queue(cfg,configsha,b/'runs/EXP-P36-001',b/'queue-p36-001'),40000)
 assert json.loads((b/'queue-p36-001/summary.json').read_text())['counts']=={'done':20},'full family incomplete'
 score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P36-001'),'--destination',str(b/'scores/EXP-P36-001')]
 execute('score-p36-001',score,900)
 result=json.loads((b/'scores/EXP-P36-001/score.json').read_text());assert result['family']=='EXP-P36-001' and result['decision'] is None
 save('complete',score=str(b/'scores/EXP-P36-001/score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
