import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# PREPARED TEMPLATE (P37 delivery, author_only), NOT LAUNCHED. EXP-P37-001 (embed-type, 15 fits, GPU3 only, 1 fit) bounded
# pipeline derived from run-p36-pipeline.TEMPLATE.py (save/interrupted/execute byte-identical). Checks the registered config,
# the passing TRAIN-only smoke gate summary, the 15 smoke identities (bound UUID set == {GPU3}); queue with exactly ONE
# --gpu GPU3 and the COMMON lock dir; score only after 15/15 and only if it can end by 22:00; else status 'incomplete'.
# Any FILL_AT_REGISTRATION placeholder fails closed. Server-side script only.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
configsha='FILL_AT_REGISTRATION'
assert len(configsha)==64,'template: configsha not filled'
codecommit='FILL_AT_REGISTRATION'  # smoke-verified P37 archive commit (= registration.source_commit)
gatesha='FILL_AT_REGISTRATION'  # = registration.smoke.gate_summary_sha256
smokeq='FILL_AT_REGISTRATION'  # P37 smoke queue dir (15/15 done)
smokeruns='FILL_AT_REGISTRATION'  # P37 smoke runs dir (ends -smoke)
assert 'FILL_AT_REGISTRATION' not in (codecommit,gatesha,smokeq,smokeruns),'template: placeholder not filled'
code=b/('code-'+codecommit)/'experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
draftsha='75265fc715f0fec998f609f0316ffe1de486c609677588d0fbd7241fc427ace2'
GPU3='GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'
LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'  # COMMON exclusive per-GPU lock dir (never the per-queue default)
SCORE_SECONDS=900;SCORE_END='22:00'  # m2: kill 21:45 + scorer 900 s <= 22:00; report incomplete by 22:30 (< 23:00)
gate=b/'p37-smoke-gate.json'
state=b/'score-p37-state.json'
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
 cmd=[python,str(code/'scripts/cuda_direct_queue.py'),'--config',str(config),'--config-sha256',sha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--output',str(output),'--queue-dir',str(queue_dir),'--lock-dir',LOCKS,'--gpu',GPU3,'run']
 assert cmd.count('--gpu')==1 and cmd[cmd.index('--gpu')+1]==GPU3 and cmd.count('--lock-dir')==1,'exactly one --gpu GPU3 and the common lock dir'
 return cmd

def clock(q,now,projection):
 """Launch/score bounds (KST, one date). Returns (queue timeout s, latest score start)."""
 kst=now.tzinfo;at=lambda hhmm:datetime.datetime.strptime(q['date']+' '+hhmm,'%Y-%m-%d %H:%M').replace(tzinfo=kst)
 assert now.date().isoformat()==q['date'],'actual launch is not on queue.date'
 assert now<=at(q['launch_by']),'actual launch after launch_by (19:00)'
 assert now+datetime.timedelta(seconds=projection)<=at(q['no_new_fit_after']),'projection does not finish before no_new_fit_after'
 assert at(q['kill_own_after'])+datetime.timedelta(seconds=SCORE_SECONDS)<=at(SCORE_END)<=at(q['aggregate_by']),'score window'
 return int((at(q['kill_own_after'])-now).total_seconds())+300,at(SCORE_END)-datetime.timedelta(seconds=SCORE_SECONDS)

try:
 save('verifying_registration_gate_and_smoke')
 draft=b/('code-'+codecommit)/'configs/DRAFT-CUDA-EMBED-TYPE-20261005.json';cfg=b/'EXP-P37-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules','exposure','queue'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert new['status']=='REGISTERED' and new['execution']['enabled'] is True
 assert new['device']['allowed_gpu_uuids']==[GPU3] and new['device']['max_parallel']==1,'device must be GPU3 only, 1 fit'
 reg=new['registration'];assert reg['source_commit']==codecommit and reg['smoke']['gate_summary_sha256']==gatesha and reg['draft_sha256']==draftsha
 assert hashlib.sha256(gate.read_bytes()).hexdigest()==gatesha
 g=json.loads(gate.read_text())
 assert g['protocol']=='p37_train_smoke_gate_v1' and g['pass'] is True and g['errors']==[] and g['tasks']==15 and g['bound_uuids']==[GPU3],'smoke gate not passed'
 assert g['config_sha256']==draftsha and g['export_manifest_sha256']==manifest and g['linear_over_periodic']<=g['ratio_max']==1.5
 assert json.loads((b/smokeq/'summary.json').read_text())['counts']=={'done':15},'all 15 smoke tasks must finish'
 assert json.loads((b/'queue-p36-001/summary.json').read_text())['counts']=={'done':20},'P36 must be terminal'
 assert not (b/'queue-p37-001').exists() and not (b/'runs/EXP-P37-001').exists() and not (b/'scores/EXP-P37-001').exists(),'P37 output already present'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py|timing_probe_|score_cuda_direct'],capture_output=True).stdout.strip(),'another queue/probe/scorer is running: start_after not satisfied'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 from pitchmdp import direct_embed_type_registry as et
 assert et.DEVICE['allowed_gpu_uuids']==[GPU3] and GPU3 not in et.RELEASED_GPUS
 sources=runner.own_hashes();assert sources==g['source_sha256'],'sources differ from the smoke gate'
 uuids=set()
 for arm in old['arms']:
  for seed in old['seeds']:
   d=runner.task_dir(b/smokeruns,arm['name'],seed,True)
   runner.verify_task(d,runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',sources))
   uuids.add(json.loads((d/'fit.json').read_text())['runtime']['bound_uuid'])
 assert uuids=={GPU3}==set(old['device']['allowed_gpu_uuids']),'smoke did not use exactly {GPU3}'
 kst=datetime.timezone(datetime.timedelta(hours=9))
 qtimeout,score_by=clock(new['queue'],datetime.datetime.now(kst),g['family_projection_seconds'])
 execute('p37-full-001',queue(cfg,configsha,b/'runs/EXP-P37-001',b/'queue-p37-001'),qtimeout)
 counts=json.loads((b/'queue-p37-001/summary.json').read_text())['counts']
 if counts!={'done':15}:
  save('incomplete',counts=counts,note='not scored (scoring only after 15/15); report incomplete by 22:30')
 elif datetime.datetime.now(kst)>score_by:
  save('incomplete',counts=counts,note='15/15 but scorer cannot end by 22:00; not scored; report by 22:30')
 else:
  score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P37-001'),'--destination',str(b/'scores/EXP-P37-001')]
  execute('score-p37-001',score,SCORE_SECONDS)
  result=json.loads((b/'scores/EXP-P37-001/score.json').read_text());assert result['family']=='EXP-P37-001' and result['decision'] is None
  save('complete',score=str(b/'scores/EXP-P37-001/score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
