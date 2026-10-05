import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# PREPARED TEMPLATE (P38 integration-002, author_only; p38-recovery-001 v2 resource check), NOT LAUNCHED. EXP-P38-001 (pitcher-ID, 10 fits, GPU3 only, 1 fit)
# bounded pipeline derived from run-p37-pipeline.TEMPLATE.py (save/interrupted/execute/queue byte-identical). Requires P37
# terminal (score-p37-state.json complete/incomplete/blocked) and GPU3 without any compute process; checks the registered
# config, the passing TRAIN-only resource gate summary, the 10 smoke identities (bound UUID set == {GPU3}); queue with
# exactly ONE --gpu GPU3 and the COMMON lock dir; score only after 10/10 and only if it can end before aggregate_by - 5 min;
# else status 'incomplete'. Any FILL_AT_REGISTRATION placeholder fails closed. Server-side script only.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
configsha='FILL_AT_REGISTRATION'
assert len(configsha)==64,'template: configsha not filled'
codecommit='FILL_AT_REGISTRATION'  # smoke-verified P38 archive commit (= registration.source_commit)
gatesha='FILL_AT_REGISTRATION'  # = registration.smoke.gate_summary_sha256
smokeq='FILL_AT_REGISTRATION'  # P38 smoke queue dir (10/10 done)
smokeruns='FILL_AT_REGISTRATION'  # P38 smoke runs dir (ends -smoke)
ressha='FILL_AT_REGISTRATION'  # = registration.resource.measurement_sha256 (v2 fixed-step TRAIN measurement, p38-recovery-001)
assert 'FILL_AT_REGISTRATION' not in (codecommit,gatesha,smokeq,smokeruns,ressha),'template: placeholder not filled'
code=b/('code-'+codecommit)/'experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
draftsha='8518f754236447fd054550dee75a5c3da7c510c3dd646f028662f097a02ee537'
GPU3='GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'
LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'  # COMMON exclusive per-GPU lock dir (never the per-queue default)
SCORE_SECONDS=900  # scorer must end by aggregate_by - 5 min (clock()); else 'incomplete'
gate=b/'p38-smoke-gate.json'  # v1 smoke gate: its VALIDITY checks are required; its resource ratio (REFUSED 1.2528) is kept as-is
resource=b/'p38-resource-v2-001/measure.json'  # the ONE v2 resource measurement; must pass (ratio <= 1.25, unchanged)
state=b/'score-p38-state.json'
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
 assert now<=at(q['launch_by']),'actual launch after launch_by'
 assert now+datetime.timedelta(seconds=projection)<=at(q['no_new_fit_after']),'projection does not finish before no_new_fit_after'
 score_end=at(q['aggregate_by'])-datetime.timedelta(minutes=5)
 assert at(q['no_new_fit_after'])<at(q['kill_own_after'])<score_end,'score window'
 return int((at(q['kill_own_after'])-now).total_seconds())+300,score_end-datetime.timedelta(seconds=SCORE_SECONDS)

try:
 save('verifying_registration_gate_and_smoke')
 draft=b/('code-'+codecommit)/'configs/DRAFT-CUDA-PITCHER-ID-20261005.json';cfg=b/'EXP-P38-001.json'
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 for key in ('arms','export','training','calibration','comparison','seeds','arm_notes','fits','device','data_rules','pitcher_id_data_rule','pitcher_id_family','exposure','queue'):
  assert old[key]==new[key],key+' differs from smoke configuration'
 assert new['status']=='REGISTERED' and new['execution']=={'enabled':True,'executable':True}
 assert new['device']['allowed_gpu_uuids']==[GPU3] and new['device']['max_parallel']==1,'device must be GPU3 only, 1 fit'
 reg=new['registration'];assert reg['source_commit']==codecommit and reg['smoke']['gate_summary_sha256']==gatesha and reg['draft_sha256']==draftsha
 assert reg['resource']['measurement_sha256']==ressha and reg['resource']['protocol']=='p38_train_fixed_step_resource_v2'
 assert hashlib.sha256(gate.read_bytes()).hexdigest()==gatesha
 g=json.loads(gate.read_text())
 assert g['protocol']=='p38_train_smoke_gate_v1' and g['errors']==[] and g['tasks']==10 and g['bound_uuids']==[GPU3],'smoke gate validity not satisfied'
 assert g['config_sha256']==draftsha and g['export_manifest_sha256']==manifest and g['ratio_max']==1.25
 assert hashlib.sha256(resource.read_bytes()).hexdigest()==ressha
 r=json.loads(resource.read_text());rd=r['decision']
 assert r['protocol']==rd['protocol']=='p38_train_fixed_step_resource_v2' and rd['pass'] is True and rd['errors']==[],'v2 resource measurement not passed'
 assert rd['candidate_over_reference']<=rd['ratio_max']==1.25 and rd['bound_uuids']==[GPU3] and rd['tasks']==4
 assert r['serial']['max_live']==1,'v2 measurement was not serial'
 assert json.loads((b/smokeq/'summary.json').read_text())['counts']=={'done':10},'all 10 smoke tasks must finish'
 assert json.loads((b/'score-p37-state.json').read_text())['status'] in ('complete','incomplete','blocked'),'P37 must be terminal'
 assert not (b/'queue-p38-001').exists() and not (b/'runs/EXP-P38-001').exists() and not (b/'scores/EXP-P38-001').exists(),'P38 output already present'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py|timing_probe_|score_cuda_direct|run-p37-pipeline|p38_resource_child|gpu3_bench_child'],capture_output=True).stdout.strip(),'another queue/probe/scorer/pipeline is running: start_after not satisfied'
 apps=subprocess.run(['nvidia-smi','--id='+GPU3,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout  # GPU3 only: no other GPU is queried
 assert GPU3 not in apps,'GPU3 has a compute process (foreign or ours): wait, never another GPU'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 from pitchmdp import direct_pitcher_id_registry as pid
 assert pid.DEVICE['allowed_gpu_uuids']==[GPU3] and GPU3 not in pid.RELEASED_GPUS
 sources=runner.own_hashes();assert sources==g['source_sha256'],'sources differ from the smoke gate'
 assert json.loads((b/'p38-resource-v2-001/spec-copy.json').read_text())['spec']['pins']['source_sha256']==sources,'v2 measured other sources'
 uuids=set()
 for arm in old['arms']:
  for seed in old['seeds']:
   d=runner.task_dir(b/smokeruns,arm['name'],seed,True)
   runner.verify_task(d,runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',sources))
   uuids.add(json.loads((d/'fit.json').read_text())['runtime']['bound_uuid'])
 assert uuids=={GPU3}==set(old['device']['allowed_gpu_uuids']),'smoke did not use exactly {GPU3}'
 kst=datetime.timezone(datetime.timedelta(hours=9))
 qtimeout,score_by=clock(new['queue'],datetime.datetime.now(kst),g['family_projection_seconds'])
 execute('p38-full-001',queue(cfg,configsha,b/'runs/EXP-P38-001',b/'queue-p38-001'),qtimeout)
 counts=json.loads((b/'queue-p38-001/summary.json').read_text())['counts']
 if counts!={'done':10}:
  save('incomplete',counts=counts,note='not scored (scoring only after 10/10); report incomplete by aggregate_by')
 elif datetime.datetime.now(kst)>score_by:
  save('incomplete',counts=counts,note='10/10 but scorer cannot end before aggregate_by - 5 min; not scored')
 else:
  score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs/EXP-P38-001'),'--destination',str(b/'scores/EXP-P38-001')]
  execute('score-p38-001',score,SCORE_SECONDS)
  result=json.loads((b/'scores/EXP-P38-001/score.json').read_text());assert result['family']=='EXP-P38-001' and result['decision'] is None
  save('complete',score=str(b/'scores/EXP-P38-001/score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
