import os,json,subprocess,pathlib,datetime,signal,hashlib,sys
# PREPARED TEMPLATE (P43 batter-ID, author_only next-research worker), NOT LAUNCHED. EXP-P43-001 (5 arms x seeds 70-72 =
# 15 fits) or the clock-only extension EXP-P43-002 (5 arms x seeds 73-74 = 10 fits, only after EXP-P43-001 is complete);
# GPU3 only, 1 fit. Derived from run-p40-pipeline.TEMPLATE.py (save/interrupted/execute/queue unchanged). Checks
# the registered config against the DATED DRAFT (p43_launch dated-check), the batter sidecar on the server (sha pinned),
# the VALID TRAIN-only smoke check of every task and its source map, no other Pitcheezy queue / probe / scorer (P40 / P42
# finished), GPU3 without any compute process; queue with exactly ONE --gpu GPU3 and the COMMON lock dir; score only
# after all tasks and only if it can end before aggregate_by - 5 min; else status 'incomplete'. Placeholders fail closed.
b=pathlib.Path('/home/sds733/pitcheezy-autonomy/20261004')
configsha='FILL_AT_REGISTRATION'
assert len(configsha)==64,'template: configsha not filled'
codecommit='FILL_AT_REGISTRATION'  # smoke-verified P43 archive commit (= registration.source_commit)
smokesha='FILL_AT_REGISTRATION'  # = registration.smoke.check_sha256
familyid='FILL_AT_REGISTRATION'  # 'EXP-P43-001' (main) or 'EXP-P43-002' (extension)
smokeq='FILL_AT_REGISTRATION'  # P43 smoke queue dir (all tasks done)
smokeruns='FILL_AT_REGISTRATION'  # P43 smoke runs dir (ends -smoke)
draftsha='FILL_AT_REGISTRATION'  # sha of the DATED DRAFT that was smoked (= registration.draft_sha256)
draftname='FILL_AT_REGISTRATION'  # its file name under code-<commit>/configs/
assert 'FILL_AT_REGISTRATION' not in (familyid,codecommit,smokesha,smokeq,smokeruns,draftsha,draftname),'template: placeholder not filled'
TASKS={'EXP-P43-001':15,'EXP-P43-002':10}[familyid];tag=familyid[-3:]  # 001 / 002
code=b/('code-'+codecommit)/'experiments/pitchmdp'
python='/home/sds733/venv/bin/python'
manifest='49097109b4f522a5a94d1557e00c52d88a8089c725977fe263c592214fb1f578'
UNDATED={'EXP-P43-001':('DRAFT-CUDA-BATTER-ID-20261006.json','bc94ae86a2482c920ebd3cb8a898774bcbbb4c5862d58639556621f4b7a442e2'),'EXP-P43-002':('DRAFT-CUDA-BATTER-ID-EXT-20261006.json','389c81b369a206460202aeda8b683a66da5df3e565dda0c380417e61f998782a')}
undatedname,undatedsha=UNDATED[familyid]
sidecarsha='ad2eacc64d46985a57527167486d7762a435f02d51fc5ed86a9e2e2644c160a4'
GPU3='GPU-c334d905-8ec9-0029-b206-46ce3ac49cd9'
LOCKS='/home/sds733/pitcheezy-autonomy/gpu-locks'  # COMMON exclusive per-GPU lock dir (never the per-queue default)
SCORE_SECONDS=1200  # 5 arms x 3 (or 2) seeds; scorer must end by aggregate_by - 5 min (clock()); else 'incomplete'
check=b/('p43-'+tag+'-smoke-check.json')
state=b/('score-p43-'+tag+'-state.json')
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
 save('verifying_registration_and_smoke')
 draft=b/('code-'+codecommit)/'configs'/draftname;cfg=b/(familyid+'.json')
 assert hashlib.sha256(draft.read_bytes()).hexdigest()==draftsha
 assert hashlib.sha256((b/('code-'+codecommit)/'configs'/undatedname).read_bytes()).hexdigest()==undatedsha
 assert hashlib.sha256((b/'batter-sidecar-49097109/sidecar.json').read_bytes()).hexdigest()==sidecarsha,'server batter sidecar missing or differs (rebuild with batter_sidecar.py; never edit)'
 assert hashlib.sha256(cfg.read_bytes()).hexdigest()==configsha
 old,new=json.loads(draft.read_text()),json.loads(cfg.read_text())
 assert {k for k in set(old)|set(new) if old.get(k)!=new.get(k)}=={'status','decision','execution','registration'},'registered config differs beyond registration paths'
 assert new['status']=='REGISTERED' and new['execution']=={'enabled':True,'executable':True}
 assert new['device']['allowed_gpu_uuids']==[GPU3] and new['device']['max_parallel']==1,'device must be GPU3 only, 1 fit'
 reg=new['registration'];assert reg['source_commit']==codecommit and reg['smoke']['check_sha256']==smokesha and reg['draft_sha256']==draftsha and reg['undated_draft_sha256']==undatedsha
 assert new['batter_sidecar']['sidecar_sha256']==sidecarsha
 assert hashlib.sha256(check.read_bytes()).hexdigest()==smokesha
 g=json.loads(check.read_text())
 assert g['protocol']=='p43_train_smoke_validity_v1' and g['valid'] is True and g['errors']==[] and g['tasks']==TASKS and g['bound_uuids']==[GPU3],'smoke check not valid'
 assert g['config_sha256']==draftsha and g['export_manifest_sha256']==manifest and g['batter_sidecar_sha256']==sidecarsha
 assert json.loads((b/smokeq/'summary.json').read_text())['counts']=={'done':TASKS},'all smoke tasks must finish'
 assert not (b/('queue-p43-'+tag)).exists() and not (b/'runs'/familyid).exists() and not (b/'scores'/familyid).exists(),'P43 output already present'
 if familyid=='EXP-P43-002':assert json.loads((b/'score-p43-001-state.json').read_text())['status']=='complete','extension only after EXP-P43-001 is complete'
 for other in ('score-p40-state.json','score-p42-state.json'):  # P40 / P42 must be finished (or stopped by their owner) first
  if (b/other).exists():assert json.loads((b/other).read_text())['status'] in ('complete','incomplete','blocked'),other+' still running'
 assert not subprocess.run(['pgrep','-f','cuda_direct_queue.py|timing_probe_|score_cuda_direct|run-p3|run-p40-pipeline|run-p42-pipeline|p38_resource_child|gpu3_bench_child'],capture_output=True).stdout.strip(),'another queue/probe/scorer/pipeline is running: start_after not satisfied'
 apps=subprocess.run(['nvidia-smi','--id='+GPU3,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout  # GPU3 only: no other GPU is queried
 assert GPU3 not in apps,'GPU3 has a compute process (foreign or ours): wait, never another GPU'
 sys.path[:0]=[str(code),str(code/'scripts')]
 import run_cuda_direct as runner
 from pitchmdp import direct_pitcher_id_registry as pid
 assert pid.DEVICE['allowed_gpu_uuids']==[GPU3] and GPU3 not in pid.RELEASED_GPUS
 sources=runner.own_hashes();assert sources==g['source_sha256']==reg['smoke']['source_sha256'],'sources differ from the smoke check'
 uuids=set()
 for arm in old['arms']:
  for seed in old['seeds']:
   d=runner.task_dir(b/smokeruns,arm['name'],seed,True)
   runner.verify_task(d,runner.expected_identity(old,manifest,arm['name'],seed,True,'cuda',sources))
   uuids.add(json.loads((d/'fit.json').read_text())['runtime']['bound_uuid'])
 assert uuids=={GPU3}==set(old['device']['allowed_gpu_uuids']),'smoke did not use exactly {GPU3}'
 kst=datetime.timezone(datetime.timedelta(hours=9))
 qtimeout,score_by=clock(new['queue'],datetime.datetime.now(kst),g['family_projection_seconds'])
 execute('p43-full-'+tag,queue(cfg,configsha,b/'runs'/familyid,b/('queue-p43-'+tag)),qtimeout)
 counts=json.loads((b/('queue-p43-'+tag)/'summary.json').read_text())['counts']
 if counts!={'done':TASKS}:
  save('incomplete',counts=counts,note='not scored (scoring only after all tasks); report incomplete by aggregate_by')
 elif datetime.datetime.now(kst)>score_by:
  save('incomplete',counts=counts,note='all tasks done but scorer cannot end before aggregate_by - 5 min; not scored')
 else:
  score=[python,str(code/'scripts/score_cuda_direct.py'),'--config',str(cfg),'--config-sha256',configsha,'--export',str(b/'export-49097109'),'--export-manifest-sha256',manifest,'--family',str(b/'runs'/familyid),'--destination',str(b/'scores'/familyid)]
  execute('score-p43-'+tag,score,SCORE_SECONDS)
  result=json.loads((b/'scores'/familyid/'score.json').read_text());assert result['family']==familyid and result['decision'] is None
  save('complete',score=str(b/'scores'/familyid/'score.json'))
except BaseException as exc:
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=60)
  except subprocess.TimeoutExpired:pass
 save('blocked',error=repr(exc),child_pid=child.pid if child else None)
 raise
