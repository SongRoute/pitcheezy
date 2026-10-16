"""U2: synthetic owned loopback only. Events hold real requests; no recorded feeds or external hosts."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest
from observer_app import publisher, upload_worker
from test_publisher import make, world  # pytest world depends on the imported make fixture
from test_publisher_upload import Endpoint, TOKEN, GAME, HEALTH, config


@pytest.fixture(autouse=True)
def synthetic_boundary(monkeypatch):
    import test_publisher, live_fixture
    for key in list(os.environ):
        if key.startswith('PITCHEEZY_'):monkeypatch.delenv(key)
    def forbidden(*args, **kwargs):raise AssertionError('recorded fixture access forbidden')
    monkeypatch.setattr(test_publisher,'listed_dir',forbidden)
    monkeypatch.setattr(live_fixture,'listed_dir',forbidden)


@pytest.fixture
def server():
    endpoint = Endpoint()
    endpoint.entered, endpoint.release = threading.Event(), threading.Event()
    endpoint.hold, endpoint.after_store = None, False
    endpoint.arrivals = []
    guard = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self): self.handle_request()
        def do_PUT(self): self.handle_request()
        def do_DELETE(self): self.handle_request()
        def handle_request(self):
            endpoint.arrivals.append((self.command,self.path.removeprefix('/channel/')))
            data = self.rfile.read(int(self.headers.get('Content-Length',0))) if self.command=='PUT' else None
            hold = endpoint.hold == (self.command,self.path.removeprefix('/channel/'))
            if hold and not endpoint.after_store:
                endpoint.entered.set();endpoint.release.wait(10)
            with guard:
                status=endpoint.handle(self.command,self.path,data,dict(self.headers))
                headers,body=dict(endpoint.response_headers),endpoint.response_body
            if hold and endpoint.after_store:
                endpoint.entered.set();endpoint.release.wait(10)
            try:
                self.send_response(status)
                for key,value in headers.items(): self.send_header(key,value)
                self.send_header('Content-Length',str(len(body)));self.end_headers()
                if self.command!='HEAD':self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    endpoint.url=f'http://127.0.0.1:{http.server_port}/channel'
    thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
    try:yield endpoint
    finally:
        endpoint.release.set();http.shutdown();http.server_close();thread.join(2)


def attach_background(world, server, monkeypatch, **options):
    monkeypatch.setenv('PITCHEEZY_PUBLISH_TOKEN',TOKEN)
    world.config['upload']=config(server,background=True,operation_deadline_s=3,shutdown_grace_s=0,**options)
    world.start()
    return world.publisher.sink


def until(predicate, seconds=10):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.01)
    raise AssertionError('bounded wait expired')


def test_held_upload_does_not_block_local_cycle_and_health_is_partial(world, server, monkeypatch):
    server.hold=('PUT',GAME)
    sink=attach_background(world,server,monkeypatch)
    world.cycle(settle=False)
    assert server.entered.wait(5)
    before=world.publisher.state['sequence']
    completed=threading.Event()
    thread=threading.Thread(target=lambda:(world.cycle(settle=False),completed.set()))
    thread.start()
    assert completed.wait(2), 'local cycle blocked by held network request'
    thread.join()
    assert world.publisher.state['sequence']==before+1
    assert world.read(HEALTH)['upload']['pending']>0 and GAME in world.publisher.state['upload_pending']
    snapshots=list(sink.local.staging.glob('upload_*.tmp'))
    assert len(snapshots)==1 and snapshots[0].stat().st_size<=publisher.UPLOAD_MAX_BYTES
    sink.cleanup()
    assert snapshots[0].exists(), 'startup cleanup must not delete an active snapshot'
    server.release.set()
    until(lambda:GAME not in world.publisher.state['upload_pending'])
    assert not snapshots[0].exists()


def test_mailbox_coalesces_before_insertion_and_overflow_keeps_durable_pending(world, server, monkeypatch):
    server.hold=('PUT',GAME)
    sink=attach_background(world,server,monkeypatch)
    world.cycle(settle=False);assert server.entered.wait(5)
    with world.publisher._state_lock:
        for i in range(400):
            path=f'v1/game/{990000+i}.json'
            sink.write(path,json.dumps({'synthetic':i}).encode())
            sink.write(path,json.dumps({'synthetic':i,'newer':True}).encode())
        assert len(sink.worker.mailbox)==upload_worker.MAX_MAILBOX
        assert sink.worker.peak_entries==upload_worker.MAX_MAILBOX
        assert len(world.publisher.state['upload_pending'])>=400
        assert all(value is None for value in sink.worker.mailbox.values())
    server.release.set()


@pytest.mark.parametrize('after_store',[False,True])
def test_timeout_restart_preserves_uncertain_attempt_and_never_resends_old_bytes(world, server, monkeypatch, after_store):
    server.hold=('PUT',GAME);server.after_store=after_store
    sink=attach_background(world,server,monkeypatch)
    sink.worker.deadline=1
    sink.write(GAME,b'{"synthetic":true}');assert server.entered.wait(5)
    until(lambda:world.publisher.state['upload_attempts'].get(GAME,{}).get('phase')=='blocked',3)
    frozen=dict(world.publisher.state['upload_attempts'][GAME])
    assert sink.worker.process is None or not sink.worker.process.is_alive()
    world.start();server.release.set();server.hold=None
    before=server.arrivals.count(('PUT',GAME))
    world.cycle(settle=False)
    time.sleep(.2)
    assert world.publisher.state['upload_attempts'][GAME]==frozen
    assert server.arrivals.count(('PUT',GAME))==before


def test_close_reaps_owned_child_before_releasing_output_lock(world, server, monkeypatch):
    server.hold=('PUT',GAME)
    sink=attach_background(world,server,monkeypatch)
    world.cycle(settle=False);assert server.entered.wait(5)
    child=sink.worker.process
    assert child is not None and child.is_alive()
    assert not os.get_inheritable(world.publisher._lock.fileno())
    with pytest.raises(publisher.AlreadyRunning):
        publisher.Publisher(None,publisher.LocalDirSink(world.out),config=world.config)
    started=time.monotonic();world.publisher.close()
    assert time.monotonic()-started<3 and not sink.worker.thread.is_alive() and sink.worker.process is None
    replacement=publisher.Publisher(None,publisher.LocalDirSink(world.out),config=world.config)
    replacement.close()
    requests=len(server.requests);time.sleep(.1);assert len(server.requests)==requests
    assert not list(sink.local.staging.glob('upload_*.tmp'))
    server.release.set()


def test_head_missing_delete_does_not_erase_newer_put_intent(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    path='v1/game/991111.json';server.hold=('HEAD',path)
    sink.delete(path);assert server.entered.wait(5)
    sink.write(path,b'{"new":1}')
    server.release.set();server.hold=None
    until(lambda:path in server.objects)
    assert server.objects[path]==b'{"new":1}'


@pytest.mark.parametrize('before_newer',[False,True])
def test_delayed_cas_put_cannot_overwrite_newer_delete_restore(world, server, monkeypatch, before_newer):
    sink=attach_background(world,server,monkeypatch)
    sink.write(GAME,b'{"first":1}')
    until(lambda:GAME not in sink.pending)
    old_etag=server.tags[GAME]
    server.hold=('PUT',GAME);server.after_store=before_newer
    sink.write(GAME,b'{"delayed":1}');assert server.entered.wait(5)
    # A distinct fixed-condition request orders the remote completion explicitly; never unconditioned writes.
    remote=sink.remote
    server.hold=None  # only the already-entered old request remains held
    meta=remote.head(GAME)
    remote.delete(GAME,expected=meta['etag'])
    deleted=remote.head(GAME)
    remote.write(GAME,b'{"newest":3}',expected=deleted['etag'])
    latest_tag=server.tags[GAME]
    server.hold=None;server.release.set()
    until(lambda:sink.worker.process is None)
    assert server.objects[GAME]==b'{"newest":3}' and server.tags[GAME]==latest_tag
    if not before_newer:
        until(lambda:sink.state['upload_attempts'].get(GAME,{}).get('phase')=='blocked')
    assert latest_tag!=old_etag


def test_spawn_start_failure_is_reaped_and_keeps_pending(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    from multiprocessing.context import SpawnProcess
    def broken(*args,**kwargs):raise OSError('synthetic spawn failure')
    monkeypatch.setattr(SpawnProcess,'start',broken)
    sink.write(GAME,b'{}')
    until(lambda:not sink.worker.active and not sink.worker.mailbox)
    assert GAME in sink.pending and sink.worker.process is None
    world.publisher.close()


def test_oversized_ipc_frame_is_rejected_before_allocation():
    import multiprocessing
    receive,send=multiprocessing.Pipe(duplex=False)
    try:
        send.send_bytes(b'x'*(upload_worker.MAX_IPC+1))
        with pytest.raises(OSError):receive.recv_bytes(upload_worker.MAX_IPC)
    finally:receive.close();send.close()


def test_blocked_prefix_cannot_starve_later_eligible_path(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    with world.publisher._state_lock:
        for i in range(140):
            path=f'v1/game/{910000+i}.json'
            publisher._queue_upload(sink.state,path,'put',hashlib.sha256(b'{}').hexdigest())
            sink.state['upload_attempts'][path]={'phase':'blocked','reason':'legacy_unknown'}
        path='v1/game/999999.json'
        sink.local.write(path,b'{}')
        publisher._queue_upload(sink.state,path,'put',hashlib.sha256(b'{}').hexdigest())
        sink.save();sink.retry(HEALTH)
    until(lambda:path in server.objects)
    assert len(sink.pending)==140 and sink.worker.peak_entries==1


@pytest.mark.parametrize('new_action',['put','delete'])
def test_late_success_keeps_and_processes_newer_local_intent(world, server, monkeypatch, new_action):
    sink=attach_background(world,server,monkeypatch)
    server.hold=('PUT',GAME);server.after_store=True
    sink.write(GAME,b'{"old":1}');assert server.entered.wait(5)
    old_revision=sink.state['upload_intents'][GAME]['revision']
    if new_action=='put':sink.write(GAME,b'{"new":2}')
    else:sink.delete(GAME)
    assert sink.state['upload_intents'][GAME]['revision']>old_revision
    server.hold=None;server.release.set()
    until(lambda:GAME not in sink.pending)
    assert server.objects.get(GAME)==(b'{"new":2}' if new_action=='put' else None)
    assert sink.remote.head(GAME)['deleted']==(new_action=='delete')


def test_partially_started_child_is_reaped_on_start_exception(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    from multiprocessing.context import SpawnProcess
    original=SpawnProcess.start
    started=[]
    def partial(process):
        original(process);started.append(process.pid)
        raise OSError('synthetic error after child started')
    monkeypatch.setattr(SpawnProcess,'start',partial)
    sink.write(GAME,b'{}')
    until(lambda:not sink.worker.active and not sink.worker.mailbox)
    assert started and sink.worker.process is None and GAME in sink.pending
    for pid in started:
        with pytest.raises(ProcessLookupError):os.kill(pid,0)


def test_close_failure_retains_lock_and_blocks_new_local_writes(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    original=sink.worker.close
    def failure():raise publisher.UploadError('upload_worker_not_stopped')
    monkeypatch.setattr(sink.worker,'close',failure)
    with pytest.raises(publisher.UploadError):world.publisher.close()
    with pytest.raises(publisher.AlreadyRunning):
        publisher.Publisher(None,publisher.LocalDirSink(world.out),config=world.config)
    with pytest.raises(publisher.Unsafe):world.publisher.cycle()
    with pytest.raises(publisher.UploadError):sink.write(GAME,b'{}')
    monkeypatch.setattr(sink.worker,'close',original)
    world.publisher.close()


def test_start_initialization_exception_closes_created_publisher(world, monkeypatch):
    from types import SimpleNamespace
    def failure():raise RuntimeError('synthetic init failure')
    app=SimpleNamespace(state=SimpleNamespace(live_parts=failure))
    monkeypatch.setattr(publisher,'build',lambda *args,**kwargs:(app,world.publisher))
    args=SimpleNamespace(config=None,out=str(world.out),interval=None,delay=None,dry_run_upload=False,
        date=None,feed_dir=None,watch_dir=None,intent_dir=None,demo_dir=None)
    with pytest.raises(RuntimeError):publisher._start(args)
    assert world.publisher._lock is None
    with pytest.raises(publisher.Unsafe):world.publisher.cycle()


def test_queued_path_with_failed_local_delete_never_sends_remote_delete(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    sink.write(GAME,b'{"first":1}');until(lambda:GAME not in sink.pending)
    other='v1/game/999998.json';server.hold=('HEAD',other)
    sink.write(other,b'{}');assert server.entered.wait(5)
    sink.write(GAME,b'{"queued":2}')
    original=sink.local.delete
    def fail_game(path):
        if path==GAME:raise OSError('synthetic local unlink failure')
        return original(path)
    monkeypatch.setattr(sink.local,'delete',fail_game)
    with pytest.raises(OSError):sink.delete(GAME)
    server.hold=None;server.release.set()
    until(lambda:not sink.worker.active and not sink.worker.mailbox)
    assert sink.pending[GAME]=='delete' and sink.local.read(GAME)==b'{"queued":2}'
    assert ('DELETE',GAME) not in server.arrivals
    sink.attempted.clear();sink.retry(HEALTH)
    until(lambda:not sink.worker.active and not sink.worker.mailbox)
    assert ('DELETE',GAME) not in server.arrivals


def test_consumed_unreaped_error_preserves_handle_snapshot_and_output_lock(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    class Unreaped:
        pid=123  # simulated handle: methods never signal any OS process
        alive=True
        def is_alive(self):return self.alive
        def terminate(self):pass
        def kill(self):pass
        def join(self,*args):pass
        def close(self):assert not self.alive
    handle=Unreaped();calls=[];original=sink.worker.request
    def request(method,path,**options):
        calls.append((method,path))
        if method=='HEAD':return {'etag':None,'sha256':None,'deleted':False}
        sink.worker.process=handle
        raise publisher.UploadError('upload_worker_not_stopped',outcome='blocked')
    monkeypatch.setattr(sink.worker,'request',request)
    with sink.lock:
        sink.write(GAME,b'{}')
        sink.write('v1/game/999999.json',b'{}')
    until(lambda:not sink.worker.thread.is_alive())
    assert calls==[('HEAD',GAME),('PUT',GAME)] and sink.worker.process is handle
    assert sink.state['upload_attempts'][GAME]['phase']=='blocked'
    snapshots=list(sink.local.staging.glob('upload_*.tmp'));assert len(snapshots)==1
    with pytest.raises(publisher.UploadError,match='upload_worker_not_stopped'):
        original('HEAD','v1/game/999999.json')
    with pytest.raises(publisher.UploadError):world.publisher.close()
    with pytest.raises(publisher.AlreadyRunning):
        publisher.Publisher(None,publisher.LocalDirSink(world.out),config=world.config)
    handle.alive=False;sink.worker._reap();world.publisher.close()
    replacement=publisher.Publisher(None,publisher.LocalDirSink(world.out),config=world.config)
    assert not snapshots[0].exists()  # startup sweep only after the previous owner has stopped
    replacement.close()


def test_retry_cursor_passes_large_head_failure_prefix(world, server, monkeypatch):
    sink=attach_background(world,server,monkeypatch)
    healthy='v1/game/999999.json';original=sink.worker.request
    def request(method,path,**options):
        if path!=healthy:raise publisher.UploadError('upload_unknown',outcome='blocked')
        return original(method,path,**options)
    monkeypatch.setattr(sink.worker,'request',request)
    with sink.lock:
        for path in [f'v1/game/{910000+i}.json' for i in range(140)]+[healthy]:
            sink.local.write(path,b'{}')
            publisher._queue_upload(sink.state,path,'put',hashlib.sha256(b'{}').hexdigest())
        sink.save();sink.retry(HEALTH)
    until(lambda:not sink.worker.active and not sink.worker.mailbox)
    sink.attempted.clear();sink.retry(HEALTH)
    until(lambda:healthy in server.objects)
    assert len(sink.pending)==140
