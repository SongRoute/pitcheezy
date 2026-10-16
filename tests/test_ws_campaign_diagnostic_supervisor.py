"""Lock inheritance and bounded unreapable-child behavior without a GPU."""
import json
import subprocess
from types import SimpleNamespace
from experiments.ws_campaign import diagnostic_supervisor as ds


def test_lock_fd_passed_and_unreapable_child_stops(monkeypatch,tmp_path):
    class Child:
        pid=123; returncode=None
        def poll(self): return None
        def terminate(self): pass
        def kill(self): pass
        def wait(self,timeout):
            assert timeout == 5
            raise subprocess.TimeoutExpired('owned-child',timeout)
    calls=[]
    def popen(command,**kw):
        calls.append(kw); return Child()
    monkeypatch.setattr(ds.subprocess,'Popen',popen)
    result=ds.supervise(['generated-only'],tmp_path,0,1024,'digest',17)
    assert calls == [{'pass_fds':(17,)}]
    assert result == 87
    receipt=json.loads((tmp_path/'SUPERVISOR.json').read_text())
    assert receipt['reason'] == 'unreaped-child-lock-retained'
    assert receipt['child_reaped'] is False


def test_output_byte_cap_stops_only_owned_child(monkeypatch,tmp_path):
    class Child:
        pid=124; returncode=None
        def poll(self):return self.returncode
        def terminate(self):self.returncode=-15
        def wait(self,timeout):return self.returncode
    child=Child()
    monkeypatch.setattr(ds.subprocess,'Popen',lambda *a,**kw:child)
    (tmp_path/'generated.bin').write_bytes(b'oversize')
    assert ds.supervise(['generated-only'],tmp_path,60,1024,'digest',17,output_cap=1)==87
    receipt=json.loads((tmp_path/'SUPERVISOR.json').read_text())
    assert receipt['reason']=='output-bytes' and receipt['child_reaped'] is True


def test_finished_child_cannot_bypass_output_cap(monkeypatch,tmp_path):
    class Child:
        pid=125; returncode=0
        def poll(self):return 0
    monkeypatch.setattr(ds.subprocess,'Popen',lambda *a,**kw:Child())
    (tmp_path/'receipt.json').write_text(json.dumps({'status':'COMPLETED'}))
    assert ds.supervise(['generated-only'],tmp_path,60,1024,'digest',17,output_cap=1)==87
    receipt=json.loads((tmp_path/'SUPERVISOR.json').read_text())
    assert receipt['reason']=='output-bytes' and receipt['status']=='INCOMPLETE'
