"""Slow storage must not block the flight callback or inference result loop."""
import io
import json
import threading
import time
import pytest
from jolgwa_ros.async_journal import AsyncJournal


class Disk:
    def __init__(self, blocked=False, fail=False):
        self.entered=threading.Event();self.release=threading.Event()
        if not blocked:self.release.set()
        self.fail=fail;self.rows=[];self.closed=False
    def write(self,text):
        self.entered.set()
        assert self.release.wait(5), 'test did not release blocked storage'
        if self.fail:raise OSError('disk full')
        self.rows.append(text)
        return len(text)
    def flush(self):pass
    def close(self):self.closed=True


def journal(disk,**kw):
    return AsyncJournal(disk,max_bytes=1024,max_records=8,
                        reserve_bytes=256,reserve_records=2,**kw)


def test_blocked_disk_does_not_block_new_records_or_flush_and_preserves_order():
    disk=Disk(blocked=True);j=journal(disk)
    j.write('first\n');assert disk.entered.wait(1)
    completed=threading.Event()
    def producer():
        j.write('second\n');j.flush();completed.set()
    t=threading.Thread(target=producer);t.start()
    try:
        assert completed.wait(.2), 'submission blocked on disk'
        assert j.status()['pending_records']==2
        assert j.status()['written']==0  # queued is never persisted/TX evidence
    finally:disk.release.set();t.join();j.close()
    assert disk.rows[:2]==['first\n','second\n']
    assert json.loads(disk.rows[-1])['pending_records']==0


def test_capacity_includes_inflight_and_reserves_critical_slots():
    disk=Disk(blocked=True);j=journal(disk)
    j.write('first\n');assert disk.entered.wait(1)
    try:
        for _ in range(5):assert j.write_diagnostic('sample\n')
        assert not j.write_diagnostic('dropped\n')
        j.write('critical1\n');j.write('critical2\n')
        with pytest.raises(OSError,match='queue_full'):j.write('overflow\n')
        assert j.status()['pending_records']==8
        assert j.status()['dropped_diagnostics']==1
    finally:disk.release.set();j.close()
    assert 'dropped\n' not in disk.rows
    assert any('journal_diagnostics_dropped' in x for x in disk.rows)


def test_utf8_byte_limit_is_bounded():
    disk=Disk(blocked=True);j=journal(disk)
    j.write('first\n');assert disk.entered.wait(1)
    try:
        assert not j.write_diagnostic('가'*300)
        with pytest.raises(OSError,match='queue_full'):j.write('가'*400)
        assert j.status()['pending_bytes']==6
    finally:disk.release.set();j.close()


def test_disk_failure_is_explicit_and_not_false_persistence():
    disk=Disk(fail=True);j=journal(disk);j.write('fail\n')
    j._thread.join(1)
    assert 'disk full' in j.status()['error']
    assert j.status()['written']==0
    with pytest.raises(OSError,match='disk full'):j.write('next\n')
    with pytest.raises(OSError,match='disk full'):j.flush()
    assert not j.write_diagnostic('best effort\n')
    with pytest.raises(OSError,match='disk full'):j.close()


def test_close_is_bounded_and_never_closes_stream_from_producer():
    disk=Disk(blocked=True);j=journal(disk,close_timeout_s=.02)
    j.write('held\n');assert disk.entered.wait(1)
    try:
        with pytest.raises(TimeoutError,match='not durable'):j.drain(timeout_s=.01)
        with pytest.raises(TimeoutError,match='not durable'):j.close()
        assert not disk.closed
        with pytest.raises(OSError,match='journal_closed'):j.write('late\n')
    finally:disk.release.set();j._thread.join(1)
    assert disk.closed


def test_production_inference_result_keeps_updating_while_disk_is_blocked(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from jolgwa_ros import scenario_incident_bench as mod
    b=mod.Bench(tmp_path);b.diagnostics.close()
    disk=Disk(blocked=True);b.diagnostics=AsyncJournal(disk)
    monkeypatch.setattr(mod,'annotated_jpeg',lambda *a,**k:b'jpeg')
    result=NS(modules={},detections=[],temporal_window_ready=True,events=[],inference_ms=20.)
    runtime=NS(health=lambda:dict(models={}))
    coverage=(lambda *a:({},True),lambda *a:True)
    def update(seq):
        frame=mod.Frame(seq,seq,time.monotonic(),b'jpeg')
        b.process_result(result,runtime,frame,image=None,cv2=None,coverage_functions=coverage)
    update(1);assert disk.entered.wait(1)
    done=threading.Event()
    t=threading.Thread(target=lambda:(update(2),done.set()));t.start()
    try:
        assert done.wait(.2)
        assert b.monitor_frame[0].sequence==2
        assert b.status=='RUNNING'
        assert b.diagnostics.status()['written']==0
    finally:disk.release.set();t.join();b.close()


def test_production_bridge_publishes_state_while_disk_is_blocked(tmp_path,monkeypatch):
    import rclpy
    from types import SimpleNamespace as NS
    from jolgwa_ros.mavlink_usb_bridge_node import MavlinkUsbBridgeNode
    monkeypatch.chdir(tmp_path);rclpy.init(args=[])
    b=MavlinkUsbBridgeNode();b._journal.close()
    disk=Disk(blocked=True);b._journal=AsyncJournal(disk)
    b._journal.write('blocked\n');assert disk.entered.wait(1)
    delivered=[];b._observer_status=NS(publish=lambda m:delivered.append(m.data))
    done=threading.Event();errors=[]
    def publish():
        try:
            for _ in range(3):b._publish_state()
        except Exception as e:errors.append(e)
        finally:done.set()
    t=threading.Thread(target=publish);t.start()
    try:
        assert done.wait(.2)
        assert not errors
        assert len(delivered)==3
        assert b._journal.status()['written']==0
    finally:disk.release.set();t.join();b.destroy_node();rclpy.shutdown()
