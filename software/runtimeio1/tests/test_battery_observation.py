import pytest
import json
from pathlib import Path
from jolgwa_uav.battery_policy import LowBatteryObservation, decode_battery
from jolgwa_uav.mavlink_route_fc_link import RouteFcState, ROUTE_STREAM_INTERVALS_US


def sample(ns, state=2, **extra):
    return dict(valid=True, received_ns=ns, charge_state=state, fault_bitmask=0, **extra)


@pytest.mark.parametrize('ms,terminal', [(4999, False), (5000, True), (5001, True)])
def test_low_deadline_and_late_ok(ms, terminal):
    p=LowBatteryObservation(); p.update(sample(0),0)
    result=p.update(sample(ms*10**6,1),ms*10**6)
    assert (result=='terminal') == terminal
    if terminal: assert p.update(sample((ms+2000)*10**6,1),(ms+2000)*10**6)=='terminal'


@pytest.mark.parametrize('ms,recovered', [(999,False),(1000,True),(1001,True)])
def test_continuous_ok_boundary(ms,recovered):
    p=LowBatteryObservation(); p.update(sample(0),0)
    for t in (200,400,600,800): p.update(sample(t*10**6,1),t*10**6)
    result=p.update(sample((200+ms)*10**6,1),(200+ms)*10**6)
    assert (result=='recovered')==recovered


def test_low_ok_chatter_does_not_extend_first_deadline():
    p=LowBatteryObservation()
    for i in range(25):
        ns=i*200_000_000
        assert p.update(sample(ns,2 if i%3==0 else 1),ns)=='observe'
        assert p.first_ns==0
    assert p.update(sample(5_000_000_000,1),5_000_000_000)=='terminal'


@pytest.mark.parametrize('state,fault',[(3,0),(4,0),(5,0),(6,0),(1,1),(2,4)])
def test_serious_no_grace(state,fault):
    p=LowBatteryObservation(); s=sample(1,state); s['fault_bitmask']=fault
    assert p.update(s,1)=='terminal'


@pytest.mark.parametrize('age,valid',[(749,True),(750,True),(751,False),(-1,False)])
def test_battery_evidence_age(age,valid):
    p=LowBatteryObservation()
    assert (p.update(sample(1_000_000_000),1_000_000_000+age*10**6,battery_fault=True)=='observe')==valid


def test_healthy_missing_or_unsupported_does_not_add_blocker():
    for s in (None,sample(0,0),sample(0,7)):
        assert LowBatteryObservation().update(s,0)=='ready'
        assert LowBatteryObservation().update(s,0,battery_fault=True)=='terminal'


def test_duplicate_ok_cannot_manufacture_recovery():
    p=LowBatteryObservation(); p.update(sample(0),0)
    s=sample(200_000_000,1)
    for now in (200_000_000,400_000_000,900_000_000): assert p.update(s,now)=='observe'
    assert p.update(s,1_200_000_000)=='terminal'


def test_first_low_uses_receipt_not_delayed_policy_callback():
    p=LowBatteryObservation()
    assert p.update(sample(1_000_000_000),1_600_000_000)=='observe'
    assert p.first_ns==1_000_000_000
    assert p.update(sample(6_000_000_000,1),6_000_000_000)=='terminal'


def test_unknown_voltage_does_not_erase_valid_critical_severity():
    s=decode_battery(dict(id=0,charge_state=3,fault_bitmask=0,voltages=[65535]*10),1,1)
    assert s['valid']
    assert LowBatteryObservation().update(s,1)=='terminal'


def test_decode_source_epoch_and_multiple_batteries():
    r=RouteFcState(); base=1_000_000_000
    data=dict(id=0,charge_state=2,fault_bitmask=0,voltages=[16000]+[65535]*9)
    assert ROUTE_STREAM_INTERVALS_US[147]==200000
    assert not r.accept('BATTERY_STATUS',data,2,1,base)
    assert r.accept('BATTERY_STATUS',data,1,1,base)
    assert r.snapshot(base).battery_sample['voltage_v']==16.
    assert not r.accept('BATTERY_STATUS',data,1,1,base)
    assert r.accept('BATTERY_STATUS',{**data,'id':1,'charge_state':1},1,1,base+1)
    assert r.snapshot(base+1).battery_sample is None
    r.transport_epoch+=1
    assert r.snapshot(base+1).battery_sample is None
    assert r.accept('BATTERY_STATUS',data,1,1,base+2)
    assert r.snapshot(base+2).battery_sample['epoch']==r.transport_epoch


@pytest.mark.parametrize('change',[dict(id=-1),dict(charge_state=True),dict(fault_bitmask=-1),dict(voltages=[float('nan')]*10)])
def test_malformed_battery_rejected(change):
    with pytest.raises(ValueError): decode_battery(dict(id=0,charge_state=1,fault_bitmask=0,voltages=[16000]+[65535]*9,**{})|change,1,1)


@pytest.mark.parametrize('index',[0,1])
def test_actual_ulog_56_57_transient_low_replay(index):
    record=json.loads((Path(__file__).parent/'fixtures/battery_flights_56_57.json').read_text())[index]
    p=LowBatteryObservation(); states=[]
    for row in record['samples']:
        ns=1_000_000_000+round(row['t_s']*1e9)
        # Firmware maps uORB warning 0..5 to MAV_BATTERY_CHARGE_STATE 1..6.
        s=sample(ns,int(row['warning'])+1)
        s['fault_bitmask']=int(row['faults'])
        states.append(p.update(s,ns,battery_fault=row['warning']!=0))
    assert 'observe' in states and 'recovered' in states
    assert 'terminal' not in states
