"""Boundary and paired-failure regressions; no physical transport."""
import copy
import io
import math
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from jolgwa_interfaces.msg import FlightCommandAck, VehicleControlState, AltitudeReferenceState
from jolgwa_ros.low_speed import AltitudeReferenceSynchronizer, accumulated_sample_age_s
from jolgwa_ros.mavlink_usb_bridge_node import MavlinkUsbBridgeNode as Bridge
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.mission_manager_node import MissionManagerNode as Manager
from test_safety_contract import terminal_fixture, controller_fixture
from test_safetycontract2 import bind


@pytest.mark.parametrize('final', [0, 2, 3, 4, 6])
def test_progress_retains_identity_until_final_result(final):
    c = terminal_fixture()
    c._pending_request_ids = {21:'land-1'}; c._sent_vehicle_commands = {21}
    c._terminal_attempts = 1
    c.get_logger = lambda: NS(error=lambda *a:None)
    bind(c, Controller, '_on_context_ack', '_on_command_ack')
    req = NS(mission_id='m', output_epoch=c._output_contract.output_epoch,
             output_sequence=c._output_contract.output_sequence, request_id='land-1')
    received=[]
    def receive(ack): received.append(ack.result); c._on_context_ack(ack)
    b=NS(_contract=c._output_contract, _pending_acks={21:(req,10_000_000_000)},
         _terminal_contract_started_ns=10_000_000_000,
         _context_ack_publisher=NS(publish=receive), _ack_publisher=NS(publish=lambda _:None),
         _journal=io.StringIO(), _timestamp_us=lambda:1)
    payload=dict(command=21,result=5,progress=50,result_param2=0,target_system=245,target_component=191)
    for seconds,result in [(10.1,5),(15.,5),(20.,final),(20.1,0)]:
        with patch('time.monotonic_ns',return_value=int(seconds*1e9)), patch('time.monotonic',return_value=seconds):
            Bridge._publish_mavlink_ack(b,{**payload,'result':result},1,1)
        if result==5:
            assert c._terminal_state==VehicleControlState.TERMINAL_PENDING
            assert c._pending_request_ids[21]=='land-1'
            c._last_px4_message_at=c._last_landed_message_at=seconds
            c._terminal_brake_transmitted=True
            c._supervise_terminal(seconds)
            assert c.commands==[]
    assert received==[5,5,final]
    assert c._terminal_state==(2 if final==0 else 3)
    assert 21 not in b._pending_acks and 21 not in c._pending_request_ids


@pytest.mark.parametrize('landed', [False, True])
def test_progress_with_no_final_ack_keeps_original_90_second_deadline(landed):
    c=terminal_fixture(); c._terminal_ack_in_progress=True
    c._external_mode_fenced=True; c._native_land=True
    c._landed=landed; c._gate.armed=not landed
    for now in (11., 50., 99.999):
        c._last_px4_message_at=c._last_landed_message_at=now
        c._supervise_terminal(now)
        assert c._terminal_state==VehicleControlState.TERMINAL_PENDING
    c._supervise_terminal(100.)
    assert c._terminal_state==VehicleControlState.TERMINAL_FAILED


@pytest.mark.parametrize('defect', ['mission','epoch','sequence','request','stale'])
def test_unrelated_or_late_ack_cannot_change_terminal(defect):
    c=terminal_fixture(); c._pending_request_ids={21:'r'}; c._sent_vehicle_commands={21}
    c._on_command_ack=lambda _:pytest.fail('unrelated ACK accepted')
    message=FlightCommandAck(mission_id='m', output_epoch=c._output_contract.output_epoch,
        output_sequence=c._output_contract.output_sequence,request_id='r',command=21,result=0,
        transmitted=True,received_monotonic_ns=10_000_000_000)
    if defect=='mission': message.mission_id='previous'
    if defect=='epoch': message.output_epoch='previous'
    if defect=='sequence': message.output_sequence+=1
    if defect=='request': message.request_id='previous'
    with patch('time.monotonic_ns',return_value=10_600_000_000 if defect=='stale' else 10_000_000_000):
        Controller._on_context_ack(c,message)
    assert c._pending_request_ids=={21:'r'}


@pytest.mark.parametrize('source,receipt,now,expected', [
    (300,10.,10.3,.6),(0,10.,10.5,.5),(0,10.,10.500001,.500001),
    (-1,10.,10.,math.inf),(math.nan,10.,10.,math.inf),(0,10.,9.,math.inf),
    (0,0.,10.,math.inf)])
def test_accumulated_sample_age(source,receipt,now,expected):
    assert accumulated_sample_age_s(source,receipt,now)==pytest.approx(expected)


def test_duplicate_and_reversed_altitude_messages_cannot_renew_age():
    import threading
    m=NS(_lock=threading.RLock(),_altitude_reference_state=None,_altitude_reference_state_received_at=0.)
    first=AltitudeReferenceState(transport_epoch=2,sequence=10)
    with patch('time.monotonic',return_value=10.): Manager._on_altitude_reference_state(m,first)
    for epoch,sequence in [(2,10),(2,9),(1,100)]:
        with patch('time.monotonic',return_value=20.):
            Manager._on_altitude_reference_state(m,AltitudeReferenceState(transport_epoch=epoch,sequence=sequence))
        assert m._altitude_reference_state_received_at==10.


@pytest.mark.parametrize('delta_ns,rejected',[(399_999_999,False),(400_000_000,False),(400_000_001,True)])
def test_home_deadline_is_checked_before_good_metadata(delta_ns,rejected):
    s=AltitudeReferenceSynchronizer()
    meta=dict(frozen_altitude_amsl_m=40.,frozen_z_ned_m=0.,current_altitude_amsl_m=40.,
        current_z_ned_m=0.,correction_valid=True,correction_revision=0,opposition_error_m=0.,
        estimator_reset_counter_valid=True,estimator_reset_counter=0,detail='locked',
        execution_home_mission_id='first',execution_home_lock_revision=1)
    s.set_home_reference(**meta)
    s._start_correction_event(10_000_000_000,-1.)
    with patch('time.monotonic_ns',return_value=10_000_000_000+delta_ns):
        accepted=s.set_home_reference(**{**meta,'correction_revision':1,'current_altitude_amsl_m':39.,'current_z_ned_m':1.})
    assert accepted != rejected
    assert s._epoch_failure_latched == rejected
    if not rejected:
        s.release_execution_home()
        assert s.set_provisional_home(altitude_amsl_m=39.,z_ned_m=1.,revision=2)
        new={**meta,'execution_home_mission_id':'second','execution_home_lock_revision':2,'provisional_home_revision':2,
             'frozen_altitude_amsl_m':39.,'frozen_z_ned_m':1.,'current_altitude_amsl_m':39.,'current_z_ned_m':1.}
        assert s.set_home_reference(**new)
        assert not s.set_home_reference(**meta)
        assert s._execution_home_mission_id=='second'


@pytest.mark.parametrize('fence', ['rc','reentry','manual','unapproved','other_mission'])
def test_terminal_exception_does_not_bypass_other_ownership_fences(fence):
    c=controller_fixture(); c._active_command.flight_profile=2
    c._gate=NS(armed=True,manual_override=fence=='manual',connected=True,
        preflight_checks_pass=True,is_approved=lambda _:fence!='unapproved')
    c._offboard=True; c._px4_failsafe_active=False
    c._autonomy_resume_required=True
    c._autonomy_reentry_required=fence=='reentry'; c._external_mode_fenced=fence=='rc'
    c._local_navigation_fresh=lambda _:True; c._receipt_age_usable=lambda *a:True
    c._status_timeout_s=.5; c._last_px4_message_at=10.
    bind(c,Controller,'_terminal_home_position_valid')
    assert not Controller._owns_terminal_handoff(c,10.,'old' if fence=='other_mission' else 'm', allow_app_resume=True)


def test_app_resume_exception_is_opt_in_for_land_only():
    c=controller_fixture(); c._active_command.flight_profile=2
    c._gate=NS(armed=True,manual_override=False,connected=True,
        preflight_checks_pass=True,is_approved=lambda _:True)
    c._offboard=True; c._px4_failsafe_active=False
    c._autonomy_resume_required=True
    c._autonomy_reentry_required=False; c._external_mode_fenced=False
    c._local_navigation_fresh=lambda _:True; c._receipt_age_usable=lambda *a:True
    c._status_timeout_s=.5; c._last_px4_message_at=10.
    bind(c,Controller,'_terminal_home_position_valid')
    assert not Controller._owns_terminal_handoff(c,10.,'m')
    assert Controller._owns_terminal_handoff(c,10.,'m',allow_app_resume=True)
