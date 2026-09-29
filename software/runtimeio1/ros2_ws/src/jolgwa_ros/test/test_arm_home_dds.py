"""Deliver ARM ACK and pending Home state over isolated localhost DDS."""
import threading
import time

from jolgwa_interfaces.msg import AltitudeReferenceState, FlightCommandAck
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from test_battery_landing import controller
from test_safetycontract2_dds import dds, deliver


def test_dds_accepted_arm_then_pending_home_keeps_disarmed_and_valid(dds):
    c = controller()
    c._gate.armed = False
    c._terminal_command_id = 21
    c._pending_request_ids = {400: 'arm-dds'}
    c._sent_vehicle_commands = {400}
    c._on_command_ack = Controller._on_command_ack.__get__(c)
    c._altitude_reference_state = None
    c._altitude_reference_state_received_at = 0.
    ack_event, altitude_event = threading.Event(), threading.Event()
    dds.create_subscription(FlightCommandAck, '/armhomeorder1/test/ack',
        lambda m: (c._on_context_ack(m), ack_event.set()), 10)
    dds.create_subscription(AltitudeReferenceState, '/armhomeorder1/test/altitude',
        lambda m: (Controller._on_altitude_reference_state(c, m), altitude_event.set()), 10)
    ap = dds.create_publisher(FlightCommandAck, '/armhomeorder1/test/ack', 10)
    hp = dds.create_publisher(AltitudeReferenceState, '/armhomeorder1/test/altitude', 10)
    deadline = time.monotonic()+3.
    while (ap.get_subscription_count() == 0 or hp.get_subscription_count() == 0) and time.monotonic() < deadline:
        time.sleep(.01)
    assert ap.get_subscription_count() and hp.get_subscription_count()
    c._arm_request_started_at = time.monotonic()
    contract = c._output_contract
    deliver(dds, ap, FlightCommandAck(mission_id='m', output_epoch=contract.output_epoch,
        output_sequence=contract.output_sequence, request_id='arm-dds', command=400,
        result=0, transmitted=True, received_monotonic_ns=time.monotonic_ns()), ack_event)
    deliver(dds, hp, AltitudeReferenceState(
        state=AltitudeReferenceState.STATE_HOME_CORRECTION_PENDING,
        valid=True, stable=True, transport_epoch=c._active_command.altitude_reference_epoch,
        sequence=1, local_z_ned_m=0., normalized_fc_altitude_home_relative_m=0.,
        home_correction_pending_age_ms=10), altitude_event)
    valid, detail, *_ = Controller._altitude_reference_status(c, c._active_command, time.monotonic())
    assert valid, detail
    assert not c._gate.armed and c._terminal_started_at is None
