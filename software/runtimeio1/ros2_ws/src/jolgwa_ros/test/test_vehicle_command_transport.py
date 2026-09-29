"""Generated ROS requests through production Bridge and MAVLink write guards.

Only the serial port and, by default, wire encoder are in memory. Set
JOLGWA_TEST_PYMAVLINK=1 to require real MAVLink 2 encoding/decoding as well.
No USB device is opened and no flight process is started.
"""
from dataclasses import replace
import io
import json
import math
import os
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from rclpy.executors import SingleThreadedExecutor
from rclpy.serialization import serialize_message, deserialize_message
from jolgwa_interfaces.msg import FlightCommandRequest
from jolgwa_uav.mavlink_route_fc_link import MavlinkRouteTransport, RouteFcSnapshot
from jolgwa_ros.px4_offboard_controller import Px4OffboardController as Controller
from jolgwa_ros.topic_names import FLIGHT_COMMAND_REQUEST
from test_safety_contract import bridge, contract, pair


def memory_transport():
    t = object.__new__(MavlinkRouteTransport)
    t.flight_output_enabled = True
    t._owner_thread = threading.get_ident()
    t.last_setpoint_ns = 0
    t.tx_count = 0
    t.journal = io.StringIO()
    t.packets = []
    t.port = NS(write=lambda packet: t.packets.append(packet) or len(packet))
    t.close = lambda: None
    if os.environ.get('JOLGWA_TEST_PYMAVLINK') == '1':
        from pymavlink.dialects.v20 import common
        t.common = common
        t.encoder = common.MAVLink(None, srcSystem=245, srcComponent=191)
    else:
        def message(*args):
            return NS(pack=lambda encoder, force_mavlink1: b'memory-packet')
        t.common = NS(MAVLink_command_long_message=message,
            MAVLink_set_position_target_local_ned_message=message)
        t.encoder = NS(seq=0)
    return t


def fc_snapshot(**changes):
    return replace(RouteFcSnapshot(connected=True, preflight_checks_pass=True,
        position_valid=True, armed=False, offboard=False, landed=True,
        failsafe=False, position=(0., 0., 0.), velocity=(0., 0., 0.), yaw=0.,
        time_boot_ms=1000, position_source='LOCAL_POSITION_NED',
        position_received_ns=time.monotonic_ns(), transport_epoch=1), **changes)


def prepare(b, command, params):
    b._transport = memory_transport()
    terminal = command in (20, 21)
    contract(b, code=3 if command == 21 else 4 if command == 20 else 1)
    snapshot = fc_snapshot(armed=terminal, offboard=terminal or command == 400,
                           landed=not terminal)
    base = time.monotonic_ns()
    # Exercise production setpoint validation/write and one-second TX warmup.
    for i in range(21):
        now = base + i * 50_000_000
        pair(b, now)
        b._stream_setpoint(snapshot, now)
    published = []
    controller = NS(_usb_output_contract=True, _output_contract=b._contract,
        _external_mode_fenced=False, _pending_request_ids={},
        _sent_vehicle_commands=set(),
        _command_request_publisher=NS(publish=published.append))
    with patch('time.monotonic_ns', return_value=now):
        Controller._send_vehicle_command(controller, command,
            **{f'param{i}': value for i, value in enumerate(params, 1)})
    message = published[0]
    # Humble deserializes float64[7] to numpy.ndarray, not a Python list.
    message = deserialize_message(serialize_message(message), FlightCommandRequest)
    return snapshot, message, now


@pytest.mark.parametrize('command,params', [
    (176, [1., 6., 0., 0., 0., 0., 0.]),
    (400, [1., 0., 0., 0., 0., 0., 0.]),
    (400, [0.] * 7),
    (21, [0.] * 7),
    (20, [0.] * 7),
])
def test_ros_request_reaches_guarded_command_write(bridge, command, params):
    b = bridge
    snapshot, message, now = prepare(b, command, params)
    b._on_command_request(message)
    before = len(b._transport.packets)
    b._drain_command_queue(snapshot, now)
    assert len(b._transport.packets) == before + 1, b._transport.journal.getvalue()
    record = json.loads(b._transport.journal.getvalue().splitlines()[-1])
    assert record['kind'] == 'tx_route_vehicle_command'
    assert record['command'] == command and record['params'] == params
    assert b._pending_acks[command][0].request_id == message.request_id
    if os.environ.get('JOLGWA_TEST_PYMAVLINK') == '1':
        packet = b._transport.packets[-1]
        assert packet[0] == 0xFD
        decoded = b._transport.common.MAVLink(None).parse_buffer(packet)[0]
        assert decoded.get_type() == 'COMMAND_LONG'
        assert decoded.command == command
        assert decoded.target_system == decoded.target_component == 1
        assert [getattr(decoded, f'param{i}') for i in range(1, 8)] == params


@pytest.mark.parametrize('index', range(7))
@pytest.mark.parametrize('invalid', [math.nan, math.inf, -math.inf])
def test_nonfinite_ros_parameter_still_rejected(bridge, index, invalid):
    params = [1., 6., 0., 0., 0., 0., 0.]
    params[index] = invalid
    snapshot, message, now = prepare(bridge, 176, params)
    bridge._on_command_request(message)
    before = len(bridge._transport.packets)
    bridge._drain_command_queue(snapshot, now)
    assert len(bridge._transport.packets) == before
    assert not bridge._pending_acks


def test_dds_request_reaches_production_bridge_and_transport(bridge):
    b = bridge
    # Only the real request subscription runs: never allow _poll to open USB.
    for timer in b.timers:
        timer.cancel()
    executor = SingleThreadedExecutor()
    executor.add_node(b)
    publisher = b.create_publisher(FlightCommandRequest, FLIGHT_COMMAND_REQUEST, 10)
    try:
        deadline = time.monotonic() + 3.
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert publisher.get_subscription_count() > 0
        snapshot, message, now = prepare(b, 176, [1., 6., 0., 0., 0., 0., 0.])
        publisher.publish(message)
        deadline = time.monotonic() + 2.
        while not b._commands and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=.02)
        assert len(b._commands) == 1
        before = len(b._transport.packets)
        b._drain_command_queue(snapshot, now)
        assert len(b._transport.packets) == before + 1
        assert b._pending_acks[176][0].request_id == message.request_id
    finally:
        executor.remove_node(b)
        executor.shutdown()


@pytest.mark.parametrize('fault', ['force_arm', 'mode', 'land_params',
    'disabled', 'preflight', 'old_epoch', 'expired'])
def test_normalization_does_not_bypass_command_guards(bridge, fault):
    command = 400 if fault == 'force_arm' else 21 if fault == 'land_params' else 176
    params = [0.] * 7
    if command == 176:
        params[:2] = [1., 3. if fault == 'mode' else 6.]
    elif command == 400:
        params[:2] = [1., 21196.]
    else:
        params[0] = 1.
    snapshot, message, now = prepare(bridge, command, params)
    if fault == 'disabled':
        bridge._transport.flight_output_enabled = False
    if fault == 'preflight':
        snapshot = replace(snapshot, preflight_checks_pass=False)
    if fault == 'old_epoch':
        message.output_epoch = 'retired'
    if fault == 'expired':
        message.issued_monotonic_ns = now - 1_000_000_000
    bridge._on_command_request(message)
    before = len(bridge._transport.packets)
    bridge._drain_command_queue(snapshot, now)
    assert len(bridge._transport.packets) == before
    assert not bridge._pending_acks
