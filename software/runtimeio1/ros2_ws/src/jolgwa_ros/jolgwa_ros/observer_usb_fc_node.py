"""Observe-only USB telemetry adapter. No /fmu/* publishers or ROS command inputs."""
import json
from pathlib import Path
import time
import uuid
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import String
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity
from jolgwa_uav.observer_fc_telemetry import TelemetryState, TOPIC_PREFIX, STREAM_INTERVALS
from jolgwa_uav.observer_fc_transport import UsbTelemetryTransport
from jolgwa_uav.observer_fc_timesync import ObserveTimesync, KINDS
from jolgwa_uav.observer_fc_worker import ObserverFcWorker, ObserverRadioSender
from .evidence_clock import local_evidence_clock_id


class ObserverUsbFcNode(Node):
    def __init__(self):
        super().__init__('observer_usb_fc')
        readonly = ParameterDescriptor(read_only=True)
        self.declare_parameter('observe_timesync_enabled', False, readonly)
        self.declare_parameter('observe_timesync_protocol', 'targeted_v2', readonly)
        self.declare_parameter('observe_timesync_legacy_exclusive_link_attested', False, readonly)
        self.declare_parameter('observe_timesync_expected_flight_sw_version', 0, readonly)
        self.declare_parameter('observe_timesync_expected_custom_version_hex', '', readonly)
        self.declare_parameter('observer_radio_enabled', False, readonly)
        self.radio_sender = (ObserverRadioSender(local_evidence_clock_id())
            if self.get_parameter('observer_radio_enabled').value else None)
        self.radio_subscription = (self.create_subscription(String,
            '/jolgwa/observer/radio/status', lambda message: self.radio_sender.submit(message.data),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
            if self.radio_sender is not None else None)
        timing = ObserveTimesync(
            enabled=self.get_parameter('observe_timesync_enabled').value,
            protocol=self.get_parameter('observe_timesync_protocol').value,
            legacy_exclusive_link_attested=self.get_parameter('observe_timesync_legacy_exclusive_link_attested').value,
            expected_flight_sw_version=self.get_parameter('observe_timesync_expected_flight_sw_version').value,
            expected_custom_version_hex=self.get_parameter('observe_timesync_expected_custom_version_hex').value)
        # The opt-in worker exclusively owns mutable timing/state after start.
        # Default disabled mode keeps its original synchronous behavior.
        self.state = TelemetryState(timing=timing) if not timing.enabled else None
        self.worker = None
        self._next_worker_status_ns = 0
        self.timing_clock_id = local_evidence_clock_id() if timing.enabled else ''
        if timing.enabled and not self.timing_clock_id:
            raise ValueError('TIMESYNC observe requires a local monotonic clock identity')
        self.transport = None
        self.requested = False
        self.next_open = 0.
        self.connection_error = 'waiting_for_usb'
        self.positions = self.create_publisher(VehicleLocalPosition, TOPIC_PREFIX+'/vehicle_local_position', qos_profile_sensor_data)
        self.attitudes = self.create_publisher(VehicleAttitude, TOPIC_PREFIX+'/vehicle_attitude', qos_profile_sensor_data)
        self.rates = self.create_publisher(VehicleAngularVelocity, TOPIC_PREFIX+'/vehicle_angular_velocity', qos_profile_sensor_data)
        self.status = self.create_publisher(String, TOPIC_PREFIX+'/status', 1)
        self.timing_publisher = (self.create_publisher(String, TOPIC_PREFIX+'/timing_evidence', qos_profile_sensor_data)
                                 if timing.enabled else None)
        directory = Path('outputs/observer_fc')
        journal_path = directory / (time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex+'.jsonl')
        self.journal = None
        if timing.enabled:
            self.worker = ObserverFcWorker(timing=timing, journal_path=journal_path,
                                           evidence_clock_id=self.timing_clock_id,
                                           radio_sender=self.radio_sender)
        else:
            directory.mkdir(parents=True, exist_ok=True)
            self.journal = journal_path.open('x',buffering=1)
        self.create_timer(.01, self.poll)
        if self.worker is None:
            self.create_timer(.1, self.report)
        if self.worker is not None:
            self.worker.start()

    def poll(self):
        if self.worker is not None:
            # Keep status ahead of potentially blocking pose DDS calls. Latest
            # batch status is capped to 10 Hz here; source deadlines stay fixed.
            self.report()
            self._publish_worker_poses()
            return
        now = time.monotonic()
        if self.transport is None:
            if now < self.next_open:
                return
            self.next_open = now+2.
            try:
                self.transport = UsbTelemetryTransport(self.journal, timing=self.state.timing)
                self.connection_error = ''
            except (OSError, ValueError) as exc:
                self.connection_error = str(exc)
                if self.state.timing.mapper.fault:
                    # A retired epoch cannot recover by retrying this object.
                    # Exit so the existing supervisor restarts the whole session.
                    raise RuntimeError(self.state.timing.mapper.fault) from exc
                return
        # Any established link exception exits this component; the supervisor
        # restarts ALL observer components so old geometry/history is discarded.
        for message in self.transport.read():
            now = time.monotonic()
            kind = message.get_type()
            if kind == 'COMMAND_ACK' and (message.get_srcSystem(), message.get_srcComponent()) == (1,1):
                self.journal.write(json.dumps({'kind':'command_ack','data':message.to_dict()})+'\n')
            if not self.state.accept(kind, message.to_dict(), message.get_srcSystem(), message.get_srcComponent(), now,
                                     received_ns=getattr(self.transport, 'last_read_monotonic_ns', None)):
                if self.state.fault:
                    raise RuntimeError(self.state.fault)
                continue
            if kind == 'HEARTBEAT' and not self.requested:
                for message_id, interval in STREAM_INTERVALS.items():
                    self.transport.request(511,message_id,interval)
                self.transport.request(512,148)
                self.requested = True
            if self.state.timing.enabled and kind in KINDS:
                # No new lease at publication; failed evidence publication must
                # not let the associated pose bypass this opt-in proof path.
                now = time.monotonic()
                evidence = self.state.timing_evidence(kind, now, self.timing_clock_id)
                if evidence is None:
                    continue
                msg = String()
                msg.data = json.dumps(evidence, allow_nan=False)
                self.timing_publisher.publish(msg)
            if kind == 'ATTITUDE_QUATERNION':
                if self.state.timing.enabled:
                    now = time.monotonic()
                v = self.state.attitude(now)
                if v:
                    att = VehicleAttitude()
                    att.timestamp = att.timestamp_sample = v['timestamp']
                    att.q = v['q']
                    self.attitudes.publish(att)
                    rates = VehicleAngularVelocity()
                    rates.timestamp = rates.timestamp_sample = v['timestamp']
                    rates.xyz = v['xyz']
                    if not self.state.timing.enabled or self.state.attitude(time.monotonic()) is not None:
                        self.rates.publish(rates)
            if kind == 'LOCAL_POSITION_NED':
                if self.state.timing.enabled:
                    now = time.monotonic()
                v = self.state.position(now)
                if v:
                    pos = VehicleLocalPosition()
                    pos.timestamp = pos.timestamp_sample = v['timestamp']
                    pos.xy_valid = pos.z_valid = pos.v_xy_valid = pos.v_z_valid = v['valid']
                    for field in ('x','y','z','vx','vy','vz','heading'):
                        setattr(pos,field,float(v[field]))
                    self.positions.publish(pos)
        if self.state.timing.enabled and self.state.fresh('HEARTBEAT', time.monotonic(), 2.5):
            self.transport.request_timesync()
        if self.state.timing.mapper.fault:
            raise RuntimeError(self.state.timing.mapper.fault)
        if self.radio_sender is not None:
            self.radio_sender.poll(self.transport,
                heartbeat_fresh=self.state.fresh('HEARTBEAT', time.monotonic(), 2.5))

    def report(self):
        if self.worker is not None:
            mailbox = self.worker.mailbox
            mailbox.raise_if_failed()
            if time.monotonic_ns() < self._next_worker_status_ns:
                return
            snapshot = mailbox.take('status')
            if snapshot is not None:
                msg = String()
                msg.data = snapshot.payload_json
                if mailbox.is_current(snapshot, time.monotonic_ns()):
                    started_ns = time.monotonic_ns()
                    self.status.publish(msg)
                    mailbox.record_publication('status', started_ns, time.monotonic_ns())
                    self._next_worker_status_ns = started_ns + 100_000_000
            mailbox.raise_if_failed()
            return
        data = self.state.diagnostic(time.monotonic())
        data['connection_error'] = self.connection_error
        data['tx_telemetry_requests'] = self.transport.tx_count if self.transport else 0
        data['rx_bad_frames'] = self.transport.bad_data if self.transport else 0
        data['startup_discarded_messages'] = self.transport.startup_discarded_messages if self.transport else 0
        if self.radio_sender is not None:
            data['observer_radio'] = self.radio_sender.diagnostic(self.transport)
        serialized = json.dumps(data,allow_nan=False)
        self.journal.write(serialized+'\n')
        msg = String()
        msg.data = serialized
        self.status.publish(msg)

    def _publish_worker_poses(self):
        mailbox = self.worker.mailbox
        mailbox.raise_if_failed()
        for kind in ('ATTITUDE_QUATERNION', 'LOCAL_POSITION_NED', 'ESTIMATOR_STATUS'):
            snapshot = mailbox.take(kind)
            if snapshot is None or not mailbox.is_current(snapshot, time.monotonic_ns()):
                continue
            evidence = String()
            evidence.data = snapshot.evidence_json
            if not mailbox.is_current(snapshot, time.monotonic_ns()):
                continue
            started_ns = time.monotonic_ns()
            self.timing_publisher.publish(evidence)
            mailbox.record_publication('evidence', started_ns, time.monotonic_ns())
            # First DDS publication can block. Recheck ORIGINAL immutable expiry
            # and worker revocation after each publish; never call mutable state.
            value = json.loads(snapshot.payload_json)
            if kind == 'ATTITUDE_QUATERNION':
                att = VehicleAttitude()
                att.timestamp = att.timestamp_sample = value['timestamp']
                att.q = value['q']
                if mailbox.is_current(snapshot, time.monotonic_ns()):
                    started_ns = time.monotonic_ns()
                    self.attitudes.publish(att)
                    mailbox.record_publication('attitude', started_ns, time.monotonic_ns())
                rates = VehicleAngularVelocity()
                rates.timestamp = rates.timestamp_sample = value['timestamp']
                rates.xyz = value['xyz']
                if mailbox.is_current(snapshot, time.monotonic_ns()):
                    started_ns = time.monotonic_ns()
                    self.rates.publish(rates)
                    mailbox.record_publication('rates', started_ns, time.monotonic_ns())
            elif kind == 'LOCAL_POSITION_NED':
                pos = VehicleLocalPosition()
                pos.timestamp = pos.timestamp_sample = value['timestamp']
                pos.xy_valid = pos.z_valid = pos.v_xy_valid = pos.v_z_valid = value['valid']
                for field in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'heading'):
                    setattr(pos, field, float(value[field]))
                if mailbox.is_current(snapshot, time.monotonic_ns()):
                    started_ns = time.monotonic_ns()
                    self.positions.publish(pos)
                    mailbox.record_publication('position', started_ns, time.monotonic_ns())
        mailbox.raise_if_failed()

    def destroy_node(self):
        if self.radio_sender is not None:
            self.radio_sender.stop()
        if self.worker is not None:
            self.worker.stop()
        if self.transport:
            self.transport.close()
        if self.journal is not None:
            self.journal.close()
        return super().destroy_node()


def main():
    rclpy.init()
    node = ObserverUsbFcNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
