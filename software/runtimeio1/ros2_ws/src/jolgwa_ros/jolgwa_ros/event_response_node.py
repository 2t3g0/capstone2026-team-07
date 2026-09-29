from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from jolgwa_interfaces.msg import EventObservation, MissionStatus
from jolgwa_interfaces.srv import ReleaseEventControl, RequestEventControl
from rclpy.executors import MultiThreadedExecutor
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from .airsim_camera import AirSimSceneCapture, AirSimVirtualGimbal
from .event_clip import EventClipPolicy, EventClipRecorder, EventClipResult, record_event_bounded
from .event_control import PHASE1_EVENT_TYPES, event_terminal_intent
from .evidence_clock import evidence_deadline_error
from .gimbal_tracking import (
    GimbalTargetTracker,
    GimbalTrackingPolicy,
)
from .ros_image_capture import FixedForwardGimbal, RosCompressedImageCapture
from .topic_names import (
    EVENT_OBSERVATION,
    MISSION_STATUS,
    RELEASE_EVENT_CONTROL_SERVICE,
    REQUEST_EVENT_CONTROL_SERVICE,
)


@dataclass(frozen=True)
class _ObservedEvent:
    event_id: str
    event_type: str
    track_id: str
    confidence: float
    bbox: tuple[float, float, float, float]
    target_visible: bool
    received_at: float
    source: str
    evidence_clock_id: str = ""
    evidence_expires_monotonic_ns: int = 0


@dataclass(frozen=True)
class _ActiveCapture:
    lease_id: str
    event: _ObservedEvent
    mission_id: str = ""
    request_source: str = ""


@dataclass
class _PendingRelease:
    lease_id: str
    capture_succeeded: bool
    detail: str
    last_attempt_at: float = 0.0
    in_flight: bool = False
    mission_id: str = ""
    request_source: str = ""


class EventResponseNode(Node):
    """Own an AirSim-backed bounded Phase 1 event-capture lease."""

    def __init__(self) -> None:
        super().__init__("event_response")
        for name, default in (("simple_obstacle_demo", False),
                              ("simple_demo_phase1_events", False),
                              ("diagnostic_relaxed_timing", False),
                              ("simulation_only", False), ("allow_real_hardware", False)):
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        self.declare_parameter("enabled", False)
        self.declare_parameter("recording_enabled", True)
        self.declare_parameter("gimbal_commands_enabled", True)
        self.declare_parameter("camera_backend", "airsim")
        self.declare_parameter("gimbal_backend", "airsim")
        self.declare_parameter(
            "ros_camera_topic", "/camera/front/compressed"
        )
        self.declare_parameter("allow_local_capture_without_control", True)
        self.declare_parameter(
            "storage_root",
            "/mnt/c/work/jolgwajetson/outputs/event_captures",
        )
        self.declare_parameter("capture_duration_s", 5.0)
        self.declare_parameter("capture_finalize_timeout_s", 2.0,
                               ParameterDescriptor(read_only=True))
        self.declare_parameter("completion_policy", "legacy_rejoin")
        self.declare_parameter("capture_fps", 25)
        self.declare_parameter("capture_bitrate_bps", 8_000_000)
        self.declare_parameter("airsim_host", "127.0.0.1")
        self.declare_parameter("airsim_port", 41451)
        self.declare_parameter("airsim_vehicle_name", "PatrolDrone")
        self.declare_parameter("airsim_camera_name", "front_center")
        self.declare_parameter("airsim_mount_x_m", 0.25)
        self.declare_parameter("airsim_mount_y_m", 0.0)
        self.declare_parameter("airsim_mount_z_m", 0.0)
        self.declare_parameter("yaw_min_deg", -90.0)
        self.declare_parameter("yaw_max_deg", 90.0)
        self.declare_parameter("pitch_min_deg", -80.0)
        self.declare_parameter("pitch_max_deg", 20.0)
        self.declare_parameter("max_yaw_rate_deg_s", 30.0)
        self.declare_parameter("max_pitch_rate_deg_s", 20.0)
        self.declare_parameter("target_lost_grace_s", 1.0)
        self.declare_parameter("lease_request_timeout_s", 2.0)
        self.declare_parameter("hold_ready_timeout_s", 4.0)
        self.declare_parameter("horizontal_fov_deg", 90.0)
        self.declare_parameter("vertical_fov_deg", 58.7)
        self.declare_parameter("request_source", "airsim-event-node")

        self._enabled = bool(self.get_parameter("enabled").value)
        self._completion_policy = str(self.get_parameter("completion_policy").value)
        self._simple_obstacle_demo = self.get_parameter("simple_obstacle_demo").value
        self._simple_demo_phase1_events = self.get_parameter("simple_demo_phase1_events").value
        if self._simple_demo_phase1_events and not self._simple_obstacle_demo:
            raise ValueError("simple_demo_phase1_events requires simple_obstacle_demo")
        if self._simple_obstacle_demo and not (
            self.get_parameter("diagnostic_relaxed_timing").value is True
            and self.get_parameter("simulation_only").value is True
            and self.get_parameter("allow_real_hardware").value is False
            and self.get_parameter("use_sim_time").value is True
            and self.get_parameter("camera_backend").value == "ros_compressed"
            and self.get_parameter("gimbal_backend").value == "fixed"
            and self._completion_policy == "capture_then_rtl"
        ):
            raise ValueError("simple_obstacle_demo requires the explicit SIM diagnostic camera capture/Home profile")
        self._capture_finalize_timeout_s = float(
            self.get_parameter("capture_finalize_timeout_s").value)
        if not 0.0 < self._capture_finalize_timeout_s <= 10.0:
            raise ValueError("capture_finalize_timeout_s must be finite and within (0, 10]")
        if self._capture_finalize_timeout_s != 2.0 and not (
            self._simple_obstacle_demo and self._simple_demo_phase1_events):
            raise ValueError("nondefault capture finalization budget requires the explicit SIM photo profile")
        event_terminal_intent(policy=self._completion_policy,
                              capture_succeeded=False, autonomy_available=False)
        self._terminal_event_id = ""
        self._terminal_mission_id = ""
        self._episode_mission_id = ""
        self._pending_mission_id = ""
        self._latest_status_received_at = float("-inf")
        self._latest_status_source_ns = 0
        self._last_lease_attempt_at = float("-inf")
        self._retired_release_outcomes = deque(maxlen=32)
        self._recording_enabled = bool(
            self.get_parameter("recording_enabled").value
        )
        self._request_source = str(
            self.get_parameter("request_source").value
        ).strip()
        if not self._request_source:
            raise ValueError("request_source must not be empty")
        camera_backend = str(
            self.get_parameter("camera_backend").value
        ).strip().lower()
        gimbal_backend = str(
            self.get_parameter("gimbal_backend").value
        ).strip().lower()
        if camera_backend not in {"airsim", "ros_compressed"}:
            raise ValueError("camera_backend must be airsim or ros_compressed")
        if gimbal_backend not in {"airsim", "fixed"}:
            raise ValueError("gimbal_backend must be airsim or fixed")
        self._camera_backend = camera_backend
        self._gimbal_backend = gimbal_backend
        self._allow_local_capture_without_control = bool(
            self.get_parameter("allow_local_capture_without_control").value
        )

        clip_policy = EventClipPolicy(
            duration_s=float(
                self.get_parameter("capture_duration_s").value
            ),
            fps=int(self.get_parameter("capture_fps").value),
            codec="h264",
            container="matroska",
            bitrate_bps=int(
                self.get_parameter("capture_bitrate_bps").value
            ),
        )
        self._recorder = EventClipRecorder(
            str(self.get_parameter("storage_root").value), clip_policy
        )
        tracking_policy = GimbalTrackingPolicy(
            yaw_min_deg=float(self.get_parameter("yaw_min_deg").value),
            yaw_max_deg=float(self.get_parameter("yaw_max_deg").value),
            pitch_min_deg=float(self.get_parameter("pitch_min_deg").value),
            pitch_max_deg=float(self.get_parameter("pitch_max_deg").value),
            max_yaw_rate_deg_s=float(
                self.get_parameter("max_yaw_rate_deg_s").value
            ),
            max_pitch_rate_deg_s=float(
                self.get_parameter("max_pitch_rate_deg_s").value
            ),
            target_lost_grace_s=float(
                self.get_parameter("target_lost_grace_s").value
            ),
            horizontal_fov_deg=float(
                self.get_parameter("horizontal_fov_deg").value
            ),
            vertical_fov_deg=float(
                self.get_parameter("vertical_fov_deg").value
            ),
        )
        self._tracker = GimbalTargetTracker(tracking_policy)
        airsim_host = str(self.get_parameter("airsim_host").value)
        airsim_port = int(self.get_parameter("airsim_port").value)
        vehicle_name = str(
            self.get_parameter("airsim_vehicle_name").value
        )
        camera_name = str(
            self.get_parameter("airsim_camera_name").value
        )
        if gimbal_backend == "airsim":
            self._gimbal = AirSimVirtualGimbal(
                host=airsim_host,
                port=airsim_port,
                vehicle_name=vehicle_name,
                camera_name=camera_name,
                mount_x_m=float(
                    self.get_parameter("airsim_mount_x_m").value
                ),
                mount_y_m=float(
                    self.get_parameter("airsim_mount_y_m").value
                ),
                mount_z_m=float(
                    self.get_parameter("airsim_mount_z_m").value
                ),
                enabled=bool(
                    self.get_parameter("gimbal_commands_enabled").value
                ),
            )
        else:
            self._gimbal = FixedForwardGimbal()

        self._lock = threading.Lock()
        self._pending: _ObservedEvent | None = None
        self._pending_started_at: float | None = None
        self._active: _ActiveCapture | None = None
        self._latest: _ObservedEvent | None = None
        self._latest_revision = 0
        self._processed_revision = 0
        self._capture_thread: threading.Thread | None = None
        self._recording_lease_id = ""
        self._active_wait_started_at: float | None = None
        self._pending_release: _PendingRelease | None = None
        self._latest_status: MissionStatus | None = None
        self._local_recording = False
        self._local_event: _ObservedEvent | None = None
        self._shutdown = threading.Event()
        self._queued_events: deque[_ObservedEvent] = deque()
        self._known_event_ids: set[str] = set()
        self._last_backend_error = ""
        self._backend_error_log_at = 0.0
        self._capture = None

        self._request_client = self.create_client(
            RequestEventControl, REQUEST_EVENT_CONTROL_SERVICE
        )
        self._release_client = self.create_client(
            ReleaseEventControl, RELEASE_EVENT_CONTROL_SERVICE
        )
        self.create_subscription(
            EventObservation,
            EVENT_OBSERVATION,
            self._on_observation,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            MissionStatus,
            MISSION_STATUS,
            self._on_mission_status,
            10,
        )
        # Target loss/lease expiration must not pause with simulation time.
        self._tracking_watchdog_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.05, self._on_gimbal_timer, clock=self._tracking_watchdog_clock)

        if self._enabled and self._recording_enabled:
            if camera_backend == "airsim":
                self._capture = AirSimSceneCapture(
                    host=airsim_host,
                    port=airsim_port,
                    vehicle_name=vehicle_name,
                    camera_name=camera_name,
                    fps=float(self.get_parameter("capture_fps").value),
                ).start()
            else:
                self._capture = RosCompressedImageCapture(
                    self,
                    str(self.get_parameter("ros_camera_topic").value),
                ).start()

        if not self._enabled:
            self.get_logger().warning(
                "event response node is inert (enabled=false)"
            )
        if gimbal_backend == "airsim" and not self._gimbal.enabled:
            self.get_logger().warning(
                "AirSim camera pose commands are inert "
                "(gimbal_commands_enabled=false)"
            )
        elif gimbal_backend == "fixed":
            self.get_logger().info(
                "fixed-forward Gazebo camera selected; tracking commands are diagnostic only"
            )

    def _on_observation(self, message: EventObservation) -> None:
        try:
            event = self._convert_observation(message)
        except ValueError as exc:
            self.get_logger().warning(f"invalid event observation: {exc}")
            return

        with self._lock:
            active_id = self._active.event.event_id if self._active else ""
            pending_id = self._pending.event_id if self._pending else ""
            local_id = (
                self._local_event.event_id if self._local_event else ""
            )
            if event.event_id in {active_id, pending_id, local_id}:
                self._latest = event
                self._latest_revision += 1
                return
            if not self._enabled or message.state != "CONFIRMED":
                return
            if self._completion_policy == "capture_then_rtl":
                if not self._mission_for_request_locked():
                    return
                if self._active is not None or self._pending is not None or self._local_recording:
                    return  # Do not reserve or queue another terminal episode.
            if event.event_id in self._known_event_ids:
                return
            self._known_event_ids.add(event.event_id)
            if (
                self._active is not None
                or self._pending is not None
                or self._local_recording
            ):
                self._queued_events.append(event)
                self.get_logger().info(
                    f"event {event.event_id} queued behind the active capture"
                )
                return

        self._dispatch_event(event)

    def _dispatch_event(self, event: _ObservedEvent) -> None:
        with self._lock:
            if self._shutdown.is_set():
                return
            mission_id = ""
            if self._completion_policy == "capture_then_rtl":
                mission_id = self._mission_for_request_locked()
                if (not mission_id or time.monotonic()-self._last_lease_attempt_at < .5
                        or evidence_deadline_error(event.evidence_clock_id, event.evidence_expires_monotonic_ns)):
                    self._known_event_ids.discard(event.event_id)
                    return
                self._last_lease_attempt_at = time.monotonic()
            if not self._request_client.service_is_ready():
                if self._completion_policy == "capture_then_rtl":
                    self._known_event_ids.discard(event.event_id)
                    self.get_logger().warning("event lease service unavailable; terminal episode not consumed")
                    return
                if not self._allow_local_capture_without_control:
                    self.get_logger().error(
                        "mission event-control service is not ready; fail-closed without recording"
                    )
                    return
                self.get_logger().warning(
                    "mission event-control service is not ready; recording locally"
                )
                self._local_recording = True
                self._latest = event
                self._latest_revision += 1
                self._start_local_recording(event, "service unavailable")
                return
            self._pending = event
            self._pending_mission_id = mission_id
            self._pending_started_at = time.monotonic()
            self._latest = event
            self._latest_revision += 1

        request = RequestEventControl.Request()
        request.event_id = event.event_id
        request.event_type = event.event_type
        request.track_id = event.track_id
        request.confidence = event.confidence
        request.source = self._event_lease_source(event)
        request.mission_id = mission_id
        request.evidence_clock_id = event.evidence_clock_id
        request.evidence_expires_monotonic_ns = event.evidence_expires_monotonic_ns
        try:
            future = self._request_client.call_async(request)
        except Exception as exc:
            with self._lock:
                if self._pending == event:
                    self._pending = None
                    self._pending_started_at = None
                    self._pending_mission_id = ""
                    self._known_event_ids.discard(event.event_id)
            self.get_logger().error(f"event lease dispatch failed without terminal latch: {exc}")
            return
        future.add_done_callback(
            lambda completed, expected=event, requested_mission=mission_id, requested_source=request.source: self._on_lease_response(
                completed, expected, requested_mission, requested_source
            )
        )

    def _event_lease_source(self, event: _ObservedEvent) -> str:
        return event.source if event.event_type == "DEMO_PERSON" else self._request_source

    def _on_lease_response(self, future, expected: _ObservedEvent, requested_mission="", requested_source="") -> None:
        try:
            response = future.result()
        except Exception as exc:
            with self._lock:
                if self._pending == expected:
                    self._pending = None
                    self._pending_started_at = None
                    self._pending_mission_id = ""
                    self._known_event_ids.discard(expected.event_id)
                    if self._allow_local_capture_without_control and self._completion_policy == "legacy_rejoin":
                        self._local_recording = True
                        self._start_local_recording(
                            expected, f"lease request failed: {exc}"
                        )
            suffix = (
                "; recording locally"
                if self._allow_local_capture_without_control and self._completion_policy == "legacy_rejoin"
                else "; fail-closed without recording"
            )
            self.get_logger().error(f"event lease request failed{suffix}: {exc}")
            return
        with self._lock:
            if self._pending != expected:
                return
            self._pending = None
            self._pending_started_at = None
            self._pending_mission_id = ""
            if not response.accepted:
                self._known_event_ids.discard(expected.event_id)
                if self._allow_local_capture_without_control and self._completion_policy == "legacy_rejoin":
                    self.get_logger().warning(
                        f"event lease rejected; recording locally: {response.message}"
                    )
                    self._local_recording = True
                    self._start_local_recording(expected, response.message)
                else:
                    self.get_logger().error(
                        f"event lease rejected; fail-closed without recording: {response.message}"
                    )
                return
            if self._completion_policy == "capture_then_rtl":
                if (not requested_mission or not response.lease_id
                        or getattr(response, "mission_id", "") != requested_mission
                        or requested_mission != self._episode_mission_id):
                    self.get_logger().error("accepted event lease has unverified mission identity; capture prohibited")
                    # No unrelated lease is released or used. Its manager-owned
                    # finite lease expires; this response is not authority proof.
                    self._known_event_ids.discard(expected.event_id)
                    return
                self._terminal_event_id = expected.event_id
                self._terminal_mission_id = requested_mission
                self._queued_events.clear()
                evidence_error = evidence_deadline_error(expected.evidence_clock_id, expected.evidence_expires_monotonic_ns)
                if evidence_error:
                    self._pending_release = _PendingRelease(response.lease_id, False,
                        "accepted lease response arrived after event evidence expired; terminal_intent=RTL",
                        mission_id=requested_mission,
                        request_source=requested_source or self._event_lease_source(expected))
                    return  # Timer releases; never start a stale-evidence clip.
            active = _ActiveCapture(response.lease_id, expected, requested_mission,
                                    requested_source or self._event_lease_source(expected))
            self._active = active
            self._active_wait_started_at = time.monotonic()
            start_capture = self._capture_is_ready_locked(active)
        self.get_logger().info(
            f"event lease {active.lease_id} accepted for {expected.event_type}"
        )
        if start_capture:
            self._start_capture(active)

    def _on_mission_status(self, message: MissionStatus) -> None:
        start_capture = None
        with self._lock:
            if self._completion_policy == "capture_then_rtl":
                source_ns = int(message.stamp.sec)*1_000_000_000+int(message.stamp.nanosec)
                if source_ns <= self._latest_status_source_ns:
                    return  # Old/duplicate status does not renew its lease.
                self._latest_status_source_ns = source_ns
                self._latest_status_received_at = time.monotonic()
                worker_busy = (self._active is not None or self._pending is not None or self._local_recording
                    or getattr(self._recorder, "_worker_unresolved", False)
                    or (self._capture_thread is not None and self._capture_thread.is_alive()))
                new_patrol = (message.mission_id and message.mission_id != self._episode_mission_id
                        and message.phase == MissionStatus.PHASE_PATROL
                        and message.approved and message.armed and message.offboard
                        and not message.manual_override)
                pending = self._pending_release
                if (new_patrol and not worker_busy and pending is not None
                        and self._terminal_mission_id == self._episode_mission_id
                        and getattr(pending, "mission_id", "") == self._episode_mission_id):
                    # New approved mission proves the old mission no longer
                    # owns this manager. Do not retry an old lease forever or
                    # relabel its capture outcome as success.
                    self._retired_release_outcomes.append(pending)
                    self.get_logger().warning(
                        f"new approved mission; retiring old lease release {pending.lease_id}; "
                        f"capture_succeeded={pending.capture_succeeded}; release unconfirmed")
                    self._pending_release = None
                if new_patrol and not worker_busy and self._pending_release is None:
                    self._episode_mission_id = message.mission_id
                    self._terminal_event_id = self._terminal_mission_id = ""
                    self._known_event_ids.clear()
                    self._queued_events.clear()
            self._latest_status = message
            if self._active is not None and self._capture_is_ready_locked(
                self._active
            ):
                start_capture = self._active
        if start_capture is not None:
            self._start_capture(start_capture)

    def _mission_for_request_locked(self):
        status = self._latest_status
        if (self._terminal_event_id or self._pending_release is not None
                or getattr(self._recorder, "_worker_unresolved", False)
                or status is None or not 0 <= time.monotonic()-self._latest_status_received_at <= 2.
                or not status.mission_id or status.mission_id != self._episode_mission_id
                or status.phase != MissionStatus.PHASE_PATROL or not status.approved
                or not status.armed or not status.offboard or status.manual_override):
            return ""
        return status.mission_id

    def _capture_is_ready_locked(self, active: _ActiveCapture) -> bool:
        status = self._latest_status
        if getattr(self, "_completion_policy", "legacy_rejoin") == "capture_then_rtl":
            if (status is None or status.mission_id != active.mission_id
                    or not 0 <= time.monotonic()-self._latest_status_received_at <= 2.
                    or not status.approved or status.manual_override
                    or not status.armed or not status.offboard):
                return False
        if (
            self._recording_lease_id
            or status is None
            or status.event_lease_id != active.lease_id
            or status.phase != MissionStatus.PHASE_EVENT_CAPTURE
            or status.control_owner != "JETSON_EVENT_CAPTURE"
        ):
            return False
        self._recording_lease_id = active.lease_id
        return True

    def _start_capture(self, active: _ActiveCapture) -> None:
        self.get_logger().info(
            f"confirmed HOLD for lease {active.lease_id}; starting "
            f"{self._recorder.policy.duration_s:g}-second capture"
        )
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            args=(active,),
            name="event-clip-recorder",
            daemon=True,
        )
        self._capture_thread.start()

    def _start_local_recording(
        self, event: _ObservedEvent, reason: str
    ) -> None:
        self._local_event = event
        self._capture_thread = threading.Thread(
            target=self._local_capture_worker,
            args=(event, reason),
            name="local-event-clip-recorder",
            daemon=True,
        )
        self._capture_thread.start()

    def _local_capture_worker(
        self, event: _ObservedEvent, reason: str
    ) -> None:
        try:
            self._local_capture_worker_impl(event, reason)
        except Exception as exc:
            self.get_logger().error(f"local recording worker failed: {exc}")
        finally:
            with self._lock:
                if self._local_event == event:
                    self._local_recording = False
                    self._local_event = None

    def _local_capture_worker_impl(
        self, event: _ObservedEvent, reason: str
    ) -> None:
        if self._capture is None:
            self.get_logger().error(
                "local event recording skipped: camera capture unavailable"
            )
        else:
            result = record_event_bounded(
                self._recorder, self._capture,
                event_id=event.event_id,
                event_type=event.event_type,
                track_id=event.track_id,
                confidence=event.confidence,
                source=event.source,
                extra_metadata={
                    "control_lease": "rejected",
                    "reason": reason,
                    "camera_backend": self._camera_backend,
                    "gimbal_backend": self._gimbal_backend,
                    "gimbal_tracking": self._gimbal.enabled,
                },
                cancel_event=self._shutdown,
            )
            log = (
                self.get_logger().info
                if result.succeeded
                else self.get_logger().error
            )
            log(f"local event recording: {result.detail}")
        with self._lock:
            self._local_recording = False
            self._local_event = None
            self._tracker.begin_return(now=time.monotonic())
            next_event = (
                self._queued_events.popleft()
                if (self._queued_events and not self._shutdown.is_set()
                    and self._completion_policy == "legacy_rejoin"
                    and not getattr(self._recorder, "_worker_unresolved", False))
                else None
            )
        if next_event is not None:
            self._dispatch_event(next_event)

    def _capture_worker(self, active: _ActiveCapture) -> None:
        try:
            self._capture_worker_impl(active)
        except Exception as exc:
            detail = f"recording worker failed: {type(exc).__name__}: {exc}"
            detail += "; terminal_intent=" + event_terminal_intent(
                policy=self._completion_policy, capture_succeeded=False, autonomy_available=True)
            with self._lock:
                self._pending_release = _PendingRelease(active.lease_id, False, detail, mission_id=active.mission_id,
                    request_source=active.request_source or self._event_lease_source(active.event))
            self.get_logger().error(detail)
        finally:
            with self._lock:
                if self._active == active:
                    self._active = None
                    self._active_wait_started_at = None
                    self._recording_lease_id = ""
            self._send_pending_release()

    def _capture_worker_impl(self, active: _ActiveCapture) -> None:
        if self._capture is None:
            started = time.monotonic()
            self._shutdown.wait(self._recorder.policy.duration_s)
            result = EventClipResult(
                succeeded=False,
                video_path="",
                metadata_path="",
                frames_written=0,
                elapsed_s=time.monotonic() - started,
                detail="recording is disabled or camera capture is unavailable",
            )
        else:
            result = record_event_bounded(
                self._recorder, self._capture,
                finalize_timeout_s=getattr(self, "_capture_finalize_timeout_s", 2.0),
                event_id=active.event.event_id,
                event_type=active.event.event_type,
                track_id=active.event.track_id,
                confidence=active.event.confidence,
                source=active.event.source,
                extra_metadata={
                    "lease_id": active.lease_id,
                    "mission_id": active.mission_id,
                    "completion_policy": self._completion_policy,
                    "camera_backend": self._camera_backend,
                    "gimbal_backend": self._gimbal_backend,
                    "camera_source": (
                        str(self.get_parameter("ros_camera_topic").value)
                        if self._camera_backend == "ros_compressed"
                        else "%s/%s"
                        % (
                            self.get_parameter("airsim_vehicle_name").value,
                            self.get_parameter("airsim_camera_name").value,
                        )
                    ),
                    "gimbal_policy": {
                        "yaw_range_deg": [-90.0, 90.0],
                        "pitch_range_deg": [-80.0, 20.0],
                        "max_yaw_rate_deg_s": 30.0,
                        "max_pitch_rate_deg_s": 20.0,
                        "target_lost_action": "RETURN_FORWARD",
                    },
                },
                cancel_event=self._shutdown,
            )

        with self._lock:
            if self._active == active:
                self._active = None
                self._active_wait_started_at = None
                self._recording_lease_id = ""
                self._tracker.begin_return(now=time.monotonic())
            detail = result.detail
            detail += "; terminal_intent=" + event_terminal_intent(
                policy=self._completion_policy, capture_succeeded=result.succeeded,
                # This is intent only. The manager checks live ownership.
                autonomy_available=True,
            )
            if result.metadata_path:
                detail += f"; metadata={result.metadata_path}"
            self._pending_release = _PendingRelease(
                lease_id=active.lease_id,
                capture_succeeded=result.succeeded,
                detail=detail,
                mission_id=active.mission_id,
                request_source=active.request_source or self._event_lease_source(active.event),
            )
        self._send_pending_release()

    def _send_pending_release(self) -> None:
        with self._lock:
            pending = self._pending_release
            now = time.monotonic()
            if (
                pending is None
                or pending.in_flight
                or now - pending.last_attempt_at < 0.5
                or not self._release_client.service_is_ready()
            ):
                return
            pending.in_flight = True
            pending.last_attempt_at = now
            lease_id = pending.lease_id
            request = ReleaseEventControl.Request()
            request.lease_id = lease_id
            request.source = pending.request_source or self._request_source
            request.capture_succeeded = pending.capture_succeeded
            request.detail = pending.detail
        try:
            future = self._release_client.call_async(request)
        except Exception as exc:
            with self._lock:
                if self._pending_release is pending:
                    pending.in_flight = False
            self.get_logger().error(
                f"event lease release dispatch failed; will retry: {exc}"
            )
            return
        future.add_done_callback(
            lambda completed, expected=lease_id: self._on_release_response(
                completed, expected
            )
        )

    def _on_release_response(self, future, expected_lease_id: str) -> None:
        try:
            response = future.result()
        except Exception as exc:
            with self._lock:
                pending = self._pending_release
                if pending is not None and pending.lease_id == expected_lease_id:
                    pending.in_flight = False
            self.get_logger().error(
                f"event lease release failed; will retry: {exc}"
            )
            return
        next_event = None
        with self._lock:
            pending = self._pending_release
            if pending is None or pending.lease_id != expected_lease_id:
                return
            if not response.accepted:
                pending.in_flight = False
            else:
                self._pending_release = None
                next_event = (
                    self._queued_events.popleft()
                    if (self._queued_events and not self._shutdown.is_set()
                        and self._completion_policy == "legacy_rejoin"
                        and not getattr(self._recorder, "_worker_unresolved", False))
                    else None
                )
        log = self.get_logger().info if response.accepted else self.get_logger().warning
        log(f"event lease release: {response.message}")
        if next_event is not None:
            self._dispatch_event(next_event)

    def _on_gimbal_timer(self) -> None:
        command = None
        failed_hold = None
        with self._lock:
            evidence_expired = (self._completion_policy == "capture_then_rtl" and self._pending is not None
                and bool(evidence_deadline_error(self._pending.evidence_clock_id,
                                                self._pending.evidence_expires_monotonic_ns)))
            if (
                self._pending is not None
                and self._pending_started_at is not None
                and (evidence_expired or time.monotonic() - self._pending_started_at
                >= float(
                    self.get_parameter("lease_request_timeout_s").value
                ))
            ):
                timed_out = self._pending
                self._pending = None
                self._pending_started_at = None
                self._pending_mission_id = ""
                self._known_event_ids.discard(timed_out.event_id)
                if self._allow_local_capture_without_control and self._completion_policy == "legacy_rejoin":
                    self._local_recording = True
                    self.get_logger().warning(
                        "event lease request timed out; recording locally"
                    )
                    self._start_local_recording(
                        timed_out, "lease request timed out"
                    )
                else:
                    self.get_logger().error(
                        "event lease request timed out; fail-closed without recording"
                    )
            if (
                self._active is not None
                and not self._recording_lease_id
                and self._active_wait_started_at is not None
                and time.monotonic() - self._active_wait_started_at
                >= float(
                    self.get_parameter("hold_ready_timeout_s").value
                )
            ):
                failed_hold = self._active
                self._active = None
                self._active_wait_started_at = None
                self._tracker.begin_return(now=time.monotonic())
                self._pending_release = _PendingRelease(
                    lease_id=failed_hold.lease_id,
                    capture_succeeded=False,
                    detail="capture not started because aircraft HOLD was not confirmed",
                    mission_id=failed_hold.mission_id,
                    request_source=failed_hold.request_source or self._event_lease_source(failed_hold.event),
                )
            active = self._active
            tracking_event = (
                active.event if active is not None else self._local_event
            )
            if (
                tracking_event is not None
                and self._latest is not None
                and self._latest.event_id == tracking_event.event_id
                and self._latest.target_visible
                and self._latest_revision != self._processed_revision
            ):
                command = self._tracker.observe_bbox(
                    self._latest.bbox,
                    now=self._latest.received_at,
                )
                self._processed_revision = self._latest_revision
            else:
                command = self._tracker.tick(now=time.monotonic())
        if failed_hold is not None:
            self.get_logger().error(
                f"HOLD confirmation timed out for lease {failed_hold.lease_id}; "
                "requesting fail-closed release"
            )
        self._send_pending_release()
        self._report_backend_errors()
        if command is None:
            return
        try:
            self._gimbal.set_attitude(
                command.yaw_deg, command.pitch_deg
            )
        except Exception as exc:
            self.get_logger().error(
                f"event camera tracking command failed: {exc}"
            )

    def _report_backend_errors(self) -> None:
        errors = []
        capture_error = (
            self._capture.last_error if self._capture is not None else ""
        )
        if capture_error:
            errors.append(f"capture: {capture_error}")
        if self._gimbal.last_error:
            errors.append(f"gimbal: {self._gimbal.last_error}")
        current = "; ".join(errors)
        now = time.monotonic()
        if current and (
            current != self._last_backend_error
            or now - self._backend_error_log_at >= 5.0
        ):
            self.get_logger().error(f"event camera backend unavailable: {current}")
            self._backend_error_log_at = now
        elif not current and self._last_backend_error:
            self.get_logger().info("event camera backend recovered")
        self._last_backend_error = current

    def _convert_observation(
        self, message: EventObservation
    ) -> _ObservedEvent:
        event_id = message.event_id.strip()
        event_type = message.event_type.strip()
        if not event_id:
            raise ValueError("event_id is empty")
        allowed = ({"DEMO_PERSON"} if (getattr(self, "_simple_obstacle_demo", False)
                   and not getattr(self, "_simple_demo_phase1_events", False))
                   else PHASE1_EVENT_TYPES)
        if event_type not in allowed:
            raise ValueError(f"unsupported Phase 1 event type: {event_type}")
        if event_type == "DEMO_PERSON" and message.source != "demo_person":
            raise ValueError("DEMO_PERSON source must be demo_person")
        bbox = (
            float(message.bbox_x1),
            float(message.bbox_y1),
            float(message.bbox_x2),
            float(message.bbox_y2),
        )
        if message.target_visible and not (
            0.0 <= bbox[0] < bbox[2] <= 1.0
            and 0.0 <= bbox[1] < bbox[3] <= 1.0
        ):
            raise ValueError("visible target bbox is not normalized")
        confidence = float(message.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence is outside [0, 1]")
        evidence_clock_id = getattr(message, "evidence_clock_id", "")
        evidence_deadline = getattr(message, "evidence_expires_monotonic_ns", 0)
        if self._completion_policy == "capture_then_rtl":
            error = evidence_deadline_error(evidence_clock_id, evidence_deadline)
            if error:
                raise ValueError("incident evidence unavailable: " + error)
        return _ObservedEvent(
            event_id=event_id,
            event_type=event_type,
            track_id=message.track_id.strip() or "untracked",
            confidence=confidence,
            bbox=bbox,
            target_visible=bool(message.target_visible),
            received_at=time.monotonic(),
            source=message.source.strip() or "phase1-demo-local",
            evidence_clock_id=evidence_clock_id,
            evidence_expires_monotonic_ns=evidence_deadline,
        )

    def destroy_node(self) -> bool:
        self._shutdown.set()
        if self._capture is not None:
            self._capture.stop()
        if (
            self._capture_thread is not None
            and self._capture_thread is not threading.current_thread()
        ):
            self._capture_thread.join(
                timeout=self._recorder.policy.duration_s + 2.0
            )
        try:
            self._gimbal.stop()
        finally:
            self._gimbal.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EventResponseNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
