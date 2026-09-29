"""Startup-only incident judgment/recording appliance; no flight interfaces."""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import uuid

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import String

from jolgwa_uav.observer_jpeg import validate_observer_jpeg
from .event_clip import EventClipRecorder, record_event_bounded
from .incident_observer import (INCIDENT_OBSERVER_TOPIC, capture_terminal_report,
                                current_incident_report, unknown_incident, validate_incident_result)
from .ros_image_capture import RosCompressedImageCapture


class _StrictIncidentCapture(RosCompressedImageCapture):
    """Reject repeated or regressing source stamps before inference and recording."""
    def __init__(self, node, topic, on_frame, on_fault):
        super().__init__(node, topic)
        self._last_stamp = 0
        self._on_frame = on_frame
        self._on_fault = on_fault

    def _on_image(self, message):
        arrived_at = time.monotonic()
        try:
            validate_observer_jpeg(message.data, message.format)
            stamp = int(message.header.stamp.sec)*1_000_000_000 + int(message.header.stamp.nanosec)
            if stamp <= self._last_stamp:
                raise ValueError("incident_frame_not_advancing_restart_on_clock_reset")
            self._last_stamp = stamp
        except (ValueError, TypeError, BufferError, OverflowError) as exc:
            with self._condition:
                self._latest = None
            self._on_fault(str(exc))
            return
        super()._on_image(message)
        with self._condition:
            latest = self._latest
            if latest is not None:
                latest = (latest[0], arrived_at, latest[2])
                self._latest = latest
        if latest is not None:
            self._on_frame(stamp, latest[1], latest[2])


class IncidentObserverNode(Node):
    """Only camera input, diagnostic String output and local file recording.

    This class deliberately does not inherit the flight perception bridge and
    never constructs an EventObservation publisher or control service client.
    """
    def __init__(self, *, jetson_url="http://127.0.0.1:8776",
                 camera_topic="/camera/camera/color/image_raw/compressed",
                 storage_root="outputs/incident_observer", recording_enabled=True):
        super().__init__("incident_observer")
        parsed = urlparse(jetson_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("incident observe backend must be dedicated local HTTP")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("incident backend URL must not include credentials/query/fragment")
        self._url = jetson_url.rstrip("/")
        self._session = str(uuid.uuid4())
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame = None
        self._last_requested_stamp = 0
        self._last_request_at = float("-inf")
        self._request_thread = None
        self._record_thread = None
        self._completed = deque()
        self._seen_events = set()
        self._recording_enabled = bool(recording_enabled)
        self._recording_state = "IDLE"
        self._last_recording = None
        self._last_report = unknown_incident("waiting_for_rgb_and_phase1")
        self._last_report_received_at = 0.0
        self._journal_failed = False
        self._sequence = 0
        root = Path(storage_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self._directory = root / ("session_" + self._session)
        self._directory.mkdir(exist_ok=False)
        self._journal = (self._directory / "diagnostics.jsonl").open("x", encoding="utf-8", buffering=1)
        self._recorder = EventClipRecorder(self._directory / "captures")
        self._publisher = self.create_publisher(String, INCIDENT_OBSERVER_TOPIC, 10)
        self._capture = _StrictIncidentCapture(self, camera_topic, self._on_frame, self._on_fault).start()
        self._timer_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(.1, self._tick, clock=self._timer_clock)
        self.get_logger().warning("INCIDENT_OBSERVE_ONLY: judgment/local recording; no flight interfaces")
        self._emit(self._last_report)

    def _on_frame(self, stamp, received_at, jpeg):
        with self._lock:
            self._latest_frame = (stamp, received_at, jpeg)

    def _on_fault(self, reason):
        with self._lock:
            self._latest_frame = None
            self._completed.append(("diagnostic", unknown_incident(reason), time.monotonic()))

    def _request(self, frame):
        stamp, received_at, jpeg = frame
        started = time.monotonic()
        completed_at = started
        report = unknown_incident("incident_request_failed")
        try:
            with urlopen(self._url + "/health", timeout=.25) as response:
                health = json.loads(response.read(1_048_577))
            if (health.get("observation_only") is not True
                    or health.get("purpose") != "incident_observe"
                    or health.get("physical_fc_commands") != 0):
                raise ValueError("incident_requires_dedicated_diagnostic_backend")
            sent_at = time.monotonic()
            age = sent_at-received_at
            if age < 0 or age > .5:
                raise ValueError("incident_frame_expired_before_post")
            query = urlencode({"frame_timestamp_ns": stamp, "frame_age_s": age})
            request = Request(self._url + "/v1/incident-observe?" + query, data=jpeg,
                              headers={"Content-Type": "image/jpeg",
                                       "X-Jolgwa-Incident-Session": self._session}, method="POST")
            with urlopen(request, timeout=max(.01, 2.0-age)) as response:
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("incident_response_too_large")
            payload = json.loads(raw)
            completed_at = time.monotonic()
            report = validate_incident_result(payload, frame_timestamp_ns=stamp,
                frame_received_at=received_at, requested_at=sent_at, completed_at=completed_at)
        except Exception as exc:
            report = unknown_incident(f"incident_request_failed:{type(exc).__name__}:{exc}")
            report["response_elapsed_s"] = max(0.0, time.monotonic()-started)
        finally:
            with self._lock:
                # Do not rebase a validated report's TTL after scheduling or
                # queue-lock delay. Its absolute expiry is already fixed.
                self._completed.append(("diagnostic", report, completed_at))

    def _record(self, event):
        report = {"recording_state": "FAILED", "event_capture_succeeded": False,
                  "reason": "recording_worker_interrupted", "terminal_intent": "RTL",
                  "terminal_command_sent": False,
                  "terminal_suppression": "observe_only_no_control_connection"}
        try:
            result = record_event_bounded(self._recorder, self._capture,
                event_id=event["event_id"], event_type=event["event_type"],
                track_id=",".join(str(x) for x in event.get("track_ids", [])) or "untracked",
                confidence=event["confidence"], source="jetson-incident-observe",
                extra_metadata={"mode": "incident-observe", "session": self._session,
                                "physical_fc_commands": 0, "completion_policy": "capture_then_rtl",
                                "terminal_suppression": "observe_only_no_control_connection"},
                cancel_event=self._shutdown)
            report = capture_terminal_report(result)
        except Exception as exc:
            report = {"recording_state": "FAILED", "event_capture_succeeded": False,
                      "detail": f"unexpected recording failure:{exc}", "terminal_intent": "RTL",
                      "terminal_command_sent": False, "terminal_suppression": "observe_only_no_control_connection"}
        finally:
            with self._lock:
                self._completed.append(("recording", dict(report, event_id=event["event_id"]), time.monotonic()))

    def _tick(self):
        if self._shutdown.is_set():
            return
        now = time.monotonic()
        with self._lock:
            completed = list(self._completed)
            self._completed.clear()
            latest = self._latest_frame
        for kind, report, completed_at in completed:
            if kind == "recording":
                self._last_recording = report
                self._recording_state = report["recording_state"]
            else:
                consumed_at = time.monotonic()
                report = current_incident_report(report, now=consumed_at)
                if (latest is None or not 0 <= consumed_at-latest[1] <= .5
                        or completed_at > consumed_at
                        or (report.get("assessment") != "UNKNOWN"
                            and completed_at != report.get("evidence_completed_monotonic_s"))):
                    report = unknown_incident("incident_result_or_camera_expired_before_consumption")
                self._last_report, self._last_report_received_at = report, completed_at
                for event in report.get("events", []):
                    if event["event_id"] in self._seen_events:
                        continue
                    self._seen_events.add(event["event_id"])
                    if len(self._seen_events) > 10000:
                        self._journal_failed = True
                        self._last_report = unknown_incident("incident_event_capacity_restart_required")
                        break
                    if (self._recording_enabled and not self._journal_failed
                            and not getattr(self._recorder, "_worker_unresolved", False)
                            and (self._record_thread is None or not self._record_thread.is_alive())):
                        self._recording_state = "RECORDING"
                        self._record_thread = threading.Thread(target=self._record, args=(event,), daemon=True)
                        self._record_thread.start()
                    else:
                        # No hidden queue of stale scenes. Record the explicit omission.
                        self._last_recording = {"event_id": event["event_id"],
                            "recording_state": "SKIPPED", "event_capture_succeeded": False,
                            "reason": "recording_disabled_busy_or_unavailable",
                            "terminal_intent": "RTL", "terminal_command_sent": False,
                            "terminal_suppression": "observe_only_no_control_connection"}
            self._emit(self._last_report)
        now = time.monotonic()
        if latest is None or not 0 <= now-latest[1] <= .5:
            self._last_report = unknown_incident("incident_rgb_missing_or_stale")
        else:
            self._last_report = current_incident_report(self._last_report, now=now)
        if (latest is not None and 0 <= now-latest[1] <= .5
                and latest[0] > self._last_requested_stamp and now-self._last_request_at >= .25
                and (self._request_thread is None or not self._request_thread.is_alive())):
            self._last_requested_stamp = latest[0]
            self._last_request_at = now
            self._request_thread = threading.Thread(target=self._request, args=(latest,), daemon=True)
            self._request_thread.start()
        self._emit(self._last_report)

    def _emit(self, report):
        self._sequence += 1
        published_at = time.monotonic()
        report = current_incident_report(report, now=published_at)
        payload = dict(report, session=self._session, sequence=self._sequence,
                       published_monotonic_s=published_at, recording_state=self._recording_state,
                       last_recording=self._last_recording)
        if self._journal_failed:
            payload.update(assessment="UNKNOWN", reason="incident_journal_unavailable",
                           events=[], valid_for_s=0.0)
        try:
            text = json.dumps(payload, ensure_ascii=True, allow_nan=False)
            self._journal.write(text + "\n")
        except Exception:
            self._journal_failed = True
            payload = dict(unknown_incident("incident_journal_unavailable"),
                           session=self._session, sequence=self._sequence,
                           published_monotonic_s=time.monotonic())
            text = json.dumps(payload)
        # Journal I/O may block. Re-age the outgoing observation after it too;
        # the journal retains the earlier snapshot, not a renewed permission.
        published_at = time.monotonic()
        if self._journal_failed:
            outgoing = unknown_incident("incident_journal_unavailable")
        else:
            outgoing = current_incident_report(report, now=published_at)
        payload = dict(outgoing, session=self._session, sequence=self._sequence,
                       published_monotonic_s=published_at)
        if not self._journal_failed:
            payload.update(recording_state=self._recording_state, last_recording=self._last_recording)
        text = json.dumps(payload, ensure_ascii=True, allow_nan=False)
        message = String()
        message.data = text
        self._publisher.publish(message)

    def destroy_node(self):
        self._shutdown.set()
        self._capture.stop()
        for worker in (self._request_thread, self._record_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=.3)
        self._journal.close()
        return super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("incident-observe",), default="incident-observe")
    parser.add_argument("--jetson-url", default="http://127.0.0.1:8776")
    parser.add_argument("--camera-topic", default="/camera/camera/color/image_raw/compressed")
    parser.add_argument("--storage-root", default="outputs/incident_observer")
    parser.add_argument("--no-record", action="store_true")
    options, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = None
    try:
        node = IncidentObserverNode(jetson_url=options.jetson_url, camera_topic=options.camera_topic,
                                    storage_root=options.storage_root, recording_enabled=not options.no_record)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
