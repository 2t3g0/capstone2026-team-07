"""D435 RGB / paced-video rehearsal using original Phase1 and event_clip.

No vehicle, ROS publisher, network client, approval or control interface exists.
Importing this module and planning do not import models, codecs or camera drivers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import uuid

EVENTS = ("FIRE_SMOKE", "HUMAN_VIOLENCE", "LITTERING", "INTRUSION_ATTEMPT",
          "CALL_FOR_HELP", "VEHICLE_ACCIDENT")


def classify_session(label, expected_event, diagnostics, recordings):
    """Labels are evaluation annotations, never model inputs or injected events."""
    valid = [d for d in diagnostics if d.get("status") == "OBSERVED"]
    events = {e["event_id"]: e for d in diagnostics for e in d.get("events", [])}
    span = valid[-1]["monotonic_s"] - valid[0]["monotonic_s"] if valid else 0.0
    sufficient = len(valid) >= 20 and span >= 5.0 and len(valid) / max(1, len(diagnostics)) >= .8
    errors = any(d.get("error") for d in diagnostics)
    completed = {r["event_id"] for r in recordings
                 if r.get("succeeded") is True and r.get("worker_finished") is True}
    if label == "negative":
        passed = sufficient and not events and not errors
    else:
        matches = {i for i, e in events.items() if e["event_type"] == expected_event}
        passed = sufficient and bool(matches & completed) and not errors and all(
            e["event_type"] == expected_event for e in events.values())
    contrary = errors or any(r.get("succeeded") is not True for r in recordings)
    contrary |= bool(events) if label == "negative" else any(
        e["event_type"] != expected_event for e in events.values())
    return dict(status="PASS" if passed else "BLOCKED" if contrary or sufficient else "UNVERIFIED",
                passed=passed, annotation=label, expected_event=expected_event,
                evaluated_frames=len(diagnostics), valid_frames=len(valid), valid_span_s=span,
                sufficient_live_coverage=sufficient, confirmed_events=list(events.values()),
                capture_results=recordings, inference_errors=errors,
                physical_fc_commands=0, real_flight_verified=False)


@dataclass(frozen=True)
class Frame:
    sequence: int
    captured_at: float
    image: object
    source_timestamp_ns: int


class FreshFrames:
    """One source owner, independent acquisition while inference/recording run."""
    def __init__(self, reader):
        self.reader = reader
        self.condition = threading.Condition()
        self.cancel = threading.Event()
        self.latest = None
        self.error = ""
        self.ended = False
        self.thread = threading.Thread(target=self._pump, name="event-bench-camera", daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _pump(self):
        previous = 0
        sequence = 0
        try:
            while not self.cancel.is_set():
                value = self.reader.read(self.cancel)
                if value is None:
                    break
                image, stamp = value
                if type(stamp) is not int or stamp <= previous:
                    raise ValueError("source timestamp not advancing; no replay/reset reuse")
                previous = stamp
                sequence += 1
                with self.condition:
                    self.latest = Frame(sequence, time.monotonic(), image, stamp)
                    self.condition.notify_all()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.reader.close()
            except Exception as exc:
                self.error = (self.error + "; " if self.error else "") + f"source close failed: {type(exc).__name__}: {exc}"
            with self.condition:
                self.ended = True
                self.condition.notify_all()

    def read_after(self, sequence, *, timeout_s=.5):
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while not self.cancel.is_set():
                if self.latest is not None and self.latest.sequence > sequence:
                    return self.latest
                remaining = deadline - time.monotonic()
                if self.ended or remaining <= 0:
                    return None
                self.condition.wait(remaining)
        return None

    def stop(self):
        self.cancel.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=3.0)
        return not self.thread.is_alive()


class VideoReader:
    def __init__(self, path):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("recorded video could not be opened")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 120:
            self.capture.release()
            raise RuntimeError("video has no usable 1..120 FPS timing")
        self.index = 0
        self.started = time.monotonic()

    def read(self, cancel):
        due = self.started + self.index / self.fps
        if cancel.wait(max(0.0, due - time.monotonic())):
            return None
        ok, image = self.capture.read()
        if not ok:
            return None
        self.index += 1
        return image, round(self.index * 1_000_000_000 / self.fps)

    def close(self):
        self.capture.release()


class D435Reader:
    """RGB only. Exact D435 device identification; no IMU or flight connection."""
    def __init__(self, serial=None):
        import numpy as np
        import pyrealsense2 as rs
        self.np = np
        devices = [d for d in rs.context().query_devices()
                   if d.get_info(rs.camera_info.name).strip() == "Intel RealSense D435"
                   and (serial is None or d.get_info(rs.camera_info.serial_number) == serial)]
        if len(devices) != 1:
            raise RuntimeError("require exactly one matching D435; provide --serial; D435i is not substituted")
        self.serial = devices[0].get_info(rs.camera_info.serial_number)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(config)  # Busy devices fail; other owners are never stopped.

    def read(self, cancel):
        if cancel.is_set():
            return None
        frame = self.pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
        if not frame:
            raise RuntimeError("D435 RGB frame missing")
        return self.np.asanyarray(frame.get_data()).copy(), round(frame.get_timestamp() * 1_000_000)

    def close(self):
        self.pipeline.stop()


def _recording_imports():
    # Reuse the actual ROS package's pure recorder, without importing rclpy.
    root = Path(__file__).resolve().parents[2] / "ros2_ws/src/jolgwa_ros"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from jolgwa_ros.event_clip import EventClipRecorder, EventClipPolicy, record_event_bounded
    return EventClipRecorder, EventClipPolicy, record_event_bounded


def _permanent_root(path):
    if str(path).startswith(("\\\\", "//")):
        raise ValueError("UNC/network output is not internal storage")
    root = Path(path).expanduser().resolve()
    temporary = Path(tempfile.gettempdir()).resolve()
    if root == temporary or temporary in root.parents:
        raise ValueError("permanent internal output must not be a temporary directory")
    if sys.platform != "win32" and root.parts[1:2] in [("media",), ("mnt",), ("run",), ("dev",), ("proc",), ("sys",)]:
        raise ValueError("use internal persistent storage, not removable/network/runtime mount roots")
    return root


def run_rehearsal(args):
    os.environ.update(ULTRALYTICS_AUTOINSTALL="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from .phase1_contract import phase1_server_response, validate_phase1_response
    from .phase1_inference import Phase1JetsonRuntime
    Recorder, Policy, record_bounded = _recording_imports()
    root = _permanent_root(args.output_dir)
    session = str(uuid.uuid4())
    directory = root / ("session_" + session)
    directory.mkdir(parents=True, exist_ok=False)
    details = dict(schema_version=1, session_id=session, session_label=args.session_label,
                   expected_event=args.expected_event, source_kind=args.source,
                   camera_model="D435" if args.source == "d435" else None,
                   imu_used=False, video=str(Path(args.video).resolve()) if args.video else None,
                   recorded_video_is_live_camera=False if args.source == "video" else None,
                   vehicle_control_connected=False, physical_fc_commands=0, retention="permanent",
                   physical_storage_mount_verified=False,
                   requested_duration_s=args.duration_s, started_utc=datetime.now(timezone.utc).isoformat())
    with (directory / "session.json").open("x", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    diagnostics, recordings, seen = [], [], set()
    capture = None
    error = ""
    cleanup_confirmed = True
    with (directory / "diagnostics.jsonl").open("x", encoding="utf-8", buffering=1) as journal:
        def log(kind, value):
            journal.write(json.dumps(dict(kind=kind, monotonic_s=time.monotonic(), payload=value),
                                     allow_nan=False) + "\n")
        try:
            runtime = Phase1JetsonRuntime(args.phase1_root, device=args.device)
            runtime.warmup()
            log("model_health", runtime.health())
            # Annotation/expected-event are deliberately not passed to runtime.
            reader = D435Reader(args.serial) if args.source == "d435" else VideoReader(args.video)
            capture = FreshFrames(reader).start()
            recorder = Recorder(directory / "captures", Policy(duration_s=5.0, fps=25))
            end = time.monotonic() + args.duration_s
            sequence = 0
            while time.monotonic() < end:
                frame = capture.read_after(sequence, timeout_s=.25)
                if frame is None:
                    if capture.ended:
                        if time.monotonic() < end-.25:
                            raise RuntimeError("source ended before requested observation duration")
                        break
                    continue
                sequence = frame.sequence
                requested = time.monotonic()
                if requested - frame.captured_at > .5:
                    continue
                try:
                    result = runtime.evaluate_bgr(frame.image)
                    completed = time.monotonic()
                    payload = phase1_server_response(result, session_id=session,
                        request_sequence=sequence, frame_timestamp_ns=frame.source_timestamp_ns,
                        input_frame_age_s=requested-frame.captured_at, elapsed_s=completed-requested)
                    report = validate_phase1_response(payload, session_id=session,
                        request_sequence=sequence, frame_timestamp_ns=frame.source_timestamp_ns,
                        frame_received_at=frame.captured_at, requested_at=requested, completed_at=completed)
                    report["monotonic_s"] = completed
                    report["source_frame_sha256"] = hashlib.sha256(frame.image.tobytes()).hexdigest()
                    report["source_timestamp_ns"] = frame.source_timestamp_ns
                except Exception as exc:
                    report = dict(status="UNKNOWN", events=[], monotonic_s=time.monotonic(),
                                  error=f"{type(exc).__name__}: {exc}")
                diagnostics.append(report)
                log("phase1", report)
                for event in report.get("events", []):
                    if event["event_id"] in seen:
                        continue
                    seen.add(event["event_id"])
                    if time.monotonic() >= report.get("_local_evidence", {}).get("expires_at", 0):
                        item = dict(event_id=event["event_id"], succeeded=False, worker_finished=True,
                                    detail="event expired after prior capture; not queued or re-injected")
                        recordings.append(item)
                        log("capture_skipped", item)
                        continue
                    recording = record_bounded(recorder, capture, finalize_timeout_s=4.0,
                        event_id=event["event_id"], event_type=event["event_type"],
                        track_id=",".join(map(str, event.get("track_ids", []))) or "untracked",
                        confidence=event["confidence"], source="phase1-event-rehearsal",
                        extra_metadata=dict(session_id=session, session_label=args.session_label,
                                            source_kind=args.source, physical_fc_commands=0,
                                            completion_policy="record_only_no_vehicle_control"))
                    item = dict(asdict(recording), event_id=event["event_id"])
                    if recording.succeeded and recording.worker_finished:
                        from .field_event_evidence import verify_capture_video
                        metadata = json.loads(Path(recording.metadata_path).read_text(encoding="utf-8"))
                        item["video_verification"] = verify_capture_video(recording.metadata_path, metadata)
                        item["succeeded"] = item["video_verification"]["passed"]
                    recordings.append(item)
                    log("capture", item)
                    if not recording.worker_finished:
                        raise RuntimeError("capture worker unresolved; stop rehearsal, preserve failure evidence")
            if capture.error:
                raise RuntimeError(capture.error)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log("error", {"error": error})
        finally:
            if capture is not None:
                cleanup_confirmed = capture.stop()
                error = error or capture.error
    report = classify_session(args.session_label, args.expected_event, diagnostics, recordings)
    report.update(session_id=session, source_kind=args.source, output_directory=str(directory),
                  error=error, acquisition_stopped=cleanup_confirmed, retention="permanent")
    if error or not cleanup_confirmed:
        report.update(status="BLOCKED", passed=False)
    with (directory / "result.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("video", "d435"), default="video")
    parser.add_argument("--video")
    parser.add_argument("--serial")
    parser.add_argument("--camera-owner-confirmed", action="store_true")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-label", choices=("positive", "negative"), required=True)
    parser.add_argument("--expected-event", choices=EVENTS)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or not 10 <= args.duration_s <= 600:
        parser.error("duration must be 10..600 seconds")
    if args.session_label == "positive" and not args.expected_event:
        parser.error("positive sessions require --expected-event (evaluation label only)")
    if args.session_label == "negative" and args.expected_event:
        parser.error("negative sessions must not specify an expected positive event")
    if args.execute and args.source == "d435" and not args.camera_owner_confirmed:
        parser.error("D435 requires --camera-owner-confirmed; never stop other camera owners")
    if args.execute and args.source == "video" and not args.video:
        parser.error("video source requires --video")
    plan = dict(status="PLAN_ONLY", execute=False,
        source_kind=args.source, session_label=args.session_label, expected_event=args.expected_event,
        recording_duration_s=5.0, fps=25, physical_fc_commands=0, network_requests=0,
        requirements=["original Phase1 models/dependencies", "OpenCV", "H264 encoder",
                      "pyrealsense2 and exclusive D435 owner for live RGB", "persistent internal output"])
    try:
        result = run_rehearsal(args) if args.execute else plan
    except Exception as exc:
        result = dict(status="BLOCKED", passed=False, physical_fc_commands=0,
                      error=f"{type(exc).__name__}: {exc}")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not args.execute or result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
