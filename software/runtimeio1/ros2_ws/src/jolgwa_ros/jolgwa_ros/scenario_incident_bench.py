"""Observe-only incident/photo bench; reuses the already-owned ROS JPEG stream.

No camera SDK, FC publisher, flight goal, motor, video, or command endpoint.
Model loading stays in the existing Phase1JetsonRuntime with original votes.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen
import uuid
from .async_journal import AsyncJournal

MODE = "INCIDENT_PHOTO_OBSERVE_ONLY"
MAX_JPEG = 8 * 1024 * 1024
TOPIC = "/camera/camera/color/image_raw/compressed"
PERSON_COLOR_BGR = (0, 220, 0)
LITTERING_CANDIDATE_COLOR_BGR = (0, 165, 255)
LITTERING_CONFIRMED_COLOR_BGR = (0, 0, 255)
LIVE_JPEG_QUALITY = 65
EVIDENCE_JPEG_QUALITY = 90
SHADOW_MODE = "RGB_POSEC3D_SHADOW_OBSERVE_ONLY"
SHADOW_CLASSES = {
    "punching", "kicking", "pushing", "attacking_with_object",
    "littering", "lock_picking", "call_for_help",
}
SHADOW_MAX_AGE_S = 1.0


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_id(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


def jpeg_dimensions(payload):
    """Bound SOF dimensions before decoding/preview, without opening a camera."""
    if (not isinstance(payload, bytes) or not 4 <= len(payload) <= MAX_JPEG
            or payload[:2] != b"\xff\xd8" or payload[-2:] != b"\xff\xd9"):
        raise ValueError("invalid_jpeg_payload")
    offset = 2
    while offset < len(payload)-2:
        if payload[offset] != 255:
            break
        while offset < len(payload) and payload[offset] == 255:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in (0xda, 0xd9):
            break
        if marker == 1 or 0xd0 <= marker <= 0xd7:
            continue
        length = int.from_bytes(payload[offset:offset+2], "big")
        if length < 2 or offset+length > len(payload):
            break
        if marker in (0xc0,0xc1,0xc2):
            if length < 8:
                break
            h,w = int.from_bytes(payload[offset+3:offset+5], "big"), int.from_bytes(payload[offset+5:offset+7], "big")
            if payload[offset+2] == 8 and payload[offset+7] == 3 and 2 <= w <= 1920 and 2 <= h <= 1080:
                return w,h
            break
        offset += length
    raise ValueError("invalid_jpeg_header_dimensions")


def atomic_new(path, content):
    """Durable new-file commit; hardlink refuses to replace an existing target."""
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalized_bbox(value):
    """Return a strict normalized xyxy box, never a guessed full-frame box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(item) for item in bbox)
            or not 0.0 <= bbox[0] < bbox[2] <= 1.0
            or not 0.0 <= bbox[1] < bbox[3] <= 1.0):
        return None
    return bbox


def _confidence(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return float(value)


def validate_shadow_candidate(value):
    if not isinstance(value, dict) or value.get("class_name") not in SHADOW_CLASSES:
        raise ValueError("invalid_shadow_class")
    bbox, confidence = normalized_bbox(value.get("bbox")), _confidence(value.get("confidence"))
    track_ids = value.get("track_ids")
    if (bbox is None or confidence is None or not isinstance(track_ids, list)
            or not 1 <= len(track_ids) <= 2
            or any(type(item) is not int or item < 0 for item in track_ids)
            or len(set(track_ids)) != len(track_ids)
            or value.get("state") != "CANDIDATE"
            or value.get("source") != "rgb_posec3d_shadow"
            or type(value.get("source_timestamp_ns")) is not int
            or value["source_timestamp_ns"] <= 0
            or _confidence(value.get("result_age_s")) is None):
        raise ValueError("invalid_shadow_candidate")
    return dict(value, bbox=list(bbox), confidence=confidence,
                result_age_s=float(value["result_age_s"]))


def validate_shadow_status(value, nonce):
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("mode") != SHADOW_MODE
            or value.get("flight_commands_enabled") is not False
            or value.get("physical_fc_commands") != 0
            or value.get("request_nonce") != nonce):
        raise ValueError("shadow_contract_mismatch")
    if not canonical_id(value.get("session_id")):
        raise ValueError("invalid_shadow_session")
    if type(value.get("sequence")) is not int or value["sequence"] < 0:
        raise ValueError("invalid_shadow_sequence")
    if value.get("state") not in {
            "STARTING", "WARMING_UP", "NORMAL_OBSERVATION", "CANDIDATE", "UNKNOWN", "ERROR"}:
        raise ValueError("invalid_shadow_state")
    valid_for = _confidence(value.get("display_valid_for_s"))
    result_age = value.get("result_age_s")
    if result_age is not None and (_confidence(result_age) is None or result_age > SHADOW_MAX_AGE_S):
        raise ValueError("invalid_shadow_result_age")
    candidates = value.get("behavior_candidates")
    hashes = value.get("model_hashes")
    if (valid_for is None or not isinstance(candidates, list) or len(candidates) > 20
            or not isinstance(hashes, dict) or set(hashes) != {"rgb", "posec3d", "yolo_pose"}
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in "0123456789abcdef" for char in digest)
                   for digest in hashes.values())
            or not isinstance(value.get("warnings"), list)
            or any(not isinstance(item, str) for item in value["warnings"])):
        raise ValueError("invalid_shadow_diagnostics")
    clean = [validate_shadow_candidate(item) for item in candidates]
    if value["state"] != "CANDIDATE" and clean:
        raise ValueError("shadow_candidates_without_candidate_state")
    return dict(deepcopy(value), behavior_candidates=clean)


class ShadowCache:
    """Best-effort loopback cache.  Failure never changes the incident pipeline."""
    def __init__(self, origin="http://127.0.0.1:8879", *, clock=time.monotonic,
                 poll_interval=.25, timeout=.15):
        self.origin = origin.rstrip("/")
        self.clock, self.poll_interval, self.timeout = clock, poll_interval, timeout
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.value = None
        self.expires = float("-inf")
        self.error = "shadow_unavailable"
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="shadow-status-cache", daemon=True)
        self.thread.start()
        return self

    def poll_once(self, opener=urlopen):
        nonce = uuid.uuid4().hex
        started = self.clock()
        request = Request(self.origin + "/status?nonce=" + nonce,
                          headers={"Cache-Control": "no-cache", "Connection": "close"})
        with opener(request, timeout=self.timeout) as response:
            if response.status != 200 or response.headers.get_content_type() != "application/json":
                raise ValueError("shadow_http_contract_mismatch")
            payload = response.read(256 * 1024 + 1)
        received = self.clock()
        if len(payload) > 256 * 1024:
            raise ValueError("shadow_status_too_large")
        value = validate_shadow_status(json.loads(payload), nonce)
        remaining = min(SHADOW_MAX_AGE_S, float(value["display_valid_for_s"]))-(received-started)
        if value["state"] == "CANDIDATE" and remaining <= 0:
            raise ValueError("shadow_status_expired_in_transit")
        with self.lock:
            self.value = value
            self.expires = received+max(0.0, remaining)
            self.error = ""
        return value

    def fail(self, reason):
        with self.lock:
            self.value = None
            self.expires = float("-inf")
            self.error = str(reason)

    def snapshot(self):
        now = self.clock()
        with self.lock:
            value = deepcopy(self.value)
            error = self.error
            expires = self.expires
        if value is None or now >= expires:
            return ({
                "state": "UNKNOWN", "reason": error or "shadow_status_expired",
                "source": "rgb_posec3d_shadow", "current": False,
                "display_valid_for_s": 0.0, "model_hashes": {},
                "warnings": ["shadow unavailable; existing incident pipeline remains active"],
            }, [])
        diagnostics = {
            "state": value["state"], "reason": value.get("reason", ""),
            "source": "rgb_posec3d_shadow", "current": True,
            "session_id": value["session_id"], "sequence": value["sequence"],
            "source_timestamp_ns": value.get("source_timestamp_ns"),
            "result_age_s": value.get("result_age_s"),
            "display_valid_for_s": max(0.0, expires-now),
            "model_hashes": deepcopy(value["model_hashes"]),
            "warnings": deepcopy(value["warnings"]),
            "metrics": deepcopy(value.get("metrics", {})),
        }
        candidates = deepcopy(value["behavior_candidates"]) if value["state"] == "CANDIDATE" else []
        return diagnostics, candidates

    def _run(self):
        while not self.stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.fail(f"{type(exc).__name__}: {exc}")
            self.stop.wait(self.poll_interval)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def person_records(detections):
    records = []
    for detection in detections or ():
        if not isinstance(detection, dict) or detection.get("class_name") != "person":
            continue
        bbox, confidence = normalized_bbox(detection.get("bbox")), _confidence(detection.get("confidence"))
        track_id = detection.get("track_id", -1)
        if bbox is None or confidence is None or type(track_id) is not int:
            continue
        records.append(dict(class_name="person", confidence=confidence,
                            bbox=list(bbox), track_id=track_id))
    return records


def littering_prediction_records(predictions):
    records = []
    for prediction in predictions or ():
        if hasattr(prediction, "to_dict"):
            prediction = prediction.to_dict()
        if not isinstance(prediction, dict) or str(prediction.get("subtype", "")).lower() != "littering":
            continue
        bbox, confidence = normalized_bbox(prediction.get("bbox")), _confidence(prediction.get("confidence"))
        track_ids = prediction.get("track_ids", [])
        if (bbox is None or confidence is None or not isinstance(track_ids, (list, tuple))
                or any(type(item) is not int for item in track_ids)):
            continue
        records.append(dict(event_type="LITTERING", subtype="littering", state="CANDIDATE",
                            confidence=confidence, bbox=list(bbox), track_ids=list(track_ids)))
    return records


def confirmed_littering_records(events):
    records = []
    for event in events or ():
        if (not isinstance(event, dict) or event.get("event_type") != "LITTERING"
                or event.get("state") != "CONFIRMED"):
            continue
        bbox, confidence = normalized_bbox(event.get("bbox")), _confidence(event.get("confidence"))
        track_ids = event.get("track_ids", [])
        if (bbox is None or confidence is None or not isinstance(track_ids, (list, tuple))
                or any(type(item) is not int for item in track_ids)):
            continue
        records.append(dict(event_type="LITTERING", subtype=str(event.get("subtype", "littering")),
                            state="CONFIRMED", confidence=confidence, bbox=list(bbox),
                            track_ids=list(track_ids)))
    return records


def annotation_operations(width, height, people, candidates, confirmed):
    """Build deterministic drawing operations; kept pure for PC-side tests."""
    if type(width) is not int or type(height) is not int or width < 2 or height < 2:
        raise ValueError("invalid_annotation_dimensions")
    operations = []
    groups = (
        (people, PERSON_COLOR_BGR, "person"),
        (candidates, LITTERING_CANDIDATE_COLOR_BGR, "LITTERING candidate"),
        (confirmed, LITTERING_CONFIRMED_COLOR_BGR, "LITTERING confirmed"),
    )
    for records, color, label in groups:
        for record in records:
            bbox = normalized_bbox(record.get("bbox")) if isinstance(record, dict) else None
            confidence = _confidence(record.get("confidence")) if isinstance(record, dict) else None
            if bbox is None or confidence is None:
                continue
            x1 = max(0, min(width - 1, int(round(bbox[0] * (width - 1)))))
            y1 = max(0, min(height - 1, int(round(bbox[1] * (height - 1)))))
            x2 = max(x1 + 1, min(width - 1, int(round(bbox[2] * (width - 1)))))
            y2 = max(y1 + 1, min(height - 1, int(round(bbox[3] * (height - 1)))))
            suffix = ""
            if label == "person" and record.get("track_id", -1) >= 0:
                suffix = " #" + str(record["track_id"])
            operations.append(dict(box=(x1, y1, x2, y2), color=color,
                                   label=f"{label}{suffix} {confidence:.2f}"))
    return operations


def annotated_jpeg(image, people, candidates, confirmed, cv2, *, quality=EVIDENCE_JPEG_QUALITY):
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("invalid_jpeg_quality")
    canvas = image.copy()
    for operation in annotation_operations(int(canvas.shape[1]), int(canvas.shape[0]),
                                            people, candidates, confirmed):
        x1, y1, x2, y2 = operation["box"]
        color = operation["color"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        text_y = max(18, y1 - 7)
        cv2.putText(canvas, operation["label"], (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("annotated_jpeg_encode_failed")
    payload = encoded.tobytes()
    jpeg_dimensions(payload)
    return payload


@dataclass(frozen=True)
class Frame:
    sequence: int
    source_timestamp_ns: int
    received_at: float
    jpeg: bytes


class Bench:
    def __init__(self, output_root, *, cooldown_s=5., event_filter="LITTERING",
                 clock=time.monotonic, shadow_cache=None):
        if not math.isfinite(cooldown_s) or not 0 <= cooldown_s <= 3600:
            raise ValueError("invalid_photo_cooldown")
        if event_filter != "LITTERING":
            raise ValueError("this bench supports the LITTERING experiment only")
        self.clock, self.cooldown_s, self.event_filter = clock, cooldown_s, event_filter
        self.shadow_cache = shadow_cache
        self.session_id = str(uuid.uuid4())
        self.directory = Path(output_root).expanduser().resolve() / ("session_" + self.session_id)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.photos = self.directory / "photos"
        self.photos.mkdir()
        self.journal = (self.directory / "events.jsonl").open("x", encoding="utf-8", buffering=1)
        self.diagnostics = AsyncJournal.open(self.directory / "diagnostics.jsonl")
        self.lock, self.condition = threading.RLock(), threading.Condition()
        self.stop = threading.Event()
        self.latest = None
        self.received_frames = self.last_stamp = self.sequence = self.consumed = 0
        self.dropped_stale = self.rejected_frames = 0
        self.status, self.reason = "STARTING", "starting"
        self.error = ""
        self.width = self.height = None
        self.inferred_at = self.last_source_received = None
        self.inferred_input_received = None
        self.inference = dict(inference_ms=None, temporal_window_ready=False, person_tracks=[],
                              raw_scores=[], raw_scores_available=False, modules={}, model_hashes={},
                              warnings=["UNVALIDATED MODEL: general_detection (VAL_REJECTED)",
                                        "UNVALIDATED MODEL: abnormal_behavior (VAL_REJECTED)"])
        self.monitor_frame = None
        self.monitor_people, self.monitor_predictions, self.monitor_confirmed = [], [], []
        self.events, self.seen, self.last_photo = deque(maxlen=20), set(), {}
        self.photo_map, self.annotated_photo_map = {}, {}
        atomic_new(self.directory / "session.json", json.dumps(dict(
            schema_version=1, session_id=self.session_id, mode=MODE, flight_commands_enabled=False,
            physical_fc_commands=0, started_utc=utc_now(), rgb_topic=TOPIC,
            event_filter=event_filter, same_class_photo_cooldown_s=cooldown_s,
            retention="indefinite", photos_only=True), indent=2).encode())

    def submit(self, jpeg, source_timestamp_ns):
        now = self.clock()
        try:
            dimensions = jpeg_dimensions(jpeg)
        except (ValueError, IndexError):
            with self.lock:
                self.rejected_frames += 1
            return False
        with self.lock:
            if (type(source_timestamp_ns) is not int or source_timestamp_ns <= self.last_stamp
                    or not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= MAX_JPEG
                    or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9")):
                self.rejected_frames += 1
                return False
            self.last_stamp = source_timestamp_ns
            self.received_frames += 1
            self.last_source_received = now
            self.latest = Frame(self.received_frames, source_timestamp_ns, now, jpeg)
            self.width, self.height = dimensions
        with self.condition:
            self.condition.notify()
        return True

    def snapshot(self, nonce=""):
        now = self.clock()
        with self.lock:
            self.sequence += 1
            age = None if self.last_source_received is None else max(0., now-self.last_source_received)
            inference_age = None if self.inferred_at is None else max(0., now-self.inferred_at)
            input_age = None if self.inferred_input_received is None else max(0.,now-self.inferred_input_received)
            state, reason = self.status, self.error or self.reason
            if state == "RUNNING" and (age is None or age > .5):
                state, reason = "NO_RGB", "camera_input_missing_or_stale"
            elif state == "RUNNING" and (input_age is None or input_age > 2.):
                state, reason = "WARMING_UP", "inference_result_missing_or_stale"
            return dict(schema_version=1, mode=MODE, flight_commands_enabled=False, physical_fc_commands=0,
                        request_nonce=nonce, session_id=self.session_id, sequence=self.sequence,
                        status=state, status_reason=reason, display_valid_for_s=1.,
                        camera=dict(received_frames=self.received_frames, source_timestamp_ns=self.last_stamp or None,
                                    age_s=age, width=self.width, height=self.height,
                                    dropped_stale=self.dropped_stale, rejected_frames=self.rejected_frames),
                        inference=dict(deepcopy(self.inference), age_s=inference_age,
                                       result_input_age_s=input_age, current_result_valid=input_age is not None and input_age <= 2.,
                                       raw_scores_age_s=input_age if self.inference["raw_scores_available"] else None),
                         events=deepcopy(list(self.events)), output_directory=str(self.directory))

    def monitor_snapshot(self, nonce=""):
        now = self.clock()
        with self.lock:
            base = self.snapshot(nonce)
            frame = self.monitor_frame
            age = None if frame is None else max(0.0, now-frame[0].received_at)
            valid_for = 0.0 if age is None else max(0.0, min(1.0, 1.0-age))
            shadow, candidates = (self.shadow_cache.snapshot() if self.shadow_cache is not None
                                  else ShadowCache(clock=self.clock).snapshot())
            return dict(
                schema_version=1, mode=MODE, flight_commands_enabled=False, physical_fc_commands=0,
                request_nonce=nonce, session_id=self.session_id, sequence=base["sequence"],
                status=base["status"], status_reason=base["status_reason"],
                display_valid_for_s=valid_for,
                frame=(dict(sequence=frame[0].sequence,
                            source_timestamp_ns=frame[0].source_timestamp_ns,
                            age_s=age, annotated=True) if frame is not None else None),
                people=deepcopy(self.monitor_people),
                littering_predictions=deepcopy(self.monitor_predictions),
                confirmed_current_frame=deepcopy(self.monitor_confirmed),
                events=deepcopy(list(self.events)),
                model_hashes=deepcopy(self.inference.get("model_hashes", {})),
                warnings=deepcopy(self.inference.get("warnings", [])),
                shadow_behavior=shadow,
                behavior_candidates=candidates,
            )

    def photo_path(self, event_id):
        try:
            if not canonical_id(event_id):
                return None
        except (ValueError, AttributeError):
            return None
        with self.lock:
            return self.photo_map.get(event_id)

    def annotated_photo_path(self, event_id):
        try:
            if not canonical_id(event_id):
                return None
        except (ValueError, AttributeError):
            return None
        with self.lock:
            return self.annotated_photo_map.get(event_id)

    def preview(self):
        with self.lock:
            frame = self.latest
        if frame is None:
            return None
        age = self.clock()-frame.received_at
        return (frame,age) if 0 <= age <= 1. else None

    def annotated_preview(self):
        with self.lock:
            value = self.monitor_frame
        if value is None:
            return None
        frame, payload = value
        age = self.clock()-frame.received_at
        return (frame, payload, age) if 0 <= age <= 1. else None

    def save_event(self, event, frame, *, annotated_payload=None, annotation_error="",
                   inference_duration_s=None):
        """Called only after real current-frame coverage; exact inference-input JPEG."""
        original_model_event = deepcopy(event)
        event_id_source = "model_supplied"
        # v2 demo runtime omits IDs. Only already-CONFIRMED, ID-ABSENT model
        # records receive a session-bound storage ID; this creates no judgment.
        # An explicitly supplied invalid ID is still rejected below, not fixed.
        if "event_id" not in event and event.get("state") == "CONFIRMED":
            try:
                canonical = json.dumps(dict(timestamp_s=event.get("timestamp_s"),
                    event_type=event.get("event_type"), subtype=event.get("subtype"),
                    track_ids=event.get("track_ids"), model_sha256=event.get("model_sha256"),
                    original_model_event=event), sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False, allow_nan=False)
                event = dict(event, event_id=str(uuid.uuid5(uuid.UUID(self.session_id),
                    "phase1-confirmed-record-v2:"+canonical)))
                event_id_source = "session_derived"
            except (TypeError, ValueError) as exc:
                self.diagnostics.write(json.dumps(dict(error="invalid_model_event_for_record_id",detail=str(exc)))+"\n")
                return None
        event_id = event.get("event_id")
        if not canonical_id(event_id) or event.get("state") != "CONFIRMED":
            self.diagnostics.write(json.dumps(dict(error="invalid_confirmed_event_id_or_state", event_id=event_id))+"\n")
            return None
        confidence = event.get("confidence")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            self.diagnostics.write(json.dumps(dict(error="invalid_event_confidence",event_id=event_id))+"\n")
            return None
        if event.get("event_type") != self.event_filter or event_id in self.seen:
            return None
        self.seen.add(event_id)
        now = self.clock()
        bbox = normalized_bbox(event.get("bbox"))
        track_ids = event.get("track_ids", [])
        if not isinstance(track_ids, (list, tuple)) or any(type(item) is not int for item in track_ids):
            track_ids = []
        row = dict(event_id=event_id, event_type=event["event_type"], subtype=event.get("subtype"),
                   confidence=confidence, model_sha256=event.get("model_sha256", ""), observed_utc=utc_now(),
                   source_timestamp_ns=frame.source_timestamp_ns, source_frame_sequence=frame.sequence,
                   camera_received_monotonic_s=frame.received_at, confirmed_monotonic_s=now,
                   model_event_timestamp_s=event.get("timestamp_s"), photo_url=None, photo_sha256=None,
                   bbox=list(bbox) if bbox is not None else None, track_ids=list(track_ids),
                   raw_photo_url=None, raw_photo_sha256=None,
                   annotated_photo_url=None, annotated_photo_sha256=None,
                   annotation_status="SUPPRESSED_COOLDOWN", annotation_error="",
                   metadata_status="PENDING", metadata_error="",
                   capture_status="SUPPRESSED_COOLDOWN", error="", session_id=self.session_id,
                   original_model_event=original_model_event, event_id_source=event_id_source,
                   inference_duration_s=inference_duration_s, confirmation_input_age_s=now-frame.received_at,
                   inference_late=now-frame.received_at > 2.,
                   warning="late_confirmed_historical_evidence_not_current_judgment" if now-frame.received_at > 2. else "")
        photo = annotated_photo = None
        if now-self.last_photo.get(event["event_type"], float("-inf")) >= self.cooldown_s:
            try:
                photo = self.photos / (event_id + ".jpg")
                atomic_new(photo, frame.jpeg)
                raw_digest = hashlib.sha256(frame.jpeg).hexdigest()
                row.update(photo_sha256=raw_digest, photo_url="/photos/" + event_id + ".jpg",
                           raw_photo_sha256=raw_digest,
                           raw_photo_url="/monitor/v1/photos/" + event_id + "/raw.jpg",
                           capture_status="SAVED", photo_path=str(photo))
                if annotated_payload is not None:
                    annotated_photo = self.photos / (event_id + ".annotated.jpg")
                    atomic_new(annotated_photo, annotated_payload)
                    row.update(annotation_status="SAVED", annotation_error="",
                               annotated_photo_sha256=hashlib.sha256(annotated_payload).hexdigest(),
                               annotated_photo_url="/monitor/v1/photos/" + event_id + "/annotated.jpg",
                               annotated_photo_path=str(annotated_photo))
                else:
                    row.update(annotation_status="ERROR",
                               annotation_error=annotation_error or "annotated_frame_unavailable")
                self.last_photo[event["event_type"]] = now
            except Exception as exc:
                if photo is None or not photo.is_file():
                    row.update(capture_status="ERROR", photo_url=None, photo_sha256=None,
                               raw_photo_url=None, raw_photo_sha256=None,
                               error=f"{type(exc).__name__}: {exc}")
                else:
                    row.update(annotation_status="ERROR",
                               annotation_error=f"{type(exc).__name__}: {exc}")
        try:
            row.update(metadata_status="SAVED", metadata_error="")
            atomic_new(self.photos / (event_id + ".json"),
                       json.dumps(row, allow_nan=False, indent=2).encode())
        except Exception as exc:
            row.update(metadata_status="ERROR", metadata_error=f"{type(exc).__name__}: {exc}")
        self.journal.write(json.dumps(row, allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        with self.lock:
            if row["capture_status"] == "SAVED":
                self.photo_map[event_id] = photo
            if row["annotation_status"] == "SAVED":
                self.annotated_photo_map[event_id] = annotated_photo
            self.events.appendleft(row)
        return row

    def process_result(self, result, runtime, frame, *, image, cv2, coverage_functions,
                       inference_duration_s=None):
        coverage_fn, covered_fn = coverage_functions
        coverage, complete = coverage_fn(result.modules, result.detections, result.temporal_window_ready)
        original = getattr(getattr(runtime, "_runtime", None), "last_snapshot", None)
        predictions = getattr(original, "temporal_predictions", ()) or ()
        scores = [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in predictions]
        people = person_records(result.detections)
        candidates = littering_prediction_records(scores)
        confirmed = confirmed_littering_records(result.events)
        warnings = ["UNVALIDATED MODEL: general_detection (VAL_REJECTED)",
                    "UNVALIDATED MODEL: abnormal_behavior (VAL_REJECTED)"]
        person_inputs = [item for item in (result.detections or ())
                         if isinstance(item, dict) and item.get("class_name") == "person"]
        prediction_inputs = [item for item in scores
                             if isinstance(item, dict) and str(item.get("subtype", "")).lower() == "littering"]
        confirmed_inputs = [item for item in (result.events or ()) if isinstance(item, dict)
                            and item.get("event_type") == "LITTERING" and item.get("state") == "CONFIRMED"]
        if len(people) != len(person_inputs):
            warnings.append("invalid_person_bbox_or_fields_omitted")
        if len(candidates) != len(prediction_inputs):
            warnings.append("invalid_littering_candidate_bbox_or_fields_omitted")
        if len(confirmed) != len(confirmed_inputs):
            warnings.append("invalid_confirmed_littering_bbox_or_fields_omitted")
        annotation_error = ""
        try:
            overlay = annotated_jpeg(image, people, candidates, confirmed, cv2,
                                     quality=EVIDENCE_JPEG_QUALITY)
            live_overlay = annotated_jpeg(image, people, candidates, confirmed, cv2,
                                          quality=LIVE_JPEG_QUALITY)
        except Exception as exc:
            overlay = None
            live_overlay = None
            annotation_error = f"{type(exc).__name__}: {exc}"
        health = runtime.health()
        now = self.clock()
        with self.lock:
            self.inferred_at = now
            self.inferred_input_received = frame.received_at
            self.status, self.reason = "RUNNING", "observing_only_no_control"
            self.inference.update(inference_ms=result.inference_ms,
                temporal_window_ready=result.temporal_window_ready, person_tracks=deepcopy(people),
                raw_scores=scores, raw_scores_available=bool(scores),
                modules=deepcopy(result.modules), coverage=coverage, coverage_complete=complete,
                source_timestamp_ns=frame.source_timestamp_ns, inference_duration_s=inference_duration_s,
                result_late=now-frame.received_at > 2.,
                model_hashes={k:v.get("sha256", "") for k,v in health.get("models", {}).items()},
                warnings=warnings)
            self.monitor_frame = (frame, live_overlay) if live_overlay is not None else None
            self.monitor_people = deepcopy(people)
            self.monitor_predictions = deepcopy(candidates)
            self.monitor_confirmed = deepcopy(confirmed)
        diagnostic = dict(monotonic_s=now, source_timestamp_ns=frame.source_timestamp_ns,
                          source_frame_sequence=frame.sequence, inference=self.snapshot()["inference"],
                          actual_events=list(result.events), annotation_error=annotation_error)
        if hasattr(self.diagnostics, 'status'):
            diagnostic['journal_delivery'] = self.diagnostics.status()
        # High-rate inference diagnostics are not evidence photos. Keep photo/
        # event persistence unchanged, but never pause live inference for this log.
        writer = getattr(self.diagnostics, 'write_diagnostic', self.diagnostics.write)
        writer(json.dumps(diagnostic, allow_nan=False) + "\n")
        for event in result.events:
            if (event.get("state") == "CONFIRMED" and event.get("event_type") == self.event_filter
                    and covered_fn(event["event_type"], coverage)):
                event_bbox_valid = normalized_bbox(event.get("bbox")) is not None
                self.save_event(event, frame, annotated_payload=overlay if event_bbox_valid else None,
                                annotation_error=(annotation_error if event_bbox_valid
                                                  else "confirmed_event_bbox_invalid"),
                                inference_duration_s=inference_duration_s)

    def worker(self, phase1_root, device):
        try:
            import cv2
            import numpy as np
            from jolgwa_uav.phase1_inference import Phase1JetsonRuntime
            from jolgwa_uav.incident_coverage import incident_coverage, incident_event_is_covered
            with self.lock:
                self.status, self.reason = "WARMING_UP", "loading_original_four_models"
            runtime = Phase1JetsonRuntime(phase1_root, device=device)
            runtime.warmup()
            health = runtime.health()
            if not health.get("loaded") or not all(health.get("models", {}).get(k, {}).get("healthy") is True
                    for k in Phase1JetsonRuntime.REQUIRED_MODELS):
                raise ValueError("original_four_model_health_incomplete")
            with self.lock:
                self.status, self.reason = "RUNNING", "waiting_for_existing_camera_rgb"
                self.inference["model_hashes"] = {k:v.get("sha256", "") for k,v in health["models"].items()}
            last_run = 0.
            while not self.stop.is_set():
                with self.lock:
                    frame = self.latest
                now = self.clock()
                if frame is None or frame.sequence <= self.consumed or now-last_run < .1:
                    with self.condition:
                        self.condition.wait(timeout=.02)
                    continue
                self.consumed = frame.sequence
                if not 0 <= now-frame.received_at <= .5:
                    self.dropped_stale += 1
                    continue
                # submit() bounded the SOF dimensions before this single decode.
                dimensions = jpeg_dimensions(frame.jpeg)
                image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (dimensions[1], dimensions[0], 3):
                    raise ValueError("jpeg_dimensions_invalid")
                with self.lock:
                    self.height, self.width = image.shape[:2]
                last_run = self.clock()
                if not 0 <= last_run-frame.received_at <= .5:
                    self.dropped_stale += 1
                    continue
                result = runtime.evaluate_bgr(image, timestamp_s=frame.received_at)
                # Preserve actual confirmed evidence even after slow inference:
                # it is marked historical/late, never presented as current safety.
                self.process_result(result, runtime, frame, image=image, cv2=cv2,
                                    coverage_functions=(incident_coverage, incident_event_is_covered),
                                    inference_duration_s=self.clock()-last_run)
        except Exception as exc:
            with self.lock:
                self.status, self.error = "ERROR", f"{type(exc).__name__}: {exc}"
            self.diagnostics.write(json.dumps(dict(error=self.error, observed_utc=utc_now()))+"\n")

    def close(self):
        self.journal.close()
        self.diagnostics.close()


def handler_for(bench):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            nonce = parse_qs(parsed.query).get("nonce", [""])[0]
            if len(nonce) > 128:
                self.send_error(400)
                return
            if parsed.path == "/status":
                body = json.dumps(bench.snapshot(nonce), allow_nan=False).encode()
                mime = "application/json"
                headers = {}
            elif parsed.path == "/monitor/v1/status":
                body = json.dumps(bench.monitor_snapshot(nonce), allow_nan=False).encode()
                mime = "application/json"
                headers = {}
            elif parsed.path == "/preview.jpg":
                preview = bench.preview()
                if preview is None:
                    self.send_error(503, "camera_frame_unavailable_or_stale")
                    return
                frame, age = preview
                body, mime = frame.jpeg, "image/jpeg"
                headers = {"X-Frame-Sequence": str(frame.sequence),
                           "X-Source-Timestamp-Ns": str(frame.source_timestamp_ns),
                           "X-Frame-Age-Ms": f"{age*1000:.3f}"}
            elif parsed.path == "/monitor/v1/preview.jpg":
                preview = bench.annotated_preview()
                if preview is None:
                    self.send_error(503, "annotated_frame_unavailable_or_stale")
                    return
                frame, body, age = preview
                mime = "image/jpeg"
                headers = {"X-Frame-Sequence": str(frame.sequence),
                           "X-Source-Timestamp-Ns": str(frame.source_timestamp_ns),
                           "X-Frame-Age-Ms": f"{age*1000:.3f}",
                           "X-Annotation": "person-littering-v1"}
            elif parsed.path.startswith("/monitor/v1/photos/"):
                parts = parsed.path.strip("/").split("/")
                if len(parts) != 5 or parts[:3] != ["monitor", "v1", "photos"] or not canonical_id(parts[3]):
                    self.send_error(404)
                    return
                if parts[4] == "raw.jpg":
                    photo = bench.photo_path(parts[3])
                elif parts[4] == "annotated.jpg":
                    photo = bench.annotated_photo_path(parts[3])
                else:
                    photo = None
                if photo is None:
                    self.send_error(404)
                    return
                body, mime = photo.read_bytes(), "image/jpeg"
                headers = {}
            elif parsed.path.startswith("/photos/") and parsed.path.endswith(".jpg"):
                photo = bench.photo_path(parsed.path[8:-4])
                if photo is None:
                    self.send_error(404)
                    return
                body, mime = photo.read_bytes(), "image/jpeg"
                headers = {}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--photo-cooldown-s", type=float, default=5.)
    parser.add_argument("--duration-s", type=float, default=600.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration_s) or not 10 <= args.duration_s <= 3600:
        parser.error("duration-s must be 10..3600")
    os.environ.update(ULTRALYTICS_AUTOINSTALL="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    shadow_cache = ShadowCache().start()
    bench = Bench(args.output_root, cooldown_s=args.photo_cooldown_s,
                  shadow_cache=shadow_cache)
    server = ThreadingHTTPServer(("127.0.0.1", 8878), handler_for(bench))
    server.daemon_threads = True
    rclpy.init()
    node = Node("incident_photo_observe_only")
    def receive(message):
        stamp = message.header.stamp.sec * 1000000000 + message.header.stamp.nanosec
        bench.submit(bytes(message.data), stamp)
    node.create_subscription(CompressedImage, TOPIC, receive, qos_profile_sensor_data)
    http = threading.Thread(target=server.serve_forever, daemon=True)
    inference = threading.Thread(target=bench.worker, args=(args.phase1_root,args.device), daemon=True)
    http.start()
    inference.start()
    deadline = time.monotonic()+args.duration_s
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
    except KeyboardInterrupt:
        pass
    finally:
        bench.stop.set()
        shadow_cache.close()
        inference.join(timeout=5.)
        server.shutdown()
        server.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if not inference.is_alive():
            bench.close()


if __name__ == "__main__":
    main()
