from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class EventClipPolicy:
    duration_s: float = 5.0
    fps: int = 25
    codec: str = "h264"
    container: str = "matroska"
    bitrate_bps: int = 8_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("duration_s must be finite and positive")
        if not 1 <= int(self.fps) <= 120:
            raise ValueError("fps must be between 1 and 120")
        if self.codec.lower() != "h264":
            raise ValueError("the Jetson event recorder currently requires H.264")
        if self.container.lower() not in {"matroska", "mkv"}:
            raise ValueError("the event recorder currently requires Matroska")
        if not 100_000 <= int(self.bitrate_bps) <= 100_000_000:
            raise ValueError("bitrate_bps is outside the supported range")


@dataclass(frozen=True)
class EventClipResult:
    succeeded: bool
    video_path: str
    metadata_path: str
    frames_written: int
    elapsed_s: float
    detail: str
    source_frames_received: int = 0
    duplicated_frames: int = 0
    worker_finished: bool = True


def safe_event_id(value: str) -> str:
    cleaned = _SAFE_ID.sub("_", str(value).strip()).strip("._")
    return (cleaned or "event")[:96]


def build_h264_writer_pipeline(
    output_path: str | Path,
    *,
    width: int,
    height: int,
    fps: int = 25,
    bitrate_bps: int = 8_000_000,
    encoder: str = "nvv4l2h264enc",
) -> str:
    path = str(Path(output_path))
    if '"' in path:
        raise ValueError("output path must not contain a double quote")
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    if fps <= 0 or bitrate_bps <= 0:
        raise ValueError("fps and bitrate_bps must be positive")
    common = [
        "appsrc is-live=true format=time do-timestamp=true",
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1",
        "videoconvert",
    ]
    if encoder == "nvv4l2h264enc":
        encoding = [
            "video/x-raw,format=BGRx",
            "nvvidconv",
            "video/x-raw(memory:NVMM),format=NV12",
            (
                "nvv4l2h264enc "
                f"bitrate={bitrate_bps} insert-sps-pps=true iframeinterval={fps}"
            ),
        ]
    elif encoder == "x264enc":
        encoding = [
            "video/x-raw,format=I420",
            (
                "x264enc tune=zerolatency speed-preset=ultrafast "
                f"bitrate={max(1, bitrate_bps // 1000)} key-int-max={fps}"
            ),
        ]
    elif encoder == "openh264enc":
        encoding = [
            "video/x-raw,format=I420",
            (
                "openh264enc complexity=low "
                f"bitrate={bitrate_bps} gop-size={fps}"
            ),
        ]
    else:
        raise ValueError(f"unsupported H.264 GStreamer encoder: {encoder}")
    return " ! ".join(
        common
        + encoding
        + [
            "h264parse config-interval=-1",
            "matroskamux",
            f'filesink location="{path}" sync=false',
        ]
    )


def detect_h264_encoder() -> str:
    inspector = shutil.which("gst-inspect-1.0")
    if inspector is None:
        raise RuntimeError("gst-inspect-1.0 is not installed")
    for encoder in ("nvv4l2h264enc", "x264enc", "openh264enc"):
        completed = subprocess.run(
            [inspector, encoder],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
        if completed.returncode == 0:
            return encoder
    raise RuntimeError("no supported GStreamer H.264 encoder is installed")


def detect_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "neither GStreamer H.264 nor FFmpeg is available"
        ) from exc
    if not executable or not Path(executable).is_file():
        raise RuntimeError("imageio-ffmpeg did not provide an executable")
    return executable


class _FfmpegH264Writer:
    def __init__(
        self,
        executable: str,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate_bps: int,
    ) -> None:
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-b:v",
                str(bitrate_bps),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "matroska",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def isOpened(self) -> bool:
        return self._process.poll() is None and self._process.stdin is not None

    def write(self, image: Any) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError("FFmpeg H.264 writer stopped unexpectedly")
        self._process.stdin.write(image.tobytes())

    def release(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
            self._process.stdin = None
        try:
            _, stderr_bytes = self._process.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                _, stderr_bytes = self._process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                _, stderr_bytes = self._process.communicate(timeout=2.0)
            raise RuntimeError("FFmpeg H.264 writer did not stop in time")
        stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
        returncode = self._process.returncode
        if returncode != 0:
            raise RuntimeError(
                f"FFmpeg H.264 writer failed ({returncode}): {stderr.strip()}"
            )


class EventClipRecorder:
    """Write a fixed-duration H.264/MKV clip from a LatestFrameCapture-like source."""

    def __init__(
        self,
        storage_root: str | Path,
        policy: EventClipPolicy | None = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.policy = policy or EventClipPolicy()

    def record(
        self,
        capture: Any,
        *,
        event_id: str,
        event_type: str,
        track_id: str,
        confidence: float,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
        cancel_event: Any | None = None,
    ) -> EventClipResult:
        started = time.monotonic()
        directory = None
        try:
            self.storage_root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            prefix = f"{stamp}_{safe_event_id(event_type)}_{safe_event_id(event_id)}_"
            directory = Path(tempfile.mkdtemp(prefix=prefix, dir=self.storage_root))
            return self._record_reserved(
                capture, directory=directory, event_id=event_id,
                event_type=event_type, track_id=track_id, confidence=confidence,
                source=source, extra_metadata=extra_metadata, cancel_event=cancel_event,
            )
        except Exception as exc:
            detail = f"recording failed: {type(exc).__name__}: {exc}"
            metadata_path = ""
            if directory is not None:
                failure_path = directory / "failure.json"
                try:
                    _atomic_json_write(failure_path, {
                        "schema_version": 2, "succeeded": False, "detail": detail,
                        "event_id": event_id, "event_type": event_type,
                        "retention": "permanent", "worker_finished": True,
                    })
                    metadata_path = str(failure_path)
                except Exception as journal_exc:
                    detail += f"; failure metadata unavailable: {journal_exc}"
            return EventClipResult(False, "", metadata_path, 0,
                                   max(0.0, time.monotonic() - started), detail)

    def _record_reserved(
        self, capture, *, directory, event_id, event_type, track_id, confidence,
        source, extra_metadata, cancel_event,
    ) -> EventClipResult:
        import cv2

        video_path = directory / "clip.mkv"
        metadata_path = directory / "metadata.json"
        started_monotonic = time.monotonic()
        deadline = started_monotonic + self.policy.duration_s
        previous_sequence = -1
        frames_written = 0
        writer = None
        error = ""
        width = 0
        height = 0
        next_frame_at = started_monotonic
        encoder = ""
        writer_backend = ""
        latest_image = None
        latest_received_at: float | None = None
        source_frames_received = 0
        source_frames_used = 0
        last_written_sequence = -1
        first_source_received_at = None
        last_source_received_at = None
        maximum_source_age_s = 1.0

        try:
            while time.monotonic() < deadline and not (
                cancel_event is not None and cancel_event.is_set()
            ):
                remaining = max(0.0, deadline - time.monotonic())
                captured = capture.read_after(
                    previous_sequence,
                    timeout_s=min(1.0 / self.policy.fps, remaining),
                )
                if captured is not None:
                    sequence = int(captured.sequence)
                    captured_at = float(captured.captured_at)
                    received_now = time.monotonic()
                    if (sequence <= previous_sequence or not math.isfinite(captured_at)
                            or captured_at < 0.0 or captured_at > received_now
                            or (last_source_received_at is not None
                                and captured_at <= last_source_received_at)):
                        raise RuntimeError("camera frame sequence/timestamp is not advancing or valid")
                    if captured_at > deadline:
                        break
                    previous_sequence = sequence
                    latest_image = captured.image
                    latest_received_at = captured_at
                    if received_now - captured_at > maximum_source_age_s:
                        continue
                    source_frames_received += 1
                    if first_source_received_at is None:
                        first_source_received_at = captured_at
                    last_source_received_at = captured_at
                now = time.monotonic()
                if latest_image is None:
                    continue
                if (
                    latest_received_at is None
                    or now - latest_received_at > maximum_source_age_s
                ):
                    continue
                image_height, image_width = latest_image.shape[:2]
                if writer is None:
                    height, width = image_height, image_width
                    try:
                        encoder = detect_h264_encoder()
                    except RuntimeError:
                        encoder = "ffmpeg-libx264"
                        writer_backend = "ffmpeg"
                        writer = _FfmpegH264Writer(
                            detect_ffmpeg(),
                            video_path,
                            width=width,
                            height=height,
                            fps=self.policy.fps,
                            bitrate_bps=self.policy.bitrate_bps,
                        )
                    else:
                        writer_backend = "gstreamer"
                        pipeline = build_h264_writer_pipeline(
                            video_path,
                            width=width,
                            height=height,
                            fps=self.policy.fps,
                            bitrate_bps=self.policy.bitrate_bps,
                            encoder=encoder,
                        )
                        writer = cv2.VideoWriter(
                            pipeline,
                            cv2.CAP_GSTREAMER,
                            0,
                            float(self.policy.fps),
                            (width, height),
                            True,
                        )
                    if not writer.isOpened():
                        raise RuntimeError(
                            "unable to open the H.264 event clip writer"
                        )
                elif (image_width, image_height) != (width, height):
                    latest_image = cv2.resize(latest_image, (width, height))
                while next_frame_at < min(now, deadline) - 1e-9:
                    writer.write(latest_image)
                    if last_written_sequence != previous_sequence:
                        source_frames_used += 1
                        last_written_sequence = previous_sequence
                    frames_written += 1
                    next_frame_at += 1.0 / self.policy.fps
            while (
                writer is not None
                and latest_image is not None
                and next_frame_at < deadline - 1e-6
                and latest_received_at is not None
                and deadline - latest_received_at <= maximum_source_age_s
                and not (
                    cancel_event is not None and cancel_event.is_set()
                )
            ):
                writer.write(latest_image)
                if last_written_sequence != previous_sequence:
                    source_frames_used += 1
                    last_written_sequence = previous_sequence
                frames_written += 1
                next_frame_at += 1.0 / self.policy.fps
        except Exception as exc:
            error = str(exc)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as exc:
                    if not error:
                        error = str(exc)

        elapsed = max(0.0, time.monotonic() - started_monotonic)
        if frames_written and not error and (not video_path.is_file() or video_path.stat().st_size == 0):
            error = "encoder produced no non-empty video file"
        expected_frames = max(
            1, round(self.policy.duration_s * self.policy.fps)
        )
        minimum_source_frames = max(1, math.ceil(self.policy.duration_s))
        cancelled = cancel_event is not None and cancel_event.is_set()
        succeeded = (
            frames_written >= expected_frames
            and source_frames_received >= minimum_source_frames
            and not error
            and not cancelled
        )
        if cancelled and not error:
            error = "capture cancelled during shutdown"
        elif not succeeded and not error:
            error = (
                "insufficient fresh source frames: "
                f"received={source_frames_received}, stored={frames_written}, "
                f"required_source={minimum_source_frames}, "
                f"required_stored={expected_frames}"
            )
        detail = (
            f"{frames_written} frames stored as H.264/MKV at {self.policy.fps} FPS"
            if succeeded
            else (error or "no camera frames received")
        )
        duplicated_frames = max(0, frames_written - source_frames_used)
        metadata: dict[str, Any] = {
            "schema_version": 2,
            "event_id": event_id,
            "event_type": event_type,
            "track_id": track_id,
            "confidence": float(confidence),
            "source": source,
            "retention": "permanent",
            "video_file": video_path.name if frames_written else "",
            "frames_written": frames_written,
            "expected_frames": expected_frames,
            "source_frames_received": source_frames_received,
            "unique_source_frames_used": source_frames_used,
            "duplicated_frames": duplicated_frames,
            "source_clock": "capture adapter host monotonic receipt; not exposure time",
            "first_source_received_monotonic_s": first_source_received_at,
            "last_source_received_monotonic_s": last_source_received_at,
            "source_span_s": (None if first_source_received_at is None else
                              last_source_received_at - first_source_received_at),
            "record_started_monotonic_s": started_monotonic,
            "record_deadline_monotonic_s": deadline,
            "finalized_monotonic_s": time.monotonic(),
            "stored_duration_s": frames_written / self.policy.fps,
            "worker_finished": True,
            "maximum_source_frame_age_s": maximum_source_age_s,
            "width": width,
            "height": height,
            "elapsed_s": round(elapsed, 3),
            "capture_policy": asdict(self.policy),
            "gstreamer_encoder": encoder,
            "writer_backend": writer_backend,
            "succeeded": succeeded,
            "detail": detail,
        }
        if extra_metadata:
            metadata["extra"] = extra_metadata
        _atomic_json_write(metadata_path, metadata)
        return EventClipResult(
            succeeded=succeeded,
            video_path=str(video_path) if frames_written else "",
            metadata_path=str(metadata_path),
            frames_written=frames_written,
            elapsed_s=elapsed,
            detail=detail,
            source_frames_received=source_frames_received,
            duplicated_frames=duplicated_frames,
        )


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False,
                      indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing existing evidence.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_event_bounded(recorder, capture, *, cancel_event=None,
                         finalize_timeout_s=2.0, **metadata) -> EventClipResult:
    """Bound the caller's wait, including codec stalls and disk exceptions.

    A blocked native encoder cannot safely be killed from a Python thread. On
    timeout the result is a failure, cancellation is latched and this recorder
    instance is quarantined; a process restart is required before further clips.
    A late worker result must not replace this terminal timeout evidence.
    """
    if not math.isfinite(finalize_timeout_s) or not 0 < finalize_timeout_s <= 10:
        raise ValueError("finalize timeout must be finite and in (0, 10]")
    if getattr(recorder, "_worker_unresolved", False):
        return EventClipResult(False, "", "", 0, 0.0,
                               "previous recording worker unresolved; restart required",
                               worker_finished=False)
    local_cancel = threading.Event()
    class Cancellation:
        def is_set(self):
            return local_cancel.is_set() or (cancel_event is not None and cancel_event.is_set())
    done = threading.Event()
    result = []
    started = time.monotonic()
    def run():
        try:
            result.append(recorder.record(capture, cancel_event=Cancellation(), **metadata))
        except Exception as exc:
            result.append(EventClipResult(False, "", "", 0,
                          max(0.0, time.monotonic() - started),
                          f"recording worker failed: {type(exc).__name__}: {exc}"))
        finally:
            done.set()
    worker = threading.Thread(target=run, name="bounded-event-writer", daemon=True)
    worker.start()
    deadline = started + recorder.policy.duration_s + finalize_timeout_s
    while not done.is_set() and time.monotonic() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            break
        done.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
    if done.is_set():
        return result[0]
    local_cancel.set()
    recorder._worker_unresolved = True
    detail = "recording deadline/cancellation reached; worker completion unconfirmed; restart required"
    path_text = ""
    try:
        recorder.storage_root.mkdir(parents=True, exist_ok=True)
        path = recorder.storage_root / ("recording_terminal_" + uuid.uuid4().hex + ".json")
        _atomic_json_write(path, {
            "schema_version": 2, "succeeded": False, "detail": detail,
            "event_id": metadata.get("event_id", ""),
            "event_type": metadata.get("event_type", ""),
            "worker_finished": False, "terminal_result": True,
            "frames_written": None, "source_frames_received": None,
            "retention": "permanent",
        })
        path_text = str(path)
    except Exception as exc:
        detail += f"; terminal evidence unavailable: {exc}"
    return EventClipResult(False, "", path_text, 0, max(0.0, time.monotonic() - started),
                           detail, worker_finished=False)
