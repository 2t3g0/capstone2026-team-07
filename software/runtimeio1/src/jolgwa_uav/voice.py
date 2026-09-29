from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2


class VoiceCaptureError(RuntimeError):
    """Base error raised by the voice capture module."""


class AudioDeviceUnavailableError(VoiceCaptureError):
    """Raised when no usable input device or sounddevice backend is available."""


class AudioRecordingError(VoiceCaptureError):
    """Raised when an available input device fails while recording."""


@dataclass(frozen=True, slots=True)
class VoiceRecording:
    """A 16 kHz, mono, signed 16-bit PCM recording and its WAV encoding."""

    pcm_bytes: bytes
    wav_bytes: bytes
    device_name: str | None = None

    @property
    def frame_count(self) -> int:
        return len(self.pcm_bytes) // SAMPLE_WIDTH_BYTES

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / SAMPLE_RATE_HZ

    @property
    def sample_rate_hz(self) -> int:
        return SAMPLE_RATE_HZ

    @property
    def channels(self) -> int:
        return CHANNELS

    @property
    def sample_width_bytes(self) -> int:
        return SAMPLE_WIDTH_BYTES


def pcm16_to_wav_bytes(pcm: bytes | bytearray | memoryview) -> bytes:
    """Serialize little-endian signed 16-bit mono PCM as a 16 kHz WAV file."""

    try:
        pcm_bytes = bytes(pcm)
    except (TypeError, ValueError) as exc:
        raise TypeError("pcm must be a bytes-like object") from exc

    if len(pcm_bytes) % SAMPLE_WIDTH_BYTES:
        raise ValueError("16-bit PCM byte length must be divisible by 2")

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm_bytes)
    return output.getvalue()


def recording_from_pcm16(
    pcm: bytes | bytearray | memoryview,
    *,
    device_name: str | None = None,
) -> VoiceRecording:
    """Build a recording result from PCM supplied by another audio source."""

    pcm_bytes = bytes(pcm)
    return VoiceRecording(
        pcm_bytes=pcm_bytes,
        wav_bytes=pcm16_to_wav_bytes(pcm_bytes),
        device_name=device_name,
    )


def record_fixed_duration(
    duration_seconds: float,
    *,
    device: int | str | None = None,
) -> VoiceRecording:
    """Record a fixed duration from a system input device using sounddevice."""

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a positive finite number")

    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise AudioDeviceUnavailableError(
            "sounddevice is unavailable; install the 'voice' extra and PortAudio"
        ) from exc

    device_info = _query_input_device(sd, device)
    frame_count = max(1, round(duration_seconds * SAMPLE_RATE_HZ))

    try:
        samples = sd.rec(
            frame_count,
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            dtype="int16",
            device=device,
            blocking=True,
        )
    except Exception as exc:
        raise AudioRecordingError(
            f"audio recording failed on input device '{device_info['name']}'"
        ) from exc

    pcm_bytes = np.asarray(samples, dtype="<i2").reshape(-1).tobytes()
    expected_bytes = frame_count * SAMPLE_WIDTH_BYTES
    if len(pcm_bytes) != expected_bytes:
        raise AudioRecordingError(
            f"input device returned {len(pcm_bytes)} bytes; expected {expected_bytes}"
        )
    return recording_from_pcm16(pcm_bytes, device_name=str(device_info["name"]))


def _query_input_device(sounddevice: Any, device: int | str | None) -> dict[str, Any]:
    try:
        device_info = sounddevice.query_devices(device, "input")
    except Exception as exc:
        requested = "default" if device is None else repr(device)
        raise AudioDeviceUnavailableError(
            f"no usable {requested} audio input device was found"
        ) from exc

    if int(device_info.get("max_input_channels", 0)) < CHANNELS:
        name = device_info.get("name", device if device is not None else "default")
        raise AudioDeviceUnavailableError(
            f"audio device '{name}' has no input channels"
        )
    return device_info
