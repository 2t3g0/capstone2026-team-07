from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class AirSimCameraError(RuntimeError):
    """Raised when the AirSim virtual camera cannot be used."""


class _AirSimClient(Protocol):
    def ping(self) -> bool: ...

    def simGetImages(
        self, requests: Sequence[Any], vehicle_name: str = ""
    ) -> Sequence[Any]: ...

    def simSetCameraPose(
        self, camera_name: str, pose: Any, vehicle_name: str = ""
    ) -> None: ...


@dataclass(frozen=True)
class AirSimCapturedFrame:
    sequence: int
    captured_at: float
    image: Any


def _load_sdk() -> Any:
    try:
        return importlib.import_module("cosysairsim")
    except ImportError as exc:
        raise AirSimCameraError(
            "cosysairsim is required for the AirSim camera backend"
        ) from exc


def _new_client(sdk: Any, host: str, port: int, timeout_s: float) -> Any:
    return sdk.MultirotorClient(
        ip=host, port=port, timeout_value=timeout_s
    )


class AirSimSceneCapture:
    """Continuously decode the latest compressed AirSim Scene frame."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "PatrolDrone",
        camera_name: str = "front_center",
        fps: float = 25.0,
        timeout_s: float = 1.0,
        reconnect_delay_s: float = 0.5,
        sdk_loader: Callable[[], Any] = _load_sdk,
        client_factory: Callable[[Any, str, int, float], Any] = _new_client,
    ) -> None:
        if not host or not vehicle_name or not camera_name:
            raise ValueError("AirSim host, vehicle and camera must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError("AirSim port must be between 1 and 65535")
        if fps <= 0 or timeout_s <= 0 or reconnect_delay_s < 0:
            raise ValueError("AirSim timing values are invalid")
        self.host = host
        self.port = int(port)
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.fps = float(fps)
        self.timeout_s = float(timeout_s)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._sdk_loader = sdk_loader
        self._client_factory = client_factory
        self._sdk: Any | None = None
        self._client: _AirSimClient | None = None
        self._frame: AirSimCapturedFrame | None = None
        self._sequence = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self.last_error = ""

    def start(self) -> "AirSimSceneCapture":
        with self._condition:
            if self._running:
                return self
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                name="airsim-scene-capture",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.timeout_s + 0.5))
        self._thread = None
        self._client = None

    def read_after(
        self, sequence: int, *, timeout_s: float = 0.5
    ) -> AirSimCapturedFrame | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._running:
                if self._frame is not None and self._frame.sequence > sequence:
                    return self._frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return None

    def _connect(self) -> None:
        if self._sdk is None:
            self._sdk = self._sdk_loader()
        client = self._client_factory(
            self._sdk, self.host, self.port, self.timeout_s
        )
        if not bool(client.ping()):
            raise AirSimCameraError(
                f"AirSim ping failed at {self.host}:{self.port}"
            )
        self._client = client

    def _read_scene(self) -> Any:
        import cv2
        import numpy as np

        if self._client is None:
            self._connect()
        assert self._client is not None and self._sdk is not None
        requests = [
            self._sdk.ImageRequest(
                self.camera_name,
                self._sdk.ImageType.Scene,
                pixels_as_float=False,
                compress=True,
            )
        ]
        try:
            responses = self._client.simGetImages(
                requests, vehicle_name=self.vehicle_name
            )
        except Exception as exc:
            if "invalid number of arguments" not in str(exc):
                raise
            rpc = getattr(self._client, "client", None)
            response_type = getattr(self._sdk, "ImageResponse", None)
            if rpc is None or response_type is None:
                raise
            raw = rpc.call(
                "simGetImages", requests, self.vehicle_name, False
            )
            responses = [response_type.from_msgpack(item) for item in raw]
        if len(responses) != 1:
            raise AirSimCameraError(
                f"expected one Scene response, received {len(responses)}"
            )
        encoded = bytes(getattr(responses[0], "image_data_uint8", b""))
        if not encoded:
            message = str(getattr(responses[0], "message", "")).strip()
            raise AirSimCameraError(message or "AirSim Scene frame was empty")
        image = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise AirSimCameraError("AirSim Scene image decode failed")
        return image

    def _run(self) -> None:
        interval = 1.0 / self.fps
        while True:
            with self._condition:
                if not self._running:
                    return
            started = time.monotonic()
            try:
                image = self._read_scene()
                captured_at = time.monotonic()
                with self._condition:
                    self._sequence += 1
                    self._frame = AirSimCapturedFrame(
                        self._sequence, captured_at, image
                    )
                    self.last_error = ""
                    self._condition.notify_all()
            except Exception as exc:
                self._client = None
                self.last_error = str(exc)
                time.sleep(self.reconnect_delay_s)
                continue
            delay = interval - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)


class AirSimVirtualGimbal:
    """Map yaw/pitch commands to a vehicle-relative AirSim camera pose."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "PatrolDrone",
        camera_name: str = "front_center",
        mount_x_m: float = 0.25,
        mount_y_m: float = 0.0,
        mount_z_m: float = 0.0,
        timeout_s: float = 1.0,
        enabled: bool = True,
        sdk_loader: Callable[[], Any] = _load_sdk,
        client_factory: Callable[[Any, str, int, float], Any] = _new_client,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.mount = (
            float(mount_x_m),
            float(mount_y_m),
            float(mount_z_m),
        )
        self.timeout_s = float(timeout_s)
        self.enabled = bool(enabled)
        self._sdk_loader = sdk_loader
        self._client_factory = client_factory
        self._sdk: Any | None = None
        self._client: _AirSimClient | None = None
        self._condition = threading.Condition()
        self._running = self.enabled
        self._pending: tuple[int, float, float] | None = None
        self._generation = 0
        self._applied_generation = 0
        self._thread: threading.Thread | None = None
        self.last_yaw_deg = 0.0
        self.last_pitch_deg = 0.0
        self.last_error = ""
        if self.enabled:
            self._thread = threading.Thread(
                target=self._run,
                name="airsim-virtual-gimbal",
                daemon=True,
            )
            self._thread.start()

    def _connect(self) -> None:
        if self._sdk is None:
            self._sdk = self._sdk_loader()
        client = self._client_factory(
            self._sdk, self.host, self.port, self.timeout_s
        )
        if not bool(client.ping()):
            raise AirSimCameraError(
                f"AirSim ping failed at {self.host}:{self.port}"
            )
        self._client = client

    def set_attitude(self, yaw_deg: float, pitch_deg: float) -> None:
        if not math.isfinite(yaw_deg) or not math.isfinite(pitch_deg):
            raise ValueError("virtual gimbal angles must be finite")
        self.last_yaw_deg = float(yaw_deg)
        self.last_pitch_deg = float(pitch_deg)
        if not self.enabled:
            return
        with self._condition:
            self._generation += 1
            self._pending = (
                self._generation,
                self.last_yaw_deg,
                self.last_pitch_deg,
            )
            self._condition.notify_all()

    def _apply(self, yaw_deg: float, pitch_deg: float) -> None:
        if self._client is None:
            self._connect()
        assert self._client is not None and self._sdk is not None
        position = self._sdk.Vector3r(*self.mount)
        orientation = self._sdk.euler_to_quaternion(
            0.0,
            math.radians(pitch_deg),
            math.radians(yaw_deg),
        )
        pose = self._sdk.Pose(position, orientation)
        try:
            self._client.simSetCameraPose(
                self.camera_name,
                pose,
                vehicle_name=self.vehicle_name,
            )
        except Exception:
            self._client = None
            raise

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running:
                    return
                generation, yaw_deg, pitch_deg = self._pending
                self._pending = None
            try:
                self._apply(yaw_deg, pitch_deg)
                error = ""
            except Exception as exc:
                error = str(exc)
            with self._condition:
                self.last_error = error
                self._applied_generation = max(
                    self._applied_generation, generation
                )
                self._condition.notify_all()

    def wait_until_applied(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            expected = self._generation
            while self._applied_generation < expected and self._running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return self._applied_generation >= expected and not self.last_error

    def stop(self) -> None:
        """A pose command has no velocity to stop; retained for backend parity."""

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.timeout_s + 0.5)
        self._thread = None
        self._client = None
