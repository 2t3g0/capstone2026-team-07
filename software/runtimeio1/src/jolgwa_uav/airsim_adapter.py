from __future__ import annotations

import importlib
import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol


CoordinateFrame = Literal["NED"]
FrameKind = Literal["rgb", "depth"]


class AirSimAdapterError(RuntimeError):
    """Base error raised by the read-only Cosys-AirSim adapter."""


class AirSimSdkUnavailableError(AirSimAdapterError):
    """Raised when the optional Cosys-AirSim Python SDK is unavailable."""


class AirSimConnectionError(AirSimAdapterError):
    """Raised when the simulator API cannot be reached."""


class AirSimFrameError(AirSimAdapterError):
    """Raised when Cosys-AirSim returns an incomplete frame."""


class _AirSimClient(Protocol):
    def ping(self) -> bool: ...

    def simGetImages(
        self, requests: Sequence[Any], vehicle_name: str = ""
    ) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class PoseNed:
    """Camera pose in AirSim's local NED frame.

    Position is north/east/down in metres. Orientation is an (x, y, z, w)
    quaternion expressed in the same NED frame.
    """

    position_ned_m: tuple[float, float, float]
    orientation_ned_xyzw: tuple[float, float, float, float]
    coordinate_frame: CoordinateFrame = "NED"


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """One read-only camera product and its acquisition metadata."""

    kind: FrameKind
    width: int
    height: int
    timestamp_ns: int
    pose: PoseNed
    encoding: str
    data: bytes | Sequence[float]
    received_monotonic_s: float
    stale: bool = False
    stale_reason: str | None = None

    @property
    def sample_count(self) -> int:
        return len(self.data)


@dataclass(frozen=True, slots=True)
class AirSimFrameSet:
    """Synchronized RGB and metric-depth products from one API request."""

    rgb: SensorFrame
    depth: SensorFrame
    vehicle_name: str
    camera_name: str
    coordinate_frame: CoordinateFrame = "NED"

    @property
    def stale(self) -> bool:
        return self.rgb.stale or self.depth.stale


@dataclass(slots=True)
class _StreamClock:
    timestamp_ns: int
    last_advance_monotonic_s: float


class CosysAirSimInputAdapter:
    """Read-only RGB/depth input adapter for Cosys-AirSim 3.4.1.

    The SDK import is delayed until a connection or read is requested. No
    vehicle-control API is exposed or invoked by this class.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "PatrolDrone",
        camera_name: str = "front_center",
        timeout_s: float = 10.0,
        stale_after_s: float = 1.0,
        reconnect_attempts: int = 1,
        reconnect_delay_s: float = 0.25,
        read_only: bool = True,
        client_factory: Callable[[str, int, float], _AirSimClient] | None = None,
        sdk_loader: Callable[[], Any] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not vehicle_name or not camera_name:
            raise ValueError("vehicle_name and camera_name must not be empty")
        if timeout_s <= 0 or stale_after_s <= 0:
            raise ValueError("timeout_s and stale_after_s must be positive")
        if reconnect_attempts < 0 or reconnect_delay_s < 0:
            raise ValueError("reconnect settings must not be negative")
        if not read_only:
            raise ValueError("CosysAirSimInputAdapter only supports read-only mode")

        self.host = host
        self.port = port
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.timeout_s = timeout_s
        self.stale_after_s = stale_after_s
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self.read_only = True
        self._client_factory = client_factory
        self._sdk_loader = sdk_loader or self._load_sdk
        self._monotonic = monotonic
        self._sleep = sleep
        self._sdk: Any | None = None
        self._client: _AirSimClient | None = None
        self._connected = False
        self._stream_clocks: dict[FrameKind, _StreamClock] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _load_sdk() -> Any:
        try:
            return importlib.import_module("cosysairsim")
        except ImportError as exc:
            raise AirSimSdkUnavailableError(
                "cosysairsim is required to read the simulator; install it in the active environment"
            ) from exc

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """Create an API client and verify the server with a read-only ping."""

        with self._lock:
            try:
                sdk = self._get_sdk()
                if self._client_factory is None:
                    client = sdk.MultirotorClient(
                        ip=self.host,
                        port=self.port,
                        timeout_value=self.timeout_s,
                    )
                else:
                    client = self._client_factory(
                        self.host, self.port, self.timeout_s
                    )
                if not bool(client.ping()):
                    raise AirSimConnectionError(
                        f"Cosys-AirSim ping failed at {self.host}:{self.port}"
                    )
            except AirSimAdapterError:
                self._drop_connection()
                raise
            except Exception as exc:
                self._drop_connection()
                raise AirSimConnectionError(
                    f"Could not connect to Cosys-AirSim at {self.host}:{self.port}: {exc}"
                ) from exc

            self._client = client
            self._connected = True
            self._stream_clocks.clear()

    def disconnect(self) -> None:
        """Forget the client without invoking any simulator-side API."""

        with self._lock:
            self._drop_connection()

    def read(self) -> AirSimFrameSet:
        """Read compressed RGB and float depth, reconnecting on transport errors."""

        with self._lock:
            last_error: Exception | None = None
            for attempt in range(self.reconnect_attempts + 1):
                try:
                    if not self._connected or self._client is None:
                        self.connect()
                    return self._read_once()
                except AirSimFrameError:
                    raise
                except (AirSimAdapterError, Exception) as exc:
                    last_error = exc
                    self._drop_connection()
                    if attempt < self.reconnect_attempts:
                        self._sleep(self.reconnect_delay_s)

            assert last_error is not None
            if isinstance(last_error, AirSimAdapterError):
                raise last_error
            raise AirSimConnectionError(
                f"Cosys-AirSim read failed after {self.reconnect_attempts + 1} attempt(s): {last_error}"
            ) from last_error

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            self._sdk = self._sdk_loader()
        return self._sdk

    def _drop_connection(self) -> None:
        self._client = None
        self._connected = False
        self._stream_clocks.clear()

    def _read_once(self) -> AirSimFrameSet:
        assert self._client is not None
        sdk = self._get_sdk()
        requests = [
            sdk.ImageRequest(
                self.camera_name,
                sdk.ImageType.Scene,
                pixels_as_float=False,
                compress=True,
            ),
            sdk.ImageRequest(
                self.camera_name,
                sdk.ImageType.DepthPerspective,
                pixels_as_float=True,
                compress=False,
            ),
        ]
        responses = self._client.simGetImages(
            requests, vehicle_name=self.vehicle_name
        )
        if len(responses) != 2:
            raise AirSimFrameError(
                f"Expected RGB and depth responses, received {len(responses)}"
            )

        received_at = self._monotonic()
        rgb = self._convert_response("rgb", responses[0], received_at)
        depth = self._convert_response("depth", responses[1], received_at)
        return AirSimFrameSet(
            rgb=rgb,
            depth=depth,
            vehicle_name=self.vehicle_name,
            camera_name=self.camera_name,
        )

    def _convert_response(
        self, kind: FrameKind, response: Any, received_at: float
    ) -> SensorFrame:
        width = int(getattr(response, "width", 0))
        height = int(getattr(response, "height", 0))
        if width <= 0 or height <= 0:
            message = str(getattr(response, "message", "")).strip()
            suffix = f": {message}" if message else ""
            raise AirSimFrameError(f"Invalid {kind} frame dimensions{suffix}")

        timestamp_ns = int(getattr(response, "time_stamp", 0))
        pose = self._pose_from_response(response)
        stale, stale_reason = self._assess_staleness(
            kind, timestamp_ns, received_at
        )

        if kind == "rgb":
            data = bytes(getattr(response, "image_data_uint8", b""))
            if not data:
                raise AirSimFrameError("RGB response contained no image bytes")
            encoding = "png"
        else:
            data = getattr(response, "image_data_float", ())
            try:
                sample_count = len(data)
            except TypeError as exc:
                raise AirSimFrameError("Depth response was not a float sequence") from exc
            expected = width * height
            if sample_count != expected:
                raise AirSimFrameError(
                    f"Depth response had {sample_count} samples; expected {expected}"
                )
            encoding = "float32_m"

        return SensorFrame(
            kind=kind,
            width=width,
            height=height,
            timestamp_ns=timestamp_ns,
            pose=pose,
            encoding=encoding,
            data=data,
            received_monotonic_s=received_at,
            stale=stale,
            stale_reason=stale_reason,
        )

    @staticmethod
    def _pose_from_response(response: Any) -> PoseNed:
        position = getattr(response, "camera_position", None)
        orientation = getattr(response, "camera_orientation", None)
        try:
            position_ned = (
                float(position.x_val),
                float(position.y_val),
                float(position.z_val),
            )
            orientation_ned = (
                float(orientation.x_val),
                float(orientation.y_val),
                float(orientation.z_val),
                float(orientation.w_val),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AirSimFrameError("Frame did not contain a valid camera pose") from exc

        if not all(math.isfinite(value) for value in (*position_ned, *orientation_ned)):
            raise AirSimFrameError("Frame camera pose contained a non-finite value")
        return PoseNed(position_ned, orientation_ned)

    def _assess_staleness(
        self, kind: FrameKind, timestamp_ns: int, received_at: float
    ) -> tuple[bool, str | None]:
        if timestamp_ns <= 0:
            return True, "missing_timestamp"

        clock = self._stream_clocks.get(kind)
        if clock is None:
            self._stream_clocks[kind] = _StreamClock(timestamp_ns, received_at)
            return False, None
        if timestamp_ns > clock.timestamp_ns:
            clock.timestamp_ns = timestamp_ns
            clock.last_advance_monotonic_s = received_at
            return False, None
        if timestamp_ns < clock.timestamp_ns:
            return True, "timestamp_regressed"
        if received_at - clock.last_advance_monotonic_s >= self.stale_after_s:
            return True, "timestamp_not_advancing"
        return False, None

