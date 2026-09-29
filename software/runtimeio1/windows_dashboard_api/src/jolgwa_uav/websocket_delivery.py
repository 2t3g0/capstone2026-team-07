"""Bounded single-writer delivery. Times are local monotonic residence only."""
import asyncio
import copy
import math
import logging
import time
from collections import OrderedDict, deque

SEND_TIMEOUT_S = 0.250
TELEMETRY_TYPES = {"vehicle.state", "mission.status", "mission.route_frame_status"}


def telemetry_key(message):
    kind = message.get("type")
    if kind not in TELEMETRY_TYPES:
        return None
    return (kind, str(message.get("proposal_id", "")) if kind == "mission.route_frame_status" else "")


def with_residence_age(message, queued_at, now=None):
    payload = copy.deepcopy(message)
    elapsed = (time.monotonic() if now is None else now)-queued_at
    fields = ("gateway_vehicle_age_ms", "geo_age_ms") if payload.get("type") == "vehicle.state" else ()
    for field in fields:
        value = payload.get(field)
        if field in payload:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 and math.isfinite(elapsed) and elapsed >= 0
            payload[field] = min(2**32-1, math.ceil(value+elapsed*1000)) if valid else 2**32-1
    if payload.get("type") == "vehicle.state":
        idle_ground = (payload.get("armed") is False and payload.get("landed") is True
            and payload.get("arming_state_valid") is True and payload.get("landed_state_valid") is True
            and not payload.get("active_execution_mission_id") and payload.get("terminal_in_progress") is False)
        age = payload.get("gateway_vehicle_age_ms", 2**32-1)
        if age >= 500 and "emergency_land_available" in payload:
            payload["emergency_land_available"] = False
        if age >= (1000 if idle_ground else 500):
            for field in ("low_speed_safety_valid", "altitude_alignment_ready"):
                if field in payload:
                    payload[field] = False
        if payload.get("geo_age_ms", 2**32-1) > 750 and "geo_valid" in payload:
            payload["geo_valid"] = False
    return payload


class SocketWriter:
    """One owner per socket; broadcast enqueues without waiting for other peers."""
    def __init__(self, send_json, on_failure):
        self._send_json = send_json
        self._on_failure = on_failure
        self._critical = deque()
        self._telemetry = OrderedDict()
        self._wake = asyncio.Event()
        self._closed = False
        self._sequence = 0
        self._task = asyncio.create_task(self._run())

    def enqueue(self, message):
        future = asyncio.get_running_loop().create_future()
        if self._closed:
            future.set_result(False)
            return future
        self._sequence += 1
        item = (self._sequence, copy.deepcopy(message), time.monotonic(), future)
        key = telemetry_key(message)
        if key is None:
            if len(self._critical) >= 256:
                logging.getLogger(__name__).warning("socket_queue_overflow generation=%s critical=%s telemetry=%s oldest_age_s=%s", id(self), len(self._critical), len(self._telemetry), time.monotonic()-self._critical[0][2])
                future.set_result(False)
                self._closed = True
            else:
                self._critical.append(item)
        else:
            replaced = self._telemetry.pop(key, None)
            if replaced is not None and not replaced[3].done():
                replaced[3].set_result(False)
            self._telemetry[key] = item
            if len(self._telemetry) > 64:
                dropped = self._telemetry.popitem(last=False)[1][3]
                if not dropped.done():
                    dropped.set_result(False)
        self._wake.set()
        return future

    async def send(self, message):
        return await self.enqueue(message)

    async def close(self):
        already_closed = self._closed
        self._closed = True
        self._wake.set()
        if self._task is not asyncio.current_task():
            if not already_closed:
                self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    def discard_telemetry(self):
        for item in self._telemetry.values():
            if not item[3].done():
                item[3].set_result(False)
        self._telemetry.clear()

    async def _run(self):
        active = None
        failed = False
        phase = "idle"
        send_started = None
        try:
            while not self._closed:
                if not self._critical and not self._telemetry:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                first = next(iter(self._telemetry.values()), None)
                if self._critical and (first is None or self._critical[0][0] < first[0]):
                    active = self._critical.popleft()
                else:
                    active = self._telemetry.popitem(last=False)[1]
                _, message, queued_at, completion = active
                send_started = None
                phase = "queue"
                budget = SEND_TIMEOUT_S
                if telemetry_key(message) is None:
                    budget -= time.monotonic()-queued_at
                    if budget <= 0:
                        raise TimeoutError("critical frame expired in output queue")
                phase = "send"
                send_started = time.monotonic()
                await asyncio.wait_for(self._send_json(with_residence_age(message, queued_at)), budget)
                if not completion.done():
                    completion.set_result(True)
                active = None
            failed = True  # Queue overflow closes the connection, not just one frame.
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            item_age = time.monotonic()-active[2] if active else None
            logging.getLogger(__name__).warning("socket_writer_failed phase=%s generation=%s type=%s close_code=%s queue_critical=%s queue_telemetry=%s age_s=%s send_s=%s detail=%r",
                phase, id(self), type(exc).__name__, getattr(exc, 'code', None), len(self._critical),
                len(self._telemetry), item_age, time.monotonic()-send_started if send_started is not None else None, exc)
            failed = True
        finally:
            self._closed = True
            remaining = list(self._critical)+list(self._telemetry.values())+([active] if active else [])
            self._critical.clear()
            self._telemetry.clear()
            for item in remaining:
                if not item[3].done():
                    item[3].set_result(False)
            if failed:
                await self._on_failure()
