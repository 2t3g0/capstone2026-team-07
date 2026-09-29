"""Read-only display data; native sensor statistics are NOT flight decisions."""
from __future__ import annotations
import math
import time
import uuid
import numpy as np
from .sensor_observer import ObserverReportLease, unknown_report


def depth_statistics(message):
    from jolgwa_uav.native_depth import native_corridor_stats
    from jolgwa_uav.vertical_avoidance import VerticalAvoidanceConfig
    if message.encoding != "16UC1":
        raise ValueError("requires_raw_16UC1")
    h, w, step = int(message.height), int(message.width), int(message.step)
    if not (0 < h <= 1080 and 0 < w <= 1920 and w*2 <= step <= w*2+4096):
        raise ValueError("invalid_depth_dimensions")
    if len(message.data) != h*step:
        raise ValueError("invalid_depth_length")
    depth = np.ndarray((h,w), dtype=">u2" if message.is_bigendian else "<u2",
                       buffer=bytes(message.data), strides=(step,2)).astype(np.float32)*.001
    valid = np.isfinite(depth) & (depth >= .15) & (depth <= 20)
    stats = native_corridor_stats(depth, valid, VerticalAvoidanceConfig(near_percentile=2), 3.0)
    if stats is None:
        raise ValueError("native_depth_library_unavailable")
    finite = lambda x: float(x) if math.isfinite(x) else None
    return {"front_near_m": finite(stats.center.near_distance_m),
            "front_median_m": finite(stats.center.median_distance_m),
            "upper_roi_near_m": finite(stats.upper.near_distance_m),
            "valid_fraction": float(valid.mean()), "width": w, "height": h,
            "scope": "native camera ROI only; not a vertical gap or avoidance judgment"}


class StatusStore:
    """Use the Jetson's monotonic clock, not the PC's, for source expiration."""
    def __init__(self):
        self.session = str(uuid.uuid4())
        self.lease = ObserverReportLease()
        self.depth_at = self.rgb_at = self.pose_at = float("-inf")
        self.depth_stamp = 0
        self.stats = {}
        self.depth_error = "waiting_for_depth"
        self.rgb_error = "waiting_for_rgb"
        self.rgb_stamp = 0
        self.started_at = time.monotonic()

    def rgb(self, message, now):
        from jolgwa_uav.observer_jpeg import validate_observer_jpeg
        stamp = int(message.header.stamp.sec)*1000000000 + int(message.header.stamp.nanosec)
        if stamp <= self.rgb_stamp:
            self.rgb_error = "rgb_timestamp_not_advancing"
            self.rgb_at = float("-inf")
            return
        self.rgb_stamp = stamp
        try:
            validate_observer_jpeg(message.data, message.format)
            self.rgb_at = now
            self.rgb_error = ""
        except (ValueError, TypeError, BufferError, OverflowError) as exc:
            self.rgb_error = str(exc)
            self.rgb_at = float("-inf")

    def depth(self, message, now):
        stamp = int(message.header.stamp.sec)*1000000000 + int(message.header.stamp.nanosec)
        if stamp <= self.depth_stamp:
            self.depth_error = "depth_timestamp_not_advancing"
            self.depth_at = float("-inf")
            return
        self.depth_stamp = stamp
        try:
            self.stats = depth_statistics(message)
            self.depth_at = now
            self.depth_error = ""
        except (ValueError, TypeError, BufferError, OverflowError) as exc:
            self.depth_error = str(exc)
            self.depth_at = float("-inf")

    def snapshot(self, now):
        report = dict(self.lease.current(now=now))
        if report["assessment"] != "UNKNOWN":
            remaining = report["published_monotonic_s"] + report["valid_for_s"] - now
            report["valid_for_s"] = max(0., min(.5, remaining))
        depth_ok = 0 <= now-self.depth_at < .5
        rgb_ok = 0 <= now-self.rgb_at < .5
        pose_ok = 0 <= now-self.pose_at < .25
        # Independent display guard; source loss must not leave a green status.
        if not depth_ok or not rgb_ok or not pose_ok:
            report = unknown_report("waiting_for_" + "_".join(
                name for name, ok in (("depth",depth_ok),("rgb",rgb_ok),("px4_pose",pose_ok)) if not ok))
        ages = {name: now-value if math.isfinite(value) and now >= value else None
                for name,value in (("depth",self.depth_at),("rgb",self.rgb_at),("px4_pose",self.pose_at))}
        return {"schema_version":1, "mode":"OBSERVE_ONLY", "flight_commands_enabled":False,
                "server_session":self.session, "server_uptime_s":max(0.,now-self.started_at),
                "report":report, "camera_ok":depth_ok and rgb_ok, "px4_pose_received":pose_ok,
                "sensor":dict(self.stats) if depth_ok else {}, "depth_error":self.depth_error,
                "rgb_error":self.rgb_error,
                "receipt_age_s":ages, "sensor_valid_for_s":max(0.,.5-(now-self.depth_at)) if depth_ok else 0.,
                "time_basis":"Jetson ROS receipt time; not hardware exposure-to-display latency"}
