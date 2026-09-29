"""Read-only display data; native sensor statistics are NOT flight decisions."""
from __future__ import annotations
import math
import time
import uuid
import numpy as np
from .sensor_observer import ObserverReportLease, unknown_report


def classify_depth_perception(*, near_distance_m, obstacle_fraction,
                              valid_fraction, upper_near_distance_m,
                              upper_obstacle_fraction, upper_valid_fraction,
                              lower_valid_fraction, trigger_distance_m=3.0,
                              release_distance_m=5.5,
                              min_obstacle_fraction=0.02,
                              min_valid_fraction=0.25):
    """Classify fresh upper/center/lower depth ROIs without aircraft pose.

    This is a camera observation for the operator display, never a movement
    decision.  Thresholds are supplied by the fixed field-demo policy below.
    """
    center_values = (near_distance_m, obstacle_fraction, valid_fraction,
                     trigger_distance_m, release_distance_m,
                     min_obstacle_fraction, min_valid_fraction)
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in center_values):
        return "UNKNOWN", "depth_center_roi_nonfinite"
    if not (near_distance_m >= 0.0
            and 0.0 < trigger_distance_m < release_distance_m
            and 0.0 <= obstacle_fraction <= 1.0
            and 0.0 <= valid_fraction <= 1.0
            and 0.0 <= min_obstacle_fraction <= 1.0
            and 0.0 <= min_valid_fraction <= 1.0):
        return "UNKNOWN", "depth_center_roi_invalid_fraction"
    if valid_fraction < min_valid_fraction:
        return "UNKNOWN", "depth_center_roi_quality_insufficient"
    obstacle = (near_distance_m <= trigger_distance_m
                and obstacle_fraction >= min_obstacle_fraction)
    if not obstacle:
        return "CLEAR", "depth_center_roi_clear"

    # A valid center hit is enough to stop.  The upper ROI may be mostly sky
    # outdoors, where active stereo cannot provide depth.  Missing upper depth
    # must therefore prevent CLIMB_REQUIRED, not erase the confirmed obstacle.
    upper_values = (upper_near_distance_m, upper_obstacle_fraction,
                    upper_valid_fraction)
    upper_valid = (
        all(type(value) in (int, float) and math.isfinite(value)
            for value in upper_values)
        and upper_near_distance_m >= 0.0
        and 0.0 <= upper_obstacle_fraction <= 1.0
        and 0.0 <= upper_valid_fraction <= 1.0
        and upper_valid_fraction >= min_valid_fraction
    )
    if not upper_valid:
        return "BLOCKED", "front_obstacle_upper_roi_unavailable"
    upper_clear = (upper_near_distance_m >= release_distance_m
                   and upper_obstacle_fraction < min_obstacle_fraction)
    if upper_clear:
        return "CLIMB_REQUIRED", "front_obstacle_upper_roi_clear_climb_candidate"
    return "BLOCKED", "front_obstacle_upper_roi_not_clear"


def depth_statistics(message):
    from jolgwa_uav.native_depth import native_corridor_stats
    from jolgwa_uav.small_obstacle_demo_guard import demo_policy_config
    if message.encoding != "16UC1":
        raise ValueError("requires_raw_16UC1")
    h, w, step = int(message.height), int(message.width), int(message.step)
    if not (0 < h <= 1080 and 0 < w <= 1920 and w*2 <= step <= w*2+4096):
        raise ValueError("invalid_depth_dimensions")
    if len(message.data) != h*step:
        raise ValueError("invalid_depth_length")
    depth = np.ndarray((h,w), dtype=">u2" if message.is_bigendian else "<u2",
                       buffer=bytes(message.data), strides=(step,2)).astype(np.float32)*.001
    config = demo_policy_config()
    import os
    from jolgwa_uav.front_climb_profile import PROFILE, configure
    if os.environ.get("JOLGWA_SCENARIO_PROFILE") == PROFILE:
        config = configure(config)
    valid = np.isfinite(depth) & (depth >= config.min_depth_m) & (depth <= config.max_depth_m)
    stats = native_corridor_stats(depth, valid, config, config.trigger_distance_m)
    if stats is None:
        raise ValueError("native_depth_library_unavailable")
    finite = lambda x: float(x) if math.isfinite(x) else None
    return {"front_near_m": finite(stats.center.near_distance_m),
            "front_median_m": finite(stats.center.median_distance_m),
            "upper_roi_near_m": finite(stats.upper.near_distance_m),
            "front_obstacle_fraction": float(stats.center.obstacle_fraction),
            "center_valid_fraction": float(stats.center.valid_fraction),
            "upper_obstacle_fraction": float(stats.upper.obstacle_fraction),
            "upper_valid_fraction": float(stats.upper.valid_fraction),
            "lower_obstacle_fraction": float(stats.lower.obstacle_fraction),
            "lower_valid_fraction": float(stats.lower.valid_fraction),
            "valid_fraction": float(valid.mean()), "width": w, "height": h,
            "perception_thresholds": {
                "trigger_distance_m": float(config.trigger_distance_m),
                "release_distance_m": float(config.release_distance_m),
                "min_obstacle_fraction": float(config.min_obstacle_fraction),
                "min_valid_fraction": float(config.min_valid_fraction),
            },
            "scope": "native upper/center/lower depth ROI observation only; independent of PX4 pose"}


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
        if depth_ok:
            thresholds = self.stats.get("perception_thresholds", {})
            assessment, reason = classify_depth_perception(
                near_distance_m=self.stats.get("front_near_m"),
                obstacle_fraction=self.stats.get("front_obstacle_fraction"),
                valid_fraction=self.stats.get("center_valid_fraction"),
                upper_near_distance_m=self.stats.get("upper_roi_near_m"),
                upper_obstacle_fraction=self.stats.get("upper_obstacle_fraction"),
                upper_valid_fraction=self.stats.get("upper_valid_fraction"),
                lower_valid_fraction=self.stats.get("lower_valid_fraction"),
                trigger_distance_m=thresholds.get("trigger_distance_m"),
                release_distance_m=thresholds.get("release_distance_m"),
                min_obstacle_fraction=thresholds.get("min_obstacle_fraction"),
                min_valid_fraction=thresholds.get("min_valid_fraction"),
            )
            perception_valid_for = max(0., min(.5, .5-(now-self.depth_at)))
        else:
            thresholds = self.stats.get("perception_thresholds", {})
            assessment, reason, perception_valid_for = "UNKNOWN", "depth_missing_or_expired", 0.
        perception = {
            "schema_version": 1,
            "assessment": assessment,
            "reason": reason,
            "sequence": self.depth_stamp if self.depth_stamp > 0 else 0,
            "source_timestamp_ns": self.depth_stamp if self.depth_stamp > 0 else 0,
            "valid_for_s": perception_valid_for,
            "source": "D435 native upper/center/lower depth ROIs; no PX4 pose used",
            "front_near_m": self.stats.get("front_near_m") if depth_ok else None,
            "front_obstacle_fraction": self.stats.get("front_obstacle_fraction") if depth_ok else None,
            "center_valid_fraction": self.stats.get("center_valid_fraction") if depth_ok else None,
            "upper_near_distance_m": self.stats.get("upper_roi_near_m") if depth_ok else None,
            "upper_obstacle_fraction": self.stats.get("upper_obstacle_fraction") if depth_ok else None,
            "upper_valid_fraction": self.stats.get("upper_valid_fraction") if depth_ok else None,
            "lower_valid_fraction": self.stats.get("lower_valid_fraction") if depth_ok else None,
            "trigger_distance_m": thresholds.get("trigger_distance_m", 3.0),
            "release_distance_m": thresholds.get("release_distance_m", 5.5),
            "min_obstacle_fraction": thresholds.get("min_obstacle_fraction", 0.02),
            "min_valid_fraction": thresholds.get("min_valid_fraction", 0.25),
            "climb_path_verified": False,
            "next_observation_step_m": None,
        }
        # Independent display guard; source loss must not leave a green status.
        if not depth_ok or not rgb_ok or not pose_ok:
            report = unknown_report("waiting_for_" + "_".join(
                name for name, ok in (("depth",depth_ok),("rgb",rgb_ok),("px4_pose",pose_ok)) if not ok))
        ages = {name: now-value if math.isfinite(value) and now >= value else None
                for name,value in (("depth",self.depth_at),("rgb",self.rgb_at),("px4_pose",self.pose_at))}
        return {"schema_version":1, "mode":"OBSERVE_ONLY", "flight_commands_enabled":False,
                "server_session":self.session, "server_uptime_s":max(0.,now-self.started_at),
                "perception":perception, "report":report,
                "build_id":"low-speed-runtimeio1-20260929",
                "scenario_profile":"FRONT_DETECT_2M_PASS_3M_V1",
                "depth_ok":depth_ok, "rgb_ok":rgb_ok,
                "camera_ok":depth_ok and rgb_ok, "px4_pose_received":pose_ok,
                "sensor":dict(self.stats) if depth_ok else {}, "depth_error":self.depth_error,
                "rgb_error":self.rgb_error,
                "receipt_age_s":ages, "sensor_valid_for_s":max(0.,.5-(now-self.depth_at)) if depth_ok else 0.,
                "time_basis":"Jetson ROS receipt time; not hardware exposure-to-display latency"}
