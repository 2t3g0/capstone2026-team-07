"""Motorless measurements only: never a CLEAR, climb, or calibration certificate.

Plane orientation uses PX4's already-estimated attitude, not raw IMU fusion.
Receipt pairing is deliberately labelled approximate. No position is invented
when indoor navigation is unavailable; all distances here start at the camera.
"""
from __future__ import annotations

import math
import numpy as np

from .depth_geometry import quaternion_rotation


def unknown(reason):
    return {"assessment": "UNKNOWN", "reason": reason,
            "diagnostic_only": True, "automatic_flight_authorized": False,
            "acquisition_time_verified": False, "calibration_certified": False}


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def attitude_from_diagnostic(value, *, now_s):
    """Retain the FC producer's original receipt; never renew it on ROS delivery."""
    if (not isinstance(value, dict) or value.get("mode") != "OBSERVE_ONLY"
            or value.get("flight_commands_enabled") is not False
            or value.get("source") != "physical_px4_usb_mavlink"
            or value.get("attitude_received") is not True
            or value.get("heartbeat_received") is not True):
        raise ValueError("physical_observe_attitude_unavailable")
    try:
        diagnostic = value["rx_diagnostics"]
        observed = diagnostic["observed_monotonic_s"]
        age = diagnostic["accepted_receipt_age_s"]["ATTITUDE_QUATERNION"]
        if not all(_finite_number(v) for v in (now_s, observed, age)):
            raise ValueError("invalid_fc_receipt_metadata")
        if not (0 <= age < .25 and 0 <= now_s-observed < .25):
            raise ValueError("fc_diagnostic_expired")
        received = observed-age
        if not 0 <= now_s-received < .25:
            raise ValueError("original_attitude_receipt_expired")
        rpy, rates = value["attitude_rpy_deg"], value["angular_velocity_rad_s"]
        if (not isinstance(rpy, (tuple, list)) or len(rpy) != 3
                or not isinstance(rates, (tuple, list)) or len(rates) != 3
                or not all(_finite_number(v) for v in [*rpy, *rates])):
            raise ValueError("invalid_attitude_values")
        if any(abs(v) > 180 for v in rpy):
            raise ValueError("invalid_attitude_angles")
        roll, pitch, yaw = [math.radians(v)/2 for v in rpy]
        cr, cp, cy = math.cos(roll), math.cos(pitch), math.cos(yaw)
        sr, sp, sy = math.sin(roll), math.sin(pitch), math.sin(yaw)
        q = (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
             cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy)
        rate = math.hypot(*rates)
        if not math.isfinite(rate) or rate > 20.:
            raise ValueError("invalid_motorless_angular_velocity")
        return {"quaternion": q, "rpy_deg": list(rpy), "received_s": received,
                "expires_s": received+.25,
                "sampled_rate_rad_s": rate}
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("invalid_fc_diagnostic") from exc


def decode_depth(*, data, width, height, step, encoding, bigendian):
    if (type(width) is not int or type(height) is not int or type(step) is not int
            or not 0 < width <= 1280 or not 0 < height <= 720
            or encoding not in ("16UC1", "32FC1")):
        raise ValueError("unsupported_depth_format")
    size = 2 if encoding == "16UC1" else 4
    if not width*size <= step <= width*size+4096 or len(data) != step*height:
        raise ValueError("invalid_depth_buffer")
    dtype = (">" if bigendian else "<") + ("u2" if size == 2 else "f4")
    depth = np.ndarray((height, width), dtype=dtype, buffer=bytes(data),
                       strides=(step, size)).astype(np.float64)
    return depth*.001 if size == 2 else depth


def plane_measurement(depth, *, fx, fy, cx, cy, quaternion, target="wall"):
    """Fit a dominant centre-ROI plane for a user-selected flat wall/floor.

    Robust trimming is not general obstacle segmentation. Mixed surfaces,
    inadequate coverage, non-planarity and degenerate geometry are rejected.
    Mount is the user-confirmed nominal level/forward rotation. Translation is
    intentionally NOT applied: this is not PX4-body-origin clearance.
    """
    if target not in ("wall", "floor"):
        raise ValueError("target_must_be_wall_or_floor")
    array = np.asarray(depth)
    if array.ndim != 2 or min(array.shape) < 16 or max(array.shape) > 1280:
        raise ValueError("invalid_depth_shape")
    if not all(_finite_number(v) for v in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("invalid_intrinsics")
    h, w = array.shape
    if not (0 <= cx < w and 0 <= cy < h):
        raise ValueError("invalid_principal_point")
    rotation = quaternion_rotation(quaternion)
    stride = max(1, math.ceil(math.sqrt((h*.6)*(w*.6)/4096)))
    v, u = np.mgrid[int(h*.2):int(h*.8):stride, int(w*.2):int(w*.8):stride]
    z = array[v, u]
    valid = np.isfinite(z) & (z >= .2) & (z <= 6.)
    fraction = float(valid.mean())
    if fraction < .7 or int(valid.sum()) < 128:
        return {**unknown("insufficient_valid_depth"), "valid_fraction": fraction}
    points = np.column_stack(((u[valid]-cx)*z[valid]/fx,
                              (v[valid]-cy)*z[valid]/fy, z[valid]))
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        if int(keep.sum()) < 128:
            return unknown("insufficient_plane_inliers")
        centre = points[keep].mean(axis=0)
        _, singular, axes = np.linalg.svd(points[keep]-centre, full_matrices=False)
        if singular[1]/math.sqrt(int(keep.sum())) < .03:
            return unknown("degenerate_plane_extent")
        normal = axes[-1]
        residual = np.abs((points-centre) @ normal)
        keep = residual <= min(.04, max(.01, 2.5*float(np.median(residual))))
    if float(keep.mean()) < .75:
        return unknown("mixed_or_nonplanar_surface")
    # Refit the final inliers; report conservative residual on ALL ROI returns.
    centre = points[keep].mean(axis=0)
    _, _, axes = np.linalg.svd(points[keep]-centre, full_matrices=False)
    normal = axes[-1]
    if np.dot(normal, centre) < 0:
        normal = -normal
    residual = np.abs((points-centre) @ normal)
    p90 = float(np.percentile(residual, 90))
    if p90 > .04:
        return unknown("plane_residual_too_large")
    nominal_optical_to_frd = np.array(((0., 0., 1.), (1., 0., 0.), (0., 1., 0.)))
    normal_body = nominal_optical_to_frd @ normal
    normal_ned = rotation @ normal_body
    elevation = math.degrees(math.asin(min(1., abs(float(normal_ned[2])))))
    return {**unknown("receipt_paired_measurement_not_flight_evidence"),
            "assessment": "UNVERIFIED_MEASUREMENT", "target": target,
            "valid_fraction": fraction, "sampled_points": len(points),
            "plane_inlier_fraction": float(keep.mean()), "plane_p90_residual_m": p90,
            "camera_perpendicular_distance_m": abs(float(np.dot(normal, centre))),
            "camera_optical_z_median_m": float(np.median(z[valid])),
            "normal_optical": normal.tolist(), "normal_body_frd": normal_body.tolist(),
            "normal_ned": normal_ned.tolist(),
            "normal_elevation_abs_deg": elevation,
            "expected_plane_angle_error_deg": elevation if target == "wall" else 90.-elevation,
            "position_required": False, "mount_rotation": "nominal_user_level_forward",
            "distance_origin": "D435_depth_optical_origin_not_PX4_body_origin"}


def evaluate(depth, *, intrinsics, fc_diagnostic, depth_received_s, now_s, target):
    """Diagnostic receipt gates; NEVER label the resulting transform certified."""
    try:
        if not all(_finite_number(v) for v in (now_s, depth_received_s)):
            raise ValueError("invalid_depth_receipt")
        if not 0 <= now_s-depth_received_s < .25:
            raise ValueError("depth_receipt_expired")
        attitude = attitude_from_diagnostic(fc_diagnostic, now_s=now_s)
        skew = abs(depth_received_s-attitude["received_s"])
        if skew > .075:
            raise ValueError("depth_attitude_receipt_skew")
        result = plane_measurement(depth, **intrinsics,
                                   quaternion=attitude["quaternion"], target=target)
        result.update(receipt_skew_s=skew, attitude_rpy_deg=attitude["rpy_deg"],
                      sampled_rate_rad_s=attitude["sampled_rate_rad_s"],
                      receipt_expires_monotonic_s=min(depth_received_s+.25, attitude["expires_s"]))
        return result
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
        return unknown(str(exc))
