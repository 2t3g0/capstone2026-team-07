"""Calibrated metric-Z geometry and bounded, observation-only roof memory.

No simulator ground truth, image ROI slant-distance clearance, or flight
commands are used. Roof height is supported by observed horizontal patches;
a visible wall top pixel alone is not evidence that the roof has been seen.
Passage means the *observed* roof extent is behind the vehicle and the current
descent footprint has positive below-roof ray evidence. It does not assert
that no other building, roof island, overhang, or invisible obstacle exists.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


def _finite(values) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=float)).all())


def quaternion_rotation(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(q))
    if not 0.99 <= norm <= 1.01:
        raise ValueError("quaternion must be unit length")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


@dataclass(frozen=True, slots=True)
class DepthIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rectified: bool = True

    def __post_init__(self):
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) < 2:
            raise ValueError("intrinsics require integer dimensions >= 2")
        if not _finite((self.fx, self.fy, self.cx, self.cy)) or min(self.fx, self.fy) <= 0:
            raise ValueError("intrinsics must be finite with positive focal lengths")
        if not (-0.5 <= self.cx <= self.width-0.5 and -0.5 <= self.cy <= self.height-0.5):
            raise ValueError("principal point lies outside the calibrated image")


@dataclass(frozen=True, slots=True)
class CameraExtrinsics:
    """Rotation maps optical XYZ to body FRD; translation is camera origin in body FRD."""
    rotation_optical_to_body_frd: tuple[tuple[float, float, float], ...]
    translation_body_frd_m: tuple[float, float, float]

    def __post_init__(self):
        r = np.asarray(self.rotation_optical_to_body_frd, dtype=float)
        t = np.asarray(self.translation_body_frd_m, dtype=float)
        if r.shape != (3, 3) or t.shape != (3,) or not _finite(r) or not _finite(t):
            raise ValueError("invalid camera extrinsics")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-6):
            raise ValueError("camera rotation must be right-handed orthonormal")
        object.__setattr__(self,"rotation_optical_to_body_frd",tuple(tuple(float(v) for v in row) for row in r))
        object.__setattr__(self,"translation_body_frd_m",tuple(float(v) for v in t))

    @classmethod
    def from_mount_quaternion(cls, quaternion_wxyz=(1.0, 0.0, 0.0, 0.0), translation_body_frd_m=(0.0, 0.0, 0.0)):
        # Nominal optical [right, down, forward] -> FRD [forward, right, down].
        nominal = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        rotation = quaternion_rotation(quaternion_wxyz) @ nominal
        return cls(tuple(tuple(float(v) for v in row) for row in rotation), tuple(translation_body_frd_m))


@dataclass(frozen=True, slots=True)
class VehiclePose:
    position_ned_m: tuple[float, float, float]
    quaternion_body_to_ned_wxyz: tuple[float, float, float, float]
    timestamp_s: float

    def __post_init__(self):
        if np.asarray(self.position_ned_m).shape != (3,) or not _finite(self.position_ned_m) or not math.isfinite(self.timestamp_s):
            raise ValueError("invalid vehicle pose")
        quaternion_rotation(self.quaternion_body_to_ned_wxyz)


@dataclass(frozen=True, slots=True)
class DepthGeometryConfig:
    min_depth_m: float = 0.15
    max_depth_m: float = 20.0
    tile_size_px: int = 8
    body_half_extents_frd_m: tuple[float, float, float] = (0.5, 0.5, 0.25)
    position_uncertainty_m: float = 0.10
    depth_uncertainty_m: float = 0.10
    angular_uncertainty_rad: float = math.radians(3.0)
    required_roof_gap_m: float = 1.0
    corridor_lateral_margin_m: float = 0.25
    max_forward_distance_m: float = 15.0
    min_roof_height_m: float = 0.5
    roof_normal_tolerance_rad: float = math.radians(15.0)
    roof_patch_planarity_m: float = 0.06
    min_roof_patches: int = 3
    ground_below_roof_margin_m: float = 0.30
    memory_ttl_s: float = 20.0
    frame_timeout_s: float = 0.5
    pose_timeout_s: float = 0.25
    max_pose_speed_mps: float = 15.0
    pose_jump_margin_m: float = 2.0
    cell_size_m: float = 0.25
    max_cells: int = 4096

    def __post_init__(self):
        positive = (self.min_depth_m, self.max_depth_m, self.required_roof_gap_m,
                    self.max_forward_distance_m, self.roof_patch_planarity_m,
                    self.ground_below_roof_margin_m, self.memory_ttl_s,
                    self.frame_timeout_s, self.pose_timeout_s, self.cell_size_m,
                    self.max_pose_speed_mps,self.pose_jump_margin_m)
        if not _finite(positive) or min(positive) <= 0 or self.min_depth_m >= self.max_depth_m:
            raise ValueError("geometry distances and timeouts must be finite and positive")
        nonnegative = (self.position_uncertainty_m, self.depth_uncertainty_m,
                       self.angular_uncertainty_rad, self.corridor_lateral_margin_m,
                       self.min_roof_height_m)
        if not _finite(nonnegative) or min(nonnegative) < 0:
            raise ValueError("uncertainties and margins must be finite and nonnegative")
        if not 0 <= self.angular_uncertainty_rad < math.pi/4 or not 0 < self.roof_normal_tolerance_rad < math.pi/4:
            raise ValueError("invalid geometry angular bounds")
        if not all(isinstance(v,int) for v in (self.tile_size_px,self.min_roof_patches,self.max_cells)) or self.tile_size_px < 2 or self.min_roof_patches < 1 or self.max_cells < 1:
            raise ValueError("invalid geometry sampling/memory limits")
        if np.asarray(self.body_half_extents_frd_m).shape != (3,) or not _finite(self.body_half_extents_frd_m) or min(self.body_half_extents_frd_m) <= 0:
            raise ValueError("body envelope must have three positive finite half extents")


@dataclass(frozen=True, slots=True)
class DepthGeometryObservation:
    geometry_valid: bool = False
    roof_clearance_verified: bool = False
    roof_vertical_gap_m: float | None = None
    roof_height_m: float | None = None
    obstacle_extent_valid: bool = False
    obstacle_far_north_m: float | None = None
    obstacle_far_east_m: float | None = None
    roof_passage_verified: bool = False
    reason: str = "geometry_unavailable"
    roof_observation_age_s: float | None = None
    sampled_points: int = 0
    roof_patch_count: int = 0
    footprint_free_fraction: float = 0.0
    observed_front_distance_m: float | None = None

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def _camera_transform(extrinsics, pose):
    body_rotation = quaternion_rotation(pose.quaternion_body_to_ned_wxyz)
    camera_rotation = body_rotation @ np.asarray(extrinsics.rotation_optical_to_body_frd)
    camera_origin = np.asarray(pose.position_ned_m) + body_rotation @ np.asarray(extrinsics.translation_body_frd_m)
    return camera_origin, camera_rotation, body_rotation


def deproject_pixels(u, v, metric_z_m, *, intrinsics: DepthIntrinsics,
                     extrinsics: CameraExtrinsics, pose: VehiclePose) -> np.ndarray:
    """World NED points; native D435 Z is optical-axis depth, not ray length."""
    if not intrinsics.rectified:
        raise ValueError("depth must use its rectified calibrated intrinsics")
    u, v, z = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(metric_z_m, float))
    optical = np.stack(((u-intrinsics.cx)*z/intrinsics.fx,
                        (v-intrinsics.cy)*z/intrinsics.fy, z), axis=-1)
    origin, rotation, _ = _camera_transform(extrinsics, pose)
    return optical @ rotation.T + origin


class DepthGeometryTracker:
    """Bounded observed roof/occupied/free-cell memory for one avoidance session.

    Call reset() for a new obstacle/session or a local-position reset. A timeout
    forgets evidence without ever granting passage. Extents are observed points,
    never a guessed building length. Calibration changes also discard memory.
    """
    def __init__(self, config: DepthGeometryConfig | None = None):
        self.config = config or DepthGeometryConfig()
        self.reset()

    _FREE_SLICE_COUNT = 8
    _FREE_SLAB_QUANTUM_M = 0.05

    def reset(self):
        self._roof_height = None
        self._roof_seen = None
        self._roof_uncertainty = 0.0
        # XY cell -> (lower NED bound, timestamp, upper NED bound, nominal top).
        # include the pose/range uncertainty AT observation, not the latest pose.
        self._roof_points = {}
        # A free XY cell requires every vertical slice of one fixed world-Z
        # epoch slab. Timestamps remain independent; completing one layer may
        # never refresh the other seven layers' leases.
        self._free_cells = {}  # XY cell -> OLDEST constituent certificate time
        self._free_layers = {}  # XY cell -> fixed eight timestamps or None
        self._free_slab = None  # conservative world-NED (low Z, high Z)
        self._calibration = None
        self._last_now = None
        self._last_pose = None
        self._memory_config = self.config

    def _clear_free_evidence(self):
        self._free_cells.clear()
        self._free_layers.clear()
        self._free_slab = None

    def _adopt_free_slab(self, roof_down, roof_uncertainty):
        """Outward-only epoch avoids reusing evidence for an expanded slab.

        A fit jittering INSIDE this epoch is already covered by the stronger
        certificate. Expansion discards every layer; it never relabels old
        certificates to a different world-Z layer or narrows an uncertainty.
        """
        low, high = roof_down-roof_uncertainty, roof_down+roof_uncertainty
        if self._free_slab is not None:
            old_low, old_high = self._free_slab
            if old_low <= low and high <= old_high:
                return self._free_slab
            low, high = min(low, old_low), max(high, old_high)
        quantum = self._FREE_SLAB_QUANTUM_M
        slab = (math.floor(low/quantum)*quantum, math.ceil(high/quantum)*quantum)
        self._clear_free_evidence()
        self._free_slab = slab
        return slab

    def _prune_free_evidence(self, now):
        cutoff = now-self.config.memory_ttl_s
        layers = {}
        for key, stamps in self._free_layers.items():
            fresh = tuple(stamp if stamp is not None and stamp >= cutoff else None
                          for stamp in stamps)
            if any(stamp is not None for stamp in fresh):
                layers[key] = fresh
        if len(layers) > self.config.max_cells:
            def newest(item):
                return max(stamp for stamp in item[1] if stamp is not None)
            for key, _ in sorted(layers.items(), key=newest)[:-self.config.max_cells]:
                del layers[key]
        self._free_layers = layers
        self._free_cells = {key: min(stamps) for key, stamps in layers.items()
                            if all(stamp is not None for stamp in stamps)}

    def _invalidate_occupied_free_layers(self, low, high):
        if self._free_slab is None or not self._free_layers:
            return
        boundaries = np.linspace(*self._free_slab, self._FREE_SLICE_COUNT+1)
        overlaps = (boundaries[:-1] <= high[2]) & (boundaries[1:] >= low[2])
        if not np.any(overlaps):
            return
        size = self.config.cell_size_m
        start = np.floor(low[:2]/size).astype(int)
        stop = np.floor(high[:2]/size).astype(int)
        # Include touching cell boundaries conservatively. Choose the smaller
        # bounded search, avoiding all-map scans for every sparse roof point.
        count = int(np.prod(stop-start+2))
        keys = ([(n,e) for n in range(int(start[0])-1, int(stop[0])+1)
                 for e in range(int(start[1])-1, int(stop[1])+1)]
                if count < len(self._free_layers) else list(self._free_layers))
        for key in keys:
            if key not in self._free_layers:
                continue
            if all(start[i]-1 <= key[i] <= stop[i] for i in (0,1)):
                stamps = tuple(None if overlaps[i] else stamp
                               for i, stamp in enumerate(self._free_layers[key]))
                self._free_cells.pop(key, None)
                if any(stamp is not None for stamp in stamps):
                    self._free_layers[key] = stamps
                else:
                    del self._free_layers[key]

    def _unoccupied_certificate_keys(self, keys, layer_low, layer_high, now):
        """Recent occupied intervals take priority over later free proposals.

        A complete ground-ray proof may overlap the independent uncertainty
        interval of a sparse return. Fail closed on that disagreement; never
        revive a layer invalidated earlier in this frame. Batching bounds the
        temporary comparisons even at both persistent memory limits.
        """
        if not keys or not self._roof_points:
            return keys
        cutoff = now-self.config.memory_ttl_s
        occupied = [(item[0][:2], item[2][:2]) for item in self._roof_points.values()
                    if item[1] >= cutoff and item[0][2] <= layer_high
                    and item[2][2] >= layer_low]
        if not occupied:
            return keys
        boxes = np.asarray(occupied, dtype=float)
        result = []
        size = self.config.cell_size_m
        for begin in range(0, len(keys), 128):
            batch = keys[begin:begin+128]
            low = np.asarray(batch, dtype=float)*size
            overlaps = ((low[:,None,:] <= boxes[None,:,1,:])
                        & (low[:,None,:]+size >= boxes[None,:,0,:]))
            blocked = np.any(np.all(overlaps, axis=2), axis=1)
            result.extend(key for key, stop in zip(batch, blocked) if not stop)
        return tuple(result)

    def _uncertainty(self, distance, ray_scale=1.0, origin_uncertainty=None):
        """Euclidean point error; depth uncertainty is native optical-Z error.

        ||(R_true-R_est)v|| <= 2||v||sin(theta/2) for any full-attitude
        rotation within theta. Optical-Z error scales by the optical ray norm.
        """
        c = self.config
        origin = c.position_uncertainty_m if origin_uncertainty is None else origin_uncertainty
        return origin+c.depth_uncertainty_m*ray_scale+2*distance*math.sin(c.angular_uncertainty_rad/2)

    def _remember_occupied(self, point, uncertainty, now):
        key = self._cell(*point[:2])
        low, high = point-uncertainty, point+uncertainty
        nominal_top = -float(point[2])
        old = self._roof_points.get(key)
        if old is not None:
            # Retain the union: choosing only a highest nominal point can lose
            # another point's farther edge or larger uncertainty in this cell.
            low, high = np.minimum(low, old[0]), np.maximum(high, old[2])
            nominal_top = max(nominal_top,old[3])
        self._roof_points[key] = (tuple(float(v) for v in low), now,
                                  tuple(float(v) for v in high),nominal_top)
        self._invalidate_occupied_free_layers(low, high)

    def _cell(self, north, east):
        size = self.config.cell_size_m
        return int(math.floor(north/size)), int(math.floor(east/size))

    def _prune(self, now):
        cutoff = now-self.config.memory_ttl_s
        self._roof_points = {key: item for key, item in self._roof_points.items() if item[1] >= cutoff}
        self._prune_free_evidence(now)
        for memory, timestamp in ((self._roof_points, lambda item: item[1][1]),):
            if len(memory) > self.config.max_cells:
                for key, _ in sorted(memory.items(), key=timestamp)[:-self.config.max_cells]:
                    del memory[key]
        if self._roof_seen is not None and self._roof_seen < cutoff:
            self._roof_height = None
            self._roof_seen = None
            self._roof_uncertainty = 0.0
            self._clear_free_evidence()

    def _fill_free_quad(self, quad, now, body_xy, *, rays, origin,
                        roof_down, roof_uncertainty, origin_uncertainty):
        """Combine separately proven 3-D layers, never merely free pixels.

        Each layer uses the unchanged full-cone eight-vertex certificate.
        Their union covers the complete conservative slab; evidence can come
        from different frames but every layer must remain within its own TTL.
        """
        slab = self._adopt_free_slab(roof_down, roof_uncertainty)
        boundaries = np.linspace(*slab, self._FREE_SLICE_COUNT+1)
        for layer in range(self._FREE_SLICE_COUNT):
            midpoint = float((boundaries[layer]+boundaries[layer+1])/2)
            half = float((boundaries[layer+1]-boundaries[layer])/2)
            if np.any(np.abs(rays[:,2]) < 1e-10):
                continue
            crossing = origin+(midpoint-origin[2])/rays[:,2,None]*rays
            keys = self._certify_free_quad(
                crossing[:,:2], body_xy, rays=rays, origin=origin,
                roof_down=midpoint, roof_uncertainty=half,
                origin_uncertainty=origin_uncertainty)
            keys = self._unoccupied_certificate_keys(
                keys, boundaries[layer], boundaries[layer+1], now)
            for key in keys:
                stamps = list(self._free_layers.get(key, (None,)*self._FREE_SLICE_COUNT))
                stamps[layer] = now
                self._free_layers[key] = tuple(stamps)
            # Bound persistent memory even between consecutive image runs.
            if len(self._free_layers) > self.config.max_cells:
                self._prune_free_evidence(now)
        self._prune_free_evidence(now)

    def _certify_free_quad(self, quad, body_xy, *, rays, origin,
                           roof_down, roof_uncertainty, origin_uncertainty):
        """Certify cells against every allowed ray frustum and roof height.

        A nominal ray-plane intersection alone is NOT proof near a grazing
        denominator. For each unit frustum boundary normal n and cell corner p
        at BOTH ends of the roof-height interval, require
          n.(p-origin) >= origin_error + 2*sin(theta/2)*||p-origin||.
        Since ||n_true-n|| <= 2*sin(theta/2), this places the complete cell slab
        inside every allowed camera frustum. The left side minus the norm is
        concave, so the eight vertices certify its entire convex volume.
        The nominal quad only limits enumeration; it never grants free space.
        """
        size = self.config.cell_size_m
        direction_error = 2*math.sin(self.config.angular_uncertainty_rad/2)
        rays = rays/np.linalg.norm(rays,axis=1)[:,None]
        if (np.any(rays[:,2] <= direction_error)
                or origin[2]+origin_uncertainty >= roof_down-roof_uncertainty):
            return ()  # a permitted ray can graze/upturn, or the plane can be above camera
        normals = np.cross(rays,np.roll(rays,-1,axis=0))
        lengths = np.linalg.norm(normals,axis=1)
        if np.any(lengths < 1e-10):
            return ()
        normals /= lengths[:,None]
        normals *= np.where(normals @ rays.mean(axis=0) >= 0,1.,-1.)[:,None]
        # Keep free memory local and bounded even for distant nearly horizontal rays.
        lo = np.maximum(np.min(quad, axis=0), body_xy-self.config.max_forward_distance_m)
        hi = np.minimum(np.max(quad, axis=0), body_xy+self.config.max_forward_distance_m)
        if np.any(lo >= hi):
            return ()
        start = np.floor(lo/size).astype(int)
        end = np.ceil(hi/size).astype(int)
        if np.prod(end-start) > self.config.max_cells*4:
            return ()
        north, east = np.meshgrid(np.arange(start[0], end[0]), np.arange(start[1], end[1]), indexing="ij")
        indices = np.stack((north.ravel(), east.ravel()), axis=1)
        corners = indices[:, None, :]*size + np.array([[0,0], [size,0], [size,size], [0,size]])
        vertices = np.empty((len(indices),8,3),dtype=float)
        vertices[:,:4,:2] = vertices[:,4:,:2] = corners
        vertices[:,:4,2] = roof_down-roof_uncertainty
        vertices[:,4:,2] = roof_down+roof_uncertainty
        relative = vertices-origin
        margin = origin_uncertainty+direction_error*np.linalg.norm(relative,axis=2)
        signed = relative @ normals.T
        inside = np.all(signed >= margin[:,:,None],axis=(1,2))
        return tuple(tuple(int(v) for v in key) for key in indices[inside])

    def observe(self, depth_m, *, intrinsics: DepthIntrinsics | None,
                extrinsics: CameraExtrinsics | None, pose: VehiclePose | None,
                depth_timestamp_s: float, now_s: float,
                route_direction_ned=(1.0, 0.0, 0.0),
                nominal_altitude_m: float | None = None,
                avoidance_active: bool = False) -> DepthGeometryObservation:
        c = self.config
        if c != self._memory_config:
            self.reset()  # evidence generated under another uncertainty is not reusable
        if intrinsics is None or extrinsics is None or pose is None:
            return DepthGeometryObservation(reason="calibration_or_pose_missing")
        if not intrinsics.rectified:
            return DepthGeometryObservation(reason="depth_intrinsics_not_rectified")
        if not _finite((now_s, depth_timestamp_s)) or now_s < depth_timestamp_s or now_s < pose.timestamp_s:
            return DepthGeometryObservation(reason="geometry_clock_invalid")
        if now_s-depth_timestamp_s > c.frame_timeout_s or now_s-pose.timestamp_s > c.pose_timeout_s:
            return DepthGeometryObservation(reason="geometry_observation_stale")
        if self._last_now is not None and now_s < self._last_now:
            self.reset()
            return DepthGeometryObservation(reason="geometry_clock_reset")
        self._last_now = now_s
        calibration = (intrinsics, extrinsics)
        if self._calibration is not None and calibration != self._calibration:
            self.reset()
        self._calibration = calibration
        if self._last_pose is not None:
            elapsed = pose.timestamp_s-self._last_pose.timestamp_s
            movement = float(np.linalg.norm(np.asarray(pose.position_ned_m)-self._last_pose.position_ned_m))
            if elapsed < 0 or movement > c.pose_jump_margin_m+c.max_pose_speed_mps*elapsed:
                self.reset()
                return DepthGeometryObservation(reason="pose_discontinuity_reset")
        self._last_pose = pose
        self._prune(now_s)
        try:
            depth = np.asarray(depth_m, dtype=np.float32)
            direction = np.asarray(route_direction_ned, dtype=float)
        except (TypeError, ValueError):
            return DepthGeometryObservation(reason="geometry_input_invalid")
        if depth.shape != (intrinsics.height, intrinsics.width):
            return DepthGeometryObservation(reason="depth_calibration_shape_mismatch")
        if direction.shape != (3,) or not _finite(direction) or abs(direction[2]) > 1e-6 or np.linalg.norm(direction[:2]) < 1e-6:
            return DepthGeometryObservation(reason="route_direction_invalid")
        direction = direction/np.linalg.norm(direction)
        # The requested corridor can rotate (the current service sends measured
        # heading). Stored certificates are world-NED 3-D cells, not route-frame
        # cells, so yaw jitter does not transform or invalidate their geometry.
        # Current corridor membership, observed far extent and body footprint
        # are all recomputed below for EVERY query. Unobserved cells remain
        # unknown after a turn; calibration/pose/clock/slab resets still purge.
        cross_direction = np.array([-direction[1], direction[0], 0.])
        origin, rotation, body_rotation = _camera_transform(extrinsics, pose)
        position = np.asarray(pose.position_ned_m)
        direction_error = 2*math.sin(c.angular_uncertainty_rad/2)
        origin_uncertainty = c.position_uncertainty_m+direction_error*float(np.linalg.norm(extrinsics.translation_body_frd_m))
        body_half = np.asarray(c.body_half_extents_frd_m)
        # Every envelope vertex can move by at most this radius under the same
        # full-attitude error cone. Apply in Z, XY footprint AND route extent.
        body_world_half = np.abs(body_rotation) @ body_half+direction_error*float(np.linalg.norm(body_half))
        body_down_extent = float(body_world_half[2])
        body_route_extent = float(np.sum(np.abs(direction[:2])*body_world_half[:2]))
        lateral_limit = float(np.sum(np.abs(cross_direction[:2])*body_world_half[:2]))+c.corridor_lateral_margin_m
        nominal_ok = nominal_altitude_m is not None and math.isfinite(nominal_altitude_m) and nominal_altitude_m > 0
        roof_floor = max(c.min_roof_height_m, nominal_altitude_m-c.required_roof_gap_m-body_down_extent) if nominal_ok else c.min_roof_height_m
        valid = np.isfinite(depth) & (depth >= c.min_depth_m) & (depth <= c.max_depth_m)
        if not np.any(valid):
            return DepthGeometryObservation(reason="depth_has_no_valid_returns")
        # Invalid / Gazebo no-hit pixels never supply occupied or free proof.
        # Use zero only for masked intermediate arithmetic, avoiding inf*0
        # warnings without converting a missing ray into a valid return.
        finite_depth = np.where(valid, depth, 0.)
        if (not avoidance_active and self._roof_height is None and self._roof_seen is None
                and not self._roof_points and not self._free_cells):
            # Nominal flight needs calibration/pose/data validity, not a roof
            # map. Native corridor ROI still decides the first EVADE command.
            # Keep every validation above and return unknown, never free. Any
            # remembered roof/occupied/free evidence requires the full path.
            return DepthGeometryObservation(
                geometry_valid=True,
                reason="roof_session_inactive_or_nominal_altitude_missing")
        tile = c.tile_size_px
        rows, cols = math.ceil(depth.shape[0]/tile), math.ceil(depth.shape[1]/tile)
        padded = np.full((rows*tile, cols*tile), np.inf, dtype=np.float32)
        padded[:depth.shape[0], :depth.shape[1]] = np.where(valid, depth, np.inf)
        blocks = padded.reshape(rows,tile,cols,tile).transpose(0,2,1,3).reshape(rows,cols,tile*tile)
        minimum, maximum = blocks.min(axis=2), blocks.max(axis=2)
        all_valid = np.isfinite(maximum)
        gy, gx = np.indices((rows,cols))
        # Camera rays are affine in pixel u/v. Compute scalar projections for
        # the full image, retaining the highest world-height return even when
        # it is farther in optical Z than the nearest return in the same tile.
        ux = (np.arange(depth.shape[1])-intrinsics.cx)/intrinsics.fx
        vy = (np.arange(depth.shape[0])-intrinsics.cy)/intrinsics.fy
        ray_scale = np.sqrt(1+ux[None,:]**2+vy[:,None]**2)
        dense_uncertainty = self._uncertainty(finite_depth*ray_scale,ray_scale,origin_uncertainty)
        def projection(axis):
            coefficients = axis @ rotation
            ray_coefficient = coefficients[0]*ux[None,:]+coefficients[1]*vy[:,None]+coefficients[2]
            return float((origin-position) @ axis)+finite_depth*ray_coefficient
        dense_along, dense_lateral = projection(direction), projection(cross_direction)
        relative_uncertainty = dense_uncertainty+c.position_uncertainty_m
        corridor_pixels = (valid & (dense_along+relative_uncertainty >= -body_route_extent)
                           & (dense_along-relative_uncertainty <= c.max_forward_distance_m)
                           & (np.abs(dense_lateral) <= lateral_limit+relative_uncertainty))
        closest_padded = np.full_like(padded,np.inf)
        closest_padded[:depth.shape[0],:depth.shape[1]] = np.where(corridor_pixels,depth,np.inf)
        closest_blocks = closest_padded.reshape(rows,tile,cols,tile).transpose(0,2,1,3).reshape(rows,cols,tile*tile)
        closest = closest_blocks.min(axis=2)
        index = closest_blocks.argmin(axis=2)
        u = gx*tile+index%tile
        v = gy*tile+index//tile
        selected = np.isfinite(closest)
        coefficient_z = rotation[2,0]*ux[None,:]+rotation[2,1]*vy[:,None]+rotation[2,2]
        world_height = -origin[2]-finite_depth*coefficient_z
        height_padded = np.full_like(padded,-np.inf)
        height_padded[:depth.shape[0],:depth.shape[1]] = np.where(corridor_pixels,world_height+dense_uncertainty,-np.inf)
        height_blocks = height_padded.reshape(rows,tile,cols,tile).transpose(0,2,1,3).reshape(rows,cols,tile*tile)
        high_index = height_blocks.argmax(axis=2)
        high_u, high_v = gx*tile+high_index%tile, gy*tile+high_index//tile
        # Farthest occupied extent is independent of highest/nearest surface.
        # Retain its worst route-forward bound even for a thin, lower obstacle.
        far_padded = np.full_like(padded,-np.inf)
        far_mask = corridor_pixels & (world_height+dense_uncertainty >= roof_floor)
        far_padded[:depth.shape[0],:depth.shape[1]] = np.where(far_mask,dense_along+dense_uncertainty,-np.inf)
        far_blocks = far_padded.reshape(rows,tile,cols,tile).transpose(0,2,1,3).reshape(rows,cols,tile*tile)
        far_index = far_blocks.argmax(axis=2)
        far_selected = np.isfinite(far_blocks.max(axis=2))
        far_u,far_v = gx*tile+far_index%tile,gy*tile+far_index//tile
        all_u = np.r_[u[selected],high_u[selected],far_u[far_selected]]
        all_v = np.r_[v[selected],high_v[selected],far_v[far_selected]]
        points = deproject_pixels(all_u,all_v,depth[all_v,all_u],intrinsics=intrinsics,extrinsics=extrinsics,pose=pose)
        point_uncertainty = dense_uncertainty[all_v,all_u]
        relative = points-position
        along, lateral = relative @ direction, relative @ cross_direction
        relative_error = point_uncertainty+c.position_uncertainty_m
        in_corridor = ((along+relative_error >= -body_route_extent)
                       & (along-relative_error <= c.max_forward_distance_m)
                       & (np.abs(lateral) <= lateral_limit+relative_error))
        # Near min pooling retains a one-pixel close obstacle inside a tile.
        # These sparse points alone never grant free-space or roof clearance.
        relevant = points[in_corridor]
        relevant_uncertainty = point_uncertainty[in_corridor]
        front_candidates = along[in_corridor & (along >= 0)]
        front_distance = float(front_candidates.min()) if len(front_candidates) else None
        if avoidance_active and nominal_ok:
            for point, error in zip(relevant,relevant_uncertainty):
                if -point[2]+error >= roof_floor:
                    self._remember_occupied(point,float(error),now_s)
        # Tile-corner normals distinguish actually seen horizontal roofs from walls.
        u0, v0 = gx*tile, gy*tile
        u1, v1 = np.minimum(u0+tile-1,depth.shape[1]-1), np.minimum(v0+tile-1,depth.shape[0]-1)
        corners_uv = [(u0,v0), (u1,v0), (u1,v1), (u0,v1)]
        corners = np.stack([deproject_pixels(uu,vv,np.where(valid[vv,uu],depth[vv,uu],0),
                             intrinsics=intrinsics,extrinsics=extrinsics,pose=pose) for uu,vv in corners_uv], axis=2)
        corner_uncertainty = np.stack([dense_uncertainty[vv,uu] for uu,vv in corners_uv],axis=2)
        normal = np.cross(corners[:,:,1]-corners[:,:,0],corners[:,:,3]-corners[:,:,0])
        norm = np.linalg.norm(normal,axis=2)
        normal_unit = normal/np.maximum(norm[:,:,None],1e-12)
        planarity = np.abs(np.sum((corners[:,:,2]-corners[:,:,0])*normal_unit,axis=2))
        centers = corners.mean(axis=2)
        center_along = (centers-position) @ direction
        center_lateral = (centers-position) @ cross_direction
        normal_tolerance = max(0.,c.roof_normal_tolerance_rad-c.angular_uncertainty_rad)
        candidate = (all_valid & (c.angular_uncertainty_rad < c.roof_normal_tolerance_rad)
                     & (norm > 1e-8) & (np.abs(normal_unit[:,:,2]) >= math.cos(normal_tolerance))
                     & (planarity <= c.roof_patch_planarity_m) & (-centers[:,:,2] >= roof_floor)
                     & (-centers[:,:,2] < -origin[2]-0.02) & (center_along >= -body_route_extent)
                     & (center_along <= c.max_forward_distance_m) & (np.abs(center_lateral) <= lateral_limit))
        patches = int(candidate.sum())
        if avoidance_active and nominal_ok and patches >= c.min_roof_patches:
            observed = corners[candidate].reshape(-1,3)
            observed_uncertainty = corner_uncertainty[candidate].reshape(-1)
            height = float(np.max(-observed[:,2]))
            uncertainty = float(np.max(observed_uncertainty))
            if self._roof_height is None or height+uncertainty > self._roof_height+self._roof_uncertainty:
                self._roof_height, self._roof_uncertainty = height, uncertainty
            self._roof_seen = now_s
            for point, error in zip(observed,observed_uncertainty):
                self._remember_occupied(point,float(error),now_s)
        roof_known = self._roof_height is not None and self._roof_seen is not None
        if roof_known and self._roof_points:
            # A distant sparse roof/edge return can have a larger uncertainty
            # than the planar patches used for recognition. Its upper bound
            # must also constrain height and invalidate narrower free evidence.
            occupied_upper_height = max(-item[0][2] for item in self._roof_points.values())
            expanded = max(self._roof_uncertainty,occupied_upper_height-self._roof_height)
            if expanded > self._roof_uncertainty:
                self._roof_uncertainty = expanded
        if roof_known:
            # Require every pixel in a candidate tile to terminate below roof.
            # The affine ray-z coefficient extrema occur at the tile corners;
            # checking both min/max optical Z bounds all interior samples.
            slab = self._adopt_free_slab(-self._roof_height, self._roof_uncertainty)
            # Ground endpoints must support the entire OUTWARD epoch slab,
            # not only the latest slightly narrower roof fit.
            roof_down = (slab[0]+slab[1])/2
            free_roof_uncertainty = (slab[1]-slab[0])/2
            rays = np.stack([np.stack(((uu-intrinsics.cx)/intrinsics.fx,(vv-intrinsics.cy)/intrinsics.fy,np.ones_like(uu)),axis=2) @ rotation.T
                             for uu,vv in corners_uv],axis=2)
            safe_minimum, safe_maximum = np.where(all_valid,minimum,0), np.where(all_valid,maximum,0)
            endpoint_min_z = np.minimum(rays[:,:,:,2]*safe_minimum[:,:,None],rays[:,:,:,2]*safe_maximum[:,:,None]).min(axis=2)+origin[2]
            max_ray_scale = np.linalg.norm(rays,axis=3).max(axis=2)
            # The terminating GROUND return has its own range/pose error;
            # using only the previous roof observation's error is insufficient.
            endpoint_error = self._uncertainty(safe_maximum*max_ray_scale,max_ray_scale,origin_uncertainty)
            ray_unit_z = rays[:,:,:,2]/np.linalg.norm(rays,axis=3)
            free_tiles = (all_valid & (ray_unit_z.min(axis=2) > direction_error)
                          & (origin[2]+origin_uncertainty < roof_down-free_roof_uncertainty))
            free_tiles &= endpoint_min_z-endpoint_error > roof_down+c.ground_below_roof_margin_m+free_roof_uncertainty
            # Merge identical horizontal safe runs across rows into rectangles.
            active_runs = {}
            rectangles = []
            for row in range(rows+1):
                mask = free_tiles[row] if row < rows else np.zeros(cols,dtype=bool)
                boundaries = np.diff(np.r_[False,mask,False].astype(np.int8))
                runs = set(zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)))
                for run in list(active_runs):
                    if run not in runs:
                        rectangles.append((*run,active_runs.pop(run),row))
                for run in runs:
                    active_runs.setdefault(run,row)
            for x_start,x_end,y_start,y_end in rectangles:
                uv = np.array([[x_start*tile,y_start*tile], [x_end*tile-1,y_start*tile],
                               [x_end*tile-1,y_end*tile-1], [x_start*tile,y_end*tile-1]],dtype=float)
                ray = np.column_stack(((uv[:,0]-intrinsics.cx)/intrinsics.fx,(uv[:,1]-intrinsics.cy)/intrinsics.fy,np.ones(4))) @ rotation.T
                if np.any(ray[:,2]/np.linalg.norm(ray,axis=1) <= direction_error):
                    continue
                crossing = origin+(roof_down-origin[2])/ray[:,2,None]*ray
                self._fill_free_quad(crossing[:,:2],now_s,position[:2],rays=ray,
                                     origin=origin,roof_down=-self._roof_height,
                                     roof_uncertainty=self._roof_uncertainty,
                                     origin_uncertainty=origin_uncertainty)
        self._prune(now_s)
        occupied_low = np.asarray([item[0] for item in self._roof_points.values()],dtype=float).reshape(-1,3)
        occupied_high = np.asarray([item[2] for item in self._roof_points.values()],dtype=float).reshape(-1,3)
        occupied_nominal_top = np.asarray([item[3] for item in self._roof_points.values()],dtype=float)
        far_point = None
        if len(occupied_low):
            centers = (occupied_low+occupied_high)/2
            half = (occupied_high-occupied_low)/2
            corridor_mask = np.abs((centers-position) @ cross_direction) <= lateral_limit+half @ np.abs(cross_direction)+c.position_uncertainty_m
            occupied_low, occupied_high = occupied_low[corridor_mask], occupied_high[corridor_mask]
            occupied_nominal_top = occupied_nominal_top[corridor_mask]
            if len(occupied_low):
                # Return the route-forward extreme of each stored interval, NOT
                # a nominal observed point. Caller can add its existing +5m.
                far_bounds = np.where(direction[None,:] >= 0,occupied_high,occupied_low)
                far_point = far_bounds[np.argmax(far_bounds @ direction)]
        roof_known = self._roof_height is not None and self._roof_seen is not None
        gap = (-position[2]-body_down_extent-self._roof_height-self._roof_uncertainty-c.position_uncertainty_m) if roof_known else None
        # Occupied interval uppers already expanded the gap/slab bound above,
        # including differences BELOW the planarity tolerance. This additional
        # nominal-height gate prevents treating a newly seen chimney/wall edge
        # as a recognized roof plane merely by inflating its uncertainty.
        higher_obstacle = bool(roof_known and len(occupied_low) and np.max(occupied_nominal_top) > self._roof_height+c.roof_patch_planarity_m)
        clearance = bool(roof_known and not higher_obstacle and gap > c.required_roof_gap_m)
        footprint = body_world_half[:2]+c.position_uncertainty_m
        low = np.floor((position[:2]-footprint)/c.cell_size_m).astype(int)
        high = np.floor((position[:2]+footprint)/c.cell_size_m).astype(int)
        footprint_cells = [(n,e) for n in range(low[0],high[0]+1) for e in range(low[1],high[1]+1)]
        free_count = sum(key in self._free_cells for key in footprint_cells)
        free_fraction = free_count/len(footprint_cells)
        observed_extent_behind = far_point is not None and float((position-far_point) @ direction) > body_route_extent+c.position_uncertainty_m
        # Once physically past the observed roof, descending below its height
        # must not revoke positive free-footprint evidence. Ground/altitude
        # limits are separate controller guards, not roof-height clearance.
        passage = bool(roof_known and not higher_obstacle and observed_extent_behind and free_fraction == 1.0)
        if not nominal_ok or not avoidance_active:
            reason = "roof_session_inactive_or_nominal_altitude_missing"
        elif not roof_known:
            reason = "roof_top_not_observed_or_memory_expired"
        elif higher_obstacle:
            reason = "higher_obstacle_surface_unresolved"
        elif not clearance:
            reason = "observed_roof_gap_insufficient"
        elif passage:
            reason = "observed_extent_passed_and_descent_footprint_observed_free"
        else:
            reason = "roof_gap_verified_but_passage_unverified"
        return DepthGeometryObservation(
            geometry_valid=True,roof_clearance_verified=clearance,
            roof_vertical_gap_m=float(gap) if gap is not None else None,
            roof_height_m=float(self._roof_height+self._roof_uncertainty) if roof_known else None,
            obstacle_extent_valid=far_point is not None,
            obstacle_far_north_m=float(far_point[0]) if far_point is not None else None,
            obstacle_far_east_m=float(far_point[1]) if far_point is not None else None,
            roof_passage_verified=passage,reason=reason,
            roof_observation_age_s=now_s-self._roof_seen if roof_known else None,
            sampled_points=len(points),roof_patch_count=patches,
            footprint_free_fraction=free_fraction,observed_front_distance_m=front_distance)
