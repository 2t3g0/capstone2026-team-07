"""Conditional time/motion budget calculation; NOT acquisition authentication.

Receipt timestamps and a latest angular-rate sample cannot prove exposure-time
alignment or a rate bound over an interval. Current ROS bridge inputs therefore
have no strict-mode proof provider. The pure calculation below is for testing
an explicitly supplied *continuous* envelope after an external provider has
validated its source/clock mapping. It never authorizes physical flight.
"""
from dataclasses import dataclass
import math

LEGACY_RECEIPT = "legacy_receipt"
STRICT_ACQUISITION = "strict_acquisition"
MAX_PAIR_SKEW_S = .075
MAX_ACQUISITION_AGE_S = .250
ANGULAR_BUDGET_RAD = math.radians(3.)
MISSING_PROOF = "depth_pose_motion_unknown_acquisition_interval_and_rate_envelope_unavailable"


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_mode(value):
    if value not in (LEGACY_RECEIPT, STRICT_ACQUISITION):
        raise ValueError("unsupported depth_pose_motion_mode")
    return value


def bridge_unavailable_reason(mode):
    """No message/header/remote response may manufacture the missing proof."""
    checked_mode(mode)
    return MISSING_PROOF if mode == STRICT_ACQUISITION else ""


@dataclass(frozen=True)
class AcquisitionInterval:
    earliest_s: float
    latest_s: float
    clock_id: str
    source_id: str
    epoch_id: str
    evidence_id: str
    basis: str = "validated_exposure_interval"


@dataclass(frozen=True)
class ContinuousRateEnvelope:
    start_s: float
    end_s: float
    upper_rad_s: float
    clock_id: str
    source_id: str
    epoch_id: str
    evidence_id: str
    expires_s: float
    basis: str = "validated_continuous_bound"


@dataclass(frozen=True)
class MotionAssessment:
    status: str
    reason: str
    separation_upper_s: float | None = None
    angular_error_upper_rad: float | None = None
    # Even a numerically valid input is conditional on an external proof source.
    physical_flight_authorized: bool = False
    evidence_authenticated: bool = False


def assess_motion_bound(*, now_s, clock_id, epoch_id, depth_source_id,
                        pose_source_id, depth=None, pose=None, envelope=None,
                        base_angular_error_rad=None):
    """Worst-case attitude displacement between two mapped exposure intervals.

    For a true continuous |omega(t)|<=R envelope, rotation path length is at
    most R*dt. Interval endpoints must include exposure and clock-mapping
    uncertainty, not callback receipt time. Base error includes extrinsic and
    estimator angular error. No unverified base-error default of zero is used.
    Source IDs/epochs are caller-validated expectations, not self-claimed trust.
    """
    def unknown(reason):
        return MotionAssessment("UNKNOWN", reason)

    expected = (clock_id, epoch_id, depth_source_id, pose_source_id)
    if (not finite(now_s) or now_s < 0 or
            any(not isinstance(v, str) or not v.strip() for v in expected)):
        return unknown("expected_clock_source_epoch_unavailable")
    if not isinstance(depth, AcquisitionInterval) or not isinstance(pose, AcquisitionInterval):
        return unknown("acquisition_interval_unavailable")
    for sample, source in ((depth, depth_source_id), (pose, pose_source_id)):
        if (sample.basis != "validated_exposure_interval" or sample.clock_id != clock_id
                or sample.epoch_id != epoch_id or sample.source_id != source
                or not isinstance(sample.evidence_id, str) or not sample.evidence_id.strip()):
            return unknown("acquisition_source_clock_epoch_or_basis_invalid")
        if (not all(finite(v) for v in (sample.earliest_s, sample.latest_s))
                or not 0 <= sample.earliest_s <= sample.latest_s <= now_s):
            return unknown("acquisition_interval_invalid_or_future")
        if now_s-sample.earliest_s > MAX_ACQUISITION_AGE_S:
            return unknown("acquisition_age_unavailable_or_expired")
    if not isinstance(envelope, ContinuousRateEnvelope):
        return unknown("continuous_rate_envelope_unavailable")
    if (envelope.basis != "validated_continuous_bound" or envelope.clock_id != clock_id
            or envelope.epoch_id != epoch_id or envelope.source_id != pose_source_id
            or not isinstance(envelope.evidence_id, str) or not envelope.evidence_id.strip()):
        return unknown("rate_source_clock_epoch_or_basis_invalid")
    if (not all(finite(v) for v in (envelope.start_s, envelope.end_s,
                                   envelope.upper_rad_s, envelope.expires_s))
            or not 0 <= envelope.start_s <= envelope.end_s <= now_s
            or envelope.upper_rad_s < 0 or envelope.expires_s <= now_s):
        return unknown("continuous_rate_envelope_invalid_or_expired")
    if (envelope.start_s > min(depth.earliest_s, pose.earliest_s)
            or envelope.end_s < max(depth.latest_s, pose.latest_s)):
        return unknown("continuous_rate_envelope_does_not_cover_acquisitions")
    if not finite(base_angular_error_rad) or base_angular_error_rad < 0:
        return unknown("base_angular_error_bound_unavailable")
    separation = max(abs(depth.earliest_s-pose.latest_s),
                     abs(depth.latest_s-pose.earliest_s))
    if separation > MAX_PAIR_SKEW_S:
        return MotionAssessment("BLOCKED", "acquisition_skew_exceeds_75ms", separation)
    bound = base_angular_error_rad + envelope.upper_rad_s*separation
    if not math.isfinite(bound):
        return unknown("angular_bound_nonfinite")
    if bound > ANGULAR_BUDGET_RAD:
        return MotionAssessment("BLOCKED", "angular_error_exceeds_3deg", separation, bound)
    return MotionAssessment("CONDITIONAL_WITHIN_BUDGET",
        "external_acquisition_and_continuous_bound_proof_still_required", separation, bound)


def receipt_diagnostic(*, depth_received_at, attitude_received_at, sampled_rate_rad_s):
    """Diagnostic arithmetic ONLY. Never feeds assess_motion_bound as proof."""
    valid = (all(finite(v) and v >= 0 for v in
                 (depth_received_at, attitude_received_at, sampled_rate_rad_s)))
    skew = abs(depth_received_at-attitude_received_at) if valid else None
    estimate = sampled_rate_rad_s*skew if valid else None
    if estimate is not None and not math.isfinite(estimate):
        estimate = None
    degrees = math.degrees(estimate) if estimate is not None else None
    if degrees is not None and not math.isfinite(degrees):
        degrees = None
    return dict(status="UNKNOWN", acquisition_age_proven=False,
        guaranteed_interval_rate_bound=False, physical_flight_authorized=False,
        receipt_skew_s=skew, sampled_rate_rad_s=sampled_rate_rad_s if valid else None,
        sampled_rate_times_receipt_skew_deg=degrees,
        meaning="receipt-time arithmetic, not exposure alignment or an error upper bound")
