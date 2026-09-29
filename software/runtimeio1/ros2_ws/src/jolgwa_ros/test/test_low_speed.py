from unittest.mock import patch
import math
from types import SimpleNamespace

import pytest

from jolgwa_ros.low_speed import (
    ALTITUDE_ALIGNMENT_HOME_CORRECTION_PENDING,
    ALTITUDE_ALIGNMENT_READY, ALTITUDE_ALIGNMENT_UNSTABLE,
    FLIGHT_OUTPUT_PAIR_SKEW_NS,
    AltitudeReferenceSynchronizer,
    FORWARD_TEST_1M, LOW_SPEED_1M_V1, LOW_SPEED_2M_V1, ROUTE,
    aligned_home_z_ned, altitude_reference_error_m,
    altitude_reference_sample_error, envelope_allows,
    flight_output_confirmation_ready, flight_output_pair_skew_valid,
    force_low_speed_route,
    forward_endpoint_ned,
    forward_test_target_within_camera_bypass, geo_is_fresh,
    low_speed_envelope_profiles_match, low_speed_readiness_error,
    low_speed_pose_coherence_reason,
    obstacle_guard_required_for_mission, preview_valid,
    px4_land_speed_is_valid, px4_offboard_loss_action_is_land,
    setpoint_contract_matches, takeoff_handshake_action,
    validate_low_speed_plan,
)
from jolgwa_ros.operator_protocol import ProtocolError, parse_browser_message


def test_flight_output_pair_skew_accepts_150ms_boundary_only():
    assert FLIGHT_OUTPUT_PAIR_SKEW_NS == 150_000_000
    assert flight_output_pair_skew_valid(1_000_000_000, 1_150_000_000)
    assert not flight_output_pair_skew_valid(1_000_000_000, 1_150_000_001)
    assert not flight_output_pair_skew_valid(-1, 1_000_000_000)


def _feed_altitude_alignment(sync, candidates, *, start_boot_ms=1000,
                             interval_ms=200, start_ns=1_000_000_000):
    observation = None
    for index, candidate in enumerate(candidates):
        boot_ms = (start_boot_ms+index*interval_ms) % 2**32
        received_ns = start_ns+index*interval_ms*1_000_000
        local_z = -1.0
        sync.add_local(
            time_boot_ms=boot_ms, z_ned_m=local_z,
            received_ns=received_ns)
        observation = sync.add_global(
            time_boot_ms=boot_ms,
            fc_altitude_home_relative_m=candidate-local_z,
            received_ns=received_ns, now_ns=received_ns)
    return observation


def test_altitude_alignment_requires_two_seconds_ten_samples_and_one_metre_span():
    sync = AltitudeReferenceSynchronizer()
    aligning = _feed_altitude_alignment(sync, [-0.97]*10)
    assert not aligning.stable
    assert aligning.window_duration_ms == 1800

    ready = _feed_altitude_alignment(
        sync, [-0.97], start_boot_ms=3000, start_ns=3_000_000_000)
    assert ready.state == ALTITUDE_ALIGNMENT_READY
    assert ready.stable
    assert ready.sample_count >= 10
    assert ready.window_duration_ms >= 2000
    assert ready.stable_aligned_home_z_ned_m == pytest.approx(-0.97)

    boundary_sync = AltitudeReferenceSynchronizer()
    boundary = _feed_altitude_alignment(
        boundary_sync, [-0.97+0.1*index for index in range(11)])
    assert boundary.state == ALTITUDE_ALIGNMENT_READY
    assert boundary.stable
    assert boundary.candidate_span_m == pytest.approx(1.00)

    unstable_sync = AltitudeReferenceSynchronizer()
    unstable = _feed_altitude_alignment(
        unstable_sync, [-0.97+0.1001*index for index in range(11)])
    assert unstable.state == ALTITUDE_ALIGNMENT_UNSTABLE
    assert not unstable.stable
    assert unstable.candidate_span_m == pytest.approx(1.001)


def test_altitude_alignment_ready_does_not_chatter_on_fc_millisecond_jitter():
    sync = AltitudeReferenceSynchronizer()
    # Captured GLOBAL_POSITION_INT timestamps from the physical PX4.  The
    # 99/100/101/110 ms quantisation made the old retained-endpoint check
    # alternate 2000 -> 1999/1901 ms and READY -> ALIGNING every few samples.
    boot_times_ms = [
        592110, 592220, 592319, 592419, 592518, 592617, 592717,
        592816, 592916, 593015, 593115, 593215, 593314, 593414,
        593513, 593612, 593712, 593811, 593922, 594020, 594120,
        594219, 594319, 594418, 594518, 594618,
    ]
    states_after_ready = []
    ready_seen = False
    for boot_ms in boot_times_ms:
        received_ns = 1_000_000_000+(boot_ms-boot_times_ms[0])*1_000_000
        sync.add_local(
            time_boot_ms=boot_ms-20, z_ned_m=9.49,
            received_ns=received_ns)
        observation = sync.add_global(
            time_boot_ms=boot_ms,
            fc_altitude_home_relative_m=0.56,
            received_ns=received_ns, now_ns=received_ns)
        if observation.state == ALTITUDE_ALIGNMENT_READY:
            ready_seen = True
        if ready_seen:
            states_after_ready.append(observation.state)

    assert ready_seen
    assert states_after_ready
    assert set(states_after_ready) == {ALTITUDE_ALIGNMENT_READY}
    assert observation.window_duration_ms == 2000
    assert observation.candidate_span_m == pytest.approx(0.0)


def test_altitude_alignment_span_excludes_outlier_older_than_exact_window():
    sync = AltitudeReferenceSynchronizer()
    observations = []
    for index in range(23):
        boot_ms = 1000+index*100
        received_ns = 1_000_000_000+index*100_000_000
        candidate = -0.48 if index == 0 else -0.97
        sync.add_local(
            time_boot_ms=boot_ms, z_ned_m=-1.0,
            received_ns=received_ns)
        observations.append(sync.add_global(
            time_boot_ms=boot_ms,
            fc_altitude_home_relative_m=candidate+1.0,
            received_ns=received_ns, now_ns=received_ns))

    # At 2.2 s the first outlier is outside the exact recent two-second
    # candidate window.  It may prove time coverage but must not affect span.
    assert observations[-1].state == ALTITUDE_ALIGNMENT_READY
    assert observations[-1].window_duration_ms == 2000
    assert observations[-1].candidate_span_m == pytest.approx(0.0)


def test_altitude_alignment_pairs_nearest_fc_timestamp_not_ros_callback_order():
    sync = AltitudeReferenceSynchronizer()
    sync.add_local(time_boot_ms=1000, z_ned_m=-1.2,
                   received_ns=1_000_000_000)
    sync.add_local(time_boot_ms=1100, z_ned_m=-1.0,
                   received_ns=1_100_000_000)
    observation = sync.add_global(
        time_boot_ms=1080, fc_altitude_home_relative_m=0.03,
        received_ns=1_120_000_000, now_ns=1_120_000_000)
    assert observation.local_time_boot_ms == 1100
    assert observation.source_skew_ms == 20
    assert observation.candidate_aligned_home_z_ned_m == pytest.approx(-0.97)


def test_altitude_alignment_accepts_boot_clock_wrap_and_resets_on_duplicate():
    sync = AltitudeReferenceSynchronizer()
    epoch = sync.transport_epoch
    sync.add_local(time_boot_ms=2**32-40, z_ned_m=-1.0,
                   received_ns=1_000_000_000)
    sync.add_global(time_boot_ms=2**32-40,
                    fc_altitude_home_relative_m=0.03,
                    received_ns=1_000_000_000,
                    now_ns=1_000_000_000)
    sync.add_local(time_boot_ms=10, z_ned_m=-1.0,
                   received_ns=1_050_000_000)
    sync.add_global(time_boot_ms=10, fc_altitude_home_relative_m=0.03,
                    received_ns=1_050_000_000,
                    now_ns=1_050_000_000)
    assert sync.transport_epoch == epoch

    sync.add_local(time_boot_ms=10, z_ned_m=-1.0,
                   received_ns=1_060_000_000)
    assert sync.transport_epoch != epoch
    assert sync.snapshot(1_060_000_000).sample_count == 0


def test_altitude_alignment_resets_stability_window_after_source_gap():
    sync = AltitudeReferenceSynchronizer()
    ready = _feed_altitude_alignment(sync, [-0.97]*11)
    assert ready.stable

    receipt_ns = 4_000_000_000
    sync.add_local(time_boot_ms=4000, z_ned_m=-1.0,
                   received_ns=receipt_ns)
    observation = sync.add_global(
        time_boot_ms=4000, fc_altitude_home_relative_m=0.03,
        received_ns=receipt_ns, now_ns=receipt_ns)
    assert not observation.stable
    assert observation.sample_count == 1
    assert observation.window_duration_ms == 0


def test_altitude_alignment_discontinuity_advances_epoch_and_realigns():
    sync = AltitudeReferenceSynchronizer()
    ready = _feed_altitude_alignment(sync, [-0.97]*11)
    assert ready.stable
    old_epoch = ready.transport_epoch

    receipt_ns = 3_200_000_000
    sync.add_local(time_boot_ms=3200, z_ned_m=-0.3,
                   received_ns=receipt_ns)
    observation = sync.add_global(
        time_boot_ms=3200, fc_altitude_home_relative_m=0.03,
        received_ns=receipt_ns, now_ns=receipt_ns)
    assert observation.transport_epoch != old_epoch
    assert not observation.stable
    assert observation.sample_count == 1
    assert observation.candidate_aligned_home_z_ned_m == pytest.approx(-0.27)


def _ready_sync_with_frozen_px4_home():
    sync = AltitudeReferenceSynchronizer()
    sync.set_home_reference(
        frozen_altitude_amsl_m=43.065216,
        frozen_z_ned_m=-6.157299,
        current_altitude_amsl_m=43.065216,
        current_z_ned_m=-6.157299,
        correction_valid=True,
        correction_revision=0,
        opposition_error_m=0.0,
        estimator_reset_counter_valid=True,
        estimator_reset_counter=3,
        detail="home_reference_locked",
    )
    observation = None
    for index in range(21):
        boot_ms = 1000+index*100
        receipt = 1_000_000_000+index*100_000_000
        sync.add_local(
            time_boot_ms=boot_ms, z_ned_m=-6.157299,
            received_ns=receipt)
        observation = sync.add_global(
            time_boot_ms=boot_ms,
            fc_altitude_home_relative_m=0.0,
            global_altitude_amsl_m=43.065216,
            received_ns=receipt, now_ns=receipt)
    assert observation.state == ALTITUDE_ALIGNMENT_READY
    return sync, observation


@patch("time.monotonic_ns", return_value=3_100_000_000)
def test_px4_home_correction_global_first_keeps_epoch_and_normalized_altitude(_clock):
    sync, ready = _ready_sync_with_frozen_px4_home()
    original_epoch = ready.transport_epoch
    receipt = 3_100_000_000
    sync.add_local(
        time_boot_ms=3100, z_ned_m=-6.158,
        received_ns=receipt)
    pending = sync.add_global(
        time_boot_ms=3100,
        fc_altitude_home_relative_m=1.082,
        global_altitude_amsl_m=43.067,
        received_ns=receipt, now_ns=receipt)
    assert pending.state == ALTITUDE_ALIGNMENT_HOME_CORRECTION_PENDING
    assert pending.transport_epoch == original_epoch
    assert pending.fc_altitude_home_relative_m == pytest.approx(1.082)
    assert pending.normalized_fc_altitude_home_relative_m == pytest.approx(
        43.067-43.065216)

    assert sync.set_home_reference(
        frozen_altitude_amsl_m=43.065216,
        frozen_z_ned_m=-6.157299,
        current_altitude_amsl_m=41.984894,
        current_z_ned_m=-5.076977,
        correction_valid=True,
        correction_revision=1,
        opposition_error_m=0.0,
        estimator_reset_counter_valid=True,
        estimator_reset_counter=3,
        detail="px4_home_only_correction_applied",
    )
    receipt += 100_000_000
    sync.add_local(
        time_boot_ms=3200, z_ned_m=-6.158,
        received_ns=receipt)
    corrected = sync.add_global(
        time_boot_ms=3200,
        fc_altitude_home_relative_m=1.082,
        global_altitude_amsl_m=43.067,
        received_ns=receipt, now_ns=receipt)
    assert corrected.transport_epoch == original_epoch
    assert corrected.state == ALTITUDE_ALIGNMENT_READY
    assert corrected.home_correction_revision == 1
    assert corrected.normalized_fc_altitude_home_relative_m == pytest.approx(
        1.082+(41.984894-43.065216))
    assert corrected.candidate_aligned_home_z_ned_m == pytest.approx(
        -6.156322, abs=0.002)


@patch("time.monotonic_ns", return_value=3_100_000_000)
def test_px4_home_correction_home_first_never_enters_pending(_clock):
    sync, ready = _ready_sync_with_frozen_px4_home()
    original_epoch = ready.transport_epoch
    assert sync.set_home_reference(
        frozen_altitude_amsl_m=43.065216,
        frozen_z_ned_m=-6.157299,
        current_altitude_amsl_m=41.984894,
        current_z_ned_m=-5.076977,
        correction_valid=True,
        correction_revision=1,
        opposition_error_m=0.0,
        estimator_reset_counter_valid=True,
        estimator_reset_counter=3,
        detail="px4_home_only_correction_applied",
    )
    receipt = 3_100_000_000
    sync.add_local(time_boot_ms=3100, z_ned_m=-6.158,
                   received_ns=receipt)
    corrected = sync.add_global(
        time_boot_ms=3100,
        fc_altitude_home_relative_m=1.082,
        global_altitude_amsl_m=43.067,
        received_ns=receipt, now_ns=receipt)
    assert corrected.transport_epoch == original_epoch
    assert corrected.state == ALTITUDE_ALIGNMENT_READY


def test_unconfirmed_px4_home_correction_advances_epoch_after_400ms():
    sync, ready = _ready_sync_with_frozen_px4_home()
    original_epoch = ready.transport_epoch
    receipt = 3_100_000_000
    sync.add_local(time_boot_ms=3100, z_ned_m=-6.158,
                   received_ns=receipt)
    pending = sync.add_global(
        time_boot_ms=3100,
        fc_altitude_home_relative_m=1.082,
        global_altitude_amsl_m=43.067,
        received_ns=receipt, now_ns=receipt)
    assert pending.state == ALTITUDE_ALIGNMENT_HOME_CORRECTION_PENDING
    rejected = sync.snapshot(receipt+401_000_000)
    assert rejected.transport_epoch != original_epoch
    assert rejected.detail == "altitude_home_correction_unconfirmed"
    assert not rejected.valid
    rejected_epoch = rejected.transport_epoch
    # The same failed correction is latched; polling must not create an epoch
    # storm like homecorr1 did on the physical vehicle.
    assert sync.snapshot(receipt+900_000_000).transport_epoch == rejected_epoch


def test_provisional_home_refinement_restarts_once_without_freezing_home():
    sync = AltitudeReferenceSynchronizer()
    first_epoch = sync.transport_epoch
    assert sync.set_provisional_home(
        altitude_amsl_m=43.0, z_ned_m=-5.0, revision=1)
    refined_epoch = sync.transport_epoch
    assert refined_epoch != first_epoch
    assert sync.set_provisional_home(
        altitude_amsl_m=43.0, z_ned_m=-5.0, revision=1)
    assert sync.transport_epoch == refined_epoch
    observation = sync.snapshot()
    assert not observation.execution_home_lock_valid
    assert observation.provisional_home_revision == 1
    assert math.isnan(observation.frozen_px4_home_altitude_amsl_m)

    assert sync.set_provisional_home(
        altitude_amsl_m=41.9, z_ned_m=-3.9, revision=2)
    assert sync.transport_epoch != refined_epoch
    assert sync.snapshot().home_correction_detail != "frozen_home_changed"


def test_altitude_alignment_stable_fixed_offset_but_rejects_time_varying_offset():
    stable = AltitudeReferenceSynchronizer()
    # local Z=-1.0 and FC relative altitude=0.03 produce a stable raw
    # Home/NED offset of -0.97 m; its magnitude is not itself an error.
    observation = _feed_altitude_alignment(stable, [-0.97]*11)
    assert observation.stable
    assert observation.stable_aligned_home_z_ned_m == pytest.approx(-0.97)

    drifting = AltitudeReferenceSynchronizer()
    observation = _feed_altitude_alignment(
        drifting, [-0.97 + 0.1001*index for index in range(11)])
    assert observation.state == ALTITUDE_ALIGNMENT_UNSTABLE
    assert not observation.stable
    assert observation.candidate_span_m == pytest.approx(1.001)


def test_altitude_alignment_ready_state_becomes_stale_without_new_fc_samples():
    sync = AltitudeReferenceSynchronizer()
    ready = _feed_altitude_alignment(sync, [-0.97]*11)
    assert ready.stable
    stale_now_ns = 1_000_000_000 + 10*200_000_000 + 501_000_000
    stale = sync.snapshot(stale_now_ns)
    assert stale.state != ALTITUDE_ALIGNMENT_READY
    assert not stale.valid
    assert not stale.stable


def test_low_speed_route_copy_is_immutable_and_forces_two_metres():
    source = {"route_waypoints_enu": [[0, 1, 15], [2, 3, 20, 90]], "route_revision": 7}
    result = force_low_speed_route(source)
    assert source["route_waypoints_enu"] == [[0, 1, 15], [2, 3, 20, 90]]
    assert result["route_waypoints_enu"] == [[0.0, 1.0, 2.0], [2.0, 3.0, 2.0, 90.0]]


def test_one_metre_profile_is_discrete_and_server_validated():
    source = {"route_waypoints_enu": [[0, 1, 15], [2, 3, 20]], "route_revision": 7}
    result = force_low_speed_route(source, 1)
    assert result["flight_profile"] == LOW_SPEED_1M_V1
    assert result["route_waypoints_enu"] == [[0.0, 1.0, 1.0], [2.0, 3.0, 1.0]]
    assert result["low_speed_limits"]["max_altitude_home_m"] == 1.5
    validate_low_speed_plan(result)
    result["low_speed_limits"]["target_altitude_home_m"] = 1.1
    with pytest.raises(ValueError, match="server constants"):
        validate_low_speed_plan(result)


@pytest.mark.parametrize("value", [0.5, 1.5, 2.5, float("nan"), None])
def test_low_speed_route_rejects_arbitrary_target_altitudes(value):
    with pytest.raises(ValueError, match="exactly 1 or 2"):
        force_low_speed_route({"route_waypoints_enu": [[0, 0, 1], [1, 0, 1]]}, value)


@pytest.mark.parametrize(("yaw", "expected"), [
    (0.0, (12.0, 20.0, -2.0)),
    (math.pi/2, (10.0, 22.0, -2.0)),
    (math.pi, (8.0, 20.0, -2.0)),
])
def test_forward_endpoint_uses_ned_heading(yaw, expected):
    assert forward_endpoint_ned((10, 20, -2), yaw) == pytest.approx(expected)


def test_preview_expires_and_rejects_position_or_heading_drift():
    preview = {"created_at": 100.0, "start_ned": (0, 0, 0), "heading_rad": 0.0}
    assert preview_valid(preview, (0.1, 0, 0), math.radians(4), now=110)[0]
    assert not preview_valid(preview, (0, 0, 0), 0, now=116)[0]
    assert not preview_valid(preview, (0.21, 0, 0), 0, now=110)[0]
    assert not preview_valid(preview, (0, 0, 0), math.radians(6), now=110)[0]
    # A landed vehicle can have local-Z estimator drift without moving in XY.
    assert preview_valid(preview, (0, 0, 0.5), 0, now=110)[0]


def test_envelope_rejects_expiry_speed_and_altitude():
    base = dict(profile=LOW_SPEED_2M_V1, velocity=(0.3, 0.4, -0.5),
                target_position=(0, 0, -2), current_local_z_ned=-2,
                home_z_ned=0, envelope_age_s=0.1)
    assert envelope_allows(**base)[0]
    assert not envelope_allows(**{**base, "envelope_age_s": 0.3})[0]
    assert not envelope_allows(**{**base, "velocity": (0.51, 0, 0)})[0]
    assert envelope_allows(**{
        **base, "target_position": (0, 0, -2.51),
    })[1] == "command_target_altitude_limit_exceeded"
    one_metre = {
        **base, "profile": LOW_SPEED_1M_V1,
        "target_position": (0, 0, -1.5), "current_local_z_ned": -1.5,
    }
    assert envelope_allows(**one_metre)[0]
    assert envelope_allows(**{
        **one_metre, "target_position": (0, 0, -1.51),
    })[1] == "command_target_altitude_limit_exceeded"
    assert envelope_allows(**{
        **one_metre, "current_local_z_ned": -1.51,
    })[1] == "local_home_altitude_limit_exceeded"


def test_aligned_home_datum_converts_observed_offsets_to_absolute_home_targets():
    # Raw PX4 Home NED can imply 1.14 m while relative_alt says 0.03 m.
    local_z = -1.14
    aligned = aligned_home_z_ned(local_z, 0.03)
    assert aligned-local_z == pytest.approx(0.03)
    target_z = aligned-1.0
    assert local_z-target_z == pytest.approx(0.97)

    below_home_aligned = aligned_home_z_ned(local_z, -0.12)
    assert local_z-(below_home_aligned-1.0) == pytest.approx(1.12)


def test_altitude_reference_is_frozen_and_rejects_stale_or_diverged_samples():
    aligned = aligned_home_z_ned(-1.14, 0.03)
    assert altitude_reference_error_m(
        aligned_home_z=aligned, local_z_ned=-1.14,
        fc_altitude_home_relative_m=0.03) == pytest.approx(0.0)
    base = dict(
        aligned_home_z=aligned, local_z_ned=-1.14,
        fc_altitude_home_relative_m=0.03,
        local_age_s=0.1, geo_age_s=0.1, sample_skew_s=0.05,
    )
    assert altitude_reference_sample_error(**base)[0] == ""
    assert altitude_reference_sample_error(
        **{**base, "fc_altitude_home_relative_m": 1.53})[0] == ""
    assert altitude_reference_sample_error(
        **{**base, "fc_altitude_home_relative_m": 1.531})[0] \
        == "altitude_reference_mismatch"
    assert altitude_reference_sample_error(
        **{**base, "geo_age_s": 0.501})[0] == "altitude_reference_stale"
    assert altitude_reference_sample_error(
        **{**base, "sample_skew_s": 0.251})[0] == "altitude_reference_stale"


def test_fc_relative_altitude_ceiling_has_distinct_rejection_reason():
    base = dict(
        profile=LOW_SPEED_1M_V1, velocity=(0.0, 0.0, 0.0),
        target_position=(0.0, 0.0, -1.0), current_local_z_ned=-1.0,
        home_z_ned=0.0,
        envelope_age_s=0.1,
    )
    assert envelope_allows(
        **base, fc_altitude_home_relative_m=1.5)[0]
    assert envelope_allows(
        **base, fc_altitude_home_relative_m=1.501)[1] \
        == "fc_home_altitude_limit_exceeded"


def test_low_speed_pose_requires_local_ned_same_epoch_and_150ms():
    base = dict(
        position_source="LOCAL_POSITION_NED",
        position_received_age_s=0.1,
        position_time_boot_ms=1000,
        position_transport_epoch=7,
        altitude_local_time_boot_ms=1150,
        altitude_transport_epoch=7,
    )
    assert low_speed_pose_coherence_reason(**base) == ""
    assert low_speed_pose_coherence_reason(**{
        **base, "position_time_boot_ms": 999,
    }) == "altitude_pose_time_skew"
    assert low_speed_pose_coherence_reason(**{
        **base, "position_source": "ODOMETRY",
    }) == "altitude_pose_source_mismatch"
    assert low_speed_pose_coherence_reason(**{
        **base, "position_transport_epoch": 6,
    }) == "altitude_pose_time_skew"
    assert low_speed_pose_coherence_reason(**{
        **base, "position_received_age_s": 0.501,
    }) == "altitude_pose_time_skew"


def test_relaxed_reference_mismatch_does_not_bypass_local_ceiling():
    aligned_home_z = 0.0
    reference_reason, error = altitude_reference_sample_error(
        aligned_home_z=aligned_home_z,
        local_z_ned=-1.6,
        fc_altitude_home_relative_m=0.1,
        local_age_s=0.1,
        geo_age_s=0.1,
        sample_skew_s=0.05,
    )
    assert reference_reason == ""
    assert error == pytest.approx(1.5)
    allowed, reason = envelope_allows(
        profile=LOW_SPEED_1M_V1,
        velocity=(0.0, 0.0, 0.0),
        target_position=(0.0, 0.0, -1.0),
        current_local_z_ned=-1.6,
        home_z_ned=aligned_home_z,
        envelope_age_s=0.1,
        fc_altitude_home_relative_m=0.1,
    )
    assert not allowed
    assert reason == "local_home_altitude_limit_exceeded"


def test_global_position_freshness_expires_after_750_ms():
    assert geo_is_fresh(True, 0)
    assert geo_is_fresh(True, 750)
    assert not geo_is_fresh(True, 751)
    assert not geo_is_fresh(False, 10)
    assert not geo_is_fresh(True, None)


@pytest.mark.parametrize(("value", "expected"), [
    (0.5, True),
    (0.6, True),
    (0.6000000238418579, True),
    (0.6002, False),
    (0.7, False),
    (0.0, False),
    (float("nan"), False),
    (None, False),
])
def test_px4_land_speed_accepts_stock_minimum_with_float_tolerance(value, expected):
    assert px4_land_speed_is_valid(value) is expected


@pytest.mark.parametrize(("value", "expected"), [
    (4, True),
    (4.0, True),
    (4.00000001, True),
    (3, False),
    (5, False),
    (float("nan"), False),
    (None, False),
])
def test_px4_offboard_loss_action_requires_land_enum(value, expected):
    assert px4_offboard_loss_action_is_land(value) is expected


def _ready_state(**overrides):
    values = dict(command_output_enabled=True, connected=True,
                  vehicle_status_fresh=True, arming_state_valid=True,
                  landed_state_valid=True,
                  preflight_checks_pass=True, position_valid=True,
                  heading_fresh=True, manual_override=False, armed=False,
                  landed=True, low_speed_obstacle_guard_enabled=True,
                  jetson_safety_fresh=True, jetson_safety_state="CLEAR")
    values.update(overrides)
    return SimpleNamespace(**values)


def _readiness(state=None, **overrides):
    values = dict(
        state=_ready_state() if state is None else state,
        geo=SimpleNamespace(valid=True, age_ms=100),
        safety=SimpleNamespace(valid=True, age_ms=100, detail="ready"),
        now=10.0, vehicle_state_received_at=9.8, active_mission=False,
        allow_active=False, require_ground=True, physical=True,
        home_fresh=True,
    )
    values.update(overrides)
    return low_speed_readiness_error(**values)


def test_low_speed_readiness_accepts_only_complete_fresh_ground_state():
    assert _readiness() == ""
    assert "stale" in _readiness(vehicle_state_received_at=9.0)
    assert "status" in _readiness(state=_ready_state(vehicle_status_fresh=False))
    assert "arming" in _readiness(state=_ready_state(arming_state_valid=False))
    assert "landed state" in _readiness(state=_ready_state(landed_state_valid=False))
    assert "heading" in _readiness(state=_ready_state(heading_fresh=False))
    assert "manual" in _readiness(state=_ready_state(manual_override=True))
    assert "disarmed" in _readiness(state=_ready_state(armed=True, landed=False))
    assert "another mission" in _readiness(active_mission=True)
    assert "obstacle" in _readiness(
        state=_ready_state(low_speed_obstacle_guard_enabled=False))
    assert "stale" in _readiness(
        state=_ready_state(jetson_safety_fresh=False))
    assert "not CLEAR" in _readiness(
        state=_ready_state(jetson_safety_state="HOLD"))
    assert "global" in _readiness(geo=SimpleNamespace(valid=True, age_ms=751))
    assert "QGC" in _readiness(
        safety=SimpleNamespace(valid=False, age_ms=100, detail="QGC parameters invalid"))
    assert "Home" in _readiness(home_fresh=False)


def test_forward_test_can_explicitly_skip_only_the_obstacle_guard_checks():
    state = _ready_state(
        low_speed_obstacle_guard_enabled=False,
        jetson_safety_fresh=False,
        jetson_safety_state="STALE",
    )
    assert _readiness(
        state=state,
        require_obstacle_guard=False,
    ) == ""
    # PX4 failsafe, geo and Home checks remain mandatory in camera-off mode.
    assert "QGC" in _readiness(
        state=state,
        require_obstacle_guard=False,
        safety=SimpleNamespace(
            valid=False, age_ms=100, detail="QGC parameters invalid"
        ),
    )


@pytest.mark.parametrize("profile", [LOW_SPEED_1M_V1, LOW_SPEED_2M_V1])
@pytest.mark.parametrize("camera_required", [True, False])
def test_forward_test_camera_on_off_readiness_for_both_profiles(
        profile, camera_required):
    # Fixed limits prove that both selectable profiles take this same gate.
    plan = force_low_speed_route(
        {"route_waypoints_enu": [[0, 0, 10], [1, 0, 10]]},
        1 if profile == LOW_SPEED_1M_V1 else 2,
    )
    assert plan["flight_profile"] == profile
    camera_missing = _ready_state(
        low_speed_obstacle_guard_enabled=False,
        jetson_safety_fresh=False,
        jetson_safety_state="STALE",
    )
    error = _readiness(
        state=camera_missing,
        require_obstacle_guard=obstacle_guard_required_for_mission(
            FORWARD_TEST_1M, camera_required
        ),
    )
    assert bool(error) is camera_required


def test_camera_off_never_bypasses_a_low_speed_route():
    assert obstacle_guard_required_for_mission(ROUTE, False)
    assert not obstacle_guard_required_for_mission(FORWARD_TEST_1M, False)
    error = _readiness(
        state=_ready_state(
            low_speed_obstacle_guard_enabled=False,
            jetson_safety_fresh=False,
            jetson_safety_state="STALE",
        ),
        require_obstacle_guard=obstacle_guard_required_for_mission(ROUTE, False),
    )
    assert "obstacle" in error


def test_camera_bypass_rejects_targets_beyond_2_2_metres():
    assert forward_test_target_within_camera_bypass((10, 20), (12.2, 20))
    assert not forward_test_target_within_camera_bypass((10, 20), (12.201, 20))
    assert not forward_test_target_within_camera_bypass(
        (10, 20), (float("nan"), 20)
    )


@pytest.mark.parametrize(("command_profile", "envelope_profile", "expected"), [
    (None, None, True),
    (LOW_SPEED_1M_V1, LOW_SPEED_1M_V1, True),
    (LOW_SPEED_2M_V1, LOW_SPEED_2M_V1, True),
    (LOW_SPEED_1M_V1, LOW_SPEED_2M_V1, False),
    (LOW_SPEED_2M_V1, LOW_SPEED_1M_V1, False),
    (LOW_SPEED_1M_V1, None, False),
    (None, LOW_SPEED_2M_V1, False),
])
def test_command_and_envelope_profiles_must_match(
        command_profile, envelope_profile, expected):
    assert low_speed_envelope_profiles_match(
        command_profile, envelope_profile
    ) is expected


def test_gateway_protocol_rejects_arbitrary_forward_test_altitude():
    with pytest.raises(ProtocolError, match="exactly 1 or 2"):
        parse_browser_message(
            '{"type":"mission.forward_test_prepare",'
            '"operator_id":"operator",'
            '"target_altitude_home_m":1.2}'
        )


def test_gateway_protocol_validates_emergency_land_identity():
    message = parse_browser_message(
        '{"type":"mission.emergency_land",'
        '"mission_id":"mission-1","operator_id":"operator",'
        '"reason":"unsafe motion"}'
    )
    assert message == {
        "type": "mission.emergency_land",
        "mission_id": "mission-1",
        "operator_id": "operator",
        "reason": "unsafe motion",
    }
    with pytest.raises(ProtocolError):
        parse_browser_message(
            '{"type":"mission.emergency_land",'
            '"mission_id":"","operator_id":"operator","reason":"x"}'
        )


def test_takeoff_handshake_never_arms_before_tx_and_offboard_confirmation():
    decide = takeoff_handshake_action
    assert decide(output_ready=False, offboard=False, armed=False) == "NONE"
    assert decide(output_ready=True, offboard=False, armed=False) == "MODE"
    assert decide(output_ready=True, offboard=True, armed=False) == "ARM"
    assert decide(output_ready=True, offboard=True, armed=True) == "NONE"
    assert decide(output_ready=True, offboard=False, armed=False,
                  mode_timed_out=True) == "NONE"
    assert decide(output_ready=True, offboard=True, armed=False,
                  arm_timed_out=True) == "NONE"


def test_flight_output_confirmation_requires_fresh_continuous_exact_tx():
    base = dict(
        receipt_age_s=0.05, continuous_age_s=1.0,
        setpoint_age_ms=50, consecutive_transmissions=20,
        exact_match=True, transport_connected=True,
        command_graph_ready=True, envelope_valid=True,
        setpoint_transmitted=True, warmup_s=1.0,
    )
    assert flight_output_confirmation_ready(**base)
    assert not flight_output_confirmation_ready(
        **{**base, "continuous_age_s": 0.99})
    assert not flight_output_confirmation_ready(
        **{**base, "receipt_age_s": 0.151})
    assert not flight_output_confirmation_ready(
        **{**base, "exact_match": False})
    assert not flight_output_confirmation_ready(
        **{**base, "envelope_valid": False})
    assert not flight_output_confirmation_ready(
        **{**base, "consecutive_transmissions": 1})


@pytest.mark.parametrize(("kind", "mode", "position", "velocity"), [
    (1, {"position": True, "velocity": False}, (1, 2, -3),
     (math.nan, math.nan, math.nan)),
    (2, {"position": False, "velocity": True},
     (math.nan, math.nan, math.nan), (0.1, 0.2, -0.3)),
])
def test_flight_contract_matches_position_and_velocity(kind, mode, position, velocity):
    valid, reason = setpoint_contract_matches(
        setpoint_kind=kind, mode=mode, position=position, velocity=velocity,
        yaw=0.25, expected_position=position, expected_velocity=velocity,
        expected_yaw=0.25,
    )
    assert valid, reason


def test_flight_contract_rejects_kind_values_and_nan_mask_mismatch():
    base = dict(
        setpoint_kind=2, mode={"position": False, "velocity": True},
        position=(math.nan, math.nan, math.nan), velocity=(0.0, 0.0, 0.0),
        yaw=0.0, expected_position=(math.nan, math.nan, math.nan),
        expected_velocity=(0.0, 0.0, 0.0), expected_yaw=0.0,
    )
    assert not setpoint_contract_matches(
        **{**base, "setpoint_kind": 1})[0]
    assert not setpoint_contract_matches(
        **{**base, "expected_velocity": (0.01, 0.0, 0.0)})[0]
    assert not setpoint_contract_matches(
        **{**base, "expected_position": (0.0, math.nan, math.nan)})[0]
    assert not setpoint_contract_matches(
        **{**base, "expected_yaw": 0.1})[0]
