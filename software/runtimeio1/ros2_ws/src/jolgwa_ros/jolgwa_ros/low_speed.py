"""Fixed, fail-closed contract for the physical low-speed flight profile."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import statistics
import time


NORMAL = "NORMAL"
LOW_SPEED_1M_V1 = "LOW_SPEED_1M_V1"
LOW_SPEED_2M_V1 = "LOW_SPEED_2M_V1"
ROUTE = "ROUTE"
FORWARD_TEST_1M = "FORWARD_TEST_1M"
# Keep the published mission-kind/authority identifiers for protocol 13.
# Their legacy names do not set the distance of this bundled forward test.
FORWARD_TEST_DISTANCE_M = 2.0
TARGET_ALTITUDE_M = 2.0
MAX_ALTITUDE_M = 2.5
LOW_SPEED_PROFILE_LIMITS = {
    LOW_SPEED_1M_V1: (1.0, 1.5),
    LOW_SPEED_2M_V1: (2.0, 2.5),
}
MAX_HORIZONTAL_SPEED_M_S = 0.5
MAX_VERTICAL_SPEED_M_S = 0.5
# Stock PX4 constrains MPC_LAND_SPEED to a minimum of 0.6 m/s.  This is
# deliberately separate from the Offboard vertical-speed envelope above:
# native LAND remains available after companion-link loss.
PX4_LAND_SPEED_MAX_M_S = 0.6
PX4_PARAMETER_FLOAT_TOLERANCE = 1e-4
PX4_COM_OBL_RC_ACT_LAND = 4
ACCEPTANCE_RADIUS_M = 0.25
SETTLE_ALTITUDE_ERROR_M = 0.15
SETTLE_SPEED_M_S = 0.1
SETTLE_TIME_S = 0.5
PREVIEW_TTL_S = 15.0
PREVIEW_POSITION_DRIFT_M = 0.2
PREVIEW_HEADING_DRIFT_RAD = math.radians(5.0)
GEO_FRESHNESS_MS = 750
FORWARD_TEST_CAMERA_BYPASS_MAX_TARGET_M = (
    FORWARD_TEST_DISTANCE_M + PREVIEW_POSITION_DRIFT_M)
SETPOINT_CONTRACT_TOLERANCE = 1e-5
# The three flight-output topics are published sequentially but delivered by
# separate ROS subscriptions.  Allow one Jetson scheduling stall while still
# requiring every member of the contract to remain inside the 250 ms lease.
FLIGHT_OUTPUT_PAIR_SKEW_NS = 150_000_000
ALTITUDE_REFERENCE_MAX_ERROR_M = 1.50
ALTITUDE_REFERENCE_FRESHNESS_S = 0.5
ALTITUDE_REFERENCE_MAX_SAMPLE_SKEW_S = 0.25
ALTITUDE_ALIGNMENT_SOURCE_SKEW_MS = 100
ALTITUDE_ALIGNMENT_SOURCE_FRESHNESS_MS = 500
ALTITUDE_ALIGNMENT_LOCAL_HISTORY_MS = 1000
ALTITUDE_ALIGNMENT_SAMPLE_GAP_MS = 250
ALTITUDE_ALIGNMENT_WINDOW_MS = 2000
ALTITUDE_ALIGNMENT_MIN_SAMPLES = 10
ALTITUDE_ALIGNMENT_MAX_SPAN_M = 1.00
ALTITUDE_ALIGNMENT_DISCONTINUITY_M = 0.50
LOW_SPEED_ALTITUDE_POSE_SKEW_MS = 150
HOME_CORRECTION_CONFIRMATION_MS = 400
HOME_CORRECTION_CROSSCHECK_M = 0.25

ALTITUDE_ALIGNMENT_ALIGNING = 0
ALTITUDE_ALIGNMENT_READY = 1
ALTITUDE_ALIGNMENT_STALE = 2
ALTITUDE_ALIGNMENT_UNSTABLE = 3
ALTITUDE_ALIGNMENT_HOME_CORRECTION_PENDING = 4
HOME_CORRECTION_NONE = 0
HOME_CORRECTION_PENDING = 1
HOME_CORRECTION_APPLIED = 2
HOME_CORRECTION_REJECTED = 3
HOME_PHASE_PROVISIONAL = 0
HOME_PHASE_EXECUTION_LOCKED = 1
HOME_PHASE_CORRECTION_PENDING = 2
HOME_PHASE_REJECTED = 3
_BOOT_TIME_MODULUS = 2**32
_BOOT_TIME_HALF_RANGE = 2**31


def accumulated_sample_age_s(source_age_ms, received_at, now):
    """Source age plus local residence; no cross-host clock assumption."""
    try:
        source, receipt, current = float(source_age_ms)/1000.0, float(received_at), float(now)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if not all(math.isfinite(v) for v in (source, receipt, current)) or source < 0 or receipt <= 0 or current < receipt:
        return math.inf
    return source + (current-receipt)


def altitude_sample_advances(previous, incoming):
    if previous is None:
        return True
    old_epoch, new_epoch = int(previous.transport_epoch), int(incoming.transport_epoch)
    if old_epoch != new_epoch:
        return boot_time_forward_delta_ms(new_epoch, old_epoch) is not None
    return boot_time_forward_delta_ms(int(incoming.sequence), int(previous.sequence)) is not None


def flight_output_pair_skew_valid(first_received_ns, second_received_ns):
    """Return whether two flight-output messages belong to the same live pair."""
    try:
        first = int(first_received_ns)
        second = int(second_received_ns)
    except (TypeError, ValueError, OverflowError):
        return False
    return first >= 0 and second >= 0 \
        and abs(first-second) <= FLIGHT_OUTPUT_PAIR_SKEW_NS


def boot_time_forward_delta_ms(new_time_boot_ms, previous_time_boot_ms):
    """Return a wrap-aware forward delta, or None for duplicate/reversal."""
    try:
        new = int(new_time_boot_ms)
        previous = int(previous_time_boot_ms)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (0 <= new < _BOOT_TIME_MODULUS
            and 0 <= previous < _BOOT_TIME_MODULUS):
        return None
    delta = (new-previous) % _BOOT_TIME_MODULUS
    if delta == 0 or delta >= _BOOT_TIME_HALF_RANGE:
        return None
    return delta


def boot_time_distance_ms(first_time_boot_ms, second_time_boot_ms):
    """Return the shortest unsigned distance between two PX4 boot clocks."""
    try:
        first = int(first_time_boot_ms)
        second = int(second_time_boot_ms)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if not (0 <= first < _BOOT_TIME_MODULUS
            and 0 <= second < _BOOT_TIME_MODULUS):
        return math.inf
    delta = (first-second) % _BOOT_TIME_MODULUS
    return min(delta, _BOOT_TIME_MODULUS-delta)


@dataclass(frozen=True)
class AltitudeReferenceObservation:
    state: int
    valid: bool
    stable: bool
    transport_epoch: int
    sequence: int
    local_time_boot_ms: int
    global_time_boot_ms: int
    local_age_ms: int
    global_age_ms: int
    source_skew_ms: int
    local_z_ned_m: float
    fc_altitude_home_relative_m: float
    candidate_aligned_home_z_ned_m: float
    stable_aligned_home_z_ned_m: float
    candidate_span_m: float
    sample_count: int
    window_duration_ms: int
    detail: str
    normalized_fc_altitude_home_relative_m: float = math.nan
    global_altitude_amsl_m: float = math.nan
    home_correction_state: int = HOME_CORRECTION_NONE
    home_correction_valid: bool = False
    home_correction_revision: int = 0
    home_correction_pending_age_ms: int = 0
    frozen_px4_home_altitude_amsl_m: float = math.nan
    current_px4_home_altitude_amsl_m: float = math.nan
    frozen_px4_home_z_ned_m: float = math.nan
    current_px4_home_z_ned_m: float = math.nan
    home_altitude_correction_m: float = 0.0
    home_z_correction_m: float = 0.0
    home_correction_opposition_error_m: float = math.nan
    frozen_home_crosscheck_error_m: float = math.nan
    estimator_reset_counter_valid: bool = False
    estimator_reset_counter: int = 0
    home_correction_detail: str = "home_correction_unavailable"
    home_phase: int = HOME_PHASE_PROVISIONAL
    execution_home_lock_valid: bool = False
    execution_home_mission_id: str = ""
    execution_home_lock_revision: int = 0
    provisional_home_revision: int = 0
    provisional_px4_home_altitude_amsl_m: float = math.nan
    provisional_px4_home_z_ned_m: float = math.nan
    altitude_epoch_failure_latched: bool = False


class AltitudeReferenceSynchronizer:
    """Pair FC-clock telemetry and prove a stable low-speed vertical datum."""

    def __init__(self):
        self.confirmation_ms = HOME_CORRECTION_CONFIRMATION_MS
        self.transport_epoch = 0
        self.sequence = 0
        self._local_history = deque(maxlen=64)
        self._window = deque(maxlen=128)
        self._last_local_boot_ms = None
        self._last_global_boot_ms = None
        self._global_unwrapped_ms = 0
        self._last_pair = None
        self._window_has_full_coverage = False
        self._forced_detail = "waiting_for_synchronized_altitude_samples"
        self._frozen_home_altitude_amsl_m = math.nan
        self._frozen_home_z_ned_m = math.nan
        self._current_home_altitude_amsl_m = math.nan
        self._current_home_z_ned_m = math.nan
        self._home_correction_m = 0.0
        self._home_z_correction_m = 0.0
        self._home_correction_opposition_error_m = math.nan
        self._home_correction_revision = 0
        self._home_correction_valid = False
        self._home_correction_state = HOME_CORRECTION_NONE
        self._home_correction_detail = "home_correction_unavailable"
        self._home_correction_pending_since_ns = 0
        self._correction_event_id = 0
        self._correction_event_revision = 0
        self._correction_expected_delta_m = math.nan
        self._last_global_altitude_amsl_m = math.nan
        self._last_crosscheck_error_m = math.nan
        self._estimator_reset_counter_valid = False
        self._estimator_reset_counter = 0
        self._home_phase = HOME_PHASE_PROVISIONAL
        self._execution_home_lock_valid = False
        self._execution_home_mission_id = ""
        self._execution_home_lock_revision = 0
        self._retired_home_generations = set()
        self._provisional_home_revision = 0
        self._provisional_home_altitude_amsl_m = math.nan
        self._provisional_home_z_ned_m = math.nan
        self._epoch_failure_latched = False
        self._epoch_failure_detail = ""
        self.begin_epoch(self._forced_detail)

    def begin_epoch(self, detail="altitude_reference_epoch_reset"):
        self.transport_epoch = (self.transport_epoch + 1) % _BOOT_TIME_MODULUS
        if self.transport_epoch == 0:
            self.transport_epoch = 1
        self._local_history.clear()
        self._window.clear()
        self._last_local_boot_ms = None
        self._last_global_boot_ms = None
        self._global_unwrapped_ms = 0
        self._last_pair = None
        self._window_has_full_coverage = False
        self._home_correction_pending_since_ns = 0
        self._forced_detail = str(detail)

    def mark_stale(self, detail):
        self._forced_detail = str(detail)

    def set_provisional_home(self, *, altitude_amsl_m, z_ned_m,
                             revision, detail="provisional_home"):
        """Follow PX4 Home before a flight contract exists.

        Each distinct revision restarts stabilization exactly once.  It does
        not establish an immutable/frozen Home and therefore cannot produce
        ``frozen_home_changed`` during normal PX4 startup refinement.
        """
        try:
            altitude = float(altitude_amsl_m)
            z_value = float(z_ned_m)
            candidate_revision = max(0, int(revision))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(altitude) or not math.isfinite(z_value):
            return False
        if self._execution_home_lock_valid:
            return False
        if candidate_revision < self._provisional_home_revision:
            return False
        changed = candidate_revision != self._provisional_home_revision
        self._provisional_home_altitude_amsl_m = altitude
        self._provisional_home_z_ned_m = z_value
        self._current_home_altitude_amsl_m = altitude
        self._current_home_z_ned_m = z_value
        self._home_phase = HOME_PHASE_PROVISIONAL
        self._home_correction_state = HOME_CORRECTION_NONE
        self._home_correction_valid = False
        self._home_correction_detail = str(detail)
        self._epoch_failure_latched = False
        self._epoch_failure_detail = ""
        if changed:
            self._provisional_home_revision = candidate_revision
            self._frozen_home_altitude_amsl_m = math.nan
            self._frozen_home_z_ned_m = math.nan
            self._home_correction_m = 0.0
            self._home_z_correction_m = 0.0
            self._home_correction_revision = 0
            self._correction_event_revision = 0
            self.begin_epoch("provisional_home_refined")
        return True

    def set_home_reference(self, *, frozen_altitude_amsl_m,
                           frozen_z_ned_m, current_altitude_amsl_m,
                           current_z_ned_m, correction_valid,
                           correction_revision, opposition_error_m,
                           estimator_reset_counter_valid,
                           estimator_reset_counter, detail,
                           home_phase="EXECUTION_LOCKED",
                           execution_home_lock_valid=True,
                           execution_home_mission_id="",
                           execution_home_lock_revision=0,
                           provisional_home_revision=0,
                           provisional_altitude_amsl_m=math.nan,
                           provisional_z_ned_m=math.nan,
                           epoch_failure_latched=False):
        """Update PX4 Home metadata without changing the altitude epoch.

        The frozen Home is immutable.  A caller may update the current PX4
        Home only after independently proving that the change is metadata-only.
        """
        generation = (str(execution_home_mission_id), int(execution_home_lock_revision))
        current = (self._execution_home_mission_id, self._execution_home_lock_revision)
        if (generation in self._retired_home_generations
                or (self._execution_home_lock_valid and generation != current)
                or int(provisional_home_revision) < self._provisional_home_revision):
            return False
        if self._correction_deadline_expired(time.monotonic_ns()):
            return False
        if int(correction_revision) < self._home_correction_revision:
            return False  # Delayed metadata cannot roll back confirmed evidence.
        values = tuple(float(value) for value in (
            frozen_altitude_amsl_m, frozen_z_ned_m,
            current_altitude_amsl_m, current_z_ned_m,
        ))
        if not all(math.isfinite(value) for value in values):
            self._home_correction_valid = False
            self._home_correction_state = HOME_CORRECTION_REJECTED
            self._home_correction_detail = "home_correction_metadata_invalid"
            return False
        frozen_alt, frozen_z, current_alt, current_z = values
        if math.isfinite(self._frozen_home_altitude_amsl_m) and (
            abs(self._frozen_home_altitude_amsl_m-frozen_alt) > 1e-6
            or abs(self._frozen_home_z_ned_m-frozen_z) > 1e-6
        ):
            self._home_correction_valid = False
            self._home_correction_state = HOME_CORRECTION_REJECTED
            self._home_correction_detail = "frozen_home_changed"
            return False
        previous_revision = self._home_correction_revision
        if int(correction_revision) > previous_revision and not self._epoch_failure_latched:
            # HOME-first uses the same event as GLOBAL-first.  Do this before
            # updating revision: confirmation requires genuinely new evidence.
            self._start_correction_event(time.monotonic_ns(), current_alt-frozen_alt)
            if not math.isfinite(self._correction_expected_delta_m):
                self._correction_expected_delta_m = current_alt-frozen_alt
        self._frozen_home_altitude_amsl_m = frozen_alt
        self._frozen_home_z_ned_m = frozen_z
        self._current_home_altitude_amsl_m = current_alt
        self._current_home_z_ned_m = current_z
        self._home_correction_m = current_alt-frozen_alt
        self._home_z_correction_m = current_z-frozen_z
        self._home_correction_opposition_error_m = float(opposition_error_m)
        self._home_correction_revision = max(0, int(correction_revision))
        self._home_phase = {
            "PROVISIONAL": HOME_PHASE_PROVISIONAL,
            "EXECUTION_LOCKED": HOME_PHASE_EXECUTION_LOCKED,
            "CORRECTION_PENDING": HOME_PHASE_CORRECTION_PENDING,
            "REJECTED": HOME_PHASE_REJECTED,
        }.get(str(home_phase), HOME_PHASE_EXECUTION_LOCKED)
        self._execution_home_lock_valid = bool(
            execution_home_lock_valid)
        self._execution_home_mission_id = str(
            execution_home_mission_id)
        self._execution_home_lock_revision = max(
            0, int(execution_home_lock_revision))
        self._provisional_home_revision = max(
            self._provisional_home_revision,
            max(0, int(provisional_home_revision)))
        provisional_altitude = float(provisional_altitude_amsl_m)
        provisional_z = float(provisional_z_ned_m)
        if math.isfinite(provisional_altitude):
            self._provisional_home_altitude_amsl_m = provisional_altitude
        if math.isfinite(provisional_z):
            self._provisional_home_z_ned_m = provisional_z
        self._epoch_failure_latched = bool(
            self._epoch_failure_latched or epoch_failure_latched)
        self._estimator_reset_counter_valid = bool(
            estimator_reset_counter_valid)
        self._estimator_reset_counter = int(estimator_reset_counter) & 0xFF
        self._home_correction_valid = bool(correction_valid)
        self._home_correction_state = (
            HOME_CORRECTION_APPLIED
            if self._home_correction_valid and (
                abs(self._home_correction_m) > 1e-6
                or abs(self._home_z_correction_m) > 1e-6)
            else HOME_CORRECTION_NONE
            if self._home_correction_revision == 0
            else HOME_CORRECTION_REJECTED
        )
        if self._home_phase == HOME_PHASE_CORRECTION_PENDING:
            self._home_correction_state = HOME_CORRECTION_PENDING
            self._start_correction_event(time.monotonic_ns(), math.nan)
        elif self._home_phase == HOME_PHASE_REJECTED:
            self._home_correction_state = HOME_CORRECTION_REJECTED
        self._home_correction_detail = str(detail)
        if self._epoch_failure_latched:
            self._home_correction_valid = False
            self._home_correction_state = HOME_CORRECTION_REJECTED
            self._home_phase = HOME_PHASE_REJECTED
            if self._epoch_failure_detail:
                self._home_correction_detail = self._epoch_failure_detail
        # Repeated old Home metadata is not evidence for a pending event.
        # Only a paired GLOBAL sample can complete its independent crosscheck.
        if self._home_correction_pending_since_ns and not self._epoch_failure_latched:
            self._home_correction_state = HOME_CORRECTION_PENDING
            self._home_phase = HOME_PHASE_CORRECTION_PENDING
        # Metadata acceptance is distinct from evidence for a Home correction.
        # Revision zero can legitimately lack an estimator reset counter.
        return bool(not self._epoch_failure_latched
            and (self._home_correction_valid or self._home_correction_revision == 0))

    def release_execution_home(self):
        """Caller has proven explicit contract end plus fresh ground state."""
        self._retired_home_generations.add((self._execution_home_mission_id, self._execution_home_lock_revision))
        self._execution_home_lock_valid = False
        self._execution_home_mission_id = ""
        self._frozen_home_altitude_amsl_m = math.nan
        self._frozen_home_z_ned_m = math.nan
        self._home_correction_revision = 0
        self._correction_event_revision = 0
        self._home_correction_valid = False
        self._home_correction_pending_since_ns = 0

    def _correction_deadline_expired(self, now_ns):
        pending = self._home_correction_pending_since_ns
        if pending and (now_ns < pending or now_ns-pending > self.confirmation_ms*1_000_000):
            self.reject_home_correction("altitude_home_correction_unconfirmed")
        return self._epoch_failure_latched

    def _start_correction_event(self, now_ns, expected_delta):
        if not self._home_correction_pending_since_ns:
            self._correction_event_id += 1
            self._correction_event_revision = self._home_correction_revision
            self._correction_expected_delta_m = float(expected_delta)
            self._home_correction_pending_since_ns = now_ns

    def reject_home_correction(self, detail):
        if self._epoch_failure_latched:
            return False
        self._home_correction_valid = False
        self._home_correction_state = HOME_CORRECTION_REJECTED
        self._home_phase = HOME_PHASE_REJECTED
        self._home_correction_detail = str(detail)
        self._epoch_failure_latched = True
        self._epoch_failure_detail = str(detail)
        self.begin_epoch(str(detail))
        return True

    def add_local(self, *, time_boot_ms, z_ned_m, received_ns):
        try:
            boot_ms = int(time_boot_ms)
            z_value = float(z_ned_m)
            receipt = int(received_ns)
        except (TypeError, ValueError, OverflowError):
            self.begin_epoch("local_altitude_sample_invalid")
            return False
        if (not 0 <= boot_ms < _BOOT_TIME_MODULUS
                or not math.isfinite(z_value) or receipt <= 0):
            self.begin_epoch("local_altitude_sample_invalid")
            return False
        if self._last_local_boot_ms is not None:
            delta = boot_time_forward_delta_ms(boot_ms, self._last_local_boot_ms)
            if delta is None:
                self.begin_epoch("local_time_boot_duplicate_or_reversed")
        self._last_local_boot_ms = boot_ms
        self._local_history.append((boot_ms, z_value, receipt))
        self._forced_detail = "collecting_synchronized_altitude_samples"
        return True

    def add_global(self, *, time_boot_ms, fc_altitude_home_relative_m,
                   received_ns, now_ns=None, global_altitude_amsl_m=math.nan):
        now_ns = int(received_ns if now_ns is None else now_ns)
        if self._correction_deadline_expired(now_ns):
            return self.snapshot(now_ns)
        try:
            boot_ms = int(time_boot_ms)
            fc_altitude = float(fc_altitude_home_relative_m)
            global_altitude = float(global_altitude_amsl_m)
            receipt = int(received_ns)
        except (TypeError, ValueError, OverflowError):
            self.begin_epoch("global_altitude_sample_invalid")
            return self.snapshot(now_ns)
        if (not 0 <= boot_ms < _BOOT_TIME_MODULUS
                or not math.isfinite(fc_altitude) or receipt <= 0):
            self.begin_epoch("global_altitude_sample_invalid")
            return self.snapshot(now_ns)

        if self._last_global_boot_ms is None:
            delta = None
        else:
            delta = boot_time_forward_delta_ms(
                boot_ms, self._last_global_boot_ms)
            if delta is None:
                self.begin_epoch("global_time_boot_duplicate_or_reversed")
            elif delta > ALTITUDE_ALIGNMENT_SAMPLE_GAP_MS:
                self._window.clear()
                self._window_has_full_coverage = False
                self._forced_detail = "altitude_sample_gap_exceeded"
        if self._last_global_boot_ms is None:
            self._global_unwrapped_ms = 0
        elif delta is not None:
            self._global_unwrapped_ms += delta
        self._last_global_boot_ms = boot_ms

        minimum_receipt = now_ns-ALTITUDE_ALIGNMENT_LOCAL_HISTORY_MS*1_000_000
        while self._local_history and self._local_history[0][2] < minimum_receipt:
            self._local_history.popleft()
        if not self._local_history:
            self._forced_detail = "matching_local_altitude_sample_missing"
            return self.snapshot(now_ns)
        local = min(
            self._local_history,
            key=lambda item: boot_time_distance_ms(item[0], boot_ms),
        )
        source_skew = boot_time_distance_ms(local[0], boot_ms)
        if source_skew > ALTITUDE_ALIGNMENT_SOURCE_SKEW_MS:
            self._forced_detail = "altitude_source_skew_exceeded"
            return self.snapshot(now_ns)
        local_age_ms = max(0, (now_ns-local[2])//1_000_000)
        global_age_ms = max(0, (now_ns-receipt)//1_000_000)
        if (local_age_ms > ALTITUDE_ALIGNMENT_SOURCE_FRESHNESS_MS
                or global_age_ms > ALTITUDE_ALIGNMENT_SOURCE_FRESHNESS_MS):
            self._forced_detail = "altitude_source_sample_stale"
            return self.snapshot(now_ns)

        normalized_fc_altitude = fc_altitude
        crosscheck_error = math.nan
        frozen_home_available = math.isfinite(
            self._frozen_home_altitude_amsl_m)
        global_altitude_available = math.isfinite(global_altitude)
        if frozen_home_available:
            normalized_fc_altitude = fc_altitude+self._home_correction_m
            if global_altitude_available:
                frozen_home_altitude = (
                    global_altitude-self._frozen_home_altitude_amsl_m)
                crosscheck_error = abs(
                    normalized_fc_altitude-frozen_home_altitude)
                # A relative-altitude jump can precede the HOME_POSITION that
                # proves PX4's metadata-only correction.  Use the independent
                # AMSL/frozen-Home value provisionally while waiting for that
                # proof; never silently advance the altitude epoch here.
                if self._epoch_failure_latched:
                    return self.snapshot(now_ns)
                if crosscheck_error > HOME_CORRECTION_CROSSCHECK_M+1e-6:
                    self._start_correction_event(
                        now_ns, frozen_home_altitude-fc_altitude)
                    pending_age_ms = max(
                        0, (now_ns-self._home_correction_pending_since_ns)
                        // 1_000_000)
                    if pending_age_ms > self.confirmation_ms:
                        self.reject_home_correction(
                            "altitude_home_correction_unconfirmed")
                        return self.snapshot(now_ns)
                    self._home_correction_state = HOME_CORRECTION_PENDING
                    self._home_phase = HOME_PHASE_CORRECTION_PENDING
                    self._home_correction_detail = (
                        "waiting_for_px4_home_correction_confirmation")
                    normalized_fc_altitude = frozen_home_altitude
                elif (self._home_correction_state == HOME_CORRECTION_PENDING
                      and self._home_correction_valid
                      and self._home_correction_revision > self._correction_event_revision
                      and abs(self._home_correction_m-self._correction_expected_delta_m)
                          <= HOME_CORRECTION_CROSSCHECK_M):
                    # The route-state Home proof may have arrived before this
                    # next GLOBAL_POSITION_INT sample.
                    self._home_correction_pending_since_ns = 0
                    self._home_correction_state = (
                        HOME_CORRECTION_APPLIED
                        if abs(self._home_correction_m) > 1e-6
                        else HOME_CORRECTION_NONE)
                    self._home_phase = HOME_PHASE_EXECUTION_LOCKED
        self._last_global_altitude_amsl_m = global_altitude
        self._last_crosscheck_error_m = crosscheck_error
        candidate = aligned_home_z_ned(local[1], normalized_fc_altitude)
        previous_pair = self._last_pair
        if (
            previous_pair is not None
            and delta is not None
            and delta <= ALTITUDE_ALIGNMENT_SAMPLE_GAP_MS
            and self._home_correction_state != HOME_CORRECTION_PENDING
            and abs(candidate-previous_pair["candidate"])
                > ALTITUDE_ALIGNMENT_DISCONTINUITY_M+1e-6
        ):
            # A continuous FC clock with a discontinuous vertical datum is a
            # local-frame/Home-relative reset, not ordinary barometer noise.
            # Start a new immutable altitude epoch and seed it with the pair
            # that exposed the reset; never silently rebase an active mission.
            self.begin_epoch("altitude_reference_discontinuity")
            self._last_local_boot_ms = local[0]
            self._local_history.append(local)
            self._last_global_boot_ms = boot_ms
            self._global_unwrapped_ms = 0
        self.sequence = (self.sequence + 1) % _BOOT_TIME_MODULUS
        self._last_pair = {
            "sequence": self.sequence,
            "local_time_boot_ms": local[0],
            "global_time_boot_ms": boot_ms,
            "local_received_ns": local[2],
            "global_received_ns": receipt,
            "source_skew_ms": int(source_skew),
            "local_z_ned_m": local[1],
            "fc_altitude_home_relative_m": fc_altitude,
            "normalized_fc_altitude_home_relative_m": normalized_fc_altitude,
            "global_altitude_amsl_m": global_altitude,
            "candidate": candidate,
        }
        self._window.append((self._global_unwrapped_ms, candidate))
        cutoff_ms = self._window[-1][0]-ALTITUDE_ALIGNMENT_WINDOW_MS
        # Prove that samples cover the complete two-second interval before
        # discarding values older than its exact lower bound.  With a normal
        # 10 Hz stream there is rarely a sample exactly on that boundary:
        # millisecond timestamp quantisation otherwise makes the retained
        # endpoint duration alternate between 2000 and 1900/1999 ms, causing
        # READY to chatter back to ALIGNING despite continuous fresh data.
        # The discarded boundary predecessor is used only as coverage proof;
        # candidate stability below is still calculated from the exact recent
        # window, so an old outlier cannot keep the datum unstable.
        # Retain at most one predecessor below the lower bound.  It proves
        # continuous coverage up to the first in-window sample, but snapshot()
        # excludes it from the candidate span and sample count.
        while (len(self._window) >= 2
               and self._window[1][0] <= cutoff_ms):
            self._window.popleft()
        self._window_has_full_coverage = bool(
            self._window and self._window[0][0] <= cutoff_ms)
        self._forced_detail = "collecting_synchronized_altitude_samples"
        return self.snapshot(now_ns)

    def snapshot(self, now_ns=None):
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        self._correction_deadline_expired(now_ns)
        pending_age_ms = (
            max(0, (now_ns-self._home_correction_pending_since_ns)//1_000_000)
            if self._home_correction_pending_since_ns > 0 else 0
        )
        if (self._home_correction_state == HOME_CORRECTION_PENDING
                and pending_age_ms > self.confirmation_ms):
            self.reject_home_correction(
                "altitude_home_correction_unconfirmed")
            pending_age_ms = 0
        pair = self._last_pair
        if pair is None:
            return AltitudeReferenceObservation(
                ALTITUDE_ALIGNMENT_ALIGNING, False, False,
                self.transport_epoch, self.sequence, 0, 0,
                2**32-1, 2**32-1, 2**32-1,
                math.nan, math.nan, math.nan, math.nan, math.nan,
                len(self._window), 0, self._forced_detail,
                normalized_fc_altitude_home_relative_m=math.nan,
                global_altitude_amsl_m=self._last_global_altitude_amsl_m,
                home_correction_state=self._home_correction_state,
                home_correction_valid=self._home_correction_valid,
                home_correction_revision=self._home_correction_revision,
                home_correction_pending_age_ms=int(pending_age_ms),
                frozen_px4_home_altitude_amsl_m=(
                    self._frozen_home_altitude_amsl_m),
                current_px4_home_altitude_amsl_m=(
                    self._current_home_altitude_amsl_m),
                frozen_px4_home_z_ned_m=self._frozen_home_z_ned_m,
                current_px4_home_z_ned_m=self._current_home_z_ned_m,
                home_altitude_correction_m=self._home_correction_m,
                home_z_correction_m=self._home_z_correction_m,
                home_correction_opposition_error_m=(
                    self._home_correction_opposition_error_m),
                frozen_home_crosscheck_error_m=self._last_crosscheck_error_m,
                estimator_reset_counter_valid=(
                    self._estimator_reset_counter_valid),
                estimator_reset_counter=self._estimator_reset_counter,
                home_correction_detail=self._home_correction_detail,
                home_phase=self._home_phase,
                execution_home_lock_valid=self._execution_home_lock_valid,
                execution_home_mission_id=self._execution_home_mission_id,
                execution_home_lock_revision=(
                    self._execution_home_lock_revision),
                provisional_home_revision=self._provisional_home_revision,
                provisional_px4_home_altitude_amsl_m=(
                    self._provisional_home_altitude_amsl_m),
                provisional_px4_home_z_ned_m=(
                    self._provisional_home_z_ned_m),
                altitude_epoch_failure_latched=self._epoch_failure_latched,
            )
        local_age = max(0, (now_ns-pair["local_received_ns"])//1_000_000)
        global_age = max(0, (now_ns-pair["global_received_ns"])//1_000_000)
        cutoff_ms = self._window[-1][0]-ALTITUDE_ALIGNMENT_WINDOW_MS
        recent_window = [
            item for item in self._window if item[0] >= cutoff_ms]
        retained_duration = (recent_window[-1][0]-recent_window[0][0]
                             if len(recent_window) >= 2 else 0)
        duration = (ALTITUDE_ALIGNMENT_WINDOW_MS
                    if self._window_has_full_coverage
                    else retained_duration)
        candidates = [item[1] for item in recent_window]
        span = max(candidates)-min(candidates) if candidates else math.inf
        fresh = bool(
            local_age <= ALTITUDE_ALIGNMENT_SOURCE_FRESHNESS_MS
            and global_age <= ALTITUDE_ALIGNMENT_SOURCE_FRESHNESS_MS
            and pair["source_skew_ms"] <= ALTITUDE_ALIGNMENT_SOURCE_SKEW_MS
        )
        enough = bool(
            len(candidates) >= ALTITUDE_ALIGNMENT_MIN_SAMPLES
            and self._window_has_full_coverage
        )
        home_evidence_ready = bool(
            not math.isfinite(self._frozen_home_altitude_amsl_m)
            or self._home_correction_valid
            or self._home_correction_state == HOME_CORRECTION_PENDING)
        stable = bool(
            fresh and enough and home_evidence_ready
            and span <= ALTITUDE_ALIGNMENT_MAX_SPAN_M+1e-6)
        if not fresh:
            state, detail = ALTITUDE_ALIGNMENT_STALE, "altitude_reference_state_stale"
        elif not enough:
            state, detail = ALTITUDE_ALIGNMENT_ALIGNING, "altitude_reference_aligning"
        elif not home_evidence_ready:
            state, detail = (
                ALTITUDE_ALIGNMENT_UNSTABLE,
                "home_correction_reset_counter_missing")
        elif not stable:
            state, detail = ALTITUDE_ALIGNMENT_UNSTABLE, "altitude_reference_unstable"
        else:
            state, detail = ALTITUDE_ALIGNMENT_READY, "ready"
        if (fresh and stable
                and self._home_correction_state == HOME_CORRECTION_PENDING):
            state = ALTITUDE_ALIGNMENT_HOME_CORRECTION_PENDING
            detail = "altitude_home_correction_pending"
        stable_home = statistics.median(candidates) if stable else math.nan
        return AltitudeReferenceObservation(
            state, fresh, stable, self.transport_epoch, pair["sequence"],
            pair["local_time_boot_ms"], pair["global_time_boot_ms"],
            min(int(local_age), 2**32-1), min(int(global_age), 2**32-1),
            pair["source_skew_ms"], pair["local_z_ned_m"],
            pair["fc_altitude_home_relative_m"], pair["candidate"],
            stable_home, span, len(candidates), int(duration), detail,
            normalized_fc_altitude_home_relative_m=pair.get(
                "normalized_fc_altitude_home_relative_m", math.nan),
            global_altitude_amsl_m=pair.get(
                "global_altitude_amsl_m", math.nan),
            home_correction_state=self._home_correction_state,
            home_correction_valid=self._home_correction_valid,
            home_correction_revision=self._home_correction_revision,
            home_correction_pending_age_ms=int(pending_age_ms),
            frozen_px4_home_altitude_amsl_m=(
                self._frozen_home_altitude_amsl_m),
            current_px4_home_altitude_amsl_m=(
                self._current_home_altitude_amsl_m),
            frozen_px4_home_z_ned_m=self._frozen_home_z_ned_m,
            current_px4_home_z_ned_m=self._current_home_z_ned_m,
            home_altitude_correction_m=self._home_correction_m,
            home_z_correction_m=self._home_z_correction_m,
            home_correction_opposition_error_m=(
                self._home_correction_opposition_error_m),
            frozen_home_crosscheck_error_m=self._last_crosscheck_error_m,
            estimator_reset_counter_valid=(
                self._estimator_reset_counter_valid),
            estimator_reset_counter=self._estimator_reset_counter,
            home_correction_detail=self._home_correction_detail,
            home_phase=self._home_phase,
            execution_home_lock_valid=self._execution_home_lock_valid,
            execution_home_mission_id=self._execution_home_mission_id,
            execution_home_lock_revision=self._execution_home_lock_revision,
            provisional_home_revision=self._provisional_home_revision,
            provisional_px4_home_altitude_amsl_m=(
                self._provisional_home_altitude_amsl_m),
            provisional_px4_home_z_ned_m=(
                self._provisional_home_z_ned_m),
            altitude_epoch_failure_latched=self._epoch_failure_latched,
        )


def aligned_home_z_ned(local_z_ned, fc_altitude_home_relative_m):
    """Capture one immutable local-NED Z datum aligned to PX4 relative_alt."""
    try:
        local_z = float(local_z_ned)
        relative_altitude = float(fc_altitude_home_relative_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite local Z and FC Home-relative altitude are required") from exc
    if not all(math.isfinite(value) for value in (local_z, relative_altitude)):
        raise ValueError("finite local Z and FC Home-relative altitude are required")
    return local_z + relative_altitude


def altitude_reference_error_m(*, aligned_home_z, local_z_ned,
                               fc_altitude_home_relative_m):
    """Return disagreement between a frozen NED datum and FC relative_alt."""
    try:
        values = tuple(float(value) for value in (
            aligned_home_z, local_z_ned, fc_altitude_home_relative_m))
    except (TypeError, ValueError, OverflowError):
        return math.inf
    if not all(math.isfinite(value) for value in values):
        return math.inf
    home_z, local_z, fc_altitude = values
    return abs((home_z-local_z)-fc_altitude)


def altitude_reference_sample_error(*, aligned_home_z, local_z_ned,
                                    fc_altitude_home_relative_m,
                                    local_age_s, geo_age_s, sample_skew_s,
                                    max_error_m=ALTITUDE_REFERENCE_MAX_ERROR_M):
    """Validate freshness/skew and then the frozen low-speed altitude datum."""
    try:
        local_age = float(local_age_s)
        geo_age = float(geo_age_s)
        skew = float(sample_skew_s)
        maximum = float(max_error_m)
    except (TypeError, ValueError, OverflowError):
        return "altitude_reference_stale", math.inf
    if (not all(math.isfinite(value) for value in (
            local_age, geo_age, skew, maximum))
            or not 0.0 <= local_age <= ALTITUDE_REFERENCE_FRESHNESS_S
            or not 0.0 <= geo_age <= ALTITUDE_REFERENCE_FRESHNESS_S
            or not 0.0 <= skew <= ALTITUDE_REFERENCE_MAX_SAMPLE_SKEW_S
            or not 0.0 < maximum <= ALTITUDE_REFERENCE_MAX_ERROR_M):
        return "altitude_reference_stale", math.inf
    error = altitude_reference_error_m(
        aligned_home_z=aligned_home_z,
        local_z_ned=local_z_ned,
        fc_altitude_home_relative_m=fc_altitude_home_relative_m,
    )
    if error > maximum + 1e-6:
        return "altitude_reference_mismatch", error
    return "", error


def px4_land_speed_is_valid(value):
    """Accept the configured PX4 LAND speed with MAVLink float tolerance."""
    try:
        speed = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(speed) and speed > 0.0
            and speed <= PX4_LAND_SPEED_MAX_M_S + PX4_PARAMETER_FLOAT_TOLERANCE)


def px4_offboard_loss_action_is_land(value):
    """Match PX4's COM_OBL_RC_ACT Land enum without accepting nearby values."""
    try:
        action = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return (math.isfinite(action)
            and abs(action-PX4_COM_OBL_RC_ACT_LAND) <= PX4_PARAMETER_FLOAT_TOLERANCE)


def profile_for_plan(plan):
    return plan.get("flight_profile", NORMAL)


def is_low_speed_profile(profile):
    return profile in LOW_SPEED_PROFILE_LIMITS


def profile_for_target_altitude(value):
    try:
        altitude = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("low-speed target must be exactly 1 or 2 m Home-relative") from exc
    if abs(altitude-1.0) <= 1e-6:
        return LOW_SPEED_1M_V1
    if abs(altitude-2.0) <= 1e-6:
        return LOW_SPEED_2M_V1
    raise ValueError("low-speed target must be exactly 1 or 2 m Home-relative")


def target_altitude_for_profile(profile):
    try:
        return LOW_SPEED_PROFILE_LIMITS[profile][0]
    except KeyError as exc:
        raise ValueError("unsupported low-speed flight_profile") from exc


def max_altitude_for_profile(profile):
    try:
        return LOW_SPEED_PROFILE_LIMITS[profile][1]
    except KeyError as exc:
        raise ValueError("unsupported low-speed flight_profile") from exc


def mission_kind_for_plan(plan):
    return plan.get("mission_kind", ROUTE)


def obstacle_guard_required_for_mission(
        mission_kind, forward_test_camera_required=True):
    """Limit the camera-off exception to deterministic forward tests."""
    return (bool(forward_test_camera_required)
            or mission_kind != FORWARD_TEST_1M)


def forward_test_target_within_camera_bypass(anchor_xy, target_xy):
    """Keep a camera-bypassed target inside the fixed forward-test corridor."""
    try:
        anchor = tuple(float(value) for value in anchor_xy)
        target = tuple(float(value) for value in target_xy)
    except (TypeError, ValueError, OverflowError):
        return False
    if (len(anchor) != 2 or len(target) != 2
            or not all(math.isfinite(value) for value in (*anchor, *target))):
        return False
    return math.hypot(target[0]-anchor[0], target[1]-anchor[1]) \
        <= FORWARD_TEST_CAMERA_BYPASS_MAX_TARGET_M + 1e-6


def low_speed_envelope_profiles_match(command_profile, envelope_profile):
    """Require both sides of a low-speed setpoint to name one exact profile."""
    if command_profile is None and envelope_profile is None:
        return True
    return (is_low_speed_profile(command_profile)
            and command_profile == envelope_profile)


def setpoint_contract_matches(*, setpoint_kind, mode, position, velocity, yaw,
                              expected_position, expected_velocity,
                              expected_yaw, tolerance=SETPOINT_CONTRACT_TOLERANCE):
    """Compare a ROS setpoint with its envelope, including every NaN mask.

    NaN is an instruction in PX4 setpoints, not merely an absent value.  The
    contract therefore requires both sides to use NaN in exactly the same
    components and compares finite values within a small serialization margin.
    """
    try:
        kind = int(setpoint_kind)
        actual_position = tuple(float(value) for value in position)
        actual_velocity = tuple(float(value) for value in velocity)
        contract_position = tuple(float(value) for value in expected_position)
        contract_velocity = tuple(float(value) for value in expected_velocity)
        actual_yaw = float(yaw)
        contract_yaw = float(expected_yaw)
        limit = float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return False, "invalid_setpoint_contract"
    if (not isinstance(mode, dict) or len(actual_position) != 3
            or len(actual_velocity) != 3 or len(contract_position) != 3
            or len(contract_velocity) != 3 or not math.isfinite(limit)
            or limit < 0.0):
        return False, "invalid_setpoint_contract"
    position_kind = kind == 1
    velocity_kind = kind == 2
    if not (position_kind or velocity_kind):
        return False, "unsupported_setpoint_kind"
    if position_kind != (mode.get("position") is True):
        return False, "setpoint_kind_mode_mismatch"
    if velocity_kind != (mode.get("velocity") is True):
        return False, "setpoint_kind_mode_mismatch"
    if position_kind and mode.get("velocity") is not False:
        return False, "setpoint_kind_mode_mismatch"
    if velocity_kind and mode.get("position") is not False:
        return False, "setpoint_kind_mode_mismatch"

    def same_number(left, right):
        left_nan, right_nan = math.isnan(left), math.isnan(right)
        if left_nan or right_nan:
            return left_nan and right_nan
        return math.isfinite(left) and math.isfinite(right) \
            and abs(left-right) <= limit

    if not all(same_number(left, right) for left, right in zip(
            actual_position, contract_position)):
        return False, "position_contract_mismatch"
    if not all(same_number(left, right) for left, right in zip(
            actual_velocity, contract_velocity)):
        return False, "velocity_contract_mismatch"
    if not same_number(actual_yaw, contract_yaw):
        return False, "yaw_contract_mismatch"
    return True, ""


def takeoff_handshake_action(*, output_ready, offboard, armed,
                             mode_timed_out=False, arm_timed_out=False):
    """Return the only command allowed at the current takeoff handshake step."""
    if not output_ready or (offboard and armed):
        return "NONE"
    if not offboard:
        return "NONE" if mode_timed_out else "MODE"
    if not armed:
        return "NONE" if arm_timed_out else "ARM"
    return "NONE"


def flight_output_confirmation_ready(*, receipt_age_s, continuous_age_s,
                                     setpoint_age_ms, consecutive_transmissions,
                                     exact_match, transport_connected,
                                     command_graph_ready, envelope_valid,
                                     setpoint_transmitted, warmup_s):
    """Validate the short-lived bridge proof used before requesting Offboard."""
    try:
        receipt_age = float(receipt_age_s)
        continuous_age = float(continuous_age_s)
        setpoint_age = int(setpoint_age_ms)
        consecutive = int(consecutive_transmissions)
        warmup = float(warmup_s)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        exact_match and transport_connected and command_graph_ready
        and envelope_valid and setpoint_transmitted
        and 0.0 <= receipt_age <= 0.15
        and 0.0 <= setpoint_age <= 150
        and consecutive >= 2
        and warmup >= 0.0 and continuous_age >= warmup
    )


def capture_then_home_for_profile(active_flight_profile=NORMAL,
                                  event_completion_policy="legacy_rejoin"):
    """Keep legacy capture/Home behavior out of every low-speed mission."""
    return (not is_low_speed_profile(active_flight_profile)
            and event_completion_policy == "capture_then_rtl")


def geo_is_fresh(valid, age_ms):
    """Use the same 750 ms flight-readiness lease in ROS and the dashboard."""
    try:
        age = int(age_ms)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(valid) and 0 <= age <= GEO_FRESHNESS_MS


def low_speed_readiness_error(*, state, geo, safety, now,
                              vehicle_state_received_at, active_mission,
                              allow_active, require_ground, physical,
                              home_fresh, require_obstacle_guard=True):
    """Return the first fail-closed reason for a low-speed mission."""
    if active_mission and not allow_active:
        return "another mission is active"
    if state is None or not 0 <= now-vehicle_state_received_at <= 0.5:
        return "vehicle state is missing or stale"
    for name, detail in (("command_output_enabled", "command output disabled"),
                         ("connected", "FC disconnected"),
                         ("vehicle_status_fresh", "PX4 vehicle status is stale"),
                         ("arming_state_valid", "PX4 arming state is unknown"),
                         ("landed_state_valid", "PX4 landed state is unknown"),
                         ("preflight_checks_pass", "preflight checks failed"),
                         ("position_valid", "local position invalid"),
                         ("heading_fresh", "heading invalid or stale")):
        if not bool(getattr(state, name, False)):
            return detail
    if bool(getattr(state, "manual_override", False)):
        return "manual control is active"
    if require_ground and (bool(getattr(state, "armed", False))
                           or not bool(getattr(state, "landed", False))):
        return "low-speed mission must start disarmed and landed"
    if physical and require_obstacle_guard:
        if not bool(getattr(state, "low_speed_obstacle_guard_enabled", False)):
            return "low-speed obstacle/safety guard is not enabled"
        if not bool(getattr(state, "jetson_safety_fresh", False)):
            return "low-speed obstacle/safety input is missing or stale"
        if str(getattr(state, "jetson_safety_state", "")).upper() != "CLEAR":
            return "low-speed obstacle/safety input is not CLEAR"
    if physical:
        if geo is None or not geo_is_fresh(getattr(geo, "valid", False),
                                           getattr(geo, "age_ms", None)):
            return "global/Home-relative position is missing or stale"
        try:
            safety_age_ms = int(getattr(safety, "age_ms", 2**32-1))
        except (TypeError, ValueError, OverflowError):
            safety_age_ms = 2**32-1
        if (safety is None or not bool(getattr(safety, "valid", False))
                or not 0 <= safety_age_ms <= 10_000):
            return str(getattr(
                safety, "detail", "PX4 low-speed failsafe parameters unavailable"))
        if not home_fresh:
            return "fresh validated PX4 Home is required"
    return ""


def force_low_speed_route(plan, target_altitude_m=TARGET_ALTITUDE_M):
    """Return an approval snapshot without mutating the stored route plan."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    points = plan.get("route_waypoints_enu")
    if not isinstance(points, list) or not 2 <= len(points) <= 300:
        raise ValueError("low-speed mode requires 2 to 300 resolved route_waypoints_enu")
    profile = profile_for_target_altitude(target_altitude_m)
    target_altitude = target_altitude_for_profile(profile)
    output = deepcopy(plan)
    converted = []
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) not in (3, 4):
            raise ValueError("route waypoint %d must contain east, north, up and optional yaw" % index)
        if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in point):
            raise ValueError("route waypoint values must be finite numbers")
        converted.append([float(point[0]), float(point[1]), target_altitude, *[float(v) for v in point[3:]]])
    output.update({
        "mission_kind": ROUTE,
        "flight_profile": profile,
        "route_waypoints_enu": converted,
        "low_speed_limits": fixed_limits(profile),
        "completion_policy": "LAND_AT_FINAL_WAYPOINT",
        "after_response": "LAND_AT_FINAL_WAYPOINT",
    })
    return output


def fixed_limits(profile=LOW_SPEED_2M_V1):
    target_altitude = target_altitude_for_profile(profile)
    max_altitude = max_altitude_for_profile(profile)
    return {
        "target_altitude_home_m": target_altitude,
        "max_altitude_home_m": max_altitude,
        "max_horizontal_speed_m_s": MAX_HORIZONTAL_SPEED_M_S,
        "max_vertical_speed_m_s": MAX_VERTICAL_SPEED_M_S,
        "acceptance_radius_m": ACCEPTANCE_RADIUS_M,
    }


def forward_endpoint_ned(start_ned, yaw_rad):
    values = tuple(float(value) for value in start_ned)
    yaw = float(yaw_rad)
    if len(values) != 3 or not all(math.isfinite(v) for v in (*values, yaw)):
        raise ValueError("fresh finite local position and heading are required")
    return (
        values[0] + FORWARD_TEST_DISTANCE_M*math.cos(yaw),
        values[1] + FORWARD_TEST_DISTANCE_M*math.sin(yaw),
        values[2],
    )


def angle_distance(a, b):
    return abs(math.atan2(math.sin(float(a)-float(b)), math.cos(float(a)-float(b))))


def validate_low_speed_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("plan must be an object")
    profile = profile_for_plan(plan)
    kind = mission_kind_for_plan(plan)
    if profile != NORMAL and not is_low_speed_profile(profile):
        raise ValueError("unsupported flight_profile")
    if kind not in (ROUTE, FORWARD_TEST_1M):
        raise ValueError("unsupported mission_kind")
    if profile == NORMAL:
        if kind != ROUTE:
            raise ValueError("FORWARD_TEST_1M requires a low-speed profile")
        return
    if "test_scenario" in plan:
        from .scenario_contract import validate_spec
        validate_spec(plan)
    target_altitude = target_altitude_for_profile(profile)
    if plan.get("low_speed_limits") != fixed_limits(profile):
        raise ValueError("low-speed limits do not match server constants")
    if plan.get("completion_policy") != "LAND_AT_FINAL_WAYPOINT":
        raise ValueError("low-speed missions must land at the final waypoint")
    if kind == ROUTE:
        points = plan.get("route_waypoints_enu")
        if not isinstance(points, list) or not 2 <= len(points) <= 300:
            raise ValueError("low-speed route requires resolved route_waypoints_enu")
        if any(not isinstance(p, list) or len(p) not in (3, 4)
               or p[2] != target_altitude for p in points):
            raise ValueError(
                "every low-speed route waypoint must match the selected Home-relative target")
    else:
        required = ("preview_token", "preview_start_ned_m", "preview_end_ned_m", "preview_heading_rad")
        if any(key not in plan for key in required):
            raise ValueError("forward-test preview is incomplete")


def preview_valid(preview, position_ned, heading_rad, now=None):
    now = time.monotonic() if now is None else float(now)
    if now < preview["created_at"] or now-preview["created_at"] > PREVIEW_TTL_S:
        return False, "forward-test preview expired"
    # The preview freezes the horizontal start and heading.  Preparation and
    # execution both require landed=true, so a change in local NED Z while on
    # the ground is estimator/barometer drift rather than vehicle movement.
    # Home-relative altitude remains independently validated and capped.
    distance = math.hypot(
        float(position_ned[0])-float(preview["start_ned"][0]),
        float(position_ned[1])-float(preview["start_ned"][1]),
    )
    if distance > PREVIEW_POSITION_DRIFT_M:
        return False, "vehicle moved more than 0.2 m after preview"
    if angle_distance(heading_rad, preview["heading_rad"]) > PREVIEW_HEADING_DRIFT_RAD:
        return False, "heading changed more than 5 degrees after preview"
    return True, ""


def low_speed_pose_coherence_reason(*, position_source,
                                    position_received_age_s,
                                    position_time_boot_ms,
                                    position_transport_epoch,
                                    altitude_local_time_boot_ms,
                                    altitude_transport_epoch):
    """Require low-speed current pose and altitude to share one FC epoch."""
    if position_source != "LOCAL_POSITION_NED":
        return "altitude_pose_source_mismatch"
    try:
        age_s = float(position_received_age_s)
        position_epoch = int(position_transport_epoch)
        altitude_epoch = int(altitude_transport_epoch)
    except (TypeError, ValueError, OverflowError):
        return "altitude_pose_time_skew"
    if (not math.isfinite(age_s) or not 0.0 <= age_s <= 0.5
            or position_epoch <= 0 or position_epoch != altitude_epoch
            or boot_time_distance_ms(
                position_time_boot_ms,
                altitude_local_time_boot_ms,
            ) > LOW_SPEED_ALTITUDE_POSE_SKEW_MS):
        return "altitude_pose_time_skew"
    return ""


def envelope_allows(*, profile, velocity, target_position,
                    current_local_z_ned, home_z_ned, envelope_age_s,
                    fc_altitude_home_relative_m=None,
                    validate_command_target=True, maximum_altitude_override=None):
    """Validate independent command-target, local-pose and FC altitude caps."""
    if not is_low_speed_profile(profile):
        return True, ""
    if not math.isfinite(envelope_age_s) or not 0 <= envelope_age_s <= 0.25:
        return False, "low-speed flight envelope missing or stale"
    values = tuple(float(v) for v in (
        *velocity, *target_position, current_local_z_ned, home_z_ned))
    if not all(math.isfinite(v) for v in values):
        return False, "low-speed setpoint is not finite"
    if math.hypot(velocity[0], velocity[1]) > MAX_HORIZONTAL_SPEED_M_S + 1e-6:
        return False, "horizontal speed exceeds low-speed envelope"
    if abs(velocity[2]) > MAX_VERTICAL_SPEED_M_S + 1e-6:
        return False, "vertical speed exceeds low-speed envelope"
    maximum_altitude = max_altitude_for_profile(profile)
    if maximum_altitude_override is not None:
        # Only the Bridge's validated v3 scenario supplies this override.
        # Ordinary profiles still take the unchanged default ceiling above.
        if maximum_altitude_override != maximum_altitude+2.0:
            return False, "invalid scenario altitude extension"
        maximum_altitude = maximum_altitude_override
    if validate_command_target:
        target_altitude = float(home_z_ned)-float(target_position[2])
        if target_altitude > maximum_altitude + 1e-6:
            return False, "command_target_altitude_limit_exceeded"
    local_altitude = float(home_z_ned)-float(current_local_z_ned)
    if local_altitude > maximum_altitude + 1e-6:
        return False, "local_home_altitude_limit_exceeded"
    if fc_altitude_home_relative_m is not None:
        try:
            fc_altitude = float(fc_altitude_home_relative_m)
        except (TypeError, ValueError, OverflowError):
            return False, "altitude_reference_stale"
        if not math.isfinite(fc_altitude):
            return False, "altitude_reference_stale"
        if fc_altitude > maximum_altitude + 1e-6:
            return False, "fc_home_altitude_limit_exceeded"
    return True, ""
