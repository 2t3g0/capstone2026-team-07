builtin_interfaces/Time stamp
string mission_id
bool command_output_enabled
bool approved
bool manual_override
bool connected
bool preflight_checks_pass
bool position_valid
bool armed
bool offboard
bool landed
float32[3] position_ned_m
float32[3] velocity_ned_m_s
string active_authority
string jetson_safety_state
bool jetson_safety_fresh
string last_error
string active_execution_mission_id
bool terminal_in_progress
bool emergency_land_available
string emergency_land_detail
bool arm_request_transmitted
bool prearm_terminated
uint64 terminal_started_monotonic_ns
bool frame_altitude_valid
bool altitude_reference_error_valid
string first_fault
string terminal_detail
# Capability/state for obstacle-aware event return. Older publishers do not
# establish these proofs; new consumers must fail closed when unavailable.
bool avoidance_active
bool flight_epoch_retired
float32 heading_rad
bool heading_fresh
bool route_heading_gate_enabled
bool terminal_owned_handoff_enabled
bool low_speed_obstacle_guard_enabled
# False is the fail-closed/default contract.  When true, only a command marked
# AUTHORITY_FORWARD_TEST_1M may run without a fresh CLEAR camera decision.
bool forward_test_camera_bypass_enabled
# Freshness is explicit: the legacy booleans retain the last observed value,
# but must never be interpreted when their corresponding validity is false.
bool vehicle_status_fresh
bool arming_state_valid
bool landed_state_valid

uint8 TERMINAL_NONE=0
uint8 TERMINAL_PENDING=1
uint8 TERMINAL_ACCEPTED=2
uint8 TERMINAL_FALLBACK=3
uint8 TERMINAL_FAILED=4
uint8 terminal_state

bool flight_output_ready
string flight_output_detail

# Low-speed vertical datum diagnostics.  The FC value remains Home-relative,
# never AGL.  Consumers must ignore the numeric values unless valid is true.
bool altitude_reference_valid
float32 fc_altitude_home_relative_m
float32 normalized_fc_altitude_home_relative_m
float32 frame_altitude_home_relative_m
float32 altitude_reference_error_m
string altitude_reference_detail
uint8 home_correction_state
bool home_correction_valid
uint32 home_correction_revision
float32 home_altitude_correction_m
float32 home_correction_opposition_error_m
string home_correction_detail
uint8 home_phase
bool execution_home_lock_valid
string execution_home_mission_id
uint32 execution_home_lock_revision
uint32 provisional_home_revision
float32 provisional_px4_home_altitude_amsl_m
float32 provisional_px4_home_z_ned_m
float32 current_px4_home_altitude_amsl_m
float32 current_px4_home_z_ned_m
float32 home_z_correction_m
bool altitude_epoch_failure_latched

# Pre-arm FC-time-paired altitude-reference alignment state.  These fields
# describe whether a new low-speed reference may be captured; they do not
# replace the immutable in-flight diagnostics above.
bool altitude_alignment_ready
uint8 altitude_alignment_state
uint32 altitude_alignment_epoch
uint32 altitude_alignment_sample_count
uint32 altitude_alignment_window_ms
float32 altitude_alignment_candidate_span_m
uint32 altitude_alignment_source_skew_ms
string altitude_alignment_detail
