uint8 COMMAND_HOLD=0
uint8 COMMAND_TAKEOFF=1
uint8 COMMAND_GOTO=2
uint8 COMMAND_LAND=3
uint8 COMMAND_RTL=4
uint8 COMMAND_ABORT=5

uint8 AUTHORITY_LLM_ROUTE=0
uint8 AUTHORITY_JETSON_EVENT_CAPTURE=1
# A deterministic dashboard-generated test marker.  This is still route
# authority; the distinct value lets the controller constrain a camera
# bypass to FORWARD_TEST_1M instead of weakening every low-speed route.
uint8 AUTHORITY_FORWARD_TEST_1M=2

uint8 FLIGHT_PROFILE_NORMAL=0
uint8 FLIGHT_PROFILE_LOW_SPEED_2M_V1=1
uint8 FLIGHT_PROFILE_LOW_SPEED_1M_V1=2

uint8 ALTITUDE_REFERENCE_NONE=0
uint8 ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V1=1
uint8 ALTITUDE_REFERENCE_FC_HOME_ALIGNED_V2=2

builtin_interfaces/Time stamp
string mission_id
uint8 flight_output_handshake_version
uint32 sequence
uint8 command
uint8 requested_authority
bool approved
float32[3] position_ned_m
float32 yaw_rad
float32 acceptance_radius_m
uint8 flight_profile
float32 home_z_ned_m
uint8 altitude_reference
float32 altitude_reference_max_error_m
uint32 altitude_reference_epoch
uint32 altitude_reference_sequence
uint32 altitude_reference_local_time_boot_ms
uint32 altitude_reference_global_time_boot_ms
