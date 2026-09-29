uint8 STATE_CLEAR=0
uint8 STATE_SLOW=1
uint8 STATE_HOLD=2
uint8 STATE_EVADE=3
uint8 STATE_STALE=4

builtin_interfaces/Time stamp
string source
uint32 sequence
# Original sensor observation age at final bridge publication (not HTTP age).
float32 observation_age_s
uint8 state
float32 confidence
float32 max_speed_mps
float32[3] velocity_ned_mps
float32 front_distance_m
float32 upper_clearance_m
float32 lower_clearance_m
float32 effective_trigger_distance_m
float32 effective_release_distance_m
bool descent_corridor_clear
# Calibrated metric depth + capture-time pose in the PX4 local frame.
# Invalid/unobserved geometry must use false flags and NaN numeric values.
bool geometry_valid
bool roof_clearance_verified
# Conservative vertical separation after airframe/uncertainty deductions.
float32 roof_vertical_gap_m
# Highest observed roof height, positive UP from the PX4 local origin.
float32 roof_height_m
bool obstacle_extent_valid
float32 obstacle_far_north_m
float32 obstacle_far_east_m
# Positive evidence that the tracked roof's far boundary has been passed;
# absence of visible roof pixels alone is not sufficient.
bool roof_passage_verified
string reason
