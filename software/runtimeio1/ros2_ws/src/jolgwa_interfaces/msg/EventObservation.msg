std_msgs/Header header

string event_id
string event_type
string state
string track_id
float32 confidence

# Normalized image coordinates in [0, 1]. The box may be repeated while
# state is ACTIVE so the Jetson can keep the gimbal on the same target.
float32 bbox_x1
float32 bbox_y1
float32 bbox_x2
float32 bbox_y2
bool target_visible
string source
string evidence_clock_id
uint64 evidence_expires_monotonic_ns
