# Local correlation of a received MAVLink ACK with the outstanding USB request.
# MAVLink COMMAND_ACK itself does not echo a request UUID.
string mission_id
string output_epoch
uint64 output_sequence
string request_id
uint16 command
uint8 result
uint64 received_monotonic_ns
bool transmitted
