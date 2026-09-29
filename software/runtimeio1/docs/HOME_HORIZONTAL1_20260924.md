# Home horizontal metadata candidate (2026-09-24)

Build ID: `low-speed-homehorizontal1-20260924` (protocol 13).

The two 2026-09-24 field attempts accepted PX4 Mode and ARM commands, then
observed a roughly 1 m HOME_POSITION change while the execution Home was
locked. The previous 0.20 m horizontal Home gate rejected the update; the
altitude synchronizer subsequently exceeded its 400 ms confirmation deadline,
and the controller issued LAND. The earlier ARM/Home order candidate only
addressed HOME_POSITION arriving before the armed HEARTBEAT.

This candidate permits a horizontal HOME_POSITION metadata refinement of at
most 2 m only when its WGS84 and local-NED displacements agree within 0.20 m,
the frozen route transform remains continuous, the recent local/global samples
remain continuous, and fresh odometry shows no estimator reset. The frozen
execution Home is never moved. Larger, inconsistent, stale, or reset-related
changes still fail closed. The Bridge journal now records both horizontal
displacements, their disagreement, and the frame-continuity delta.

The previous field journals record only the maximum HOME_POSITION difference;
they do not contain the raw WGS84 and local-NED pairs. Thus they establish the
failure sequence but cannot prove the actual update satisfies this new gate.
The field stack was inactive and the Jetson observer service active at the
latest read-only check; this explains the then-current dashboard UNKNOWN but
does not establish the first cause of transient UNKNOWN during flight.

This is a local validation candidate, not a completed 1 m flight verification.
Before a field attempt, inspect the new Home transition journal diagnostics and
confirm a valid horizontal refinement and uninterrupted FC status stream. If
the representations disagree, keep the rejection and inspect the PX4 ULog and
raw HOME_POSITION telemetry rather than enlarging the bounds.
