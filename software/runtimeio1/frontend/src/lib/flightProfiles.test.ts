import { describe, expect, it } from 'vitest';
import { approvalHoldState, toLowSpeedPlan } from './flightProfiles';

describe('toLowSpeedPlan', () => {
  it('creates a 2 m execution snapshot without mutating the stored route', () => {
    const original = { status: 'OK', route_id: 'r1', route_revision: 3,
      route_waypoints_enu: [[1, 2, 15], [3, 4, 20, 90]] };
    const result = toLowSpeedPlan(original);
    expect(result.route_waypoints_enu).toEqual([[1, 2, 2], [3, 4, 2, 90]]);
    expect(original.route_waypoints_enu).toEqual([[1, 2, 15], [3, 4, 20, 90]]);
    expect(result).toMatchObject({ flight_profile: 'LOW_SPEED_2M_V1',
      completion_policy: 'LAND_AT_FINAL_WAYPOINT' });
  });

  it('creates only the approved 1 m profile and 1.5 m ceiling', () => {
    const original = { status: 'OK', route_waypoints_enu: [[1, 2, 15], [3, 4, 20]] };
    const result = toLowSpeedPlan(original, 1);
    expect(result.route_waypoints_enu).toEqual([[1, 2, 1], [3, 4, 1]]);
    expect(result).toMatchObject({
      flight_profile: 'LOW_SPEED_1M_V1',
      low_speed_limits: { target_altitude_home_m: 1, max_altitude_home_m: 1.5 },
    });
  });
});

describe('forward-test press and hold', () => {
  it('does not dispatch when released before three seconds', () => {
    expect(approvalHoldState(2999, true).dispatch).toBe(false);
    expect(approvalHoldState(2999, false)).toEqual({ progress: 0, dispatch: false });
  });

  it('dispatches only while continuously held for three seconds', () => {
    expect(approvalHoldState(3000, true)).toEqual({ progress: 1, dispatch: true });
    expect(approvalHoldState(4000, false).dispatch).toBe(false);
  });
});
