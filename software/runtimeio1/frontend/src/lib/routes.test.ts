import { describe, expect, it } from 'vitest';
import {
  appendFreeWaypoint,
  appendRoadGeometry,
  planCaptureActions,
  planRouteEnuToGeo,
  planRouteToEditableWaypoints,
  routeToMissionPlan,
  type StoredRoute,
} from './routes';

describe('route editing helpers', () => {
  it('joins a road segment in the direction closest to the current tail', () => {
    const current = [{ latitude_deg: 35, longitude_deg: 129.001, altitude_m: 15 }];
    const result = appendRoadGeometry(current, [[35, 129], [35, 129.001]], 20);
    expect(result).toHaveLength(2);
    expect(result[1].longitude_deg).toBe(129);
    expect(result[1].altitude_m).toBe(20);
  });

  it('builds an approved-plan-compatible route snapshot', () => {
    const route: StoredRoute = {
      id: 'route-1', revision: 2, name: '중앙도로', description: '',
      created_at: '', updated_at: '', waypoints: [],
      waypoints_enu: [[1, 2, 15], [3, 4, 15]],
    };
    const plan = routeToMissionPlan(route, 3) as Record<string, unknown>;
    expect(plan.request_purpose).toBe('EXECUTE_MISSION');
    expect(plan.route_id).toBe('route-1');
    expect(plan.route_revision).toBe(2);
    expect(plan.route_name).toBe('중앙도로');
    expect(plan.route_waypoints_enu).toEqual(route.waypoints_enu);
    expect(plan.patrol_limit).toEqual({ type: 'LAPS', value: 3 });
    expect(plan.after_response).toBe('RETURN_HOME');
    expect(plan.monitor_events).toEqual(['ALL']);
    expect(plan.response_rules).toEqual({ DEFAULT: ['RECORD', 'ALERT'] });
    // A registered snapshot must not also select the unrelated legacy zone A.
    expect(plan.patrol_zones).toEqual([]);
  });

  it('defaults a prepared route mission to one lap', () => {
    const route: StoredRoute = {
      id: 'route-default', revision: 1, name: '기본 경로', description: '',
      created_at: '', updated_at: '', waypoints: [],
      waypoints_enu: [[0, 0, 15], [10, 0, 15]],
    };
    const plan = routeToMissionPlan(route) as Record<string, unknown>;
    expect(plan.patrol_limit).toEqual({ type: 'LAPS', value: 1 });
    expect(plan.after_response).toBe('RETURN_HOME');
  });

  it('creates the selected 1 m route snapshot without changing the stored route', () => {
    const route: StoredRoute = {
      id: 'route-low', revision: 1, name: '저속 경로', description: '',
      created_at: '', updated_at: '', waypoints: [],
      waypoints_enu: [[0, 0, 15], [10, 0, 15]],
    };
    const plan = routeToMissionPlan(route, 1, 1) as Record<string, unknown>;
    expect(plan.flight_profile).toBe('LOW_SPEED_1M_V1');
    expect(plan.route_waypoints_enu).toEqual([[0, 0, 1], [10, 0, 1]]);
    expect(plan.low_speed_limits).toMatchObject({
      target_altitude_home_m: 1,
      max_altitude_home_m: 1.5,
    });
    expect(route.waypoints_enu).toEqual([[0, 0, 15], [10, 0, 15]]);
  });

  it('adds exactly one free-flight point and ignores an immediate duplicate', () => {
    const start = { latitude_deg: 35, longitude_deg: 129, altitude_m: 15 };
    const once = appendFreeWaypoint([], start);
    const duplicate = appendFreeWaypoint(once, { ...start });
    const next = appendFreeWaypoint(duplicate, {
      latitude_deg: 35.0001,
      longitude_deg: 129.0001,
      altitude_m: 15,
    });
    expect(once).toHaveLength(1);
    expect(duplicate).toBe(once);
    expect(next).toHaveLength(2);
  });

  it('converts ENU plan waypoints around the catalog reference', () => {
    const reference = { latitude_deg: 35.235, longitude_deg: 129.075, elevation_m: 12 };
    const result = planRouteEnuToGeo([[0, 0, 15], [100, 200, 20, 90]], reference);
    expect(result).not.toBeNull();
    expect(result?.[0]).toMatchObject({ latitude_deg: 35.235, longitude_deg: 129.075, up_m: 15 });
    expect(result?.[1].latitude_deg).toBeCloseTo(35.23679663, 7);
    expect(result?.[1].longitude_deg).toBeCloseTo(129.07609981, 7);
    expect(result?.[1].yaw_deg).toBe(90);
  });

  it.each([
    [[0, 0]],
    [[0, 0, 15, 90, 1]],
    [[0, Number.NaN, 15]],
    [[0, 0, Number.POSITIVE_INFINITY]],
    [[Number.MAX_VALUE, 0, 15]],
    [['0', 0, 15]],
  ])('rejects an entire plan route containing malformed waypoint %j', (waypoint) => {
    const reference = { latitude_deg: 35.235, longitude_deg: 129.075, elevation_m: 12 };
    expect(planRouteEnuToGeo(waypoint, reference)).toBeNull();
  });

  it('extracts only valid CAPTURE_STILL waypoint actions', () => {
    expect(planCaptureActions([
      { waypoint_index: 1, action: 'CAPTURE_STILL', landmark_id: ' BIOLOGY ' },
      { waypoint_index: 2, action: 'ORBIT', landmark_id: 'HUMANITIES' },
      { waypoint_index: 3, action: 'CAPTURE_STILL' },
      { waypoint_index: 0.5, action: 'CAPTURE_STILL' },
    ], 3)).toEqual([{ waypoint_index: 1, landmark_id: 'BIOLOGY' }]);
  });

  it('copies a generated plan route into editable stored-route waypoints', () => {
    expect(planRouteToEditableWaypoints([
      {
        east_m: 10,
        north_m: 20,
        up_m: 25,
        yaw_deg: -45,
        latitude_deg: 35.23,
        longitude_deg: 129.08,
      },
      {
        east_m: 30,
        north_m: 40,
        up_m: 20,
        latitude_deg: 35.24,
        longitude_deg: 129.09,
      },
    ])).toEqual([
      { latitude_deg: 35.23, longitude_deg: 129.08, altitude_m: 25, yaw_deg: -45 },
      { latitude_deg: 35.24, longitude_deg: 129.09, altitude_m: 20 },
    ]);
  });
});
