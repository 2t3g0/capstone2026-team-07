export interface RoadSegment {
  id: string;
  osm_way_id: number;
  name: string;
  category: 'vehicle' | 'pedestrian';
  highway: string;
  geometry: [number, number][];
  waypoints_enu: [number, number][];
}

export interface RouteWaypoint {
  latitude_deg: number;
  longitude_deg: number;
  altitude_m: number;
  yaw_deg?: number | null;
}

export interface StoredRoute {
  id: string;
  revision: number;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  waypoints: RouteWaypoint[];
  waypoints_enu: number[][];
}

export interface RouteCatalog {
  schema_version: string;
  site_id: string;
  simulation_only: boolean;
  generated_from_osm_at: string;
  source: string;
  license: string;
  reference: {
    latitude_deg: number;
    longitude_deg: number;
    elevation_m: number;
  };
  geofence_geo: [number, number][];
  segments: RoadSegment[];
  custom_routes: StoredRoute[];
}

export const EARTH_RADIUS_M = 6_378_137;
const MAX_PLAN_WAYPOINTS = 300;

export interface PlanRouteWaypoint {
  east_m: number;
  north_m: number;
  up_m: number;
  yaw_deg?: number;
  latitude_deg: number;
  longitude_deg: number;
}

export interface PlanCaptureAction {
  waypoint_index: number;
  landmark_id?: string;
}

export function planRouteToEditableWaypoints(points: PlanRouteWaypoint[]): RouteWaypoint[] {
  return points.map((point) => ({
    latitude_deg: point.latitude_deg,
    longitude_deg: point.longitude_deg,
    altitude_m: point.up_m,
    ...(point.yaw_deg === undefined ? {} : { yaw_deg: point.yaw_deg }),
  }));
}

/** Converts a complete, valid ENU route. A malformed waypoint rejects the whole route. */
export function planRouteEnuToGeo(
  value: unknown,
  reference: RouteCatalog['reference'],
): PlanRouteWaypoint[] | null {
  if (!Array.isArray(value) || value.length > MAX_PLAN_WAYPOINTS) return null;
  const { latitude_deg: latitude, longitude_deg: longitude } = reference;
  if (
    !Number.isFinite(latitude)
    || !Number.isFinite(longitude)
    || latitude < -90
    || latitude > 90
    || longitude < -180
    || longitude > 180
  ) return null;

  const latitudeRad = (latitude * Math.PI) / 180;
  const longitudeRadius = EARTH_RADIUS_M * Math.cos(latitudeRad);
  if (Math.abs(longitudeRadius) < 1) return null;

  const converted: PlanRouteWaypoint[] = [];
  for (const waypoint of value) {
    if (!Array.isArray(waypoint) || (waypoint.length !== 3 && waypoint.length !== 4)) return null;
    if (!waypoint.every((coordinate) => typeof coordinate === 'number' && Number.isFinite(coordinate))) return null;
    const [east_m, north_m, up_m, yaw_deg] = waypoint as [number, number, number, number?];
    const convertedLatitude = latitude + ((north_m / EARTH_RADIUS_M) * 180) / Math.PI;
    const convertedLongitude = longitude + ((east_m / longitudeRadius) * 180) / Math.PI;
    if (
      !Number.isFinite(convertedLatitude)
      || !Number.isFinite(convertedLongitude)
      || convertedLatitude < -90
      || convertedLatitude > 90
      || convertedLongitude < -180
      || convertedLongitude > 180
    ) return null;
    converted.push({
      east_m,
      north_m,
      up_m,
      ...(yaw_deg === undefined ? {} : { yaw_deg }),
      latitude_deg: convertedLatitude,
      longitude_deg: convertedLongitude,
    });
  }
  return converted;
}

export function planCaptureActions(value: unknown, waypointCount: number): PlanCaptureAction[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((candidate): PlanCaptureAction[] => {
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) return [];
    const action = candidate as Record<string, unknown>;
    const waypointIndex = action.waypoint_index;
    if (
      action.action !== 'CAPTURE_STILL'
      || typeof waypointIndex !== 'number'
      || !Number.isInteger(waypointIndex)
      || waypointIndex < 0
      || waypointIndex >= waypointCount
    ) return [];
    return [{
      waypoint_index: waypointIndex,
      ...(typeof action.landmark_id === 'string' && action.landmark_id.trim()
        ? { landmark_id: action.landmark_id.trim() }
        : {}),
    }];
  });
}

function distanceSquared(a: RouteWaypoint, b: RouteWaypoint): number {
  const latitudeScale = 111_320;
  const longitudeScale = latitudeScale * Math.cos((a.latitude_deg * Math.PI) / 180);
  const north = (a.latitude_deg - b.latitude_deg) * latitudeScale;
  const east = (a.longitude_deg - b.longitude_deg) * longitudeScale;
  return north * north + east * east;
}

export function appendRoadGeometry(
  current: RouteWaypoint[],
  geometry: [number, number][],
  altitudeM: number,
  maximum = 300,
): RouteWaypoint[] {
  if (geometry.length < 2) return current;
  const incoming = geometry.map(([latitude_deg, longitude_deg]) => ({
    latitude_deg,
    longitude_deg,
    altitude_m: altitudeM,
  }));
  if (current.length) {
    const tail = current[current.length - 1];
    if (distanceSquared(tail, incoming[incoming.length - 1]) < distanceSquared(tail, incoming[0])) {
      incoming.reverse();
    }
    if (distanceSquared(tail, incoming[0]) < 1) incoming.shift();
  }
  const output = [...current];
  for (const waypoint of incoming) {
    const previous = output[output.length - 1];
    if (!previous || distanceSquared(previous, waypoint) >= 0.25) output.push(waypoint);
    if (output.length >= maximum) break;
  }
  return output;
}

export function appendFreeWaypoint(
  current: RouteWaypoint[],
  waypoint: RouteWaypoint,
  maximum = 300,
): RouteWaypoint[] {
  if (current.length >= maximum) return current;
  const previous = current[current.length - 1];
  if (previous && distanceSquared(previous, waypoint) < 0.25) return current;
  return [...current, waypoint];
}

export function routeToMissionPlan(
  route: StoredRoute,
  laps = 1,
  lowSpeedAltitude: LowSpeedAltitude | null = null,
): object {
  const plan = {
    request_purpose: 'EXECUTE_MISSION',
    status: 'OK',
    // Registered routes carry their own immutable waypoint snapshot. Combining
    // that with a legacy patrol zone is ambiguous and rejected by ROS.
    patrol_zones: [],
    patrol_limit: { type: 'LAPS', value: Math.max(1, Math.min(100, Math.trunc(laps))) },
    monitor_events: ['ALL'],
    response_rules: { DEFAULT: ['RECORD', 'ALERT'] },
    after_response: 'RETURN_HOME',
    missing_fields: [],
    unsupported_values: [],
    message: `등록 경로: ${route.name}`,
    route_id: route.id,
    route_revision: route.revision,
    route_name: route.name,
    route_waypoints_enu: route.waypoints_enu,
  };
  if (lowSpeedAltitude === null) return { ...plan, mission_kind: 'ROUTE', flight_profile: 'NORMAL' };
  return {
    ...plan,
    mission_kind: 'ROUTE',
    flight_profile: profileForAltitude(lowSpeedAltitude),
    route_waypoints_enu: route.waypoints_enu.map((point) => [
      point[0], point[1], lowSpeedAltitude, ...point.slice(3),
    ]),
    low_speed_limits: limitsForAltitude(lowSpeedAltitude),
    completion_policy: 'LAND_AT_FINAL_WAYPOINT',
    after_response: 'LAND_AT_FINAL_WAYPOINT',
  };
}
import { limitsForAltitude, profileForAltitude, type LowSpeedAltitude } from './flightProfiles';
