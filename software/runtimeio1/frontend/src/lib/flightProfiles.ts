import type { JsonObject } from './protocol';

export const LOW_SPEED_1M_PROFILE = 'LOW_SPEED_1M_V1';
export const LOW_SPEED_2M_PROFILE = 'LOW_SPEED_2M_V1';
export const LOW_SPEED_PROFILE = LOW_SPEED_2M_PROFILE;
export type LowSpeedAltitude = 1 | 2;
export const FORWARD_TEST_HOLD_MS = 3000;
const COMMON_LOW_SPEED_LIMITS = Object.freeze({
  max_horizontal_speed_m_s: 0.5,
  max_vertical_speed_m_s: 0.5,
  acceptance_radius_m: 0.25,
});

export function profileForAltitude(altitude: LowSpeedAltitude): string {
  return altitude === 1 ? LOW_SPEED_1M_PROFILE : LOW_SPEED_2M_PROFILE;
}

export function limitsForAltitude(altitude: LowSpeedAltitude) {
  return {
    target_altitude_home_m: altitude,
    max_altitude_home_m: altitude + 0.5,
    ...COMMON_LOW_SPEED_LIMITS,
  };
}

export function isLowSpeedProfile(value: unknown): boolean {
  return value === LOW_SPEED_1M_PROFILE || value === LOW_SPEED_2M_PROFILE;
}

export function lowSpeedAltitudeForPlan(plan: JsonObject | null): LowSpeedAltitude | null {
  if (plan?.flight_profile === LOW_SPEED_1M_PROFILE) return 1;
  if (plan?.flight_profile === LOW_SPEED_2M_PROFILE) return 2;
  return null;
}

export function toLowSpeedPlan(plan: JsonObject, altitude: LowSpeedAltitude = 2): JsonObject {
  const source = plan.route_waypoints_enu;
  if (!Array.isArray(source) || source.length < 2 || source.length > 300) {
    throw new Error('저속 모드는 완전히 해석된 2개 이상의 경로 좌표가 필요합니다.');
  }
  const route = source.map((value, index) => {
    if (!Array.isArray(value) || (value.length !== 3 && value.length !== 4)) {
      throw new Error(`웨이포인트 ${index + 1} 형식이 올바르지 않습니다.`);
    }
    const values = value.map(Number);
    if (!values.every(Number.isFinite)) throw new Error('경로 좌표는 유한한 숫자여야 합니다.');
    return [values[0], values[1], altitude, ...values.slice(3)];
  });
  return {
    ...structuredClone(plan),
    mission_kind: 'ROUTE',
    flight_profile: profileForAltitude(altitude),
    route_waypoints_enu: route,
    low_speed_limits: limitsForAltitude(altitude),
    completion_policy: 'LAND_AT_FINAL_WAYPOINT',
    after_response: 'LAND_AT_FINAL_WAYPOINT',
  };
}

export function isForwardTestPlan(plan: JsonObject | null): boolean {
  return plan?.mission_kind === 'FORWARD_TEST_1M'
    && isLowSpeedProfile(plan?.flight_profile);
}

export function approvalHoldState(elapsedMs: number, stillHeld: boolean) {
  if (!stillHeld || !Number.isFinite(elapsedMs) || elapsedMs < 0) {
    return { progress: 0, dispatch: false };
  }
  const progress = Math.min(1, elapsedMs / FORWARD_TEST_HOLD_MS);
  return { progress, dispatch: progress >= 1 };
}
