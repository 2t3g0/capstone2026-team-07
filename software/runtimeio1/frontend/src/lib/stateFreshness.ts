export function accumulatedAgeMs(sourceAge: unknown, receivedAt: number, now: number): number {
  if (typeof sourceAge !== 'number' || !Number.isFinite(sourceAge) || sourceAge < 0
    || !Number.isFinite(receivedAt) || !Number.isFinite(now) || now < receivedAt) return Infinity;
  return sourceAge + now-receivedAt;
}

export function vehicleStateFresh(sourceAge: unknown, receivedAt: number, now: number,
  socketConnected: boolean, rosConnected: boolean, fcConnected: boolean, fcStatusFresh: boolean): boolean {
  return socketConnected && rosConnected && fcConnected && fcStatusFresh
    && accumulatedAgeMs(sourceAge, receivedAt, now) < 500;
}

// Only an idle, explicitly disarmed/landed vehicle gets the longer display and
// preview-preparation window. Execution and airborne controls use vehicleStateFresh.
export function groundPreparationStateFresh(vehicle: Record<string, unknown> | null,
  receivedAt: number, now: number, socketConnected: boolean, rosConnected: boolean): boolean {
  const idleGround = vehicle?.armed === false && vehicle?.landed === true
    && vehicle?.arming_state_valid === true && vehicle?.landed_state_valid === true
    && !vehicle?.active_execution_mission_id && vehicle?.terminal_in_progress === false;
  return socketConnected && rosConnected && vehicle?.connected === true
    && vehicle?.vehicle_status_fresh === true
    && accumulatedAgeMs(vehicle?.gateway_vehicle_age_ms, receivedAt, now) < (idleGround ? 1000 : 500);
}

// Display diagnostics only: readiness and execution still use the existing gates.
export function armingStateUnknownReason(vehicle: Record<string, unknown> | null,
  fresh: boolean, socketConnected: boolean, rosConnected: boolean): string | null {
  if (!socketConnected) return '브라우저 연결 끊김';
  if (!rosConnected) return 'ROS 연결 끊김';
  if (!vehicle) return '기체 상태 수신 대기';
  if (vehicle.connected === false) return 'FC 연결 끊김';
  if (vehicle.connected !== true || vehicle.vehicle_status_fresh !== true
    || vehicle.arming_state_valid !== true) return 'FC 상태 확인 불가';
  if (!fresh) return '상태 수신 지연';
  return null;
}

export function emergencyLandReady(vehicle: Record<string, unknown> | null, fresh: boolean): boolean {
  return fresh && Boolean(vehicle?.active_execution_mission_id)
    && vehicle?.emergency_land_available === true
    && vehicle?.command_output_enabled === true && vehicle?.connected === true
    && vehicle?.vehicle_status_fresh === true && vehicle?.arming_state_valid === true
    && vehicle?.armed === true;
}
