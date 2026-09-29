import { describe, it, expect } from 'vitest';
import { accumulatedAgeMs, emergencyLandReady, groundPreparationStateFresh, vehicleStateFresh } from './stateFreshness';

describe('retained source age', () => {
  it('keeps the 500ms boundary for execution controls', () => {
    expect(vehicleStateFresh(300, 1000, 1199, true, true, true, true)).toBe(true);
    expect(vehicleStateFresh(300, 1000, 1200, true, true, true, true)).toBe(false);
    expect(vehicleStateFresh(300, 1000, 1201, true, true, true, true)).toBe(false);
  });
  it('never calls a fresh Gateway message LIVE when FC state is stale', () => {
    expect(vehicleStateFresh(0, 1000, 1001, true, true, false, true)).toBe(false);
    expect(vehicleStateFresh(0, 1000, 1001, true, true, true, false)).toBe(false);
  });
  it('invalidates old readiness across disconnect and tab suspension', () => {
    expect(vehicleStateFresh(0, 1000, 1001, false, true, true, true)).toBe(false);
    expect(vehicleStateFresh(0, 1000, 1001, true, false, true, true)).toBe(false);
    expect(vehicleStateFresh(undefined, 0, 1002, true, true, true, true)).toBe(false);
    expect(vehicleStateFresh(0, 1000, 60000, true, true, true, true)).toBe(false);
  });
  it('cannot make a replay LIVE', () => expect(accumulatedAgeMs(30000, 10, 10)).toBe(30000));
  it('expires in the browser without another message', () => {
    expect(accumulatedAgeMs(100, 1000, 1400)).toBe(500);
  });
  it('fails closed on absent, negative or nonfinite age', () => {
    for (const age of [undefined, null, -1, NaN, Infinity]) expect(accumulatedAgeMs(age, 0, 1)).toBe(Infinity);
  });
  it('allows an existing ERROR mission to request LAND, not an idle/stale craft', () => {
    const v = {active_execution_mission_id:'m', phase:'ERROR', emergency_land_available:true,
      command_output_enabled:true, connected:true, vehicle_status_fresh:true, arming_state_valid:true, armed:true};
    expect(emergencyLandReady(v, true)).toBe(true);
    expect(emergencyLandReady(v, false)).toBe(false);
    expect(emergencyLandReady({...v, active_execution_mission_id:''}, true)).toBe(false);
  });
});

describe('idle ground preparation timeout', () => {
  const ground = {gateway_vehicle_age_ms: 300, connected: true, vehicle_status_fresh: true,
    armed: false, landed: true, arming_state_valid: true, landed_state_valid: true,
    active_execution_mission_id: '', terminal_in_progress: false};
  it('allows preparation through 999ms but expires at 1000ms; execution still expires at 500ms', () => {
    expect(groundPreparationStateFresh(ground, 1000, 1699, true, true)).toBe(true);
    expect(groundPreparationStateFresh(ground, 1000, 1700, true, true)).toBe(false);
    expect(groundPreparationStateFresh(ground, 1000, 1701, true, true)).toBe(false);
    expect(vehicleStateFresh(300, 1000, 1699, true, true, true, true)).toBe(false);
  });
  it('does not extend armed, airborne, active or unknown ground states', () => {
    for (const change of [{armed: true}, {landed: false}, {arming_state_valid: false},
      {landed_state_valid: false}, {armed: undefined}, {active_execution_mission_id: 'm'},
      {terminal_in_progress: true}]) {
      expect(groundPreparationStateFresh({...ground, ...change}, 1000, 1200, true, true)).toBe(false);
    }
  });
  it('cannot extend a disconnect, stale FC state, invalid age or suspended tab', () => {
    expect(groundPreparationStateFresh(ground, 1000, 1001, false, true)).toBe(false);
    expect(groundPreparationStateFresh(ground, 1000, 1001, true, false)).toBe(false);
    for (const change of [{connected: false}, {vehicle_status_fresh: false},
      {gateway_vehicle_age_ms: NaN}, {gateway_vehicle_age_ms: -1}]) {
      expect(groundPreparationStateFresh({...ground, ...change}, 1000, 1001, true, true)).toBe(false);
    }
    expect(groundPreparationStateFresh(ground, 1000, 60000, true, true)).toBe(false);
  });
});
