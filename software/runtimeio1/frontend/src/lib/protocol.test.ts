import { describe, expect, it } from 'vitest';
import {
  audioResponseToProposal,
  getDefaultEndpoints,
  isRouteDraftPlan,
  manualVelocity,
  parseInboundMessage,
  parseProposalPlan,
  readManualOverride,
  readPositionNed,
} from './protocol';

describe('operator protocol', () => {
  it('uses current host with secure transports on HTTPS', () => {
    expect(getDefaultEndpoints({ protocol: 'https:', hostname: 'ops.local' } as Location)).toEqual({
      apiBase: 'https://ops.local:9293',
      webSocketUrl: 'wss://ops.local:9293/ws/operator',
    });
  });

  it('rejects malformed and unknown inbound messages', () => {
    expect(parseInboundMessage('{')).toBeNull();
    expect(parseInboundMessage('{"type":"unknown"}')).toBeNull();
    expect(parseInboundMessage('{"type":"vehicle.state","armed":true}')).toMatchObject({ armed: true });
    expect(parseInboundMessage('{"type":"control.status","owned":true}')).toMatchObject({ owned: true });
    expect(parseInboundMessage('{"type":"mission.resume_result","accepted":true}')).toMatchObject({ accepted: true });
    expect(parseInboundMessage('{"type":"mission.emergency_result","action":"LAND","accepted":true,"mission_id":"m1"}')).toMatchObject({ action: 'LAND', accepted: true });
    expect(parseInboundMessage('{"type":"mission.execution_health","mission_id":"m1","stale":true,"feedback_age_ms":3001,"detail":"stale"}')).toMatchObject({ stale: true });
  });

  it('zeroes motion whenever deadman is released', () => {
    expect(manualVelocity(3.9, { forward: 2, right: -2, up: 0.4, yaw: 0.2 }, false)).toEqual({
      type: 'manual.velocity',
      seq: 3,
      forward: 0,
      right: 0,
      up: 0,
      yaw: 0,
      deadman: false,
    });
  });

  it('converts audio API output to a mission proposal message', () => {
    expect(audioResponseToProposal({ status: 'OK', patrol_zones: ['A'] })).toEqual({
      type: 'mission.propose',
      raw_command: '음성 명령',
      plan: { status: 'OK', patrol_zones: ['A'] },
    });
  });

  it('keeps compatibility with a wrapped audio plan response', () => {
    expect(audioResponseToProposal({ transcript: 'A 구역 순찰', plan: { zone: 'A' } })).toEqual({
      type: 'mission.propose', raw_command: 'A 구역 순찰', plan: { zone: 'A' },
    });
  });

  it('distinguishes route drafting from an executable mission plan', () => {
    expect(isRouteDraftPlan({ request_purpose: 'CREATE_ROUTE', status: 'OK' })).toBe(true);
    expect(isRouteDraftPlan({ request_purpose: 'EXECUTE_MISSION', status: 'OK' })).toBe(false);
    expect(isRouteDraftPlan(null)).toBe(false);
  });

  it('parses ROS mission proposal plan_json when no plan object is present', () => {
    expect(parseProposalPlan({ proposal_id: 'p-1', plan_json: '{"status":"OK","patrol_zones":["B"]}' })).toEqual({
      status: 'OK', patrol_zones: ['B'],
    });
    expect(parseProposalPlan({ plan_json: '{bad json' })).toBeNull();
  });

  it('reads the authoritative manual state and NED array contract', () => {
    expect(readManualOverride({ manual_override: true })).toBe(true);
    expect(readManualOverride({ manual_override: 'true' })).toBeNull();
    expect(readPositionNed({ position_ned_m: [1.5, -2, -12] })).toEqual([1.5, -2, -12]);
    expect(readPositionNed({ x: 3, position_ned_m: [1, 2, 3] })).toEqual([3, 2, 3]);
  });
});
