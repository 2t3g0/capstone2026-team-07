import { clampUnit, type ManualAxes, ZERO_AXES } from './control';

export type JsonObject = Record<string, unknown>;

export type OutboundMessage =
  | { type: 'command.text'; command: string }
  | { type: 'mission.propose'; raw_command: string; plan: object }
  | { type: 'mission.approve'; proposal_id: string; approved: boolean; operator_id: string }
  | { type: 'mission.execute'; proposal_id: string; mission_id: string; plan_json: string }
  | { type: 'mission.resume'; mission_id: string; operator_id: string; reason: string }
  | { type: 'mission.emergency_land'; mission_id: string; operator_id: string; reason: string }
  | { type: 'mission.forward_test_prepare'; operator_id: string; target_altitude_home_m: 1 | 2 }
  | { type: 'control.claim' }
  | { type: 'manual.override'; active: boolean; source: string; reason: string }
  | ({ type: 'manual.velocity'; seq: number; deadman: boolean } & ManualAxes)
  | { type: 'heartbeat'; timestamp: number };

export type InboundType =
  | 'mission.proposal'
  | 'mission.status'
  | 'vehicle.state'
  | 'mission.approval_result'
  | 'mission.execution_result'
  | 'mission.execution_health'
  | 'mission.emergency_result'
  | 'mission.execution_started'
  | 'mission.route_frame_status'
  | 'gateway.capabilities'
  | 'mission.resume_result'
  | 'mission.forward_test_prepare_result'
  | 'gateway.status'
  | 'control.status'
  | 'error';

export interface InboundMessage extends JsonObject {
  type: InboundType;
}

const INBOUND_TYPES = new Set<InboundType>([
  'mission.proposal',
  'mission.status',
  'vehicle.state',
  'mission.approval_result',
  'mission.execution_result',
  'mission.execution_health',
  'mission.emergency_result',
  'mission.execution_started',
  'mission.route_frame_status',
  'gateway.capabilities',
  'mission.resume_result',
  'mission.forward_test_prepare_result',
  'gateway.status',
  'control.status',
  'error',
]);

export function getDefaultEndpoints(location: Pick<Location, 'protocol' | 'hostname'>): {
  apiBase: string;
  webSocketUrl: string;
} {
  const secure = location.protocol === 'https:';
  return {
    apiBase: `${secure ? 'https' : 'http'}://${location.hostname}:9293`,
    webSocketUrl: `${secure ? 'wss' : 'ws'}://${location.hostname}:9293/ws/operator`,
  };
}

export function parseInboundMessage(raw: string): InboundMessage | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const type = (value as JsonObject).type;
    if (typeof type !== 'string' || !INBOUND_TYPES.has(type as InboundType)) return null;
    return value as InboundMessage;
  } catch {
    return null;
  }
}

export function manualVelocity(seq: number, axes: ManualAxes, deadman: boolean): OutboundMessage {
  const safeAxes = deadman ? axes : ZERO_AXES;
  return {
    type: 'manual.velocity',
    seq: Math.max(0, Math.trunc(seq)),
    forward: clampUnit(safeAxes.forward),
    right: clampUnit(safeAxes.right),
    up: clampUnit(safeAxes.up),
    yaw: clampUnit(safeAxes.yaw),
    deadman,
  };
}

export function serializeOutbound(message: OutboundMessage): string {
  return JSON.stringify(message);
}

export function audioResponseToProposal(payload: unknown): OutboundMessage {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error('음성 계획 응답 형식이 올바르지 않습니다.');
  }
  const data = payload as JsonObject;
  const wrappedPlan = data.plan;
  const plan = wrappedPlan && typeof wrappedPlan === 'object' && !Array.isArray(wrappedPlan)
    ? wrappedPlan
    : data;
  const rawCommand = data.raw_command ?? data.command ?? data.transcript ?? '음성 명령';
  return {
    type: 'mission.propose',
    raw_command: typeof rawCommand === 'string' ? rawCommand : String(rawCommand),
    plan,
  };
}

export function isRouteDraftPlan(payload: unknown): payload is JsonObject {
  return Boolean(
    payload
    && typeof payload === 'object'
    && !Array.isArray(payload)
    && (payload as JsonObject).request_purpose === 'CREATE_ROUTE',
  );
}

export function parseProposalPlan(proposal: JsonObject | null): object | null {
  if (!proposal) return null;
  const plan = proposal.plan;
  if (plan && typeof plan === 'object' && !Array.isArray(plan)) return plan;
  if (typeof proposal.plan_json !== 'string') return null;
  try {
    const parsed: unknown = JSON.parse(proposal.plan_json);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

export function readManualOverride(vehicle: JsonObject | null): boolean | null {
  return typeof vehicle?.manual_override === 'boolean' ? vehicle.manual_override : null;
}

export function readPositionNed(vehicle: JsonObject | null): [unknown, unknown, unknown] {
  const array = Array.isArray(vehicle?.position_ned_m) ? vehicle.position_ned_m : [];
  return [
    vehicle?.x ?? vehicle?.north_m ?? array[0],
    vehicle?.y ?? vehicle?.east_m ?? array[1],
    vehicle?.z ?? vehicle?.down_m ?? vehicle?.altitude ?? array[2],
  ];
}
