// @vitest-environment jsdom
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

// Mount the production App and socket hook. Only the transport and maps are fake.
vi.mock('./components/RoutePlanner', () => ({ RoutePlanner: () => null }));
vi.mock('./components/LivePositionMap', () => ({ LivePositionMap: () => null }));

class TestSocket {
  static OPEN = 1;
  static instances: TestSocket[] = [];
  readyState = 0;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  sent: Record<string, unknown>[] = [];
  constructor() { TestSocket.instances.push(this); }
  open() { this.readyState = 1; this.onopen?.(); }
  close() { this.readyState = 3; this.onclose?.(); }
  receive(message: Record<string, unknown>) { this.onmessage?.({ data: JSON.stringify(message) }); }
  send(raw: string) { this.sent.push(JSON.parse(raw)); }
}

const capabilities = {
  type: 'gateway.capabilities', build_id: 'low-speed-runtimeio1-20260929',
  protocol_version: 13, route_frame_version: 1, flight_output_handshake_version: 8,
  altitude_reference_version: 6, home_correction_version: 3, status_freshness_version: 3,
  emergency_control_version: 2, auto_execute: true, live_geo_position: true,
  low_speed_profile_version: 2, low_speed_target_altitudes_m: [1, 2],
  forward_test: true, forward_test_camera_switch: true,
};
const ground = {
  type: 'vehicle.state', gateway_vehicle_age_ms: 0, connected: true,
  vehicle_status_fresh: true, arming_state_valid: true, landed_state_valid: true,
  armed: false, landed: true, active_execution_mission_id: '', terminal_in_progress: false,
  command_output_enabled: true, preflight_checks_pass: true, position_valid: true,
  heading_fresh: true, heading_rad: 0, low_speed_safety_valid: true,
  altitude_alignment_ready: true, geo_valid: true, geo_age_ms: 0,
  latitude_deg: 37, longitude_deg: 127, forward_test_camera_bypass_enabled: true,
};

let root: Root;
let container: HTMLDivElement;
let socket: TestSocket;
function receive(message: Record<string, unknown>, target = socket) {
  act(() => target.receive(message));
}
function ready(target = socket) {
  receive({ type: 'gateway.status', ros_connected: true }, target);
  receive(capabilities, target);
  receive(ground, target);
}
function button(label: string) {
  const found = [...container.querySelectorAll('button')].find(b => b.textContent === label);
  expect(found, `button ${label}`).toBeDefined();
  return found!;
}
function header() { return container.querySelector('.top-statuses')!.textContent; }
function advance(ms: number) { act(() => vi.advanceTimersByTime(ms)); }
function claim() {
  act(() => button('운영권 가져오기').click());
  expect(socket.sent).toEqual([{ type: 'control.claim' }]);
  receive({ type: 'control.status', owned: true });
}

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date', 'performance', 'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval'] });
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  vi.stubGlobal('WebSocket', TestSocket);
  TestSocket.instances = [];
  container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root.render(createElement(App)));
  socket = TestSocket.instances[0];
  act(() => socket.open());
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('ownership independent of telemetry freshness', () => {
  it('shows telemetry delay instead of a yellow ROS connected message', () => {
    ready(); claim();
    receive({type:'gateway.status',ros_connected:true,detail:'ROS gateway connected'});
    advance(1100);
    expect(container.querySelector('.connection-banner')?.textContent ?? container.textContent).not.toContain('ROS gateway connected');
    expect(header()).toContain('상태 수신 지연');
  });
  it('deduplicates old results and does not clear a new execution', () => {
    ready(); claim();
    const result={type:'mission.execution_result',mission_id:'old',proposal_id:'old-p',success:false,message:'OLD_RESULT'};
    receive(result); receive(result);
    expect(container.textContent?.split('OLD_RESULT').length).toBe(2);
    expect(container.textContent).toContain('이전 임무 결과 [old]');
    receive({type:'mission.execution_started',mission_id:'new',proposal_id:'new-p'});
    receive(result);
    expect(button('HOME 2 m').disabled).toBe(true);
  });
  it('prepares the scenario without a camera readiness gate and shows the whole envelope', () => {
    ready(); claim();
    receive({ ...capabilities, test_scenario_v1: true });
    receive({ ...ground, forward_test_camera_bypass_enabled: false,
      jetson_safety_fresh: false, jetson_safety_state: '' });
    expect(button('시험 준비').disabled).toBe(false);
    expect(container.textContent).toContain('최대 이동 범위 11m');
    expect(container.textContent).toContain('회피 상승 HOME 3m');
    expect(container.textContent).toContain('2m 감지 · 전방 CLEAR까지 상승 후 3m 전진');
    expect(container.textContent).toContain('미검출 대기 20초');
    expect(container.textContent).not.toContain('카메라 장애물 보호 OFF');
    act(() => button('시험 준비').click());
    expect(socket.sent.at(-1)).toEqual({ type: 'mission.forward_test_prepare',
      operator_id: expect.any(String), target_altitude_home_m: 1 });
  });
  it('labels the forward test as 2 m independently of HOME altitude', () => {
    ready();
    expect(container.textContent).toContain('기수 방향 2 m 시험');
    expect(container.textContent).toContain('HOME 1 m');
    expect(container.textContent).toContain('HOME 2 m');
  });

  it('keeps the server owner through a 1.1s gap and restores preparation without another claim', () => {
    ready(); claim();
    expect(button('시험 준비').disabled).toBe(false);
    advance(1100);
    expect(button('운영권 보유').disabled).toBe(true);
    expect(header()).toContain('ARMING STATE UNKNOWN · 상태 수신 지연');
    expect(button('시험 준비').disabled).toBe(true);
    receive(ground);
    expect(header()).toContain('DISARMED');
    expect(button('시험 준비').disabled).toBe(false);
    expect(socket.sent).toEqual([{ type: 'control.claim' }]);
  });

  it.each(['운영권 이전', '운영권 만료'])('honors server revocation (%s), including after stale recovery', detail => {
    ready(); claim(); advance(1100);
    receive({ type: 'control.status', owned: false, detail });
    receive(ground);
    expect(button('운영권 가져오기').disabled).toBe(false);
    expect(button('시험 준비').disabled).toBe(true);
    expect(socket.sent).toEqual([{ type: 'control.claim' }]);
  });

  it('does not lose or duplicate a pending claim during a telemetry gap', () => {
    ready();
    act(() => button('운영권 가져오기').click());
    advance(1100);
    expect(button('인수 중').disabled).toBe(true);
    receive({ type: 'control.status', owned: true });
    expect(button('운영권 보유').disabled).toBe(true);
    receive(ground);
    expect(button('시험 준비').disabled).toBe(false);
    expect(socket.sent).toEqual([{ type: 'control.claim' }]);
  });

  it('retains server ownership across ROS loss while invalidating preparation', () => {
    ready(); claim();
    receive({ type: 'gateway.status', ros_connected: false });
    expect(header()).toContain('ARMING STATE UNKNOWN · ROS 연결 끊김');
    expect(button('운영권 보유').disabled).toBe(true);
    expect(button('시험 준비').disabled).toBe(true);
    ready();
    expect(button('시험 준비').disabled).toBe(false);
    expect(socket.sent).toEqual([{ type: 'control.claim' }]);
  });

  it('clears owner on browser disconnect and ignores previous-connection messages after reconnect', () => {
    ready(); claim();
    const oldSocket = socket;
    act(() => oldSocket.close());
    expect(header()).toContain('브라우저 연결 끊김');
    expect(button('운영권 가져오기').disabled).toBe(true);
    advance(500);
    socket = TestSocket.instances[1];
    act(() => socket.open());
    ready();
    receive({ type: 'control.status', owned: true }, oldSocket);
    expect(button('운영권 가져오기').disabled).toBe(false);
    expect(button('시험 준비').disabled).toBe(true);
    expect(socket.sent).toEqual([]);
    receive({ type: 'control.status', owned: true });
    expect(button('시험 준비').disabled).toBe(false);
  });
});

describe('UNKNOWN diagnostics and unchanged boundaries', () => {
  it.each([
    [999, false, false], [1000, false, true], [499, true, false], [500, true, true],
  ])('source age %d ms, armed %s -> UNKNOWN %s', (age, armed, unknown) => {
    ready(); claim();
    receive({ ...ground, gateway_vehicle_age_ms: age, armed });
    expect(header()?.includes('ARMING STATE UNKNOWN')).toBe(unknown);
    expect(button('시험 준비').disabled).toBe(unknown || armed);
  });

  it.each([
    [{ connected: false }, 'FC 연결 끊김'],
    [{ vehicle_status_fresh: false }, 'FC 상태 확인 불가'],
    [{ arming_state_valid: false }, 'FC 상태 확인 불가'],
  ])('blocks invalid FC state %o without revoking ownership', (change, reason) => {
    ready(); claim();
    receive({ ...ground, ...change });
    expect(header()).toContain(`ARMING STATE UNKNOWN · ${reason}`);
    expect(button('운영권 보유').disabled).toBe(true);
    expect(button('시험 준비').disabled).toBe(true);
    receive(ground);
    expect(button('시험 준비').disabled).toBe(false);
  });

  it('distinguishes initial data wait from known FC or ROS loss', () => {
    receive({ type: 'gateway.status', ros_connected: true });
    expect(header()).toContain('ARMING STATE UNKNOWN · 기체 상태 수신 대기');
    expect(button('시험 준비').disabled).toBe(true);
  });
});

function preparedScenario() {
  ready(); claim(); receive({ ...capabilities, test_scenario_v1: true });
  const plan = { status: 'OK', mission_kind: 'FORWARD_TEST_1M', flight_profile: 'LOW_SPEED_1M_V1',
    low_speed_limits: { target_altitude_home_m: 1 }, preview_created_unix_ms: Date.now(),
    preview_start_ned_m: [0,0,0], preview_end_ned_m: [10,0,0], preview_heading_rad: 0,
    test_scenario: { version: 1, build_id: capabilities.build_id, search_m: 10, final_m: 1 } };
  receive({ type:'mission.proposal',proposal_id:'review',status:1,requires_approval:true,plan_json:JSON.stringify(plan) });
  receive({ type:'mission.route_frame_status',proposal_id:'review',valid:true,altitude_reference_valid:true });
  return container.querySelector<HTMLButtonElement>('.hold-to-run-button')!;
}

describe('review: configuration changes retire preparation', () => {
  it('does not dispatch after telemetry expires during the hold', () => {
    const execute=preparedScenario();
    act(() => execute.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true})));
    advance(3100);
    expect(socket.sent.some(m => m.type==='mission.approve' && m.approved===true)).toBe(false);
  });
  it('locks settings as soon as positive approval is dispatched, before airborne telemetry', () => {
    const execute=preparedScenario();
    act(() => execute.dispatchEvent(new KeyboardEvent('keydown',{key:'Enter',bubbles:true})));
    for(let i=0;i<31;i++) { receive(ground); advance(100); }
    expect(socket.sent.some(m => m.type==='mission.approve' && m.approved===true)).toBe(true);
    expect(button('HOME 2 m').disabled).toBe(true);
  });
  it('discards a delayed prepared proposal after settings changed', () => {
    ready(); claim(); receive({...capabilities,test_scenario_v1:true});
    act(() => button('시험 준비').click());
    act(() => button('HOME 2 m').click());
    receive({type:'mission.proposal',proposal_id:'late',requires_approval:true,
      plan_json:JSON.stringify({mission_kind:'FORWARD_TEST_1M',flight_profile:'LOW_SPEED_1M_V1'})});
    expect(container.querySelector('.hold-to-run-button')).toBeNull();
    expect(socket.sent).toContainEqual({type:'mission.approve',proposal_id:'late',approved:false,operator_id:expect.any(String)});
    receive({type:'mission.approval_result',proposal_id:'late',accepted:true});
    expect(container.textContent).not.toContain('승인 완료; 실행 요청 중');
  });
  it('R6 changing 1m to 2m and back requires new preparation', () => {
    const execute=preparedScenario();
    expect(execute.disabled).toBe(false);
    act(() => button('HOME 2 m').click());
    expect(container.querySelector('.hold-to-run-button')).toBeNull();
    act(() => button('HOME 1 m').click());
    expect(container.querySelector('.hold-to-run-button')).toBeNull();
    expect(socket.sent.some(m => m.type==='mission.forward_test_prepare')).toBe(false);
  });
  it('R6 a running three-second approval timer is cancelled after altitude change', () => {
    const execute=preparedScenario();
    act(() => execute.dispatchEvent(new KeyboardEvent('keydown',{ key:'Enter',bubbles:true })));
    advance(100);
    act(() => button('HOME 2 m').click());
    expect(container.querySelector('.hold-to-run-button')).toBeNull();
    advance(3000);
    expect(socket.sent).not.toContainEqual({ type:'mission.approve',proposal_id:'review',approved:true,operator_id:expect.any(String) });
  });
});
