import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CircleStop,
  Link,
  Link2Off,
  KeyRound,
  LoaderCircle,
  Gauge,
  Navigation,
  MapPinned,
  Mic,
  Plane,
  PlaneLanding,
  Radio,
  RefreshCw,
  Route as RouteIcon,
  Send,
  ShieldCheck,
  Square,
  Wifi,
  X,
} from 'lucide-react';
import { RoutePlanner } from './components/RoutePlanner';
import { LivePositionMap } from './components/LivePositionMap';
import { StatusPill } from './components/StatusPill';
import { useOperatorSocket } from './hooks/useOperatorSocket';
import { accumulatedAgeMs, armingStateUnknownReason, emergencyLandReady, groundPreparationStateFresh, vehicleStateFresh } from './lib/stateFreshness';
import {
  audioResponseToProposal,
  getDefaultEndpoints,
  isRouteDraftPlan,
  parseProposalPlan,
  readPositionNed,
  type InboundMessage,
  type JsonObject,
} from './lib/protocol';
import {
  approvalHoldState,
  isForwardTestPlan,
  isLowSpeedProfile,
  lowSpeedAltitudeForPlan,
  toLowSpeedPlan,
  type LowSpeedAltitude,
} from './lib/flightProfiles';

interface Notice {
  id: number;
  tone: 'info' | 'error' | 'success';
  message: string;
}

const OPERATOR_ID = 'web-operator';
const EXPECTED_GATEWAY_BUILD = 'low-speed-runtimeio1-20260929';
const EXPECTED_PROTOCOL_VERSION = 13;
const EXPECTED_ROUTE_FRAME_VERSION = 1;
const PHASE_LABELS: Record<number, string> = {
  0: 'IDLE',
  1: 'AWAITING APPROVAL',
  2: 'TAKEOFF',
  3: 'PATROL',
  4: 'RETURNING HOME',
  5: 'PAUSED MANUAL',
  6: 'COMPLETED',
  7: 'ABORTED',
  8: 'ERROR',
  9: 'MOVING TO START',
  10: 'PAUSED EVENT',
  11: 'EVENT CAPTURE',
  12: 'REJOINING ROUTE',
  13: 'HOLD — RESUME REQUIRED',
  14: 'FORWARD TEST 1 M',
  15: 'LANDING',
};
const defaultEndpoints = getDefaultEndpoints(window.location);
const apiBase = import.meta.env.VITE_API_BASE || defaultEndpoints.apiBase;
const webSocketUrl = import.meta.env.VITE_WS_URL || defaultEndpoints.webSocketUrl;

function objectValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '--';
  if (typeof value === 'boolean') return value ? 'YES' : 'NO';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(2);
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function firstValue(source: JsonObject | null, keys: string[]): unknown {
  for (const key of keys) {
    if (source?.[key] !== undefined) return source[key];
  }
  return undefined;
}

function titleCase(value: unknown): string {
  return objectValue(value).replaceAll('_', ' ').toUpperCase();
}

function phaseLabel(value: unknown): string {
  return typeof value === 'number' && PHASE_LABELS[value]
    ? PHASE_LABELS[value]
    : titleCase(value ?? 'IDLE');
}

function proposalId(proposal: JsonObject | null): string {
  return String(proposal?.proposal_id ?? proposal?.id ?? '');
}

export default function App() {
  const [gateway, setGateway] = useState<JsonObject | null>(null);
  const [vehicle, setVehicle] = useState<JsonObject | null>(null);
  const [vehicleReceivedAt, setVehicleReceivedAt] = useState(0);
  const [missionStatus, setMissionStatus] = useState<JsonObject | null>(null);
  const [proposal, setProposal] = useState<JsonObject | null>(null);
  const [approval, setApproval] = useState<JsonObject | null>(null);
  const [execution, setExecution] = useState<JsonObject | null>(null);
  const [executionStarted, setExecutionStarted] = useState<JsonObject | null>(null);
  const [executionHealth, setExecutionHealth] = useState<JsonObject | null>(null);
  const [capabilities, setCapabilities] = useState<JsonObject | null>(null);
  const [routeFrameStatus, setRouteFrameStatus] = useState<JsonObject | null>(null);
  const [command, setCommand] = useState('');
  const [notices, setNotices] = useState<Notice[]>([]);
  const [controlOwned, setControlOwned] = useState(false);
  const [claimPending, setClaimPending] = useState(false);
  const [approvalPending, setApprovalPending] = useState(false);
  const [recording, setRecording] = useState(false);
  const [audioBusy, setAudioBusy] = useState(false);
  const [textBusy, setTextBusy] = useState(false);
  const [routePlannerOpen, setRoutePlannerOpen] = useState(false);
  const [routeDraftPlan, setRouteDraftPlan] = useState<JsonObject | null>(null);
  const [routeDraftRequest, setRouteDraftRequest] = useState(0);
  const [forwardPreparePending, setForwardPreparePending] = useState(false);
  const [selectedLowSpeedAltitude, setSelectedLowSpeedAltitude] = useState<LowSpeedAltitude>(1);
  const [holdProgress, setHoldProgress] = useState(0);
  const [emergencyHoldAction, setEmergencyHoldAction] = useState<'STOP' | 'LAND' | null>(null);
  const [emergencyHoldProgress, setEmergencyHoldProgress] = useState(0);
  const [nowMs, setNowMs] = useState(Date.now());
  const [keyDialogOpen, setKeyDialogOpen] = useState(false);
  const [keyConfigured, setKeyConfigured] = useState<boolean | null>(null);
  const [keyValue, setKeyValue] = useState('');
  const [keySaving, setKeySaving] = useState(false);
  const [keyError, setKeyError] = useState('');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const pttHeldRef = useRef(false);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const noticeIdRef = useRef(0);
  const holdTimerRef = useRef<number | null>(null);
  const holdStartedRef = useRef(0);
  const holdCompletedRef = useRef(false);
  const emergencyTimerRef = useRef<number | null>(null);
  const emergencyStartedRef = useRef(0);
  const emergencyCompletedRef = useRef(false);
  const preparationGeneration = useRef(0);
  const retiredProposals = useRef(new Set<string>());
  const pendingPreparation = useRef<number | null>(null);
  const receivedResultKeys = useRef(new Set<string>());
  const currentExecutionMission = useRef<string | null>(null);
  const approvalDispatchReady = useRef(false);
  const approvalFreshnessCheck = useRef<() => boolean>(() => false);
  const sendRef = useRef<(message: Parameters<ReturnType<typeof useOperatorSocket>['send']>[0]) => boolean>(() => false);

  const addNotice = useCallback((message: string, tone: Notice['tone'] = 'info') => {
    const id = ++noticeIdRef.current;
    setNotices((current) => [...current.slice(-3), { id, tone, message }]);
    window.setTimeout(() => setNotices((current) => current.filter((item) => item.id !== id)), 6_000);
  }, []);

  const onSocketMessage = useCallback((message: InboundMessage) => {
    switch (message.type) {
      case 'gateway.status':
        setGateway(message);
        if (message.ros_connected !== true) {
          setVehicle(null); setVehicleReceivedAt(0); setCapabilities(null); setRouteFrameStatus(null);
        }
        break;
      case 'gateway.capabilities': setCapabilities(message); break;
      case 'control.status': {
        const owned = message.owned === true;
        setControlOwned(owned);
        setClaimPending(false);
        addNotice(
          String(message.detail ?? (owned ? '이 세션이 운영권을 획득했습니다.' : '운영권이 다른 세션으로 이동했습니다.')),
          owned ? 'success' : 'info',
        );
        break;
      }
      case 'vehicle.state': setVehicleReceivedAt(performance.now()); setVehicle(message); break;
      case 'mission.status': setMissionStatus(message); break;
      case 'mission.proposal':
        if (retiredProposals.current.has(proposalId(message))
            || (isForwardTestPlan(parseProposalPlan(message) as JsonObject | null) && pendingPreparation.current !== null
              && pendingPreparation.current !== preparationGeneration.current)) {
          retiredProposals.current.add(proposalId(message));
          sendRef.current({ type: 'mission.approve', proposal_id: proposalId(message), approved: false, operator_id: OPERATOR_ID });
          pendingPreparation.current = null;
          setForwardPreparePending(false);
          break;
        }
        pendingPreparation.current = null;
        if (holdTimerRef.current !== null) window.clearInterval(holdTimerRef.current);
        holdTimerRef.current = null;
        holdCompletedRef.current = false;
        preparationGeneration.current += 1;
        setRouteDraftPlan(null);
        setProposal(message);
        setApproval(null);
        setApprovalPending(false);
        setExecution(null);
        setExecutionStarted(null);
        setExecutionHealth(null);
        setRouteFrameStatus(null);
        addNotice('새 임무 계획을 수신했습니다.');
        setForwardPreparePending(false);
        setHoldProgress(0);
        break;
      case 'mission.forward_test_prepare_result':
        if (!message.accepted) { pendingPreparation.current = null; setForwardPreparePending(false); }
        if (!message.accepted) addNotice(String(message.message ?? '2 m 시험 준비가 거부되었습니다.'), 'error');
        break;
      case 'mission.approval_result':
        if (retiredProposals.current.has(String(message.proposal_id ?? ''))) break;
        setApproval(message);
        setApprovalPending(false);
        addNotice(
          message.accepted
            ? '승인 완료; 실행 요청 중'
            : String(message.message ?? '승인이 거부되었습니다.'),
          message.accepted ? 'success' : 'error',
        );
        break;
      case 'mission.execution_started':
        if (retiredProposals.current.has(String(message.proposal_id ?? ''))) break;
        currentExecutionMission.current = String(message.mission_id ?? "");
        setExecutionStarted(message);
        addNotice('자동 실행을 시작했습니다.', 'success');
        break;
      case 'mission.route_frame_status':
        if (String(message.proposal_id ?? '') === proposalId(proposal)) {
          setRouteFrameStatus(message);
        }
        break;
      case 'mission.execution_result': {
        const resultKey = String(message.mission_id ?? '') + ':' + String(message.proposal_id ?? '') + ':' + String(message.success) + ':' + String(message.message ?? message.detail ?? '');
        if (receivedResultKeys.current.has(resultKey)) break;
        receivedResultKeys.current.add(resultKey);
        if (receivedResultKeys.current.size > 256) receivedResultKeys.current.delete(receivedResultKeys.current.values().next().value!);
        if (currentExecutionMission.current !== String(message.mission_id ?? '')) {
          addNotice('이전 임무 결과 [' + String(message.mission_id ?? '') + ']: ' + String(message.message ?? ''), 'info');
          break;
        }
        currentExecutionMission.current = null;
        setApproval(null); setApprovalPending(false); setExecutionStarted(null);
        setExecution(message);
        setExecutionHealth(null);
        addNotice(String(message.message ?? '임무 실행 결과를 수신했습니다.'), message.success ? 'success' : 'error');
        break;
      }
      case 'mission.execution_health':
        setExecutionHealth(message);
        break;
      case 'mission.emergency_result':
        addNotice(
          String(message.message ?? (message.accepted ? '긴급 착륙이 승인되었습니다.' : '긴급 착륙이 거부되었습니다.')),
          message.accepted ? 'success' : 'error',
        );
        break;
      case 'mission.resume_result':
        addNotice(
          String(message.message ?? (message.accepted ? '임무 재개가 승인되었습니다.' : '임무 재개가 거부되었습니다.')),
          message.accepted ? 'success' : 'error',
        );
        break;
      case 'error':
        if (String(message.code ?? '').includes('forward_test_prepare')) {
          pendingPreparation.current = null; setForwardPreparePending(false);
        }
        addNotice(`${message.code ?? 'ERROR'}: ${message.message ?? '알 수 없는 오류'}`, 'error');
        break;
    }
  }, [addNotice, proposal]);

  const socket = useOperatorSocket(webSocketUrl, onSocketMessage);
  const send = socket.send;
  sendRef.current = send;
  useEffect(() => {
    if (socket.state !== 'connected') {
      setControlOwned(false);
      setClaimPending(false);
      setVehicle(null);
      setVehicleReceivedAt(0);
      setCapabilities(null);
      setGateway(null);
      setRouteFrameStatus(null);
      pendingPreparation.current = null;
      setForwardPreparePending(false);
    }
  }, [socket.state]);
  const socketConnected = socket.state === 'connected';
  const rosConnected = gateway?.ros_connected === true;
  const vehicleAgeMs = accumulatedAgeMs(vehicle?.gateway_vehicle_age_ms, vehicleReceivedAt, performance.now());
  const executionVehicleFresh = vehicleStateFresh(vehicle?.gateway_vehicle_age_ms, vehicleReceivedAt, performance.now(), socketConnected, rosConnected, vehicle?.connected === true, vehicle?.vehicle_status_fresh === true);
  const vehicleFresh = groundPreparationStateFresh(vehicle, vehicleReceivedAt, performance.now(), socketConnected, rosConnected);
  const controlsAvailable = socketConnected && rosConnected && executionVehicleFresh;
  const preparationAvailable = socketConnected && rosConnected && vehicleFresh;

  useEffect(() => () => {
    if (holdTimerRef.current !== null) window.clearInterval(holdTimerRef.current);
    if (emergencyTimerRef.current !== null) window.clearInterval(emergencyTimerRef.current);
  }, []);

  useEffect(() => {
    if (!keyDialogOpen) return;
    const controller = new AbortController();
    setKeyConfigured(null);
    setKeyError('');
    fetch(`${apiBase}/v1/settings/gemini-key`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`키 상태 확인 실패 (${response.status})`);
        return response.json() as Promise<{ configured?: boolean }>;
      })
      .then((payload) => setKeyConfigured(payload.configured === true))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        setKeyError(error instanceof Error ? error.message : '키 상태를 확인하지 못했습니다.');
      });
    return () => controller.abort();
  }, [keyDialogOpen]);

  const saveGeminiKey = async (event: React.FormEvent) => {
    event.preventDefault();
    const value = keyValue.trim();
    if (value.length < 8 || keySaving) return;
    setKeySaving(true);
    setKeyError('');
    try {
      const response = await fetch(`${apiBase}/v1/settings/gemini-key`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: value }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(payload?.detail || `Gemini API 키 저장 실패 (${response.status})`);
      }
      setKeyValue('');
      setKeyConfigured(true);
      setKeyDialogOpen(false);
      addNotice('Gemini API 키를 저장했습니다. 다음 명령부터 적용됩니다.', 'success');
    } catch (error) {
      setKeyError(error instanceof Error ? error.message : 'Gemini API 키를 저장하지 못했습니다.');
    } finally {
      setKeySaving(false);
    }
  };

  const openRouteDraft = useCallback((payload: JsonObject) => {
    if (payload.status !== 'OK' || !Array.isArray(payload.route_waypoints_enu)) {
      throw new Error(String(payload.message ?? '경로 초안을 생성하지 못했습니다.'));
    }
    setRouteDraftPlan(payload);
    setRouteDraftRequest((value) => value + 1);
    setRoutePlannerOpen(true);
  }, []);

  const submitText = async (event: React.FormEvent) => {
    event.preventDefault();
    const value = command.trim();
    if (!value || !socketConnected || textBusy) return;
    setTextBusy(true);
    try {
      const response = await fetch(`${apiBase}/v1/plan/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: value }),
      });
      if (!response.ok) throw new Error(`텍스트 계획 요청 실패 (${response.status})`);
      const payload: unknown = await response.json();
      if (isRouteDraftPlan(payload)) {
        openRouteDraft(payload);
        setCommand('');
        addNotice('경로 제작 명령을 새 편집 초안으로 열었습니다.', 'success');
        return;
      }
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
        throw new Error('텍스트 계획 응답 형식이 올바르지 않습니다.');
      }
      if (!send({ type: 'mission.propose', raw_command: value, plan: payload })) {
        throw new Error('게이트웨이 연결이 끊어졌습니다.');
      }
      setCommand('');
      addNotice('명령을 전송했습니다.');
    } catch (error) {
      addNotice(error instanceof Error ? error.message : '텍스트 명령 처리 실패', 'error');
    } finally {
      setTextBusy(false);
    }
  };

  const claimControl = () => {
    if (!socketConnected || claimPending) return;
    if (send({ type: 'control.claim' })) {
      setClaimPending(true);
    }
  };

  const cancelEmergencyHold = () => {
    if (emergencyTimerRef.current !== null) window.clearInterval(emergencyTimerRef.current);
    emergencyTimerRef.current = null;
    emergencyCompletedRef.current = false;
    setEmergencyHoldAction(null);
    setEmergencyHoldProgress(0);
  };

  const dispatchEmergency = (action: 'STOP' | 'LAND') => {
    const missionId = String(vehicle?.active_execution_mission_id ?? '');
    if (!missionId) {
      addNotice('실행 중인 임무가 없어 긴급 명령을 보낼 수 없습니다.', 'error');
      return;
    }
    if (action === 'STOP') {
      if (send({
        type: 'manual.override',
        active: true,
        source: 'web-emergency-stop',
        reason: 'operator emergency motion stop and position hold',
      })) {
        addNotice('긴급 이동 정지를 요청했습니다. 현재 위치 HOLD이며 모터 kill이 아닙니다.', 'error');
      }
      return;
    }
    if (send({
      type: 'mission.emergency_land',
      mission_id: missionId,
      operator_id: OPERATOR_ID,
      reason: 'operator requested dashboard emergency LAND',
    })) {
      addNotice('긴급 착륙 요청을 전송했습니다. LAND 승인과 착륙 상태를 확인하세요.', 'error');
    }
  };

  const startEmergencyHold = (action: 'STOP' | 'LAND', durationMs: number) => {
    if (emergencyTimerRef.current !== null || emergencyCompletedRef.current) return;
    emergencyStartedRef.current = performance.now();
    emergencyCompletedRef.current = false;
    setEmergencyHoldAction(action);
    setEmergencyHoldProgress(0);
    emergencyTimerRef.current = window.setInterval(() => {
      const progress = Math.min(1, (performance.now() - emergencyStartedRef.current) / durationMs);
      setEmergencyHoldProgress(progress);
      if (progress >= 1) {
        if (emergencyTimerRef.current !== null) window.clearInterval(emergencyTimerRef.current);
        emergencyTimerRef.current = null;
        emergencyCompletedRef.current = true;
        dispatchEmergency(action);
      }
    }, 40);
  };

  const startRecording = async () => {
    if (!socketConnected || audioBusy || recording) return;
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      addNotice('이 브라우저에서는 마이크 녹음을 지원하지 않습니다.', 'error');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferred = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(stream, preferred ? { mimeType: preferred } : undefined);
      chunksRef.current = [];
      streamRef.current = stream;
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        setRecording(false);
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        if (blob.size === 0) return;
        setAudioBusy(true);
        try {
          const form = new FormData();
          const extension = recorder.mimeType.includes('ogg') ? 'ogg' : 'webm';
          form.append('audio', blob, `command.${extension}`);
          const response = await fetch(`${apiBase}/v1/plan/audio`, { method: 'POST', body: form });
          if (!response.ok) throw new Error(`음성 계획 요청 실패 (${response.status})`);
          const payload: unknown = await response.json();
          if (isRouteDraftPlan(payload)) {
            openRouteDraft(payload);
            addNotice('음성 경로 제작 명령을 새 편집 초안으로 열었습니다.', 'success');
            return;
          }
          const message = audioResponseToProposal(payload);
          if (!send(message)) throw new Error('게이트웨이 연결이 끊어졌습니다.');
          addNotice('음성 명령을 계획으로 전송했습니다.', 'success');
        } catch (error) {
          addNotice(error instanceof Error ? error.message : '음성 명령 처리 실패', 'error');
        } finally {
          setAudioBusy(false);
        }
      };
      recorder.start();
      setRecording(true);
      if (!pttHeldRef.current) recorder.stop();
    } catch (error) {
      addNotice(error instanceof Error ? error.message : '마이크 권한을 얻지 못했습니다.', 'error');
    }
  };

  const stopRecording = () => {
    pttHeldRef.current = false;
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
  };

  const approve = (approved: boolean) => {
    const id = proposalId(proposal);
    if (!id) return;
    if (send({ type: 'mission.approve', proposal_id: id, approved, operator_id: OPERATOR_ID })) {
      setApprovalPending(true);
    }
  };

  const scenarioEnabled = capabilities?.test_scenario_v1 === true;
  const prepareForwardTest = () => {
    if (forwardPreparePending) return;
    if (send({
      type: 'mission.forward_test_prepare',
      operator_id: OPERATOR_ID,
      target_altitude_home_m: selectedLowSpeedAltitude,
    })) {
      setForwardPreparePending(true);
      pendingPreparation.current = preparationGeneration.current;
      addNotice(scenarioEnabled ? `HOME ${selectedLowSpeedAltitude}m 통합 시험: 탐색 10m + 마지막 1m, 감지 2m, 상승 최대 +2m, CLEAR 후 전진 3m, 미검출 대기 20초 · LOW 지속 5초 · Home 확인 2초` : `HOME 기준 ${selectedLowSpeedAltitude} m와 현재 기수각으로 2 m 시험 미리보기를 준비합니다.`);
    }
  };

  const prepareLowSpeedCopy = () => {
    if (!plan) return;
    try {
      const snapshot = toLowSpeedPlan(plan, selectedLowSpeedAltitude);
      if (!send({ type: 'mission.propose', raw_command: `${String(proposal?.raw_command ?? '등록 경로')} — HOME 기준 ${selectedLowSpeedAltitude} m 저속 실행 사본`, plan: snapshot })) {
        throw new Error('게이트웨이 연결이 끊어졌습니다.');
      }
    } catch (error) {
      addNotice(error instanceof Error ? error.message : '저속 실행 사본을 만들지 못했습니다.', 'error');
    }
  };

  const cancelApprovalHold = () => {
    if (holdTimerRef.current !== null) window.clearInterval(holdTimerRef.current);
    holdTimerRef.current = null;
    holdCompletedRef.current = false;
    setHoldProgress(0);
  };

  const startApprovalHold = () => {
    if (holdTimerRef.current !== null || holdCompletedRef.current) return;
    const generation = preparationGeneration.current;
    const heldProposal = proposalId(proposal);
    holdStartedRef.current = performance.now();
    holdTimerRef.current = window.setInterval(() => {
      if (generation !== preparationGeneration.current || retiredProposals.current.has(heldProposal)
          || !approvalDispatchReady.current || !approvalFreshnessCheck.current()) { cancelApprovalHold(); return; }
      const state = approvalHoldState(performance.now() - holdStartedRef.current, true);
      setHoldProgress(state.progress);
      if (state.dispatch) {
        if (holdTimerRef.current !== null) window.clearInterval(holdTimerRef.current);
        holdTimerRef.current = null;
        holdCompletedRef.current = true;
        approve(true);
      }
    }, 50);
  };

  const resumeMission = () => {
    const missionId = String(missionStatus?.mission_id ?? '');
    if (!missionId) return;
    send({
      type: 'mission.resume',
      mission_id: missionId,
      operator_id: OPERATOR_ID,
      reason: 'operator confirmed autonomous resume',
    });
  };

  const plan = parseProposalPlan(proposal) as JsonObject | null;
  const positionNed = readPositionNed(vehicle);
  const planRows = useMemo(() => plan ? Object.entries(plan).slice(0, 10) : [], [plan]);
  const proposalReady = plan !== null && proposal?.requires_approval === true;
  const capabilitiesReceived = capabilities !== null;
  const capabilitiesReady = capabilities?.build_id === EXPECTED_GATEWAY_BUILD
    && capabilities?.protocol_version === EXPECTED_PROTOCOL_VERSION
    && capabilities?.route_frame_version === EXPECTED_ROUTE_FRAME_VERSION
    && capabilities?.flight_output_handshake_version === 8
    && capabilities?.altitude_reference_version === 6
    && capabilities?.home_correction_version === 3
    && capabilities?.status_freshness_version === 3
    && capabilities?.emergency_control_version === 2
    && capabilities?.auto_execute === true;
  const lowSpeedCapabilitiesReady = capabilitiesReady
    && capabilities?.live_geo_position === true
    && capabilities?.low_speed_profile_version === 2
    && Array.isArray(capabilities?.low_speed_target_altitudes_m)
    && capabilities.low_speed_target_altitudes_m.includes(1)
    && capabilities.low_speed_target_altitudes_m.includes(2)
    && capabilities?.forward_test === true
    && capabilities?.forward_test_camera_switch === true
    && capabilities?.flight_output_handshake_version === 8
    && capabilities?.altitude_reference_version === 6
    && capabilities?.home_correction_version === 3
    && capabilities?.status_freshness_version === 3;
  const routeFrameReady = routeFrameStatus?.proposal_id === proposalId(proposal)
    && routeFrameStatus?.valid === true;
  const routeAltitudeReferenceReady = !isLowSpeedProfile(plan?.flight_profile)
    || routeFrameStatus?.altitude_reference_valid === true;
  const finalApprovalReady = proposalReady && capabilitiesReady && routeFrameReady
    && routeAltitudeReferenceReady;
  const reviewHint = !controlsAvailable
    ? 'HUB와 ROS 연결이 필요합니다.'
      : !controlOwned
      ? '우상단의 운영권 가져오기를 먼저 누르세요.'
      : !capabilitiesReceived
        ? 'Jetson 실행 프로토콜/빌드 정보를 기다리는 중입니다.'
      : !capabilitiesReady
        ? `Jetson 실행 프로토콜/빌드가 일치하지 않아 승인을 차단했습니다. `
          + `(실제 build=${String(capabilities?.build_id ?? '--')}, `
          + `protocol=${String(capabilities?.protocol_version ?? '--')}, `
          + `frame=${String(capabilities?.route_frame_version ?? '--')})`
      : !proposalReady
        ? '실행 가능한 OK 계획이 아닙니다.'
      : !routeFrameReady
          ? `좌표 프레임 검증 대기/실패: ${String(routeFrameStatus?.reason ?? '상태 없음')}`
        : !routeAltitudeReferenceReady
          ? `저속 고도 기준 정렬 실패: ${String(routeFrameStatus?.altitude_reference_detail ?? '상태 없음')}`
        : approvalPending
          ? '승인 결과를 기다리는 중입니다.'
          : approval
            ? approval.accepted
              ? executionStarted ? '자동 실행을 시작했습니다.' : '승인 완료; 실행 요청 중'
              : '이 제안은 거절되었습니다.'
            : '거절 또는 승인 중 하나를 선택하세요. 승인하면 즉시 실행됩니다.';
  const phase = firstValue(missionStatus, ['phase', 'status', 'mission_phase']);
  const currentWaypoint = firstValue(missionStatus, ['waypoint_index', 'current_waypoint']);
  const totalWaypoints = firstValue(missionStatus, ['total_waypoints']);
  const waypointLabel = typeof currentWaypoint === 'number' && typeof totalWaypoints === 'number' && totalWaypoints > 0
    ? `${currentWaypoint} / ${totalWaypoints}`
    : objectValue(currentWaypoint);
  const reportedProgress = firstValue(missionStatus, ['progress', 'progress_percent']);
  const progressLabel = reportedProgress !== undefined
    ? objectValue(reportedProgress)
    : typeof currentWaypoint === 'number' && typeof totalWaypoints === 'number' && totalWaypoints > 0
      ? `${Math.round((currentWaypoint / totalWaypoints) * 100)}%`
      : '--';
  const armed = firstValue(vehicle, ['armed', 'is_armed']) === true;
  const fcStateFresh = vehicleFresh && vehicle?.connected === true
    && vehicle?.vehicle_status_fresh === true && vehicle?.arming_state_valid === true;
  const armedStateLabel = fcStateFresh
    ? (armed ? 'ARMED' : 'DISARMED')
    : 'ARMING STATE UNKNOWN';
  const unknownReason = armingStateUnknownReason(vehicle, vehicleFresh, socketConnected, rosConnected);
  const landed = firstValue(vehicle, ['landed']) === true;
  const geoAgeMs = accumulatedAgeMs(vehicle?.geo_age_ms, vehicleReceivedAt, performance.now());
  const latitude = typeof vehicle?.latitude_deg === 'number' && Number.isFinite(vehicle.latitude_deg) ? vehicle.latitude_deg : null;
  const longitude = typeof vehicle?.longitude_deg === 'number' && Number.isFinite(vehicle.longitude_deg) ? vehicle.longitude_deg : null;
  const headingRad = typeof vehicle?.heading_rad === 'number' && Number.isFinite(vehicle.heading_rad) ? vehicle.heading_rad : null;
  const homeAltitude = typeof vehicle?.altitude_home_relative_m === 'number'
    && Number.isFinite(vehicle.altitude_home_relative_m)
    ? vehicle.altitude_home_relative_m : null;
  const estimatedClimb = homeAltitude === null ? null : selectedLowSpeedAltitude-homeAltitude;
  const frameHomeNed = Array.isArray(routeFrameStatus?.home_ned_m)
    ? routeFrameStatus.home_ned_m.map(Number) : null;
  const currentPositionZ = positionNed && positionNed.length >= 3
    ? Number(positionNed[2]) : Number.NaN;
  const approvalFrameAltitude = frameHomeNed && positionNed
    && frameHomeNed.length >= 3 && positionNed.length >= 3
    && Number.isFinite(frameHomeNed[2]) && Number.isFinite(currentPositionZ)
    ? frameHomeNed[2]-currentPositionZ : null;
  const alignedApprovalAltitude = vehicle?.altitude_reference_valid === true
    && typeof vehicle.frame_altitude_home_relative_m === 'number'
    ? vehicle.frame_altitude_home_relative_m
    : (routeFrameStatus?.altitude_reference_valid === true
        || routeFrameStatus?.altitude_reference_diagnostics_available === true)
      && typeof routeFrameStatus.frame_altitude_home_relative_m === 'number'
      ? routeFrameStatus.frame_altitude_home_relative_m : null;
  const rawHomeOffset = approvalFrameAltitude !== null && homeAltitude !== null
    ? approvalFrameAltitude-homeAltitude : null;
  const altitudeReferenceRawDetail = vehicle?.altitude_reference_valid === true
    ? String(vehicle.altitude_reference_detail ?? 'ready')
    : routeFrameStatus?.altitude_reference_valid === true
      ? String(routeFrameStatus.altitude_reference_detail ?? 'ready')
      : String(routeFrameStatus?.altitude_reference_detail ?? 'aligning');
  const altitudeReferenceDetail = !isLowSpeedProfile(plan?.flight_profile)
    ? 'NOT REQUIRED'
    : !vehicleFresh ? 'STALE'
    : vehicle?.altitude_reference_valid === true
      || routeFrameStatus?.altitude_reference_valid === true
      ? 'READY'
      : altitudeReferenceRawDetail === 'altitude_reference_mismatch'
        || altitudeReferenceRawDetail === 'altitude_reference_epoch_changed'
        ? 'DIVERGED'
        : altitudeReferenceRawDetail === 'altitude_reference_stale'
          ? 'STALE'
          : altitudeReferenceRawDetail === 'altitude_reference_unstable'
            ? 'UNSTABLE'
          : 'ALIGNING';
  const altitudeAlignmentState = Number(vehicle?.altitude_alignment_state);
  const altitudeAlignmentSampleCount = Number(vehicle?.altitude_alignment_sample_count ?? 0);
  const altitudeAlignmentLabel = !vehicleFresh ? '오래된 상태 / 확인 불가'
    : vehicle?.altitude_alignment_ready === true
    ? '준비 완료'
    : altitudeAlignmentState === 4
      ? 'HOME 보정 확인 중'
    : altitudeAlignmentState === 2
      ? '오래됨'
      : altitudeAlignmentState === 3
        ? '불안정'
        : altitudeAlignmentSampleCount > 0
          ? '안정화 중'
          : '동기화 중';
  const homeCorrectionState = Number(vehicle?.home_correction_state ?? 0);
  const homeCorrectionLabel = homeCorrectionState === 1
    ? 'PENDING'
    : homeCorrectionState === 2
      ? 'APPLIED'
      : homeCorrectionState === 3
        ? 'REJECTED'
         : 'NONE';
  const homePhase = Number(vehicle?.home_phase ?? 0);
  const homePhaseLabel = homePhase === 1
    ? 'EXECUTION LOCKED'
    : homePhase === 2
      ? 'CORRECTION PENDING'
      : homePhase === 3
        ? 'REJECTED'
        : 'PROVISIONAL';
  const geoStale = vehicle?.geo_valid !== true || geoAgeMs > 750 || latitude === null || longitude === null;
  const forwardPlan = isForwardTestPlan(plan);
  const planLowSpeedAltitude = lowSpeedAltitudeForPlan(plan);
  const forwardSelectionMatches = !forwardPlan
    || planLowSpeedAltitude === selectedLowSpeedAltitude;
  const previewStart = forwardPlan && Array.isArray(plan?.preview_start_ned_m) ? plan.preview_start_ned_m.map(Number) : null;
  const previewEnd = forwardPlan && Array.isArray(plan?.preview_end_ned_m) ? plan.preview_end_ned_m.map(Number) : null;
  const previewCreated = typeof plan?.preview_created_unix_ms === 'number' ? plan.preview_created_unix_ms : 0;
  const previewRemaining = forwardPlan ? Math.max(0, 15 - (nowMs-previewCreated)/1000) : 0;
  const missionActive = Boolean(missionStatus?.mission_id)
    && ![0, 6, 7, 8].includes(typeof phase === 'number' ? phase : -1);
  const emergencyControlsReady = controlsAvailable && controlOwned
    && capabilitiesReady && capabilities?.emergency_control_version === 2
    && emergencyLandReady(vehicle, executionVehicleFresh);
  const lowSpeedBaseReady = lowSpeedCapabilitiesReady && preparationAvailable && controlOwned
    && vehicle?.command_output_enabled === true && vehicle?.connected === true
    && vehicle?.vehicle_status_fresh === true
    && vehicle?.preflight_checks_pass === true && vehicle?.position_valid === true
    && vehicle?.heading_fresh === true && !geoStale && vehicle?.low_speed_safety_valid === true
    && vehicle?.altitude_alignment_ready === true
    && vehicle?.arming_state_valid === true && vehicle?.landed_state_valid === true
    && !armed && landed && !missionActive;
  const cameraBypassEnabled = vehicle?.forward_test_camera_bypass_enabled === true;
  const cameraSafetyReady = vehicle?.low_speed_obstacle_guard_enabled === true
    && vehicle?.jetson_safety_fresh === true
    && vehicle?.jetson_safety_state === 'CLEAR';
  const forwardTestReady = lowSpeedBaseReady
    && (scenarioEnabled || cameraBypassEnabled || cameraSafetyReady);
  const lowSpeedReady = lowSpeedBaseReady && cameraSafetyReady;
  const forwardTestBlockedReason = !preparationAvailable ? (armingStateUnknownReason(vehicle, vehicleFresh, socketConnected, rosConnected) || '기체 상태 수신 대기')
    : vehicle?.position_valid !== true ? '현재 위치 확인 불가'
    : vehicle?.heading_fresh !== true ? '기수 방향 확인 불가'
    : vehicle?.altitude_alignment_ready !== true
    ? `고도 기준 ${altitudeAlignmentLabel}: ${String(vehicle?.altitude_alignment_detail ?? '동기화 상태를 기다리세요.')}`
    : String(vehicle?.low_speed_safety_detail ?? '비행 준비 조건을 확인하세요.');
  const resumeRequired = phase === 5 || phase === 13;
  approvalDispatchReady.current = controlsAvailable && controlOwned && finalApprovalReady
    && forwardTestReady && forwardSelectionMatches && !approvalPending && !approval && previewRemaining > 0;
  approvalFreshnessCheck.current = () => vehicleStateFresh(vehicle?.gateway_vehicle_age_ms,
    vehicleReceivedAt, performance.now(), socketConnected, rosConnected,
    vehicle?.connected === true, vehicle?.vehicle_status_fresh === true)
    && (!forwardPlan || Date.now()-previewCreated < 15000);
  const selectAltitude = (altitude: LowSpeedAltitude) => {
    if (missionActive || armed || approvalPending || approval?.accepted === true || executionStarted || altitude === selectedLowSpeedAltitude) return;
    preparationGeneration.current += 1;
    cancelApprovalHold();
    const id = proposalId(proposal);
    if (id) {
      retiredProposals.current.add(id);
      send({ type: 'mission.approve', proposal_id: id, approved: false, operator_id: OPERATOR_ID });
    }
    setProposal(null); setApproval(null); setApprovalPending(false); setRouteFrameStatus(null);
    setSelectedLowSpeedAltitude(altitude);
  };

  useEffect(() => {
    const timer = window.setInterval(() => setNowMs(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <Plane size={21} strokeWidth={1.8} />
          <div><strong>명령 대시보드</strong><span>FLIGHT OPERATIONS</span></div>
        </div>
        <div className="top-statuses">
          <StatusPill tone={socketConnected ? 'good' : socket.state === 'connecting' ? 'warn' : 'bad'}>
            {socketConnected ? <Link size={13} /> : <Link2Off size={13} />} HUB {socket.state.toUpperCase()}
          </StatusPill>
          <StatusPill tone={rosConnected ? 'good' : 'bad'}><Radio size={13} /> ROS {rosConnected ? 'ONLINE' : 'OFFLINE'}</StatusPill>
          <StatusPill tone={!fcStateFresh ? 'bad' : armed ? 'warn' : 'neutral'}>
            {armedStateLabel}{!fcStateFresh && unknownReason ? ` · ${unknownReason}` : ''}
          </StatusPill>
          <StatusPill tone={vehicleFresh && vehicle?.low_speed_safety_valid === true ? 'good' : 'warn'}><Gauge size={13} /> LOW-SPEED {!vehicleFresh ? 'STALE' : vehicle?.low_speed_safety_valid === true ? 'READY' : 'BLOCKED'}</StatusPill>
        </div>
        <button
          className={`control-claim-button ${controlOwned ? 'is-owned' : ''}`}
          type="button"
          disabled={!socketConnected || claimPending || controlOwned}
          onClick={claimControl}
        >
          {claimPending ? <LoaderCircle className="spin" size={15} /> : <ShieldCheck size={15} />}
          {controlOwned ? '운영권 보유' : claimPending ? '인수 중' : '운영권 가져오기'}
        </button>
        <button
          className="emergency-button emergency-stop-button"
          type="button"
          disabled={!emergencyControlsReady || vehicle?.manual_override === true}
          title="1초간 유지: 이동을 멈추고 현재 위치를 HOLD합니다. 모터 정지가 아닙니다."
          style={{ '--emergency-progress': `${emergencyHoldAction === 'STOP' ? emergencyHoldProgress * 100 : 0}%` } as React.CSSProperties}
          onPointerDown={() => startEmergencyHold('STOP', 1_000)}
          onPointerUp={cancelEmergencyHold}
          onPointerLeave={cancelEmergencyHold}
          onPointerCancel={cancelEmergencyHold}
          onKeyDown={(event) => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) startEmergencyHold('STOP', 1_000); }}
          onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') cancelEmergencyHold(); }}
        >
          <CircleStop size={15} />
          {emergencyHoldAction === 'STOP' ? '1초 유지' : vehicle?.manual_override === true ? '정지됨' : '긴급 정지'}
        </button>
        <button
          className="emergency-button emergency-land-button"
          type="button"
          disabled={!emergencyControlsReady}
          title="2초간 유지: 이동 정지 후 PX4 LAND 상태 머신을 실행합니다."
          style={{ '--emergency-progress': `${emergencyHoldAction === 'LAND' ? emergencyHoldProgress * 100 : 0}%` } as React.CSSProperties}
          onPointerDown={() => startEmergencyHold('LAND', 2_000)}
          onPointerUp={cancelEmergencyHold}
          onPointerLeave={cancelEmergencyHold}
          onPointerCancel={cancelEmergencyHold}
          onKeyDown={(event) => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) startEmergencyHold('LAND', 2_000); }}
          onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') cancelEmergencyHold(); }}
        >
          <PlaneLanding size={15} />
          {emergencyHoldAction === 'LAND' ? '2초 유지' : '긴급 착륙'}
        </button>
        <button className="route-open-button" type="button" onClick={() => setRoutePlannerOpen(true)}><RouteIcon size={15} /> 경로</button>
        <button
          className="icon-button"
          type="button"
          title="Gemini API 키 설정"
          aria-label="Gemini API 키 설정"
          onClick={() => {
            setRoutePlannerOpen(false);
            setKeyValue('');
            setKeyDialogOpen(true);
          }}
        >
          <KeyRound size={17} />
        </button>
        <button className="icon-button" title="연결 다시 시도" onClick={() => socket.reconnect()}><RefreshCw size={17} /></button>
      </header>

      {!preparationAvailable && (
        <div className="connection-banner" role="alert">
          <AlertTriangle size={17} />
          <span>{armingStateUnknownReason(vehicle, vehicleFresh, socketConnected, rosConnected) || '기체 상태 수신 대기'}</span>
          <button onClick={() => socket.reconnect()}>재연결</button>
        </div>
      )}

      {executionHealth?.stale === true && (
        <div className="connection-banner" role="alert" data-testid="execution-stale-banner">
          <AlertTriangle size={17} />
          <span>
            EXECUTION STATUS STALE — {String(executionHealth.detail ?? '실행 피드백이 3초 이상 없습니다.')}
            {' '}({objectValue(executionHealth.feedback_age_ms)} ms)
          </span>
        </div>
      )}

      <main className="console-grid">
        <section className="panel mission-panel">
          <div className="panel-heading"><div><MapPinned size={17} /><h2>임무 명령</h2></div><StatusPill tone={phase === 7 || phase === 8 ? 'bad' : phase ? 'good' : 'neutral'}>{phaseLabel(phase)}</StatusPill></div>
          <form className="command-form" onSubmit={submitText}>
            <textarea
              value={command}
              onChange={(event) => setCommand(event.target.value)}
              placeholder="순찰, 촬영 또는 경로 제작 명령 입력"
              rows={3}
              disabled={!socketConnected || textBusy}
            />
            <div className="command-actions">
              <button className="primary-button" type="submit" disabled={!command.trim() || !socketConnected || textBusy}>
                {textBusy ? <LoaderCircle className="spin" size={16} /> : <Send size={16} />}
                {textBusy ? '처리 중' : '전송'}
              </button>
              <button
                className={`ptt-button ${recording ? 'is-recording' : ''}`}
                type="button"
                disabled={!socketConnected || audioBusy}
                onPointerDown={(event) => { pttHeldRef.current = true; event.currentTarget.setPointerCapture(event.pointerId); void startRecording(); }}
                onPointerUp={stopRecording}
                onPointerCancel={stopRecording}
                onKeyDown={(event) => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) { pttHeldRef.current = true; void startRecording(); } }}
                onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') stopRecording(); }}
              >
                {audioBusy ? <LoaderCircle className="spin" size={16} /> : recording ? <Square size={15} fill="currentColor" /> : <Mic size={16} />}
                {audioBusy ? '처리 중' : recording ? '놓아서 전송' : '눌러서 말하기'}
              </button>
            </div>
          </form>

          <section className="forward-test-strip" aria-label={scenarioEnabled ? '통합 시나리오 시험' : '기수 방향 2미터 저속 시험'}>
            <div>
              <Navigation size={18} />
              <div><strong>{scenarioEnabled ? "장애물·투기 통합 시험" : "기수 방향 2 m 시험"}</strong><span>선택 목표 HOME {selectedLowSpeedAltitude.toFixed(1)} m · 현재 {homeAltitude === null ? '--' : `${homeAltitude.toFixed(2)} m`} · 예상 상승 {estimatedClimb === null ? '--' : `${estimatedClimb.toFixed(2)} m`}</span></div>
            </div>
            <div className="low-speed-altitude-selector" aria-label="저속 목표고도 선택">
              <button type="button" disabled={missionActive || armed || approvalPending || approval?.accepted === true || Boolean(executionStarted)} className={selectedLowSpeedAltitude === 1 ? 'active' : ''} onClick={() => selectAltitude(1)}>HOME 1 m</button>
              <button type="button" disabled={missionActive || armed || approvalPending || approval?.accepted === true || Boolean(executionStarted)} className={selectedLowSpeedAltitude === 2 ? 'active' : ''} onClick={() => selectAltitude(2)}>HOME 2 m</button>
            </div>
            <button type="button" disabled={!forwardTestReady || forwardPreparePending} onClick={prepareForwardTest} title={forwardTestReady ? '현재 위치와 기수각으로 미리보기 생성' : forwardTestBlockedReason}>
              {forwardPreparePending ? <LoaderCircle className="spin" size={15} /> : <MapPinned size={15} />}
              {forwardPreparePending ? '준비 중' : '시험 준비'}
            </button>
          </section>
          {scenarioEnabled && (
            <div className="result-box">
              <strong>탐색 10m + 마지막 1m · 최대 이동 범위 11m</strong><br />
              순항 HOME {selectedLowSpeedAltitude}m · 회피 상승 HOME {selectedLowSpeedAltitude + 2}m · 고도 상한 {selectedLowSpeedAltitude + 2.5}m · 통과 후 원고도 복귀<br />
              투기 검출 → 정지·사진 → 경로 복귀 → 1m 전진·착륙 · 끝 지점 미검출 대기 20초 · LOW 지속 5초 · Home 확인 2초<br />
              시험 전용: 2m 감지 · 전방 CLEAR까지 상승 후 3m 전진 · 입력 공백 정지 대기 5초 · 사진 저장 10초 · v5 고정 NED 경로<br />
              카메라는 프로그램 화면에서 직접 확인 후 시작하세요. 비행 중 깊이·검출 입력을 사용합니다.
            </div>
          )}
          {cameraBypassEnabled && !scenarioEnabled && (
            <div className="result-box error">
              <strong>카메라 장애물 보호 OFF</strong> — 이 예외는 기수 방향 2 m 시험에만 적용됩니다. 일반 저속 경로는 계속 차단됩니다.
            </div>
          )}

          <div className="section-divider" />
          <div className="subheading"><h3>계획 검토</h3><span>{proposalId(proposal) || 'NO PROPOSAL'}</span></div>
          {!proposal ? (
            <div className="empty-state"><MapPinned size={25} /><span>계획 대기 중</span></div>
          ) : (
            <>
              <div className="raw-command">{String(proposal.raw_command ?? proposal.command ?? '명령 원문 없음')}</div>
              {isLowSpeedProfile(plan?.flight_profile) && planLowSpeedAltitude !== null && (
                <div className="low-speed-profile-banner"><Gauge size={17} /><div><strong>{String(plan?.flight_profile)}</strong><span>실행 경로 전체 HOME {planLowSpeedAltitude.toFixed(1)} m · 이동 수평/수직 0.5 m/s · 상한 {(planLowSpeedAltitude+0.5).toFixed(1)} m · PX4 LAND 0.6 m/s</span></div></div>
              )}
              <div className="result-box" aria-label="사건 대응 정책">
                <div>사건 후 동작: <strong>{objectValue(plan?.after_response)}</strong></div>
                {typeof proposal.message === 'string' && proposal.message.trim()
                  ? <div>{proposal.message}</div>
                  : null}
              </div>
              <dl className="plan-grid">
                {planRows.map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{objectValue(value)}</dd></div>)}
              </dl>
              <div className={`result-box ${routeFrameReady ? 'success' : 'error'}`}>
                <div>좌표 프레임: <strong>{['CONSUMED', 'COMPLETED'].includes(String(routeFrameStatus?.preview_state))
                  ? String(routeFrameStatus?.preview_state) : routeFrameReady ? 'VALID' : 'BLOCKED'}</strong></div>
                <div>Home WGS84: {routeFrameReady ? objectValue(routeFrameStatus?.home_wgs84) : '--'}</div>
                <div>Home NED: {routeFrameReady ? objectValue(routeFrameStatus?.home_ned_m) : '--'}</div>
                <div>정렬 Home Z: {routeFrameStatus?.altitude_reference_diagnostics_available === true ? `${objectValue(routeFrameStatus?.aligned_home_z_ned_m)} m` : '--'}</div>
                <div>고도 기준: <strong>{altitudeReferenceDetail}</strong></div>
                <div>FC RAW / NORMALIZED / 정렬 HOME ALT: {routeFrameStatus?.altitude_reference_diagnostics_available === true ? `${objectValue(routeFrameStatus?.fc_altitude_home_relative_m)} m / ${objectValue(routeFrameStatus?.normalized_fc_altitude_home_relative_m)} m / ${objectValue(routeFrameStatus?.frame_altitude_home_relative_m)} m` : '--'}</div>
                <div>PX4 HOME 보정: {routeFrameStatus ? `${objectValue(routeFrameStatus.home_altitude_correction_m)} m / ${titleCase(routeFrameStatus.home_correction_detail)}` : '--'}</div>
                <div>HOME phase: {routeFrameStatus ? objectValue(routeFrameStatus.home_phase) : '--'} / lock {routeFrameStatus?.execution_home_lock_valid === true ? 'VALID' : 'OPEN'}</div>
                <div>고도 기준 오차: {routeFrameStatus?.altitude_reference_error_valid === true ? `${objectValue(routeFrameStatus?.altitude_reference_error_m)} m` : '--'}</div>
                <div>첫 목표 NED: {routeFrameReady ? objectValue(routeFrameStatus?.first_target_ned_m) : '--'}</div>
                <div>첫 구간 / 최대 구간: {routeFrameReady ? `${objectValue(routeFrameStatus?.first_leg_m)} m / ${objectValue(routeFrameStatus?.maximum_leg_m)} m` : '--'}</div>
                {!routeFrameReady && <div>{String(routeFrameStatus?.reason ?? '검증 결과 대기 중')}</div>}
              </div>
              {plan ? (
                <details className="json-details"><summary>PLAN JSON</summary><pre>{JSON.stringify(plan, null, 2)}</pre></details>
              ) : (
                <div className="result-box error">계획 JSON을 생성하지 못했습니다. 오류 메시지를 확인한 뒤 명령을 다시 전송하세요.</div>
              )}
              {plan && !isLowSpeedProfile(plan.flight_profile) && Array.isArray(plan.route_waypoints_enu) && (
                <button className="low-speed-copy-button" type="button" disabled={!lowSpeedCapabilitiesReady || approvalPending || Boolean(approval)} onClick={prepareLowSpeedCopy}><Gauge size={15} /> 이 경로의 {selectedLowSpeedAltitude} m 저속 실행 사본 만들기</button>
              )}
              <div className={`review-hint ${controlOwned && finalApprovalReady ? 'is-ready' : ''}`}>{reviewHint}</div>
              <div className="review-actions">
                <button title={reviewHint} className="danger-secondary" disabled={!controlsAvailable || !controlOwned || !proposalReady || approvalPending || Boolean(approval)} onClick={() => approve(false)}><X size={16} /> 거절</button>
                {forwardPlan ? (
                  <button
                    title={previewRemaining > 0 ? '3초간 계속 누르면 승인 후 즉시 실행됩니다.' : '미리보기가 만료되었습니다.'}
                    className="hold-to-run-button"
                    style={{ '--hold-progress': `${holdProgress * 100}%` } as React.CSSProperties}
                    disabled={!controlsAvailable || !controlOwned || !finalApprovalReady || !forwardTestReady || !forwardSelectionMatches || approvalPending || Boolean(approval) || previewRemaining <= 0}
                    onPointerDown={startApprovalHold}
                    onPointerUp={cancelApprovalHold}
                    onPointerLeave={cancelApprovalHold}
                    onPointerCancel={cancelApprovalHold}
                    onKeyDown={(event) => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) startApprovalHold(); }}
                    onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') cancelApprovalHold(); }}
                  ><Navigation size={16} /> {holdProgress > 0 ? `${Math.ceil((1-holdProgress)*3)}초 유지` : `3초 눌러 실행 · ${previewRemaining.toFixed(1)}초`}</button>
                ) : (
                  <button title={reviewHint} className="primary-button" disabled={!controlsAvailable || !controlOwned || !finalApprovalReady || (isLowSpeedProfile(plan?.flight_profile) && !lowSpeedReady) || approvalPending || Boolean(approval)} onClick={() => approve(true)}><Check size={16} /> 승인</button>
                )}
              </div>
            </>
          )}
        </section>

        <section className="panel telemetry-panel">
          <div className="panel-heading"><div><Wifi size={17} /><h2>기체 상태</h2></div><span className="updated-label">{vehicleFresh ? 'LIVE' : 'STALE · 마지막 관측값'}</span></div>
          <LivePositionMap latitude={latitude} longitude={longitude} headingRad={headingRad} stale={geoStale} previewStartNed={previewStart} previewEndNed={previewEnd} />
          <div className="telemetry-grid">
            <Metric label="FLIGHT MODE" value={firstValue(vehicle, ['nav_state', 'flight_mode', 'mode'])} />
            <Metric label="BATTERY" value={firstValue(vehicle, ['battery_percent', 'battery', 'remaining'])} suffix={typeof firstValue(vehicle, ['battery_percent']) === 'number' ? '%' : ''} />
            <Metric label="POSITION" value={firstValue(vehicle, ['position_valid', 'local_position_valid'])} />
            <Metric label="OFFBOARD" value={firstValue(vehicle, ['offboard_active', 'offboard'])} />
            <Metric label="NORTH X" value={positionNed[0]} suffix=" m" />
            <Metric label="EAST Y" value={positionNed[1]} suffix=" m" />
            <Metric label="DOWN Z" value={positionNed[2]} suffix=" m" />
            <Metric label="LATITUDE" value={latitude} suffix="°" />
            <Metric label="LONGITUDE" value={longitude} suffix="°" />
            <Metric label="FC HOME ALT (RAW) — AGL 아님" value={firstValue(vehicle, ['fc_altitude_home_relative_m', 'altitude_home_relative_m'])} suffix=" m" />
            <Metric label="고정 HOME 기준 ALT (NORMALIZED)" value={firstValue(vehicle, ['normalized_fc_altitude_home_relative_m'])} suffix=" m" />
            <Metric label="PX4 HOME 보정량" value={firstValue(vehicle, ['home_altitude_correction_m'])} suffix=" m" />
            <Metric label="HOME 보정 상태" value={homeCorrectionLabel} />
            <Metric label="HOME PHASE" value={homePhaseLabel} />
            <Metric label="HOME LOCK" value={vehicle?.execution_home_lock_valid === true ? 'VALID' : 'OPEN'} />
            <Metric label="정렬된 승인 HOME ALT" value={alignedApprovalAltitude} suffix=" m" />
            <Metric label="원본 HOME/NED 오프셋" value={rawHomeOffset} suffix=" m" />
            <Metric label="고도 기준 상태" value={altitudeReferenceDetail} />
            <Metric label="고도 동기화" value={altitudeAlignmentLabel} />
            <Metric label="동기화 표본" value={firstValue(vehicle, ['altitude_alignment_sample_count'])} />
            <Metric label="안정화 시간" value={typeof vehicle?.altitude_alignment_window_ms === 'number' ? vehicle.altitude_alignment_window_ms / 1000 : null} suffix=" s" />
            <Metric label="정렬 후보 범위" value={firstValue(vehicle, ['altitude_alignment_candidate_span_m'])} suffix=" m" />
            <Metric label="FC 시각 오차" value={firstValue(vehicle, ['altitude_alignment_source_skew_ms'])} suffix=" ms" />
            <Metric label="AMSL ALT" value={firstValue(vehicle, ['altitude_amsl_m'])} suffix=" m" />
            <Metric label="HEADING" value={headingRad === null ? null : headingRad * 180 / Math.PI} suffix="°" />
            <Metric label="AUTHORITY" value={firstValue(vehicle, ['active_authority'])} />
            <Metric label="FLIGHT OUTPUT" value={String(vehicle?.flight_output_detail ?? '').startsWith('battery_')
              ? ({ battery_low_observing: 'LOW 관찰 중 · 5초 지속 시 착륙', battery_low_recovery_observing: '배터리 회복 확인 중 · 1초',
                  battery_low_recovered: '배터리 LOW 회복', battery_health_terminal_only: '배터리 경고 · 제동/LAND 인계',
                  battery_terminal_handoff_unavailable: '배터리 착륙 인계 불가' } as Record<string, string>)[String(vehicle?.flight_output_detail)] || String(vehicle?.flight_output_detail)
              : vehicle?.flight_output_ready === true ? 'READY' : String(vehicle?.flight_output_detail ?? '--')} />
            <Metric label="JETSON SAFETY" value={firstValue(vehicle, ['jetson_safety_state']) || (firstValue(vehicle, ['jetson_safety_fresh']) === true ? 'CLEAR' : '--')} />
          </div>
          <div className="section-divider" />
          <div className="subheading"><h3>임무 진행</h3><span>{String(missionStatus?.mission_id ?? '--')}</span></div>
          <dl className="status-list">
            <div><dt>PHASE</dt><dd>{phaseLabel(phase)}</dd></div>
            <div><dt>WAYPOINT</dt><dd>{waypointLabel}</dd></div>
            <div><dt>PROGRESS</dt><dd>{progressLabel}</dd></div>
            <div><dt>MESSAGE</dt><dd>{objectValue(firstValue(missionStatus, ['message', 'detail']))}</dd></div>
          </dl>
          {resumeRequired && (
            <button
              className="execute-button"
              type="button"
              disabled={!controlsAvailable || !controlOwned || !missionStatus?.mission_id}
              onClick={resumeMission}
            >
              <RefreshCw size={16} /> 자율 임무 재개
            </button>
          )}
          {execution && <div className={`result-box ${execution.success ? 'success' : 'error'}`}>{String(execution.message ?? execution.final_phase ?? '실행 결과 수신')}</div>}
        </section>
      </main>

      <div className="notice-stack" aria-live="polite">
        {notices.map((notice) => <div key={notice.id} className={`notice notice-${notice.tone}`}>{notice.tone === 'error' ? <AlertTriangle size={16} /> : <Check size={16} />}{notice.message}</div>)}
      </div>

      <RoutePlanner
        apiBase={apiBase}
        open={routePlannerOpen}
        plan={routeDraftPlan ?? plan}
        socketConnected={socketConnected}
        lowSpeedAvailable={lowSpeedCapabilitiesReady}
        importPlanRequest={routeDraftRequest}
        onPlanImported={() => setRouteDraftPlan(null)}
        onClose={() => setRoutePlannerOpen(false)}
        onPrepare={(rawCommand, routePlan) => send({ type: 'mission.propose', raw_command: rawCommand, plan: routePlan })}
        onNotice={addNotice}
      />

      {keyDialogOpen && (
        <div className="settings-backdrop" role="presentation" onMouseDown={() => setKeyDialogOpen(false)}>
          <section
            className="settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="gemini-key-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div><KeyRound size={18} /><h2 id="gemini-key-title">Gemini API 키</h2></div>
              <button className="icon-button" type="button" title="닫기" aria-label="닫기" onClick={() => setKeyDialogOpen(false)}><X size={17} /></button>
            </header>
            <form onSubmit={saveGeminiKey}>
              <div className={`key-status ${keyConfigured ? 'is-configured' : ''}`}>
                {keyConfigured === null ? '상태 확인 중' : keyConfigured ? '현재 키가 설정되어 있습니다.' : 'Gemini API 키가 필요합니다.'}
              </div>
              <label htmlFor="gemini-api-key">새 API 키</label>
              <input
                id="gemini-api-key"
                type="password"
                autoComplete="off"
                autoFocus
                value={keyValue}
                onChange={(event) => setKeyValue(event.target.value)}
                placeholder="Google AI Studio API 키 입력"
                minLength={8}
                maxLength={512}
              />
              <p>보안을 위해 기존 키는 표시하지 않습니다. 저장한 키는 현재 Windows 사용자에게만 적용됩니다.</p>
              {keyError && <div className="settings-error" role="alert">{keyError}</div>}
              <div className="settings-actions">
                <button className="danger-secondary" type="button" onClick={() => setKeyDialogOpen(false)}>취소</button>
                <button className="primary-button" type="submit" disabled={keyValue.trim().length < 8 || keySaving}>
                  {keySaving ? <LoaderCircle className="spin" size={16} /> : <Check size={16} />}
                  {keySaving ? '저장 중' : '저장'}
                </button>
              </div>
            </form>
          </section>
        </div>
      )}

    </div>
  );
}

function Metric({ label, value, suffix = '' }: { label: string; value: unknown; suffix?: string }) {
  const rendered = objectValue(value);
  return <div className="metric"><span>{label}</span><strong>{rendered}{rendered === '--' ? '' : suffix}</strong></div>;
}
