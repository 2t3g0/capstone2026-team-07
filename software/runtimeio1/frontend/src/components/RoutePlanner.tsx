import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CircleMarker,
  MapContainer,
  Marker,
  Pane,
  Polygon,
  Polyline,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet';
import { divIcon } from 'leaflet';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  ArrowUpDown,
  Camera,
  CopyPlus,
  Layers2,
  LoaderCircle,
  Gauge,
  MapPinned,
  MousePointer2,
  PencilRuler,
  Plus,
  RotateCcw,
  Redo2,
  Save,
  Send,
  Trash2,
  Undo2,
  Waypoints,
  WandSparkles,
  X,
} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import {
  appendFreeWaypoint,
  appendRoadGeometry,
  planCaptureActions,
  planRouteEnuToGeo,
  planRouteToEditableWaypoints,
  routeToMissionPlan,
  type RoadSegment,
  type RouteCatalog,
  type RouteWaypoint,
  type StoredRoute,
} from '../lib/routes';
import type { LowSpeedAltitude } from '../lib/flightProfiles';

interface RoutePlannerProps {
  apiBase: string;
  open: boolean;
  plan: Record<string, unknown> | null;
  socketConnected: boolean;
  lowSpeedAvailable: boolean;
  onClose: () => void;
  onPrepare: (rawCommand: string, plan: object) => boolean;
  importPlanRequest?: number;
  onPlanImported?: () => void;
  onNotice: (message: string, tone?: 'info' | 'error' | 'success') => void;
}

type EditMode = 'road' | 'point';
type RouteLayerMode = 'plan' | 'edit' | 'compare';

const captureMarkerIcon = divIcon({
  className: 'plan-capture-marker',
  html: renderToStaticMarkup(<Camera aria-hidden="true" size={15} strokeWidth={2.2} />),
  iconAnchor: [14, 14],
  iconSize: [28, 28],
});

function MapClick({ active, onPoint }: { active: boolean; onPoint: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(event) {
      if (active) onPoint(event.latlng.lat, event.latlng.lng);
    },
  });
  return null;
}

function PlanRouteViewport({ points }: { points: { latitude_deg: number; longitude_deg: number }[] | null }) {
  const map = useMap();
  useEffect(() => {
    if (!points?.length) return;
    map.invalidateSize();
    map.fitBounds(
      points.map((point): [number, number] => [point.latitude_deg, point.longitude_deg]),
      { padding: [28, 28], maxZoom: 17 },
    );
  }, [map, points]);
  return null;
}

export function RoutePlanner({
  apiBase,
  open,
  plan,
  socketConnected,
  lowSpeedAvailable,
  onClose,
  onPrepare,
  importPlanRequest = 0,
  onPlanImported,
  onNotice,
}: RoutePlannerProps) {
  const [catalog, setCatalog] = useState<RouteCatalog | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [waypoints, setWaypoints] = useState<RouteWaypoint[]>([]);
  const [history, setHistory] = useState<RouteWaypoint[][]>([]);
  const [future, setFuture] = useState<RouteWaypoint[][]>([]);
  const waypointsRef = useRef<RouteWaypoint[]>([]);
  const historyRef = useRef<RouteWaypoint[][]>([]);
  const futureRef = useRef<RouteWaypoint[][]>([]);
  const importedPlanRequestRef = useRef(0);
  const [altitude, setAltitude] = useState(15);
  const [laps, setLaps] = useState(1);
  const [mode, setMode] = useState<EditMode>('point');
  const [showVehicle, setShowVehicle] = useState(true);
  const [showPedestrian, setShowPedestrian] = useState(false);
  const [routeLayerMode, setRouteLayerMode] = useState<RouteLayerMode>('plan');
  const [busy, setBusy] = useState(false);
  const [lowSpeedAltitude, setLowSpeedAltitude] = useState<LowSpeedAltitude | null>(null);

  const loadCatalog = useCallback(async () => {
    setBusy(true);
    try {
      const response = await fetch(`${apiBase}/v1/routes`);
      if (!response.ok) throw new Error(`경로 목록 요청 실패 (${response.status})`);
      setCatalog(await response.json() as RouteCatalog);
    } catch (error) {
      onNotice(error instanceof Error ? error.message : '경로 목록을 불러오지 못했습니다.', 'error');
    } finally {
      setBusy(false);
    }
  }, [apiBase, onNotice]);

  useEffect(() => {
    if (open) void loadCatalog();
  }, [loadCatalog, open]);

  const replaceWaypoints = useCallback((next: RouteWaypoint[]) => {
    waypointsRef.current = next;
    setWaypoints(next);
  }, []);

  const resetEditHistory = useCallback((next: RouteWaypoint[]) => {
    historyRef.current = [];
    futureRef.current = [];
    setHistory([]);
    setFuture([]);
    replaceWaypoints(next);
  }, [replaceWaypoints]);

  const commitWaypoints = useCallback((next: RouteWaypoint[]) => {
    const snapshots = [...historyRef.current.slice(-49), waypointsRef.current];
    historyRef.current = snapshots;
    futureRef.current = [];
    setHistory(snapshots);
    setFuture([]);
    replaceWaypoints(next);
  }, [replaceWaypoints]);

  const undoRouteEdit = useCallback(() => {
    if (!historyRef.current.length) return;
    const previous = historyRef.current[historyRef.current.length - 1];
    const nextHistory = historyRef.current.slice(0, -1);
    const nextFuture = [waypointsRef.current, ...futureRef.current].slice(0, 50);
    historyRef.current = nextHistory;
    futureRef.current = nextFuture;
    setHistory(nextHistory);
    setFuture(nextFuture);
    replaceWaypoints(previous);
  }, [replaceWaypoints]);

  const redoRouteEdit = useCallback(() => {
    if (!futureRef.current.length) return;
    const next = futureRef.current[0];
    const nextFuture = futureRef.current.slice(1);
    const nextHistory = [...historyRef.current.slice(-49), waypointsRef.current];
    historyRef.current = nextHistory;
    futureRef.current = nextFuture;
    setHistory(nextHistory);
    setFuture(nextFuture);
    replaceWaypoints(next);
  }, [replaceWaypoints]);

  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      const target = event.target;
      if (
        target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || (target instanceof HTMLElement && target.isContentEditable)
      ) return;
      const key = event.key.toLowerCase();
      if (key === 'z' && !event.shiftKey) {
        event.preventDefault();
        undoRouteEdit();
      } else if (key === 'y' || (key === 'z' && event.shiftKey)) {
        event.preventDefault();
        redoRouteEdit();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, redoRouteEdit, undoRouteEdit]);

  const visibleSegments = useMemo(() => catalog?.segments.filter((segment) => (
    (segment.category === 'vehicle' && showVehicle)
    || (segment.category === 'pedestrian' && showPedestrian)
  )) ?? [], [catalog, showPedestrian, showVehicle]);

  const planRouteValue = plan?.route_waypoints_enu;
  const planRoute = useMemo(
    () => catalog ? planRouteEnuToGeo(planRouteValue, catalog.reference) : null,
    [catalog, planRouteValue],
  );
  const captureActions = useMemo(
    () => planCaptureActions(plan?.route_actions, planRoute?.length ?? 0),
    [plan?.route_actions, planRoute?.length],
  );
  const planRouteState = planRouteValue === undefined || (Array.isArray(planRouteValue) && planRouteValue.length === 0)
    ? 'empty'
    : planRoute === null
      ? 'invalid'
      : 'ready';
  const hasPlanRoute = planRouteState === 'ready';
  const hasEditRoute = waypoints.length > 0;
  const showPlanRoute = hasPlanRoute && (routeLayerMode === 'plan' || routeLayerMode === 'compare');
  const showEditRoute = hasEditRoute && (routeLayerMode === 'edit' || routeLayerMode === 'compare');
  const mapFocusPoints = showPlanRoute && showEditRoute
    ? [...(planRoute ?? []), ...waypoints]
    : showPlanRoute
      ? planRoute
      : showEditRoute
        ? waypoints
        : null;

  useEffect(() => {
    if (hasPlanRoute) setRouteLayerMode('plan');
  }, [hasPlanRoute, plan?.route_id]);

  const importCurrentPlan = useCallback(() => {
    if (!planRoute?.length) return;
    const routeName = typeof plan?.route_name === 'string' && plan.route_name.trim()
      ? plan.route_name.trim()
      : '현재 계획 경로';
    const routeId = typeof plan?.route_id === 'string' ? plan.route_id : '';
    const limit = plan?.patrol_limit;
    const limitValue = limit && typeof limit === 'object' && !Array.isArray(limit)
      ? (limit as Record<string, unknown>).value
      : null;
    setSelectedId(null);
    setName(routeName.slice(0, 80));
    setDescription(routeId ? `계획 ${routeId}에서 가져온 경로` : '현재 계획에서 가져온 경로');
    resetEditHistory(planRouteToEditableWaypoints(planRoute));
    setAltitude(planRoute[0]?.up_m ?? 15);
    setLaps(typeof limitValue === 'number' && Number.isInteger(limitValue)
      ? Math.max(1, Math.min(100, limitValue))
      : 1);
    setMode('point');
    setRouteLayerMode('edit');
    onNotice('현재 계획을 새 경로 초안으로 가져왔습니다. 촬영 동작은 경로 저장에 포함되지 않습니다.', 'success');
    onPlanImported?.();
  }, [onNotice, onPlanImported, plan, planRoute, resetEditHistory]);

  useEffect(() => {
    if (
      !open
      || importPlanRequest <= 0
      || importedPlanRequestRef.current === importPlanRequest
      || !planRoute?.length
    ) return;
    importedPlanRequestRef.current = importPlanRequest;
    importCurrentPlan();
  }, [importCurrentPlan, importPlanRequest, open, planRoute]);

  if (!open) return null;

  const selectedRoute = catalog?.custom_routes.find((route) => route.id === selectedId) ?? null;
  const center: [number, number] = catalog
    ? [catalog.reference.latitude_deg, catalog.reference.longitude_deg]
    : [35.2350126, 129.0748631];

  const resetDraft = () => {
    setSelectedId(null);
    setName('새 순찰 경로');
    setDescription('');
    resetEditHistory([]);
    setAltitude(15);
    setLaps(1);
    setRouteLayerMode(hasPlanRoute ? 'plan' : 'edit');
  };

  const selectRoute = (route: StoredRoute) => {
    setSelectedId(route.id);
    setName(route.name);
    setDescription(route.description);
    resetEditHistory(route.waypoints);
    setAltitude(route.waypoints[0]?.altitude_m ?? 15);
    setLaps(1);
    setRouteLayerMode('edit');
  };

  const appendSegment = (segment: RoadSegment) => {
    if (mode !== 'road') return;
    const current = waypointsRef.current;
    const updated = appendRoadGeometry(current, segment.geometry, altitude);
    if (updated.length === current.length) return;
    if (updated.length === 300) onNotice('웨이포인트 최대 300개에 도달했습니다.', 'error');
    commitWaypoints(updated);
    setRouteLayerMode('edit');
  };

  const appendPoint = (latitude_deg: number, longitude_deg: number) => {
    if (waypointsRef.current.length >= 300) {
      onNotice('웨이포인트 최대 300개에 도달했습니다.', 'error');
      return;
    }
    const updated = appendFreeWaypoint(waypointsRef.current, {
      latitude_deg,
      longitude_deg,
      altitude_m: altitude,
    });
    if (updated !== waypointsRef.current) {
      commitWaypoints(updated);
      setRouteLayerMode('edit');
    }
  };

  const payload = () => ({ name, description, waypoints });

  const saveRoute = async () => {
    if (!name.trim() || waypoints.length < 2 || busy) return;
    setBusy(true);
    try {
      const response = await fetch(
        selectedRoute ? `${apiBase}/v1/routes/${selectedRoute.id}` : `${apiBase}/v1/routes`,
        {
          method: selectedRoute ? 'PUT' : 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(selectedRoute
            ? { ...payload(), revision: selectedRoute.revision }
            : payload()),
        },
      );
      if (!response.ok) {
        const detail = await response.json().catch(() => null) as { detail?: string } | null;
        throw new Error(detail?.detail || `경로 저장 실패 (${response.status})`);
      }
      const saved = await response.json() as StoredRoute;
      await loadCatalog();
      setSelectedId(saved.id);
      setName(saved.name);
      setDescription(saved.description);
      resetEditHistory(saved.waypoints);
      onNotice('경로를 저장했습니다.', 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : '경로 저장 실패', 'error');
    } finally {
      setBusy(false);
    }
  };

  const deleteRoute = async () => {
    if (!selectedRoute || busy) return;
    setBusy(true);
    try {
      const response = await fetch(
        `${apiBase}/v1/routes/${selectedRoute.id}?revision=${selectedRoute.revision}`,
        { method: 'DELETE' },
      );
      if (!response.ok) throw new Error(`경로 삭제 실패 (${response.status})`);
      resetDraft();
      await loadCatalog();
      onNotice('경로를 삭제했습니다.', 'success');
    } catch (error) {
      onNotice(error instanceof Error ? error.message : '경로 삭제 실패', 'error');
    } finally {
      setBusy(false);
    }
  };

  const prepareMission = () => {
    if (!selectedRoute || !socketConnected) return;
    const rawCommand = lowSpeedAltitude !== null
      ? `${selectedRoute.name} 등록 경로를 Home 기준 ${lowSpeedAltitude} m 저속으로 순찰 후 마지막 지점 착륙`
      : `${selectedRoute.name} 등록 경로를 ${laps}바퀴 순찰`;
    if (onPrepare(rawCommand, routeToMissionPlan(selectedRoute, laps, lowSpeedAltitude))) {
      onNotice(lowSpeedAltitude !== null
        ? `${lowSpeedAltitude} m 저속 실행 사본을 계획 검토로 보냈습니다.`
        : '등록 경로를 계획 검토로 보냈습니다.', 'success');
      onClose();
    }
  };

  return (
    <div className="route-modal" role="dialog" aria-modal="true" aria-label="순찰 경로 편집">
      <header className="route-modal-header">
        <div><MapPinned size={18} /><strong>부산캠퍼스 경로</strong><span>{catalog?.segments.length ?? 0} ROAD SEGMENTS</span></div>
        <button className="icon-button" title="닫기" aria-label="경로 편집 닫기" onClick={onClose}><X size={18} /></button>
      </header>
      <div className="route-workspace">
        <aside className="route-sidebar">
          <div className="route-sidebar-toolbar">
            <h2>등록 경로</h2>
            <button className="route-icon-button" title="새 경로" onClick={resetDraft}><Plus size={16} /></button>
          </div>
          <div className="route-list">
            {catalog?.custom_routes.map((route) => (
              <button key={route.id} className={selectedId === route.id ? 'active' : ''} onClick={() => selectRoute(route)}>
                <strong>{route.name}</strong><span>{route.waypoints.length} WP · R{route.revision}</span>
              </button>
            ))}
            {!busy && catalog?.custom_routes.length === 0 && <div className="route-empty">NO SAVED ROUTES</div>}
          </div>
          <div className="route-form">
            <button
              className="route-import-plan-button"
              disabled={planRouteState !== 'ready' || busy}
              onClick={importCurrentPlan}
            >
              <CopyPlus size={15} /> 현재 계획 가져오기
            </button>
            <label><span>경로명</span><input value={name} maxLength={80} onChange={(event) => setName(event.target.value)} /></label>
            <label><span>설명</span><input value={description} maxLength={300} onChange={(event) => setDescription(event.target.value)} /></label>
            <div className="route-numbers">
              <label><span>고도</span><input type="number" min="1" max="120" step="1" value={altitude} onChange={(event) => setAltitude(Number(event.target.value))} /></label>
              <label><span>반복</span><input type="number" min="1" max="100" step="1" value={laps} onChange={(event) => setLaps(Number(event.target.value))} /></label>
            </div>
            <div className="route-stats"><span>WAYPOINTS</span><strong>{waypoints.length} / 300</strong></div>
            <div className="flight-profile-selector">
              <button className={lowSpeedAltitude === null ? 'active' : ''} onClick={() => setLowSpeedAltitude(null)}>NORMAL</button>
              <button className={lowSpeedAltitude === 1 ? 'active low' : ''} disabled={!lowSpeedAvailable} title={lowSpeedAvailable ? 'HOME 기준 1 m 저속 프로필' : 'Jetson 저속 프로토콜 v2가 필요합니다.'} onClick={() => setLowSpeedAltitude(1)}><Gauge size={13} /> 저속 1 m</button>
              <button className={lowSpeedAltitude === 2 ? 'active low' : ''} disabled={!lowSpeedAvailable} title={lowSpeedAvailable ? 'HOME 기준 2 m 저속 프로필' : 'Jetson 저속 프로토콜 v2가 필요합니다.'} onClick={() => setLowSpeedAltitude(2)}><Gauge size={13} /> 저속 2 m</button>
            </div>
            {lowSpeedAltitude !== null && <div className="low-speed-note">HOME 기준 {lowSpeedAltitude.toFixed(1)} m · 이동 수평/수직 0.5 m/s · 상한 {(lowSpeedAltitude + 0.5).toFixed(1)} m · PX4 LAND 0.6 m/s</div>}
            <div className="route-edit-actions">
              <button title="이전 편집 취소 (Ctrl+Z)" disabled={!history.length} onClick={undoRouteEdit}><Undo2 size={15} /></button>
              <button title="편집 다시 실행 (Ctrl+Shift+Z)" disabled={!future.length} onClick={redoRouteEdit}><Redo2 size={15} /></button>
              <button title="경로 역순" disabled={waypoints.length < 2} onClick={() => commitWaypoints([...waypoints].reverse())}><ArrowUpDown size={15} /></button>
              <button title="경로 비우기" disabled={!waypoints.length} onClick={() => commitWaypoints([])}><RotateCcw size={15} /></button>
            </div>
            <div className="route-save-actions">
              <button className="danger-secondary" disabled={!selectedRoute || busy} onClick={() => void deleteRoute()}><Trash2 size={15} /> 삭제</button>
              <button className="primary-button" disabled={busy || !name.trim() || waypoints.length < 2} onClick={() => void saveRoute()}>
                {busy ? <LoaderCircle className="spin" size={15} /> : <Save size={15} />} 저장
              </button>
            </div>
            <button className={`route-prepare-button ${lowSpeedAltitude !== null ? 'is-low-speed' : ''}`} disabled={!selectedRoute || !socketConnected || busy} onClick={prepareMission}><Send size={15} /> {lowSpeedAltitude !== null ? '저속 임무로 준비' : '임무로 준비'}</button>
          </div>
        </aside>
        <section className="route-map-shell">
          <div className="route-map-toolbar">
            <div className="segmented">
              <button className={mode === 'point' ? 'active' : ''} onClick={() => setMode('point')}><Waypoints size={14} /> 자유 경로</button>
              <button className={mode === 'road' ? 'active' : ''} onClick={() => setMode('road')}><MousePointer2 size={14} /> 도로 구간</button>
            </div>
            <label><input type="checkbox" checked={showVehicle} onChange={(event) => setShowVehicle(event.target.checked)} /> 차량도로</label>
            <label><input type="checkbox" checked={showPedestrian} onChange={(event) => setShowPedestrian(event.target.checked)} /> 보행로</label>
            <div className="segmented route-layer-segmented" aria-label="경로 표시 모드">
              <button title="현재 계획만 표시" className={routeLayerMode === 'plan' ? 'active plan-layer' : ''} disabled={!hasPlanRoute} onClick={() => setRouteLayerMode('plan')}><WandSparkles size={14} /> 현재 계획</button>
              <button title="등록 또는 편집 중인 경로만 표시" className={routeLayerMode === 'edit' ? 'active edit-layer' : ''} disabled={!hasEditRoute} onClick={() => setRouteLayerMode('edit')}><PencilRuler size={14} /> 편집 경로</button>
              <button title="두 경로를 함께 표시" className={routeLayerMode === 'compare' ? 'active compare-layer' : ''} disabled={!hasPlanRoute || !hasEditRoute} onClick={() => setRouteLayerMode('compare')}><Layers2 size={14} /> 비교</button>
            </div>
            <span className={`plan-route-status is-${routeLayerMode}`}>
              {routeLayerMode === 'compare' && hasPlanRoute && hasEditRoute
                ? `계획 ${planRoute?.length ?? 0} / 편집 ${waypoints.length} WP`
                : showPlanRoute
                  ? `현재 계획 ${planRoute?.length ?? 0} WP`
                  : showEditRoute
                    ? `편집 경로 ${waypoints.length} WP`
                    : planRouteState === 'invalid' ? '현재 계획 경로 오류' : '표시할 경로 없음'}
            </span>
          </div>
          <MapContainer className={`route-map ${mode === 'point' ? 'is-point-mode' : ''}`} center={center} zoom={16} zoomControl doubleClickZoom={false}>
            <PlanRouteViewport points={mapFocusPoints} />
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            {catalog && <Polygon positions={catalog.geofence_geo} pathOptions={{ color: '#ef6868', weight: 2, fillOpacity: 0.03, interactive: false }} />}
            {visibleSegments.map((segment) => (
              <Polyline
                key={segment.id}
                positions={segment.geometry}
                pathOptions={{
                  color: segment.category === 'vehicle' ? '#edb84f' : '#7e8d99',
                  weight: segment.category === 'vehicle' ? 5 : 3,
                  opacity: mode === 'road' ? 0.72 : 0.34,
                  interactive: mode === 'road',
                }}
                eventHandlers={{ click: () => appendSegment(segment) }}
              ><Tooltip sticky>{segment.name} · {segment.highway}</Tooltip></Polyline>
            ))}
            {showPlanRoute && planRoute && (
              <Pane name="current-plan-route" style={{ zIndex: 460 }}>
                {planRoute.length > 1 && (
                  <>
                    <Polyline
                      positions={planRoute.map((point): [number, number] => [point.latitude_deg, point.longitude_deg])}
                      pathOptions={{ color: '#120b1a', weight: 9, opacity: 0.72, interactive: false }}
                    />
                    <Polyline
                      positions={planRoute.map((point): [number, number] => [point.latitude_deg, point.longitude_deg])}
                      pathOptions={{ color: '#bb6cff', weight: 5, opacity: 0.98, interactive: false }}
                    />
                  </>
                )}
                {planRoute.map((point, index) => (
                  <CircleMarker
                    key={`plan-${index}-${point.east_m}-${point.north_m}`}
                    center={[point.latitude_deg, point.longitude_deg]}
                    radius={index === 0 || index === planRoute.length - 1 ? 7 : 3}
                    pathOptions={{
                      color: index === 0 ? '#ff3030' : index === planRoute.length - 1 ? '#22c55e' : '#f1dcff',
                      fillColor: index === 0 ? '#ff3030' : index === planRoute.length - 1 ? '#22c55e' : '#7d3bb3',
                      fillOpacity: 1,
                      weight: 2,
                      interactive: false,
                    }}
                  >
                    <Tooltip>{index === 0 ? '현재 계획 시작' : index === planRoute.length - 1 ? '현재 계획 종료' : `현재 계획 ${index + 1}`} · {point.up_m} m</Tooltip>
                  </CircleMarker>
                ))}
                {captureActions.map((action) => {
                  const point = planRoute[action.waypoint_index];
                  return (
                    <Marker
                      key={`capture-${action.waypoint_index}-${action.landmark_id ?? ''}`}
                      position={[point.latitude_deg, point.longitude_deg]}
                      icon={captureMarkerIcon}
                      interactive
                      keyboard={false}
                    >
                      <Tooltip direction="top" offset={[0, -12]}>
                        사진 촬영 · WP {action.waypoint_index + 1}{action.landmark_id ? ` · ${action.landmark_id}` : ''}
                      </Tooltip>
                    </Marker>
                  );
                })}
              </Pane>
            )}
            {showEditRoute && (
              <Pane name="editable-route" style={{ zIndex: 470 }}>
                {waypoints.length > 1 && (
                  <Polyline
                    positions={waypoints.map((point): [number, number] => [point.latitude_deg, point.longitude_deg])}
                    pathOptions={{
                      color: '#42c4d4',
                      weight: routeLayerMode === 'compare' ? 3 : 5,
                      opacity: 0.98,
                      dashArray: routeLayerMode === 'compare' ? '9 7' : undefined,
                      interactive: false,
                    }}
                  />
                )}
                {waypoints.map((point, index) => (
                  <CircleMarker
                    key={`${index}-${point.latitude_deg}-${point.longitude_deg}`}
                    center={[point.latitude_deg, point.longitude_deg]}
                    radius={index === 0 || index === waypoints.length - 1 ? 6 : routeLayerMode === 'compare' ? 2 : 4}
                    pathOptions={{
                      color: index === 0 ? '#ef4444' : index === waypoints.length - 1 ? '#22c55e' : '#d9fbff',
                      fillColor: index === 0 ? '#ef4444' : index === waypoints.length - 1 ? '#22c55e' : '#26747f',
                      fillOpacity: 1,
                      weight: routeLayerMode === 'compare' ? 1 : 2,
                      interactive: false,
                    }}
                  ><Tooltip>{index === 0 ? '시작' : index === waypoints.length - 1 ? '종료' : index + 1} · {point.altitude_m} m</Tooltip></CircleMarker>
                ))}
              </Pane>
            )}
            <MapClick active={mode === 'point'} onPoint={appendPoint} />
          </MapContainer>
        </section>
      </div>
    </div>
  );
}
