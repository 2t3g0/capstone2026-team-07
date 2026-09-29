import { useEffect, useMemo } from 'react';
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet';
import { divIcon } from 'leaflet';
import 'leaflet/dist/leaflet.css';

interface Props {
  latitude: number | null;
  longitude: number | null;
  headingRad: number | null;
  stale: boolean;
  previewStartNed?: number[] | null;
  previewEndNed?: number[] | null;
}

const FALLBACK: [number, number] = [35.2350126, 129.0748631];

function FollowVehicle({ position }: { position: [number, number] | null }) {
  const map = useMap();
  useEffect(() => {
    if (position) map.setView(position, Math.max(map.getZoom(), 18), { animate: false });
  }, [map, position?.[0], position?.[1]]);
  return null;
}

function nedDeltaToGeo(origin: [number, number], start: number[], end: number[]): [number, number] {
  const north = end[0] - start[0];
  const east = end[1] - start[1];
  return [
    origin[0] + north / 6_378_137 * 180 / Math.PI,
    origin[1] + east / (6_378_137 * Math.cos(origin[0] * Math.PI / 180)) * 180 / Math.PI,
  ];
}

export function LivePositionMap({ latitude, longitude, headingRad, stale,
  previewStartNed, previewEndNed }: Props) {
  const valid = latitude !== null && longitude !== null;
  const position: [number, number] | null = valid ? [latitude, longitude] : null;
  const endpoint = useMemo(() => (
    position && previewStartNed?.length === 3 && previewEndNed?.length === 3
      ? nedDeltaToGeo(position, previewStartNed, previewEndNed)
      : null
  ), [position?.[0], position?.[1], previewStartNed, previewEndNed]);
  const icon = useMemo(() => divIcon({
    className: 'live-aircraft-marker-wrap',
    html: `<div class="live-aircraft-marker ${stale ? 'is-stale' : ''}" style="transform:rotate(${(headingRad ?? 0) * 180 / Math.PI}deg)"><span></span></div>`,
    iconSize: [34, 34], iconAnchor: [17, 17],
  }), [headingRad, stale]);

  return (
    <div className={`live-map-shell ${stale ? 'is-stale' : ''}`}>
      <MapContainer className="live-map" center={position ?? FALLBACK} zoom={valid ? 18 : 16} zoomControl={false} attributionControl>
        <FollowVehicle position={position} />
        <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {position && <Marker position={position} icon={icon}><Tooltip permanent direction="top" offset={[0, -18]}>{stale ? 'STALE' : '현재 기체'}</Tooltip></Marker>}
        {position && endpoint && <>
          <Polyline positions={[position, endpoint]} pathOptions={{ color: '#e0b85c', weight: 4, dashArray: '7 6' }} />
          <CircleMarker center={endpoint} radius={6} pathOptions={{ color: '#f5d98f', fillColor: '#b98628', fillOpacity: 1 }}><Tooltip>시험 준비에서 확정한 탐색 종점</Tooltip></CircleMarker>
        </>}
      </MapContainer>
      {!valid && <div className="live-map-empty">GLOBAL POSITION 대기 중</div>}
      <div className={`live-map-freshness ${stale ? 'is-stale' : ''}`}>{stale ? 'STALE — 비행 준비 사용 불가' : 'LIVE POSITION'}</div>
    </div>
  );
}
