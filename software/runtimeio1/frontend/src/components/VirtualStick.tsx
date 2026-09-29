import { useRef, type ReactNode } from 'react';
import { normalizeStickPointer, releaseStick, type StickVector } from '../lib/control';

interface VirtualStickProps {
  label: string;
  xLabel: string;
  yLabel: string;
  value: StickVector;
  disabled?: boolean;
  latchXOnRelease?: boolean;
  headerAction?: ReactNode;
  onChange: (value: StickVector) => void;
}

export function VirtualStick({
  label,
  xLabel,
  yLabel,
  value,
  disabled,
  latchXOnRelease = false,
  headerAction,
  onChange,
}: VirtualStickProps) {
  const gateRef = useRef<HTMLDivElement>(null);
  const pointerRef = useRef<number | null>(null);

  const update = (clientX: number, clientY: number) => {
    if (!gateRef.current || disabled) return;
    onChange(normalizeStickPointer(clientX, clientY, gateRef.current.getBoundingClientRect()));
  };

  const release = (event: React.PointerEvent) => {
    if (pointerRef.current !== event.pointerId) return;
    pointerRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    onChange(releaseStick(value, latchXOnRelease));
  };

  return (
    <div className={`stick-control ${disabled ? 'is-disabled' : ''}`}>
      <div className="stick-heading">
        <div><strong>{label}</strong>{headerAction}</div>
        <span>{xLabel} / {yLabel}</span>
      </div>
      <div
        ref={gateRef}
        className="stick-gate"
        role="slider"
        aria-label={label}
        aria-disabled={disabled}
        tabIndex={disabled ? -1 : 0}
        onPointerDown={(event) => {
          if (disabled) return;
          pointerRef.current = event.pointerId;
          event.currentTarget.setPointerCapture(event.pointerId);
          update(event.clientX, event.clientY);
        }}
        onPointerMove={(event) => {
          if (pointerRef.current === event.pointerId) update(event.clientX, event.clientY);
        }}
        onPointerUp={release}
        onPointerCancel={release}
        onKeyDown={(event) => {
          if (disabled) return;
          const step = event.shiftKey ? 0.25 : 0.1;
          const next = { ...value };
          if (event.key === 'ArrowLeft') next.x -= step;
          else if (event.key === 'ArrowRight') next.x += step;
          else if (event.key === 'ArrowUp') next.y += step;
          else if (event.key === 'ArrowDown') next.y -= step;
          else return;
          event.preventDefault();
          onChange({ x: Math.max(-1, Math.min(1, next.x)), y: Math.max(-1, Math.min(1, next.y)) });
        }}
        onKeyUp={(event) => {
          if (event.key.startsWith('Arrow')) onChange(releaseStick(value, latchXOnRelease));
        }}
      >
        <span className="stick-axis stick-axis-x" />
        <span className="stick-axis stick-axis-y" />
        <span
          className="stick-knob"
          style={{ transform: `translate(calc(-50% + ${value.x * 66}px), calc(-50% - ${value.y * 66}px))` }}
        />
      </div>
      <div className="stick-values">
        <span>{xLabel} {value.x.toFixed(2)}</span>
        <span>{yLabel} {value.y.toFixed(2)}</span>
      </div>
    </div>
  );
}
