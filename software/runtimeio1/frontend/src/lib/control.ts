export interface StickVector {
  x: number;
  y: number;
}

export interface ManualAxes {
  forward: number;
  right: number;
  up: number;
  yaw: number;
}

export const ZERO_STICK: StickVector = Object.freeze({ x: 0, y: 0 });
export const ZERO_AXES: ManualAxes = Object.freeze({ forward: 0, right: 0, up: 0, yaw: 0 });

export function clampUnit(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-1, Math.min(1, value));
}

export function applyDeadzone(value: number, deadzone = 0.08): number {
  const safe = clampUnit(value);
  const threshold = Math.max(0, Math.min(0.95, deadzone));
  if (Math.abs(safe) <= threshold) return 0;
  return clampUnit(Math.sign(safe) * ((Math.abs(safe) - threshold) / (1 - threshold)));
}

export function normalizeStickPointer(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>,
): StickVector {
  const radius = Math.max(1, Math.min(rect.width, rect.height) / 2);
  const centerX = rect.left + rect.width / 2;
  const centerY = rect.top + rect.height / 2;
  let x = (clientX - centerX) / radius;
  let y = (centerY - clientY) / radius;
  const magnitude = Math.hypot(x, y);
  if (magnitude > 1) {
    x /= magnitude;
    y /= magnitude;
  }
  return { x: clampUnit(x), y: clampUnit(y) };
}

export function mapMode2(left: StickVector, right: StickVector, deadzone = 0.08): ManualAxes {
  return {
    forward: applyDeadzone(right.y, deadzone),
    right: applyDeadzone(right.x, deadzone),
    up: applyDeadzone(left.y, deadzone),
    yaw: applyDeadzone(left.x, deadzone),
  };
}

export function mapGamepadMode2(axes: readonly number[], deadzone = 0.08): ManualAxes {
  return mapMode2(
    { x: axes[0] ?? 0, y: -(axes[1] ?? 0) },
    { x: axes[2] ?? 0, y: -(axes[3] ?? 0) },
    deadzone,
  );
}

export function hasMeaningfulManualInput(axes: ManualAxes, threshold = 0.01): boolean {
  return Object.values(axes).some((value) => Math.abs(value) > threshold);
}

export function latchYaw(previousYaw: number, sampledYaw: number): number {
  return sampledYaw === 0 ? clampUnit(previousYaw) : clampUnit(sampledYaw);
}

export function releaseStick(value: StickVector, latchX = false): StickVector {
  return { x: latchX ? clampUnit(value.x) : 0, y: 0 };
}
