import { describe, expect, it } from 'vitest';
import {
  applyDeadzone,
  hasMeaningfulManualInput,
  latchYaw,
  mapGamepadMode2,
  mapMode2,
  normalizeStickPointer,
  releaseStick,
} from './control';

describe('Mode 2 control mapping', () => {
  it('maps left stick to altitude/yaw and right stick to forward/right', () => {
    expect(mapMode2({ x: -1, y: 1 }, { x: 0.5, y: -0.5 }, 0)).toEqual({
      forward: -0.5,
      right: 0.5,
      up: 1,
      yaw: -1,
    });
  });

  it('maps standard gamepad Y axes from screen-down to flight-up', () => {
    expect(mapGamepadMode2([0.2, -1, -0.4, 0.5], 0)).toEqual({
      forward: -0.5,
      right: -0.4,
      up: 1,
      yaw: 0.2,
    });
  });

  it('applies a rescaled deadzone', () => {
    expect(applyDeadzone(0.07, 0.08)).toBe(0);
    expect(applyDeadzone(0.54, 0.08)).toBeCloseTo(0.5);
  });

  it('normalizes and clamps pointer input to a circular gate', () => {
    const rect = { left: 100, top: 100, width: 200, height: 200 };
    expect(normalizeStickPointer(200, 100, rect)).toEqual({ x: 0, y: 1 });
    const corner = normalizeStickPointer(400, -100, rect);
    expect(Math.hypot(corner.x, corner.y)).toBeCloseTo(1);
  });

  it('detects meaningful manual input after mapping', () => {
    expect(hasMeaningfulManualInput({ forward: 0, right: 0, up: 0, yaw: 0 })).toBe(false);
    expect(hasMeaningfulManualInput({ forward: 0.02, right: 0, up: 0, yaw: 0 })).toBe(true);
  });

  it('latches yaw when a gamepad returns to center', () => {
    expect(latchYaw(0.62, 0)).toBe(0.62);
    expect(latchYaw(0.62, -0.35)).toBe(-0.35);
  });

  it('centers vertical input while retaining latched yaw on release', () => {
    expect(releaseStick({ x: 0.7, y: -0.4 }, true)).toEqual({ x: 0.7, y: 0 });
    expect(releaseStick({ x: 0.7, y: -0.4 })).toEqual({ x: 0, y: 0 });
  });
});
