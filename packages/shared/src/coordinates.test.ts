import { describe, expect, it } from 'vitest';
import { metersToCanvas } from './coordinates';

describe('metersToCanvas', () => {
  it('maps pitch center to canvas center and flips y', () => {
    expect(metersToCanvas(0, 0, 105, 68, { x: 10, y: 20, width: 1050, height: 680 })).toEqual({ x: 535, y: 360 });
    expect(metersToCanvas(0, 34, 105, 68, { x: 0, y: 0, width: 1050, height: 680 }).y).toBe(0);
  });
});
