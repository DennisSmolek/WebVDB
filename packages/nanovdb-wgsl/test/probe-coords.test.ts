import { describe, expect, it } from "vitest";
import { probeCoords, probePoints } from "../src/cpu/probe-coords.js";

/**
 * `probeCoords`/`probePoints` (`src/cpu/probe-coords.ts`) exist so a fixed
 * seed reproduces the exact same coordinate sequence across hosts (TS today,
 * WGSL/other languages later). The snapshot below pins that sequence for
 * this seed/bbox combination — if it ever changes, either the splitmix64
 * constants or the draw order changed, and every downstream port needs to be
 * updated in lockstep.
 */

const SEED = 0x1234567890abcdefn;
const BBOX_MIN: [number, number, number] = [-10, -10, -10];
const BBOX_MAX: [number, number, number] = [10, 10, 10];
const DILATE = 4; // default

describe("probeCoords", () => {
  it("first 5 coords match the pinned golden sequence for a fixed seed", () => {
    const coords = probeCoords({ seed: SEED, count: 5, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(coords).toEqual([
      [14, 0, -11],
      [-12, 1, 10],
      [9, 2, -11],
      [10, 3, 11],
      [0, 8, -3],
    ]);
  });

  it("is deterministic: two calls with the same seed produce identical sequences", () => {
    const a = probeCoords({ seed: SEED, count: 50, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    const b = probeCoords({ seed: SEED, count: 50, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(b).toEqual(a);
  });

  it("different seeds produce different sequences", () => {
    const a = probeCoords({ seed: SEED, count: 20, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    const b = probeCoords({ seed: SEED + 1n, count: 20, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(b).not.toEqual(a);
  });

  it("respects the dilated bbox (inclusive) on every axis", () => {
    const coords = probeCoords({ seed: SEED, count: 500, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(coords.length).toBe(500);
    for (const [x, y, z] of coords) {
      expect(x).toBeGreaterThanOrEqual(BBOX_MIN[0] - DILATE);
      expect(x).toBeLessThanOrEqual(BBOX_MAX[0] + DILATE);
      expect(y).toBeGreaterThanOrEqual(BBOX_MIN[1] - DILATE);
      expect(y).toBeLessThanOrEqual(BBOX_MAX[1] + DILATE);
      expect(z).toBeGreaterThanOrEqual(BBOX_MIN[2] - DILATE);
      expect(z).toBeLessThanOrEqual(BBOX_MAX[2] + DILATE);
      expect(Number.isInteger(x)).toBe(true);
      expect(Number.isInteger(y)).toBe(true);
      expect(Number.isInteger(z)).toBe(true);
    }
  });

  it("honors a custom dilate value", () => {
    const coords = probeCoords({
      seed: SEED,
      count: 500,
      bboxMin: BBOX_MIN,
      bboxMax: BBOX_MAX,
      dilate: 0,
    });
    for (const [x, y, z] of coords) {
      expect(x).toBeGreaterThanOrEqual(BBOX_MIN[0]);
      expect(x).toBeLessThanOrEqual(BBOX_MAX[0]);
      expect(y).toBeGreaterThanOrEqual(BBOX_MIN[1]);
      expect(y).toBeLessThanOrEqual(BBOX_MAX[1]);
      expect(z).toBeGreaterThanOrEqual(BBOX_MIN[2]);
      expect(z).toBeLessThanOrEqual(BBOX_MAX[2]);
    }
  });
});

describe("probePoints", () => {
  it("first 5 points match the pinned golden sequence for a fixed seed", () => {
    const points = probePoints({ seed: SEED, count: 5, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(points).toEqual([
      [14.75, -10.25, 1.5],
      [9, -11, 3.5],
      [0.25, -2.25, 7],
      [-13.25, -6, -3],
      [-2.75, 13.75, -7],
    ]);
  });

  it("is deterministic: two calls with the same seed produce identical sequences", () => {
    const a = probePoints({ seed: SEED, count: 50, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    const b = probePoints({ seed: SEED, count: 50, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(b).toEqual(a);
  });

  it("integer base part respects the dilated bbox and fraction is one of {0, .25, .5, .75}", () => {
    const points = probePoints({ seed: SEED, count: 500, bboxMin: BBOX_MIN, bboxMax: BBOX_MAX });
    expect(points.length).toBe(500);
    const allowedFractions = [0, 0.25, 0.5, 0.75];
    for (const [x, y, z] of points) {
      for (const [v, lo, hi] of [
        [x, BBOX_MIN[0] - DILATE, BBOX_MAX[0] + DILATE],
        [y, BBOX_MIN[1] - DILATE, BBOX_MAX[1] + DILATE],
        [z, BBOX_MIN[2] - DILATE, BBOX_MAX[2] + DILATE],
      ] as const) {
        const base = Math.floor(v);
        const frac = v - base;
        expect(base).toBeGreaterThanOrEqual(lo);
        expect(base).toBeLessThanOrEqual(hi);
        const matches = allowedFractions.some((f) => Math.abs(f - frac) < 1e-9);
        expect(matches, `fraction ${frac} (from ${v}) not in {0,.25,.5,.75}`).toBe(true);
      }
    }
  });
});
