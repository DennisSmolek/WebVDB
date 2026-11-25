import { describe, expect, it } from "vitest";
import {
  VDB_FRAME_AMBIENT_MAX_PAIRS,
  vdbFrameAmbientDirections,
  vdbFrameAmbientIrradiance,
  vdbFrameAmbientPairTotals,
  vdbFrameAmbientVisibility,
  vdbFrameShadowMarch,
} from "../../src/tools/internal.js";

// K6.Q1 shadowed ambient (V85): the direction set, the per-pair totals and the distant-light irradiance the core
// kernels twin. Each test pins a property a port error (or a mutation) would break.

type V3 = [number, number, number];
const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;

describe("K6.Q1 vdbFrameAmbientDirections", () => {
  it("gives unit directions, the sky first, the ring at the equal-area centroid, evenly in azimuth", () => {
    const sky: V3 = [0.2, 0.9, -0.3];
    const s = sky.map((v) => v / Math.hypot(...sky));
    for (let n = 1; n <= VDB_FRAME_AMBIENT_MAX_PAIRS; n++) {
      const d = vdbFrameAmbientDirections(sky, n);
      expect(d.length).toBe(n);
      for (const v of d) expect(Math.hypot(...v)).toBeCloseTo(1, 12);
      expect(dot(d[0]!, s)).toBeCloseTo(1, 12);
      for (const v of d.slice(1)) expect(dot(v, s)).toBeCloseTo((1 - 1 / n) / 2, 12);
      // the ring's projections onto the tangent plane sum to zero (even azimuths)
      if (n > 2) {
        const sum = d.slice(1).reduce((a, v) => a.map((x, i) => x + v[i]! - s[i]! * dot(v, s)), [0, 0, 0]);
        expect(Math.hypot(...sum)).toBeLessThan(1e-12);
      }
    }
    expect(vdbFrameAmbientDirections([0, 1, 0], 99).length).toBe(VDB_FRAME_AMBIENT_MAX_PAIRS);
    expect(vdbFrameAmbientDirections([0, 1, 0], 0).length).toBe(1);
    // a sky along +y still gets a tangent frame
    expect(vdbFrameAmbientDirections([0, 1, 0], 4).every((v) => v.every(Number.isFinite))).toBe(true);
  });

  it("is a quadrature of the spherical mean of transmittance: a homogeneous ball, off-centre (error shrinks with pairs)", () => {
    // T(ω) = exp(−σ · distance to the unit sphere's surface from p along ω); exact mean by dense quadrature
    const p: V3 = [0.35, -0.2, 0.15];
    const sigma = 2.5;
    const exit = (w: readonly number[]) => {
      const b = dot(p, w);
      return -b + Math.sqrt(b * b - (dot(p, p) - 1));
    };
    let exact = 0;
    const M = 400;
    for (let i = 0; i < M; i++)
      for (let j = 0; j < 2 * M; j++) {
        const u = -1 + (2 * (i + 0.5)) / M;
        const phi = (Math.PI * (j + 0.5)) / M;
        const r = Math.sqrt(1 - u * u);
        exact += Math.exp(-sigma * exit([r * Math.cos(phi), u, r * Math.sin(phi)]));
      }
    exact /= 2 * M * M;
    const estimate = (n: number) => {
      const totals = vdbFrameAmbientDirections([0, 1, 0], n).map((w) => [Math.exp(-sigma * exit(w)), Math.exp(-sigma * exit(w.map((x) => -x)))] as [number, number]);
      const v = vdbFrameAmbientVisibility(totals);
      return (v[0] + v[1]) / 2;
    };
    const err = [1, 4, 7].map((n) => Math.abs(estimate(n) / exact - 1));
    // 1 pair (his sky up / down) is off by tens of percent here; 4 and 7 pairs within a few
    expect(err[1]!).toBeLessThan(0.05);
    expect(err[2]!).toBeLessThan(0.04);
    expect(err[1]!).toBeLessThan(err[0]!);
  });
});

describe("K6.Q1 pair totals and irradiance", () => {
  it("diffusion pairs pass the white furnace at albedo 1 (V up + V down = 2) and fall below with absorption", () => {
    const steps = Array.from({ length: 24 }, (_, i) => 0.05 + 0.04 * Math.sin(i));
    const back = Array.from({ length: 24 }, (_, i) => 0.08 + 0.02 * Math.cos(i * 0.7));
    const white: V3 = [1, 1, 1];
    const f = vdbFrameShadowMarch(steps, white, 0.2);
    const b = vdbFrameShadowMarch(back, white, 0.2);
    const [up, down] = vdbFrameAmbientPairTotals(f, b, true);
    expect(up + down).toBeCloseTo(2, 12);
    const grey: V3 = [0.5, 0.5, 0.5];
    const [u2, d2] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, grey, 0.2), vdbFrameShadowMarch(back, grey, 0.2), true);
    expect(u2 + d2).toBeLessThan(1.9);
    // no multiple scattering: each side's own total (the direct transmittance)
    const [u3, d3] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, white, 0), vdbFrameShadowMarch(back, white, 0), false);
    expect(u3).toBeCloseTo(Math.exp(-steps.reduce((a, x) => a + x, 0)), 12);
    expect(d3).toBeCloseTo(Math.exp(-back.reduce((a, x) => a + x, 0)), 12);
    // octaves: the octave total of each side's 1 / (1 + depth), over the open direction's (1 + 0.5 + 0.25)
    const oct: [number, number, number, number] = [2, 0.5, 0.5, 0.5];
    const depth = steps.reduce((a, x) => a + x, 0);
    const [u4] = vdbFrameAmbientPairTotals(vdbFrameShadowMarch(steps, white, 0, oct), vdbFrameShadowMarch(back, white, 0, oct), false, oct);
    expect(u4).toBeCloseTo((Math.exp(-depth) + 0.5 * Math.exp(-0.5 * depth) + 0.25 * Math.exp(-0.25 * depth)) / 1.75, 6);
  });

  it("unoccluded, the irradiance is the unshadowed renderer's ambient + hemisphere light; occluded it scales per side", () => {
    const light = { ambient: [0.3, 0.2, 0.1] as V3, hemiSky: [0.5, 0.6, 0.7] as V3, hemiGround: [0.1, 0.05, 0.02] as V3, sky: [0, 1, 0] as V3 };
    const open = vdbFrameAmbientIrradiance(light, [1, 1]);
    for (let c = 0; c < 3; c++) expect(open[c]).toBeCloseTo(light.ambient[c]! + (light.hemiSky[c]! + light.hemiGround[c]!) / 2, 12);
    const half = vdbFrameAmbientIrradiance(light, [0.5, 0.2]);
    for (let c = 0; c < 3; c++) expect(half[c]).toBeCloseTo(light.ambient[c]! * 0.35 + light.hemiSky[c]! * 0.25 + light.hemiGround[c]! * 0.1, 12);
    // a constant probe (c0 only): irradiance 0.886227 c0 at any visibility mean
    const probe = Array.from({ length: 9 }, (_, i) => (i === 0 ? ([1, 2, 3] as V3) : ([0, 0, 0] as V3)));
    const p = vdbFrameAmbientIrradiance({ ambient: [0, 0, 0], hemiSky: [0, 0, 0], hemiGround: [0, 0, 0], sky: [0, 1, 0], probe }, [0.6, 0.4]);
    for (let c = 0; c < 3; c++) expect(p[c]).toBeCloseTo(0.886227 * (c + 1) * 0.5, 9);
  });
});
